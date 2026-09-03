# -*- coding: utf-8 -*-
"""KataGo-gomoku 蒸馏数据加载器（renju15x，有禁手 15×15）。

把 KataGo 格式的 npz 转成监督训练所需的 ``(状态, 策略, 价值)`` 三元组：

- 状态 (N, 4, 15, 15)：[我的棋子, 对手棋子, 禁手生效, 阶段=0]
- 策略 (N, 225)：软策略分布（去掉 pass 通道并逐样本归一化）
- 价值 (N, 1)：P(己胜) - P(己负)，范围 [-1, 1]

张量对应关系（见 katago-gomoku-distill-2025.5/README.md）：
  binaryInputNCHWPacked[:,1] / [:,2]  → 我的 / 对手棋子
  globalInputNC[:,5]（己黑=-1 / 己白=+1）→ 禁手生效
  globalTargetsNC[:,0:2]（己胜 / 己负）→ 价值
  policyTargetsNCMove[:,0,:225] → 软策略
"""
import glob
import os

import numpy as np

BOARD_SIZE = 15
N_POS = BOARD_SIZE * BOARD_SIZE  # 225


def _unpack_board(packed: np.ndarray) -> np.ndarray:
    """binaryInputNCHWPacked (N, 22, L) uint8 → (N, 22, 15, 15) uint8。"""
    bits = np.unpackbits(packed, axis=2)
    return bits[:, :, :N_POS].reshape(packed.shape[0], 22, BOARD_SIZE, BOARD_SIZE)


def _file_to_samples(path: str):
    """把单个 npz 转成 (states, policies, values)。"""
    d = np.load(path)
    n = d['binaryInputNCHWPacked'].shape[0]

    bits = _unpack_board(d['binaryInputNCHWPacked'])
    my = bits[:, 1].astype(np.float32)
    opp = bits[:, 2].astype(np.float32)

    # 禁手生效：globalInputNC[:,5] < 0（己方执黑）→ 1.0，否则 0.0
    forbidden = (d['globalInputNC'][:, 5] < 0).astype(np.float32)
    forbidden_plane = np.tile(forbidden[:, None, None], (1, BOARD_SIZE, BOARD_SIZE))
    phase_plane = np.zeros((n, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    states = np.stack([my, opp, forbidden_plane, phase_plane], axis=1)  # (N, 4, 15, 15)

    # 软策略：去掉 pass（索引 225），逐样本归一化（KataGo 数据未归一化）
    p = d['policyTargetsNCMove'][:, 0, :N_POS].astype(np.float32)
    p_sum = p.sum(axis=1, keepdims=True)
    p_sum = np.where(p_sum == 0, 1.0, p_sum)
    policies = p / p_sum

    # 价值：P(己胜) - P(己负)
    values = (d['globalTargetsNC'][:, 0] - d['globalTargetsNC'][:, 1]).astype(np.float32)
    values = values[:, None]  # (N, 1)

    return states, policies, values


class KatagoRenjuDataset:
    """按文件顺序迭代 renju15x 数据，可限制总样本数（内存友好）。"""

    def __init__(self, root: str, split: str = "train", max_samples: int = None):
        self.root = root
        self.split = split
        self.max_samples = max_samples
        files = sorted(glob.glob(os.path.join(root, split, "*.npz")))
        if not files:
            raise FileNotFoundError(f"未找到数据：{os.path.join(root, split)}/*.npz")
        self.files = files

    def __len__(self):
        return len(self.files)

    def iter_batches(self, batch_size: int = 256, shuffle_files: bool = True):
        """逐文件产出 (states (B,4,15,15), policies (B,225), values (B,1))。"""
        files = list(self.files)
        if shuffle_files:
            files = list(np.random.permutation(files))
        emitted = 0
        for fp in files:
            states, policies, values = _file_to_samples(fp)
            n = states.shape[0]
            for s in range(0, n, batch_size):
                if self.max_samples is not None and emitted >= self.max_samples:
                    return
                end = min(s + batch_size, n)
                if self.max_samples is not None:
                    end = min(end, s + (self.max_samples - emitted))
                if end <= s:
                    return
                yield states[s:end], policies[s:end], values[s:end]
                emitted += (end - s)
