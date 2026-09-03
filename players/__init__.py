# -*- coding: utf-8 -*-
"""按规则名自动解析最新模型快照，供各 AI 选手（player_alpha*）复用。"""

import glob
import os
import re
from typing import Optional

# 正式快照（train.py / train_supervised.py 保存）
_OFFICIAL_PATTERNS = ("snapshot_iter*.pt", "sl_pretrain_*.pt")
# 训练中断遗留的候选 checkpoint（train.py 正常流程会删除），仅在无兼容正式快照时回退
_CANDIDATE_PATTERN = "_selfplay_candidate_iter*.pt"


def _checkpoint_compatible(path: str, board_size, action_size) -> bool:
    """读取 checkpoint 元数据（board_size/action_size），判断与目标网络是否兼容。

    旧版 save 无 in_channels 字段（3 通道），由 network.load 自动迁移，这里不校验。
    """
    if board_size is None and action_size is None:
        return True
    try:
        import torch  # 延迟导入：纯 MCTS 选手（player_mcts）不依赖 torch

        state = torch.load(path, map_location="cpu")
        if board_size is not None and state.get("board_size") != board_size:
            return False
        if action_size is not None and state.get("action_size") != action_size:
            return False
        return True
    except Exception:
        return False


def latest_model_path(model_dir: str,
                      board_size: Optional[int] = None,
                      action_size: Optional[int] = None) -> Optional[str]:
    """取目录下与目标网络结构兼容的最新模型快照。

    - 正式快照（snapshot_iter* / sl_pretrain_*）按文件名时间戳（YYYYMMDD_HHMMSS）
      排序，不依赖文件系统 mtime（文件同步/上传后 mtime 可能错乱）。
    - 无兼容正式快照时，回退到遗留候选 checkpoint（按迭代号排序）。
    - 传入 board_size / action_size 可跳过不兼容尺寸的快照（如 8×8 冒烟测试模型）。
    - 找不到任何兼容快照时返回 None。
    """
    official = []
    for pattern in _OFFICIAL_PATTERNS:
        for path in glob.glob(os.path.join(model_dir, pattern)):
            m = re.search(r"_(\d{8}_\d{6})\.pt$", os.path.basename(path))
            if not m or not _checkpoint_compatible(path, board_size, action_size):
                continue
            official.append((m.group(1), path))
    if official:
        return max(official)[1]

    candidates = []
    for path in glob.glob(os.path.join(model_dir, _CANDIDATE_PATTERN)):
        m = re.search(r"_iter(\d+)\.pt$", os.path.basename(path))
        if not m or not _checkpoint_compatible(path, board_size, action_size):
            continue
        candidates.append((int(m.group(1)), path))
    if candidates:
        return max(candidates)[1]
    return None


def default_model_path(rules: str,
                       board_size: int = 15,
                       action_size: Optional[int] = None) -> Optional[str]:
    """按规则选择模型目录并取最新兼容快照：renju -> models_renju，其他 -> models。

    action_size 未指定时按规则推导：renju = board_size² + 4（开局决策动作），
    其他 = board_size²。
    """
    rules = (rules or "").lower()
    if action_size is None:
        n_pos = board_size * board_size
        action_size = n_pos + 4 if rules == "renju" else n_pos
    model_dir = "models_renju" if rules == "renju" else "models"
    return latest_model_path(model_dir, board_size=board_size, action_size=action_size)
