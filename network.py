from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


def infer_in_channels(action_size: int, board_size: int) -> int:
    """根据动作空间推断输入通道数：含决策动作（renju，action_size > size²）→ 4，否则 3。"""
    return 4 if action_size > board_size * board_size else 3


def _migrate_in_channels(net_state: dict, target_in_channels: int):
    """把旧 3 通道 checkpoint 的 conv 权重迁移到新的 4 通道网络。

    旧编码 [my, opp, phase] → 新编码 [my, opp, forbidden, phase]：
    通道 0/1 不变，旧通道 2（phase）移到新通道 3，新通道 2（forbidden）补零。
    返回 (net_state, migrated)：migrated=True 表示发生了通道迁移（此时应丢弃优化器状态）。
    """
    key = "conv.weight"
    if key not in net_state:
        return net_state, False
    w = net_state[key]
    actual = int(w.shape[1])
    if actual == target_in_channels:
        return net_state, False
    if actual == 3 and target_in_channels == 4:
        out = w.new_zeros(w.shape[0], 4, w.shape[2], w.shape[3])
        out[:, 0] = w[:, 0]
        out[:, 1] = w[:, 1]
        out[:, 3] = w[:, 2]
        migrated = dict(net_state)
        migrated[key] = out
        return migrated, True
    raise ValueError(f"无法迁移输入通道数 {actual} -> {target_in_channels}（仅支持 3 -> 4）")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """
    小型AlphaZero风格的残差网络。

    参数:
    - in_channels: 输入通道数（默认为3）
    - board_size: H == W == board_size
    - action_size: 离散动作数量 (board_size * board_size)
    - n_res_blocks: 残差块数量（可调整以平衡速度与性能）
    - channels: 主体中卷积滤波器的数量
    """

    def __init__(self,
                 in_channels: int = 3,
                 board_size: int = 15,
                 action_size: int = 15*15,
                 n_res_blocks: int = 6,
                 channels: int = 128):
        super().__init__()

        self.board_size = board_size
        self.action_size = action_size
        self.channels = channels

        # 初始卷积块
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

        # 残差塔
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_res_blocks)])

        # 策略头
        # 小卷积 -> 展平 -> 线性层到action_size
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        # 价值头
        # 小卷积 -> 展平 -> 两层MLP -> 标量
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        # 对卷积层使用Kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入:
            x: (B, in_channels, H, W) float32
        返回:
            policy_logits: (B, action_size)  (原始logits，在softmax之前)
            value: (B, 1) float 在tanh之后的值，范围[-1,1]
        """
        # 主体
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        for blk in self.res_blocks:
            out = blk(out)

        # 策略头
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.shape[0], -1)
        policy_logits = self.policy_fc(p)  # (B, action_size)

        # 价值头
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.shape[0], -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        value = torch.tanh(v)

        return policy_logits, value
    
    def predict(self, state):
        """
        预测方法，接收状态并返回策略和价值。
        """
        # 将状态（如果仍是numpy数组）转换为PyTorch张量
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 添加额外的维度用于批次大小(1)

        # 将张量传入网络
        policy_logits, value = self(state_tensor)

        return policy_logits, value


class PyTorchModel:
    """
    围绕AlphaZeroNet的包装类，提供:
     - predict(encoded_states) -> (policy_np, value_np)
     - predict_batch(list_of_encoded_states) -> (policy_np, value_np)
     - train_batch(...) -> 损失字典和反向传播步骤
     - save / load
    """

    def __init__(self,
                 board_size: int = 15,
                 action_size: Optional[int] = None,
                 device: Optional[str] = None,
                 n_res_blocks: int = 3,
                 channels: int = 64,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 in_channels: Optional[int] = None):
        self.board_size = board_size
        self.action_size = action_size if action_size is not None else board_size * board_size
        # 未显式指定时按动作空间推断：renju（action_size=size²+4）→ 4 通道，否则 3 通道
        self.in_channels = in_channels if in_channels is not None else infer_in_channels(self.action_size, self.board_size)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = AlphaZeroNet(
            in_channels=self.in_channels,
            board_size=board_size,
            action_size=self.action_size,
            n_res_blocks=n_res_blocks,
            channels=channels
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')  # log_probs vs target probs
        self._value_only = False

    def set_value_only_finetune(self, enabled: bool = True) -> None:
        """微调模式：冻结 trunk + 策略头（含 BN 统计量），只训练价值头。

        用于强冠军模型的继续训练：策略分布保持不变（MCTS 选路不变），
        仅提升价值评估精度——消除全量更新把已收敛策略训坏的问题。
        """
        self._value_only = enabled
        lr = self.optimizer.defaults["lr"]
        wd = self.optimizer.defaults.get("weight_decay", 1e-4)
        if enabled:
            for name, p in self.net.named_parameters():
                p.requires_grad = name.startswith("value")
            params = [p for p in self.net.parameters() if p.requires_grad]
        else:
            for p in self.net.parameters():
                p.requires_grad = True
            params = list(self.net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    # -------------------------
    # 预测（用于MCTS的批次）
    # -------------------------
    def predict(self, encoded_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        encoded_states: numpy数组 (B, C, H, W) float32
        返回:
            policy_probs: np.array (B, action_size) float32
            values: np.array (B, 1) float32
        """
        training = self.net.training
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(encoded_states.astype(np.float32)).to(self.device)
            logits, value = self.net(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            values = value.cpu().numpy()
        self.net.train(training)
        return probs, values

    # -------------------------
    # 便利方法：从列表创建批次
    # -------------------------
    def predict_batch(self, states_list: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        states_list: np数组列表 (C,H,W)
        返回: policy_probs, values 作为np数组
        """
        batch = self.make_batch_from_states(states_list)
        return self.predict(batch)

    # -------------------------
    # 单个训练批次
    # -------------------------
    def train_batch(self,
                    states: np.ndarray,
                    target_pis: np.ndarray,
                    target_vs: np.ndarray,
                    epochs: int = 1) -> dict:
        self.net.train()
        if self._value_only:
            # 冻结模块切回 eval：防止 BN 在训练模式下漂移运行统计量、间接改变冻结的策略
            for name, m in self.net.named_modules():
                if not name.startswith("value") and isinstance(m, nn.BatchNorm2d):
                    m.eval()
        states_t = torch.from_numpy(states.astype(np.float32)).to(self.device)
        target_pis_t = torch.from_numpy(target_pis.astype(np.float32)).to(self.device)
        target_vs_t = torch.from_numpy(target_vs.astype(np.float32)).to(self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0

        for _ in range(epochs):
            self.optimizer.zero_grad()
            logits, values = self.net(states_t)
            log_probs = F.log_softmax(logits, dim=1)

            policy_loss = self.policy_loss_fn(log_probs, target_pis_t)
            value_loss = self.value_loss_fn(values, target_vs_t)

            loss = value_loss if self._value_only else (policy_loss + value_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 3.0)
            self.optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_loss += float(loss.item())

        ne = float(epochs)
        return {
            "policy_loss": total_policy_loss / ne,
            "value_loss": total_value_loss / ne,
            "total_loss": total_loss / ne
        }

    # -------------------------
    # 保存 / 加载
    # -------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "net": self.net.state_dict(),
            "opt": self.optimizer.state_dict(),
            "board_size": self.board_size,
            "action_size": self.action_size,
            "in_channels": self.in_channels,
        }
        torch.save(state, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        map_location = map_location or self.device
        state = torch.load(path, map_location=map_location)
        net_state, migrated = _migrate_in_channels(state["net"], self.in_channels)
        self.net.load_state_dict(net_state)
        # 发生通道迁移时丢弃优化器状态（动量张量形状与参数不匹配），仅保留权重。
        if not migrated and "opt" in state and state["opt"] is not None:
            try:
                self.optimizer.load_state_dict(state["opt"])
            except Exception:
                pass

    # -------------------------
    # 工具方法：将列表转换为批次
    # -------------------------
    @staticmethod
    def make_batch_from_states(list_of_encoded_states: list) -> np.ndarray:
        return np.stack(list_of_encoded_states, axis=0).astype(np.float32)