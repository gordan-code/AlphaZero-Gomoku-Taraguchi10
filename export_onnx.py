# -*- coding: utf-8 -*-
"""把训练好的 renju 快照导出为 ONNX，供 Electron 前端 onnxruntime 推理。

用法：
    python export_onnx.py --checkpoint models_renju/snapshot_iter31_xxx.pt --out models_renju/model.onnx

输入：(1, in_channels, 15, 15) float32，in_channels=4（我的棋子/对手棋子/禁手生效/阶段）。
输出：policy_logits (1, 229) 与 value (1, 1)。
"""
import argparse

import torch
import torch.nn as nn

from network import PyTorchModel


class ExportNet(nn.Module):
    """导出壳：AlphaZeroNet 原本返回 (policy_logits, value) 元组，这里原样导出双输出。"""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        logits, value = self.net(x)
        return logits, value


def main() -> None:
    p = argparse.ArgumentParser(description="导出 renju 快照为 ONNX")
    p.add_argument("--checkpoint", required=True, help="snapshot_iter*.pt 或 sl_pretrain_*.pt")
    p.add_argument("--out", default="model.onnx")
    p.add_argument("--board-size", type=int, default=15)
    p.add_argument("--action-size", type=int, default=229)
    p.add_argument("--opset", type=int, default=13)
    args = p.parse_args()

    m = PyTorchModel(board_size=args.board_size, action_size=args.action_size)
    m.load(args.checkpoint)  # 3 通道旧快照会自动迁移为 4 通道
    m.net.cpu()  # ONNX 面向 CPU/WASM 推理，统一放到 CPU 导出
    m.net.eval()

    export = ExportNet(m.net)
    x = torch.randn(1, m.in_channels, args.board_size, args.board_size)

    torch.onnx.export(
        export,
        x,
        args.out,
        input_names=["input"],
        output_names=["policy", "value"],
        opset_version=args.opset,
    )
    print(f"已导出 ONNX: {args.out}")
    print(f"  in_channels={m.in_channels}, action_size={args.action_size}")
    print(f"  输入 (1,{m.in_channels},{args.board_size},{args.board_size}) -> policy (1,{args.action_size}), value (1,1)")


if __name__ == "__main__":
    main()
