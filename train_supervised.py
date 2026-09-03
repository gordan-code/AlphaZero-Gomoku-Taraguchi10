# -*- coding: utf-8 -*-
"""用 KataGo renju15x 蒸馏数据做监督预训练（行为克隆），预热策略与价值网络。

监督阶段只训练 225 个落子策略 + 价值；塔拉山口-10 的 4 个决策动作（交换/走法）留给自对弈学。
产出快照后，接 ``train.py --game renju --pretrained-model <快照>`` 做自对弈微调。

示例：
    python train_supervised.py --data-root D:/Develop/datasets/katago-gomoku-distill-2025.5/renju15x_label28b \
        --max-samples 2000000 --epochs 1 --batch-size 256 --model-out models_renju
"""
import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from data.katago_loader import KatagoRenjuDataset
from network import PyTorchModel


def train_supervised(
    data_root: str,
    split: str = "train",
    max_samples: int = None,
    batch_size: int = 256,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    n_res_blocks: int = 3,
    channels: int = 64,
    model_out: str = "models_renju",
    log_every: int = 50,
    pretrained: str = None,
):
    board_size = 15
    action_size = board_size * board_size + 4  # renju = 229

    model = PyTorchModel(
        board_size=board_size,
        action_size=action_size,
        n_res_blocks=n_res_blocks,
        channels=channels,
        lr=lr,
        weight_decay=weight_decay,
    )
    # in_channels 自动为 4（action_size > 225），无需手写

    if pretrained and os.path.exists(pretrained):
        print(f"从已有快照 warm-start：{pretrained}（3 通道自动迁移为 4 通道）")
        model.load(pretrained)

    ds = KatagoRenjuDataset(data_root, split=split, max_samples=max_samples)
    print(f"数据文件数: {len(ds)} | in_channels={model.in_channels} | action_size={action_size} | device={model.device}")

    device = model.device
    global_step = 0
    t0 = time.time()
    n_pos = action_size - 4  # 225

    for epoch in range(epochs):
        total_p_loss = 0.0
        total_v_loss = 0.0
        n_batch = 0
        for states, policies, values in ds.iter_batches(batch_size):
            states_t = torch.from_numpy(states).to(device)
            policies_t = torch.from_numpy(policies).to(device)
            values_t = torch.from_numpy(values).to(device)

            model.optimizer.zero_grad()
            logits, pred_v = model.net(states_t)  # logits (B,229), pred_v (B,1)

            # 只对 225 个落子位置做软交叉熵；决策动作 225-228 不参与监督
            log_probs = F.log_softmax(logits[:, :n_pos], dim=1)
            p_loss = -(policies_t * log_probs).sum(dim=1).mean()

            v_loss = F.mse_loss(pred_v, values_t)

            loss = p_loss + v_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), 3.0)
            model.optimizer.step()

            total_p_loss += float(p_loss.item())
            total_v_loss += float(v_loss.item())
            n_batch += 1
            global_step += 1

            if global_step % log_every == 0:
                el = time.time() - t0
                print(
                    f"epoch {epoch + 1}/{epochs} step {global_step} | "
                    f"p_loss={total_p_loss / n_batch:.4f} v_loss={total_v_loss / n_batch:.4f} | {el:.0f}s"
                )

        avg_p = total_p_loss / max(1, n_batch)
        avg_v = total_v_loss / max(1, n_batch)
        print(f"== epoch {epoch + 1}/{epochs} 完成 | avg p_loss={avg_p:.4f} avg v_loss={avg_v:.4f} ==")

    os.makedirs(model_out, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(model_out, f"sl_pretrain_{ts}.pt")
    model.save(out_path)
    print(f"已保存监督预训练快照: {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KataGo renju15x 监督预训练", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", required=True, help="renju15x_label28b 所在目录（含 train/ 与 val/）")
    p.add_argument("--split", default="train", help="train 或 val")
    p.add_argument("--max-samples", type=int, default=None, help="最多用多少样本（None=全部，约 6480 万）")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-res-blocks", type=int, default=3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--model-out", default="models_renju")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--pretrained", default=None, help="可选：从已有快照（3 通道自动迁移）继续监督训练")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    train_supervised(
        data_root=args.data_root,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        n_res_blocks=args.n_res_blocks,
        channels=args.channels,
        model_out=args.model_out,
        log_every=args.log_every,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
