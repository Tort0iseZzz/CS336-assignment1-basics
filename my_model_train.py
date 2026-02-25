import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

from cs336_basics.transformer_arch import Transformer_lm
from cs336_basics.transformer_train import AdamW, cross_entropy, learning_rate_schedule, gradient_clipping
from cs336_basics.data_loader import data_loading
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint


def train():
    parser = argparse.ArgumentParser(description="CS336 Assignment 1 Training Script")
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to training .bin file"
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to validation .bin file"
    )
    parser.add_argument(
        "--out_dir", type=str, default="checkpoints", help="Where to save checkpoints"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    # 1. 硬件准备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # 2. 数据加载 (Memory Mapping)
    # 假设数据是以 uint16 存储的 token ID
    train_data = np.memmap(args.train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_path, dtype=np.uint16, mode="r")

    # 3. 初始化模型、优化器
    # 注意：这里的模型参数需根据你的 Transformer 实现进行配置
    model_config = {
        "vocab_size": 10000,
        "num_layers": 6,
        "num_heads": 8,
        "d_model": 512,
        "context_length": args.context_length,
    }
    model = Transformer_lm(**model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    start_iter = 0
    if args.resume:
        ckpt_path = os.path.join(args.out_dir, "latest_checkpoint.pt")
        if os.path.exists(ckpt_path):
            start_iter = load_checkpoint(ckpt_path, model, optimizer)
            print(f"Resumed from iteration {start_iter}")

    # 4. 损失函数 (你之前实现的 Cross Entropy)
    criterion = cross_entropy

    # 5. 训练主循环 (带有 tqdm 进度条)
    # total 设置为剩余的迭代次数
    pbar = tqdm(range(start_iter, args.max_iters), desc="Training")

    for it in pbar:
        model.train()
        # 获取 Batch
        x, y = data_loading(train_data, args.batch_size, args.context_length, device)

        # 前向传播
        logits = model(x)  # 预期形状: (B, T, V)
        # CrossEntropy 要求输入为 (N, C)，所以需要 reshape
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 建议添加梯度裁剪以保证 RTX 5060 训练稳定性
        gradient_clipping(model.parameters(), 1.0)
        optimizer.step()

        # 更新 tqdm 进度条右侧的统计信息
        if it % 10 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 定期验证
        if it > 0 and it % args.eval_interval == 0:
            val_loss = estimate_loss(
                model, val_data, args.batch_size, args.context_length, device, criterion
            )
            print(f"\nStep {it}: Val Loss = {val_loss:.4f}")
            # 如果你有 wandb，可以在这里 log

        # 定期保存
        if it > 0 and it % args.save_interval == 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_{it}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
            # 同时更新一个 latest 指针方便 resume
            save_checkpoint(
                model, optimizer, it, os.path.join(args.out_dir, "latest_checkpoint.pt")
            )


@torch.no_grad()
def estimate_loss(
    model, data, batch_size, context_length, device, criterion, eval_iters=100
):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = data_loading(data, batch_size, context_length, device)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    return np.mean(losses)


if __name__ == "__main__":
    train()
