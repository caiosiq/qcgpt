# scripts/train_supervised.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import os
import argparse
import time
import csv

from qcgpt1.training.supervised import (
    build_simplified_dataloader,
    train_supervised_epoch,
)
from qcgpt1.models.policy import CircuitPolicy
from qcgpt1.gates import VOCAB
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--raw_max_depth", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--basis_only", action="store_true")
    parser.add_argument("--n_random_states", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="model_checkpoints")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="transformer_v1")
    parser.add_argument("--use_unitary_loss", action="store_true")
    parser.add_argument("--lambda_sup", type=float, default=1.0)
    parser.add_argument("--lambda_U", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none","cosine","step","exp"]) 
    parser.add_argument("--t_max", type=int, default=50)  # for cosine
    parser.add_argument("--step_size", type=int, default=50)  # for step
    parser.add_argument("--gamma", type=float, default=0.5)  # for step/exp
    parser.add_argument("--mem_log_interval", type=int, default=0)  # 0 disables per-batch GPU mem logging
    parser.add_argument("--softmax_temp", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(VOCAB)

    model = CircuitPolicy(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=4,
        max_spec_len=256,
        max_circ_len=128,
    ).to(device)

    if args.ckpt is not None and os.path.exists(args.ckpt):
        try:
            state = torch.load(args.ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "exp":
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)

    train_loader = build_simplified_dataloader(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        n_qubits=3,
        raw_max_depth=args.raw_max_depth,
        include_basis_states=True if args.basis_only else True,
        n_random_states=(0 if args.basis_only else args.n_random_states),
        num_workers=16,
        pin_memory=True,
    )

    best_loss = float("inf")

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or ts
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    loss_csv = os.path.join(run_dir, f"{args.prefix}_loss.csv")
    config_txt = os.path.join(run_dir, f"{args.prefix}_config.txt")
    with open(loss_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"]) 
    with open(config_txt, "w") as f:
        f.write(f"use_unitary_loss={args.use_unitary_loss}\n")
        f.write(f"lambda_sup={args.lambda_sup}\n")
        f.write(f"lambda_U={args.lambda_U}\n")
        f.write(f"raw_max_depth={args.raw_max_depth}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"scheduler={args.scheduler}\n")
        f.write(f"t_max={args.t_max}\n")
        f.write(f"step_size={args.step_size}\n")
        f.write(f"gamma={args.gamma}\n")
        f.write(f"softmax_temp={args.softmax_temp}\n")

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_supervised_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            use_unitary_loss=args.use_unitary_loss,
            lambda_sup=args.lambda_sup,
            lambda_U=args.lambda_U,
            softmax_temp=args.softmax_temp,
        )
        if scheduler is not None:
            scheduler.step()
        print(f"[Supervised] Epoch {epoch:03d}  TrainLoss={train_loss:.4f}  LR={optimizer.param_groups[0]['lr']:.6f}")

        with open(loss_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.8f}"])

        if train_loss < best_loss:
            best_loss = train_loss
            best_path = os.path.join(run_dir, f"{args.prefix}_best.pt")
            torch.save({"model_state_dict": model.state_dict()}, best_path)

        if epoch % 10 == 0:
            e_path = os.path.join(run_dir, f"{args.prefix}_e{epoch}.pt")
            torch.save({"model_state_dict": model.state_dict()}, e_path)
            files = [f for f in os.listdir(run_dir) if f.startswith(f"{args.prefix}_e") and f.endswith(".pt")]
            def parse_e(f):
                m = re.search(r"_e(\d+)\.pt$", f)
                return int(m.group(1)) if m else -1
            files_sorted = sorted(files, key=parse_e)
            while len(files_sorted) > 3:
                old = files_sorted.pop(0)
                try:
                    os.remove(os.path.join(run_dir, old))
                except FileNotFoundError:
                    pass

    


if __name__ == "__main__":
    main()
