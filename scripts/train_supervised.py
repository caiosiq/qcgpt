# scripts/train_supervised.py
import torch
import torch.optim as optim
import os
import argparse
import time
import csv

from qcgpt.training.supervised import (
    build_simplified_dataloader,
    train_supervised_epoch,
)
from qcgpt.models.policy import CircuitPolicy
from qcgpt.gates import VOCAB


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
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    train_loader = build_simplified_dataloader(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        n_qubits=2,
        raw_max_depth=args.raw_max_depth,
        include_basis_states=True if args.basis_only else True,
        n_random_states=(0 if args.basis_only else args.n_random_states),
        num_workers=0,
    )

    best_loss = float("inf")

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or ts
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    loss_csv = os.path.join(run_dir, f"{args.prefix}_loss.csv")
    with open(loss_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"]) 

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_supervised_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        print(f"[Supervised] Epoch {epoch:03d}  TrainLoss={train_loss:.4f}")

        with open(loss_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.8f}"])

        if train_loss < best_loss:
            best_loss = train_loss
            best_path = os.path.join(run_dir, f"{args.prefix}_best.pt")
            torch.save({"model_state_dict": model.state_dict()}, best_path)

    final_path = os.path.join(run_dir, f"{args.prefix}_final.pt")
    torch.save({"model_state_dict": model.state_dict()}, final_path)


if __name__ == "__main__":
    main()
