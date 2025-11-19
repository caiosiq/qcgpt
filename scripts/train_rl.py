# scripts/train_rl.py
import os
import torch
import torch.optim as optim
import argparse

from qcgpt.models.policy import CircuitPolicy
from qcgpt.gates import VOCAB
from qcgpt.training.rl import train_rl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--lambda_len", type=float, default=0.1)
    parser.add_argument("--max_gates_ref", type=int, default=6)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--use_blackbox", action="store_true")
    parser.add_argument("--method", type=str, default="statevector")
    parser.add_argument("--use_noise", action="store_true")
    parser.add_argument("--p1", type=float, default=0.0)
    parser.add_argument("--p2", type=float, default=0.0)
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

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    train_rl(
        model=model,
        optimizer=optimizer,
        device=device,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        max_len=args.max_len,
        lambda_len=args.lambda_len,
        max_gates_ref=args.max_gates_ref,
        log_every=args.log_every,
        use_blackbox=args.use_blackbox,
        method=args.method,
        use_noise=args.use_noise,
        p1=args.p1,
        p2=args.p2,
    )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, "checkpoints/rl_finetuned.pt")


if __name__ == "__main__":
    main()
