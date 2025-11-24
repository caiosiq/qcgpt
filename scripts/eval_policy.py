# scripts/eval_policy.py

import os
import argparse
import numpy as np
import torch

from qcgpt.gates import VOCAB, PAD_ID, BOS_CIRC_ID, EOS_CIRC_ID
from qcgpt.models.policy import CircuitPolicy
from qcgpt.data.qiskit_utils import sample_task
from qcgpt.encoding import tokens_to_circuit
from qcgpt.data.specs import build_spec_sequence_batch
from qcgpt.evaluation.metrics import (
    quantum_fidelity_from_spec,
    gate_count,
)
from qcgpt.evaluation.visualize import (
    format_circuit,
)


def load_model(ckpt_path: str, device: torch.device) -> CircuitPolicy:
    vocab_size = len(VOCAB)
    model = CircuitPolicy(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=4,
        max_spec_len=256,
        max_circ_len=128,
    ).to(device)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    else:
        print("WARNING: No checkpoint found; evaluating random-initialized model.")

    model.eval()
    return model


def evaluate_single_task(
    model: CircuitPolicy,
    device: torch.device,
    max_len: int = 32,
    max_gates_ref: int = 6,
    use_qiskit: bool = True,
    decode_strategy: str = "greedy",
    beam_width: int = 5,
    top_k: int = 5,
    top_p: float = 0.9,
):
    spec_tensor, ref_circ = sample_task(max_gates=max_gates_ref)
    spec_batch_np, spec_pad_mask_np = build_spec_sequence_batch([spec_tensor])
    spec_batch = torch.tensor(spec_batch_np, dtype=torch.float32, device=device)
    spec_pad_mask = torch.tensor(spec_pad_mask_np, dtype=torch.bool, device=device)

    # 3) Sample circuit tokens from policy using selected strategy
    if decode_strategy == "greedy":
        with torch.no_grad():
            sampled_tokens, _ = model.sample_circuit_tokens(
                spec_batch=spec_batch,
                spec_pad_mask=spec_pad_mask,
                bos_id=BOS_CIRC_ID,
                eos_id=EOS_CIRC_ID,
                max_len=max_len,
            )
    elif decode_strategy == "beam":
        sampled_tokens = beam_search_decode(model, spec_batch, spec_pad_mask, max_len=max_len, bos_id=BOS_CIRC_ID, eos_id=EOS_CIRC_ID, beam_width=beam_width)
    elif decode_strategy == "topk":
        sampled_tokens = top_k_decode(model, spec_batch, spec_pad_mask, max_len=max_len, bos_id=BOS_CIRC_ID, eos_id=EOS_CIRC_ID, k=top_k)
    elif decode_strategy == "topp":
        sampled_tokens = top_p_decode(model, spec_batch, spec_pad_mask, max_len=max_len, bos_id=BOS_CIRC_ID, eos_id=EOS_CIRC_ID, p=top_p)
    else:
        with torch.no_grad():
            sampled_tokens, _ = model.sample_circuit_tokens(
                spec_batch=spec_batch,
                spec_pad_mask=spec_pad_mask,
                bos_id=BOS_CIRC_ID,
                eos_id=EOS_CIRC_ID,
                max_len=max_len,
            )
    seq = sampled_tokens[0].tolist()
    # remove padding
    seq = [t for t in seq if t != PAD_ID]
    candidate_circ = tokens_to_circuit(seq)

    fid_ref = quantum_fidelity_from_spec(spec_tensor, ref_circ) if use_qiskit else float("nan")
    fid_cand = quantum_fidelity_from_spec(spec_tensor, candidate_circ) if use_qiskit else float("nan")

    gc_ref = gate_count(ref_circ)
    gc_cand = gate_count(candidate_circ)

    # 5) Pretty-print everything
    print("=" * 60)
    print("Amplitude-based mapping spec with basis pairs")
    print("-" * 60)
    print("Reference circuit (random task generator):")
    print(format_circuit(ref_circ))
    print(f"  Classical acc: {acc_ref:.3f}  |  Quantum fid: {fid_ref:.3f}  |  Gates: {gc_ref}")
    print("-" * 60)
    print(f"QCGPT proposed circuit (decode={decode_strategy}):")
    print(format_circuit(candidate_circ))
    print(f"  Classical acc: {acc_cand:.3f}  |  Quantum fid: {fid_cand:.3f}  |  Gates: {gc_cand}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/rl_finetuned.pt",
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of random tasks to evaluate.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=32,
        help="Max circuit token length for sampling.",
    )
    parser.add_argument(
        "--max_gates_ref",
        type=int,
        default=6,
        help="Max gates in reference circuit when sampling tasks.",
    )
    parser.add_argument(
        "--no_qiskit",
        action="store_true",
        help="Skip Qiskit quantum fidelity computation (faster).",
    )
    parser.add_argument("--decode_strategy", type=str, default="greedy", choices=["greedy","beam","topk","topp"])
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.ckpt, device=device)

    use_qiskit = not args.no_qiskit

    print(f"Evaluating {args.num_examples} random tasks.")
    print(f"Using Qiskit fidelity: {use_qiskit}")

    for i in range(args.num_examples):
        print(f"\n### Example {i+1}/{args.num_examples}")
        evaluate_single_task(
            model=model,
            device=device,
            max_len=args.max_len,
            max_gates_ref=args.max_gates_ref,
            use_qiskit=use_qiskit,
            decode_strategy=args.decode_strategy,
            beam_width=args.beam_width,
            top_k=args.top_k,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
from qcgpt.decoding import beam_search_decode, top_k_decode, top_p_decode
