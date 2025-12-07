import os
import argparse
import torch
import numpy as np

from qcgpt2.gates2 import BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2, VOCAB2
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.data.qiskit_utils2 import sample_task2 as sample_task
from qcgpt2.data.specs2 import build_spec_sequence_batch
from qcgpt2.encoding2 import tokens_to_circuit2
from qcgpt1.evaluation.metrics import gate_count
from qcgpt2.unitaries2 import build_circuit_unitary2
from qcgpt1.decoding import beam_search_decode, top_k_decode, top_p_decode


def load_model(ckpt_path: str, device: torch.device) -> CircuitPolicy2:
    model = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        sd = state.get("model_state_dict", state)
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--max_gates_ref", type=int, default=6)
    parser.add_argument("--decode_strategy", type=str, default="greedy", choices=["greedy","beam","topk","topp"])
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)

    for i in range(args.num_examples):
        spec_tensor, ref_circ = sample_task(max_gates=args.max_gates_ref)
        spec_batch_np, spec_pad_mask_np = build_spec_sequence_batch([spec_tensor])
        spec_batch = torch.tensor(spec_batch_np, dtype=torch.float32, device=device)
        spec_pad_mask = torch.tensor(spec_pad_mask_np, dtype=torch.bool, device=device)
        if args.decode_strategy == "greedy":
            sampled_tokens, _ = model.sample_circuit_tokens(spec_batch, spec_pad_mask, BOS_CIRC_ID2, EOS_CIRC_ID2, args.max_len)
        elif args.decode_strategy == "beam":
            sampled_tokens = beam_search_decode(model, spec_batch, spec_pad_mask, max_len=args.max_len, bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2, beam_width=args.beam_width)
        elif args.decode_strategy == "topk":
            sampled_tokens = top_k_decode(model, spec_batch, spec_pad_mask, max_len=args.max_len, bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2, k=args.top_k)
        else:
            sampled_tokens = top_p_decode(model, spec_batch, spec_pad_mask, max_len=args.max_len, bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2, p=args.top_p)
        toks = sampled_tokens[0].tolist()
        seq = []
        for t in toks:
            if t in (PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2):
                continue
            seq.append(t)
        cand = tokens_to_circuit2(seq)
        U_ref = build_circuit_unitary2(ref_circ, n_qubits=3)
        U_cand = build_circuit_unitary2(cand, n_qubits=3)
        trace = torch.einsum("ij,ij->", U_ref.conj(), U_cand)
        d = U_ref.size(0)
        fid = (trace.abs() ** 2) / (d ** 2)
        print(f"Example {i+1}: gates={gate_count(ref_circ)} → cand_gates={gate_count(cand)}  fidelity={float(fid):.4f}")


if __name__ == "__main__":
    main()
