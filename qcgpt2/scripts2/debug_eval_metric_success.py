"""QCGPT2 Fidelity Debugging Script

Runs two evaluation paths (real-world Qiskit fidelity vs training-path
fidelity) and reports discrepancies to help diagnose metric drift.
"""
import torch
import numpy as np
import argparse
import sys
import pandas as pd
import torch.nn.functional as F
import json, os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
# Imports
from qcgpt2.gates2 import BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2, VOCAB2
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.training2.supervised import (
    build_simplified_dataloader2, 
    calculate_physical_fidelity_components,
    parallel_unitary_product
)
from qcgpt2.encoding2 import tokens_to_circuit2
from qcgpt2.unitaries2 import build_circuit_unitary2, get_unitary_for_token_id

# --- Method A: Generate Integers AND Return Them ---
def eval_method_a_standard_with_seqs(model, spec_batch, spec_pad_mask, U_tgt_batch, device):
    B = spec_batch.size(0)
    memory = model.encoder(spec_batch, spec_pad_mask)
    curr_seq = torch.full((B, 1), BOS_CIRC_ID2, dtype=torch.long, device=device)
    unfinished = torch.ones(B, dtype=torch.bool, device=device)
    
    generated_tokens = [] # List of (B,) tensors
    
    for _ in range(32):
        logits = model.decoder(curr_seq, memory, memory_key_padding_mask=spec_pad_mask)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
        curr_seq = torch.cat([curr_seq, next_tokens.unsqueeze(1)], dim=1)
        generated_tokens.append(next_tokens) # Save for Method B
        unfinished = unfinished & (next_tokens != EOS_CIRC_ID2)
        if not unfinished.any(): break
            
    # Fidelity via Qiskit
    fidelities = []
    seqs = curr_seq.cpu().numpy()
    U_tgts = U_tgt_batch.cpu().numpy()
    
    for b in range(B):
        clean = [t for t in seqs[b] if t not in [BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2]]
        try:
            circ = tokens_to_circuit2(clean)
            U_pred = build_circuit_unitary2(circ, n_qubits=3).numpy()
            trace = np.trace(U_tgts[b].conj().T @ U_pred)
            fidelities.append( (np.abs(trace)**2) / 64.0 )
        except:
            fidelities.append(0.0)
            
    # Return both Scores and the Raw Sequence (excluding BOS) for Method B
    # Stack generated tokens: (B, L)
    seqs_tensor = torch.stack(generated_tokens, dim=1)
    return torch.tensor(fidelities), seqs_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading: {args.ckpt}")
    model = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    st = torch.load(args.ckpt, map_location=device)
    if 'model_state_dict' in st: st = st['model_state_dict']
    model.load_state_dict(st, strict=False)
    model.eval()
    
    dataloader = build_simplified_dataloader2(num_samples=100, batch_size=20, n_qubits=3, 
                                              raw_max_depth=8, include_basis_states=True,num_workers=4,
                                              pin_memory=True)
    
    mats = []
    for tid in range(len(VOCAB2)):
        if tid in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
            mats.append(torch.eye(8, dtype=torch.complex64))
        else:
            mats.append(get_unitary_for_token_id(tid, n_qubits=3).to(dtype=torch.complex64))
    U_stack = torch.stack(mats, dim=0).to(device)
    cost_tensor = torch.zeros(len(VOCAB2), device=device) 
    
    spec_batch, spec_pad_mask, _, circ_tgt = next(iter(dataloader))
    spec_batch = spec_batch.to(device)
    spec_pad_mask = spec_pad_mask.to(device)
    circ_tgt = circ_tgt.to(device)
    
    B = spec_batch.size(0)
    U_tgt = torch.eye(8, dtype=torch.complex64, device=device).unsqueeze(0).repeat(B, 1, 1)
    with torch.no_grad():
        for t in range(circ_tgt.size(1)):
            U_tgt = torch.matmul(U_stack[circ_tgt[:, t].clamp(0, len(VOCAB2)-1)], U_tgt)
            
    print("\n--- Running Perfect Input Test ---")
    
    with torch.no_grad():
        # 1. Run Method A (Standard) -> Get Scores AND Sequences
        fids_A, seqs_A = eval_method_a_standard_with_seqs(model, spec_batch, spec_pad_mask, U_tgt, device)
        
        # 2. Feed EXACT Sequences to Method B Physics Engine
        # Convert Integers to One-Hot Float
        probs_perfect = F.one_hot(seqs_A, num_classes=len(VOCAB2)).float()
        
        # Pad to 32 if needed (Physics engine expects shape)
        if probs_perfect.size(1) < 32:
            pad_len = 32 - probs_perfect.size(1)
            pad = torch.zeros(B, pad_len, len(VOCAB2), device=device)
            pad[:, :, PAD_ID2] = 1.0 
            probs_perfect = torch.cat([probs_perfect, pad], dim=1)
            
        U_pred_B, _ = calculate_physical_fidelity_components(
            probs=probs_perfect, 
            U_stack=U_stack, 
            cost_tensor=cost_tensor, 
            device=device
        )
        
        trace = torch.einsum("bij,bij->b", U_tgt.conj(), U_pred_B)
        fids_B = (trace.abs() ** 2) / 64.0
        
    print(f"\nBatch Size: {B}")
    print(f"Fidelity A (Qiskit): {fids_A.mean().item():.4f}")
    print(f"Fidelity B (Tensor w/ Perfect Inputs): {fids_B.mean().item():.4f}")
    
    diff = fids_A - fids_B.cpu()
    print(f"Mean Discrepancy: {diff.mean().item():.4f}")

    if diff.abs().mean() < 1e-4:
        print("\n>>> CONCLUSION: Physics Engine is CORRECT. The problem is in the Generator.")
    else:
        print("\n>>> CONCLUSION: Physics Engine is BROKEN. It calculates the wrong matrix.")

if __name__ == "__main__":
    main()
