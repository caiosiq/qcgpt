import torch
import numpy as np
import argparse
import sys
import pandas as pd
import json, os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
# Imports
from qcgpt2.gates2 import BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2, VOCAB2, GATE_COST_REGISTRY
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.training2.supervised import (
    build_simplified_dataloader2, 
    generate_differentiable_logits, 
    calculate_physical_fidelity_components,
    parallel_unitary_product
)
from qcgpt2.encoding2 import tokens_to_circuit2
from qcgpt2.unitaries2 import build_circuit_unitary2, get_unitary_for_token_id

# --- Method A: The "Real World" (Qiskit/Integer) Evaluation ---
def eval_method_a_standard(model, spec_batch, spec_pad_mask, U_tgt_batch, device):
    """
    Standard generation (Integer IDs) -> Qiskit Circuit -> Unitary -> Fidelity.
    """
    B = spec_batch.size(0)
    memory = model.encoder(spec_batch, spec_pad_mask)
    curr_seq = torch.full((B, 1), BOS_CIRC_ID2, dtype=torch.long, device=device)
    unfinished = torch.ones(B, dtype=torch.bool, device=device)
    
    # 1. Greedy Loop (Integers)
    for _ in range(32):
        logits = model.decoder(curr_seq, memory, memory_key_padding_mask=spec_pad_mask)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
        curr_seq = torch.cat([curr_seq, next_tokens.unsqueeze(1)], dim=1)
        unfinished = unfinished & (next_tokens != EOS_CIRC_ID2)
        if not unfinished.any(): break
            
    # 2. Fidelity via Qiskit/Numpy
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
            
    return torch.tensor(fidelities)

# --- Method B: The "Training" (Differentiable/Gumbel) Evaluation ---
def eval_method_b_training(model, spec_batch, spec_pad_mask, U_tgt_batch, U_stack, cost_tensor, device):
    """
    Differentiable generation (Embeddings) -> Physics Engine -> Fidelity.
    This exactly mimics your 'evaluate_supervised_epoch2'.
    """
    # 1. Greedy Generation (Embeddings / Gumbel)
    # Note: We use greedy=True to match what we assume Eval is doing
    probs_gen = generate_differentiable_logits(
        model, spec_batch, spec_pad_mask,
        bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2,
        max_len=32,
        greedy=True # Deterministic
    )
    
    # 2. Physics Engine
    # Note: This does NOT convert to Qiskit. It multiplies tensors.
    U_pred, _ = calculate_physical_fidelity_components(
        probs=probs_gen, 
        U_stack=U_stack, 
        cost_tensor=cost_tensor, 
        device=device
    )
    
    # 3. Fidelity via Torch
    trace = torch.einsum("bij,bji->b", U_tgt_batch.conj(), U_pred)
    fidelities = (trace.abs() ** 2) / 64.0
    return fidelities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Pre-Trained Model")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    print(f"Loading: {args.ckpt}")
    model = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    st = torch.load(args.ckpt, map_location=device)
    if 'model_state_dict' in st: st = st['model_state_dict']
    model.load_state_dict(st, strict=False)
    model.eval()
    
    # Setup Data (Small batch is enough to verify)
    dataloader = build_simplified_dataloader2(num_samples=100, batch_size=20, n_qubits=3, 
                                              raw_max_depth=8, include_basis_states=True,num_workers=4,
                                              pin_memory=True)
    
    # Setup Matrices for Method B
    mats = []
    for tid in range(len(VOCAB2)):
        if tid in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
            mats.append(torch.eye(8, dtype=torch.complex64))
        else:
            mats.append(get_unitary_for_token_id(tid, n_qubits=3).to(dtype=torch.complex64))
    U_stack = torch.stack(mats, dim=0).to(device)
    cost_tensor = torch.zeros(len(VOCAB2), device=device) # Dummy cost
    
    print("\n--- Running Comparison (Method A vs Method B) ---")
    
    # Get one batch
    spec_batch, spec_pad_mask, _, circ_tgt = next(iter(dataloader))
    spec_batch = spec_batch.to(device)
    spec_pad_mask = spec_pad_mask.to(device)
    circ_tgt = circ_tgt.to(device)
    
    # Build Ground Truth Unitary (Shared)
    B = spec_batch.size(0)
    U_tgt = torch.eye(8, dtype=torch.complex64, device=device).unsqueeze(0).repeat(B, 1, 1)
    with torch.no_grad():
        for t in range(circ_tgt.size(1)):
            U_tgt = torch.matmul(U_stack[circ_tgt[:, t].clamp(0, len(VOCAB2)-1)], U_tgt)
            
    # Run Comparisons
    with torch.no_grad():
        fids_A = eval_method_a_standard(model, spec_batch, spec_pad_mask, U_tgt, device)
        fids_B = eval_method_b_training(model, spec_batch, spec_pad_mask, U_tgt, U_stack, cost_tensor, device)
        
    # Report
    print(f"\nBatch Size: {B}")
    print(f"Avg Fidelity (Method A - Real World): {fids_A.mean().item():.4f}")
    print(f"Avg Fidelity (Method B - Training Eval): {fids_B.mean().item():.4f}")
    
    diff = fids_A - fids_B.cpu()
    print(f"\nMean Discrepancy (A - B): {diff.mean().item():.4f}")
    
    # Show individual failures
    df = pd.DataFrame({
        "Real_Fid": fids_A.cpu().numpy(),
        "Train_Fid": fids_B.cpu().numpy(),
        "Diff": diff.numpy()
    })
    print("\nSample Results:")
    print(df.head(10))

if __name__ == "__main__":
    main()