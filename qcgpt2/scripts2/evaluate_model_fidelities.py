"""
qcgpt2/evaluation/evaluate_fidelity_recovery.py
Generates fidelity histograms for Section 5.1 using the VALIDATED method 
(reconstructing circuits from the training distribution).
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse

from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.gates2 import VOCAB2, BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2
from qcgpt2.unitaries2 import get_unitary_for_token_id
from qcgpt2.training2.supervised import build_simplified_dataloader2
from qcgpt2.training2.supervised import generate_differentiable_logits, parallel_unitary_product

def load_model(ckpt_path, device):
    print(f"Loading model from {ckpt_path}...")
    model = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    try:
        # weights_only=False to support legacy checkpoints
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
        return None
    model.eval()
    return model

def calculate_fidelity_batch(U_target, U_pred):
    # Batch fidelity calc
    # U_target: (B, 8, 8)
    # U_pred: (B, 8, 8)
    d = 8.0
    # Trace: sum(conj(A) * B) along last two dims
    # einsum 'bij,bij->b' computes trace(A^dag B)
    trace = torch.einsum('bij,bij->b', U_target.conj(), U_pred)
    fid = (trace.abs() ** 2) / (d ** 2)
    return fid

def run_evaluation(phase1_ckpt, phase2_ckpt, output_dir, n_samples=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Models
    model_p1 = load_model(phase1_ckpt, device)
    model_p2 = load_model(phase2_ckpt, device)
    
    # 2. Build Dataset (The "Easy" Distribution)
    # We use the DataLoader to get (Spec, TargetCircuit) pairs
    # This ensures U_target is exactly representable by depth <= 8
    print("Building Dataset...")
    dataloader = build_simplified_dataloader2(
        num_samples=n_samples,
        batch_size=100, # Efficient batching
        n_qubits=3,
        raw_max_depth=8,
        include_basis_states=True,
        n_random_states=0,
        num_workers=4
    )
    
    # 3. Preload Unitary Stack
    mats = []
    for tid in range(len(VOCAB2)):
        if tid in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
            mats.append(torch.eye(8, dtype=torch.complex128))
        else:
            mats.append(get_unitary_for_token_id(tid, n_qubits=3).to(dtype=torch.complex128))
    U_stack = torch.stack(mats, dim=0).to(device)

    all_fids_p1 = []
    all_fids_p2 = []

    print(f"Evaluating models on {n_samples} samples...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            spec_batch, spec_mask, circ_in, circ_tgt = batch
            spec_batch = spec_batch.to(device)
            spec_mask = spec_mask.to(device)
            circ_tgt = circ_tgt.to(device)
            
            # A. Calculate Ground Truth Unitary from Tokens
            # This is the "Previous Script" logic: U_tgt comes from the circuit
            # Shape: (B, L) -> (B, L, 8, 8) -> (B, 8, 8)
            U_tgt_seq = U_stack[circ_tgt.clamp(0, len(VOCAB2)-1)]
            U_target = parallel_unitary_product(U_tgt_seq)
            
            # B. Inference P1
            probs_p1 = generate_differentiable_logits(
                model_p1, spec_batch, spec_mask, 
                bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2, 
                max_len=16, greedy=True
            )
            tokens_p1 = torch.argmax(probs_p1, dim=-1) # (B, L)
            # Truncate at EOS? Parallel product handles Identity padding naturally
            # Just map EOS/PAD to Identity
            U_seq_p1 = U_stack[tokens_p1]
            U_pred_p1 = parallel_unitary_product(U_seq_p1)
            
            # C. Inference P2
            probs_p2 = generate_differentiable_logits(
                model_p2, spec_batch, spec_mask, 
                bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2, 
                max_len=16, greedy=True
            )
            tokens_p2 = torch.argmax(probs_p2, dim=-1)
            U_seq_p2 = U_stack[tokens_p2]
            U_pred_p2 = parallel_unitary_product(U_seq_p2)
            
            # D. Fidelity
            fid_p1 = calculate_fidelity_batch(U_target, U_pred_p1)
            fid_p2 = calculate_fidelity_batch(U_target, U_pred_p2)
            
            all_fids_p1.extend(fid_p1.cpu().numpy())
            all_fids_p2.extend(fid_p2.cpu().numpy())

    # --- Plotting ---
    fids_p1 = np.array(all_fids_p1)
    fids_p2 = np.array(all_fids_p2)
    
    # Calculate Stats
    mu1, med1 = fids_p1.mean(), np.median(fids_p1)
    mu2, med2 = fids_p2.mean(), np.median(fids_p2)
    
    plt.figure(figsize=(10, 6))
    
    label_p1 = f'Phase I (Pre-trained)\nMean: {mu1:.3f}, Median: {med1:.3f}'
    label_p2 = f'Phase II (Fine-tuned)\nMean: {mu2:.3f}, Median: {med2:.3f}'
    
    plt.hist(fids_p1, bins=30, alpha=0.6, label=label_p1, color='gray', edgecolor='k')
    plt.hist(fids_p2, bins=30, alpha=0.6, label=label_p2, color='#1f77b4', edgecolor='k')
    
    plt.xlabel('Unitary Fidelity', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.title('Fidelity Recovery: Impact of Unitary Fine-Tuning')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    hist_path = os.path.join(output_dir, 'fidelity_histogram.png')
    plt.savefig(hist_path, dpi=300)
    print(f"Saved histogram to {hist_path}")
    
    # Save Data
    df = pd.DataFrame({'fid_phase1': fids_p1, 'fid_phase2': fids_p2})
    df.to_csv(os.path.join(output_dir, 'fidelity_results.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", type=str, required=True, help="Path to Phase 1 Checkpoint")
    parser.add_argument("--p2", type=str, required=True, help="Path to Phase 2 Checkpoint")
    parser.add_argument("--out", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--samples", type=int, default=500, help="Number of test circuits")
    args = parser.parse_args()
    
    run_evaluation(args.p1, args.p2, args.out, args.samples)