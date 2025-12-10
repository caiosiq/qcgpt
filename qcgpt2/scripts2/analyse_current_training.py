import os
import sys
"""QCGPT2 Training Analysis Script

Loads multiple checkpoints (pre/mid/fine), evaluates them on a sampled
dataset, and produces histograms/CSV summaries for fidelity and cost
under consistent generation settings.
"""
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# --- PATH FIX: Ensure qcgpt2 is in path ---
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)

# --- Imports ---
from qcgpt2.gates2 import BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2, VOCAB2
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.training2.supervised import build_simplified_dataloader2
from qcgpt2.encoding2 import tokens_to_circuit2
from qcgpt2.unitaries2 import build_circuit_unitary2, get_unitary_for_token_id

# --- 1. ROBUST LOCAL COST REGISTRY ---
def build_local_cost_registry():
    registry = {}
    for i, token in enumerate(VOCAB2):
        t_str = str(token).lower()
        cost = 0.0
        # IBM Heron/Eagle Error Rates
        if "cx" in t_str or "cz" in t_str or "ecr" in t_str: cost = 0.01
        elif "swap" in t_str: cost = 0.03
        elif "ccx" in t_str: cost = 0.06
        elif any(g in t_str for g in ["h", "rx", "ry", "sx", "x", "y"]): cost = 0.001
        registry[i] = cost
    return registry

LOCAL_COST_REGISTRY = build_local_cost_registry()

def calculate_circuit_cost(tokens):
    cost = 0.0
    for t in tokens:
        cost += LOCAL_COST_REGISTRY.get(int(t), 0.0)
    return cost

# --- Helper: Batch Evaluation ---
@torch.no_grad()
def evaluate_batch(model, spec_batch, spec_pad_mask, U_tgt_batch, device, max_len=64):
    B = spec_batch.size(0)
    memory = model.encoder(spec_batch, spec_pad_mask)
    curr_seq = torch.full((B, 1), BOS_CIRC_ID2, dtype=torch.long, device=device)
    unfinished = torch.ones(B, dtype=torch.bool, device=device)
    
    for _ in range(max_len):
        logits = model.decoder(curr_seq, memory, memory_key_padding_mask=spec_pad_mask)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
        curr_seq = torch.cat([curr_seq, next_tokens.unsqueeze(1)], dim=1)
        unfinished = unfinished & (next_tokens != EOS_CIRC_ID2)
        if not unfinished.any(): break
            
    fidelities = []
    costs = []
    generated_seqs = curr_seq.cpu().numpy()
    U_tgts = U_tgt_batch.cpu().numpy()
    
    for b in range(B):
        raw_toks = generated_seqs[b]
        clean_toks = []
        for t in raw_toks:
            if t == BOS_CIRC_ID2: continue
            if t == EOS_CIRC_ID2: break
            if t == PAD_ID2: continue
            clean_toks.append(t)
            
        costs.append(calculate_circuit_cost(clean_toks))
        
        try:
            circ = tokens_to_circuit2(clean_toks)
            U_pred = build_circuit_unitary2(circ, n_qubits=3).numpy()
            U_tgt = U_tgts[b]
            trace = np.trace(U_tgt.conj().T @ U_pred)
            fid = (np.abs(trace) ** 2) / (8.0 ** 2)
            fidelities.append(fid)
        except:
            fidelities.append(0.0)
            
    return fidelities, costs

def load_model(path, device):
    print(f"Loading: {path}")
    model = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    st = torch.load(path, map_location=device)
    if 'model_state_dict' in st: st = st['model_state_dict']
    st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
    model.load_state_dict(st, strict=False)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_pre", type=str, required=True, help="Imitation Model")
    parser.add_argument("--ckpt_mid", type=str, required=True, help="Physics Model")
    parser.add_argument("--ckpt_fine", type=str, required=True, help="Optimizer Model")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--raw_max_depth", type=int, default=8, help="Depth 8 or 32")
    parser.add_argument("--out_dir", type=str, default="final_report_histograms")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load Models
    model_pre = load_model(args.ckpt_pre, device)
    model_mid = load_model(args.ckpt_mid, device)
    model_fine = load_model(args.ckpt_fine, device)

    # Setup Data
    gen_max_len = max(32, args.raw_max_depth * 3) 
    dataloader = build_simplified_dataloader2(
        num_samples=args.num_samples,
        batch_size=100,
        n_qubits=3,
        raw_max_depth=args.raw_max_depth,
        include_basis_states=True,
        num_workers=4
    )
    
    # Unitary Stack
    mats = []
    for tid in range(len(VOCAB2)):
        if tid in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
            mats.append(torch.eye(8, dtype=torch.complex64))
        else:
            mats.append(get_unitary_for_token_id(tid, n_qubits=3).to(dtype=torch.complex64))
    U_stack = torch.stack(mats, dim=0).to(device)

    results = []
    print(f"Starting 3-Way Histogram Comparison on {args.num_samples} samples (Depth {args.raw_max_depth})...")
    
    for _, (spec_batch, spec_mask, _, circ_tgt) in enumerate(tqdm(dataloader)):
        spec_batch = spec_batch.to(device)
        spec_mask = spec_mask.to(device)
        circ_tgt = circ_tgt.to(device)
        B_curr = circ_tgt.size(0)
        
        # 1. Build Reference
        U_tgt = torch.eye(8, dtype=torch.complex64, device=device).unsqueeze(0).repeat(B_curr, 1, 1)
        ref_costs = []
        circ_tgt_np = circ_tgt.cpu().numpy()
        for b in range(B_curr):
            toks = [t for t in circ_tgt_np[b] if t not in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]]
            ref_costs.append(calculate_circuit_cost(toks))
        with torch.no_grad():
            for t in range(circ_tgt.size(1)):
                U_tgt = torch.matmul(U_stack[circ_tgt[:, t].clamp(0, len(VOCAB2)-1)], U_tgt)

        # 2. Evaluate Models
        fids_pre, costs_pre = evaluate_batch(model_pre, spec_batch, spec_mask, U_tgt, device, gen_max_len)
        fids_mid, costs_mid = evaluate_batch(model_mid, spec_batch, spec_mask, U_tgt, device, gen_max_len)
        fids_fine, costs_fine = evaluate_batch(model_fine, spec_batch, spec_mask, U_tgt, device, gen_max_len)
        
        for i in range(len(fids_pre)):
            results.append({
                "Ref_Cost": ref_costs[i],
                "Pre_Fid": fids_pre[i], "Pre_Cost": costs_pre[i],
                "Mid_Fid": fids_mid[i], "Mid_Cost": costs_mid[i],
                "Fine_Fid": fids_fine[i], "Fine_Cost": costs_fine[i]
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, "histogram_data.csv"), index=False)
    
    # --- PLOTTING: 3 Histograms Side-by-Side ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
    
    # Define a helper to plot each model
    def plot_model_hist(ax, fid_col, cost_col, model_name, color):
        # 1. Filter: Only count "Valid" solutions (Fidelity > 0.9)
        valid_df = df[df[fid_col] > 0.9]
        
        if len(valid_df) == 0:
            ax.text(0.5, 0.5, "No Valid Solutions", ha='center', fontsize=12)
            return
            
        # 2. Calculate Savings: (Reference - Model)
        # Positive = Cheaper than Qiskit
        savings = valid_df['Ref_Cost'] - valid_df[cost_col]
        
        # 3. Calculate Win Rate
        # Wins: Cheaper than Qiskit (> 0.0001 to handle float noise)
        wins = (savings > 0.0001).sum()
        win_rate = (wins / len(valid_df)) * 100
        avg_saving = savings[savings > 0.0001].mean() if wins > 0 else 0.0
        
        # 4. Plot
        ax.hist(savings, bins=30, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Parity (Qiskit)')
        
        # 5. Styling
        ax.set_title(f"{model_name}\nWin Rate: {win_rate:.2f}% (Avg Saving: {avg_saving:.4f})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Cost Savings (Positive = Cheaper)", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("Count of Circuits", fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    # Plot 1: Imitation (Pre)
    plot_model_hist(axes[0], 'Pre_Fid', 'Pre_Cost', 'Imitation (Pre-Trained)', 'blue')
    
    # Plot 2: Physics (Mid)
    plot_model_hist(axes[1], 'Mid_Fid', 'Mid_Cost', 'Physics (Mid-Trained)', 'green')
    
    # Plot 3: Optimizer (Fine)
    plot_model_hist(axes[2], 'Fine_Fid', 'Fine_Cost', 'Optimizer (Fine-Tuned)', 'red')

    plt.suptitle(f"Optimization Performance vs Qiskit (Depth {args.raw_max_depth})", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "three_way_comparison.png"))
    
    print(f"\nHistogram Grid saved to {args.out_dir}/three_way_comparison.png")

if __name__ == "__main__":
    main()
