import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg") # Headless plotting
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Library Imports
from qcgpt2.gates2 import BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2, VOCAB2
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.training2.supervised import build_simplified_dataloader2
from qcgpt2.encoding2 import tokens_to_circuit2
from qcgpt2.unitaries2 import build_circuit_unitary2, get_unitary_for_token_id

# --- Helper: Greedy Generation & Fidelity ---
@torch.no_grad()
def evaluate_batch(model, spec_batch, spec_pad_mask, U_tgt_batch, device, max_len=48):
    """
    Generates circuits greedily and calculates their fidelity against the target.
    """
    B = spec_batch.size(0)
    
    # 1. Greedy Generation (Autoregressive)
    memory = model.encoder(spec_batch, spec_pad_mask)
    curr_seq = torch.full((B, 1), BOS_CIRC_ID2, dtype=torch.long, device=device)
    unfinished = torch.ones(B, dtype=torch.bool, device=device)
    
    for _ in range(max_len):
        logits = model.decoder(curr_seq, memory, memory_key_padding_mask=spec_pad_mask)
        next_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_logits, dim=-1)
        curr_seq = torch.cat([curr_seq, next_tokens.unsqueeze(1)], dim=1)
        unfinished = unfinished & (next_tokens != EOS_CIRC_ID2)
        if not unfinished.any():
            break
            
    # 2. Fidelity Calculation (CPU Side)
    fidelities = []
    generated_seqs = curr_seq.cpu().numpy()
    U_tgts = U_tgt_batch.cpu().numpy()
    
    for b in range(B):
        toks = generated_seqs[b]
        clean_toks = []
        for t in toks:
            if t == BOS_CIRC_ID2: continue
            if t == EOS_CIRC_ID2: break
            if t == PAD_ID2: continue
            clean_toks.append(t)
            
        try:
            circ = tokens_to_circuit2(clean_toks)
            U_pred = build_circuit_unitary2(circ, n_qubits=3).numpy()
            U_tgt = U_tgts[b]
            trace = np.trace(U_tgt.conj().T @ U_pred)
            fid = (np.abs(trace) ** 2) / (8.0 ** 2)
            fidelities.append(fid)
        except Exception as e:
            print(f"Error parsing: {clean_toks}") # Uncomment to see
            failures += 1
            fidelities.append(0.0)
            
    return fidelities

def load_model_weights(model, path):
    print(f"Loading weights from: {path}")
    st = torch.load(path, map_location='cpu')
    if 'model_state_dict' in st: st = st['model_state_dict']
    st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
    model.load_state_dict(st, strict=False)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_pre", type=str, required=True, help="Path to Pre-Trained Model")
    parser.add_argument("--ckpt_mid", type=str, required=True, help="Path to Mid-Trained Model")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="comparison_results")
    parser.add_argument("--raw_max_depth", type=int, default=32, help="Circuit depth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model_pre = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    model_mid = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)

    load_model_weights(model_pre, args.ckpt_pre)
    load_model_weights(model_mid, args.ckpt_mid)

    dataloader = build_simplified_dataloader2(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        n_qubits=3,
        raw_max_depth=args.raw_max_depth,
        include_basis_states=True,
        n_random_states=0,
        num_workers=4, # <--- ADD THIS LINE (Fixes the crash and speeds up eval)
        pin_memory=True # <--- Good to add for GPU speed
    )
    
    mats = []
    for tid in range(len(VOCAB2)):
        if tid in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
            mats.append(torch.eye(8, dtype=torch.complex64))
        else:
            mats.append(get_unitary_for_token_id(tid, n_qubits=3).to(dtype=torch.complex64))
    U_stack = torch.stack(mats, dim=0).to(device)

    results = []
    print(f"Starting Comparison on {args.num_samples} samples...")
    
    for batch_idx, (spec_batch, spec_mask, circ_in, circ_tgt) in enumerate(tqdm(dataloader)):
        spec_batch = spec_batch.to(device)
        spec_mask = spec_mask.to(device)
        
        B_curr = circ_tgt.size(0)
        U_tgt_batch = torch.eye(8, dtype=torch.complex64, device=device).unsqueeze(0).repeat(B_curr, 1, 1)
        
        with torch.no_grad():
            for t in range(circ_tgt.size(1)):
                toks = circ_tgt[:, t]
                u_gates = U_stack[toks.clamp(0, len(VOCAB2)-1)]
                U_tgt_batch = torch.matmul(u_gates, U_tgt_batch)

        fids_pre = evaluate_batch(model_pre, spec_batch, spec_mask, U_tgt_batch, device)
        fids_mid = evaluate_batch(model_mid, spec_batch, spec_mask, U_tgt_batch, device)
        
        for fp, fm in zip(fids_pre, fids_mid):
            results.append({
                "Fidelity_Pre": fp,
                "Fidelity_Mid": fm,
                "Delta": fm - fp,
                "Winner": "Mid" if fm > fp else ("Pre" if fp > fm else "Tie")
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.out_dir, "comparison_raw.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n=== FINAL RESULTS ===")
    print(f"Pre-Trained Avg Fidelity: {df['Fidelity_Pre'].mean():.4f}")
    print(f"Mid-Trained Avg Fidelity: {df['Fidelity_Mid'].mean():.4f}")
    print(f"Absolute Improvement:    {df['Fidelity_Mid'].mean() - df['Fidelity_Pre'].mean():.4f}")
    print(f"Mid-Trained Win Rate:     {(df['Fidelity_Mid'] > df['Fidelity_Pre']).mean() * 100:.2f}%")
    print(f"Perfect Solutions (Pre):  {(df['Fidelity_Pre'] > 0.99).mean() * 100:.2f}%")
    print(f"Perfect Solutions (Mid):  {(df['Fidelity_Mid'] > 0.99).mean() * 100:.2f}%")

    # --- Plotting with Pure Matplotlib (No Seaborn) ---
    plt.figure(figsize=(10, 6))
    
    # Histogram instead of KDE
    plt.hist(df['Fidelity_Pre'], bins=20, alpha=0.5, label='Pre-Trained (Imitation)', density=True, color='blue')
    plt.hist(df['Fidelity_Mid'], bins=20, alpha=0.5, label='Mid-Trained (Physics)', density=True, color='green')
    
    plt.title("Fidelity Distribution: Imitation vs. Self-Generated Optimization")
    plt.xlabel("Unitary Fidelity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, "fidelity_dist.png"))
    
    # Box Plot
    plt.figure(figsize=(6, 6))
    plt.boxplot([df['Fidelity_Pre'], df['Fidelity_Mid']], labels=['Pre-Trained', 'Mid-Trained'])
    plt.title("Fidelity Comparison")
    plt.ylabel("Fidelity")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, "fidelity_boxplot.png"))
    
    print(f"Plots saved to {args.out_dir}")

if __name__ == "__main__":
    main()