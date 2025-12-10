import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)

# Imports
from qcgpt2.gates2 import BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2, VOCAB2, ID_TO_TOKEN2
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.training2.supervised import build_simplified_dataloader2

def analyze_batch_divergence(model, spec_batch, spec_pad_mask, device):
    """
    Compares Integer vs Embedding forward pass for a single batch.
    Returns: (total_steps, total_mismatches, max_diff)
    """
    B = spec_batch.size(0)
    
    # 1. Setup Inputs
    memory = model.encoder(spec_batch, spec_pad_mask)
    w_emb = model.decoder.token_emb.weight
    
    # Path A: Integer History
    curr_seq_A = torch.full((B, 1), BOS_CIRC_ID2, dtype=torch.long, device=device)
    
    # Path B: Embedding History
    curr_input_B = torch.full((B, 1), BOS_CIRC_ID2, dtype=torch.long, device=device)
    curr_embeds_B = model.decoder.token_emb(curr_input_B) 
    history_embeds_B = curr_embeds_B
    
    batch_mismatches = 0
    batch_steps = 0
    batch_max_diff = 0.0
    
    # Track which sequences in the batch are finished
    unfinished = torch.ones(B, dtype=torch.bool, device=device)

    for t in range(32):
        # --- PATH A: Standard Integer Forward ---
        logits_A = model.decoder(curr_seq_A, memory, memory_key_padding_mask=spec_pad_mask)
        next_logits_A = logits_A[:, -1, :] 
        token_A = torch.argmax(next_logits_A, dim=-1)
        
        # --- PATH B: Differentiable Embedding Forward ---
        logits_B = model.decoder.forward_embeds(history_embeds_B, memory, memory_key_padding_mask=spec_pad_mask)
        next_logits_B = logits_B[:, -1, :] 
        token_B_predicted = torch.argmax(next_logits_B, dim=-1)
        
        # --- Measure Divergence ---
        # Only count mismatches on active sequences
        mismatches = ((token_A != token_B_predicted) & unfinished).sum().item()
        
        diff = (next_logits_A - next_logits_B).abs()
        current_max_diff = diff.max().item()
        
        batch_mismatches += mismatches
        batch_steps += unfinished.sum().item() # Only count valid steps
        batch_max_diff = max(batch_max_diff, current_max_diff)
        
        # --- Force Synchronization ---
        # We assume Method B 'accepts' the token from A to continue the trace
        one_hot_A = F.one_hot(token_A, num_classes=len(VOCAB2)).float()
        next_embed_B = torch.matmul(one_hot_A, w_emb).unsqueeze(1)
        
        curr_seq_A = torch.cat([curr_seq_A, token_A.unsqueeze(1)], dim=1)
        history_embeds_B = torch.cat([history_embeds_B, next_embed_B], dim=1)
        
        # Update unfinished status
        unfinished = unfinished & (token_A != EOS_CIRC_ID2)
        if not unfinished.any():
            break
            
    return batch_steps, batch_mismatches, batch_max_diff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000, help="Total samples to test")
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    print(f"Loading: {args.ckpt}")
    model = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    st = torch.load(args.ckpt, map_location=device)
    if 'model_state_dict' in st: st = st['model_state_dict']
    model.load_state_dict(st, strict=False)
    model.eval()
    
    # Setup Data
    print(f"Generating {args.num_samples} samples for divergence testing...")
    dataloader = build_simplified_dataloader2(
        num_samples=args.num_samples, 
        batch_size=args.batch_size, 
        n_qubits=3, 
        raw_max_depth=8, 
        include_basis_states=True,
        num_workers=4,
        pin_memory=True
    )
    
    total_steps_global = 0
    total_mismatches_global = 0
    global_max_diff = 0.0
    
    with torch.no_grad():
        for i, (spec_batch, spec_pad_mask, _, _) in enumerate(tqdm(dataloader)):
            spec_batch = spec_batch.to(device)
            spec_pad_mask = spec_pad_mask.to(device)
            
            steps, miss, max_d = analyze_batch_divergence(model, spec_batch, spec_pad_mask, device)
            
            total_steps_global += steps
            total_mismatches_global += miss
            global_max_diff = max(global_max_diff, max_d)

    print("\n" + "="*40)
    print("ROBUST DIVERGENCE REPORT")
    print("="*40)
    print(f"Samples Tested:       {args.num_samples}")
    print(f"Total Token Steps:    {total_steps_global}")
    print(f"Total Mismatches:     {total_mismatches_global}")
    if total_steps_global > 0:
        print(f"Mismatch Rate:        {(total_mismatches_global / total_steps_global) * 100:.4f}%")
    print(f"Max Logit Difference: {global_max_diff:.8f}")
    print("="*40)
    
    if total_mismatches_global == 0:
        print("CONCLUSION: Floating point noise has ZERO effect on generation.")
    else:
        print("CONCLUSION: Floating point noise IS causing decision flips.")

if __name__ == "__main__":
    main()