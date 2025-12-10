"""QCGPT2 Gumbel Reconstruct Script

Demonstrates differentiable generation via Gumbel-Softmax from logits,
building a circuit token sequence and inspecting selected IDs.
"""
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import os
# Adjust path to find your library
sys.path.append(".") 

from qcgpt2.gates2 import BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2, VOCAB2, ID_TO_TOKEN2
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.data.specs2 import build_spec_sequence_batch
from qcgpt2.data.qiskit_utils2 import sample_task2 as sample_task

# --- 1. The Standard Greedy Generator (Control Group) ---
@torch.no_grad()
def generate_standard_greedy(model, spec_batch, spec_pad_mask, max_len=32):
    """
    Standard autoregressive generation using discrete token IDs.
    """
    B = spec_batch.size(0)
    device = spec_batch.device
    
    # Encoder
    memory = model.encoder(spec_batch, spec_pad_mask)
    
    # Decoder Loop
    curr_seq = torch.full((B, 1), BOS_CIRC_ID2, dtype=torch.long, device=device)
    
    for _ in range(max_len):
        # Forward pass (Discrete IDs)
        logits = model.decoder(curr_seq, memory, memory_key_padding_mask=spec_pad_mask)
        next_token_logits = logits[:, -1, :]
        
        # Greedy Choice
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        curr_seq = torch.cat([curr_seq, next_token], dim=1)
        
        if next_token.item() == EOS_CIRC_ID2:
            break
            
    return curr_seq[0].tolist()

# --- 2. The Differentiable Generator (Experimental Group) ---
# Note: We simulate "Training Mode" logic but wrap in no_grad for comparison
@torch.no_grad()
def generate_differentiable_debug(model, spec_batch, spec_pad_mask, max_len=32, temp=1e-2):
    """
    Autoregressive generation using Embeddings + Gumbel(Hard).
    At low temp, this should behave exactly like Greedy.
    """
    B = spec_batch.size(0)
    device = spec_batch.device
    
    # Encoder
    memory = model.encoder(spec_batch, spec_pad_mask)
    
    # Access Weights
    w_emb = model.decoder.token_emb.weight
    
    # Initial Input: BOS Embedding
    curr_input_ids = torch.full((B, 1), BOS_CIRC_ID2, dtype=torch.long, device=device)
    curr_embeds = model.decoder.token_emb(curr_input_ids) # (B, 1, D)
    
    history_embeds = curr_embeds
    generated_ids = [BOS_CIRC_ID2]
    
    for _ in range(max_len):
        # Forward Pass (Continuous Embeddings)
        # Using the new method you added to CircuitPolicy2
        logits_seq = model.decoder_forward_embeds(history_embeds, memory, memory_key_padding_mask=spec_pad_mask)
        next_token_logits = logits_seq[:, -1, :]
        
        # Gumbel Softmax (Hard=True)
        # At temp=1e-5, this acts as a differentiable argmax
        
        next_token_onehot = F.gumbel_softmax(next_token_logits, tau=temp, hard=True, dim=-1)
        # 1. Find the index
        # idx = torch.argmax(next_token_logits, dim=-1)
        
        # # 2. Create One-Hot
        # next_token_onehot = F.one_hot(idx, num_classes=len(VOCAB2)).float()
        
        # Track what ID was selected (by finding the 1.0)
        selected_id = torch.argmax(next_token_onehot, dim=-1).item()
        generated_ids.append(selected_id)
        
        # Prepare Next Input (Matrix Multiplication)
        # This is the "Differentiable Connection"
        next_embed = torch.matmul(next_token_onehot, w_emb).unsqueeze(1)
        
        # Append
        history_embeds = torch.cat([history_embeds, next_embed], dim=1)
        
        if selected_id == EOS_CIRC_ID2:
            break
            
    return generated_ids

# --- Main Test Loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    if os.path.exists(args.ckpt):
        print(f"Loading {args.ckpt}...")
        st = torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(st.get("model_state_dict", st), strict=False)
    model.eval() # Disable dropout for deterministic comparison!
    
    print("\n--- Running Consistency Check (Greedy vs Diff-AutoReg) ---")
    
    for i in range(10):
        print(f"\nTest Case {i}:")
        spec_tensor, _ = sample_task(max_gates=8)
        spec_batch, spec_mask = build_spec_sequence_batch([spec_tensor])
        spec_batch = torch.tensor(spec_batch, dtype=torch.float32, device=device)
        spec_mask = torch.tensor(spec_mask, dtype=torch.bool, device=device)
        
        # 1. Run Standard
        seq_std = generate_standard_greedy(model, spec_batch, spec_mask)
        print(f"Standard: {[ID_TO_TOKEN2.get(t, t) for t in seq_std]}")
        
        # 2. Run Differentiable (Temp ~ 0)
        seq_diff = generate_differentiable_debug(model, spec_batch, spec_mask, temp=1e-1)
        print(f"DiffProp: {[ID_TO_TOKEN2.get(t, t) for t in seq_diff]}")
        
        # 3. Verify
        if seq_std == seq_diff:
            print(">>> MATCH: PERFECT")
        else:
            print(">>> MISMATCH: ERROR")
            # Find divergence point
            min_len = min(len(seq_std), len(seq_diff))
            for k in range(min_len):
                if seq_std[k] != seq_diff[k]:
                    print(f"    Diverged at index {k}: Standard {seq_std[k]} vs Diff {seq_diff[k]}")
                    break

if __name__ == "__main__":
    main()
