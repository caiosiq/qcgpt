import os
"""QCGPT2 Unitary Debug Script

Provides debugging utilities for differentiable (Gumbel-Softmax) circuit
generation and soft/teacher-forcing unitary reconstruction to analyze
training-time vs real-world fidelities.
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

# Imports from your library
from qcgpt2.gates2 import BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2, VOCAB2
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.data.qiskit_utils2 import sample_task2 as sample_task
from qcgpt2.data.specs2 import build_spec_sequence_batch
from qcgpt2.encoding2 import tokens_to_circuit2, ID_TO_TOKEN2, circuit2_to_tokens
from qcgpt2.circuits2 import Circuit2, Gate2
from qcgpt2.unitaries2 import build_circuit_unitary2, get_unitary_for_token_id

def generate_differentiable_logits(model, spec_batch, spec_pad_mask, 
                                   bos_id, eos_id, max_len=32, temp=0.5):
    """
    Generates a circuit autoregressively using Gumbel-Softmax (Straight-Through).
    Returns: 'soft_probs' sequence (B, L, Vocab) that has gradients attached.
    """
    B = spec_batch.size(0)
    device = spec_batch.device
    
    # 1. Encode Truth Table (Standard)
    # We do this once.
    memory = model.encoder(spec_batch, spec_pad_mask)
    
    # 2. Setup Embedding Access
    # We need the raw weights to do the differentiable lookup: Soft_OneHot @ Matrix
    # Assumes model.decoder has an attribute 'token_embedding' which is nn.Embedding
    w_emb = model.decoder.token_embedding.weight # (Vocab, D_model)
    
    # 3. Start with <BOS>
    # Initial input is just the BOS embedding
    curr_input = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    curr_embeds = model.decoder.token_embedding(curr_input) # (B, 1, D)
    
    history_embeds = curr_embeds
    soft_probs_list = []
    
    # 4. The Autoregressive Loop (O(L))
    for t in range(max_len):
        # A. Forward Pass using Embeddings
        # Uses your new 'forward_embeds' function
        # We pass the WHOLE history so the Transformer can attend to it
        logits = model.decoder.forward_embeds(history_embeds, memory, memory_key_padding_mask=spec_pad_mask)
        
        # B. Get prediction for the *next* token (the last one in the sequence)
        next_token_logits = logits[:, -1, :] # (B, Vocab)
        
        # C. Gumbel-Softmax (The Magic Step)
        # hard=True: Forward pass sees a perfect One-Hot vector (Discrete).
        #            Backward pass sees the Softmax probability gradients.
        next_token_onehot = F.gumbel_softmax(next_token_logits, tau=temp, hard=True, dim=-1)
        
        # Save for Unitary Calculation
        soft_probs_list.append(next_token_onehot)
        
        # D. Prepare Input for Next Step
        # Differentiable Embedding Lookup: OneHot @ Weights
        # (B, V) @ (V, D) -> (B, D) -> (B, 1, D)
        next_embed = torch.matmul(next_token_onehot, w_emb).unsqueeze(1)
        
        # Append to history
        history_embeds = torch.cat([history_embeds, next_embed], dim=1)
        
        # Optimization: We could break early if all batches hit EOS, 
        # but for batched diff-prog, fixed length is often numerically more stable.
        
    # 5. Stack into Sequence
    # Shape: (B, MaxLen, Vocab) - This looks exactly like "One-Hot Logits"
    full_probs = torch.stack(soft_probs_list, dim=1)
    
    return full_probs
# --- Local Helper for Unitary Product ---
def parallel_unitary_product(U_seq):
    """
    Computes the product of a sequence of matrices: U_n @ ... @ U_2 @ U_1
    U_seq shape: [B, L, 8, 8]
    Returns: [B, 8, 8]
    """
    B, L, D, _ = U_seq.shape
    curr = torch.eye(D, dtype=U_seq.dtype, device=U_seq.device).unsqueeze(0).repeat(B, 1, 1)
    
    # We multiply in sequence order. 
    # Usually quantum circuits are applied |psi> = U_n ... U_1 |0>
    # So new state = U_t @ current_state_matrix
    for t in range(L):
        u_t = U_seq[:, t, :, :]
        curr = torch.matmul(u_t, curr)
        
    return curr

# --- Helper to turn Reference Circuit into Tensor Batch ---
def ref_to_tensor(ref_circ, device):
    c2 = Circuit2(nqubits=ref_circ.nqubits)
    for g in ref_circ.gates:
        c2.add_gate(Gate2(g.gate_type, g.targets))
    toks = circuit2_to_tokens(c2)
    return torch.tensor([toks], dtype=torch.long, device=device)

def load_model(ckpt: str, device: torch.device) -> CircuitPolicy2:
    model = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    if ckpt and os.path.exists(ckpt):
        print(f"Loading checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device, weights_only=True)
        sd = state.get("model_state_dict", state)
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model

# --- DEBUG FUNCTION: ENFORCED T=0 ---
def debug_calculate_physical_fidelity_components(logits, U_stack, device, verbose=False):
    """
    A T=0 hard version of the fidelity calculation that logs every step.
    """
    B, Lc, V = logits.shape
    
    # 1. Enforce Hard Argmax (T=0)
    # No softmax, just take the max logit
    ids_hard = logits.argmax(dim=-1) # [B, Lc]
    
    if verbose:
        print(f"\n[DEBUG FUNC] Sequence Length: {Lc}")
        print(f"[DEBUG FUNC] IDs selected by Argmax: {ids_hard[0].tolist()}")
        readable = [ID_TO_TOKEN2.get(t, f"ID_{t}") for t in ids_hard[0].tolist()]
        print(f"[DEBUG FUNC] Tokens: {readable}")

    # 2. Build Sequence of Unitaries manually to debug
    # We will replicate the vectorization but print details
    
    batch_U_accum = torch.eye(8, dtype=torch.complex64, device=device).unsqueeze(0) # [1, 8, 8]
    alive = True
    
    # We iterate purely for logging purposes, though we could vectorize
    # Using 0-th batch element for logs
    
    final_mats = []
    
    for t in range(Lc):
        token_id = ids_hard[0, t].item()
        token_name = ID_TO_TOKEN2.get(token_id, "UNK")
        
        # Check EOS
        if token_id == EOS_CIRC_ID2:
            alive = False
            if verbose: print(f"  Step {t}: <EOS> encountered. Stopping updates.")
        
        # Get Matrix from Stack
        # U_stack is [V, 8, 8]
        current_u = U_stack[token_id] 
        
        # Logic Check: If not alive, current_u effectively becomes Identity
        effective_u = current_u if alive else torch.eye(8, dtype=torch.complex64, device=device)
        
        # Accumulate
        prev_accum = batch_U_accum.clone()
        batch_U_accum = torch.matmul(effective_u, batch_U_accum)
        
        if verbose and alive:
            # Print stats about the added matrix
            is_identity = torch.allclose(current_u, torch.eye(8, device=device, dtype=current_u.dtype))
            print(f"  Step {t}: Token={token_name} (ID={token_id}) | IsIdentity={is_identity}")
            
            # OPTIONAL: Check against build_circuit_unitary2 to ensure U_stack is correct
            # We create a dummy circuit with JUST this gate
            if token_id not in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
                try:
                    dummy_circ = tokens_to_circuit2([BOS_CIRC_ID2, token_id, EOS_CIRC_ID2])
                    dummy_U = build_circuit_unitary2(dummy_circ, n_qubits=3).to(device)
                    # dummy_U is the product, so it should match current_u
                    diff = torch.norm(dummy_U - current_u)
                    if diff > 1e-5:
                        print(f"    WARNING: U_stack mismatch for {token_name}! Diff: {diff.item()}")
                except Exception as e:
                    print(f"    Could not verify {token_name}: {e}")

    return batch_U_accum, ids_hard

# --- 1. Teacher Forcing Soft Unitary (Wrapper) ---
def soft_unitary_teacher_forcing_debug(model: CircuitPolicy2, device: torch.device, 
                                 spec_tensor: np.ndarray, ref_tokens: torch.Tensor,
                                 verbose: bool = False):
    
    spec_batch_np, spec_pad_mask_np = build_spec_sequence_batch([spec_tensor])
    spec_batch = torch.tensor(spec_batch_np, dtype=torch.float32, device=device)
    spec_pad_mask = torch.tensor(spec_pad_mask_np, dtype=torch.bool, device=device)
    
    # Input to Transformer: Ref[:-1]
    circ_in = ref_tokens[:, :-1] 
    
    # Forward Pass
    logits = model(spec_batch, spec_pad_mask, circ_in) # [1, L-1, V]
    
    # Prepare Unitary Stack
    mats = [] 
    for tid in range(len(VOCAB2)):
        if tid in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
            mats.append(torch.eye(8, dtype=torch.complex64))
        else:
            mats.append(get_unitary_for_token_id(tid, n_qubits=3).to(dtype=torch.complex64))
    U_stack = torch.stack(mats, dim=0).to(device)
    
    # Call the DEBUG function
    U_pred, selected_ids = debug_calculate_physical_fidelity_components(
        logits, U_stack, device, verbose=verbose
    )
    
    return U_pred.squeeze(0), selected_ids

# --- 2. Hard Reconstruction ---
@torch.no_grad()
def reconstruct_tokens_hard(model, device, spec_tensor, max_len):
    spec_batch_np, spec_pad_mask_np = build_spec_sequence_batch([spec_tensor])
    spec_batch = torch.tensor(spec_batch_np, dtype=torch.float32, device=device)
    spec_pad_mask = torch.tensor(spec_pad_mask_np, dtype=torch.bool, device=device)
    tokens, _ = model.sample_circuit_tokens(spec_batch, spec_pad_mask, BOS_CIRC_ID2, EOS_CIRC_ID2, max_len)
    seq = [t for t in tokens[0].tolist() if t != PAD_ID2]
    return seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(edgeitems=4, precision=3, linewidth=150)
    
    model = load_model(args.ckpt, device=device)
    
    print(f"--- Debugging {args.ckpt} with T=0 Enforcement ---")
    
    for i in range(args.num_samples):
        spec_tensor, ref_circ = sample_task(max_gates=6)
        
        # A. Ground Truth
        U_ref = build_circuit_unitary2(ref_circ, n_qubits=3).to(device)
        
        # B. Hard Generation (Autoregressive)
        token_ids_hard_gen = reconstruct_tokens_hard(model, device, spec_tensor, max_len=32)
        cand_circ = tokens_to_circuit2(token_ids_hard_gen)
        U_hard_gen = build_circuit_unitary2(cand_circ, n_qubits=3).to(device)
        
        # C. Teacher Forcing "Hard" Debug (Using T=0 logic inside the differentiable flow)
        ref_tokens = ref_to_tensor(ref_circ, device)
        verbose = (i % 5 == 0)
        
        if verbose:
            print(f"\n{'='*20} SAMPLE {i} {'='*20}")
            print(f"Reference: {[ID_TO_TOKEN2[t] for t in ref_tokens[0].tolist()]}")
        
        U_tf_debug, token_ids_tf = soft_unitary_teacher_forcing_debug(
            model, device, spec_tensor, ref_tokens, verbose=verbose
        )
        
        # --- COMPARISONS ---
        
        # 1. Fidelity Checks
        fid_hard_gen = (torch.einsum("ij,ji->", U_ref.conj().T, U_hard_gen).abs()**2 / 64.0).item()
        fid_tf_debug = (torch.einsum("ij,ji->", U_ref.conj().T, U_tf_debug).abs()**2 / 64.0).item()
        
        # 2. Matrix Equality Check (TF Debug vs Hard Gen)
        # Note: These are likely different because TF sees ground truth, Hard Gen sees its own history.
        # But if the model is perfect, they should match.
        mat_diff = torch.norm(U_tf_debug - U_hard_gen).item()
        
        if verbose:
            print("-" * 30)
            print(f"Tokens (Hard Gen):      {[ID_TO_TOKEN2.get(t,'UNK') for t in token_ids_hard_gen]}")
            print(f"Tokens (TF Debug T=0):  {[ID_TO_TOKEN2.get(t.item(),'UNK') for t in token_ids_tf[0]]}")
            print("-" * 30)
            print(f"Fidelity (Hard Gen):    {fid_hard_gen:.4f}")
            print(f"Fidelity (TF Debug):    {fid_tf_debug:.4f}")
            print(f"Matrix Diff (TF vs Gen):{mat_diff:.4f}")
            
            # 3. Check if TF Debug actually outputs a valid unitary
            # U @ U_dagger should be Identity
            is_unitary = torch.allclose(
                U_tf_debug @ U_tf_debug.conj().T, 
                torch.eye(8, dtype=torch.complex64, device=device), 
                atol=1e-4
            )
            print(f"TF Debug Output is Unitary? {is_unitary}")
            
            if not is_unitary:
                print("!!! CRITICAL: The debug construction function is not producing a unitary matrix.")

if __name__ == "__main__":
    main()
