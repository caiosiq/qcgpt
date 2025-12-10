import json, os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
import torch
import torch.nn.functional as F
from qcgpt2.gates2 import PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2, VOCAB2
from qcgpt2.unitaries2 import get_unitary_for_token_id

# --- 1. Re-implement the suspected functions locally to verify ---

def naive_sequential_product(indices, U_stack):
    """The Slow, Trusted Way (Ground Truth)"""
    B, L = indices.shape
    results = []
    for b in range(B):
        U_accum = torch.eye(8, dtype=torch.complex128) # High precision
        for t in range(L):
            token_id = indices[b, t].item()
            
            # STOP LOGIC: mimicking what we expect the model to learn
            if token_id == EOS_CIRC_ID2 or token_id == PAD_ID2:
                # If we hit EOS, we stop processing (multiply by Identity)
                break
            
            # Multiply
            u_gate = U_stack[token_id].to(dtype=torch.complex128)
            U_accum = u_gate @ U_accum # Apply gate U_accum -> U_new @ U_accum
            
        results.append(U_accum)
    return torch.stack(results)

def parallel_unitary_product_debug(seq):
    """Your current implementation"""
    # ... (Copying your parallel logic) ...
    B, L, D, _ = seq.shape
    target_L = 1
    while target_L < L: target_L *= 2
    if target_L > L:
        padding = torch.eye(D, dtype=seq.dtype, device=seq.device).view(1, 1, D, D)
        padding = padding.expand(B, target_L - L, D, D)
        seq = torch.cat([seq, padding], dim=1)
    
    current_seq = seq
    while current_seq.shape[1] > 1:
        left = current_seq[:, 0::2]
        right = current_seq[:, 1::2]
        current_seq = right @ left
    return current_seq.squeeze(1)

# --- 2. The Test Harness ---

def run_debug_check(model_path, device_str="cuda"):
    device = torch.device(device_str)
    
    print("--- 1. Building Unitary Stack ---")
    mats = []
    for tid in range(len(VOCAB2)):
        if tid in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
            mats.append(torch.eye(8, dtype=torch.complex128)) # Force Double Precision
        else:
            mats.append(get_unitary_for_token_id(tid, n_qubits=3).to(dtype=torch.complex128))
    U_stack = torch.stack(mats, dim=0).to(device)
    
    print(f"Checking EOS Matrix (ID={EOS_CIRC_ID2})...")
    is_identity = torch.allclose(U_stack[EOS_CIRC_ID2], torch.eye(8, dtype=torch.complex128, device=device))
    print(f"Is EOS mapped to Identity? {is_identity}")
    if not is_identity:
        print("CRITICAL BUG FOUND: EOS is not Identity!")
        return

    # --- 2. Generate a Fake Batch ---
    # Create a sequence: H_0 (Valid) -> X_0 (Valid) -> EOS (Stop) -> Z_0 (Garbage after stop)
    # We want to ensure Z_0 is IGNORED.
    
    # Let's find IDs
    try:
        id_H = VOCAB2.index("H_0")
        id_X = VOCAB2.index("X_0")
        id_Z = VOCAB2.index("Z_0") # Garbage
    except:
        print("Could not find standard gates in vocab, using indices 1,2,3")
        id_H, id_X, id_Z = 10, 11, 12 # Fallback
        
    print(f"Testing Sequence: H_0 -> X_0 -> EOS -> Z_0 (Garbage)")
    
    # Create Batch of size 2
    # Seq 1: H -> X -> EOS -> Z (Should be XH)
    # Seq 2: H -> H -> EOS -> EOS (Should be I)
    input_ids = torch.tensor([
        [id_H, id_X, EOS_CIRC_ID2, id_Z],
        [id_H, id_H, EOS_CIRC_ID2, EOS_CIRC_ID2]
    ], device=device)
    
    B, L = input_ids.shape
    
    # --- 3. Run Ground Truth (Naive Loop) ---
    print("\n--- Running Ground Truth (Naive Loop) ---")
    U_true = naive_sequential_product(input_ids, U_stack)
    
    # --- 4. Run Differentiable Logic (Hard Mode) ---
    print("--- Running Differentiable Logic (Hard Mode) ---")
    
    # Create One-Hot (Simulating Hard Argmax)
    one_hot = F.one_hot(input_ids, num_classes=len(VOCAB2)).float()
    
    # Apply Life Mask Logic (Copy-pasting your function logic)
    probs = one_hot # In hard mode, probs IS one_hot
    
    p_eos = probs[:, :, EOS_CIRC_ID2]
    p_continue = 1.0 - p_eos
    
    # Mask Logic check
    life_mask = torch.cumprod(p_continue, dim=1)
    life_mask = torch.roll(life_mask, shifts=1, dims=1)
    life_mask[:, 0] = 1.0
    
    print(f"Life Mask generated:\n{life_mask.cpu().numpy()}")
    # Expectation for Seq 1: [1, 1, 1, 0] 
    # (H is alive, X is alive, EOS is alive/Identity, Z is dead/Identity)
    
    # Matmul Logic
    U_seq = torch.einsum("blv,vij->blij", probs.to(dtype=torch.complex128), U_stack)
    I = torch.eye(8, dtype=torch.complex128, device=device).view(1, 1, 8, 8)
    
    life_mask_U = life_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=torch.complex128)
    U_effective = life_mask_U * U_seq + (1.0 - life_mask_U) * I
    
    # Parallel Product
    U_diff = parallel_unitary_product_debug(U_effective)
    
    # --- 5. Compare ---
    diff = (U_true - U_diff).abs().sum()
    print(f"\nDifference betweeen Naive Loop and Differentiable Logic: {diff.item():.8f}")
    
    if diff.item() > 1e-5:
        print("FAIL: The differentiable logic does not match the naive loop.")
        print("Possible causes: Parallel product order, or Masking logic.")
    else:
        print("PASS: The logic is mathematically consistent.")
        
    # --- 6. Compare Parallel Product vs Sequential Product (Math Only) ---
    print("\n--- Checking Pure Matrix Multiplication Order ---")
    # Take raw sequence without masking
    U_raw_seq = U_stack[input_ids]
    U_raw_parallel = parallel_unitary_product_debug(U_raw_seq)
    
    # Sequential equivalent (No stopping, just multiply all)
    U_raw_naive = []
    for b in range(B):
        acc = torch.eye(8, dtype=torch.complex128, device=device)
        for t in range(L):
            acc = U_stack[input_ids[b,t]] @ acc
        U_raw_naive.append(acc)
    U_raw_naive = torch.stack(U_raw_naive)
    
    diff_raw = (U_raw_parallel - U_raw_naive).abs().sum()
    print(f"Raw Matrix Product Difference: {diff_raw.item():.8f}")

if __name__ == "__main__":
    run_debug_check(None, device_str="cuda" if torch.cuda.is_available() else "cpu")