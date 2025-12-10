import torch

def test_fidelity_math():
    # Example: S Gate (Phase Gate)
    # [1, 0]
    # [0, i]
    U_tgt = torch.tensor([[[1.0, 0.0], [0.0, 1.0j]]], dtype=torch.complex64)
    
    # Perfect Prediction
    U_pred = U_tgt.clone()
    
    print("--- Math Debug ---")
    print(f"Target (S Gate):\n{U_tgt}")
    
    # 1. Your Current Code (BROKEN)
    # Tr(U* @ U) -> Tr([[1, 0], [0, -i]] @ [[1, 0], [0, i]])
    #            -> Tr([[1, 0], [0, 1]]) -> 2.
    # Wait, for Diagonal matrices, Transpose doesn't matter.
    # We need a non-diagonal, complex matrix. Y Gate.
    
    # Y Gate: [[0, -i], [i, 0]]
    U_tgt = torch.tensor([[[0.0, -1.0j], [1.0j, 0.0]]], dtype=torch.complex64)
    U_pred = U_tgt.clone()
    print(f"\nTarget (Y Gate):\n{U_tgt}")
    
    # Current Implementation: einsum("bij,bji", conj, pred) -> Tr(U* U)
    # U* = [[0, i], [-i, 0]]
    # U  = [[0, -i], [i, 0]]
    # U* @ U = [[(i)(i), 0], [0, (-i)(-i)]] = [[-1, 0], [0, -1]]
    # Trace = -2. |Trace|^2 = 4. Scaled by 1/d^2 (1/4) = 1.0.
    
    # Wait... Y is also symmetric (mostly).
    # We need a NON-SYMMETRIC matrix to fail "bij,bji" vs "bij,bij".
    # Rx(theta)? Or just a random unitary.
    
    U_tgt = torch.randn(1, 2, 2, dtype=torch.complex64)
    # Orthonormalize to make it unitary-ish (optional, but cleaner)
    U_tgt = torch.linalg.qr(U_tgt)[0]
    U_pred = U_tgt.clone()
    
    print("\nRandom Unitary Test:")
    
    # METHOD A (Yours): Tr(U* U)
    trace_wrong = torch.einsum("bij,bji->b", U_tgt.conj(), U_pred)
    fid_wrong = (trace_wrong.abs() ** 2) / 4.0
    print(f"Your Method (bij,bji): Fidelity = {fid_wrong.item():.4f} (Should be 1.0)")
    
    # METHOD B (Correct): Tr(U_dag U)
    # Use standard matrix multiplication to prove ground truth
    U_dag = U_tgt.conj().transpose(-2, -1)
    prod = U_dag @ U_pred
    trace_truth = prod.diagonal(dim1=-2, dim2=-1).sum(-1)
    fid_truth = (trace_truth.abs() ** 2) / 4.0
    print(f"Ground Truth (Matmul): Fidelity = {fid_truth.item():.4f}")
    
    # METHOD C (Proposed Fix): Element-wise Sum
    trace_fix = torch.einsum("bij,bij->b", U_tgt.conj(), U_pred)
    fid_fix = (trace_fix.abs() ** 2) / 4.0
    print(f"Proposed Fix (bij,bij): Fidelity = {fid_fix.item():.4f}")

if __name__ == "__main__":
    test_fidelity_math()