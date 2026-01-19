
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from qcgpt3 import GateRegistry, TensorUnitaryBackend, QDPE, Circuit
from qcgpt3.data.dataset import HighPerformanceDataset
from qcgpt3.models.policy import CircuitPolicy
from qcgpt3.training.objectives import TrainingContext

def check_dataset_consistency():
    print("\n=== Checking Dataset Consistency (Augmentations) ===")
    registry = GateRegistry(n_qubits=3)
    # CPU backend for validation
    backend = TensorUnitaryBackend(registry, device='cpu')
    qdpe = QDPE(registry, backend, device='cpu')
    
    # Enable augmentations
    dataset = HighPerformanceDataset(
        registry=registry,
        qdpe=None, 
        num_samples=20,
        n_qubits=3,
        raw_max_depth=8,
        augment_commutation=True,
        augment_permutation=True
    )
    
    mismatches = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        spec_tensor = sample["spec_tensor"] # (dim, 2, dim, 2)
        circ_tokens = sample["circ_tokens"]
        
        # 1. Reconstruct Unitary from tokens (Target)
        try:
            U_tokens = qdpe.compute_unitary_from_tokens(circ_tokens)
        except Exception as e:
            print(f"Sample {i}: Failed to compute unitary from tokens. {e}")
            continue

        # 2. Reconstruct Unitary from spec_tensor (Input)
        # spec_tensor shape: (dim, 2, dim, 2) -> (dim, 2, dim) complex
        spec_c = torch.complex(spec_tensor[..., 0], spec_tensor[..., 1])
        inputs = spec_c[:, 0, :] # (dim, dim)
        outputs = spec_c[:, 1, :] # (dim, dim)
        
        # U = Outputs * Inputs.H
        # Note: Inputs should be orthonormal (basis states). 
        # If permutation happened, they are still orthonormal basis states, just shuffled.
        U_spec = outputs.T @ inputs.conj()
        
        # 3. Compare
        diff = torch.norm(U_tokens - U_spec)
        
        if diff > 1e-4:
            print(f"Sample {i} [MISMATCH]: Diff = {diff.item():.6f}")
            print(f"  Tokens: {circ_tokens.tolist()}")
            mismatches += 1
        else:
            print(f"Sample {i} [OK]: Diff = {diff.item():.6e}")
            
    if mismatches == 0:
        print(">> Dataset Consistency: PASS")
    else:
        print(f">> Dataset Consistency: FAIL ({mismatches} mismatches found)")

def check_model_gradients():
    print("\n=== Checking Model Gradients & Forward Pass ===")
    device = torch.device("cpu") # Keep it simple for debug
    registry = GateRegistry(n_qubits=3)
    model = CircuitPolicy(registry=registry).to(device)
    
    # Dummy Batch
    B = 2
    dim = 8
    # spec: [B, n_pairs, 2, dim, 2] -> flattened in collator usually
    # The model expects: spec_batch [B, n_pairs, 4*dim]
    spec_batch = torch.randn(B, dim, 4*dim).to(device)
    spec_pad_mask = torch.zeros(B, dim).bool().to(device)
    
    # Circuit inputs
    Lc = 10
    circ_in = torch.randint(0, 10, (B, Lc)).to(device)
    
    print("Running Forward Pass...")
    try:
        logits, physics_preds = model(spec_batch, spec_pad_mask, circ_in, return_physics=True)
        print(f"Logits Shape: {logits.shape}")
        print(f"Physics Preds Shape: {physics_preds.shape}")
        
        if torch.isnan(logits).any():
            print("!! NaN detected in logits !!")
        else:
            print("Logits are numeric (No NaNs).")
            
    except Exception as e:
        print(f"Forward Pass Failed: {e}")
        return

    print("Running Backward Pass (Gradient Check)...")
    try:
        loss = logits.sum()
        loss.backward()
        
        has_grads = True
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"Warning: No grad for {name}")
                has_grads = False
            elif torch.isnan(param.grad).any():
                print(f"!! NaN grad for {name} !!")
                has_grads = False
                
        if has_grads:
            print(">> Gradients: PASS")
        else:
            print(">> Gradients: ISSUES FOUND")
            
    except Exception as e:
        print(f"Backward Pass Failed: {e}")

if __name__ == "__main__":
    check_dataset_consistency()
    check_model_gradients()
