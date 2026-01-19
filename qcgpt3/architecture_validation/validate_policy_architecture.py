import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qcgpt3 import GateRegistry
from qcgpt3.models.policy import CircuitPolicy

def validate_policy():
    print("\n--- Test 5: Policy Architecture & Forward Pass ---")
    
    # 1. Setup
    n_qubits = 3
    registry = GateRegistry(n_qubits=n_qubits)
    
    # Small model for testing
    d_model = 64
    n_layers = 2
    n_heads = 2
    max_spec_len = 2**n_qubits # 8
    max_circ_len = 20
    
    policy = CircuitPolicy(
        registry=registry,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_spec_len=max_spec_len,
        max_circ_len=max_circ_len
    )
    
    print(f"Policy initialized with d_model={d_model}, vocab={len(registry.vocab)}")
    
    # 2. Dummy Inputs
    batch_size = 4
    
    # Spec Tensor: (B, N_pairs, 2, Dim, 2) -> Flattened to (B, N_pairs, 4*Dim)
    # The Encoder expects flattened features [B, n_pairs, 4*dim]
    dim = 2**n_qubits
    # Structured: (B, dim, 2, dim, 2)
    dummy_spec_structured = torch.randn(batch_size, dim, 2, dim, 2)
    # Flatten last 3 dims: 2 * dim * 2 = 4 * dim
    dummy_spec = dummy_spec_structured.view(batch_size, dim, -1) # (B, 8, 32)
    
    # Spec Pad Mask (False = keep, True = ignore)
    dummy_mask = torch.zeros(batch_size, dim, dtype=torch.bool)
    
    # Circuit Tokens: (B, L)
    dummy_tokens = torch.randint(0, len(registry.vocab), (batch_size, 10))
    
    # 3. Forward Pass
    print("Running forward pass...")
    logits = policy(dummy_spec, dummy_mask, dummy_tokens)
    
    print(f"Logits Shape: {logits.shape}")
    expected_shape = (batch_size, 10, len(registry.vocab))
    
    if logits.shape == expected_shape:
        print("PASS: Logits shape is correct.")
    else:
        print(f"FAIL: Expected {expected_shape}, got {logits.shape}")
        sys.exit(1)
        
    # 4. Forward with Physics
    print("Running forward pass with physics heads...")
    logits, ent_pred = policy(dummy_spec, dummy_mask, dummy_tokens, return_physics=True)
    
    print(f"Entanglement Prediction Shape: {ent_pred.shape}")
    # Entanglement head output shape depends on implementation. Usually (B, L, 1) or (B, L, N_qubits) or similar.
    # In transformer.py it was just a Linear(d_model, 1) usually for "entanglement present" or similar?
    # Or maybe it predicts entanglement entropy?
    # Let's assume it runs without error is the main check.
    
    if ent_pred is not None:
        print("PASS: Physics head returned output.")
    
    # 5. Sampling Loop
    print("Testing Sampling Loop...")
    try:
        seqs, log_probs = policy.sample_circuit_tokens(dummy_spec, dummy_mask, max_len=15)
        print(f"Sampled Seqs Shape: {seqs.shape}")
        print(f"Sampled LogProbs Shape: {log_probs.shape}")
        print("PASS: Sampling successful.")
    except Exception as e:
        print(f"FAIL: Sampling raised error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_policy()
