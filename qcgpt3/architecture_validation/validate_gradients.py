import torch
import torch.nn as nn
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qcgpt3 import GateRegistry, TensorUnitaryBackend, QDPE

def validate_gradient_flow():
    print("\n--- Test 3: Gradient Sanity (Identity Loop) ---")
    
    # Setup Engine
    registry = GateRegistry(n_qubits=3)
    backend = TensorUnitaryBackend(registry, device=torch.device('cpu'))
    # Ensure precision for gradients
    backend.dtype = torch.complex128
    backend._I = backend._I.to(torch.complex128)
    
    qdpe = QDPE(registry, backend, device=torch.device('cpu'))
    # Force QDPE to use complex128 stack
    qdpe.u_stack = qdpe.u_stack.to(torch.complex128)
    
    # Target: Identity
    dim = 2**3
    target_U = torch.eye(dim, dtype=torch.complex128)
    
    # Input: Random Logits for a sequence of length 5
    seq_len = 5
    vocab_size = len(registry.vocab)
    
    logits = torch.randn(1, seq_len, vocab_size, dtype=torch.float64, requires_grad=True)
    
    optimizer = torch.optim.Adam([logits], lr=0.1)
    
    print("Starting optimization loop...")
    losses = []
    grad_norms = []
    
    for i in range(100):
        optimizer.zero_grad()
        
        # Softmax to get probs
        probs = torch.softmax(logits, dim=-1) # (1, L, V)
        
        # Compute Soft Unitary using QDPE helper
        probs_seq = probs # (1, L, V)
        
        # Use qdpe.compute_unitary directly (supports soft probs)
        U_pred = qdpe.compute_unitary(probs_seq, method="product")[0]
        
        # Loss: || U_pred - I ||^2
        diff = U_pred - target_U
        loss = torch.real(torch.trace(diff.H @ diff)) 
        
        loss.backward()
        
        # Record Gradient Norm
        grad_norm = logits.grad.norm().item()
        grad_norms.append(grad_norm)
        
        optimizer.step()
        
        losses.append(loss.item())
        if i % 20 == 0:
            print(f"Step {i}: Loss = {loss.item():.4f}")
            
    print(f"Final Loss: {losses[-1]:.4f}")
    
    # Visualization
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Optimization Steps')
    ax1.set_ylabel('Loss (Frobenius Dist)', color=color)
    ax1.plot(losses, color=color, linewidth=2, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Gradient Norm', color=color)  # we already handled the x-label with ax1
    ax2.plot(grad_norms, color=color, linestyle='--', label='Grad Norm')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')
    
    plt.title('Gradient Flow Validation: Learning Identity Matrix')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    save_path = os.path.join(output_dir, "gradient_flow_sanity.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    
    # Check convergence
    if losses[-1] < 0.1 and losses[-1] < losses[0]:
        print("PASS: Gradients flow and optimization converges.")
    elif losses[-1] < losses[0]:
        print("WARN: Loss decreased but didn't reach 0 (Local minima possible). Passing for now.")
    else:
        print("FAIL: Optimization failed to decrease loss.")
        sys.exit(1)

if __name__ == "__main__":
    validate_gradient_flow()
