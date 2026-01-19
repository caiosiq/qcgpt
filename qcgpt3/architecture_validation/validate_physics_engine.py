import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from qiskit.quantum_info import Operator

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qcgpt3 import GateRegistry, TensorUnitaryBackend, QDPE
from qcgpt3.simulators.qiskit_sim import QiskitEngine
from qcgpt3.encoding import CircuitEncoder

def fix_global_phase(U_pred, U_ref):
    """
    Adjusts U_pred by a global phase to match U_ref if they are physically identical.
    Finds scalar alpha such that U_pred * alpha approx U_ref.
    """
    u_pred_flat = U_pred.flatten()
    u_ref_flat = U_ref.flatten()
    
    idx = np.argmax(np.abs(u_ref_flat) > 1e-5)
    if np.abs(u_ref_flat[idx]) < 1e-5: return U_pred

    val_ref = u_ref_flat[idx]
    val_pred = u_pred_flat[idx]
    
    if np.abs(val_pred) < 1e-5: return U_pred
    
    phase_diff = val_ref / val_pred
    phase_diff /= np.abs(phase_diff)
    
    return U_pred * phase_diff

def validate_hard_consistency():
    print("\n--- Test 1.A: Hard Consistency (Matrix Test) ---")
    registry = GateRegistry(n_qubits=3)
    backend = TensorUnitaryBackend(registry, device=torch.device('cpu'))
    
    # Ensure backend uses complex128 for validation precision
    backend.dtype = torch.complex128 
    backend._I = backend._I.to(torch.complex128)
    
    qdpe = QDPE(registry, backend, device=torch.device('cpu'))
    qiskit_engine = QiskitEngine(registry)
    encoder = CircuitEncoder(registry)

    # Visualization Data
    depths = range(2, 21, 2)
    diffs = []
    
    print("Running depth sweep for Hard Consistency...")
    
    # Track "safe" gates
    safe_gates = set()
    failed_circuits = []
    
    for depth in depths:
        # Generate random circuit
        seed = 42 + depth
        rng = np.random.RandomState(seed)
        
        qc_qiskit = qiskit_engine.sample_random_circuit(n_qubits=3, max_depth=depth, rng=rng)
        circ = qiskit_engine.qiskit_to_circuit(qc_qiskit)
        
        # 1. QDPE
        tokens = encoder.encode(circ)
        tokens_t = torch.tensor(tokens, dtype=torch.long)
        
        with torch.no_grad():
            U_qdpe = qdpe.compute_unitary_from_tokens(tokens_t)
            U_qdpe_np = U_qdpe.numpy().astype(np.complex128)

        # 2. Qiskit
        U_qiskit = Operator(qc_qiskit).data.astype(np.complex128)

        # 3. Compare
        U_qdpe_aligned = fix_global_phase(U_qdpe_np, U_qiskit)
        diff = np.linalg.norm(U_qdpe_aligned - U_qiskit)
        diffs.append(diff)
        
        if diff > 1e-5:
            print(f"DEBUG: High diff at depth {depth}: {diff:.2e}")
            print("Circuit Gates:", circ.gates)
            print("Qiskit Ops:", [op.name for op in qc_qiskit.data])
            failed_circuits.append(circ)
        else:
            # If passed, add gates to safe set
            for g in circ.gates:
                safe_gates.add(g.gate_type)
                
    # After loop, analyze
    print("\n--- Analysis ---")
    print(f"Safe Gates (appeared in passing circuits): {sorted(list(safe_gates))}")
    
    all_failed_gates = set()
    for fc in failed_circuits:
        for g in fc.gates:
            all_failed_gates.add(g.gate_type)
            
    # Potential culprits: Gates in failed circuits that are NOT in safe set (or rarely in safe set)
    # But strictly: Gates that appear in failed circuits but NEVER in safe circuits?
    # Or maybe it's a combination.
    
    suspicious = all_failed_gates - safe_gates
    if suspicious:
        print(f"SUSPICIOUS GATES (Only in failed circuits): {suspicious}")
    else:
        print("No single gate is exclusively in failed circuits. Likely an interaction or phase issue.")
        
    # Plotting
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(depths, diffs, 'o-', label='Hard Consistency (QDPE vs Qiskit)')
    plt.yscale('log')
    plt.xlabel('Circuit Depth')
    plt.ylabel('Frobenius Norm Difference')
    plt.title('Physics Engine Precision vs Depth')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "hard_consistency_depth.png"))
    print(f"Hard consistency plot saved.")

    if max(diffs) < 1e-5:
        print("PASS: Hard Physics matches Qiskit across depths.")
    else:
        print(f"FAIL: Max diff {max(diffs):.2e} exceeds threshold.")
        sys.exit(1)

def validate_soft_consistency():
    print("\n--- Test 1.B: Soft Consistency (Interpolation Test) ---")
    registry = GateRegistry(n_qubits=3)
    backend = TensorUnitaryBackend(registry, device=torch.device('cpu'))
    # Ensure complex128
    backend.dtype = torch.complex128
    backend._I = backend._I.to(torch.complex128)
    
    qdpe = QDPE(registry, backend, device=torch.device('cpu'))
    
    try:
        id_x = registry.vocab.index("X_0")
        id_z = registry.vocab.index("Z_0")
    except ValueError:
        print("Skipping Soft Test: X_0 or Z_0 not in vocab.")
        return

    # Sweep interpolation alpha
    alphas = np.linspace(0, 1, 20)
    diffs = []
    
    U_stack = qdpe.u_stack.to(torch.complex128)
    U_x = backend.get_unitary_for_token_id(id_x)
    U_z = backend.get_unitary_for_token_id(id_z)
    
    vocab_size = len(registry.vocab)
    
    print("Running interpolation sweep...")
    
    for alpha in alphas:
        # Create Soft Probability Vector: alpha * X + (1-alpha) * Z
        probs = torch.zeros((1, vocab_size), dtype=torch.float64) 
        probs[0, id_x] = alpha
        probs[0, id_z] = 1.0 - alpha
        
        # 1. QDPE Interpolation (via compute_unitary)
        # We need to manually call the contraction logic or use a helper that accepts soft probs.
        # QDPE.compute_unitary is designed for this.
        # Note: compute_unitary expects (B, L, V)
        probs_seq = probs.unsqueeze(1) # (1, 1, V)
        
        # We need to ensure QDPE uses complex128 stack for this test if backend was updated
        qdpe.u_stack = U_stack 
        
        with torch.no_grad():
            U_soft_engine = qdpe.compute_unitary(probs_seq, method="product")[0]
        
        # 2. Manual Calculation
        U_manual = alpha * U_x + (1.0 - alpha) * U_z
        
        diff = torch.norm(U_soft_engine - U_manual).item()
        diffs.append(diff)
        
    # Plotting
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    
    plt.figure(figsize=(10, 5))
    plt.plot(alphas, diffs, 'x-', color='orange', label='Soft Consistency (QDPE vs Linear Mix)')
    plt.yscale('log')
    plt.xlabel('Interpolation Alpha (X vs Z)')
    plt.ylabel('Frobenius Norm Difference')
    plt.title('Soft Gate Interpolation Accuracy')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "soft_consistency_sweep.png"))
    print(f"Soft consistency plot saved.")
    
    if max(diffs) < 1e-6:
        print("PASS: Soft Physics interpolates correctly.")
    else:
        print(f"FAIL: Soft Physics mismatch max diff {max(diffs):.2e}")
        sys.exit(1)

if __name__ == "__main__":
    validate_hard_consistency()
    validate_soft_consistency()
