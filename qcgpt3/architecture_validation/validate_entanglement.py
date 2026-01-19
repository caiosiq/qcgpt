import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qcgpt3 import GateRegistry, TensorUnitaryBackend, QDPE, Circuit, Gate
from qcgpt3.simulators.qiskit_sim import QiskitEngine
from qcgpt3.encoding import CircuitEncoder
from qiskit.quantum_info import entropy, partial_trace, Statevector

def validate_entanglement():
    print("\n--- Test 4: Entanglement & Truth Table ---")
    registry = GateRegistry(n_qubits=3)
    backend = TensorUnitaryBackend(registry, device=torch.device('cpu'))
    # Ensure precision
    backend.dtype = torch.complex128
    backend._I = backend._I.to(torch.complex128)
    
    qdpe = QDPE(registry, backend, device=torch.device('cpu'))
    qiskit_engine = QiskitEngine(registry)
    
    # 1. Create GHZ Circuit: H(0), CX(0,1), CX(0,2)
    circ = Circuit(3)
    circ.add_gate(Gate("H", [0]))
    circ.add_gate(Gate("CX", [0, 1]))
    circ.add_gate(Gate("CX", [0, 2]))
    
    print("Circuit: GHZ State (H-CX-CX)")
    
    # 2. QDPE Execution
    encoder = CircuitEncoder(registry)
    tokens = encoder.encode(circ)
    tokens_t = torch.tensor(tokens, dtype=torch.long)
    
    with torch.no_grad():
        U_qdpe = qdpe.compute_unitary_from_tokens(tokens_t) # (D, D)
        U_qdpe = U_qdpe.numpy().astype(np.complex128)
        
    # Apply to |000> (First column of U)
    psi_qdpe = U_qdpe[:, 0]
    
    # 3. Qiskit Execution
    qc = qiskit_engine.circuit_to_qiskit(circ)
    psi_qiskit = Statevector.from_label('000').evolve(qc).data
    
    # 4. Compare Statevectors (up to global phase)
    # We check overlap
    overlap = np.abs(np.vdot(psi_qdpe, psi_qiskit))
    print(f"State Overlap: {overlap:.6f}")
    
    if overlap < 0.999:
        print("FAIL: Statevectors do not match.")
        sys.exit(1)
        
    # 5. Check Entropy (Entanglement)
    sv_qdpe = Statevector(psi_qdpe)
    sv_qiskit = Statevector(psi_qiskit)
    
    rho_qdpe = partial_trace(sv_qdpe, [1, 2])
    rho_qiskit = partial_trace(sv_qiskit, [1, 2])
    
    ent_qdpe = entropy(rho_qdpe)
    ent_qiskit = entropy(rho_qiskit)
    
    print(f"Entropy (QDPE):   {ent_qdpe:.4f}")
    print(f"Entropy (Qiskit): {ent_qiskit:.4f}")
    
    if abs(ent_qdpe - ent_qiskit) < 1e-4:
        print("PASS: Entanglement entropy matches.")
    else:
        print("FAIL: Entropy mismatch.")
        sys.exit(1)

    # 6. Visualization: Statevector Amplitudes
    # Plot Real and Imaginary parts of both statevectors to visually confirm match
    
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Align phase for visualization
    phase_diff = np.angle(np.vdot(psi_qdpe, psi_qiskit))
    psi_qdpe_aligned = psi_qdpe * np.exp(-1j * phase_diff)
    
    basis_labels = [f"|{i:03b}>" for i in range(8)]
    x = np.arange(8)
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Real Part
    ax1.bar(x - width/2, psi_qdpe_aligned.real, width, label='QDPE', alpha=0.7)
    ax1.bar(x + width/2, psi_qiskit.real, width, label='Qiskit', alpha=0.7)
    ax1.set_ylabel('Real Amplitude')
    ax1.set_title(f'GHZ State Reconstruction (Overlap={overlap:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Imaginary Part
    ax2.bar(x - width/2, psi_qdpe_aligned.imag, width, label='QDPE', alpha=0.7)
    ax2.bar(x + width/2, psi_qiskit.imag, width, label='Qiskit', alpha=0.7)
    ax2.set_ylabel('Imag Amplitude')
    ax2.set_xticks(x)
    ax2.set_xticklabels(basis_labels)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "entanglement_state_vis.png")
    plt.savefig(save_path)
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    validate_entanglement()
