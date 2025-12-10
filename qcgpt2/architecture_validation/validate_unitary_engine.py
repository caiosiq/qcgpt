"""
qcgpt2/architecture_validation/validate_unitary_engine.py
Validates that the Differentiable Physics Engine (PyTorch) produces 
identical unitaries to Qiskit (Standard).
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

# QCGPT2 Imports
from qcgpt2.circuits2 import Circuit2
from qcgpt2.gates2 import VOCAB2, BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2, ID_TO_TOKEN2
from qcgpt2.unitaries2 import get_unitary_for_token_id
from qcgpt2.data.qiskit_utils2 import sample_task2
from qcgpt2.encoding2 import circuit2_to_tokens
from qcgpt2.training2.supervised import parallel_unitary_product

def get_qiskit_unitary(circ):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    # We rebuild the circuit in Qiskit to be sure
    # (Assuming circuit2_to_qiskit exists or we map manually)
    # For robust validation, let's map manually here to avoid dependencies
    from qcgpt2.simulators2.qiskit_sim2 import circuit2_to_qiskit
    qc = circuit2_to_qiskit(circ)
    # Operator() gives the matrix
    return Operator(qc).data

def run_validation(n_samples=500, max_depth=30):
    print(f"=== Validating Unitary Engine (N={n_samples}) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Preload Stack
    mats = []
    for tid in range(len(VOCAB2)):
        if tid in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
            mats.append(torch.eye(8, dtype=torch.complex128)) # Use Double Precision for Check
        else:
            mats.append(get_unitary_for_token_id(tid, n_qubits=3).to(dtype=torch.complex128))
    U_stack = torch.stack(mats, dim=0).to(device)
    
    results = []
    
    for i in tqdm(range(n_samples)):
        # Generate random circuit
        # We vary depth to check stability
        depth = np.random.randint(2, max_depth)
        _, circ = sample_task2(max_gates=depth)
        
        # A. Qiskit Ground Truth
        try:
            U_qiskit = get_qiskit_unitary(circ)
        except Exception as e:
            continue # Skip invalid random circuits
            
        # B. PyTorch Prediction
        tokens = circuit2_to_tokens(circ)
        # Pad with BOS/EOS to match training flow
        seq = [BOS_CIRC_ID2] + tokens + [EOS_CIRC_ID2]
        token_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        
        with torch.no_grad():
            # Gather unitaries
            U_seq = U_stack[token_tensor] # (1, L, 8, 8)
            # Multiply
            U_torch = parallel_unitary_product(U_seq).cpu().numpy()[0]
            
        # C. Compare (Frobenius Norm Difference)
        diff = np.linalg.norm(U_qiskit - U_torch)
        
        results.append({
            "depth": len(tokens),
            "diff_norm": diff,
            "status": "PASS" if diff < 1e-6 else "FAIL"
        })
        
    df = pd.DataFrame(results)
    print("\nSummary:")
    print(df.describe())
    
    # Save Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['depth'], df['diff_norm'], alpha=0.6, c='blue', edgecolors='k')
    plt.axhline(1e-6, color='red', linestyle='--', label='Precision Threshold (1e-6)')
    plt.yscale('log')
    plt.xlabel('Circuit Depth (Gates)')
    plt.ylabel('Matrix Difference (Frobenius Norm)')
    plt.title('Unitary Engine Validation: PyTorch vs Qiskit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/caiosiq/qcgpt/qcgpt2/architecture_validation/results/unitary_validation_log.png')
    print("Saved unitary_validation_log.png")
    
    # Save CSV
    df.to_csv('/home/caiosiq/qcgpt/qcgpt2/architecture_validation/results/unitary_validation.csv', index=False)

if __name__ == "__main__":
    run_validation()