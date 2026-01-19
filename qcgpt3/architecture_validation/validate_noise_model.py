import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from qcgpt3 import GateRegistry, TensorUnitaryBackend, QDPE
from qcgpt3.simulators.qiskit_sim import QiskitEngine
from qcgpt3.encoding import CircuitEncoder

def validate_noise_correlation():
    print("\n--- Test 2: Noise Model (Fidelity Proxy) ---")
    
    # 1. Setup Architecture
    registry = GateRegistry(n_qubits=3)
    qiskit_engine = QiskitEngine(registry)
    backend = TensorUnitaryBackend(registry, device=torch.device('cpu'))
    qdpe = QDPE(registry, backend, device=torch.device('cpu'))
    encoder = CircuitEncoder(registry)

    # 2. Define Cost Tensor (Simulating "Learned" Costs)
    # 1q = 0.001, 2q = 0.01
    cost_tensor = torch.zeros(len(registry.vocab))
    for i, name in enumerate(registry.vocab):
        if "CX" in name or "SWAP" in name or "CZ" in name:
            cost_tensor[i] = 0.01
        elif "ID" in name:
            cost_tensor[i] = 0.0
        else:
            cost_tensor[i] = 0.001
            
    # Inject cost tensor into QDPE manually for this test
    qdpe.cost_tensor = cost_tensor.to(qdpe.device)
            
    # 3. Generate Circuits
    num_circuits = 50 
    pred_costs = []
    real_infidelities = []
    
    rng = np.random.RandomState(42)
    
    print(f"Simulating {num_circuits} circuits with variable depths...")
    
    # Generate circuits with depths ranging from 5 to 50 to get a good spread
    depths = np.linspace(5, 50, num_circuits, dtype=int)
    
    for depth in depths:
        # Generate
        qc_raw = qiskit_engine.sample_random_circuit(n_qubits=3, max_depth=int(depth), rng=rng)
        
        # 1. Convert to Circuit first (required for simplify_circuit now)
        try:
             circ_raw = qiskit_engine.qiskit_to_circuit(qc_raw)
        except:
             continue
             
        # 2. Simplify
        qc_simp = qiskit_engine.simplify_circuit(circ_raw)
        
        # 3. Use result
        circ = qc_simp # simplify_circuit returns Circuit object
        
        if len(circ.gates) == 0: continue
        
        # A. Predict Cost (QDPE Proxy)
        tokens = encoder.encode(circ)
        tokens_t = torch.tensor(tokens, dtype=torch.long)
        
        # Hard tokens -> One-hot probs
        probs = torch.nn.functional.one_hot(tokens_t, num_classes=len(registry.vocab)).float().unsqueeze(0) # (1, L, V)
        
        # Use QDPE compute_noise
        with torch.no_grad():
            gate_noise = qdpe.compute_noise(probs)
            
        # QDPE returns 1 - exp(-total_cost). This is the "fidelity loss".
        # If we want to check correlation with real infidelity (1 - F), this is the correct metric.
        pred_loss = gate_noise.item()
        
        # B. Real Infidelity (Qiskit Noise Simulation)
        fid = qiskit_engine.noisy_fidelity_vs_ideal(circ, n_qubits=3, cost_tensor=cost_tensor)
        
        if np.isnan(fid):
            continue
            
        real_infidelity = 1.0 - fid
        
        pred_costs.append(pred_loss)
        real_infidelities.append(real_infidelity)

    # 4. Analysis
    pred_costs = np.array(pred_costs)
    real_infidelities = np.array(real_infidelities)
    
    corr, _ = pearsonr(pred_costs, real_infidelities)
    print(f"Pearson Correlation: {corr:.4f}")
    
    # 5. Visualization
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(pred_costs, real_infidelities, alpha=0.7, label='Circuits')
    
    # Plot y=x line
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5, label='Ideal 1:1')
    
    plt.xlabel('QDPE Predicted Loss (1 - exp(-cost))')
    plt.ylabel('Real Infidelity (1 - Fidelity)')
    plt.title(f'Noise Model Validation\nPearson R = {corr:.4f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = os.path.join(output_dir, "noise_model_correlation.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    
    if corr > 0.85:
        print("PASS: Strong correlation between Proxy Cost and Real Noise.")
    else:
        print("FAIL: Correlation too weak.")
        if corr < 0.7: sys.exit(1)

if __name__ == "__main__":
    validate_noise_correlation()
