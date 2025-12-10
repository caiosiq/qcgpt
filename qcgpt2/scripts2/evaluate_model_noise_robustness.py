"""
qcgpt2/evaluation/evaluate_noise_robustness.py
Generates the 'Fidelity Gain' histogram for Section 5.2.
Compares Phase III (Noise-Adaptive) model vs Qiskit Optimization Level 3.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import state_fidelity, Statevector

# QCGPT Imports
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.gates2 import VOCAB2, BOS_CIRC_ID2, EOS_CIRC_ID2
from qcgpt2.data.qiskit_utils2 import sample_task2
from qcgpt2.encoding2 import tokens_to_circuit2
from qcgpt2.training2.supervised import generate_differentiable_logits, _convert_circuit_to_tokens2
from qcgpt2.simulators2.qiskit_sim2 import circuit2_to_qiskit

def load_model(ckpt_path, device):
    print(f"Loading model from {ckpt_path}...")
    model = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    try:
        # weights_only=False to support legacy checkpoints
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
        return None
    model.eval()
    return model

def build_noise_model(scale_factor=1.0):
    """
    Builds a depolarizing noise model scaled by the user factor.
    Base rates are approximations of standard IBM hardware (Heron/Eagle).
    """
    nm = NoiseModel()
    
    # 1-qubit gates: ~0.001 base error
    p1 = min(1.0, 0.001 * scale_factor)
    error_1q = depolarizing_error(p1, 1)
    nm.add_all_qubit_quantum_error(error_1q, ['rx', 'ry', 'rz', 'h', 's', 'x', 'y', 'z'])
    
    # 2-qubit gates: ~0.01 base error (10x 1-qubit)
    p2 = min(1.0, 0.01 * scale_factor)
    error_2q = depolarizing_error(p2, 2)
    nm.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'swap'])
    
    # 3-qubit gates (if used): ~0.1 base error (expensive!)
    p3 = min(1.0, 0.1 * scale_factor)
    error_3q = depolarizing_error(p3, 3)
    nm.add_all_qubit_quantum_error(error_3q, ['ccx', 'cswap'])
    
    return nm

def simulate_fidelity(qc_qiskit, noise_model, target_unitary_circ):
    """
    Runs the circuit on a noisy Density Matrix simulator and compares 
    output state to the ideal target state.
    """
    if qc_qiskit is None:
        return 0.0

    n_qubits = qc_qiskit.num_qubits
    
    # 1. Ideal State (Ground Truth)
    # We simulate the *target* circuit ideally to get the expected vector
    psi_ideal = Statevector.from_int(0, 2**n_qubits).evolve(target_unitary_circ)
    
    # 2. Noisy Simulation
    # Append density matrix save
    qc_sim = qc_qiskit.copy()
    qc_sim.save_density_matrix()
    
    # Transpile for the simulator (handles basis gates)
    backend = AerSimulator(method='density_matrix', noise_model=noise_model)
    t_qc = transpile(qc_sim, backend, optimization_level=0) # Don't optimize further, measure exactly what was given
    
    try:
        result = backend.run(t_qc).result()
        rho_noisy = result.data(0)['density_matrix']
        # Compute Fidelity
        fid = state_fidelity(psi_ideal, rho_noisy)
        return fid
    except Exception as e:
        # print(f"Sim error: {e}")
        return 0.0

def run_comparison(ckpt_path, output_dir, noise_scale=5.0, n_samples=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Model
    model = load_model(ckpt_path, device)
    if model is None: return

    # 2. Setup Environment
    noise_model = build_noise_model(noise_scale)
    print(f"Noise Model Configured (Scale={noise_scale}x)")
    
    results = []
    
    print(f"Benchmarking {n_samples} circuits against Qiskit O3...")
    
    for i in tqdm(range(n_samples)):
        # Generate Task
        depth = np.random.randint(4, 12) # Use slightly deeper circuits to make optimization interesting
        spec, target_circ_raw = sample_task2(max_gates=depth)
        
        # Convert raw target to Qiskit object for ground truth
        qc_target = circuit2_to_qiskit(target_circ_raw)
        
        # --- A. Qiskit Baseline ---
        # "Standard Compilation": Transpile with optimization_level=3
        # We use a standard basis to give Qiskit a fair target
        basis_gates = ['u1', 'u2', 'u3', 'cx', 'cz'] 
        qc_qiskit = transpile(qc_target, basis_gates=basis_gates, optimization_level=3)
        
        fid_qiskit = simulate_fidelity(qc_qiskit, noise_model, qc_target)
        
        # --- B. Model Prediction ---
        # Prepare Input
        if isinstance(spec, list): spec = np.array(spec)
        spec_tensor = torch.tensor(spec, dtype=torch.float32).to(device)
        if spec_tensor.dim() > 2: spec_tensor = spec_tensor.view(spec_tensor.size(0), -1)
        if spec_tensor.dim() == 2: spec_tensor = spec_tensor.unsqueeze(0)
        spec_mask = torch.zeros(1, spec_tensor.size(1), dtype=torch.bool).to(device)
        
        # Inference
        with torch.no_grad():
            probs = generate_differentiable_logits(
                model, spec_tensor, spec_mask, 
                bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2, 
                max_len=32, greedy=True
            )
            tokens = torch.argmax(probs, dim=-1)[0]
            if (tokens == EOS_CIRC_ID2).any():
                tokens = tokens[:(tokens == EOS_CIRC_ID2).nonzero(as_tuple=True)[0][0]]
        
        # Convert to Circuit
        try:
            model_circ_tokens = tokens.cpu().numpy()
            qc_model_raw = tokens_to_circuit2(model_circ_tokens)
            qc_model = circuit2_to_qiskit(qc_model_raw)
            fid_model = simulate_fidelity(qc_model, noise_model, qc_target)
        except Exception:
            # Model generated invalid circuit -> 0 fidelity
            fid_model = 0.0
            
        # Record
        delta_f = fid_model - fid_qiskit
        results.append({
            "depth": depth,
            "fid_qiskit": fid_qiskit,
            "fid_model": fid_model,
            "delta_f": delta_f
        })

    # --- Analysis & Plotting ---
    df = pd.DataFrame(results)
    
    # Win Rate
    wins = len(df[df['delta_f'] > 0.01])
    ties = len(df[(df['delta_f'] >= -0.01) & (df['delta_f'] <= 0.01)])
    losses = len(df[df['delta_f'] < -0.01])
    print(f"\nResults: Model Wins: {wins}, Ties: {ties}, Qiskit Wins: {losses}")
    print(f"Average Fidelity Gain: {df['delta_f'].mean():.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Histogram of Delta F
    # We want to see positive skew
    counts, bins, patches = plt.hist(df['delta_f'], bins=30, color='teal', alpha=0.7, edgecolor='black')
    
    # Highlight the zero line
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Equal Performance')
    
    # Add stats to legend/title
    plt.title(f'NISQ Advantage: Noise-Adaptive Model vs. Qiskit O3\n(Noise Scale={noise_scale}x, Mean Gain={df["delta_f"].mean():.3f})')
    plt.xlabel('Fidelity Gain (Model - Qiskit)', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, 'fidelity_gain_histogram.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")
    
    df.to_csv(os.path.join(output_dir, 'noise_comparison.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Phase 3 Model")
    parser.add_argument("--out", type=str, default="evaluations_models/results_section_5_2")
    parser.add_argument("--scale", type=float, default=5.0, help="Noise scale factor")
    parser.add_argument("--samples", type=int, default=300)
    args = parser.parse_args()
    
    run_comparison(args.ckpt, args.out, args.scale, args.samples)