"""
qcgpt2/architecture_validation/validate_noise_model.py
Benchmarks the Gate Cost Proxy against real Qiskit Depolarizing Noise simulations.
Optimized for speed using batching and multiprocessing.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from qcgpt2.gates2 import VOCAB2, GATE_COST_REGISTRY, TOKEN_TO_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2
from qcgpt2.data.qiskit_utils2 import sample_task2
from qcgpt2.encoding2 import circuit2_to_tokens
from qcgpt2.simulators2.qiskit_sim2 import noisy_fidelity_vs_ideal
from qcgpt2.training2.supervised import calculate_physical_fidelity_components

# --- Helper for Multiprocessing ---
def run_qiskit_sim_wrapper(args):
    """
    Wrapper to run Qiskit simulation in a separate process.
    args: (circuit, n_qubits, cost_tensor_numpy, depth)
    """
    circ, n_qubits, cost_values, depth = args
    # Reconstruct tensor inside process to avoid pickling issues with CUDA tensors
    # (Though we pass numpy array for safety)
    cost_tensor = torch.tensor(cost_values, dtype=torch.float32)
    
    try:
        fid = noisy_fidelity_vs_ideal(circ, n_qubits, cost_tensor)
        return (depth, fid)
    except Exception:
        return (depth, float("nan"))

def run_noise_benchmark(n_samples=300, max_workers=8):
    print(f"=== Validating Noise Model (N={n_samples}) ===")
    print(f"Parallelizing Qiskit simulation with {max_workers} workers.")
    
    device = torch.device("cpu") # Validation is CPU-bound by Qiskit anyway
    
    # 1. Setup Cost Tensor
    noise_scale = 1.0
    cost_values = [GATE_COST_REGISTRY[tok] for tok in VOCAB2]
    vocab_size = len(VOCAB2)
    cost_tensor = torch.tensor(cost_values, device=device, dtype=torch.float32) * noise_scale
    
    # 2. Generate Data Batch
    print("Generating circuits...")
    tasks = []
    for _ in range(n_samples):
        depth = np.random.randint(2, 25)
        _, circ = sample_task2(max_gates=depth)
        tasks.append((circ, depth))
        
    # 3. Calculate QDPE Predictions (Batched)
    print("Calculating QDPE predictions...")
    pred_infidelities = []
    
    # We process sequentially for QDPE since it's fast
    for circ, depth in tasks:
        tokens = circuit2_to_tokens(circ)
        # One-Hot Probs
        probs = torch.zeros(1, len(tokens), vocab_size, device=device)
        for j, tok in enumerate(tokens):
            probs[0, j, tok] = 1.0
            
        U_stack = torch.zeros(vocab_size, 8, 8, device=device) # Dummy
        _, pred_loss = calculate_physical_fidelity_components(probs, U_stack, cost_tensor, device)
        pred_infidelities.append(pred_loss.item())

    # 4. Calculate Qiskit Reality (Parallel)
    print("Running Qiskit simulations...")
    real_infidelities = [None] * n_samples
    valid_indices = []
    
    # Prepare args for workers (convert tensor to list/numpy for pickling)
    sim_args = [(circ, 3, cost_values, depth) for circ, depth in tasks]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(run_qiskit_sim_wrapper, arg): i for i, arg in enumerate(sim_args)}
        
        for future in tqdm(as_completed(future_to_idx), total=n_samples):
            idx = future_to_idx[future]
            try:
                depth_res, fid = future.result()
                if not np.isnan(fid):
                    real_infidelities[idx] = 1.0 - fid
                    valid_indices.append(idx)
            except Exception as e:
                # print(f"Sim failed: {e}")
                pass

    # 5. Compile Results
    results = []
    for i in valid_indices:
        results.append({
            "depth": tasks[i][1],
            "pred_infidelity": pred_infidelities[i],
            "real_infidelity": real_infidelities[i]
        })
        
    df = pd.DataFrame(results)
    
    # Analysis
    if len(df) > 1:
        corr, _ = pearsonr(df['pred_infidelity'], df['real_infidelity'])
        print(f"\nCorrelation (Predicted vs Real): {corr:.4f}")
    else:
        print("\nNot enough valid samples.")
        corr = 0.0
    
    # Save
    output_dir = "/orcd/home/002/caiosiq/qcgpt/qcgpt2/architecture_validation/results"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['pred_infidelity'], df['real_infidelity'], c=df['depth'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Circuit Depth')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction (y=x)')
    plt.xlabel('QDPE Predicted Infidelity (1 - exp(-S))')
    plt.ylabel('Qiskit Measured Infidelity (1 - F)')
    plt.title(f'Noise Model Validation (R={corr:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.savefig(os.path.join(output_dir, 'noise_validation_scatter.png'))
    df.to_csv(os.path.join(output_dir, 'noise_validation.csv'), index=False)
    print(f"Saved results to {output_dir}")

if __name__ == "__main__":
    # Check CPU count for workers
    workers = min(16, os.cpu_count() or 1)
    run_noise_benchmark(n_samples=300, max_workers=workers)