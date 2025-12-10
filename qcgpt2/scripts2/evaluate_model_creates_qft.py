"""
qcgpt2/evaluation/evaluate_algorithms.py
Generates Figure 5.4: Zero-shot synthesis of standard quantum algorithms.
Tests QFT-3, Toffoli, and GHZ State Preparation.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Operator, Statevector

# QCGPT
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.gates2 import VOCAB2, BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2
from qcgpt2.encoding2 import tokens_to_circuit2
from qcgpt2.training2.supervised import generate_differentiable_logits, parallel_unitary_product
from qcgpt2.unitaries2 import get_unitary_for_token_id
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

def get_standard_algorithms():
    """Defines the ground truth circuits for standard algorithms."""
    algos = {}
    
    # 1. QFT-3
    qc_qft = QFT(3).decompose()
    algos["QFT-3"] = qc_qft
    
    # 2. Toffoli (CCX)
    qc_toffoli = QuantumCircuit(3)
    qc_toffoli.ccx(0, 1, 2)
    algos["Toffoli"] = qc_toffoli
    
    # 3. GHZ State Prep
    qc_ghz = QuantumCircuit(3)
    qc_ghz.h(0)
    qc_ghz.cx(0, 1)
    qc_ghz.cx(1, 2)
    algos["GHZ Prep"] = qc_ghz
    
    return algos

def extract_truth_table(qc, n_qubits=3):
    """
    Simulates the circuit to get input/output pairs (The Spec).
    Crucially ensures the feature vector format matches training:
    [In_Re0, In_Im0, In_Re1, In_Im1, ..., Out_Re0, Out_Im0...]
    """
    # Get full Unitary
    U = Operator(qc).data # (8, 8)
    
    dim = 2**n_qubits
    
    # Generate full basis truth table
    input_states = []
    output_states = []
    
    for i in range(dim):
        in_vec = np.zeros(dim, dtype=np.complex64)
        in_vec[i] = 1.0
        out_vec = U @ in_vec
        
        input_states.append(in_vec)
        output_states.append(out_vec)
    
    pairs = []
    for i in range(dim):
        # 1. Input State Features
        # Must interleave Real/Imag: [r0, i0, r1, i1...]
        v_in = input_states[i]
        in_feat = np.column_stack((v_in.real, v_in.imag)).flatten()
        
        # 2. Output State Features
        v_out = output_states[i]
        out_feat = np.column_stack((v_out.real, v_out.imag)).flatten()
        
        # Concatenate Input + Output features
        feat = np.concatenate([in_feat, out_feat]) # (16 + 16 = 32)
        pairs.append(feat)
        
    spec = np.stack(pairs) # (8, 32)
    return spec

def run_synthesis(ckpt_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Model
    model = load_model(ckpt_path, device)
    if model is None: return

    # 2. Load Algos
    algos = get_standard_algorithms()
    
    # 3. Preload Unitary Stack for Fidelity Check
    mats = []
    for tid in range(len(VOCAB2)):
        if tid in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
            mats.append(torch.eye(8, dtype=torch.complex128))
        else:
            mats.append(get_unitary_for_token_id(tid, n_qubits=3).to(dtype=torch.complex128))
    U_stack = torch.stack(mats, dim=0).to(device)

    # 4. Generate
    results = {}
    
    for name, target_qc in algos.items():
        print(f"Synthesizing {name}...")
        
        # A. Get Spec
        spec = extract_truth_table(target_qc) # (8, 32)
        spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(device) # (1, 8, 32)
        
        # Mask: All 8 pairs are valid
        spec_mask = torch.zeros(1, 8, dtype=torch.bool).to(device)
        
        # B. Inference
        with torch.no_grad():
            probs = generate_differentiable_logits(
                model, spec_tensor, spec_mask, 
                bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2, 
                max_len=32, greedy=True
            )
            tokens = torch.argmax(probs, dim=-1)[0]
            if (tokens == EOS_CIRC_ID2).any():
                tokens = tokens[:(tokens == EOS_CIRC_ID2).nonzero(as_tuple=True)[0][0]]
        
        # C. Fidelity Check
        # Target U
        U_target = torch.tensor(Operator(target_qc).data, device=device, dtype=torch.complex128)
        
        # Pred U
        if len(tokens) > 0:
            U_seq = U_stack[tokens].unsqueeze(0)
            U_pred = parallel_unitary_product(U_seq)[0]
        else:
            U_pred = torch.eye(8, dtype=torch.complex128).to(device)
            
        # Standard Unitary Fidelity
        # d = 8.0
        # fid = |Tr(U_tgt^dag U_pred)|^2 / d^2
        trace = torch.trace(U_target.conj().T @ U_pred)
        fid = (torch.abs(trace) ** 2) / (8.0 ** 2)
        
        # D. Convert to Circuit for Plotting
        try:
            pred_qc_raw = tokens_to_circuit2(tokens.cpu().numpy())
            pred_qc = circuit2_to_qiskit(pred_qc_raw)
        except:
            pred_qc = QuantumCircuit(3) # Empty/Fail
            
        results[name] = {
            "target": target_qc,
            "pred": pred_qc,
            "fidelity": fid.item()
        }
        print(f"  > Fidelity: {fid.item():.4f}")

    # 5. Plotting
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 4 * len(results)))
    if len(results) == 1: axes = [axes]
    
    idx = 0
    for name, res in results.items():
        # Left: Target (Truth)
        ax_tgt = axes[idx][0]
        # Draw target (simple decomposition for visualization)
        try:
            res["target"].draw('mpl', ax=ax_tgt)
        except: 
            ax_tgt.text(0.5, 0.5, "Draw Error", ha='center')
        ax_tgt.set_title(f"Target: {name}", fontweight='bold')
        
        # Right: Generated (Model)
        ax_pred = axes[idx][1]
        try:
            res["pred"].draw('mpl', ax=ax_pred)
            ax_pred.set_title(f"Generated (F={res['fidelity']:.3f})", fontweight='bold', color='blue')
        except:
            ax_pred.text(0.5, 0.5, "Invalid Circuit", ha='center')
            
        idx += 1
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, "generated_algorithms.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved algorithm synthesis plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Model Checkpoint")
    parser.add_argument("--out", type=str, default="evaluation_results/section_5_3")
    args = parser.parse_args()
    
    run_synthesis(args.ckpt, args.out)