import torch
import numpy as np
import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
from qcgpt2.gates2 import PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2, VOCAB2, GATE_COST_REGISTRY, TOKEN_TO_ID2
from qcgpt2.training2.supervised import calculate_physical_fidelity_components
from qcgpt2.encoding2 import tokens_to_circuit2
from qcgpt2.simulators2.qiskit_sim2 import noisy_fidelity_vs_ideal

 

def run_integration_test():
    device = torch.device("cpu")
    vocab_size = len(VOCAB2)
    cost_values = [GATE_COST_REGISTRY[tok] for tok in VOCAB2]
    cost_tensor = torch.tensor(cost_values, device=device, dtype=torch.float32)
    id_h = TOKEN_TO_ID2["H_0"]
    id_cx = TOKEN_TO_ID2["CX_0_1"]
    id_cswap = TOKEN_TO_ID2["CSWAP_1_0_2"]
    id_ccx = TOKEN_TO_ID2["CCX_0_1_2"]
    toks = [id_h, id_cx, id_ccx,id_cswap,id_h,id_cswap,id_ccx, EOS_CIRC_ID2]
    seq = [t for t in toks if t not in (PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2)]
    circ = tokens_to_circuit2(seq, n_qubits=3)
    print("[Test] Tokens:", [VOCAB2[t] for t in seq])
    print("[Test] Gate costs (mean by type): H=", float(cost_tensor[id_h]), "CX=", float(cost_tensor[id_cx]), "CSWAP=", float(cost_tensor[id_cswap]),"CCX=",float(cost_tensor[id_ccx]) )
    probs = torch.zeros(1, len(toks), vocab_size, device=device)
    for i,tok in enumerate(toks):
        probs[0,i,tok]=1.0
    
    U_stack = torch.zeros(vocab_size, 8, 8)
    _, noise_sum = calculate_physical_fidelity_components(probs, U_stack, cost_tensor, device, noise_scale=1.0)
    expected_noise = float(noise_sum)
    print("[Physics] Expected noise sum:", expected_noise)
    fid_noisy = noisy_fidelity_vs_ideal(circ, 3, cost_tensor)
    print("[Qiskit] Noisy fidelity vs ideal:", fid_noisy)
    if not np.isnan(fid_noisy):
        loss_due_to_noise = 1.0 - fid_noisy
        print("[Qiskit] Loss due to noise:", loss_due_to_noise)
        assert abs(loss_due_to_noise - expected_noise) < 0.02
    else:
        print("[Qiskit] Aer not available; verifying registry baseline only")
        assert abs(expected_noise - 0.011) < 1e-6

if __name__ == "__main__":
    run_integration_test()
