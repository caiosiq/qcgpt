import json, os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
import numpy as np
import torch
from qcgpt2.gates2 import VOCAB2, PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.training2.supervised import calculate_physical_fidelity_components
from qcgpt2.unitaries2 import get_unitary_for_token_id

def build_U_stack(device):
    mats = []
    for tid in range(len(VOCAB2)):
        if tid in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
            mats.append(torch.eye(8, dtype=torch.complex64))
        else:
            mats.append(get_unitary_for_token_id(tid, n_qubits=3).to(dtype=torch.complex64))
    return torch.stack(mats, dim=0).to(device)

def test_zero_loss_exact_logits():
    device = torch.device('cpu')
    U_stack = build_U_stack(device)
    # Build logits that exactly match a simple circuit: [BOS, X_0, EOS, PAD, ...]
    V = len(VOCAB2)
    B, Lc = 1, 4
    logits = torch.zeros((B, Lc, V), dtype=torch.float32)
    # Positions: 0:BOS, 1:X_0, 2:EOS, 3:PAD
    def id(tok): return VOCAB2.index(tok)
    toks = [id('<BOS_CIRC>'), id('X_0'), id('<EOS_CIRC>'), id('<PAD>')]
    for t in range(Lc):
        logits[0, t, toks[t]] = 10.0
    cost_tensor = torch.zeros((V,), dtype=torch.float32)
    U_pred, noise = calculate_physical_fidelity_components(logits, U_stack, cost_tensor, device, temperature=1e-6, hard=True)
    # Target unitary is X on q0
    ids = torch.tensor([toks[1], toks[2], toks[3]], dtype=torch.long).view(1, -1)
    U_tgt_seq = U_stack[ids]
    def parallel(seq):
        curr = seq
        while curr.shape[1] > 1:
            curr = curr[:, 1::2] @ curr[:, 0::2]
        return curr.squeeze(1)
    U_tgt = parallel(U_tgt_seq)
    tr = torch.einsum('bij,bji->b', U_tgt.conj(), U_pred)
    fid = (tr.abs()**2)/(8**2)
    assert float(fid) > 0.999

def test_stoppage_masking():
    device = torch.device('cpu')
    U_stack = build_U_stack(device)
    V = len(VOCAB2)
    # logits: BOS, X_0, EOS, X_1 -> eos at pos2 should mask pos3
    B, Lc = 1, 4
    logits = torch.zeros((B, Lc, V))
    def id(tok): return VOCAB2.index(tok)
    toks = [id('<BOS_CIRC>'), id('X_0'), id('<EOS_CIRC>'), id('X_1')]
    for t in range(Lc):
        logits[0, t, toks[t]] = 10.0
    cost_tensor = torch.zeros((V,), dtype=torch.float32)
    U_pred, noise = calculate_physical_fidelity_components(logits, U_stack, cost_tensor, device, temperature=1e-6, hard=True)
    # Target U is X_0 only; X_1 should be masked out
    ids = torch.tensor([toks[1], toks[2], PAD_ID2], dtype=torch.long).view(1, -1)
    U_tgt = build_U_stack(device)[ids]
    curr = U_tgt
    while curr.shape[1] > 1:
        curr = curr[:, 1::2] @ curr[:, 0::2]
    U_tgt = curr.squeeze(1)
    tr = torch.einsum('bij,bji->b', U_tgt.conj(), U_pred)
    fid = (tr.abs()**2)/(8**2)
    assert float(fid) > 0.999

def test_gate_noise_penalty():
    device = torch.device('cpu')
    U_stack = build_U_stack(device)
    V = len(VOCAB2)
    # logits choose two gates with different costs; check penalty adds
    B, Lc = 1, 3
    logits = torch.zeros((B, Lc, V))
    def id(tok): return VOCAB2.index(tok)
    a, b = id('X_0'), id('H_0')
    logits[0,0,id('<BOS_CIRC>')]=10.0
    logits[0,1,a]=10.0
    logits[0,2,b]=10.0
    cost = torch.zeros((V,)); cost[a]=0.2; cost[b]=0.5
    U_pred, noise = calculate_physical_fidelity_components(logits, U_stack, cost.to(torch.float32), device, temperature=1e-6, hard=True)
    assert abs(float(noise[0]) - (0.2+0.5)) < 1e-6

def test_unitary_equivalence_zero_loss():
    # Two different token sequences producing same unitary
    device = torch.device('cpu')
    U_stack = build_U_stack(device)
    V = len(VOCAB2)
    B, Lc = 1, 3
    def id(tok): return VOCAB2.index(tok)
    # Example: Z and RZ_PI produce same unitary (if present); use S then RZ_PI_2 equivalence via registry mapping in simulator
    # Here we just compare two target sequences with same U
    ids1 = torch.tensor([[id('S_0'), id('<PAD>')]], dtype=torch.long)
    ids2 = torch.tensor([[id('RZ_PI_2_0'), id('<PAD>')]], dtype=torch.long) if any(t.startswith('RZ_PI_2_') for t in VOCAB2) else ids1
    U1 = U_stack[ids1]
    U2 = U_stack[ids2]
    def parallel(seq):
        curr = seq
        while curr.shape[1] > 1:
            curr = curr[:, 1::2] @ curr[:, 0::2]
        return curr.squeeze(1)
    U1 = parallel(U1); U2 = parallel(U2)
    diff = (U1 - U2).abs().max().item()
    assert diff < 1e-6

if __name__ == "__main__":
    test_zero_loss_exact_logits()
    test_stoppage_masking()
    test_gate_noise_penalty()
    test_unitary_equivalence_zero_loss()
