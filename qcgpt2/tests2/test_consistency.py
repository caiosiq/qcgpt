import json, os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
import torch
from qcgpt2.gates2 import VOCAB2, PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2
from qcgpt2.encoding2 import token_to_gate, circuit2_to_tokens, tokens_to_circuit2
from qcgpt2.circuits2 import Circuit2
from qcgpt2.unitaries2 import build_circuit_unitary2

def test_roundtrip_unitary():
    c = Circuit2(nqubits=3)
    c.add_gate(token_to_gate("RY_PI_2_0"))
    c.add_gate(token_to_gate("H_1"))
    c.add_gate(token_to_gate("CCZ_0_1_2"))
    toks = circuit2_to_tokens(c)
    toks = [t for t in toks if t not in (PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2)]
    c2 = tokens_to_circuit2(toks)
    U1 = build_circuit_unitary2(c,3)
    U2 = build_circuit_unitary2(c2,3)
    tr = torch.einsum("ij,ij->", U1.conj(), U2)
    fid = (tr.abs()**2)/(U1.size(0)**2)
    assert float(fid) > 0.999

def test_canonicalization_ccz():
    g1 = token_to_gate("CCZ_2_1_0")
    assert g1.targets == [0,1,2]
