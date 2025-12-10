import json, os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
import numpy as np
import torch
from qcgpt2.circuits2 import Circuit2, Gate2
from qcgpt2.encoding2 import circuit2_to_tokens, tokens_to_circuit2
from qcgpt2.unitaries2 import build_circuit_unitary2
from qcgpt2.simulators2.qiskit_sim2 import circuit2_to_qiskit, basis_bits_to_statevector

def bits_from_index(idx, n):
    return [(idx >> i) & 1 for i in range(n)]

def index_from_bits(bits):
    idx = 0
    for i, b in enumerate(bits):
        idx |= (b << i)
    return idx

def expected_cswap(bits, ctrl, a, b):
    out = bits.copy()
    if bits[ctrl] == 1:
        out[a], out[b] = out[b], out[a]
    return out

def test_cswap_truth_table():
    n = 3
    ctrl, a, b = 0, 1, 2
    c = Circuit2(nqubits=n)
    c.add_gate(Gate2("CSWAP", [ctrl, a, b]))
    qc = circuit2_to_qiskit(c)
    for idx in range(2**n):
        bits = bits_from_index(idx, n)
        psi_in = basis_bits_to_statevector(bits)
        psi_out = psi_in.evolve(qc)
        probs = np.abs(psi_out.data)**2
        out_idx = int(np.argmax(probs))
        out_bits = bits_from_index(out_idx, n)
        assert out_bits == expected_cswap(bits, ctrl, a, b)
    toks = circuit2_to_tokens(c)
    c2 = tokens_to_circuit2([t for t in toks if t not in [0, toks[0], toks[-1]]])
    U1 = build_circuit_unitary2(c, n)
    U2 = build_circuit_unitary2(c2, n)
    tr = torch.einsum("ij,ij->", U1.conj(), U2)
    fid = (tr.abs()**2)/(U1.size(0)**2)
    assert float(fid) > 0.999

def test_ccz_phase():
    n = 3
    c = Circuit2(nqubits=n)
    c.add_gate(Gate2("CCZ", [0,1,2]))
    qc = circuit2_to_qiskit(c)
    for idx in range(2**n):
        bits = bits_from_index(idx, n)
        psi_in = basis_bits_to_statevector(bits)
        psi_out = psi_in.evolve(qc)
        amp = psi_out.data[index_from_bits(bits)]
        if bits == [1,1,1]:
            assert np.isclose(amp.real, -1.0, atol=1e-6)
        else:
            assert np.isclose(amp.real, 1.0, atol=1e-6)
    toks = circuit2_to_tokens(c)
    c2 = tokens_to_circuit2([t for t in toks if t not in [0, toks[0], toks[-1]]])
    U1 = build_circuit_unitary2(c, n)
    U2 = build_circuit_unitary2(c2, n)
    tr = torch.einsum("ij,ij->", U1.conj(), U2)
    fid = (tr.abs()**2)/(U1.size(0)**2)
    assert float(fid) > 0.999

def test_cx_truth_table():
    n = 3
    c = Circuit2(nqubits=n)
    c.add_gate(Gate2("CX", [0,1]))
    qc = circuit2_to_qiskit(c)
    for idx in range(2**n):
        bits = bits_from_index(idx, n)
        psi_in = basis_bits_to_statevector(bits)
        psi_out = psi_in.evolve(qc)
        probs = np.abs(psi_out.data)**2
        out_idx = int(np.argmax(probs))
        out_bits = bits_from_index(out_idx, n)
        exp = bits.copy()
        if bits[0] == 1:
            exp[1] ^= 1
        assert out_bits == exp

if __name__ == "__main__":
    print('testing cswap_truth_table')
    test_cswap_truth_table()
    print('testing ccz_phase')
    test_ccz_phase()
    print('testing cx_truth_table')
    test_cx_truth_table()