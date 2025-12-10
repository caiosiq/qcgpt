import json, os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
import numpy as np
import torch
from qcgpt2.gates2 import VOCAB2, PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2
from qcgpt2.gate_registry2 import token_to_gate_parts, rotation_to_gate, apply_to_qiskit
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

def check_cswap(ctrl, a, b, n=3):
    c = Circuit2(nqubits=n)
    c.add_gate(Gate2("CSWAP", [ctrl, a, b]))
    qc = circuit2_to_qiskit(c)
    for idx in range(2**n):
        bits = bits_from_index(idx, n)
        psi_in = basis_bits_to_statevector(bits)
        psi_out = psi_in.evolve(qc)
        probs = np.abs(psi_out.data)**2
        out_idx = int(np.argmax(probs))
        if bits[ctrl] == 1:
            exp = bits.copy(); exp[a], exp[b] = exp[b], exp[a]
        else:
            exp = bits
        assert bits_from_index(out_idx, n) == exp
    toks = circuit2_to_tokens(c)
    toks_clean = [t for t in toks if t not in (PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2)]
    c2 = tokens_to_circuit2(toks_clean)
    U1 = build_circuit_unitary2(c, n)
    U2 = build_circuit_unitary2(c2, n)
    tr = torch.einsum("ij,ij->", U1.conj(), U2)
    fid = (tr.abs()**2)/(U1.size(0)**2)
    print("CSWAP", ctrl, a, b, float(fid))

def check_ccz(n=3):
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
    toks_clean = [t for t in toks if t not in (PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2)]
    c2 = tokens_to_circuit2(toks_clean)
    U1 = build_circuit_unitary2(c, n)
    U2 = build_circuit_unitary2(c2, n)
    tr = torch.einsum("ij,ij->", U1.conj(), U2)
    fid = (tr.abs()**2)/(U1.size(0)**2)
    print("CCZ", float(fid))

def check_ccx(n=3):
    c = Circuit2(nqubits=n)
    c.add_gate(Gate2("CCX", [0,1,2]))
    qc = circuit2_to_qiskit(c)
    for idx in range(2**n):
        bits = bits_from_index(idx, n)
        psi_in = basis_bits_to_statevector(bits)
        psi_out = psi_in.evolve(qc)
        probs = np.abs(psi_out.data)**2
        out_idx = int(np.argmax(probs))
        exp = bits.copy()
        if bits[0] == 1 and bits[1] == 1:
            exp[2] ^= 1
        assert bits_from_index(out_idx, n) == exp
    toks = circuit2_to_tokens(c)
    toks_clean = [t for t in toks if t not in (PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2)]
    c2 = tokens_to_circuit2(toks_clean)
    U1 = build_circuit_unitary2(c, n)
    U2 = build_circuit_unitary2(c2, n)
    tr = torch.einsum("ij,ij->", U1.conj(), U2)
    fid = (tr.abs()**2)/(U1.size(0)**2)
    print("CCX", float(fid))

def check_rotations_map():
    for theta in [np.pi, np.pi/2, np.pi/4, np.pi/8, np.pi/16]:
        print("rx", theta, rotation_to_gate("rx", theta))
        print("ry", theta, rotation_to_gate("ry", theta))
        print("rz", theta, rotation_to_gate("rz", theta))

def main():
    check_cswap(0,1,2)
    check_ccz()
    check_ccx()
    check_rotations_map()

if __name__ == "__main__":
    main()
