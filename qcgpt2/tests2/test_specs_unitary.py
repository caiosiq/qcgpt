import json, os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
import numpy as np
import torch
from qcgpt2.circuits2 import Circuit2, Gate2
from qcgpt2.unitaries2 import build_circuit_unitary2
from qcgpt2.simulators2.qiskit_sim2 import circuit2_to_qiskit, basis_bits_to_statevector
from qcgpt2.data.qiskit_utils2 import build_mapping_spec_from_circuit2

def build_unitary_via_qiskit(circ: Circuit2, n_qubits: int) -> torch.Tensor:
    qc = circuit2_to_qiskit(circ)
    D = 2 ** n_qubits
    U = np.zeros((D, D), dtype=np.complex64)
    for idx in range(D):
        psi_in = basis_bits_to_statevector([(idx >> i) & 1 for i in range(n_qubits)])
        psi_out = psi_in.evolve(qc)
        U[:, idx] = psi_out.data.astype(np.complex64)
    return torch.tensor(U)

def compare_matrices(U1: torch.Tensor, U2: torch.Tensor, atol=1e-6):
    diff = (U1 - U2).abs().max().item()
    assert diff < atol, f"Unitary mismatch max diff {diff}"

def compare_specs(circ: Circuit2, U: torch.Tensor, n_qubits: int):
    spec = build_mapping_spec_from_circuit2(circ)
    D = 2 ** n_qubits
    # Validate spec pairs match U application for basis states
    for idx in range(D):
        # psi_in one-hot basis
        e = np.zeros((D,), dtype=np.complex64); e[idx] = 1.0 + 0j
        psi_out = (U.numpy() @ e)
        rin = spec[idx, 0, :, 0]; iin = spec[idx, 0, :, 1]
        rout = spec[idx, 1, :, 0]; iout = spec[idx, 1, :, 1]
        vin = rin + 1j * iin
        vout = rout + 1j * iout
        assert np.allclose(vin, e, atol=1e-6)
        assert np.allclose(vout, psi_out, atol=1e-6)

def test_identity_and_single_gates():
    n = 3
    c = Circuit2(nqubits=n)
    U_q = build_unitary_via_qiskit(c, n)
    U_m = build_circuit_unitary2(c, n)
    compare_matrices(U_m, U_q)
    compare_specs(c, U_m, n)
    # Single-qubit gate
    for gt in ["X", "Y", "Z", "H", "S", "T", "RX_PI_2", "RY_PI_2", "RZ_PI_16"]:
        c = Circuit2(nqubits=n); c.add_gate(Gate2(gt, [0]))
        U_q = build_unitary_via_qiskit(c, n)
        U_m = build_circuit_unitary2(c, n)
        compare_matrices(U_m, U_q)
        compare_specs(c, U_m, n)

def test_two_qubit_gates():
    n = 3
    for gt, a, b in [("CX",0,1),("CZ",0,2),("SWAP",1,2)]:
        c = Circuit2(nqubits=n); c.add_gate(Gate2(gt, [a,b]))
        U_q = build_unitary_via_qiskit(c, n)
        U_m = build_circuit_unitary2(c, n)
        compare_matrices(U_m, U_q)
        compare_specs(c, U_m, n)

def test_three_qubit_gates():
    n = 3
    for gt, t in [("CCX", [0,1,2]), ("CCZ", [0,1,2]), ("CSWAP", [0,1,2])]:
        c = Circuit2(nqubits=n); c.add_gate(Gate2(gt, t))
        U_q = build_unitary_via_qiskit(c, n)
        U_m = build_circuit_unitary2(c, n)
        compare_matrices(U_m, U_q)
        compare_specs(c, U_m, n)

if __name__ == "__main__":
    print('testing identity and single qubit gates')
    test_identity_and_single_gates()
    print('testing two qubit gates')
    test_two_qubit_gates()
    print('testing three qubit gates')
    test_three_qubit_gates()