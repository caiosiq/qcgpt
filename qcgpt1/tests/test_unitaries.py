import torch
from qcgpt1.circuits import Circuit
from qcgpt1.gates import Gate
from qcgpt1.unitaries import build_circuit_unitary


def test_build_circuit_unitary_identity():
    circ = Circuit(nqubits=3)
    U = build_circuit_unitary(circ, n_qubits=3)
    I = torch.eye(8, dtype=torch.complex64)
    assert torch.allclose(U, I)


def test_build_circuit_unitary_simple():
    circ = Circuit(nqubits=3)
    circ.add_gate(Gate("X", [0]))
    U = build_circuit_unitary(circ, n_qubits=3)
    assert U.shape == (8, 8)
