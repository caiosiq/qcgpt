import numpy as np
from typing import Tuple, List

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler.exceptions import TranspilerError

from ..circuits import Circuit
from ..gates import Gate
from ..simulators.qiskit_sim import basis_bits_to_statevector, circuit_to_qiskit, qiskit_to_circuit
from .specs import state_pairs_to_spec_tensor

ALL_2BIT_INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)

def random_reference_circuit(max_gates: int = 6) -> Circuit:
    from ..gates import GATE_TYPES
    circ = Circuit(nqubits=2)
    for _ in range(np.random.randint(1, max_gates + 1)):
        gt = np.random.choice(GATE_TYPES)
        if gt in {"X", "Y", "Z", "H", "S", "T", "ID"}:
            q = np.random.randint(0, 2)
            circ.add_gate(Gate(gt, [q]))
        else:
            qs = [0, 1]
            np.random.shuffle(qs)
            circ.add_gate(Gate(gt, qs))
    return circ

def build_mapping_spec_from_circuit(circ: Circuit) -> np.ndarray:
    qc = circuit_to_qiskit(circ)
    psi_pairs = []
    for bits in ALL_2BIT_INPUTS:
        psi_in = basis_bits_to_statevector(bits)
        psi_out = psi_in.evolve(qc)
        psi_pairs.append((psi_in, psi_out))
    spec = state_pairs_to_spec_tensor(psi_pairs, n_qubits=2)
    return spec

def sample_task(max_gates: int = 6) -> Tuple[np.ndarray, Circuit]:
    circ = random_reference_circuit(max_gates=max_gates)
    spec = build_mapping_spec_from_circuit(circ)
    return spec, circ


def sample_random_qiskit_circuit(n_qubits: int = 2, max_depth: int = 8) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    gates = ["x", "y", "z", "h", "s", "t", "cx", "cz", "swap"]
    for _ in range(np.random.randint(1, max_depth + 1)):
        g = np.random.choice(gates)
        if g in {"x", "y", "z", "h", "s", "t"}:
            qc.__getattribute__(g)(np.random.randint(0, n_qubits))
        elif g in {"cx", "cz", "swap"}:
            q0, q1 = np.random.choice(range(n_qubits), size=2, replace=False)
            getattr(qc, g)(q0, q1)
    return qc


def simplify_qiskit_circuit(
    qc_raw: QuantumCircuit,
    basis_gates: List[str] | None = None,
    optimization_level: int = 3,
) -> QuantumCircuit:
    if basis_gates is None:
        basis_gates = ["id", "x", "y", "z", "h", "s", "t", "cx", "cz", "swap"]
    try:
        return transpile(qc_raw, basis_gates=basis_gates, optimization_level=optimization_level)
    except TranspilerError:
        for lvl in [2, 1, 0]:
            try:
                return transpile(qc_raw, optimization_level=lvl)
            except TranspilerError:
                continue
        return qc_raw


def build_spec_from_circuit(
    circ: Circuit,
    n_qubits: int = 2,
    use_basis_states: bool = True,
    n_random_states: int = 0,
) -> np.ndarray:
    qc = circuit_to_qiskit(circ)
    psi_pairs = []
    if use_basis_states:
        for b in ALL_2BIT_INPUTS[: 2 ** n_qubits]:
            psi_in = basis_bits_to_statevector(b)
            psi_out = psi_in.evolve(qc)
            psi_pairs.append((psi_in, psi_out))
    for _ in range(n_random_states):
        vec = np.random.randn(2 ** n_qubits) + 1j * np.random.randn(2 ** n_qubits)
        vec = vec / np.linalg.norm(vec)
        psi_in = Statevector(vec)
        psi_out = psi_in.evolve(qc)
        psi_pairs.append((psi_in, psi_out))
    return state_pairs_to_spec_tensor(psi_pairs, n_qubits=n_qubits)
