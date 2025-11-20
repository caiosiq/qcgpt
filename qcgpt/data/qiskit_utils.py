import numpy as np
from typing import Tuple, List

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler.exceptions import TranspilerError

from ..circuits import Circuit
from ..gates import Gate
from ..simulators.qiskit_sim import basis_bits_to_statevector, circuit_to_qiskit, qiskit_to_circuit
from .specs import state_pairs_to_spec_tensor

def all_basis_inputs(n_qubits: int) -> np.ndarray:
    xs = []
    for idx in range(2 ** n_qubits):
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
        xs.append(bits)
    return np.array(xs, dtype=np.int64)

def random_reference_circuit(max_gates: int = 6, n_qubits: int = 3) -> Circuit:
    from ..gates import GATE_TYPES
    circ = Circuit(nqubits=n_qubits)
    for _ in range(np.random.randint(1, max_gates + 1)):
        gt = np.random.choice(GATE_TYPES)
        if gt in {"CX", "CZ", "SWAP"}:
            qs = list(np.random.choice(range(n_qubits), size=2, replace=False))
            circ.add_gate(Gate(gt, qs))
        else:
            q = np.random.randint(0, n_qubits)
            circ.add_gate(Gate(gt, [q]))
    return circ

def build_mapping_spec_from_circuit(circ: Circuit) -> np.ndarray:
    qc = circuit_to_qiskit(circ)
    psi_pairs = []
    for bits in all_basis_inputs(circ.nqubits):
        psi_in = basis_bits_to_statevector(bits)
        psi_out = psi_in.evolve(qc)
        psi_pairs.append((psi_in, psi_out))
    spec = state_pairs_to_spec_tensor(psi_pairs, n_qubits=circ.nqubits)
    return spec

def sample_task(max_gates: int = 6, n_qubits: int = 3) -> Tuple[np.ndarray, Circuit]:
    circ = random_reference_circuit(max_gates=max_gates, n_qubits=n_qubits)
    spec = build_mapping_spec_from_circuit(circ)
    return spec, circ


def sample_random_qiskit_circuit(n_qubits: int = 3, max_depth: int = 8) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    gates = ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "cx", "cz", "swap", "ccx", "cswap", "ccz"]
    angles = [np.pi/16, np.pi/8, np.pi/4, np.pi/2, np.pi]
    for _ in range(np.random.randint(1, max_depth + 1)):
        g = np.random.choice(gates)
        if g in {"x", "y", "z", "h", "s", "t"}:
            qc.__getattribute__(g)(np.random.randint(0, n_qubits))
        elif g in {"rx", "ry", "rz"}:
            q = np.random.randint(0, n_qubits)
            theta = np.random.choice(angles)
            getattr(qc, g)(theta, q)
        elif g in {"cx", "cz", "swap"}:
            q0, q1 = np.random.choice(range(n_qubits), size=2, replace=False)
            getattr(qc, g)(q0, q1)
        elif g in {"ccx", "cswap"}:
            q0, q1, q2 = np.random.choice(range(n_qubits), size=3, replace=False)
            getattr(qc, g)(q0, q1, q2)
        elif g == "ccz":
            q0, q1, q2 = np.random.choice(range(n_qubits), size=3, replace=False)
            qc.h(q2); qc.ccx(q0, q1, q2); qc.h(q2)
    return qc


def simplify_qiskit_circuit(
    qc_raw: QuantumCircuit,
    basis_gates: List[str] | None = None,
    optimization_level: int = 3,
) -> QuantumCircuit:
    if basis_gates is None:
        basis_gates = ["id", "x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "cx", "cz", "swap", "ccx", "cswap"]
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
    n_qubits: int = 3,
    use_basis_states: bool = True,
    n_random_states: int = 0,
) -> np.ndarray:
    qc = circuit_to_qiskit(circ)
    psi_pairs = []
    if use_basis_states:
        for b in all_basis_inputs(n_qubits):
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
