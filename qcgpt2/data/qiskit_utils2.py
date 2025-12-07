import numpy as np
from typing import Tuple, List

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qcgpt2.simulators2.qiskit_sim2 import circuit2_to_qiskit as circuit_to_qiskit
from qcgpt2.gate_registry2 import canonicalize_targets, ONEQ_STD, ONEQ_RX, ONEQ_RY, ONEQ_RZ, TWOQ, THREEQ
from qcgpt1.data.qiskit_utils import all_basis_inputs
from .specs2 import state_pairs_to_spec_tensor
from ..circuits2 import Circuit2, Gate2


def random_reference_circuit2(max_gates: int = 6, n_qubits: int = 3) -> Circuit2:
    ONEQ = ONEQ_STD + ONEQ_RX + ONEQ_RY + ONEQ_RZ
    circ = Circuit2(nqubits=n_qubits)
    for _ in range(np.random.randint(1, max_gates + 1)):
        r = np.random.rand()
        if r < 0.6:
            gt = np.random.choice(ONEQ)
            q = np.random.randint(0, n_qubits)
            circ.add_gate(Gate2(gt, [q]))
        elif r < 0.9:
            gt = np.random.choice(TWOQ)
            a, b = np.random.choice(range(n_qubits), size=2, replace=False)
            targets = canonicalize_targets(gt, [a, b])
            circ.add_gate(Gate2(gt, targets))
        else:
            gt = np.random.choice(THREEQ)
            a, b, c = np.random.choice(range(n_qubits), size=3, replace=False)
            targets = canonicalize_targets(gt, [a, b, c])
            circ.add_gate(Gate2(gt, targets))
    return circ


def build_mapping_spec_from_circuit2(circ: Circuit2) -> np.ndarray:
    qc = circuit_to_qiskit(circ)  # adapter handles Circuit2 gates
    psi_pairs = []
    for bits in all_basis_inputs(circ.nqubits):
        psi_in = Statevector.from_int(sum(bits[i] << i for i in range(circ.nqubits)), dims=2**circ.nqubits)
        psi_out = psi_in.evolve(qc)
        psi_pairs.append((psi_in, psi_out))
    return state_pairs_to_spec_tensor(psi_pairs, n_qubits=circ.nqubits)


def sample_task2(max_gates: int = 6, n_qubits: int = 3) -> Tuple[np.ndarray, Circuit2]:
    circ = random_reference_circuit2(max_gates=max_gates, n_qubits=n_qubits)
    spec = build_mapping_spec_from_circuit2(circ)
    return spec, circ
