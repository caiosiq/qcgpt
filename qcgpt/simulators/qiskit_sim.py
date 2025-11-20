# qcgpt/simulators/qiskit_sim.py

from typing import List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

from ..circuits import Circuit
from ..gates import Gate


def circuit_to_qiskit(circ: Circuit) -> QuantumCircuit:
    """
    Convert our internal Circuit representation to a Qiskit QuantumCircuit.
    For now we assume no measurements; just a unitary circuit on n qubits.
    """
    qc = QuantumCircuit(circ.nqubits)

    for gate in circ.gates:
        gt = gate.gate_type
        qs = gate.targets

        if gt == "ID":
            continue
        elif gt == "X":
            qc.x(qs[0])
        elif gt == "Y":
            qc.y(qs[0])
        elif gt == "Z":
            qc.z(qs[0])
        elif gt == "H":
            qc.h(qs[0])
        elif gt == "S":
            qc.s(qs[0])
        elif gt == "T":
            qc.t(qs[0])
        elif gt == "CX":
            qc.cx(qs[0], qs[1])
        elif gt == "CZ":
            qc.cz(qs[0], qs[1])
        elif gt == "SWAP":
            qc.swap(qs[0], qs[1])
        elif gt == "CCX":
            qc.ccx(qs[0], qs[1], qs[2])
        elif gt == "CSWAP":
            qc.cswap(qs[0], qs[1], qs[2])
        elif gt == "CCZ":
            qc.h(qs[2]); qc.ccx(qs[0], qs[1], qs[2]); qc.h(qs[2])
        elif gt.startswith("RX_"):
            if gt == "RX_PI_16":
                qc.rx(np.pi/16, qs[0])
            elif gt == "RX_PI_8":
                qc.rx(np.pi/8, qs[0])
            elif gt == "RX_PI_4":
                qc.rx(np.pi/4, qs[0])
            elif gt == "RX_PI_2":
                qc.rx(np.pi/2, qs[0])
            elif gt == "RX_PI":
                qc.rx(np.pi, qs[0])
            else:
                raise ValueError(f"Unsupported RX variant: {gt}")
        elif gt.startswith("RY_"):
            if gt == "RY_PI_16":
                qc.ry(np.pi/16, qs[0])
            elif gt == "RY_PI_8":
                qc.ry(np.pi/8, qs[0])
            elif gt == "RY_PI_4":
                qc.ry(np.pi/4, qs[0])
            elif gt == "RY_PI_2":
                qc.ry(np.pi/2, qs[0])
            elif gt == "RY_PI":
                qc.ry(np.pi, qs[0])
            else:
                raise ValueError(f"Unsupported RY variant: {gt}")
        elif gt.startswith("RZ_"):
            if gt == "RZ_PI_16":
                qc.rz(np.pi/16, qs[0])
            elif gt == "RZ_PI_8":
                qc.rz(np.pi/8, qs[0])
            elif gt == "RZ_PI_4":
                qc.rz(np.pi/4, qs[0])
            elif gt == "RZ_PI_2":
                qc.rz(np.pi/2, qs[0])
            elif gt == "RZ_PI":
                qc.rz(np.pi, qs[0])
            else:
                raise ValueError(f"Unsupported RZ variant: {gt}")
        else:
            raise ValueError(f"Unsupported gate type for Qiskit: {gt}")

    return qc


def qiskit_to_circuit(qc: QuantumCircuit) -> Circuit:
    """
    Convert a Qiskit QuantumCircuit (restricted to our basis gates)
    into our internal Circuit representation.
    Uses the modern CircuitInstruction API.
    """
    circ = Circuit(nqubits=qc.num_qubits)
    for instr in qc.data:
        name = instr.operation.name.lower()
        qargs = instr.qubits
        qs = [qc.qubits.index(q) for q in qargs]
        if name in {"id", "x", "y", "z", "h", "s", "t"}:
            circ.add_gate(Gate(name.upper(), [qs[0]]))
        elif name in {"cx", "cz", "swap"}:
            circ.add_gate(Gate(name.upper(), [qs[0], qs[1]]))
        elif name in {"ccx", "cswap", "ccz"}:
            circ.add_gate(Gate(name.upper(), [qs[0], qs[1], qs[2]]))
        elif name in {"rx", "ry", "rz"}:
            theta = float(instr.operation.params[0])
            angles = [np.pi/16, np.pi/8, np.pi/4, np.pi/2, np.pi]
            tokens = {
                "rx": ["RX_PI_16", "RX_PI_8", "RX_PI_4", "RX_PI_2", "RX_PI"],
                "ry": ["RY_PI_16", "RY_PI_8", "RY_PI_4", "RY_PI_2", "RY_PI"],
                "rz": ["RZ_PI_16", "RZ_PI_8", "RZ_PI_4", "RZ_PI_2", "RZ_PI"],
            }
            diffs = [abs(theta - a) for a in angles]
            idx = int(np.argmin(diffs))
            circ.add_gate(Gate(tokens[name][idx], [qs[0]]))
        elif name in {"barrier", "measure", "reset", "snapshot"}:
            continue
        else:
            raise ValueError(f"Unsupported gate in qiskit_to_circuit: {name}")
    return circ


def basis_bits_to_statevector(bits: np.ndarray) -> Statevector:
    """
    bits: shape [nqubits], values {0,1}
    Returns the corresponding basis state |bits> as a Qiskit Statevector.

    Note: Qiskit uses little-endian ordering by default. For 2 qubits:
      |q1 q0>  corresponds to basis index (2*q1 + q0).
    Here we treat bits[0] as q0, bits[1] as q1, etc.
    """
    bits = np.asarray(bits, dtype=int)
    nqubits = bits.shape[0]
    index = 0
    for i in range(nqubits):
        index |= (bits[i] << i)  # q0 is least significant bit
    return Statevector.from_int(index, dims=2**nqubits)


def apply_circuit_to_state(
    circ: Circuit,
    psi_in: Statevector,
) -> Statevector:
    """
    Apply the Qiskit circuit corresponding to 'circ' to an input statevector.
    """
    qc = circuit_to_qiskit(circ)
    return psi_in.evolve(qc)


def average_basis_mapping_fidelity(
    spec_states: np.ndarray,
    circ: Circuit,
) -> float:
    """
    spec_states: [nstates, 2, nqubits], values {0,1}
      spec_states[i,0,:] = input bits
      spec_states[i,1,:] = desired output bits

    We interpret each pair as basis states |x_i>, |y_i> and compute:
      F_i = |<y_i | U | x_i>|^2
    Return the average over i.
    """
    nstates, two, nqubits = spec_states.shape
    assert two == 2

    qc = circuit_to_qiskit(circ)

    fids = []
    for i in range(nstates):
        x_bits = spec_states[i, 0, :]
        y_bits = spec_states[i, 1, :]

        psi_in = basis_bits_to_statevector(x_bits)
        psi_target = basis_bits_to_statevector(y_bits)

        psi_out = psi_in.evolve(qc)

        F = state_fidelity(psi_out, psi_target)
        fids.append(F)

    return float(np.mean(fids))


def average_state_fidelity_arbitrary(
    psi_in_list: List[Statevector],
    psi_out_target_list: List[Statevector],
    circ: Circuit,
) -> float:
    """
    More general: given arbitrary input/target statevectors,
    compute average fidelity of circ acting on psi_in_list vs psi_out_target_list.
    """
    assert len(psi_in_list) == len(psi_out_target_list)
    qc = circuit_to_qiskit(circ)

    fids = []
    for psi_in, psi_target in zip(psi_in_list, psi_out_target_list):
        psi_out = psi_in.evolve(qc)
        F = state_fidelity(psi_out, psi_target)
        fids.append(F)

    return float(np.mean(fids))
