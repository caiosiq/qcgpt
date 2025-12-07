from typing import List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from ..circuits2 import Circuit2, Gate2
from ..gate_registry2 import canonicalize_targets, rotation_to_gate


def circuit2_to_qiskit(circ: Circuit2) -> QuantumCircuit:
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
            if len(qs) < 3:
                continue
            qc.ccx(qs[0], qs[1], qs[2])
        elif gt == "CSWAP":
            if len(qs) < 3:
                continue
            qc.cswap(qs[0], qs[1], qs[2])
        elif gt == "CCZ":
            if len(qs) < 3:
                continue
            qc.h(qs[2]); qc.ccx(qs[0], qs[1], qs[2]); qc.h(qs[2])
        elif gt.startswith("RX_"):
            angles = {
                "RX_PI_16": np.pi/16, "RX_PI_8": np.pi/8, "RX_PI_4": np.pi/4,
                "RX_PI_2": np.pi/2, "RX_PI": np.pi,
            }
            qc.rx(angles[gt], qs[0])
        elif gt.startswith("RY_"):
            angles = {
                "RY_PI_16": np.pi/16, "RY_PI_8": np.pi/8, "RY_PI_4": np.pi/4,
                "RY_PI_2": np.pi/2, "RY_PI": np.pi,
            }
            qc.ry(angles[gt], qs[0])
        elif gt.startswith("RZ_"):
            angles = {
                "RZ_PI_16": np.pi/16, "RZ_PI_8": np.pi/8, "RZ_PI_4": np.pi/4,
                "RZ_PI_2": np.pi/2, "RZ_PI": np.pi,
            }
            qc.rz(angles[gt], qs[0])
        else:
            raise ValueError(f"Unsupported gate type for Qiskit: {gt}")
    return qc


def qiskit_to_circuit2(qc: QuantumCircuit) -> Circuit2:
    circ = Circuit2(nqubits=qc.num_qubits)
    for instr in qc.data:
        name = instr.operation.name.lower()
        qs = [qc.qubits.index(q) for q in instr.qubits]
        if name in {"id", "x", "y", "z", "h", "s", "t"}:
            circ.add_gate(Gate2(name.upper(), [qs[0]]))
        elif name in {"cx", "cz", "swap"}:
            a, b = qs[0], qs[1]
            gt = name.upper()
            if gt in {"CZ", "SWAP"} and a > b:
                a, b = b, a
            circ.add_gate(Gate2(gt, [a, b]))
        elif name == "ccx":
            targets = canonicalize_targets("CCX", [qs[0], qs[1], qs[2]])
            circ.add_gate(Gate2("CCX", targets))
        elif name == "cswap":
            targets = canonicalize_targets("CSWAP", [qs[0], qs[1], qs[2]])
            circ.add_gate(Gate2("CSWAP", targets))
        elif name == "ccz":
            targets = canonicalize_targets("CCZ", [qs[0], qs[1], qs[2]])
            circ.add_gate(Gate2("CCZ", targets))
        elif name in {"rx", "ry", "rz"}:
            theta = float(instr.operation.params[0])
            tok = rotation_to_gate(name, theta)
            circ.add_gate(Gate2(tok, [qs[0]]))
        elif name in {"barrier", "measure", "reset", "snapshot"}:
            continue
        else:
            raise ValueError(f"Unsupported gate in qiskit_to_circuit2: {name}")
    return circ


def basis_bits_to_statevector(bits: List[int]) -> Statevector:
    bits = np.asarray(bits, dtype=int)
    nqubits = bits.shape[0]
    index = 0
    for i in range(nqubits):
        index |= (bits[i] << i)
    return Statevector.from_int(index, dims=(2,) * nqubits)
