# qcgpt/circuits.py
from dataclasses import dataclass, field
from typing import List
import numpy as np

from .gates import Gate

@dataclass
class Circuit:
    nqubits: int = 3
    gates: List[Gate] = field(default_factory=list)

    def add_gate(self, gate: Gate):
        assert all(0 <= q < self.nqubits for q in gate.targets)
        self.gates.append(gate)

def apply_gate_to_bits(bits: np.ndarray, gate: Gate) -> np.ndarray:
    """bits: shape [2], values in {0,1} for 2 qubits."""
    res = bits.copy()
    gt = gate.gate_type
    qs = gate.targets

    if gt == "X":
        q = qs[0]; res[q] ^= 1

    elif gt == "Z":
        # Z changes phase only in computational basis; ignore at bit level
        pass

    elif gt == "H":
        # True H creates superpositions; for classical bits we treat as no-op
        pass

    elif gt in {"Y", "S", "T"}:
        # These are phase/superposition gates in the computational basis;
        # for bit-level mapping they don't change the observed bits.
        pass

    elif gt.startswith("RX_") or gt.startswith("RY_") or gt.startswith("RZ_"):
        pass

    elif gt == "CX":
        c, t = qs
        if res[c] == 1:
            res[t] ^= 1

    elif gt == "CZ":
        # Controlled-phase; no change in computational basis bits
        pass

    elif gt == "SWAP":
        q0, q1 = qs
        res[q0], res[q1] = res[q1], res[q0]

    elif gt == "ID":
        pass

    elif gt == "CCX":
        c1, c2, t = qs
        if res[c1] == 1 and res[c2] == 1:
            res[t] ^= 1

    elif gt == "CCZ":
        # Triple-controlled phase; no change in computational basis bits
        pass

    elif gt == "CSWAP":
        c, a, b = qs
        if res[c] == 1:
            res[a], res[b] = res[b], res[a]

    else:
        raise ValueError(f"Unsupported gate type in classical sim: {gt}")

    return res

def run_circuit_on_bitstrings(circ: Circuit, xs: np.ndarray) -> np.ndarray:
    """xs: [N, 2] array of bits."""
    ys = xs.copy()
    for gate in circ.gates:
        for i in range(xs.shape[0]):
            ys[i] = apply_gate_to_bits(ys[i], gate)
    return ys
