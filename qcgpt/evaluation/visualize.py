# qcgpt/evaluation/visualize.py

from typing import List
import numpy as np

from ..circuits import Circuit
from ..gates import Gate


def bits_to_str(bits: np.ndarray) -> str:
    return "".join(str(int(b)) for b in bits)


def format_mapping_spec(spec_states: np.ndarray) -> str:
    """
    spec_states: [nstates, 2, nqubits]
    Returns a multiline string like:
      00 -> 01
      01 -> 10
      ...
    """
    nstates, two, nqubits = spec_states.shape
    lines = []
    for i in range(nstates):
        x = bits_to_str(spec_states[i, 0, :])
        y = bits_to_str(spec_states[i, 1, :])
        lines.append(f"{x} -> {y}")
    return "\n".join(lines)


def format_circuit(circ: Circuit) -> str:
    """
    Simple textual representation:
      0: H q0
      1: CX q0 q1
      ...
    """
    lines: List[str] = []
    for idx, gate in enumerate(circ.gates):
        if len(gate.targets) == 1:
            lines.append(f"{idx:02d}: {gate.gate_type} q{gate.targets[0]}")
        else:
            t_str = " ".join(f"q{q}" for q in gate.targets)
            lines.append(f"{idx:02d}: {gate.gate_type} {t_str}")
    if not lines:
        return "<empty circuit>"
    return "\n".join(lines)
