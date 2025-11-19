# qcgpt/encoding.py
from typing import List, Dict
import numpy as np

from .gates import (
    TOKEN_TO_ID, ID_TO_TOKEN,
    BOS_SPEC_ID, EOS_SPEC_ID,
    BOS_CIRC_ID, EOS_CIRC_ID,
)

def spec_to_tokens(states: np.ndarray) -> List[int]:
    """
    states: shape [nstates, 2, nqubits], values 0/1
    Encodes as:
    <BOS_SPEC> x0_bits -> y0_bits ; x1_bits -> y1_bits ; ... <EOS_SPEC>
    """
    nstates, two, nqubits = states.shape
    assert two == 2
    tokens: List[int] = [BOS_SPEC_ID]

    for i in range(nstates):
        x_bits = states[i, 0, :]
        y_bits = states[i, 1, :]

        for b in x_bits:
            tokens.append(TOKEN_TO_ID[str(int(b))])
        tokens.append(TOKEN_TO_ID["->"])
        for b in y_bits:
            tokens.append(TOKEN_TO_ID[str(int(b))])
        tokens.append(TOKEN_TO_ID[";"])

    tokens.append(EOS_SPEC_ID)
    return tokens

def circuit_to_tokens(circ) -> List[int]:
    """
    Encode Circuit as:
    <BOS_CIRC> GATE q* q* ... <EOS_CIRC>
    For 1-qubit gates: GATE q
    For 2-qubit gates: GATE q_i q_j
    """
    from .gates import GATE_TYPES  # for sanity
    tokens = [BOS_CIRC_ID]
    for gate in circ.gates:
        assert gate.gate_type in GATE_TYPES
        tokens.append(TOKEN_TO_ID[gate.gate_type])
        for q in gate.targets:
            tokens.append(TOKEN_TO_ID[f"q{q}"])
    tokens.append(EOS_CIRC_ID)
    return tokens

def tokens_to_circuit(tokens) -> "Circuit":
    """Inverse of circuit_to_tokens."""
    from .gates import Gate
    from .circuits import Circuit

    # strip BOS/EOS
    # assume BOS_CIRC at pos 0, EOS somewhere later
    seq = [t for t in tokens if t != BOS_CIRC_ID and t != EOS_CIRC_ID]

    circ = Circuit(nqubits=2)
    i = 0
    while i < len(seq):
        tok = ID_TO_TOKEN.get(seq[i], None)
        if tok is None:
            i += 1
            continue
        if tok in {"X","Y","Z","H","S","T","ID"}:
            if i + 1 >= len(seq):
                break
            qtok = ID_TO_TOKEN.get(seq[i+1], "")
            if not (isinstance(qtok, str) and qtok.startswith("q")):
                i += 1
                continue
            try:
                q_idx = int(qtok[1])
            except Exception:
                i += 1
                continue
            circ.add_gate(Gate(tok, [q_idx]))
            i += 2
        elif tok in {"CX","CZ","SWAP"}:
            if i + 2 >= len(seq):
                break
            qtok1 = ID_TO_TOKEN.get(seq[i+1], "")
            qtok2 = ID_TO_TOKEN.get(seq[i+2], "")
            if not (isinstance(qtok1, str) and qtok1.startswith("q")):
                i += 1
                continue
            if not (isinstance(qtok2, str) and qtok2.startswith("q")):
                i += 1
                continue
            try:
                q1 = int(qtok1[1])
                q2 = int(qtok2[1])
            except Exception:
                i += 1
                continue
            circ.add_gate(Gate(tok, [q1, q2]))
            i += 3
        else:
            i += 1
            continue
    return circ
