from typing import List
from .gates2 import TOKEN_TO_ID2, ID_TO_TOKEN2, BOS_CIRC_ID2, EOS_CIRC_ID2
from .gate_registry2 import canonicalize_targets
from .circuits2 import Circuit2, Gate2


def token_to_gate(tok: str) -> Gate2:
    parts = tok.split("_")
    if tok in {"<PAD>", "<BOS_CIRC>", "<EOS_CIRC>"}:
        return Gate2(tok, [])
    if parts[0] in {"RX", "RY", "RZ"} and len(parts) >= 4 and parts[1] == "PI":
        gate_type = f"{parts[0]}_PI_{parts[2]}"
        target = int(parts[3])
        return Gate2(gate_type, [target])
    if parts[0] in {"RX", "RY", "RZ"} and len(parts) == 3 and parts[1] == "PI":
        gate_type = f"{parts[0]}_PI"
        target = int(parts[2])
        return Gate2(gate_type, [target])
    gate_type = parts[0]
    if len(parts) == 2:
        return Gate2(gate_type, [int(parts[1])])
    if len(parts) == 3:
        a, b = int(parts[1]), int(parts[2])
        if gate_type in {"CZ", "SWAP"} and a > b:
            a, b = b, a
        return Gate2(gate_type, [a, b])
    if len(parts) == 4:
        a, b, c = int(parts[1]), int(parts[2]), int(parts[3])
        if gate_type == "CCZ":
            a, b, c = sorted([a, b, c])
            return Gate2(gate_type, [a, b, c])
        if gate_type == "CSWAP":
            ctrl, t1, t2 = a, b, c
            t1, t2 = sorted([t1, t2])
            return Gate2(gate_type, [ctrl, t1, t2])
        return Gate2(gate_type, [a, b, c])
    return Gate2(gate_type, [])


def tokens_to_circuit2(tokens: List[int], n_qubits: int = 3) -> Circuit2:
    circ = Circuit2(nqubits=n_qubits)
    seq_names = [ID_TO_TOKEN2[t] for t in tokens]
    for name in seq_names:
        if name in {"<PAD>", "<BOS_CIRC>", "<EOS_CIRC>"}:
            continue
        g = token_to_gate(name)
        circ.add_gate(g)
    return circ


def circuit2_to_tokens(circ: Circuit2) -> List[int]:
    toks: List[int] = [BOS_CIRC_ID2]
    for g in circ.gates:
        targets = canonicalize_targets(g.gate_type, g.targets)
        name = g.gate_type + ("_" + "_".join(str(q) for q in targets) if targets else "")
        toks.append(TOKEN_TO_ID2[name])
    toks.append(EOS_CIRC_ID2)
    return toks
