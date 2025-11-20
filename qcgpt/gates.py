# qcgpt/gates.py
from dataclasses import dataclass
from typing import List, Dict

# Core gate types (discrete, no parameters for v0)
GATE_TYPES = [
    "ID",
    "X", "Y", "Z",
    "H", "S", "T",
    "RX_PI_16", "RX_PI_8", "RX_PI_4", "RX_PI_2", "RX_PI",
    "RY_PI_16", "RY_PI_8", "RY_PI_4", "RY_PI_2", "RY_PI",
    "RZ_PI_16", "RZ_PI_8", "RZ_PI_4", "RZ_PI_2", "RZ_PI",
    "CX", "CZ", "SWAP",
    "CCX", "CCZ", "CSWAP",
]

# Special tokens
SPECIAL_TOKENS = [
    "<PAD>",
    "<BOS_SPEC>", "<EOS_SPEC>",
    "<BOS_CIRC>", "<EOS_CIRC>",
    "->", ";",
]

# Qubit index tokens for 3 qubits
QUBIT_TOKENS = ["q0", "q1", "q2"]

# Bit tokens for 0/1 in mapping spec
BIT_TOKENS = ["0", "1"]

# Build vocabulary
VOCAB = SPECIAL_TOKENS + GATE_TYPES + QUBIT_TOKENS + BIT_TOKENS
TOKEN_TO_ID: Dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {i: tok for tok, i in TOKEN_TO_ID.items()}

PAD_ID        = TOKEN_TO_ID["<PAD>"]
BOS_SPEC_ID   = TOKEN_TO_ID["<BOS_SPEC>"]
EOS_SPEC_ID   = TOKEN_TO_ID["<EOS_SPEC>"]
BOS_CIRC_ID   = TOKEN_TO_ID["<BOS_CIRC>"]
EOS_CIRC_ID   = TOKEN_TO_ID["<EOS_CIRC>"]

@dataclass
class Gate:
    gate_type: str        # e.g., "X", "CX"
    targets: List[int]    # e.g., [0] or [0,1]

    def arity(self) -> int:
        if self.gate_type in {"CX","CZ","SWAP"}:
            return 2
        if self.gate_type in {"CCX","CCZ","CSWAP"}:
            return 3
        return 1
