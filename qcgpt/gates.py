# qcgpt/gates.py
from dataclasses import dataclass
from typing import List, Dict

# Core gate types (discrete, no parameters for v0)
GATE_TYPES = [
    "ID",      # identity
    "X", "Y", "Z",
    "H", "S", "T",
    "CX", "CZ", "SWAP",
]

# Special tokens
SPECIAL_TOKENS = [
    "<PAD>",
    "<BOS_SPEC>", "<EOS_SPEC>",
    "<BOS_CIRC>", "<EOS_CIRC>",
    "->", ";",
]

# Qubit index tokens for 2 qubits
QUBIT_TOKENS = ["q0", "q1"]

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
        return 1 if self.gate_type in {"ID","X","Y","Z","H","S","T"} else 2
