from typing import Dict, List
from .gate_registry2 import build_vocab, build_gate_costs

SPECIAL_TOKENS = [
    "<PAD>", "<BOS_CIRC>", "<EOS_CIRC>",
]

ONEQ = ["ID", "X", "Y", "Z", "H", "S", "T",
         # RX/RY fine angles; drop PI (equivalent to X/Y up to global phase)
         "RX_PI_16", "RX_PI_8", "RX_PI_4", "RX_PI_2",
         "RY_PI_16", "RY_PI_8", "RY_PI_4", "RY_PI_2",
         # RZ keep fine angles; drop PI (Z), PI/2 (S), PI/4 (T)
         "RZ_PI_16", "RZ_PI_8"]
TWOQ_ORDERED = ["CX", "CZ", "SWAP"]
THREEQ = ["CCX", "CCZ", "CSWAP"]

def build_vocab_wrapper(n_qubits: int = 3) -> List[str]:
    return SPECIAL_TOKENS + build_vocab(n_qubits)

VOCAB2 = build_vocab_wrapper(3)
GATE_COST_REGISTRY = build_gate_costs(VOCAB2)
TOKEN_TO_ID2: Dict[str, int] = {tok: i for i, tok in enumerate(VOCAB2)}
ID_TO_TOKEN2: Dict[int, str] = {i: tok for tok, i in TOKEN_TO_ID2.items()}

PAD_ID2 = TOKEN_TO_ID2["<PAD>"]
BOS_CIRC_ID2 = TOKEN_TO_ID2["<BOS_CIRC>"]
EOS_CIRC_ID2 = TOKEN_TO_ID2["<EOS_CIRC>"]
