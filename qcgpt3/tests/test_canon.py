import torch
from qcgpt2.gates2 import VOCAB2, PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2
from qcgpt2.encoding2 import token_to_gate, circuit2_to_tokens, tokens_to_circuit2
from qcgpt2.data.qiskit_utils2 import sample_task2
from qcgpt2.unitaries2 import build_circuit_unitary2

def is_canonical(tok: str) -> bool:
    parts = tok.split("_")
    gt = parts[0]
    if gt == "CCZ" and len(parts) == 4:
        a,b,c = int(parts[1]), int(parts[2]), int(parts[3])
        return [a,b,c] == sorted([a,b,c])
    if gt == "CSWAP" and len(parts) == 4:
        ctrl,a,b = int(parts[1]), int(parts[2]), int(parts[3])
        return a < b and ctrl not in (a,b)
    if gt == "CCX" and len(parts) == 4:
        a,b,t = int(parts[1]), int(parts[2]), int(parts[3])
        return a < b and t not in (a,b)
    return True

def test_vocab_is_canonical():
    for tok in VOCAB2:
        assert is_canonical(tok)

def test_dataset_tokens_canonical_and_unitary():
    spec, circ = sample_task2(max_gates=4)
    toks = circuit2_to_tokens(circ)
    toks_clean = [t for t in toks if t not in (PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2)]
    names = [VOCAB2[t] for t in toks_clean]
    for n in names:
        assert is_canonical(n)
    circ2 = tokens_to_circuit2(toks_clean)
    U1 = build_circuit_unitary2(circ,3)
    U2 = build_circuit_unitary2(circ2,3)
    tr = torch.einsum("ij,ij->", U1.conj(), U2)
    fid = (tr.abs()**2)/(U1.size(0)**2)
    assert float(fid) > 0.999
