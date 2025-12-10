"""QCGPT2 Vocabulary Mapping Utility

Prints a JSON list mapping each token to its gate type and targets.
Useful for inspecting the gate-with-target vocabulary.
"""
import json, os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root)
from qcgpt2.gates2 import VOCAB2
from qcgpt2.encoding2 import token_to_gate

def main():
    mapping = []
    for tok in VOCAB2:
        g = token_to_gate(tok)
        mapping.append({
            "token": tok,
            "gate_type": g.gate_type,
            "targets": g.targets,
        })
    print(json.dumps(mapping, indent=2))

if __name__ == "__main__":
    main()
