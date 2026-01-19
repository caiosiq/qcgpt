from typing import List, Tuple

class Gate:
    def __init__(self, gate_type: str, targets: List[int]):
        self.gate_type = gate_type
        self.targets = targets

    def __repr__(self):
        return f"Gate({self.gate_type}, {self.targets})"

class Circuit:
    def __init__(self, n_qubits: int = 3):
        self.n_qubits = n_qubits
        self.gates: List[Gate] = []

    def add_gate(self, gate: Gate):
        self.gates.append(gate)

    def __repr__(self):
        return f"Circuit(n={self.n_qubits}, gates={len(self.gates)})"
