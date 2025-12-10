from dataclasses import dataclass, field
from typing import List


@dataclass
class Gate2:
    gate_type: str
    targets: List[int]

    def arity(self) -> int:
        if self.gate_type in {"CX","CZ","SWAP"}:
            return 2
        if self.gate_type in {"CCX","CCZ","CSWAP"}:
            return 3
        return 1


@dataclass
class Circuit2:
    nqubits: int = 3
    gates: List[Gate2] = field(default_factory=list)

    def add_gate(self, gate: Gate2):
        assert all(0 <= q < self.nqubits for q in gate.targets)
        if len(set(gate.targets)) != len(gate.targets):
            raise ValueError("Repeated targets not allowed for multi-qubit gates")
        self.gates.append(gate)

