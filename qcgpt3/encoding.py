from typing import List
from .gate_registry import GateRegistry
from .circuits import Circuit, Gate

class CircuitEncoder:
    def __init__(self, registry: GateRegistry):
        self.registry = registry

    def encode(self, circuit: Circuit) -> List[int]:
        toks: List[int] = [self.registry.bos_id]
        for g in circuit.gates:
            targets = self.registry.canonicalize_targets(g.gate_type, g.targets)
            name = g.gate_type + ("_" + "_".join(str(q) for q in targets) if targets else "")
            if name in self.registry.token_to_id:
                toks.append(self.registry.token_to_id[name])
            else:
                # Handle unknown gate or raise error?
                # For now skip or maybe raise.
                raise ValueError(f"Gate {name} not in vocabulary")
        toks.append(self.registry.eos_id)
        return toks

    def decode(self, tokens: List[int]) -> Circuit:
        circ = Circuit(n_qubits=self.registry.n_qubits)
        for t in tokens:
            name = self.registry.id_to_token.get(t, "<PAD>")
            if name in self.registry.special_tokens:
                continue
            
            gate_type, targets = self.registry.token_to_gate_parts(name)
            circ.add_gate(Gate(gate_type, targets))
        return circ
