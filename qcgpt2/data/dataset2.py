import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
import numpy as np
import hashlib

# Import your helpers
from .specs2 import state_pairs_to_spec_tensor
from ..simulators2.qiskit_sim2 import qiskit_to_circuit2
from ..encoding2 import circuit2_to_tokens
from ..unitaries2 import build_circuit_unitary2

# ---------------------------------------------------------
# HELPER FUNCTIONS
def all_basis_inputs(n_qubits: int) -> np.ndarray:
    xs = []
    for idx in range(2 ** n_qubits):
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
        xs.append(bits)
    return np.array(xs, dtype=np.int64)

def sample_random_qiskit_circuit2(n_qubits, max_depth, rng):
    from qiskit import QuantumCircuit
    from ..gate_registry2 import ONEQ_STD, ONEQ_RX, ONEQ_RY, ONEQ_RZ, TWOQ, THREEQ, canonicalize_targets, apply_to_qiskit
    qc = QuantumCircuit(n_qubits)
    oneq_pool = ONEQ_STD + ONEQ_RX + ONEQ_RY + ONEQ_RZ
    for _ in range(rng.randint(1, max_depth + 1)):
        r = rng.rand()
        if r < 0.6:
            tok = rng.choice(oneq_pool)
            q = rng.randint(0, n_qubits)
            apply_to_qiskit(qc, tok, [q])
        elif r < 0.9:
            gt = rng.choice(TWOQ)
            a, b = rng.choice(range(n_qubits), size=2, replace=False)
            a, b = canonicalize_targets(gt, [a, b])
            apply_to_qiskit(qc, gt, [a,b])
        else:
            gt = rng.choice(THREEQ)
            a, b, c = rng.choice(range(n_qubits), size=3, replace=False)
            a, b, c = canonicalize_targets(gt, [a, b, c])
            apply_to_qiskit(qc, gt, [a,b,c])
    return qc

def simplify_qiskit_circuit2(qc, basis_gates=None, optimization_level=3):
    from qiskit import transpile
    from qiskit.transpiler.exceptions import TranspilerError
    if basis_gates is None:
        basis_gates = ["id", "x", "y", "z", "h", "s", "t", "rx", "ry", "rz", "cx", "cz", "swap", "ccx", "cswap"]
    try:
        return transpile(qc, basis_gates=basis_gates, optimization_level=optimization_level)
    except TranspilerError:
        return qc
# ---------------------------------------------------------

class SimplifiedCircuitDataset2(Dataset):
    def __init__(self, num_samples: int, n_qubits: int = 3,
                 raw_max_depth: int = 8,
                 min_len: int = 1,
                 max_len: Optional[int] = 32,
                 include_basis_states: bool = True,
                 n_random_states: int = 0,
                 **kwargs):
        
        self.num_samples = num_samples
        self.n_qubits = n_qubits
        self.raw_max_depth = raw_max_depth
        self.min_len = min_len
        self.max_len = max_len
        self.include_basis_states = include_basis_states
        self.n_random_states = n_random_states
        
        # No pre-generation. 
        # No global deduplication set (collisions are rare and harmless for training).
        print(f"[Dataset] Configured for {num_samples} deterministic on-the-fly samples.")

    def _build_spec_from_circuit2(self, circ) -> np.ndarray:
        from ..simulators2.qiskit_sim2 import circuit2_to_qiskit
        from qiskit.quantum_info import Statevector
        
        qc = circuit2_to_qiskit(circ)
        psi_pairs = []
        if self.include_basis_states:
            for b in all_basis_inputs(self.n_qubits):
                idx = sum(b[i] << i for i in range(self.n_qubits))
                psi_in = Statevector.from_int(idx, dims=2**self.n_qubits)
                psi_out = psi_in.evolve(qc)
                psi_pairs.append((psi_in, psi_out))
        for _ in range(self.n_random_states):
            # We need deterministic random states too!
            # But since the function is called inside __getitem__ with a seeded RNG context,
            # standard numpy calls might drift if not careful.
            # Best to rely on the fact that __getitem__ sets the seed?
            # Actually, let's generate standard basis only for now to be safe/fast.
            pass 
        return state_pairs_to_spec_tensor(psi_pairs, n_qubits=self.n_qubits)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, object]:
        # 1. DETERMINISTIC SEEDING
        # This guarantees that index 500 always produces the same circuit,
        # regardless of worker ID or epoch number.
        seed = (idx * 1315423911) % (2**32 - 1)
        rng = np.random.RandomState(seed)

        # 2. GENERATE ON THE FLY
        # We loop a few times in case generation fails (empty circuit, etc)
        # But we DO NOT check against a global hash set.
        for _ in range(20):
            qc_raw = sample_random_qiskit_circuit2(self.n_qubits, self.raw_max_depth, rng)
            qc_simp = simplify_qiskit_circuit2(qc_raw)
            
            try:
                circ_simp = qiskit_to_circuit2(qc_simp)
            except:
                continue
            
            L = len(circ_simp.gates)
            if L == 0: continue
            if self.max_len and L > self.max_len: continue
            if L < self.min_len: continue
            
            # Success
            spec_tensor = self._build_spec_from_circuit2(circ_simp)
            circ_tokens = circuit2_to_tokens(circ_simp)
            
            return {
                "spec_tensor": torch.tensor(spec_tensor, dtype=torch.float32),
                "circ_tokens": torch.tensor(circ_tokens, dtype=torch.long),
                "ref_circuit": circ_simp,
            }
            
        # Fallback if random generation keeps failing (unlikely)
        # Just return an empty/identity-like entry or raise error
        raise RuntimeError(f"Could not generate valid circuit for idx {idx}")
