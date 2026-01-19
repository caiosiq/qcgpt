import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, List
import numpy as np
import hashlib

# Import your helpers
from .specs import state_pairs_to_spec_tensor
from ..simulators.qiskit_sim import QiskitEngine
from ..encoding import CircuitEncoder
from .. import GateRegistry, Circuit, Gate

# Helper registry
_REGISTRY = GateRegistry()
_ENCODER = CircuitEncoder(_REGISTRY)

# ---------------------------------------------------------
# HELPER FUNCTIONS
def all_basis_inputs(n_qubits: int) -> np.ndarray:
    xs = []
    for idx in range(2 ** n_qubits):
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
        xs.append(bits)
    return np.array(xs, dtype=np.int64)

# ---------------------------------------------------------

class HighPerformanceDataset(Dataset):
    def __init__(self, registry: GateRegistry, qdpe: Optional[object], num_samples: int, n_qubits: int = 3,
                 raw_max_depth: int = 8,
                 min_len: int = 1,
                 max_len: Optional[int] = 32,
                 include_basis_states: bool = True,
                 n_random_states: int = 0,
                 augment_commutation: bool = False,
                 augment_permutation: bool = False,
                 **kwargs):
        
        self.registry = registry
        # We need a CPU-based QDPE for the workers to use
        # If qdpe is provided, we try to extract its stack or create a new one on CPU
        # If qdpe is None, we create a fresh one.
        # Note: 'qdpe' argument might be the GPU instance from training script.
        # We should NOT store a CUDA tensor in the dataset if we use num_workers > 0
        
        from qcgpt3 import TensorUnitaryBackend, QDPE
        # We always create a fresh CPU backend/QDPE for the dataset
        cpu_backend = TensorUnitaryBackend(registry, device=torch.device('cpu'))
        self.cpu_qdpe = QDPE(registry, cpu_backend, device=torch.device('cpu'))
        
        # We also need an encoder
        self.encoder = CircuitEncoder(registry)
        
        # Qiskit Engine for generation
        self.qiskit_engine = QiskitEngine(registry)

        self.num_samples = num_samples
        self.n_qubits = n_qubits
        self.raw_max_depth = raw_max_depth
        self.min_len = min_len
        self.max_len = max_len
        self.include_basis_states = include_basis_states
        self.n_random_states = n_random_states
        
        self.augment_commutation = augment_commutation
        self.augment_permutation = augment_permutation
        
        print(f"[HighPerformanceDataset] Configured for {num_samples} samples. QDPE-accelerated.")

    def _build_spec_from_circuit_qdpe(self, circ) -> np.ndarray:
        # 1. Encode circuit
        tokens = self.encoder.encode(circ)
        tokens_t = torch.tensor(tokens, dtype=torch.long)
        
        # 2. Compute Unitary (Physics)
        # Returns (D, D) Complex Tensor
        U = self.cpu_qdpe.compute_unitary_from_tokens(tokens_t)
        U_np = U.numpy() # (Rows, Cols)
        
        dim = U_np.shape[0]
        
        # 3. Vectorized Spec Construction
        # We need shape: (n_pairs, 2 (in/out), dim, 2 (re/im))
        # n_pairs = dim (since we use all basis states)
        
        # INPUT STATE: Identity Matrix (Transposed so rows = specific basis state)
        # I_basis shape: (dim, dim) -> Row k is the state |k>
        I_basis = np.eye(dim, dtype=np.complex64) 
        
        # OUTPUT STATE: The Unitary (Transposed)
        # The output of input |k> is the k-th column of U. 
        # We want that in the k-th row of our data tensor.
        U_basis = U_np.T 
        
        # Stack Input and Output along axis 1
        # Shape: (dim, 2, dim)
        pairs_complex = np.stack([I_basis, U_basis], axis=1)
        
        # Split Real and Imaginary
        # Shape: (dim, 2, dim, 2)
        spec = np.stack([pairs_complex.real, pairs_complex.imag], axis=-1)
        
        return spec.astype(np.float32)

    def apply_commutation_jitter(self, circ: Circuit, rng: np.random.RandomState):
        """
        Swaps adjacent gates if they act on disjoint qubits.
        Iterates through the circuit multiple times or just once? 
        A single pass with random swaps is usually sufficient for augmentation.
        """
        if len(circ.gates) < 2:
            return circ
        
        # We can do a few passes or a probability based single pass.
        # Let's do a single probabilistic pass.
        gates = list(circ.gates)
        # Iterate from 0 to N-2
        for i in range(len(gates) - 1):
            g1 = gates[i]
            g2 = gates[i+1]
            
            # Check disjointness
            # We assume targets are integers.
            s1 = set(g1.targets)
            s2 = set(g2.targets)
            
            if s1.isdisjoint(s2):
                # Flip a coin
                if rng.rand() < 0.5:
                    gates[i], gates[i+1] = gates[i+1], gates[i]
                    
        circ.gates = gates
        return circ

    def _get_permuted_indices(self, perm):
        """
        Vectorized calculation of index permutation.
        perm: Array of shape (n_qubits,)
        """
        n = len(perm)
        # Create a matrix of bit values [1, 2, 4, 8...]
        powers = 1 << np.arange(n)
        
        # Create all indices [0...2^n-1]
        indices = np.arange(1 << n)
        
        # Expand bits: (2^n, n) matrix of 0s and 1s
        # checks if j-th bit is set in index i
        bits = (indices[:, None] >> np.arange(n)) & 1
        
        # Reconstruct indices using the permuted powers
        # We want the bit that WAS at 'i' to now contribute weight 2^{perm[i]}
        new_powers = 1 << perm
        
        # Sum (bits * new_weights)
        new_indices = np.sum(bits * new_powers, axis=1)
        return new_indices

    def apply_qubit_relabeling(self, circ: Circuit, spec_tensor: np.ndarray, rng: np.random.RandomState):
        """
        Permutes qubit labels in both circuit and spec_tensor.
        """
        # 1. Generate random permutation P
        # P[i] = new label for qubit i
        perm = rng.permutation(self.n_qubits)
        
        # 2. Apply to Circuit
        new_gates = []
        for g in circ.gates:
            new_targets = [perm[t] for t in g.targets]
            new_targets_canon = self.registry.canonicalize_targets(g.gate_type, new_targets)
            new_gates.append(Gate(g.gate_type, new_targets_canon))
            
        circ.gates = new_gates
        
        # 3. Apply to Spec Tensor
        # spec_tensor shape: (n_pairs, 2, dim, 2)
        # dim = 2**n_qubits
        # We need to permute the 'dim' axis.
        # The index k = sum(b_i * 2^i) maps to k' = sum(b_i * 2^{perm[i]})
        
        dim = 2 ** self.n_qubits        
        # Compute new indices
        new_indices = self._get_permuted_indices(perm)
        
        new_spec_tensor = np.zeros_like(spec_tensor)
        new_spec_tensor[:, :, new_indices, :] = spec_tensor
        
        return circ, new_spec_tensor

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, object]:
        # 1. DETERMINISTIC SEEDING
        seed = (idx * 1315423911) % (2**32 - 1)
        rng = np.random.RandomState(seed)

        # 2. GENERATE ON THE FLY
        for _ in range(20):
            qc_qiskit_raw = self.qiskit_engine.sample_random_circuit(self.n_qubits, self.raw_max_depth, rng)
            
            # Simplify Circuit: Takes Circuit -> Returns Circuit
            # But sample_random_circuit returns Qiskit QuantumCircuit
            # We need to convert first
            try:
                circ_raw = self.qiskit_engine.qiskit_to_circuit(qc_qiskit_raw)
                circ_simp = self.qiskit_engine.simplify_circuit(circ_raw)
            except:
                continue
            
            L = len(circ_simp.gates)
            if L == 0: continue
            if self.max_len and L > self.max_len: continue
            if L < self.min_len: continue
            
            # Success
            # Use QDPE to build spec
            spec_tensor = self._build_spec_from_circuit_qdpe(circ_simp)
            
            # --- AUGMENTATIONS ---
            if self.augment_commutation:
                circ_simp = self.apply_commutation_jitter(circ_simp, rng)
                
            if self.augment_permutation:
                circ_simp, spec_tensor = self.apply_qubit_relabeling(circ_simp, spec_tensor, rng)
            # ---------------------
            
            # Use encoder
            circ_tokens = self.encoder.encode(circ_simp)
            
            return {
                "spec_tensor": torch.tensor(spec_tensor, dtype=torch.float32),
                "circ_tokens": torch.tensor(circ_tokens, dtype=torch.long),
                "ref_circuit": circ_simp,
            }
            
        raise RuntimeError(f"Could not generate valid circuit for idx {idx}")
