# qcgpt/data/dataset.py
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
from qiskit import QuantumCircuit

from .qiskit_utils import sample_task, sample_random_qiskit_circuit, simplify_qiskit_circuit
from ..simulators.qiskit_sim import qiskit_to_circuit
from ..encoding import circuit_to_tokens

class MappingCircuitDataset(Dataset):
    def __init__(self, size: int, max_gates: int = 6):
        self.size = size
        self.max_gates = max_gates

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, object]:
        spec_tensor, circ = sample_task(max_gates=self.max_gates)
        return {
            "spec_tensor": torch.tensor(spec_tensor, dtype=torch.float32),
            "ref_circuit": circ,
        }


class SimplifiedCircuitDataset(Dataset):
    """
    Dataset that yields (spec_tensor, teacher circuit tokens, ref_circuit) built from
    Qiskit-simplified circuits, with amplitude-based specs.

    Args:
      num_samples: total items to generate
      n_qubits: number of qubits (spec basis size = 2**n_qubits)
      raw_max_depth: upper bound on random raw circuit depth BEFORE Qiskit simplification
        - affects the distribution of teacher circuit lengths but does not directly change
          spec tensor shape
      min_len: minimum gate count in the simplified teacher circuit (filter)
      max_len: maximum gate count in the simplified teacher circuit (filter)
      include_basis_states: whether to include ALL computational basis inputs in the spec
        - if True, contributes 2**n_qubits state pairs
      n_random_states: additional random pure input states to include in spec
        - contributes n_random_states state pairs

    Spec shape:
      n_states = (2**n_qubits if include_basis_states else 0) + n_random_states
      spec_tensor has shape [n_states, 2, 2**n_qubits, 2]
        - second dim: (input, output)
        - last dim: (real, imag) amplitudes
    """
    def __init__(self, num_samples: int, n_qubits: int = 2,
                 raw_max_depth: int = 8,
                 min_len: int = 1,
                 max_len: Optional[int] = 32,
                 include_basis_states: bool = True,
                 n_random_states: int = 0):
        self.num_samples = num_samples
        self.n_qubits = n_qubits
        self.raw_max_depth = raw_max_depth
        self.min_len = min_len
        self.max_len = max_len
        self.include_basis_states = include_basis_states
        self.n_random_states = n_random_states

    def __len__(self):
        return self.num_samples

    def _sample_valid_simplified(self) -> Dict[str, object]:
        """
        Sample a raw circuit, simplify it, filter by length, then build:
          - spec_tensor using configured basis/random states
          - circ_tokens from the simplified teacher circuit
        Returns a dict suitable for collation.
        """
        for _ in range(20):
            qc_raw = sample_random_qiskit_circuit(n_qubits=self.n_qubits, max_depth=self.raw_max_depth)
            qc_simp = simplify_qiskit_circuit(qc_raw)
            try:
                circ_simp = qiskit_to_circuit(qc_simp)
            except Exception:
                continue
            L = len(circ_simp.gates)
            if L == 0:
                continue
            if self.max_len is not None and L > self.max_len:
                continue
            if L < self.min_len:
                continue
            # Build spec from simplified circuit
            from .qiskit_utils import build_spec_from_circuit
            spec_tensor = build_spec_from_circuit(
                circ_simp,
                n_qubits=self.n_qubits,
                use_basis_states=self.include_basis_states,
                n_random_states=self.n_random_states,
            )
            circ_tokens = circuit_to_tokens(circ_simp)
            return {
                "spec_tensor": torch.tensor(spec_tensor, dtype=torch.float32),
                "circ_tokens": torch.tensor(circ_tokens, dtype=torch.long),
                "ref_circuit": circ_simp,
            }
        # Fallback: use internal random circuit to ensure non-empty teacher
        from .qiskit_utils import build_spec_from_circuit
        from .qiskit_utils import random_reference_circuit
        fallback_circ = random_reference_circuit(max_gates=self.raw_max_depth)
        spec_tensor = build_spec_from_circuit(
            fallback_circ,
            n_qubits=self.n_qubits,
            use_basis_states=self.include_basis_states,
            n_random_states=self.n_random_states,
        )
        circ_tokens = circuit_to_tokens(fallback_circ)
        return {
            "spec_tensor": torch.tensor(spec_tensor, dtype=torch.float32),
            "circ_tokens": torch.tensor(circ_tokens, dtype=torch.long),
            "ref_circuit": fallback_circ,
        }

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """
        Returns:
          {
            "spec_tensor": float32 [n_states, 2, 2**n_qubits, 2],
            "circ_tokens": int64 token ids for teacher circuit,
            "ref_circuit": internal Circuit (teacher)
          }
        """
        return self._sample_valid_simplified()
