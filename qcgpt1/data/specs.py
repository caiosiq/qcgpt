import numpy as np
from typing import List, Tuple

def state_pairs_to_spec_tensor(
    psi_pairs: List[Tuple["Statevector", "Statevector"]],
    n_qubits: int,
) -> np.ndarray:
    """
    Convert a list of statevector pairs into an amplitude-based spec tensor.

    Args:
        psi_pairs: list of (psi_in, psi_out) Qiskit Statevectors
        n_qubits: number of qubits

    Returns:
        spec_tensor: float32 array of shape [n_states, 2, 2**n_qubits, 2]
        where the last dimension holds [real, imag] amplitudes.
    """
    n_states = len(psi_pairs)
    dim = 2 ** n_qubits
    spec = np.zeros((n_states, 2, dim, 2), dtype=np.float32)
    for i, (psi_in, psi_out) in enumerate(psi_pairs):
        amp_in = np.asarray(psi_in.data, dtype=np.complex64)
        amp_out = np.asarray(psi_out.data, dtype=np.complex64)
        assert amp_in.shape[0] == dim
        assert amp_out.shape[0] == dim
        spec[i, 0, :, 0] = amp_in.real
        spec[i, 0, :, 1] = amp_in.imag
        spec[i, 1, :, 0] = amp_out.real
        spec[i, 1, :, 1] = amp_out.imag
    return spec

def spec_tensor_to_state_pairs(
    spec_tensor: np.ndarray,
    n_qubits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Inverse of state_pairs_to_spec_tensor.

    Args:
        spec_tensor: float32 array [n_states, 2, 2**n_qubits, 2]
        n_qubits: number of qubits

    Returns:
        A list of (psi_in, psi_out) complex numpy arrays with shape [2**n_qubits].
    """
    dim = 2 ** n_qubits
    n_states = spec_tensor.shape[0]
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_states):
        rin = spec_tensor[i, 0, :, 0]
        iin = spec_tensor[i, 0, :, 1]
        rout = spec_tensor[i, 1, :, 0]
        iout = spec_tensor[i, 1, :, 1]
        psi_in = rin.astype(np.float32) + 1j * iin.astype(np.float32)
        psi_out = rout.astype(np.float32) + 1j * iout.astype(np.float32)
        assert psi_in.shape[0] == dim
        assert psi_out.shape[0] == dim
        out.append((psi_in, psi_out))
    return out

def build_spec_sequence_batch(
    spec_tensors: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert amplitude spec tensors into padded continuous sequences for the encoder.

    Args:
        spec_tensors: list of arrays each with shape [n_states, 2, n_basis, 2]

    Returns:
        spec_batch: float32 array [B, L_max, 4]
        spec_pad_mask: bool array [B, L_max] where True indicates padding.
    """
    seqs: List[np.ndarray] = []
    for spec in spec_tensors:
        n_states, _, n_basis, _ = spec.shape
        L = n_states * n_basis
        seq = np.zeros((L, 4), dtype=np.float32)
        t = 0
        for i in range(n_states):
            for j in range(n_basis):
                seq[t, 0] = spec[i, 0, j, 0]
                seq[t, 1] = spec[i, 0, j, 1]
                seq[t, 2] = spec[i, 1, j, 0]
                seq[t, 3] = spec[i, 1, j, 1]
                t += 1
        seqs.append(seq)
    L_max = max(s.shape[0] for s in seqs)
    B = len(seqs)
    batch = np.zeros((B, L_max, 4), dtype=np.float32)
    pad_mask = np.ones((B, L_max), dtype=bool)
    for b, s in enumerate(seqs):
        L = s.shape[0]
        batch[b, :L, :] = s
        pad_mask[b, :L] = False
    return batch, pad_mask