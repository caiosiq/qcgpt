import numpy as np
from typing import List, Tuple


def state_pairs_to_spec_tensor(
    psi_pairs: List[Tuple["Statevector", "Statevector"]],
    n_qubits: int,
) -> np.ndarray:
    n_states = len(psi_pairs)
    dim = 2 ** n_qubits
    spec = np.zeros((n_states, 2, dim, 2), dtype=np.float32)
    for i, (psi_in, psi_out) in enumerate(psi_pairs):
        amp_in = np.asarray(psi_in.data, dtype=np.complex64)
        amp_out = np.asarray(psi_out.data, dtype=np.complex64)
        spec[i, 0, :, 0] = amp_in.real
        spec[i, 0, :, 1] = amp_in.imag
        spec[i, 1, :, 0] = amp_out.real
        spec[i, 1, :, 1] = amp_out.imag
    return spec


def build_spec_sequence_batch(
    spec_tensors: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    # Build pair-wise tokens: each pair encodes all basis amplitudes
    pairs: List[np.ndarray] = []
    for spec in spec_tensors:
        n_pairs, two, dim, two2 = spec.shape
        assert two == 2 and two2 == 2
        # Flatten per pair: [Re_in, Im_in, Re_out, Im_out] for all dim indices
        P = np.zeros((n_pairs, 4 * dim), dtype=np.float32)
        for i in range(n_pairs):
            # Interleaved flattening: [in(b0.real), in(b0.imag), in(b1.real), in(b1.imag), ..., out(...)]
            in_flat = spec[i, 0].reshape(-1)
            out_flat = spec[i, 1].reshape(-1)
            P[i, : in_flat.shape[0]] = in_flat
            P[i, in_flat.shape[0] : in_flat.shape[0] + out_flat.shape[0]] = out_flat
        pairs.append(P)
    L_max = max(p.shape[0] for p in pairs)
    B = len(pairs)
    batch = np.zeros((B, L_max, pairs[0].shape[1]), dtype=np.float32)
    pad_mask = np.ones((B, L_max), dtype=bool)
    for b, P in enumerate(pairs):
        L = P.shape[0]
        batch[b, :L, :] = P
        pad_mask[b, :L] = False
    return batch, pad_mask
