# qcgpt/training/rollouts.py
import torch
import numpy as np
from typing import List, Tuple

from ..models.policy import CircuitPolicy
from ..encoding import tokens_to_circuit
from ..data.qiskit_utils import sample_task
from ..data.specs import build_spec_sequence_batch
from ..gates import PAD_ID, BOS_CIRC_ID, EOS_CIRC_ID
from ..simulators.qiskit_sim import circuit_to_qiskit
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
try:
    from qiskit_aer import AerSimulator
    from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
    AER_AVAILABLE = True
except Exception:
    AER_AVAILABLE = False


def build_batch_specs(batch_size: int, max_gates_ref: int = 6):
    spec_list = []
    for _ in range(batch_size):
        spec_tensor, _ = sample_task(max_gates=max_gates_ref)
        spec_list.append(spec_tensor)
    spec_batch_np, spec_pad_mask_np = build_spec_sequence_batch(spec_list)
    spec_states_batch = np.stack(spec_list, axis=0)
    spec_batch = torch.tensor(spec_batch_np, dtype=torch.float32)
    spec_pad_mask = torch.tensor(spec_pad_mask_np, dtype=torch.bool)
    return spec_states_batch, spec_batch, spec_pad_mask


def compute_reward_for_circuit(
    spec_tensor: np.ndarray,
    circ,
    lambda_len: float = 0.1,
) -> float:
    n_states, _, dim, _ = spec_tensor.shape
    qc = circuit_to_qiskit(circ)
    fids = []
    for i in range(n_states):
        rin = spec_tensor[i, 0, :, 0]
        iin = spec_tensor[i, 0, :, 1]
        rout = spec_tensor[i, 1, :, 0]
        iout = spec_tensor[i, 1, :, 1]
        psi_in = rin.astype(np.float32) + 1j * iin.astype(np.float32)
        psi_out_target = rout.astype(np.float32) + 1j * iout.astype(np.float32)
        psi_out_pred = Statevector(psi_in).evolve(qc)
        fids.append(state_fidelity(Statevector(psi_out_target), psi_out_pred))
    F = float(np.mean(fids))
    length_penalty = lambda_len * len(circ.gates)
    return F - length_penalty


def _build_noise_model(p1: float, p2: float):
    if not AER_AVAILABLE:
        return None
    nm = NoiseModel()
    if p1 and p1 > 0:
        nm.add_quantum_error(depolarizing_error(p1, 1), ["x","y","z","h","s","t"])
    if p2 and p2 > 0:
        nm.add_quantum_error(depolarizing_error(p2, 2), ["cx","cz","swap"])
    return nm

def _simulate_output_with_blackbox(circ, psi_in: np.ndarray, method: str, noise_model):
    qc = circuit_to_qiskit(circ)
    base = Statevector(psi_in)
    if not AER_AVAILABLE or (not noise_model and method == "statevector"):
        return base.evolve(qc)
    backend = AerSimulator(method=method, noise_model=noise_model) if noise_model else AerSimulator(method=method)
    from qiskit import QuantumCircuit
    qcall = QuantumCircuit(circ.nqubits)
    qcall.initialize(psi_in, list(range(circ.nqubits)))
    qcall.compose(qc, inplace=True)
    if method == "density_matrix":
        try:
            qcall.save_density_matrix()
            res = backend.run(qcall).result()
            rho = res.data(0)["density_matrix"]
            return DensityMatrix(rho)
        except Exception:
            return base.evolve(qc)
    qcall.save_statevector()
    res = backend.run(qcall).result()
    sv = res.data(0)["statevector"]
    return Statevector(sv)

def compute_reward_qiskit_blackbox(
    spec_tensor: np.ndarray,
    circ,
    lambda_len: float,
    method: str = "statevector",
    use_noise: bool = False,
    p1: float = 0.0,
    p2: float = 0.0,
) -> float:
    noise_model = _build_noise_model(p1, p2) if use_noise else None
    n_states = spec_tensor.shape[0]
    fids = []
    for i in range(n_states):
        rin = spec_tensor[i, 0, :, 0]
        iin = spec_tensor[i, 0, :, 1]
        rout = spec_tensor[i, 1, :, 0]
        iout = spec_tensor[i, 1, :, 1]
        psi_in = rin.astype(np.float32) + 1j * iin.astype(np.float32)
        psi_target = rout.astype(np.float32) + 1j * iout.astype(np.float32)
        pred = _simulate_output_with_blackbox(circ, psi_in, method, noise_model)
        fids.append(state_fidelity(pred, Statevector(psi_target)))
    F = float(np.mean(fids))
    return F - lambda_len * len(circ.gates)


class RewardBaseline:
    def __init__(self, momentum: float = 0.9):
        self.value = 0.0
        self.momentum = momentum

    def update(self, batch_mean_reward: float):
        self.value = self.momentum * self.value + (1 - self.momentum) * batch_mean_reward
