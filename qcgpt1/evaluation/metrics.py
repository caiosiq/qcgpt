# qcgpt/evaluation/metrics.py

from typing import Dict
import numpy as np

from ..circuits import Circuit
from ..simulators.qiskit_sim import circuit_to_qiskit
from qiskit.quantum_info import Statevector, state_fidelity


def quantum_fidelity_from_spec(spec_tensor: np.ndarray, circ: Circuit) -> float:
    qc = circuit_to_qiskit(circ)
    n_states = spec_tensor.shape[0]
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
    return float(np.mean(fids))


def gate_count(circ: Circuit) -> int:
    return len(circ.gates)


def summarize_metrics(spec_tensor: np.ndarray, circ: Circuit) -> Dict[str, float]:
    return {
        "fid_quantum": quantum_fidelity_from_spec(spec_tensor, circ),
        "gate_count": gate_count(circ),
    }
