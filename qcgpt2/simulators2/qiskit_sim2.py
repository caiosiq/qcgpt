from typing import List, Dict
import numpy as np
import torch

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix

from ..circuits2 import Circuit2, Gate2
from ..gate_registry2 import canonicalize_targets, rotation_to_gate, ONEQ_STD, TWOQ, THREEQ
from ..gates2 import ID_TO_TOKEN2, VOCAB2


def circuit2_to_qiskit(circ: Circuit2) -> QuantumCircuit:
    qc = QuantumCircuit(circ.nqubits)
    for gate in circ.gates:
        gt = gate.gate_type
        qs = gate.targets
        if gt == "ID":
            continue
        elif gt == "X":
            qc.x(qs[0])
        elif gt == "Y":
            qc.y(qs[0])
        elif gt == "Z":
            qc.z(qs[0])
        elif gt == "H":
            qc.h(qs[0])
        elif gt == "S":
            qc.s(qs[0])
        elif gt == "T":
            qc.t(qs[0])
        elif gt == "CX":
            qc.cx(qs[0], qs[1])
        elif gt == "CZ":
            qc.cz(qs[0], qs[1])
        elif gt == "SWAP":
            qc.swap(qs[0], qs[1])
        elif gt == "CCX":
            if len(qs) < 3:
                continue
            qc.ccx(qs[0], qs[1], qs[2])
        elif gt == "CSWAP":
            if len(qs) < 3:
                continue
            qc.cswap(qs[0], qs[1], qs[2])
        elif gt == "CCZ":
            if len(qs) < 3:
                continue
            qc.h(qs[2]); qc.ccx(qs[0], qs[1], qs[2]); qc.h(qs[2])
        elif gt.startswith("RX_"):
            angles = {
                "RX_PI_16": np.pi/16, "RX_PI_8": np.pi/8, "RX_PI_4": np.pi/4,
                "RX_PI_2": np.pi/2, "RX_PI": np.pi,
            }
            qc.rx(angles[gt], qs[0])
        elif gt.startswith("RY_"):
            angles = {
                "RY_PI_16": np.pi/16, "RY_PI_8": np.pi/8, "RY_PI_4": np.pi/4,
                "RY_PI_2": np.pi/2, "RY_PI": np.pi,
            }
            qc.ry(angles[gt], qs[0])
        elif gt.startswith("RZ_"):
            angles = {
                "RZ_PI_16": np.pi/16, "RZ_PI_8": np.pi/8, "RZ_PI_4": np.pi/4,
                "RZ_PI_2": np.pi/2, "RZ_PI": np.pi,
            }
            qc.rz(angles[gt], qs[0])
        else:
            raise ValueError(f"Unsupported gate type for Qiskit: {gt}")
    return qc


def qiskit_to_circuit2(qc: QuantumCircuit) -> Circuit2:
    circ = Circuit2(nqubits=qc.num_qubits)
    for instr in qc.data:
        name = instr.operation.name.lower()
        qs = [qc.qubits.index(q) for q in instr.qubits]
        if name in {"id", "x", "y", "z", "h", "s", "t"}:
            circ.add_gate(Gate2(name.upper(), [qs[0]]))
        elif name in {"cx", "cz", "swap"}:
            a, b = qs[0], qs[1]
            gt = name.upper()
            if gt in {"CZ", "SWAP"} and a > b:
                a, b = b, a
            circ.add_gate(Gate2(gt, [a, b]))
        elif name == "ccx":
            targets = canonicalize_targets("CCX", [qs[0], qs[1], qs[2]])
            circ.add_gate(Gate2("CCX", targets))
        elif name == "cswap":
            targets = canonicalize_targets("CSWAP", [qs[0], qs[1], qs[2]])
            circ.add_gate(Gate2("CSWAP", targets))
        elif name == "ccz":
            targets = canonicalize_targets("CCZ", [qs[0], qs[1], qs[2]])
            circ.add_gate(Gate2("CCZ", targets))
        elif name in {"rx", "ry", "rz"}:
            theta = float(instr.operation.params[0])
            tok = rotation_to_gate(name, theta)
            circ.add_gate(Gate2(tok, [qs[0]]))
        elif name in {"barrier", "measure", "reset", "snapshot"}:
            continue
        else:
            raise ValueError(f"Unsupported gate in qiskit_to_circuit2: {name}")
    return circ


def basis_bits_to_statevector(bits: List[int]) -> Statevector:
    bits = np.asarray(bits, dtype=int)
    nqubits = bits.shape[0]
    index = 0
    for i in range(nqubits):
        index |= (bits[i] << i)
    return Statevector.from_int(index, dims=(2,) * nqubits)


def build_unitary_via_qiskit(circ: Circuit2, n_qubits: int) -> np.ndarray:
    qc = circuit2_to_qiskit(circ)
    D = 2 ** n_qubits
    U = np.zeros((D, D), dtype=np.complex64)
    for idx in range(D):
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
        psi_in = basis_bits_to_statevector(bits)
        psi_out = psi_in.evolve(qc)
        U[:, idx] = psi_out.data.astype(np.complex64)
    return U


def noisy_fidelity_vs_ideal(circ: Circuit2, n_qubits: int, cost_tensor: torch.Tensor) -> float:
    from qiskit import transpile, QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit.quantum_info import state_fidelity, Statevector
    import numpy as np

    qc = circuit2_to_qiskit(circ)
    nm = NoiseModel()
    
    # --- 1. Map Registry Costs to Gate Types ---
    gate_types = set(ONEQ_STD + ["RX", "RY", "RZ"] + TWOQ + THREEQ)
    type_to_ids: Dict[str, List[int]] = {}
    for tok_id, name in ID_TO_TOKEN2.items():
        parts = name.split("_")
        gt = parts[0]
        if gt in gate_types:
            type_to_ids.setdefault(gt, []).append(tok_id)
            
    type_to_cost: Dict[str, float] = {}
    for gt, ids in type_to_ids.items():
        if len(ids) == 0: continue
        # Average cost for this gate type (e.g. mean of RX(pi/2), RX(pi/4)...)
        vals = cost_tensor[torch.tensor(ids, dtype=torch.long, device=cost_tensor.device)].float()
        type_to_cost[gt] = float(vals.mean().item())

    # --- 2. Build Noise Model ---
    # We apply noise to the 'atomic' names. 
    # When Qiskit decomposes CSWAP -> CXs, the CSWAP error is ignored, 
    # but the CXs will pick up the 'cx' error defined below.
    for name, p in type_to_cost.items():
        name_lower = name.lower()
        if p > 0.0:
            if name_lower in {"x","y","z","h","s","t", "rx", "ry", "rz"}:
                nm.add_all_qubit_quantum_error(depolarizing_error(p*1/2, 1), [name_lower])
            elif name_lower in {"cx","cz","swap"}:
                nm.add_all_qubit_quantum_error(depolarizing_error(p*4/3, 2), [name_lower])
            elif name_lower in {"ccx", "cswap", "ccz"}:
                nm.add_all_qubit_quantum_error(depolarizing_error(p*8/7, 3), [name_lower])

    backend = AerSimulator(method="density_matrix", noise_model=nm)
    D = 2 ** n_qubits
    fids = []
    
    for idx in range(D):
        bits = [(idx >> i) & 1 for i in range(n_qubits)]
        psi_in = basis_bits_to_statevector(bits)
        ideal_out = psi_in.evolve(qc)
        
        qc_run = QuantumCircuit(n_qubits)
        qc_run.initialize(psi_in.data, list(range(n_qubits)))
        qc_run.compose(qc, inplace=True)
        try:
            qc_run.save_density_matrix()
        except Exception:
            return float("nan")
            
        # FIX: Standard transpile. 
        # Allows Qiskit to decompose CSWAP into CX/U3 which the simulator can handle.
        transpiled = transpile(qc_run, backend, optimization_level=0)
        
        result = backend.run(transpiled).result()
        try:
            # FIX: Correct data access for modern Qiskit Aer
            rho = result.data(0)['density_matrix']
        except KeyError:
            return float("nan")
            
        fids.append(state_fidelity(rho, Statevector(ideal_out)))
        
    return float(np.mean(fids))




# def noisy_fidelity_vs_ideal(circ: Circuit2, n_qubits: int, cost_tensor: torch.Tensor) -> float:
#     from qiskit import transpile, QuantumCircuit  # FIX: Import transpile
#     from qiskit_aer import AerSimulator
#     from qiskit_aer.noise import NoiseModel, depolarizing_error
#     from qiskit.quantum_info import state_fidelity, Statevector
#     import numpy as np

#     qc = circuit2_to_qiskit(circ)
#     nm = NoiseModel()
#     # Build cost per gate type from cost_tensor over VOCAB2
#     gate_types = set(ONEQ_STD + ["RX", "RY", "RZ"] + TWOQ + THREEQ)
#     type_to_ids: Dict[str, List[int]] = {}
#     for tok_id, name in ID_TO_TOKEN2.items():
#         parts = name.split("_")
#         gt = parts[0]
#         if gt in gate_types:
#             type_to_ids.setdefault(gt, []).append(tok_id)
#     type_to_cost: Dict[str, float] = {}
#     for gt, ids in type_to_ids.items():
#         if len(ids) == 0:
#             continue
#         vals = cost_tensor[torch.tensor(ids, dtype=torch.long, device=cost_tensor.device)].float()
#         type_to_cost[gt] = float(vals.mean().item())

#     names = set(inst.operation.name.lower() for inst in qc.data)
    
#     for name in names:
#         if name in {"x","y","z","h","s","t"}:
#             p = type_to_cost.get(name.upper(), 0.0)
#             if p > 0.0:
#                 # FIX: Use add_all_qubit_quantum_error for generic application
#                 error = depolarizing_error(p, 1)
#                 nm.add_all_qubit_quantum_error(error, [name])
                
#         elif name in {"rx","ry","rz"}:
#             p = type_to_cost.get(name.upper(), 0.0)
#             if p > 0.0:
#                 error = depolarizing_error(p, 1)
#                 nm.add_all_qubit_quantum_error(error, [name])
                
#         elif name in {"cx","cz","swap"}:
#             p = type_to_cost.get(name.upper(), 0.0)
#             if p > 0.0:
#                 # Note: Depolarizing error dimension must match gate (2 qubits)
#                 error = depolarizing_error(p, 2)
#                 nm.add_all_qubit_quantum_error(error, [name])

#     backend = AerSimulator(method="density_matrix", noise_model=nm)
#     D = 2 ** n_qubits
#     fids = []
    
#     for idx in range(D):
#         bits = [(idx >> i) & 1 for i in range(n_qubits)]
#         psi_in = basis_bits_to_statevector(bits)
#         ideal_out = psi_in.evolve(qc)
        
#         qc_run = QuantumCircuit(n_qubits)
#         qc_run.initialize(psi_in.data, list(range(n_qubits)))
#         qc_run.compose(qc, inplace=True)
        
#         try:
#             qc_run.save_density_matrix()
#         except Exception:
#             return float("nan")
            
#         # FIX: Use the standalone transpile function
#         transpiled = transpile(qc_run, backend)
        
#         result = backend.run(transpiled).result()
#         rho = result.data(0)['density_matrix']
#         fids.append(state_fidelity(rho, Statevector(ideal_out)))
#     return float(np.mean(fids))