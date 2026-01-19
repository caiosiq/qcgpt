from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import torch
import math

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix, partial_trace, entropy
from qiskit.transpiler.exceptions import TranspilerError

# Local Imports (Ensure these match your project structure)
from qcgpt3 import Circuit, Gate, GateRegistry

class QiskitEngine:
    """
    Encapsulates all interactions with Qiskit for validation and sanity checks.
    """
    def __init__(self, registry: GateRegistry):
        self.registry = registry

    # --- Core Conversions ---
    def circuit_to_qiskit(self, circ: Circuit) -> QuantumCircuit:
        """
        Converts internal Circuit representation to Qiskit QuantumCircuit.
        """
        qc = QuantumCircuit(circ.n_qubits)
        mapping = self.registry.qiskit_map
        
        for gate in circ.gates:
            gt = gate.gate_type
            qs = gate.targets
            
            if gt not in mapping:
                continue

            info = mapping[gt]
            q_name = info['qiskit_name']
            params = info['params']
            
            # Special Case: ID 
            if q_name == 'id' or gt == 'ID':
                continue
                
            # Special Case: Decomposed CCZ (H-CCX-H)
            if q_name == 'ccz_decomp':
                qc.h(qs[2])
                qc.ccx(qs[0], qs[1], qs[2])
                qc.h(qs[2])
                continue
            
            # Standard Application
            if hasattr(qc, q_name):
                method = getattr(qc, q_name)
                args = []
                if params is not None:
                    args.extend(params)
                args.extend(qs)
                method(*args)
                
        return qc

    def qiskit_to_circuit(self, qc: QuantumCircuit) -> Circuit:
        circ = Circuit(n_qubits=qc.num_qubits)
        reverse_map = self.registry.reverse_qiskit_map
        
        for instr in qc.data:
            name = instr.operation.name.lower()
            qs = [qc.qubits.index(q) for q in instr.qubits]
            
            # 1. Standard Gates (No Params)
            if name in reverse_map:
                # e.g., name='sdg' -> gt='S_dag'
                gt = reverse_map[name]
                targets = self.registry.canonicalize_targets(gt, qs)
                circ.add_gate(Gate(gt, targets))
                
            # 2. Rotations (With Params)
            elif name in {"rx", "ry", "rz", "cp"}:
                # Standard Qiskit rotation
                theta = float(instr.operation.params[0])
                if name == "cp":
                    tok = self.registry.get_closest_rotation_token("cp", theta)
                    circ.add_gate(Gate(tok, [qs[0], qs[1]]))
                else:
                    tok = self.registry.get_closest_rotation_token(name, theta)
                    circ.add_gate(Gate(tok, [qs[0]]))
                
            # 3. Special/Ignored
            elif name in {"barrier", "measure", "reset", "snapshot"}:
                continue
                
        return circ

    def simplify_circuit(self, circ: Circuit) -> Circuit:
        """
        Simplifies a circuit using Qiskit's transpiler (optimization level 3).
        Returns a new Circuit object with optimized gates mapped back to our registry.
        """
        qc = self.circuit_to_qiskit(circ)
        
        # Get basis gates from registry
        # We need a list of string names: ['cx', 'u3', 'id', ...]
        # QCGPT3 basis usually: ['cx', 'u3'] or specific set?
        # If we provide all our gates as basis gates, transpiler might use them.
        
        # However, Qiskit transpiler works best with standard basis gates like ['cx', 'u3']
        # and then we decompose/approximate back?
        
        # Or we can provide our full gate set if they are standard Qiskit gates.
        all_basis = []
        g1, g2, g3, _ = self.registry.get_supported_qiskit_gates()
        all_basis.extend(g1)
        all_basis.extend(g2)
        all_basis.extend(g3)
        
        # Transpile
        try:
            # optimization_level=3 is heavy optimization
            qc_simp = transpile(qc, basis_gates=all_basis, optimization_level=3)
        except TranspilerError:
            # Fallback if transpilation fails (e.g. empty circuit or weird constraints)
            return circ
            
        return self.qiskit_to_circuit(qc_simp)

    # --- Analysis & Validation Helpers ---

    def get_final_statevector(self, circ: Circuit) -> np.ndarray:
        qc = self.circuit_to_qiskit(circ)
        sv = Statevector.from_int(0, dims=(2,) * circ.n_qubits)
        return sv.evolve(qc).data

    def get_truth_table(self, circ: Circuit) -> np.ndarray:
        sv = self.get_final_statevector(circ)
        return np.abs(sv) ** 2

    # --- Noise & Fidelity ---

    def noisy_fidelity_vs_ideal(self, circ: Circuit, n_qubits: int, cost_tensor: torch.Tensor) -> float:
        """
        Runs the circuit on a noisy simulator where error rates are derived 
        from the cost_tensor (simulating the 'unlearned' noise).
        """
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel, depolarizing_error
        except ImportError:
            print("WARNING: qiskit-aer not installed. Skipping noise simulation.")
            return 1.0

        qc = self.circuit_to_qiskit(circ)
        nm = NoiseModel()
        
        # Build noise model from cost tensor
        vocab = self.registry.vocab
        type_to_ids: Dict[str, List[int]] = {}
        
        for tok_id, name in enumerate(vocab):
            # FIX: Use the robust parser here too!
            # Old code: gt = name.split("_")[0] -> would break on S_dag
            gt, _ = self.registry.token_to_gate_parts(name)
            
            type_to_ids.setdefault(gt, []).append(tok_id)
                
        # Calculate average cost per gate type
        type_to_cost: Dict[str, float] = {}
        for gt, ids in type_to_ids.items():
            if len(ids) == 0: continue
            vals = cost_tensor[torch.tensor(ids, dtype=torch.long, device=cost_tensor.device)].float()
            type_to_cost[gt] = float(vals.mean().item())

        # Apply noise
        # Keep track of which qiskit instructions we've already added error to
        added_errors = set()

        for name, p in type_to_cost.items():
            # Qiskit names are lowercase, Registry names are Upper (mostly)
            # We need to map Registry Name -> Qiskit Name for the noise model
            if name not in self.registry.qiskit_map:
                continue
                
            qiskit_name = self.registry.qiskit_map[name]['qiskit_name']
            if qiskit_name == 'ccz_decomp': qiskit_name = 'ccx' # Approximation
            
            # Avoid adding multiple errors for same instruction (e.g. RX_PI_2 vs RX_PI_4 both map to 'rx')
            if qiskit_name in added_errors:
                continue
            
            if p > 0.0:
                if self.registry.qiskit_map[name]['num_qubits'] == 2:
                     nm.add_all_qubit_quantum_error(depolarizing_error(min(p, 1.0), 2), [qiskit_name])
                elif self.registry.qiskit_map[name]['num_qubits'] == 3:
                     nm.add_all_qubit_quantum_error(depolarizing_error(min(p*1.5, 1.0), 3), [qiskit_name])
                elif qiskit_name not in {"id", "barrier", "measure"}: 
                     nm.add_all_qubit_quantum_error(depolarizing_error(min(p/10, 1.0), 1), [qiskit_name])
                
                added_errors.add(qiskit_name)

        backend = AerSimulator(method="density_matrix", noise_model=nm)
        D = 2 ** n_qubits
        fids = []
        
        check_indices = [0, D-1] if D >= 2 else [0]
        if D > 4: check_indices = [0, D//2, D-1]

        for idx in check_indices:
            sv_in = Statevector.from_int(idx, dims=(2,) * n_qubits)
            ideal_out = sv_in.evolve(qc)
            
            qc_run = QuantumCircuit(n_qubits)
            qc_run.initialize(sv_in.data, list(range(n_qubits)))
            qc_run.compose(qc, inplace=True)
            qc_run.save_density_matrix()
            
            transpiled = transpile(qc_run, backend, optimization_level=0)
            result = backend.run(transpiled).result()
            
            if not result.success: return 0.0
            
            rho = result.data(0)['density_matrix']
            fids.append(state_fidelity(rho, Statevector(ideal_out)))
            
        return float(np.mean(fids))

    def sample_random_circuit(self, n_qubits: int, max_depth: int, rng: np.random.RandomState) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        
        # Get supported gates from registry dynamically
        gates_1q, gates_2q, gates_3q, angles = self.registry.get_supported_qiskit_gates()
        
        for _ in range(max_depth):
            # Pick number of qubits involved (1, 2, or 3)
            # Bias towards 1q and 2q
            q_type = rng.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
            
            if q_type == 1:
                g = rng.choice(gates_1q)
                q = rng.randint(0, n_qubits)
                
                if g in ['rx', 'ry', 'rz']:
                    theta = rng.choice(angles)
                    getattr(qc, g)(theta, q)
                else:
                    getattr(qc, g)(q)
                    
            elif q_type == 2 and n_qubits >= 2:
                g = rng.choice(gates_2q)
                q1, q2 = rng.choice(n_qubits, 2, replace=False)
                
                if g == 'cp':
                     theta = rng.choice(angles)
                     getattr(qc, g)(theta, q1, q2)
                else:
                     getattr(qc, g)(q1, q2)
                
            elif q_type == 3 and n_qubits >= 3:
                g = rng.choice(gates_3q)
                q1, q2, q3 = rng.choice(n_qubits, 3, replace=False)
                getattr(qc, g)(q1, q2, q3)
                
        return qc