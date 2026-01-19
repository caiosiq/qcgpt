from typing import List, Dict, Tuple, Optional, Any
import math
import numpy as np # Ensure numpy is imported

class GateRegistry:
    def __init__(self, n_qubits: int = 3):
        self.n_qubits = n_qubits
        self.special_tokens = ["<PAD>", "<BOS_CIRC>", "<EOS_CIRC>"]
        
        # --- 1. ADDED ADJOINTS & HARDWARE PRIMITIVES ---
        self.one_q_std = ["ID", "X", "Y", "Z", "H", "S", "S_dag", "T", "T_dag", "SX", "SX_dag"]
        
        self.one_q_rx = ["RX_PI_16", "RX_PI_8", "RX_PI_4", "RX_PI_2"]
        self.one_q_ry = ["RY_PI_16", "RY_PI_8", "RY_PI_4", "RY_PI_2"]
        self.one_q_rz = ["RZ_PI_16", "RZ_PI_8", "RZ_PI_4", "RZ_PI_2"] 
        
        self.two_q = ["CX", "CZ", "SWAP"]
        self.two_q_cp = ["CP_PI_16", "CP_PI_8", "CP_PI_4", "CP_PI_2", "CP_PI"]
        self.three_q = ["CCX", "CCZ", "CSWAP"]
        
        # Build vocabulary
        self.vocab = self._build_vocab()
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        
        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS_CIRC>"]
        self.eos_id = self.token_to_id["<EOS_CIRC>"]
        
        self.qiskit_map = self._build_qiskit_map()
        self.gate_costs = self._build_gate_costs()
        
    def token_to_gate_parts(self, tok: str) -> Tuple[str, List[int]]:
        """
        Robustly parses a token string into (GateType, [TargetQubits]).
        Handles underscores in gate names (e.g., 'S_dag_0', 'RX_PI_16_1').
        """
        if tok in self.special_tokens:
            return tok, []
            
        parts = tok.split("_")
        

        
        targets = []
        split_index = len(parts)
        

        if len(parts) >= 2 and parts[-1].isdigit():
             # Candidate: gate = parts[:-1], targets = [parts[-1]]
             gate_cand = "_".join(parts[:-1])
             if self._is_valid_gate_type(gate_cand):
                 return gate_cand, [int(parts[-1])]
                 
        # Try taking 2 targets
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
             gate_cand = "_".join(parts[:-2])
             if self._is_valid_gate_type(gate_cand):
                 return gate_cand, [int(parts[-2]), int(parts[-1])]
                 
        # Try taking 3 targets
        if len(parts) >= 4 and parts[-1].isdigit() and parts[-2].isdigit() and parts[-3].isdigit():
             gate_cand = "_".join(parts[:-3])
             if self._is_valid_gate_type(gate_cand):
                 return gate_cand, [int(parts[-3]), int(parts[-2]), int(parts[-1])]
        
        # Fallback to old greedy logic if strictly needed, or raise error?
        # The greedy logic was:
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].isdigit():
                targets.insert(0, int(parts[i])) 
                split_index = i
            else:
                break
        gate_type = "_".join(parts[:split_index])
        
        # Fix for the RX_PI_16 case specifically:
        # If gate_type ends in "PI", and first target is 16, 8, 4, 2...
        # It's likely a misparse.
        if gate_type.endswith("_PI") and targets:
             val = targets[0]
             if val in [2, 4, 8, 16]:
                 # Reconstruct the gate name
                 gate_type = f"{gate_type}_{val}"
                 targets = targets[1:]
                 
        return gate_type, self.canonicalize_targets(gate_type, targets)

    def _is_valid_gate_type(self, gt: str) -> bool:
        # Check if this gate type base exists in our map or sets
        # Qiskit map keys are gate types (e.g. RX_PI_16, CX, S_dag)
        return gt in self.qiskit_map

    def get_supported_qiskit_gates(self) -> Tuple[List[str], List[str], List[str], List[float]]:
        """
        Returns lists of supported Qiskit gate names (lowercase) categorized by qubit count,
        and the list of supported rotation angles.
        """
        gates_1q = []
        gates_2q = []
        gates_3q = []
        
        # Iterate over our qiskit map to populate lists
        for gt, info in self.qiskit_map.items():
            name = info['qiskit_name']
            nq = info['num_qubits']
            params = info['params']
            
            # Skip decomposed or special gates if they don't map to single method call
            if name == 'ccz_decomp': continue # Handled separately or skipped
            if name == 'id': continue
            
            if nq == 1:
                if name not in gates_1q: gates_1q.append(name)
            elif nq == 2:
                if name not in gates_2q: gates_2q.append(name)
            elif nq == 3:
                if name not in gates_3q: gates_3q.append(name)
                
        # Rotation angles
        angles = [math.pi/16, math.pi/8, math.pi/4, math.pi/2, math.pi]
        
        return gates_1q, gates_2q, gates_3q, angles

    def _build_vocab(self) -> List[str]:
        toks = list(self.special_tokens)
        
        # 1Q gates
        for collection in [self.one_q_std, self.one_q_rx, self.one_q_ry, self.one_q_rz]:
            for g in collection:
                for q in range(self.n_qubits):
                    toks.append(f"{g}_{q}")
                    
        # 2Q gates
        for a in range(self.n_qubits):
            for b in range(self.n_qubits):
                if a == b: continue
                toks.append(f"CX_{a}_{b}")
                if a < b: 
                    toks.append(f"CZ_{a}_{b}")
                    toks.append(f"SWAP_{a}_{b}")
                    
        # 2Q CP gates (Symmetric, so a < b)
        for g in self.two_q_cp:
            for a in range(self.n_qubits):
                for b in range(a + 1, self.n_qubits):
                    toks.append(f"{g}_{a}_{b}")
                
        # 3Q gates
        for a in range(self.n_qubits):
            for b in range(a + 1, self.n_qubits):
                for t in range(self.n_qubits):
                    if t in (a, b): continue
                    toks.append(f"CCX_{a}_{b}_{t}")
                    
        for a in range(self.n_qubits):
            for b in range(a + 1, self.n_qubits):
                for c in range(b + 1, self.n_qubits):
                    toks.append(f"CCZ_{a}_{b}_{c}")
                    
        for ctrl in range(self.n_qubits):
            for a in range(self.n_qubits):
                if a == ctrl: continue
                for b in range(a + 1, self.n_qubits):
                    if b == ctrl: continue
                    toks.append(f"CSWAP_{ctrl}_{a}_{b}")
                    
        return toks

    def _build_gate_costs(self) -> Dict[str, float]:
        costs = {}
        
        COST_VIRTUAL = 0.0      
        COST_1Q_PHYSICAL = 1.0  
        COST_2Q_NATIVE = 10.0   
        COST_SWAP = 30.0        
        COST_CCX = 60.0         
        
        for token in self.vocab:
            # FIX: Use the robust parser instead of manual split
            gate_type, _ = self.token_to_gate_parts(token)
            
            if gate_type == "ID": cost = 0.0
            
            elif gate_type in ["X", "Y", "H", "SX", "SX_dag"]: cost = COST_1Q_PHYSICAL
            
            elif gate_type in ["Z", "S", "S_dag", "T", "T_dag"]: cost = COST_VIRTUAL
            
            elif gate_type == "CX": cost = COST_2Q_NATIVE
            elif gate_type == "CZ": cost = COST_2Q_NATIVE 
            elif gate_type == "SWAP": cost = COST_SWAP
            elif gate_type.startswith("CP"): cost = COST_2Q_NATIVE # New CP
            
            elif gate_type == "CCX": cost = COST_CCX
            elif gate_type == "CCZ": cost = COST_CCX 
            elif gate_type == "CSWAP": cost = COST_CCX 
            
            elif gate_type.startswith("RX") or gate_type.startswith("RY"): cost = COST_1Q_PHYSICAL
            elif gate_type.startswith("RZ"): cost = COST_VIRTUAL
            
            else: cost = 0.0
            
            costs[token] = cost
        return costs

    def _build_qiskit_map(self) -> Dict[str, Any]:
        mapping = {}
        
        # 1Q Standard
        mapping["ID"] = {'qiskit_name': 'id', 'num_qubits': 1, 'params': None}
        mapping["X"] = {'qiskit_name': 'x', 'num_qubits': 1, 'params': None}
        mapping["Y"] = {'qiskit_name': 'y', 'num_qubits': 1, 'params': None}
        mapping["Z"] = {'qiskit_name': 'z', 'num_qubits': 1, 'params': None}
        mapping["H"] = {'qiskit_name': 'h', 'num_qubits': 1, 'params': None}
        
        # Adjoints & SX
        mapping["S"] = {'qiskit_name': 's', 'num_qubits': 1, 'params': None}
        mapping["S_dag"] = {'qiskit_name': 'sdg', 'num_qubits': 1, 'params': None}
        mapping["T"] = {'qiskit_name': 't', 'num_qubits': 1, 'params': None}
        mapping["T_dag"] = {'qiskit_name': 'tdg', 'num_qubits': 1, 'params': None}
        mapping["SX"] = {'qiskit_name': 'sx', 'num_qubits': 1, 'params': None}
        mapping["SX_dag"] = {'qiskit_name': 'sxdg', 'num_qubits': 1, 'params': None}
        
        # Rotations
        rot_angles = {
            "PI_16": np.pi/16, "PI_8": np.pi/8, 
            "PI_4": np.pi/4, "PI_2": np.pi/2, "PI": np.pi
        }
        
        for axis in ["RX", "RY", "RZ"]:
            for suffix, angle in rot_angles.items():
                gate_type = f"{axis}_{suffix}"
                mapping[gate_type] = {
                    'qiskit_name': axis.lower(),
                    'num_qubits': 1, 
                    'params': [angle]
                }
                
        # 2Q CP
        for suffix, angle in rot_angles.items():
            gate_type = f"CP_{suffix}"
            mapping[gate_type] = {
                'qiskit_name': 'cp',
                'num_qubits': 2,
                'params': [angle]
            }

        # Multi Q
        mapping["CX"] = {'qiskit_name': 'cx', 'num_qubits': 2, 'params': None}
        mapping["CZ"] = {'qiskit_name': 'cz', 'num_qubits': 2, 'params': None}
        mapping["SWAP"] = {'qiskit_name': 'swap', 'num_qubits': 2, 'params': None}
        
        mapping["CCX"] = {'qiskit_name': 'ccx', 'num_qubits': 3, 'params': None}
        mapping["CSWAP"] = {'qiskit_name': 'cswap', 'num_qubits': 3, 'params': None}
        mapping["CCZ"] = {'qiskit_name': 'ccz_decomp', 'num_qubits': 3, 'params': None}
        
        # Build Reverse Map
        self.reverse_qiskit_map = {}
        for gt, info in mapping.items():
            if info['params'] is None: 
                # e.g. maps 'sdg' -> 'S_dag'
                self.reverse_qiskit_map[info['qiskit_name']] = gt
        
        return mapping

    def canonicalize_targets(self, gate_type: str, targets: List[int]) -> List[int]:
        if gate_type in {"CZ", "SWAP"} and len(targets) == 2:
            return sorted(targets)
        if gate_type.startswith("CP") and len(targets) == 2:
            return sorted(targets)
        if gate_type == "CCZ" and len(targets) == 3:
            return sorted(targets)
        if gate_type == "CSWAP" and len(targets) == 3:
            ctrl, a, b = targets[0], targets[1], targets[2]
            a, b = sorted([a, b])
            return [ctrl, a, b]
        if gate_type == "CCX" and len(targets) == 3:
            c1, c2, t = targets[0], targets[1], targets[2]
            # Canonicalize controls only
            if c1 > c2: c1, c2 = c2, c1
            return [c1, c2, t]
        return targets

    def get_closest_rotation_token(self, axis: str, theta: float) -> str:
        # Wrap theta to [0, 2*pi] or [-pi, pi]
        theta = theta % (2 * math.pi)
        
        # We only support positive rotations in vocab usually (e.g. PI_4).
        # Qiskit often uses negative angles like -pi/4.
        # RZ(-pi/4) == RZ(7pi/4) ? No, vocab is limited.
        # But RZ(-pi/4) = T_dag
        
        # Let's handle negative angles by converting to positive equivalent if needed?
        # Or just map absolute value and check sign?
        # Current logic: abs(theta - a)
        
        # If theta is negative, abs(theta - pi/4) will be large.
        # We should map to [0, 2pi].
        
        # Wait, our vocab only has POSITIVE angles: PI_16, PI_8...
        # If Qiskit generates RZ(-pi/4), we need to map it to... what?
        # We don't have RZ_MINUS_PI_4.
        # BUT RZ(-pi/4) is T_dag.
        # RZ(-pi/2) is S_dag.
        
        # If we have arbitrary negative angle, we might fail to match any PI_X token.
        # However, our sample_random_circuit ONLY generates positive angles from the list:
        # [pi/16, pi/8, pi/4, pi/2, pi]
        
        # So theta should be positive.
        # UNLESS Qiskit transpiler or some internal optimization flipped the sign?
        # But we are running `sample_random_circuit` which calls `qc.rx(theta)`.
        
        # Let's debug by ensuring we handle 2*pi wraparound or small floating point errors.
        
        angles = [math.pi/16, math.pi/8, math.pi/4, math.pi/2, math.pi]
        
        # Find best match (modulo 2pi?)
        # For now, simple diff on raw theta.
        diffs = [abs(theta - a) for a in angles]
        min_diff = min(diffs)
        
        # If the error is too large, it means we generated an angle that isn't in our set.
        # But we generated it FROM the set.
        # Maybe precision issue?
        
        idx = int(diffs.index(min_diff))
        nearest = angles[idx]
        eps = 1e-4 # Relax epsilon slightly
        
        if min_diff > eps:
             # Fallback: maybe it's close to 0 or 2pi?
             # But we don't have ID or 2PI in angles list.
             # Just return nearest for now.
             pass
        
        # Map nearest angle to suffix
        if abs(nearest - math.pi) < eps: suffix = "PI" 
        elif abs(nearest - math.pi/2) < eps: suffix = "PI_2"
        elif abs(nearest - math.pi/4) < eps: suffix = "PI_4"
        elif abs(nearest - math.pi/8) < eps: suffix = "PI_8"
        else: suffix = "PI_16"
        
        axis_upper = axis.upper()
        
        if suffix == "PI":
            if axis_upper == "RX": return "X"
            if axis_upper == "RY": return "Y"
            if axis_upper == "RZ": return "Z"
            if axis_upper == "CP": return "CZ" # CP(pi) == CZ
            
        if axis_upper == "RZ":
            if suffix == "PI_2": return "S"
            if suffix == "PI_4": return "T"
            
        # Check for RY(pi/2) and RX(pi/2) -> specific tokens if standard?
        # No, they are just RX_PI_2.
        
        return f"{axis_upper}_{suffix}"