import torch
import math
from typing import List, Optional
from abc import ABC, abstractmethod

# Assuming GateRegistry is in .gate_registry
# from .gate_registry import GateRegistry 

class UnitaryBackend(ABC):
    def __init__(self, registry, device=torch.device('cpu')):
        self.registry = registry
        self.device = device

    @abstractmethod
    def get_unitary_for_token_id(self, tok_id: int) -> torch.Tensor:
        pass

class TensorUnitaryBackend(UnitaryBackend):
    def __init__(self, registry, device=torch.device('cpu')):
        super().__init__(registry, device)
        self.dtype = torch.complex64
        
        # --- 1. Basic Gates ---
        self._I = torch.eye(2, dtype=self.dtype, device=device)
        self._X = torch.tensor([[0, 1], [1, 0]], dtype=self.dtype, device=device)
        self._Y = torch.tensor([[0, -1j], [1j, 0]], dtype=self.dtype, device=device)
        self._Z = torch.tensor([[1, 0], [0, -1]], dtype=self.dtype, device=device)
        self._H = (1.0 / math.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], dtype=self.dtype, device=device)
        
        # --- 2. Phase Gates & Adjoints ---
        # S = diag(1, i)
        self._S = torch.tensor([[1, 0], [0, 1j]], dtype=self.dtype, device=device)
        # S_dag = diag(1, -i)
        self._S_dag = torch.tensor([[1, 0], [0, -1j]], dtype=self.dtype, device=device)
        
        # T = diag(1, exp(i*pi/4))
        self._T = torch.tensor([[1, 0], [0, torch.exp(1j * torch.tensor(math.pi / 4))]], dtype=self.dtype, device=device)
        # T_dag = diag(1, exp(-i*pi/4))
        self._T_dag = torch.tensor([[1, 0], [0, torch.exp(-1j * torch.tensor(math.pi / 4))]], dtype=self.dtype, device=device)

        # --- 3. Hardware Native SX (Sqrt(X)) ---
        # SX = 0.5 * [[1+i, 1-i], [1-i, 1+i]]
        # This is the standard Qiskit definition: https://qiskit.org/documentation/stubs/qiskit.circuit.library.SXGate.html
        # SX = 1/2 * [ [1+i, 1-i], [1-i, 1+i] ]
        val_sx = 0.5 * torch.tensor([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=self.dtype, device=device)
        self._SX = val_sx
        
        # SX_dag
        # Adjoint of SX
        self._SX_dag = val_sx.conj().T.contiguous()

    def _lift_one_qubit(self, U1: torch.Tensor, target: int) -> torch.Tensor:
        """Lifts a 1Q gate to N-qubits (Qiskit Little-Endian Order)."""
        # Start with the MSB (qubit N-1)
        # If MSB is the target, use U1, else Identity.
        # Kron downwards to LSB (qubit 0).
        curr = U1 if (self.registry.n_qubits - 1) == target else self._I
        
        for q in range(self.registry.n_qubits - 2, -1, -1):
            next_gate = U1 if q == target else self._I
            curr = torch.kron(curr, next_gate)
        return curr

    def _lift_two_qubit(self, U2: torch.Tensor, targets: List[int]) -> torch.Tensor:
        """Lifts 4x4 matrix to N-qubits. Handles arbitrary control/target positions."""
        a, b = targets
        if a == b: return torch.eye(2 ** self.registry.n_qubits, dtype=self.dtype, device=self.device)
        
        D = 2 ** self.registry.n_qubits
        U = torch.zeros((D, D), dtype=self.dtype, device=self.device)
        
        # We iterate through all basis states |idx>
        for idx in range(D):
            # Extract bits of the current state index
            bits = [(idx >> i) & 1 for i in range(self.registry.n_qubits)]
            
            # Form the local 2-bit index (Control 'a' is MSB of local, Target 'b' is LSB)
            # This aligns with standard CX definitions where Control is top wire.
            local_idx = (bits[a] << 1) | bits[b]
            
            # Get the column from the small U2 matrix corresponding to this local state
            # This tells us what amplitude transitions to where locally
            col = U2[:, local_idx]
            
            # Place these amplitudes into the large matrix
            for out_local_idx in range(4):
                val = col[out_local_idx]
                if val == 0: continue
                
                # Construct the output global index
                out_bits = bits.copy()
                out_bits[a] = (out_local_idx >> 1) & 1
                out_bits[b] = out_local_idx & 1
                
                out_global_idx = 0
                for i in range(self.registry.n_qubits):
                    out_global_idx |= (out_bits[i] << i)
                
                U[out_global_idx, idx] += val
        return U

    def _permute_three_qubit(self, U3: torch.Tensor, targets: List[int]) -> torch.Tensor:
        """
        Lifts a generic 8x8 matrix (defined on q0, q1, q2) to specific targets in N-qubits.
        Currently optimized for N=3 cases (Permutation only).
        """
        if self.registry.n_qubits != 3:
             # Fallback for N > 3 would require a similar bit-logic to _lift_two_qubit
             # For QCGPT3 prototype, N=3 is the standard constraint.
             raise NotImplementedError("3Q gates currently only supported for N=3 systems.")

        D = 8
        P = torch.zeros((D, D), dtype=self.dtype, device=self.device)
        
        # targets = [phy_q0, phy_q1, phy_q2] for the gate's logical input
        # We build a permutation matrix P that swaps logical <-> physical
        
        for idx in range(D):
            bits = [(idx >> i) & 1 for i in range(3)]
            
            # Map logical bit 'k' to physical bit 'targets[k]'
            out_bits = [0, 0, 0]
            for logical_k in range(3):
                physical_k = targets[logical_k]
                out_bits[physical_k] = bits[logical_k]
                
            out_idx = out_bits[0] | (out_bits[1] << 1) | (out_bits[2] << 2)
            P[out_idx, idx] = 1.0
            
        # U_physical = P @ U_logical @ P.T
        return P @ U3 @ P.T

    def _get_base_gate(self, gate_type: str) -> torch.Tensor:
        # Standard
        if gate_type == "X": return self._X
        if gate_type == "Y": return self._Y
        if gate_type == "Z": return self._Z
        if gate_type == "H": return self._H
        
        # Phases
        if gate_type == "S": return self._S
        if gate_type == "S_dag": return self._S_dag # NEW
        if gate_type == "T": return self._T
        if gate_type == "T_dag": return self._T_dag # NEW
        
        # Native
        if gate_type == "SX": return self._SX # NEW
        if gate_type == "SX_dag": return self._SX_dag # NEW
        
        # Parameterized
        if gate_type.startswith("RX"):
            theta = self._parse_angle(gate_type)
            c = math.cos(theta / 2); s = -1j * math.sin(theta / 2)
            return torch.tensor([[c, s], [s, c]], dtype=self.dtype, device=self.device)
            
        if gate_type.startswith("RY"):
            theta = self._parse_angle(gate_type)
            c = math.cos(theta / 2); s = math.sin(theta / 2)
            return torch.tensor([[c, -s], [s, c]], dtype=self.dtype, device=self.device)
            
        if gate_type.startswith("RZ"):
            theta = self._parse_angle(gate_type)
            # RZ = diag(e^{-i t/2}, e^{i t/2})
            return torch.tensor([
                [torch.exp(-0.5j * torch.tensor(theta)), 0], 
                [0, torch.exp(0.5j * torch.tensor(theta))]
            ], dtype=self.dtype, device=self.device)
            
        raise ValueError(f"Unknown base gate: {gate_type}")

    def _parse_angle(self, gate_type: str) -> float:
        # Helper to clean up angle parsing
        angles = {
            "PI_16": math.pi/16, "PI_8": math.pi/8, 
            "PI_4": math.pi/4, "PI_2": math.pi/2, "PI": math.pi
        }
        # e.g., RX_PI_16 -> PI_16
        suffix = "_".join(gate_type.split("_")[1:])
        return angles.get(suffix, 0.0)

    def get_unitary_for_token_id(self, tok_id: int) -> torch.Tensor:
        name = self.registry.id_to_token[tok_id]
        if name in self.registry.special_tokens:
            return torch.eye(2 ** self.registry.n_qubits, dtype=self.dtype, device=self.device)
            
        gate_type, targets = self.registry.token_to_gate_parts(name)
        
        # 1. Single Qubit Logic
        if gate_type in ["ID", "X", "Y", "Z", "H", "S", "S_dag", "T", "T_dag", "SX", "SX_dag"] or gate_type.startswith(("RX", "RY", "RZ")):
             if gate_type == "ID": return torch.eye(2 ** self.registry.n_qubits, dtype=self.dtype, device=self.device)
             base = self._get_base_gate(gate_type)
             # Reshape base to ensure it is (2, 2)
             # _get_base_gate returns (2, 2), but verify.
             if base.dim() != 2:
                 base = base.reshape(2, 2)
             return self._lift_one_qubit(base, targets[0])
             
        # 2. Two Qubit Logic
        if gate_type in ["CX", "CZ", "SWAP"] or gate_type.startswith("CP"):
            if gate_type == "CX":
                # Standard CNOT: |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
                U = torch.eye(4, dtype=self.dtype, device=self.device)
                U[2,2]=0; U[2,3]=1; U[3,2]=1; U[3,3]=0
            elif gate_type == "CZ":
                U = torch.diag(torch.tensor([1, 1, 1, -1], dtype=self.dtype, device=self.device))
            elif gate_type == "SWAP":
                U = torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=self.dtype, device=self.device)
            elif gate_type.startswith("CP"):
                # Controlled Phase: diag(1, 1, 1, e^i theta)
                theta = self._parse_angle(gate_type)
                # Note: Qiskit 'cp' adds phase only to |11>
                phase = torch.exp(1j * torch.tensor(theta))
                U = torch.diag(torch.tensor([1, 1, 1, phase], dtype=self.dtype, device=self.device))
                
            return self._lift_two_qubit(U, targets)
            
        # 3. Three Qubit Logic
        if gate_type in ["CCX", "CCZ", "CSWAP"]:
            I8 = torch.eye(8, dtype=self.dtype, device=self.device)
            if gate_type == "CCX": # Toffoli
                U = I8.clone()
                # Toffoli is typically defined as Controlled-Controlled-NOT.
                # If we assume the gate logic is (c1, c2, t), and we use the
                # _permute_three_qubit function which maps logical (0, 1, 2) to physical targets,
                # then we must define the matrix U assuming logical bits 0 and 1 are controls,
                # and logical bit 2 is the target.
                
                # In standard binary order (q0, q1, q2) where q2 is MSB? 
                # No, usually Kronecker product order implies q0 is MSB or LSB depending on convention.
                # PyTorch/Numpy Kronecker: A kron B -> A is "top/left" (MSB).
                # Our _lift functions assume q(N-1) is MSB.
                
                # So for 3 qubits: q2 (MSB), q1, q0 (LSB).
                # Logical input to _permute is [c1, c2, t].
                # So logical q0=c1, q1=c2, q2=t.
                
                # Wait, _permute_three_qubit maps logical indices 0,1,2 to physical targets.
                # If we define U such that controls are on logical 0 and 1, and target on logical 2...
                
                # Let's check _permute_three_qubit logic:
                # targets = [phy_for_log0, phy_for_log1, phy_for_log2]
                
                # So if we define U for CCX(0,1->2):
                # We want the state where logical 0 is 1 AND logical 1 is 1 to flip logical 2.
                # Indices where (bit0=1, bit1=1):
                # 011 (3), 111 (7) ? No, bits are usually LSB at index 0 or right.
                
                # Let's stick to the convention in _lift_two_qubit:
                # bits = [(idx >> i) & 1 for i in range(N)] -> i=0 is LSB.
                
                # So logical 0 is LSB. Logical 2 is MSB.
                # bits[0]=1, bits[1]=1 -> flip bits[2].
                # Indices:
                # 011 (binary) = 3 -> 111 (7)
                # 110 (binary 6: b2=1,b1=1,b0=0) ? No.
                
                # Let's look at indices idx such that (idx & 1) and (idx & 2).
                # idx=3 (011): b0=1, b1=1, b2=0. -> Flip b2 -> 111 (7).
                # idx=7 (111): b0=1, b1=1, b2=1. -> Flip b2 -> 011 (3).
                
                # So we swap 3 and 7.
                U[3,3]=0; U[3,7]=1; U[7,3]=1; U[7,7]=0
                
            elif gate_type == "CCZ":
                # Apply Z to target (logical 2) if c1 (logical 0) and c2 (logical 1) are 1.
                # Phase flip on state |111> -> index 7.
                # Also |011> -> index 3? No, Z is diagonal.
                # Controlled-Controlled-Z is symmetric. Only |111> gets -1.
                U = I8.clone(); U[7,7] = -1
                
            elif gate_type == "CSWAP": # Fredkin
                # Controlled-Swap.
                # Usually Control is logical 0. Swap logical 1 and 2.
                # If bit0=1, swap bit1 and bit2.
                # States where bit0=1:
                # 1 (001): b0=1, b1=0, b2=0. No swap.
                # 3 (011): b0=1, b1=1, b2=0. Swap b1,b2 -> 101 (5).
                # 5 (101): b0=1, b1=0, b2=1. Swap b1,b2 -> 011 (3).
                # 7 (111): b0=1, b1=1, b2=1. No swap.
                
                # So we swap 3 and 5.
                U = I8.clone()
                U[3,3]=0; U[3,5]=1; U[5,3]=1; U[5,5]=0
            
            return self._permute_three_qubit(U, targets)
            
        return torch.eye(2 ** self.registry.n_qubits, dtype=self.dtype, device=self.device)