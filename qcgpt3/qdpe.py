import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from .gate_registry import GateRegistry
from .unitaries import UnitaryBackend

class QDPE:
    """
    Quantum Differentiable Physics Engine (QDPE).
    Handles the reconstruction of unitaries and noise from probability distributions over gates.
    """
    def __init__(self, 
                 registry: GateRegistry, 
                 backend: UnitaryBackend, 
                 device: torch.device = torch.device('cpu'),
                 noise_scale: float = 1.0):
        self.registry = registry
        self.backend = backend
        self.device = device
        self.noise_scale = noise_scale
        
        # Precompute stacks
        self.n_qubits = registry.n_qubits
        self.u_stack = self._build_u_stack()
        self.cost_tensor = self._build_cost_tensor()
        
        # Placeholder for H_stack if needed later
        self.h_stack: Optional[torch.Tensor] = None

    def _build_u_stack(self) -> torch.Tensor:
        mats = []
        for tok_id in range(len(self.registry.vocab)):
            mats.append(self.backend.get_unitary_for_token_id(tok_id))
        return torch.stack(mats, dim=0).to(self.device)

    def _build_cost_tensor(self) -> torch.Tensor:
        costs = [self.registry.gate_costs[tok] for tok in self.registry.vocab]
        return torch.tensor(costs, dtype=torch.float32, device=self.device)

    def ensure_h_stack(self):
        if self.h_stack is None:
            mats = []
            for tok_id in range(len(self.registry.vocab)):
                try:
                    mats.append(self.backend.get_hamiltonian_for_token_id(tok_id))
                except NotImplementedError:
                    # Fallback or zero? For now, let's fail or use zeros
                    # Assuming backend handles it or we skip this method
                    mats.append(torch.zeros((2**self.registry.n_qubits, 2**self.registry.n_qubits), 
                                          dtype=torch.complex64, device=self.device))
            self.h_stack = torch.stack(mats, dim=0).to(self.device)

    @staticmethod
    def parallel_unitary_product(seq: torch.Tensor) -> torch.Tensor:
        """
        Computes the product of a sequence of matrices using tree reduction.
        Input: (B, L, D, D)
        Output: (B, D, D)
        """
        B, L, D, _ = seq.shape
        target_L = 1
        while target_L < L:
            target_L *= 2
            
        if target_L > L:
            padding = torch.eye(D, dtype=seq.dtype, device=seq.device).view(1, 1, D, D)
            padding = padding.expand(B, target_L - L, D, D)
            seq = torch.cat([seq, padding], dim=1)
            
        current_seq = seq
        while current_seq.shape[1] > 1:
            left = current_seq[:, 0::2]
            right = current_seq[:, 1::2]
            current_seq = right @ left
            
        return current_seq.squeeze(1)

    def calculate_fidelity(self, 
                           probs: torch.Tensor, 
                           target_unitary: torch.Tensor,
                           method: str = "product") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the predicted unitary, noise penalty, and fidelity loss.
        
        Args:
            probs: (B, L, V) Soft probability distribution.
            target_unitary: (B, D, D) Target unitary matrix.
            method: "product" (default) or "hamiltonian_sum".
            
        Returns:
            U_pred: (B, D, D) Predicted unitary.
            fidelity_loss: Scalar tensor (1 - fidelity).
            noise_penalty: Scalar tensor (noise term).
        """
        # Calculate shared quantities
        B, L, V = probs.shape
        life_mask = self._get_life_mask(probs)
        
        # 1. Compute Noise
        gate_noise = self.compute_noise(probs, life_mask)
        
        # 2. Compute Unitary
        U_pred = self.compute_unitary(probs, life_mask, method=method)
        
        # 3. Fidelity
        trace = torch.einsum("bij,bij->b", target_unitary.conj(), U_pred)
        dim = target_unitary.size(-1)
        fidelity = (trace.abs() ** 2) / (dim ** 2)
        loss_U = 1.0 - fidelity.mean()
        
        return U_pred, loss_U, gate_noise

    def _get_life_mask(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Calculates the life mask (B, L, 1) from probabilities.
        """
        eos_id = self.registry.eos_id
        p_eos = probs[:, :, eos_id]
        
        # cumprod is stable enough for this sequence length
        life_mask = torch.cumprod(1.0 - p_eos, dim=1)
        life_mask = torch.roll(life_mask, shifts=1, dims=1)
        life_mask[:, 0] = 1.0
        return life_mask.unsqueeze(-1)

    def compute_noise(self, probs: torch.Tensor, life_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the noise penalty.
        """
        if life_mask is None:
            life_mask = self._get_life_mask(probs)
            
        step_costs = torch.einsum("blv,v->bl", probs, self.cost_tensor)
        total_noise = (step_costs * life_mask.squeeze(-1)).sum(dim=1) * self.noise_scale
        fidelity_loss_noise = 1.0 - torch.exp(-total_noise)
        return fidelity_loss_noise

    def compute_unitary(self, probs: torch.Tensor, life_mask: Optional[torch.Tensor] = None, method: str = "product") -> torch.Tensor:
        """
        Computes the final unitary matrix.
        """
        if life_mask is None:
            life_mask = self._get_life_mask(probs)
            
        B, L, V = probs.shape
        probs = probs.float()
        
        if method == "product":
            # Find Identity index (Fallback to PAD if not explicit, or 0)
            id_idx = getattr(self.registry, 'pad_id', 0) 
            
            # Create a "soft identity" distribution: [0, 0, ... 1, ... 0]
            I_dist = torch.zeros(V, device=probs.device, dtype=probs.dtype)
            I_dist[id_idx] = 1.0
            I_dist = I_dist.view(1, 1, V) # Broadcastable
            
            # Interpolate probabilities
            probs_effective = life_mask * probs + (1.0 - life_mask) * I_dist
            
            D = self.u_stack.size(-1)
            flat_probs = probs_effective.view(-1, V).to(self.u_stack.dtype)
            flat_stack = self.u_stack.view(V, -1)
            
            # This is the "Soft Gate" construction
            U_flat = torch.mm(flat_probs, flat_stack)
            U_seq = U_flat.view(B, L, D, D)
            
            # Parallel Product
            U_final = self.parallel_unitary_product(U_seq)

        elif method == "hamiltonian_sum":
            self.ensure_h_stack()
            
            probs_effective = probs * life_mask 
            
            H_seq = torch.einsum("blv,vij->blij", 
                               probs_effective.to(self.h_stack.dtype), 
                               self.h_stack)
            
            H_total = H_seq.sum(dim=1)
            U_final = torch.linalg.matrix_exp(-1j * H_total)
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return U_final

    def forward(self, probs: torch.Tensor, method: str = "product") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Legacy forward pass wrapper.
        """
        life_mask = self._get_life_mask(probs)
        noise = self.compute_noise(probs, life_mask)
        unitary = self.compute_unitary(probs, life_mask, method)
        return unitary, noise

    def compute_unitary_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Computes the final unitary from a sequence of token IDs.
        Input: tokens (L,) or (B, L) LongTensor
        Output: (D, D) or (B, D, D) ComplexTensor
        """
        is_batch = tokens.dim() == 2
        if not is_batch:
            tokens = tokens.unsqueeze(0)
            
        B, L = tokens.shape
        
        # We need to handle device mismatch if tokens are on CPU but u_stack on GPU
        device = self.u_stack.device
        if tokens.device != device:
            tokens = tokens.to(device)
            
        # Select: (B, L, D, D)
        # Advanced indexing: u_stack[tokens]
        U_seq = self.u_stack[tokens] 

        # 3. Product
        U_final = self.parallel_unitary_product(U_seq)
        
        if not is_batch:
            return U_final.squeeze(0)
        return U_final

    def compute_meyer_wallach(self, unitary_batch: torch.Tensor) -> torch.Tensor:
        """
        Vectorized Meyer-Wallach calculation.
        Computes entanglement for all qubits in parallel using a 'Super-Batch'.
        """
        # 1. Get State Vectors |psi> (B, Dim)
        psi = unitary_batch[:, :, 0] 
        B = psi.shape[0]
        n = self.n_qubits
        
        # Reshape to tensor of qubits: (B, q0, q1, ... qN)
        psi_tensor = psi.view([B] + [2] * n)

        # 2. Create the 'Super-Batch' via Stacking
        # We need to create N views of the state, where for the k-th view, 
        # the k-th qubit is moved to the front (to be kept) and others are pushed back (to be traced).
        # Note: torch.stack forces a copy, but this is necessary for contiguous memory in matmul anyway.
        
        permuted_states = []
        for k in range(n):
            # Target dims: (Batch, Qubit_k, Rest...)
            dims = [0, k + 1] + [i + 1 for i in range(n) if i != k]
            permuted_states.append(psi_tensor.permute(dims))
            
        # Stack shape: (B, n, 2, 2, ..., 2)
        # We merge B and n into a single batch dimension for the matmul
        stacked = torch.stack(permuted_states, dim=1)
        
        # Flatten to: (B*n, 2, 2^(n-1))
        # This prepares us for: rho = M @ M.dag
        # M shape is (Batch_Size * N_Qubits, Rows=2, Cols=Rest)
        super_batch_psi = stacked.reshape(B * n, 2, -1)
        
        # 3. Compute Purity in one massive shot
        # rho_k = psi_mat @ psi_mat^H
        # Result shape: (B*n, 2, 2) - The 2x2 density matrix for every qubit in every batch
        rho_k = torch.matmul(super_batch_psi, super_batch_psi.conj().transpose(1, 2))
        
        # Purity = Tr(rho^2)
        # rho^2 shape: (B*n, 2, 2)
        rho_sq = torch.matmul(rho_k, rho_k)
        
        # Trace of each matrix
        purity_flat = rho_sq.diagonal(dim1=-2, dim2=-1).sum(-1) # Shape: (B*n,)
        
        # 4. Aggregate
        # Reshape back to separate Batch and Qubits: (B, n)
        purity_per_qubit = purity_flat.view(B, n)
        
        # Sum purities across qubits
        total_purity = purity_per_qubit.sum(dim=1) # (B,)
        
        # Formula: Q = 2 * (1 - 1/n * sum(purity))
        Q = 2.0 * (1.0 - (total_purity.real / n))
        
        return Q.unsqueeze(-1)

    def compute_cumulative_entanglement(self, token_seqs: torch.Tensor) -> torch.Tensor:
        """
        Computes the Meyer-Wallach entanglement measure for every step in the sequence.
        Input: (B, L) LongTensor of token IDs.
        Output: (B, L) FloatTensor of entanglement scores [0, 1].
        """
        B, L = token_seqs.shape
        device = token_seqs.device
        
        # We need to simulate the circuit step-by-step.
        # This is expensive: O(L) matmuls per batch.
        
        # 1. Get Unitaries for all tokens: (B, L, D, D)
        # Handle control tokens (Identity)
        # Assuming u_stack has Identity for BOS/EOS/PAD
        # If not, we should check. But typically UnitaryBackend provides it.
        if token_seqs.device != self.u_stack.device:
            token_seqs = token_seqs.to(self.u_stack.device)
            
        U_steps = self.u_stack[token_seqs] # (B, L, D, D)
        
        # 2. Cumulative Product (Prefix Scan)
        # U_cum[t] = U_t @ ... @ U_0
        # We need to compute this efficiently.
        # torch.cumprod doesn't work for matrices.
        # We have to loop or use a scan? PyTorch doesn't have a native scan for matmul.
        # Loop is okay for L ~ 32.
        
        # Initialize with Identity
        dim = U_steps.size(-1)
        current_U = torch.eye(dim, dtype=U_steps.dtype, device=U_steps.device).unsqueeze(0).expand(B, dim, dim)
        
        states_history = []
    
        # 1. Sequential Physics Evolution (Keep this light)
        for t in range(L):
            step_U = U_steps[:, t]
            current_U = torch.bmm(step_U, current_U)
            states_history.append(current_U) # Store reference
            
        # 2. Parallel Measurement (Heavy Lifting)
        # Stack all time steps into a larger batch
        # Shape: (B * L, D, D)
        all_states_flat = torch.cat(states_history, dim=0) 
        
        # Run MW once on the massive batch
        all_scores_flat = self.compute_meyer_wallach(all_states_flat)
        
        # Reshape back to (B, L)
        return all_scores_flat.view(L, B).t()

    def get_physics_labels(self, probs: torch.Tensor):
        """
        Helper to generate all physics targets for auxiliary heads.
        No gradients required here.
        """
        with torch.no_grad():
            U_pred, _ = self.forward(probs)
            entanglement_scores = self.compute_meyer_wallach(U_pred)
        
        return entanglement_scores