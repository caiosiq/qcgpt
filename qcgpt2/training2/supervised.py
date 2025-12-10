import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from typing import List, Dict

from qcgpt2.data.specs2 import build_spec_sequence_batch
from qcgpt2.data.dataset2 import SimplifiedCircuitDataset2 as SimplifiedCircuitDataset
from qcgpt2.gates2 import PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2, VOCAB2, GATE_COST_REGISTRY
from qcgpt2.gate_registry2 import token_to_gate_parts
from qcgpt2.circuits2 import Circuit2, Gate2
from qcgpt2.encoding2 import circuit2_to_tokens
from qcgpt2.unitaries2 import get_unitary_for_token_id, build_circuit_unitary2
import torch.nn.functional as F

def _convert_circuit_to_tokens2(circ) -> torch.Tensor:
    c2 = Circuit2(nqubits=circ.nqubits)
    for g in circ.gates:
        c2.add_gate(Gate2(g.gate_type, g.targets))
    toks2 = circuit2_to_tokens(c2)
    return torch.tensor(toks2, dtype=torch.long)
def parallel_unitary_product(seq: torch.Tensor) -> torch.Tensor:
    """
    Computes the product of a sequence of matrices using tree reduction.
    Input: (B, L, D, D)
    Output: (B, D, D)
    Complexity: O(log L) steps instead of O(L)
    """
    # 1. Pad to next power of 2 for easy tree reduction (Optional but simpler)
    B, L, D, _ = seq.shape
    
    # Calculate next power of 2
    target_L = 1
    while target_L < L:
        target_L *= 2
        
    if target_L > L:
        # Pad with Identity matrices
        padding = torch.eye(D, dtype=seq.dtype, device=seq.device).view(1, 1, D, D)
        padding = padding.expand(B, target_L - L, D, D)
        seq = torch.cat([seq, padding], dim=1)
        
    current_seq = seq
    
    # 2. Tree Reduction Loop
    # We reduce the sequence length by half in every iteration
    while current_seq.shape[1] > 1:
        # Reshape to (B, L/2, 2, D, D) to group neighbors
        # We want to multiply U_{t+1} @ U_t (Left multiplication convention)
        B_curr, L_curr, _, _ = current_seq.shape
        half_L = L_curr // 2
        
        # Split into left and right halves
        # left: indices 0, 2, 4... (Applied First)
        # right: indices 1, 3, 5... (Applied Second)
        left = current_seq[:, 0::2] 
        right = current_seq[:, 1::2]
        
        # Multiply: Right @ Left
        current_seq = right @ left 
        
    # Result is now (B, 1, D, D), squeeze to (B, D, D)
    return current_seq.squeeze(1)

def generate_differentiable_logits(model, spec_batch, spec_pad_mask, 
                                   bos_id, eos_id, max_len=32, temp=0.5, greedy=False):
    """
    Generates a circuit autoregressively using Gumbel-Softmax (Straight-Through).
    Returns: 'soft_probs' sequence (B, L, Vocab) that has gradients attached.
    """
    B = spec_batch.size(0)
    device = spec_batch.device
    
    # 1. Encode Truth Table (Standard)
    # We do this once.
    memory = model.encoder(spec_batch, spec_pad_mask)
    
    # 2. Setup Embedding Access
    # We need the raw weights to do the differentiable lookup: Soft_OneHot @ Matrix
    # Assumes model.decoder has an attribute 'token_embedding' which is nn.Embedding
    w_emb = model.decoder.token_emb.weight # (Vocab, D_model)
    
    # 3. Start with <BOS>
    # Initial input is just the BOS embedding
    curr_input = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    curr_embeds = model.decoder.token_emb(curr_input) # (B, 1, D)
    
    history_embeds = curr_embeds
    soft_probs_list = []
    
    # 4. The Autoregressive Loop (O(L))
    for t in range(max_len):
        # A. Forward Pass using Embeddings
        # Uses your new 'forward_embeds' function
        # We pass the WHOLE history so the Transformer can attend to it
        logits = model.decoder.forward_embeds(history_embeds, memory, memory_key_padding_mask=spec_pad_mask)
        
        # B. Get prediction for the *next* token (the last one in the sequence)
        next_token_logits = logits[:, -1, :] # (B, Vocab)
        
        # C. Gumbel-Softmax (The Magic Step)
        # hard=True: Forward pass sees a perfect One-Hot vector (Discrete).
        #            Backward pass sees the Softmax probability gradients.
        if greedy:
            # --- DETERMINISTIC PATH (Eval) ---
            # Pick exactly the max logit. No noise.
            token_id = torch.argmax(next_token_logits, dim=-1)
            next_token_onehot = F.one_hot(token_id, num_classes=next_token_logits.size(-1)).float()
        else:
            # --- DIFFERENTIABLE PATH (Training) ---
            # Add noise to explore and allow gradients to flow
            next_token_onehot = F.gumbel_softmax(next_token_logits, tau=temp, hard=True, dim=-1)
        
        # Save for Unitary Calculation
        soft_probs_list.append(next_token_onehot)
        
        # D. Prepare Input for Next Step
        # Differentiable Embedding Lookup: OneHot @ Weights
        # (B, V) @ (V, D) -> (B, D) -> (B, 1, D)
        next_embed = torch.matmul(next_token_onehot, w_emb).unsqueeze(1)
        
        # Append to history
        history_embeds = torch.cat([history_embeds, next_embed], dim=1)
        
        # Optimization: We could break early if all batches hit EOS, 
        # but for batched diff-prog, fixed length is often numerically more stable.
        
    # 5. Stack into Sequence
    # Shape: (B, MaxLen, Vocab) - This looks exactly like "One-Hot Logits"
    full_probs = torch.stack(soft_probs_list, dim=1).float()
    
    return full_probs
def calculate_physical_fidelity_components(probs, 
                                           U_stack=None, cost_tensor=None, device=None,
                                           noise_scale=1.0):
    """
    Flexible physics engine with Hardware-Aware Scaling.
    
    Args:
        probs: (B, L, V) Soft probability distribution from Gumbel-Softmax.
        U_stack: (V, 2^n, 2^n) Tensor of unitary matrices.
        cost_tensor: (V,) Tensor of raw gate error probabilities (p).
        device: torch.device
        noise_scale (float): A scalar calibration factor to map 'Raw p' to 'Fidelity Loss'.
                             For Depolarizing Noise on 3 qubits, use ~2.0.
                             For simple sum-of-errors, use 1.0.
    """
    
    # --- 1. Determine Probabilities ---
    probs_gates = probs.float()
    B, Lc, V = probs.shape
    
    # Extract EOS prob for Life Masking
    p_eos_ste = probs[:, :, EOS_CIRC_ID2]

    # --- 2. Compute Life Mask ---
    # Standard autoregressive masking: signal dies after EOS
    p_continue = 1.0 - p_eos_ste
    life_mask = torch.cumprod(p_continue, dim=1)
    
    # Shift logic: If EOS is at t, the circuit is valid up to t.
    # We shift right so the mask drops to 0 *after* the EOS token.
    life_mask = torch.roll(life_mask, shifts=1, dims=1)
    life_mask[:, 0] = 1.0
    
    # Expand for broadcasting
    life_mask_U = life_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=U_stack.dtype)
    life_mask_noise = life_mask.to(dtype=cost_tensor.dtype)

    # --- 3. Compute Unitary (Ideal Physics) ---
    # Weighted sum of gates
    U_seq = torch.einsum("blv,vij->blij", probs_gates.to(U_stack.dtype), U_stack)
    I = torch.eye(8, dtype=U_stack.dtype, device=device).view(1, 1, 8, 8)
    
    # Apply Mask: (Alive * Gate) + (Dead * Identity)
    U_effective_seq = life_mask_U * U_seq + (1.0 - life_mask_U) * I
    
    # Parallel Product (O(log L))
    U_final = parallel_unitary_product(U_effective_seq)

    # --- 4. Compute Noise (Hardware Physics) ---
    # A. Expected raw cost per step (e.g. 0.01 for CNOT)
    step_costs = torch.einsum("blv,v->bl", probs_gates.float(), cost_tensor)
    
    # B. Mask out costs after EOS
    valid_step_costs = step_costs * life_mask_noise
    
    # C. Sum and Scale
    # We multiply by noise_scale to allow for testing on different noise scales. This is different from a lambda on the loss because the fidelity loss can only be from 0 to 1
    total_noise = valid_step_costs.sum(dim=1) * noise_scale
    fidelity_loss = 1.0 - torch.exp(-total_noise)

    return U_final, fidelity_loss

def collate_supervised2(batch: List[Dict[str, object]]):
    spec_tensors = [item["spec_tensor"].numpy() for item in batch]
    spec_batch_np, spec_pad_mask_np = build_spec_sequence_batch(spec_tensors)
    spec_batch = torch.tensor(spec_batch_np, dtype=torch.float32)
    spec_pad_mask = torch.tensor(spec_pad_mask_np, dtype=torch.bool)
    circ_seqs = []
    for item in batch:
        ref_circ = item.get("ref_circuit", None)
        if ref_circ is None:
            seq = torch.tensor([BOS_CIRC_ID2, EOS_CIRC_ID2], dtype=torch.long)
        else:
            seq = _convert_circuit_to_tokens2(ref_circ)
        if seq.size(0) < 2:
            seq = torch.tensor([BOS_CIRC_ID2, EOS_CIRC_ID2], dtype=torch.long)
        circ_seqs.append(seq)
    max_Lc = max(2, max(seq.size(0) for seq in circ_seqs))
    B = len(circ_seqs)
    circ_in = torch.full((B, max_Lc - 1), PAD_ID2, dtype=torch.long)
    circ_tgt = torch.full((B, max_Lc - 1), PAD_ID2, dtype=torch.long)
    for i, seq in enumerate(circ_seqs):
        Lc = seq.size(0)
        circ_in[i, :Lc - 1] = seq[:-1]
        circ_tgt[i, :Lc - 1] = seq[1:]
    return spec_batch, spec_pad_mask, circ_in, circ_tgt


def build_simplified_dataloader2(num_samples: int, batch_size: int, n_qubits: int = 3,
                                 raw_max_depth: int = 8, include_basis_states: bool = True,
                                 n_random_states: int = 0, num_workers: int = 0,
                                 pin_memory: bool = False) -> DataLoader:
    dataset = SimplifiedCircuitDataset(
        num_samples=num_samples,
        n_qubits=n_qubits,
        raw_max_depth=raw_max_depth,
        include_basis_states=include_basis_states,
        n_random_states=n_random_states,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2,
        collate_fn=collate_supervised2,
    )


def train_supervised_epoch2(model, dataloader, optimizer, device,
                            use_unitary_loss=False, lambda_sup=1.0, lambda_U=0.0,
                            softmax_temp=1.0, use_noise=False, lambda_noise=0.0,
                            noise_scale: float = 1.0):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID2)
    total_loss = 0.0
    total_tokens = 0
    cost_values = [GATE_COST_REGISTRY[tok] for tok in VOCAB2]
    cost_tensor = torch.tensor(cost_values, device=device, dtype=torch.float32)
    # Pre-load Unitary Stack ONCE
    # IMPORTANT: Ensure PAD, BOS, EOS map to Identity in U_stack
    mats = []
    for tid in range(len(VOCAB2)):
        if tid in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
            mats.append(torch.eye(8, dtype=torch.complex64))
        else:
            mats.append(get_unitary_for_token_id(tid, n_qubits=3).to(dtype=torch.complex64))
            
    U_stack = torch.stack(mats, dim=0).to(device)

    for spec_batch, spec_pad_mask, circ_in, circ_tgt in dataloader:
        # ... setup inputs ...
        spec_batch = spec_batch.to(device)
        spec_pad_mask = spec_pad_mask.to(device)
        circ_in = circ_in.to(device)
        circ_tgt = circ_tgt.to(device)
        # --- A. Supervised Path (Teacher Forcing) ---
        # This keeps the model anchored to syntax.
        # We assume lambda_sup is small (e.g. 0.05) during fine-tuning.
        if lambda_sup > 0:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits_sup = model(spec_batch, spec_pad_mask, circ_in)
            B, Lc, V = logits_sup.shape
            loss_sup = criterion(logits_sup.view(B * Lc, V), circ_tgt.view(B * Lc))
        else:
            loss_sup = 0.0

        loss = lambda_sup * loss_sup

        # --- B. Unitary Path (Self-Generation) ---
        if use_unitary_loss and lambda_U > 0:
            # We must use Float32 for the physics engine
            with torch.cuda.amp.autocast(enabled=False) if device.type == "cuda" else torch.no_grad():
                
                # 1. Generate the circuit Autoregressively (The "Free" Path)
                # This replaces the old 'model(circ_in)' call for this section
                probs_gen = generate_differentiable_logits(
                    model, spec_batch, spec_pad_mask, 
                    bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2, 
                    max_len=32,
                    temp=softmax_temp # Temp is handled inside the generator
                )
                
                # 2. Physics Engine (Direct Probs)
                U_pred, gate_noise = calculate_physical_fidelity_components(
                    probs=probs_gen, 
                    U_stack=U_stack, 
                    cost_tensor=cost_tensor, 
                    device=device,
                    noise_scale=noise_scale,
                )
                
                
                # 4. Target Unitary
                # We need U_tgt. Since circ_tgt is the Reference, we use it.
                # (Ideally, U_tgt comes from the Truth Table, but Reference is a valid proxy for the Target Matrix)
                with torch.no_grad():
                    ids = circ_tgt.clamp(min=0, max=len(VOCAB2)-1)
                    U_tgt_seq = U_stack[ids]
                    U_tgt = parallel_unitary_product(U_tgt_seq)

                # 5. Loss Calculation
                trace = torch.einsum("bij,bij->b", U_tgt.conj(), U_pred)
                fidelity = (trace.abs() ** 2) / (8 ** 2)
                loss_U = 1.0 - fidelity.mean()
                
                if use_noise and lambda_noise > 0:
                    loss_U = loss_U + lambda_noise/lambda_U * gate_noise.mean()

            loss = loss + lambda_U * loss_U

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            num_non_pad = (circ_tgt != PAD_ID2).sum().item()
            total_loss += loss.item() * num_non_pad
            total_tokens += num_non_pad

    return total_loss / max(1, total_tokens)


def evaluate_supervised_epoch2(model, dataloader, device,
                               use_unitary_loss=False, lambda_sup=1.0, lambda_U=0.0,
                               softmax_temp=1.0, use_noise=False, lambda_noise=0.0,
                               noise_scale: float = 1.0):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID2)
    total_loss = 0.0
    total_tokens = 0
    
    # Init Cost & Stack (Same as before)
    cost_values = [GATE_COST_REGISTRY[tok] for tok in VOCAB2]
    cost_tensor = torch.tensor(cost_values, device=device, dtype=torch.float32)
    mats = []
    for tid in range(len(VOCAB2)):
        if tid in [PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2]:
            mats.append(torch.eye(8, dtype=torch.complex64))
        else:
            mats.append(get_unitary_for_token_id(tid, n_qubits=3).to(dtype=torch.complex64))
    U_stack = torch.stack(mats, dim=0).to(device)

    with torch.no_grad():
        for spec_batch, spec_pad_mask, circ_in, circ_tgt in dataloader:
            spec_batch = spec_batch.to(device)
            spec_pad_mask = spec_pad_mask.to(device)
            circ_in = circ_in.to(device)
            circ_tgt = circ_tgt.to(device)
            
            # 1. Supervised Loss (Standard)
            logits = model(spec_batch, spec_pad_mask, circ_in)
            B, Lc, V = logits.shape
            ce = criterion(logits.view(B * Lc, V), circ_tgt.view(B * Lc))
            loss = lambda_sup * ce 
            
            # 2. Unitary Loss (Autoregressive Self-Gen)
            if use_unitary_loss and lambda_U > 0:
                # Use the SAME generation function as training
                # but with hard=True (greedy-ish) or temp=0.01 for robust eval
                # Since we are in no_grad, we don't need gradients, 
                # but using the differentiable generator is the easiest way to reuse code.
                
                # Use low temp for Eval to check "Best Attempt"
                
                probs_gen = generate_differentiable_logits(
                    model, spec_batch, spec_pad_mask, 
                    bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2, 
                    max_len=32,
                    greedy=True
                )
                
                # Physics Engine (Direct Probs)
                U_pred, gate_noise = calculate_physical_fidelity_components(
                    probs=probs_gen, 
                    U_stack=U_stack, 
                    cost_tensor=cost_tensor, 
                    device=device,
                    noise_scale=noise_scale,
                )
                
                # Target Unitary (From Reference)
                ids = circ_tgt.clamp(min=0, max=len(VOCAB2)-1)
                U_tgt_seq = U_stack[ids]
                U_tgt = parallel_unitary_product(U_tgt_seq)

                # Fidelity
                trace = torch.einsum("bij,bij->b", U_tgt.conj(), U_pred)
                fidelity = (trace.abs() ** 2) / (8 ** 2)
                loss_U = 1.0 - fidelity.mean()
                
                if use_noise and lambda_noise > 0:
                    loss_U = loss_U + lambda_noise * gate_noise.mean()
                    
                loss = loss + lambda_U * loss_U
                
            num_non_pad = (circ_tgt != PAD_ID2).sum().item()
            total_loss += loss.item() * num_non_pad
            total_tokens += num_non_pad
            
    return total_loss / max(1, total_tokens)
