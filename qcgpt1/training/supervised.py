# qcgpt/training/supervised.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import List, Dict

from ..gates import PAD_ID, BOS_CIRC_ID, EOS_CIRC_ID
from ..models.policy import CircuitPolicy
from ..data.dataset import MappingCircuitDataset, SimplifiedCircuitDataset
from ..encoding import circuit_to_tokens
from ..data.specs import build_spec_sequence_batch
from ..unitaries import build_circuit_unitary
import math


def collate_supervised(batch: List[Dict[str, object]]):
    spec_tensors = [item["spec_tensor"].numpy() for item in batch]
    spec_batch_np, spec_pad_mask_np = build_spec_sequence_batch(spec_tensors)
    spec_batch = torch.tensor(spec_batch_np, dtype=torch.float32)
    spec_pad_mask = torch.tensor(spec_pad_mask_np, dtype=torch.bool)

    circ_seqs = []
    for item in batch:
        if "circ_tokens" in item:
            seq = item["circ_tokens"]
            if seq.size(0) < 2:
                seq = torch.tensor([BOS_CIRC_ID, EOS_CIRC_ID], dtype=torch.long)
            circ_seqs.append(seq)
        else:
            circ = item["ref_circuit"]
            tokens = circuit_to_tokens(circ)
            if len(tokens) < 2:
                tokens = [BOS_CIRC_ID, EOS_CIRC_ID]
            circ_seqs.append(torch.tensor(tokens, dtype=torch.long))

    max_Lc = max(2, max(seq.size(0) for seq in circ_seqs))
    B = len(circ_seqs)
    circ_in = torch.full((B, max_Lc - 1), PAD_ID, dtype=torch.long)
    circ_tgt = torch.full((B, max_Lc - 1), PAD_ID, dtype=torch.long)
    for i, seq in enumerate(circ_seqs):
        Lc = seq.size(0)
        circ_in[i, :Lc - 1] = seq[:-1]
        circ_tgt[i, :Lc - 1] = seq[1:]

    # Optional logging info: count dedupe skips if present
    # We attach minimal metadata via attributes to avoid timing cost
    collate_supervised.last_dedupe_skips = 0
    for item in batch:
        if "dedupe_skips" in item:
            collate_supervised.last_dedupe_skips = max(collate_supervised.last_dedupe_skips, int(item["dedupe_skips"]))
    return spec_batch, spec_pad_mask, circ_in, circ_tgt


def train_supervised_epoch(
    model: CircuitPolicy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_unitary_loss: bool = False,
    lambda_sup: float = 1.0,
    lambda_U: float = 0.1,
    softmax_temp: float = 1.0,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    total_loss = 0.0
    total_tokens = 0

    for spec_batch, spec_pad_mask, circ_in, circ_tgt in dataloader:
        spec_batch = spec_batch.to(device)
        spec_pad_mask = spec_pad_mask.to(device)
        circ_in    = circ_in.to(device)
        circ_tgt   = circ_tgt.to(device)

        logits = model(spec_batch, spec_pad_mask, circ_in)
        B, Lc, V = logits.shape

        ce = criterion(
            logits.view(B * Lc, V),
            circ_tgt.view(B * Lc)
        )
        loss = ce

        if use_unitary_loss:
            # Identify gate positions from target tokens (teacher)
            from ..gates import TOKEN_TO_ID, GATE_TYPES
            gate_ids = torch.tensor([TOKEN_TO_ID[g] for g in GATE_TYPES], device=device)
            is_gate = torch.zeros((B, Lc), dtype=torch.bool, device=device)
            for gid in gate_ids:
                is_gate |= (circ_tgt == gid)
            # Extract per-example gate logits over gate-only vocabulary
            gate_logits = []
            for b in range(B):
                idxs = torch.nonzero(is_gate[b]).squeeze(-1)
                if idxs.numel() == 0:
                    gate_logits.append(torch.empty((0, gate_ids.numel()), device=device))
                else:
                    gl = logits[b, idxs][:, gate_ids]
                    gate_logits.append(gl)
            # Build soft unitary: simplified relaxation (ignores qubit indices)
            D = 2 ** int(spec_batch.size(-2).bit_length() - 1)  # approximate; we use n_qubits=3 constant below
            n_qubits = 3
            U_pred = torch.eye(2 ** n_qubits, dtype=torch.complex64, device=device).unsqueeze(0).repeat(B, 1, 1)
            # Map gate types to full 3-qubit unitaries once
            from ..unitaries import get_gate_unitary
            gate_unitaries = {g: get_gate_unitary(g, [0], n_qubits=n_qubits).to(device) if g not in {"CX","CZ","SWAP","CCX","CCZ","CSWAP"} else get_gate_unitary(g, [0,1] if g in {"CX","CZ","SWAP"} else [0,1,2], n_qubits=n_qubits).to(device) for g in GATE_TYPES}
            GU_stack = torch.stack([gate_unitaries[g] for g in GATE_TYPES], dim=0)  # [Kg, D, D]
            for b in range(B):
                Ub = torch.eye(2 ** n_qubits, dtype=torch.complex64, device=device)
                gl = gate_logits[b]
                for t in range(gl.size(0)):
                    logits_t = gl[t] / max(softmax_temp, 1e-6)
                    alpha = torch.softmax(logits_t, dim=-1)
                    alpha = alpha.to(dtype=GU_stack.dtype)
                    Ut = torch.einsum("k,kij->ij", alpha, GU_stack)
                    Ub = Ut @ Ub
                U_pred[b] = Ub
            # Build U_target from ref circuits in dataloader, if available
            # Since supervised dataloader returns circ_in/circ_tgt only, we reconstruct teacher circuit tokens back to Circuit
            from ..encoding import tokens_to_circuit
            U_tgt_list = []
            for b in range(B):
                toks = circ_tgt[b].tolist()
                toks = [t for t in toks if t != TOKEN_TO_ID["<PAD>"]]
                try:
                    circ = tokens_to_circuit(toks)
                    U_tgt = build_circuit_unitary(circ, n_qubits=n_qubits).to(device)
                except Exception:
                    U_tgt = torch.eye(2 ** n_qubits, dtype=torch.complex64, device=device)
                U_tgt_list.append(U_tgt)
            U_tgt = torch.stack(U_tgt_list, dim=0)
            trace = torch.einsum("bij,bji->b", U_tgt.conj(), U_pred)
            fidelity = (trace.abs() ** 2) / ((2 ** n_qubits) ** 2)
            loss_U = 1.0 - fidelity.mean()
            loss = lambda_sup * ce + lambda_U * loss_U

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            num_non_pad = (circ_tgt != PAD_ID).sum().item()
            total_loss += loss.item() * num_non_pad
            total_tokens += num_non_pad
            # Lightweight print of dedupe status without overhead
            if hasattr(collate_supervised, "last_dedupe_skips"):
                skips = collate_supervised.last_dedupe_skips
                if skips > 0:
                    print(f"[Dedupe] Skips so far: {skips}")
            # Optional GPU memory logging
            if device.type == "cuda":
                try:
                    import os
                    interval = int(os.environ.get("MEM_LOG_INTERVAL", "0"))
                except Exception:
                    interval = 0
                if interval > 0:
                    if (total_tokens // max(1, num_non_pad)) % interval == 0:
                        used = torch.cuda.memory_allocated() / (1024 ** 2)
                        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                        print(f"[GPU] mem_alloc={used:.1f}MB  mem_reserved={reserved:.1f}MB")

    return total_loss / max(1, total_tokens)


def build_supervised_dataloader(
    size: int,
    batch_size: int,
    max_gates: int = 6,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = MappingCircuitDataset(size=size, max_gates=max_gates)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_supervised,
    )


def build_simplified_dataloader(
    num_samples: int,
    batch_size: int,
    n_qubits: int = 3,
    raw_max_depth: int = 8,
    include_basis_states: bool = True,
    n_random_states: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
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
        collate_fn=collate_supervised,
    )
