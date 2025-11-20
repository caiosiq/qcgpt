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

    return spec_batch, spec_pad_mask, circ_in, circ_tgt


def train_supervised_epoch(
    model: CircuitPolicy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
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

        loss = criterion(
            logits.view(B * Lc, V),
            circ_tgt.view(B * Lc)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            num_non_pad = (circ_tgt != PAD_ID).sum().item()
            total_loss += loss.item() * num_non_pad
            total_tokens += num_non_pad

    return total_loss / max(1, total_tokens)


def build_supervised_dataloader(
    size: int,
    batch_size: int,
    max_gates: int = 6,
    num_workers: int = 0,
) -> DataLoader:
    dataset = MappingCircuitDataset(size=size, max_gates=max_gates)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
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
        collate_fn=collate_supervised,
    )
