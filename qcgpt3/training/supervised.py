import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Optional

from qcgpt3 import GateRegistry, QDPE, CircuitEncoder, Circuit, Gate
from qcgpt3.data.dataset import HighPerformanceDataset
from qcgpt3.data.specs import build_spec_sequence_batch
from qcgpt3.training.objectives import Objective

class SupervisedCollator:
    def __init__(self, registry: GateRegistry):
        self.registry = registry
        self.encoder = CircuitEncoder(registry)

    def __call__(self, batch: List[Dict[str, object]]):
        spec_tensors = [item["spec_tensor"].numpy() for item in batch]
        spec_batch_np, spec_pad_mask_np = build_spec_sequence_batch(spec_tensors)
        spec_batch = torch.tensor(spec_batch_np, dtype=torch.float32)
        spec_pad_mask = torch.tensor(spec_pad_mask_np, dtype=torch.bool)
        
        circ_seqs = []
        for item in batch:
            ref_circ = item.get("ref_circuit", None)
            if ref_circ is None:
                # Default empty circuit: BOS, EOS
                seq = torch.tensor([self.registry.bos_id, self.registry.eos_id], dtype=torch.long)
            else:
                # Convert ref_circuit (likely qcgpt2/1 object) to qcgpt3 Circuit if needed
                # Or if ref_circuit is already compatible.
                # Assuming ref_circuit has gates with gate_type and targets.
                # We rebuild it as a qcgpt3 Circuit to be safe and use the encoder.
                c3 = Circuit(n_qubits=self.registry.n_qubits)
                for g in ref_circ.gates:
                    c3.add_gate(Gate(g.gate_type, g.targets))
                
                toks = self.encoder.encode(c3)
                seq = torch.tensor(toks, dtype=torch.long)
            
            if seq.size(0) < 2:
                seq = torch.tensor([self.registry.bos_id, self.registry.eos_id], dtype=torch.long)
            circ_seqs.append(seq)
            
        max_Lc = max(2, max(seq.size(0) for seq in circ_seqs))
        B = len(circ_seqs)
        circ_in = torch.full((B, max_Lc - 1), self.registry.pad_id, dtype=torch.long)
        circ_tgt = torch.full((B, max_Lc - 1), self.registry.pad_id, dtype=torch.long)
        
        for i, seq in enumerate(circ_seqs):
            Lc = seq.size(0)
            circ_in[i, :Lc - 1] = seq[:-1]
            circ_tgt[i, :Lc - 1] = seq[1:]
            
        return spec_batch, spec_pad_mask, circ_in, circ_tgt

def build_high_performance_dataloader(registry: GateRegistry, qdpe: Optional[QDPE], num_samples: int, batch_size: int, n_qubits: int = 3,
                                 raw_max_depth: int = 8, include_basis_states: bool = True,
                                 n_random_states: int = 0, num_workers: int = 0,
                                 pin_memory: bool = False,
                                 augment_commutation: bool = False,
                                 augment_permutation: bool = False) -> DataLoader:
    dataset = HighPerformanceDataset(
        registry=registry,
        qdpe=qdpe,
        num_samples=num_samples,
        n_qubits=n_qubits,
        raw_max_depth=raw_max_depth,
        include_basis_states=include_basis_states,
        n_random_states=n_random_states,
        augment_commutation=augment_commutation,
        augment_permutation=augment_permutation,
    )
    
    collator = SupervisedCollator(registry)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2,
        collate_fn=collator,
    )

def train_supervised_epoch(model, dataloader, optimizer, device,
                            registry: GateRegistry, qdpe: QDPE,
                            objective: Objective):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_metrics = {}
    num_batches = 0

    for spec_batch, spec_pad_mask, circ_in, circ_tgt in dataloader:
        spec_batch = spec_batch.to(device)
        spec_pad_mask = spec_pad_mask.to(device)
        circ_in = circ_in.to(device)
        circ_tgt = circ_tgt.to(device)

        loss, metrics = objective(
            model=model,
            qdpe=qdpe,
            spec_batch=spec_batch,
            spec_pad_mask=spec_pad_mask,
            circ_in=circ_in,
            circ_tgt=circ_tgt,
            is_training=True
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            num_non_pad = (circ_tgt != registry.pad_id).sum().item()
            total_loss += loss.item() * num_non_pad
            total_tokens += num_non_pad
            
            # Accumulate metrics (sum for simple batch average)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            num_batches += 1

    avg_loss = total_loss / max(1, total_tokens)
    avg_metrics = {k: v / max(1, num_batches) for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics


def evaluate_supervised_epoch(model, dataloader, device,
                               registry: GateRegistry, qdpe: QDPE,
                               objective: Objective):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_metrics = {}
    num_batches = 0
    
    with torch.no_grad():
        for spec_batch, spec_pad_mask, circ_in, circ_tgt in dataloader:
            spec_batch = spec_batch.to(device)
            spec_pad_mask = spec_pad_mask.to(device)
            circ_in = circ_in.to(device)
            circ_tgt = circ_tgt.to(device)
            
            loss, metrics = objective(
                model=model,
                qdpe=qdpe,
                spec_batch=spec_batch,
                spec_pad_mask=spec_pad_mask,
                circ_in=circ_in,
                circ_tgt=circ_tgt,
                is_training=False
            )
            
            num_non_pad = (circ_tgt != registry.pad_id).sum().item()
            total_loss += loss.item() * num_non_pad
            total_tokens += num_non_pad
            
            # Accumulate metrics
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            num_batches += 1
            
    avg_loss = total_loss / max(1, total_tokens)
    avg_metrics = {k: v / max(1, num_batches) for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics
