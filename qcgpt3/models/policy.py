import torch
import torch.nn as nn
from typing import Tuple, Optional

# Import the new classes from transformer.py
from .transformer import DualViewEncoder, CircuitDecoderPhysics
from ..gate_registry import GateRegistry

class CircuitPolicy(nn.Module):
    def __init__(self, registry: GateRegistry,
                 d_model: int = 256,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 max_spec_len: int = 256,
                 max_circ_len: int = 128):
        super().__init__()
        self.registry = registry
        vocab_size = len(registry.vocab)
        
        # Use the new Physics-Aware Encoder
        self.encoder = DualViewEncoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads, max_pairs=max_spec_len, n_qubits=registry.n_qubits)
        
        # Use the new Decoder with Physics Heads
        self.decoder = CircuitDecoderPhysics(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, max_len=max_circ_len)

    def forward(self, spec_batch: torch.Tensor, spec_pad_mask: torch.Tensor, circ_tokens: torch.Tensor, return_physics: bool = False):
        """
        Standard forward pass. 
        If return_physics=True, returns tuple (logits, entanglement_preds)
        """
        enc_out = self.encoder(spec_batch, spec_pad_mask)
        
        if return_physics:
            logits, ent_pred = self.decoder(circ_tokens, enc_out, memory_key_padding_mask=spec_pad_mask, return_physics=True)
            return logits, ent_pred
        else:
            logits = self.decoder(circ_tokens, enc_out, memory_key_padding_mask=spec_pad_mask, return_physics=False)
            return logits

    def decoder_forward_embeds(self, tgt_embeds: torch.Tensor, enc_out: torch.Tensor, memory_key_padding_mask: torch.Tensor = None):
        """
        Helper for Differentiable Autoregressive Training (Gumbel-Softmax).
        Takes continuous embeddings directly.
        """
        return self.decoder.forward_embeds(tgt_embeds, enc_out, memory_key_padding_mask=memory_key_padding_mask)

    def sample_circuit_tokens(self, spec_batch: torch.Tensor, spec_pad_mask: torch.Tensor,
                              max_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference Loop (Greedy/Sampling).
        """
        device = spec_batch.device
        B = spec_batch.size(0)
        
        bos_id = self.registry.bos_id
        eos_id = self.registry.eos_id
        pad_id = self.registry.pad_id
        
        # Encode once
        enc_out = self.encoder(spec_batch, spec_pad_mask)
        
        # Initialize sequence with BOS
        seqs = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        log_probs = torch.zeros(B, dtype=torch.float32, device=device)
        
        for _ in range(max_len):
            # Pass current sequence to decoder
            # Note: We don't need return_physics during sampling usually
            logits = self.decoder(seqs, enc_out, memory_key_padding_mask=spec_pad_mask)
            next_logits = logits[:, -1, :]
            
            probs = torch.softmax(next_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            
            logp = dist.log_prob(actions)
            log_probs = log_probs + (~done) * logp
            
            seqs = torch.cat([seqs, actions.unsqueeze(1)], dim=1)
            
            # Check for EOS
            done = done | (actions == eos_id)
            if done.all():
                break
        
        # Pad results if they finished early
        maxL = seqs.size(1)
        if maxL < max_len + 1:
            pad = torch.full((B, max_len+1-maxL), pad_id, dtype=torch.long, device=device)
            seqs = torch.cat([seqs, pad], dim=1)
            
        return seqs, log_probs
