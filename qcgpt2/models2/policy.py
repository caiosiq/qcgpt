import torch
import torch.nn as nn
from typing import Tuple
from .transformer import SpecEncoder, CircuitDecoder2
from ..gates2 import PAD_ID2


class CircuitPolicy2(nn.Module):
    def __init__(self, vocab_size: int,
                 d_model: int = 256,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 max_spec_len: int = 256,
                 max_circ_len: int = 128):
        super().__init__()
        self.encoder = SpecEncoder(d_model, n_layers, n_heads, max_spec_len)
        self.decoder = CircuitDecoder2(vocab_size, d_model, n_layers, n_heads, max_circ_len)

    def forward(self, spec_batch: torch.Tensor, spec_pad_mask: torch.Tensor, circ_tokens: torch.Tensor):
        enc_out = self.encoder(spec_batch, spec_pad_mask)
        logits = self.decoder(circ_tokens, enc_out, memory_key_padding_mask=spec_pad_mask)
        return logits
    def decoder_forward_embeds(self, tgt_embeds: torch.Tensor, enc_out: torch.Tensor, memory_key_padding_mask: torch.Tensor = None):
        """
        Helper for Differentiable Autoregressive Training.
        Takes embeddings directly, skips encoder (assumes enc_out is passed).
        """
        return self.decoder.forward_embeds(tgt_embeds, enc_out, memory_key_padding_mask=memory_key_padding_mask)
    def sample_circuit_tokens(self, spec_batch: torch.Tensor, spec_pad_mask: torch.Tensor,
                              bos_id: int, eos_id: int, max_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        device = spec_batch.device
        B = spec_batch.size(0)
        enc_out = self.encoder(spec_batch, spec_pad_mask)
        seqs = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        log_probs = torch.zeros(B, dtype=torch.float32, device=device)
        for _ in range(max_len):
            logits = self.decoder(seqs, enc_out, memory_key_padding_mask=spec_pad_mask)
            next_logits = logits[:, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            logp = dist.log_prob(actions)
            log_probs = log_probs + (~done) * logp
            seqs = torch.cat([seqs, actions.unsqueeze(1)], dim=1)
            done = done | (actions == eos_id)
            if done.all():
                break
        maxL = seqs.size(1)
        if maxL < max_len + 1:
            pad = torch.full((B, max_len+1-maxL), PAD_ID2, dtype=torch.long, device=device)
            seqs = torch.cat([seqs, pad], dim=1)
        return seqs, log_probs
