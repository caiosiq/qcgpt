# qcgpt/models/transformer.py
import torch
import torch.nn as nn
from ..gates import PAD_ID

class SpecEncoder(nn.Module):
    def __init__(self, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 4, max_len: int = 256):
        super().__init__()
        self.input_proj = nn.Linear(4, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, spec_batch: torch.Tensor, spec_pad_mask: torch.Tensor) -> torch.Tensor:
            B, L, _ = spec_batch.shape
            pos = torch.arange(L, device=spec_batch.device).unsqueeze(0).expand(B, L)
            x = self.input_proj(spec_batch) + self.pos_emb(pos)
            return self.encoder(
                x,
                src_key_padding_mask=spec_pad_mask,
            )

class CircuitDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 4, max_len: int = 128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, circ_tokens: torch.Tensor,
                enc_out: torch.Tensor,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        circ_tokens: [B, Lc]
        enc_out:     [B, Ls, d_model]
        enc_padding_mask: [B, Ls] (True for PAD)
        """
        B, Lc = circ_tokens.shape
        pos = torch.arange(Lc, device=circ_tokens.device).unsqueeze(0).expand(B, Lc)
        x = self.token_emb(circ_tokens) + self.pos_emb(pos)

        # causal mask for autoregressive decoding
        tgt_mask = torch.triu(torch.ones(Lc, Lc, device=x.device), 1).bool()

        # mask out PAD tokens in target
        tgt_key_padding_mask = (circ_tokens == PAD_ID)  # [B, Lc]

        logits = self.decoder(
            x,
            enc_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(logits)

