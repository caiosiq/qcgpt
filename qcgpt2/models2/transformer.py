import torch
import torch.nn as nn


class SpecEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=4, max_pairs=64, n_qubits: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        dim = 2 ** n_qubits
        self.input_proj = nn.Linear(4 * dim, d_model)
        self.pos_emb = nn.Embedding(max_pairs, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=4*d_model)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, spec_batch: torch.Tensor, spec_pad_mask: torch.Tensor):
        # spec_batch: [B, n_pairs, 4*dim]; spec_pad_mask: [B, n_pairs]
        B, n_pairs, F = spec_batch.shape
        dim = 2 ** self.n_qubits
        assert F == 4 * dim
        x = self.input_proj(spec_batch)
        pos = torch.arange(n_pairs, device=spec_batch.device).unsqueeze(0).expand(B, n_pairs)
        x = x + self.pos_emb(pos)
        return self.encoder(x, src_key_padding_mask=spec_pad_mask)


class CircuitDecoder2(nn.Module):
    def __init__(self, vocab_size: int, d_model=256, n_layers=4, n_heads=4, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=4*d_model)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, circ_tokens: torch.Tensor, enc_out: torch.Tensor, memory_key_padding_mask: torch.Tensor = None):
        B, Lc = circ_tokens.shape
        pos = torch.arange(Lc, device=circ_tokens.device).unsqueeze(0).expand(B, Lc)
        x = self.token_emb(circ_tokens) + self.pos_emb(pos)
        tgt_mask = torch.triu(torch.ones(Lc, Lc, device=x.device), 1).bool()
        return self.out_proj(self.decoder(x, enc_out, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask))
    def forward_embeds(self, tgt_embeds, memory, memory_key_padding_mask=None):
        """
        Alternative forward pass that accepts continuous embeddings 
        instead of discrete token IDs.
        
        tgt_embeds: (Batch, SeqLen, D_model) - Already looked up via Gumbel/Matmul
        memory: Encoder output
        """
        B, Lc, _ = tgt_embeds.shape
        
        # 1. Apply Positional Encoding
        # We generate indices [0, 1, ... Lc-1] just like in standard forward
        pos = torch.arange(Lc, device=tgt_embeds.device).unsqueeze(0).expand(B, Lc)
        
        # Add the learned position vectors to the input embeddings
        x = tgt_embeds + self.pos_emb(pos)
        
        # 2. Generate Causal Mask (Triangular mask)
        # Using the same logic as your forward: triu(1).bool()
        # Note: PyTorch Transformer expects a boolean mask or float mask. 
        # Your forward uses .bool(), so we stick to that for consistency.
        tgt_mask = torch.triu(torch.ones(Lc, Lc, device=x.device), 1).bool()
        
        # 3. Transformer Decoder Pass
        # Using self.decoder (not self.transformer_decoder)
        out = self.decoder(
            x, 
            memory, 
            tgt_mask=tgt_mask, 
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # 4. Project to Logits
        # Using self.out_proj (not self.output_head)
        return self.out_proj(out)
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask