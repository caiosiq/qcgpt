import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper: Differentiable Walsh-Hadamard Transform ---
def fast_walsh_hadamard(x):
    """
    Computes the Walsh-Hadamard Transform along the last dimension.
    Used to extract the 'Pauli Spectrum' from the Truth Table.
    """
    h = 1
    x = x.clone()
    while h < x.shape[-1]:
        for i in range(0, x.shape[-1], h * 2):
            a = x[..., i : i + h]
            b = x[..., i + h : i + 2 * h]
            x[..., i : i + h] = a + b
            x[..., i + h : i + 2 * h] = a - b
        h *= 2
    return x / math.sqrt(x.shape[-1])

# --- Component: Entanglement Physics Head ---
class EntanglementHead(nn.Module):
    """
    Auxiliary head that predicts a propriety of the circuit state. (Ranging from 0 to 1)
    Forces the decoder to learn quantum correlations, not just syntax.
    """
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Output: 0.0 (Separable) -> 1.0 (Maximally Entangled)
        )

    def forward(self, decoder_out):
        return self.net(decoder_out)

# --- Component: Rotary-Style Positional Embedding ---
class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal embeddings that generalize better to longer circuits than learned embeddings.
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, SeqLen, D_model]
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


# --- MAIN ENCODER: Physics-Aware Dual View ---
class DualViewEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=4, max_pairs=64, n_qubits: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        dim = 2 ** n_qubits
        
        # 1. Spatial Projection (Standard View)
        # Assuming input spec_batch is [B, n_pairs, 4*dim] or similar flattened structure
        self.spatial_proj = nn.Linear(4 * dim, d_model // 2)
        
        # 2. Spectral Projection (Frequency View)
        # We project the raw truth table values to a matching dimension
        self.spectral_proj = nn.Linear(4 * dim, d_model // 2)

        # 3. Basis/Positional Embedding
        self.pos_emb = nn.Embedding(max_pairs, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=4*d_model)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, spec_batch: torch.Tensor, spec_pad_mask: torch.Tensor):
        """
        spec_batch: [B, n_pairs, 4*dim] - The Truth Table / Unitary Spec
        """
        B, n_pairs, F = spec_batch.shape
        
        # A. Spatial View (Time Domain)
        spatial = self.spatial_proj(spec_batch) # [B, n_pairs, d/2]
        
        # B. Spectral View (Frequency Domain)
        # Apply Walsh-Hadamard Transform to the SEQUENCE dimension (rows of truth table)
        # We want to see correlations across basis states, not across features.
        
        # Transpose so n_pairs is last: [B, F, n_pairs]
        spec_transposed = spec_batch.transpose(1, 2)
        
        # Apply WHT on n_pairs dimension
        spectral_raw_transposed = fast_walsh_hadamard(spec_transposed)
        
        # Transpose back: [B, n_pairs, F]
        spectral_raw = spectral_raw_transposed.transpose(1, 2)
        
        spectral = self.spectral_proj(spectral_raw) # [B, n_pairs, d/2]
        
        # C. Fusion
        x = torch.cat([spatial, spectral], dim=-1) # [B, n_pairs, d_model]
        
        # D. Add Position/Basis Info
        pos = torch.arange(n_pairs, device=spec_batch.device).unsqueeze(0).expand(B, n_pairs)
        x = x + self.pos_emb(pos)
        
        return self.encoder(x, src_key_padding_mask=spec_pad_mask)


# --- MAIN DECODER: Physics-Equipped ---
class LieAlgebraProjection(nn.Module):
    """
    The 'Quantum FFT' Layer.
    
    Transforms the linear embedding vector into the 'Lie Algebra' representation
    (Pauli Basis) and computes 'Commutator Interactions'.
    
    Analogy:
    - Neural Ops: Space Domain -> FFT -> Freq Domain -> Multiply -> IFFT
    - QCGPT:      Vector Domain -> Project -> Lie Algebra -> Commutator -> Project Back
    """
    def __init__(self, d_model, n_qubits=3):
        super().__init__()
        self.n_qubits = n_qubits
        
        # We define the dimension of the Lie Algebra su(2^N).
        # For N=3, there are 4^3 - 1 = 63 Pauli strings (plus Identity = 64).
        self.algebra_dim = 4 ** n_qubits
        
        # 1. The "Log-Map": Projects embedding into Pauli Coefficients
        self.to_algebra = nn.Linear(d_model, self.algebra_dim)
        
        # 2. The "Structure Constants": Learnable commutator weights
        # Instead of a dense matrix, we use a specialized interaction layer
        # that mimics [A, B] physics.
        self.lie_mixing = nn.Linear(self.algebra_dim, self.algebra_dim)
        
        # 3. The "Exp-Map": Projects back to embedding space
        self.from_algebra = nn.Linear(self.algebra_dim, d_model)
        
        # Learnable 'Temperature' for the basis
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: [Batch, SeqLen, d_model]
        residual = x
        
        # --- Step 1: Move to Lie Algebra Space (The "Coefficients") ---
        # Analogy: This is like taking the FFT.
        # We implicitly assume the output vector represents coefficients alpha_i 
        # for Pauli strings P_i.
        coeffs = self.to_algebra(x) # [B, L, 4^N]
        
        # --- Step 2: The Commutator Mixing (The "Physics") ---
        # In Lie Algebra, the fundamental operation is not addition, it's the Bracket.
        # [A, B] = AB - BA.
        # We simulate this by mixing the coefficients non-linearly.
        
        # We use a Tanh activation because Lie Algebras are bounded/compact logic
        # unlike ReLUs which go to infinity.
        # This mimics the "closure" of the Pauli group (Pauli * Pauli = Pauli).
        interacted = torch.tanh(self.lie_mixing(coeffs)) * self.scale
        
        # --- Step 3: Move back to Vector Space (The "Inverse") ---
        # Analogy: The IFFT.
        out = self.from_algebra(interacted)
        
        # Skip connection to preserve original features
        return residual + out
class CircuitDecoderPhysics(nn.Module):
    def __init__(self, vocab_size: int, d_model=256, n_layers=4, n_heads=4, max_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Switched to Rotary-Style Fixed Embeddings (Better for variable length circuits)
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_len)
        self.lie_transform = LieAlgebraProjection(d_model, n_qubits=3)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=4*d_model)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        
        self.out_proj = nn.Linear(d_model, vocab_size)
        
        # NEW: Entanglement Auxiliary Head
        self.entanglement_head = EntanglementHead(d_model)

    def forward(self, circ_tokens: torch.Tensor, enc_out: torch.Tensor, memory_key_padding_mask: torch.Tensor = None, return_physics=False):
        B, Lc = circ_tokens.shape
        
        # Embed + Rotary Position
        x = self.token_emb(circ_tokens)
        x = self.lie_transform(x)
        x = self.pos_emb(x)
        
        # Causal Mask
        tgt_mask = torch.triu(torch.ones(Lc, Lc, device=x.device), 1).bool()
        
        # Decode
        decoded = self.decoder(x, enc_out, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        logits = self.out_proj(decoded)
        
        if return_physics:
            # Predict entanglement at every step [B, Lc, 1]
            ent_pred = self.entanglement_head(decoded)
            return logits, ent_pred
            
        return logits

    def forward_raw(self, tgt_embeds, memory, memory_key_padding_mask=None):
        """
        Returns raw decoder hidden states (before output projection).
        Used by CircuitGenerator for physics heads during sampling.
        """
        # Embeds are already continuous, just add position
        x = self.lie_transform(tgt_embeds)
        x = self.pos_emb(x)
        
        Lc = x.size(1)
        tgt_mask = torch.triu(torch.ones(Lc, Lc, device=x.device), 1).bool()
        
        out = self.decoder(x, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        return out

    def forward_embeds(self, tgt_embeds, memory, memory_key_padding_mask=None):
        """
        Differentiable forward pass for Gumbel-Softmax inputs.
        """
        # Embeds are already continuous, just add position
        x = self.lie_transform(tgt_embeds)
        x = self.pos_emb(x)
        
        Lc = x.size(1)
        tgt_mask = torch.triu(torch.ones(Lc, Lc, device=x.device), 1).bool()
        
        out = self.decoder(x, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.out_proj(out)