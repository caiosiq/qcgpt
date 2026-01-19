import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from qcgpt3 import GateRegistry

class CircuitGenerator:
    """
    Handles all model generation logic:
    1. Teacher Forcing (for supervised training)
    2. Autoregressive Generation (Greedy / Sampling / Gumbel-Softmax for differentiable physics)
    3. Entanglement/Physics predictions
    """
    def __init__(self, model: nn.Module, registry: GateRegistry):
        self.model = model
        self.registry = registry
        
        # Ensure model has output_head if we rely on it
        if not hasattr(self.model, "output_head"):
            # Usually policy.decoder.out_proj is the head
            # But policy doesn't expose it directly as 'output_head'
            # We can create a wrapper or just access decoder.out_proj
            pass

    def _get_logits_from_hidden(self, hidden):
        if hasattr(self.model, "output_head"):
            return self.model.output_head(hidden)
        elif hasattr(self.model.decoder, "out_proj"):
            return self.model.decoder.out_proj(hidden)
        else:
            raise AttributeError("Model must have output_head or decoder.out_proj")

    def get_teacher_forcing_logits(self, 
                                   spec_batch: torch.Tensor, 
                                   spec_pad_mask: torch.Tensor, 
                                   circ_in: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for cross-entropy loss.
        """
        # (B, L, V)
        logits = self.model(spec_batch, spec_pad_mask, circ_in)
        return logits

    def generate_autoregressive(self, 
                                spec_batch: torch.Tensor, 
                                spec_pad_mask: torch.Tensor, 
                                max_len: int = 32, 
                                temp: float = 1.0, 
                                greedy: bool = False,
                                return_hard_tokens: bool = False,
                                return_physics: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generates a circuit autoregressively.
        Supports Gumbel-Softmax for differentiable gradients.
        """
        B = spec_batch.size(0)
        device = spec_batch.device
        
        # 1. Encode Truth Table
        memory = self.model.encoder(spec_batch, spec_pad_mask)
        
        # 2. Setup Embedding Access
        w_emb = self.model.decoder.token_emb.weight # (Vocab, D_model)
        
        # 3. Start with <BOS>
        curr_input = torch.full((B, 1), self.registry.bos_id, dtype=torch.long, device=device)
        curr_embeds = self.model.decoder.token_emb(curr_input) # (B, 1, D)
        
        history_embeds = curr_embeds
        soft_probs_list = []
        token_list = []
        ent_preds_list = []
        
        # 4. The Autoregressive Loop
        for t in range(max_len):
            # Use forward_raw to get hidden states if available, else forward_embeds
            # Assuming Policy has access to decoder which has forward_raw
            if hasattr(self.model.decoder, "forward_raw"):
                decoder_out = self.model.decoder.forward_raw(history_embeds, memory, memory_key_padding_mask=spec_pad_mask)
                last_step_hidden = decoder_out[:, -1, :] # (B, D)
                next_token_logits = self._get_logits_from_hidden(last_step_hidden)
            else:
                # Fallback (slower or less capable)
                decoder_out = self.model.decoder.forward_embeds(history_embeds, memory, memory_key_padding_mask=spec_pad_mask)
                last_step_hidden = decoder_out[:, -1, :]
                next_token_logits = self._get_logits_from_hidden(last_step_hidden)

            if greedy:
                token_id = torch.argmax(next_token_logits, dim=-1)
                next_token_onehot = F.one_hot(token_id, num_classes=next_token_logits.size(-1)).float()
            else:
                next_token_onehot = F.gumbel_softmax(next_token_logits, tau=temp, hard=True, dim=-1)
                token_id = torch.argmax(next_token_onehot, dim=-1)
            
            soft_probs_list.append(next_token_onehot)
            if return_hard_tokens:
                token_list.append(token_id)
            
            if return_physics and last_step_hidden is not None:
                # Use model's decoder physics head if available
                if hasattr(self.model.decoder, "entanglement_head"):
                    ent_pred = torch.sigmoid(self.model.decoder.entanglement_head(last_step_hidden))
                    ent_preds_list.append(ent_pred)
                elif hasattr(self.model, "entanglement_head"):
                    ent_pred = torch.sigmoid(self.model.entanglement_head(last_step_hidden))
                    ent_preds_list.append(ent_pred)

            # Next Step Input
            next_embed = torch.matmul(next_token_onehot, w_emb).unsqueeze(1)
            history_embeds = torch.cat([history_embeds, next_embed], dim=1)
            
        full_probs = torch.stack(soft_probs_list, dim=1).float() # (B, L, V)
        
        outputs = {"soft_probs": full_probs}
        if return_hard_tokens:
            outputs["hard_tokens"] = torch.stack(token_list, dim=1)
        if return_physics and ent_preds_list:
             outputs["ent_preds"] = torch.stack(ent_preds_list, dim=1).squeeze(-1)
            
        return outputs

    def get_physics_predictions(self, 
                                spec_batch: torch.Tensor, 
                                spec_pad_mask: torch.Tensor, 
                                circ_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, entanglement_preds) using teacher forcing.
        """
        logits, ent_pred = self.model(spec_batch, spec_pad_mask, circ_tokens, return_physics=True)
        return logits, ent_pred
