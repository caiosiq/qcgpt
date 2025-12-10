import torch
from typing import Tuple
from .gates import PAD_ID


def _step_logits(policy, seqs: torch.Tensor, enc_out: torch.Tensor, spec_pad_mask: torch.Tensor) -> torch.Tensor:
    logits = policy.decoder(seqs, enc_out, memory_key_padding_mask=spec_pad_mask)
    return logits[:, -1, :]


@torch.no_grad()
def greedy_decode(policy, spec_batch: torch.Tensor, spec_pad_mask: torch.Tensor, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = spec_batch.device
    enc_out = policy.encoder(spec_batch, spec_pad_mask)
    B = spec_batch.size(0)
    seqs = torch.full((B, 1), policy.gates.BOS_CIRC_ID if hasattr(policy, 'gates') else 0, dtype=torch.long, device=device)
    log_probs = torch.zeros(B, dtype=torch.float32, device=device)
    for _ in range(max_len):
        logits = _step_logits(policy, seqs, enc_out, spec_pad_mask)
        next_tok = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        logp = torch.log(torch.gather(probs, 1, next_tok.unsqueeze(1)).squeeze(1) + 1e-12)
        log_probs = log_probs + logp
        seqs = torch.cat([seqs, next_tok.unsqueeze(1)], dim=1)
    return seqs, log_probs


@torch.no_grad()
def beam_search_decode(policy, spec_batch: torch.Tensor, spec_pad_mask: torch.Tensor, max_len: int, bos_id: int, eos_id: int, beam_width: int = 5, length_penalty: float = 1.0) -> torch.Tensor:
    device = spec_batch.device
    enc_out = policy.encoder(spec_batch, spec_pad_mask)
    B = spec_batch.size(0)
    final = []
    for b in range(B):
        beams = [(torch.tensor([[bos_id]], device=device, dtype=torch.long), 0.0, False)]
        for _ in range(max_len):
            new_beams = []
            for seq, score, done in beams:
                if done:
                    new_beams.append((seq, score, done))
                    continue
                logits = policy.decoder(seq, enc_out[b:b+1], memory_key_padding_mask=spec_pad_mask[b:b+1])[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                topk = torch.topk(log_probs, k=min(beam_width, log_probs.size(-1)))
                for i in range(topk.indices.size(0)):
                    tok = topk.indices[i]
                    sc = score + float(topk.values[i].item())
                    new_seq = torch.cat([seq, tok.view(1, 1)], dim=1)
                    new_done = (tok.item() == eos_id)
                    new_beams.append((new_seq, sc, new_done))
            beams = sorted(new_beams, key=lambda x: x[1] / (len(x[0][0]) ** length_penalty), reverse=True)[:beam_width]
            if all(d for _, _, d in beams):
                break
        best_seq = beams[0][0]
        final.append(best_seq)
    maxL = max(seq.size(1) for seq in final)
    out = torch.full((B, maxL), PAD_ID, dtype=torch.long, device=device)
    for i, seq in enumerate(final):
        out[i, :seq.size(1)] = seq[0]
    return out


@torch.no_grad()
def top_k_decode(policy, spec_batch: torch.Tensor, spec_pad_mask: torch.Tensor, max_len: int, bos_id: int, eos_id: int, k: int = 5) -> torch.Tensor:
    device = spec_batch.device
    enc_out = policy.encoder(spec_batch, spec_pad_mask)
    B = spec_batch.size(0)
    seqs = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        logits = policy.decoder(seqs, enc_out, memory_key_padding_mask=spec_pad_mask)[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        vals, idxs = torch.topk(probs, k=k, dim=-1)
        choice_idx = torch.multinomial(vals, num_samples=1).squeeze(1)
        next_tok = torch.gather(idxs, 1, choice_idx.unsqueeze(1)).squeeze(1)
        seqs = torch.cat([seqs, next_tok.unsqueeze(1)], dim=1)
        done |= (next_tok == eos_id)
        if done.all():
            break
    return seqs


@torch.no_grad()
def top_p_decode(policy, spec_batch: torch.Tensor, spec_pad_mask: torch.Tensor, max_len: int, bos_id: int, eos_id: int, p: float = 0.9) -> torch.Tensor:
    device = spec_batch.device
    enc_out = policy.encoder(spec_batch, spec_pad_mask)
    B = spec_batch.size(0)
    seqs = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        logits = policy.decoder(seqs, enc_out, memory_key_padding_mask=spec_pad_mask)[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idxs = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum < p
        mask[:, 0] = True
        filtered_idxs = [sorted_idxs[i, mask[i]] for i in range(B)]
        next_tok = torch.stack([fi[torch.randint(0, fi.size(0), (1,)).item()] for fi in filtered_idxs])
        seqs = torch.cat([seqs, next_tok.unsqueeze(1)], dim=1)
        done |= (next_tok == eos_id)
        if done.all():
            break
    return seqs