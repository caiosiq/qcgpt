import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.gates2 import VOCAB2, BOS_CIRC_ID2, EOS_CIRC_ID2

def main():
    rng = np.random.RandomState(0)
    spec_raw = rng.randn(1, 2, 8, 2).astype(np.float32)
    target_tokens = torch.tensor([[BOS_CIRC_ID2, 5, 6, EOS_CIRC_ID2]], dtype=torch.long)
    spec_list = [spec_raw]
    pairs = []
    for s in spec_list:
        n_pairs, two, dim, two2 = s.shape
        P = np.zeros((n_pairs, 4 * dim), dtype=np.float32)
        in_flat = s[0, 0].reshape(-1)
        out_flat = s[0, 1].reshape(-1)
        P[0, : in_flat.shape[0]] = in_flat
        P[0, in_flat.shape[0] : in_flat.shape[0] + out_flat.shape[0]] = out_flat
        pairs.append(P)
    spec_batch_np = np.stack(pairs, axis=0)
    spec_batch = torch.tensor(spec_batch_np)
    spec_pad_mask = torch.zeros((1, 1), dtype=torch.bool)
    circ_in = target_tokens[:, :-1]
    circ_tgt = target_tokens[:, 1:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    spec_batch = spec_batch.to(device)
    spec_pad_mask = spec_pad_mask.to(device)
    circ_in = circ_in.to(device)
    circ_tgt = circ_tgt.to(device)
    for i in range(100):
        optimizer.zero_grad()
        logits = model(spec_batch, spec_pad_mask, circ_in)
        loss = criterion(logits.reshape(-1, len(VOCAB2)), circ_tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Step {i}: Loss {loss.item():.6f}")
    if loss.item() < 0.1:
        print("SUCCESS: Model can memorize.")
    else:
        print("FAILURE: Model cannot memorize even 1 item.")

if __name__ == "__main__":
    main()
