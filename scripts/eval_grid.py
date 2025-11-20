import os
import argparse
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit.exceptions import MissingOptionalLibraryError

from qcgpt.gates import VOCAB, PAD_ID, BOS_CIRC_ID, EOS_CIRC_ID
from qcgpt.models.policy import CircuitPolicy
from qcgpt.data.qiskit_utils import sample_task
from qcgpt.data.specs import build_spec_sequence_batch
from qcgpt.encoding import tokens_to_circuit
from qcgpt.evaluation.metrics import quantum_fidelity_from_spec
from qcgpt.evaluation.visualize import format_circuit
from qcgpt.simulators.qiskit_sim import circuit_to_qiskit


def load_model(ckpt: str, device: torch.device) -> CircuitPolicy:
    model = CircuitPolicy(
        vocab_size=len(VOCAB),
        d_model=256,
        n_layers=4,
        n_heads=4,
        max_spec_len=256,
        max_circ_len=128,
    ).to(device)
    if ckpt and os.path.exists(ckpt):
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=device)
        sd = state["model_state_dict"]
        try:
            model.load_state_dict(sd)
        except RuntimeError:
            with torch.no_grad():
                if "decoder.token_emb.weight" in sd:
                    old_emb = sd["decoder.token_emb.weight"]
                    new_emb = model.decoder.token_emb.weight
                    n = min(old_emb.shape[0], new_emb.shape[0])
                    new_emb[:n].copy_(old_emb[:n])
                if "decoder.out_proj.weight" in sd:
                    old_out_w = sd["decoder.out_proj.weight"]
                    new_out_w = model.decoder.out_proj.weight
                    n = min(old_out_w.shape[0], new_out_w.shape[0])
                    new_out_w[:n].copy_(old_out_w[:n])
                if "decoder.out_proj.bias" in sd:
                    old_out_b = sd["decoder.out_proj.bias"]
                    new_out_b = model.decoder.out_proj.bias
                    n = min(old_out_b.shape[0], new_out_b.shape[0])
                    new_out_b[:n].copy_(old_out_b[:n])
            for k in [
                "decoder.token_emb.weight",
                "decoder.out_proj.weight",
                "decoder.out_proj.bias",
            ]:
                if k in sd:
                    del sd[k]
            model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def reconstruct_circuit(model: CircuitPolicy, device: torch.device, spec_tensor: np.ndarray, max_len: int) -> list:
    spec_batch_np, spec_pad_mask_np = build_spec_sequence_batch([spec_tensor])
    spec_batch = torch.tensor(spec_batch_np, dtype=torch.float32, device=device)
    spec_pad_mask = torch.tensor(spec_pad_mask_np, dtype=torch.bool, device=device)
    tokens, _ = model.sample_circuit_tokens(
        spec_batch=spec_batch,
        spec_pad_mask=spec_pad_mask,
        bos_id=BOS_CIRC_ID,
        eos_id=EOS_CIRC_ID,
        max_len=max_len,
    )
    seq = [t for t in tokens[0].tolist() if t != PAD_ID]
    return seq


def render_qc_image(qc) -> np.ndarray:
    try:
        fig = qc.draw(output="mpl")
    except MissingOptionalLibraryError:
        text = qc.draw(output="text")
        fig = plt.figure(figsize=(6, 2))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0, 1, str(text), family="monospace", va="top", ha="left", fontsize=6)
        fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h, w, 4)
    plt.close(fig)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_rows", type=int, default=16)
    parser.add_argument("--num_cols", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--max_gates_ref", type=int, default=6)
    parser.add_argument("--out_dir", type=str, default="model_evaluations")
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device=device)

    ts = args.run_name or ("eval_" + str(np.datetime64("now").astype(str).replace("T","_")))
    out_dir = os.path.join(args.out_dir, ts)
    os.makedirs(out_dir, exist_ok=True)

    # Prepare CSV
    csv_path = os.path.join(out_dir, "fidelities.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "fid_ref", "fid_recon"]) 

    # Build 16x16 figure of circuits
    nrows, ncols = args.num_rows, args.num_cols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*2.0, nrows*2.0))
    fig.tight_layout(pad=0.8)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    fid_refs = []
    fid_cands = []
    # For each cell, sample a task, reconstruct, compute fidelity, and render text
    for r in range(nrows):
        for c in range(ncols):
            spec_tensor, ref_circ = sample_task(max_gates=args.max_gates_ref)
            seq = reconstruct_circuit(model, device, spec_tensor, args.max_len)
            cand_circ = tokens_to_circuit(seq)
            fid_ref = quantum_fidelity_from_spec(spec_tensor, ref_circ)
            fid_cand = quantum_fidelity_from_spec(spec_tensor, cand_circ)
            fid_refs.append(fid_ref)
            fid_cands.append(fid_cand)

            # Save CSV row
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([r, c, f"{fid_ref:.6f}", f"{fid_cand:.6f}"]) 

            qc_ref = circuit_to_qiskit(ref_circ)
            qc_cand = circuit_to_qiskit(cand_circ)
            img_ref = render_qc_image(qc_ref)
            img_cand = render_qc_image(qc_cand)
            hmin = min(img_ref.shape[0], img_cand.shape[0])
            img_ref = img_ref[:hmin, :, :]
            img_cand = img_cand[:hmin, :, :]
            img_pair = np.concatenate([img_ref, img_cand], axis=1)
            ax = axes[r][c]
            ax.axis("off")
            ax.imshow(img_pair)
            ax.set_title(f"R {fid_ref:.3f} | C {fid_cand:.3f}", fontsize=6)

    fig_path = os.path.join(out_dir, "circuits_grid.png")
    fig.savefig(fig_path, dpi=200)

    avg_csv = os.path.join(out_dir, "avg_fidelity.csv")
    with open(avg_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mean_fid_ref", "mean_fid_cand"]) 
        mref = float(np.mean(fid_refs)) if len(fid_refs) > 0 else float("nan")
        mcand = float(np.mean(fid_cands)) if len(fid_cands) > 0 else float("nan")
        writer.writerow([f"{mref:.6f}", f"{mcand:.6f}"])

    # Also write a simple README-like text with summary counts
    # (kept minimal; no markdown creation beyond this file)
    print(f"Saved grid to {fig_path}")
    print(f"Saved fidelities CSV to {csv_path}")
    print(f"Saved average fidelity CSV to {avg_csv}")


if __name__ == "__main__":
    main()