import os
import argparse
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qcgpt2.gates2 import BOS_CIRC_ID2, EOS_CIRC_ID2, PAD_ID2, VOCAB2
from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.data.qiskit_utils2 import sample_task2 as sample_task
from qcgpt2.data.specs2 import build_spec_sequence_batch
from qcgpt2.encoding2 import tokens_to_circuit2, ID_TO_TOKEN2, circuit2_to_tokens
from qcgpt2.circuits2 import Circuit2, Gate2
from qcgpt2.unitaries2 import build_circuit_unitary2
from qcgpt1.evaluation.metrics import gate_count
from qcgpt1.decoding import beam_search_decode, top_k_decode, top_p_decode


def load_model(ckpt: str, device: torch.device) -> CircuitPolicy2:
    model = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=True)
        sd = state.get("model_state_dict", state)
        # Handle torch.compile prefixes ("_orig_mod.")
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model


@torch.no_grad()
def reconstruct_tokens(model: CircuitPolicy2, device: torch.device, spec_tensor: np.ndarray, max_len: int,
                       decode_strategy: str = "greedy", beam_width: int = 5, top_k: int = 5, top_p: float = 0.9) -> list:
    spec_batch_np, spec_pad_mask_np = build_spec_sequence_batch([spec_tensor])
    spec_batch = torch.tensor(spec_batch_np, dtype=torch.float32, device=device)
    spec_pad_mask = torch.tensor(spec_pad_mask_np, dtype=torch.bool, device=device)
    if decode_strategy == "greedy":
        tokens, _ = model.sample_circuit_tokens(spec_batch, spec_pad_mask, BOS_CIRC_ID2, EOS_CIRC_ID2, max_len)
    elif decode_strategy == "beam":
        tokens = beam_search_decode(model, spec_batch, spec_pad_mask, max_len=max_len, bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2, beam_width=beam_width)
    elif decode_strategy == "topk":
        tokens = top_k_decode(model, spec_batch, spec_pad_mask, max_len=max_len, bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2, k=top_k)
    elif decode_strategy == "topp":
        tokens = top_p_decode(model, spec_batch, spec_pad_mask, max_len=max_len, bos_id=BOS_CIRC_ID2, eos_id=EOS_CIRC_ID2, p=top_p)
    else:
        tokens, _ = model.sample_circuit_tokens(spec_batch, spec_pad_mask, BOS_CIRC_ID2, EOS_CIRC_ID2, max_len)
    toks = tokens[0].tolist()
    seq = []
    for t in toks:
        if t in (PAD_ID2, BOS_CIRC_ID2, EOS_CIRC_ID2):
            continue
        seq.append(t)
    return seq


def tokens_to_human(seq_ids: list) -> list:
    names = [ID_TO_TOKEN2[t] for t in seq_ids]
    out = []
    for name in names:
        if name in {"<PAD>", "<BOS_CIRC>", "<EOS_CIRC>"}:
            continue
        parts = name.split("_")
        gate = parts[0].lower()
        qs = parts[1:]
        out.append(gate + "".join(q for q in qs))
    return out


def ref_circuit_to_tokens_human(ref_circ) -> list:
    c2 = Circuit2(nqubits=ref_circ.nqubits)
    for g in ref_circ.gates:
        c2.add_gate(Gate2(g.gate_type, g.targets))
    ids = circuit2_to_tokens(c2)
    ids = [t for t in ids if ID_TO_TOKEN2[t] not in {"<PAD>", "<BOS_CIRC>", "<EOS_CIRC>"}]
    return tokens_to_human(ids)


def render_mpl_pair(ref_circ, cand_circ, title: str) -> np.ndarray:
    from qcgpt2.simulators2.qiskit_sim2 import circuit2_to_qiskit
    try:
        ref_qc = circuit2_to_qiskit(ref_circ)
        cand_qc = circuit2_to_qiskit(cand_circ)
        fig = plt.figure(figsize=(8, 3.5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for ax in (ax1, ax2):
            ax.axis("off")
        ref_qc.draw(output="mpl", ax=ax1)
        cand_qc.draw(output="mpl", ax=ax2)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(h, w, 4)
        plt.close(fig)
        return img
    except Exception:
        # Fallback to text if optional mpl deps missing
        ref_text = "Reference\n" + f"gates={len(ref_circ.gates)}"
        cand_ids = circuit2_to_tokens(cand_circ)
        cand_ids = [t for t in cand_ids if ID_TO_TOKEN2[t] not in {"<PAD>", "<BOS_CIRC>", "<EOS_CIRC>"}]
        cand_text = "\n".join(tokens_to_human(cand_ids))
        return render_text_pair(ref_text, cand_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_rows", type=int, default=8)
    parser.add_argument("--num_cols", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--max_gates_ref", type=int, default=6)
    parser.add_argument("--out_dir", type=str, default="model_evaluations")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--decode_strategy", type=str, default="greedy", choices=["greedy","beam","topk","topp"])
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device=device)

    ts = args.run_name or ("eval_" + str(np.datetime64("now").astype(str).replace("T","_")))
    out_dir = os.path.join(args.out_dir, ts)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "fidelities.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["row", "col", "fid_ref", "fid_cand"]) 

    nrows, ncols = args.num_rows, args.num_cols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4.5, nrows*4.5))
    fig.tight_layout(pad=0.8)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    fid_refs = []
    fid_cands = []

    for r in range(nrows):
        for c in range(ncols):
            spec_tensor, ref_circ = sample_task(max_gates=args.max_gates_ref)
            seq = reconstruct_tokens(model, device, spec_tensor, args.max_len, args.decode_strategy, args.beam_width, args.top_k, args.top_p)
            cand_circ = tokens_to_circuit2(seq)
            U_ref = build_circuit_unitary2(ref_circ, n_qubits=3)
            U_cand = build_circuit_unitary2(cand_circ, n_qubits=3)
            U_ref_dag = U_ref.conj().transpose(0, 1) # or .T

            # 2. Compute Inner Product: Trace(U_ref_dag @ U_cand)
            # "ij,ji->" effectively does the matmul diagonal sum
            trace = torch.einsum("ij,ji->", U_ref_dag, U_cand)

            # 3. Normalize
            fid_cand = (trace.abs() ** 2) / (8.0 ** 2) # 8.0 for 3 qubits
            fid_cands.append(fid_cand)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([r, c, "NA", f"{fid_cand:.6f}"]) 
            human_ref = ref_circuit_to_tokens_human(ref_circ)
            human_cand = tokens_to_human(seq)
            title = f"ref: {human_ref} | cand: {human_cand} | fid {fid_cand:.3f}"
            img_pair = render_mpl_pair(ref_circ, cand_circ, title)
            ax = axes[r][c]
            ax.axis("off")
            ax.imshow(img_pair)
            ax.set_title("")

    fig_path = os.path.join(out_dir, "circuits_grid.png")
    fig.savefig(fig_path, dpi=200)

    avg_csv = os.path.join(out_dir, "avg_fidelity.csv")
    with open(avg_csv, "w", newline="") as f:
        w = csv.writer(f)
        mcand = float(np.mean(fid_cands)) if len(fid_cands) > 0 else float("nan")
        w.writerow(["mean_fid_cand"]) 
        w.writerow([f"{mcand:.6f}"])

    print(f"Saved grid to {fig_path}")
    print(f"Saved fidelities CSV to {csv_path}")
    print(f"Saved average fidelity CSV to {avg_csv}")


if __name__ == "__main__":
    main()
