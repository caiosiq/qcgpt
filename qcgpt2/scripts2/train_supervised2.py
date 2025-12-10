"""QCGPT2 Supervised Training Script

Trains the QCGPT2 `CircuitPolicy2` in a supervised manner from synthetic
specification–circuit pairs.

- Builds datasets using Qiskit-based amplitude specs
- Supports curriculum (depth/noise), unitary loss, and scheduler options
- Writes only `*_best.pt` and `*_final.pt` checkpoints inside `model_checkpoints/<RUN_NAME>`

Run:
    python qcgpt2/scripts2/train_supervised2.py --num_epochs 50 --batch_size 512 \
        --out_dir model_checkpoints --run_name <stamp> --prefix transformer_v2
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, LinearLR, SequentialLR
import argparse
import os
import time
import csv
from torch.utils.data import DataLoader

from qcgpt2.models2.policy import CircuitPolicy2
from qcgpt2.gates2 import VOCAB2
from qcgpt2.training2.supervised import build_simplified_dataloader2, train_supervised_epoch2, evaluate_supervised_epoch2


def get_curriculum_params(epoch, total_epochs, start_depth=8, end_depth=32,
                          max_noise_weight=1.0, noise_warmup_epochs=20,
                          depth_step=4, increase_every=5):
    """
    Compute curriculum learning parameters for a given epoch.
    
    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        start_depth: Starting maximum circuit depth
        end_depth: Final maximum circuit depth
        max_noise_weight: Maximum lambda_noise value
        noise_warmup_epochs: Number of epochs to warmup noise weight
        depth_step: Depth increment per step
        increase_every: Epochs between depth increases
    
    Returns:
        current_depth: Current maximum depth for this epoch
        noise_weight: Current lambda_noise value for this epoch
    """
    # 1. Depth Schedule (Step-wise increase)
    steps = (epoch - 1) // increase_every
    current_depth = min(end_depth, start_depth + (steps * depth_step))
    
    # 2. Noise Schedule (Linear Warmup)
    if epoch < noise_warmup_epochs:
        # Ramp from 0.0 to max_noise_weight
        noise_weight = max_noise_weight * (epoch / noise_warmup_epochs)
    else:
        noise_weight = max_noise_weight
        
    return current_depth, noise_weight


def main():
    print("Starting Training")
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--raw_max_depth", type=int, default=32)
    parser.add_argument("--basis_only", action="store_true")
    parser.add_argument("--n_random_states", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="model_checkpoints")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="transformer_v2")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_enc", type=float, default=None)
    parser.add_argument("--lr_dec", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none","cosine","step","exp","exp_warmup"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam","sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--t_max", type=int, default=50)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--use_unitary_loss", action="store_true")
    parser.add_argument("--lambda_sup", type=float, default=1.0)
    parser.add_argument("--lambda_U", type=float, default=0.0)
    parser.add_argument("--use_noise", action="store_true")
    parser.add_argument("--lambda_noise", type=float, default=0.0)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--use_curriculum", action="store_true")
    parser.add_argument("--curriculum_start_depth", type=int, default=8)
    parser.add_argument("--curriculum_end_depth", type=int, default=32)
    parser.add_argument("--curriculum_max_noise", type=float, default=1.0)
    parser.add_argument("--curriculum_noise_warmup", type=int, default=20)
    parser.add_argument("--curriculum_depth_step", type=int, default=4)
    parser.add_argument("--curriculum_increase_every", type=int, default=5)
    parser.add_argument("--softmax_temp", type=float, default=1.0)
    parser.add_argument("--temp_min", type=float, default=0.1)
    parser.add_argument("--temp_schedule", type=str, default="cosine", choices=["cosine","linear","exp","none", "exp_warmup"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Epochs to heat up LR")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass
    # Determine run directory and optionally load config for resume
    if args.resume_dir is not None and os.path.isdir(args.resume_dir):
        print("Resume Training, Getting Parameters")
        run_dir = args.resume_dir
        write_header = False
        config_txt = os.path.join(run_dir, f"{args.prefix}_config.txt")
        if os.path.exists(config_txt):
            cfg = {}
            with open(config_txt, "r") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        cfg[k] = v
            args.prefix = cfg.get("prefix", args.prefix)
            args.lr = float(cfg.get("lr", args.lr))
            args.lr_enc = float(cfg.get("lr_enc", args.lr_enc)) if cfg.get("lr_enc") else args.lr_enc
            args.lr_dec = float(cfg.get("lr_dec", args.lr_dec)) if cfg.get("lr_dec") else args.lr_dec
            args.scheduler = cfg.get("scheduler", args.scheduler)
            args.optimizer = cfg.get("optimizer", args.optimizer)
            if "momentum" in cfg:
                try:
                    args.momentum = float(cfg["momentum"])
                except Exception:
                    pass
            args.t_max = int(cfg.get("t_max", args.t_max))
            args.step_size = int(cfg.get("step_size", args.step_size))
            args.gamma = float(cfg.get("gamma", args.gamma))
            args.warmup_epochs = int(cfg.get("warmup_epochs", args.warmup_epochs))
            args.num_workers = int(cfg.get("num_workers", args.num_workers))
            args.pin_memory = (cfg.get("pin_memory", str(args.pin_memory)).lower() in {"1","true","yes"})
            args.raw_max_depth = int(cfg.get("raw_max_depth", args.raw_max_depth))
            args.basis_only = (cfg.get("basis_only", str(args.basis_only)).lower() in {"1","true","yes"})
            args.n_random_states = int(cfg.get("n_random_states", args.n_random_states))
            args.use_unitary_loss = (cfg.get("use_unitary_loss", str(args.use_unitary_loss)).lower() in {"1","true","yes"})
            args.lambda_sup = float(cfg.get("lambda_sup", args.lambda_sup))
            args.lambda_U = float(cfg.get("lambda_U", args.lambda_U))
            args.use_noise = (cfg.get("use_noise", str(args.use_noise)).lower() in {"1","true","yes"})
            args.lambda_noise = float(cfg.get("lambda_noise", args.lambda_noise))
            args.use_curriculum = (cfg.get("use_curriculum", str(args.use_curriculum)).lower() in {"1","true","yes"})
            args.curriculum_start_depth = int(cfg.get("curriculum_start_depth", args.curriculum_start_depth))
            args.curriculum_end_depth = int(cfg.get("curriculum_end_depth", args.curriculum_end_depth))
            args.curriculum_max_noise = float(cfg.get("curriculum_max_noise", args.curriculum_max_noise))
            args.curriculum_noise_warmup = int(cfg.get("curriculum_noise_warmup", args.curriculum_noise_warmup))
            args.curriculum_depth_step = int(cfg.get("curriculum_depth_step", args.curriculum_depth_step))
            args.curriculum_increase_every = int(cfg.get("curriculum_increase_every", args.curriculum_increase_every))
            args.softmax_temp = float(cfg.get("softmax_temp", args.softmax_temp))
            args.temp_min = float(cfg.get("temp_min", args.temp_min))
            args.temp_schedule = cfg.get("temp_schedule", args.temp_schedule)
            args.batch_size = int(cfg.get("batch_size", args.batch_size)) if cfg.get("batch_size") else args.batch_size
            args.num_samples = int(cfg.get("num_samples", args.num_samples)) if cfg.get("num_samples") else args.num_samples
            args.val_split = float(cfg.get("val_split", args.val_split)) if cfg.get("val_split") else args.val_split
            total_planned = int(cfg.get("num_epochs_total", args.num_epochs)) if cfg.get("num_epochs_total") else args.num_epochs
        else:
            total_planned = args.num_epochs
    else:
        print("New Training, Creating Folder")
        ts = time.strftime("%Y%m%d_%H%M%S")
        run = args.run_name or ts
        run_dir = os.path.join(args.out_dir, run)
        os.makedirs(run_dir, exist_ok=True)
        write_header = True
        config_txt = os.path.join(run_dir, f"{args.prefix}_config.txt")
        total_planned = args.num_epochs

    # Initialize model and optimizer/scheduler per (possibly overridden) args
    model = CircuitPolicy2(vocab_size=len(VOCAB2)).to(device)
    # Disable torch.compile for faster startup and fewer state_dict surprises
    if args.lr_enc is not None and args.lr_dec is not None:
        enc_ids = set(map(id, model.encoder.parameters()))
        decoder_params = [p for p in model.parameters() if id(p) not in enc_ids]
        if args.optimizer == "sgd":
            optimizer = optim.SGD([
                {"params": list(model.encoder.parameters()), "lr": args.lr_enc},
                {"params": decoder_params, "lr": args.lr_dec},
            ], weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            optimizer = optim.Adam([
                {"params": list(model.encoder.parameters()), "lr": args.lr_enc},
                {"params": decoder_params, "lr": args.lr_dec},
            ], weight_decay=args.weight_decay)
    else:
        if args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "exp":
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == "exp_warmup":
        # 1. Warmup: Heat from 1% of LR to 100% of LR over 'warmup_epochs'
        # start_factor=0.01 means we start very cold.
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs)

        # 2. Decay: Cool down exponentially after warmup
        decay = ExponentialLR(optimizer, gamma=args.gamma)

        # 3. Chain them
        # milestones=[args.warmup_epochs] tells it when to switch from warmup to decay
        scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[args.warmup_epochs])

    # Determine initial depth for curriculum learning
    if args.use_curriculum:
        # If resuming, compute the depth for the resume epoch
        if args.resume_dir is not None and os.path.isdir(args.resume_dir):
            # start_epoch will be set later, but we can compute depth for it
            # We'll recompute this after start_epoch is determined, but set a placeholder
            initial_depth = args.curriculum_start_depth
        else:
            initial_depth = args.curriculum_start_depth
        print(f"[Train2] Curriculum learning enabled: starting depth={initial_depth}, end depth={args.curriculum_end_depth}")
    else:
        initial_depth = args.raw_max_depth
    
    # Helper function to rebuild dataset with given depth
    def rebuild_dataloaders(current_depth):
        print(f"[Train2] Building dataset with max_depth={current_depth}...")
        full_loader = build_simplified_dataloader2(
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            n_qubits=3,
            raw_max_depth=current_depth,
            include_basis_states=True if args.basis_only else True,
            n_random_states=(0 if args.basis_only else args.n_random_states),
            num_workers=min(args.num_workers, 16),
            pin_memory=True if torch.cuda.is_available() else args.pin_memory,
        )
        dataset = full_loader.dataset
        N = len(dataset)
        Nv = max(1, int(args.val_split * N))
        Nt = N - Nv
        gen = torch.Generator().manual_seed(12345)
        train_subset, val_subset = torch.utils.data.random_split(dataset, [Nt, Nv], generator=gen)
        print(f"[Train2] Split sizes: train={Nt} val={Nv}")
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=min(args.num_workers, 16), pin_memory=True if torch.cuda.is_available() else args.pin_memory,
                                  collate_fn=full_loader.collate_fn)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                                num_workers=min(args.num_workers, 16), pin_memory=True if torch.cuda.is_available() else args.pin_memory,
                                collate_fn=full_loader.collate_fn)
        return train_loader, val_loader
    
    # Build initial train/val loaders
    # Note: If resuming, we'll recompute depth after start_epoch is determined
    print("[Train2] Building initial dataset...")
    train_loader, val_loader = rebuild_dataloaders(initial_depth)
    current_depth = initial_depth
    
    # If resuming with curriculum, recompute the correct depth for the resume epoch
    # This happens after start_epoch is determined in the resume block below

    loss_csv = os.path.join(run_dir, f"{args.prefix}_loss.csv")
    print(f"[Train2] Writing logs to {loss_csv}")
    if write_header or not os.path.exists(loss_csv):
        # Build CSV header based on whether curriculum is enabled
        csv_header = ["epoch", "train_loss", "val_loss"]
        if args.use_curriculum:
            csv_header.extend(["depth", "lambda_noise"])
        # Always include noise_scale in header
        csv_header.append("noise_scale")
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow(csv_header) 
    if not os.path.exists(config_txt):
        with open(config_txt, "w") as f:
            f.write(f"prefix={args.prefix}\n")
            f.write(f"num_epochs_total={total_planned}\n")
            f.write(f"batch_size={args.batch_size}\n")
            f.write(f"num_samples={args.num_samples}\n")
            f.write(f"val_split={args.val_split}\n")
            f.write(f"lr={args.lr}\n")
            if args.lr_enc is not None: f.write(f"lr_enc={args.lr_enc}\n")
            if args.lr_dec is not None: f.write(f"lr_dec={args.lr_dec}\n")
            f.write(f"weight_decay={args.weight_decay}\n")
            f.write(f"scheduler={args.scheduler}\n")
            f.write(f"optimizer={args.optimizer}\n")
            f.write(f"momentum={args.momentum:.6f}\n")
            f.write(f"t_max={args.t_max}\n")
            f.write(f"step_size={args.step_size}\n")
            f.write(f"gamma={args.gamma}\n")
            f.write(f"warmup_epochs={args.warmup_epochs}\n")
            f.write(f"num_workers={args.num_workers}\n")
            f.write(f"pin_memory={args.pin_memory}\n")
            f.write(f"raw_max_depth={args.raw_max_depth}\n")
            f.write(f"basis_only={args.basis_only}\n")
            f.write(f"n_random_states={args.n_random_states}\n")
            f.write(f"use_unitary_loss={args.use_unitary_loss}\n")
            f.write(f"lambda_sup={args.lambda_sup}\n")
            f.write(f"lambda_U={args.lambda_U}\n")
            f.write(f"use_noise={args.use_noise}\n")
            f.write(f"lambda_noise={args.lambda_noise}\n")
            f.write(f"use_curriculum={args.use_curriculum}\n")
            f.write(f"curriculum_start_depth={args.curriculum_start_depth}\n")
            f.write(f"curriculum_end_depth={args.curriculum_end_depth}\n")
            f.write(f"curriculum_max_noise={args.curriculum_max_noise}\n")
            f.write(f"curriculum_noise_warmup={args.curriculum_noise_warmup}\n")
            f.write(f"curriculum_depth_step={args.curriculum_depth_step}\n")
            f.write(f"curriculum_increase_every={args.curriculum_increase_every}\n")
            f.write(f"softmax_temp={args.softmax_temp}\n")
            f.write(f"temp_min={args.temp_min}\n")
            f.write(f"temp_schedule={args.temp_schedule}\n")
            f.write(f"noise_scale={args.noise_scale}\n")

    start_epoch = 1
    # Resolve initial checkpoint: allow directory or file
    ckpt_path = None
    if args.ckpt:
        if os.path.isdir(args.ckpt):
            files = [f for f in os.listdir(args.ckpt) if f.endswith('.pt')]
            best = [f for f in files if f.endswith('_best.pt')]
            if best:
                ckpt_path = os.path.join(args.ckpt, best[0])
            else:
                # pick highest epoch matching prefix_e{N}.pt
                candidates = []
                for f in files:
                    try:
                        if '_e' in f:
                            e = int(f.split('_e')[-1].split('.')[0])
                            candidates.append((e, f))
                    except Exception:
                        continue
                if candidates:
                    candidates.sort()
                    ckpt_path = os.path.join(args.ckpt, candidates[-1][1])
        elif os.path.isfile(args.ckpt):
            ckpt_path = args.ckpt

    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state))
        print(f"[Train2] Loaded initial checkpoint {ckpt_path}")
    if args.resume_dir is not None and os.path.isdir(args.resume_dir):
        files = [f for f in os.listdir(run_dir) if f.startswith(args.prefix + "_e") and f.endswith(".pt")]
        if files:
            epochs = []
            for f in files:
                try:
                    e = int(f.split("_e")[-1].split(".")[0])
                    epochs.append((e, f))
                except:
                    pass
            if epochs:
                epochs.sort()
                last_e, last_f = epochs[-1]
                ckpt_path = os.path.join(run_dir, last_f)
                state = torch.load(ckpt_path, map_location=device)
                scheduler_loaded = False
                # state may be a raw state_dict or a dict payload
                if isinstance(state, dict) and "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"])
                    # Load optimizer/scheduler states if present
                    opt_sd = state.get("optimizer_state_dict", None)
                    sch_sd = state.get("scheduler_state_dict", None)
                    if opt_sd is not None:
                        try:
                            optimizer.load_state_dict(opt_sd)
                            print(f"[Train2] Loaded optimizer state from {ckpt_path}")
                        except Exception as e:
                            print(f"[Train2] Optimizer state load failed: {e}")
                    if sch_sd is not None and scheduler is not None:
                        try:
                            scheduler.load_state_dict(sch_sd)
                            scheduler_loaded = True
                            print(f"[Train2] Loaded scheduler state from {ckpt_path}")
                        except Exception as e:
                            print(f"[Train2] Scheduler state load failed: {e}")
                else:
                    model.load_state_dict(state)
                start_epoch = last_e + 1
                print(f"[Train2] Resuming from {ckpt_path} at epoch {start_epoch}")
                if scheduler is not None and not scheduler_loaded:
                    print(f"[Train2] Fast-forwarding scheduler to epoch {start_epoch - 1}...")
                    try:
                        for _ in range(start_epoch - 1):
                            scheduler.step()
                    except Exception as e:
                        print(f"[Train2] Scheduler fast-forward failed: {e}")
                # Truncate CSV rows beyond last_e
                # Also ensure header matches current curriculum settings
        if os.path.exists(loss_csv):
                    try:
                        with open(loss_csv, "r") as f:
                            rows = list(csv.reader(f))
                        # Build expected header based on curriculum settings
                        expected_header = ["epoch", "train_loss", "val_loss"]
                        if args.use_curriculum:
                            expected_header.extend(["depth", "lambda_noise"])
                        # Always include noise_scale column if not present
                        if "noise_scale" not in expected_header:
                            expected_header.append("noise_scale")
                        header = rows[0] if rows else expected_header
                        # Update header if it doesn't match (e.g., curriculum was enabled/disabled)
                        if header != expected_header:
                            print(f"[Train2] Updating CSV header: {header} -> {expected_header}")
                            header = expected_header
                        kept = [header]
                        for row in rows[1:]:
                            try:
                                ep_row = int(row[0])
                                if ep_row <= last_e:
                                    # Pad row if needed (e.g., curriculum was just enabled)
                                    while len(row) < len(header):
                                        row.append("")
                                    # Truncate row if too long (e.g., curriculum was disabled)
                                    row = row[:len(header)]
                                    kept.append(row)
                            except:
                                continue
                        with open(loss_csv, "w", newline="") as f:
                            csv.writer(f).writerows(kept)
                        print(f"[Train2] Truncated loss CSV to epochs <= {last_e}")
                    except Exception as e:
                        print(f"[Train2] CSV truncation failed: {e}")
        if start_epoch > total_planned:
            print(f"Resume requested but already completed: last_epoch={start_epoch-1} >= total={total_planned}")
            return
        args.num_epochs = total_planned - (start_epoch - 1)
        print(f"[Train2] Planned remaining epochs: {args.num_epochs}")

        # Load noise_scale from config if present to ensure consistency on resume
        try:
            import sys as _sys
            cli_has_noise_scale = any(arg.startswith("--noise_scale") for arg in _sys.argv)
            if not cli_has_noise_scale:
                with open(config_txt, "r") as f:
                    for line in f:
                        if line.startswith("noise_scale="):
                            val = float(line.strip().split("=", 1)[1])
                            args.noise_scale = val
                            print(f"[Train2] Loaded noise_scale={args.noise_scale} from config")
                            break
        except Exception:
            pass
        
        # If curriculum is enabled, recompute the correct depth for the resume epoch
        if args.use_curriculum:
            resume_depth, _ = get_curriculum_params(
                epoch=start_epoch,
                total_epochs=total_planned,
                start_depth=args.curriculum_start_depth,
                end_depth=args.curriculum_end_depth,
                max_noise_weight=args.curriculum_max_noise,
                noise_warmup_epochs=args.curriculum_noise_warmup,
                depth_step=args.curriculum_depth_step,
                increase_every=args.curriculum_increase_every
            )
            if resume_depth != current_depth:
                print(f"[Train2] Resume: adjusting depth from {current_depth} to {resume_depth} for epoch {start_epoch}")
                train_loader, val_loader = rebuild_dataloaders(resume_depth)
                current_depth = resume_depth
    best_loss = float("inf")

    # Use a *global* epoch index for temperature scheduling so that
    # resumes continue the same annealing schedule instead of
    # restarting it from T=softmax_temp.
    #
    # We already tracked the originally planned total epochs in
    # `total_planned` and wrote it to the config as `num_epochs_total`.
    # Above, on resume, we loaded that value from the config and
    # recomputed `args.num_epochs` as the *remaining* epochs.  Here we
    # build a small closure that uses the absolute epoch index `ep`
    # together with `total_planned` to recover the same temperature
    # that would have been used in the original uninterrupted run.
    def get_temperature(ep: int) -> float:
        # Map epochs 1..total_planned to t in [0, 1].  Using (ep-1)
        # ensures epoch 1 starts at exactly softmax_temp and epoch
        # total_planned reaches temp_min.
        T = max(1, total_planned - 1)
        t = float(max(0, ep - 1)) / T

        if args.temp_schedule == "cosine":
            # Cosine decay from softmax_temp -> temp_min over t in [0,1]
            return (
                args.softmax_temp
                - (args.softmax_temp - args.temp_min)
                * 0.5
                * (1 - torch.cos(torch.tensor(t * 3.1415926535))).item()
            )
        elif args.temp_schedule == "linear":
            return args.softmax_temp + t * (args.temp_min - args.softmax_temp)
        elif args.temp_schedule == "exp":
            base = args.temp_min / max(1e-6, args.softmax_temp)
            return args.softmax_temp * (base ** t)
        else:
            return args.softmax_temp

    for ep in range(start_epoch, start_epoch + args.num_epochs):
        # Temperature scheduler (global across whole training run)
        temp = get_temperature(ep)
        
        # Curriculum learning: update depth and noise weight
        effective_lambda_noise = 0.0
        if args.use_curriculum:
            new_depth, curriculum_noise_weight = get_curriculum_params(
                epoch=ep,
                total_epochs=total_planned,
                start_depth=args.curriculum_start_depth,
                end_depth=args.curriculum_end_depth,
                max_noise_weight=args.curriculum_max_noise,
                noise_warmup_epochs=args.curriculum_noise_warmup,
                depth_step=args.curriculum_depth_step,
                increase_every=args.curriculum_increase_every
            )
            
            # Rebuild dataset if depth changed
            if new_depth != current_depth:
                print(f"[Train2] Curriculum: depth changed {current_depth} -> {new_depth} at epoch {ep}")
                train_loader, val_loader = rebuild_dataloaders(new_depth)
                current_depth = new_depth
            
            # Use curriculum noise weight if use_noise is enabled
            effective_lambda_noise = curriculum_noise_weight if args.use_noise else 0.0
        else:
            effective_lambda_noise = args.lambda_noise if args.use_noise else 0.0

        train_loss = train_supervised_epoch2(
            model, train_loader, optimizer, device,
            use_unitary_loss=args.use_unitary_loss,
            lambda_sup=args.lambda_sup,
            lambda_U=args.lambda_U,
            softmax_temp=temp,
            use_noise=args.use_noise,
            lambda_noise=effective_lambda_noise,
            noise_scale=args.noise_scale,
        )
        val_loss = evaluate_supervised_epoch2(
            model, val_loader, device,
            use_unitary_loss=args.use_unitary_loss,
            lambda_sup=args.lambda_sup,
            lambda_U=args.lambda_U,
            softmax_temp=temp,
            use_noise=args.use_noise,
            lambda_noise=effective_lambda_noise,
            noise_scale=args.noise_scale,
        )
        if scheduler is not None:
            scheduler.step()
        group_lrs = ",".join(f"{pg['lr']:.3g}" for pg in optimizer.param_groups)
        curriculum_info = ""
        if args.use_curriculum:
            curriculum_info = f"  Depth={current_depth}  NoiseW={effective_lambda_noise:.4f}"
        print(f"[Supervised2] Epoch {ep:03d}  TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  LR[{group_lrs}]  Temp={temp:.3f}{curriculum_info}")
        with open(loss_csv, "a", newline="") as f:
            row = [ep, f"{train_loss:.8f}", f"{val_loss:.8f}"]
            if args.use_curriculum:
                row.extend([current_depth, f"{effective_lambda_noise:.8f}"])
            row.append(f"{args.noise_scale:.6f}")
            csv.writer(f).writerow(row)
        if (val_loss if args.use_unitary_loss else train_loss) < best_loss:
            best_loss = val_loss if args.use_unitary_loss else train_loss
            best_path = os.path.join(run_dir, f"{args.prefix}_best.pt")
            torch.save({"model_state_dict": model.state_dict()}, best_path)
        if ep % 10 == 0:
            e_path = os.path.join(run_dir, f"{args.prefix}_e{ep}.pt")
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            }, e_path)


if __name__ == "__main__":
    main()
