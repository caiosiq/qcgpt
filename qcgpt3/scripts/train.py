import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, LinearLR, SequentialLR
import argparse
import os
import time
import csv
import json
import sys
from typing import Tuple, List
from dataclasses import asdict
# QCGPT3 Imports
from qcgpt3.models.policy import CircuitPolicy
from qcgpt3 import GateRegistry, TensorUnitaryBackend, QDPE
from qcgpt3.training.supervised import build_high_performance_dataloader, train_supervised_epoch, evaluate_supervised_epoch
from qcgpt3.training.objectives import (
    SupervisedLoss, 
    UnitaryFidelityLoss, 
    NoisePenaltyLoss, 
    EntanglementTeacherLoss,
    EntanglementConsistencyLoss,
    WeightedSumObjective
)
from qcgpt3.training.config import TrainingConfig

def get_curriculum_params(epoch: int, config: TrainingConfig) -> Tuple[int, float]:
    """Compute curriculum learning parameters for a given epoch."""
    # Check for Staged Curriculum
    stage = config.get_stage(epoch)
    if stage:
        return stage.max_depth, config.lambda_noise # For now, we don't stage noise separately

    # Fallback to Legacy Linear Schedule
    # 1. Depth Schedule
    steps = (epoch - 1) // config.curriculum_increase_every
    current_depth = min(config.curriculum_end_depth, 
                        config.curriculum_start_depth + (steps * config.curriculum_depth_step))
    
    # 2. Noise Schedule
    if epoch < config.curriculum_noise_warmup:
        noise_weight = config.curriculum_max_noise * (epoch / config.curriculum_noise_warmup)
    else:
        noise_weight = config.curriculum_max_noise
        
    return current_depth, noise_weight

def get_temperature(epoch: int, total_epochs: int, config: TrainingConfig) -> float:
    """Compute Softmax Temperature."""
    T = max(1, total_epochs - 1)
    t = float(max(0, epoch - 1)) / T

    if config.temp_schedule == "cosine":
        return (
            config.softmax_temp
            - (config.softmax_temp - config.temp_min)
            * 0.5
            * (1 - torch.cos(torch.tensor(t * 3.1415926535))).item()
        )
    elif config.temp_schedule == "linear":
        return config.softmax_temp + t * (config.temp_min - config.softmax_temp)
    elif config.temp_schedule == "exp":
        base = config.temp_min / max(1e-6, config.softmax_temp)
        return config.softmax_temp * (base ** t)
    else:
        return config.softmax_temp

def setup_optimizer(model: CircuitPolicy, config: TrainingConfig, start_epoch: int = 1):
    # Always separate groups to allow curriculum to adjust them independently later
    # This ensures "group_name" is present for the training loop logic
    enc_ids = set(map(id, model.encoder.parameters()))
    decoder_params = [p for p in model.parameters() if id(p) not in enc_ids]
    
    # Determine initial LRs
    # 1. Try top-level config
    lr_enc = config.lr_enc
    lr_dec = config.lr_dec
    lr_base = config.lr
    
    # 2. If None, try to infer from the current curriculum stage
    if lr_enc is None or lr_dec is None:
        stage = config.get_stage(start_epoch)
        if stage:
            print(f"Initializing Optimizer with stage '{stage.name}' parameters.")
            if lr_enc is None:
                lr_enc = stage.lr_enc if stage.lr_enc is not None else stage.lr
            if lr_dec is None:
                lr_dec = stage.lr_dec if stage.lr_dec is not None else stage.lr
                
    # 3. Fallback to base lr or default 3e-4
    if lr_enc is None: lr_enc = lr_base
    if lr_dec is None: lr_dec = lr_base
    
    params = [
        {"params": list(model.encoder.parameters()), "lr": lr_enc, "group_name": "encoder"},
        {"params": decoder_params, "lr": lr_dec, "group_name": "decoder"},
    ]

    if config.optimizer == "sgd":
        optimizer = optim.SGD(params, lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
    else:
        optimizer = optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    return optimizer

def setup_scheduler(optimizer, config: TrainingConfig):
    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.t_max)
    elif config.scheduler == "step":
        return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.scheduler == "exp":
        return ExponentialLR(optimizer, gamma=config.gamma)
    elif config.scheduler == "exp_warmup":
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_epochs)
        decay = ExponentialLR(optimizer, gamma=config.gamma)
        return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[config.warmup_epochs])
    return None

def build_objectives(registry: GateRegistry, config: TrainingConfig) -> WeightedSumObjective:
    objectives = []
    weights = []
    
    # 1. Supervised (Teacher Forcing)
    objectives.append(SupervisedLoss(registry))
    weights.append(config.lambda_sup)
    
    # 2. Unitary Fidelity
    if config.use_unitary_loss:
        objectives.append(UnitaryFidelityLoss(registry, softmax_temp=config.softmax_temp))
        weights.append(config.lambda_U)
        
    # 3. Noise Penalty
    if config.use_noise:
        objectives.append(NoisePenaltyLoss(registry))
        weights.append(config.lambda_noise)
        
    # 4. Entanglement Teacher (Regression on TF state)
    if config.use_entanglement:
        objectives.append(EntanglementTeacherLoss(registry))
        weights.append(config.lambda_ent_teacher)
        
    # 5. Entanglement Consistency (Regression on AR state)
    if config.use_entanglement:
        objectives.append(EntanglementConsistencyLoss(registry))
        weights.append(config.lambda_ent_consistency)
        
    return WeightedSumObjective(objectives, weights)

def parse_args():
    parser = argparse.ArgumentParser(description="QCGPT3 Training")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Allow overriding any config field via CLI
    # We construct a dummy Config to inspect fields
    defaults = TrainingConfig()
    for field_name, field_def in defaults.__dataclass_fields__.items():
        val = getattr(defaults, field_name)
        t = type(val) if val is not None else str
        if t == bool:
             parser.add_argument(f"--{field_name}", action="store_true", default=None)
             parser.add_argument(f"--no-{field_name}", dest=field_name, action="store_false", default=None)
        else:
             parser.add_argument(f"--{field_name}", type=t, default=None)
             
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Configuration
    if args.config:
        print(f"Loading config from {args.config}")
        config = TrainingConfig.from_json(args.config)
    else:
        config = TrainingConfig()
        
    # Override with CLI args
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            setattr(config, key, value)
            
    # Resume Logic
    start_epoch = 1
    run_dir = None
    
    if config.resume_dir:
        if os.path.isdir(config.resume_dir):
            print(f"Resuming from {config.resume_dir}")
            run_dir = config.resume_dir
            # Load saved config if exists
            cfg_path = os.path.join(run_dir, "config.json")
            if os.path.exists(cfg_path):
                # We load the saved config but respect CLI overrides for restart parameters
                saved_config = TrainingConfig.from_json(cfg_path)
                # Keep some restart-specific params from current config/CLI
                saved_config.num_epochs = config.num_epochs
                saved_config.resume_dir = config.resume_dir
                config = saved_config
            
            # Find checkpoint
            files = [f for f in os.listdir(run_dir) if f.startswith(config.prefix) and f.endswith(".pt")]
            candidates = []
            for f in files:
                if "_e" in f:
                    try:
                        ep = int(f.split("_e")[-1].split(".")[0])
                        candidates.append((ep, f))
                    except: pass
            if candidates:
                candidates.sort()
                last_ep, last_ckpt = candidates[-1]
                config.ckpt = os.path.join(run_dir, last_ckpt)
                start_epoch = last_ep + 1
        else:
            print(f"Resume directory {config.resume_dir} not found.")
            
    if run_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = config.run_name or ts
        run_dir = os.path.join(config.out_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Created run directory: {run_dir}")
        
    # Save current config
    config.save(os.path.join(run_dir, "config.json"))

    # 2. Setup Device & Backend
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("medium")
        except: pass

    registry = GateRegistry(n_qubits=3)
    backend = TensorUnitaryBackend(registry, device=device)
    qdpe = QDPE(registry, backend, device=device, noise_scale=config.noise_scale)

    # 3. Model, Optimizer, Scheduler
    model = CircuitPolicy(registry=registry).to(device)
    optimizer = setup_optimizer(model, config, start_epoch)
    scheduler = setup_scheduler(optimizer, config)

    # Load Checkpoint
    if config.ckpt and os.path.isfile(config.ckpt):
        print(f"Loading checkpoint: {config.ckpt}")
        state = torch.load(config.ckpt, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            if "optimizer_state_dict" in state:
                try: optimizer.load_state_dict(state["optimizer_state_dict"])
                except: pass
            if "scheduler_state_dict" in state and scheduler:
                try: scheduler.load_state_dict(state["scheduler_state_dict"])
                except: pass
            if "epoch" in state and config.resume_dir:
                 # If we are resuming, trust the epoch from file
                 # But we calculated start_epoch above from filename, which is usually safer
                 pass
        else:
            model.load_state_dict(state)

    # 4. Objectives
    objective_fn = build_objectives(registry, config)

    # 5. Data Loaders
    def get_loader(depth):
        print(f"Building dataloader with max_depth={depth}")
        return build_high_performance_dataloader(
            registry=registry,
            qdpe=qdpe,
            num_samples=config.num_samples,
            batch_size=config.batch_size,
            n_qubits=3,
            raw_max_depth=depth,
            include_basis_states=True,
            n_random_states=(0 if config.basis_only else config.n_random_states),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory and torch.cuda.is_available(),
            augment_commutation=config.augment_commutation,
            augment_permutation=config.augment_permutation
        )

    # Initial Depth
    current_depth = config.raw_max_depth
    if config.use_curriculum:
        current_depth, _ = get_curriculum_params(start_epoch, config)
    
    full_loader = get_loader(current_depth)
    
    # Split Train/Val
    def split_loader(loader):
        dataset = loader.dataset
        N = len(dataset)
        Nv = max(1, int(config.val_split * N))
        Nt = N - Nv
        gen = torch.Generator().manual_seed(config.seed)
        train_ds, val_ds = torch.utils.data.random_split(dataset, [Nt, Nv], generator=gen)
        
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            collate_fn=loader.collate_fn
        )
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=config.pin_memory,
            collate_fn=loader.collate_fn
        )
        return train_dl, val_dl

    train_loader, val_loader = split_loader(full_loader)

    # CSV Logging
    loss_csv = os.path.join(run_dir, "loss.csv")
    csv_header = ["epoch", "train_loss", "val_loss", "depth", "noise_weight", "temp", "lr"]
    
    if not os.path.exists(loss_csv):
        with open(loss_csv, "w", newline="") as f:
            csv.writer(f).writerow(csv_header)

    best_loss = float("inf")

    # 6. Training Loop
    print(f"Starting training from epoch {start_epoch} to {config.num_epochs}")
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        # Update Curriculum
        noise_weight = config.lambda_noise
        
        # Check for Staged Curriculum
        stage = config.get_stage(epoch)
        if stage:
            new_depth = stage.max_depth
            
            # Staged LR/Scheduler Update Logic
            if epoch == stage.start_epoch:
                # 1. Determine LRs for this stage
                # Priority: stage.lr_enc/dec > stage.lr > None (keep current)
                s_lr = stage.lr
                s_enc = stage.lr_enc if stage.lr_enc is not None else s_lr
                s_dec = stage.lr_dec if stage.lr_dec is not None else s_lr
                
                updated_any = False
                if s_enc is not None or s_dec is not None:
                     print(f"Curriculum Stage '{stage.name}': Updating LRs -> Enc: {s_enc}, Dec: {s_dec}")
                     for group in optimizer.param_groups:
                         name = group.get("group_name")
                         if name == "encoder" and s_enc is not None:
                             group["lr"] = s_enc
                             updated_any = True
                         elif name == "decoder" and s_dec is not None:
                             group["lr"] = s_dec
                             updated_any = True
                         # Fallback for groups without name (e.g. if not setup by new setup_optimizer)
                         elif name is None and s_lr is not None:
                             group["lr"] = s_lr
                             updated_any = True

                # 2. Re-init Scheduler for this stage
                # We do this if we updated LRs OR if we just entered a new stage (warm restart)
                if scheduler:
                      stage_len = stage.end_epoch - stage.start_epoch + 1
                      print(f"Re-initializing Cosine Scheduler for {stage_len} epochs (Stage: {stage.name})")
                      # CosineAnnealingLR reads the CURRENT lr from optimizer.param_groups as base_lr
                      scheduler = CosineAnnealingLR(optimizer, T_max=stage_len)
            
            if new_depth != current_depth:
                print(f"Curriculum Update: Depth {current_depth} -> {new_depth} (Stage: {stage.name})")
                current_depth = new_depth
                full_loader = get_loader(current_depth)
                train_loader, val_loader = split_loader(full_loader)
        
        elif config.use_curriculum:
            # Legacy Logic
            new_depth, nw = get_curriculum_params(epoch, config)
            if config.use_noise:
                noise_weight = nw
                
            if new_depth != current_depth:
                print(f"Curriculum Update: Depth {current_depth} -> {new_depth}")
                current_depth = new_depth
                full_loader = get_loader(current_depth)
                train_loader, val_loader = split_loader(full_loader)
                
        # Update Temperature
        temp = get_temperature(epoch, config.num_epochs, config)
        
        # Update Objective Params
        objective_fn.set_curriculum_params(softmax_temp=temp)
        
        # Update Weights (dynamic noise)
        # Weights order: [Sup, U, Noise, EntT, EntC]
        current_weights = [config.lambda_sup]
        if config.use_unitary_loss: current_weights.append(config.lambda_U)
        if config.use_noise: current_weights.append(noise_weight)
        if config.use_entanglement: current_weights.append(config.lambda_ent_teacher)
        if config.use_entanglement: current_weights.append(config.lambda_ent_consistency)
        
        objective_fn.update_weights(current_weights)
        
        # Update Global Noise Scale
        qdpe.noise_scale = config.noise_scale

        # Train & Eval
        train_loss, train_metrics = train_supervised_epoch(
            model, train_loader, optimizer, device, registry, qdpe, objective_fn
        )
        
        val_loss, val_metrics = evaluate_supervised_epoch(
            model, val_loader, device, registry, qdpe, objective_fn
        )
        
        if scheduler:
            scheduler.step()
            lrs = scheduler.get_last_lr()
            if len(lrs) == 2:
                lr_str = f"LR_Enc: {lrs[0]:.2e} | LR_Dec: {lrs[1]:.2e}"
                csv_lrs = f"{lrs[0]:.2e},{lrs[1]:.2e}"
            else:
                lr_str = f"LR: {lrs[0]:.2e}"
                csv_lrs = f"{lrs[0]:.2e}"
        else:
            if config.lr_enc is not None and config.lr_dec is not None:
                lr_str = f"LR_Enc: {config.lr_enc:.2e} | LR_Dec: {config.lr_dec:.2e}"
                csv_lrs = f"{config.lr_enc:.2e},{config.lr_dec:.2e}"
            else:
                lr_str = f"LR: {config.lr:.2e}"
                csv_lrs = f"{config.lr:.2e}"

        # Logging
        def fmt_metrics(metrics):
            return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

        print(f"Epoch {epoch:03d} | T_Loss: {train_loss:.4f} ({fmt_metrics(train_metrics)}) | V_Loss: {val_loss:.4f} ({fmt_metrics(val_metrics)}) | "
              f"D: {current_depth} | N_W: {noise_weight:.2f} | Temp: {temp:.2f} | {lr_str}")
        
        with open(loss_csv, "a", newline="") as f:
            # We add metrics to CSV too? For now, stick to original schema or extend it?
            # User only asked for printing in .out.
            # Let's keep CSV stable for now to avoid breaking analysis scripts.
            csv.writer(f).writerow([epoch, train_loss, val_loss, current_depth, noise_weight, temp, csv_lrs])
            
        # Checkpointing
        metric = val_loss if config.use_unitary_loss else train_loss
        if metric < best_loss:
            best_loss = metric
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "epoch": epoch
            }, os.path.join(run_dir, f"{config.prefix}_best.pt"))
            
        if epoch % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "config": asdict(config),
                "epoch": epoch
            }, os.path.join(run_dir, f"{config.prefix}_e{epoch}.pt"))

if __name__ == "__main__":
    main()
