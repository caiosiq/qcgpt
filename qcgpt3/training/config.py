
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

@dataclass
class CurriculumStage:
    name: str
    start_epoch: int
    end_epoch: int
    max_depth: int
    lr: Optional[float] = None
    lr_enc: Optional[float] = None
    lr_dec: Optional[float] = None
    goal: Optional[str] = None
    rationale: Optional[str] = None

@dataclass
class TrainingConfig:
    # Experiment
    run_name: Optional[str] = None
    out_dir: str = "model_checkpoints"
    prefix: str = "transformer_v3"
    seed: int = 42

    # Data
    num_samples: int = 10000
    val_split: float = 0.05
    batch_size: int = 32
    num_workers: int = 16
    pin_memory: bool = True
    raw_max_depth: int = 32
    basis_only: bool = False
    n_random_states: int = 0
    
    # Augmentation
    augment_commutation: bool = False
    augment_permutation: bool = False

    # Model
    # (Model params are currently hardcoded in CircuitPolicy defaults, 
    # but we could add them here if needed)

    # Optimization
    num_epochs: int = 10
    lr: float = 3e-4
    lr_enc: Optional[float] = None
    lr_dec: Optional[float] = None
    weight_decay: float = 0.01
    optimizer: str = "adam" # adam, sgd
    momentum: float = 0.9
    
    # Scheduler
    scheduler: str = "none" # none, cosine, step, exp, exp_warmup
    t_max: int = 50
    step_size: int = 50
    gamma: float = 0.5
    warmup_epochs: int = 10

    # Objectives & Weights
    lambda_sup: float = 1.0
    
    use_unitary_loss: bool = False
    lambda_U: float = 0.0
    
    use_noise: bool = False
    lambda_noise: float = 0.0
    noise_scale: float = 1.0
    
    use_entanglement: bool = False
    lambda_ent_teacher: float = 0.0
    lambda_ent_consistency: float = 0.0

    # Temperature
    softmax_temp: float = 1.0
    temp_min: float = 0.1
    temp_schedule: str = "cosine" # cosine, linear, exp, none

    # Curriculum
    use_curriculum: bool = False
    curriculum_stages: List[Dict] = field(default_factory=list)
    # Legacy fields (kept for backward compatibility or if stages are empty)
    curriculum_start_depth: int = 8
    curriculum_end_depth: int = 32
    curriculum_max_noise: float = 1.0
    curriculum_noise_warmup: int = 20
    curriculum_depth_step: int = 4
    curriculum_increase_every: int = 5

    # Checkpointing
    ckpt: Optional[str] = None
    resume_dir: Optional[str] = None

    @classmethod
    def from_json(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
            
    def update_from_args(self, args):
        """Override config with CLI arguments"""
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
    
    def get_stage(self, epoch: int) -> Optional[CurriculumStage]:
        for stage_data in self.curriculum_stages:
            stage = CurriculumStage(**stage_data)
            if stage.start_epoch <= epoch <= stage.end_epoch:
                return stage
        return None
