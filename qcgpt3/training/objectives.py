import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Any
from abc import ABC, abstractmethod
from qcgpt3 import GateRegistry, QDPE
from qcgpt3.models.generation import CircuitGenerator

# --- Training Context ---

class TrainingContext:
    """
    Manages cached computations and shared state for a single training step.
    This replaces ad-hoc dictionary lookups with structured access methods.
    """
    def __init__(self, 
                 generator: CircuitGenerator, 
                 qdpe: QDPE, 
                 spec_batch: torch.Tensor, 
                 spec_pad_mask: torch.Tensor, 
                 circ_in: torch.Tensor, 
                 circ_tgt: torch.Tensor, 
                 is_training: bool):
        
        self.generator = generator
        self.qdpe = qdpe
        self.spec_batch = spec_batch
        self.spec_pad_mask = spec_pad_mask
        self.circ_in = circ_in
        self.circ_tgt = circ_tgt
        self.is_training = is_training
        
        # Cache Storage
        self._tf_logits = None
        self._tf_physics = None
        self._ar_output = None # dict
        self._target_unitary = None
        self._fidelity_result = None # (U_pred, loss_U, gate_noise)
        self._target_ent_curve_tf = None
        self._target_ent_curve_ar = None

    @property
    def teacher_forcing_logits(self) -> torch.Tensor:
        """Returns logits for teacher forcing (cross-entropy)."""
        if self._tf_logits is None:
            self._tf_logits = self.generator.get_teacher_forcing_logits(
                self.spec_batch, self.spec_pad_mask, self.circ_in
            )
        return self._tf_logits

    @property
    def teacher_forcing_physics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, ent_preds) for teacher forcing."""
        if self._tf_physics is None:
            self._tf_physics = self.generator.get_physics_predictions(
                self.spec_batch, self.spec_pad_mask, self.circ_in
            )
        return self._tf_physics

    def get_autoregressive_output(self, softmax_temp: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Returns generation output (soft_probs, hard_tokens, ent_preds).
        Note: Caches result based on FIRST call. If temp varies, this might be inexact,
        but typically we use one temperature per step.
        """
        if self._ar_output is None:
            # We determine max_len from target input for consistency
            max_len = self.circ_in.size(1) if self.circ_in is not None else 32
            
            self._ar_output = self.generator.generate_autoregressive(
                self.spec_batch, 
                self.spec_pad_mask,
                max_len=max_len,
                temp=softmax_temp,
                greedy=(not self.is_training),
                return_physics=True 
            )
        return self._ar_output

    @property
    def target_unitary(self) -> torch.Tensor:
        """Computes and caches the target unitary for the ground truth circuit."""
        if self._target_unitary is None:
            with torch.no_grad():
                ids = self.circ_tgt.clamp(min=0, max=len(self.qdpe.registry.vocab)-1)
                U_tgt_seq = self.qdpe.u_stack[ids]
                self._target_unitary = self.qdpe.parallel_unitary_product(U_tgt_seq)
        return self._target_unitary

    def get_fidelity_data(self, softmax_temp: float = 1.0) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Returns (U_pred, loss_U, gate_noise) for the autoregressive generation.
        Caches computation to avoid re-running QDPE for noise/unitary separately if possible.
        """
        if self._fidelity_result is None:
            ar_out = self.get_autoregressive_output(softmax_temp)
            probs_gen = ar_out["soft_probs"]
            U_tgt = self.target_unitary
            
            # calculate_fidelity now uses shared mask internally
            U_pred, loss_U, gate_noise = self.qdpe.calculate_fidelity(probs_gen, U_tgt)
            self._fidelity_result = (U_pred, loss_U, gate_noise)
            
        return self._fidelity_result

    def get_noise_only(self, softmax_temp: float = 1.0) -> torch.Tensor:
        """
        Efficiently calculates ONLY the noise penalty without building the full unitary.
        """
        # If we already have the full result, return it
        if self._fidelity_result is not None:
            return self._fidelity_result[2]
            
        # Otherwise, compute just noise
        ar_out = self.get_autoregressive_output(softmax_temp)
        probs_gen = ar_out["soft_probs"]
        return self.qdpe.compute_noise(probs_gen)

    @property
    def target_entanglement_curve_tf(self) -> torch.Tensor:
        """Ground truth entanglement for Teacher Forcing input."""
        if self._target_ent_curve_tf is None:
            with torch.no_grad():
                ids = self.circ_in.clamp(min=0, max=len(self.qdpe.registry.vocab)-1)
                self._target_ent_curve_tf = self.qdpe.compute_cumulative_entanglement(ids)
        return self._target_ent_curve_tf

    @property
    def target_entanglement_curve_ar(self) -> torch.Tensor:
        """Ground truth entanglement for Autoregressive output."""
        if self._target_ent_curve_ar is None:
            ar_out = self.get_autoregressive_output() # uses default/cached temp
            probs_gen = ar_out["soft_probs"]
            gen_ids = torch.argmax(probs_gen, dim=-1)
            with torch.no_grad():
                self._target_ent_curve_ar = self.qdpe.compute_cumulative_entanglement(gen_ids)
        return self._target_ent_curve_ar


# --- Objectives ---

class Objective(ABC):
    @abstractmethod
    def __call__(self, ctx: TrainingContext) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates loss using the provided TrainingContext.
        """
        pass

    def set_curriculum_params(self, **kwargs):
        pass

class TeacherForcingObjective(Objective):
    pass

class AutoregressiveObjective(Objective):
    pass

# --- Implementations ---

class SupervisedLoss(TeacherForcingObjective):
    def __init__(self, registry: GateRegistry):
        self.registry = registry
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=registry.pad_id)

    def __call__(self, ctx: TrainingContext):
        logits = ctx.teacher_forcing_logits
        circ_tgt = ctx.circ_tgt
        
        B, Lc, V = logits.shape
        loss = self.ce_criterion(logits.view(B * Lc, V), circ_tgt.view(B * Lc))
        return loss, {"loss_sup": loss.item()}

class UnitaryFidelityLoss(AutoregressiveObjective):
    def __init__(self, registry: GateRegistry, softmax_temp: float = 1.0):
        self.registry = registry
        self.softmax_temp = softmax_temp

    def set_curriculum_params(self, softmax_temp: float = None, **kwargs):
        if softmax_temp is not None:
            self.softmax_temp = softmax_temp

    def __call__(self, ctx: TrainingContext):
        _, loss_U, _ = ctx.get_fidelity_data(self.softmax_temp)
        return loss_U, {"loss_U": loss_U.item()}

class NoisePenaltyLoss(AutoregressiveObjective):
    def __init__(self, registry: GateRegistry):
        self.registry = registry

    def set_curriculum_params(self, **kwargs):
        # Noise scale is handled globally by QDPE, not here
        pass

    def __call__(self, ctx: TrainingContext):
        # Assumes ctx.qdpe.noise_scale is already set correctly by the training loop
        
        # Use efficient noise-only calculation if possible
        # Note: If UnitaryFidelityLoss runs first, get_fidelity_data will cache everything.
        # If NoisePenaltyLoss runs alone/first, get_noise_only is faster.
        loss_noise = ctx.get_noise_only().mean()
        return loss_noise, {"loss_noise": loss_noise.item()}

class EntanglementTeacherLoss(TeacherForcingObjective):
    def __init__(self, registry: GateRegistry):
        self.registry = registry
        # Change from MSELoss to L1Loss (Mean Absolute Error)
        # 0.1 error -> 0.1 loss (instead of 0.01)
        self.loss_fn = nn.L1Loss(reduction='none') 

    def __call__(self, ctx: TrainingContext):
        # 1. Model Prediction
        _, pred_ent = ctx.teacher_forcing_physics
        pred_ent = pred_ent.squeeze(-1)
        
        # 2. Ground Truth
        target_ent = ctx.target_entanglement_curve_tf
        
        # 3. Loss
        mask = (ctx.circ_in != self.registry.pad_id).float()
        min_len = min(pred_ent.shape[1], target_ent.shape[1])
        
        loss_raw = self.loss_fn(pred_ent[:, :min_len], target_ent[:, :min_len])
        
        # Normalize by mask sum
        loss = (loss_raw * mask[:, :min_len]).sum() / (mask[:, :min_len].sum() + 1e-6)
        
        return loss, {"loss_ent_teacher": loss.item()}

class EntanglementConsistencyLoss(AutoregressiveObjective):
    def __init__(self, registry: GateRegistry):
        self.registry = registry
        # Change from MSELoss to L1Loss (Mean Absolute Error)
        # 0.1 error -> 0.1 loss (instead of 0.01)
        self.loss_fn = nn.L1Loss() 

    def __call__(self, ctx: TrainingContext):
        if not ctx.is_training:
            return torch.tensor(0.0, device=ctx.spec_batch.device), {}

        # 1. Model Prediction (from generation)
        ar_out = ctx.get_autoregressive_output()
        if "ent_preds" not in ar_out:
             # Should not happen if return_physics=True in context
             return torch.tensor(0.0, device=ctx.spec_batch.device), {}
        pred_ent = ar_out["ent_preds"]

        # 2. Ground Truth (of generated sequence)
        target_ent = ctx.target_entanglement_curve_ar
        
        # 3. Loss
        min_len = min(pred_ent.shape[1], target_ent.shape[1])
        loss = self.loss_fn(pred_ent[:, :min_len], target_ent[:, :min_len])
        
        return loss, {"loss_ent_consistency": loss.item()}

class WeightedSumObjective(Objective):
    def __init__(self, objectives: List[Objective], weights: List[float]):
        self.objectives = objectives
        self.weights = weights
        
    def update_weights(self, new_weights: List[float]):
        self.weights = new_weights
        
    def set_curriculum_params(self, **kwargs):
        for obj in self.objectives:
            obj.set_curriculum_params(**kwargs)
        if "weights" in kwargs:
            self.weights = kwargs["weights"]

    def __call__(self, 
                 model: nn.Module, 
                 qdpe: QDPE, 
                 spec_batch: torch.Tensor, 
                 spec_pad_mask: torch.Tensor, 
                 circ_in: torch.Tensor, 
                 circ_tgt: torch.Tensor, 
                 is_training: bool,
                 context: Any = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        # 1. Initialize Context
        generator = CircuitGenerator(model, qdpe.registry)
        ctx = TrainingContext(generator, qdpe, spec_batch, spec_pad_mask, circ_in, circ_tgt, is_training)
        
        total_loss = torch.tensor(0.0, device=spec_batch.device)
        all_metrics = {}
        
        # 2. Iterate
        for obj, w in zip(self.objectives, self.weights):
            if w > 0:
                l, m = obj(ctx)
                total_loss += w * l
                all_metrics.update(m)
                
        return total_loss, all_metrics
