# QCGPT: Quantum Circuit Generation with Transformer Policies

## Overview

**Goal**: Given a specification of input→output quantum state pairs, generate a discrete quantum circuit that approximately implements the mapping while being as short/simple as possible.

**Current scope**: 3-qubit circuits using a comprehensive gate set including standard gates (ID, X, Y, Z, H, S, T), fine-angle rotations (RX/RY/RZ with π/16, π/8, π/4, π/2), and multi-qubit gates (CX, CZ, SWAP, CCX, CCZ, CSWAP).

**Approach**: An encoder–decoder Transformer where the spec side is a continuous sequence (amplitudes) and the circuit side is a token sequence. Supports supervised seq2seq training with advanced features like temperature annealing, separate encoder/decoder learning rates, and unitary reconstruction loss.

---

## Project Structure

The codebase contains two main versions:

### QCGPT1 (`qcgpt1/`)
- Original architecture with separate gate and qubit tokens
- Legacy implementation for reference

### QCGPT2 (`qcgpt2/`) - **Current Active Version**
- **Improved tokenization**: Gate-with-target tokens (e.g., `X_0`, `CX_0_1`, `CCX_0_1_2`)
- **Unitary-based training**: Direct unitary matrix reconstruction loss
- **Advanced training features**: Temperature annealing, separate encoder/decoder learning rates
- **Better canonicalization**: Automatic gate target ordering and canonical forms

**Key QCGPT2 Files:**
- `gates2.py`: Gate+target vocabulary and token mappings
- `gate_registry2.py`: Gate canonicalization and vocabulary building
- `circuits2.py`: `Circuit2` and `Gate2` classes
- `encoding2.py`: Token ↔ circuit conversions
- `unitaries2.py`: Exact unitary matrices for all gates
- `models2/transformer.py`: `SpecEncoder` and `CircuitDecoder2`
- `models2/policy.py`: `CircuitPolicy2` wrapper
- `training2/supervised.py`: Supervised training with unitary loss
- `scripts2/train_supervised2.py`: Main training script with resume support
- `scripts2/eval_policy2.py`: Model evaluation
- `scripts2/eval_grid2.py`: Grid-based evaluation

---

## Architecture

### Dataflow

1. **Spec Construction**: Build amplitude spec from circuits → `spec_tensor` `[n_states, 2, 2^n, 2]`
2. **Batching**: Convert to continuous sequences → `spec_batch [B, L, 4]`, `spec_pad_mask [B, L]`
3. **Encoding**: `SpecEncoder` produces `enc_out [B, L, d_model]`
4. **Decoding**: `CircuitDecoder2` autoregressively produces logits over vocabulary
5. **Sampling**: Policy wrapper generates circuit tokens with EOS termination

### Transformer Components

**SpecEncoder** (`qcgpt2/models2/transformer.py`):
- Input: `spec_batch ∈ ℝ^{B × L_spec_max × 4}` (Re/Im amplitudes for input/output states)
- Projects 4 features → `d_model` via learned linear layer
- Adds learned positional embeddings
- `n_layers` Transformer encoder blocks with `n_heads` attention
- Output: `enc_out ∈ ℝ^{B × L_spec_max × d_model}`

**CircuitDecoder2** (`qcgpt2/models2/transformer.py`):
- Token embeddings + positional embeddings
- Causal self-attention mask (autoregressive)
- Cross-attention to encoder output with padding masks
- Output projection: `ℝ^{d_model} → ℝ^{|V|}` logits

**CircuitPolicy2** (`qcgpt2/models2/policy.py`):
- Wires encoder/decoder together
- Provides `sample_circuit_tokens()` for autoregressive generation
- Supports differentiable embedding-based forward pass for training

---

## Problem Specification

### Spec Tensor Format

Shape: `spec_tensor ∈ ℝ^{n_states × 2 × 2^{n_qubits} × 2}`

- `n_states`: Number of input→output state pairs
- Dimension 2: Index 0 = input state, Index 1 = output state
- `2^{n_qubits}`: Number of computational basis vectors
- Last dimension: `[real, imag]` per amplitude

**Semantics**:
- `spec[i, 0, j, 0] = Re(ψ_in^(i)[j])`
- `spec[i, 0, j, 1] = Im(ψ_in^(i)[j])`
- `spec[i, 1, j, 0] = Re(ψ_out^(i)[j])`
- `spec[i, 1, j, 1] = Im(ψ_out^(i)[j])`

**Basis ordering**: Qiskit computational basis (e.g., for 2 qubits: `|00⟩, |01⟩, |10⟩, |11⟩`)

---

## Tokenization (QCGPT2)

### Vocabulary Structure

**Special tokens**: `<PAD>`, `<BOS_CIRC>`, `<EOS_CIRC>`

**Gate tokens** (gate-with-target format):
- One-qubit: `ID_0`, `X_0`, `Y_0`, `Z_0`, `H_0`, `S_0`, `T_0`, `RX_PI_16_0`, etc. (for each qubit 0..n-1)
- Two-qubit: `CX_0_1`, `CZ_0_1`, `SWAP_0_1` (canonicalized: `a < b` for symmetric gates)
- Three-qubit: `CCX_0_1_2`, `CCZ_0_1_2`, `CSWAP_0_1_2` (canonicalized ordering)

**Key advantages over QCGPT1**:
- No separate qubit index tokens needed
- Automatic canonicalization (e.g., `CZ_1_0` → `CZ_0_1`)
- Simpler decoding (one token = one complete gate application)

---

## Training

### Supervised Training (QCGPT2)

**Main script**: `qcgpt2/scripts2/train_supervised2.py`

**Key features**:
- **Separate learning rates**: `--lr_enc` and `--lr_dec` for encoder/decoder
- **Learning rate schedulers**: `none`, `cosine`, `step`, `exp`, `exp_warmup` (with warmup epochs)
- **Temperature annealing**: `--temp_schedule` (cosine/linear/exp/none), `--softmax_temp`, `--temp_min`
- **Unitary reconstruction loss**: `--use_unitary_loss` with `--lambda_U` weight
- **Noise loss**: `--use_noise` with `--lambda_noise` weight (penalizes gate count/complexity)
- **Curriculum learning**: Progressive depth and noise weight scheduling
- **Supervised loss**: Cross-entropy with `--lambda_sup` weight
- **Resume support**: Automatically loads config, optimizer, scheduler states
- **Checkpoint management**: Saves best model, periodic epoch checkpoints, loss CSV with curriculum info

**Training modes**:
- **Supervised only**: `lambda_sup > 0`, `lambda_U = 0`
- **Unitary loss only**: `lambda_sup = 0`, `lambda_U > 0`
- **Mixed**: Both losses combined (recommended for fine-tuning)
- **With noise penalty**: `use_noise=True`, `lambda_noise > 0` adds gate count penalty to unitary loss

**Loss components**:
1. **Supervised loss**: `L_sup = CrossEntropy(logits, target_tokens)` (teacher forcing)
2. **Unitary loss**: `L_U = ||U_pred - U_target||² + λ_noise · gate_noise` where:
   - `U_pred` = unitary from autoregressively generated circuit (via Gumbel-Softmax)
   - `U_target` = unitary from reference circuit
   - `gate_noise` = weighted sum of gate costs (penalizes complex gates)
   - Uses parallel tree reduction for efficient matrix multiplication

**Temperature annealing**:
- Global schedule across total planned epochs (not restarted on resume)
- Cosine: `T(ep) = T_max - (T_max - T_min) * 0.5 * (1 - cos(π * (ep-1)/(total-1)))`
- Linear: `T(ep) = T_max + (ep-1)/(total-1) * (T_min - T_max)`
- Exponential: `T(ep) = T_max * (T_min/T_max)^((ep-1)/(total-1))`
- None: Fixed temperature (no annealing)

**Curriculum Learning**:
- Progressive difficulty scheduling for depth and noise weight
- **Depth schedule**: Step-wise increase from `start_depth` to `end_depth`
  - Increases by `depth_step` every `increase_every` epochs
  - Dataset is automatically rebuilt when depth changes
- **Noise schedule**: Linear warmup from 0 to `max_noise_weight` over `noise_warmup_epochs`
- Automatically adjusts on resume to match the correct epoch

### Command-Line Usage

**Basic training**:
```bash
python qcgpt2/scripts2/train_supervised2.py \
  --num_epochs 200 \
  --num_samples 200000 \
  --batch_size 512 \
  --lr_enc 5e-6 \
  --lr_dec 2e-5 \
  --weight_decay 0.01 \
  --scheduler exp \
  --gamma 0.985 \
  --use_unitary_loss \
  --lambda_sup 0.1 \
  --lambda_U 1.0 \
  --softmax_temp 1.0 \
  --temp_min 0.1 \
  --temp_schedule cosine \
  --raw_max_depth 8 \
  --basis_only
```

**Resume training**:
```bash
python qcgpt2/scripts2/train_supervised2.py \
  --resume_dir /path/to/checkpoint/folder
```

**Training with curriculum learning**:
```bash
python qcgpt2/scripts2/train_supervised2.py \
  --num_epochs 200 \
  --num_samples 200000 \
  --batch_size 512 \
  --lr_enc 5e-6 \
  --lr_dec 2e-5 \
  --use_unitary_loss \
  --lambda_sup 0.1 \
  --lambda_U 1.0 \
  --use_noise \
  --use_curriculum \
  --curriculum_start_depth 8 \
  --curriculum_end_depth 32 \
  --curriculum_max_noise 0.1 \
  --curriculum_noise_warmup 30 \
  --curriculum_depth_step 4 \
  --curriculum_increase_every 10 \
  --softmax_temp 1.0 \
  --temp_min 0.1 \
  --temp_schedule cosine
```

**Training with warmup scheduler**:
```bash
python qcgpt2/scripts2/train_supervised2.py \
  --num_epochs 200 \
  --scheduler exp_warmup \
  --warmup_epochs 15 \
  --gamma 0.98 \
  --lr_enc 1e-7 \
  --lr_dec 4e-6
```

The resume script automatically:
- Loads config from `{prefix}_config.txt`
- Finds latest epoch checkpoint
- Restores optimizer and scheduler states
- Continues temperature annealing from correct epoch
- Truncates loss CSV to avoid duplicates

**Key arguments**:
- `--num_epochs`: Total epochs for new runs, remaining epochs for resume
- `--num_samples`: Dataset size
- `--batch_size`: Batch size
- `--lr_enc`, `--lr_dec`: Separate learning rates (or use `--lr` for single rate)
- `--scheduler`: `none`, `cosine`, `step`, `exp`, `exp_warmup` (exponential with warmup)
- `--warmup_epochs`: Epochs for LR warmup (used with `exp_warmup` scheduler)
- `--use_unitary_loss`: Enable unitary reconstruction loss
- `--lambda_sup`, `--lambda_U`: Loss weights
- `--use_noise`: Enable noise/gate complexity penalty
- `--lambda_noise`: Weight for noise penalty (only used if `--use_noise`)
- `--use_curriculum`: Enable curriculum learning
- `--curriculum_start_depth`, `--curriculum_end_depth`: Depth range for curriculum
- `--curriculum_max_noise`: Maximum noise weight in curriculum
- `--curriculum_noise_warmup`: Epochs to warmup noise weight
- `--curriculum_depth_step`: Depth increment per step
- `--curriculum_increase_every`: Epochs between depth increases
- `--softmax_temp`, `--temp_min`, `--temp_schedule`: Temperature annealing
- `--raw_max_depth`: Maximum circuit depth in dataset (ignored if curriculum enabled)
- `--basis_only`: Use only computational basis states
- `--n_random_states`: Additional random states beyond basis
- `--ckpt`: Initial checkpoint to load (for transfer learning)
- `--resume_dir`: Directory containing previous training run

### Batch Jobs (SLURM)

**Training scripts**:
- `batch_1pretrain2.slurm`: Initial pretraining
- `batch_2midtrain2.slurm`: Mid-training with curriculum
- `batch_3teach_noise2.slurm`: Fine-tuning with noise penalty (fixed temperature)
- `batch_curriculum_learning2.slurm`: Full curriculum learning setup

**Resume** (`batch_jobs/gpt2_jobs/resume_batch_train2.slurm`):
```bash
sbatch batch_jobs/gpt2_jobs/resume_batch_train2.slurm /path/to/checkpoint/folder
```

**Configuration**: Edit SLURM scripts or set environment variables:
- `LR_ENC`, `LR_DEC`: Learning rates
- `SCHEDULER`: `none`, `cosine`, `step`, `exp`, `exp_warmup`
- `WARMUP_EPOCHS`: Epochs for LR warmup (with `exp_warmup` scheduler)
- `SOFTMAX_TEMP`, `TEMP_MIN`, `TEMP_SCHEDULE`: Temperature annealing
- `LAMBDA_SUP`, `LAMBDA_U`: Loss weights
- `USE_NOISE`, `LAMBDA_NOISE`: Noise penalty configuration
- `USE_CURRICULUM`: Enable curriculum learning
- `CURRICULUM_START_DEPTH`, `CURRICULUM_END_DEPTH`: Depth range
- `CURRICULUM_MAX_NOISE`, `CURRICULUM_NOISE_WARMUP`: Noise curriculum
- `CURRICULUM_DEPTH_STEP`, `CURRICULUM_INCREASE_EVERY`: Depth schedule
- `EPOCHS`, `SAMPLES`, `BATCH`: Training parameters

### Checkpoint Structure

Each training run creates a directory (e.g., `model_checkpoints/qcgpt2_mid_20251202_211824/`):

- `{prefix}_config.txt`: All training hyperparameters (saved automatically)
- `{prefix}_loss.csv`: Training/validation loss per epoch
  - Without curriculum: `epoch,train_loss,val_loss`
  - With curriculum: `epoch,train_loss,val_loss,depth,lambda_noise`
- `{prefix}_best.pt`: Best model (lowest validation loss)
- `{prefix}_e{N}.pt`: Epoch checkpoints (every 10 epochs)
  - Contains: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`

**Config file format** (automatically saved/loaded):
```
prefix=transformer_v2
num_epochs_total=200
batch_size=512
lr_enc=5e-06
lr_dec=2e-05
scheduler=exp_warmup
warmup_epochs=15
gamma=0.98
softmax_temp=1.0
temp_min=0.1
temp_schedule=cosine
lambda_sup=0.1
lambda_U=1.0
use_noise=True
lambda_noise=1.0
use_curriculum=True
curriculum_start_depth=8
curriculum_end_depth=32
curriculum_max_noise=1.0
curriculum_noise_warmup=20
curriculum_depth_step=4
curriculum_increase_every=5
...
```

---

## Evaluation

**Key QCGPT2 Scripts**

- `qcgpt2/scripts2/train_supervised2.py`
  - Supervised training of `CircuitPolicy2` from synthetic spec–circuit pairs
  - Options: curriculum (depth/noise), unitary loss, schedulers, separate LR for encoder/decoder
  - Outputs: `model_checkpoints/<RUN_NAME>/<PREFIX>_best.pt` and `<PREFIX>_final.pt`; loss CSV logs (with curriculum info when enabled)
  - Example:
    ```bash
    python qcgpt2/scripts2/train_supervised2.py \
      --num_epochs 200 --num_samples 200000 --batch_size 512 \
      --use_unitary_loss --lambda_sup 0.1 --lambda_U 1.0 \
      --use_curriculum --curriculum_start_depth 8 --curriculum_end_depth 32
    ```

- `qcgpt2/scripts2/eval_policy2.py`
  - Evaluates a trained policy by sampling candidate circuits for random tasks
  - Reports unitary-based fidelity and gate counts; supports decode strategies `greedy`, `beam`, `topk`, `topp`
  - Example:
    ```bash
    python qcgpt2/scripts2/eval_policy2.py --ckpt /path/to/model.pt --num_examples 100 --max_len 64
    ```

- `qcgpt2/scripts2/eval_grid2.py`
  - Generates a grid figure of Qiskit-drawn circuits comparing reference vs reconstructed candidates
  - Saves `model_evaluations/<RUN_NAME>/circuits_grid.png` and `fidelities.csv` with per-cell fidelities
  - Example:
    ```bash
    python qcgpt2/scripts2/eval_grid2.py \
      --ckpt /path/to/model.pt --num_rows 16 --num_cols 16 --max_len 32 --max_gates_ref 6 \
      --out_dir model_evaluations --run_name eval_$(date +%Y%m%d_%H%M%S)
    ```

- `qcgpt2/scripts2/compare_models.py`
  - Compares two checkpoints across a sampled dataset; produces CSV and plots in `comparison_results/`
  - Example:
    ```bash
    python qcgpt2/scripts2/compare_models.py \
      --ckpt_pre /path/to/pre.pt --ckpt_mid /path/to/mid.pt \
      --num_samples 1000 --batch_size 100 --out_dir comparison_results
    ```

- `qcgpt2/scripts2/analyse_current_training.py`
  - Loads pre/mid/fine checkpoints and evaluates them under consistent generation; writes histograms and summaries

- `qcgpt2/scripts2/eval_unitary_debug.py`
  - Diagnostic script for Gumbel-Softmax-based differentiable generation and soft unitary computation; contrasts training vs real-world fidelities

- `qcgpt2/scripts2/eval_gumbel_reconstruct.py`
  - Minimal example of constructing token sequences via Gumbel-Softmax from decoder logits

- `qcgpt2/scripts2/vocab_map2.py`
  - Prints a JSON mapping of tokens to gate types and targets (vocabulary introspection)

---

## Quantum Fidelity

Given `spec_tensor` and a candidate circuit:

1. Reconstruct complex states: `ψ_in^(i)`, `ψ_out_target^(i)` from amplitudes
2. Convert circuit to Qiskit and evolve: `ψ_out_pred^(i) = U_circ · ψ_in^(i)`
3. Compute fidelity: `F_i = |⟨ψ_out_target^(i) | ψ_out_pred^(i)⟩|²`
4. Average: `F = (1/n_states) ∑ F_i`

**Unitary-based loss** (QCGPT2):
- Directly compares `U_pred` (from generated circuit) vs `U_target` (from reference)
- Uses Frobenius norm: `||U_pred - U_target||²_F`
- More efficient than state-by-state fidelity for training

---

## Installation

**Dependencies**:
```bash
pip install torch numpy qiskit matplotlib pandas
```

**Environment setup**:
- Python 3.8+
- PyTorch with CUDA support (recommended)
- Qiskit for quantum simulation

**Path setup**:
```bash
export PYTHONPATH=/path/to/qcgpt:$PYTHONPATH
```

---

## Project Structure (Detailed)

### QCGPT2 Core (`qcgpt2/`)

**Models**:
- `models2/transformer.py`: `SpecEncoder`, `CircuitDecoder2`
- `models2/policy.py`: `CircuitPolicy2`

**Data**:
- `data/dataset2.py`: `SimplifiedCircuitDataset2` (generates random circuits)
- `data/specs2.py`: Spec tensor construction and batching

**Gates & Circuits**:
- `gates2.py`: Vocabulary and token mappings
- `gate_registry2.py`: Gate canonicalization, vocabulary building
- `circuits2.py`: `Circuit2`, `Gate2` classes
- `encoding2.py`: Token ↔ circuit conversions
- `unitaries2.py`: Exact unitary matrices for all gates

**Training**:
- `training2/supervised.py`: Training loop, unitary loss, Gumbel-Softmax generation
- `scripts2/train_supervised2.py`: Main training script with resume support

**Evaluation**:
- `scripts2/eval_policy2.py`: Single model evaluation
- `scripts2/eval_grid2.py`: Grid-based evaluation
- `scripts2/compare_models.py`: Multi-model comparison

**Simulators**:
- `simulators2/`: Qiskit-based quantum simulation

**Tests**:
- `tests2/`: Unit tests for components

### QCGPT1 (Legacy) (`qcgpt1/`)

Similar structure but with separate gate/qubit tokenization. See original README sections for details.

### Batch Jobs (`batch_jobs/`)

- `gpt2_jobs/`: SLURM scripts for QCGPT2 training/evaluation
  - `batch_1pretrain2.slurm`: Initial pretraining
  - `batch_2midtrain2.slurm`: Mid-training with curriculum
  - `batch_3teach_noise2.slurm`: Fine-tuning with noise penalty
  - `batch_curriculum_learning2.slurm`: Curriculum learning configuration
  - `batch_eval2.slurm`: Model evaluation
  - `batch_analyse_training.slurm`: Training analysis
  - `resume_batch_train2.slurm`: Resume training from checkpoint
- `gpt1_jobs/`: SLURM scripts for QCGPT1 (legacy)
- `logs/`: Job output logs

### Output Directories

- `model_checkpoints/`: Training runs (one directory per run)
- `model_evaluations/`: Evaluation results
- `comparison_results/`: Model comparison outputs
- `saved_models/`: Final exported models

---

## Mathematical Summary

### Spec Sequence Construction

For each pair `i ∈ {1..n_states}` and basis index `j ∈ {0..2^n−1}`:
- Features: `v_{i,j} = (Re ψ_in^(i)[j], Im ψ_in^(i)[j], Re ψ_out^(i)[j], Im ψ_out^(i)[j]) ∈ ℝ^4`
- Flatten to timestep: `t = i·2^n + j`
- Sequence length: `L_spec = n_states · 2^n`

### Encoder

- Input projection: `h_t^0 = W_in v_t + p_t` where `W_in ∈ ℝ^{d_model×4}`, `p_t ∈ ℝ^{d_model}`
- Transformer layers: `h^ℓ = TFEnc(h^{ℓ−1})`
- Output: `H ∈ ℝ^{L_spec×d_model}`

### Decoder

- Token embeddings: `e_u ∈ ℝ^{d_model}`, positions: `q_t ∈ ℝ^{d_model}`
- Input: `z_t = e_{u_t} + q_t`
- Causal self-attention on `z`, cross-attention to encoder `H`
- Output logits: `o_t = W_out y_t` where `W_out ∈ ℝ^{|V|×d_model}`

### Loss Functions

**Supervised loss**:
- `L_sup = − ∑_{t∈valid} log softmax(o_t)[y_t]`

**Unitary loss** (QCGPT2):
- Generate circuit via Gumbel-Softmax: `probs = GumbelSoftmax(logits, temp)`
- Compute unitary: `U_pred = ∏_t U_gate(probs[t])` (parallel tree reduction)
- Loss: `L_U = ||U_pred - U_target||²_F + λ_noise · gate_noise`

**Total loss**:
- `L = λ_sup · L_sup + λ_U · L_U`
- If `use_noise=True`: `L_U` includes gate complexity penalty

---

## Design Notes

### QCGPT2 Improvements

1. **Gate-with-target tokens**: Simpler vocabulary, no grammar constraints needed
2. **Unitary loss**: Direct physics-based training signal
3. **Noise penalty**: Gate complexity regularization via `lambda_noise`
4. **Curriculum learning**: Progressive difficulty scheduling for depth and noise
5. **Temperature annealing**: Smooth exploration → exploitation transition
6. **Learning rate warmup**: `exp_warmup` scheduler with configurable warmup epochs
7. **Separate learning rates**: Fine-grained control over encoder/decoder training
8. **Resume support**: Seamless continuation with state preservation and curriculum adjustment
9. **Enhanced logging**: CSV includes curriculum depth and noise weight when enabled

### Future Extensions

- **Constrained decoding**: Enforce valid gate sequences via masks
- **More qubits**: Scale to 4+ qubits (note exponential growth in spec length)
- **Beam search**: Improve generation quality
- **RL fine-tuning**: REINFORCE with fidelity rewards
- **Noise models**: Train on noisy quantum hardware

---

## License and Contributions

Contributions welcome! Please open PRs with tests demonstrating improvements.
