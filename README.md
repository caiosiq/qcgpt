QCGPT: Quantum Circuit Generation with Transformer Policies

Overview

- Goal: Given a specification of input→output quantum state pairs, generate a discrete quantum circuit that approximately implements the mapping while being as short/simple as possible.
- Current scope: 2-qubit circuits using a small gate set: `ID, X, Y, Z, H, S, T, CX, CZ, SWAP`.
- Approach: An encoder–decoder Transformer where the spec side is a continuous sequence (amplitudes) and the circuit side is a token sequence (gates + qubit indices). Supports supervised seq2seq training and REINFORCE.

Problem Specification

- Spec tensor represents full quantum states over the computational basis.
- Shape: `spec_tensor ∈ ℝ^{n_states × 2 × 2^{n_qubits} × 2}`
  - `n_states`: number of pairs used for supervision/evaluation (e.g., 4 for two-qubit basis inputs).
  - Second dim size 2: index 0 is input state, index 1 is output state.
  - `2^{n_qubits}`: number of basis vectors in the Hilbert space.
  - Last dim size 2: `[real, imag]` per amplitude.
- Semantics:
  - `spec[i, 0, j, 0] = Re(ψ_in^(i)[j])`, `spec[i, 0, j, 1] = Im(ψ_in^(i)[j])`
  - `spec[i, 1, j, 0] = Re(ψ_out^(i)[j])`, `spec[i, 1, j, 1] = Im(ψ_out^(i)[j])`
- Basis ordering: Qiskit computational basis, e.g. for 2 qubits indices 0..3 map to `|00>,|01>,|10>,|11>`.
- Primary helpers:
  - Build spec from Qiskit states: `qcgpt/data/specs.py:1-33`.
  - Inverse helper for validation: `qcgpt/data/specs.py:35-58`.
  - Batch spec into continuous sequences: `qcgpt/data/specs.py:60-97`.

Tokenization (Circuit Side)

- Vocabulary: special tokens, gate tokens, and qubit index tokens.
- Special tokens: `PAD, BOS_SPEC, EOS_SPEC, BOS_CIRC, EOS_CIRC, '->', ';'` (`qcgpt/gates.py:13-19`).
- Gate tokens: `ID, X, Y, Z, H, S, T, CX, CZ, SWAP` (`qcgpt/gates.py:5-11`).
- Qubit tokens: for 2 qubits: `q0, q1` (`qcgpt/gates.py:21-23`).
- Circuit tokenization:
  - `circuit_to_tokens(circ)`: `BOS_CIRC`, then `GATE` followed by qubit indices, then `EOS_CIRC` (`qcgpt/encoding.py:35-50`).
  - `tokens_to_circuit(tokens)`: robust decoding with arity checks and qubit token validation (`qcgpt/encoding.py:52-79`).

Data Generation

- Random reference circuits: `qcgpt/data/qiskit_utils.py:16-25`.
- Convert circuit to Qiskit and build amplitude spec using basis inputs: `qcgpt/data/qiskit_utils.py:27-33`.
- Dataset items: `{"spec_tensor": float32 tensor, "ref_circuit": Circuit}` (`qcgpt/data/dataset.py:17-22`).

Model Architecture

- Continuous Spec Encoder (`SpecEncoder`): `qcgpt/models/transformer.py:6-31`
  - Input: `spec_batch ∈ ℝ^{B × L_spec_max × 4}` where each timestep holds `(Re ψ_in, Im ψ_in, Re ψ_out, Im ψ_out)` for a `(state, basis)` pair.
  - Projection: `input_proj: ℝ^4 → ℝ^{d_model}`.
  - Positional embedding: learnable `[max_len, d_model]` added to inputs.
  - Transformer encoder: `n_layers`, `n_heads`, feedforward `4× d_model`, batch-first.
  - Padding mask: `spec_pad_mask ∈ {False,True}^{B×L}`, used as `src_key_padding_mask` to ignore padded steps.
  - Output: `enc_out ∈ ℝ^{B × L_spec_max × d_model}`.

- Circuit Decoder (`CircuitDecoder`): `qcgpt/models/transformer.py:32-70`
  - Input tokens: `circ_tokens ∈ ℤ^{B × L_circ}`; token + positional embedding.
  - Causal mask: upper-triangular to enforce autoregressive decoding.
  - Target padding mask: masks `PAD` in target.
  - Memory key padding mask: set from `spec_pad_mask` to ignore padded encoder timesteps.
  - Output projection: `ℝ^{d_model} → ℝ^{|V|}` logits per position.

- Circuit Policy (`CircuitPolicy`): `qcgpt/models/policy.py`
  - Forward:
    - Inputs: `spec_batch, spec_pad_mask, circ_tokens`.
    - Outputs: `logits ∈ ℝ^{B × L_circ × |V|}` (`qcgpt/models/policy.py:20-23`).
  - Sampling:
    - Inputs: `spec_batch, spec_pad_mask, bos_id, eos_id, max_len`.
    - Outputs: `(sampled_tokens ∈ ℤ^{B×(≤max_len+1)} padded with PAD, log_probs ∈ ℝ^{B})` (`qcgpt/models/policy.py:25-59`).
    - Log-probabilities are accumulated per step for REINFORCE.

Training Objectives

- Supervised seq2seq (teacher forcing):
  - For each batch: build `(spec_batch, spec_pad_mask)` and circuit token inputs/targets.
  - Loss: token-level cross-entropy with `ignore_index=PAD_ID`:
    - `L_sup = (1/N) ∑ CE(logits[b, t, :], target[b, t])` over non-PAD positions.
  - Optimization: AdamW, gradient clipping at norm 1.0 (`qcgpt/training/supervised.py:72-75`).

- Reinforcement Learning (REINFORCE):
  - Reward: `R = F − λ ⋅ L` where `F` is average fidelity over spec states, `L` is gate count, `λ` is length penalty (`qcgpt/training/rollouts.py:42-57`).
  - Baseline: exponential moving average `b_t` (`RewardBaseline`, `qcgpt/training/rollouts.py:75-81`).
  - Advantage: `A = R − b_t`.
  - Objective: maximize `E[R] ≈ E[A ⋅ log π(a|s)]`. Minimize loss:
    - `L_rl = − (1/B) ∑ A_i ⋅ log_probs_i` (`qcgpt/training/rl.py:72-75`).
  - Optimization: AdamW, gradient clipping at norm 1.0.

Quantum Fidelity

- Given `spec_tensor` and a candidate `circ`:
  - Reconstruct complex states per tuple `(ψ_in^(i), ψ_out_target^(i))` from amplitudes.
  - Convert `circ` to Qiskit and evolve `ψ_in^(i)` to `ψ_out_pred^(i)`.
  - Fidelity per pair: `F_i = fidelity(ψ_out_target^(i), ψ_out_pred^(i))`.
  - Average over `i`: `F = (1/n_states) ∑ F_i` (`qcgpt/evaluation/metrics.py:10-23`).

Simulators

- Qiskit statevector-based: circuit conversion and fidelity computation (`qcgpt/simulators/qiskit_sim.py:13-47`, `qcgpt/simulators/qiskit_sim.py:78-109`).
- Classical bit-level simulator retained for legacy and speed in `qcgpt/circuits.py:17-45`, but the main training path uses amplitude-based specs and quantum fidelity.

Scripts

- Supervised training: `scripts/train_supervised.py`
  - Builds dataloader from Qiskit-based specs and trains `CircuitPolicy` with CE loss.
  - Saves checkpoints to `checkpoints/supervised_epoch_XXX.pt`.

- RL training: `scripts/train_rl.py`
  - Optionally loads a supervised checkpoint.
  - Runs REINFORCE with fidelity-based rewards.
  - Saves final model to `checkpoints/rl_finetuned.pt`.

- Evaluation: `scripts/eval_policy.py`
  - Samples tasks, generates candidate circuits, prints reference vs candidate metrics and circuit listings.

End-to-End Tests (Notebook)

- `notebooks/01_qcgpt_quantum_spec_tests.ipynb` validates the full pipeline:
  - Qiskit states → spec tensor → inverse reconstruction (tolerance check).
  - Batching continuous spec sequences → SpecEncoder output shape.
  - Sampling circuit tokens and converting to `Circuit`.
  - Fidelity comparison vs reference circuit.
  - Single RL step runs with finite loss and gradients.

Project Structure

- `qcgpt/gates.py`: vocabulary, tokens, `Gate`.
- `qcgpt/circuits.py`: `Circuit`, bit-level simulator.
- `qcgpt/encoding.py`: circuit tokenization and robust decoding.
- `qcgpt/models/transformer.py`: `SpecEncoder` (continuous), `CircuitDecoder` (token-based).
- `qcgpt/models/policy.py`: encoder–decoder wiring and sampling.
- `qcgpt/data/qiskit_utils.py`: random circuits and amplitude spec generation.
- `qcgpt/data/specs.py`: amplitude spec helpers and batching.
- `qcgpt/data/dataset.py`: dataset yielding amplitude specs + reference circuits.
- `qcgpt/training/supervised.py`: dataloader, collation, training epoch.
- `qcgpt/training/rollouts.py`: spec batching for RL, reward, baseline.
- `qcgpt/training/rl.py`: REINFORCE step and loop.
- `qcgpt/simulators/qiskit_sim.py`: circuit conversion, statevector helpers, fidelities.
- `qcgpt/evaluation/metrics.py`: fidelity and gate count.
- `qcgpt/evaluation/visualize.py`: simple formatting utilities.
- `scripts/`: supervised, RL, evaluation scripts.
- `notebooks/`: sanity tests and qualitative examples.

Usage

- Install dependencies: `pip install qiskit torch numpy`.
- Supervised: `python scripts/train_supervised.py`.
- RL: `python scripts/train_rl.py` (ensure Qiskit installed; consider CUDA for speed).
- Eval: `python scripts/eval_policy.py --ckpt checkpoints/rl_finetuned.pt --num_examples 10`.

Command-Line RL Fine-Tuning

- Standard RL:
  - `python scripts/train_rl.py --num_steps 5000 --batch_size 16 --max_len 32 --lambda_len 0.1`
- Start from supervised checkpoint:
  - `python scripts/train_rl.py --ckpt checkpoints/supervised_final.pt --num_steps 5000`
- Qiskit black-box with noise (requires `qiskit-aer`):
  - `python scripts/train_rl.py --use_blackbox --method density_matrix --use_noise --p1 0.001 --p2 0.005`
  - Parameters:
    - `--method`: `statevector` or `density_matrix`
    - `--use_noise`: enable NoiseModel
    - `--p1`, `--p2`: depolarizing rates for 1q/2q gates
  - Output: `checkpoints/rl_finetuned.pt`

Command-Line Training (Supervised)

- Train from scratch and save checkpoints:
  - `python scripts/train_supervised.py --num_epochs 20 --num_samples 10000 --batch_size 64 --raw_max_depth 16`
- Resume training from a checkpoint:
  - `python scripts/train_supervised.py --ckpt checkpoints/supervised_current.pt --num_epochs 10 --num_samples 5000 --batch_size 64`
- Basis-only spec (computational basis only):
  - `python scripts/train_supervised.py --basis_only`
- Add random states to the spec (in addition to basis):
  - `python scripts/train_supervised.py --n_random_states 4`
- Checkpoints written:
  - `checkpoints/supervised_current.pt` after each epoch from epoch 10 onward
  - `checkpoints/supervised_best.pt` whenever training loss improves
  - `checkpoints/supervised_final.pt` at the end of training
- GPU usage:
  - Script auto-selects CUDA if available; ensure the environment has GPU drivers and PyTorch with CUDA support installed.

Batch Job Example (GPU)

- Example job command:
  - `python scripts/train_supervised.py --num_epochs 50 --num_samples 50000 --batch_size 128 --raw_max_depth 16`
- Resume job if preempted:
  - `python scripts/train_supervised.py --ckpt checkpoints/supervised_current.pt --num_epochs 20 --num_samples 50000 --batch_size 128`
- Tips:
  - Use larger `--batch_size` on GPU and scale `--num_samples` for better coverage.
  - Monitor logs for loss; best checkpoint is updated automatically.

Design Notes and Extensions

- Continuous spec input avoids lossy tokenization of amplitudes; circuit side remains token-based for discrete gate synthesis.
- Constrained decoding (future): enforce valid token grammars (gate → required qubit tokens) via masks.
- More qubits: parameterize qubit tokens (`q0..q{n-1}`), increase `n_basis = 2^n`, and adapt batching/masks; note exponential growth in sequence length.
- Advanced RL: actor-critic/PPO to reduce variance; learned value head; curriculum over gate count.
- Noise models: integrate Qiskit noise for robust circuit generation; adjust rewards accordingly.
- Beam search/top-k decoding: improve circuit quality and reduce degenerate tokens.

Mathematical Summary

- Spec sequence construction:
  - For each pair `i ∈ {1..n_states}` and basis index `j ∈ {0..2^n−1}`, construct features `v_{i,j} = (Re ψ_in^(i)[j], Im ψ_in^(i)[j], Re ψ_out^(i)[j], Im ψ_out^(i)[j]) ∈ ℝ^4`.
  - Flatten `(i,j)` to timestep `t = i⋅2^n + j`. Sequence length `L_spec = n_states ⋅ 2^n`.

- Encoder:
  - `h_t^0 = W_in v_t + p_t`, `W_in ∈ ℝ^{d_model×4}`, `p_t ∈ ℝ^{d_model}`.
  - Transformer layers: `h^ℓ = TFEnc(h^{ℓ−1})`, output `H ∈ ℝ^{L_spec×d_model}`.

- Decoder:
  - Token embeddings `e_u ∈ ℝ^{d_model}`, positions `q_t ∈ ℝ^{d_model}`, input `z_t = e_{u_t} + q_t`.
  - Causal self-attention on `z`, cross-attention to encoder `H` with memory masks.
  - Output logits: `o_t = W_out y_t`, `W_out ∈ ℝ^{|V|×d_model}`.

- Supervised loss:
  - `L_sup = − ∑_{t∈valid} log softmax(o_t)[y_t]`.

- REINFORCE loss:
  - Sample tokens `u_{1:T} ~ π(u_t | u_{<t}, H)` with BOS/EOS.
  - Compute `R = (1/n_states) ∑ fidelity(ψ_out_target^(i), evolve(circ, ψ_in^(i))) − λ ⋅ |circ|`.
  - Loss: `L_rl = − (R − b) ⋅ ∑_{t=1}^T log π(u_t | u_{<t}, H)`.

License and Contributions

- Contributions welcome: open PRs with tests/notebooks demonstrating improvements.