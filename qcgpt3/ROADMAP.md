# QCGPT3 Roadmap

This document outlines the future development plan for QCGPT3, focusing on scalability, advanced optimization, and usability.

## Phase 1: Usability & Consolidation (Short Term)

- [ ] **Inference CLI Tool**: Create a `scripts/solve.py` or `scripts/generate.py` that allows users to input a Truth Table (or unitary matrix) and receive a generated circuit, without running the training loop.
- [ ] **Gate Set Expansion**: Extend `GateRegistry` to support a wider range of gates, including parametric gates ($R_x(\theta)$, $R_y(\theta)$, $R_z(\theta)$) to enable variational circuit synthesis.
- [ ] **Unified Validation Script**: Consolidate the scripts in `architecture_validation/` into a single test suite (`tests/run_validation_suite.sh`) to ensure physics engine and gradient consistency before every release.
- [ ] **Pre-trained Weights**: Release a standard set of weights for 3-qubit synthesis to serve as a baseline.

## Phase 2: Advanced Optimization & RL (Medium Term)

- [ ] **Reinforcement Learning Integration**: 
    - While Differentiable Physics provides excellent gradients, it can get stuck in local optima for discrete gate sequences.
    - Implement **PPO (Proximal Policy Optimization)** using the `CircuitPolicy` to fine-tune circuits after the initial supervised/physics phase.
- [ ] **Dynamic Entanglement Curriculum**:
    - Instead of a fixed schedule, adaptively increase the target circuit complexity (entanglement) based on the model's current success rate (fidelity).
- [ ] **Beam Search Decoding**: Implement Beam Search in `CircuitPolicy.sample_circuit_tokens` to improve inference quality over greedy/sampling methods.

## Phase 3: Scalability (Long Term)

- [ ] **Sparse Encoders for N > 3**:
    - The current `DualViewEncoder` uses dense Truth Tables ($2^N$ rows). For $N=10$, this is unfeasible.
    - Develop a **Sparse Transformer Encoder** that only attends to non-zero/significant amplitudes or uses a latent representation of the unitary.
- [ ] **Multi-GPU Distributed Training**: Optimize `QDPE` and the Data Pipeline for `DistributedDataParallel` (DDP) to train on clusters.
- [ ] **Real Hardware Backend**: Interface `QDPE` with Qiskit Runtime to fine-tune circuits for specific real quantum processors (accounting for device-specific noise).

## Phase 4: Application Specificity

- [ ] **Algorithm-Specific Datasets**: Train specialized models for:
    - State Preparation
    - Oracle Synthesis
    - QFT/Arithmetic Circuits
- [ ] **OpenQASM 3.0 Support**: Full import/export support for the modern OpenQASM standard.
