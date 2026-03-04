# QCGPT3 Architecture

## Overview
QCGPT3 is the **Modular, High-Performance** evolution of the QCGPT series. While it builds on the differentiable physics core of QCGPT2, it completely refactors the system into reusable components, introduces a physics-informed Transformer architecture, adds advanced physics capabilities (Entanglement), and solves the data generation bottleneck.

## Core Innovations

### 1. QDPE (Quantum Differentiable Physics Engine)
*   **Engine as a Class:** The physics logic is encapsulated in the `QDPE` class (`qcgpt3/qdpe.py`).
*   **State & Entanglement:** Added capabilities to simulate state evolution and compute **Meyer-Wallach Entanglement** (`compute_cumulative_entanglement`), enabling physics-informed auxiliary heads.
*   **Differentiable Unitaries:** Supports both `product` (sequence of gates) and `hamiltonian_sum` (Lie Algebra generator) methods for computing the final unitary, fully differentiable for backpropagation.
*   **Reusable:** The engine can be instantiated anywhere (Training, Evaluation, Dataset Generation) with consistent behavior.

### 2. Physics-Informed Transformer Architecture
QCGPT3 introduces inductive biases specifically for quantum circuits into the neural network itself (`qcgpt3/models/transformer.py`):
*   **DualViewEncoder:** 
    *   **Spatial View:** Standard projection of the Truth Table (Input/Output pairs).
    *   **Spectral View:** Applies a **Walsh-Hadamard Transform** to the input specification to capture global correlations and "frequency" domain features of the Boolean function/Unitary.
*   **LieAlgebraProjection (Quantum FFT Layer):** 
    *   Transforms linear embeddings into the **Lie Algebra** representation ($su(2^N)$).
    *   Simulates "commutator interactions" (mimicking $[A, B]$) via a specialized mixing layer before projecting back.
*   **EntanglementHead:** An auxiliary decoder head that predicts the entanglement state of the qubits at every step, forcing the model to learn quantum correlations.

### 3. Modular Objectives System
*   **Problem:** Previous versions had hardcoded loss logic.
*   **Solution:** Extensible `Objective` hierarchy (`qcgpt3/training/objectives.py`).
    *   **`TrainingContext`:** Efficiently manages shared computations (e.g., generating a circuit once and reusing the trajectory for Fidelity, Noise, and Entanglement losses).
    *   **Pluggable Losses:** 
        *   `SupervisedLoss`: Teacher Forcing (Cross Entropy).
        *   `UnitaryFidelityLoss`: Differentiable physics loss against ground truth unitary.
        *   `EntanglementConsistencyLoss`: Ensures the model's internal physics predictions match the generated circuit's actual physics.

### 4. High-Performance Data Pipeline
*   **Bottleneck Solved:** QCGPT3's `HighPerformanceDataset` (`qcgpt3/data/dataset.py`) uses a CPU-optimized instance of `QDPE` to generate synthetic data, removing the Qiskit overhead.
*   **Augmentations:** 
    *   **Commutation Jitter:** Randomly swaps commuting gates.
    *   **Qubit Relabeling:** Permutes qubit indices and corresponding truth table entries to encourage invariance.

### 5. Staged Curriculum Learning
*   The training loop (`qcgpt3/scripts/train.py`) supports complex curriculum schedules defined in JSON.
*   **Stages:** Can vary circuit depth, noise levels, and learning rates (Encoder/Decoder split) independently over epochs.

## Summary of Upgrade
| Feature | QCGPT1 | QCGPT2 | QCGPT3 |
| :--- | :--- | :--- | :--- |
| **Paradigm** | Text Seq2Seq | Differentiable Physics (Procedural) | **Differentiable Physics (OO & Modular)** |
| **Neural Arch** | Standard Transformer | Standard Transformer | **Physics-Informed (DualView, LieAlgebra)** |
| **Physics Engine** | None | Functions in `supervised.py` | **`QDPE` Class (State, Unitary, Entanglement)** |
| **Dataset Gen** | Qiskit (Slow) | Qiskit (Slow) | **QDPE-CPU (Fast) + Augmentations** |
| **Loss Logic** | Hardcoded | Hardcoded | **Modular `Objective` & `TrainingContext`** |
| **New Physics** | None | Fidelity & Noise | **+ Entanglement & Consistency** |
