# QCGPT3 Architecture

## Overview
QCGPT3 is the **Modular, High-Performance** evolution of the QCGPT series. While it builds on the differentiable physics core of QCGPT2, it completely refactors the system into reusable components, adds advanced physics capabilities (Entanglement), and solves the data generation bottleneck.

## Core Innovations

### 1. QDPE (Quantum Differentiable Physics Engine)
*   **Engine as a Class:** The physics logic (formerly scattered functions in QCGPT2) is now encapsulated in the `QDPE` class (`qcgpt3/qdpe.py`).
*   **State & Entanglement:** Added capabilities to simulate state evolution and compute **Meyer-Wallach Entanglement** (`compute_cumulative_entanglement`), enabling physics-informed auxiliary heads.
*   **Reusable:** The engine can be instantiated anywhere (Training, Evaluation, Dataset Generation) with consistent behavior.

### 2. Modular Objectives System
*   **Problem:** QCGPT2 had hardcoded loss logic (`if use_unitary_loss...`).
*   **Solution:** QCGPT3 introduces an extensible `Objective` hierarchy (`qcgpt3/training/objectives.py`).
    *   **`TrainingContext`:** Efficiently manages shared computations (e.g., generating a circuit once and reusing the trajectory for Fidelity, Noise, and Entanglement losses).
    *   **Pluggable Losses:** New losses like `EntanglementConsistencyLoss` can be added without touching the training loop.

### 3. High-Performance Data Pipeline
*   **Bottleneck Solved:** QCGPT2 relied on Qiskit for creating ground-truth datasets. QCGPT3's `HighPerformanceDataset` (`qcgpt3/data/dataset.py`) uses a CPU-optimized instance of `QDPE` to generate synthetic data.
*   **Speed:** This removes the Python-loop overhead of Qiskit, allowing for massive on-the-fly dataset generation.
*   **Augmentations:** Adds quantum-specific augmentations (Commutation Jitter, Qubit Relabeling) directly into the pipeline.

### 4. Generator Abstraction
*   **`CircuitGenerator`:** A dedicated class (`qcgpt3/models/generation.py`) handles all model sampling modes (Greedy, Temperature, Gumbel-Softmax, Teacher Forcing). This cleans up the model code and standardizes how different objectives interact with the policy.

## Summary of Upgrade
| Feature | QCGPT1 | QCGPT2 | QCGPT3 |
| :--- | :--- | :--- | :--- |
| **Paradigm** | Text Seq2Seq | Differentiable Physics (Procedural) | **Differentiable Physics (OO & Modular)** |
| **Physics Engine** | None | Functions in `supervised.py` | **`QDPE` Class (State, Unitary, Entanglement)** |
| **Dataset Gen** | Qiskit (Slow) | Qiskit (Slow) | **QDPE-CPU (Fast)** |
| **Loss Logic** | Hardcoded | Hardcoded | **Modular `Objective` & `TrainingContext`** |
| **New Physics** | None | Fidelity & Noise | **+ Entanglement & Consistency** |
