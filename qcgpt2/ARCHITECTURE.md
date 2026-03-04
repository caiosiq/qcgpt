# QCGPT2 Architecture

## Overview
QCGPT2 represents a major leap from QCGPT1 by introducing **Differentiable Physics** and **Hardware Awareness**. Unlike QCGPT1, which treats circuits purely as text, QCGPT2 simulates the quantum mechanics of the generated circuits within the training loop, allowing gradients to flow from physical metrics (Fidelity) back to the model parameters.

## Core Innovations

### 1. Differentiable Physics (DiffPhys)
*   **Implementation:** Implemented directly in `qcgpt2/training2/supervised.py`.
*   **Gumbel-Softmax:** Uses the Straight-Through Gumbel Estimator to generate "soft" gate sequences. This allows the discrete choices of the Transformer to be differentiable.
*   **Unitary Reconstruction:** Maps soft gate tokens to their Unitary Matrices and computes the final circuit unitary $U_{pred}$ using parallel tensor contraction (`parallel_unitary_product`).
*   **Fidelity Loss:** Trains the model to maximize $|\text{Tr}(U_{target}^\dagger U_{pred})|^2$, effectively learning "how to build a unitary" rather than just "how to copy a string."

### 2. Hardware-Aware Noise Modeling
*   **Cost Injection:** Uses a `GateRegistry` (`qcgpt2/gate_registry2.py`) to assign error costs to gates (e.g., CNOT is 10x more expensive than 1-qubit gates).
*   **Noise Penalty:** The training objective includes a differentiable noise term. The model learns to avoid expensive gates (like SWAPs) unless they are necessary for the unitary, effectively performing **Differentiable Architecture Search**.

### 3. Architecture
*   **Model:** Standard Transformer Encoder-Decoder (`CircuitPolicy2`).
*   **Tokenization:** Refined vocabulary (`CX_0_1` vs `CX_1_0`) handled by the registry.

## Limitations
*   **Monolithic Code:** The physics engine, loss calculation, and training loop are tightly coupled in `supervised.py`, making it hard to extend or debug.
*   **Dataset Bottleneck:** While *training* uses a fast differentiable simulator, the *dataset generation* (ground truth creation) still relies on Qiskit, which is slow.
*   **Limited Physics:** Focuses primarily on Unitary Fidelity and simple gate error sums. It lacks advanced metrics like Entanglement or state-vector evolution.
