# QCGPT1 Architecture

## Overview
QCGPT1 is the baseline implementation of a Transformer-based model for Quantum Circuit Generation. It treats quantum circuit synthesis as a sequence-to-sequence translation task, mapping input/output state specifications to a sequence of discrete gate tokens.

## Architecture Components

### 1. Model (`qcgpt1/models/policy.py`)
*   **Transformer Encoder-Decoder:** Standard architecture.
    *   **SpecEncoder:** Encodes the input specification (pairs of input/output quantum states).
    *   **CircuitDecoder:** Autoregressively generates the quantum circuit, token by token.
*   **Tokenization:** Uses a simple, hardcoded vocabulary (mapped in `qcgpt1/gates.py`). Gates are treated as atomic tokens (e.g., `CX_0_1`, `H_0`).

### 2. Data Pipeline
*   **Input:** Pairs of `(psi_in, psi_out)` state vectors.
*   **Output:** Sequence of gate tokens.
*   **Training:** Standard Supervised Learning (Teacher Forcing) using Cross-Entropy Loss.

### 3. Simulation
*   **Backend:** Relies heavily on **Qiskit** for circuit validation and statevector simulation during data generation.
*   **Validation:** Deterministic simulation of generated circuits to check if they match the target unitary.

## Limitations
*   **Scalability:** Relying on Qiskit for state evolution is slow for large datasets.
*   **Rigidity:** Vocabulary and gate definitions are hardcoded global variables.
*   **Lack of Physics:** The model treats gates as abstract text tokens, with no understanding of their underlying physical properties (unitary matrices) or costs (noise).
