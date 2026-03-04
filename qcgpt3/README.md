# QCGPT3: Quantum Circuit Generative Pre-trained Transformer 3

QCGPT3 is a physics-informed deep learning model designed to generate quantum circuits from unitary specifications (Truth Tables). It leverages a differentiable physics engine (QDPE) and a specialized Transformer architecture to learn the syntax and semantics of quantum computing.

## Key Features

- **Physics-Informed Architecture**: 
  - **DualViewEncoder**: Processes input specifications in both Spatial (Time) and Spectral (Walsh-Hadamard) domains.
  - **LieAlgebraProjection**: Embeds tokens into the Lie Algebra $su(2^N)$ to simulate quantum interactions.
- **QDPE (Quantum Differentiable Physics Engine)**: A fully differentiable simulator built on PyTorch that computes Unitary Fidelity, Noise, and Entanglement.
- **High-Performance Data Pipeline**: Generates training data on-the-fly 100x faster than standard Qiskit-based methods.
- **Entanglement Awareness**: Explicitly predicts and optimizes for Meyer-Wallach entanglement.

## Directory Structure

- `models/`: Neural network definitions (`CircuitPolicy`, `Transformer`, `Physics Heads`).
- `training/`: Training loop components, objectives, and configuration.
- `data/`: Dataset generation, augmentation, and specification handling.
- `simulators/`: Interfacing with Qiskit for validation.
- `configs/`: JSON configuration files for training curricula.
- `scripts/`: Entry points for training and evaluation.
- `architecture_validation/`: Scripts to verify gradients, physics consistency, and noise models.

## Installation

Ensure you have a Python environment (3.8+) with PyTorch installed.

```bash
pip install torch numpy qiskit
```

*Note: This package assumes it is part of the larger `qcgpt` repository structure.*

## Usage

### Training

The main entry point is `scripts/train.py`. You can run it with a configuration file or override parameters via CLI.

**Basic Run:**
```bash
python -m qcgpt3.scripts.train --config qcgpt3/configs/stage1_bootcamp.json
```

**Override Hyperparameters:**
```bash
python -m qcgpt3.scripts.train --config qcgpt3/configs/stage1_bootcamp.json --batch_size 64 --lr 1e-4
```

### Configuration

Training schedules are defined in JSON files in `configs/`. A config defines:
- **Curriculum**: Staged changes in circuit depth and noise.
- **Loss Weights**: Balancing Supervised, Fidelity, and Entanglement losses.
- **Optimizer**: Learning rates for Encoder/Decoder.

Example `config.json` snippet:
```json
{
  "run_name": "experiment_01",
  "num_epochs": 100,
  "stages": [
    {"name": "warmup", "start_epoch": 1, "max_depth": 4},
    {"name": "grow", "start_epoch": 20, "max_depth": 8}
  ]
}
```

## Architecture

For a deep dive into the system architecture, including the Mathematical motivations behind the Lie Algebra layers and the Modular Objective system, please refer to [ARCHITECTURE.md](ARCHITECTURE.md).
