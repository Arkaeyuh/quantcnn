# Hybrid Quantum-Classical Image Classifier

A research project comparing quantum and classical approaches to image classification on MNIST. A shared convolutional neural network extracts image features, which are then classified by one of three interchangeable heads — a standard linear layer, a small neural network, or a variational quantum circuit. Experiments are run across multiple data sizes and random seeds to produce statistically meaningful comparisons.

## What is a variational quantum circuit?

A quantum circuit is a sequence of operations applied to quantum bits (qubits). A *variational* circuit has tunable rotation angles as its parameters — these are learned by gradient descent just like weights in a neural network. The circuit encodes the image features as rotation angles on the qubits, applies entangling operations that allow qubits to interact, and reads out measurement values that are passed to a final classification layer.

This hybrid setup (classical CNN → quantum circuit → output) is a leading near-term architecture in quantum machine learning research because it can run on today's quantum simulators without requiring a fault-tolerant quantum computer.

## Results

All experiments use MNIST (handwritten digits, 10 classes). Accuracy reported as the best validation accuracy across 25 epochs, averaged over 3 random seeds.

| Training samples | Linear | MLP | Quantum (no re-upload) |
|:---:|:---:|:---:|:---:|
| 250 | 61.5% | 36.7% | **54.5%** |
| 500 | 79.3% | 53.2% | **71.2%** |
| 1,000 | 89.6% | 76.4% | **83.3%** |

**Key findings:**

- The quantum head substantially outperforms a classical MLP matched to the same parameter count (~154 parameters each), by 15–20 percentage points across all data sizes.
- It closes to within 6–8% of the linear baseline, a gap that stays roughly constant as training data grows.
- Data re-uploading — a technique where input features are re-encoded between circuit layers to increase expressivity — consistently hurts performance in this setting. The plain (no re-upload) circuit is the most accurate and the most stable across seeds.
- At 1,000 samples, the quantum head achieves 83.1–83.6% across all three seeds (std 0.2%), demonstrating that a well-initialised VQC can be highly reproducible.

## Architecture

```
Input image (28×28)
       │
  SmallCNN backbone          3 convolutional layers → global average pool
       │
  Bottleneck (8-dim)         Linear projection to a compact embedding
       │
  ┌────┴─────────────────────────────────┐
  │ Linear head   MLP head   Quantum head │
  │  (baseline)  (baseline)  (VQC + linear readout)
  └────┬─────────────────────────────────┘
       │
  Class logits (10)
```

The quantum head encodes the 8-dimensional embedding into rotation angles on 8 qubits, applies 4 layers of learned rotations and entangling gates, measures each qubit's expectation value, and maps the 8 outputs to class logits via a final linear layer.

The MLP baseline is sized to match the quantum head's total parameter count so the comparison is fair.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=.
```

## Usage

**Single training run**

```bash
PYTHONPATH=. python -m scripts.train \
  --head quantum \
  --reupload_mode none \
  --train_subset_size 1000 \
  --epochs 25 \
  --seed 0
```

Available options for `--head`: `linear`, `mlp`, `quantum`  
Available options for `--reupload_mode`: `none`, `light`, `strong`

**Full sweep** (reproduces the results table above)

```bash
# Quick smoke-test (~minutes)
PYTHONPATH=. python -m scripts.run_sweep --quick

# Full sweep — parallel workers significantly reduce wall time
PYTHONPATH=. python -m scripts.run_sweep --epochs 25 --n_workers 4
```

Each run is saved to `runs/logs/sweep_<timestamp>/` with a `metrics.csv` and `best_model.pt` per configuration.

**Plot results**

```bash
PYTHONPATH=. python -m scripts.plot_results --log_dir runs/logs/<sweep_folder>
```

Produces `accuracy_vs_subset.png` — a curve of best validation accuracy vs training set size for each head type.

## Project structure

```
quantcnn/
  data.py                  Stratified subset sampling for MNIST and CIFAR-10
  training_utils.py        Training loop, evaluation, gradient norm logging, CSV output
  models/
    backbone.py            CNN feature extractor (SmallCNN for MNIST, CifarCNN for CIFAR-10)
    classical_heads.py     Linear and MLP classifier heads
    quantum_head.py        Variational quantum circuit head (PennyLane)
    hybrid_classifier.py   Combines backbone + any head into one model

scripts/
  train.py                 Single-run training script with CLI
  run_sweep.py             Grid sweep with optional parallel execution
  plot_results.py          Aggregates sweep logs and generates plots
```

## Implementation notes

- Quantum circuits run on PennyLane's `lightning.qubit` simulator (C++ backend) with adjoint differentiation, which is significantly faster than the default Python backend.
- The backbone and classical heads run on GPU via Apple MPS (Metal) or CUDA when available; the quantum circuit always runs on CPU, as PennyLane's statevector simulators require it.
- VQC weights are initialised from `uniform(-0.1, 0.1)` rather than PyTorch's default uninitialised memory, keeping rotation angles near zero where gradients are well-defined.
- Training uses AdamW with a cosine annealing learning rate schedule. The best checkpoint (by validation accuracy) is saved separately from the final epoch weights.
- Per-component gradient norms (backbone vs. head) are logged each epoch, making it straightforward to detect optimisation pathologies like barren plateaus in the quantum head.
