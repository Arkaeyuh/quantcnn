# Hybrid Quantum-Classical Image Classifier

A research project comparing quantum and classical approaches to image
classification on MNIST and CIFAR-10. A shared convolutional neural network
extracts image features, which are then classified by one of three
interchangeable heads: a standard linear layer, a small neural network, or a
variational quantum circuit. Experiments are run across multiple data sizes
and random seeds to produce statistically meaningful comparisons.

A full write-up of the methodology and findings is in
[`paper/paper.pdf`](paper/paper.pdf).

## What is a variational quantum circuit?

A quantum circuit is a sequence of operations applied to quantum bits
(qubits). A *variational* circuit has tunable rotation angles as its
parameters, learned by gradient descent just like weights in a neural
network. The circuit encodes the image features as rotation angles on the
qubits, applies entangling operations that allow qubits to interact, and
reads out measurement values that are passed to a final classification layer.

This hybrid setup (classical CNN, then quantum circuit, then output) is a
leading near-term architecture in quantum machine learning research because
it can run on today's quantum simulators without requiring a
fault-tolerant quantum computer.

## Results

Accuracy is the best test-split accuracy across all training epochs, averaged
over 3 random seeds. Full per-seed numbers, standard deviations, and
generalization-gap analysis are in the paper; the tables below summarize the
headline comparisons.

**MNIST** (25 epochs)

| Training samples | Linear | MLP | Quantum (no re-upload) |
|:---:|:---:|:---:|:---:|
| 250 | 61.5% | 36.7% | **54.5%** |
| 500 | 79.3% | 53.2% | **71.2%** |
| 1,000 | 89.6% | 76.4% | **83.3%** |

**CIFAR-10** (80 epochs)

| Training samples | Linear | MLP | Quantum (no re-upload) |
|:---:|:---:|:---:|:---:|
| 1,000 | **57.5%** | 51.2% | 37.1% |
| 2,000 | **64.6%** | 60.6% | 46.8% |
| 4,000 | **71.2%** | 67.2% | 58.9% |

**Key findings:**

- On MNIST, the quantum head outperforms a classical MLP of similar parameter
  count (154 vs. 162 parameters) by 7 to 18 percentage points, and comes
  within 6 to 8 points of the simple linear head. On CIFAR-10 the ordering
  flips: the quantum head loses to both classical heads at every data size,
  though the gap narrows as more training data is added.
- The quantum head never outperforms the plain linear head on either
  dataset. Its advantage, where it exists, is only over a similarly sized
  classical network, not over the simplest possible baseline.
- Looking at trained checkpoints, the bottleneck features that feed the
  quantum circuit are far more saturated on CIFAR-10 (16.7% of values past
  the steep part of the encoding curve) than on MNIST (4.3%). This saturation
  is a plausible mechanism for the CIFAR-10 gap: it flattens gradients and
  collapses distinct feature values onto nearly identical rotation angles.
- Data re-uploading, a technique where input features are re-encoded between
  circuit layers to increase expressivity, consistently hurts performance in
  this setting on both datasets. The plain (no re-upload) circuit is the most
  accurate and, on MNIST, the most stable across seeds.
- Quantum heads show noticeably higher seed-to-seed variance than classical
  heads, especially on CIFAR-10, consistent with a harder optimization
  landscape. This variance is real but secondary: even the best-seed quantum
  run falls short of the average linear head on CIFAR-10, so encoding
  capacity, not optimization luck, is the main story.

## Architecture

```
Input image
       |
  CNN backbone               Convolutional layers -> global average pool
       |
  Bottleneck (8-dim)         Linear projection to a compact embedding
       |
  +----+-----------------------------------+
  | Linear head   MLP head   Quantum head   |
  |  (baseline)  (baseline)  (VQC + linear readout)
  +----+-----------------------------------+
       |
  Class logits
```

The quantum head encodes the 8-dimensional embedding into rotation angles on
8 qubits, applies 4 layers of learned rotations and entangling gates,
measures each qubit's expectation value, and maps the 8 outputs to class
logits via a final linear layer.

The MLP baseline is sized so its total parameter count is as close as
possible to the quantum head's, though the two architectures cannot be made
exactly equal; the linear head is smaller still and serves as a simple
reference point rather than a matched baseline.

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
  --dataset mnist \
  --head quantum \
  --reupload_mode none \
  --train_subset_size 1000 \
  --epochs 25 \
  --seed 0
```

Available options for `--dataset`: `mnist`, `cifar10`
Available options for `--head`: `linear`, `mlp`, `quantum`
Available options for `--reupload_mode`: `none`, `light`, `strong`

**Full sweep** (reproduces the results tables above)

```bash
# Quick smoke-test (~minutes)
PYTHONPATH=. python -m scripts.run_sweep --quick

# MNIST sweep, parallel workers significantly reduce wall time
PYTHONPATH=. python -m scripts.run_sweep --dataset mnist --epochs 25 --n_workers 8

# CIFAR-10 sweep
PYTHONPATH=. python -m scripts.run_sweep \
  --dataset cifar10 \
  --epochs 80 \
  --gradient_clip_norm 2.0 \
  --subsets 1000 2000 4000 \
  --n_workers 8
```

Each run is saved to `runs/logs/sweep_<timestamp>/` with a `metrics.csv` and
`best_model.pt` per configuration.

**Plot results**

```bash
PYTHONPATH=. python -m scripts.plot_results --log_dir runs/logs/<sweep_folder>
```

Produces `accuracy_vs_subset.png`, a curve of best test-split accuracy vs
training set size for each head type.

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

paper/
  paper.tex                Workshop-style write-up of methodology and results
  paper.pdf                Compiled version
  figures/                 Accuracy plots used in the paper
```

## Implementation notes

- Quantum circuits run on PennyLane's `lightning.qubit` simulator (C++
  backend) with adjoint differentiation, which is significantly faster than
  the default Python backend.
- The backbone and classical heads run on GPU via Apple MPS (Metal) or CUDA
  when available; the quantum circuit always runs on CPU, since PennyLane's
  statevector simulators require it.
- VQC weights are initialized from `uniform(-0.1, 0.1)` rather than
  PyTorch's default uninitialized memory, keeping rotation angles near zero
  where gradients are well defined. This was a significant fix: without it,
  seed-to-seed variance on the quantum head was extreme.
- Training uses AdamW with a cosine annealing learning rate schedule. The
  best checkpoint (by test-split accuracy) is saved separately from the
  final epoch weights.
- Per-component gradient norms (backbone vs. head) are logged each epoch,
  making it straightforward to detect optimization pathologies like barren
  plateaus in the quantum head.
- The current evaluation protocol selects the best checkpoint using the
  official test split, so reported numbers are best described as test-split
  accuracy rather than held-out test accuracy. The paper quantifies this
  effect directly and finds it small (under 1.3 points everywhere); a
  proper train/validation/test split is planned.

## Limitations and planned work

The paper's discussion section covers this in more depth, but briefly:

- The backbone is trained jointly with each head, so head comparisons
  include any difference in the embedding each head induces. A
  frozen-backbone experiment, training all heads on identical fixed
  embeddings, would isolate the head's contribution more cleanly.
- A classical baseline using sinusoidal (Fourier) features would help test
  whether the quantum head's behavior comes from its encoding rather than
  anything specifically quantum.
- All quantum results are noiseless simulations; real hardware would add
  noise on top of these numbers, not remove it.