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

The headline comparison uses a train/validation/test split: a 2,000-image
validation set is drawn from the unused part of the training set, and the
checkpoint epoch and the learning rate are both chosen on validation. Every
head gets the same learning-rate search. Numbers are test accuracy, mean over
3 seeds. Standard deviations, the full grid, and the exploratory sweep that
preceded this are in the paper.

**MNIST** (50 epochs)

| Training samples | Linear | MLP (162 params) | Quantum (no re-upload) |
|:---:|:---:|:---:|:---:|
| 250 | **90.3%** | 78.7% | 76.6% |
| 500 | **94.3%** | 86.7% | 90.0% |
| 1,000 | **96.4%** | 93.9% | 94.6% |

**CIFAR-10** (80 epochs)

| Training samples | Linear | MLP (162 params) | Quantum (no re-upload) |
|:---:|:---:|:---:|:---:|
| 1,000 | **59.0%** | 52.3% | 43.1% |
| 2,000 | **67.6%** | 62.0% | 53.1% |
| 4,000 | **74.6%** | 70.8% | 67.7% |

**Key findings:**

- The linear head has the highest mean accuracy in every condition, and in
  every selected (tuned) configuration the quantum head never beats it.
- An earlier sweep that used one shared recipe for all heads (25 epochs, one
  learning rate, checkpoints picked on the test set) showed the quantum head
  beating the MLP on MNIST by 7 to 18 points. Most of that came from the
  recipe. Longer training with validation-based selection removes much of it
  even before tuning, and with each head's learning rate tuned what remains
  is a mean difference of at most about 3 points. On CIFAR-10, tuning the learning rate alone
  shrinks the quantum head's deficit to the MLP from 10–14 points to 3–9.
- Training every head on the *same* frozen embedding separates the head from
  the backbone it induces. On an embedding from a quantum-trained backbone,
  a linear head still beats the quantum head by 1.6–1.7 points, for every
  seed. Most of the original CIFAR-10 gap came from the worse backbone that
  joint training through the quantum circuit produces.
- On embeddings with large values, the quantum head's `π·tanh(x)` angle
  encoding saturates and the head collapses. A purely classical head using
  the same encoding (cos/sin features) largely collapses too. Standardizing
  the input recovers most of the loss and a learned rescaling a little more;
  tripling training time recovers little (0.5 points). On embeddings already trained
  through the circuit, standardizing hurts. What matters is the match
  between the embedding's distribution and the encoding.
- Data re-uploading reduced accuracy in the shared-recipe sweep. It was not
  re-tested with tuning, so treat that result as provisional.

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

**Full sweep** (the exploratory shared-learning-rate sweep in the paper)

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

**Paper experiments** (validation protocol, frozen embeddings, tables)

```bash
# Train/validation/test re-runs with a matched learning-rate search.
# Checkpoints go to runs/protocol_v2/<grid>; logs are then copied to results/protocol_v2/<grid>.
PYTHONPATH=. python -m scripts.run_protocol_v2 --grid mnist_search --n_workers 4
PYTHONPATH=. python -m scripts.run_protocol_v2 --grid cifar_val --n_workers 4

# Re-evaluate saved checkpoints and train heads on frozen embeddings
PYTHONPATH=. python -m scripts.analyze_checkpoints runs/logs/sweep_20260627_095118 --out results/checkpoints_mnist.csv
PYTHONPATH=. python -m scripts.analyze_checkpoints runs/logs/sweep_20260628_223253 --out results/checkpoints_cifar10.csv
# Heads on frozen embeddings of the 12 released source checkpoints (results/frozen_sources),
# full learning-rate grid; rows already in results/frozen_heads.csv are skipped
PYTHONPATH=. python -m scripts.frozen_heads

# Regenerate the paper's tables, summaries and figures from results/
PYTHONPATH=. python -m scripts.make_tables results/raw/sweep_20260627_095118 results/raw/sweep_20260628_223253
PYTHONPATH=. python -m scripts.summarize_v2
PYTHONPATH=. python -m scripts.make_figures
```

`results/` holds the per-epoch metrics and metadata for every run reported
in the paper, plus the 12 backbone checkpoints the frozen-embedding experiment
starts from (`results/frozen_sources`, 24 MB). Other checkpoints are not
released.

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
    quantum_head.py        Variational quantum circuit head (PennyLane or PyTorch simulator)
    torch_vqc.py           Batched PyTorch statevector simulator of the circuit
    hybrid_classifier.py   Combines backbone + any head into one model

scripts/
  train.py                 Single-run training script with CLI
  run_sweep.py             Grid sweep with optional parallel execution
  plot_results.py          Aggregates sweep logs and generates plots
  run_protocol_v2.py       Train/validation/test re-runs with a matched lr search
  analyze_checkpoints.py   Clean-image checkpoint re-evaluation and saturation
  frozen_heads.py          Heads trained on identical frozen embeddings
  make_tables.py           Exploratory-sweep tables from raw logs
  summarize_v2.py          Protocol, frozen-head and checkpoint summaries
  make_figures.py          Paper figures

results/                   Released metrics and metadata for all paper runs
tests/                     PyTorch simulator vs. PennyLane equivalence test

paper/
  paper.tex                Workshop-style write-up of methodology and results
  paper.pdf                Compiled version
  figures/                 Accuracy plots used in the paper
```

## Implementation notes

- Quantum circuits can run on PennyLane's `lightning.qubit` simulator with
  adjoint differentiation (`--quantum_sim pennylane`, the default), or on a
  batched PyTorch statevector simulator of the same circuit
  (`--quantum_sim torch`), which is about 10x faster.
  `tests/test_torch_vqc.py` checks it against PennyLane to 1e-6 in outputs
  and gradients.
- The backbone and classical heads run on Apple MPS or CUDA when available;
  the quantum circuit always runs on CPU in float64.
- VQC rotation angles are initialized from `uniform(-0.1, 0.1)`, which
  starts every rotation gate near the identity, instead of PennyLane's default
  `uniform(0, 2π)`. In preliminary runs the default gave much larger
  seed-to-seed variance.
- The head stores 154 parameters, but only 146 affect the output. The last
  layer's RZ rotations only change phases; the CNOT ring that follows permutes
  basis states and the Z measurements read only amplitude magnitudes, so those
  phases never reach the outputs.
- With `--val_size N`, `scripts/train.py` holds out a stratified validation
  set, selects the checkpoint on it, and writes `final.json` with clean
  train / validation / test accuracy of the selected checkpoint. Without it
  (the legacy default), the checkpoint is selected on the test split.

## Limitations

- All quantum results are noiseless simulations; real hardware would add
  noise on top of these numbers, not remove it.
- One circuit (8 qubits, 4 layers), one bottleneck size, two datasets, three
  seeds per setting. Only the learning rate was tuned.
- Data re-uploading was evaluated only with a shared learning rate.

