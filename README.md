# Hybrid quantum-classical image classifier

Classical CNN feature extractor with a bottleneck embedding, paired with:

- **Linear** and **MLP** classification heads (baselines).
- **Variational quantum circuit (VQC)** head implemented with PennyLane (`default.qubit` + backprop differentiation).

Experiments compare **no / light / strong data re-uploading** schedules on small labeled subsets of MNIST.

## Setup

Python 3.10+ recommended.

```bash
cd quantCnn
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=.  # or prefix each command below
```

## Train

Single run (example):

```bash
PYTHONPATH=. python -m scripts.train \
  --dataset mnist \
  --head quantum \
  --reupload_mode none \
  --train_subset_size 1000 \
  --seed 0 \
  --epochs 15 \
  --log_csv runs/logs/single_seed0.csv
```

## Sweep (multi-seed, subset × head × re-upload)

Produces CSV logs suitable for plotting:

```bash
PYTHONPATH=. python -m scripts.run_sweep
PYTHONPATH=. python -m scripts.run_sweep --quick --epochs 4
```

Output defaults to `runs/logs/sweep_<timestamp>/` (printed at the end).

## Plot results

Reads all `metrics.csv` files under a directory (recursive):

```bash
PYTHONPATH=. python -m scripts.plot_results --log_dir runs/logs/<your_sweep_folder>
```

## Paper

Portfolio-style write-up: [paper/portfolio_paper.md](paper/portfolio_paper.md).

## Scope

- Claims in the paper are framed against **capacity-matched classical heads** under a fixed training budget — not general “quantum advantage.”
- Default training uses exact expectation values (noiseless simulator). Hardware noise models can extend the Discussion section.
