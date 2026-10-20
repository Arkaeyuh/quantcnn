# Products of Cosines: Quantum Classifier Heads and the CNN Representations Trained Through Them

A research project on hybrid quantum–classical image classifiers. A convolutional
network (CNN) turns an image into an 8-number summary, and a small *variational quantum
circuit* (VQC) on 8 simulated qubits classifies it. Because the CNN is trained *through*
the quantum head, the head also shapes what the CNN learns. This project measures that
effect and traces it to the structure of the circuit.

The paper (ICML format) is [`paper/main.pdf`](paper/main.pdf); its source is in `paper/`.

## What is a variational quantum circuit?

A quantum circuit is a sequence of operations on quantum bits (qubits). A *variational*
circuit has tunable rotation angles that are learned by gradient descent, like neural
network weights. Here the circuit encodes the 8 CNN features as rotation angles, applies
trainable rotations and *entangling* CNOT gates that let qubits interact, and measures each
qubit. A final linear layer turns the 8 measurements into class scores. Everything runs in
an exact, noiseless simulator.

## Main findings

1. **Tune every head before comparing them.** With one shared training recipe, the quantum
   head appeared to beat a small classical network (MLP) on MNIST by up to 18 points. With
   longer training, validation-based model selection and each head's learning rate tuned,
   the largest remaining mean advantage is 3.3 points, and a plain linear head has the
   highest mean test accuracy in every setting tested.
2. **The quantum head lowers the quality of the CNN it trains.** Freezing each trained CNN
   and fitting the *same* linear probe to its 8-number summary, the default quantum head's
   CNN scores 4.7 points lower than a linear head's on CIFAR-10 (4,000 images, 5 seeds,
   every seed lower). A two-layer MLP probe finds the same gap, so the loss is not just one
   of linear readability.
3. **The CNOT gates are responsible, through one quantity.** With all trainable rotations
   at zero, each measurement is exactly a product of cosines of the inputs, over a set of
   inputs fixed by the CNOT gates. The average size of that set, the *product order*, tracks
   the cost: probe accuracy is 73.6 at order 1, 69.8 for the default circuit (order 2.5) and
   60.1 at order 5.25. Removing the CNOTs recovers 3.8 of the 4.7 points; a circuit that
   keeps its CNOTs but has product order 1 recovers 3.1. The cost persists when the rotations
   start far from zero, and higher-order heads concentrate the gradient the CNN receives into
   fewer directions.
4. **Without CNOTs, the quantum head is exactly a classical model**: a per-input
   combination of `cos` and `sin` of the encoded features, which classical heads match
   within about a point.
5. **It holds on a pretrained network.** On an ImageNet-pretrained ResNet-18 the same
   ordering holds. With the number of optimizer updates matched, the default quantum head
   costs 6–7 points with 1,000 to 10,000 training images, while the CNOT-free head stays
   within 0.5 points of a linear head. At a fixed 30 epochs the cost looks much larger
   with little data (22.3 points at 1,000 images), largely because small sets get fewer
   updates.

Every number above is regenerated from the files in `results/` by the scripts listed
below. Limitations (noiseless simulation, one circuit family, small datasets, 3–5 seeds)
are discussed in the paper.

## Architecture

```
image -> CNN backbone -> bottleneck z (8 numbers) -> head -> class scores
                                    |
                                    +--> (after training, backbone frozen) common linear probe
heads: linear | MLP (one hidden layer of 8) | VQC: R_y(pi*tanh z_j) encoding,
       4 layers of trainable R_y R_z rotations each followed by a CNOT layout, <Z> readout,
       linear layer | classical products of cosines (no circuit)
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=.
```

## Reproducing the paper

All commands run from the repository root with `PYTHONPATH=.`.

**1. Training grids.** Each grid trains its runs into `runs/protocol_v2/<grid>/` and
exports per-run metrics and metadata (`meta.json`, `final.json`, `metrics.csv`) to
`results/protocol_v2/<grid>/`. Already-finished runs are skipped.

```bash
python -m scripts.run_protocol_v2 --grid <grid> --n_workers 6 --threads_per_worker 2
```

| Grids | What they cover |
|---|---|
| `mnist_search`, `cifar_val`, `mnist_bn`, `cifar_bn` | head comparison under matched tuning (Table 1), BatchNorm control |
| `cifar_circuit`, `cifar_entangle`, `cifar_pairs`, `mnist_circuit`, `mnist_order` | circuit variants: no CNOT, last-layer ring, 1–2 layers, CNOT pairs, encodings, MNIST |
| `cifar_classical`, `cifar_cosprod`, `cifar_support` | classical heads on cosines and on circuit/designed product supports |
| `cifar_mech`, `cifar_long`, `cifar_long2`, `cifar_sg` | controls: backbone learning rate, frozen circuit, 2x training budget, gradient decoupling |
| `cifar_pretrained`, `cifar_pretrained2`, `cifar_pretrained_1k`, `cifar_pretrained_10k` (+ `_1k_lr`, `_10k_lr`, edge-step grids) | ImageNet-pretrained ResNet-18 backbone at 1k / 4k / 10k images |
| `cifar_seeds`, `night_seeds`, `pretrained2_seeds` | extra seeds (3–4) for the headline cells |
| `cifar_init`, `cifar_pretrained_zero` | rotations initialized in U(−π, π); ResNet-18 VQC with rotations frozen at zero |
| `cifar_pretrained_budget`, `scripts/lockstep_pretrained.py` | ResNet-18 at a matched number of updates (1k × 120, 10k × 12 epochs); three heads trained in lockstep from copied weights |

**2. Probes and analyses** on the trained backbones:

```bash
python -m scripts.probe_backbones --datasets cifar10 --grids <grid> --heads all --out results/<name>_probe.csv
python -m scripts.probe_backbones --probe mlp --datasets cifar10 --grids <grid> --heads all --out results/<name>_mlp_probe.csv
python -m scripts.zero_angle_check      # trained circuits vs their zero-rotation product model
python -m scripts.sign_symmetry --circuit_lrs quantum+half=0.01 quantum+half_sym=0.01 quantum+noent=0.01
python -m scripts.zero_angle_check --cases pretrained --out results/zero_angle_check_pretrained.csv
python -m scripts.grad_rank           # effective rank of the backbone's loss gradient by head
python -m scripts.lockstep_pretrained --n_workers 5
python -m scripts.probe_backbones --all_lrs --heads all --datasets cifar10 --grids cifar_pretrained_lockstep --out results/lockstep_probe.csv
```

**3. Summaries, tables, figures and the paper:**

```bash
python -m scripts.summarize_v2 > results/summary_v2.txt
python -m scripts.make_tracker          # EXPERIMENT_TRACKER.md: run counts and DONE status per grid
for s in paper/figures/scripts/gen_*.py; do python $s; done
cd paper && latexmk -pdf main.tex
```

**4. Tests** of the simulator and the analytical identities the paper relies on:

```bash
python tests/test_torch_vqc.py      # PyTorch simulator vs PennyLane, outputs and gradients
python tests/test_identities.py     # zero-rotation product identity, ring-free sinusoid,
                                    # designed-support statistics, gradient-decoupled heads
```

The earlier exploratory sweep (one shared recipe) and the frozen-embedding study are
reproduced with `scripts/run_sweep.py`, `scripts/analyze_checkpoints.py`,
`scripts/frozen_heads.py` and `scripts/make_tables.py`; their outputs are in `results/`.
The 12 backbone checkpoints the frozen-embedding study starts from are released in
`results/frozen_sources`; other checkpoints are not.

## Project structure

```
quantcnn/
  data.py                  Stratified subsets and validation splits for MNIST and CIFAR-10
  training_utils.py        Training loop, evaluation, gradient-norm logging
  models/
    backbone.py            Small CNNs and the pretrained ResNet-18 backbone
    classical_heads.py     Linear, MLP, cosine and product-of-cosines heads; CNOT supports
    quantum_head.py        VQC head (PennyLane or PyTorch simulator)
    torch_vqc.py           Batched PyTorch statevector simulator of the circuit
    hybrid_classifier.py   Backbone + head
scripts/                   Training grids, probes, analyses, summaries (see above)
results/                   Released metrics and metadata for every run in the paper
tests/                     Simulator equivalence and identity tests
paper/                     ICML-format paper; figures/ holds the figure/table generators
```

## Implementation notes

- The PyTorch statevector simulator matches PennyLane's `lightning.qubit` to 1e-6 in
  outputs and input gradients (1e-5 in rotation gradients) and is about 10x faster; the
  paper's protocol runs use it.
- The backbone runs on Apple MPS or CUDA when available; the circuit runs in float64.
- Rotation angles start in `uniform(-0.1, 0.1)`, near the identity, so training starts
  close to the zero-rotation product model.
- Learning rates are selected on a 2,000-image validation set held out from the training
  pool; test accuracy is reported once, at the selected rate and epoch.

## License

Code is released under the MIT License (see `LICENSE`). MNIST and CIFAR-10 are downloaded at
runtime and are not redistributed here.
