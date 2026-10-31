# Provenance

Settings the result CSVs do not record, exported on 2026-10-31 by `scripts/export_provenance.py` from the code
that produced the results (or from the git revision named in each record), and a reproduction of the main probes.

| File | Contents |
|---|---|
| `probes.json` | Linear probe (8→10) and MLP probe (standardize, 8→256→256→10, ReLU): layer shapes, AdamW (weight decay 1e-4), cosine schedule, 60 epochs, batch 128, learning-rate grids, selection rule; captured by running the probe trainer. Git history of the probe code and the first commit of every probe CSV. |
| `models.csv` | Backbones: conv layers, parameter counts (small CNNs 94,152 and 964,360), input resolution at the first convolution (128×128 for both pretrained networks), bottleneck width. |
| `tests.csv` | Every test in `tests/`, its outcome on this machine and the tolerances it asserts (`torch.allclose` default rtol 1e-5). |
| `environment.json` | Host and library versions, recorded post hoc (run metadata has no host field). |
| `measurements.json` | Init-gradient batch size (32 in train mode), toy-model architecture and data, the committed effective-order cutoff rule (1e-12), each read from the revision that produced the CSV. |
| `repro_backbone_probe.csv`, `repro_mlp_probe.csv`, `probe_candidates.csv` | Re-run of `results/backbone_probe.csv` (90 rows, linear probe, MNIST and CIFAR-10 matched-tuning runs) and of the 27 `cifar_val` rows of `results/mlp_probe.csv`, with every candidate probe learning rate logged (648 candidates). All 117 selected rows reproduce the original CSVs exactly (same selected rate, identical test accuracy). |
