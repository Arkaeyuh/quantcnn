"""Dataset loaders and stratified subset construction."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def stratified_indices(
    labels: torch.Tensor | np.ndarray,
    n_take: int,
    seed: int,
) -> list[int]:
    """Return stratified subset indices preserving class proportions."""
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    rng = np.random.default_rng(seed)
    lbl = np.asarray(labels)
    classes = sorted(np.unique(lbl).tolist())
    pools = [np.where(lbl == c)[0] for c in classes]
    k = len(classes)
    proportions = []
    counts = [(lbl == c).sum() for c in classes]
    total = sum(counts)
    for c in counts:
        proportions.append(c / total)
    # sample per class proportional to prevalence, rounding with largest remainder fix
    target = []
    frac = []
    assigned = 0
    for p in proportions:
        raw = n_take * p
        target.append(int(np.floor(raw)))
        frac.append(raw - target[-1])
        assigned += target[-1]
    remainder = n_take - assigned
    if remainder > 0:
        order = np.argsort(-np.array(frac))
        for j in range(remainder):
            target[int(order[j % len(target)])] += 1
    indices: list[int] = []
    for c_idx, t in enumerate(target):
        pool = pools[c_idx]
        if t <= 0:
            continue
        if t >= len(pool):
            chosen = pool
        else:
            chosen = rng.choice(pool, size=t, replace=False)
        indices.extend(chosen.astype(int).tolist())
    rng.shuffle(indices)
    return indices


def get_mnist(
    subset_size: int | None,
    seed: int,
    data_root: str = "./data",
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    val_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
    if subset_size is None or subset_size >= len(train_ds):
        return train_ds, val_ds

    lbls = getattr(train_ds, "targets", None)
    if lbls is None:
        lbls = np.array([train_ds[i][1] for i in range(min(60000, len(train_ds)))])
    ids = stratified_indices(lbls, subset_size, seed)
    train_sub = Subset(train_ds, ids)
    return train_sub, val_ds


def get_cifar10(
    subset_size: int | None,
    seed: int,
    data_root: str = "./data",
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    tfm_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    tfm_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm_train)
    val_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm_test)

    if subset_size is None or subset_size >= len(train_ds):
        return train_ds, val_ds

    lbls = np.array(train_ds.targets)
    ids = stratified_indices(lbls, subset_size, seed)
    train_sub = Subset(train_ds, ids)
    return train_sub, val_ds


def dataset_num_classes(ds: torch.utils.data.Dataset) -> int:
    base = getattr(ds, "dataset", ds)  # unwrap Subset
    if hasattr(base, "classes"):
        return len(base.classes)
    if hasattr(base, "targets"):
        targets = base.targets
        if isinstance(targets, torch.Tensor):
            return int(targets.unique().numel())
        return len(set(targets))
    raise ValueError(f"Cannot infer num_classes from {type(base).__name__}.")


def get_splits(
    dataset: str,
    subset_size: int,
    seed: int,
    val_size: int,
    data_root: str = "./data",
) -> dict[str, torch.utils.data.Dataset]:
    """Train subset plus a held-out validation split drawn from the *training* pool.

    Returns ``train`` (augmented for CIFAR-10), ``train_clean`` (same images, test
    preprocessing), ``val`` (``val_size`` stratified training images disjoint from
    the subset, test preprocessing) and ``test`` (official test split). The train
    subset uses the same ``stratified_indices(labels, subset_size, seed)`` call as
    ``get_mnist``/``get_cifar10``, so it is identical to the subset of the original
    sweeps for a given seed.
    """
    loader = get_mnist if dataset == "mnist" else get_cifar10
    train_full, test_ds = loader(None, seed, data_root=data_root)
    clean_tfm = test_ds.transform
    cls = type(train_full)
    clean_full = cls(root=data_root, train=True, download=False, transform=clean_tfm)

    labels = np.asarray(train_full.targets)
    train_ids = stratified_indices(labels, subset_size, seed)
    rest = np.setdiff1d(np.arange(len(labels)), np.asarray(train_ids))
    # Seed offset keeps the validation draw independent of the subset draw.
    val_local = stratified_indices(labels[rest], val_size, seed + 10_000)
    val_ids = rest[np.asarray(val_local)].tolist()
    return {
        "train": Subset(train_full, train_ids),
        "train_clean": Subset(clean_full, train_ids),
        "val": Subset(clean_full, val_ids),
        "test": test_ds,
    }
