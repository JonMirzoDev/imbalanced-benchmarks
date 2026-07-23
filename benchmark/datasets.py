"""Benchmark datasets with real class imbalance."""

from __future__ import annotations

import numpy as np
from imblearn.datasets import fetch_datasets
from sklearn.datasets import fetch_openml


def load_all() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {name: (X, y)} with y in {0, 1}, 1 = minority/positive class."""
    zenodo = fetch_datasets(filter_data=("mammography", "thyroid_sick"))
    datasets = {
        name: (bunch.data, (bunch.target == 1).astype(int))
        for name, bunch in zenodo.items()
    }

    pima = fetch_openml("diabetes", version=1, as_frame=False, parser="auto")
    datasets["pima_diabetes"] = (
        pima.data.astype(float),
        (pima.target == "tested_positive").astype(int),
    )
    return datasets


def describe(datasets: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    for name, (X, y) in datasets.items():
        positive = y.mean()
        print(f"{name:16s}  n={len(y):5d}  features={X.shape[1]:3d}  positive={positive:.1%}")
