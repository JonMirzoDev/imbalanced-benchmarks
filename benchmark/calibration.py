"""Probability calibration after imbalance treatments.

Weighting and resampling change the class balance the model sees in training,
so its predicted probabilities no longer match the real positive rate. This
script measures how far off they are and whether a sigmoid (Platt)
recalibration step, fitted inside each training fold, brings them back.

Writes results/calibration.csv and results/reliability.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .datasets import load_all
from .methods import RANDOM_STATE, build_methods

FOLDS = 5
BINS = 10


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = BINS) -> float:
    """ECE with equal-count bins, so rare positives are not left in empty bins."""
    order = np.argsort(p)
    error = 0.0
    for chunk in np.array_split(order, bins):
        error += abs(p[chunk].mean() - y[chunk].mean()) * len(chunk) / len(y)
    return error


def reliability_points(y: np.ndarray, p: np.ndarray, bins: int = BINS):
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return [p[c].mean() for c in chunks], [y[c].mean() for c in chunks]


def build_candidates() -> dict:
    """Original pipelines plus a recalibrated copy of each one.

    The baselines are recalibrated too, as a control: otherwise any gain from
    "treatment + Platt" could just be Platt scaling smoothing the scores.
    """
    candidates = dict(build_methods())
    for name, pipeline in build_methods().items():
        candidates[f"{name}+platt"] = CalibratedClassifierCV(pipeline, method="sigmoid", cv=3)
    return candidates


def main() -> None:
    datasets = load_all()
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows, curves = [], {}
    for data_name, (X, y) in datasets.items():
        for method_name, model in build_candidates().items():
            p = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
            rows.append(
                {
                    "dataset": data_name,
                    "method": method_name,
                    "prevalence": y.mean(),
                    "mean_predicted": p.mean(),
                    "brier": brier_score_loss(y, p),
                    "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                    "ece": expected_calibration_error(y, p),
                    "pr_auc": average_precision_score(y, p),
                }
            )
            curves[(data_name, method_name)] = reliability_points(y, p)
            r = rows[-1]
            print(
                f"{data_name:16s} {method_name:26s} mean_p={r['mean_predicted']:.3f} "
                f"(prev {r['prevalence']:.3f})  brier={r['brier']:.4f}  ece={r['ece']:.3f}  "
                f"pr_auc={r['pr_auc']:.3f}"
            )

    out = Path("results")
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "calibration.csv", index=False)
    plot_reliability(curves, list(datasets), out / "reliability.png")


def plot_reliability(curves: dict, dataset_names: list[str], path: Path) -> None:
    shown = ["logreg_baseline", "logreg_weighted", "logreg_smote", "logreg_smote+platt"]
    fig, axes = plt.subplots(1, len(dataset_names), figsize=(4.2 * len(dataset_names), 4.2))
    for ax, data_name in zip(axes, dataset_names):
        ax.plot([0, 1], [0, 1], color="0.7", linestyle="--", linewidth=1)
        for method_name in shown:
            predicted, observed = curves[(data_name, method_name)]
            ax.plot(predicted, observed, marker="o", markersize=3, label=method_name)
        ax.set_title(data_name)
        ax.set_xlabel("mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed positive rate")
    axes[-1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
