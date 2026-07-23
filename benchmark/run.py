"""Run the full benchmark: datasets x methods under stratified cross-validation.

Writes results/results.csv and prints a summary sorted by PR-AUC per dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from .datasets import describe, load_all
from .methods import RANDOM_STATE, build_methods

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
    "recall": "recall",
}
FOLDS = 5


def main() -> None:
    datasets = load_all()
    describe(datasets)
    methods = build_methods()
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for data_name, (X, y) in datasets.items():
        for method_name, pipeline in methods.items():
            scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
            row = {"dataset": data_name, "method": method_name}
            for key in SCORING:
                row[key] = scores[f"test_{key}"].mean()
                row[f"{key}_std"] = scores[f"test_{key}"].std()
            rows.append(row)
            print(
                f"{data_name:16s} {method_name:20s} "
                f"pr_auc={row['pr_auc']:.3f}  f1={row['f1']:.3f}  recall={row['recall']:.3f}"
            )

    results = pd.DataFrame(rows).sort_values(["dataset", "pr_auc"], ascending=[True, False])
    out = Path("results")
    out.mkdir(exist_ok=True)
    results.to_csv(out / "results.csv", index=False)

    print("\n=== best method per dataset (by PR-AUC) ===")
    print(results.groupby("dataset").head(1)[["dataset", "method", "pr_auc", "f1", "recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
