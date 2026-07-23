# imbalanced-benchmarks

Systematic comparison of techniques for class-imbalanced classification on real datasets, including medical ones.

When 2% of samples are positive, accuracy is a misleading metric and standard training under-serves the minority class — usually the class that matters (disease, failure, fraud). This benchmark compares the common remedies under one consistent evaluation protocol: stratified cross-validation, scored with metrics that remain informative under imbalance (PR-AUC, balanced accuracy, F1, recall).

## Methods compared

- **Baseline** — untreated logistic regression and random forest
- **Cost-sensitive weighting** — `class_weight="balanced"` (misclassification costs inversely proportional to class frequency)
- **Oversampling** — SMOTE
- **Undersampling** — random majority undersampling

## Datasets

| Dataset | Domain | Positive class |
|---|---|---|
| mammography | medical imaging follow-up | ~2.3% |
| thyroid_sick | clinical thyroid screening | ~6% |
| pima diabetes | clinical tabular | ~35% (mild-imbalance contrast) |

## Usage

```
pip install -r requirements.txt
python -m benchmark.run          # writes results/results.csv and prints a summary table
```

## Results

PR-AUC, 5-fold stratified CV (seed 42, scikit-learn 1.9). Full metrics with fold std in `results/results.csv`.

| Method | mammography | thyroid_sick | pima_diabetes |
|---|---|---|---|
| logreg_baseline | 0.618 | 0.702 | 0.723 |
| logreg_weighted | 0.547 | 0.655 | **0.725** |
| logreg_smote | 0.541 | 0.644 | 0.723 |
| forest_baseline | **0.755** | 0.952 | 0.687 |
| forest_weighted | 0.736 | **0.952** | 0.693 |
| forest_smote | 0.711 | 0.945 | 0.682 |
| forest_undersample | 0.643 | 0.865 | 0.688 |

Takeaways from this run:

- Ranked by PR-AUC alone, untreated baselines look best — but that hides the operating point. The treatments buy **recall**: on thyroid_sick, `forest_weighted` matches the baseline's PR-AUC (0.952) while lifting recall from 0.78 to 0.89; on mammography, weighting lifts logistic-regression recall from 0.42 to 0.85 at a real precision cost.
- Undersampling is the bluntest instrument: highest recall everywhere (0.89–0.95 on the two rare-positive datasets), paid for with the largest F1/PR-AUC drops.
- On mildly imbalanced pima_diabetes (35% positive), all methods land within noise of each other — imbalance treatments matter when imbalance is severe.

## Roadmap

- [ ] cost-sensitive SVM formulations
- [ ] probability calibration after resampling
- [ ] missing-value handling strategies
