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

Run pending — table updated from `results/results.csv` as runs complete.

## Roadmap

- [ ] cost-sensitive SVM formulations
- [ ] probability calibration after resampling
- [ ] missing-value handling strategies
