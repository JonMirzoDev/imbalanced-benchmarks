"""Model/treatment configurations under comparison."""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def _logreg(**kwargs):
    return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **kwargs)


def _forest(**kwargs):
    return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, **kwargs)


def build_methods() -> dict[str, Pipeline]:
    """Each pipeline scales features and applies one imbalance treatment."""
    return {
        "logreg_baseline": Pipeline(
            [("scale", StandardScaler()), ("clf", _logreg())]
        ),
        "logreg_weighted": Pipeline(
            [("scale", StandardScaler()), ("clf", _logreg(class_weight="balanced"))]
        ),
        "logreg_smote": Pipeline(
            [
                ("scale", StandardScaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("clf", _logreg()),
            ]
        ),
        "forest_baseline": Pipeline([("clf", _forest())]),
        "forest_weighted": Pipeline([("clf", _forest(class_weight="balanced"))]),
        "forest_smote": Pipeline(
            [("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", _forest())]
        ),
        "forest_undersample": Pipeline(
            [("under", RandomUnderSampler(random_state=RANDOM_STATE)), ("clf", _forest())]
        ),
    }
