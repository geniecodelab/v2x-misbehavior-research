"""Train Decision Tree / Logistic Regression / XGBoost baselines for comparison
against the existing RandomForest model, reusing the cached feature splits
produced by scripts/train_veremi_baseline.py.

Usage (run from the repo root, after train_veremi_baseline.py has been run at
least once so that data/processed/veremi_nextgen/*.pkl.gz exist):

    python scripts/train_baselines.py

Outputs:
    reports/phase2_baselines_metrics.json
    reports/phase2_baselines_evaluation.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from v2x_ids.evaluation import (
    binary_metrics,
    choose_threshold,
    detection_latency,
    node_metrics,
    per_attack_metrics,
)
from v2x_ids.features import feature_matrix

try:
    # pyrefly: ignore [missing-import]
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train baseline classifiers for comparison against the RandomForest model"
    )
    parser.add_argument("--processed-dir", default="data/processed/veremi_nextgen")
    parser.add_argument("--output-metrics", default="reports/phase2_baselines_metrics.json")
    parser.add_argument("--output-report", default="reports/phase2_baselines_evaluation.md")
    parser.add_argument("--threshold-grid-step", type=float, default=0.01)  # matches phase2.yaml
    parser.add_argument("--node-score-quantile", type=float, default=0.95)  # matches phase2.yaml
    parser.add_argument("--random-state", type=int, default=42)  # matches phase2.yaml
    return parser.parse_args()


def load_split(processed_dir: Path, split: str) -> pd.DataFrame:
    cache = processed_dir / f"{split}_features.pkl.gz"
    if not cache.is_file():
        raise FileNotFoundError(
            f"{cache} not found. Run scripts/train_veremi_baseline.py first "
            "so the cached feature splits exist."
        )
    return pd.read_pickle(cache, compression="gzip")


def anomaly_score(classifier: Any, frame: pd.DataFrame, benign_index: int) -> np.ndarray:
    """1 - P(benign), matching the RandomForest baseline's anomaly definition."""
    probabilities = classifier.predict_proba(feature_matrix(frame))
    return 1.0 - probabilities[:, benign_index]


def build_models(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
    }
    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=random_state,
        )
    else:
        print("xgboost not installed (pip install xgboost) - skipping XGBoost baseline")
    return models


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    train = load_split(processed_dir, "train")
    validation = load_split(processed_dir, "validation")
    test = load_split(processed_dir, "test")

    models = build_models(random_state=args.random_state)
    metrics: dict[str, Any] = {"methods": {}}
    train_labels = train["attack_class"].to_numpy()

    from sklearn.preprocessing import LabelEncoder

    for name, classifier in models.items():
        print(f"training {name}...")
        started = time.perf_counter()
        if name == "xgboost":
            encoder = LabelEncoder()
            encoded_labels = encoder.fit_transform(train_labels)
            benign_index = int(np.where(encoder.classes_ == "benign")[0][0])
            sample_weight = compute_sample_weight("balanced", train_labels)
            classifier.fit(feature_matrix(train), encoded_labels, sample_weight=sample_weight)
        else:
            classifier.fit(feature_matrix(train), train_labels)
            benign_index = list(classifier.classes_).index("benign")
        fit_seconds = time.perf_counter() - started

        validation_scores = anomaly_score(classifier, validation, benign_index)
        threshold = choose_threshold(
            validation["label"].to_numpy(), validation_scores, args.threshold_grid_step
        )

        test_scores = anomaly_score(classifier, test, benign_index)
        test_labels = test["label"].to_numpy()

        metrics["methods"][name] = {
            "fit_seconds": fit_seconds,
            "threshold": threshold,
            "message": binary_metrics(test_labels, test_scores, threshold),
            "node": node_metrics(test, test_scores, threshold, args.node_score_quantile),
            "per_attack": per_attack_metrics(test, test_scores, threshold),
            "detection_latency": detection_latency(test, test_scores, threshold),
        }
        message = metrics["methods"][name]["message"]
        print(
            f"{name}: precision={message['precision']:.4f} recall={message['recall']:.4f} "
            f"f1={message['f1']:.4f} fpr={message['false_positive_rate']:.4f} "
            f"pr_auc={message['pr_auc']:.4f} (fit {fit_seconds:.2f}s)"
        )

    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metrics).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    rows = []
    for name, values in metrics["methods"].items():
        m = values["message"]
        n = values["node"]
        rows.append(
            f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['false_positive_rate']:.4f} | {m['pr_auc']:.4f} | {n['f1']:.4f} | "
            f"{values['fit_seconds']:.2f}s |"
        )
    report = (
        "# Baseline model comparison (Decision Tree / Logistic Regression / XGBoost)\n\n"
        "Compare these rows against RandomForest's `methods.ml_only` entry in "
        "`reports/phase2_metrics.json` (same Test set, same message/node metric definitions).\n\n"
        "| Method | Precision | Recall | F1 | FPR | PR-AUC | Node F1 | Fit time |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + "\n"
    )
    Path(args.output_report).write_text(report, encoding="utf-8")
    print(f"wrote {args.output_metrics}")
    print(f"wrote {args.output_report}")


if __name__ == "__main__":
    main()
