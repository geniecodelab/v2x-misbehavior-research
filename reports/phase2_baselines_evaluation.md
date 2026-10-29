# Baseline model comparison (Decision Tree / Logistic Regression / XGBoost)

Compare these rows against RandomForest's `methods.ml_only` entry in `reports/phase2_metrics.json` (same Test set, same message/node metric definitions).

| Method | Precision | Recall | F1 | FPR | PR-AUC | Node F1 | Fit time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decision_tree | 0.9495 | 0.7341 | 0.8280 | 0.0303 | 0.8896 | 0.8423 | 1.37s |
| logistic_regression | 0.8748 | 0.7333 | 0.7978 | 0.0813 | 0.8831 | 0.6801 | 35.21s |
| xgboost | 0.9584 | 0.7279 | 0.8274 | 0.0245 | 0.8976 | 0.8582 | 8.56s |
