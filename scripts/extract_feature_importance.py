"""Extract RandomForest feature_importances_ and save a seaborn bar chart.

Usage:
    python scripts/extract_feature_importance.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODEL_PATH = "models/veremi_nextgen_rf.joblib"
print(f"using model: {MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)
classifier = bundle.classifier
feature_names = list(bundle.feature_names)

importances = classifier.feature_importances_
df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

Path("reports").mkdir(exist_ok=True)
df.to_csv("reports/feature_importance.csv", index=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=df, x="importance", y="feature", palette="viridis")
plt.title("RandomForest Feature Importance")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=150)

print(df.to_string(index=False))
print("saved reports/feature_importance.csv, reports/feature_importance.png")