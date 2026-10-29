import _paths
import pandas as pd

parts = [pd.read_parquet(_paths.P(f"{s}_features.parquet")) for s in ("train", "validation", "test")]
df = pd.concat(parts, ignore_index=True)
df.to_parquet(_paths.PROCESSED / "all_features.parquet", index=False)
print(df.shape)
print(df["split"].value_counts())
print(df["dataset_variant"].value_counts())
print(df["label"].value_counts())