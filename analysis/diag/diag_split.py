import _paths
import pandas as pd
df = pd.read_parquet(_paths.A("scored.parquet"))
print(df.groupby("split")[["pos_x", "pos_y"]].agg(["min", "max"]).round(0))
b = df[df["dataset_variant"] == "benign"]
print(b.groupby("split")["f5_d_off"].describe(percentiles=[.5, .9, .99]).round(2))