import _paths
import pandas as pd, numpy as np
df = pd.read_parquet(_paths.A("scored.parquet"))
b = df[(df["dataset_variant"] == "benign") & (df["split"] == "validation")].copy()
b["herr_deg"] = np.degrees(b["f5_heading_err"])
print(b["herr_deg"].describe(percentiles=[.5, .9, .99, .995, .999]).round(1))
bad = b[b["herr_deg"] > 90]
print("\n90도 초과:", len(bad), "행 /", bad["sender_id"].nunique(), "송신자")
print("그중 도로 폴리곤 내부(d_off=0) 비율:", (bad["f5_d_off"] == 0).mean().round(3))
print("\n송신자별 건수:\n", bad.groupby("sender_id").size().sort_values(ascending=False).head(10))
print("\n위치 분포:\n", bad[["pos_x", "pos_y"]].describe().round(0))
print(bad[["sender_id", "timestamp", "pos_x", "pos_y", "speed", "f5_d_off", "herr_deg"]].head(15).to_string())