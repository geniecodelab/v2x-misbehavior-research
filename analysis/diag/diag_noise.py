import _paths
import numpy as np, pandas as pd
import f5_map_check as f

m = f.load_map(_paths.A("map.pkl"))
df = pd.read_parquet(_paths.A("scored_tol25.parquet"))
b = df[(df["dataset_variant"] == "benign") & (df["split"] == "validation")]
b = b.sample(min(20000, len(b)), random_state=0)
x, y = b["pos_x"].to_numpy(float), b["pos_y"].to_numpy(float)
nx, ny = b["position_noise_x"].to_numpy(float), b["position_noise_y"].to_numpy(float)
print("보고 위치        :", f.fit_quality(m, x, y))
print("보고 위치 - noise:", f.fit_quality(m, x - nx, y - ny))
print("보고 위치 + noise:", f.fit_quality(m, x + nx, y + ny))
s = df[(df["dataset_variant"] == "benign") & (df["split"] == "validation")]
s = s.drop_duplicates(["sender_id", "timestamp"]).sort_values(["sender_id", "timestamp"])
ac = s.groupby("sender_id")["position_noise_x"].apply(lambda v: v.autocorr() if len(v) > 3 else np.nan)
print("송신자 내 position_noise_x lag-1 자기상관 중앙값:", round(float(ac.median()), 3))
