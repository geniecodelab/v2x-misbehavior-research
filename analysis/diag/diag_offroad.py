import _paths
import pandas as pd
df = pd.read_parquet(_paths.A("scored.parquet"))
b = df[df["dataset_variant"] == "benign"]
d = b["f5_d_off"]
print("benign n =", len(b))
print("도로 밖 비율:", (d > 0).mean().round(3))
print("d_off 분위수(m):", d.quantile([0.5, 0.9, 0.95, 0.99, 0.995, 0.999]).round(2).to_dict())
print("도로 밖 중 1m 이내:", ((d > 0) & (d <= 1)).sum() / max((d > 0).sum(), 1))
print("도로 밖 중 3m 초과:", (d > 3).sum() / max((d > 0).sum(), 1))
print("노이즈 필드 요약:\n", b[["position_noise_x", "position_noise_y", "position_noise_magnitude"]].describe().round(3))
far = b[d > 3]
print("3m 초과 이탈 메시지를 가진 송신자 수:", far["sender_id"].nunique(), "/", b["sender_id"].nunique())
print(far.groupby("sender_id").size().sort_values(ascending=False).head(10))