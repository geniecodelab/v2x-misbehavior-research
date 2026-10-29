import _paths
import numpy as np, pandas as pd

df = pd.read_parquet(_paths.A("scored_tol5.parquet"))
b = df[(df["dataset_variant"] == "benign") & (df["split"] == "validation")]

# 1) 같은 메시지를 여러 수신자가 받았을 때 노이즈가 같은가
g = b.groupby(["sender_id", "message_id"])["position_noise_x"].agg(["size", "nunique", "std"])
multi = g[g["size"] > 1]
print(f"여러 수신자가 받은 메시지: {len(multi)} / {len(g)}")
print(f"  그중 수신자 간 노이즈가 다른 비율: {(multi['nunique'] > 1).mean():.3f}")
print(f"  수신자 간 노이즈 표준편차 중앙값: {multi['std'].median():.3f} m")

# 2) (수신자, 송신자) 시퀀스 안에서의 시간 상관
s = b.sort_values(["receiver_id", "sender_id", "timestamp"])
grp = s.groupby(["receiver_id", "sender_id"])["position_noise_x"]
ac = grp.apply(lambda v: v.autocorr() if len(v) > 5 else np.nan)
d = grp.diff()
print(f"(수신자,송신자) 시퀀스 내 lag-1 자기상관 중앙값: {ac.median():.3f}")
print(f"  노이즈 표준편차 {s['position_noise_x'].std():.3f} m, 연속 차분 표준편차 {d.std():.3f} m "
      f"(독립이면 약 {s['position_noise_x'].std() * np.sqrt(2):.3f} m)")

# 3) 송신자 단위로 중복 제거했을 때(이전 진단 방식)
u = b.drop_duplicates(["sender_id", "timestamp"]).sort_values(["sender_id", "timestamp"])
ac2 = u.groupby("sender_id")["position_noise_x"].apply(lambda v: v.autocorr() if len(v) > 5 else np.nan)
print(f"송신자 단위 중복 제거 후 자기상관 중앙값(이전 방식): {ac2.median():.3f}")
