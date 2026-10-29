import _paths
import numpy as np, pandas as pd, shapely
import f5_map_check as f

m = f.load_map(_paths.A("map.pkl"))
T = f.load_transform(_paths.A("transform.json"))
df = pd.read_parquet(_paths.A("scored.parquet"))
b = df[(df["dataset_variant"] == "benign") & (df["split"] == "validation") & (df["speed"] > 3)]
b = b.sample(min(20000, len(b)), random_state=0)
px, py = b["pos_x"].to_numpy(float), b["pos_y"].to_numpy(float)
hm = f.heading_math_from(b, -1, 1, T)
pts = shapely.points(px, py)

# 10 m 이내 차로 중 진행 방향이 30도 이내로 같은 차로
pi, li = m["tree"].query(pts, predicate="dwithin", distance=10)
lh = f.lane_heading_at(m, li, px[pi], py[pi])
same = np.abs(f.wrap_angle(hm[pi] - lh)) < np.radians(30)
pi, li, lh = pi[same], li[same], lh[same]
d = shapely.distance(pts[pi], m["lines"][li])
order = np.lexsort((d, pi))
sel = order[np.unique(pi[order], return_index=True)[1]]   # 점마다 가장 가까운 같은 방향 차로
P, L, H = pi[sel], m["lines"][li[sel]], lh[sel]
proj = shapely.line_interpolate_point(L, shapely.line_locate_point(L, pts[P]))
dx, dy = px[P] - shapely.get_x(proj), py[P] - shapely.get_y(proj)
lat = -dx * np.sin(H) + dy * np.cos(H)                    # +: 진행 방향 왼쪽

print(f"같은 방향 차로가 10 m 내에 있는 비율: {len(P) / len(b):.3f}")
print("같은 방향 차로 중심선 대비 횡방향 편차(m, +=왼쪽):")
print(pd.Series(lat).describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(2))
h, e = np.histogram(lat, bins=np.arange(-6, 6.5, 1))
for c, lo in zip(h, e[:-1]):
    print(f"  [{lo:+.0f},{lo + 1:+.0f}) {c:6d} " + "#" * int(60 * c / h.max()))
print("\n위치 노이즈 필드:")
print(b[["position_noise_x", "position_noise_y", "position_noise_magnitude"]].describe().round(3))
