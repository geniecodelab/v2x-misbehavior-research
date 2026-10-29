#!/usr/bin/env python3
"""
꺾은선(SUMO) vs 평활 곡선 도로 표현 비교 (분석 전용).

질문 1. benign 헤딩은 꺾은선 접선과 평활 접선 중 어느 쪽에 더 가까운가?
        -> 판정 기준으로 어떤 도로 표현을 써야 하는가
질문 2. SUMO 차량의 헤딩은 곡선에서 계단식으로 바뀌는가(꺾은선 운동)?
        -> 평활 궤적(클로소이드 등)으로 만든 공격을 SUMO benign과 비교하면 안 되는 근거

참위치(pos - position_noise)와 참헤딩 후보(heading - heading_noise)는 오라클 정보이며
여기서는 분석에만 사용한다.

사용 예
  python spline_compare.py --map map.pkl --scored scored_tol5.parquet --transform transform.json --out-dir spline_cmp
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

import f5_map_check as f
import road_geom as rg

C = f.COL


def q(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0)
    return dict(n=len(v), median=np.median(v), p90=np.quantile(v, .9), p99=np.quantile(v, .99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--transform")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--sigmas", default="0,2,5,10,20", help="평활 표준편차(m), 0 = 꺾은선")
    ap.add_argument("--curve-kappa", type=float, default=1 / 300, help="곡선 구간 판정 |kappa| 하한(1/m)")
    ap.add_argument("--max-rows", type=int, default=30000)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    m = f.load_map(a.map)
    T = f.load_transform(a.transform)
    meta = json.load(open(Path(a.scored).with_suffix(".meta.json")))
    s_h, k_h = meta["heading_s"], meta["heading_k"]

    df = f.read_table(a.scored)
    b = df[(df[C["variant"]] == f.BENIGN_VARIANT) & (df[C["split"]] == a.split)]
    b = b.drop_duplicates([C["sender"], C["t"]]).sort_values([C["sender"], C["t"]]).reset_index(drop=True)
    has_noise = {"position_noise_x", "position_noise_y"} <= set(b.columns)
    tx = b[C["x"]] - (b["position_noise_x"] if has_noise else 0)
    ty = b[C["y"]] - (b["position_noise_y"] if has_noise else 0)
    px, py = f.apply_transform(tx, ty, T)
    hm = f.heading_math_from(b, s_h, k_h, T)

    # 참헤딩 후보: heading_noise가 실제 노이즈 값이라면 빼면 참헤딩 (단위 미상 -> rad/deg 둘 다 시험)
    cands = {"reported": hm}
    if "heading_noise" in b.columns:
        n = b["heading_noise"].to_numpy(float)
        sgn = -1 if T["flip_y"] else 1
        cands["minus_noise_rad"] = f.wrap_angle(hm - sgn * s_h * n)
        cands["minus_noise_deg"] = f.wrap_angle(hm - sgn * s_h * np.radians(n))

    moving = b[C["speed"]].abs().to_numpy() > 3
    li = rg.nearest_same_direction_lane(m, px, py, hm)
    ok = moving & (li >= 0)
    f.log(f"{a.split} benign 고유 메시지 {len(b)}개 중 분석 대상 {ok.sum()}개 (참위치 보정={has_noise})")

    sub = np.flatnonzero(ok)
    if len(sub) > a.max_rows:
        sub = np.sort(np.random.default_rng(0).choice(sub, a.max_rows, replace=False))

    rows, smooth = [], {}
    for sig in [float(v) for v in a.sigmas.split(",")]:
        sm = rg.build_smooth(m, sig)
        smooth[sig] = sm
        th, _ = rg.lookup(m, sm, li[sub], px[sub], py[sub])
        for name, h in cands.items():
            e = np.degrees(np.abs(f.wrap_angle(h[sub] - th)))
            rows.append(dict(sigma_m=sig, heading=name, **q(e)))
        f.log(f"sigma={sig:g} m 완료")
    res = pd.DataFrame(rows)
    res.to_csv(out / "heading_error_by_sigma.csv", index=False)
    f.log("헤딩 오차(deg) - 도로 표현(sigma)별, 헤딩 후보별:\n" + res.round(3).to_string(index=False))

    # 가장 잘 맞는 참헤딩 후보 선택 (sigma=0 기준 median 최소)
    r0 = res[res["sigma_m"] == res["sigma_m"].min()].set_index("heading")["median"]
    best = r0.idxmin()
    f.log(f"참헤딩 후보 중 꺾은선 기준 median 최소: {best} ({r0[best]:.3f} deg)")

    # 질문 2: 곡선 구간에서 스텝당 헤딩 변화가 계단식인가
    sig_ref = 5.0 if 5.0 in smooth else max(smooth)
    _, kap = rg.lookup(m, smooth[sig_ref], np.where(li >= 0, li, 0), px, py)
    kap = np.where(li >= 0, kap, np.nan)
    h = cands[best]
    same = (b[C["sender"]] == b[C["sender"]].shift()).to_numpy()
    dt = (b[C["t"]] - b[C["t"]].shift()).to_numpy()
    dh = np.degrees(f.wrap_angle(h - np.roll(h, 1)))
    v_prev = b[C["speed"]].shift().to_numpy()
    pred = np.degrees(v_prev * kap * dt)
    valid = same & moving & np.isfinite(kap) & (np.abs(dt - 1) < 0.2)
    curve = valid & (np.abs(kap) > a.curve_kappa)
    straight = valid & (np.abs(kap) < a.curve_kappa / 10)
    stats = pd.DataFrame([
        dict(segment="curve", n=curve.sum(),
             frac_abs_dh_lt_0p2deg=np.mean(np.abs(dh[curve]) < 0.2),
             median_abs_dh=np.median(np.abs(dh[curve])),
             median_abs_pred=np.median(np.abs(pred[curve])),
             corr_dh_pred=np.corrcoef(dh[curve], pred[curve])[0, 1] if curve.sum() > 10 else np.nan),
        dict(segment="straight", n=straight.sum(),
             frac_abs_dh_lt_0p2deg=np.mean(np.abs(dh[straight]) < 0.2),
             median_abs_dh=np.median(np.abs(dh[straight])),
             median_abs_pred=np.median(np.abs(pred[straight])), corr_dh_pred=np.nan),
    ])
    stats.to_csv(out / "stepwise_heading_check.csv", index=False)
    f.log(f"스텝당 헤딩 변화 (헤딩 후보={best}, 곡률 sigma={sig_ref:g} m):\n" + stats.round(3).to_string(index=False)
          + "\n  해석: 곡선 구간에서 |dh|<0.2도 비율이 높고 예측과 상관이 약하면 꺾은선(계단식) 운동, "
            "비율이 낮고 상관이 높으면 연속 곡선 운동")
    f.log(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
