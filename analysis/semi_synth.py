#!/usr/bin/env python3
"""
준합성 실험: test benign 궤적에 크기·방향을 통제한 상수 오프셋 b를 주입하고
f5_map_check.py evaluate와 같은 임계값·판정 로직으로 탐지율을 측정한다.

- 방향 기준: 송신자의 첫 이동 메시지의 진행 방향 h0.
  angle 0 = 앞(종방향), 90 = 왼쪽(횡방향), 180 = 뒤(종방향), 270 = 오른쪽(횡방향)
- b는 전역 좌표에서 상수(= Constant Position Offset 정의). 곡선 구간에서는
  국소 종/횡 성분이 달라지므로 메시지별 국소 성분(b_par, b_perp)도 기록한다.
- mag 0은 대조군: 탐지율이 benign FPR과 같아야 한다(따름정리 1의 지도판).

사용 예
  python semi_synth.py --map map.pkl --scored scored_tol5.parquet --transform transform.json \
      --thresholds results_v5/thresholds.json --out-dir semi_v1
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, shapely
import f5_map_check as f

C = f.COL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--scored", required=True, help="score 단계 출력(헤딩 규약·반경 메타 포함)")
    ap.add_argument("--transform")
    ap.add_argument("--thresholds", required=True, help="evaluate가 저장한 thresholds.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mags", default="0,2,5,10,15,20,30,40,60,80")
    ap.add_argument("--angles", default="0,45,90,135,180,225,270,315")
    ap.add_argument("--max-senders", type=int, default=0, help="0이면 전체")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    m = f.load_map(a.map)
    T = f.load_transform(a.transform)
    th = json.load(open(a.thresholds))
    meta = json.load(open(Path(a.scored).with_suffix(".meta.json")))
    eps, theta = th["eps_m"], np.radians(th["theta_deg"])
    window, onroad_only = th["window"], th.get("head_onroad_only", False)
    s_h, k_h, tol = meta["heading_s"], meta["heading_k"], meta.get("head_tol_m", 1.0)
    f.log(f"eps={eps:.2f} m, theta={th['theta_deg']:.1f} deg, window={window}, "
          f"head_onroad_only={onroad_only}, head_tol_m={tol}, heading s={s_h},k={k_h}")

    df = f.read_table(a.scored)
    ben = df[(df[C["variant"]] == f.BENIGN_VARIANT) & (df[C["split"]] == a.split)].copy()
    if a.max_senders:
        rng = np.random.default_rng(a.seed)
        keep = rng.choice(ben[C["sender"]].unique(),
                          min(a.max_senders, ben[C["sender"]].nunique()), replace=False)
        ben = ben[ben[C["sender"]].isin(keep)]
    ben["h_math"] = f.heading_math_from(ben, s_h, k_h, T)
    ben = ben.sort_values([C["sender"], C["t"]])
    moving = ben[ben[C["speed"]].abs() > 3]
    h0 = moving.groupby(C["sender"])["h_math"].first()
    ben = ben[ben[C["sender"]].isin(h0.index)]
    ben["h0"] = ben[C["sender"]].map(h0)
    px0, py0 = f.apply_transform(ben[C["x"]], ben[C["y"]], T)
    f.log(f"{a.split} benign {len(ben)}행, 송신자 {ben[C['sender']].nunique()}명")

    mags = [float(v) for v in a.mags.split(",")]
    angs = [float(v) for v in a.angles.split(",")]
    parts = []
    t0 = time.time()
    for mag in mags:
        for ang in (angs if mag > 0 else [0.0]):
            g = ben[[C["receiver"], C["sender"], C["t"], C["speed"], "h_math", "h0"]].copy()
            dirn = g["h0"].to_numpy() + np.radians(ang)
            bx, by = mag * np.cos(dirn), mag * np.sin(dirn)
            px, py = px0 + bx, py0 + by
            g["f5_d_off"] = shapely.distance(m["area"], shapely.points(px, py))
            herr = f.heading_error(m, px, py, g["h_math"].to_numpy(), tol_m=tol)
            herr[g[C["speed"]].abs().to_numpy() < f.MIN_SPEED_FOR_HEADING] = np.nan
            g["f5_heading_err"] = herr
            h = g["h_math"].to_numpy()
            g["b_par"] = bx * np.cos(h) + by * np.sin(h)
            g["b_perp"] = -bx * np.sin(h) + by * np.cos(h)
            g["mag"], g["angle"] = mag, ang
            parts.append(g)
            f.log(f"mag={mag:g} angle={ang:g} 완료 ({time.time() - t0:.0f}s)")
    res = pd.concat(parts, ignore_index=True)

    key = [C["receiver"], C["sender"], "mag", "angle"]
    res = res.sort_values(key + [C["t"]]).reset_index(drop=True)
    for c in ("f5_d_off", "f5_heading_err"):
        res[c + "_w"] = (res[c] if window <= 1 else
                         res.groupby(key, sort=False)[c].transform(
                             lambda s: s.rolling(window, min_periods=1).median()))
    res["flag_off"] = res["f5_d_off_w"] > eps
    gate = (res["f5_d_off"] == 0) if onroad_only else (res["f5_d_off_w"] <= eps)
    res["flag_head"] = gate & (res["f5_heading_err_w"] > theta)
    res["flag"] = res["flag_off"] | res["flag_head"]

    by_cfg = res.groupby(["mag", "angle"]).agg(
        n=("flag", "size"), n_senders=(C["sender"], "nunique"),
        recall=("flag", "mean"), recall_off=("flag_off", "mean"), recall_head=("flag_head", "mean"))
    by_cfg.to_csv(out / "by_config.csv")
    piv = by_cfg["recall"].unstack("angle")
    piv.to_csv(out / "recall_mag_x_angle.csv")
    f.log("탐지율 (행: |b| m, 열: 진행 방향 기준 각도, 0=앞 90=왼쪽 180=뒤 270=오른쪽):\n" + piv.round(3).to_string())

    # 송신자 단위 부트스트랩 95% CI (종방향 vs 횡방향 요약)
    res["axis"] = np.where(res["angle"].isin([0.0, 180.0]), "longitudinal",
                   np.where(res["angle"].isin([90.0, 270.0]), "lateral", "diagonal"))
    summ = []
    rng = np.random.default_rng(a.seed)
    for (mag, axis), g in res[res["mag"] > 0].groupby(["mag", "axis"]):
        per = g.groupby(C["sender"])["flag"].agg(["sum", "size"])
        sums, sizes = per["sum"].to_numpy(), per["size"].to_numpy()
        boots = []
        for _ in range(500):
            i = rng.integers(0, len(per), len(per))
            boots.append(sums[i].sum() / sizes[i].sum())
        summ.append(dict(mag=mag, axis=axis, recall=sums.sum() / sizes.sum(),
                         ci_lo=np.quantile(boots, 0.025), ci_hi=np.quantile(boots, 0.975),
                         n_senders=len(per)))
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "axis_summary_ci.csv", index=False)
    f.log("축별 요약(송신자 단위 부트스트랩 95% CI):\n" + summ.round(3).to_string(index=False))

    bins = [0, 1, 2, 3, 5, 8, 12, 20, 40, np.inf]
    nz = res[res["mag"] > 0].copy()
    nz["abs_b_perp_bin"] = pd.cut(nz["b_perp"].abs(), bins, right=False)
    nz["abs_b_par_bin"] = pd.cut(nz["b_par"].abs(), bins, right=False)
    loc = nz.pivot_table(index="abs_b_perp_bin", columns="abs_b_par_bin", values="flag",
                         aggfunc="mean", observed=True)
    loc.to_csv(out / "recall_local_perp_x_par.csv")
    f.log("국소 성분 기준 탐지율 (행: |b_perp|, 열: |b_par|):\n" + loc.round(2).to_string())
    f.log(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
