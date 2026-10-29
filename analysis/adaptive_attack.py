#!/usr/bin/env python3
"""
지도 인지 적응형 공격자(A1/A2) 준합성 주입 실험.

공격 생성(SUMO 궤적 기반, 오라클 불필요):
  control : 원래 benign 메시지 그대로 (FPR 대조군)
  A1(tau) : 위치만 tau초 뒤 보고값, 속도·헤딩은 현재값        (순진형)
  A1h(tau): 위치·헤딩을 tau초 뒤 보고값, 속도만 현재값          (헤딩까지 일관)
  A2(tau) : 위치·속도·헤딩 전부 tau초 뒤 보고값                 (일관형)
  미래 보고값은 같은 (수신자, 송신자) 시퀀스에서 가져온다(수신자별 노이즈 차이에 안전).
  노이즈는 메시지마다 독립(자기상관 ~0)이므로 미래 메시지의 노이즈를 그대로 써도 통계가 같다.
  비교 공정성을 위해 모든 조건을 max(tau)초 뒤 메시지가 존재하는 행으로 제한한다.

탐지기(임계값은 --thr-split benign control에서 각자 목표 FPR로 선정):
  R_scalar : | |dp| - v_prev*dt |              (방향 정보 없음)
  R_vector : | dp - v_mean*dt*u(h_mid) |       (보고 헤딩 사용, h_mid = 직전·현재 헤딩의 중간각)
  F5_off   : 도로 이탈 거리
  F5_head  : 보고 헤딩 vs 보고 위치의 차로 방향 오차
  COMB     : R_scalar | R_vector | F5_off | F5_head
  window>1이면 잔차는 인과적 이동 평균, 나머지는 인과적 이동 중앙값.

분석(오라클 사용 = 분석 전용):
  map_turn_deg : 참위치에서 본 t-1 과 t+tau-1 사이 도로 접선각 변화(평활 도로, sigma)
  pred_resid   : |v(t+tau-1)u(h(t+tau-1)) - v(t-1)u(h(t-1))| * dt   (A1 잔차 이론값)

사용 예
  python adaptive_attack.py --map map.pkl --scored scored_tol5.parquet --transform transform.json --out-dir adaptive_v1
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, shapely

import f5_map_check as f
import road_geom as rg

C = f.COL
KEY = [C["receiver"], C["sender"], "config"]


# ---------------------------------------------------------------------------
PAIR = [C["receiver"], C["sender"]]


def sender_states(b, s_h, k_h, T, m, sm):
    """(수신자, 송신자, 시각) 단위 보고 상태 표 (+분석용 참위치 도로 접선각·곡률)."""
    st = b.drop_duplicates(PAIR + [C["t"]]).copy()
    st["hm"] = f.heading_math_from(st, s_h, k_h, T)
    st["x_rep"], st["y_rep"] = f.apply_transform(st[C["x"]], st[C["y"]], T)
    nx = st["position_noise_x"] if "position_noise_x" in st else 0.0
    ny = st["position_noise_y"] if "position_noise_y" in st else 0.0
    tx, ty = f.apply_transform(st[C["x"]] - nx, st[C["y"]] - ny, T)
    li = rg.nearest_same_direction_lane(m, tx, ty, st["hm"].to_numpy())
    th, kap = rg.lookup(m, sm, np.where(li >= 0, li, 0), tx, ty)
    st["theta_map"] = np.where(li >= 0, th, np.nan)
    st["kappa_map"] = np.where(li >= 0, kap, np.nan)
    cols = PAIR + [C["t"], "x_rep", "y_rep", C["speed"], "hm", "theta_map", "kappa_map"]
    if "position_residual" in st:
        cols.append("position_residual")
    return st[cols].sort_values(C["t"]).reset_index(drop=True)


def attach_future(st, tau, tol=0.2):
    fut = st.rename(columns={c: "f_" + c for c in st.columns if c not in PAIR + [C["t"]]})
    left = st[PAIR + [C["t"]]].copy()
    left["t_target"] = left[C["t"]] + tau
    fut = fut.rename(columns={C["t"]: "t_target"})
    j = pd.merge_asof(left.sort_values("t_target"), fut.sort_values("t_target"),
                      on="t_target", by=PAIR, tolerance=tol, direction="nearest")
    return j.drop(columns="t_target")


def build_configs(df, split, taus, s_h, k_h, T, m, sm, max_senders, seed):
    rows = df[(df[C["variant"]] == f.BENIGN_VARIANT) & (df[C["split"]] == split)]
    if max_senders:
        rng = np.random.default_rng(seed)
        keep = rng.choice(rows[C["sender"]].unique(), min(max_senders, rows[C["sender"]].nunique()), replace=False)
        rows = rows[rows[C["sender"]].isin(keep)]
    base = sender_states(rows, s_h, k_h, T, m, sm)

    futs = {tau: attach_future(base, tau) for tau in taus}
    valid = np.ones(len(base), bool)
    fcols = {}
    for tau in taus:
        fb = base[PAIR + [C["t"]]].merge(futs[tau], on=PAIR + [C["t"]], how="left")
        fcols[tau] = fb
        valid &= fb["f_x_rep"].notna().to_numpy()
    f.log(f"{split}: 수신 행 {len(base)}개 중 tau={max(taus)}초 뒤 메시지 존재 {valid.sum()}개 사용")

    out = []
    b0 = base[valid].copy()
    for c in ("x_rep", "y_rep", C["speed"], "hm"):
        b0["r_" + c] = b0[c]
    b0["kappa_rep"] = b0["kappa_map"]
    b0["config"], b0["attacker"], b0["tau"] = "control", "control", 0
    b0["o_theta"], b0["f_theta"], b0["f_speed_a"], b0["f_hm_a"] = b0["theta_map"], b0["theta_map"], b0[C["speed"]], b0["hm"]
    out.append(b0)
    for tau in taus:
        fb = fcols[tau][valid]
        for att in ("A1", "A1h", "A2"):
            g = base[valid].copy()
            g["r_x_rep"], g["r_y_rep"] = fb["f_x_rep"].to_numpy(), fb["f_y_rep"].to_numpy()
            fut_speed, fut_hm = fb["f_" + C["speed"]].to_numpy(), fb["f_hm"].to_numpy()
            g["r_" + C["speed"]] = fut_speed if att == "A2" else g[C["speed"]]
            g["r_hm"] = g["hm"] if att == "A1" else fut_hm
            g["o_theta"], g["f_theta"] = g["theta_map"], fb["f_theta_map"].to_numpy()
            g["f_speed_a"], g["f_hm_a"] = fb["f_" + C["speed"]].to_numpy(), fb["f_hm"].to_numpy()
            g["kappa_rep"] = fb["f_kappa_map"].to_numpy()   # 보고 위치(=실제로 있었던 위치)의 도로 곡률
            g["config"], g["attacker"], g["tau"] = f"{att}_tau{tau:g}", att, tau
            out.append(g)
    res = pd.concat(out, ignore_index=True)
    return res.sort_values(KEY + [C["t"]]).reset_index(drop=True)


# ---------------------------------------------------------------------------
def features(res, m, head_tol, v_mode):
    g = res.groupby(KEY, sort=False)
    prev = lambda c: g[c].shift()
    dt = res[C["t"]] - prev(C["t"])
    dx, dy = res["r_x_rep"] - prev("r_x_rep"), res["r_y_rep"] - prev("r_y_rep")
    vp, vc = prev("r_" + C["speed"]), res["r_" + C["speed"]]
    v = {"prev": vp, "cur": vc, "mean": (vp + vc) / 2}[v_mode]
    hp, hc = prev("r_hm"), res["r_hm"]
    hmid = hp + f.wrap_angle(hc - hp) / 2
    res["R_scalar"] = (np.hypot(dx, dy) - v * dt).abs()
    res["R_vector"] = np.hypot(dx - v * dt * np.cos(hmid), dy - v * dt * np.sin(hmid))
    px, py = res["r_x_rep"].to_numpy(), res["r_y_rep"].to_numpy()
    res["F5_off"] = shapely.distance(m["area"], shapely.points(px, py))
    herr = f.heading_error(m, px, py, res["r_hm"].to_numpy(), tol_m=head_tol)
    herr[res["r_" + C["speed"]].abs().to_numpy() < f.MIN_SPEED_FOR_HEADING] = np.nan
    res["F5_head"] = np.degrees(herr)
    # 분석용 이론값
    res["map_turn_deg"] = np.degrees(np.abs(f.wrap_angle(prev("f_theta") - prev("o_theta"))))
    fv, fh = prev("f_speed_a"), prev("f_hm_a")
    ov, oh = prev(C["speed"]), prev("hm")
    res["pred_resid"] = np.hypot(fv * np.cos(fh) - ov * np.cos(oh), fv * np.sin(fh) - ov * np.sin(oh)) * dt
    res["abs_dv"] = (fv - ov).abs()
    res["dt"] = dt
    return res


def windowed(res, window):
    for c, how in (("R_scalar", "mean"), ("R_vector", "mean"), ("F5_off", "median"), ("F5_head", "median")):
        if window <= 1:
            res[c + "_w"] = res[c]
            continue
        r = res.groupby(KEY, sort=False)[c].rolling(window, min_periods=1)
        r = r.mean() if how == "mean" else r.median()
        res[c + "_w"] = r.reset_index(level=list(range(len(KEY))), drop=True)
    return res


def calibrate_residual(res):
    ctl = res[(res["config"] == "control")]
    if "position_residual" not in ctl or ctl["position_residual"].notna().sum() == 0:
        f.log("position_residual 컬럼 없음 -> v_prev, 스칼라/벡터 둘 다 평가")
        return "prev", None
    rep = {}
    g = ctl.groupby(KEY, sort=False)
    dt = ctl[C["t"]] - g[C["t"]].shift()
    dx, dy = ctl["r_x_rep"] - g["r_x_rep"].shift(), ctl["r_y_rep"] - g["r_y_rep"].shift()
    vp, vc = g["r_" + C["speed"]].shift(), ctl["r_" + C["speed"]]
    hp = g["r_hm"].shift()
    hp = hp + f.wrap_angle(ctl["r_hm"] - hp) / 2
    for mode, v in (("prev", vp), ("cur", vc), ("mean", (vp + vc) / 2)):
        sc = (np.hypot(dx, dy) - v * dt).abs()
        ve = np.hypot(dx - v * dt * np.cos(hp), dy - v * dt * np.sin(hp))
        for kind, val in (("scalar", sc), ("vector", ve)):
            d = (val - ctl["position_residual"]).abs()
            d = d[np.isfinite(d)]
            rep[(mode, kind)] = (float(np.median(d)), float(np.mean(d < 1e-3))) if len(d) else (np.inf, 0.0)
    for k, (md, fr) in rep.items():
        f.log(f"  기존 position_residual 대조: v={k[0]:4s} {k[1]:6s} median|차이|={md:.4f}, 일치(<1e-3) 비율={fr:.3f}")
    best = min(rep, key=lambda k: rep[k][0])
    f.log(f"기존 구현과 가장 가까운 정의: v={best[0]}, {best[1]} 잔차")
    if rep[best][1] < 0.9:
        f.log("경고: 어떤 정의도 기존 컬럼과 정확히 일치하지 않음(순서·dt 처리 차이 가능). v=prev로 진행")
        return "prev", best[1]
    return best[0], best[1]


DETECTORS = ["R_scalar", "R_vector", "F5_off", "F5_head"]


def thresholds(ctl, fpr, theta_deg):
    th = {}
    for d in DETECTORS:
        v = ctl[d + "_w"].dropna()
        th[d] = float(np.quantile(v, 1 - fpr))
    if theta_deg is not None:
        th["F5_head"] = float(theta_deg)
    return th


def flags(res, th):
    for d in DETECTORS:
        res["flag_" + d] = res[d + "_w"] > th[d]
    res["flag_COMB"] = res["flag_R_scalar"] | res["flag_R_vector"] | res["flag_F5_off"] | res["flag_F5_head"]
    return res


def boot_ci(g, col, n=500, seed=0):
    per = g.groupby(C["sender"])[col].agg(["sum", "size"])
    s, z = per["sum"].to_numpy(), per["size"].to_numpy()
    rng = np.random.default_rng(seed)
    bs = [s[i].sum() / z[i].sum() for i in (rng.integers(0, len(per), len(per)) for _ in range(n))]
    return np.quantile(bs, 0.025), np.quantile(bs, 0.975)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--scored", required=True)
    ap.add_argument("--transform")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--thr-split", default="validation")
    ap.add_argument("--taus", default="1,2,3,5")
    ap.add_argument("--fpr", type=float, default=0.01, help="탐지기별 목표 FPR")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--theta-deg", type=float, default=None, help="F5_head 임계값 고정(미지정 시 분위수)")
    ap.add_argument("--sigma", type=float, default=5.0, help="분석용 도로 평활 sigma(m)")
    ap.add_argument("--max-senders", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    m = f.load_map(a.map)
    T = f.load_transform(a.transform)
    meta = json.load(open(Path(a.scored).with_suffix(".meta.json")))
    s_h, k_h, head_tol = meta["heading_s"], meta["heading_k"], meta.get("head_tol_m", 1.0)
    sm = rg.build_smooth(m, a.sigma)
    df = f.read_table(a.scored)
    taus = [float(v) for v in a.taus.split(",")]
    f.log(f"준비 완료 ({time.time() - t0:.0f}s). head_tol_m={head_tol}, window={a.window}, 목표 FPR={a.fpr}")

    # 임계값 split (control만 필요)
    thr = build_configs(df, a.thr_split, taus[:1], s_h, k_h, T, m, sm, 0, a.seed)
    thr = thr[thr["config"] == "control"].copy()
    v_mode, _ = calibrate_residual(thr)
    thr = windowed(features(thr, m, head_tol, v_mode), a.window)
    th = thresholds(thr, a.fpr, a.theta_deg)
    f.log("임계값: " + ", ".join(f"{k}={v:.3f}" for k, v in th.items()) + f"  ({time.time() - t0:.0f}s)")

    # 평가 split
    res = build_configs(df, a.eval_split, taus, s_h, k_h, T, m, sm, a.max_senders, a.seed)
    f.log(f"평가 행 {len(res)}개 특징 계산 중")
    res = flags(windowed(features(res, m, head_tol, v_mode), a.window), th)
    res = res[res["dt"].notna()]  # 시퀀스 첫 메시지 제외

    fl = ["flag_" + d for d in DETECTORS] + ["flag_COMB"]
    tab = res.groupby(["attacker", "tau"])[fl].mean()
    tab.insert(0, "n_senders", res.groupby(["attacker", "tau"])[C["sender"]].nunique())
    ci = {k: boot_ci(g, "flag_COMB", seed=a.seed) for k, g in res.groupby(["attacker", "tau"])}
    tab["COMB_ci_lo"] = [ci[k][0] for k in tab.index]
    tab["COMB_ci_hi"] = [ci[k][1] for k in tab.index]
    tab.to_csv(out / "detection_by_attacker_tau.csv")
    f.log("탐지율 (control 행 = FPR):\n" + tab.round(3).to_string())

    # A1/A2: 도로 접선각 변화 구간별
    bins = [0, 2, 5, 10, 20, 45, 90, 180.01]
    att = res[res["attacker"] != "control"].copy()
    att["turn_bin"] = pd.cut(att["map_turn_deg"], bins, right=False)
    by_turn = att.groupby(["attacker", "tau", "turn_bin"], observed=True).agg(
        n=("flag_COMB", "size"), n_senders=(C["sender"], "nunique"),
        pred_resid_med=("pred_resid", "median"), R_vector_med=("R_vector", "median"),
        R_scalar=("flag_R_scalar", "mean"), R_vector=("flag_R_vector", "mean"),
        F5_head=("flag_F5_head", "mean"), COMB=("flag_COMB", "mean"))
    by_turn.to_csv(out / "detection_by_map_turn.csv")
    f.log("도로 접선각 변화(|turn|, deg) 구간별:\n" + by_turn.round(3).to_string())

    # 같은 도로 곡률 조건에서 대조군과 비교 (A2가 대조군과 같으면 곡선에서도 탐지 불가)
    rad = 1.0 / res["kappa_rep"].abs().clip(lower=1e-6)
    res["radius_bin"] = pd.cut(rad, [0, 50, 100, 200, 500, 1000, np.inf], right=False)
    by_k = res.groupby(["radius_bin", "attacker", "tau"], observed=True).agg(
        n=("flag_COMB", "size"), n_senders=(C["sender"], "nunique"),
        R_vector=("flag_R_vector", "mean"), F5_head=("flag_F5_head", "mean"), COMB=("flag_COMB", "mean"))
    by_k.to_csv(out / "detection_by_radius_vs_control.csv")
    f.log("보고 위치의 도로 곡률 반경(m) 구간별, 대조군 포함:\n" + by_k.round(3).to_string())

    # 속도 변화가 작은 경우만(|dv|<0.5): 곡률 효과 분리
    cs = att[att["abs_dv"] < 0.5]
    by_turn_cs = cs.groupby(["attacker", "tau", "turn_bin"], observed=True).agg(
        n=("flag_COMB", "size"), n_senders=(C["sender"], "nunique"),
        R_scalar=("flag_R_scalar", "mean"), R_vector=("flag_R_vector", "mean"),
        F5_head=("flag_F5_head", "mean"), COMB=("flag_COMB", "mean"))
    by_turn_cs.to_csv(out / "detection_by_map_turn_const_speed.csv")
    f.log("등속(|dv|<0.5 m/s)만, 도로 접선각 변화 구간별:\n" + by_turn_cs.round(3).to_string())

    # A1: 속도 변화 구간별 (곡률과 분리해서 보기 위해 |turn|<2도만)
    a1s = att[(att["attacker"] == "A1") & (att["map_turn_deg"] < 2)].copy()
    a1s["dv_bin"] = pd.cut(a1s["abs_dv"], [0, 0.5, 1, 2, 4, 8, np.inf], right=False)
    by_dv = a1s.groupby(["tau", "dv_bin"], observed=True).agg(
        n=("flag_COMB", "size"), R_scalar=("flag_R_scalar", "mean"),
        R_vector=("flag_R_vector", "mean"), COMB=("flag_COMB", "mean"))
    by_dv.to_csv(out / "A1_straight_by_speed_change.csv")
    f.log("A1, 직선 구간(|turn|<2도)에서 속도 변화 |dv| 구간별:\n" + by_dv.round(3).to_string())

    # 이론값 대비 실측 잔차 (A1, window 없는 R_vector)
    a1 = att[att["attacker"] == "A1"][["pred_resid", "R_vector"]].dropna()
    ctl = res[res["attacker"] == "control"]["R_vector"].dropna()
    if len(a1) > 10:
        r = np.corrcoef(a1["pred_resid"], a1["R_vector"])[0, 1]
        f.log(f"A1 이론 잔차 vs 실측 R_vector 상관 r={r:.3f} "
              f"(노이즈 영향: control R_vector median={ctl.median():.2f} m)")

    json.dump(dict(thresholds=th, v_mode=v_mode, window=a.window, fpr=a.fpr, taus=taus,
                   sigma=a.sigma, head_tol_m=head_tol), open(out / "config.json", "w"), indent=2)
    f.log(f"결과 저장: {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()