#!/usr/bin/env python3
"""
pilot_theory.py: 적응형 시간 왜곡 공격의 두 이론 관계식 검증.

실험 1 (자기 일관성 비용): mode "a"에서 tau를 일정 비율로 바꿀 때
    이론  r ~ v * |tau_dot| * dt
    측정  스칼라/벡터 잔차의 중앙값, 이론값과의 비(ratio)와 상관, 탐지율

실험 2 (물리적 회수 비용): mode "b"에서 tau를 dtau 만큼 줄일 때
    이론  T_min = 2 * sqrt(dtau * v / a_lim)
    측정  실제 궤적에서 보고 가속도 |a_rep| <= a_lim 를 만족하는 최소 T (이분 탐색)

사용 예
  python pilot_theory.py --scored scored_tol5.parquet --max-senders 100
"""
import argparse

import numpy as np
import pandas as pd

import common as cm
import tau_model as tm


def exp1(trajs, rates, thr_scalar, thr_vector, tau0=2.0):
    rows = []
    for sid, d in trajs.items():
        tr = tm.Traj(**d)
        t = tr.t
        for rate in rates:
            tau, td, tdd = tm.tau_linear(t, tau0, rate)
            rep = tm.report(tr, t, tau, td, tdd, mode="a")
            r_s, r_v = tm.residuals(rep)
            ok = rep["ok"] & np.isfinite(r_s)
            ok[0] = False
            if ok.sum() < 5:
                continue
            dt = np.median(np.diff(t))
            pred = tr.spd(t) * abs(rate) * dt
            rows.append(dict(sender=sid, rate=rate, n=int(ok.sum()),
                             v_med=float(np.median(tr.spd(t)[ok])),
                             pred_med=float(np.median(pred[ok])),
                             r_scalar_med=float(np.median(r_s[ok])),
                             r_vector_med=float(np.median(r_v[ok])),
                             det_scalar=float(np.mean(r_s[ok] > thr_scalar)),
                             det_vector=float(np.mean(r_v[ok] > thr_vector))))
    return pd.DataFrame(rows)


def max_arep(tr, t, tau0, dtau, t_start, T):
    tau, td, tdd = tm.tau_ramp(t, tau0, dtau, t_start, T)
    rep = tm.report(tr, t, tau, td, tdd, mode="b")
    m = rep["ok"] & (t >= t_start - 1) & (t <= t_start + T + 1)
    if m.sum() < 3:
        return np.inf, np.inf
    return float(np.nanmax(np.abs(rep["a"][m]))), float(np.nanmax(rep["v"][m]))


def exp2(trajs, dtaus, a_lims, tau0=5.0, T_hi=60.0):
    rows = []
    for sid, d in trajs.items():
        tr = tm.Traj(**d)
        t = tr.t
        t_start = t[0] + 2.0
        v0 = float(tr.spd(t_start + 1))
        if t[-1] - t_start < 15:
            continue
        for dtau in dtaus:
            for a_lim in a_lims:
                lo, hi = 0.5, min(T_hi, t[-1] - t_start - 1)
                a_hi, _ = max_arep(tr, t, tau0, dtau, t_start, hi)
                if a_hi > a_lim:            # 가장 느린 변화로도 한계 초과 -> 불가
                    rows.append(dict(sender=sid, dtau=dtau, a_lim=a_lim, v0=v0,
                                     T_min=np.nan, T_theory=tm.theory_min_time(dtau, v0, a_lim),
                                     feasible=False))
                    continue
                for _ in range(30):         # 이분 탐색
                    mid = (lo + hi) / 2
                    a_m, v_m = max_arep(tr, t, tau0, dtau, t_start, mid)
                    if a_m <= a_lim and v_m <= cm.V_MAX:
                        hi = mid
                    else:
                        lo = mid
                rows.append(dict(sender=sid, dtau=dtau, a_lim=a_lim, v0=v0, T_min=hi,
                                 T_theory=tm.theory_min_time(dtau, v0, a_lim), feasible=True))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="scored_tol5.parquet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-senders", type=int, default=100)
    ap.add_argument("--thr-scalar", type=float, default=3.018, help="adaptive_v2 임계값")
    ap.add_argument("--thr-vector", type=float, default=6.372)
    ap.add_argument("--rates", default="0,0.02,0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--dtaus", default="1,2,3,5")
    ap.add_argument("--a-lims", default="2,3,4")
    a = ap.parse_args()

    trajs = cm.load_benign(a.scored, a.split, a.max_senders)
    if not trajs:
        raise SystemExit("궤적을 불러오지 못했습니다. --scored/--split 확인")

    cm.log("실험 1: 자기 일관성 비용 (mode a)")
    e1 = exp1(trajs, [float(v) for v in a.rates.split(",")], a.thr_scalar, a.thr_vector)
    cm.save(e1, "exp1_selfconsistency.csv")
    s1 = e1.groupby("rate").agg(n_senders=("sender", "nunique"), v_med=("v_med", "median"),
                                pred=("pred_med", "median"), r_scalar=("r_scalar_med", "median"),
                                r_vector=("r_vector_med", "median"),
                                det_scalar=("det_scalar", "mean"), det_vector=("det_vector", "mean"))
    base = float(s1.loc[0.0, "r_scalar"]) if 0.0 in s1.index else 0.0
    s1["scalar_excess"] = s1["r_scalar"] - base          # tau 변화가 없을 때의 기본 잔차 제거
    s1["excess/pred"] = s1["scalar_excess"] / s1["pred"].replace(0, np.nan)
    cm.log("tau 변화율별 (값은 중앙값, 탐지율은 평균). excess/pred가 1 근처면 이론 r = v|tau_dot|dt 성립:\n"
           + s1.round(3).to_string())
    m = e1[e1["rate"] > 0]
    if len(m) > 3:
        cm.log(f"이론값 대비 실측 스칼라 잔차 상관 r={np.corrcoef(m['pred_med'], m['r_scalar_med'])[0, 1]:.3f}")

    cm.log("실험 2: 물리적 회수 비용 (mode b)")
    e2 = exp2(trajs, [float(v) for v in a.dtaus.split(",")], [float(v) for v in a.a_lims.split(",")])
    cm.save(e2, "exp2_recovery_time.csv")
    ok = e2[e2["feasible"]]
    s2 = ok.groupby(["dtau", "a_lim"]).agg(n=("sender", "nunique"),
                                           T_min_med=("T_min", "median"),
                                           T_theory_med=("T_theory", "median"),
                                           ratio_med=("T_min", "median"))
    s2["ratio_med"] = (ok.assign(r=ok["T_min"] / ok["T_theory"]).groupby(["dtau", "a_lim"])["r"].median())
    s2["infeasible_frac"] = (e2.groupby(["dtau", "a_lim"])["feasible"]
                             .apply(lambda s: 1 - s.mean()))
    cm.log("dtau(초) x a_lim(m/s^2) 별 최소 회수 시간:\n" + s2.round(2).to_string())
    cm.log("해석: T_min/T_theory 비가 1 근처면 이론이 실제 궤적에서도 성립. "
           "1보다 크면 가감속 중인 구간에서 추가 비용이 있다는 뜻.")


if __name__ == "__main__":
    main()
