#!/usr/bin/env python3
"""
pilot2.py: 파일럿 2단계.

E1. 탐지 조건을 논문 1(adaptive_v2)과 일치시켜 재측정
    - 잔차에 창 5의 인과적 이동 평균 적용, 임계값 R_scalar=3.018 / R_vector=6.372
    - mode "a"(비보정)와 mode "b"(완전 보정)를 함께 측정
      -> mode b의 탐지율이 기준선(rate=0)과 같으면 "자기 일관성은 유지된다"는 모델 가정 확인

E2. "불가능(infeasible)" 판정의 원인 분리
    tau 고정 시 s_dot=1, s_ddot=0 이므로 a_rep = a(s), 즉 그 차량의 실제 가속도다.
    따라서 max|a_true| > a_lim 인 구간은 tau 변화와 무관한 "기준선 위반"이다.
    - baseline_violation : tau를 고정해도 a_lim 초과 (공격자의 비용이 아님)
    - ramp_infeasible    : 기준선은 통과하나 어떤 T로도 tau 변화가 불가능
    - feasible           : T_min 측정 -> 이론값 2*sqrt(dtau*v/a_lim)과 비교

사용 예
  python pilot2.py --scored scored_tol5.parquet --max-senders 150
"""
import argparse

import numpy as np
import pandas as pd

import common as cm
import tau_model as tm


def windowed(r, window):
    return pd.Series(r).rolling(window, min_periods=1).mean().to_numpy()


# ---------------------------------------------------------------------------
def e1(trajs, rates, thr_s, thr_v, window, tau0):
    rows = []
    for sid, d in trajs.items():
        tr = tm.Traj(**d)
        t = tr.t
        dt = float(np.median(np.diff(t)))
        for rate in rates:
            tau, td, tdd = tm.tau_linear(t, tau0, rate)
            for mode in ("a", "b"):
                rep = tm.report(tr, t, tau, td, tdd, mode=mode)
                r_s, r_v = tm.residuals(rep)
                ok = rep["ok"] & np.isfinite(r_s)
                ok[0] = False
                if ok.sum() < 5:
                    continue
                ws, wv = windowed(r_s, window), windowed(r_v, window)
                rows.append(dict(sender=sid, rate=rate, mode=mode, n=int(ok.sum()),
                                 pred=float(np.median(tr.spd(t)[ok] * abs(rate) * dt)),
                                 r_scalar=float(np.median(r_s[ok])),
                                 r_vector=float(np.median(r_v[ok])),
                                 det_scalar=float(np.mean(ws[ok] > thr_s)),
                                 det_vector=float(np.mean(wv[ok] > thr_v)),
                                 det_any=float(np.mean((ws[ok] > thr_s) | (wv[ok] > thr_v)))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def max_report(tr, t, tau0, dtau, t_start, T, win):
    t0, t1 = float(t[win][0]), float(t[win][-1])
    return tm.peak_constraints(tr, t0, t1, tau0, dtau, t_start, T)


def e2(trajs, dtaus, a_lims, tau0, T_hi):
    rows = []
    for sid, d in trajs.items():
        tr = tm.Traj(**d)
        t = tr.t
        t_start = t[0] + 2.0
        if t[-1] - t_start < 15:
            continue
        v0 = float(tr.spd(t_start + 1))
        for dtau in dtaus:
            T_max = min(T_hi, t[-1] - t_start - 1)
            win = (t >= t_start - 1) & (t <= t_start + T_max + 1)
            if win.sum() < 3:
                continue
            base_a = float(np.nanmax(np.abs(tr.acc(t[win] + tau0))))   # tau 고정 시 보고 가속도
            for a_lim in a_lims:
                rec = dict(sender=sid, dtau=dtau, a_lim=a_lim, v0=v0, base_max_a=base_a,
                           T_theory=tm.theory_min_time(dtau, v0, a_lim))
                if base_a > a_lim:
                    rows.append(dict(rec, status="baseline_violation", T_min=np.nan))
                    continue
                a_hi, v_hi = max_report(tr, t, tau0, dtau, t_start, T_max, win)
                if a_hi > a_lim or v_hi > cm.V_MAX:
                    rows.append(dict(rec, status="ramp_infeasible", T_min=np.nan))
                    continue
                lo, hi = 0.5, T_max
                for _ in range(30):
                    mid = (lo + hi) / 2
                    a_m, v_m = max_report(tr, t, tau0, dtau, t_start, mid, win)
                    if a_m <= a_lim and v_m <= cm.V_MAX:
                        hi = mid
                    else:
                        lo = mid
                rows.append(dict(rec, status="feasible", T_min=hi))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="scored_tol5.parquet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-senders", type=int, default=150)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--thr-scalar", type=float, default=3.018)
    ap.add_argument("--thr-vector", type=float, default=6.372)
    ap.add_argument("--tau0", type=float, default=5.0)
    ap.add_argument("--rates", default="0,0.01,0.02,0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--dtaus", default="1,2,3,5")
    ap.add_argument("--a-lims", default="3,4,6,9,12")
    ap.add_argument("--T-hi", type=float, default=60.0)
    a = ap.parse_args()

    trajs = cm.load_benign(a.scored, a.split, a.max_senders)
    if not trajs:
        raise SystemExit("궤적을 불러오지 못했습니다.")

    # 참조: 정상 궤적의 가속도 분포
    acc = np.concatenate([np.abs(np.gradient(d["v"], d["t"])) for d in trajs.values()])
    cm.log(f"정상 궤적 |a| 분포: median {np.median(acc):.2f}, p95 {np.quantile(acc, .95):.2f}, "
           f"p99 {np.quantile(acc, .99):.2f}, max {acc.max():.2f} m/s^2")

    cm.log(f"E1: 창 {a.window} 이동 평균 적용 탐지 (임계값 {a.thr_scalar}/{a.thr_vector})")
    d1 = e1(trajs, [float(v) for v in a.rates.split(",")], a.thr_scalar, a.thr_vector, a.window, a.tau0)
    cm.save(d1, "e1_windowed_detection.csv")
    s1 = d1.pivot_table(index="rate", columns="mode",
                        values=["r_scalar", "det_scalar", "det_vector", "det_any"], aggfunc="median")
    cm.log("tau 변화율 x 보고 방식:\n" + s1.round(3).to_string())
    base = d1[(d1["rate"] == 0) & (d1["mode"] == "a")]["det_any"].median()
    cm.log(f"기준선(rate=0) 탐지율 = {base:.3f}  <- 논문 1의 대조군 수준과 비슷해야 함")
    ma = d1[(d1["mode"] == "a") & (d1["rate"] > 0)].groupby("rate")["det_any"].median()
    onset = ma[ma > base + 0.1]
    cm.log(f"탐지 시작 변화율(기준선 +0.1 초과): {onset.index.min() if len(onset) else '없음'}")

    cm.log("E2: 불가능 판정의 원인 분리")
    d2 = e2(trajs, [float(v) for v in a.dtaus.split(",")], [float(v) for v in a.a_lims.split(",")],
            a.tau0, a.T_hi)
    cm.save(d2, "e2_recovery_decomposed.csv")
    st = d2.pivot_table(index=["dtau", "a_lim"], columns="status", values="sender",
                        aggfunc="count", fill_value=0)
    st = st.div(st.sum(axis=1), axis=0)
    fe = d2[d2["status"] == "feasible"].copy()
    fe["ratio"] = fe["T_min"] / fe["T_theory"]
    summ = fe.groupby(["dtau", "a_lim"]).agg(n=("sender", "nunique"), T_min=("T_min", "median"),
                                             T_theory=("T_theory", "median"), ratio=("ratio", "median"))
    cm.log("상태 비율:\n" + st.round(3).to_string())
    cm.log("feasible 인 경우의 최소 회수 시간:\n" + summ.round(2).to_string())
    cm.log("해석: baseline_violation 비율이 높으면 그 a_lim은 정상 주행도 넘는 값이므로 "
           "제약으로 쓰기에 부적절하다. ramp_infeasible 만이 tau 변화 자체의 비용이다.")


if __name__ == "__main__":
    main()
