#!/usr/bin/env python3
"""
pilot4.py: 파일럿 4단계.

E3+. 거짓말 예산 (탐색 상한 확장, 포화 표시)
E4+. 예측 차수별 공격자 비교
     order 1 (CV)  : 등속 직선 외삽            d = v*tau
     order 2 (CA)  : 등가속도 + 등요레이트      d = v*tau + a*tau^2/2,      h = h + psi_dot*tau
     order 3 (CJ)  : 등저크 + 선형 요레이트     d = ... + j*tau^3/6,         h = ... + psi_ddot*tau^2/2
       -> 곡률이 호 길이에 선형인 클로소이드에 해당. 이론상 잔차가 저크에서 스냅으로 밀린다.
     각 차수에서 보고 속도·헤딩도 같은 차수로 일관되게 만들어 자기 일관성을 유지한다.
     참조로 oracle(실제 미래를 아는 mode b)도 함께 측정한다.
     급감속 기준을 여러 값으로 나누어 "위험한 순간"의 탐지율을 본다.

사용 예
  python pilot4.py --scored scored_tol5.parquet --max-senders 150
"""
import argparse

import numpy as np
import pandas as pd

import common as cm
import tau_model as tm
from pilot3 import add_noise


# ---------------------------------------------------------------------------
def e3(trajs, horizons, a_lims, tau_hi):
    rows = []
    for sid, d in trajs.items():
        tr = tm.Traj(**d)
        t = tr.t
        t_start = t[0] + 2.0
        if t[-1] - t_start < max(horizons) + 3:
            continue
        for H in horizons:
            t0f, t1f = t_start - 1, t_start + H + 1
            tf = np.arange(t0f, t1f, 0.05)
            v0 = float(tr.spd(t_start))
            for a_lim in a_lims:
                if float(np.nanmax(np.abs(tr.acc(tf)))) > a_lim:
                    continue
                lo, hi = 0.0, tau_hi
                for _ in range(30):
                    mid = (lo + hi) / 2
                    a_m, v_m = tm.peak_constraints(tr, t0f, t1f, mid, mid, t_start, H)
                    lo, hi = (lo, mid) if (a_m > a_lim or v_m > cm.V_MAX) else (mid, hi)
                rows.append(dict(sender=sid, H=H, a_lim=a_lim, v0=v0, tau_max=lo, lie_m=lo * v0,
                                 lie_theory=a_lim * H ** 2 / 4, saturated=bool(lo > tau_hi * 0.99)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def predict_report(tr, t, tau, order, sub=20):
    """차수 order의 테일러 외삽으로 tau초 뒤 상태를 예측해 보고. 위치는 예측 운동을 수치 적분."""
    v, h = tr.spd(t), tr.head(t)
    a = tr.acc(t)
    j = np.gradient(a, t)
    hh = np.unwrap(tr.head(t))
    psi1 = np.gradient(hh, t)
    psi2 = np.gradient(psi1, t)
    if order == 1:
        a = np.zeros_like(a); j = np.zeros_like(j); psi1 = np.zeros_like(psi1); psi2 = np.zeros_like(psi2)
    elif order == 2:
        j = np.zeros_like(j); psi2 = np.zeros_like(psi2)

    x, y = tr._hermite(t, tr.x, tr.mx), tr._hermite(t, tr.y, tr.my)
    du = tau / sub
    for k in range(sub):                      # 예측 운동 적분
        u = (k + 0.5) * du
        vu = v + a * u + 0.5 * j * u ** 2
        hu = hh + psi1 * u + 0.5 * psi2 * u ** 2
        x = x + vu * np.cos(hu) * du
        y = y + vu * np.sin(hu) * du
    v_rep = v + a * tau + 0.5 * j * tau ** 2
    h_rep = hh + psi1 * tau + 0.5 * psi2 * tau ** 2
    return dict(t=t, x=x, y=y, v=v_rep, h=f_wrap(h_rep), a=a + j * tau, ok=tr.valid(t + tau))


def f_wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def e4(trajs, taus, orders, brake_levels, thr_s, thr_v, window, brake_win=3.0):
    rows = []
    for sid, d in trajs.items():
        tr = tm.Traj(**d)
        t = tr.t
        acc = tr.acc(t)
        for tau in taus:
            reps = {f"order{o}": predict_report(tr, t, tau, o) for o in orders}
            tau_a, td, tdd = tm.tau_linear(t, tau, 0.0)
            reps["oracle"] = tm.report(tr, t, tau_a, td, tdd, mode="b")
            for name, rep in reps.items():
                rep = add_noise(rep, tr, t)
                r_s, r_v = tm.residuals(rep)
                ok = rep["ok"] & np.isfinite(r_s)
                ok[0] = False
                if ok.sum() < 5:
                    continue
                ws = pd.Series(r_s).rolling(window, min_periods=1).mean().to_numpy()
                wv = pd.Series(r_v).rolling(window, min_periods=1).mean().to_numpy()
                hit = (ws > thr_s) | (wv > thr_v)
                for bl in brake_levels:
                    brake = np.zeros(len(t), bool)
                    for i in np.flatnonzero(acc < -bl):
                        brake |= np.abs(t - t[i]) <= brake_win
                    for seg, mask in (("brake", ok & brake), ("other", ok & ~brake)):
                        if mask.sum() >= 3:
                            rows.append(dict(sender=sid, tau=tau, attacker=name, brake_level=bl,
                                             segment=seg, n=int(mask.sum()), hit=int(hit[mask].sum()),
                                             r_vector=float(np.median(r_v[mask]))))
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
    ap.add_argument("--horizons", default="1,2,3,5,8,12,20")
    ap.add_argument("--a-lims", default="6,9,12")
    ap.add_argument("--tau-hi", type=float, default=40.0)
    ap.add_argument("--taus", default="1,2,3,5")
    ap.add_argument("--orders", default="1,2,3")
    ap.add_argument("--brake-levels", default="2,4,6")
    a = ap.parse_args()

    trajs = cm.load_benign(a.scored, a.split, a.max_senders)
    if not trajs:
        raise SystemExit("궤적을 불러오지 못했습니다.")

    cm.log("E3+: 거짓말 예산 (탐색 상한 확장)")
    d3 = e3(trajs, [float(v) for v in a.horizons.split(",")],
            [float(v) for v in a.a_lims.split(",")], a.tau_hi)
    cm.save(d3, "e3b_lie_budget.csv")
    s3 = d3.groupby(["a_lim", "H"]).agg(n=("sender", "nunique"), v0=("v0", "median"),
                                        tau_max=("tau_max", "median"), lie_m=("lie_m", "median"),
                                        lie_theory=("lie_theory", "median"),
                                        saturated=("saturated", "mean"))
    s3["lie/theory"] = s3["lie_m"] / s3["lie_theory"]
    cm.log("거짓말 예산:\n" + s3.round(2).to_string())

    cm.log("E4+: 예측 차수별 공격자")
    d4 = e4(trajs, [float(v) for v in a.taus.split(",")], [int(v) for v in a.orders.split(",")],
            [float(v) for v in a.brake_levels.split(",")], a.thr_scalar, a.thr_vector, a.window)
    cm.save(d4, "e4b_predictor_orders.csv")
    for bl, g0 in d4.groupby("brake_level"):
        g = g0.groupby(["tau", "attacker", "segment"])
        s4 = g.agg(n_msg=("n", "sum"), r_vector=("r_vector", "median"))
        s4["det"] = g["hit"].sum() / g["n"].sum()
        piv = s4["det"].unstack(["segment"])
        cm.log(f"급감속 기준 {bl} m/s^2 에서의 탐지율:\n" + piv.round(3).to_string())
    cm.log("해석: order가 높을수록(=클로소이드급 예측) 탐지율이 낮아야 하고, "
           "oracle(실제 미래)은 기준선 수준이어야 한다. brake 구간이 other보다 높으면 "
           "'위험한 순간에 예측이 드러난다'는 주장이 성립한다.")


if __name__ == "__main__":
    main()
