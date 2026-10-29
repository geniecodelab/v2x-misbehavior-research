#!/usr/bin/env python3
"""
pilot3.py: 파일럿 3단계.

E1'. 조건을 논문 1과 맞춘 탐지율 재측정
     - 보고 메시지에 원래 메시지의 노이즈를 다시 더한다(위치·속도·헤딩)
     - 진짜 기준선 "none"(tau = 0) 포함. 이 값이 논문 1의 대조군 수준이어야 보간 오차가 없다는 뜻
     - A2_fixed(자기 일관, tau 고정), A2_rate(tau 변화), A1_fixed(위치만 왜곡)를 비교
     - 집계는 송신자 중앙값이 아니라 전체 메시지 평균(논문 1과 동일)

E3.  "사건까지 남은 시간 H vs 유지 가능한 거짓말 크기" 곡선
     tau를 0으로 되돌리는 최소 시간이 T = 2*sqrt(tau*v/a_lim) 이므로
     H초 안에 되돌릴 수 있는 최대 tau는 tau_max = a_lim * H^2 / (4v),
     이때 공간 거짓말은 d = v * tau_max = a_lim * H^2 / 4  (속도와 무관)
     -> 실제 궤적에서 이 관계가 성립하는지 이분 탐색으로 확인한다.

E4.  예측 기반 공격자 (미래를 모르는 현실적 tau>0 공격자)
     등가속도 외삽으로 tau초 뒤를 예측해 보고. 이론 잔차 ~ (1/2) * j * tau^2 * dt
     급감속 사건 부근과 그 밖에서 탐지율을 비교한다.

사용 예
  python pilot3.py --scored scored_tol5.parquet --max-senders 150
"""
import argparse

import numpy as np
import pandas as pd

import common as cm
import tau_model as tm


def add_noise(rep, tr, t):
    """보고 메시지에 그 시각 메시지의 노이즈 실현값을 더한다(논문 1과 같은 조건)."""
    n = tr.noise
    if not n:
        return rep
    idx = np.clip(np.searchsorted(tr.t, t), 0, len(tr.t) - 1)
    out = dict(rep)
    out["x"] = rep["x"] + n["x"][idx]
    out["y"] = rep["y"] + n["y"][idx]
    out["v"] = rep["v"] + n["v"][idx]
    out["h"] = rep["h"] + n["h"][idx]
    return out


def detect(rep, thr_s, thr_v, window):
    r_s, r_v = tm.residuals(rep)
    ok = rep["ok"] & np.isfinite(r_s)
    ok[0] = False
    if ok.sum() < 5:
        return None
    ws = pd.Series(r_s).rolling(window, min_periods=1).mean().to_numpy()
    wv = pd.Series(r_v).rolling(window, min_periods=1).mean().to_numpy()
    return dict(n=int(ok.sum()), r_scalar=float(np.median(r_s[ok])), r_vector=float(np.median(r_v[ok])),
                hit_s=int(np.sum(ws[ok] > thr_s)), hit_v=int(np.sum(wv[ok] > thr_v)),
                hit_any=int(np.sum((ws[ok] > thr_s) | (wv[ok] > thr_v))))


# ---------------------------------------------------------------------------
def e1(trajs, rates, tau0, thr_s, thr_v, window):
    rows = []
    for sid, d in trajs.items():
        tr = tm.Traj(**d)
        t = tr.t
        cfgs = [("none", 0.0, 0.0, "b"), ("A2_fixed", tau0, 0.0, "b"), ("A1_fixed", tau0, 0.0, "a")]
        cfgs += [(f"A2_rate{r:g}", tau0, r, "b") for r in rates if r > 0]
        for name, t0, rate, mode in cfgs:
            tau, td, tdd = tm.tau_linear(t, t0, rate)
            rep = add_noise(tm.report(tr, t, tau, td, tdd, mode=mode), tr, t)
            r = detect(rep, thr_s, thr_v, window)
            if r:
                rows.append(dict(sender=sid, config=name, rate=rate, mode=mode, **r))
    return pd.DataFrame(rows)


def agg(df, by="config"):
    g = df.groupby(by)
    out = g.agg(n_senders=("sender", "nunique"), n_msg=("n", "sum"),
                r_scalar=("r_scalar", "median"), r_vector=("r_vector", "median"))
    for c in ("hit_s", "hit_v", "hit_any"):
        out[c.replace("hit", "det")] = g[c].sum() / g["n"].sum()
    return out


# ---------------------------------------------------------------------------
def e3(trajs, horizons, a_lims, tau_hi=15.0):
    """H초 안에 tau를 0으로 되돌릴 수 있는 최대 tau와 그때의 거짓말 크기."""
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
                    continue                      # 기준선 위반 구간은 제외
                lo, hi = 0.0, tau_hi
                for _ in range(25):               # H초 안에 되돌릴 수 있는 최대 tau 이분 탐색
                    mid = (lo + hi) / 2
                    a_m, v_m = tm.peak_constraints(tr, t0f, t1f, mid, mid, t_start, H)
                    bad = (a_m > a_lim) or (v_m > cm.V_MAX)
                    lo, hi = (lo, mid) if bad else (mid, hi)
                rows.append(dict(sender=sid, H=H, a_lim=a_lim, v0=v0, tau_max=lo,
                                 saturated=bool(lo > tau_hi * 0.99),
                                 lie_m=lo * v0, lie_theory=a_lim * H ** 2 / 4,
                                 tau_theory=a_lim * H ** 2 / (4 * v0)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def e4(trajs, taus, thr_s, thr_v, window, brake_a=3.0, brake_win=3.0):
    """등가속도 외삽으로 미래를 예측해 보고하는 공격자."""
    rows = []
    for sid, d in trajs.items():
        tr = tm.Traj(**d)
        t = tr.t
        a = tr.acc(t)
        jerk = np.gradient(a, t)
        brake = np.zeros(len(t), bool)
        for i in np.flatnonzero(a < -brake_a):
            brake |= np.abs(t - t[i]) <= brake_win
        for tau in taus:
            # 등가속도 외삽: 위치·속도는 현재 상태에서 tau초 외삽, 헤딩은 현재값
            v, h = tr.spd(t), tr.head(t)
            x = tr.x if False else tr._hermite(t, tr.x, tr.mx)
            y = tr._hermite(t, tr.y, tr.my)
            x = x + (v * tau + 0.5 * a * tau ** 2) * np.cos(h)
            y = y + (v * tau + 0.5 * a * tau ** 2) * np.sin(h)
            rep = dict(t=t, x=x, y=y, v=v + a * tau, h=h, a=a, ok=tr.valid(t + tau))
            rep = add_noise(rep, tr, t)
            r_s, r_v = tm.residuals(rep)
            ok = rep["ok"] & np.isfinite(r_s)
            ok[0] = False
            if ok.sum() < 5:
                continue
            ws = pd.Series(r_s).rolling(window, min_periods=1).mean().to_numpy()
            wv = pd.Series(r_v).rolling(window, min_periods=1).mean().to_numpy()
            hit = (ws > thr_s) | (wv > thr_v)
            for seg, mask in (("brake", ok & brake), ("other", ok & ~brake)):
                if mask.sum() >= 3:
                    rows.append(dict(sender=sid, tau=tau, segment=seg, n=int(mask.sum()),
                                     hit=int(hit[mask].sum()),
                                     pred_resid=float(np.median(0.5 * np.abs(jerk[mask]) * tau ** 2)),
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
    ap.add_argument("--tau0", type=float, default=5.0)
    ap.add_argument("--rates", default="0.01,0.02,0.05,0.1,0.2,0.3")
    ap.add_argument("--horizons", default="1,2,3,5,8,12")
    ap.add_argument("--a-lims", default="6,9,12")
    ap.add_argument("--taus", default="1,2,3,5")
    a = ap.parse_args()

    trajs = cm.load_benign(a.scored, a.split, a.max_senders)
    if not trajs:
        raise SystemExit("궤적을 불러오지 못했습니다.")
    rates = [float(v) for v in a.rates.split(",")]
    a_lims = [float(v) for v in a.a_lims.split(",")]

    cm.log("E1': 노이즈 조건을 맞춘 탐지율 (전체 메시지 평균)")
    d1 = e1(trajs, rates, a.tau0, a.thr_scalar, a.thr_vector, a.window)
    cm.save(d1, "e1b_detection_matched.csv")
    s1 = agg(d1)
    cm.log("설정별:\n" + s1.round(3).to_string())
    base = float(s1.loc["none", "det_any"])
    cm.log(f"기준선(tau=0) 탐지율 {base:.3f} <- 논문 1 대조군(0.02 안팎)과 비슷해야 보간 오차가 없다는 뜻")
    rr = s1[s1.index.str.startswith("A2_rate")].copy()
    rr["rate"] = [float(i.replace("A2_rate", "")) for i in rr.index]
    over = rr[rr["det_any"] > float(s1.loc["A2_fixed", "det_any"]) + 0.05].sort_values("rate")
    cm.log(f"tau 변화가 드러나기 시작하는 |tau_dot|: {over['rate'].iloc[0] if len(over) else '없음(0.3까지)'}")

    cm.log("E3: 사건까지 남은 시간 H vs 유지 가능한 거짓말 크기")
    d3 = e3(trajs, [float(v) for v in a.horizons.split(",")], a_lims)
    cm.save(d3, "e3_lie_budget.csv")
    s3 = d3.groupby(["a_lim", "H"]).agg(n=("sender", "nunique"), v0=("v0", "median"),
                                        tau_max=("tau_max", "median"), lie_m=("lie_m", "median"),
                                        lie_theory=("lie_theory", "median"))
    s3["lie/theory"] = s3["lie_m"] / s3["lie_theory"]
    cm.log("거짓말 예산 (중앙값):\n" + s3.round(2).to_string())
    cm.log("해석: lie_m 이 논문 1의 GNSS 편향 규모(약 5 m)보다 작으면, "
           "그 시간 여유에서는 탐지 가능한 크기의 거짓말을 유지할 수 없다는 뜻")

    cm.log("E4: 예측 기반 공격자 (등가속도 외삽)")
    d4 = e4(trajs, [float(v) for v in a.taus.split(",")], a.thr_scalar, a.thr_vector, a.window)
    cm.save(d4, "e4_prediction_attacker.csv")
    g = d4.groupby(["tau", "segment"])
    s4 = g.agg(n_senders=("sender", "nunique"), n_msg=("n", "sum"),
               pred_resid=("pred_resid", "median"), r_vector=("r_vector", "median"))
    s4["det"] = g["hit"].sum() / g["n"].sum()
    cm.log("tau x 구간(brake: 급감속 ±3초):\n" + s4.round(3).to_string())


if __name__ == "__main__":
    main()
