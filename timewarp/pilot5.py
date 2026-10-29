#!/usr/bin/env python3
"""
pilot5.py: 예측 차수별 tau 최적점 곡선과 그림 생성.

tau를 촘촘히 훑으며 차수(1: 등속, 2: 원호, 3: 클로소이드)별 탐지율을 재고,
  (a) 탐지율 vs tau : 차수별 최적점과 역전 지점
  (b) 탐지율 vs 거짓말 크기(v*tau) : 공격자가 고를 수 있는 최선(하위 포락선)
을 그린다. 포락선이 공격자의 실효 한계이며, "위력과 은닉의 상충"을 보여주는 그림이다.

사용 예
  python pilot5.py --scored scored_tol5.parquet --max-senders 150 --taus 0.5,1,1.5,2,2.5,3,4,5,6
"""
import argparse

import numpy as np
import pandas as pd

import common as cm
import tau_model as tm
from pilot3 import add_noise
from pilot4 import predict_report

ORDER_LABEL = {1: "1차 (등속)", 2: "2차 (원호)", 3: "3차 (클로소이드)", 0: "oracle (실제 미래)"}


def sweep(trajs, taus, orders, brake_level, brake_win, thr_s, thr_v, window):
    rows = []
    for sid, d in trajs.items():
        tr = tm.Traj(**d)
        t = tr.t
        acc = tr.acc(t)
        brake = np.zeros(len(t), bool)
        for i in np.flatnonzero(acc < -brake_level):
            brake |= np.abs(t - t[i]) <= brake_win
        v_med = float(np.median(tr.spd(t)))
        for tau in taus:
            reps = {o: predict_report(tr, t, tau, o) for o in orders}
            ta, td, tdd = tm.tau_linear(t, tau, 0.0)
            reps[0] = tm.report(tr, t, ta, td, tdd, mode="b")
            for o, rep in reps.items():
                rep = add_noise(rep, tr, t)
                r_s, r_v = tm.residuals(rep)
                ok = rep["ok"] & np.isfinite(r_s)
                ok[0] = False
                if ok.sum() < 5:
                    continue
                ws = pd.Series(r_s).rolling(window, min_periods=1).mean().to_numpy()
                wv = pd.Series(r_v).rolling(window, min_periods=1).mean().to_numpy()
                hit = (ws > thr_s) | (wv > thr_v)
                for seg, mask in (("brake", ok & brake), ("other", ok & ~brake), ("all", ok)):
                    if mask.sum() >= 3:
                        rows.append(dict(sender=sid, tau=tau, order=o, segment=seg, v_med=v_med,
                                         n=int(mask.sum()), hit=int(hit[mask].sum())))
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby(["segment", "order", "tau"])
    s = g.agg(n_msg=("n", "sum"), n_senders=("sender", "nunique"), v_med=("v_med", "median"))
    s["det"] = g["hit"].sum() / g["n"].sum()
    s["lie_m"] = s.index.get_level_values("tau") * s["v_med"]
    return s.reset_index()


def make_figure(s, out_png, brake_level):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    for cand in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "AppleGothic"):
        if any(cand == f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False

    b = s[s["segment"] == "brake"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), dpi=300)
    colors = {0: "#888888", 1: "#555555", 2: "#d9822b", 3: "#1f5aa6"}
    for o in sorted(b["order"].unique()):
        g = b[b["order"] == o].sort_values("tau")
        ax[0].plot(g["tau"], g["det"], marker="o", ms=3, lw=1.3, color=colors.get(o),
                   ls="--" if o == 0 else "-", label=ORDER_LABEL.get(o, str(o)))
        ax[1].plot(g["lie_m"], g["det"], marker="o", ms=3, lw=1.3, color=colors.get(o),
                   ls="--" if o == 0 else "-")
    env = (b[b["order"] > 0].sort_values("lie_m").groupby("lie_m")["det"].min()
           .rolling(3, min_periods=1).min())
    ax[1].plot(env.index, env.values, color="#c0392b", lw=2, alpha=0.8, label="공격자의 최선(포락선)")

    ax[0].set_xlabel("예측 지평 τ (초)", fontsize=8)
    ax[0].set_ylabel(f"급감속(|a|>{brake_level:g} m/s²) 구간 탐지율", fontsize=7.5)
    ax[1].set_xlabel("공간 거짓말 크기 ≈ v·τ (m)", fontsize=8)
    for a_ in ax:
        a_.grid(lw=0.3, alpha=0.5)
        a_.tick_params(labelsize=7)
        a_.set_ylim(-0.02, 1.02)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    ax[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    ax[1].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png)
    cm.log(f"그림 저장: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="scored_tol5.parquet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-senders", type=int, default=150)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--thr-scalar", type=float, default=3.018)
    ap.add_argument("--thr-vector", type=float, default=6.372)
    ap.add_argument("--taus", default="0.5,1,1.5,2,2.5,3,3.5,4,5,6")
    ap.add_argument("--orders", default="1,2,3")
    ap.add_argument("--brake-level", type=float, default=4.0)
    ap.add_argument("--brake-win", type=float, default=3.0)
    a = ap.parse_args()

    trajs = cm.load_benign(a.scored, a.split, a.max_senders)
    if not trajs:
        raise SystemExit("궤적을 불러오지 못했습니다.")
    taus = [float(v) for v in a.taus.split(",")]
    orders = [int(v) for v in a.orders.split(",")]

    df = sweep(trajs, taus, orders, a.brake_level, a.brake_win, a.thr_scalar, a.thr_vector, a.window)
    cm.save(df, "e5_tau_sweep_raw.csv")
    s = summarize(df)
    cm.save(s, "e5_tau_sweep_summary.csv")

    for seg in ("brake", "other"):
        piv = s[s["segment"] == seg].pivot_table(index="tau", columns="order", values="det")
        piv.columns = [ORDER_LABEL.get(c, c) for c in piv.columns]
        cm.log(f"[{seg}] 탐지율 (행: tau, 열: 예측 차수):\n" + piv.round(3).to_string())

    b = s[(s["segment"] == "brake") & (s["order"] > 0)]
    cm.log("차수별 최적점 (급감속 구간 탐지율 최소):")
    for o, g in b.groupby("order"):
        i = g["det"].idxmin()
        cm.log(f"  {ORDER_LABEL[o]}: tau*={g.loc[i, 'tau']:.1f}s, 탐지율 {g.loc[i, 'det']:.3f}, "
               f"거짓말 {g.loc[i, 'lie_m']:.1f} m")
    for lim in (0.1, 0.2, 0.3):
        ok = b[b["det"] <= lim]
        best = f"{ok['lie_m'].max():.1f} m" if len(ok) else "없음"
        cm.log(f"  탐지율 {lim:.1f} 이하를 유지하며 만들 수 있는 최대 거짓말: {best}")

    make_figure(s, str(cm.RESULTS / "fig_tau_optimum.png"), a.brake_level)


if __name__ == "__main__":
    main()
