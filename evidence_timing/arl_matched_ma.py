#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arl_matched_ma.py
메모 X-2 의 엄밀판.

sweep_alpha_and_ma.py 는 두 탐지기(이동 평균 / 비이동 평균)의 임계값을 각각
"메시지당 오탐률 1%" 로 맞췄다. 그런데 이동 평균 통계량은 서로 강하게 상관되어
경보가 뭉쳐 나므로, 같은 메시지당 오탐률에서도 평균 오탐 간격(ARL)은 다르다
(325개 대 154개). 식 (3)의 alpha 는 ARL 로 정의된 양이므로, 둘을 정말 같은
오탐 수준에 놓으려면 ARL 을 맞춰야 한다.

여기서는 비이동평균 탐지기의 임계값을 이동평균 탐지기와 같은 ARL 이 되도록
검증 분할에서 다시 정하고, 두 탐지기의 지연을 k* 와 다시 비교한다.
결론(고유 지연이 사실상 이동 평균에서 온다)이 이 조건에서도 성립하는지 본다.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v2x_evidence_timing as E


def build(df, smooth, jitter):
    trs = [E.Trajectory(g, smooth=smooth, max_dt_jitter=jitter) for _, g in df.groupby("sender")]
    return [tr for tr in trs if tr.ok]


def arl_of(stats, thr):
    n_msg = sum(int(np.sum(~np.isnan(x))) for x in stats)
    n_on = sum(int(np.sum((np.nan_to_num(x, nan=-1)[1:] >= thr) &
                          (np.nan_to_num(x, nan=-1)[:-1] < thr))) for x in stats)
    return float(n_msg / max(n_on, 1))


def thr_for_arl(stats, target, lo, hi, iters=60):
    """단조 증가하는 임계값에 대해 ARL(thr) = target 이 되는 thr 을 이분 탐색."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if arl_of(stats, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", default="results_sweep")
    p.add_argument("--taus", type=float, nargs="+", default=[1.0, 2.0, 2.5, 3.0, 4.0])
    p.add_argument("--order", type=int, default=1)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--fpr", type=float, default=0.01)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--brake-a", type=float, default=4.0)
    p.add_argument("--onset-a", type=float, default=0.5)
    p.add_argument("--horizon", type=float, default=15.0)
    p.add_argument("--smooth", type=float, default=0.5)
    p.add_argument("--max-dt-jitter", type=float, default=0.2)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    jit = a.max_dt_jitter if a.max_dt_jitter > 0 else None
    test = build(E.load_nextgen(a.test), a.smooth, jit)
    val = build(E.load_nextgen(a.val), a.smooth, jit)

    c = a.c
    rb_val = E.benign_residuals(val, c)
    allr = np.concatenate([r[~np.isnan(r)] for r in rb_val])
    sigma = float(np.std(allr))
    mu_B = float(np.mean(allr))
    ma_list = [E.moving_abs_mean(r, a.window) for r in rb_val]
    abs_list = [np.abs(r) for r in rb_val]

    eta_ma = float(np.nanquantile(np.concatenate(ma_list), 1 - a.fpr))
    eta_raw_fpr = float(np.nanquantile(np.abs(allr), 1 - a.fpr))
    arl_ma = arl_of(ma_list, eta_ma)
    # 이동평균 탐지기와 같은 ARL 이 되도록 비이동평균 임계값을 다시 정한다
    eta_raw_arl = thr_for_arl(abs_list, arl_ma, eta_raw_fpr, float(np.nanmax(np.abs(allr))))

    print(f"sigma={sigma:.4f}")
    print(f"이동평균      eta={eta_ma:.4f} ({eta_ma/sigma:.2f}σ)  ARL={arl_ma:.0f}")
    print(f"비이동(FPR1%) eta={eta_raw_fpr:.4f} ({eta_raw_fpr/sigma:.2f}σ)  ARL={arl_of(abs_list, eta_raw_fpr):.0f}")
    print(f"비이동(ARL맞춤) eta={eta_raw_arl:.4f} ({eta_raw_arl/sigma:.2f}σ)  ARL={arl_of(abs_list, eta_raw_arl):.0f}")

    thr_g = math.log(1.0 / a.alpha)
    rows = []
    for tau in a.taus:
        for si, tr in enumerate(test):
            t, r, D, Dp = E.messages(tr, tau, a.order, c, "path")
            if len(t) < a.window + 2:
                continue
            ma = E.moving_abs_mean(r, a.window)
            absr = np.abs(np.nan_to_num(r, nan=0.0))
            for e in E.braking_events(tr, a.brake_a, a.onset_a):
                if not ((t > e["t_on"]) & (t <= e["t_on"] + a.horizon)).sum() >= 2:
                    continue
                if bool(E.value_at(t, np.nan_to_num(ma, nan=-1), e["t_on"]) >= eta_ma):
                    continue
                sel_g = t > e["t_on"]
                S_g = E.genie_cusum(r[sel_g] - mu_B, np.nan_to_num(D[sel_g]), sigma)
                rows.append(dict(
                    tau=tau, sender=si,
                    genie=E.first_cross_time(t[sel_g], S_g, e["t_on"], a.horizon, thr_g),
                    d_ma=E.first_cross_time(t, np.nan_to_num(ma, nan=-1), e["t_on"], a.horizon, eta_ma),
                    d_raw_fpr=E.first_cross_time(t, absr, e["t_on"], a.horizon, eta_raw_fpr),
                    d_raw_arl=E.first_cross_time(t, absr, e["t_on"], a.horizon, eta_raw_arl)))
        print(f"  tau={tau} 완료")

    ev = pd.DataFrame(rows)
    print(f"\n사건 {len(ev)}건")
    out = dict(sigma=sigma, arl_target=arl_ma,
               eta=dict(moving_average=eta_ma, no_ma_fpr_matched=eta_raw_fpr,
                        no_ma_arl_matched=eta_raw_arl),
               median_k_star_genie=float(ev.genie.median()), by_detector={})
    print(f"k*(최적 탐지기) 중앙값 = {ev.genie.median():.2f}s\n")
    for label, col in (("이동평균(FPR 1%)", "d_ma"),
                       ("비이동평균(FPR 1% 맞춤)", "d_raw_fpr"),
                       ("비이동평균(ARL 맞춤)", "d_raw_arl")):
        ok = ev.genie.notna() & ev[col].notna()
        gap = (ev[col] - ev.genie)[ok]
        d = dict(n=int(ok.sum()), median_delay=float(ev[col][ok].median()),
                 mean_delay=float(ev[col][ok].mean()),
                 median_gap=float(gap.median()), mean_gap=float(gap.mean()),
                 share_ge=float((gap >= 0).mean()))
        out["by_detector"][label] = d
        print(f"  {label:<24} 지연중앙={d['median_delay']:.2f}s 평균={d['mean_delay']:.2f}s | "
              f"차이중앙={d['median_gap']:.2f}s 평균={d['mean_gap']:.2f}s | k*이상 {d['share_ge']*100:.1f}%")

    ev.to_csv(os.path.join(a.out, "events_arl_matched.csv"), index=False)
    with open(os.path.join(a.out, "arl_matched_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {a.out}/arl_matched_summary.json")


if __name__ == "__main__":
    main()
