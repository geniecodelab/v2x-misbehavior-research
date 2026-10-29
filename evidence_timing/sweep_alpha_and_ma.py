#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_alpha_and_ma.py
원고 v19 메모 X절의 재실행 두 가지.

X-1. alpha 스윕
    같은 급감속 사건에서 alpha 만 바꿔 식 (3)을 다시 풀고, ln(1/alpha) 에 대한
    k* 의 로그-로그 기울기를 이론값 +1/3 과 비교한다.
    (sigma^2 과 ln(1/alpha) 는 식 (4)에 곱으로 들어가므로, 이 지수가 +1/3 이면
     점이 셋뿐인 sigma 축의 약점을 메워 준다.)

X-2. 이동 평균 제거 비교의 임계값 재보정
    이동 평균 탐지기와 비이동평균 탐지기의 임계값을 각각 검증 분할에서 FPR 1% 로
    따로 정하고(= 같은 오탐 수준), 두 탐지기의 지연을 k* 와 비교한다.
    덤으로 검정 통계량의 분포를 적어 eta_ma 가 sigma 의 몇 배인지 설명한다.

실행 (v2x_evidence_timing.py 와 같은 폴더에서):
    python sweep_alpha_and_ma.py --test <Test 폴더> --val <Validation 폴더> \
                                 --out results_sweep
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", default="results_sweep")
    p.add_argument("--taus", type=float, nargs="+", default=[1.0, 2.0, 2.5, 3.0, 4.0])
    p.add_argument("--order", type=int, default=1)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--alphas", type=float, nargs="+", default=[1e-1, 1e-2, 1e-3, 1e-4])
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
    print(f"test {len(test)}명, val {len(val)}명")

    c = a.c
    # ---- 검증 분할에서 sigma 와 두 탐지기의 임계값을 각각 FPR 1% 로 따로 정한다 ----
    rb_val = E.benign_residuals(val, c)
    allr = np.concatenate([r[~np.isnan(r)] for r in rb_val])
    sigma = float(np.std(allr))
    mu_B = float(np.mean(allr))
    ma_list = [E.moving_abs_mean(r, a.window) for r in rb_val]
    ma_all = np.concatenate(ma_list)
    eta_ma = float(np.nanquantile(ma_all, 1 - a.fpr))
    abs_list = [np.abs(r) for r in rb_val]
    eta_raw = float(np.nanquantile(np.abs(allr), 1 - a.fpr))

    def arl(stats, thr):
        n_msg = sum(int(np.sum(~np.isnan(x))) for x in stats)
        n_on = sum(int(np.sum((np.nan_to_num(x, nan=-1)[1:] >= thr) &
                              (np.nan_to_num(x, nan=-1)[:-1] < thr))) for x in stats)
        return float(n_msg / max(n_on, 1))

    # 시험 분할에서의 실측 오탐률 (두 탐지기가 같은 오탐 수준인지 확인)
    rb_test = E.benign_residuals(test, c)
    ma_t = np.concatenate([E.moving_abs_mean(r, a.window) for r in rb_test])
    raw_t = np.abs(np.concatenate([r[~np.isnan(r)] for r in rb_test]))

    stat = dict(
        sigma=sigma, mean_benign_residual=mu_B,
        detector_moving_average=dict(
            statistic="최근 5개 메시지의 |r| 이동 평균, 단측",
            eta=eta_ma, eta_over_sigma=eta_ma / sigma,
            val_fpr=float(np.nanmean(ma_all >= eta_ma)),
            test_fpr=float(np.nanmean(ma_t >= eta_ma)),
            ARL_messages=arl(ma_list, eta_ma)),
        detector_no_moving_average=dict(
            statistic="|r| 그 자체, 단측",
            eta=eta_raw, eta_over_sigma=eta_raw / sigma,
            val_fpr=float(np.nanmean(np.abs(allr) >= eta_raw)),
            test_fpr=float(np.nanmean(raw_t >= eta_raw)),
            ARL_messages=arl(abs_list, eta_raw)),
        benign_abs_r=dict(
            mean=float(np.mean(np.abs(allr))), median=float(np.median(np.abs(allr))),
            p99=float(np.quantile(np.abs(allr), .99)),
            mean_over_sigma=float(np.mean(np.abs(allr)) / sigma),
            kurtosis=float(pd.Series(allr).kurtosis())),
        benign_ma=dict(mean=float(np.nanmean(ma_all)), p99=eta_ma,
                       sd=float(np.nanstd(ma_all))),
    )
    print(f"sigma={sigma:.3f}  eta_ma={eta_ma:.3f}({eta_ma/sigma:.2f}σ)  "
          f"eta_raw={eta_raw:.3f}({eta_raw/sigma:.2f}σ)")
    print(f"  ARL: 이동평균 {stat['detector_moving_average']['ARL_messages']:.0f}개, "
          f"비이동평균 {stat['detector_no_moving_average']['ARL_messages']:.0f}개 메시지")

    # ---- 사건별로 alpha 마다 k* 와 최적 탐지기 지연을 다시 계산 ----
    rows = []
    for tau in a.taus:
        for si, tr in enumerate(test):
            t, r, D, Dp = E.messages(tr, tau, a.order, c, "path")
            if len(t) < a.window + 2:
                continue
            ma = E.moving_abs_mean(r, a.window)
            info = D ** 2 / (2 * sigma ** 2)
            absr = np.abs(np.nan_to_num(r, nan=0.0))
            for e in E.braking_events(tr, a.brake_a, a.onset_a):
                sel = (t > e["t_on"]) & (t <= e["t_on"] + a.horizon)
                if sel.sum() < 2:
                    continue
                pre_alarm = bool(E.value_at(t, np.nan_to_num(ma, nan=-1), e["t_on"]) >= eta_ma)
                if pre_alarm:
                    continue                      # 본 분석과 같은 기준으로 제외
                tm = t[sel]
                cum = np.nancumsum(info[sel])
                sel_g = t > e["t_on"]
                S_g = E.genie_cusum(r[sel_g] - mu_B, np.nan_to_num(D[sel_g]), sigma)
                # 두 탐지기의 지연 (임계값은 각각 FPR 1% 로 따로 보정한 것)
                d_ma = E.first_cross_time(t, np.nan_to_num(ma, nan=-1), e["t_on"], a.horizon, eta_ma)
                d_raw = E.first_cross_time(t, absr, e["t_on"], a.horizon, eta_raw)
                row = dict(tau=tau, sender=si, t_on=e["t_on"], jerk=e["jerk"],
                           delay_detector=d_ma, delay_no_ma=d_raw)
                for al in a.alphas:
                    thr = math.log(1.0 / al)
                    row[f"k_meas_{al:g}"] = E.first_cross_time(tm, cum, e["t_on"] - 1e-9, a.horizon, thr)
                    row[f"genie_{al:g}"] = E.first_cross_time(t[sel_g], S_g, e["t_on"], a.horizon, thr)
                rows.append(row)
        print(f"  tau={tau} 완료")

    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(a.out, "events_alpha_sweep.csv"), index=False)
    print(f"사건 {len(ev)}건 (pre_alarm 제외)")

    # ---- X-1 결과: ln(1/alpha) 에 대한 기울기 ----
    ln_inv = np.array([math.log(1.0 / x) for x in a.alphas])
    med_k = np.array([ev[f"k_meas_{x:g}"].median() for x in a.alphas])
    med_g = np.array([ev[f"genie_{x:g}"].median() for x in a.alphas])
    n_det = [int(ev[f"genie_{x:g}"].notna().sum()) for x in a.alphas]

    def slope(x, y):
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0]) if ok.sum() >= 2 else np.nan

    out = dict(
        n_events=int(len(ev)), sigma=sigma, detector_stats=stat,
        alpha_sweep=dict(
            alphas=list(a.alphas), ln_inv_alpha=ln_inv.tolist(),
            median_k_star=med_k.tolist(), median_delay_genie=med_g.tolist(),
            n_detected=n_det,
            slope_k_star_vs_ln_inv_alpha=slope(ln_inv, med_k),
            slope_genie_vs_ln_inv_alpha=slope(ln_inv, med_g),
            theory=1.0 / 3.0),
    )
    print("\n== X-1. alpha 스윕 ==")
    for al, li, k, g, nd in zip(a.alphas, ln_inv, med_k, med_g, n_det):
        print(f"  alpha={al:<8g} ln(1/a)={li:5.2f}  k*중앙값={k:.3f}s  최적탐지기={g:.3f}s  탐지={nd}건")
    print(f"  기울기: k* {out['alpha_sweep']['slope_k_star_vs_ln_inv_alpha']:.3f}, "
          f"최적탐지기 {out['alpha_sweep']['slope_genie_vs_ln_inv_alpha']:.3f}  (이론 +1/3 = 0.333)")

    # ---- X-2 결과: 두 탐지기의 지연을 k* 와 비교 ----
    base = 0.01
    kg = ev[f"genie_{base:g}"]
    res = {}
    for name, col in (("이동평균", "delay_detector"), ("비이동평균", "delay_no_ma")):
        ok = kg.notna() & ev[col].notna()
        gap = (ev[col] - kg)[ok]
        res[name] = dict(n=int(ok.sum()),
                         median_delay=float(ev[col][ok].median()),
                         mean_delay=float(ev[col][ok].mean()),
                         median_gap=float(gap.median()), mean_gap=float(gap.mean()),
                         share_ge=float((gap >= 0).mean()))
    out["ma_recalibrated"] = dict(alpha=base, median_k_star_genie=float(kg.median()), by_detector=res)
    print("\n== X-2. 임계값을 각각 FPR 1% 로 재보정한 비교 (alpha=0.01) ==")
    print(f"  k*(최적 탐지기) 중앙값 = {kg.median():.2f}s")
    for name, d in res.items():
        print(f"  {name:<8} n={d['n']:3d} 지연중앙값={d['median_delay']:.2f}s "
              f"차이중앙값={d['median_gap']:.2f}s 평균차이={d['mean_gap']:.2f}s "
              f"(k*이상 {d['share_ge']*100:.1f}%)")

    with open(os.path.join(a.out, "sweep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n저장: {a.out}/sweep_summary.json, events_alpha_sweep.csv")


if __name__ == "__main__":
    main()
