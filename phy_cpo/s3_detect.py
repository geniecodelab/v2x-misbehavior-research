#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s3_detect.py — 3단계: 검사기와 임계값.

    d_rep = |p_rep - q|
    P_hat = P_tx - L0 - 10 n log10(d_rep)
    rho   = P_rx - P_hat            ( = 10 n log10(d_rep/d) + X )
    s     = 같은 (tx, rx) 쌍 안에서 최근 K 개 rho 의 이동 평균
    판정   = |s| > tau        (양측)   /   s > tau_1s   (단측: 실제보다 멀다고 주장)

tau 는 **검증 분할의 정상 메시지**에서만 정한다 (|s| 의 99 백분위수).
K_eff 는 정상 s 의 표준편차 감소로 추정한다: K_eff = (sd(s_1)/sd(s_K))^2.

산출: work/detections_<tag>.parquet, out/03_threshold.md
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK, OUT = os.path.join(ROOT, "work"), os.path.join(ROOT, "out")
P_TX, L0 = 20.0, 47.8
KS = (1, 5, 10)
ALPHA = 0.01


def add_s(df, K):
    """(split, cond, tx, rx) 쌍 안에서 시간순 최근 K 개 rho 의 이동 평균."""
    df = df.sort_values(["split", "cond", "tx_id", "rx_id", "rcv_t"])
    g = df.groupby(["split", "cond", "tx_id", "rx_id"], sort=False)["rho"]
    return df.assign(**{f"s{K}": g.transform(lambda x: x.rolling(K, min_periods=K).mean())})


def main():
    meta = json.load(open(os.path.join(WORK, "params.json"), encoding="utf-8"))
    report = []
    for m in meta:
        tag, n = m["tag"], m["n"]
        df = pd.read_parquet(os.path.join(WORK, f"rssi_{tag}.parquet"))
        df["P_hat"] = P_TX - L0 - 10 * n * np.log10(df.d_rep)
        df["rho"] = df.P_rx - df.P_hat
        for K in KS:
            df = add_s(df, K)

        # ---- 임계값: 검증 분할의 정상 메시지에서만 ----
        val_b = df[(df.split == "Validation") & (df.label == 0)]
        info = dict(tag=tag, **{k: m[k] for k in ("n", "sigma_dB", "d_corr_m")})
        for K in KS:
            s = val_b[f"s{K}"].dropna()
            tau = float(np.quantile(np.abs(s), 1 - ALPHA))
            tau1 = float(np.quantile(s, 1 - ALPHA))
            df[f"det{K}"] = (df[f"s{K}"].abs() > tau)
            df[f"det{K}_1s"] = (df[f"s{K}"] > tau1)
            info[f"tau_K{K}"] = tau
            info[f"tau1s_K{K}"] = tau1
            info[f"sd_s_K{K}"] = float(s.std())
            # 시험 분할 정상 메시지에서의 실측 오탐률
            te = df[(df.split == "Test") & (df.label == 0)]
            info[f"fpr_val_K{K}"] = float((np.abs(s) > tau).mean())
            info[f"fpr_test_K{K}"] = float((te[f"s{K}"].abs() > tau).mean())
            info[f"fpr_test_1s_K{K}"] = float((te[f"s{K}"] > tau1).mean())
        for K in KS:
            info[f"K_eff_K{K}"] = (info["sd_s_K1"] / info[f"sd_s_K{K}"]) ** 2
        report.append(info)
        df.to_parquet(os.path.join(WORK, f"detections_{tag}.parquet"), index=False)
        print(f"{tag}: tau(K=5)={info['tau_K5']:.3f} dB, K_eff(K=5)={info['K_eff_K5']:.2f}, "
              f"test FPR={info['fpr_test_K5']:.4f}")

    with open(os.path.join(WORK, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT, "03_threshold.md"), "w", encoding="utf-8") as f:
        w = f.write
        w("# 3단계 — 검사기와 임계값\n\n")
        w("`rho = P_rx - (P_tx - L0 - 10 n log10(d_rep))`, `s` = 같은 (tx, rx) 쌍의 최근 K개 이동 평균.\n")
        w(f"임계값은 **검증 분할 정상 메시지**의 |s| {100*(1-ALPHA):.0f} 백분위수 (오탐률 {ALPHA:.0%}).\n\n")
        w("## 임계값과 오탐률\n\n")
        w("| 태그 | K | τ 양측 (dB) | τ 단측 (dB) | sd(s) (dB) | K_eff | 검증 FPR | 시험 FPR | 시험 FPR(단측) |\n")
        w("|---|---|---|---|---|---|---|---|---|\n")
        for r in report:
            for K in KS:
                w(f"| `{r['tag']}` | {K} | {r[f'tau_K{K}']:.3f} | {r[f'tau1s_K{K}']:.3f} | "
                  f"{r[f'sd_s_K{K}']:.3f} | {r[f'K_eff_K{K}']:.2f} | {r[f'fpr_val_K{K}']:.4f} | "
                  f"{r[f'fpr_test_K{K}']:.4f} | {r[f'fpr_test_1s_K{K}']:.4f} |\n")
        w("\n## K_eff 해석\n\n")
        base = next(r for r in report if r["tag"] == "n2_s4_dc25")
        wht = next((r for r in report if r["tag"] == "n2_s4_dc0"), None)
        w(f"기본 조합(`n2_s4_dc25`)에서 K = 5의 K_eff = **{base['K_eff_K5']:.2f}**, "
          f"K = 10의 K_eff = **{base['K_eff_K10']:.2f}** 이다.\n")
        w("섀도잉이 공간 상관을 가지므로 이동 평균으로 얻는 유효 표본 수가 K 보다 작다.\n")
        if wht:
            w(f"\n흰 잡음(`d_corr = 0`) 조합에서는 K = 5의 K_eff = **{wht['K_eff_K5']:.2f}**, "
              f"K = 10은 **{wht['K_eff_K10']:.2f}** 로 K 에 가깝다. ")
            w("지침서가 경고한 대로 **흰 잡음으로 만들면 이동 평균의 이득이 과대평가된다**는 것이 확인된다.\n")
        w("\n## 단측 판정\n\n")
        w("단측은 `s > τ₁`(보고 거리가 실제보다 멀다고 주장하는 방향)만 잡는다. CPO 의 b_r 부호는\n")
        w("주행에 따라 양·음 모두 나오므로 단측은 절반가량을 놓친다. 논문 식 (3)은 |ρ| 를 쓰므로\n")
        w("**양측이 기준**이며 z_α = 2.576 이다 (단측이면 2.326).\n")
    print("작성: out/03_threshold.md")


if __name__ == "__main__":
    main()
