#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2_rssi.py — 2단계: 수신 신호 세기 합성.

    P_rx = P_tx - L0 - 10 n log10(d) + X      (d = 실제 거리. 보고 거리를 쓰지 않는다)

섀도잉 X 는 (송신, 수신) 쌍마다 1차 자기회귀로 만든다.
    X[k] = a X[k-1] + sqrt(1-a^2) sigma e[k],  a = exp(-delta_dist[k]/d_corr)
    delta_dist[k] = 연속 두 메시지 사이 '상대 위치 벡터'의 변화량 |Δ(p-q)|
쌍의 첫 메시지는 X ~ N(0, sigma^2).

조건 A 와 B 는 실제 위치가 같으므로 **같은 X 실현값**을 공유한다(공정 비교).
따라서 X 는 (tx, rx, msg) 단위로 한 번만 만들어 두 조건에 함께 쓴다.

산출: work/rssi_<tag>.parquet, out/02_params.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK, OUT = os.path.join(ROOT, "work"), os.path.join(ROOT, "out")

P_TX, L0, RX_SENS = 20.0, 47.8, -89.0


def tag_of(n, sigma, d_corr):
    return f"n{n:g}_s{sigma:g}_dc{d_corr:g}"


def shadowing(df, sigma, d_corr, seed):
    """(tx, rx) 쌍 안에서 시간순으로 AR(1) 섀도잉을 만든다. 실제 위치 기준."""
    rng = np.random.default_rng(seed)
    # 조건별 중복을 없애고 (tx, rx, msg) 단위의 고유 기하만 남긴다
    geo = (df[df.cond != "B"]                       # benign + A 가 전체 (tx,rx,msg) 를 덮는다
           .drop_duplicates(["split", "tx_id", "rx_id", "msg_id"])
           .sort_values(["split", "tx_id", "rx_id", "rcv_t"])
           .reset_index(drop=True))
    relx = geo.p_true_x - geo.qx_true
    rely = geo.p_true_y - geo.qy_true
    pair = geo.split + "|" + geo.tx_id + "|" + geo.rx_id
    newpair = pair.ne(pair.shift()).to_numpy()
    step = np.hypot(relx.diff().to_numpy(), rely.diff().to_numpy())
    step[newpair] = np.inf                           # 쌍의 첫 메시지 -> a = 0
    a = np.exp(-step / d_corr) if d_corr > 0 else np.zeros(len(geo))
    a = np.nan_to_num(a, nan=0.0, posinf=0.0)
    e = rng.standard_normal(len(geo))
    X = np.empty(len(geo))
    prev = 0.0
    for i in range(len(geo)):
        if newpair[i]:
            X[i] = sigma * e[i]                      # 첫 메시지: N(0, sigma^2)
        else:
            X[i] = a[i] * prev + np.sqrt(max(1 - a[i] ** 2, 0.0)) * sigma * e[i]
        prev = X[i]
    geo = geo.assign(X=X, ar_a=a)
    return geo[["split", "tx_id", "rx_id", "msg_id", "X", "ar_a"]]


def run(df, n, sigma, d_corr, seed):
    sh = shadowing(df, sigma, d_corr, seed)
    r = df.merge(sh, on=["split", "tx_id", "rx_id", "msg_id"], how="left")
    r["P_rx"] = P_TX - L0 - 10 * n * np.log10(r.d) + r.X
    keep = r.P_rx >= RX_SENS
    drop_rate = float((~keep).mean())
    r = r[keep].copy()
    return r, drop_rate, float(sh.ar_a.replace(0, np.nan).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--ns", type=float, nargs="+", default=[2.0, 2.7])
    p.add_argument("--sigmas", type=float, nargs="+", default=[2.0, 4.0, 6.0])
    p.add_argument("--dcorrs", type=float, nargs="+", default=[25.0, 0.0])
    a = p.parse_args()

    df = pd.read_parquet(os.path.join(WORK, "messages.parquet"))
    combos, meta = [], []
    for n, sg, dc in itertools.product(a.ns, a.sigmas, a.dcorrs):
        # 전체 격자는 낭비이므로 기본값 축을 중심으로만 돈다
        if not (n == 2.0 and dc == 25.0) and not (sg == 4.0 and dc == 25.0) and not (n == 2.0 and sg == 4.0):
            continue
        combos.append((n, sg, dc))
    for n, sg, dc in combos:
        r, drop, mean_a = run(df, n, sg, dc, a.seed)
        t = tag_of(n, sg, dc)
        r.to_parquet(os.path.join(WORK, f"rssi_{t}.parquet"), index=False)
        meta.append(dict(tag=t, n=n, sigma_dB=sg, d_corr_m=dc, seed=a.seed,
                         rows=int(len(r)), drop_rate_below_sens=drop, mean_ar_coeff=mean_a))
        print(f"{t}: {len(r):,}행, 감도 미달 제외 {drop*100:.2f}%, 평균 AR 계수 {mean_a:.3f}")

    with open(os.path.join(WORK, "params.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "02_params.md"), "w", encoding="utf-8") as f:
        w = f.write
        w("# 2단계 — 신호 세기 합성\n\n")
        w(f"`P_rx = P_tx - L0 - 10 n log10(d) + X`, P_tx = {P_TX} dBm, L0 = {L0} dB, "
          f"수신 감도 {RX_SENS} dBm, seed = {a.seed}\n\n")
        w("**실제 거리 d 로 만든다.** 보고 거리는 3단계의 탐지기만 쓴다.\n\n")
        w("섀도잉은 (송신, 수신) 쌍 안에서 AR(1)로 생성한다. `a = exp(-Δ/d_corr)`이고\n")
        w("Δ 는 연속 두 메시지 사이 **상대 위치 벡터 `p_true - q` 의 변화량**이다\n")
        w("(두 차량이 같이 움직여 상대 기하가 그대로면 섀도잉도 그대로 유지된다).\n")
        w("조건 A 와 B 는 실제 위치가 같으므로 **같은 X 실현값**을 공유해 공정하게 비교된다.\n\n")
        w("## 파라미터 조합\n\n| 태그 | n | σ (dB) | d_corr (m) | 행 수 | 감도 미달 제외 | 평균 AR 계수 |\n")
        w("|---|---|---|---|---|---|---|\n")
        for m in meta:
            w(f"| `{m['tag']}` | {m['n']} | {m['sigma_dB']} | {m['d_corr_m']} | {m['rows']:,} | "
              f"{m['drop_rate_below_sens']*100:.2f}% | {m['mean_ar_coeff']:.3f} |\n")
        w("\n기본 조합은 `n2_s4_dc25` (n = 2.0, σ = 4 dB, d_corr = 25 m)이다.\n")
        w("`dc0` 은 흰 잡음 비교용이며, 이동 평균의 이득이 과대평가되는지 보는 데 쓴다.\n")
    print("작성: out/02_params.md")


if __name__ == "__main__":
    main()
