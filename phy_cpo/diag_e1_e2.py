#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_e1_e2.py — E1·E2 불합격 원인 진단.

의심 순서(지침서 8.3): 섀도잉 상관 → 이동 평균 K_eff → 양측/단측 → b 추정
여기에 더해 근사식 자체의 유효 범위(|b| << d)를 함께 본다.
"""
import json, math, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "work")
TAG, K, LN10 = "n2_s4_dc25", 5, math.log(10)
n, sigma = 2.0, 4.0

df = pd.read_parquet(os.path.join(WORK, f"detections_{TAG}.parquet"))
te = df[df.split == "Test"].copy()
atk = te[te.label == 1]

print("=" * 78)
print("1) 근사식이 성립할 조건: |b| << d 인가")
print("=" * 78)
for c in ("A", "B"):
    g = atk[atk.cond == c]
    print(f"  조건 {c}: |b| 중앙값 {g.b_abs.median():7.1f} m, d 중앙값 {g.d.median():6.1f} m")
    print(f"           |b|/d  중앙값 {(g.b_abs/g.d).median():.2f},  |b|/d > 0.5 비율 {(g.b_abs/g.d > .5).mean():.2f}")
    print(f"           |b_t|/d 중앙값 {(g.b_t_abs/g.d).median():.2f}, |b_t|>d 비율 {(g.b_t_abs > g.d).mean():.2f}")

print()
print("=" * 78)
print("2) E1 을 조건별·|b|/d 구간별로 다시 본다")
print("=" * 78)
obs = 10 * n * np.log10(atk.d_rep / atk.d)
lin = (10 * n / LN10) * (atk.b_r / atk.d)
quad = (10 * n / LN10) * (atk.b_t_abs ** 2 / (2 * atk.d ** 2))
app = lin + quad
a2 = atk.assign(obs=obs, app=app, lin=lin, quad=quad, err=(obs - app).abs(),
                bd=atk.b_abs / atk.d)
for c in ("A", "B"):
    g = a2[a2.cond == c]
    r = np.corrcoef(g.obs, g.app)[0, 1]
    print(f"  조건 {c}: r={r:.4f}  MAE={g.err.mean():7.3f} dB")
    gb = g.assign(bin=pd.cut(g.bd, [0, .1, .25, .5, 1, 2, np.inf]))
    t = gb.groupby("bin", observed=True).agg(n=("err", "size"), mae=("err", "mean"),
                                             r=("obs", lambda s: np.nan))
    for b, row in t.iterrows():
        sub = gb[gb.bin == b]
        rr = np.corrcoef(sub.obs, sub.app)[0, 1] if len(sub) > 5 else np.nan
        print(f"      |b|/d {str(b):<12} n={int(row['n']):6d}  MAE={row['mae']:9.3f} dB  r={rr:.4f}")

print()
print("  |b|/d <= 0.25 로 제한했을 때 (근사식이 유효한 영역):")
g = a2[a2.bd <= 0.25]
print(f"      전체   n={len(g):6d}  r={np.corrcoef(g.obs, g.app)[0,1]:.5f}  MAE={g.err.mean():.4f} dB")
for c in ("A", "B"):
    h = g[g.cond == c]
    if len(h) > 5:
        print(f"      조건{c} n={len(h):6d}  r={np.corrcoef(h.obs,h.app)[0,1]:.5f}  MAE={h.err.mean():.4f} dB")

print()
print("=" * 78)
print("3) 접선 성분 2차 항이 조건 B 에서 얼마나 큰가 (E2 곡선이 안 겹치는 원인 후보)")
print("=" * 78)
for c in ("A", "B"):
    g = a2[a2.cond == c]
    print(f"  조건 {c}: |1차항| 중앙값 {g.lin.abs().median():6.2f} dB, "
          f"2차항 중앙값 {g.quad.median():6.2f} dB, "
          f"2차항/|1차항| 중앙값 {(g.quad/g.lin.abs().replace(0,np.nan)).median():.2f}")
    print(f"           2차항이 임계값(7.58dB)을 혼자 넘는 비율 {(g.quad > 7.584).mean():.3f}")

print()
print("=" * 78)
print("4) 이동 평균이 곡선을 흐리는가 (K=1 과 비교)")
print("=" * 78)
rb = np.concatenate([[0], np.logspace(-2, 0.7, 14)])


def curve(g, kk):
    h = g.dropna(subset=[f"s{kk}"]).copy()
    h["bin"] = pd.cut(h.ratio, rb)
    t = h.groupby("bin", observed=True).agg(rate=(f"det{kk}", "mean"), n=(f"det{kk}", "size"))
    return t[t.n >= 30]


for kk in (1, 5):
    cA, cB = curve(atk[atk.cond == "A"], kk), curve(atk[atk.cond == "B"], kk)
    com = cA.index.intersection(cB.index)
    gap = np.abs(cA.loc[com, "rate"] - cB.loc[com, "rate"]).mean()
    print(f"  K={kk}: |b_r|/d 곡선 간격 {gap:.3f} (공통 구간 {len(com)}개)")

print()
print("=" * 78)
print("5) |b_t| 2차 항이 작은 행만 골라 다시 본다 (근사식 유효 영역에서 곡선이 겹치는가)")
print("=" * 78)
for cut in (1.0, 0.5, 0.25):
    sub = atk[atk.b_t_abs <= cut * atk.d]
    cA, cB = curve(sub[sub.cond == "A"], 5), curve(sub[sub.cond == "B"], 5)
    com = cA.index.intersection(cB.index)
    if len(com):
        gap = np.abs(cA.loc[com, "rate"] - cB.loc[com, "rate"]).mean()
        print(f"  |b_t| <= {cut}·d : 곡선 간격 {gap:.3f} (공통 {len(com)}개, A {len(sub[sub.cond=='A'])}행, B {len(sub[sub.cond=='B'])}행)")
    else:
        print(f"  |b_t| <= {cut}·d : 공통 구간 없음 (A {len(sub[sub.cond=='A'])}행, B {len(sub[sub.cond=='B'])}행)")

print()
print("=" * 78)
print("6) 정리 변수를 바꿔 본다 — 무엇이 탐지율을 가장 잘 정리하는가")
print("=" * 78)
a3 = atk.dropna(subset=[f"s{K}"]).copy()
# 이론상 잔차의 기대값 (섀도잉 제외) 을 sigma 로 정규화한 것
a3["mu_exact"] = 10 * n * np.log10(a3.d_rep / a3.d)
a3["mu_lin"] = (10 * n / LN10) * (a3.b_r / a3.d)
for var, label, bins in (("ratio", "|b_r|/d", rb),
                         ("b_abs", "|b|", [0, 40, 60, 80, 100, 200, 250, 300, 400]),
                         ("mu_lin_abs", "|1차항| [dB]", np.linspace(0, 30, 16)),
                         ("mu_exact_abs", "|정확한 잔차 평균| [dB]", np.linspace(0, 30, 16))):
    if var.endswith("_abs") and var not in a3:
        a3[var] = a3[var[:-4]].abs()
    cA, cB = None, None
    h = a3.copy(); h["bin"] = pd.cut(h[var], bins)
    t = h.groupby(["cond", "bin"], observed=True).agg(rate=(f"det{K}", "mean"), n=(f"det{K}", "size"))
    t = t[t.n >= 30]
    try:
        A_ = t.loc["A"]; B_ = t.loc["B"]
        com = A_.index.intersection(B_.index)
        gap = np.abs(A_.loc[com, "rate"] - B_.loc[com, "rate"]).mean() if len(com) else np.nan
        print(f"  {label:<26} 공통 구간 {len(com):2d}개, 곡선 간격 {gap:.3f}")
    except KeyError:
        print(f"  {label:<26} 공통 구간 없음")
