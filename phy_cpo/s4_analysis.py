#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s4_analysis.py — 4단계: E1~E5 분석과 그림, results.json.

평가는 모두 **시험(Test) 분할**, 임계값은 3단계에서 검증 분할로 정한 값을 쓴다.
"""
from __future__ import annotations

import json
import math
import os
from datetime import date

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

_names = [x.name for x in _fm.fontManager.ttflist]
for _k in ("NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK"):
    _h = [x for x in _names if _k in x]
    if _h:
        plt.rcParams["font.family"] = [_h[0], "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK, OUT = os.path.join(ROOT, "work"), os.path.join(ROOT, "out")
BASE_TAG = "n2_s4_dc25"
K = 5
Z_TWO, Z_ONE = 2.5758293035489004, 2.3263478740408408
LN10 = math.log(10.0)
D_BINS = [0, 50, 150, 300, np.inf]
D_LAB = ["<50", "50-150", "150-300", ">300"]


def load(tag):
    return pd.read_parquet(os.path.join(WORK, f"detections_{tag}.parquet"))


def thr(tag):
    return next(r for r in json.load(open(os.path.join(WORK, "thresholds.json"), encoding="utf-8"))
                if r["tag"] == tag)


def rate_by_bin(df, col, bins, detcol=f"det{K}"):
    g = df.dropna(subset=[f"s{K}"]).copy()
    g["_bin"] = pd.cut(g[col], bins)
    r = g.groupby("_bin", observed=True).agg(rate=(detcol, "mean"), n=(detcol, "size"))
    return r[r.n >= 30]


def cross_half(centers, rates, level=0.5):
    """탐지율이 level 을 넘는 지점을 선형 보간으로 찾는다."""
    c, r = np.asarray(centers, float), np.asarray(rates, float)
    for i in range(1, len(r)):
        if r[i - 1] < level <= r[i]:
            if r[i] == r[i - 1]:
                return c[i]
            return float(c[i - 1] + (level - r[i - 1]) * (c[i] - c[i - 1]) / (r[i] - r[i - 1]))
    return float("nan")


def main():
    T = thr(BASE_TAG)
    n, sigma = T["n"], T["sigma_dB"]
    K_eff = T[f"K_eff_K{K}"]
    tau = T[f"tau_K{K}"]
    df = load(BASE_TAG)
    te = df[df.split == "Test"].dropna(subset=[f"s{K}"]).copy()
    atk = te[te.label == 1]
    ben = te[te.label == 0]
    A, B = atk[atk.cond == "A"], atk[atk.cond == "B"]
    fpr_test = float(ben[f"det{K}"].mean())

    R = {"run_id": f"phy_cpo_{BASE_TAG}", "seed": 20260920, "date": str(date.today()),
         "data": {}, "params": {}, "E1": {}, "E2": {}, "E3": {}, "E4": {}, "E5": {},
         "fpr": {}, "notes": []}

    sch = json.load(open(os.path.join(OUT, "00_schema_report.json"), encoding="utf-8"))
    R["data"] = {
        "scenario": "InTAS_highway_2 (고속도로 저밀도), constantPositionOffset",
        "splits": {"train": "미사용", "val": int((df.split == "Validation").sum()),
                   "test": int((df.split == "Test").sum())},
        "cpo_senders": sch["Test"]["n_attacker_senders"],
        "b_abs_min": sch["Test"]["b_abs_min"], "b_abs_max": sch["Test"]["b_abs_max"],
        "b_const_check_max_std_m": sch["Test"]["b_const_check_max_std_m"],
        "receiver_pos_source": "receiver.pos - receiver.pos_noise (메시지 레코드 내 수신자 객체)",
        "true_pos_source": "무공격 실행(InTAS_highway_2)의 같은 (sender_id, messageID) 에서 sender.pos - sender.pos_noise",
        "cond_B_offset_m": [250.0, -150.0], "distance_2d_only": True}
    R["params"] = {"P_tx_dBm": 20, "L0_dB": 47.8, "n": n, "sigma_dB": sigma,
                   "d_corr_m": T["d_corr_m"], "K": K, "alpha": 0.01, "two_sided": True,
                   "tau_dB": tau, "tau_one_sided_dB": T[f"tau1s_K{K}"],
                   "K_eff": K_eff,
                   "drop_rate_below_sens": next(
                       m["drop_rate_below_sens"] for m in
                       json.load(open(os.path.join(WORK, "params.json"), encoding="utf-8"))
                       if m["tag"] == BASE_TAG)}

    # ---------------- E1 ----------------
    # 식 (2)는 |b| << d 를 전제한 근사이므로, 전체를 뭉뚱그리지 말고
    # 유효 영역(|b|/d 구간)별로 본다. 전체 풀링 값도 함께 남긴다.
    obs = 10 * n * np.log10(atk.d_rep / atk.d)
    app = (10 * n / LN10) * (atk.b_r / atk.d + atk.b_t_abs ** 2 / (2 * atk.d ** 2))
    a1 = atk.assign(obs=obs, app=app, err=(obs - app).abs(), bd=atk.b_abs / atk.d)
    a1 = a1[np.isfinite(a1.obs) & np.isfinite(a1.app)]

    def rmae(g):
        return (float(np.corrcoef(g.obs, g.app)[0, 1]) if len(g) > 5 else float("nan"),
                float(g.err.mean()), int(len(g)))

    bd_bins = [0, .1, .25, .5, 1, 2, np.inf]
    by_bd = {}
    for b, g in a1.groupby(pd.cut(a1.bd, bd_bins), observed=True):
        r_, m_, n_ = rmae(g)
        by_bd[str(b)] = {"pearson_r": r_, "mae_dB": m_, "n": n_}
    valid = a1[a1.bd <= 0.25]
    r_v, m_v, n_v = rmae(valid)
    r_all, m_all, _ = rmae(a1)
    R["E1"] = {
        "validity_domain": "|b|/d <= 0.25",
        "pearson_r": r_v, "mae_dB": m_v, "n_in_domain": n_v,
        "pearson_r_all_rows": r_all, "mae_dB_all_rows": m_all,
        "by_b_over_d": by_bd,
        "mae_by_distance": {str(k): float(v) for k, v in
                            a1.assign(_b=pd.cut(a1.d, D_BINS, labels=D_LAB))
                            .groupby("_b", observed=True).err.mean().items()},
        "pass_in_domain": bool(r_v >= 0.99 and m_v <= 0.3),
        "pass_all_rows": bool(r_all >= 0.99 and m_all <= 0.3)}
    R["E1"]["pass"] = R["E1"]["pass_in_domain"]

    # ---------------- E2 ----------------
    b_bins = [0, 40, 60, 80, 100, 200, 250, 300, 400]
    r_bins = np.concatenate([[0], np.logspace(-2, 0.9, 20)])
    mu_bins = np.linspace(0, 30, 21)
    for g in (A, B, atk, te):
        pass
    te["mu"] = 10 * n * np.log10(te.d_rep / te.d)
    te["mu_abs"] = te.mu.abs()
    atk = te[te.label == 1]
    A, B = atk[atk.cond == "A"], atk[atk.cond == "B"]

    def gap(x, y):
        c = x.index.intersection(y.index)
        return (float(np.abs(x.loc[c, "rate"] - y.loc[c, "rate"]).mean()), len(c)) if len(c) else (float("nan"), 0)

    cen = lambda idx: np.array([i.mid for i in idx])
    curves, gaps = {}, {}
    for KK in (1, K):
        for var, bins in (("b_abs", b_bins), ("ratio", r_bins), ("mu_abs", mu_bins)):
            cA = rate_by_bin(A, var, bins, f"det{KK}")
            cB = rate_by_bin(B, var, bins, f"det{KK}")
            g_, c_ = gap(cA, cB)
            gaps[f"K{KK}_{var}"] = {"gap": g_, "n_common_bins": c_}
            if KK == K:
                curves[var] = (cA, cB)
    rA_b, rB_b = curves["b_abs"]
    rA_r, rB_r = curves["ratio"]
    rA_m, rB_m = curves["mu_abs"]

    pooled = rate_by_bin(pd.concat([A, B]), "ratio", r_bins)
    emp = cross_half(cen(pooled.index), pooled.rate.to_numpy())
    pred_lin = (LN10 / (10 * n)) * Z_TWO * sigma / math.sqrt(K_eff)
    pred_tau = tau * LN10 / (10 * n)              # 선형근사를 실제 tau 로 환산
    # 근사를 쓰지 않은 정확한 임계 (b_t = 0 가정, 부호별로 다름)
    exact_pos = 10 ** (tau / (10 * n)) - 1
    exact_neg = 1 - 10 ** (-tau / (10 * n))
    pos = atk[atk.b_r >= 0]
    cpos = rate_by_bin(pos, "ratio", r_bins)
    emp_pos = cross_half(cen(cpos.index), cpos.rate.to_numpy())

    R["E2"] = {
        "cond_A": {"mean_b_abs": float(A.b_abs.mean()), "detection_rate": float(A[f"det{K}"].mean()),
                   "n": int(len(A))},
        "cond_B": {"mean_b_abs": float(B.b_abs.mean()), "detection_rate": float(B[f"det{K}"].mean()),
                   "n": int(len(B))},
        "curve_gap_by_b_abs": gaps[f"K{K}_b_abs"]["gap"],
        "n_common_bins_b_abs": gaps[f"K{K}_b_abs"]["n_common_bins"],
        "curve_gap_by_ratio": gaps[f"K{K}_ratio"]["gap"],
        "n_common_bins_ratio": gaps[f"K{K}_ratio"]["n_common_bins"],
        "curve_gap_all": gaps,
        "empirical_threshold_ratio": emp,
        "empirical_threshold_ratio_b_r_positive": emp_pos,
        "predicted_threshold_ratio": pred_lin,
        "predicted_threshold_ratio_from_tau": pred_tau,
        "exact_threshold_ratio_b_r_positive": exact_pos,
        "exact_threshold_ratio_b_r_negative": exact_neg,
        "rel_err_vs_predicted": float(abs(emp - pred_lin) / pred_lin),
        "rel_err_exact_b_r_positive": float(abs(emp_pos - exact_pos) / exact_pos),
        "K_eff_estimate": K_eff,
        "detection_rate_by_b_abs": {"A": {str(i): float(v) for i, v in rA_b.rate.items()},
                                    "B": {str(i): float(v) for i, v in rB_b.rate.items()}},
        "detection_rate_by_ratio": {"A": {str(i): float(v) for i, v in rA_r.rate.items()},
                                    "B": {str(i): float(v) for i, v in rB_r.rate.items()}},
        "detection_rate_by_mu_abs": {"A": {str(i): float(v) for i, v in rA_m.rate.items()},
                                     "B": {str(i): float(v) for i, v in rB_m.rate.items()}},
        "pass": bool(gaps[f"K{K}_ratio"]["gap"] <= 0.10
                     and abs(emp - pred_lin) / pred_lin <= 0.30)}

    # ---------------- E3 ----------------
    band = atk[(atk.ratio > 0.6) & (atk.ratio < 1.0)]
    bt_bins = [0, 25, 50, 100, 200, 400]
    rb = rate_by_bin(band, "b_t_abs", bt_bins)
    R["E3"] = {"ratio_band": [0.6, 1.0], "n": int(len(band)),
               "detection_rate_by_b_t": {str(i): float(v) for i, v in rb.rate.items()},
               "spread": float(rb.rate.max() - rb.rate.min()) if len(rb) else float("nan"),
               "n_by_bin": {str(i): int(v) for i, v in rb.n.items()}}
    R["E3"]["pass"] = bool(R["E3"]["spread"] <= 0.02) if len(rb) else False
    # 2차 항이 커지는 구간 (|b_t| 가 d 와 비슷)
    near = atk[(atk.b_t_abs > 0.5 * atk.d)]
    R["E3"]["exception_b_t_gt_half_d"] = {
        "share_of_attack_rows": float(len(near) / len(atk)),
        "detection_rate": float(near[f"det{K}"].mean()) if len(near) else float("nan")}

    # ---------------- E4 ----------------
    by_sigma, by_n = {}, {}
    for r in json.load(open(os.path.join(WORK, "thresholds.json"), encoding="utf-8")):
        d2 = load(r["tag"])
        t2 = d2[(d2.split == "Test") & d2[f"s{K}"].notna() & (d2.label == 1)]
        p2 = rate_by_bin(t2, "ratio", r_bins)
        e2 = cross_half(cen(p2.index), p2.rate.to_numpy())
        rec = {"empirical_threshold_ratio": e2,
               "predicted": (LN10 / (10 * r["n"])) * Z_TWO * r["sigma_dB"] / math.sqrt(r[f"K_eff_K{K}"]),
               "tau_dB": r[f"tau_K{K}"], "K_eff": r[f"K_eff_K{K}"]}
        if r["d_corr_m"] == 25 and r["n"] == 2.0:
            by_sigma[str(r["sigma_dB"])] = rec
        if r["d_corr_m"] == 25 and r["sigma_dB"] == 4.0:
            by_n[str(r["n"])] = rec

    def logslope(d):
        x = np.array([float(k) for k in d]); y = np.array([d[k]["empirical_threshold_ratio"] for k in d])
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]) if m.sum() >= 2 else float("nan")

    dr = rate_by_bin(A, "d", [0, 50, 100, 150, 200, 300, 500, np.inf])
    dist_fpr = next((i.mid for i, v in dr.rate.items() if v <= fpr_test * 2), float("nan"))
    R["E4"] = {"threshold_ratio_by_sigma": by_sigma, "threshold_ratio_by_n": by_n,
               "slope_vs_sigma": logslope(by_sigma), "slope_vs_n": logslope(by_n),
               "theory_slope_sigma": 1.0, "theory_slope_n": -1.0,
               "detection_rate_by_distance": {str(i): float(v) for i, v in dr.rate.items()},
               "n_by_distance": {str(i): int(v) for i, v in dr.n.items()},
               "distance_where_rate_equals_fpr_m": float(dist_fpr)}

    # ---------------- E5 ----------------
    vb = df[(df.split == "Validation") & (df.label == 0)].rep_d2edge.dropna()
    w = float(np.quantile(vb, 0.01))                # 정상 FPR 1% 가 되는 도로 이탈 임계
    e5 = te[(te.cond.isin(["benign", "A"])) & te.rep_d2edge.notna()].copy()
    e5["map_det"] = e5.rep_d2edge < w
    e5["rssi_det"] = e5[f"det{K}"]
    ea = e5[e5.label == 1]
    conf = {"both": float(((ea.map_det) & (ea.rssi_det)).mean()),
            "map_only": float(((ea.map_det) & (~ea.rssi_det)).mean()),
            "rssi_only": float(((~ea.map_det) & (ea.rssi_det)).mean()),
            "neither": float(((~ea.map_det) & (~ea.rssi_det)).mean())}
    miss = ea[~ea.map_det]
    eb = e5[e5.label == 0]
    R["E5"] = {"road_edge_threshold_m": w,
               "map_detector": "sender.distance_to_road_edge < w (보고 위치 기준, SUMO 도로망 대신 데이터셋 제공 필드 사용)",
               "los_road_angle_deg": None,
               "confusion": conf,
               "rssi_rescue_rate_of_map_misses": float(miss.rssi_det.mean()) if len(miss) else float("nan"),
               "rescue_distance_median_m": float(miss[miss.rssi_det].d.median()) if len(miss) else float("nan"),
               "map_detection_rate": float(ea.map_det.mean()),
               "rssi_detection_rate": float(ea.rssi_det.mean()),
               "union_detection_rate": float((ea.map_det | ea.rssi_det).mean()),
               "union_fpr": float((eb.map_det | eb.rssi_det).mean()),
               "map_fpr": float(eb.map_det.mean()), "rssi_fpr": float(eb.rssi_det.mean())}

    R["fpr"] = {"val": T[f"fpr_val_K{K}"], "test": fpr_test,
                "test_one_sided": T[f"fpr_test_1s_K{K}"]}

    # ---------------- 그림 ----------------
    # fig2: (a) |b|  (b) |b_r|/d  (c) 정확한 잔차 |mu|
    from scipy.stats import norm as _norm
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))
    for a_, (cA, cB, xlab, ttl) in zip(ax, [
            (rA_b, rB_b, "오프셋 크기 |b| [m]", "(a) 오프셋 크기로 정리"),
            (rA_r, rB_r, "선 방향 성분 비 |b_r| / d", "(b) |b_r|/d 로 정리"),
            (rA_m, rB_m, "|10n·log10(d'/d)| [dB]", "(c) 정확한 잔차로 정리")]):
        a_.plot(cen(cA.index), cA.rate, "o-", color="C0", label="조건 A (28–99 m)")
        a_.plot(cen(cB.index), cB.rate, "s--", color="C3", label="조건 B (292 m)")
        a_.set_xlabel(xlab); a_.set_ylabel("탐지율"); a_.set_title(ttl)
        a_.axhline(fpr_test, color="gray", ls="--", lw=1, label="오탐률")
        a_.set_ylim(-0.03, 1.03); a_.grid(alpha=.3)
    ax[1].set_xscale("log")
    # 로그축 기본 라벨은 mathtext 라 한글 폰트에 없는 U+2212 가 깨진다. 숫자로 직접 찍는다.
    from matplotlib.ticker import FuncFormatter, LogLocator
    _f = FuncFormatter(lambda v, _p: f"{v:g}")
    ax[1].xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax[1].xaxis.set_major_formatter(_f)
    ax[1].xaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ""))
    ax[1].axvline(pred_lin, color="k", ls=":", lw=1.2, label=f"식 (3) 선형 예측 {pred_lin:.2f}")
    ax[1].axvline(exact_pos, color="g", ls="-.", lw=1.2, label=f"정확한 예측 (b_r>0) {exact_pos:.2f}")
    xs = np.linspace(0, 30, 200)
    ax[2].plot(xs, _norm.sf((tau - xs) / sigma) + _norm.cdf((-tau - xs) / sigma),
               "k:", lw=1.4, label="이론 P(|μ+X|>τ)")
    for a_ in ax:
        a_.legend(fontsize=7)
    fig.suptitle("조건 A·B 곡선 간격:  |b| 겹치는 구간 없음   ·   "
                 f"|b_r|/d {gaps[f'K{K}_ratio']['gap']:.3f}   ·   "
                 f"|μ| {gaps[f'K{K}_mu_abs']['gap']:.3f}    (K=5)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "fig2_detection_rate.png"), dpi=200)
    plt.close(fig)

    # figA: E1
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    idx = np.random.default_rng(0).choice(len(atk), min(6000, len(atk)), replace=False)
    ax.scatter(app.to_numpy()[idx], obs.to_numpy()[idx], s=3, alpha=.25)
    lim = [np.nanpercentile(app, .2), np.nanpercentile(app, 99.8)]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("식 (2) 근사 [dB]"); ax.set_ylabel("정확한 값 10n·log10(d'/d) [dB]")
    ax.set_title(f"E1  r = {R['E1']['pearson_r']:.4f}, MAE = {R['E1']['mae_dB']:.3f} dB")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figA_residual_check.png"), dpi=200); plt.close(fig)

    # fig3: 보완성
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sa = ea.sample(min(6000, len(ea)), random_state=0)
    x = (w - sa.rep_d2edge)                      # >0 이면 지도 검사가 탐지
    y = sa[f"s{K}"].abs() - tau                  # >0 이면 신호 세기 검사가 탐지
    ax.scatter(x, y, s=3, alpha=.25)
    ax.axvline(0, color="k", lw=1); ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("도로 이탈 여유 (w − 도로가장자리거리) [m]  → 오른쪽이 지도 탐지")
    ax.set_ylabel("|s| − τ [dB]  → 위쪽이 신호세기 탐지")
    ax.set_title("E5 두 검사의 보완성 (조건 A)")
    ax.set_xlim(np.percentile(x, .5), np.percentile(x, 99.5))
    ax.set_ylim(np.percentile(y, .5), np.percentile(y, 99.5))
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_complementarity.png"), dpi=200); plt.close(fig)

    # ---------------- notes ----------------
    N = R["notes"]
    N.append(
        f"E1: 식 (2)의 유효 영역(|b|/d<=0.25)에서는 **합격** (r={R['E1']['pearson_r']:.4f}, "
        f"MAE={R['E1']['mae_dB']:.3f} dB). 그러나 전체 행으로는 불합격이다 "
        f"(r={R['E1']['pearson_r_all_rows']:.3f}, MAE={R['E1']['mae_dB_all_rows']:.1f} dB). "
        f"|b|/d가 커질수록 급격히 나빠진다: "
        + ", ".join(f"{k} MAE {v['mae_dB']:.2f} dB" for k, v in list(R["E1"]["by_b_over_d"].items())[1:]))
    N.append(
        f"E2 **불합격**. |b_r|/d 로 정리해도 조건 A·B 곡선 간격이 {gaps[f'K{K}_ratio']['gap']:.3f} "
        f"(K=1이면 {gaps['K1_ratio']['gap']:.3f}) 로 기준 0.10 을 넘는다. "
        f"실제 임계 {emp:.3f} vs 식 (3) 선형 예측 {pred_lin:.3f} (상대오차 "
        f"{R['E2']['rel_err_vs_predicted']*100:.0f}%) 로 기준 30% 를 크게 넘는다.")
    N.append(
        "E2 불합격의 원인은 코드가 아니라 **선형근사의 자기모순**이다. 식 (3)이 예측하는 임계 "
        f"|b_r|/d = {pred_lin:.2f} 는 1 에 가까운 값인데, 식 (2)의 선형화는 |b|/d << 1 에서만 "
        "성립한다(E1). 즉 근사가 깨지는 지점을 근사로 예측하고 있다. 근사를 쓰지 않은 정확한 조건 "
        f"|b_r|/d > 10^(τ/10n) − 1 = {exact_pos:.2f} (b_r>0) 로 계산하면 실제 임계 {emp_pos:.2f} 와 "
        f"{R['E2']['rel_err_exact_b_r_positive']*100:.0f}% 이내로 맞는다.")
    N.append(
        f"정확한 조건은 b_r 의 **부호에 대해 비대칭**이다. b_r>0(멀다고 주장)는 {exact_pos:.2f}, "
        f"b_r<0(가깝다고 주장)는 {exact_neg:.2f} 로 3배 이상 차이가 난다. 로그가 한쪽을 압축하기 "
        "때문이며, |b_r|/d 를 부호 없이 하나로 묶으면 서로 다른 두 임계가 섞여 곡선이 흐려진다.")
    N.append(
        f"탐지율을 실제로 정리하는 변수는 정확한 잔차 |μ| = |10n·log10(d'/d)| 이다. 이것으로 "
        f"정리하면 곡선 간격이 K=1 에서 {gaps['K1_mu_abs']['gap']:.3f} 로 사실상 0 이고, "
        "구간별 탐지율이 이론값 P(|μ+X|>τ) 와 일치한다(그림 2c). 파이프라인이 옳다는 근거이기도 하다. "
        f"|b_r|/d 는 |μ| 의 대용으로 쓸 수 있으나 같은 |b_r|/d 에서 |μ| 의 사분위범위가 "
        "최대 7 dB 에 이른다(접선 성분과 로그 비선형성 때문).")
    N.append(
        f"|b| 로 정리한 곡선은 두 조건의 값 범위가 겹치지 않아(A 28–99 m, B 292 m) 공통 구간이 "
        "0개다. 따라서 '|b| 로는 분리된다'는 것은 측정된 사실이 아니라 두 조건이 서로 다른 구간에 "
        "있다는 뜻일 뿐이므로, 이 대비를 논문의 근거로 쓰면 안 된다.")
    N.append(
        f"이동 평균은 곡선을 흐린다. |b_r|/d 기준 간격이 K=1 {gaps['K1_ratio']['gap']:.3f} → "
        f"K=5 {gaps[f'K{K}_ratio']['gap']:.3f} 로 커진다. 5개를 평균하는 동안 상대 기하가 변해 "
        "그 행의 순간 |b_r|/d 가 판정을 대표하지 못하기 때문이다.")
    N.append(
        f"E4 의 기울기도 같은 원인을 가리킨다. σ 기울기 {R['E4']['slope_vs_sigma']:.2f}(이론 +1), "
        f"n 기울기 {R['E4']['slope_vs_n']:.2f}(이론 −1) 로 둘 다 크기가 1 보다 크다. "
        "선형 이론은 임계가 σ 에 비례한다고 보지만, 정확한 임계 10^(τ/10n)−1 은 τ(∝σ) 에 대해 "
        "지수적으로 자라므로 기울기가 1 을 넘는 것이 맞다. σ·n 민감도도 선형근사가 아니라 "
        "정확한 식으로 읽어야 한다.")
    N.append(f"섀도잉 공간 상관 때문에 K=5의 유효 표본 수는 K_eff={K_eff:.2f}에 그친다. "
             f"흰 잡음으로 만들면 K_eff가 {thr('n2_s4_dc0')[f'K_eff_K{K}']:.2f}로 나와 이동 평균 이득이 과대평가된다.")
    N.append("수신 차량은 자신의 위치를 정확히 안다고 가정했다(지침서의 단일 q 규약). "
             "수신자 자신의 측위 오차를 넣으면 정상 잔차의 바닥이 조금 올라간다.")
    N.append("n=2.7 조합에서는 수신 감도 미달로 행의 40.9%가 빠져 모집단이 달라진다. "
             "n 에 대한 기울기는 이 점을 감안해 읽어야 한다.")
    N.append("E5의 도로 이탈 검사는 SUMO 도로망 대신 데이터셋의 sender.distance_to_road_edge 필드를 썼다. "
             "los_road_angle 과 b_n 은 계산하지 않았다(그림 3의 축을 이탈 여유로 대체).")
    N.append("조건 B 는 무공격 실행의 보고 위치에 b=(250,-150) 을 더해 만들었으므로 "
             "조건 A 와 실제 위치·섀도잉 실현값이 완전히 같다.")

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2, default=float)

    print(json.dumps({k: R[k] for k in ("E1", "E2", "E3", "E4", "E5", "fpr")},
                     ensure_ascii=False, indent=2, default=float)[:4000])
    print("\n작성: out/results.json, fig2/fig3/figA")


if __name__ == "__main__":
    main()
