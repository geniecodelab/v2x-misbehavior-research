#!/usr/bin/env python3
"""
plot_e5.py: pilot5가 저장한 results/e5_tau_sweep_summary.csv 만 읽어 그림을 그린다.
(matplotlib 설치 후 실행. 계산은 다시 하지 않는다.)

  python plot_e5.py [--brake-level 4] [--csv results/e5_tau_sweep_summary.csv]
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ORDER_LABEL = {0: "oracle (실제 미래)", 1: "1차 (등속)", 2: "2차 (원호)", 3: "3차 (클로소이드)"}
COLORS = {0: "#888888", 1: "#555555", 2: "#d9822b", 3: "#1f5aa6"}


def pick_font():
    for cand in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"):
        if any(cand == f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/e5_tau_sweep_summary.csv")
    ap.add_argument("--brake-level", type=float, default=4.0)
    ap.add_argument("--out", default="results/fig_tau_optimum.png")
    a = ap.parse_args()

    font = pick_font()
    plt.rcParams["axes.unicode_minus"] = False
    if font is None:
        print("[경고] 한글 폰트를 찾지 못했습니다. 축 이름이 깨질 수 있습니다.")

    s = pd.read_csv(a.csv)
    b = s[s["segment"] == "brake"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), dpi=300)
    for o in sorted(b["order"].unique()):
        g = b[b["order"] == o].sort_values("tau")
        st = dict(marker="o", ms=3, lw=1.3, color=COLORS.get(o), ls="--" if o == 0 else "-")
        ax[0].plot(g["tau"], g["det"], label=ORDER_LABEL.get(o, o), **st)
        ax[1].plot(g["lie_m"], g["det"], **st)

    env = b[b["order"] > 0].sort_values("lie_m").groupby("lie_m")["det"].min().cummin()
    ax[1].plot(env.index, env.values, color="#c0392b", lw=2, alpha=0.85, label="공격자의 최선(포락선)")

    ax[0].set_xlabel("예측 지평 τ (초)", fontsize=8)
    ax[0].set_ylabel(f"급감속(|a|>{a.brake_level:g} m/s²) 구간 탐지율", fontsize=7.5)
    ax[1].set_xlabel("공간 거짓말 크기 ≈ v·τ (m)", fontsize=8)
    for x in ax:
        x.grid(lw=0.3, alpha=0.5)
        x.tick_params(labelsize=7)
        x.set_ylim(-0.02, 1.02)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
    ax[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    ax[1].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"그림 저장: {a.out}")


if __name__ == "__main__":
    main()
