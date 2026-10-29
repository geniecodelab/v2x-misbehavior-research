#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1_messages.py — 1단계: (수신자, 메시지) 테이블 만들기.

조건 A = 데이터셋 기본 CPO (|b| 31~92 m)
조건 B = 같은 궤적·같은 공격자에 b = (250, -150) m 를 주입한 것 (선행연구[4] 조건)
정상 행은 두 조건이 공유한다.

좌표 규약
  p_true = 무공격 실행의 sender.pos - sender.pos_noise      (송신 실제 위치)
  p_rep  = CPO 실행의 sender.pos                            (보고 위치, 노이즈 포함)
  q      = receiver.pos - receiver.pos_noise                (수신 차량 실제 위치)
  b      = CPO 실행 sender.pos - 무공격 실행 sender.pos      (노이즈가 상쇄되어 오프셋만 남음)

산출: work/messages.parquet, out/01_dataset_stats.md
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.environ.get("VEREMI_EXTRACTED_DIR", "data/raw/veremi_nextgen/extracted")  # 압축 해제한 VeReMi NextGen 폴더
BEN = os.path.join(EXT, "InTAS_highway_2")
CPO = os.path.join(EXT, "InTAS_highway_2_constantPositionOffset")
WORK, OUT = os.path.join(ROOT, "work"), os.path.join(ROOT, "out")
B_COND_B = (250.0, -150.0)          # 선행연구[4] 조건, |b| = 291.5 m


def vec2(s):
    if isinstance(s, str):
        p = s.split(",")
        return float(p[0]), float(p[1])
    return float(s[0]), float(s[1])


def read_folder(folder):
    out = []
    for fp in sorted(glob.glob(os.path.join(folder, "*.json"))):
        for m in json.load(open(fp, encoding="utf-8")):
            sd, rc = m["sender"], m["receiver"]
            sx, sy = vec2(sd["pos"]); snx, sny = vec2(sd["pos_noise"])
            qx, qy = vec2(rc["pos"]); qnx, qny = vec2(rc["pos_noise"])
            out.append((str(m["sender_id"]), int(m["messageID"]),
                        float(m["sendTime"]) / 1e9, float(m["rcvTime"]) / 1e9,
                        int(m.get("attacker", 0) or 0),
                        sx, sy, snx, sny, qx, qy, qnx, qny,
                        float(sd.get("distance_to_road_edge", np.nan)),
                        os.path.splitext(os.path.basename(fp))[0]))
    cols = ["tx_id", "msg_id", "t", "rcv_t", "attacker", "sx", "sy", "snx", "sny",
            "qx", "qy", "qnx", "qny", "d2edge", "rx_id"]
    return pd.DataFrame(out, columns=cols)


def build(split):
    cpo = read_folder(os.path.join(CPO, split))
    ben = read_folder(os.path.join(BEN, split))
    # 무공격 실행에서 송신자 보고 위치와 참값을 가져온다 (메시지 단위로 유일)
    b1 = ben.drop_duplicates(["tx_id", "msg_id"])[["tx_id", "msg_id", "sx", "sy", "snx", "sny", "d2edge"]]
    b1 = b1.rename(columns=dict(sx="ben_sx", sy="ben_sy", snx="ben_snx", sny="ben_sny",
                                d2edge="ben_d2edge"))
    df = cpo.merge(b1, on=["tx_id", "msg_id"], how="inner")

    df["p_true_x"] = df.ben_sx - df.ben_snx          # 송신 실제 위치
    df["p_true_y"] = df.ben_sy - df.ben_sny
    df["qx_true"] = df.qx - df.qnx                   # 수신 실제 위치
    df["qy_true"] = df.qy - df.qny
    df["b_x"] = df.sx - df.ben_sx                    # 오프셋 (정상은 0)
    df["b_y"] = df.sy - df.ben_sy
    df["split"] = split

    rows = []
    # 정상 행 (두 조건 공유)
    g = df[df.attacker == 0].copy()
    g["cond"] = "benign"; g["label"] = 0
    g["p_rep_x"], g["p_rep_y"] = g.sx, g.sy
    g["bb_x"], g["bb_y"] = 0.0, 0.0
    g["rep_d2edge"] = g.d2edge
    rows.append(g)
    # 조건 A: 데이터셋 기본 CPO
    a = df[df.attacker == 1].copy()
    a["cond"] = "A"; a["label"] = 1
    a["p_rep_x"], a["p_rep_y"] = a.sx, a.sy
    a["bb_x"], a["bb_y"] = a.b_x, a.b_y
    a["rep_d2edge"] = a.d2edge
    rows.append(a)
    # 조건 B: 같은 궤적에 b=(250,-150) 주입 (무공격 보고 위치 + b)
    b = df[df.attacker == 1].copy()
    b["cond"] = "B"; b["label"] = 1
    b["p_rep_x"] = b.ben_sx + B_COND_B[0]
    b["p_rep_y"] = b.ben_sy + B_COND_B[1]
    b["bb_x"], b["bb_y"] = B_COND_B[0], B_COND_B[1]
    b["rep_d2edge"] = np.nan          # 도로망 기준 재계산 불가 (E5는 조건 A만)
    rows.append(b)

    r = pd.concat(rows, ignore_index=True)
    # 기하량
    ux, uy = r.p_true_x - r.qx_true, r.p_true_y - r.qy_true
    r["d"] = np.hypot(ux, uy)
    r["d_rep"] = np.hypot(r.p_rep_x - r.qx_true, r.p_rep_y - r.qy_true)
    ux, uy = ux / r.d, uy / r.d                       # 수신 -> 송신 실제 방향 단위벡터
    r["b_abs"] = np.hypot(r.bb_x, r.bb_y)
    r["b_r"] = r.bb_x * ux + r.bb_y * uy
    r["b_t_abs"] = np.sqrt(np.maximum(r.b_abs ** 2 - r.b_r ** 2, 0.0))
    r["cos_theta"] = np.where(r.b_abs > 0, r.b_r / np.where(r.b_abs > 0, r.b_abs, 1), np.nan)
    r["ratio"] = np.abs(r.b_r) / r.d                  # |b_r|/d  ★ 핵심 변수
    return r[["split", "cond", "label", "msg_id", "t", "rcv_t", "tx_id", "rx_id",
              "p_rep_x", "p_rep_y", "p_true_x", "p_true_y", "qx_true", "qy_true",
              "bb_x", "bb_y", "d", "d_rep", "b_r", "b_t_abs", "b_abs", "cos_theta",
              "ratio", "rep_d2edge", "ben_d2edge"]].rename(
        columns=dict(bb_x="b_x", bb_y="b_y"))


def main():
    os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    parts = []
    for split in ("Validation", "Test"):
        r = build(split)
        print(f"[{split}] {len(r):,}행  "
              + "  ".join(f"{c}={int((r['cond'] == c).sum()):,}" for c in ("benign", "A", "B")))
        parts.append(r)
    m = pd.concat(parts, ignore_index=True)
    m.to_parquet(os.path.join(WORK, "messages.parquet"), index=False)

    with open(os.path.join(OUT, "01_dataset_stats.md"), "w", encoding="utf-8") as f:
        w = f.write
        w("# 1단계 — 메시지 테이블\n\n`work/messages.parquet`\n\n")
        w("행 단위는 **(수신자, 메시지) 쌍**이다. 정상 행은 조건 A·B가 공유하고,\n")
        w("공격 행은 조건마다 따로 만들었다(같은 궤적·같은 실제 위치, 보고 위치만 다름).\n\n")
        w("## 행 수\n\n| 분할 | 정상 | 조건 A (CPO 기본) | 조건 B (250,-150) | 합계 |\n|---|---|---|---|---|\n")
        for s in ("Validation", "Test"):
            g = m[m.split == s]
            w(f"| {s} | {(g.cond=='benign').sum():,} | {(g.cond=='A').sum():,} | "
              f"{(g.cond=='B').sum():,} | {len(g):,} |\n")
        w("\n## 공격/정상 비율 (조건 A 기준)\n\n| 분할 | 공격 행 비율 | 공격자 수 | 정상 송신자 수 |\n|---|---|---|---|\n")
        for s in ("Validation", "Test"):
            g = m[(m.split == s) & (m.cond.isin(["benign", "A"]))]
            w(f"| {s} | {g.label.mean():.3f} | {g[g.label==1].tx_id.nunique()} | "
              f"{g[g.label==0].tx_id.nunique()} |\n")
        w("\n## 기하량 분포 (Test, 공격 행)\n\n")
        w("| 조건 | \\|b\\| (m) | d (m) 중앙값 | \\|b_r\\| (m) 중앙값 | \\|b_r\\|/d 중앙값 | cos θ 중앙값 |\n|---|---|---|---|---|---|\n")
        for c in ("A", "B"):
            g = m[(m.split == "Test") & (m.cond == c)]
            w(f"| {c} | {g.b_abs.min():.1f}–{g.b_abs.max():.1f} | {g.d.median():.1f} | "
              f"{g.b_r.abs().median():.1f} | {g.ratio.median():.3f} | {g.cos_theta.abs().median():.2f} |\n")
        w("\n## 수신 거리 d 분포 (Test, 전체 행)\n\n")
        q = m[m.split == "Test"].d.quantile([.05, .25, .5, .75, .95])
        w("| 5% | 25% | 50% | 75% | 95% | 최대 |\n|---|---|---|---|---|---|\n")
        w("| " + " | ".join(f"{v:.0f}" for v in q) + f" | {m[m.split=='Test'].d.max():.0f} | (m)\n")
        w("\n## 좌표 규약\n\n")
        w("- `p_true` = 무공격 실행 `sender.pos` − `sender.pos_noise` (송신 실제 위치)\n")
        w("- `p_rep` = CPO 실행 `sender.pos` (보고 위치, 측위 오차 포함)\n")
        w("- `q` = `receiver.pos` − `receiver.pos_noise` (수신 실제 위치)\n")
        w("- `b` = CPO `sender.pos` − 무공격 `sender.pos` (측위 오차가 상쇄되어 오프셋만 남음)\n")
        w("- 거리는 모두 2D(x, y). z는 전 레코드 0이라 쓰지 않았다.\n\n")
        w("정상 행에서도 `d_rep` ≠ `d` 인데, 이는 송신자의 측위 오차(수 m) 때문이며\n")
        w("탐지기가 실제로 겪는 조건이다. 이 성분이 오탐률의 바닥을 만든다.\n")
    print("작성: work/messages.parquet, out/01_dataset_stats.md")


if __name__ == "__main__":
    main()
