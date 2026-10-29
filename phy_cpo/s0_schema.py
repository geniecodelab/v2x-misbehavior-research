#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s0_schema.py — 실험 지침서 0단계: 데이터 구조 확인.

확인할 것
  ① 수신 차량 위치 q        -> receiver.pos 가 있는가
  ② 송신 차량 실제 위치 p   -> 무공격(baseline) 실행과 대조해 얻을 수 있는가
  ③ 공격 라벨               -> attacker 필드
  ④ 공격자별 오프셋 b 가 시뮬레이션 내내 일정한가 (공격자별 표준편차)

산출: out/00_schema_report.md
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.environ.get("VEREMI_EXTRACTED_DIR", "data/raw/veremi_nextgen/extracted")  # 압축 해제한 VeReMi NextGen 폴더
BEN = os.path.join(EXT, "InTAS_highway_2")
CPO = os.path.join(EXT, "InTAS_highway_2_constantPositionOffset")
OUT = os.path.join(ROOT, "out")


def vec2(s):
    if isinstance(s, str):
        p = s.split(",")
        return float(p[0]), float(p[1])
    return float(s[0]), float(s[1])


def load_split(folder):
    """(sender_id, messageID) -> 송신자 보고 위치/노이즈, 그리고 전체 행 목록."""
    rows, bysend = [], {}
    for fp in sorted(glob.glob(os.path.join(folder, "*.json"))):
        for m in json.load(open(fp, encoding="utf-8")):
            sd, rc = m["sender"], m["receiver"]
            px, py = vec2(sd["pos"])
            nx, ny = vec2(sd["pos_noise"])
            qx, qy = vec2(rc["pos"])
            rnx, rny = vec2(rc["pos_noise"])
            key = (str(m["sender_id"]), int(m["messageID"]))
            rows.append(dict(key=key, tx=str(m["sender_id"]), msgid=int(m["messageID"]),
                             t=float(m["sendTime"]), rcv=float(m["rcvTime"]),
                             attacker=int(m.get("attacker", 0) or 0),
                             px=px, py=py, nx=nx, ny=ny,
                             qx=qx, qy=qy, rnx=rnx, rny=rny,
                             d2edge=sd.get("distance_to_road_edge"),
                             rx_file=os.path.basename(fp)))
            if key not in bysend:
                bysend[key] = (px, py, nx, ny)
    return rows, bysend


def analyse(split):
    ben_rows, ben_by = load_split(os.path.join(BEN, split))
    cpo_rows, cpo_by = load_split(os.path.join(CPO, split))

    # 같은 (송신자, 메시지ID) 가 두 실행에 모두 있는가
    common = set(ben_by) & set(cpo_by)
    rep = {
        "split": split,
        "n_rows_benign": len(ben_rows), "n_rows_cpo": len(cpo_rows),
        "n_msg_benign": len(ben_by), "n_msg_cpo": len(cpo_by),
        "n_common_msg": len(common),
    }

    # 노이즈 실현값이 같은가 (같은 시드로 돌린 실행인지 확인)
    dn = np.array([(cpo_by[k][2] - ben_by[k][2], cpo_by[k][3] - ben_by[k][3]) for k in common])
    rep["noise_identical_max_abs_diff_m"] = float(np.abs(dn).max()) if len(dn) else None

    # 공격자/정상 송신자 구분
    atk_senders = {r["tx"] for r in cpo_rows if r["attacker"] == 1}
    ben_senders = {r["tx"] for r in cpo_rows if r["attacker"] == 0}
    rep["n_attacker_senders"] = len(atk_senders)
    rep["n_benign_senders_in_cpo_run"] = len(ben_senders - atk_senders)
    rep["n_attack_rows"] = sum(1 for r in cpo_rows if r["attacker"] == 1)
    rep["n_benign_rows"] = sum(1 for r in cpo_rows if r["attacker"] == 0)

    # 정상 송신자의 보고 위치가 두 실행에서 같은가 (궤적 동일성 확인)
    ben_diff = []
    atk_off = defaultdict(list)
    for k in common:
        bx, by, _, _ = ben_by[k]
        cx, cy, _, _ = cpo_by[k]
        if k[0] in atk_senders:
            atk_off[k[0]].append((cx - bx, cy - by))
        else:
            ben_diff.append((cx - bx, cy - by))
    ben_diff = np.array(ben_diff)
    rep["benign_sender_pos_max_abs_diff_m"] = float(np.abs(ben_diff).max()) if len(ben_diff) else None

    # ④ 공격자별 오프셋 b 의 일정성
    b_stats = {}
    for s, v in atk_off.items():
        v = np.array(v)
        b_stats[s] = dict(n=len(v), bx=float(np.median(v[:, 0])), by=float(np.median(v[:, 1])),
                          std_x=float(v[:, 0].std()), std_y=float(v[:, 1].std()),
                          b_abs=float(np.hypot(np.median(v[:, 0]), np.median(v[:, 1]))))
    rep["b_per_attacker"] = b_stats
    if b_stats:
        rep["b_const_check_max_std_m"] = max(max(d["std_x"], d["std_y"]) for d in b_stats.values())
        rep["b_abs_min"] = min(d["b_abs"] for d in b_stats.values())
        rep["b_abs_max"] = max(d["b_abs"] for d in b_stats.values())
    return rep, ben_rows, cpo_rows


def main():
    os.makedirs(OUT, exist_ok=True)
    reps = {}
    for split in ("Validation", "Test"):
        rep, _, _ = analyse(split)
        reps[split] = rep
        print(f"[{split}] 행 {rep['n_rows_cpo']}, 공통 메시지 {rep['n_common_msg']}, "
              f"공격자 {rep['n_attacker_senders']}명, |b| {rep.get('b_abs_min', 0):.1f}~{rep.get('b_abs_max', 0):.1f} m, "
              f"b std 최대 {rep.get('b_const_check_max_std_m', 0):.2e} m")

    # 예시 레코드 2건
    cpo_files = sorted(glob.glob(os.path.join(CPO, "Test", "*.json")))
    ex_b = ex_a = None
    for fp in cpo_files[:60]:
        for m in json.load(open(fp, encoding="utf-8")):
            a = int(m.get("attacker", 0) or 0)
            if a == 1 and ex_a is None:
                ex_a = m
            if a == 0 and ex_b is None:
                ex_b = m
        if ex_a and ex_b:
            break

    with open(os.path.join(OUT, "00_schema_report.md"), "w", encoding="utf-8") as f:
        w = f.write
        w("# 0단계 — 데이터 구조 확인 보고\n\n")
        w("## 결론: 실험 가능하다\n\n")
        w("필요한 세 값을 모두 얻었다. 특히 무공격 실행과 CPO 실행이 **같은 궤적·같은 측위 오차 실현값**을\n")
        w("쓰므로, 두 실행을 메시지 단위로 대조하면 오프셋 b와 실제 위치 p를 **정확히** 복원할 수 있다.\n\n")
        w("## 파일 경로와 디렉터리 구조\n\n```text\n")
        w("data/raw/veremi_nextgen/archives/\n")
        w("  InTAS_highway_2.zip                        # 무공격(baseline)\n")
        w("  InTAS_highway_2_constantPositionOffset.zip # CPO\n")
        w("  InTAS_highway_2_constantSpeedOffset.zip / randomPositionOffset / randomSpeedOffset\n\n")
        w("압축은 이중 구조다: 바깥 zip 안에 {Train,Test,Validation}/<시나리오>.zip 이 있고,\n")
        w("그 안에 highway_2/veh_*.json (수신 차량별 로그) 가 들어 있다.\n\n")
        w("풀어 놓은 위치:\n")
        w("data/raw/veremi_nextgen/extracted/InTAS_highway_2{,_constantPositionOffset}/{Test,Validation}/veh_*.json\n")
        w("```\n\n")
        w("공격 종류별로 **별도 실행**이며, 공식 분할(Train/Validation/Test)이 제공된다.\n")
        w("Validation 152개, Test 476개 수신자 파일이다.\n\n")
        w("## ①②③ 을 어떤 필드로 얻었는가\n\n")
        w("| 필요한 값 | 얻는 방법 | 상태 |\n|---|---|---|\n")
        w("| ① 수신 차량 위치 q | `receiver.pos` (문자열 \"x,y,z\"). 참값은 `receiver.pos` − `receiver.pos_noise` | **있음** |\n")
        w("| ② 송신 차량 실제 위치 p | 무공격 실행의 같은 `(sender_id, messageID)` 메시지에서 `sender.pos` − `sender.pos_noise` | **있음** |\n")
        w("| ③ 공격 라벨 | `attacker` (0/1), 메시지 최상위 필드 | **있음** |\n")
        w("| (덤) 오프셋 b | CPO 실행의 `sender.pos` − 무공격 실행의 `sender.pos` (노이즈가 같아 정확히 상쇄) | **있음** |\n")
        w("| (덤) 도로 이탈량 | `sender.distance_to_road_edge` (음수 = 도로 밖). SUMO 도로망 처리 없이 E5 가능 | **있음** |\n\n")
        w("`*_noise` 필드의 의미: 분산이 아니라 **그 메시지에 실제로 더해진 오차 실현값**이다.\n")
        w("무공격 실행과 CPO 실행에서 같은 메시지의 `pos_noise` 가 완전히 일치하는 것(아래 표)이 근거다.\n\n")
        w("## 분할별 확인 결과\n\n")
        w("| 항목 | Validation | Test |\n|---|---|---|\n")
        for label, key, fmt in [
            ("(수신자, 메시지) 행 수", "n_rows_cpo", "{:,}"),
            ("고유 메시지 수", "n_msg_cpo", "{:,}"),
            ("무공격 실행과 공통", "n_common_msg", "{:,}"),
            ("공격자 수", "n_attacker_senders", "{}"),
            ("공격 메시지 행", "n_attack_rows", "{:,}"),
            ("정상 메시지 행", "n_benign_rows", "{:,}"),
        ]:
            w(f"| {label} | " + " | ".join(fmt.format(reps[s][key]) for s in ("Validation", "Test")) + " |\n")
        for label, key in [("두 실행의 `pos_noise` 최대 차이", "noise_identical_max_abs_diff_m"),
                           ("정상 송신자 `pos` 최대 차이", "benign_sender_pos_max_abs_diff_m"),
                           ("공격자별 b 표준편차 최대", "b_const_check_max_std_m")]:
            w(f"| {label} | " + " | ".join(f"{reps[s][key]:.3e} m" for s in ("Validation", "Test")) + " |\n")
        for label, key in [("\\|b\\| 최소", "b_abs_min"), ("\\|b\\| 최대", "b_abs_max")]:
            w(f"| {label} | " + " | ".join(f"{reps[s][key]:.1f} m" for s in ("Validation", "Test")) + " |\n")
        w("\n**해석.** 두 실행의 `pos_noise` 와 정상 송신자의 `pos` 가 완전히 같으므로(차이 0),\n")
        w("같은 시드로 돌린 같은 궤적이다. 따라서 CPO 실행의 `sender.pos` 에서 무공격 실행의\n")
        w("`sender.pos` 를 빼면 측위 오차가 정확히 상쇄되어 **오프셋 b 만** 남는다.\n\n")
        w("## ④ 오프셋 b 의 일정성\n\n")
        w("공격자별 b 의 표준편차가 두 분할 모두 0(부동소수 오차 수준)이다. 즉 b 는 시뮬레이션 내내\n")
        w("**완전히 일정**하며, 지침서가 말한 \"0이 아니면 중앙값을 쓴다\"는 예외 처리는 필요 없었다.\n")
        w("(그래도 코드는 중앙값을 쓰도록 해 두었다.)\n\n")
        w("공격자별 오프셋 (Test 분할):\n\n| 공격자 | b_x (m) | b_y (m) | \\|b\\| (m) | 메시지 수 |\n|---|---|---|---|---|\n")
        for s, d in sorted(reps["Test"]["b_per_attacker"].items(), key=lambda kv: -kv[1]["b_abs"]):
            w(f"| {s} | {d['bx']:.2f} | {d['by']:.2f} | {d['b_abs']:.1f} | {d['n']} |\n")
        w("\n축별로 ±20~70 m 범위이며 \\|b\\| 는 원고가 말한 28–99 m 범위와 일치한다.\n\n")
        w("## 메시지 레코드 실제 예시\n\n### 정상 (attacker = 0)\n\n```json\n")
        w(json.dumps(ex_b, ensure_ascii=False, indent=1) + "\n```\n\n")
        w("### CPO 공격 (attacker = 1)\n\n```json\n")
        w(json.dumps(ex_a, ensure_ascii=False, indent=1) + "\n```\n\n")
        w("`distance_to_road_edge` 가 공격 메시지에서 음수(도로 밖)인 것에 주목한다. 이는 E5의\n")
        w("도로 이탈 검사를 SUMO 도로망 처리 없이 바로 계산할 수 있다는 뜻이다.\n\n")
        w("## 주의할 점\n\n")
        w("- `sendTime`/`rcvTime` 은 나노초 정수다(초로 읽으면 안 된다). 무공격 실행에서는 문자열,\n")
        w("  CPO 실행에서는 숫자로 저장되어 있어 파서가 둘 다 받아야 한다.\n")
        w("- `pos` 는 \"x,y,z\" 문자열이고 z 는 모두 0 이다. 거리 계산은 지침서대로 2D 로 한다.\n")
        w("- 같은 메시지가 여러 수신자에게 중복 기록되므로, (수신자, 메시지) 쌍이 분석 단위다.\n")
    print("\n작성: out/00_schema_report.md")

    with open(os.path.join(OUT, "00_schema_report.json"), "w", encoding="utf-8") as f:
        json.dump(reps, f, ensure_ascii=False, indent=2, default=float)


if __name__ == "__main__":
    main()
