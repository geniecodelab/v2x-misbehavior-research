"""
common.py: timewarp 프로젝트 공용 설정.

analysis/ 의 f5_map_check, road_geom을 복사하지 않고 그대로 불러 쓴다.
기존 산출물(map.pkl, transform.json, scored_*.parquet)은 artifacts/ 에서 참조한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent      # 저장소 루트
HERE = Path(__file__).resolve().parent             # ...\timewarp
RESULTS = HERE / "results"
ARTIFACTS = ROOT / "artifacts"                     # map.pkl, transform.json, scored_*.parquet
sys.path.insert(0, str(ROOT / "analysis"))         # f5_map_check, road_geom

import f5_map_check as f  # noqa: E402

C = f.COL
A_LIM_DEFAULT = 3.0        # 승용차 상시 사용 가능한 종방향 가감속 한계(m/s^2)
V_MAX = 55.0               # 기존 Rule의 속도 상한
DT_NOMINAL = 1.0


def log(*a):
    f.log(*a)


def path(name: str) -> Path:
    """기존 산출물 경로. 절대경로는 그대로, 이름만 주면 artifacts/ 에서 찾는다."""
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    return ARTIFACTS / name


def load_benign(scored_name="scored_tol5.parquet", split="test", max_senders=0, seed=0,
                min_len=30):
    """
    송신자별 참궤적(pos - position_noise, heading - heading_noise)을 딕셔너리로 반환.
    오라클 필드지만 공격자는 자기 참상태를 알고 있으므로 공격 생성에 사용할 수 있다.
    """
    df = f.read_table(path(scored_name))
    b = df[(df[C["variant"]] == f.BENIGN_VARIANT) & (df[C["split"]] == split)]
    b = b.drop_duplicates([C["sender"], C["t"]]).sort_values([C["sender"], C["t"]])

    hm = f.heading_math_from(b, -1, 1, dict(flip_y=False, y_ref=0.0, tx=0.0, ty=0.0))
    if "heading_noise" in b.columns:                      # 참헤딩 = 보고 - 노이즈(도)
        hm = f.wrap_angle(hm + np.radians(b["heading_noise"].to_numpy(float)))
    b = b.assign(_hm=hm)
    if {"position_noise_x", "position_noise_y"} <= set(b.columns):
        b = b.assign(_x=b[C["x"]] - b["position_noise_x"], _y=b[C["y"]] - b["position_noise_y"])
    else:
        b = b.assign(_x=b[C["x"]], _y=b[C["y"]])

    if "speed_noise" in b.columns:                        # 참속도 = 보고 - 노이즈
        b = b.assign(_v=b[C["speed"]] - b["speed_noise"])
    else:
        b = b.assign(_v=b[C["speed"]])
    for c, src in (("_nx", "position_noise_x"), ("_ny", "position_noise_y"),
                   ("_nv", "speed_noise"), ("_nh", "heading_noise")):
        b = b.assign(**{c: b[src] if src in b.columns else 0.0})

    out = {}
    for sid, g in b.groupby(C["sender"], sort=False):
        if len(g) < min_len:
            continue
        dt = np.diff(g[C["t"]].to_numpy())
        if np.nanmax(np.abs(dt - np.median(dt))) > 0.2:    # 결측 구간이 있는 시퀀스 제외
            continue
        out[sid] = dict(t=g[C["t"]].to_numpy(float), x=g["_x"].to_numpy(float),
                        y=g["_y"].to_numpy(float), v=g["_v"].to_numpy(float),
                        h=g["_hm"].to_numpy(float),
                        noise=dict(x=g["_nx"].to_numpy(float), y=g["_ny"].to_numpy(float),
                                   v=g["_nv"].to_numpy(float),
                                   h=np.radians(g["_nh"].to_numpy(float))))
    log(f"{split} benign 송신자 {len(out)}명 로드 (길이 >= {min_len} 메시지)")
    if max_senders and len(out) > max_senders:
        keys = np.random.default_rng(seed).choice(list(out), max_senders, replace=False)
        out = {k: out[k] for k in keys}
    return out


def save(df: pd.DataFrame, name: str):
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / name, index=False)
    log(f"저장: results/{name}")
