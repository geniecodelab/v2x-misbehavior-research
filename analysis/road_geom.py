"""
road_geom.py: 꺾은선(SUMO) 차로를 평활 곡선으로 재구성하고 접선 방향·곡률을 계산하는 공용 모듈.

방법: 각 차로를 1 m 간격으로 재표본화 -> x(s), y(s)에 가우시안 평활(표준편차 sigma_m)
      -> 수치 미분으로 접선각 theta(s)와 곡률 kappa(s) = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5
sigma_m = 0 이면 평활 없이 꺾은선 그대로(곡률은 꼭짓점에서만 0이 아님).
numpy/shapely만 사용(scipy 불필요).

주의: 평활은 차로 단위로 수행하므로 차로 양 끝(교차로 연결부)에서는 곡률이 과소추정될 수 있다.
"""
from __future__ import annotations

import numpy as np
import shapely

import f5_map_check as f

STEP_M = 1.0


def _gauss_kernel(sigma):
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), r


def build_smooth(m, sigma_m: float):
    """모든 차로에 대해 1 m 간격 접선각·곡률 배열을 만든다."""
    lines = m["lines"]
    lengths = shapely.length(lines)
    counts = np.maximum(np.ceil(lengths / STEP_M).astype(int) + 1, 2)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    theta = np.empty(offsets[-1])
    kappa = np.empty(offsets[-1])
    if sigma_m > 0:
        kern, r = _gauss_kernel(sigma_m / STEP_M)
    for i, (ln, L, n) in enumerate(zip(lines, lengths, counts)):
        s = np.linspace(0.0, L, n)
        pts = shapely.line_interpolate_point(ln, s)
        x, y = shapely.get_x(pts), shapely.get_y(pts)
        if sigma_m > 0 and n > 2:
            x = np.convolve(np.pad(x, r, mode="reflect", reflect_type="odd"), kern, mode="valid")
            y = np.convolve(np.pad(y, r, mode="reflect", reflect_type="odd"), kern, mode="valid")
        ds = max(L / (n - 1), 1e-3) if n > 1 else STEP_M
        dx, dy = np.gradient(x, ds), np.gradient(y, ds)
        ddx, ddy = np.gradient(dx, ds), np.gradient(dy, ds)
        theta[offsets[i]:offsets[i + 1]] = np.arctan2(dy, dx)
        kappa[offsets[i]:offsets[i + 1]] = (dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9)
    return dict(sigma_m=sigma_m, offsets=offsets, counts=counts, lengths=lengths, theta=theta, kappa=kappa)


def lookup(m, sm, li, px, py):
    """차로 li 위 (px,py) 투영점의 평활 접선각(rad)과 곡률(1/m)."""
    s = shapely.line_locate_point(m["lines"][li], shapely.points(px, py))
    n = sm["counts"][li]
    idx = np.clip(np.rint(s / np.maximum(sm["lengths"][li], 1e-9) * (n - 1)).astype(int), 0, n - 1)
    j = sm["offsets"][li] + idx
    return sm["theta"][j], sm["kappa"][j]


def nearest_same_direction_lane(m, px, py, heading_math, radius=10.0, max_dev_deg=30.0):
    """각 점에서 radius 이내, 진행 방향이 max_dev_deg 이내인 차로 중 가장 가까운 차로 인덱스(-1: 없음)."""
    n = len(px)
    pts = shapely.points(px, py)
    pi, li = m["tree"].query(pts, predicate="dwithin", distance=radius)
    out = np.full(n, -1)
    if len(pi) == 0:
        return out
    lh = f.lane_heading_at(m, li, px[pi], py[pi])
    ok = np.abs(f.wrap_angle(heading_math[pi] - lh)) < np.radians(max_dev_deg)
    pi, li = pi[ok], li[ok]
    if len(pi) == 0:
        return out
    d = shapely.distance(pts[pi], m["lines"][li])
    order = np.lexsort((d, pi))
    first = order[np.unique(pi[order], return_index=True)[1]]
    out[pi[first]] = li[first]
    return out
