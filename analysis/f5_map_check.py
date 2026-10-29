#!/usr/bin/env python3
"""
F5: 지도 정합성 기반 위치 위조 탐지 (VeReMi NextGen / InTAS)

단계
  1. build-map : SUMO net.xml -> 차로 중심선 / 주행 가능 영역(pickle)
  2. align     : 데이터 좌표계와 도로망 좌표계의 정합 검증(+필요 시 평행이동 보정)
  3. score     : 메시지별 도로 이탈 거리(d_off), 주행 방향 오차(heading_err) 계산
                 (헤딩 표기 규약은 train benign으로 자동 보정)
  4. evaluate  : validation benign으로 임계값 선정 -> test 평가,
                 오프셋의 종/횡 방향 분해 분석, 데이터셋 감사

사용 예
  python f5_map_check.py build-map --net ingolstadt.net.xml --data features.parquet --out map.pkl
  python f5_map_check.py align     --map map.pkl --data features.parquet --out transform.json
  python f5_map_check.py score     --map map.pkl --data features.parquet --transform transform.json --out scored.parquet
  python f5_map_check.py evaluate  --map map.pkl --scored scored.parquet --transform transform.json --out-dir results

의존성: numpy, pandas, shapely>=2.0, (parquet 입출력 시) pyarrow
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree
from shapely.geometry import LineString, Polygon, box

# ---------------------------------------------------------------------------
# 공통 설정 (데이터 컬럼명이 다르면 여기만 수정)
# ---------------------------------------------------------------------------
COL = dict(
    split="split",
    variant="dataset_variant",
    label="label",            # 0 = benign, 1 = attack
    receiver="receiver_id",
    sender="sender_id",
    msg="message_id",
    t="timestamp",
    x="pos_x",
    y="pos_y",
    speed="speed",
    hsin="heading_sin",
    hcos="heading_cos",
    road_edge="distance_to_road_edge",
)
BENIGN_VARIANT = "benign"
SEQ_KEY = [COL["receiver"], COL["sender"], COL["variant"]]   # 논문 5.2절 시퀀스 키
MATCH_KEY = [COL["split"], COL["receiver"], COL["sender"], COL["msg"], COL["t"]]
DEFAULT_LANE_WIDTH = 3.2      # SUMO 기본 차로 폭
MIN_SPEED_FOR_HEADING = 1.0   # m/s, 정지 차량은 헤딩 신뢰 불가
VEHICLE_CLASS = "passenger"


def log(*a):
    print("[f5]", *a, file=sys.stderr, flush=True)


def read_table(path: str | Path, columns=None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=columns)
    else:
        df = pd.read_csv(path, usecols=columns)
    if COL["split"] in df.columns:
        df[COL["split"]] = df[COL["split"]].astype(str).str.lower()
    return df


def write_table(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def wrap_angle(a):
    """[-pi, pi) 로 감기"""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# 1. 도로망 파싱
# ---------------------------------------------------------------------------
def _parse_shape(s: str) -> list[tuple[float, float]]:
    return [tuple(map(float, p.split(",")[:2])) for p in s.split()]


def _allows_vehicle(allow: str | None, disallow: str | None) -> bool:
    if allow is not None:
        toks = allow.split()
        return "all" in toks or VEHICLE_CLASS in toks
    if disallow is not None:
        toks = disallow.split()
        return "all" not in toks and VEHICLE_CLASS not in toks
    return True


def build_map(net_path: str, bbox: tuple[float, float, float, float] | None, margin: float):
    """net.xml -> dict(lines, widths, lane_ids, area, location)"""
    log(f"net 파싱: {net_path}")
    type_width, type_allow = {}, {}
    lines, widths, ids = [], [], []
    junction_polys = []
    location = {}
    clip = None
    if bbox is not None:
        clip = box(bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)

    for _, el in ET.iterparse(net_path, events=("end",)):
        tag = el.tag
        if tag == "location":
            location = dict(el.attrib)
        elif tag == "type":
            if "width" in el.attrib:
                type_width[el.get("id")] = float(el.get("width"))
            type_allow[el.get("id")] = (el.get("allow"), el.get("disallow"))
        elif tag == "edge":
            etype = el.get("type")
            t_allow, t_dis = type_allow.get(etype, (None, None))
            for ln in el.findall("lane"):
                allow = ln.get("allow", t_allow)
                dis = ln.get("disallow", t_dis)
                if not _allows_vehicle(allow, dis):
                    continue
                pts = _parse_shape(ln.get("shape", ""))
                if len(pts) < 2:
                    continue
                geom = LineString(pts)
                if clip is not None and not geom.intersects(clip):
                    continue
                w = float(ln.get("width", type_width.get(etype, DEFAULT_LANE_WIDTH)))
                lines.append(geom)
                widths.append(w)
                ids.append(ln.get("id"))
            el.clear()
        elif tag == "junction":
            if el.get("type") not in ("internal", "dead_end") and el.get("shape"):
                pts = _parse_shape(el.get("shape"))
                if len(pts) >= 3:
                    poly = Polygon(pts)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if clip is None or poly.intersects(clip):
                        junction_polys.append(poly)
            el.clear()

    lines_arr = np.array(lines, dtype=object)
    widths_arr = np.array(widths)
    log(f"차량 통행 차로 {len(lines_arr)}개, 교차로 폴리곤 {len(junction_polys)}개")
    lane_polys = shapely.buffer(lines_arr, widths_arr / 2.0, cap_style="flat", join_style="round")
    area = shapely.union_all(np.concatenate([lane_polys, np.array(junction_polys, dtype=object)]))
    log(f"주행 가능 영역 면적 {area.area / 1e6:.2f} km^2")
    return dict(lines=lines_arr, widths=widths_arr, lane_ids=np.array(ids),
                area=area, location=location)


def cmd_build_map(args):
    bbox = None
    if args.data:
        df = read_table(args.data, columns=[COL["x"], COL["y"]])
        bbox = (df[COL["x"]].min(), df[COL["y"]].min(), df[COL["x"]].max(), df[COL["y"]].max())
        log(f"데이터 범위 x[{bbox[0]:.0f},{bbox[2]:.0f}] y[{bbox[1]:.0f},{bbox[3]:.0f}]")
    m = build_map(args.net, bbox, args.margin)
    loc = m["location"]
    if loc:
        log(f"net location: convBoundary={loc.get('convBoundary')} netOffset={loc.get('netOffset')}")
        if bbox is not None and "convBoundary" in loc:
            cb = list(map(float, loc["convBoundary"].split(",")))
            inside = cb[0] <= bbox[0] and bbox[2] <= cb[2] and cb[1] <= bbox[1] and bbox[3] <= cb[3]
            log(f"데이터 범위가 convBoundary 안에 {'있음 -> SUMO 로컬 좌표일 가능성 높음' if inside else '없음 -> 좌표 변환 필요'}")
    with open(args.out, "wb") as f:
        pickle.dump(m, f)
    log(f"저장: {args.out}")


def load_map(path):
    with open(path, "rb") as f:
        m = pickle.load(f)
    m["tree"] = STRtree(m["lines"])
    shapely.prepare(m["area"])
    return m


# ---------------------------------------------------------------------------
# 2. 좌표 정합
# ---------------------------------------------------------------------------
def load_transform(path):
    if not path:
        return dict(flip_y=False, y_ref=0.0, tx=0.0, ty=0.0)
    with open(path) as f:
        return json.load(f)


def apply_transform(x, y, T):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if T["flip_y"]:
        y = T["y_ref"] - y
    return x + T["tx"], y + T["ty"]


def transform_vector(dx, dy, T):
    """평행이동은 벡터에 영향 없음, 반전만 반영"""
    return dx, (-dy if T["flip_y"] else dy)


def _nearest_on_lines(m, px, py):
    pts = shapely.points(px, py)
    idx = m["tree"].query_nearest(pts, return_distance=False, all_matches=False)
    # idx: [point_idx, line_idx]
    order = np.argsort(idx[0])
    li = idx[1][order]
    near = shapely.shortest_line(pts, m["lines"][li])
    end = shapely.get_point(near, 1)
    return shapely.get_x(end), shapely.get_y(end), li


def fit_quality(m, px, py):
    nx, ny, _ = _nearest_on_lines(m, px, py)
    d_center = np.hypot(nx - px, ny - py)
    on_road = shapely.contains_xy(m["area"], px, py)
    return dict(median_center_dist=float(np.median(d_center)),
                p90_center_dist=float(np.quantile(d_center, 0.9)),
                on_road_frac=float(on_road.mean()))


def icp_translation(m, px, py, T, iters=15):
    T = dict(T)
    for _ in range(iters):
        qx, qy = apply_transform(px, py, T)
        nx, ny, _ = _nearest_on_lines(m, qx, qy)
        dx, dy = np.median(nx - qx), np.median(ny - qy)
        T["tx"] += dx
        T["ty"] += dy
        if np.hypot(dx, dy) < 0.05:
            break
    return T


def cmd_align(args):
    m = load_map(args.map)
    df = read_table(args.data)
    b = df[(df[COL["variant"]] == BENIGN_VARIANT) & (df[COL["split"]] == args.fit_split)]
    if len(b) == 0:
        sys.exit(f"'{args.fit_split}' split의 benign 행이 없습니다. --fit-split 확인")
    b = b.sample(min(args.sample, len(b)), random_state=0)
    px, py = b[COL["x"]].to_numpy(float), b[COL["y"]].to_numpy(float)

    cb = m["area"].bounds
    candidates = {
        "identity": dict(flip_y=False, y_ref=0.0, tx=0.0, ty=0.0),
        # Veins 방식(y축 반전) 가설: 데이터 y 중심을 도로망 y 중심에 맞춘 뒤 ICP
        "flip_y": dict(flip_y=True, y_ref=float(py.mean() + (cb[1] + cb[3]) / 2), tx=0.0, ty=0.0),
    }
    report = {}
    for name, T0 in candidates.items():
        q0 = fit_quality(m, *apply_transform(px, py, T0))
        T1 = icp_translation(m, px, py, T0)
        q1 = fit_quality(m, *apply_transform(px, py, T1))
        report[name] = dict(initial=q0, refined=q1, transform=T1)
        log(f"[{name}] 초기 {q0} -> ICP 후 {q1}, 이동량 ({T1['tx']:.2f}, {T1['ty']:.2f})")

    best = min(report, key=lambda k: report[k]["refined"]["median_center_dist"])
    T = report[best]["transform"]
    shift = float(np.hypot(T["tx"], T["ty"]))
    q = report[best]["refined"]
    log(f"선택: {best}, ICP 이동량 {shift:.2f} m")
    if shift >= args.snap:
        log(f"경고: ICP 이동량 {shift:.2f} m. 직선 도로에서는 종방향 이동이 관측 불가능하고(논문 예측 1) "
            "평행 차로로 잘못 수렴할 수 있으므로, 이 보정값을 그대로 신뢰하지 마세요. "
            "항등 변환 품질이 나쁘다면 좌표계 정의(netOffset, Veins 변환)를 먼저 확인하는 것이 우선입니다.")
    if not T["flip_y"] and shift < args.snap:
        log(f"이동량 {shift:.2f} m < {args.snap} m -> 항등 변환으로 고정(노이즈 수준)")
        T = candidates["identity"]
    if q["median_center_dist"] > 3.0 or q["on_road_frac"] < 0.9:
        log("경고: 정합 품질이 낮습니다. 도로망 버전 불일치 또는 다른 좌표계 가능성. 결과 해석 주의.")
    with open(args.out, "w") as f:
        json.dump(dict(T, _report=report, _selected=best), f, indent=2)
    log(f"저장: {args.out}")


# ---------------------------------------------------------------------------
# 3. 메시지별 점수
# ---------------------------------------------------------------------------
def lane_heading_at(m, li, px, py):
    """차로 li 위 (px,py)의 투영점에서 진행 방향(수학 각도, rad). SUMO shape는 주행 방향 순서."""
    lines = m["lines"][li]
    pts = shapely.points(px, py)
    s = shapely.line_locate_point(lines, pts)
    L = shapely.length(lines)
    a = shapely.line_interpolate_point(lines, np.clip(s - 1.0, 0, L))
    b = shapely.line_interpolate_point(lines, np.clip(s + 1.0, 0, L))
    return np.arctan2(shapely.get_y(b) - shapely.get_y(a), shapely.get_x(b) - shapely.get_x(a))


def heading_error(m, px, py, heading_math, tol_m=1.0):
    """보고 위치에서 (차로 반폭 + tol_m) 이내의 차로들 중 최소 방향 오차. 해당 차로 없으면 NaN."""
    n = len(px)
    pts = shapely.points(px, py)
    maxw = float(m["widths"].max())
    pi, li = m["tree"].query(pts, predicate="dwithin", distance=maxw / 2 + tol_m)
    # 차로별 폭으로 재필터
    d = shapely.distance(pts[pi], m["lines"][li])
    keep = d <= m["widths"][li] / 2 + tol_m
    pi, li = pi[keep], li[keep]
    err = np.full(n, np.nan)
    if len(pi) == 0:
        return err
    lh = lane_heading_at(m, li, px[pi], py[pi])
    e = np.abs(wrap_angle(heading_math[pi] - lh))
    tmp = np.full(n, np.inf)
    np.minimum.at(tmp, pi, e)
    err[np.isfinite(tmp)] = tmp[np.isfinite(tmp)]
    return err


# 헤딩 규약 후보: math = s * a + k * pi/2  (a = atan2(heading_sin, heading_cos))
#   s=+1,k=0 : 이미 수학 각도(+x 기준 반시계)
#   s=-1,k=1 : 나침 각도(북 기준 시계, SUMO/TraCI 규약)
HEADING_CANDIDATES = [(s, k) for s in (1, -1) for k in range(4)]


def heading_math_from(df, s, k, T):
    a = np.arctan2(df[COL["hsin"]].to_numpy(float), df[COL["hcos"]].to_numpy(float))
    h = s * a + k * np.pi / 2
    if T["flip_y"]:
        h = -h
    return wrap_angle(h)


def calibrate_heading(m, df, T, sample=20000):
    b = df[(df[COL["variant"]] == BENIGN_VARIANT) & (df[COL["speed"]].abs() > 3.0)]
    b = b.sample(min(sample, len(b)), random_state=0)
    px, py = apply_transform(b[COL["x"]], b[COL["y"]], T)
    nx, ny, li = _nearest_on_lines(m, px, py)
    lh = lane_heading_at(m, li, px, py)
    res = {}
    for s, k in HEADING_CANDIDATES:
        h = heading_math_from(b, s, k, T)
        res[(s, k)] = float(np.median(np.abs(wrap_angle(h - lh))))
    best = min(res, key=res.get)
    log("헤딩 규약 후보별 median 오차(deg): " +
        ", ".join(f"s={s:+d},k={k}:{np.degrees(v):.1f}" for (s, k), v in res.items()))
    if np.degrees(res[best]) > 20:
        log("경고: 어떤 규약도 도로 방향과 잘 맞지 않습니다. heading 컬럼 정의를 확인하세요.")
    return best, res


def cmd_score(args):
    m = load_map(args.map)
    T = load_transform(args.transform)
    df = read_table(args.data)
    fit = df[df[COL["split"]] == args.fit_split]
    (s, k), _ = calibrate_heading(m, fit, T)
    log(f"헤딩 규약 선택: s={s}, k={k}")

    px, py = apply_transform(df[COL["x"]], df[COL["y"]], T)
    log(f"d_off 계산 중 ({len(df)}행)")
    pts = shapely.points(px, py)
    df["f5_d_off"] = shapely.distance(m["area"], pts)
    log("heading_err 계산 중")
    hm = heading_math_from(df, s, k, T)
    herr = heading_error(m, px, py, hm, tol_m=args.head_tol_m)
    herr[df[COL["speed"]].abs().to_numpy() < MIN_SPEED_FOR_HEADING] = np.nan
    df["f5_heading_err"] = herr
    df.attrs["heading_convention"] = f"s={s},k={k}"
    write_table(df, args.out)
    with open(Path(args.out).with_suffix(".meta.json"), "w") as f:
        json.dump(dict(heading_s=s, heading_k=k, transform=args.transform, head_tol_m=args.head_tol_m), f)
    log(f"저장: {args.out}")


# ---------------------------------------------------------------------------
# 4. 평가
# ---------------------------------------------------------------------------
def add_windowed(df, window):
    """시퀀스별 인과적(과거만) 이동 중앙값. window=1이면 원값."""
    df = df.sort_values(SEQ_KEY + [COL["t"]]).reset_index(drop=True)
    for c in ("f5_d_off", "f5_heading_err"):
        if window <= 1:
            df[c + "_w"] = df[c]
        else:
            df[c + "_w"] = (df.groupby(SEQ_KEY, sort=False)[c]
                              .transform(lambda s: s.rolling(window, min_periods=1).median()))
    return df


def select_thresholds(df, split, fpr, min_eps=0.0, min_theta_deg=0.0, head_onroad_only=False):
    v = df[(df[COL["split"]] == split) & (df[COL["label"]] == 0)]
    if len(v) == 0:
        sys.exit(f"'{split}' split의 benign 행이 없습니다. --threshold-split 확인")
    eps = float(np.nanquantile(v["f5_d_off_w"], 1 - fpr / 2))
    hv = v.loc[v["f5_d_off"] == 0, "f5_heading_err_w"] if head_onroad_only else v["f5_heading_err_w"]
    theta = float(np.nanquantile(hv.dropna(), 1 - fpr / 2))
    return max(eps, min_eps), max(theta, np.radians(min_theta_deg))


def apply_flags(df, eps, theta, base_flag_col=None, head_onroad_only=False):
    off = df["f5_d_off_w"] > eps
    gate = (df["f5_d_off"] == 0) if head_onroad_only else (df["f5_d_off_w"] <= eps)
    head = gate & (df["f5_heading_err_w"] > theta)  # NaN 비교는 False
    df["f5_flag_off"] = off
    df["f5_flag_head"] = head
    df["f5_flag"] = off | head
    if base_flag_col:
        df["combined_flag"] = df[base_flag_col].astype(bool) | df["f5_flag"]
    return df


def per_variant(df, flag):
    rows = []
    for v, g in df.groupby(COL["variant"]):
        rate = g[flag].mean()
        rows.append(dict(variant=v, n=len(g),
                         metric="FPR" if v == BENIGN_VARIANT else "Recall", value=rate))
    return pd.DataFrame(rows)


def overall(df, flag):
    y = df[COL["label"]].astype(int).to_numpy()
    p = df[flag].astype(bool).to_numpy()
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(flag=flag, precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12), fpr=fp / max(fp + tn, 1))


def decompose_offsets(m, df, T, variants, eps=0.0):
    """공격 메시지를 benign과 매칭해 b = p_att - p_ben 을 차로 접선/법선 성분으로 분해 (분석 전용)."""
    ben = df[df[COL["variant"]] == BENIGN_VARIANT][MATCH_KEY + [COL["x"], COL["y"]]]
    ben = ben.drop_duplicates(MATCH_KEY)
    att = df[df[COL["variant"]].isin(variants)]
    j = att.merge(ben, on=MATCH_KEY, suffixes=("", "_ben"), how="inner")
    log(f"매칭: 공격 {len(att)}행 중 {len(j)}행 ({len(j) / max(len(att), 1):.1%})")
    if len(j) == 0:
        return j
    bx, by = apply_transform(j[COL["x"] + "_ben"], j[COL["y"] + "_ben"], T)
    dx, dy = transform_vector((j[COL["x"]] - j[COL["x"] + "_ben"]).to_numpy(),
                              (j[COL["y"]] - j[COL["y"] + "_ben"]).to_numpy(), T)
    _, _, li = _nearest_on_lines(m, bx, by)
    h = lane_heading_at(m, li, bx, by)
    tx, ty = np.cos(h), np.sin(h)            # 접선
    nx, ny = -ty, tx                         # 법선(좌측)
    j["b_mag"] = np.hypot(dx, dy)
    j["b_par"] = dx * tx + dy * ty
    j["b_perp"] = dx * nx + dy * ny
    # 이론 예측(논문 예측 1·2): 성분 하나만 적용했을 때 판정과 같은 허용오차 eps를 넘어 도로를 벗어나는가
    def off(px, py):
        return shapely.distance(m["area"], shapely.points(px, py)) > eps
    j["pred_offroad_perp"] = off(bx + j["b_perp"].to_numpy() * nx, by + j["b_perp"].to_numpy() * ny)
    j["pred_offroad_par"] = off(bx + j["b_par"].to_numpy() * tx, by + j["b_par"].to_numpy() * ty)
    j["pred_offroad_full"] = off(bx + dx, by + dy)
    return j


def cmd_evaluate(args):
    m = load_map(args.map)
    T = load_transform(args.transform)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_table(args.scored)
    df = add_windowed(df, args.window)
    eps, theta = select_thresholds(df, args.threshold_split, args.fpr, args.min_eps, args.min_theta_deg,
                                   args.head_onroad_only)
    if args.theta_deg is not None:
        theta = float(np.radians(args.theta_deg))
    log(f"임계값({args.threshold_split} benign, 목표 FPR {args.fpr}): eps={eps:.2f} m, theta={np.degrees(theta):.1f} deg")
    df = apply_flags(df, eps, theta, args.base_flag_col, args.head_onroad_only)
    test = df[df[COL["split"]] == args.eval_split].copy()

    flags = ["f5_flag_off", "f5_flag_head", "f5_flag"]
    if args.base_flag_col:
        flags = [args.base_flag_col] + flags + ["combined_flag"]
    pd.DataFrame([overall(test, f) for f in flags]).to_csv(out / "overall.csv", index=False)
    pv = pd.concat([per_variant(test, f).assign(flag=f) for f in flags])
    pv.pivot_table(index="variant", columns="flag", values="value").to_csv(out / "per_variant.csv")
    log("공격 유형별 결과:\n" + pv.pivot_table(index="variant", columns="flag", values="value").round(3).to_string())

    # --- 오프셋 분해 (예측 1·2 검증)
    variants = [v for v in test[COL["variant"]].unique() if "PositionOffset" in str(v)]
    dec = decompose_offsets(m, test, T, variants, eps=eps)
    if len(dec):
        if args.save_parquet:
            dec.to_parquet(out / "offset_decomposition.parquet", index=False)
        # 일관성 검사: constant offset은 송신자별 b가 일정해야 함
        cpo = dec[dec[COL["variant"]].str.contains("constant", case=False)]
        if len(cpo):
            raw = cpo.assign(bx=cpo[COL["x"]] - cpo[COL["x"] + "_ben"], by=cpo[COL["y"]] - cpo[COL["y"] + "_ben"])
            sd = raw.groupby(COL["sender"])[["bx", "by"]].std().max(axis=1)
            log(f"constantPositionOffset 송신자별 b 표준편차: median {sd.median():.4f} m, max {sd.max():.4f} m "
                "(≈0이면 benign이 참값 대용으로 유효)")
        bins = [0, 1, 2, 3, 4, 6, 8, 12, 20, 40, np.inf]
        dec["abs_b_perp_bin"] = pd.cut(dec["b_perp"].abs(), bins, right=False)
        dec["abs_b_par_bin"] = pd.cut(dec["b_par"].abs(), bins, right=False)
        agg = dict(n=("f5_flag", "size"), n_senders=(COL["sender"], "nunique"),
                   recall=("f5_flag", "mean"),
                   recall_off=("f5_flag_off", "mean"),
                   pred_offroad_perp=("pred_offroad_perp", "mean"),
                   pred_offroad_full=("pred_offroad_full", "mean"))
        g_perp = dec.groupby([COL["variant"], "abs_b_perp_bin"], observed=True).agg(**agg)
        g_perp.to_csv(out / "recall_by_abs_b_perp.csv")
        small = dec[dec["b_perp"].abs() < args.small_perp]
        g_par = small.groupby([COL["variant"], "abs_b_par_bin"], observed=True).agg(
            **{k: v for k, v in agg.items() if k != "pred_offroad_perp"},
            pred_offroad_par=("pred_offroad_par", "mean"))
        g_par.to_csv(out / f"recall_by_abs_b_par_perp_lt{args.small_perp:g}m.csv")
        log("횡방향 |b_perp| 구간별:\n" + g_perp.round(3).to_string())
        log(f"|b_perp|<{args.small_perp} m 샘플의 종방향 |b_par| 구간별:\n" + g_par.round(3).to_string())
        ct = pd.crosstab([dec[COL["variant"]], dec["pred_offroad_full"]], dec["f5_flag_off"], normalize="index")
        ct.to_csv(out / "prediction_vs_detection.csv")
        log(f"이론 예측(eps={eps:.2f} m 기준 도로 이탈) vs 실제 도로이탈 플래그 (행 정규화):\n" + ct.round(3).to_string()
            + ("\n  주의: window>1이면 random 계열은 과거 메시지 중앙값으로 판정되므로 현재 메시지 예측과 직접 대응하지 않음"
               if args.window > 1 else ""))

    # --- 데이터셋 감사: distance_to_road_edge
    if COL["road_edge"] in test.columns:
        re = test.groupby(COL["variant"])[COL["road_edge"]].agg(
            n="size", nonzero_frac=lambda s: (s != 0).mean(), mean="mean", max="max")
        re.to_csv(out / "audit_distance_to_road_edge.csv")
        log("distance_to_road_edge 감사:\n" + re.round(3).to_string())
        nz = test[test[COL["road_edge"]] != 0]
        if len(nz) > 10:
            r = np.corrcoef(nz[COL["road_edge"]], nz["f5_d_off"])[0, 1]
            log(f"비영(非0) 행에서 distance_to_road_edge vs f5_d_off 상관계수 r={r:.3f} "
                "(높으면 우리 d_off 구현의 교차검증 근거. 단 이 필드는 특징으로 쓰지 말 것)")

    with open(out / "thresholds.json", "w") as f:
        json.dump(dict(eps_m=eps, theta_deg=float(np.degrees(theta)), window=args.window,
                       head_onroad_only=args.head_onroad_only,
                       fpr_target=args.fpr, threshold_split=args.threshold_split), f, indent=2)
    log(f"결과 저장: {out}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-map")
    p.add_argument("--net", required=True)
    p.add_argument("--data", help="데이터 범위로 도로망을 잘라 속도 향상")
    p.add_argument("--margin", type=float, default=500.0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_build_map)

    p = sub.add_parser("align")
    p.add_argument("--map", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--fit-split", default="train")
    p.add_argument("--sample", type=int, default=20000)
    p.add_argument("--snap", type=float, default=0.5, help="이 값 미만의 이동은 항등 변환으로 간주(m)")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_align)

    p = sub.add_parser("score")
    p.add_argument("--map", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--transform")
    p.add_argument("--fit-split", default="train", help="헤딩 규약 보정에 쓸 split")
    p.add_argument("--head-tol-m", type=float, default=1.0,
                   help="방향 검사 반경 = 차로 반폭 + 이 값(m). 위치 오차가 크면 늘릴 것")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("evaluate")
    p.add_argument("--map", required=True)
    p.add_argument("--scored", required=True)
    p.add_argument("--transform")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fpr", type=float, default=0.01, help="F5 단독 목표 FPR (도로이탈/방향에 절반씩 배분)")
    p.add_argument("--window", type=int, default=1, help="시퀀스별 인과적 이동 중앙값 창 크기")
    p.add_argument("--threshold-split", default="validation")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--base-flag-col", help="기존 Rule+CUSUM 판정 컬럼(있으면 OR 결합 성능도 계산)")
    p.add_argument("--small-perp", type=float, default=1.0, help="종방향 분석에 쓸 |b_perp| 상한(m)")
    p.add_argument("--min-eps", type=float, default=0.0, help="도로 이탈 임계값 하한(m)")
    p.add_argument("--min-theta-deg", type=float, default=10.0, help="방향 오차 임계값 하한(deg)")
    p.add_argument("--theta-deg", type=float, default=None,
                   help="방향 임계값을 분위수 대신 고정(예: 90 = 반대 방향 판정)")
    p.add_argument("--head-onroad-only", action="store_true",
                   help="방향 검사를 도로 폴리곤 내부(d_off=0) 메시지로 제한")
    p.add_argument("--save-parquet", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()