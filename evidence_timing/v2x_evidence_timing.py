#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v2x_evidence_timing.py
정보처리학회 원고 v18 "미래 상태 예측 기반 V2X 위치 위조의 증거는 언제 드러나는가"의 실험 코드.

무엇을 하는가
  1) 정상 송신자의 실제 궤적 복원 (보고값 - 데이터셋 측위 오차)
  2) 공격 메시지 생성
       - 이상적 공격자: tau초 뒤의 실제 상태를 보고
       - 현실적 공격자: n차(1=등속, 2=등가속, 3=등저크) 외삽으로 tau초 뒤를 예측해 보고
     원래 메시지의 측위 오차에 c배를 곱해 다시 더함 (측정 잡음 조절, c=1이 원래 조건)
  3) 식 (1) 위치-속도 불일치 r, 이상적 공격자 대비 불일치 D = r_attack - r_ideal
  4) 식 (2) 예측값 D_pred = dt * tau^n/n! * d(g_n)/dt
  5) 식 (3) 정보 한계 탐지 시각 k*: 감속 시작 후 sum D^2/(2 sigma^2) >= ln(1/alpha)
     식 (4) (1차만) k*_pred = (6 sigma^2 ln(1/alpha) / (dt tau j)^2)^(1/3)
  6) 실제 탐지기(위치-속도 일관성 검사, 5개 이동 평균, FPR 1% 임계값)의 탐지 지연
  7) H1/H2/H3 결과 표와 그림 저장

사용법
  합성 데이터로 흐름 확인:
    python v2x_evidence_timing.py --synthetic --out results_synth
  VeReMi NextGen:
    python v2x_evidence_timing.py --test <시험 분할 폴더> --val <검증 분할 폴더> --out results
  옵션은 --help 참고.

주의 (원고 메모와 대응)
  - 정상 불일치의 자기상관을 summary.json에 기록한다. 크면 식 (3)의 독립 근사가 약해진다.
  - 공격자는 지도(도로 형상)를 알아 예측한 호의 길이를 실제 경로 위 위치로 바꾼다고 가정한다
    (진행 방향 1차원 분석과 일치). --extrap heading 으로 직선 외삽도 가능.
  - 감속 사건의 시작은 "가속도가 -onset_a 아래로 내려가기 시작한 시각"이며,
    사건 판정은 가속도가 -brake_a(기본 4 m/s^2)를 넘은 경우로 한다.
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import font_manager as _fm
_names = [x.name for x in _fm.fontManager.ttflist]
for _key in ("NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK", "Noto Serif CJK"):
    _hit = [n for n in _names if _key in n]
    if _hit:
        # 한글 폰트에는 U+2212(수학 마이너스)가 없어 로그축 지수 라벨이 깨진다.
        # DejaVu Sans 를 대체 폰트로 덧붙이면 한글은 한글 폰트로, 기호는 DejaVu 로 찍힌다.
        plt.rcParams["font.family"] = [_hit[0], "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------------------
# 1. 데이터 읽기
# --------------------------------------------------------------------------------------
def _vec(v):
    if isinstance(v, str):
        v = [float(p) for p in v.split(",")]
    return np.atleast_1d(np.asarray(v, dtype=float))


_TIME_DIV = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}


def _detect_time_div(df):
    """sendTime의 시간 단위를 메시지 간격으로 추정한다.
    NextGen 배포본은 OMNeT++ simtime을 정수 문자열(InTAS 배포본은 ns)로 적어 두었으므로
    그대로 초로 읽으면 dt가 1e9초가 되어 모든 물리량이 망가진다.
    비콘 주기가 0.02~5초 사이에 들어오는 배율을 고른다."""
    d = df.sort_values(["sender", "t"]).groupby("sender")["t"].diff().dropna()
    d = d[d > 0]
    if d.empty:
        return 1.0, "s"
    med = float(d.median())
    for unit in ("s", "ms", "us", "ns", "ps"):
        if 0.02 <= med / _TIME_DIV[unit] <= 5.0:
            return _TIME_DIV[unit], unit
    return 1.0, "s"


def load_nextgen(folder, sender_key="sender_id", max_files=None, time_unit="auto"):
    """VeReMi NextGen 폴더(수신 차량별 JSON)를 읽어 송신자별 메시지 표를 만든다.
    같은 메시지가 여러 수신자에게 중복되므로 (송신자, sendTime)으로 중복 제거한다.
    한 번이라도 attacker=1인 송신자는 제외한다(정상 송신자만 사용).
    time_unit: sendTime의 단위. "auto"면 메시지 간격으로 추정한다."""
    rows = []
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"JSON 파일이 없습니다: {folder}")
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            msgs = json.load(f)
        for m in msgs:
            sd = m["sender"]
            pos, pn = _vec(sd["pos"]), _vec(sd["pos_noise"])
            spd, sn = _vec(sd["spd"]), _vec(sd["spd_noise"])
            rows.append(dict(
                sender=str(m.get(sender_key, m["sender_id"])),
                t=float(m["sendTime"]),
                x=pos[0], y=pos[1], nx=pn[0], ny=pn[1],
                spd=float(np.linalg.norm(spd)) if spd.size > 1 else float(spd[0]),
                nspd=float(sn[0]),
                attacker=int(m.get("attacker", 0)),
                profile=str(sd.get("driversProfile", "Normal")),
            ))
    df = pd.DataFrame(rows)
    div, unit = _detect_time_div(df) if time_unit == "auto" else (_TIME_DIV[time_unit], time_unit)
    if div != 1.0:
        df["t"] = (df["t"] - df["t"].min()) / div
    print(f"  {folder}: {len(files)} files, sendTime 단위={unit} (÷{div:g})")
    bad = set(df.loc[df.attacker == 1, "sender"])
    df = df[~df.sender.isin(bad)]
    df = df.drop_duplicates(["sender", "t"]).sort_values(["sender", "t"]).reset_index(drop=True)
    return df


def make_synthetic(n_senders=40, duration=600.0, seed=0, noise_std=2.88, rho=0.99,
                   white_pos=0.3, spd_noise=0.3):
    """흐름 확인용 합성 데이터 (실제 결과로 쓰지 않는다).
    순항(작고 느리게 변하는 가속도) → 급감속 → 원래 속도까지 재가속을 반복하는 저크 제한 운동을
    완만한 곡선 경로에 입히고, Gauss-Markov(AR(1)) 측위 편향 + 백색 잡음, 속도 잡음을 더한다.
    운전자 유형은 최대 저크로 흉내 낸다 (신중 4, 보통 8, 공격적 12 m/s^3)."""
    rng = np.random.default_rng(seed)
    rows = []
    dt_sim = 0.05
    tt = np.arange(0, duration, dt_sim)
    for k in range(n_senders):
        jmax = float(rng.choice([4.0, 8.0, 12.0]))
        profile = {4.0: "Cautious", 8.0: "Normal", 12.0: "Aggressive"}[jmax]
        v_cruise = rng.uniform(22, 32)
        v = np.empty_like(tt); v[0] = v_cruise
        a, mode, mode_end = 0.0, "cruise", 0.0
        target, next_event, next_drift = 0.0, rng.uniform(30, 60), 0.0
        for i in range(1, len(tt)):
            t = tt[i]
            if mode == "cruise":
                if t >= next_drift:
                    target = float(np.clip(rng.normal(0, 0.15) - 0.05 * (v[i - 1] - v_cruise), -0.4, 0.4))
                    next_drift = t + 5.0
                if t >= next_event:
                    mode, target, mode_end = "brake", -rng.uniform(4.5, 7.0), t + rng.uniform(1.5, 3.0)
            elif mode == "brake" and t >= mode_end:
                mode, target = "accel", rng.uniform(0.8, 1.5)
            elif mode == "accel" and v[i - 1] >= v_cruise:
                mode, target, next_event = "cruise", 0.0, t + rng.uniform(30, 70)
            a += float(np.clip(target - a, -jmax * dt_sim, jmax * dt_sim))
            v[i] = max(1.0, v[i - 1] + a * dt_sim)
        s = np.concatenate([[0], np.cumsum(v[1:] * dt_sim)])
        th = s / rng.uniform(800, 3000) * rng.choice([-1, 1])
        ds = np.r_[0, np.diff(s)]
        x, y = np.cumsum(np.cos(th) * ds), np.cumsum(np.sin(th) * ds)
        tm = np.arange(1.0, duration - 1.0, 1.0) + rng.uniform(-0.05, 0.05, len(np.arange(1.0, duration - 1.0, 1.0)))
        xs, ys, vs = np.interp(tm, tt, x), np.interp(tm, tt, y), np.interp(tm, tt, v)
        e = np.zeros((len(tm), 2)); sig_innov = noise_std * math.sqrt(1 - rho ** 2)
        e[0] = rng.normal(0, noise_std, 2)
        for i in range(1, len(tm)):
            e[i] = rho * e[i - 1] + rng.normal(0, sig_innov, 2)
        e += rng.normal(0, white_pos, e.shape)
        ns = rng.normal(0, spd_noise, len(tm))
        for i in range(len(tm)):
            rows.append(dict(sender=f"S{k}", t=tm[i], x=xs[i] + e[i, 0], y=ys[i] + e[i, 1],
                             nx=e[i, 0], ny=e[i, 1], spd=vs[i] + ns[i], nspd=ns[i],
                             attacker=0, profile=profile))
    return pd.DataFrame(rows).sort_values(["sender", "t"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 2. 실제 궤적 (진행 방향 호의 길이 s(t)와 경로 형상)
# --------------------------------------------------------------------------------------
class Trajectory:
    def __init__(self, g, smooth=0.5, min_len=30, max_dt_jitter=0.2):
        g = g.sort_values("t")
        self.t = g.t.to_numpy()
        self.x = (g.x - g.nx).to_numpy()          # 실제 위치 = 보고값 - 측위 오차
        self.y = (g.y - g.ny).to_numpy()
        self.nx, self.ny, self.nspd = g.nx.to_numpy(), g.ny.to_numpy(), g.nspd.to_numpy()
        self.profile = g.profile.iloc[0]
        self.ok = len(self.t) >= min_len and np.all(np.diff(self.t) > 0)
        # 결측 구간이 있는 송신자는 제외한다 (timewarp/common.py 의 load_benign 과 같은 기준).
        # 수신 차량이 통신 범위를 벗어나면 비콘이 수십 초씩 끊기는데, 그 구간의 직선 거리
        # |Δp| 는 곡선 경로 길이 v̄·Δt 보다 훨씬 짧아 식 (1)의 r 이 수백 m 까지 벌어진다.
        # 그런 결측이 전체 간격의 1% 미만이어도 분산 통계인 sigma 를 완전히 망가뜨리므로
        # (InTAS highway_2 검증 분할: sigma 0.21 m -> 161 m) 반드시 걸러야 한다.
        if self.ok and max_dt_jitter is not None:
            d = np.diff(self.t)
            self.ok = bool(np.nanmax(np.abs(d - np.median(d))) <= max_dt_jitter)
        if not self.ok:
            return
        d = np.hypot(np.diff(self.x), np.diff(self.y))
        s = np.r_[0.0, np.cumsum(d)]
        # 5차 평활 스플라인: 4차 미분(저크의 변화율)까지 연속 -> 3차 예측기에도 사용 가능
        self.spl = UnivariateSpline(self.t, s, k=5, s=smooth * len(self.t))
        self.der = [self.spl.derivative(k) if k else self.spl for k in range(5)]
        # 경로 형상: 호의 길이 -> (x, y) (정지 구간 중복 제거)
        su, iu = np.unique(s, return_index=True)
        self.path_s, self.path_x, self.path_y = su, self.x[iu], self.y[iu]

    def s(self, tq, k=0):
        return self.der[k](tq)

    def xy_at_s(self, sq):
        sq = np.clip(sq, self.path_s[0], self.path_s[-1])
        return np.interp(sq, self.path_s, self.path_x), np.interp(sq, self.path_s, self.path_y)

    def heading_xy(self, tq):
        eps = 0.2
        x1, y1 = self.xy_at_s(self.s(tq - eps)); x2, y2 = self.xy_at_s(self.s(tq + eps))
        ang = np.arctan2(y2 - y1, x2 - x1)
        return np.cos(ang), np.sin(ang)


# --------------------------------------------------------------------------------------
# 3. 공격 메시지와 불일치
# --------------------------------------------------------------------------------------
def reported_state(tr, tq, tau, order, extrap="path"):
    """order = 0 이면 이상적 공격자(실제 미래), 1/2/3 이면 n차 예측.
    반환: 보고 위치 (x, y), 보고 속도 v"""
    if order == 0:
        sh = tr.s(tq + tau); vh = tr.s(tq + tau, 1)
        x, y = tr.xy_at_s(sh)
        return x, y, vh
    sh = sum(tr.s(tq, k) * tau ** k / math.factorial(k) for k in range(order + 1))
    vh = sum(tr.s(tq, k) * tau ** (k - 1) / math.factorial(k - 1) for k in range(1, order + 1))
    if extrap == "path":                     # 지도를 아는 공격자: 예측한 호의 길이를 경로 위로
        x, y = tr.xy_at_s(sh)
    else:                                    # 현재 진행 방향으로 직선 외삽
        cx, cy = tr.heading_xy(tq)
        x0, y0 = tr.xy_at_s(tr.s(tq))
        x, y = x0 + cx * (sh - tr.s(tq)), y0 + cy * (sh - tr.s(tq))
    return x, y, np.maximum(vh, 0.0)


def residual(x, y, v, t):
    """식 (1): r_t = |y_t - y_{t-1}| - (v_{t-1}+v_t)/2 * dt  (부호 유지, 첫 메시지는 nan)"""
    r = np.full(len(t), np.nan)
    d = np.hypot(np.diff(x), np.diff(y))
    r[1:] = d - 0.5 * (v[1:] + v[:-1]) * np.diff(t)
    return r


def moving_abs_mean(r, w):
    a = pd.Series(np.abs(r)).rolling(w, min_periods=w).mean().to_numpy()
    return a


def valid_mask(tr, tau):
    return (tr.t + tau <= tr.t[-1] - 0.5) & (tr.t >= tr.t[0] + 2.0)


def messages(tr, tau, order, c, extrap):
    """한 송신자에 대해 공격 메시지(측위 오차 c배 재주입)와 불일치를 계산"""
    m = valid_mask(tr, tau)
    t = tr.t[m]
    xa, ya, va = reported_state(tr, t, tau, order, extrap)
    xi, yi, vi = reported_state(tr, t, tau, 0, extrap)
    nx, ny, nv = c * tr.nx[m], c * tr.ny[m], c * tr.nspd[m]
    r_att = residual(xa + nx, ya + ny, va + nv, t)
    r_ide = residual(xi + nx, yi + ny, vi + nv, t)
    D = r_att - r_ide
    if order > 0:
        dt = np.r_[np.nan, np.diff(t)]
        tmid = t - 0.5 * np.nan_to_num(dt, nan=1.0)
        D_pred = dt * tau ** order / math.factorial(order) * tr.s(tmid, order + 1)
    else:
        D_pred = np.zeros_like(D)
    return t, r_att, D, D_pred


def benign_residuals(trajs, c):
    rs = []
    for tr in trajs:
        t = tr.t
        x, y, v = reported_state(tr, t, 0.0, 0)
        rs.append(residual(x + c * tr.nx, y + c * tr.ny, v + c * tr.nspd, t))
    return rs


# --------------------------------------------------------------------------------------
# 4. 급감속 사건
# --------------------------------------------------------------------------------------
def braking_events(tr, brake_a=4.0, onset_a=0.5, refractory=15.0, grid=0.05):
    """가속도가 -brake_a 아래로 내려간 사건마다 (감속 시작 시각, 판정 시각, 평균 저크)"""
    tg = np.arange(tr.t[0] + 3.0, tr.t[-1] - 3.0, grid)
    if len(tg) < 10:
        return []
    a = tr.s(tg, 2)
    ev, last = [], -1e9
    below = np.where((a[1:] < -brake_a) & (a[:-1] >= -brake_a))[0] + 1
    for i in below:
        tc = tg[i]
        if tc - last < refractory:
            continue
        j = i
        while j > 0 and a[j] < -onset_a:
            j -= 1
        t_on = tg[j]
        if tc - t_on <= 0:
            continue
        jerk = (-brake_a - a[j]) / (tc - t_on)     # 감속 시작~판정 구간의 평균 저크 (음수)
        ev.append(dict(t_on=t_on, t_cross=tc, jerk=abs(jerk)))
        last = tc
    return ev


def phase_labels(tr, t, a_thr=0.5, j_thr=0.5):
    a, j = tr.s(t, 2), tr.s(t, 3)
    lab = np.where(np.abs(j) >= j_thr, "transition", np.where(np.abs(a) >= a_thr, "sustained", "cruise"))
    return lab


# --------------------------------------------------------------------------------------
# 5. 실험 본체
# --------------------------------------------------------------------------------------
def genie_cusum(x, D, sigma):
    """공격 형태 D를 미리 아는 최적 탐지기(CUSUM). 관측 x(정상 평균 제거)가 정상이면 N(0,σ²),
    공격이면 N(D,σ²)라고 보고 로그 우도비를 누적한다. Lorden/Moustakides에 따라 같은 오탐 수준에서
    평균 지연이 가장 짧은 탐지기이므로, 이 지연을 '정보 한계'의 경험적 기준으로 쓴다.
    로그 우도비의 기댓값이 D²/(2σ²)이므로, 식 (3)의 k*는 이 탐지기 지연의 평균 경로 근사이다."""
    llr = (D * x - 0.5 * D ** 2) / sigma ** 2
    S, acc = np.zeros(len(x)), 0.0
    for i, l in enumerate(llr):
        acc = max(0.0, acc + (0.0 if not np.isfinite(l) else l))
        S[i] = acc
    return S


def value_at(tm, vals, t0):
    idx = np.where(tm <= t0)[0]
    return vals[idx[-1]] if len(idx) else np.nan


def benign_residual_at(tr, t, c):
    m = np.isin(tr.t, t)
    x, y, v = reported_state(tr, t, 0.0, 0)
    return residual(x + c * tr.nx[m], y + c * tr.ny[m], v + c * tr.nspd[m], t)


def first_cross_time(tm, vals, start, horizon, thr):
    sel = (tm > start) & (tm <= start + horizon)
    idx = np.where(sel & (vals >= thr))[0]
    return (tm[idx[0]] - start) if len(idx) else np.nan


def run(args):
    os.makedirs(args.out, exist_ok=True)
    if args.synthetic:
        df_test = make_synthetic(seed=1)
        df_val = make_synthetic(n_senders=20, seed=2)
    else:
        df_test = load_nextgen(args.test, args.sender_key, args.max_files, args.time_unit)
        df_val = load_nextgen(args.val, args.sender_key, args.max_files, args.time_unit) if args.val else None

    def build(df):
        jit = args.max_dt_jitter if args.max_dt_jitter > 0 else None
        trs = [Trajectory(g, smooth=args.smooth, max_dt_jitter=jit)
               for _, g in df.groupby("sender")]
        return [tr for tr in trs if tr.ok]

    test = build(df_test)
    if df_val is not None:
        val = build(df_val)
        val_note = "validation split"
    else:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(test))
        cut = int(0.3 * len(test))
        val, test = [test[i] for i in idx[:cut]], [test[i] for i in idx[cut:]]
        val_note = "WARNING: 30% of test senders held out (no --val given)"
    print(f"test senders={len(test)}, val senders={len(val)} ({val_note})")

    ln_inv_alpha = math.log(1.0 / args.alpha)
    summary = dict(val_note=val_note, alpha=args.alpha, window=args.window,
                   max_dt_jitter=args.max_dt_jitter,
                   n_senders=dict(test=len(test), val=len(val)), noise_scale={})
    ev_rows, h1_rows, rate_rows = [], [], []

    for c in args.noise_scale:
        # ---- 정상 불일치의 표준편차 sigma, 자기상관, 탐지기 임계값 (검증 분할) ----
        rb_val = benign_residuals(val, c)
        allr = np.concatenate([r[~np.isnan(r)] for r in rb_val])
        sigma = float(np.std(allr))
        mu_B = float(np.mean(allr))
        ac = {}
        for lag in (1, 2, 3):
            pairs = [(r[lag:], r[:-lag]) for r in rb_val if len(r) > lag + 5]
            a1 = np.concatenate([p[0] for p in pairs]); a0 = np.concatenate([p[1] for p in pairs])
            ok = ~np.isnan(a1) & ~np.isnan(a0)
            ac[lag] = float(np.corrcoef(a1[ok], a0[ok])[0, 1])
        ma_list = [moving_abs_mean(r, args.window) for r in rb_val]
        ma_all = np.concatenate(ma_list)
        eta_ma = float(np.nanquantile(ma_all, 1 - args.fpr))
        eta_raw = float(np.nanquantile(np.abs(allr), 1 - args.fpr))
        def arl(stats, thr):
            # 정상 메시지에서 경보가 새로 켜지는 평균 간격(메시지 수) = 경험적 평균 오탐 간격
            n_msg = sum(int(np.sum(~np.isnan(x))) for x in stats)
            n_on = sum(int(np.sum((np.nan_to_num(x, nan=-1)[1:] >= thr) & (np.nan_to_num(x, nan=-1)[:-1] < thr))) for x in stats)
            return float(n_msg / max(n_on, 1))
        summary["noise_scale"][str(c)] = dict(sigma=sigma, mean_benign_residual=mu_B, autocorr=ac, eta_ma=eta_ma, eta_raw=eta_raw,
                                              genie_threshold_ln_inv_alpha=ln_inv_alpha,
                                              empirical_ARL_messages=dict(
                                                  detector_moving_average=arl(ma_list, eta_ma),
                                                  detector_no_moving_average=arl([np.abs(r) for r in rb_val], eta_raw),
                                                  genie_lower_bound=1.0 / args.alpha))
        print(f"[c={c}] sigma={sigma:.3f} m, autocorr lag1={ac[1]:.2f}, eta_ma={eta_ma:.3f}, eta_raw={eta_raw:.3f}")

        # 시험 분할 정상 메시지의 실제 오탐률
        rb_test = benign_residuals(test, c)
        ma_t = np.concatenate([moving_abs_mean(r, args.window) for r in rb_test])
        summary["noise_scale"][str(c)]["test_fpr"] = float(np.nanmean(ma_t >= eta_ma))

        for order in [0] + list(args.orders):
            for tau in args.taus:
                det_all, det_brk = [], []
                # 최적 탐지기(CUSUM)의 임계값: h = ln(1/alpha) 이면 평균 오탐 간격이 1/alpha 메시지 이상 (Lorden)
                h_genie = ln_inv_alpha
                for si, tr in enumerate(test):
                    t, r, D, Dp = messages(tr, tau, order, c, args.extrap)
                    if len(t) < args.window + 2:
                        continue
                    ma = moving_abs_mean(r, args.window)
                    alarm = ma >= eta_ma
                    det_all.append(alarm[~np.isnan(ma)])
                    evs = braking_events(tr, args.brake_a, args.onset_a)
                    for e in evs:
                        w = (t >= e["t_cross"] - 3) & (t <= e["t_cross"] + 3) & ~np.isnan(ma)
                        det_brk.append(alarm[w])
                    # H1: 불일치의 원인 (현실적 공격자, c=1일 때만 저장)
                    if order > 0 and c == 1.0 and si % args.h1_every == 0:
                        lab = phase_labels(tr, t)
                        for i in range(1, len(t)):
                            if np.isnan(D[i]) or np.isnan(Dp[i]):
                                continue
                            h1_rows.append(dict(order=order, tau=tau, phase=lab[i],
                                                D=D[i], D_pred=Dp[i], D_over_sigma=D[i] / sigma))
                    # H2/H3: 사건별 정보 한계 탐지 시각과 실제 탐지 지연
                    if order == 0:
                        continue
                    info = D ** 2 / (2 * sigma ** 2)
                    info_p = Dp ** 2 / (2 * sigma ** 2)
                    for e in evs:
                        sel = (t > e["t_on"]) & (t <= e["t_on"] + args.horizon)
                        if sel.sum() < 2:
                            continue
                        tm = t[sel]
                        cum = np.nancumsum(info[sel]); cum_p = np.nancumsum(info_p[sel])
                        k_meas = first_cross_time(tm, cum, e["t_on"] - 1e-9, args.horizon, ln_inv_alpha)
                        k_eq2 = first_cross_time(tm, cum_p, e["t_on"] - 1e-9, args.horizon, ln_inv_alpha)
                        dtm = float(np.nanmedian(np.diff(t)))
                        k_eq4 = np.nan
                        if order == 1 and e["jerk"] > 0:
                            k_eq4 = (6 * sigma ** 2 * ln_inv_alpha / (dtm * tau * e["jerk"]) ** 2) ** (1 / 3) * dtm
                        d_ma = first_cross_time(t, np.nan_to_num(ma, nan=-1), e["t_on"], args.horizon, eta_ma)
                        # 최적 탐지기는 감속 시작 시점부터 증거를 누적한다 (식 (3)의 확률적 버전)
                        sel_g = t > e["t_on"]
                        S_g = genie_cusum(r[sel_g] - mu_B, np.nan_to_num(D[sel_g]), sigma)
                        d_genie = first_cross_time(t[sel_g], S_g, e["t_on"], args.horizon, h_genie)
                        # 감속 시작 시점에 이미 경보 상태면 지연을 잴 수 없으므로 표시하고 분석에서 제외
                        pre_alarm = bool(value_at(t, np.nan_to_num(ma, nan=-1), e["t_on"]) >= eta_ma)
                        d_raw = first_cross_time(t, np.abs(np.nan_to_num(r, nan=0)), e["t_on"], args.horizon, eta_raw)
                        ev_rows.append(dict(c=c, order=order, tau=tau, sender=si, profile=tr.profile,
                                            t_on=e["t_on"], jerk=e["jerk"], sigma=sigma,
                                            k_star_meas=k_meas, k_star_eq2=k_eq2, k_star_eq4=k_eq4,
                                            delay_genie=d_genie, delay_detector=d_ma, delay_no_ma=d_raw,
                                            pre_alarm=pre_alarm))
                rate_rows.append(dict(c=c, order=order, tau=tau,
                                      rate_all=float(np.mean(np.concatenate(det_all))) if det_all else np.nan,
                                      rate_braking=float(np.mean(np.concatenate(det_brk))) if det_brk else np.nan))
            print(f"  order={order} done")

    ev = pd.DataFrame(ev_rows); h1 = pd.DataFrame(h1_rows); rates = pd.DataFrame(rate_rows)
    ev.to_csv(os.path.join(args.out, "events_H2_H3.csv"), index=False)
    h1.to_csv(os.path.join(args.out, "messages_H1.csv"), index=False)
    rates.to_csv(os.path.join(args.out, "detection_rates.csv"), index=False)
    summary.update(analyze(ev, h1, rates, args))
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


# --------------------------------------------------------------------------------------
# 6. 분석과 그림 (원고 메모의 판정 기준)
# --------------------------------------------------------------------------------------
def loglog_slope(x, y, min_pts=5):
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_pts:
        return np.nan
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def analyze(ev, h1, rates, args):
    out = {}
    # ---------------- H1 ----------------
    if len(h1):
        h = {}
        for o, g in h1.groupby("order"):
            ok = np.isfinite(g.D) & np.isfinite(g.D_pred)
            h[int(o)] = dict(
                corr_D_vs_pred=float(np.corrcoef(g.D[ok], g.D_pred[ok])[0, 1]),
                median_abs_D_over_sigma={p: float(np.median(np.abs(gg.D_over_sigma))) for p, gg in g.groupby("phase")})
        out["H1"] = h
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        for o, g in h1.groupby("order"):
            ax[0].scatter(g.D_pred, g.D, s=3, alpha=0.3, label=f"{o}차")
        lim = np.nanpercentile(np.abs(h1.D), 99.5)
        ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
        ax[0].set_xlabel("식 (2) 예측 D [m]"); ax[0].set_ylabel("측정 D [m]"); ax[0].legend()
        g1 = h1[h1.order == 1]
        _bp = [np.abs(g1[g1.phase == p].D_over_sigma) for p in ["cruise", "sustained", "transition"]]
        _tick = ["순항", "가감속 유지", "가감속 변화"]
        # matplotlib 3.9 에서 boxplot(labels=) -> boxplot(tick_labels=) 로 이름이 바뀌었다
        try:
            ax[1].boxplot(_bp, tick_labels=_tick, showfliers=False)
        except TypeError:
            ax[1].boxplot(_bp, labels=_tick, showfliers=False)
        ax[1].set_ylabel("|D| / σ (1차, 모든 τ)")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig_H1.png"), dpi=200); plt.close(fig)

    if not len(ev):
        return out
    # ---------------- H2 ----------------
    out["n_events_total"] = int(len(ev)); out["n_events_pre_alarm_excluded"] = int(ev.pre_alarm.sum())
    ev = ev[~ev.pre_alarm]
    e1 = ev[(ev.c == 1.0)]
    h2 = {}
    for o, g in e1.groupby("order"):
        ok = g.k_star_meas.notna() & g.k_star_eq2.notna()
        rel = (g.k_star_meas[ok] - g.k_star_eq2[ok]).abs()
        okg = g.delay_genie.notna() & g.k_star_meas.notna()
        h2[int(o)] = dict(n_events=int(len(g)), n_detected=int(g.k_star_meas.notna().sum()),
                          median_k_star_eq3=float(g.k_star_meas.median()),
                          median_delay_genie=float(g.delay_genie.median()),
                          median_abs_diff_eq3_vs_genie=float((g.k_star_meas[okg] - g.delay_genie[okg]).abs().median()) if okg.any() else np.nan,
                          median_abs_diff_eq3_vs_eq2=float(rel.median()) if len(rel) else np.nan,
                          slope_genie_vs_tau_jerk=loglog_slope((g.tau * g.jerk).to_numpy(), g.delay_genie.to_numpy()),
                          slope_eq3_vs_tau_jerk=loglog_slope((g.tau * g.jerk).to_numpy(), g.k_star_meas.to_numpy()))
    g1 = e1[e1.order == 1]
    ok = g1.k_star_meas.notna() & g1.k_star_eq4.notna()
    h2["order1_eq4_median_abs_diff"] = float((g1.k_star_meas[ok] - g1.k_star_eq4[ok]).abs().median()) if ok.any() else np.nan
    # 잡음 배율에 대한 기울기 (1차, 사건별 중앙값을 c에 대해)
    med_c = ev[ev.order == 1].groupby("c").delay_genie.median()
    h2["slope_vs_noise_scale"] = loglog_slope(med_c.index.to_numpy(float), med_c.to_numpy(), min_pts=2)
    h2["median_k_star_by_c"] = {str(k): float(v) for k, v in med_c.items()}
    h2["median_delay_genie_by_profile"] = {p: float(g.delay_genie.median()) for p, g in g1.groupby("profile")}
    okg1 = g1.delay_genie.notna() & g1.k_star_eq4.notna()
    h2["order1_eq4_vs_genie_median_abs_diff"] = float((g1.delay_genie[okg1] - g1.k_star_eq4[okg1]).abs().median()) if okg1.any() else np.nan
    h2["criterion"] = ("지지: 최적 탐지기 지연의 tau*jerk 기울기 ≈ -2/3 (±0.2), c 기울기 ≈ +2/3 (±0.2), "
                       "식 (3)/(4)와 최적 탐지기 지연의 차이 중앙값 ≤ 1초(메시지 한 개 간격)")
    out["H2"] = h2

    if len(g1) == 0 or med_c.dropna().le(0).all() or med_c.dropna().empty:
        out["H2"] = h2
        out["note"] = "감속 시작 시점에 이미 경보 상태인 사건만 남아 H2 그림을 그리지 못함 (순항 중에도 탐지되는지 확인 필요)"
        return out
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    marks = {"Normal": "o", "Aggressive": "^", "Cautious": "s"}
    for (tau, prof), g in g1.groupby(["tau", "profile"]):
        ax[0].scatter(g.k_star_eq4, g.delay_genie, s=10, marker=marks.get(prof, "o"),
                      label=f"τ={tau}, {prof}")
    mx = np.nanmax([g1.k_star_eq4.max(), g1.delay_genie.max(), 1.0])
    ax[0].plot([0, mx], [0, mx], "k--", lw=0.8)
    ax[0].set_xlabel("식 (4) 예측 k* [s]"); ax[0].set_ylabel("최적 탐지기 지연 [s]")
    ax[0].set_title("(a) 1차 예측 공격자"); ax[0].legend(fontsize=6, ncol=2)
    ax[1].plot(med_c.index, med_c.values, "o-")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    # 로그축 기본 라벨은 mathtext(10^{-1})로 그려지는데, mathtext는 font.family 의 대체 폰트
    # 목록을 보지 않아 한글 폰트에 없는 U+2212가 깨진다. 값이 몇 개뿐이므로 그냥 숫자로 찍는다.
    _fmt = FuncFormatter(lambda v, _p: f"{v:g}")
    for _axis in (ax[1].xaxis, ax[1].yaxis):
        _axis.set_major_formatter(_fmt)
        _axis.set_minor_formatter(_fmt)
    ax[1].set_xlabel("측정 잡음 배율 c"); ax[1].set_ylabel("최적 탐지기 지연 중앙값 [s]")
    ax[1].set_title(f"(b) 측정 잡음의 영향 (기울기 {h2['slope_vs_noise_scale']:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig2_H2.png"), dpi=200); plt.close(fig)

    # ---------------- H3 ----------------
    h3 = {}
    for o, g in e1.groupby("order"):
        ok = g.delay_genie.notna() & g.delay_detector.notna()
        gap = (g.delay_detector - g.delay_genie)[ok]
        okr = g.delay_genie.notna() & g.delay_no_ma.notna()
        gap_raw = (g.delay_no_ma - g.delay_genie)[okr]
        h3[int(o)] = dict(n_events=int(ok.sum()),
                          mean_delay_detector=float(g.delay_detector[ok].mean()) if ok.any() else np.nan,
                          mean_delay_genie=float(g.delay_genie[ok].mean()) if ok.any() else np.nan,
                          mean_gap_detector_minus_genie=float(gap.mean()) if len(gap) else np.nan,
                          mean_gap_without_moving_average=float(gap_raw.mean()) if len(gap_raw) else np.nan,
                          share_events_detector_ge_genie=float((gap >= 0).mean()) if len(gap) else np.nan)
    h3["criterion"] = ("지지: 평균적으로 실제 탐지기 지연 ≥ 최적 탐지기 지연 (Lorden 기준은 평균에 대한 것이므로 사건 단위 비율은 참고용). "
                       "gap = 실제 탐지기를 개선해 줄일 수 있는 최대 여지, 그중 이동 평균이 차지하는 몫 = gap - gap_without_moving_average")
    out["H3"] = h3
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", help="VeReMi NextGen 시험 분할 폴더 (예: InTAS highway 2AM test)")
    p.add_argument("--val", help="검증 분할 폴더 (임계값과 sigma 추정용)")
    p.add_argument("--synthetic", action="store_true", help="합성 데이터로 흐름만 확인")
    p.add_argument("--out", default="results")
    p.add_argument("--sender-key", default="sender_id", help="송신자 구분 필드 (가명 변경 시 sender_alias 검토)")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--time-unit", choices=["auto", "s", "ms", "us", "ns", "ps"], default="auto",
                   help="sendTime의 단위 (InTAS NextGen 배포본은 ns)")
    p.add_argument("--taus", type=float, nargs="+", default=[1.0, 2.0, 2.5, 3.0, 4.0])
    p.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--noise-scale", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--alpha", type=float, default=0.01, help="오탐 수준 (식 3의 ln(1/alpha))")
    p.add_argument("--fpr", type=float, default=0.01, help="실제 탐지기 임계값의 오탐률")
    p.add_argument("--window", type=int, default=5, help="실제 탐지기의 이동 평균 메시지 수")
    p.add_argument("--brake-a", type=float, default=4.0)
    p.add_argument("--onset-a", type=float, default=0.5)
    p.add_argument("--horizon", type=float, default=15.0, help="감속 시작 후 관찰 시간 [s]")
    p.add_argument("--smooth", type=float, default=0.5, help="궤적 평활 스플라인 강도")
    p.add_argument("--max-dt-jitter", type=float, default=0.2,
                   help="메시지 간격이 중앙값에서 이만큼(초) 넘게 벗어나는 송신자는 제외. "
                        "0 이하를 주면 거르지 않는다(결측 구간이 sigma를 망가뜨린다)")
    p.add_argument("--extrap", choices=["path", "heading"], default="path")
    p.add_argument("--h1-every", type=int, default=1, help="H1 저장 시 송신자 추출 간격")
    args = p.parse_args()
    if not args.synthetic and not args.test:
        p.error("--test 또는 --synthetic 이 필요합니다")
    run(args)


if __name__ == "__main__":
    main()
