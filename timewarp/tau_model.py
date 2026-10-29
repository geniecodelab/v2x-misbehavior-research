"""
tau_model.py: 시간 왜곡(time warping) 위조 모델.

보고 궤적:  y(t) = p(s(t)),  s(t) = t + tau(t)
  mode "a" (비보정): 위치만 왜곡, 속도·헤딩은 현재 참값 -> 자기 일관성이 깨진다.
      이론: r ~ v * |tau_dot| * dt
  mode "b" (완전 보정): v_rep = s_dot * v(s), a_rep = s_ddot * v(s) + s_dot^2 * a(s)
      자기 일관성은 유지되지만 보고 가속도·속도가 물리 한계를 넘을 수 있다.
      이론: |tau_ddot| <= a_lim / v  ->  tau를 dtau 만큼 바꾸는 최소 시간
            T >= 2 * sqrt(dtau * v / a_lim)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 궤적 보간 (numpy만 사용, C1 연속인 3차 Hermite)
# ---------------------------------------------------------------------------
class Traj:
    def __init__(self, t, x, y, v, h, noise=None):
        self.noise = noise or {}
        self.t, self.x, self.y, self.v, self.h = map(np.asarray, (t, x, y, v, h))
        self.dt = float(np.median(np.diff(self.t)))
        # 중앙차분 접선 (Catmull-Rom)
        self.mx = np.gradient(self.x, self.t)
        self.my = np.gradient(self.y, self.t)
        self.mv = np.gradient(self.v, self.t)          # 종방향 가속도 a(t)

    def _hermite(self, s, f_, m_):
        s = np.clip(s, self.t[0], self.t[-1])
        i = np.clip(np.searchsorted(self.t, s) - 1, 0, len(self.t) - 2)
        h = self.t[i + 1] - self.t[i]
        u = (s - self.t[i]) / h
        u2, u3 = u * u, u ** 3
        h00, h10 = 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u
        h01, h11 = -2 * u3 + 3 * u2, u3 - u2
        return h00 * f_[i] + h10 * h * m_[i] + h01 * f_[i + 1] + h11 * h * m_[i + 1]

    def pos(self, s):
        return self._hermite(s, self.x, self.mx), self._hermite(s, self.y, self.my)

    def spd(self, s):
        return self._hermite(s, self.v, self.mv)

    def acc(self, s):
        return np.interp(np.clip(s, self.t[0], self.t[-1]), self.t, self.mv)

    def head(self, s):
        c = self._hermite(s, np.cos(self.h), np.gradient(np.cos(self.h), self.t))
        si = self._hermite(s, np.sin(self.h), np.gradient(np.sin(self.h), self.t))
        return np.arctan2(si, c)

    def valid(self, s, margin=1.0):
        return (s >= self.t[0] + margin) & (s <= self.t[-1] - margin)


# ---------------------------------------------------------------------------
# tau 프로파일
# ---------------------------------------------------------------------------
def tau_ramp(t, tau0, dtau, t_start, T):
    """
    tau0 에서 tau0 - dtau 로 시간 T 동안 bang-bang(|tau_ddot| 일정)으로 변화.
    반환: tau, tau_dot, tau_ddot
    """
    u = np.clip((t - t_start) / T, 0.0, 1.0)
    ramp = np.where(u <= 0.5, 2 * u ** 2, 1 - 2 * (1 - u) ** 2)
    d = np.where(u <= 0.5, 4 * u, 4 * (1 - u)) / T
    dd = np.where((u > 0) & (u < 1), np.where(u <= 0.5, 4.0, -4.0) / T ** 2, 0.0)
    return tau0 - dtau * ramp, -dtau * d, -dtau * dd


def tau_linear(t, tau0, rate):
    return tau0 + rate * (t - t[0]), np.full_like(t, rate), np.zeros_like(t)


# ---------------------------------------------------------------------------
# 보고 메시지 생성
# ---------------------------------------------------------------------------
def report(traj: Traj, t, tau, tau_dot, tau_ddot, mode="a"):
    s = t + tau
    sd, sdd = 1.0 + tau_dot, tau_ddot
    x, y = traj.pos(s)
    out = dict(t=t, s=s, x=x, y=y, ok=traj.valid(s))
    if mode == "a":                       # 위치만 왜곡
        out["v"] = traj.spd(t)
        out["h"] = traj.head(t)
        out["a"] = traj.acc(t)
    else:                                 # 완전 보정
        out["v"] = sd * traj.spd(s)
        out["h"] = np.where(sd >= 0, traj.head(s), traj.head(s) + np.pi)
        out["a"] = sdd * traj.spd(s) + sd ** 2 * traj.acc(s)
    return out


# ---------------------------------------------------------------------------
# 탐지 특징 (기존 파이프라인과 동일한 정의)
# ---------------------------------------------------------------------------
def residuals(rep):
    """r_scalar: | |dp| - (v_prev+v_cur)/2*dt |,  r_vector: |dp - v_mean*dt*u(h_mid)|"""
    t, x, y, v, h = rep["t"], rep["x"], rep["y"], rep["v"], rep["h"]
    dt = np.diff(t)
    dx, dy = np.diff(x), np.diff(y)
    vm = (v[:-1] + v[1:]) / 2
    hmid = h[:-1] + ((h[1:] - h[:-1] + np.pi) % (2 * np.pi) - np.pi) / 2
    r_s = np.abs(np.hypot(dx, dy) - vm * dt)
    r_v = np.hypot(dx - vm * dt * np.cos(hmid), dy - vm * dt * np.sin(hmid))
    return np.concatenate([[np.nan], r_s]), np.concatenate([[np.nan], r_v])


# ---------------------------------------------------------------------------
# 제약과 투영 (최적 회피 공격 단계에서 사용)
# ---------------------------------------------------------------------------
def project_tau(tau, dt, max_dot=None, max_ddot=None, iters=50):
    """|tau_dot|, |tau_ddot| 제약을 만족하도록 반복 투영."""
    tau = np.array(tau, float)
    for _ in range(iters):
        if max_ddot is not None:
            dd = np.diff(tau, 2) / dt ** 2
            over = np.abs(dd) > max_ddot
            if over.any():
                corr = (np.abs(dd) - max_ddot).clip(min=0) * np.sign(dd) * dt ** 2 / 4
                tau[:-2] += corr
                tau[1:-1] -= 2 * corr
                tau[2:] += corr
        if max_dot is not None:
            d = np.diff(tau) / dt
            over = np.abs(d) > max_dot
            if over.any():
                corr = (np.abs(d) - max_dot).clip(min=0) * np.sign(d) * dt / 2
                tau[:-1] += corr
                tau[1:] -= corr
        if not (max_ddot is not None and np.any(np.abs(np.diff(tau, 2) / dt ** 2) > max_ddot * 1.01)) and \
           not (max_dot is not None and np.any(np.abs(np.diff(tau) / dt) > max_dot * 1.01)):
            break
    return tau


def peak_constraints(traj, t0, t1, tau0, dtau, t_start, T, step=0.05):
    """[t0,t1] 구간을 step 간격으로 조밀하게 평가한 보고 가속도/속도의 최대 크기.
    메시지 주기(1 s)로만 검사하면 짧은 tau 변화 구간을 놓치므로 반드시 조밀 격자를 쓴다."""
    tf = np.arange(t0, t1, step)
    tau, td, tdd = tau_ramp(tf, tau0, dtau, t_start, T)
    rep = report(traj, tf, tau, td, tdd, mode="b")
    m = rep["ok"]
    if m.sum() < 3:
        return np.inf, np.inf
    return float(np.nanmax(np.abs(rep["a"][m]))), float(np.nanmax(np.abs(rep["v"][m])))


def theory_min_time(dtau, v, a_lim):
    """|tau_ddot| <= a_lim/v 하에서 tau를 dtau 만큼 바꾸는 최소 시간."""
    return 2.0 * np.sqrt(np.abs(dtau) * v / a_lim)
