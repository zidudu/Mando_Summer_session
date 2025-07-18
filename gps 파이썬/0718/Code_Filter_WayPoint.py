#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_curve_and_make_waypoints_onefig.py  (직선 = 선형회귀 스무딩)
────────────────────────────────────────────────────────────────
• Raw CSV → (곡선 : B-스플라인 / 직선 : Linear Regression) Filtered
• 동적 웨이포인트 생성
• Figure(1×3) : ① Raw vs Filtered ② Waypoints ③ Mean/RMSE/Range
"""

import sys, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from pathlib import Path

# ─── 파라미터 ────────────────────────────────────────────────
CSV_IN    = sys.argv[1] if len(sys.argv) > 1 else "raw_track_xy_22.csv"
CURV_TH   = 0.015         # 직선/곡선 임계 곡률
MIN_LEN   = 5             # 최소 세그먼트 길이
SMOOTH_S  = 0.6          # 곡선 B-스플라인 s
K1, K2    = 0.10, 0.20    # WP 곡률 구간
D_S, D_M, D_C = 0.55, 0.40, 0.28
VIEW_R    = 5.0
# ────────────────────────────────────────────────────────────

def load_xy(fname):
    d = np.loadtxt(fname, delimiter=',', skiprows=1)
    return d[:,0], d[:,1]

def curvature(x, y):
    dx, dy   = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = dx*ddy - dy*ddx
    den = (dx*dx + dy*dy)**1.5
    k   = np.zeros_like(x)
    mask = den > 1e-12
    k[mask] = np.abs(num[mask]/den[mask])
    return k

def segments(kappa):
    segs, ty = [], []
    i, N = 0, len(kappa)
    while i < N:
        mode = "line" if kappa[i] < CURV_TH else "curve"
        j = i+1
        while j < N and ((mode=="line" and kappa[j]<CURV_TH) or
                         (mode=="curve" and kappa[j]>=CURV_TH)):
            j += 1
        if j - i >= MIN_LEN:
            segs.append((i, j)); ty.append(mode)
        i = j
    return segs, ty

# ── 곡선 스플라인 스무딩 ─────────────────────────────────────
def smooth_curve(x, y):
    pts = list(dict.fromkeys(zip(x, y)))      # 중복 제거(순서 유지)
    ux, uy = zip(*pts)
    n = len(ux)
    if n < 2: return x, y
    k = min(3, n-1)
    tck, _ = splprep([ux, uy], s=SMOOTH_S, k=k)
    xs_s, ys_s = splev(np.linspace(0, 1, len(x)), tck)
    return xs_s, ys_s

# ── 직선 → 선형 회귀 직선 위로 투영 ──────────────────────────
def smooth_line_regress(x, y):
    if len(x) < 2: return x, y
    m, b = np.polyfit(x, y, 1)           # y = m x + b
    y_fit = m * x + b
    return x, y_fit

# ── 전체 트랙 스무딩 ────────────────────────────────────────
def filter_track(xr, yr):
    kappa = curvature(xr, yr)
    segs, ty = segments(kappa)
    xf, yf = xr.copy(), yr.copy()
    for (i0, i1), mode in zip(segs, ty):
        if mode == "curve":
            xs_s, ys_s = smooth_curve(xr[i0:i1], yr[i0:i1])
        else:  # line
            xs_s, ys_s = smooth_line_regress(xr[i0:i1], yr[i0:i1])
        xf[i0:i1], yf[i0:i1] = xs_s, ys_s
    return xf, yf

# ── 동적 웨이포인트 생성 ────────────────────────────────────
def dyn_wp(x, y):
    kap = curvature(x, y)
    wpts, acc = [(x[0], y[0])], 0.0
    for i in range(1, len(x)):
        seg = math.hypot(x[i]-x[i-1], y[i]-y[i-1]); acc += seg
        d_t = D_S if kap[i]<K1 else (D_M if kap[i]<K2 else D_C)
        while acc >= d_t:
            r = (seg - (acc-d_t)) / seg
            wpts.append((x[i-1] + (x[i]-x[i-1])*r,
                         y[i-1] + (y[i]-y[i-1])*r))
            acc -= d_t
    wpts.append((x[-1], y[-1]))
    return np.array(wpts)

# ── 오차 계산 ───────────────────────────────────────────────
def seg_errors(x, y):
    kappa = curvature(x, y)
    segs, ty = segments(kappa)
    dev = []
    for (i0,i1), mode in zip(segs, ty):
        xs, ys = x[i0:i1], y[i0:i1]
        if mode == "line":
            m, b = np.polyfit(xs, ys, 1)
            dev.append(np.abs(m*xs - ys + b) / math.hypot(m, -1))
        else:
            xs_s, ys_s = smooth_curve(xs, ys)
            dev.append(np.hypot(xs-xs_s, ys-ys_s))
    return np.hstack(dev) if dev else np.array([])

def stats(arr):
    return arr.mean(), math.sqrt((arr**2).mean()), np.ptp(arr)

def save_wp(arr, stem="waypoints_dynamic"):
    p = Path(f"{stem}.csv")
    for idx in range(1_000):
        if not p.exists():
            np.savetxt(p, arr, delimiter=',', header="X_m,Y_m",
                       comments='', fmt="%.6f")
            return p
        p = Path(f"{stem}_{idx}.csv")

# ── 메인 ────────────────────────────────────────────────────
if __name__ == "__main__":
    X, Y = load_xy(CSV_IN)
    xr, yr = X - X[0], Y - Y[0]

    xf, yf = filter_track(xr, yr)
    WPT    = dyn_wp(xf, yf)
    out_csv = save_wp(WPT)

    # 오차 통계
    err_raw, err_flt = seg_errors(xr, yr), seg_errors(xf, yf)
    m_r, rmse_r, rng_r = stats(err_raw)
    m_f, rmse_f, rng_f = stats(err_flt)

    # ─── Figure (1×3) ──────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(18,5))

    # (1) Raw vs Filtered
    ax[0].plot(xr, yr, '0.6', lw=1, label='Raw')
    ax[0].plot(xf, yf, 'r-', lw=1.5, label='Filtered')
    ax[0].set_title("Raw vs Filtered")
    ax[0].axis('equal'); ax[0].set_xlim(-VIEW_R, VIEW_R); ax[0].set_ylim(-VIEW_R, VIEW_R)
    ax[0].set_xlabel("ΔX [m]"); ax[0].set_ylabel("ΔY [m]"); ax[0].legend(); ax[0].grid()

    # (2) Waypoints
    ax[1].plot(xf, yf, 'k-', lw=1, alpha=0.4, label='Filtered')
    ax[1].scatter(WPT[:,0], WPT[:,1], c='orange', marker='*', s=40,
                  label=f'Waypoints ({len(WPT)})')
    ax[1].set_title("Dynamic Waypoints")
    ax[1].axis('equal'); ax[1].set_xlim(-VIEW_R, VIEW_R); ax[1].set_ylim(-VIEW_R, VIEW_R)
    ax[1].set_xlabel("ΔX [m]"); ax[1].set_ylabel("ΔY [m]"); ax[1].legend(); ax[1].grid()

    # (3) Error Statistics
    labels = ['Mean', 'RMSE', 'Range']
    raw_vals = [m_r, rmse_r, rng_r]
    flt_vals = [m_f, rmse_f, rng_f]
    x = np.arange(3); w = 0.35
    ax[2].bar(x-w/2, raw_vals, width=w, label='Raw', color='skyblue')
    ax[2].bar(x+w/2, flt_vals, width=w, label='Filtered', color='salmon')
    for i,v in enumerate(raw_vals): ax[2].text(i-w/2, v+1e-4, f"{v:.3f}", ha='center')
    for i,v in enumerate(flt_vals): ax[2].text(i+w/2, v+1e-4, f"{v:.3f}", ha='center')
    ax[2].set_xticks(x); ax[2].set_xticklabels(labels)
    ax[2].set_ylabel("[m]"); ax[2].set_title("Error Statistics")
    ax[2].legend(); ax[2].grid(axis='y', ls='--')

    plt.tight_layout(); plt.show()

    # 결과 출력
    print("\n─── 오차 통계 ───")
    print(f"[Raw]      mean={m_r:.4f}, RMSE={rmse_r:.4f}, range={rng_r:.4f}")
    print(f"[Filtered] mean={m_f:.4f}, RMSE={rmse_f:.4f}, range={rng_f:.4f}")
    print(f"\n[Waypoints] 저장 완료 → {out_csv} (총 {len(WPT)}개)")
