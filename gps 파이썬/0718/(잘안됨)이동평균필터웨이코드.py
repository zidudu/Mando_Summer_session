#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_curve_and_make_waypoints_onefig.py
────────────────────────────────────────────────────────────────
• Raw CSV → (곡선 : B-스플라인 / 직선 : Moving Average) Filtered
• 동적 웨이포인트 생성
• Figure(1×3)
   ① Raw(scatter) vs Filtered(line) [선명]
   ② Waypoints
   ③ Mean / RMSE / Range / Std
• 옵션:
   --zoom   : Raw vs Filtered 패널을 마지막 0.1 m 범위로 확대
"""

import sys, math, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from pathlib import Path

# ─── 파라미터 ────────────────────────────────────────────────
CSV_IN   = "raw_track_xy_18.csv"
CURV_TH  = 0.015
MIN_LEN  = 5
SMOOTH_S = 0.6
WIN_MEAN = 5
K1, K2   = 0.10, 0.20
D_S, D_M, D_C = 0.55, 0.40, 0.28
VIEW_R   = 5.0

# ─── CLI 옵션 ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="?", default=CSV_IN)
parser.add_argument("--zoom", action="store_true",
                    help="Raw vs Filtered 패널을 마지막 0.1 m 범위로 확대")
args   = parser.parse_args()
CSV_IN = args.csv

# ─── 유틸리티 ────────────────────────────────────────────────
def load_xy(fname):
    d = np.loadtxt(fname, delimiter=',', skiprows=1)
    return d[:, 0], d[:, 1]

def curvature(x, y):
    dx, dy   = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = dx * ddy - dy * ddx
    den = (dx**2 + dy**2)**1.5
    kap = np.zeros_like(x)
    msk = den > 1e-12
    kap[msk] = np.abs(num[msk] / den[msk])
    return kap

def segments(kappa):
    segs, ty = [], []
    i, N = 0, len(kappa)
    while i < N:
        mode = "line" if kappa[i] < CURV_TH else "curve"
        j = i + 1
        while j < N and ((mode == "line" and kappa[j] < CURV_TH) or
                         (mode == "curve" and kappa[j] >= CURV_TH)):
            j += 1
        if j - i >= MIN_LEN:
            segs.append((i, j)); ty.append(mode)
        i = j
    return segs, ty

def smooth_curve(x, y):
    pts = list(dict.fromkeys(zip(x, y)))
    ux, uy = zip(*pts)
    if len(ux) < 2: return x, y
    k = min(3, len(ux) - 1)
    tck, _ = splprep([ux, uy], s=SMOOTH_S, k=k)
    xs, ys = splev(np.linspace(0, 1, len(x)), tck)
    return xs, ys

def moving_average(x, y, win=WIN_MEAN):
    if len(x) < win: return x, y
    ker = np.ones(win) / win
    x_ma = np.convolve(x, ker, mode='same')
    y_ma = np.convolve(y, ker, mode='same')
    return x_ma, y_ma

def filter_track(xr, yr):
    kappa = curvature(xr, yr)
    segs, ty = segments(kappa)
    xf, yf = xr.copy(), yr.copy()
    for (i0, i1), mode in zip(segs, ty):
        if mode == "curve":
            xs_s, ys_s = smooth_curve(xr[i0:i1], yr[i0:i1])
        else:
            xs_s, ys_s = moving_average(xr[i0:i1], yr[i0:i1])
        len_seg = i1 - i0
        xf[i0:i1], yf[i0:i1] = xs_s[:len_seg], ys_s[:len_seg]
    return xf, yf

def dyn_wp(x, y):
    kap = curvature(x, y)
    wpts, acc = [(x[0], y[0])], 0.0
    for i in range(1, len(x)):
        seg = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]); acc += seg
        d_t = D_S if kap[i] < K1 else (D_M if kap[i] < K2 else D_C)
        while acc >= d_t:
            r = (seg - (acc - d_t)) / seg
            wpts.append((x[i - 1] + (x[i] - x[i - 1]) * r,
                         y[i - 1] + (y[i] - y[i - 1]) * r))
            acc -= d_t
    wpts.append((x[-1], y[-1]))
    return np.array(wpts)

def seg_errors(x, y):
    kappa = curvature(x, y)
    segs, ty = segments(kappa)
    dev = []
    for (i0, i1), mode in zip(segs, ty):
        xs, ys = x[i0:i1], y[i0:i1]
        if len(xs) < 2: continue
        if mode == "line":
            m, b = np.polyfit(xs, ys, 1)
            dev.append(np.abs(m * xs - ys + b) / math.hypot(m, -1))
        else:
            xs_s, ys_s = smooth_curve(xs, ys)
            dev.append(np.hypot(xs - xs_s, ys - ys_s))
    return np.hstack(dev) if dev else np.array([])

def stats(arr):
    return arr.mean(), math.sqrt((arr**2).mean()), np.ptp(arr), arr.std()

def save_wp(arr, stem="waypoints_dynamic"):
    p = Path(f"{stem}.csv")
    for idx in range(1000):
        if not p.exists():
            np.savetxt(p, arr, delimiter=',', header="X_m,Y_m",
                       comments='', fmt="%.6f")
            return p
        p = Path(f"{stem}_{idx}.csv")

# ─── 메인 ────────────────────────────────────────────────────
if __name__ == "__main__":
    X, Y = load_xy(CSV_IN)
    xr, yr = X - X[0], Y - Y[0]

    xf, yf = filter_track(xr, yr)
    WPT = dyn_wp(xf, yf)
    out_csv = save_wp(WPT)

    err_raw, err_flt = seg_errors(xr, yr), seg_errors(xf, yf)
    m_r, rmse_r, rng_r, std_r = stats(err_raw)
    m_f, rmse_f, rng_f, std_f = stats(err_flt)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Raw vs Filtered : Raw 검정 ●, Filtered 빨강 선
    ax[0].scatter(xr, yr, s=40, c='k', label='Raw', zorder=3)
    ax[0].plot(xf, yf, 'r-', lw=1.4, label='Filtered', zorder=1)
    ax[0].set_title("Raw vs Filtered")
    ax[0].axis('equal')
    if args.zoom:
        # 마지막 0.1 m 범위로 자동 확대
        ax[0].set_xlim(xf[-1] - 0.1, xf[-1])
        ax[0].set_ylim(yf[-1] - 0.1, yf[-1])
    else:
        ax[0].set_xlim(-VIEW_R, VIEW_R); ax[0].set_ylim(-VIEW_R, VIEW_R)
    ax[0].set_xlabel("ΔX [m]"); ax[0].set_ylabel("ΔY [m]")
    ax[0].legend(); ax[0].grid()

    # (2) Waypoints
    ax[1].plot(xf, yf, 'k-', lw=1, alpha=0.4, label='Filtered')
    ax[1].scatter(WPT[:, 0], WPT[:, 1], c='orange', marker='*', s=28,
                  label=f'Waypoints ({len(WPT)})')
    ax[1].set_title("Dynamic Waypoints")
    ax[1].axis('equal'); ax[1].set_xlim(-VIEW_R, VIEW_R); ax[1].set_ylim(-VIEW_R, VIEW_R)
    ax[1].set_xlabel("ΔX [m]"); ax[1].set_ylabel("ΔY [m]")
    ax[1].legend(); ax[1].grid()

    # (3) Error Statistics
    labels   = ['Mean', 'RMSE', 'Range', 'Std(σ)']
    raw_vals = [m_r, rmse_r, rng_r, std_r]
    flt_vals = [m_f, rmse_f, rng_f, std_f]
    xbar, w  = np.arange(len(labels)), 0.35

    ax[2].bar(xbar - w/2, raw_vals, width=w, label='Raw', color='skyblue')
    ax[2].bar(xbar + w/2, flt_vals, width=w, label='Filtered', color='salmon')
    for i, v in enumerate(raw_vals):
        ax[2].text(i - w/2, v + 1e-4, f"{v:.3f}", ha='center')
    for i, v in enumerate(flt_vals):
        ax[2].text(i + w/2, v + 1e-4, f"{v:.3f}", ha='center')

    ax[2].set_xticks(xbar); ax[2].set_xticklabels(labels)
    ax[2].set_ylabel("[m]"); ax[2].set_title("Error Statistics")
    ax[2].legend(); ax[2].grid(axis='y', ls='--')

    plt.tight_layout(); plt.show()

    print("\n─── 오차 통계 ───")
    print(f"[Raw]      mean={m_r:.4f}, RMSE={rmse_r:.4f}, range={rng_r:.4f}, std={std_r:.4f}")
    print(f"[Filtered] mean={m_f:.4f}, RMSE={rmse_f:.4f}, range={rng_f:.4f}, std={std_f:.4f}")
    print(f"\n[Waypoints] 저장 완료 → {out_csv} (총 {len(WPT)}개)")
