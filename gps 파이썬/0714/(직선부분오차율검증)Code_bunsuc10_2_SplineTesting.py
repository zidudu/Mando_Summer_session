#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arc_spline_visualizer.py  (COMPLETE)

• 왼쪽   : GPS 원시 궤적 - 모델(직선·원·클로소이드) 오버레이
• 가운데 : 편차 히스토그램(Count)
• 오른쪽 : 요약 통계 막대그래프(Mean·Std·RMSE·Min·Max·Range) – 소수점 6자리 표기
• Figure 2 : Sample-Index 별 직선/원호 편차 비교
"""

import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares
from scipy.interpolate import splprep, splev
try:
    import pyclothoids  # pip install pyclothoids
except ImportError:
    pyclothoids = None
    warnings.warn("pyclothoids 미설치 → 클로소이드 구간은 B-스플라인으로 대체됩니다.")

# ── 파라미터 ────────────────────────────────────────────
IN_CSV            = "raw_track_xy_6.csv"
LINE_CURV_THR     = 0.10   # κ < 0.10 → 직선
CIRCLE_CURV_THR   = 0.20   # 0.10 ≤ κ < 0.20 → 원호
MIN_SEG_LEN       = 5      # 최소 세그먼트 길이
SMOOTH_S          = 0.5    # 스플라인 스무딩 파라미터
# ────────────────────────────────────────────────────────

def load_xy(csv_file: str):
    data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    return data[:,0], data[:,1]

def curvature(xs, ys):
    dx, dy = np.gradient(xs), np.gradient(ys)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = dx * ddy - dy * ddx
    den = (dx*dx + dy*dy)**1.5
    k = np.zeros_like(xs)
    mask = den > 1e-6
    k[mask] = np.abs(num[mask] / den[mask])
    return k

def segment_by_curv(kappa):
    segs, types = [], []
    i, n = 0, len(kappa)
    while i < n:
        if kappa[i] < LINE_CURV_THR:
            mode = 'line'
        elif kappa[i] < CIRCLE_CURV_THR:
            mode = 'circle'
        else:
            mode = 'clothoid'
        j = i + 1
        while j < n and (
            (mode=='line'   and kappa[j] < LINE_CURV_THR) or
            (mode=='circle' and LINE_CURV_THR <= kappa[j] < CIRCLE_CURV_THR) or
            (mode=='clothoid' and kappa[j] >= CIRCLE_CURV_THR)
        ):
            j += 1
        if j - i >= MIN_SEG_LEN:
            segs.append((i, j))
            types.append(mode)
        i = j
    return segs, types

def fit_line(xs, ys):
    m, b = np.polyfit(xs, ys, 1)
    dev = np.abs(m*xs - ys + b) / math.hypot(m, 1)
    return dev, m, b

def fit_circle(xs, ys):
    xm, ym = xs.mean(), ys.mean()
    r0 = np.mean(np.hypot(xs-xm, ys-ym))
    fun = lambda c: np.hypot(xs-c[0], ys-c[1]) - c[2]
    xc, yc, r = least_squares(fun, [xm, ym, r0]).x
    dev = np.abs(np.hypot(xs-xc, ys-yc) - r)
    return dev, xc, yc, r

def fit_clothoid_spline(xs, ys):
    if pyclothoids is not None:
        try:
            t0 = math.atan2(ys[1]-ys[0], xs[1]-xs[0])
            t1 = math.atan2(ys[-1]-ys[-2], xs[-1]-xs[-2])
            clo = pyclothoids.Clothoid.G1Hermite(xs[0], ys[0], t0,
                                                 xs[-1], ys[-1], t1)
            s = np.linspace(0, clo.L, len(xs))
            dev = np.hypot(xs-clo.X(s), ys-clo.Y(s))
            return dev, ('clothoid', clo)
        except Exception:
            pass
    # fallback to B-spline
    k = min(3, len(xs)-1)
    tck, _ = splprep([xs, ys], s=SMOOTH_S, k=k)
    u = np.linspace(0, 1, len(xs))
    xs_s, ys_s = splev(u, tck)
    dev = np.hypot(xs-xs_s, ys-ys_s)
    return dev, ('spline', tck)

def stats(arr):
    return np.array([
        float(arr.mean()),
        float(arr.std()),
        float(np.sqrt((arr**2).mean())),
        float(arr.min()),
        float(arr.max()),
        float(np.ptp(arr)),
    ])

if __name__ == "__main__":
    # Load data
    xs, ys = load_xy(IN_CSV)
    kappa = curvature(xs, ys)
    segs, types = segment_by_curv(kappa)

    dev_line, dev_circle, dev_clo = [], [], []

    # Prepare subplots
    fig, (ax_trk, ax_hist, ax_bar) = plt.subplots(1, 3, figsize=(15,5))

    # Track + fits
    ax_trk.plot(xs, ys, 'k.-', ms=3, label='Raw')
    for (i0, i1), tp in zip(segs, types):
        sx, sy = xs[i0:i1], ys[i0:i1]
        if tp=='line':
            dev, m, b = fit_line(sx, sy)
            dev_line.append(dev)
            ax_trk.plot([sx[0],sx[-1]], [m*sx[0]+b,m*sx[-1]+b], 'g--', label='Line' if 'Line' not in ax_trk.get_legend_handles_labels()[1] else '')
        elif tp=='circle':
            dev, xc, yc, r = fit_circle(sx, sy)
            dev_circle.append(dev)
            th = np.linspace(0,2*np.pi,200)
            ax_trk.plot(xc+r*np.cos(th), yc+r*np.sin(th), 'r-', label='Circle' if 'Circle' not in ax_trk.get_legend_handles_labels()[1] else '')
        else:
            dev, (mode, obj) = fit_clothoid_spline(sx, sy)
            dev_clo.append(dev)
            if mode=='clothoid':
                s = np.linspace(0, obj.L, 200)
                ax_trk.plot(obj.X(s), obj.Y(s), 'b-', label='Clothoid' if 'Clothoid' not in ax_trk.get_legend_handles_labels()[1] else '')
            else:
                u = np.linspace(0,1,200)
                xs_s, ys_s = splev(u, obj)
                ax_trk.plot(xs_s, ys_s, 'b--', label='Spline' if 'Spline' not in ax_trk.get_legend_handles_labels()[1] else '')
    ax_trk.set_title('Track & Fits')
    ax_trk.axis('equal'); ax_trk.grid(True); ax_trk.legend()

    # Histogram
    all_dev = np.concatenate(dev_line + dev_circle + dev_clo)
    ax_hist.hist(all_dev, bins=30, color='tab:purple', alpha=0.7)
    ax_hist.set_xlabel('Deviation [m]'); ax_hist.set_ylabel('Count'); ax_hist.set_title('Deviation Distribution')
    ax_hist.grid(axis='y', linestyle='--', alpha=0.6)

    # Bar chart stats with 6 decimal places
    groups, lbls = [], []
    if dev_line:
        groups.append(stats(np.concatenate(dev_line))); lbls.append('Line')
    if dev_circle:
        groups.append(stats(np.concatenate(dev_circle))); lbls.append('Circle')
    if dev_clo:
        groups.append(stats(np.concatenate(dev_clo))); lbls.append('Clothoid')
    if groups:
        groups = np.vstack(groups)
        metrics = ['mean','std','rmse','min','max','range']
        x = np.arange(len(metrics)); w = 0.8/len(lbls)
        for idx, (lbl, row) in enumerate(zip(lbls, groups)):
            ax_bar.bar(x+idx*w, row, w, label=lbl)
            for xi, val in zip(x+idx*w, row):
                ax_bar.text(xi, val+0.0005, f'{val:.6f}', ha='center', va='bottom', fontsize=7)
        ax_bar.set_xticks(x+ w*(len(lbls)-1)/2)
        ax_bar.set_xticklabels(metrics)
        ax_bar.set_ylabel('Deviation [m]'); ax_bar.set_title('Error Statistics')
        ax_bar.grid(axis='y', linestyle='--', alpha=0.6); ax_bar.legend()
    else:
        ax_bar.text(0.5,0.5,'No segments',ha='center',va='center'); ax_bar.axis('off')

    plt.tight_layout(); plt.show()

        # Figure 2: Bar chart for Line and Circle deviations
    if dev_line or dev_circle:
        fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4))
        # Line deviation bar
        if dev_line:
            dl = np.concatenate(dev_line)
            axs2[0].bar(np.arange(len(dl)), dl, color='r')
            axs2[0].set_title('Line Deviation per Sample')
            axs2[0].set_xlabel('Sample Index')
            axs2[0].set_ylabel('Deviation [m]')
            axs2[0].grid(True)
        else:
            axs2[0].text(0.5,0.5,'No line segments',ha='center',va='center')
            axs2[0].axis('off')
        # Circle deviation bar
        if dev_circle:
            dc = np.concatenate(dev_circle)
            axs2[1].bar(np.arange(len(dc)), dc, color='b')
            axs2[1].set_title('Circle Deviation per Sample')
            axs2[1].set_xlabel('Sample Index')
            axs2[1].set_ylabel('Deviation [m]')
            axs2[1].grid(True)
        else:
            axs2[1].text(0.5,0.5,'No circle segments',ha='center',va='center')
            axs2[1].axis('off')
        plt.tight_layout()
        plt.show()
