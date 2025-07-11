#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPS Error Analysis Toolkit
==========================
* Supports linear-regression and B-spline deviation for one or more CSV tracks.
* Specify CSV file paths directly in the `csv_files` list.
* For each track:
  1. Linear fit → perpendicular deviation.
  2. B-spline (configurable smoothing) → perpendicular deviation.
* Prints statistics (mean, std, RMSE, range) for both models.
* Visualises:
  • Overlayed paths with regression line & spline.
  • Combined deviation overlay for all tracks & models.

Usage:
  - Modify `csv_files` list below with your filenames.
  - Adjust `smoothing` if needed for spline smoothing.
  - Run: python gps_error_combined.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.optimize import fminbound

# ──────────────────────────────────────────────────────────────
# 1) 파일 목록 설정
# ──────────────────────────────────────────────────────────────
# 처리할 CSV 파일명을 직접 지정하세요 (X,Y 컬럼, 헤더 한 줄 건너뜀)
csv_files = [
    Path("raw_track_xy_4.csv"),
    Path("waypoints_dynamic_1.csv"),
]

# spline smoothing parameter
def_smoothing = 5.0

# ──────────────────────────────────────────────────────────────
# 2) 데이터 로드 유틸
# ──────────────────────────────────────────────────────────────

def load_xy(path: Path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[:,0], arr[:,1]

# ──────────────────────────────────────────────────────────────
# 3A) 선형 회귀 편차
# ──────────────────────────────────────────────────────────────

def linear_dev(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    denom = np.hypot(slope, 1)
    dev   = np.abs(slope * x - y + intercept) / denom
    return dev, slope, intercept

# ──────────────────────────────────────────────────────────────
# 3B) 스플라인 편차
# ──────────────────────────────────────────────────────────────

def fit_spline(x, y, smoothing=def_smoothing, k=3):
    tck, _ = splprep([x, y], s=smoothing, k=k)
    return tck


def spline_dev(x, y, tck):
    def pt_dist(u, xi, yi):
        xs, ys = splev(u, tck)
        return np.hypot(xs - xi, ys - yi)
    return np.array([
        pt_dist(fminbound(lambda u: pt_dist(u, xi, yi), 0.0, 1.0, disp=0), xi, yi)
        for xi, yi in zip(x, y)
    ])

# ──────────────────────────────────────────────────────────────
# 4) 통계 계산
# ──────────────────────────────────────────────────────────────

def calc_stats(arr):
    return {
        "mean":  float(arr.mean()),
        "std":   float(arr.std()),
        "rmse":  float(np.sqrt(np.mean(arr**2))),
        "range": float(arr.max() - arr.min()),
    }

# ──────────────────────────────────────────────────────────────
# 5) 시각화 함수
# ──────────────────────────────────────────────────────────────

def plot_paths(ax, x, y, slope, intercept, tck, label):
    ax.scatter(x, y, s=12, alpha=0.5, label=f"{label} points")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope*xs + intercept, "--", label=f"{label} linear")
    u = np.linspace(0,1,200)
    xs_s, ys_s = splev(u, tck)
    ax.plot(xs_s, ys_s, label=f"{label} spline")


def overlay_deviation(cte_dict):
    plt.figure(figsize=(7,4))
    for name, (lin_d, spl_d) in cte_dict.items():
        idx = np.arange(len(lin_d))
        plt.plot(idx, lin_d, label=f"{name} linear")
        plt.plot(idx, spl_d, '--', label=f"{name} spline")
    plt.xlabel("Sample Index")
    plt.ylabel("Deviation [m]")
    plt.title("Deviation Overlay")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ──────────────────────────────────────────────────────────────
# 6) 메인
# ──────────────────────────────────────────────────────────────

def main():
    fig, ax = plt.subplots(figsize=(6,5))
    cte_dict = {}

    for path in csv_files:
        x, y = load_xy(path)
        lin_d, m, b = linear_dev(x, y)
        tck = fit_spline(x, y)
        spl_d = spline_dev(x, y, tck)

        stats_lin = calc_stats(lin_d)
        stats_spl = calc_stats(spl_d)
        print(f"\n[{path.name}] Linear: mean={stats_lin['mean']:.3f}, std={stats_lin['std']:.3f}, rmse={stats_lin['rmse']:.3f}, range={stats_lin['range']:.3f}")
        print(f"[{path.name}] Spline: mean={stats_spl['mean']:.3f}, std={stats_spl['std']:.3f}, rmse={stats_spl['rmse']:.3f}, range={stats_spl['range']:.3f}")

        plot_paths(ax, x, y, m, b, tck, path.stem)
        cte_dict[path.stem] = (lin_d, spl_d)

    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_title("Paths with Linear & Spline Fit")
    ax.legend(); ax.grid(True); plt.tight_layout(); plt.show()

    overlay_deviation(cte_dict)

if __name__ == "__main__":
    main()
