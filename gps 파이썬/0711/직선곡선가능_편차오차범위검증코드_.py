#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Deviation Analysis & Visualization
  - Load X, Y from a CSV file
  - (A) Linear regression (np.polyfit) → vertical deviation
  - (B) B-spline approximation (SciPy splprep/splev) → vertical deviation
  - Print statistics (mean, std, RMSE, range)
  - Plot (1) raw path with regression line & spline curve
          (2) deviation vs. sample index
※ 시각화 요소는 모두 영어, 코드 주석은 한국어
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.optimize import fminbound

# ──────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────
def load_data(fname: str):
    """CSV 파일에서 X, Y 열을 로드 (헤더 1줄 스킵)"""
    data = np.loadtxt(fname, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]

# ──────────────────────────────────────────────────────────────
# (A) 선형 회귀 기반 편차
# ──────────────────────────────────────────────────────────────
def linear_deviation(x: np.ndarray, y: np.ndarray):
    """1차 회귀선 대비 수직 편차 및 계수 반환"""
    slope, intercept = np.polyfit(x, y, 1)
    denom  = np.hypot(slope, 1)
    dist   = np.abs(slope * x - y + intercept) / denom
    return dist, slope, intercept

# ──────────────────────────────────────────────────────────────
# (B) 스플라인 기반 편차 (중복점 제거 + 스무딩)
# ──────────────────────────────────────────────────────────────
def spline_deviation(x: np.ndarray, y: np.ndarray,
                     smoothing: float = 1.0, degree: int = 3):
    """
    1) 연속된 중복점 제거
    2) splprep(..., s=smoothing) 으로 B-spline 모델 생성
    3) fminbound 으로 각 점의 곡선까지 수직거리 계산
    """
    # 1) 중복점 제거
    if len(x) > 1:
        mask = np.concatenate(([True],
                               ~((x[1:]==x[:-1]) & (y[1:]==y[:-1]))))
        x_clean = x[mask]
        y_clean = y[mask]
    else:
        x_clean, y_clean = x, y

    # 2) 스플라인 모델 생성 (smoothing>0 권장)
    tck, _ = splprep([x_clean, y_clean], s=smoothing, k=degree)

    # 3) 거리 계산
    def pt_dist(u, xi, yi):
        xs, ys = splev(u, tck)
        return np.hypot(xs - xi, ys - yi)

    dev = []
    for xi, yi in zip(x_clean, y_clean):
        u0 = fminbound(lambda u: pt_dist(u, xi, yi), 0.0, 1.0, disp=0)
        dev.append(pt_dist(u0, xi, yi))

    # 4) 복원: 원본 인덱스 대비 편차 (중복된 점엔 0 할당)
    full_dev = np.zeros_like(x, dtype=float)
    full_dev[mask] = dev
    return full_dev, tck

# ──────────────────────────────────────────────────────────────
# 통계 출력
# ──────────────────────────────────────────────────────────────
def print_stats(tag: str, errs: np.ndarray):
    mean = errs.mean()
    std  = errs.std()
    rmse = np.sqrt(np.mean(errs ** 2))
    rng  = errs.max() - errs.min()
    print(f"[{tag}] Mean: {mean:.3f} m | Std: {std:.3f} m | RMSE: {rmse:.3f} m | Range: {rng:.3f} m")

# ──────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────
def plot_path(x, y, slope, intercept, tck):
    """원본 데이터 + 회귀선 + 스플라인 곡선"""
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=10, alpha=0.6, label="Data Points")
    # 회귀선
    xs = np.linspace(x.min(), x.max(), 200)
    plt.plot(xs, slope*xs + intercept, "g--", lw=1.5, label="Regression Line")
    # 스플라인 곡선
    us = np.linspace(0,1,400)
    xs_s, ys_s = splev(us, tck)
    plt.plot(xs_s, ys_s, "r-", lw=1.5, label="Spline Curve")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.title("Data & Regression / Spline")
    plt.legend(); plt.grid(True)

def plot_deviation(line_dev: np.ndarray, spline_dev: np.ndarray):
    """선형 vs. 스플라인 편차 비교"""
    idx = np.arange(len(line_dev))
    plt.figure(figsize=(6,4))
    plt.plot(idx, line_dev, "g-o", label="Linear Deviation")
    plt.plot(idx, spline_dev, "r--s", label="Spline Deviation")
    plt.xlabel("Sample Index"); plt.ylabel("Deviation [m]")
    plt.title("Linear vs. Spline Deviation Comparison")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    csv_file = "waypoints_dynamic_3.csv"  # ← 실제 파일명으로 수정

    # 1) 데이터 로드
    x, y = load_data(csv_file)

    # 2) 선형 편차
    lin_dev, slope, intercept = linear_deviation(x, y)
    print_stats("Linear Regression", lin_dev)

    # 3) 스플라인 편차 (중복 제거 + 스무딩)
    spl_dev, tck = spline_deviation(x, y, smoothing=1.0, degree=3)
    print_stats("Spline", spl_dev)

    # 4) 시각화
    plot_path(x, y, slope, intercept, tck)
    plot_deviation(lin_dev, spl_dev)

if __name__ == "__main__":
    main()
