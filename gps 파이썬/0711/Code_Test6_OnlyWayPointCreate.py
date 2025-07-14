#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_waypoints.py
raw CSV → 동적 웨이포인트 생성 및 시각화
  - 상단 IN_CSV, OUT_CSV 변수 설정
  - raw CSV 로드
  - 곡률 기반 동적 웨이포인트 추출
  - 시각화 함수로 원본 궤적과 웨이포인트 플롯
  - CSV 저장 (중복 시 자동 인덱싱)
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── 파일명 변수 ─────────────────────────────────────────
IN_CSV  = "raw_track_xy_6.csv"     # 입력 raw CSV
OUT_CSV = "waypoints_dynamic_test1.csv" # 출력 WPT CSV

# ── 웨이포인트 생성 파라미터 ─────────────────────────────
K_TH1, K_TH2               = 0.10, 0.20
D_STRAIGHT, D_MID, D_CURVE = 0.55, 0.40, 0.28

# ── 헬퍼 함수 ────────────────────────────────────────────
def unique_filepath(dirpath: Path, basename: str, ext: str = ".csv") -> Path:
    p = dirpath / f"{basename}{ext}"
    if not p.exists():
        return p
    i = 1
    while True:
        p2 = dirpath / f"{basename}_{i}{ext}"
        if not p2.exists():
            return p2
        i += 1


def save_csv(path: Path, arr: np.ndarray, header: list):
    np.savetxt(path, arr, delimiter=",",
               header=",".join(header), comments="", fmt="%.6f")

# ── 곡률 및 동적 웨이포인트 생성 ───────────────────────────
def curvature(xs, ys):
    x = np.asarray(xs); y = np.asarray(ys)
    dx = np.gradient(x); dy = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    num = dx*ddy - dy*ddx
    den = (dx*dx + dy*dy)**1.5
    k = np.zeros_like(x)
    mask = den > 1e-6
    k[mask] = np.abs(num[mask] / den[mask])
    return k


def dynamic_waypoints(xs, ys):
    κ = curvature(xs, ys)
    wpts = [(xs[0], ys[0])]
    acc = 0.0
    for i in range(1, len(xs)):
        seg = math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
        acc += seg
        d_t = D_STRAIGHT if κ[i] < K_TH1 else (D_MID if κ[i] < K_TH2 else D_CURVE)
        while acc >= d_t:
            over = acc - d_t
            ratio = (seg - over) / seg
            x_wp = xs[i-1] + (xs[i]-xs[i-1]) * ratio
            y_wp = ys[i-1] + (ys[i]-ys[i-1]) * ratio
            wpts.append((x_wp, y_wp))
            acc -= d_t
    wpts.append((xs[-1], ys[-1]))
    return wpts

# ── 시각화 함수 ───────────────────────────────────────────
def visualize(xs, ys, wpts):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(xs, ys, '.-', label='Raw Track', alpha=0.6)
    w = np.array(wpts)
    ax.plot(w[:,0], w[:,1], '.', ms=3, label='Waypoints')
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Raw Track & Dynamic Waypoints')
    ax.legend()
    ax.grid(True)
    plt.show()

# ── 메인 실행 ─────────────────────────────────────────────
def main():
    raw_path = Path(IN_CSV)
    if not raw_path.exists():
        raise FileNotFoundError(f"입력 파일 없음: {IN_CSV}")

    # raw CSV 로드
    data = np.loadtxt(raw_path, delimiter=",", skiprows=1)
    xs, ys = data[:,0], data[:,1]

    # 웨이포인트 생성
    wpts = dynamic_waypoints(xs, ys)

    # 시각화
    visualize(xs, ys, wpts)

    # CSV 저장
    out_basename = Path(OUT_CSV).stem
    out_path = unique_filepath(raw_path.parent, out_basename)
    save_csv(out_path, np.array(wpts), ["X_m", "Y_m"])
    print(f"[Waypoint] 저장 완료 → {out_path}")

if __name__ == "__main__":
    main()
