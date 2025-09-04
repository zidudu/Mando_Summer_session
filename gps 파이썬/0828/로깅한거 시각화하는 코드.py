#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_current_xy.py
────────────────────────────────────────────
waypoint_log_xxx.csv 파일에서 current_x, current_y 컬럼만 시각화
"""

import pandas as pd
import matplotlib.pyplot as plt

# ── CSV 파일 경로 (수정하세요) ─────────────────────
CSV_PATH = "/home/root1/catkin_ws/src/rtk_waypoint_tracker/config/waypoint_log_20250903_185341.csv"

def main():
    # CSV 읽기
    df = pd.read_csv(CSV_PATH)

    # current_x, current_y 추출
    xs = df["current_x"].values
    ys = df["current_y"].values

    # 시각화
    plt.figure(figsize=(8,8))
    plt.plot(xs, ys, 'r.-', label="Current Path")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Current XY Path from CSV")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
