#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_current_vs_waypoints.py
────────────────────────────────────────────
1) waypoint_log_xxx.csv → current_x, current_y (실제 주행 경로)
2) waypoints_xxx.csv   → Lat, Lon → X, Y (목표 경로)
두 개를 같은 플롯에 표시하여 경로 비교
"""

import pandas as pd
import matplotlib.pyplot as plt
import math

# ── 파일 경로 (수정하세요) ─────────────────────────
LOG_CSV  = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/logs/waypoint_log_20250908_190906.csv"
WPT_CSV  = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/raw_track_latlon_14.csv"

# ── 위경도 → XY 변환 (Web Mercator) ────────────────
def latlon_to_meters(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def main():
    # ── 주행 로그 읽기 ──────────────────────────────
    log_df = pd.read_csv(LOG_CSV)
    xs = log_df["x"].values
    ys = log_df["y"].values

    # ── 웨이포인트 CSV 읽기 ─────────────────────────
    wpt_df = pd.read_csv(WPT_CSV)
    if not {'lat','Lon'}.issubset(wpt_df.columns):
        raise RuntimeError("웨이포인트 CSV에 'Lat','Lon' 컬럼이 필요합니다.")
    wpt_coords = [latlon_to_meters(lat, lon) for lat, lon in zip(wpt_df['Lat'], wpt_df['Lon'])]
    wpt_xs, wpt_ys = zip(*wpt_coords)

    # ── 시각화 ─────────────────────────────────────
    plt.figure(figsize=(8,8))
    
    # 목표 웨이포인트 경로
    plt.plot(wpt_xs, wpt_ys, 'b.-', label="Waypoint Path")

    # 실제 주행 경로
    plt.plot(xs, ys, 'r.-', label="Current Path")

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Waypoint vs Current Path")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
