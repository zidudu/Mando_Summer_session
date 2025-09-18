#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_waypoints_only.py
- 웨이포인트 CSV만 사용하여 플롯
- 웨이포인트 간격, 원 반경, WP 인덱스 표시
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
# GUI 없을 수도 있으니 안전하게 Agg
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------- 사용자 설정 ----------------
WPT_CSV  = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/raw_track_latlon_18.csv"
OUT_PNG  = os.path.splitext(os.path.basename(WPT_CSV))[0] + "_waypoints.png"

# 표시/분석 파라미터
WAYPOINT_SPACING   = 2.5   # [m] 재샘플링 간격 (0 이면 원래 포인트 그대로 사용)
TARGET_RADIUS_END  = 2.0   # [m] 이 반경 안이면 도달 원 표시
ANNOTATE_WP_INDEX  = True
DRAW_WP_CIRCLES    = True
# ----------------------------------------------

def latlon_to_mercator(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def find_col(df, key):
    keys = {c.lower(): c for c in df.columns}
    return keys.get(key.lower(), None)

def load_wpt_latlon(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"웨이포인트 CSV를 찾을 수 없습니다: {path}")
    wdf = pd.read_csv(path)
    lat_col = find_col(wdf, 'lat')
    lon_col = find_col(wdf, 'lon')
    if lat_col is not None and lon_col is not None:
        lats = wdf[lat_col].astype(float).values
        lons = wdf[lon_col].astype(float).values
        merc = [latlon_to_mercator(float(lat), float(lon)) for lat, lon in zip(lats, lons)]
        wx, wy = zip(*merc)
        return np.array(wx), np.array(wy)
    x_col = find_col(wdf, 'x'); y_col = find_col(wdf, 'y')
    if x_col is not None and y_col is not None:
        return wdf[x_col].astype(float).values, wdf[y_col].astype(float).values
    raise RuntimeError("웨이포인트 CSV에 'Lat','Lon' 또는 'x','y' 컬럼이 필요합니다.")

# waypoint_tracker_node.py의 generate_waypoints_along_path과 동일한 동작(선분 보간)
def generate_waypoints_along_path(path_pts, spacing=3.0):
    pts = [tuple(p) for p in path_pts]
    if spacing is None or spacing <= 0 or len(pts) < 2:
        return np.array(pts)
    new_points = [pts[0]]
    last_point = np.array(pts[0], dtype=float)
    dist_accum = 0.0
    for i in range(1, len(pts)):
        current_point = np.array(pts[i], dtype=float)
        seg = current_point - last_point
        seg_len = np.linalg.norm(seg)
        while seg_len + dist_accum >= spacing:
            remain = spacing - dist_accum
            if seg_len == 0:
                break
            direction = seg / seg_len
            new_point = last_point + direction * remain
            new_points.append(tuple(new_point))
            last_point = new_point
            seg = current_point - last_point
            seg_len = np.linalg.norm(seg)
            dist_accum = 0.0
        dist_accum += seg_len
        last_point = current_point
    if np.linalg.norm(np.array(new_points[-1]) - np.array(pts[-1])) > 1e-6:
        new_points.append(tuple(pts[-1]))
    return np.array(new_points)

def main():
    # 웨이포인트 로드
    wpt_xm, wpt_ym = load_wpt_latlon(WPT_CSV)
    if wpt_xm.size == 0:
        raise RuntimeError("웨이포인트에 좌표 데이터가 없습니다.")

    # 원점 이동(첫 좌표 = 0,0)
    origin_x = float(wpt_xm[0]); origin_y = float(wpt_ym[0])
    wpt_x = wpt_xm - origin_x
    wpt_y = wpt_ym - origin_y

    # 재샘플링
    if WAYPOINT_SPACING and WAYPOINT_SPACING > 0:
        spaced = generate_waypoints_along_path(np.column_stack((wpt_x, wpt_y)), spacing=WAYPOINT_SPACING)
        spaced_x, spaced_y = spaced[:,0], spaced[:,1]
    else:
        spaced_x, spaced_y = wpt_x, wpt_y

    # 플롯
    fig, ax = plt.subplots(figsize=(9,9))
    ax.plot(wpt_x, wpt_y, 'g-', linewidth=1.0, label='CSV Path (raw)')
    ax.plot(spaced_x, spaced_y, 'b.-', markersize=4, label=f'Waypoints spaced {WAYPOINT_SPACING:.1f} m')

    for idx, (xw, yw) in enumerate(zip(spaced_x, spaced_y), start=1):
        if ANNOTATE_WP_INDEX:
            ax.text(xw, yw, str(idx), fontsize=7, ha='center', va='center', zorder=5)
        if DRAW_WP_CIRCLES:
            circ = Circle((xw, yw), TARGET_RADIUS_END, color='blue', fill=False, linestyle='--', alpha=0.25, linewidth=0.8)
            ax.add_patch(circ)
        ax.scatter([xw], [yw], c='navy', s=20, marker='.', zorder=4)

    ax.set_xlabel("X [m] (local)")
    ax.set_ylabel("Y [m] (local)")
    ax.set_title("Waypoints Only")
    ax.axis('equal'); ax.grid(True)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"[OK] Plot saved as: {OUT_PNG}")

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
