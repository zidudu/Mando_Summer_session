#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_waypoints_compare_hardcoded.py
- 두 개의 웨이포인트 CSV 파일 경로를 코드에 하드코딩하여 비교 플롯을 생성합니다.
- 설정은 아래 '사용자 설정' 블록에서 직접 수정하세요.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
# GUI 없을 수도 있으니 안전하게 Agg 사용
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------- 사용자 설정 (여기서 하드코딩) ----------------
WPT_PATH_A = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/u-center-GMap_v2_0921.csv"
WPT_PATH_B = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/raw_track_latlon_19_용인전체_float.csv"

# 출력 파일명 (None이면 자동 생성)
OUT_PNG = None  # 예: "/home/jigu/compare_AB.png" 또는 None

# 재샘플 간격 및 원 반경, 어노테이션 설정
WAYPOINT_SPACING   = 2.0    # [m], 0 이면 재샘플링 안함
TARGET_RADIUS_END  = 2.0    # [m] 웨이포인트 도달 반경 원
ANNOTATE_WP_INDEX  = True
DRAW_WP_CIRCLES    = True
FIGSIZE = (10, 10)
# ------------------------------------------------------------

def latlon_to_mercator(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def find_col(df, key):
    keys = {c.lower(): c for c in df.columns}
    return keys.get(key.lower(), None)

def load_wpt_latlon_or_xy(path):
    """CSV에서 자동으로 lat/lon 또는 x/y를 찾아 (x_m, y_m) numpy 반환."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"웨이포인트 CSV를 찾을 수 없습니다: {path}")
    # 인코딩/파싱 유연성 위해 utf-8-sig 사용
    df = pd.read_csv(path, encoding='utf-8-sig', engine='python')
    cols = [str(c).strip() for c in df.columns]
    lcol = [c.lower() for c in cols]

    # lat/lon 탐색
    lat_col = None; lon_col = None
    for cand in ('lat', 'latitude'):
        if cand in lcol:
            lat_col = cols[lcol.index(cand)]; break
    for cand in ('lon', 'longitude', 'lng'):
        if cand in lcol:
            lon_col = cols[lcol.index(cand)]; break
    if lat_col is not None and lon_col is not None:
        lats = pd.to_numeric(df[lat_col], errors='coerce').to_numpy()
        lons = pd.to_numeric(df[lon_col], errors='coerce').to_numpy()
        mask = np.isfinite(lats) & np.isfinite(lons)
        if not np.any(mask):
            raise RuntimeError(f"{path}에서 유효한 lat/lon 값을 찾을 수 없습니다.")
        lats = lats[mask]; lons = lons[mask]
        merc = np.array([latlon_to_mercator(float(lat), float(lon)) for lat, lon in zip(lats, lons)])
        return merc[:,0], merc[:,1], 'latlon'

    # x/y 탐색
    x_col = None; y_col = None
    for cand in ('x', 'east', 'easting'):
        if cand in lcol:
            x_col = cols[lcol.index(cand)]; break
    for cand in ('y', 'north', 'northing'):
        if cand in lcol:
            y_col = cols[lcol.index(cand)]; break
    if x_col is not None and y_col is not None:
        xs = pd.to_numeric(df[x_col], errors='coerce').to_numpy()
        ys = pd.to_numeric(df[y_col], errors='coerce').to_numpy()
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            raise RuntimeError(f"{path}에서 유효한 x/y 값을 찾을 수 없습니다.")
        return xs[mask], ys[mask], 'xy'

    # 못 찾음
    raise RuntimeError(f"{path}에서 'lat/lon' 또는 'x/y' 컬럼을 찾지 못했습니다. 컬럼 목록: {', '.join(cols)}")

def generate_waypoints_along_path(path_pts, spacing=3.0):
    """선분 보간 기반 재샘플링 (원래 함수와 동일 로직)."""
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

def plot_compare_hardcoded():
    # 로드
    x1m, y1m, mode1 = load_wpt_latlon_or_xy(WPT_PATH_A)
    x2m, y2m, mode2 = load_wpt_latlon_or_xy(WPT_PATH_B)

    # 공통 원점: 첫 파일의 첫 점을 (0,0)으로 맞춤
    origin_x = float(x1m[0]); origin_y = float(y1m[0])
    x1 = x1m - origin_x; y1 = y1m - origin_y
    x2 = x2m - origin_x; y2 = y2m - origin_y

    # 재샘플링
    if WAYPOINT_SPACING and WAYPOINT_SPACING > 0:
        spaced1 = generate_waypoints_along_path(np.column_stack((x1, y1)), spacing=WAYPOINT_SPACING)
        spaced2 = generate_waypoints_along_path(np.column_stack((x2, y2)), spacing=WAYPOINT_SPACING)
        s1x, s1y = spaced1[:,0], spaced1[:,1]
        s2x, s2y = spaced2[:,0], spaced2[:,1]
    else:
        s1x, s1y = x1, y1
        s2x, s2y = x2, y2

    # 플롯
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(x1, y1, color='tab:green', linestyle='-', linewidth=1.0,
            label=f'Path A (raw) [{os.path.basename(WPT_PATH_A)}]')
    ax.plot(x2, y2, color='tab:orange', linestyle='-', linewidth=1.0,
            label=f'Path B (raw) [{os.path.basename(WPT_PATH_B)}]')

    ax.plot(s1x, s1y, 'b.-', markersize=4, label=f'Waypoints A spaced {WAYPOINT_SPACING:.1f} m')
    ax.plot(s2x, s2y, 'r.-', markersize=4, label=f'Waypoints B spaced {WAYPOINT_SPACING:.1f} m')

    # A 웨이포인트: 인덱스/원
    for idx, (xw, yw) in enumerate(zip(s1x, s1y), start=1):
        ax.scatter([xw], [yw], c='navy', s=20, marker='.', zorder=4)
        if ANNOTATE_WP_INDEX:
            ax.text(xw, yw, str(idx), fontsize=7, ha='center', va='center', zorder=6, color='navy')
        if DRAW_WP_CIRCLES:
            circ = Circle((xw, yw), TARGET_RADIUS_END, color='blue', fill=False,
                          linestyle='--', alpha=0.25, linewidth=0.8)
            ax.add_patch(circ)

    # B 웨이포인트: 인덱스/원 (약간 오프셋된 텍스트)
    for idx, (xw, yw) in enumerate(zip(s2x, s2y), start=1):
        ax.scatter([xw], [yw], c='darkred', s=20, marker='.', zorder=4)
        if ANNOTATE_WP_INDEX:
            ax.text(xw + 0.05, yw + 0.05, str(idx), fontsize=7, ha='left', va='bottom', zorder=6, color='darkred')
        if DRAW_WP_CIRCLES:
            circ = Circle((xw, yw), TARGET_RADIUS_END, color='red', fill=False,
                          linestyle=':', alpha=0.25, linewidth=0.8)
            ax.add_patch(circ)

    ax.set_xlabel("X [m] (local)")
    ax.set_ylabel("Y [m] (local)")
    ax.set_title(f"Waypoints Compare\nA: {os.path.basename(WPT_PATH_A)}  vs  B: {os.path.basename(WPT_PATH_B)}")
    ax.axis('equal'); ax.grid(True)
    ax.legend(loc='upper left')

    plt.tight_layout()
    out_png = OUT_PNG
    if out_png is None:
        base1 = os.path.splitext(os.path.basename(WPT_PATH_A))[0]
        base2 = os.path.splitext(os.path.basename(WPT_PATH_B))[0]
        out_png = f"{base1}_VS_{base2}_compare.png"
    plt.savefig(out_png, dpi=180)
    print(f"[OK] Plot saved as: {out_png}")

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    try:
        plot_compare_hardcoded()
    except Exception as e:
        print("[ERROR]", e)
