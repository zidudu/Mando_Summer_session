#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_current_vs_waypoints_with_wpinfo.py
- 로그 CSV의 lat,lon을 사용하여 웨이포인트와 함께 플롯
- 웨이포인트 간격, 원 반경, WP 인덱스, 방문 여부, 오차 통계 표시
"""
import os
import math
import numpy as np
import pandas as pd
import matplotlib
# GUI가 없을 수 있으므로 안전하게 Agg 사용
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------- 사용자 설정 ----------------
LOG_CSV  = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/logs/waypoint_log_20250908_175635.csv"
WPT_CSV  = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/raw_track_latlon_6.csv"
OUT_PNG  = os.path.splitext(os.path.basename(LOG_CSV))[0] + "_vs_wpt_augmented.png"

# 표시/분석 파라미터
WAYPOINT_SPACING   = 2.5   # [m] 재샘플링 간격 (0 이면 원래 포인트 그대로 사용)
TARGET_RADIUS_END  = 2.0   # [m] 이 반경 안이면 '해당 웨이포인트 방문'으로 간주
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

def load_log_latlon(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"로그 CSV를 찾을 수 없습니다: {path}")
    df = pd.read_csv(path)
    lat_col = find_col(df, 'lat')
    lon_col = find_col(df, 'lon')
    if lat_col is None or lon_col is None:
        raise RuntimeError("로그 CSV에 'lat','lon' 컬럼이 필요합니다. (대소문자 확인)")
    lats = df[lat_col].astype(float).values
    lons = df[lon_col].astype(float).values
    merc = [latlon_to_mercator(float(lat), float(lon)) for lat, lon in zip(lats, lons)]
    mx, my = zip(*merc)
    return np.array(mx), np.array(my), df

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
    """
    path_pts: iterable of (x,y) 좌표 (np.array 혹은 리스트)
    spacing: 원하는 등간격 [m] (spacing <= 0 이면 입력경로 그대로 반환)
    """
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
        # 만약 이전에 남은 길이와 합쳐서 spacing 이상이면 보간
        while seg_len + dist_accum >= spacing:
            remain = spacing - dist_accum
            if seg_len == 0:
                break
            direction = seg / seg_len
            new_point = last_point + direction * remain
            new_points.append(tuple(new_point))
            # 남은 세그먼트 재설정
            last_point = new_point
            seg = current_point - last_point
            seg_len = np.linalg.norm(seg)
            dist_accum = 0.0
        dist_accum += seg_len
        last_point = current_point
    # 끝점 추가
    if np.linalg.norm(np.array(new_points[-1]) - np.array(pts[-1])) > 1e-6:
        new_points.append(tuple(pts[-1]))
    return np.array(new_points)

def compute_nearest_dists(log_x, log_y, wpt_x, wpt_y):
    """각 로그 포인트마다 가장 가까운 (웨이포인트) 거리와 인덱스 반환"""
    wx = np.asarray(wpt_x); wy = np.asarray(wpt_y)
    n_log = len(log_x)
    dists = np.zeros(n_log)
    idxs  = np.zeros(n_log, dtype=int)
    for i in range(n_log):
        dx = wx - log_x[i]
        dy = wy - log_y[i]
        d2 = dx*dx + dy*dy
        j = int(np.argmin(d2))
        dists[i] = math.hypot(wx[j]-log_x[i], wy[j]-log_y[i])
        idxs[i] = j
    return dists, idxs

def compute_stats(dist_arr):
    mean = float(np.mean(dist_arr))
    std  = float(np.std(dist_arr))
    rmse = float(np.sqrt(np.mean(dist_arr**2)))
    mx   = float(np.max(dist_arr))
    mn   = float(np.min(dist_arr))
    return mean, std, rmse, mn, mx

def compute_text_offset(xs, ys):
    if len(xs) == 0 or len(ys) == 0:
        return 0.3, 0.3
    range_x = max(xs) - min(xs)
    range_y = max(ys) - min(ys)
    span = max(range_x, range_y, 1.0)
    off = span * 0.006
    return off, off

def mark_point(ax, x, y, label, color='k', marker='o', text_offset=(0.3,0.3), fontsize=10):
    ax.scatter([x], [y], c=color, s=60, marker=marker, zorder=6)
    ax.text(x + text_offset[0], y + text_offset[1], str(label),
            fontsize=fontsize, fontweight='bold', va='center', ha='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), zorder=7)

def main():
    # load
    log_xm, log_ym, log_df = load_log_latlon(LOG_CSV)
    wpt_xm, wpt_ym = load_wpt_latlon(WPT_CSV)

    if log_xm.size == 0:
        raise RuntimeError("로그에 좌표 데이터가 없습니다.")
    if wpt_xm.size == 0:
        raise RuntimeError("웨이포인트에 좌표 데이터가 없습니다.")

    # origin shift: 로그의 첫 좌표를 0,0 으로
    origin_x = float(log_xm[0]); origin_y = float(log_ym[0])
    log_x = log_xm - origin_x
    log_y = log_ym - origin_y
    wpt_x = wpt_xm - origin_x
    wpt_y = wpt_ym - origin_y

    # 재샘플링 (spacing)
    if WAYPOINT_SPACING is not None and WAYPOINT_SPACING > 0:
        spaced = generate_waypoints_along_path(np.column_stack((wpt_x, wpt_y)), spacing=WAYPOINT_SPACING)
        spaced_x = spaced[:,0]; spaced_y = spaced[:,1]
    else:
        spaced_x = wpt_x; spaced_y = wpt_y

    # 로그 포인트 -> 가장 가까운 웨이포인트(재샘플된)까지의 거리 계산
    dists, nearest_idxs = compute_nearest_dists(log_x, log_y, spaced_x, spaced_y)
    mean, std, rmse, mn, mx = compute_stats(dists)

    # 어떤 웨이포인트가 '방문'됐는지 (어떤 로그 포인트가 반경 이내인지로 판단)
    visited = np.zeros(len(spaced_x), dtype=bool)
    for i, dist in enumerate(dists):
        if dist <= TARGET_RADIUS_END:
            visited[nearest_idxs[i]] = True

    # 플롯
    fig, ax = plt.subplots(figsize=(9,9))
    # 원래 경로(원시) - 연두
    ax.plot(wpt_x, wpt_y, 'g-', linewidth=1.0, label='CSV Path (raw)')
    # 재샘플된 웨이포인트 - 파랑
    ax.plot(spaced_x, spaced_y, 'b.-', markersize=4, label=f'Waypoints spaced {WAYPOINT_SPACING:.1f} m')
    # 로그 - 빨강
    ax.plot(log_x, log_y, 'r.-', linewidth=1.0, label='Log Path (shifted)')
    ax.scatter(log_x[0], log_y[0], c='green', s=80, marker='o', label='Log Start (0)')
    ax.scatter(log_x[-1], log_y[-1], c='black', s=80, marker='s', label='Log End (1)')

    # 웨이포인트 인덱스 및 원(도달 반경)
    ox, oy = compute_text_offset(np.concatenate((log_x, spaced_x)), np.concatenate((log_y, spaced_y)))
    for idx, (xw, yw) in enumerate(zip(spaced_x, spaced_y), start=1):
        if ANNOTATE_WP_INDEX:
            ax.text(xw, yw, str(idx), fontsize=7, ha='center', va='center', zorder=5)
        if DRAW_WP_CIRCLES:
            circ = Circle((xw, yw), TARGET_RADIUS_END, color='blue', fill=False, linestyle='--', alpha=0.25, linewidth=0.8)
            ax.add_patch(circ)
        # 방문한 웨이포인트는 채워진 녹색 원으로 표시
        if visited[idx-1]:
            ax.scatter([xw], [yw], c='limegreen', s=60, marker='o', zorder=7)
        else:
            ax.scatter([xw], [yw], c='navy', s=20, marker='.', zorder=4)

    # 히트맵처럼 최근 인덱스 분포를 보여주고 싶으면(선택): 주석 처리된 코드 참고
    # ax.scatter(log_x, log_y, c=nearest_idxs, cmap='tab20', s=8)

    # 오차 통계 텍스트 (범례 대신 플롯 영역에 넣기)
    stats_txt = (
        f"Error to nearest WP (n={len(dists)}):\n"
        f" mean={mean:.3f} m  std={std:.3f} m  rmse={rmse:.3f} m\n"
        f" min={mn:.3f} m  max={mx:.3f} m"
    )
    # 오른쪽 위에 텍스트 박스
    ax.text(0.98, 0.98, stats_txt, transform=ax.transAxes, fontsize=9,
            va='top', ha='right', bbox=dict(fc='white', alpha=0.9, ec='0.5'))

    ax.set_xlabel("X [m] (local)")
    ax.set_ylabel("Y [m] (local)")
    ax.set_title("Waypoint vs Current Path — Start:0 End:1")
    ax.axis('equal'); ax.grid(True)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"[OK] Plot saved as: {OUT_PNG}")
    print(f"[INFO] mean={mean:.3f} m  std={std:.3f} m  rmse={rmse:.3f} m  visited_wp_count={visited.sum()}/{len(spaced_x)}")

    # GUI 환경이면 창 띄우기 시도
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
