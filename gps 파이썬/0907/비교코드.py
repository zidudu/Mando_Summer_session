#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_current_vs_waypoints_use_latlon_with_marks.py
- 로그 CSV의 lat,lon을 사용하여 웨이포인트와 함께 플롯합니다.
- 로그와 웨이포인트를 Web Mercator로 변환하고, 로그의 첫 위치를 원점으로 shift 합니다.
- 각 파일의 시작점(0)과 끝점(1)을 표시합니다.
"""
import os, math
import pandas as pd
import matplotlib
# GUI가 없을 수 있으므로 안전하게 Agg 사용
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- 사용자 경로 (필요시 수정) ----------------
LOG_CSV  = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/logs/waypoint_log_20250908_175635.csv"
WPT_CSV  = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/raw_track_latlon_6.csv"
# ---------------------------------------------------------

def latlon_to_mercator(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def find_col(df, key):
    # 대소문자 무시로 컬럼명 찾아서 실제 컬럼명 반환
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
    return list(mx), list(my), df

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
        return list(wx), list(wy)
    # 혹시 이미 x,y로 되어있으면 사용
    x_col = find_col(wdf, 'x'); y_col = find_col(wdf, 'y')
    if x_col is not None and y_col is not None:
        return list(wdf[x_col].astype(float).values), list(wdf[y_col].astype(float).values)
    raise RuntimeError("웨이포인트 CSV에 'Lat','Lon' 또는 'x','y' 컬럼이 필요합니다.")

def compute_text_offset(xs, ys):
    """축 크기에 상대적인 작은 오프셋 반환 (m 단위)"""
    if len(xs) == 0 or len(ys) == 0:
        return 0.3, 0.3
    range_x = max(xs) - min(xs)
    range_y = max(ys) - min(ys)
    span = max(range_x, range_y, 1.0)
    # 0.6% 정도를 오프셋으로 사용
    off = span * 0.006
    return off, off

def mark_point(ax, x, y, label, color='k', marker='o', text_offset=(0.3,0.3), fontsize=12):
    ax.scatter([x], [y], c=color, s=50, marker=marker, zorder=5)
    ax.text(x + text_offset[0], y + text_offset[1], str(label),
            fontsize=fontsize, fontweight='bold', va='center', ha='center', zorder=6,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))

def main():
    # 로드 및 변환
    log_xm, log_ym, log_df = load_log_latlon(LOG_CSV)
    wpt_xm, wpt_ym = load_wpt_latlon(WPT_CSV)

    if len(log_xm) == 0:
        raise RuntimeError("로그에 좌표 데이터가 없습니다.")
    if len(wpt_xm) == 0:
        raise RuntimeError("웨이포인트에 좌표 데이터가 없습니다.")

    # 로그의 첫 위치를 origin으로 해서 local frame으로 변환 (둘 다 같은 frame으로 맞춤)
    origin_x = log_xm[0]
    origin_y = log_ym[0]
    log_x = [xx - origin_x for xx in log_xm]
    log_y = [yy - origin_y for yy in log_ym]
    wpt_x = [xx - origin_x for xx in wpt_xm]
    wpt_y = [yy - origin_y for yy in wpt_ym]

    # 텍스트 오프셋 (로그와 웨이포인트가 겹치지 않게 방향 다르게 줄 것)
    ox, oy = compute_text_offset(log_x + wpt_x, log_y + wpt_y)

    plt.figure(figsize=(8,8))
    ax = plt.gca()

    # 웨이포인트: 파란
    ax.plot(wpt_x, wpt_y, 'b.-', label="Waypoint Path (shifted, m)", zorder=2)
    # 로그: 빨강
    ax.plot(log_x, log_y,  'r.-', label="Current Path (log lat/lon -> mercator -> shifted, m)", zorder=3)

    # 시작/끝 마커 및 라벨
    # 로그 시작(0) / 끝(1)
    mark_point(ax, log_x[0], log_y[0], 0, color='green', marker='o', text_offset=(ox, oy))
    mark_point(ax, log_x[-1], log_y[-1], 1, color='black', marker='s', text_offset=(ox, -oy))
    # 웨이포인트 시작(0) / 끝(1) — 텍스트 오프셋 방향을 반대로 주어 겹침 완화
    mark_point(ax, wpt_x[0], wpt_y[0], 0, color='blue', marker='^', text_offset=(-ox, oy))
    mark_point(ax, wpt_x[-1], wpt_y[-1], 1, color='blue', marker='v', text_offset=(-ox, -oy))

    # 시각화 설정
    ax.set_xlabel("X [m] (local)")
    ax.set_ylabel("Y [m] (local)")
    ax.set_title("Waypoint vs Current Path (using log lat/lon) — Start:0 End:1")
    ax.axis('equal')
    ax.grid(True)
    ax.legend(loc='upper right')
    plt.tight_layout()

    out_png = os.path.splitext(os.path.basename(LOG_CSV))[0] + "_vs_wpt_latlon_with_marks.png"
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Plot saved as: {out_png}")

    # GUI 환경이면 띄우기 시도
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
