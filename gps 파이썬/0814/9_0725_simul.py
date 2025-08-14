#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gps_simulation_with_info.py  (웨이포인트 반경 + 헤딩·조향각 시각화)

• gps_wp_log.csv 로부터 time, lat, lon, heading, delta, wp_idx 컬럼을 읽어
  ENU 좌표로 변환 후 Matplotlib FuncAnimation으로
  궤적 애니메이션과 함께 동적 정보 박스, 헤딩/조향각 화살표, 타겟 WP 반경 원을 표시합니다.
"""

import math
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle

# ────────────────────────────────────────────────────────────────
# 사용자 설정
# ────────────────────────────────────────────────────────────────
LOG_CSV       = 'gps_wp_log.csv'      # 로그 CSV
WP_CSV        = r"C:\Users\dlwlr\OneDrive - 한라대학교\바탕 화면\GPS_purse_Python\bocsa\waypoints_2m.csv"
INTERVAL_MS   = 50                    # 애니메이션 프레임 간격 (ms)
ARROW_LEN     = 1.0                   # 헤딩·조향 화살표 길이 (m)
ARRIVE_RADIUS = 0.5                   # 웨이포인트 도달 반경 (m)
# ────────────────────────────────────────────────────────────────

# 1) 로그 불러오기
log = pd.read_csv(LOG_CSV)
# 컬럼: time, lat, lon, heading, delta, wp_idx

# 2) 웨이포인트 불러오기
wp_df = pd.read_csv(WP_CSV)
# 컬럼: Index, Lat, Lon

# 3) 기준점 설정 (로그 첫 행)
lat0 = log['lat'].iloc[0]
lon0 = log['lon'].iloc[0]

# 4) 위경도→ENU 변환 함수
def latlon_to_enu(lat, lon, lat_ref, lon_ref):
    R = 6_378_137.0
    deg2rad = math.pi/180
    x = (lon - lon_ref) * math.cos(lat_ref * deg2rad) * deg2rad * R
    y = (lat - lat_ref) * deg2rad * R
    return x, y

# 5) 로그 ENU 및 파라미터 추출
enu = log.apply(lambda r: latlon_to_enu(r['lat'], r['lon'], lat0, lon0), axis=1)
xs = np.array([p[0] for p in enu])
ys = np.array([p[1] for p in enu])
times    = log['time'].values
headings = np.deg2rad(log['heading'].values)
deltas   = log['delta'].values
targets  = log['wp_idx'].astype(int).values - 1  # 0-based

# 6) 웨이포인트 ENU
wp_enu = wp_df.apply(lambda r: latlon_to_enu(r['Lat'], r['Lon'], lat0, lon0), axis=1)
wx = np.array([p[0] for p in wp_enu])
wy = np.array([p[1] for p in wp_enu])

# ────────────────────────────────────────────────────────────────
# 플롯 설정
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,8))
ax.set_aspect('equal', 'box')
ax.grid(True)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('GPS 궤적 시뮬레이션')

# 전체 궤적
ax.plot(xs, ys, color='lightgray', lw=1, label='Path')

# 웨이포인트 및 도달 반경 원
for i, (x_wp, y_wp) in enumerate(zip(wx, wy), start=1):
    ax.scatter(x_wp, y_wp, c='red', s=20, label='Waypoints' if i==1 else "")
    ax.text(x_wp, y_wp, str(i), fontsize=6, ha='right', va='bottom')
    circle = Circle((x_wp, y_wp), ARRIVE_RADIUS,
                    edgecolor='blue', facecolor='none', linestyle='--',
                    label='Arrive Radius' if i==1 else "")
    ax.add_patch(circle)

# 차량 위치 및 타겟 WP 마커
veh_pt,    = ax.plot([], [], 'o', c='gold', ms=8, label='Vehicle')
target_pt, = ax.plot([], [], 'X', c='magenta', ms=10, label='Target WP')

# 헤딩·조향 화살표
heading_arrow = FancyArrowPatch((0,0),(0,0))
steer_arrow   = FancyArrowPatch((0,0),(0,0))
ax.add_patch(heading_arrow)
ax.add_patch(steer_arrow)

# 정보 박스
info_txt = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                   ha='left', va='top', fontsize=9,
                   bbox=dict(fc='white', alpha=0.7))

ax.legend(loc='lower right')

# ────────────────────────────────────────────────────────────────
# 애니메이션 업데이트 함수
# ────────────────────────────────────────────────────────────────
def update(frame):
    global heading_arrow, steer_arrow

    # 현재 좌표 및 파라미터
    x, y    = xs[frame], ys[frame]
    theta   = headings[frame]
    delta   = deltas[frame]
    tgt     = targets[frame]
    tx, ty  = wx[tgt], wy[tgt]
    dist_wp = math.hypot(tx-x, ty-y)
    time_str = datetime.datetime.fromtimestamp(times[frame]).strftime('%H:%M:%S')

    # 차량 위치
    veh_pt.set_data([x], [y])
    # 타겟 WP
    target_pt.set_data([tx], [ty])

    # 이전 화살표 제거
    heading_arrow.remove()
    steer_arrow.remove()

    # 헤딩 화살표
    hx = x + ARROW_LEN * math.cos(theta)
    hy = y + ARROW_LEN * math.sin(theta)
    heading_arrow = FancyArrowPatch((x, y), (hx, hy),
                                   color='blue', lw=2,
                                   arrowstyle='-|>', mutation_scale=15)
    ax.add_patch(heading_arrow)

    # 조향 화살표
    sx = x + ARROW_LEN * math.cos(theta + math.radians(delta))
    sy = y + ARROW_LEN * math.sin(theta + math.radians(delta))
    steer_arrow = FancyArrowPatch((x, y), (sx, sy),
                                 color='red', lw=2,
                                 arrowstyle='-|>', mutation_scale=15)
    ax.add_patch(steer_arrow)

    # 정보 박스 갱신
    info_txt.set_text(
        f"Time: {time_str}\n"
        f"Heading: {math.degrees(theta):.1f}°\n"
        f"Steering: {delta:.1f}°\n"
        f"Position: ({x:.1f}, {y:.1f}) m\n"
        f"Dist→WP: {dist_wp:.1f} m\n"
        f"Target WP: {tgt+1}"
    )

    return veh_pt, target_pt, heading_arrow, steer_arrow, info_txt

# ────────────────────────────────────────────────────────────────
# FuncAnimation 생성 및 실행
# ────────────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig, update,
    frames=len(xs),
    interval=INTERVAL_MS,
    blit=False,
    repeat=False
)

plt.show()
