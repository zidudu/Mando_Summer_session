#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exit_radius_simulation.py
• Waypoint 경로와 exit 반경을 벗어났을 때
  재진입 지점을 시뮬레이션하여 시각화
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ───────────────────────────────────────────────
# 1) 재진입 후보 선정 함수 (거리+각도 cost 기반)
# ───────────────────────────────────────────────
def nearest_point_on_path(px, py, wx, wy):
    min_d2, best = 1e9, None
    for i in range(len(wx)-1):
        ax, ay = wx[i],   wy[i]
        bx, by = wx[i+1], wy[i+1]
        vx, vy = bx-ax, by-ay
        seg2 = vx*vx + vy*vy
        t = 0 if seg2==0 else max(0, min(1, ((px-ax)*vx + (py-ay)*vy)/seg2))
        cx, cy = ax + t*vx, ay + t*vy
        d2 = (px-cx)**2 + (py-cy)**2
        if d2 < min_d2:
            min_d2, best = d2, (i, t, (cx, cy), math.sqrt(d2))
    return best  # (i, t, (cx,cy), dist)

def select_rejoin_point(px, py, yaw, wx, wy,
                        current_target, N_forward=3,
                        w_d=1.0, w_a=2.0, d_thresh=5.0):
    hx, hy = math.cos(yaw), math.sin(yaw)
    best_cost, best = float('inf'), None

    start_i = max(0, current_target-1)
    end_i   = min(len(wx)-1, current_target+N_forward)
    for i in range(start_i, end_i):
        ax, ay = wx[i],   wy[i]
        bx, by = wx[i+1], wy[i+1]
        vx, vy = bx-ax, by-ay
        seg2 = vx*vx + vy*vy
        t = 0 if seg2==0 else max(0, min(1, ((px-ax)*vx + (py-ay)*vy)/seg2))
        cx, cy = ax + t*vx, ay + t*vy

        # 전방 후보만
        if (cx-px)*hx + (cy-py)*hy < 0:
            continue

        d = math.hypot(cx-px, cy-py)
        angle_to = math.atan2(cy-py, cx-px)
        da = abs((angle_to - yaw + math.pi) % (2*math.pi) - math.pi)

        cost = w_d*d + w_a*da
        if cost < best_cost:
            best_cost, best = cost, (i, t, (cx, cy), d)

    # fallback
    if best is None or best[3] > d_thresh:
        return nearest_point_on_path(px, py, wx, wy)
    return best  # (i, t, (cx,cy), dist)

# ───────────────────────────────────────────────
# 2) 시뮬레이션 설정
# ───────────────────────────────────────────────
# 예시 웨이포인트 (직선 + 곡선)
wx = np.array([0, 5, 10, 15, 20])
wy = np.array([0, 0,  0,  0,  0])

exit_r = 2.0                      # exit 반경 [m]
current_target = 1                # 현재 목표 WP 인덱스 (0-based)
yaw = math.radians(0)             # 차량 헤딩 (도로 진행 방향)

# 벗어난 위치 샘플 (exit_r 밖)
test_positions = [
    (6.0, 3.0),  # 위쪽으로 튕겨나간 경우
    (4.0,-3.0),  # 아래쪽으로 튕겨나간 경우
    (12.0, 2.5)  # 중간 곡선 구간
]

# ───────────────────────────────────────────────
# 3) 시뮬레이션 및 시각화
# ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(wx, wy, 'k.-', label='Waypoints')
for i,(x,y) in enumerate(zip(wx,wy)):
    ax.text(x,y+0.5,str(i+1), ha='center')

# exit 반경 원
ax.add_patch(Circle((wx[current_target], wy[current_target]),
                    exit_r, ec='r', ls='--', fill=False,
                    label='Exit Radius'))

for px, py in test_positions:
    # 거리 검사
    d_wp = math.hypot(px-wx[current_target], py-wy[current_target])
    if d_wp > exit_r:
        # 재진입점 계산
        i, t, (rx,ry), dist = select_rejoin_point(
            px, py, yaw, wx, wy, current_target)
        # 플롯
        ax.plot(px, py, 'ro', label='Out Position' if 'Out Position' not in ax.get_legend_handles_labels()[1] else '')
        ax.plot(rx, ry, 'bx', ms=10, mew=2,
                label='Rejoin Point' if 'Rejoin Point' not in ax.get_legend_handles_labels()[1] else '')
        ax.plot([px, rx],[py, ry],'b--')

ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Exit Radius 벗어날 때 재진입 시뮬레이션')
ax.legend()
ax.grid(True)
plt.show()
