#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pure_pursuit_full.py
──────────────────────────────────────────────────────────────
• Waypoints / Vehicle / Look-ahead 원
• 헤딩(파란)·조향(빨간) 화살표 시각화 (FancyArrowPatch)
• 인덱스 기반 원 바깥 점 중 가장 가까운 인덱스 선택
• 마지막 웨이포인트 도달 시 자동 정지
──────────────────────────────────────────────────────────────
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from pynput import keyboard

# ── 파라미터 ────────────────────────────────────────────────
MODE         = "auto"                 # 'manual' or 'auto'
CSV_WPT      = "waypoints_dynamic_17.csv"
WHEELBASE    = 0.28                   # [m]
DT           = 0.05                   # [s]
MAX_STEER    = math.radians(30)       # ±30°
MAX_SPEED    = 3.0                    # [m/s]
K_LD, K_V    = 1.0, 0.4               # look-ahead 계수
MIN_AUTO_V   = 1.0                    # AUTO 모드 진입 시 속도 [m/s]
ARROW_LEN    = 0.8 * WHEELBASE        # 화살표 길이 [m]

# ── 웨이포인트 로드 ──────────────────────────────────────────
wp = np.loadtxt(CSV_WPT, delimiter=",", skiprows=1)
wp_x, wp_y = wp[:, 0], wp[:, 1]

# ── 상태 초기화 ────────────────────────────────────────────
state = {
    "x": wp_x[0],
    "y": wp_y[0],
    "yaw": math.atan2(wp_y[1]-wp_y[0], wp_x[1]-wp_x[0]),  # 초기 헤딩
    "v": 0.0,
    "delta": 0.0,
}
mode_auto = (MODE.lower() == "auto")
if mode_auto:
    state["v"] = MIN_AUTO_V
# 목표 인덱스 초기화
target_idx = 0

# ── 인덱스 기반 타겟 선정(순방향만) ─────────────────────────────────
def find_target_nearby(idx_current, Ld):
    """
    idx_current: 이전에 선택된 인덱스
    Ld: look-ahead 반경
    가장 가까운 원 바깥 인덱스(순방향) 선택
    """
    global target_idx
    candidates = []
    for i in range(idx_current, len(wp_x)):
        dx_i = wp_x[i] - state['x']
        dy_i = wp_y[i] - state['y']
        if dx_i*dx_i + dy_i*dy_i >= Ld*Ld:
            candidates.append(i)
    if not candidates:
        new_idx = len(wp_x) - 1
    else:
        new_idx = min(candidates)
    target_idx = new_idx
    dx = wp_x[new_idx] - state['x']
    dy = wp_y[new_idx] - state['y']
    return dx, dy

# ── Pure-Pursuit 조향 계산 ─────────────────────────────────
def pure_pursuit_step():
    Ld = K_LD + K_V * state["v"]
    dx, dy = find_target_nearby(target_idx, Ld)
    alpha = math.atan2(dy, dx) - state['yaw']
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # wrap-to-pi
    delta = math.atan2(2 * WHEELBASE * math.sin(alpha), Ld)
    return max(-MAX_STEER, min(MAX_STEER, delta))

# ── 차량 모델 업데이트 ─────────────────────────────────────
def update_vehicle():
    beta = math.atan2(math.tan(state["delta"]), 1.0)
    state["x"]   += state["v"] * math.cos(state["yaw"]+beta) * DT
    state["y"]   += state["v"] * math.sin(state["yaw"]+beta) * DT
    state["yaw"] += state["v"] / WHEELBASE * math.tan(state["delta"]) * DT

# ── 키보드 콜백 ────────────────────────────────────────────
def on_press(key):
    global mode_auto
    try:
        c = key.char.lower()
        if c == 'q':
            return False
        if c == 'm':
            mode_auto = not mode_auto
            print(f"[MODE] {'AUTO' if mode_auto else 'MANUAL'}")
            if mode_auto:
                state['v'] = max(state['v'], MIN_AUTO_V)
    except AttributeError:
        pass
    if not mode_auto:
        if key == keyboard.Key.up:    state['v'] = min(MAX_SPEED, state['v']+0.2)
        elif key == keyboard.Key.down: state['v'] = max(0.0,       state['v']-0.2)
        elif key == keyboard.Key.left: state['delta'] = max(-MAX_STEER, state['delta']-math.radians(2))
        elif key == keyboard.Key.right:state['delta'] = min(MAX_STEER,  state['delta']+math.radians(2))
        elif key == keyboard.Key.space:state['v'] = 0.0

# ── 그래프 & 오브젝트 초기화 ────────────────────────────────
fig, ax = plt.subplots(figsize=(7,7))
ax.plot(wp_x, wp_y, 'k-', lw=1.2, label='Reference')
for i, (x, y) in enumerate(zip(wp_x, wp_y)):
    ax.text(x, y, str(i), fontsize=7, ha='right', va='bottom')
car_pt,  = ax.plot([], [], 'bo', ms=6, label='Vehicle')
look_pt, = ax.plot([], [], 'rs', ms=5, label='Look-ahead')
circle   = plt.Circle((0,0), 0, ec='green', fill=False, lw=1.3)
ax.add_patch(circle)
heading_arrow = FancyArrowPatch((0,0),(0,0), color='blue', mutation_scale=12, lw=1.5)
steer_arrow   = FancyArrowPatch((0,0),(0,0), color='red',  mutation_scale=12, lw=1.5)
ax.add_patch(heading_arrow)
ax.add_patch(steer_arrow)
ax.set_xlim(wp_x.min()-5, wp_x.max()+5)
ax.set_ylim(wp_y.min()-5, wp_y.max()+5)
ax.set_aspect('equal')
ax.grid(True)
ax.legend(loc='upper right')

# ── 애니메이션 루프 ────────────────────────────────────────
def animate(_):
    global mode_auto
    if mode_auto:
        state['delta'] = pure_pursuit_step()
    update_vehicle()
    
    # 마지막 웨이포인트 근접 시 정지 (실제 거리 기준만)
    dx_end = wp_x[-1] - state['x']
    dy_end = wp_y[-1] - state['y']
    if dx_end*dx_end + dy_end*dy_end < 0.5**2:
        state['v'] = 0.0
        mode_auto = False

    # 차량 위치 갱신
    car_pt.set_data([state['x']], [state['y']])

    # Look-ahead 및 원 표시
    Ld = K_LD + K_V * state['v']
    dx = wp_x[target_idx] - state['x']
    dy = wp_y[target_idx] - state['y']
    look_pt.set_data([state['x']+dx], [state['y']+dy])
    look_pt.set_visible(mode_auto)
    if mode_auto:
        circle.center, circle.radius = (state['x'], state['y']), Ld
    else:
        circle.radius = 0

    # 헤딩 화살표
    hx, hy = state['x'], state['y']
    hx2 = hx + math.cos(state['yaw']) * ARROW_LEN
    hy2 = hy + math.sin(state['yaw']) * ARROW_LEN
    heading_arrow.set_positions((hx, hy), (hx2, hy2))

    # 조향 화살표
    fx = hx + math.cos(state['yaw']) * WHEELBASE
    fy = hy + math.sin(state['yaw']) * WHEELBASE
    sx = fx + math.cos(state['yaw'] + state['delta']) * ARROW_LEN
    sy = fy + math.sin(state['yaw'] + state['delta']) * ARROW_LEN
    steer_arrow.set_positions((fx, fy), (sx, sy))

    return car_pt, look_pt, circle, heading_arrow, steer_arrow

ani = animation.FuncAnimation(fig, animate, interval=DT*1000, blit=False)

# ── 실행 ───────────────────────────────────────────────────
listener = keyboard.Listener(on_press=on_press)
listener.start()
print("시작 모드:", "AUTO" if mode_auto else "MANUAL", "–  ↑↓ 속도  ←→ 조향  Space 정지  m 토글  q 종료")
plt.show()
listener.stop()
