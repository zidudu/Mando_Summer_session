#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pure_pursuit_with_heading_and_steer.py
──────────────────────────────────────────────────────────────
• Waypoints / Vehicle / Look-ahead 원
• 헤딩(파란)·조향각(빨간) 화살표 시각화
──────────────────────────────────────────────────────────────
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pynput import keyboard

# ── 사용자 파라미터 ─────────────────────────────────────────
MODE        = "auto"              # 'manual' or 'auto'
CSV_WPT     = "waypoints_dynamic_19.csv"
WHEELBASE   = 0.28                # [m]
DT          = 0.05                # [s]
MAX_STEER   = math.radians(30)    # ±30°
MAX_SPEED   = 3.0                 # [m/s]
K_LD, K_V   = 0.5, 0.4            # look-ahead 계수
MIN_AUTO_V  = 1.0                 # AUTO 진입 시 기본 속도 [m/s]

# ── Waypoints 로드 ──────────────────────────────────────────
wp = np.loadtxt(CSV_WPT, delimiter=",", skiprows=1)
wp_x, wp_y = wp[:, 0], wp[:, 1]

# ── 상태 초기화 ────────────────────────────────────────────
state = {
    "x": wp_x[0],
    "y": wp_y[0],
    "yaw": math.atan2(wp_y[1] - wp_y[0], wp_x[1] - wp_x[0]),  # 초기 헤딩 = 첫 구간 방향
    "v":  0.0,
    "delta": 0.0,
}
mode_auto = (MODE.lower() == "auto")
if mode_auto:
    state["v"] = MIN_AUTO_V
target_idx = 0   # 웨이포인트 인덱스

# ── Pure-Pursuit 보조 함수 ──────────────────────────────────
def find_target(ix_prev, Ld, max_jump=1):
    """Ld 원 바깥 첫 웨이포인트 반환, 인덱스 점프폭 제한"""
    global target_idx
    for i in range(ix_prev, len(wp_x)):
        dx, dy = wp_x[i] - state["x"], wp_y[i] - state["y"]
        if dx*dx + dy*dy >= Ld*Ld:
            target_idx = min(i, ix_prev + max_jump)
            break
    dx = wp_x[target_idx] - state["x"]
    dy = wp_y[target_idx] - state["y"]
    return dx, dy

def pure_pursuit_step():
    Ld = K_LD + K_V * state["v"]
    dx, dy = find_target(target_idx, Ld)
    alpha  = math.atan2(dy, dx) - state["yaw"]
    alpha  = math.atan2(math.sin(alpha), math.cos(alpha))      # wrap-to-π
    delta  = math.atan2(2 * WHEELBASE * math.sin(alpha), Ld)
    return max(-MAX_STEER, min(MAX_STEER, delta))

def update_vehicle():
    beta = math.atan2(math.tan(state["delta"]), 1.0)  # 근사
    state["x"]   += state["v"] * math.cos(state["yaw"] + beta) * DT
    state["y"]   += state["v"] * math.sin(state["yaw"] + beta) * DT
    state["yaw"] += state["v"] / WHEELBASE * math.tan(state["delta"]) * DT

# ── 키보드 콜백 ────────────────────────────────────────────
def on_press(key):
    global mode_auto
    try:
        c = key.char.lower()
        if c == "q":
            return False
        if c == "m":
            mode_auto = not mode_auto
            print(f"[MODE] {'AUTO' if mode_auto else 'MANUAL'}")
            if mode_auto:
                state["v"] = max(state["v"], MIN_AUTO_V)
    except AttributeError:
        pass

    if not mode_auto:                      # 수동 조종
        if key == keyboard.Key.up:       state["v"] = min(MAX_SPEED, state["v"] + 0.2)
        elif key == keyboard.Key.down:   state["v"] = max(0.0,       state["v"] - 0.2)
        elif key == keyboard.Key.left:   state["delta"] = max(-MAX_STEER, state["delta"] - math.radians(2))
        elif key == keyboard.Key.right:  state["delta"] = min(MAX_STEER,  state["delta"] + math.radians(2))
        elif key == keyboard.Key.space:  state["v"] = 0.0

# ── 그래프 세팅 ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(wp_x, wp_y, "k.", ms=4, label="Waypoints")
for idx, (x, y) in enumerate(zip(wp_x, wp_y)):
    ax.text(x, y, str(idx), fontsize=7, ha="right", va="bottom")

car_pt, = ax.plot([], [], "bo", ms=6, label="Vehicle")
look_pt, = ax.plot([], [], "rs", ms=5, label="Look-ahead")
circle = plt.Circle((0, 0), 0, ec="green", fill=False, lw=1.5)
ax.add_patch(circle)

# 헤딩(파란), 조향(빨간) 화살표 — 데이터 단위 그대로 표시
heading_quiv = ax.quiver([], [], [], [],
                         color="blue", angles="xy",
                         scale_units="xy", scale=1,
                         width=0.007, label="Heading")
steer_quiv   = ax.quiver([], [], [], [],
                         color="red",  angles="xy",
                         scale_units="xy", scale=1,
                         width=0.007, label="Steering")

ax.set_xlim(wp_x.min() - 5, wp_x.max() + 5)
ax.set_ylim(wp_y.min() - 5, wp_y.max() + 5)
ax.set_aspect("equal"); ax.grid(True); ax.legend(loc="upper right")

# ── 애니메이션 루프 ────────────────────────────────────────
def animate(_):
    if mode_auto:
        state["delta"] = pure_pursuit_step()
    update_vehicle()

    # 차량·Look-ahead·원
    car_pt.set_data([state["x"]], [state["y"]])
    if mode_auto:
        Ld = K_LD + K_V * state["v"]
        dx, dy = find_target(target_idx, Ld)
        look_pt.set_data([state["x"] + dx], [state["y"] + dy])
        look_pt.set_visible(True)
        circle.center, circle.radius = (state["x"], state["y"]), Ld
    else:
        look_pt.set_visible(False)
        circle.radius = 0

    # 헤딩 화살표
    ux, uy = math.cos(state["yaw"]), math.sin(state["yaw"])
    heading_quiv.set_offsets([[state["x"], state["y"]]])
    heading_quiv.set_UVC([ux], [uy])

    # 조향 화살표 (전륜 위치 기준)
    fx = state["x"] + math.cos(state["yaw"]) * WHEELBASE
    fy = state["y"] + math.sin(state["yaw"]) * WHEELBASE
    sx = math.cos(state["yaw"] + state["delta"])
    sy = math.sin(state["yaw"] + state["delta"])
    steer_quiv.set_offsets([[fx, fy]])
    steer_quiv.set_UVC([sx], [sy])

    return car_pt, look_pt, circle, heading_quiv, steer_quiv

ani = animation.FuncAnimation(fig, animate, interval=DT * 1000, blit=False)

# ── 키보드 리스너 & 실행 ───────────────────────────────────
listener = keyboard.Listener(on_press=on_press)
listener.start()
print("시작 모드:", "AUTO" if mode_auto else "MANUAL",
      "–  ↑↓: 속도  ←→: 조향  Space: 정지  m: 토글  q: 종료")
plt.show()
listener.stop()
