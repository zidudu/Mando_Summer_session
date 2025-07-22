#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pure_pursuit_live_gps_viewer.py
──────────────────────────────────────────────────────────────
• 실시간 RTK-GPS(GNGGA) → Web Mercator(X,Y) ↴ Vehicle 위치 갱신
• Waypoints / Look-ahead 원 / Target(빨강) 시각화
• 헤딩(파란)·조향각(빨강) 화살표는 ‘시각화용’ 계산만 수행
──────────────────────────────────────────────────────────────
[조작키]
   q : 종료
──────────────────────────────────────────────────────────────
"""
import math, time, serial, os
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard

# ───────── 사용자 설정 ──────────────────────────────────────
PORT, BAUD, TIMEOUT = "COM10", 115200, 0.1   # GNSS 포트/보레이트
CSV_WPT             = "waypoints_dynamic_19.csv"
WHEELBASE           = 0.28          # [m] (시각화용) # 차량의 축간 거리,차량 앞바퀴와 뒷바퀴 사이의 거리
K_LD, K_V           = 0.5, 0.4      # Ld = K_LD + K_V·v 
                                    # K_LD: Look-ahead 거리 계산 계수, 차량이 추적할 목표 지점까지의 최소 거리를 계산,Look-ahead 거리는 차량이 목표를 추적할 때 차량과 목표 웨이포인트 사이의 거리입니다.
                                    # K_V: 차량의 속도에 따라 Look-ahead 거리 조정을 위한 계수, 차량이 빠를수록 Look-ahead 거리가 더 멀어지도록 조정하는 역할
MIN_LD              = 0.8           # 속도 0일 때 최소 반경, 속도가 0일 때도 너무 짧은 Look-ahead 거리가 설정되지 않도록 하는 제한값
# ────────────────────────────────────────────────────────────

# ── 보조 함수 ───────────────────────────────────────────────
def nmea2deg(val: str) -> float:
    """NMEA ddmm.mmmm → 십진도"""
    try:
        v = float(val)
        d = int(v // 100)
        m = v - 100*d
        return d + m/60.0
    except:          # 빈 문자열 등
        return float('nan')

def mercator_xy(lat: float, lon: float):
    """WGS84 위/경도 → Web Mercator X,Y [m]"""
    R = 6_378_137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.radians(90 + lat) / 2))
    return x, y

# ── Waypoints 로드 ─────────────────────────────────────────
wp = np.loadtxt(CSV_WPT, delimiter=",", skiprows=1)
wp_x, wp_y = wp[:, 0], wp[:, 1]

# ── 실시간 상태 변수 ───────────────────────────────────────
state = {           # state: 차량의 현재 상태를 추적하는 데 사용됨
    "x"      : wp_x[0],       # 첫 샘플 수신 전 임시값, x: 차량의 현재 X 좌표
    "y"      : wp_y[0],       # y: 차량의 현재 Y 좌표
    "yaw"    : 0.0,           # yaw: 차량의 **헤딩(방향)**을 나타내는 값입니다. 처음에는 0.0으로 초기화되며, 차량이 움직일 때마다 변경됩니다. 
                                    # yaw는 차량의 진행 방향을 나타내며, 라디안 단위로 설정
    "v"      : 0.0,           # m/s (샘플 간 Δs/Δt) # v: 차량의 속도입니다. 차량의 이동 속도를 나타내며, Δs/Δt로 계산
    "delta"  : 0.0            # 조향각 (시각화용)    # 이 값은 차량의 조향을 나타내며, 차량의 회전 방향을 제어
}
prev_time = None                # 이전 시각(시간)을 저장하는 변수
prev_x    = None                # 이전 X 좌표를 저장하는 변수
target_idx = 0                # Look-ahead 타깃 WP 인덱스, Look-ahead 목표 웨이포인트의 인덱스입니다. Pure-Pursuit 알고리즘에서 차량이 추적할 다음 웨이포인트의 인덱스를 나타냅니다
prev_y    = None                # 이전 Y 좌표를 저장하는 변수
# ── Look-ahead 타깃 찾기 ───────────────────────────────────
# 현재 차량 위치와 각 웨이포인트 간의 거리를 계산하여, Ld 거리 이상 떨어져 있는 첫 번째 웨이포인트를 찾습니다.
def find_target(ix_prev, Ld, max_jump=1): # 이전에 목표로 설정된 웨이포인트의 인덱스, Look-ahead 거리, 인덱스 점프의 최대 범위
    """Ld 원 바깥 첫 웨이포인트 인덱스 결정"""
    global target_idx # target_idx는 현재 목표 웨이포인트의 인덱스
    for i in range(ix_prev, len(wp_x)):
        dx = wp_x[i] - state["x"] # wp_x[i]와 wp_y[i]는 웨이포인트 파일에서 읽어들인 웨이포인트의 x, y 좌표
        dy = wp_y[i] - state["y"] # state["x"], **state["y"]**는 현재 차량의 위치
        if dx*dx + dy*dy >= Ld*Ld:        # 원 밖, # 차량의 현재 위치와 웨이포인트 i와의 제곱된 거리입니다. 이를 통해 유클리드 거리를 구하려는 것
                                          # *Ld*Ld**는 Look-ahead 거리의 제곱입니다. 이 값과 웨이포인트와의 거리 제곱을 비교하여, 차량이 추적할 목표 웨이포인트가 Ld보다 멀리 떨어져 있는지 확인
            target_idx = min(i, ix_prev + max_jump) # ix_prev부터 시작해서 Ld 거리를 초과하는 웨이포인트를 찾은 뒤, 이 웨이포인트의 인덱스를 **target_idx**에 저장
                                                    # max_jump: 웨이포인트 간 인덱스를 한 번에 얼마나 많이 건너뛰는지를 설정하는 값입니다. 
                                                    # 기본값은 1로 설정되어 있으며, 이는 웨이포인트 인덱스를 1단계씩만 진행하도록 제한합니다. 즉, 현재 target_idx를 기준으로 한 단계씩만 웨이포인트를 추적합니다. 
                                                    # max_jump를 키워서 더 많은 웨이포인트를 한 번에 건너뛰게 설정할 수 있습니다.    
                                                    # => max_jump를 초과해서 너무 멀리 있는 웨이포인트를 선택하지 않도록 인덱스를 제한
            break
    # 목표 웨이포인트와 차량의 현재 위치 간의 X, Y 좌표 차이를 계산
    dx = wp_x[target_idx] - state["x"]
    dy = wp_y[target_idx] - state["y"]
    return dx, dy
# Pure-Pursuit 알고리즘에서 목표 웨이포인트를 향해 차량이 **회전해야 할 조향각(steering angle, δ)**을 계산
def compute_pp_delta(Ld, dx, dy):
    """Pure-Pursuit 조향각(시각화용)"""
    # alpha: 목표 웨이포인트가 있는 방향과 차량의 현재 헤딩(yaw) 간의 각도 차이를 계산
    alpha = math.atan2(dy, dx) - state["yaw"] # 차량과 목표 웨이포인트 간의 각도를 계산하고, 그 값에서 차량의 헤딩 state["yaw"]를 빼서 각도 차이를 구합니다.
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # wrap, atan2는 각도를 -π에서 π 범위로 제한하므로, 이 값을 wrap하여 -π와 π 사이로 맞춰줍니다
    # delta: 차량이 목표 웨이포인트를 향해 회전해야 하는 조향각을 계산합니다. atan2(2 * WHEELBASE * sin(alpha), Ld)는 조향각을 계산하는 공식
    return math.atan2(2 * WHEELBASE * math.sin(alpha), Ld) # 2 * WHEELBASE * sin(alpha)는 차량이 목표를 추적하기 위해 얼마나 회전해야 하는지를 계산하며, 이 값이 **Look-ahead 거리(Ld)**에 비례
                                                           # 차량의 축간 거리(Wheelbase)에 기반하여 차량이 얼마나 회전해야 할지를 계산


# ── 그래프 초기 설정 ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7)) # figsize=(7, 7)는 그래프의 크기를 설정
ax.plot(wp_x, wp_y, "k.", ms=4, label="Waypoints")
for idx, (x, y) in enumerate(zip(wp_x, wp_y)): # wp_x와 wp_y의 각 요소를 x, y로 짝지어 enumerate() 함수로 순서와 값을 동시에 추출
    ax.text(x, y, str(idx), fontsize=7, ha="right", va="bottom") # => 각 웨이포인트의 인덱스 번호를 해당 웨이포인트 위에 표시
# 차량 위치
car_pt,  = ax.plot([], [], "bo", ms=6, label="Vehicle")
# Look-ahead 
look_pt, = ax.plot([], [], "rs", ms=5, label="Look-ahead")
# Look-ahead 원
circle = plt.Circle((0, 0), 0, ec="green", fill=False, lw=1.5)
# 이 원을 ax(Axes)에 추가
ax.add_patch(circle)
# ax.quiver([], [], [], [], ...): 빈 리스트를 넘기고, 실시간으로 차량의 헤딩 방향을 나타내는 화살표를 그리기 위한 설정
heading_qv = ax.quiver([], [], [], [],
                       color="blue", angles="xy",
                       scale_units="xy", scale=1,
                       width=0.007, label="Heading")
steer_qv   = ax.quiver([], [], [], [],
                       color="red",  angles="xy",
                       scale_units="xy", scale=1,
                       width=0.007, label="Steering")
# X축 범위를 웨이포인트 X 값의 최소값과 최대값에 5를 더하고 빼서 설정합니다. 
# 이는 그래프에 약간의 여유 공간을 추가하여 차량이 움직이는 영역 잘 볼수있게 함
ax.set_xlim(wp_x.min() - 5, wp_x.max() + 5) 
ax.set_ylim(wp_y.min() - 5, wp_y.max() + 5)
# X축과 Y축의 비율을 동일하게 설정하여, 정사각형 비율로 그래프가 그려지도록 합니다
# 그리드 표시를 활성화
# 범례를 우측 상단에 위치
ax.set_aspect("equal"); ax.grid(True); ax.legend(loc="upper right")

# 인터랙티브 모드 활성화
plt.ion()
# 그래프 화면에 표시
plt.show()

# ── 키 입력 (종료만) ───────────────────────────────────────
def on_press(key):
    if key == keyboard.Key.esc or (hasattr(key, "char") and key.char == 'q'):
        return False

listener = keyboard.Listener(on_press=on_press)
listener.start()

# ── 시리얼 포트 열기 ───────────────────────────────────────
def open_serial():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        print(f"[INFO] Serial opened → {PORT}")
        return ser
    except Exception as e:
        print("[ERROR] Serial open failed:", e)
        return None

ser = open_serial()
print("GNGGA 수신 대기…  (q 또는 Esc 로 종료)")

# ── 메인 루프 ──────────────────────────────────────────────
try:
    while listener.is_alive():
        # 시리얼 연결 안 됐으면 재시도
        if ser is None or not ser.is_open:
            time.sleep(1)
            ser = open_serial()
            continue

        raw = ser.readline().decode(errors="ignore").strip()
        # gngga 만 받음
        if not raw.startswith("$GNGGA"):
            continue
        parts = raw.split(",") # , 구분해 배열로 각각 넣음
        if len(parts) < 7 or parts[2] == "" or parts[4] == "": # 위도 경도 없을때 
            continue  # 위·경도 정보 없으면 skip

        # ---------------- GPS 파싱 ----------------
        lat = nmea2deg(parts[2]); lon = nmea2deg(parts[4]) # 위도 경도 값만 빼옴
        if parts[3] == "S": lat = -lat
        if parts[5] == "W": lon = -lon
        x, y = mercator_xy(lat, lon) # x y 값으로 변환

        #---------------- 속도·헤딩 계산 ----------------
        # GPS에서 받은 실시간 좌표(x, y)를 기반으로 차량의 이동을 추적하고 조향각(state["delta"])을 계산하는 작업을 수행
        now = time.time() # time.time(): 현재 시간을 초 단위로 반환
        if prev_time is not None:
            dt = now - prev_time # prev_time은 이전 위치를 기준으로 이전 시간을 기록한 값, # dt는 현재 시간과 이전 시간 간의 차이로, 이동에 소요된 시간을 계산
            dist = math.hypot(x - prev_x, y - prev_y) # a와 b의 유클리드 거리(피타고라스의 정리) 계산을 수행, **x - prev_x**는 X 좌표의 변화량, **y - prev_y**는 Y 좌표의 변화량
            
            # ------------$$$$ 수정된 코드 (곡선에 대한 거리 보정)$$$$-------------------------#
            # 예시로 단순히 곡률을 고려한 보정을 추가할 수 있음
            # curvature = compute_curvature(x, y, prev_x, prev_y)
            # dist = dist / (1 + curvature)  # 곡률에 따라 거리 보정
            # -------------------------------------------------------------------------------------#


            # (v 자동 계산)                                          # dist**는 차량이 이전 위치에서 현재 위치까지 이동한 거리
            state["v"] = dist / dt if dt > 0 else 0.0  # 속도 **v**는 이동한 거리(dist)를 이동 시간(dt)으로 나누어 계산
                                                       # *dt > 0**이면 유효한 이동 시간이므로 속도를 계산하고, **dt == 0**이면 이동하지 않았거나 시간 차이가 너무 작아서 속도를 0으로 설정
            
            #-------------------------$$수정된 코드 (수동 입력)$$-------------------------# 
            # def on_press(key):
            #     if key == keyboard.Key.up:
            #         state["v"] += 0.1  # 속도 증가
            #     elif key == keyboard.Key.down:
            #         state["v"] -= 0.1  # 속도 감소
            #     elif key == keyboard.Key.left:
            #         state["yaw"] -= 0.1  # 헤딩 왼쪽 회전
            #     elif key == keyboard.Key.right:
            #         state["yaw"] += 0.1  # 헤딩 오른쪽 회전
            #---------------------------------------------------------------------#

            if dist > 1e-3: # dist > 1e-3: 이동한 거리가 매우 작을 경우, 헤딩 계산을 생략할 수 있습니다. 1e-3은 0.001로, 너무 작은 이동을 무시하기 위한 기준값
                state["yaw"] = math.atan2(y - prev_y, x - prev_x)        # Look-ahead 반경, **math.atan2(dy, dx)**는 차량이 이동한 방향을 계산합니다. dy와 dx는 각각 Y와 X 방향의 거리 차이
                                                                         # 라디안 단위로 반환되며, **state["yaw"]**는 차량의 방향
        Ld = max(MIN_LD, K_LD + K_V * state["v"]) # MIN_LD: 차량이 정지할 때 최소 Look-ahead 거리를 설정, 이 값은 속도 0일 때 최소 거리로 설정되며, 차량이 매우 천천히 움직일 때도 최소 거리 이상을 유지
                                                  # K_LD: 기본 Look-ahead 거리 계수
                                                  # K_V: 차량 속도에 따라 Look-ahead 거리를 조정하는 계수
                                                  # state["v"]**는 속도입니다. 따라서 속도가 빠를수록 Look-ahead 거리가 커지게 됩니다.
        # -> Ld 값은 차량의 속도에 비례해서 Look-ahead 반경을 설정하며, 이 반경 내에서 차량이 목표 웨이포인트를 추적합니다.
     
        # -----------$$수정된 코드 (속도와 관계없이 일정한 Look-ahead 거리 설정)$$-----------------------------------------------#
        # ---(만약 거리 일정하게 하고 싶으면 이거 쓰기. 동적으로 바꾸고 싶으면 위에꺼)--- #
        # Ld = MIN_LD  # 또는 원하는 고정 값으로 설정
        #--------------------------------------------------------------------------------------------------------------------#

       
        
        # 목표 웨이포인트 찾기
        dx, dy = find_target(target_idx, Ld) 
        #  조향각 계산
        state["delta"] = compute_pp_delta(Ld, dx, dy)
        # ← 여기에 prev_y를 함께 저장
        prev_time, prev_x, prev_y = now, x, y
        
        # ---------------- 시각화 ------------------
        # Vehicle
        car_pt.set_data([state["x"]], [state["y"]])

        # Look-ahead 원 / 타깃
            # Look-ahead 원 (초록색 원)
        circle.center, circle.radius = (state["x"], state["y"]), Ld
            # 목표 웨이포인트 (빨간 사각형)
        look_pt.set_data([state["x"] + dx], [state["y"] + dy])

        # --------------------------------------$$ 수정된 코드 (목표 웨이포인트를 원으로 표시)$$--------------------------------------#
        # circle_target = plt.Circle((state["x"] + dx, state["y"] + dy), Ld, color="red", fill=False, lw=2)
        # ax.add_patch(circle_target)
        # ------------------------------------------------------------------------------------------------------------------#


        # 헤딩 화살표(파란색)
        ux, uy = math.cos(state["yaw"]), math.sin(state["yaw"]) # 차량이 향하는 방향을 나타내는 벡터를 계산해 ux,uy에 넣음,ux는 차량이 향하는 X축 방향의 단위 벡터, uy는 Y축 방향의 단위 벡터
        heading_qv.set_offsets([[state["x"], state["y"]]]) # 헤딩 화살표의 시작점을 차량의 현재 위치(state["x"], state["y"])로 설정
        heading_qv.set_UVC([ux], [uy]) # ux**와 **uy**를 이용하여 헤딩 화살표의 방향을 설정합니다. 헤딩 화살표는 차량이 현재 향하는 방향

        # 조향 화살표 (전륜 위치 기준, 시각화용)
        fx = state["x"] + math.cos(state["yaw"]) * WHEELBASE # fx, fy: 조향 화살표의 기준점을 계산합니다. 이는 차량의 전륜 위치,차량이 향하는 방향(state["yaw"])에 **차량의 축간 거리(WHEELBASE)**를 더한 좌표
        fy = state["y"] + math.sin(state["yaw"]) * WHEELBASE
        sx = math.cos(state["yaw"] + state["delta"]) # sx, sy: 조향각(state["delta"])에 따라 조향 화살표의 방향 계산
        sy = math.sin(state["yaw"] + state["delta"])
        steer_qv.set_offsets([[fx, fy]]) # 조향 화살표의 시작점을 전륜 위치(fx, fy)로 설정
        steer_qv.set_UVC([sx], [sy]) # 조향 화살표의 방향을 설정

        # 화면 업데이트
        plt.pause(0.01) # plt.pause(0.01): 0.01초 동안 일시 정지하여 화면을 갱신 # 이 명령은 실시간으로 그래프를 갱신하기 위해 필요

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

finally:
    if ser and ser.is_open:
        ser.close()
    listener.stop()
    print("[INFO] 종료되었습니다.")
