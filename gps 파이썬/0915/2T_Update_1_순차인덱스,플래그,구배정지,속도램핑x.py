#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 + matplotlib · GPS waypoint 추종 & 조향각/속도/구배 퍼블리시 (플래그 통합판)
- 입력: (param) ~fix_topic  (sensor_msgs/NavSatFix, default=/gps1/fix)
- 출력(통신 규격 고정):
  * /gps/steer_cmd   (std_msgs/Float32)  → LPF + "부호 반전 유지" 적용 조향 각도[deg]
  * /gps/speed_cmd   (std_msgs/Float32)  → 속도 코드(목표 코드가 9 이상일 때만 램핑)
  * /gps/status      (std_msgs/String)   → RTK 상태 (간단화: "NONE")
  * /gps/wp_index    (std_msgs/Int32)    → '원 반경' 안에 들어간 WP 인덱스(1-based), 없으면 0
  * /gps/GRADEUP_ON  (std_msgs/Int32, latch) → 구배 구간 상태(평소 0, 구배 1)  # %%추가%%
- 파라미터(일부):
  * ~csv_path     (str)   : 웨이포인트 CSV 경로 (기본: ~/다운로드/left_final.csv)
  * ~fix_topic    (str)   : NavSatFix 토픽      (기본: /gps1/fix)
  * ~enable_plot  (bool)  : matplotlib 표시     (기본: true)
  * ~dt           (float) : LPF 샘플 주기[s]    (기본: 0.05)
  * ~tau          (float) : 1차 LPF 시간상수[s] (기본: 0.25)
  * ~fs           (float) : 제어 루프 주기[Hz]  (기본: 20)                          # %%추가%%
  * ~base_speed_code (int): 기본 속도 코드      (기본: 5)                           # %%추가%%
  * ~step_per_loop    (int): 전역 램핑 스텝     (기본: 2)                           # %%추가%%
  * ~ramp_threshold_code (int): 램핑 임계(코드)  (기본: 9; 미만이면 즉시 스냅)        # %%추가%%
"""

import os, math, time, csv
import rospy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, Int32, String

# ====== 상수 ======
EARTH_RADIUS   = 6378137.0     # [m]
MIN_DISTANCE   = 2.8           # [m] 웨이포인트 간 최소 간격
TARGET_RADIUS  = 2.8           # [m] 목표 웨이포인트 도달 반경
ANGLE_LIMIT_DEG = 30.0         # [deg] 조향 제한

# %%추가%% 퍼블리시 토픽(고정 경로)
TOPIC_SPEED_CMD     = '/gps/speed_cmd'     # Float32
TOPIC_STEER_CMD     = '/gps/steer_cmd'     # Float32 (deg)
TOPIC_RTK_STATUS    = '/gps/status'        # String ("FIXED"/"FLOAT"/"NONE") - 여기선 간단화
TOPIC_WP_INDEX      = '/gps/wp_index'      # Int32 (1-based, 반경 밖=0)
TOPIC_WP_GRADEUP_ON = '/gps/GRADEUP_ON'    # Int32 (latch) ← 문자열 절대 변경 금지

# ====== 전역 상태 ======
filtered_steering_angle = 0.0
current_x, current_y = [], []
current_waypoint_index = 0        # 도달 반경 내 진입 시 +1 (순차 인덱스)
reduced_waypoints = None          # shape: (2, N)

# LPF 파라미터(파라미터 서버에서 로드)
DT   = 0.05   # [s]
TAU  = 0.25   # [s]
ALPHA = None  # dt/(tau+dt)

# %%추가%% 속도/플래그 전역
base_speed_code_param = 5
GLOBAL_STEP_PER_LOOP  = 2
RAMP_THRESHOLD_CODE   = 9

speed_cmd_current_code = 0        # 현재 퍼블 중인 속도 코드
speed_desired_code     = 0        # 이번 루프 목표 속도 코드

# %%추가%% 플래그/홀드 상태
flag_zones = []                   # build_flag_zones(FLAG_DEFS) 결과
hold_active = False
hold_until  = 0.0
hold_reason = None
last_hold_zone_name = None        # 같은 구간에서 1회만 정지

# 퍼블리셔 (init_node 이후에 실제 인스턴스 생성)
steering_pub = None
waypoint_index_pub = None
# %%추가%%
speed_pub = None
rtk_pub   = None
grade_pub = None  # latch

# %%추가%% FLAG 정의(예시: 실제 코스 인덱스에 맞게 조정하세요)
# ※ 요청대로 'step_per_loop'는 플래그에서 제거했습니다(전역만 사용).
FLAG_DEFS = [
    {'name': 'GRADE_START', 'start': 4, 'end': 5,
     'radius_scale': 1.0, 'lookahead_scale': 0.95,
     'speed_code': 5, 'speed_cap': 7,
     'stop_on_hit': False, 'stop_duration_sec': None,
     'grade_topic': 1},

    # 언덕(GRADE) — 해당 원에 처음 들어오면 1회 3초 정지, 구간에 있는 동안 grade_topic=1 유지
    {'name': 'GRADE_UP', 'start': 6, 'end': 6,
     'radius_scale': 1.0, 'lookahead_scale': 0.95,
     'speed_code': None, 'speed_cap': None,
     'stop_on_hit': True, 'stop_duration_sec': 3,
     'grade_topic': 1},

    {'name': 'GRADE_GO', 'start': 7, 'end': 9,
     'radius_scale': 1.0, 'lookahead_scale': 0.95,
     'speed_code': 5, 'speed_cap': 7,
     'stop_on_hit': False, 'stop_duration_sec': None,
     'grade_topic': 1},

    {'name': 'GRADE_END', 'start': 10, 'end': 11,
     'radius_scale': 1.0, 'lookahead_scale': 0.95,
     'speed_code': 5, 'speed_cap': 7,
     'stop_on_hit': False, 'stop_duration_sec': None,
     'grade_topic': 0},
]

# %%추가%%
STOP_FLAG_STAY_SEC = 3.0  # 기본 정지 시간(초) - 개별 구간에서 override 가능

# ====== 유틸 ======
def lat_lon_to_meters(lat, lon):
    """(deg, deg) -> (m, m) Web Mercator 근사 (소구역에서 오차 작음)"""
    x = EARTH_RADIUS * lon * math.pi / 180.0
    y = EARTH_RADIUS * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def distance_in_meters(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def _pick_column(df_cols, candidates):
    """df 컬럼 목록(df_cols)에서 candidates 중 첫 번째로 일치하는 이름을 반환 (소문자 매칭). 없으면 None."""
    for c in candidates:
        if c in df_cols:
            return c
    return None

def build_reduced_waypoints(csv_path):
    """
    CSV 컬럼 자동 인식:
      - 위/경도(도): (latitude|lat), (longitude|lon)
      - 동/북   (m): (east|easting|x), (north|northing|y)
    어떤 모드를 썼는지 INFO 로깅.
    """
    csv_path = os.path.expanduser(csv_path)
    df = pd.read_csv(csv_path, encoding='utf-8-sig', engine='python', sep=None)
    df.columns = [str(c).strip().lower() for c in df.columns]

    cols = set(df.columns)
    lat_cands  = ['latitude', 'lat']
    lon_cands  = ['longitude', 'lon']
    east_cands = ['east', 'easting', 'x']
    north_cands= ['north', 'northing', 'y']

    lat_col  = _pick_column(cols, lat_cands)
    lon_col  = _pick_column(cols, lon_cands)
    east_col = _pick_column(cols, east_cands)
    north_col= _pick_column(cols, north_cands)

    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors='coerce').to_numpy()
        lon = pd.to_numeric(df[lon_col], errors='coerce').to_numpy()
        pts = [lat_lon_to_meters(a, b) for a, b in zip(lat, lon)]
        wx, wy = zip(*pts)
        rospy.loginfo("Waypoints from lat/lon columns: (%s, %s)", lat_col, lon_col)
    elif east_col and north_col:
        wx = pd.to_numeric(df[east_col],  errors='coerce').to_numpy()
        wy = pd.to_numeric(df[north_col], errors='coerce').to_numpy()
        rospy.loginfo("Waypoints from east/north columns: (%s, %s)", east_col, north_col)
    else:
        available = ", ".join(sorted(cols))
        raise ValueError(
            "CSV에서 사용할 좌표 컬럼을 찾지 못했습니다.\n"
            f"- 허용되는 조합 1) lat/lon  2) east/north (또는 x/y)\n"
            f"- 현재 컬럼: {available}"
        )

    # 간격 축소(>= MIN_DISTANCE)
    reduced_x = [float(wx[0])]
    reduced_y = [float(wy[0])]
    for x, y in zip(wx[1:], wy[1:]):
        if distance_in_meters(reduced_x[-1], reduced_y[-1], float(x), float(y)) >= MIN_DISTANCE:
            reduced_x.append(float(x))
            reduced_y.append(float(y))

    return np.array([reduced_x, reduced_y])  # shape: (2, N)

def signed_angle_deg(v1, v2):
    """두 벡터 사이 부호 있는 각도(도). 반시계 양수, 시계 음수."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = float(np.dot(v1, v2)) / (n1 * n2)
    dot = max(min(dot, 1.0), -1.0)
    ang = math.degrees(math.acos(dot))
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return ang if cross >= 0 else -ang

# === 시간상수 기반 LPF + 부호 반전 유지 ===
def lpf_and_invert(current_angle_deg):
    """
    1차 저역필터:
      y[k] = (1-ALPHA)*y[k-1] + ALPHA*x[k],  where ALPHA = DT/(TAU+DT)
    이후 '부호 반전 유지' 적용해 반환.
    """
    global filtered_steering_angle, ALPHA
    filtered_steering_angle = (1.0 - ALPHA) * filtered_steering_angle + ALPHA * current_angle_deg
    return -filtered_steering_angle  # 부호 반전 유지

# %%추가%% 1-based → 0-based 구간 변환
def build_flag_zones(flag_defs):
    """1-based(start/end) → 0-based로 변환하여 zone dict 리스트 생성"""
    zones = []
    for fd in flag_defs:
        s0 = int(min(fd['start'], fd['end'])) - 1
        e0 = int(max(fd['start'], fd['end'])) - 1
        zones.append({
            'name': fd['name'],
            'start0': s0,
            'end0': e0,
            'radius_scale': float(fd.get('radius_scale', 1.0)),
            'lookahead_scale': float(fd.get('lookahead_scale', 1.0)),
            'speed_code': fd.get('speed_code', None),
            'speed_cap': fd.get('speed_cap', None),
            'stop_on_hit': bool(fd.get('stop_on_hit', False)),
            'stop_duration_sec': float(fd.get('stop_duration_sec', STOP_FLAG_STAY_SEC) if fd.get('stop_on_hit', False) else 0.0),
            'grade_topic': int(fd.get('grade_topic', 0)),
            'disp_range': f"{s0+1}~{e0+1}",
        })
    return zones

# %%추가%% 최근접 WP + 반경 내 표시(1-based 인덱스 반환)
def nearest_wp_and_display(x, y, base_radius, radius_scale=1.0):
    """현재 위치(x,y)에 대해 최근접 WP 인덱스(0-based)와 반경 내면 1-based index, 아니면 0 반환"""
    global reduced_waypoints
    wx, wy = reduced_waypoints
    dx = wx - x
    dy = wy - y
    d2 = dx*dx + dy*dy
    idx = int(np.argmin(d2))
    dist = math.sqrt(float(d2[idx]))
    r_eff = base_radius * float(radius_scale)
    wp_display = (idx + 1) if (dist < r_eff) else 0
    return idx, wp_display, dist, r_eff

# %%추가%% 램핑/스냅 공용 함수 (임계 미만은 무조건 스냅)
def ramp_or_snap_speed(desired_code, current_code, step_per_loop, ramp_threshold):
    """
    desired_code가 ramp_threshold 이상이면 램핑, 미만이면 즉시(current=desired).
    반환: (new_current_code, did_ramp:bool)
    """
    desired_code = int(desired_code)
    current_code = int(current_code)
    step = max(1, int(step_per_loop))
    # 임계 미만 → 즉시 스냅
    if desired_code < int(ramp_threshold):
        return desired_code, False
    # 램핑
    if current_code < desired_code:
        new_code = min(current_code + step, desired_code)
        return new_code, (new_code != desired_code)
    elif current_code > desired_code:
        new_code = max(current_code - step, desired_code)
        return new_code, (new_code != desired_code)
    else:
        return current_code, False

# ====== 콜백 ======
def gps_callback(msg: NavSatFix):
    """위치 갱신 + 조향각 계산/퍼블리시(벡터각 + LPF + 부호반전)"""
    global current_waypoint_index

    # 현재 위치(m) 누적
    x, y = lat_lon_to_meters(msg.latitude, msg.longitude)
    current_x.append(x); current_y.append(y)

    # 타깃 웨이포인트(순차 인덱스)
    tx = reduced_waypoints[0][current_waypoint_index]
    ty = reduced_waypoints[1][current_waypoint_index]

    # 도달 판정(반경 내 → 다음 인덱스)
    if distance_in_meters(x, y, tx, ty) < TARGET_RADIUS:
        if current_waypoint_index < reduced_waypoints.shape[1] - 1:
            current_waypoint_index += 1
            if waypoint_index_pub:
                waypoint_index_pub.publish(Int32(current_waypoint_index))

    # 조향각 계산 & 퍼블리시 (직전 위치가 있어야 진행)
    if len(current_x) >= 2:
        prev = np.array([current_x[-2], current_y[-2]])
        curr = np.array([current_x[-1], current_y[-1]])
        head_vec = curr - prev
        tgt_vec  = np.array([tx, ty]) - curr

        angle_deg = signed_angle_deg(head_vec, tgt_vec)
        angle_deg = max(min(angle_deg, ANGLE_LIMIT_DEG), -ANGLE_LIMIT_DEG)  # 제한
        smooth_inv = lpf_and_invert(angle_deg)  # LPF + 부호 반전 유지

        if steering_pub:
            steering_pub.publish(Float32(smooth_inv))

# %%추가%% 플래그/홀드/속도/구배 토픽 처리용 주기 루프
def control_loop(_evt):
    """주기적으로 플래그/홀드/속도 램핑/구배 토픽/인덱스 퍼블리시를 처리"""
    global hold_active, hold_until, hold_reason, last_hold_zone_name
    global speed_cmd_current_code, speed_desired_code

    if reduced_waypoints is None or len(current_x) == 0:
        return  # 위치가 아직 없음

    x, y = current_x[-1], current_y[-1]

    # 현재 구간 탐색: 최근접 WP 기준
    nearest_idx, wp_display, dist, r_eff = nearest_wp_and_display(x, y, TARGET_RADIUS, radius_scale=1.0)
    z_now = None
    for z in flag_zones:
        if z['start0'] <= nearest_idx <= z['end0']:
            z_now = z
            break

    # ── stop_on_hit: '원 밖→안' 진입 엣지에서 1회 정지 트리거 ──────────────
    stop_zone_now = None
    if wp_display > 0:
        for z in flag_zones:
            if z.get('stop_on_hit', False) and (z['start0'] <= nearest_idx <= z['end0']):
                # 반경 스케일 적용한 원 안인지 재확인
                _, wp_disp2, _, _ = nearest_wp_and_display(x, y, TARGET_RADIUS, radius_scale=z.get('radius_scale', 1.0))
                if wp_disp2 > 0:
                    stop_zone_now = z
                    break

    # 새로 진입했고 아직 hold가 아니며, 같은 구간에서 이미 1회 정지한 적 없으면 hold 시작
    if stop_zone_now and (not hold_active):
        if last_hold_zone_name != stop_zone_now['name']:
            hold_active = True
            dur = float(stop_zone_now.get('stop_duration_sec', STOP_FLAG_STAY_SEC))
            hold_until  = time.time() + max(0.0, dur)
            hold_reason = stop_zone_now['name']
            last_hold_zone_name = stop_zone_now['name']
            rospy.loginfo("[HOLD] zone=%s dur=%.1fs", hold_reason, dur)

    # hold 종료 체크
    if hold_active and time.time() >= hold_until:
        hold_active = False
        hold_reason = None
        rospy.loginfo("[HOLD] released")

    # ── 원하는 속도 코드 계산(구간/캡 반영) ────────────────────────────────
    desired = int(base_speed_code_param)
    cap     = None

    if z_now is not None:
        if z_now.get('speed_code') is not None:
            desired = int(z_now['speed_code'])
        if z_now.get('speed_cap') is not None:
            cap = int(z_now['speed_cap'])

    # hold 중이면 무조건 0
    if hold_active:
        desired = 0
        cap = 0

    # cap 적용
    if cap is not None:
        desired = max(0, min(int(cap), int(desired)))

    speed_desired_code = desired

    # 램핑/스냅 적용 (요청: 목표 코드가 임계(기본 9) 이상일 때만 램핑)
    new_code, _ = ramp_or_snap_speed(
        desired_code   = speed_desired_code,
        current_code   = speed_cmd_current_code,
        step_per_loop  = GLOBAL_STEP_PER_LOOP,   # 플래그 개별 step_per_loop는 사용하지 않음
        ramp_threshold = RAMP_THRESHOLD_CODE,
    )
    speed_cmd_current_code = int(new_code)

    # 퍼블리시: 속도/구배/RTK/인덱스 ---------------------------------------
    if speed_pub:
        speed_pub.publish(Float32(float(speed_cmd_current_code)))

    # grade 토픽: 구간에 grade_topic=1이거나 hold 이유가 GRADE_UP면 1 유지
    grade_on = 0
    if z_now is not None and int(z_now.get('grade_topic', 0)) == 1:
        grade_on = 1
    if hold_active and (hold_reason is not None) and ('GRADE_UP' in hold_reason):
        grade_on = 1
    if grade_pub:
        grade_pub.publish(Int32(int(grade_on)))  # latch

    # RTK 상태(간단화: NONE)
    if rtk_pub:
        rtk_pub.publish(String("NONE"))

    # '원 반경 내' 인덱스(1-based) 퍼블리시 (없으면 0)
    if waypoint_index_pub:
        waypoint_index_pub.publish(Int32(int(wp_display)))

# ====== 시각화 ======
def update_plot(_):
    ax = plt.gca()
    ax.clear()

    # 웨이포인트 + 반경
    ax.scatter(reduced_waypoints[0], reduced_waypoints[1], s=10, marker='o', label='Waypoints')
    for i, (wx, wy) in enumerate(zip(reduced_waypoints[0], reduced_waypoints[1]), 1):
        ax.add_patch(plt.Circle((wx, wy), TARGET_RADIUS, fill=False, linestyle='--'))
        ax.text(wx, wy, str(i), fontsize=8, ha='center', va='center')

    # 현재 위치 & 타깃
    if current_x and current_y:
        cx, cy = current_x[-1], current_y[-1]
        ax.scatter(cx, cy, s=50, marker='x', label='Current')

        tx = reduced_waypoints[0][current_waypoint_index]
        ty = reduced_waypoints[1][current_waypoint_index]
        ax.arrow(cx, cy, tx - cx, ty - cy, head_width=0.5, head_length=0.5)

        # 이동 경로(최근 200점만)
        k = max(0, len(current_x) - 200)
        for i in range(k + 1, len(current_x)):
            ax.arrow(current_x[i-1], current_y[i-1],
                     current_x[i]-current_x[i-1], current_y[i]-current_y[i-1],
                     head_width=0.2, head_length=0.2, length_includes_head=True)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True)
    ax.legend(loc='best')
    ax.set_title(f'Filtered steering (inverted) · dt={DT*1000:.0f} ms, tau={TAU:.2f} s; limit ±{ANGLE_LIMIT_DEG:.0f}°')

# ====== 메인 ======
def main():
    global steering_pub, waypoint_index_pub
    global speed_pub, rtk_pub, grade_pub
    global reduced_waypoints, DT, TAU, ALPHA
    global base_speed_code_param, GLOBAL_STEP_PER_LOOP, RAMP_THRESHOLD_CODE
    global log_csv_path  # %%추가%%

    rospy.init_node('gps_waypoint_tracker_flags', anonymous=True)

    # 파라미터
    csv_path    = rospy.get_param('~csv_path',    os.path.expanduser('~/다운로드/left_final.csv'))
    fix_topic   = rospy.get_param('~fix_topic',   '/gps1/fix')
    enable_plot = rospy.get_param('~enable_plot', True)
    DT  = float(rospy.get_param('~dt',  0.05))   # 50ms 기본
    TAU = float(rospy.get_param('~tau', 0.25))   # 0.25s 기본
    fs  = float(rospy.get_param('~fs',  20.0))   # 제어 루프 20Hz
    ALPHA = DT / (TAU + DT)

    # %%추가%% 속도/램핑 파라미터 로드
    base_speed_code_param = int(rospy.get_param('~base_speed_code', 5))
    GLOBAL_STEP_PER_LOOP  = int(rospy.get_param('~step_per_loop', 2))
    RAMP_THRESHOLD_CODE   = int(rospy.get_param('~ramp_threshold_code', 9))

    rospy.loginfo("LPF params: dt=%.3fs, tau=%.3fs, alpha=%.3f", DT, TAU, ALPHA)
    rospy.loginfo("Speed params: base=%d, step=%d, ramp_thr=%d",
                  base_speed_code_param, GLOBAL_STEP_PER_LOOP, RAMP_THRESHOLD_CODE)

    # %%추가%% 고정 로그 경로 설정 (/home/jigu/.../logs/waypoint_log_YYYYmmdd_HHMMSS.csv)
    logs_dir_fixed = '/home/jigu/catkin_ws/src/rtk_waypoint_tracker/logs'  # %%추가%%
    try:
        os.makedirs(logs_dir_fixed, exist_ok=True)  # %%추가%%
    except Exception as e:
        rospy.logwarn(f"[log] make dir failed: {e}")  # %%추가%%
    log_csv_path = os.path.join(
        logs_dir_fixed, f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )  # %%추가%%
    rospy.loginfo(f"[log] CSV path: {log_csv_path}")  # %%추가%%

    # 퍼블리셔 생성
    steering_pub       = rospy.Publisher(TOPIC_STEER_CMD, Float32, queue_size=10)   # %%추가%%
    waypoint_index_pub = rospy.Publisher(TOPIC_WP_INDEX,   Int32,   queue_size=10)  # %%추가%%
    # %%추가%% 추가 퍼블리셔
    speed_pub = rospy.Publisher(TOPIC_SPEED_CMD,   Float32, queue_size=10)
    rtk_pub   = rospy.Publisher(TOPIC_RTK_STATUS,  String,  queue_size=10)
    grade_pub = rospy.Publisher(TOPIC_WP_GRADEUP_ON, Int32, queue_size=1, latch=True)

    # %%추가%% 시작 시 래치 0 청소
    try:
        grade_pub.publish(Int32(0))
    except Exception:
        pass

    # 서브스크라이브
    rospy.Subscriber(fix_topic, NavSatFix, gps_callback)
    rospy.loginfo("Subscribed NavSatFix: %s", fix_topic)

    # 웨이포인트 구성
    global reduced_waypoints
    reduced_waypoints = build_reduced_waypoints(csv_path)
    rospy.loginfo("Waypoints loaded: %d (reduced spacing >= %.1fm)", reduced_waypoints.shape[1], MIN_DISTANCE)

    # %%추가%% 플래그 존 구성
    global flag_zones
    flag_zones = build_flag_zones(FLAG_DEFS)
    if flag_zones:
        rospy.loginfo("[flag] zones loaded: " + ", ".join([f"{z['name']}({z['disp_range']})" for z in flag_zones]))
    else:
        rospy.loginfo("[flag] no zones defined.")

    # %%추가%% 제어 루프 타이머
    rospy.Timer(rospy.Duration(1.0 / max(1.0, fs)), control_loop)

    if enable_plot:
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, update_plot, interval=300)  # 300ms 주기 업데이트
        plt.show()
    else:
        rospy.loginfo("Headless mode: plotting disabled (~enable_plot=false).")
        rospy.spin()

def _on_shutdown():
    """종료 시 안전 정지 + 래치 0으로 청소"""
    rospy.loginfo("[tracker] shutdown: stop & latch zeros")
    try:
        rate = rospy.Rate(30)
        for _ in range(15):
            if speed_pub: speed_pub.publish(Float32(0.0))
            if steering_pub: steering_pub.publish(Float32(0.0))
            if waypoint_index_pub: waypoint_index_pub.publish(Int32(0))
            if grade_pub: grade_pub.publish(Int32(0))  # 래치 0
            if rtk_pub:   rtk_pub.publish(String("NONE"))
            rate.sleep()
    except Exception:
        pass
    # 마지막으로 래치 0 보증
    try:
        if grade_pub: grade_pub.publish(Int32(0))
    except Exception:
        pass

if __name__ == '__main__':
    try:
        rospy.on_shutdown(_on_shutdown)
        main()
    except rospy.ROSInterruptException:
        pass
