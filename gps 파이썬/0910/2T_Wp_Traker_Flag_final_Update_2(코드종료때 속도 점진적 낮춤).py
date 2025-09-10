#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import time
from collections import deque

import rospy
import rospkg
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, String, Int32

import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import geopy.distance
import pandas as pd
import numpy as np
from queue import Queue

# ──────────────────────────────────────────────────────────────────
# 기본 파라미터 (필요시 rosparam으로 덮어쓰기)
# ──────────────────────────────────────────────────────────────────
# CSV 원본 경로를 따라 “보간해서 만든” 주행용 웨이포인트 간격입니다.작게 하면 곡선 추종이 매끈해지지만 계산량↑, 노이즈에 민감해질 수 있습니다.
WAYPOINT_SPACING   = 2.5         # m
# 차량이 “해당 웨이포인트에 도달했다”고 판정하는 반경입니다.너무 작으면 인덱스가 잘 안넘어가고, 너무 크면 급히 다음점으로 넘어가 진동이 생길 수 있습니다.
TARGET_RADIUS_END  = 2.0         # m  (반경 안이면 '그 WP에 있다'로 간주)
MAX_STEER_DEG      = 25.0        # deg
# 스티어링 부호 규약(좌/우 +/−)을 하드웨어/상위 제어기와 맞추기 위한 부호 반전 계수입니다.
SIGN_CONVENTION    = -1.0

# 룩어헤드
#최소 룩어헤드 길이. 아주 느릴 때도 이 정도는 앞을 보도록 강제합니다.
LOOKAHEAD_MIN      = 3.2         # m
#최대 룩어헤드 길이. 너무 멀리 보면 코너에서 늦게 도는 경향이 있어 상한을 둡니다.
LOOKAHEAD_MAX      = 4.0         # m
# 속도에 비례해 룩어헤드를 늘리는 기울기. 내부적으로 Ld_base = clamp(LOOKAHEAD_MIN, LOOKAHEAD_MAX, LOOKAHEAD_MIN + LOOKAHEAD_K * v) 로 쓰입니다.
LOOKAHEAD_K        = 0.2         # m per (m/s)

# 조향각 1차 저역통과필터의 차단 주파수(대역폭).
#값을 낮추면 조향이 더 부드럽지만 반응이 느려지고, 값을 올리면 즉각적이지만 노이즈에 민감합니다.
#일반적으로 0.5~1.5 Hz 사이에서 튜닝합니다.
LPF_FC_HZ          = 0.8         # Hz (조향 LPF)
# 최근 샘플들로 이동속도를 추정할 때 사용하는 버퍼 길이(중앙값 필터 등).
# 길게 하면 속도 추정이 안정적이지만 반응이 느려집니다.
SPEED_BUF_LEN      = 10
# 위치 점프 등으로 계산된 “말도 안 되는 순간 속도”를 무시하기 위한 상한치입니다.
# GPS 글리치로 20 m/s 같은 값이 튀면 해당 샘플을 버립니다.
# 실제 최고 주행속도에 맞춰 적당히 크게 잡되, 너무 크게 잡으면 글리치를 못 거릅니다.
MAX_JITTER_SPEED   = 4.0         # m/s
# 최근 두 위치 사이 이동거리가 이 값보다 작으면 “헤딩 업데이트”를 하지 않습니다.
#거의 정지 상태에서의 방향 추정이 요동치는 것을 방지합니다.
MIN_MOVE_FOR_HEADING = 0.05      # m
# 메인 루프/퍼블리시 주기(초당 20회).
#높이면 반응이 빨라지지만 CPU 사용량↑. 센서/차량 주기와 조율하세요(예: 10~30 Hz).
FS_DEFAULT         = 20.0        # 퍼블리시/루프 Hz
# 최근 GPS fix가 이 시간 이상 들어오지 않으면 안전 정지(속도 0, 조향 0)합니다.
GPS_TIMEOUT_SEC    = 1.0         # 최근 fix 없을 때 안전정지

# ── 속도 명령 상수 ────────────────────────────────────────────────
#  - dSPACE 요구: 0~6 정수 코드. None이면 rosparam(~speed_code)을 기본으로 사용
#  - 플래그가 속도를 오버라이드하려면 보통 None으로 두세요.
SPEED_FORCE_CODE = None

# 기본 speed_code, 플래그에 안들어가있거나 속도 안정할때 기본으로 주는 속도값
BASE_SPEED = 5   # 기본 speed_code (정수 코드, 예: 2)

SPEED_CAP_CODE_DEFAULT = 10    # 코드 상한(예: 6)
STEP_PER_LOOP_DEFAULT = 2     # 매 루프당 정수 스텝(±1이 기본)

# 시각화 옵션
ANNOTATE_WAYPOINT_INDEX = True
DRAW_WAYPOINT_CIRCLES   = True

# 퍼블리시 토픽
TOPIC_SPEED_CMD    = '/vehicle/speed_cmd'     # Float32 (코드 0~6를 그대로 실수로 전송, 값은 정수만 보냄)
TOPIC_STEER_CMD    = '/vehicle/steer_cmd'     # Float32 (deg)
TOPIC_RTK_STATUS   = '/rtk/status'            # String ("FIX"/"FLOAT"/"NONE")
TOPIC_WP_INDEX     = '/tracker/wp_index'      # Int32 (반경 밖=0, 안=해당 인덱스 1-based)

# u-blox 옵셔널 의존성 (RTK 상태)
_HAVE_RELPOSNED = False
try:
    from ublox_msgs.msg import NavRELPOSNED9 as NavRELPOSNED
    _HAVE_RELPOSNED = True
except Exception:
    try:
        from ublox_msgs.msg import NavRELPOSNED
        _HAVE_RELPOSNED = True
    except Exception:
        _HAVE_RELPOSNED = False

# ──────────────────────────────────────────────────────────────────
# 플래그(구간) 정의
#  - 1-based 인덱스로 start/end 지정 (내부에서 0-based로 변환)
#  - radius_scale, lookahead_scale: 해당 구간에서 반경/룩어헤드 스케일
#  - speed_code: 구간 내 사용할 속도 코드(0~6). None이면 기본 속도 유지
#  - speed_cap:  구간 내 상한(선택)
#  - step_per_loop: 매 루프당 코드 변화 스텝(정수). 생략 시 1(기본).
# ──────────────────────────────────────────────────────────────────
FLAG_DEFS = [
    # # 예시 S자 구간: 룩어헤드 소폭 축소, 속도 2로 제한, 매 루프 1스텝
    # {'name': 'S_CURVE', 'start': 20, 'end': 28,
    #  'radius_scale': 0.9, 'lookahead_scale': 0.8,
    #  'speed_code': 2, 'speed_cap': 3, 'step_per_loop': 1},

    # 우회전 직각 구간: 정밀 제어 위해 속도 1
    {'name': 'RIGHT_ANGLE', 'start': 11, 'end': 15,
     'radius_scale': 1.0, 'lookahead_scale': 1.0,
     'speed_code': 6, 'speed_cap': 8, 'step_per_loop': 2},

    # # 교차로: 속도 0(정지) 또는 1로 크리핑
    # {'name': 'T_JUNCTION', 'start': 50, 'end': 55,
    #  'radius_scale': 0.7, 'lookahead_scale': 1.0,
    #  'speed_code': 0, 'speed_cap': 1, 'step_per_loop': 1},

    # 정지 구간: 정지 (예: 코드 4)
    {'name': 'STOP', 'start': 20, 'end': 21,
     'radius_scale': 0.5, 'lookahead_scale': 1.0,
     'speed_code': 0, 'speed_cap': 0, 'step_per_loop': 2},

    # 직선 가속 구간: 기본보다 빠르게 (예: 코드 4)
    {'name': 'STRAIGHT_FAST', 'start': 1, 'end': 10,
     'radius_scale': 1.0, 'lookahead_scale': 1.0,
     'speed_code': 8, 'speed_cap': 10, 'step_per_loop': 2},
]

# (선택) 구간 진입/이탈시 훅 함수
def on_enter_generic(zone): rospy.loginfo(f"[flag] ENTER {zone['name']} {zone['disp_range']}")
def on_exit_generic(zone):  rospy.loginfo(f"[flag] EXIT  {zone['name']} {zone['disp_range']}")

FLAG_HOOKS = {
    'S_CURVE':        (on_enter_generic, on_exit_generic),
    'RIGHT_ANGLE':    (on_enter_generic, on_exit_generic),
    'T_JUNCTION':     (on_enter_generic, on_exit_generic),
    'STRAIGHT_FAST':  (on_enter_generic, on_exit_generic),
}

# ──────────────────────────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────────────────────────
gps_queue = Queue()
latest_filtered_angle = 0.0
pos_buf   = deque(maxlen=SPEED_BUF_LEN*2)
speed_buf = deque(maxlen=SPEED_BUF_LEN)

pub_speed = None
pub_steer = None
pub_rtk   = None
pub_wpidx = None

rtk_status_txt = "NONE"
last_fix_time  = 0.0

# 퍼블리시/표시용 WP 인덱스: 반경 밖이면 -1, 안이면 0-based 인덱스
wp_index_active = -1

# 속도 코드 상태(정수 스텝 전용)
speed_cmd_current_code = 0     # 현재 퍼블리시할 코드(정수)
speed_desired_code     = 0     # 목표 코드(정수)

# 퍼블리시된 속도(코드)를 저장해서 시각화/로그에 쓰기
last_pub_speed_code = 0.0

# 플래그(구간) 런타임 상태
flag_zones = []              # 내부 0-based 변환된 리스트
active_flag = None           # 현재 활성 플래그 dict
just_entered = False
just_exited = False

# 로그
log_csv_path = None
_last_log_wall = 0.0

# ──────────────────────────────────────────────────────────────────
# 경로/파일 경로
# ──────────────────────────────────────────────────────────────────
def _default_paths():
    try:
        pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')

    waypoint_csv = os.path.join(pkg_path, 'config', 'raw_track_latlon_17(직진-우회전 신호).csv')
    logs_dir     = os.path.join(pkg_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_csv      = os.path.join(logs_dir, f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    return waypoint_csv, log_csv

WAYPOINT_CSV_DEFAULT, LOG_CSV_DEFAULT = _default_paths()

# ──────────────────────────────────────────────────────────────────
# 좌표/웨이포인트 유틸
# ──────────────────────────────────────────────────────────────────
def latlon_to_xy_fn(ref_lat, ref_lon):
    def _to_xy(lat, lon):
        northing = geopy.distance.geodesic((ref_lat, ref_lon), (lat, ref_lon)).meters
        easting  = geopy.distance.geodesic((ref_lat, ref_lon), (ref_lat, lon)).meters
        if lat < ref_lat: northing *= -1
        if lon < ref_lon: easting  *= -1
        return easting, northing
    return _to_xy

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def generate_waypoints_along_path(path, spacing=3.0):
    if len(path) < 2:
        return path
    new_points = [path[0]]
    dist_accum = 0.0
    last_point = np.array(path[0], dtype=float)
    for i in range(1, len(path)):
        current_point = np.array(path[i], dtype=float)
        segment = current_point - last_point
        seg_len = np.linalg.norm(segment)
        while dist_accum + seg_len >= spacing:
            remain = spacing - dist_accum
            direction = segment / seg_len
            new_point = last_point + direction * remain
            new_points.append(tuple(new_point))
            last_point = new_point
            segment = current_point - last_point
            seg_len = np.linalg.norm(segment)
            dist_accum = 0.0
        dist_accum += seg_len
        last_point = current_point
    if euclidean_dist(new_points[-1], path[-1]) > 1e-6:
        new_points.append(path[-1])
    return new_points

def wrap_deg(a): return (a + 180.0) % 360.0 - 180.0

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float); v2 = np.array(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0: return 0.0
    dot = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    ang = math.degrees(math.acos(dot))
    if v1[0]*v2[1] - v1[1]*v2[0] < 0: ang = -ang
    return ang

class AngleLPF:
    def __init__(self, fc_hz=3.0, init_deg=0.0):
        self.fc = fc_hz
        self.y = init_deg
        self.t_last = None
    def update(self, target_deg, t_sec):
        if self.t_last is None:
            self.t_last = t_sec
            self.y = target_deg
            return self.y
        dt = max(1e-3, t_sec - self.t_last)
        tau = 1.0 / (2.0 * math.pi * self.fc)
        alpha = dt / (tau + dt)
        err = wrap_deg(target_deg - self.y)
        self.y = wrap_deg(self.y + alpha * err)
        self.t_last = t_sec
        return self.y

def find_nearest_index(x, y, xs, ys, start_idx, window_ahead=60, window_back=10):
    n = len(xs)
    i0, i1 = max(0, start_idx - window_back), min(n - 1, start_idx + window_ahead)
    sub_x, sub_y = np.array(xs[i0:i1+1]), np.array(ys[i0:i1+1])
    d2 = (sub_x - x)**2 + (sub_y - y)**2
    return i0 + int(np.argmin(d2))

def target_index_from_lookahead(nearest_idx, Ld, spacing, n):
    steps = max(1, int(math.ceil(Ld / max(1e-6, spacing))))
    return min(n - 1, nearest_idx + steps)

def nearest_in_radius_index(x, y, xs, ys, radius):
    xs = np.asarray(xs); ys = np.asarray(ys)
    d2 = (xs - x)**2 + (ys - y)**2
    i = int(np.argmin(d2))
    if math.hypot(xs[i]-x, ys[i]-y) <= radius:
        return i
    return -1

def in_radius_idx(x, y, idx0, xs, ys, radius):
    tx, ty = xs[idx0], ys[idx0]
    return (math.hypot(x - tx, y - ty) <= radius)

# ──────────────────────────────────────────────────────────────────
# 플래그 매니저
# ──────────────────────────────────────────────────────────────────
def build_flag_zones(flag_defs):
    """1-based → 0-based로 변환하여 내부 구조로 보관"""
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
            'step_per_loop': int(fd.get('step_per_loop', STEP_PER_LOOP_DEFAULT)),
            'disp_range': f"{fd['start']}–{fd['end']} (1-based)"
        })
    return zones

def flag_enter(zone):
    enter, _ = FLAG_HOOKS.get(zone['name'], (None, None))
    if enter: enter(zone)

def flag_exit(zone):
    _, exitf = FLAG_HOOKS.get(zone['name'], (None, None))
    if exitf: exitf(zone)

def flag_update_state(x, y, nearest_idx, xs, ys, base_radius):
    """
    리턴:
      eff_radius, lookahead_scale, active_zone(dict|None), entered(bool), exited(bool)
    """
    global active_flag, just_entered, just_exited
    just_entered = False
    just_exited  = False

    eff_radius = base_radius
    look_scl   = 1.0

    # 1) 진입 체크
    if active_flag is None:
        for z in flag_zones:
            if in_radius_idx(x, y, z['start0'], xs, ys, base_radius):
                active_flag = z
                just_entered = True
                flag_enter(z)
                break
    # 2) 종료 체크
    else:
        z = active_flag
        if nearest_idx > z['end0']:
            just_exited = True
            flag_exit(z)
            active_flag = None

    # 3) 활성 플래그 적용
    if active_flag is not None:
        z = active_flag
        eff_radius = base_radius * z['radius_scale']
        look_scl   = z['lookahead_scale']

    return eff_radius, look_scl, active_flag, just_entered, just_exited

# ──────────────────────────────────────────────────────────────────
# ROS 콜백/퍼블리시
# ──────────────────────────────────────────────────────────────────
def gps_callback(data: NavSatFix):
    global last_fix_time
    if hasattr(data, "status") and getattr(data.status, "status", 0) < 0:
        return
    lat, lon = float(data.latitude), float(data.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return
    stamp = data.header.stamp.to_sec() if data.header and data.header.stamp else rospy.Time.now().to_sec()
    gps_queue.put((lat, lon, stamp))
    last_fix_time = time.time()

def _cb_relpos(msg):
    global rtk_status_txt
    try:
        carr_soln = int((int(msg.flags) >> 3) & 0x3)
        if carr_soln == 2:   rtk_status_txt = "FIX"
        elif carr_soln == 1: rtk_status_txt = "FLOAT"
        else:                rtk_status_txt = "NONE"
    except Exception:
        rtk_status_txt = "NONE"

def publish_all(event, one_based=True):
    """주기 퍼블리시(속도/조향/RTK/인덱스) + 타임아웃 안전정지"""
    global last_pub_speed_code
    now = time.time()
    no_gps = (now - last_fix_time) > rospy.get_param('~gps_timeout_sec', GPS_TIMEOUT_SEC)

    # 퍼블리시는 Float32지만 값은 정수로 보냄
    v_out_int = 0 if no_gps else int(speed_cmd_current_code)
    steer_out = 0.0 if no_gps else float(latest_filtered_angle)

    if pub_speed: pub_speed.publish(Float32(float(v_out_int)))
    if pub_steer: pub_steer.publish(Float32(steer_out))
    if pub_rtk:   pub_rtk.publish(String(rtk_status_txt))
    if pub_wpidx:
        if wp_index_active >= 0:
            idx_pub = (wp_index_active + 1) if one_based else wp_index_active
        else:
            idx_pub = 0
        pub_wpidx.publish(Int32(int(idx_pub)))

    last_pub_speed_code = float(v_out_int)

def _on_shutdown():
    global speed_cmd_current_code
    rospy.loginfo("[tracker] shutdown: ramping down speed to 0")

    try:
        rate = rospy.Rate(20)  # 20Hz = 50ms
        while speed_cmd_current_code > 0:
            speed_cmd_current_code = max(0, speed_cmd_current_code - 4)  # 4씩 감소
            if pub_speed:
                pub_speed.publish(Float32(float(speed_cmd_current_code)))
            if pub_steer:
                pub_steer.publish(Float32(0.0))
            rate.sleep()

        # 마지막으로 0 보장
        if pub_speed: pub_speed.publish(Float32(0.0))
        if pub_steer: pub_steer.publish(Float32(0.0))

    except Exception as e:
        rospy.logwarn(f"[tracker] shutdown ramp-down failed: {e}")


# ──────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────
def main():
    global pub_speed, pub_steer, pub_rtk, pub_wpidx
    global latest_filtered_angle, log_csv_path, wp_index_active
    global _last_log_wall, last_pub_speed_code, flag_zones
    global speed_cmd_current_code, speed_desired_code

    rospy.init_node('rtk_waypoint_tracker', anonymous=False)
    rospy.on_shutdown(_on_shutdown)

    # 파라미터
    waypoint_csv = rospy.get_param('~waypoint_csv', WAYPOINT_CSV_DEFAULT)
    log_csv_path = rospy.get_param('~log_csv',      LOG_CSV_DEFAULT)
    fs           = float(rospy.get_param('~fs',     FS_DEFAULT))
    one_based    = bool(rospy.get_param('~wp_index_one_based', True))
    ublox_ns     = rospy.get_param('~ublox_ns', '/gps1')
    fix_topic    = rospy.get_param('~fix_topic',    ublox_ns + '/fix')
    relpos_topic = rospy.get_param('~relpos_topic', ublox_ns + '/navrelposned')

    base_speed_code_param = int(rospy.get_param('~speed_code', BASE_SPEED))
    global_cap = int(rospy.get_param('~speed_cap_code', SPEED_CAP_CODE_DEFAULT))
    step_per_loop_global = int(rospy.get_param('~step_per_loop', STEP_PER_LOOP_DEFAULT))

    # 퍼블리셔/구독자
    pub_speed = rospy.Publisher(TOPIC_SPEED_CMD, Float32, queue_size=10)
    pub_steer = rospy.Publisher(TOPIC_STEER_CMD, Float32, queue_size=10)
    pub_rtk   = rospy.Publisher(TOPIC_RTK_STATUS, String,  queue_size=10)
    pub_wpidx = rospy.Publisher(TOPIC_WP_INDEX,   Int32,   queue_size=10)

    rospy.Subscriber(fix_topic, NavSatFix, gps_callback, queue_size=100)
    if _HAVE_RELPOSNED:
        rospy.Subscriber(relpos_topic, NavRELPOSNED, _cb_relpos, queue_size=50)
    rospy.loginfo("[tracker] subscribe: fix=%s, relpos=%s(%s)", fix_topic, relpos_topic, "ON" if _HAVE_RELPOSNED else "OFF")

    # CSV 로드
    if not os.path.exists(waypoint_csv):
        rospy.logerr("[tracker] waypoint csv not found: %s", waypoint_csv)
        return

    df = pd.read_csv(waypoint_csv)
    ref_lat = float(df['Lat'][0]); ref_lon = float(df['Lon'][0])
    to_xy = latlon_to_xy_fn(ref_lat, ref_lon)

    csv_coords = [to_xy(row['Lat'], row['Lon']) for _, row in df.iterrows()]
    original_path = list(csv_coords)
    spaced_waypoints = generate_waypoints_along_path(original_path, spacing=WAYPOINT_SPACING)
    spaced_x, spaced_y = zip(*spaced_waypoints)

    # 플래그(1→0 기반 변환)
    flag_zones = build_flag_zones(FLAG_DEFS)
    if flag_zones:
        rospy.loginfo("[flag] zones loaded: " + ", ".join([f"{z['name']}({z['disp_range']})" for z in flag_zones]))
    else:
        rospy.loginfo("[flag] no zones defined.")

    # ── 플롯: 상단 경로 + 하단 상태패널 ──
    plt.ion()
    fig = plt.figure(figsize=(7.8, 9.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
    ax  = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[1, 0]); ax_info.axis('off')

    ax.plot([p[0] for p in csv_coords], [p[1] for p in csv_coords], 'g-', label='CSV Path')
    ax.plot(spaced_x, spaced_y, 'b.-', markersize=3, label=f'{WAYPOINT_SPACING:.0f}m Waypoints')
    live_line,   = ax.plot([], [], 'r-', linewidth=1, label='Live GPS')
    current_pt,  = ax.plot([], [], 'ro', label='Current')
    target_line, = ax.plot([], [], 'g--', linewidth=1, label='Target Line')
    target_pt,   = ax.plot([], [], marker='*', markersize=12, color='m', linestyle='None', label='Target ★')
    ax.axis('equal'); ax.grid(True); ax.legend()

    hud_text = ax.text(0.98, 0.02, "",
                       transform=ax.transAxes, ha='right', va='bottom',
                       fontsize=9, bbox=dict(fc='white', alpha=0.75, ec='0.5'))

    minx, maxx = min(min([p[0] for p in csv_coords]), min(spaced_x))-10, max(max([p[0] for p in csv_coords]), max(spaced_x))+10
    miny, maxy = min(min([p[1] for p in csv_coords]), min(spaced_y))-10, max(max([p[1] for p in csv_coords]), max(spaced_y))+10
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

    if ANNOTATE_WAYPOINT_INDEX or DRAW_WAYPOINT_CIRCLES:
        for idx, (xw, yw) in enumerate(zip(spaced_x, spaced_y), 1):
            if ANNOTATE_WAYPOINT_INDEX:
                ax.text(xw, yw, str(idx), fontsize=8, ha='center', va='center', color='black')
            if DRAW_WAYPOINT_CIRCLES:
                ax.add_patch(Circle((xw, yw), TARGET_RADIUS_END, color='blue', fill=False, linestyle='--', alpha=0.5))

    # 조향 필터/상태
    lpf = AngleLPF(fc_hz=LPF_FC_HZ)
    prev_Ld = LOOKAHEAD_MIN
    nearest_idx_prev = 0
    last_heading_vec = None

    # 속도 초기화(정수 코드)
    if SPEED_FORCE_CODE is not None:
        speed_desired_code = int(max(0, min(global_cap, int(SPEED_FORCE_CODE))))
    else:
        speed_desired_code = int(max(0, min(global_cap, int(base_speed_code_param))))
    speed_cmd_current_code = int(speed_desired_code)

    # 퍼블리시 타이머
    rospy.Timer(rospy.Duration(1.0/max(1.0, fs)), lambda e: publish_all(e, one_based=one_based))

    rate = rospy.Rate(fs)
    try:
        while not rospy.is_shutdown():
            updated = False
            while not gps_queue.empty():
                lat, lon, tsec = gps_queue.get()
                x, y = to_xy(lat, lon)
                updated = True

                # 속도 추정(스파이크 제거)
                if len(pos_buf) > 0:
                    t_prev, x_prev, y_prev = pos_buf[-1]
                    dt = max(1e-3, tsec - t_prev)
                    d  = math.hypot(x - x_prev, y - y_prev)
                    inst_v = d / dt
                    if inst_v > MAX_JITTER_SPEED:
                        continue
                    speed_buf.append(inst_v)
                pos_buf.append((tsec, x, y))
                speed_mps = float(np.median(speed_buf)) if speed_buf else 0.0

                # 제어용 타겟 인덱스(룩어헤드)
                nearest_idx = find_nearest_index(x, y, spaced_x, spaced_y, nearest_idx_prev, 80, 15)
                nearest_idx_prev = nearest_idx

                # 플래그 상태 갱신 → 반경/룩어헤드/속도 오버라이드 정보
                eff_radius, look_scl, z_active, entered, exited = flag_update_state(
                    x, y, nearest_idx, spaced_x, spaced_y, TARGET_RADIUS_END
                )

                # 룩어헤드
                Ld_target_base = max(LOOKAHEAD_MIN, min(LOOKAHEAD_MAX, LOOKAHEAD_MIN + LOOKAHEAD_K * speed_mps))
                Ld_target = max(LOOKAHEAD_MIN, min(LOOKAHEAD_MAX, Ld_target_base * look_scl))
                Ld = prev_Ld + 0.2 * (Ld_target - prev_Ld)
                prev_Ld = Ld

                # 타깃 인덱스 계산
                tgt_idx = target_index_from_lookahead(nearest_idx, Ld, WAYPOINT_SPACING, len(spaced_x))
                tx, ty = spaced_x[tgt_idx], spaced_y[tgt_idx]
                target_line.set_data([x, tx], [y, ty])
                target_pt.set_data([tx], [ty])  # ★ 별표 표시

                # 퍼블리시/표시용 인덱스: (효과 적용된 반경으로) 반경 안이면 0-based, 아니면 -1
                wp_index_active = nearest_in_radius_index(x, y, spaced_x, spaced_y, eff_radius)

                # 헤딩 추정(최근 이동)
                heading_vec = None
                for k in range(2, min(len(pos_buf), 5)+1):
                    t0, x0, y0 = pos_buf[-k]
                    if math.hypot(x - x0, y - y0) >= MIN_MOVE_FOR_HEADING:
                        heading_vec = (x - x0, y - y0)
                        break
                if heading_vec is not None:
                    last_heading_vec = heading_vec
                elif last_heading_vec is None:
                    # 아직 헤딩 부트스트랩 안됨
                    continue

                # 조향 계산 + 동적 LPF
                target_vec = (tx - x, ty - y)
                raw_angle = angle_between(last_heading_vec, target_vec)
                base_fc = LPF_FC_HZ
                lpf.fc = min(2.0, base_fc + 0.5) if abs(raw_angle) > 10 else base_fc
                filt_angle = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, lpf.update(raw_angle, tsec)))
                latest_filtered_angle = SIGN_CONVENTION * filt_angle

                # ── 속도 코드 산출(플래그 → 기본) + 정수 스텝 램핑 ──
                if SPEED_FORCE_CODE is not None:
                    desired = int(max(0, min(global_cap, int(SPEED_FORCE_CODE))))
                    cap = global_cap
                    active_name = "FORCE"
                    step_this_loop = step_per_loop_global
                else:
                    if z_active is not None and (z_active.get('speed_code') is not None):
                        desired = int(z_active['speed_code'])
                        cap = int(z_active['speed_cap']) if z_active.get('speed_cap') is not None else global_cap
                        step_this_loop = int(max(1, z_active.get('step_per_loop', step_per_loop_global)))
                        active_name = z_active['name']
                    else:
                        desired = int(base_speed_code_param)
                        cap = global_cap
                        step_this_loop = step_per_loop_global
                        active_name = None

                desired = max(0, min(cap, desired))
                speed_desired_code = desired

                # 정수 스텝: 현재 → 목표로 ±step_this_loop 만큼만 이동
                if speed_cmd_current_code < speed_desired_code:
                    speed_cmd_current_code = min(speed_desired_code, speed_cmd_current_code + step_this_loop)
                elif speed_cmd_current_code > speed_desired_code:
                    speed_cmd_current_code = max(speed_desired_code, speed_cmd_current_code - step_this_loop)
                speed_cmd_current_code = max(0, min(cap, speed_cmd_current_code))

                # 시각화 갱신
                live_line.set_data([p[1] for p in pos_buf], [p[2] for p in pos_buf])
                current_pt.set_data([x], [y])

                # ── 하단 상태 패널 + 메인 HUD ──
                ax_info.clear(); ax_info.axis('off')
                heading_deg = (math.degrees(math.atan2(last_heading_vec[1], last_heading_vec[0]))
                               if last_heading_vec is not None else float('nan'))
                wp_display = (wp_index_active + 1) if (wp_index_active >= 0) else 0
                tgt_display = int(tgt_idx + 1)  # 1-based 표시

                status = [
                    f"TGT={tgt_display}",
                    f"WP(in-radius)={wp_display}",
                    f"Speed(code): cur={speed_cmd_current_code} → des={speed_desired_code} (step={step_this_loop})",
                    f"Meas v={speed_mps:.2f} m/s",
                    f"Steer={latest_filtered_angle:+.1f}°",
                    f"Heading={heading_deg:.1f}°",
                    f"RTK={rtk_status_txt}",
                ]
                if active_name:
                    status.append(f"FLAG={active_name}")
                ax_info.text(0.02, 0.5, " | ".join(status), fontsize=11, va='center')

                hud_text.set_text(
                    f"TGT={tgt_display} | pub_v={float(speed_cmd_current_code):.1f} | meas_v={speed_mps:.2f} m/s | "
                    f"steer={latest_filtered_angle:+.1f}° | WP={wp_display}"
                    + (f" | FLAG={active_name}" if active_name else "")
                )

                # 로그 (0.5s에 한번) - 타깃 인덱스(1-based) 포함
                noww = time.time()
                if log_csv_path and (noww - _last_log_wall > 0.5):
                    try:
                        new = not os.path.exists(log_csv_path)
                        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
                        with open(log_csv_path, 'a', newline='') as f:
                            w = csv.writer(f)
                            if new:
                                w.writerow(['time','lat','lon','x','y',
                                            'wp_in_radius(1based)','tgt_idx(1based)',
                                            'tx','ty',
                                            'steer_deg','meas_speed_mps',
                                            'speed_cur_code','speed_des_code','Ld',
                                            'rtk','flag_active','flag_enter','flag_exit'])
                            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                        f"{lat:.7f}", f"{lon:.7f}", f"{x:.3f}", f"{y:.3f}",
                                        wp_display, tgt_display, f"{tx:.3f}", f"{ty:.3f}",
                                        f"{latest_filtered_angle:.2f}", f"{speed_mps:.2f}",
                                        int(speed_cmd_current_code), int(speed_desired_code),
                                        f"{Ld:.2f}", rtk_status_txt,
                                        (active_name or ""), "1" if entered else "0", "1" if exited else "0"])
                        _last_log_wall = noww
                    except Exception as e:
                        rospy.logwarn(f"[tracker] log write failed: {e}")

                # 콘솔 디버그
                rospy.loginfo_throttle(
                    0.5,
                    f"TGT={tgt_display} | WP(in-radius)={wp_display} / tgt={tgt_display}/{len(spaced_x)} | "
                    f"meas_v={speed_mps:.2f} m/s | code(cur→des)={speed_cmd_current_code}->{speed_desired_code} | "
                    f"Ld={Ld:.2f} | steer={latest_filtered_angle:+.2f} deg | RTK={rtk_status_txt}"
                    + (f" | FLAG={active_name}" if active_name else "")
                    + (" | ENTER" if entered else "")
                    + (" | EXIT" if exited else "")
                )

            if updated:
                fig.canvas.draw_idle()
            plt.pause(0.001)
            rate.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        _on_shutdown() # ctrl + c 줬을때 셧다운 종료 함수 실행. 종료될때 속도,조향각 0을 줌.

if __name__ == '__main__':
    main()
