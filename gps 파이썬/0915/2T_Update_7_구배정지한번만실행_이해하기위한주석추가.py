#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
요약
- 램핑/클램핑/hold 제거(속도는 즉시 반영)
- GRADE_UP 구간에서 '원샷' 구배정지 알고리즘:
  1) 구간(반경내 WP) 진입시: 4초 정지(속도0, 조향0, grade=1)
  2) 이어서 1초 직진: 속도 5, 조향 0(헤딩 안정화)
  3) 이후 정상 로직 복귀. 이 알고리즘은 노드 실행 중 단 한 번만 동작.
- 구배 토픽은 '/gps/GRADEUP_ON' 사용(래치=False), 종료/시작 때 0도 보내며,
  알고리즘 동작 중에는 강제로 1 유지.
"""

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
# 기본 파라미터
# ──────────────────────────────────────────────────────────────────
WAYPOINT_SPACING   = 2.5
TARGET_RADIUS_END  = 2.0
MAX_STEER_DEG      = 27.0
SIGN_CONVENTION    = -1.0

LOOKAHEAD_MIN      = 3.2
LOOKAHEAD_MAX      = 4.0
LOOKAHEAD_K        = 0.2

LPF_FC_HZ          = 0.8
SPEED_BUF_LEN      = 10
MAX_JITTER_SPEED   = 4.0
MIN_MOVE_FOR_HEADING = 0.05 # 5 cm 이상 이동했을때 헤딩 벡터 새로 갱신

FS_DEFAULT         = 20.0
GPS_TIMEOUT_SEC    = 1.0

# 구배 정지 시나리오(원샷)
GRADE_HOLD_SEC         = 4.0   # 첫 정지 4초
GRADE_STRAIGHT_SEC     = 1.0   # 이후 직진(조향0) 1초
GRADE_STRAIGHT_SPEED   = 5     # 직진 단계 속도 코드

# 속도 명령(즉시 반영, 램핑/클램핑 없음)
SPEED_FORCE_CODE = None
BASE_SPEED       = 5
SPEED_CAP_CODE_DEFAULT = 10     # (미사용) 남겨둠
STEP_PER_LOOP_DEFAULT  = 2      # (미사용) 남겨둠

# 시각화
ANNOTATE_WAYPOINT_INDEX = True
DRAW_WAYPOINT_CIRCLES   = True

# 퍼블리시 토픽
TOPIC_SPEED_CMD   = '/gps/speed_cmd'     # Float32
TOPIC_STEER_CMD   = '/gps/steer_cmd'     # Float32 (deg)
TOPIC_RTK_STATUS  = '/gps/rtk_status'    # String ("FIX"/"FLOAT"/"NONE")
TOPIC_WP_INDEX    = '/gps/wp_index'      # Int32 (1-based, 반경 밖=0)

# 통신 노드와 합의한 정식 이름(언더스코어 없이)
TOPIC_WP_GRADEUP_ON = '/gps/GRADEUP_ON'  # Int32 (0/1), latch=False

# u-blox (옵션)
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
# ──────────────────────────────────────────────────────────────────
FLAG_DEFS = [
    {'name': 'GRADE_START', 'start': 4, 'end': 5,
     'radius_scale': 1.0, 'lookahead_scale': 0.95,
     'speed_code': 5, 'speed_cap': 7,
     'stop_on_hit': False, 'stop_duration_sec': None,
     'grade_topic': 1},

    # 구배 정지 트리거 구간(반경내 WP 들어오면 1회만 실행)
    {'name': 'GRADE_UP', 'start': 6, 'end': 6,
     'radius_scale': 1.0, 'lookahead_scale': 0.95,
     'speed_code': None, 'speed_cap': None, # speed_code=None → STRAIGHT 끝나면 기본속도(base_speed, 예: 5)로 정상 복귀 ✅
                                            # speed_code=0 → STRAIGHT 끝나자마자 속도 0으로 덮어씌워져서 차가 또 멈춤 ❌
     'stop_on_hit': True, 'stop_duration_sec': GRADE_HOLD_SEC,
     'grade_topic': 1},

    {'name': 'GRADE_GO', 'start': 7, 'end': 8,
     'radius_scale': 1.0, 'lookahead_scale': 0.95,
     'speed_code': 5, 'speed_cap': 7,
     'stop_on_hit': False, 'stop_duration_sec': None,
     'grade_topic': 1},

    {'name': 'GRADE_END', 'start': 9, 'end': 11,
     'radius_scale': 1.0, 'lookahead_scale': 0.95,
     'speed_code': 5, 'speed_cap': 7,
     'stop_on_hit': False, 'stop_duration_sec': None,
     'grade_topic': 0},
]

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
pub_grade = None

rtk_status_txt = "NONE"
last_fix_time  = 0.0

wp_index_active = -1

speed_cmd_current_code = 0
speed_desired_code     = 0
last_pub_speed_code    = 0.0

flag_zones = []
active_flag = None

log_csv_path = None
_last_log_wall = 0.0

final_stop_latched = False  # (미사용)

# 구배 토픽 값(0/1)
grade_topic_value = 0

# ── 구배 정지 상태머신(원샷) ─────────────────────────────────────
STATE_IDLE     = 0
STATE_HOLD     = 1  # 4초 정지
STATE_STRAIGHT = 2  # 1초 직진(조향 0)
STATE_DONE     = 3

grade_stop_state    = STATE_IDLE
grade_stop_started  = False
grade_stop_done     = False
grade_stop_t_end    = 0.0
grade_stop_zone_rng = (0, 0)  # (start1, end1) 1-based

# ──────────────────────────────────────────────────────────────────
# 경로/파일 경로
# ──────────────────────────────────────────────────────────────────
def _default_paths():
    try:
        pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')

    waypoint_csv = os.path.join(pkg_path, 'config', 'raw_track_latlon_18.csv')
    logs_dir     = os.path.join(pkg_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_csv      = os.path.join(logs_dir, f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    return waypoint_csv, log_csv

WAYPOINT_CSV_DEFAULT, LOG_CSV_DEFAULT = _default_paths()

# ──────────────────────────────────────────────────────────────────
# 유틸
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

def find_nearest_index(x, y, xs, ys, start_idx, window_ahead=80, window_back=15):
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
# 플래그 헬퍼/상태
# ──────────────────────────────────────────────────────────────────
def build_flag_zones(flag_defs):
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
            #'step_per_loop': int(fd.get('step_per_loop', STEP_PER_LOOP_DEFAULT)),
            'stop_on_hit': bool(fd.get('stop_on_hit', False)),
            'stop_duration_sec': float(fd.get('stop_duration_sec', GRADE_HOLD_SEC)
                                       if fd.get('stop_duration_sec', None) is not None
                                       else GRADE_HOLD_SEC),
            'sequential': bool(fd.get('sequential', False)),
            'grade_topic': fd.get('grade_topic', None),
            'disp_range': f"{fd['start']}–{fd['end']} (1-based)"
        })
    return zones

def zone_contains_wp_display(zone, wp_display):
    """현재 표시 인덱스(wp_display, 1-based)가 구간 범위 안인지."""
    if wp_display <= 0: return False
    start1, end1 = zone['start0'] + 1, zone['end0'] + 1
    return (start1 <= wp_display <= end1)

def get_active_zone_by_radius(x, y, xs, ys, base_radius):
    """시작 WP 원 안에 들어오면 활성화(시각화/스케일용)."""
    for z in flag_zones:
        if in_radius_idx(x, y, z['start0'], xs, ys, base_radius):
            return z
    return None

# ──────────────────────────────────────────────────────────────────
# 구배 정지(원샷) 상태머신 함수
# ──────────────────────────────────────────────────────────────────
def grade_stop_can_start(wp_display):
    """아직 한 번도 안 돌았고, GRADE_UP(stop_on_hit) 구간 안에 들어오면 트리거."""
    if grade_stop_started or grade_stop_done: # 정지 시작, 정지 끝남이 둘 중 하나라도 true 이면 실행 x
        return False, None
    for z in flag_zones:
        # stop_on_hit == True (진입 시 멈추도록 지정된 플래그여야 함)
        # name == 'GRADE_UP' (특히 이름이 "GRADE_UP"인 플래그)
        if z.get('stop_on_hit', False) and z['name'] == 'GRADE_UP':
            # zone_contains_wp_display(z, wp_display) 함수로, 현재 표시된 웨이포인트 인덱스(wp_display)가 그 구간(z) 안에 들어왔는지 검사합니다.
            if zone_contains_wp_display(z, wp_display):
                # 들어왔다면 (True, z)를 반환 → 정지 시작 조건 충족.
                return True, z
    return False, None

def grade_stop_start(now, zone):
    """4초 정지 단계 시작."""
    global grade_stop_state, grade_stop_started, grade_stop_t_end, grade_stop_zone_rng
    grade_stop_started = True
    grade_stop_state   = STATE_HOLD
    grade_stop_t_end   = now + float(zone.get('stop_duration_sec', GRADE_HOLD_SEC))
    grade_stop_zone_rng = (zone['start0']+1, zone['end0']+1)
    rospy.loginfo(f"[GRADE] STOP start for {grade_stop_t_end - now:.1f}s at zone {zone['name']} {grade_stop_zone_rng}")

def grade_stop_update(now):
    """상태 진행 및 강제 출력값 반환.
    returns: (override_active, v_code, steer_deg, grade_flag)
    """
    global grade_stop_state, grade_stop_done, grade_stop_t_end

    if grade_stop_state == STATE_IDLE or grade_stop_state == STATE_DONE:
        return (False, None, None, None)

    # 1) HOLD: 4초 정지(속도0/조향0/grade=1)
    if grade_stop_state == STATE_HOLD:
        if now >= grade_stop_t_end:
            grade_stop_state = STATE_STRAIGHT
            grade_stop_t_end = now + GRADE_STRAIGHT_SEC
            rospy.loginfo(f"[GRADE] STRAIGHT start for {GRADE_STRAIGHT_SEC:.1f}s (v={GRADE_STRAIGHT_SPEED}, steer=0)")
            return (True, 0, 0.0, 1)
        else:
            return (True, 0, 0.0, 1)

    # 2) STRAIGHT: 1초 직진(속도5/조향0/grade=1)
    if grade_stop_state == STATE_STRAIGHT:
        if now >= grade_stop_t_end:
            grade_stop_state = STATE_DONE
            grade_stop_done  = True
            rospy.loginfo("[GRADE] sequence done (one-shot)")
            return (False, None, None, None)  # 정상 로직으로 복귀
        else:
            return (True, GRADE_STRAIGHT_SPEED, 0.0, 1)

    return (False, None, None, None)

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
    """주기 퍼블리시(구배 원샷 상태에 따라 강제값 우선)."""
    global last_pub_speed_code
    now = time.time()
    no_gps = (now - last_fix_time) > rospy.get_param('~gps_timeout_sec', GPS_TIMEOUT_SEC)

    # 기본 출력(정상)
    v_out_int = int(speed_cmd_current_code)
    steer_out = float(latest_filtered_angle)

    # GPS 없으면 안전정지
    if no_gps:
        v_out_int = 0
        steer_out = 0.0

    # 구배 원샷 상태 강제(정지/직진 단계)
    override, v_force, steer_force, grade_force = grade_stop_update(now)
    if override:
        if v_force is not None:   v_out_int = int(v_force)
        if steer_force is not None: steer_out = float(steer_force)
        # grade 토픽도 강제 1
        if pub_grade and (grade_force is not None):
            pub_grade.publish(Int32(int(grade_force)))

    # 퍼블리시
    if pub_speed: pub_speed.publish(Float32(float(v_out_int)))
    if pub_steer: pub_steer.publish(Float32(steer_out))
    if pub_rtk:   pub_rtk.publish(String(rtk_status_txt))
    if pub_wpidx:
        idx_pub = (wp_index_active + 1) if (wp_index_active >= 0 and one_based) else (wp_index_active if wp_index_active >= 0 else 0)
        pub_wpidx.publish(Int32(int(idx_pub)))

    # grade 토픽(기본: 현재 계산값)
    if pub_grade:
        # override 단계가 아니면 평상시 계산값을 퍼블리시
        if not override:
            pub_grade.publish(Int32(int(grade_topic_value)))

    last_pub_speed_code = float(v_out_int)

def _publish_idle_zeros_once():
    """시작 직후 안전 0 한 번."""
    try:
        if pub_speed: pub_speed.publish(Float32(0.0))
        if pub_steer: pub_steer.publish(Float32(0.0))
        if pub_wpidx: pub_wpidx.publish(Int32(0))
        if pub_rtk:   pub_rtk.publish(String("NONE"))
        if pub_grade: pub_grade.publish(Int32(0))
    except Exception as e:
        rospy.logwarn(f"[tracker] initial zero publish failed: {e}")

def _on_shutdown():
    """종료 시 0 한번 깔끔하게 송신."""
    try:
        if pub_speed: pub_speed.publish(Float32(0.0))
        if pub_steer: pub_steer.publish(Float32(0.0))
        if pub_wpidx: pub_wpidx.publish(Int32(0))
        if pub_grade: pub_grade.publish(Int32(0))
        if pub_rtk:   pub_rtk.publish(String("NONE"))

        rospy.sleep(0.03)   # ← 여기 추가 (30ms 전송 여유)
        rospy.loginfo("[tracker] shutdown zeros published")
    except Exception as e:
        rospy.logwarn(f"[tracker] shutdown failed: {e}")

# ──────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────
def main():
    global pub_speed, pub_steer, pub_rtk, pub_wpidx, pub_grade
    global latest_filtered_angle, log_csv_path, wp_index_active
    global _last_log_wall, last_pub_speed_code, flag_zones
    global speed_cmd_current_code, speed_desired_code
    global active_flag, grade_topic_value

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

    # 퍼블리셔/구독자
    pub_speed = rospy.Publisher(TOPIC_SPEED_CMD, Float32, queue_size=10)
    pub_steer = rospy.Publisher(TOPIC_STEER_CMD, Float32, queue_size=10)
    pub_rtk   = rospy.Publisher(TOPIC_RTK_STATUS, String,  queue_size=10)
    pub_wpidx = rospy.Publisher(TOPIC_WP_INDEX,   Int32,   queue_size=10)
    pub_grade = rospy.Publisher(TOPIC_WP_GRADEUP_ON, Int32, queue_size=10, latch=False)  # latch 끔

    rospy.Subscriber(fix_topic, NavSatFix, gps_callback, queue_size=100)
    if _HAVE_RELPOSNED:
        rospy.Subscriber(relpos_topic, NavRELPOSNED, _cb_relpos, queue_size=50)

    rospy.loginfo("[tracker] subscribe: fix=%s, relpos=%s(%s)", fix_topic, relpos_topic, "ON" if _HAVE_RELPOSNED else "OFF")

    # 시작 안전값(초기화)
    _publish_idle_zeros_once()
    rospy.sleep(0.01)

    # CSV 로드
    if not os.path.exists(waypoint_csv):
        rospy.logerr("[tracker] waypoint csv not found: %s", waypoint_csv)
        return
    # csv lat lon 변환
    df = pd.read_csv(waypoint_csv)
    ref_lat = float(df['Lat'][0]); ref_lon = float(df['Lon'][0])
    to_xy = latlon_to_xy_fn(ref_lat, ref_lon)
    # 웨이포인트 찍기
    csv_coords = [to_xy(row['Lat'], row['Lon']) for _, row in df.iterrows()]
    spaced_waypoints = generate_waypoints_along_path(csv_coords, spacing=WAYPOINT_SPACING)
    spaced_x, spaced_y = zip(*spaced_waypoints)

    # 플래그
    flag_zones = build_flag_zones(FLAG_DEFS)
    if flag_zones:
        rospy.loginfo("[flag] zones loaded: " + ", ".join([f"{z['name']}({z['disp_range']})" for z in flag_zones]))
    else:
        rospy.loginfo("[flag] no zones defined.")

    # ── 플롯 ──
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
    ax.axis('equal'); ax.grid(True)
    ax.legend()

    hud_text = ax.text(0.98, 0.02, "", transform=ax.transAxes, ha='right', va='bottom',
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

    # 속도 초기화(즉시 반영 모드)
    if SPEED_FORCE_CODE is not None:
        speed_desired_code = int(SPEED_FORCE_CODE)
    else:
        speed_desired_code = int(base_speed_code_param)
    speed_cmd_current_code = int(speed_desired_code)

    # 퍼블리시 타이머
    rospy.Timer(rospy.Duration(1.0/max(1.0, fs)), lambda e: publish_all(e, one_based=one_based))

    # 루프 (50ms 주기)
    rate = rospy.Rate(fs)
    try:
        while not rospy.is_shutdown(): # 셧다운 아닐때
            updated = False 
            while not gps_queue.empty(): # gps 값 있을때 실행
                lat, lon, tsec = gps_queue.get()
                x, y = to_xy(lat, lon)
                updated = True

                # 속도 추정(스파이크 제거), 
                # GPS 위치 변화량을 이용해서 차량 속도를 추정하는 계산 로직
                if len(pos_buf) > 0:
                    t_prev, x_prev, y_prev = pos_buf[-1]
                    dt = max(1e-3, tsec - t_prev)
                    d  = math.hypot(x - x_prev, y - y_prev)
                    inst_v = d / dt
                    if inst_v <= MAX_JITTER_SPEED:
                        speed_buf.append(inst_v)
                pos_buf.append((tsec, x, y))
                speed_mps = float(np.median(speed_buf)) if speed_buf else 0.0

                # 최근접/룩어헤드
                nearest_idx = find_nearest_index(x, y, spaced_x, spaced_y, nearest_idx_prev, 80, 15)
                nearest_idx_prev = nearest_idx

                # 룩어헤드 길이
                Ld_target_base = max(LOOKAHEAD_MIN, min(LOOKAHEAD_MAX, LOOKAHEAD_MIN + LOOKAHEAD_K * speed_mps))
                Ld = prev_Ld + 0.2 * (Ld_target_base - prev_Ld)
                prev_Ld = Ld

                # 타깃 인덱스
                tgt_idx = target_index_from_lookahead(nearest_idx, Ld, WAYPOINT_SPACING, len(spaced_x)) # 타겟 인덱스 값
                tx, ty = spaced_x[tgt_idx], spaced_y[tgt_idx]
                target_line.set_data([x, tx], [y, ty])
                target_pt.set_data([tx], [ty])

                # 반경 내 인덱스, 즉 내 위치가 어느 인덱스 wp에 있는지(1-based 표시)
                wp_index_active = nearest_in_radius_index(x, y, spaced_x, spaced_y, TARGET_RADIUS_END)
                wp_display = (wp_index_active + 1) if (wp_index_active >= 0) else 0

                # (시각화용) 시작 원 기준 활성 구간
                active_flag = get_active_zone_by_radius(x, y, spaced_x, spaced_y, TARGET_RADIUS_END)

                # ── 구배 정지 트리거(원샷) ──
                # 언덕 정지(grade stop)”를 언제 시작할지 감지하고, 그 외 평상시에는 grade 토픽 값을 계산하는 로직
                # can_start: True/False (정지 조건 충족 여부)
                # hit_zone: 조건 충족된 zone 객체 (예: {'name': 'GRADE_UP', ...})
                can_start, hit_zone = grade_stop_can_start(wp_display) # grade_stop_can_start(wp_display):지금 언덕 정지를 시작해도 되는지 검사.
                                                                       # wp_display: 현재 차량이 반경 안에 들어온 웨이포인트 번호 (1-based, 없으면 0).
                # 만약 정지를 시작할 수 있다면(can_start=True) → grade_stop_start 실행.                                                        
                if can_start and hit_zone is not None: # 
                    grade_stop_start(time.time(), hit_zone) # time.time() = 현재 시각, hit_zone = 트리거된 구간 정보.=> 이렇게 되면 STATE_HOLD(4초 정지)로 상태머신이 진입합니다.
                # ── grade 토픽 값 계산(override 단계가 아닐 때만) ──
                if active_flag is not None and (active_flag.get('grade_topic') is not None): # active_flag: 지금 차량이 들어와 있는 플래그 zone 객체 (없으면 None).
                    """
                                                                                            zone 안에 grade_topic 값이 정의돼 있다면:

                                                                                            0이 아니면 grade_topic_value = 1

                                                                                            0이면 grade_topic_value = 0
                    """                                                                        
                    try:
                        grade_topic_value = 1 if int(active_flag['grade_topic']) != 0 else 0
                    except Exception:
                        grade_topic_value = 0
                else:
                    grade_topic_value = 0

                # 헤딩 벡터 추정
                heading_vec = None
                for k in range(2, min(len(pos_buf), 5)+1):
                    t0, x0, y0 = pos_buf[-k]
                    if math.hypot(x - x0, y - y0) >= MIN_MOVE_FOR_HEADING:
                        heading_vec = (x - x0, y - y0)
                        break
                if heading_vec is not None:
                    last_heading_vec = heading_vec

                # 조향 계산(grade STRAIGHT 단계에서는 publish_all에서 강제 0됨)
                # 즉, 헤딩이 없는 초기 상황이나 정지 상태 → 조향각을 구할 수 없으니 그냥 0으로 둡니다.
                if ('last_heading_vec' not in locals()) or last_heading_vec is None:
                    latest_filtered_angle = 0.0
                else:
                    target_vec = (tx - x, ty - y) # 목표를 향하는 방향 벡터
                    raw_angle = angle_between(last_heading_vec, target_vec) # 현재 진행방향(last_heading_vec)과 목표방향(target_vec) 사이의 각도를 계산.
                    # lpf 필터
                    base_fc = LPF_FC_HZ
                    lpf.fc = min(2.0, base_fc + 0.5) if abs(raw_angle) > 10 else base_fc
                    filt_angle = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, lpf.update(raw_angle, tsec)))
                    latest_filtered_angle = SIGN_CONVENTION * filt_angle

                # ── 속도 코드 산출(즉시 반영) ──
                # active_flag에 speed_code가 있으면 그대로, 없으면 base_speed 유지
                # active_flag: 현재 차량이 들어와 있는 flag zone(플래그 구간)의 정보.
                # 예: {'name': 'GRADE_START', 'speed_code': 5, ...}
                if active_flag is not None and (active_flag.get('speed_code') is not None):
                    # 지금 zone이 있고, 그 zone에 speed_code 값이 정의돼 있다면 → 그 값을 그대로 사용.
                    desired = int(active_flag['speed_code'])
                else:
                    # 없으면(=zone이 없거나, speed_code 없음) → 기본 속도 코드(base_speed_code_param) 사용.
                    desired = int(base_speed_code_param)

                speed_desired_code     = desired
                speed_cmd_current_code = desired  # 즉시 적용

                # 시각화/HUD
                live_line.set_data([p[1] for p in pos_buf], [p[2] for p in pos_buf])
                current_pt.set_data([x], [y])

                ax_info.clear(); ax_info.axis('off')
                heading_deg = (math.degrees(math.atan2(last_heading_vec[1], last_heading_vec[0]))
                               if ('last_heading_vec' in locals() and last_heading_vec is not None) else float('nan'))
                tgt_display = int(tgt_idx + 1)

                status = [
                    f"TGT={tgt_display}",
                    f"WP(in-radius)={(wp_index_active + 1) if wp_index_active>=0 else 0}",
                    f"Speed(code): cur={speed_cmd_current_code} → des={speed_desired_code}",
                    f"Meas v={speed_mps:.2f} m/s",
                    f"Steer={latest_filtered_angle:+.1f}°",
                    f"Heading={heading_deg:.1f}°",
                    f"RTK={rtk_status_txt}",
                    f"GRADE_UP_ON={grade_topic_value}",
                    f"GS_STATE={grade_stop_state}",
                ]
                ax_info.text(0.02, 0.5, " | ".join([s for s in status if s]), fontsize=11, va='center')

                hud_text.set_text(
                    f"TGT={tgt_display} | pub_v={last_pub_speed_code:.1f} "
                    f"| meas_v={speed_mps:.2f} m/s | steer={latest_filtered_angle:+.1f}° "
                    f"| WP={(wp_index_active + 1) if wp_index_active>=0 else 0}"
                    + f" | GS_STATE={grade_stop_state}"
                    + f" | GRADE_UP_ON={grade_topic_value}"
                )

                # 로그 (0.5s)
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
                                            'rtk','grade_up_on','grade_state'])
                            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                        f"{lat:.7f}", f"{lon:.7f}", f"{x:.3f}", f"{y:.3f}",
                                        (wp_index_active + 1) if wp_index_active>=0 else 0,
                                        tgt_display, f"{tx:.3f}", f"{ty:.3f}",
                                        f"{latest_filtered_angle:.2f}", f"{speed_mps:.2f}",
                                        int(speed_cmd_current_code), int(speed_desired_code),
                                        f"{Ld:.2f}", rtk_status_txt,
                                        int(1 if grade_stop_state in (STATE_HOLD, STATE_STRAIGHT) else grade_topic_value),
                                        int(grade_stop_state)])
                        _last_log_wall = noww
                    except Exception as e:
                        rospy.logwarn(f"[tracker] log write failed: {e}")

            if updated:
                fig.canvas.draw_idle()
            plt.pause(0.001)
            rate.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        _on_shutdown()

if __name__ == '__main__':
    main()
