#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
0911_ 수정한거
1. 교차로 구간 순차 인덱스로 

추가(이번 커밋)
- GRADE_UP 구간에서 /gps/GRADE_UP_ON = 1 퍼블리시, 구간 밖 0
- 정지(hold) 3초 중/후에도 GRADE_UP 구간에 있는 동안은 1 유지
- 헤딩 미추정 상태에서도 속도 램핑은 계속 진행(조향만 0), 3초 뒤 바로 출발 보장
'''

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
MIN_MOVE_FOR_HEADING = 0.05

FS_DEFAULT         = 20.0
GPS_TIMEOUT_SEC    = 1.0

# ── 언덕 정지 시간(초) ──────────────────────────────────────────────
STOP_FLAG_STAY_SEC = 3.0

# ── 속도 명령 상수 ────────────────────────────────────────────────
SPEED_FORCE_CODE = None
SPEED_CAP_CODE_DEFAULT = 10
STEP_PER_LOOP_DEFAULT  = 2
BASE_SPEED       = 5

# 시각화 옵션
ANNOTATE_WAYPOINT_INDEX = True
DRAW_WAYPOINT_CIRCLES   = True

# 퍼블리시 토픽
TOPIC_SPEED_CMD   = '/gps/speed_cmd'     # Float32
TOPIC_STEER_CMD   = '/gps/steer_cmd'     # Float32 (deg)
TOPIC_RTK_STATUS  = '/gps/rtk_status'    # String ("FIX"/"FLOAT"/"NONE")
TOPIC_WP_INDEX    = '/gps/wp_index'      # Int32 (1-based, 반경 밖=0)
TOPIC_GRADE_UP_ON = '/gps/GRADE_UP_ON'   # Int32 (0/1)  ★ 추가

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
# ──────────────────────────────────────────────────────────────────
FLAG_DEFS = [
    # 언덕(GRADE) — 해당 원에 들어오면 1회 3초 정지, 구간에 있는 동안 grade_topic=1 유지
    {'name': 'GRADE_UP', 'start': 14, 'end': 14,
     'radius_scale': 0.3, 'lookahead_scale': 0.95,
     'speed_code': None, 'speed_cap': None, 'step_per_loop': 2,
     'stop_on_hit': True, 'stop_duration_sec': None,
     'grade_topic': 1},  # ★ 추가

    # 정지 구간(최종 도착 등)
    {'name': 'STOP', 'start': 20, 'end': 21,
     'radius_scale': 0.5, 'lookahead_scale': 1.0,
     'speed_code': 0, 'speed_cap': 0, 'step_per_loop': 1, 'sequential': False},

    # 직선 가속 예시
    {'name': 'STRAIGHT_FAST', 'start': 1, 'end': 10,
     'radius_scale': 1.0, 'lookahead_scale': 1.0,
     'speed_code': 8, 'speed_cap': 10, 'step_per_loop': 2,
     'sequential': False},

    # 교차로: 순차 인덱스
    {'name': 'INTERSECTION_1', 'start': 30, 'end': 40,
     'radius_scale': 1.0, 'lookahead_scale': 1.0,
     'speed_code': None, 'speed_cap': None, 'step_per_loop': 2,
     'sequential': True}
]

def on_enter_generic(zone): rospy.loginfo(f"[flag] ENTER {zone['name']} {zone['disp_range']}")
def on_exit_generic(zone):  rospy.loginfo(f"[flag] EXIT  {zone['name']} {zone['disp_range']}")

FLAG_HOOKS = {
    'S_CURVE':        (on_enter_generic, on_exit_generic),
    'RIGHT_ANGLE':    (on_enter_generic, on_exit_generic),
    'T_JUNCTION':     (on_enter_generic, on_exit_generic),
    'STRAIGHT_FAST':  (on_enter_generic, on_exit_generic),
    'GRADE_UP':       (on_enter_generic, on_exit_generic),
    'STOP':           (on_enter_generic, on_exit_generic),
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
pub_grade = None          # ★ 추가

rtk_status_txt = "NONE"
last_fix_time  = 0.0

wp_index_active = -1

speed_cmd_current_code = 0
speed_desired_code     = 0
last_pub_speed_code    = 0.0

flag_zones = []
active_flag = None
just_entered = False
just_exited = False

log_csv_path = None
_last_log_wall = 0.0

# 홀드/정지
last_hold_zone_name = None
last_hold_zone_range = (0, 0)
final_stop_latched = False

hold_active = False
hold_until  = 0.0
hold_reason = ""
last_hold_wp_idx = 0
zone_armed = True

# 교차로 순차 인덱스
seq_active = False
seq_idx    = -1
seq_zone   = None
_prev_seq_active = False

# 구배 토픽 값(0/1)
grade_topic_value = 0     # ★ 추가

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
# 플래그 매니저
# ──────────────────────────────────────────────────────────────────
def build_flag_zones(flag_defs):
    """1-based → 0-based 변환 + zone dict 구성"""
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
            'stop_on_hit': bool(fd.get('stop_on_hit', False)),
            'stop_duration_sec': float(fd.get('stop_duration_sec', STOP_FLAG_STAY_SEC)
                                       if fd.get('stop_duration_sec', None) is not None
                                       else STOP_FLAG_STAY_SEC),
            'sequential': bool(fd.get('sequential', False)),
            'grade_topic': fd.get('grade_topic', None),   # ★ 추가
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

    # 진입: 시작 WP 원 안에 들어오면 활성
    if active_flag is None:
        for z in flag_zones:
            if in_radius_idx(x, y, z['start0'], xs, ys, base_radius):
                active_flag = z
                just_entered = True
                flag_enter(z)
                break
    # 종료: 최근접 인덱스가 end를 넘으면 비활성
    else:
        z = active_flag
        if nearest_idx > z['end0']:
            just_exited = True
            flag_exit(z)
            active_flag = None

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
    global last_pub_speed_code, hold_active, hold_until, hold_reason, final_stop_latched
    now = time.time()
    no_gps = (now - last_fix_time) > rospy.get_param('~gps_timeout_sec', GPS_TIMEOUT_SEC)

    # 홀드 종료 체크
    if hold_active and now >= hold_until:
        hold_active = False
        rospy.loginfo(f"[flag] HOLD done ({hold_reason})")

    # 최종 결정
    if no_gps or hold_active or final_stop_latched:
        v_out_int = 0
        steer_out = 0.0
    else:
        v_out_int = int(speed_cmd_current_code)
        steer_out = float(latest_filtered_angle)

    if pub_speed: pub_speed.publish(Float32(float(v_out_int)))
    if pub_steer: pub_steer.publish(Float32(steer_out))
    if pub_rtk:   pub_rtk.publish(String(rtk_status_txt))
    if pub_wpidx:
        idx_pub = (wp_index_active + 1) if (wp_index_active >= 0 and one_based) else (wp_index_active if wp_index_active >= 0 else 0)
        pub_wpidx.publish(Int32(int(idx_pub)))

    # ★ 구배 토픽 퍼블리시
    if pub_grade:
        pub_grade.publish(Int32(int(grade_topic_value)))

    last_pub_speed_code = float(v_out_int)

def _on_shutdown():
    """노드 종료 시 안전하게 모든 제어 토픽을 0(또는 안전값)으로 퍼블리시."""
    global speed_cmd_current_code, last_pub_speed_code
    rospy.loginfo("[tracker] shutdown: publishing zeros to all topics")

    try:
        # 내부 상태 갱신
        speed_cmd_current_code = 0
        last_pub_speed_code = 0.0

        # 여러번 반복해서 전송 (ROS 메시지 전달 보장 확률을 높임)
        for _ in range(6):
            try:
                if pub_speed:
                    pub_speed.publish(Float32(0.0))
                if pub_steer:
                    pub_steer.publish(Float32(0.0))
                if pub_wpidx:
                    pub_wpidx.publish(Int32(0))
                if pub_grade:
                    pub_grade.publish(Int32(0))
                if pub_rtk:
                    pub_rtk.publish(String("NONE"))
            except Exception as inner_e:
                rospy.logwarn(f"[tracker] shutdown inner publish failed: {inner_e}")
            # 짧게 쉬어서 토픽이 전파될 시간 확보
            rospy.sleep(0.05)

        rospy.loginfo("[tracker] shutdown: all zero messages published")

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
    global hold_active, hold_until, hold_reason, last_hold_wp_idx, zone_armed
    global last_hold_zone_name, last_hold_zone_range, final_stop_latched
    global seq_active, seq_zone, seq_idx, grade_topic_value

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

    # 정지 유지시간 파라미터
    global STOP_FLAG_STAY_SEC
    STOP_FLAG_STAY_SEC = float(rospy.get_param('~stop_flag_stay_sec', STOP_FLAG_STAY_SEC))

    base_speed_code_param = int(rospy.get_param('~speed_code', BASE_SPEED))
    global_cap = int(rospy.get_param('~speed_cap_code', SPEED_CAP_CODE_DEFAULT))
    step_per_loop_global = int(rospy.get_param('~step_per_loop', STEP_PER_LOOP_DEFAULT))

    # 퍼블리셔/구독자
    pub_speed = rospy.Publisher(TOPIC_SPEED_CMD, Float32, queue_size=10)
    pub_steer = rospy.Publisher(TOPIC_STEER_CMD, Float32, queue_size=10)
    pub_rtk   = rospy.Publisher(TOPIC_RTK_STATUS, String,  queue_size=10)
    pub_wpidx = rospy.Publisher(TOPIC_WP_INDEX,   Int32,   queue_size=10)
    pub_grade = rospy.Publisher(TOPIC_GRADE_UP_ON, Int32,  queue_size=10)  # ★ 추가

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
    seq_badge, = ax.plot([], [], linestyle='-', color='orange', label='Sequential mode')
    seq_badge.set_visible(False)
    ax.legend()

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

    # 속도 초기화
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

                # 최근접/룩어헤드
                nearest_idx = find_nearest_index(x, y, spaced_x, spaced_y, nearest_idx_prev, 80, 15)
                nearest_idx_prev = nearest_idx

                # 플래그 상태
                eff_radius, look_scl, z_active, entered, exited = flag_update_state(
                    x, y, nearest_idx, spaced_x, spaced_y, TARGET_RADIUS_END
                )

                # 순차모드 진입/이탈
                if entered and (z_active is not None) and z_active.get('sequential', False):
                    seq_active = True
                    seq_zone   = z_active
                    seq_idx    = max(z_active['start0'], nearest_idx)
                    rospy.loginfo(f"[seq] ENTER {z_active['name']} {z_active['disp_range']} (start idx={seq_idx+1})")
                if exited and seq_active:
                    rospy.loginfo("[seq] EXIT sequential mode")
                    seq_active = False
                    seq_zone   = None

                # 룩어헤드
                Ld_target_base = max(LOOKAHEAD_MIN, min(LOOKAHEAD_MAX, LOOKAHEAD_MIN + LOOKAHEAD_K * speed_mps))
                Ld_target = max(LOOKAHEAD_MIN, min(LOOKAHEAD_MAX, Ld_target_base * look_scl))
                Ld = prev_Ld + 0.2 * (Ld_target - prev_Ld)
                prev_Ld = Ld

                # 타깃 인덱스
                tgt_idx = target_index_from_lookahead(nearest_idx, Ld, WAYPOINT_SPACING, len(spaced_x))
                tx, ty = spaced_x[tgt_idx], spaced_y[tgt_idx]
                target_line.set_data([x, tx], [y, ty])
                target_pt.set_data([tx], [ty])

                # 반경 내 인덱스
                wp_index_active = nearest_in_radius_index(x, y, spaced_x, spaced_y, eff_radius)
                wp_display = (wp_index_active + 1) if (wp_index_active >= 0) else 0

                # 순차모드 동작
                if seq_active and (seq_zone is not None):
                    if wp_index_active == seq_idx:
                        seq_idx = min(seq_idx + 1, seq_zone['end0'])
                    tgt_idx = seq_idx
                    tx, ty = spaced_x[tgt_idx], spaced_y[tgt_idx]
                    target_line.set_data([x, tx], [y, ty])
                    target_pt.set_data([tx], [ty])
                    seq_badge.set_visible(True)
                    nearest_idx_prev = tgt_idx
                    if nearest_idx > seq_zone['end0'] or tgt_idx >= seq_zone['end0']:
                        rospy.loginfo("[seq] DONE range → exit sequential mode")
                        seq_active = False
                        seq_zone   = None
                else:
                    seq_badge.set_visible(False)

                # ── stop_on_hit 트리거 (언덕 포함) ──
                stop_zone_now = None
                if wp_display > 0:
                    for z in flag_zones:
                        if z.get('stop_on_hit', False):
                            start1, end1 = (z['start0']+1), (z['end0']+1)
                            if start1 <= wp_display <= end1:
                                stop_zone_now = z
                                break
                if stop_zone_now:
                    if zone_armed and not hold_active:
                        dur = float(stop_zone_now.get('stop_duration_sec', STOP_FLAG_STAY_SEC))
                        hold_active = True
                        hold_until  = time.time() + dur
                        hold_reason = stop_zone_now['name']
                        last_hold_zone_name  = stop_zone_now['name']
                        last_hold_zone_range = (stop_zone_now['start0']+1, stop_zone_now['end0']+1)
                        last_hold_wp_idx = wp_display
                        zone_armed = False
                        speed_desired_code = 0
                        speed_cmd_current_code = 0
                        rospy.loginfo(f"[flag] HOLD start ({hold_reason}) at WP {last_hold_wp_idx} for {dur:.1f}s")
                else:
                    if last_hold_zone_name is None:
                        zone_armed = True
                    else:
                        s1, e1 = last_hold_zone_range
                        if not (s1 <= wp_display <= e1):
                            zone_armed = True
                            last_hold_zone_name = None

                # ★ 구배 토픽 값 결정: 활성 플래그가 값 제공 시 그 값, 없으면 0
                if z_active is not None and (z_active.get('grade_topic') is not None):
                    try:
                        grade_topic_value = 1 if int(z_active['grade_topic']) != 0 else 0
                    except Exception:
                        grade_topic_value = 0
                else:
                    grade_topic_value = 0
                # 정지 중이고 원인이 GRADE_UP이면 확실히 1 유지
                if hold_active and (hold_reason == 'GRADE_UP'):
                    grade_topic_value = 1

                # 헤딩 추정
                heading_vec = None
                for k in range(2, min(len(pos_buf), 5)+1):
                    t0, x0, y0 = pos_buf[-k]
                    if math.hypot(x - x0, y - y0) >= MIN_MOVE_FOR_HEADING:
                        heading_vec = (x - x0, y - y0)
                        break
                if heading_vec is not None:
                    last_heading_vec = heading_vec

                # 헤딩이 아직 없으면 조향만 0으로 두고(출발은 해야 하므로) 속도 램핑은 계속 진행
                if 'last_heading_vec' not in locals() or last_heading_vec is None:
                    latest_filtered_angle = 0.0
                else:
                    # 조향 계산 + 동적 LPF
                    target_vec = (tx - x, ty - y)
                    raw_angle = angle_between(last_heading_vec, target_vec)
                    base_fc = LPF_FC_HZ
                    lpf.fc = min(2.0, base_fc + 0.5) if abs(raw_angle) > 10 else base_fc
                    filt_angle = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, lpf.update(raw_angle, tsec)))
                    latest_filtered_angle = SIGN_CONVENTION * filt_angle

                # ── 속도 코드 산출 + 램핑 ──
                if z_active is not None and (z_active.get('speed_code') is not None):
                    desired = int(z_active['speed_code'])
                    cap = int(z_active['speed_cap']) if z_active.get('speed_cap') is not None else global_cap
                    step_this_loop = int(max(1, z_active.get('step_per_loop', step_per_loop_global)))
                    active_name = z_active['name']
                    if active_name == "STOP":
                        base_speed_code_param = 0
                        try:
                            final_stop_latched = True
                            speed_desired_code = 0
                            speed_cmd_current_code = 0
                            base_speed_code_param = 0
                        except Exception:
                            pass
                        rospy.loginfo("[STOP] Final stop zone → base_speed=0")
                else:
                    desired = int(base_speed_code_param)
                    cap = global_cap
                    step_this_loop = step_per_loop_global
                    active_name = None

                if hold_active:
                    desired = 0
                    cap = 0

                desired = max(0, min(cap, desired))
                speed_desired_code = desired

                if speed_cmd_current_code < speed_desired_code:
                    speed_cmd_current_code = min(speed_desired_code, speed_cmd_current_code + step_this_loop)
                elif speed_cmd_current_code > speed_desired_code:
                    speed_cmd_current_code = max(speed_desired_code, speed_cmd_current_code - step_this_loop)
                speed_cmd_current_code = max(0, min(max(cap, 0), speed_cmd_current_code))

                # 시각화
                live_line.set_data([p[1] for p in pos_buf], [p[2] for p in pos_buf])
                current_pt.set_data([x], [y])

                ax_info.clear(); ax_info.axis('off')
                heading_deg = (math.degrees(math.atan2(last_heading_vec[1], last_heading_vec[0]))
                               if ('last_heading_vec' in locals() and last_heading_vec is not None) else float('nan'))
                tgt_display = int(tgt_idx + 1)
                hold_txt = f"HOLD {hold_reason} ({max(0.0, hold_until-time.time()):.1f}s)" if hold_active else ""
                status = [
                    f"TGT={tgt_display}",
                    f"WP(in-radius)={wp_display}",
                    f"Speed(code): cur={speed_cmd_current_code} → des={speed_desired_code}",
                    f"Meas v={speed_mps:.2f} m/s",
                    f"Steer={latest_filtered_angle:+.1f}°",
                    f"Heading={heading_deg:.1f}°",
                    f"RTK={rtk_status_txt}",
                    f"GRADE_UP_ON={grade_topic_value}",     # ★ 표시
                    ("SEQ=ON" if seq_active else "SEQ=OFF"),
                    (hold_txt if hold_txt else "")
                ]
                ax_info.text(0.02, 0.5, " | ".join([s for s in status if s]), fontsize=11, va='center')

                hud_text.set_text(
                    f"TGT={tgt_display} | pub_v={last_pub_speed_code:.1f} "
                    f"| meas_v={speed_mps:.2f} m/s | steer={latest_filtered_angle:+.1f}° "
                    f"| WP={wp_display}"
                    + (f" | {hold_txt}" if hold_txt else "")
                    + (" | FINAL_STOP" if final_stop_latched else "")
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
                                            'rtk','hold','hold_wp','grade_up_on'])
                            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                        f"{lat:.7f}", f"{lon:.7f}", f"{x:.3f}", f"{y:.3f}",
                                        wp_display, tgt_display, f"{tx:.3f}", f"{ty:.3f}",
                                        f"{latest_filtered_angle:.2f}", f"{speed_mps:.2f}",
                                        int(speed_cmd_current_code), int(speed_desired_code),
                                        f"{Ld:.2f}", rtk_status_txt,
                                        "1" if hold_active else "0", last_hold_wp_idx if hold_active else 0,
                                        int(grade_topic_value)])
                        _last_log_wall = noww
                    except Exception as e:
                        rospy.logwarn(f"[tracker] log write failed: {e}")

                rospy.loginfo_throttle(
                    0.5,
                    f"TGT={tgt_display} | WP(in-radius)={wp_display} | "
                    f"meas_v={speed_mps:.2f} m/s | code(cur→des)={speed_cmd_current_code}->{speed_desired_code} | "
                    f"Ld={Ld:.2f} | steer={latest_filtered_angle:+.2f} deg | RTK={rtk_status_txt} "
                    f"| GRADE_UP_ON={grade_topic_value}"
                    + (f" | HOLD {hold_reason} ({max(0.0, hold_until-time.time()):.1f}s)" if hold_active else "")
                )

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
