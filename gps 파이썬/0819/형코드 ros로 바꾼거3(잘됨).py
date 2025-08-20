#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waypoint_tracker_ros_topics.py (ROS1 Noetic, Ubuntu)

- ROS 토픽 구독 기반:
    위치: sensor_msgs/NavSatFix (기본 /ublox/fix)
    RTK 상태(옵션): ublox_msgs/NavPVT, NavRELPOSNED 플래그 해석
- 웨이포인트 순차 추종 + 시각화(Matplotlib: plt.ion 루프, animation 미사용)
- 퍼블리시: /vehicle/speed_cmd (Float32, m/s), /vehicle/steer_cmd (Float32, deg), /rtk/status (String)
- 기본 경로:
    /home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/left_lane.csv
    /home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/waypoint_log_YYYYMMDD_HHMMSS.csv
"""

import os
import csv
import math
import time
import threading

import numpy as np
import pandas as pd

import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import rospy
import rospkg
from std_msgs.msg import Float32, String
from sensor_msgs.msg import NavSatFix

# ── ublox_msgs 임포트 가드(있으면 RTK 상태 사용) ─────────
_HAVE_RELPOSNED = False
_HAVE_NAVPVT = False
try:
    from ublox_msgs.msg import NavRELPOSNED9 as NavRELPOSNED
    _HAVE_RELPOSNED = True
except Exception:
    try:
        from ublox_msgs.msg import NavRELPOSNED
        _HAVE_RELPOSNED = True
    except Exception:
        _HAVE_RELPOSNED = False

try:
    from ublox_msgs.msg import NavPVT
    _HAVE_NAVPVT = True
except Exception:
    _HAVE_NAVPVT = False

# ── 전역 기본값 ─────────────────────────────────────
TARGET_RADIUS_DEFAULT = 1.5
MIN_WAYPOINT_DISTANCE_DEFAULT = 0.9
FC_DEFAULT = 2.0
FS_DEFAULT = 10.0
GPS_OUTLIER_THRESHOLD_DEFAULT = 1.0
STEER_LIMIT_DEG_DEFAULT = 20.0
CONST_SPEED_DEFAULT = 1.0

# 패키지/config 기본 경로 계산
def _default_paths():
    try:
        pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
    cfg = os.path.join(pkg_path, 'config')
    wp = os.path.join(cfg, 'left_lane.csv')
    log = os.path.join(cfg, f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    return cfg, wp, log

CFG_DIR_DEFAULT, WAYPOINT_CSV_DEFAULT, LOG_CSV_DEFAULT = _default_paths()

# ── 런타임 상태 ─────────────────────────────────────
params = {}
pub_speed = None
pub_steer = None
pub_rtk   = None

current_x, current_y = [], []
heading_vectors = []
waypoints_x = None
waypoints_y = None
waypoint_index = 0

alpha = 0.56                 # LPF 계수(런타임 계산)
_filtered_steering = 0.0

_prev_raw_x = None
_prev_raw_y = None
_prev_f_x = None
_prev_f_y = None

rtk_status_txt = "NONE"
_state_lock = threading.Lock()

# ── 유틸 ─────────────────────────────────────────────
def dm_to_dec(dm, direction):
    try:
        d = int(float(dm) / 100)
        m = float(dm) - d * 100
        dec = d + m / 60.0
        return -dec if direction in ['S', 'W'] else dec
    except Exception:
        return None

def latlon_to_meters(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def distance_m(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def calculate_steering_angle(v1, v2):
    v1 = np.asarray(v1, dtype=float); v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    TH = 0.05
    if n1 < TH or n2 < TH:
        return 0.0
    dot = float(np.dot(v1, v2))
    c = max(min(dot / (n1 * n2), 1.0), -1.0)
    ang = math.degrees(math.acos(c))
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    if cross < 0: ang = -ang
    ang = max(min(ang / 1.3, 25.0), -25.0)
    if abs(ang) > 20.0 and (n1 < TH or n2 < TH):
        return 0.0
    return ang

def apply_low_pass_filter(current):
    global _filtered_steering, alpha
    _filtered_steering = (1 - alpha) * _filtered_steering + alpha * current
    return _filtered_steering * -1.0

def filter_gps_signal(x, y):
    global _prev_raw_x, _prev_raw_y, _prev_f_x, _prev_f_y, alpha
    if _prev_raw_x is not None and _prev_raw_y is not None:
        if distance_m(_prev_raw_x, _prev_raw_y, x, y) > float(params['gps_outlier_th']):
            x, y = _prev_raw_x, _prev_raw_y
        else:
            _prev_raw_x, _prev_raw_y = x, y
    else:
        _prev_raw_x, _prev_raw_y = x, y
    if _prev_f_x is None or _prev_f_y is None:
        _prev_f_x, _prev_f_y = x, y
    fx = (1 - alpha) * _prev_f_x + alpha * x
    fy = (1 - alpha) * _prev_f_y + alpha * y
    _prev_f_x, _prev_f_y = fx, fy
    return fx, fy

def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

# ── 웨이포인트 처리 ─────────────────────────────────
def load_waypoints(path_csv, min_wp_dist):
    df = pd.read_csv(path_csv)
    coords = [latlon_to_meters(row['Lat'], row['Lon']) for _, row in df.iterrows()]
    if len(coords) < 1:
        raise RuntimeError("waypoints csv empty")
    fx = [float(coords[0][0])]; fy = [float(coords[0][1])]
    for xi, yi in coords[1:]:
        if distance_m(fx[-1], fy[-1], xi, yi) >= min_wp_dist:
            fx.append(float(xi)); fy.append(float(yi))
    return np.array(fx), np.array(fy)

# ── ROS 퍼블리셔 ────────────────────────────────────
def publish_speed(speed):
    if pub_speed: pub_speed.publish(Float32(data=float(speed)))

def publish_steer_deg(steer_deg):
    sd = clamp(float(steer_deg), -float(params['steer_limit_deg']), float(params['steer_limit_deg']))
    if pub_steer: pub_steer.publish(Float32(data=sd))

def publish_rtk(txt):
    if pub_rtk: pub_rtk.publish(String(data=str(txt)))

# ── ROS 구독 콜백 ───────────────────────────────────
def _cb_fix(msg: 'NavSatFix'):
    """GPS 위치 갱신 (NavSatFix)"""
    if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
        return
    x, y = latlon_to_meters(msg.latitude, msg.longitude)
    fx, fy = filter_gps_signal(x, y)
    with _state_lock:
        if (len(current_x) == 0 or
            distance_m(current_x[-1], current_y[-1], fx, fy) >= 0.001):
            current_x.append(fx); current_y.append(fy)

def _cb_relpos(msg):
    """ublox NavRELPOSNED flags → RTK 상태"""
    global rtk_status_txt
    try:
        mask  = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_MASK'))
        fixed = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_FIXED'))
        flt   = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_FLOAT'))
        bits = int(msg.flags) & mask
        rtk_status_txt = "FIX" if bits == fixed else ("FLOAT" if bits == flt else "NONE")
        publish_rtk(rtk_status_txt)
    except Exception:
        pass

def _cb_navpvt(msg):
    """ublox NavPVT flags → RTK 상태"""
    global rtk_status_txt
    try:
        mask  = int(getattr(NavPVT, 'FLAGS_CARRIER_PHASE_MASK'))
        fixed = int(getattr(NavPVT, 'CARRIER_PHASE_FIXED'))
        flt   = int(getattr(NavPVT, 'CARRIER_PHASE_FLOAT'))
        phase = int(msg.flags) & mask
        rtk_status_txt = "FIX" if phase == fixed else ("FLOAT" if phase == flt else "NONE")
        publish_rtk(rtk_status_txt)
    except Exception:
        pass

# ── 시각화(애니메이션 없이 주기 갱신) ─────────────────
def update_plot_once(ax):
    global waypoint_index
    ax.clear()

    with _state_lock:
        cx = list(current_x); cy = list(current_y)

    # 윈도우 영역
    window_size = 20
    start_index = (waypoint_index // window_size) * window_size
    end_index = min(start_index + window_size, len(waypoints_x))

    # 웨이포인트 + 도착원
    ax.scatter(waypoints_x[start_index:end_index], waypoints_y[start_index:end_index],
               label='Waypoints', s=10)
    for i in range(start_index, end_index):
        c = plt.Circle((waypoints_x[i], waypoints_y[i]), float(params['target_radius']),
                       fill=False, linestyle='--')
        ax.add_patch(c)
        ax.text(waypoints_x[i], waypoints_y[i], str(i + 1), fontsize=8, ha='center')

    smooth = 0.0
    if cx and cy:
        # 현재위치 / 타겟
        ax.scatter(cx[-1], cy[-1], color='red', s=50, label='Current')
        tx, ty = waypoints_x[waypoint_index], waypoints_y[waypoint_index]
        ax.arrow(cx[-1], cy[-1], tx - cx[-1], ty - cy[-1],
                 head_width=0.5, head_length=0.5, color='green')

        # 궤적
        for i in range(1, len(cx)):
            dx = cx[i] - cx[i-1]; dy = cy[i] - cy[i-1]
            ax.arrow(cx[i-1], cy[i-1], dx, dy,
                     head_width=0.2, head_length=0.2, color='orange')

        # 조향 계산 + LPF
        if len(cx) > 1:
            dx = cx[-1] - cx[-2]; dy = cy[-1] - cy[-2]
            heading_vectors.append((dx, dy))
            target_vec = (tx - cx[-1], ty - cy[-1])
            angle = calculate_steering_angle((dx, dy), target_vec)
            smooth = apply_low_pass_filter(angle)
        else:
            target_vec = ('', '')

        # 로깅
        if params['log_csv']:
            try:
                new = not os.path.exists(params['log_csv'])
                os.makedirs(os.path.dirname(params['log_csv']), exist_ok=True)
                with open(params['log_csv'], 'a', newline='') as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(['current_x','current_y','prev_x','prev_y',
                                    'target_vector_x','target_vector_y','waypoint_x','waypoint_y','angle'])
                    if len(cx) > 1:
                        w.writerow([cx[-1], cy[-1], cx[-2], cy[-2],
                                    target_vec[0], target_vec[1], tx, ty, smooth])
                    else:
                        w.writerow([cx[-1], cy[-1], '', '', '', '', tx, ty, smooth])
            except Exception as e:
                rospy.logwarn(f"[tracker_topics] log write failed: {e}")

        # 퍼블리시 (속도는 상수)
        publish_speed(params['const_speed'])
        publish_steer_deg(smooth)
        publish_rtk(rtk_status_txt)

        # 도착 반경 → 다음 인덱스
        if len(cx) > 1 and distance_m(cx[-1], cy[-1], tx, ty) < float(params['target_radius']):
            if waypoint_index < len(waypoints_x) - 1:
                waypoint_index += 1

    ax.set_title(f"ROS GPS Tracker  Steering: {smooth:.2f}°  RTK: {rtk_status_txt}")
    ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)')
    ax.axis('equal'); ax.grid(True); ax.legend(loc='upper right')

# ── 메인 ────────────────────────────────────────────
def main():
    global pub_speed, pub_steer, pub_rtk, waypoints_x, waypoints_y, alpha, params

    rospy.init_node('waypoint_tracker_ros_topics', anonymous=False)

    # 파라미터(기본: config 폴더, ublox 네임스페이스)
    ublox_ns = rospy.get_param('~ublox_ns', '/ublox')
    params = {
        'fix_topic':        rospy.get_param('~fix_topic',    ublox_ns + '/fix'),
        'navpvt_topic':     rospy.get_param('~navpvt_topic', ublox_ns + '/navpvt'),
        'relpos_topic':     rospy.get_param('~relpos_topic', ublox_ns + '/navrelposned'),
        'waypoint_csv':     rospy.get_param('~waypoint_csv', WAYPOINT_CSV_DEFAULT),
        'target_radius':    float(rospy.get_param('~target_radius', TARGET_RADIUS_DEFAULT)),
        'min_wp_distance':  float(rospy.get_param('~min_wp_distance', MIN_WAYPOINT_DISTANCE_DEFAULT)),
        'fc':               float(rospy.get_param('~fc', FC_DEFAULT)),
        'fs':               float(rospy.get_param('~fs', FS_DEFAULT)),
        'gps_outlier_th':   float(rospy.get_param('~gps_outlier_th', GPS_OUTLIER_THRESHOLD_DEFAULT)),
        'steer_limit_deg':  float(rospy.get_param('~steer_limit_deg', STEER_LIMIT_DEG_DEFAULT)),
        'const_speed':      float(rospy.get_param('~const_speed', CONST_SPEED_DEFAULT)),
        'log_csv':          rospy.get_param('~log_csv', LOG_CSV_DEFAULT),
    }
    # LPF 계수
    alpha = (2 * math.pi * params['fc']) / (2 * math.pi * params['fc'] + params['fs'])

    # 퍼블리셔
    pub_speed = rospy.Publisher('/vehicle/speed_cmd', Float32, queue_size=10)
    pub_steer = rospy.Publisher('/vehicle/steer_cmd',  Float32, queue_size=10)
    pub_rtk   = rospy.Publisher('/rtk/status',         String,  queue_size=10)

    # 웨이포인트 로드
    try:
        os.makedirs(os.path.dirname(params['waypoint_csv']), exist_ok=True)
        waypoints_x, waypoints_y = load_waypoints(params['waypoint_csv'], params['min_wp_distance'])
    except Exception as e:
        rospy.logerr(f"[tracker_topics] failed to load waypoints: {e}")
        return

    # 구독자
    rospy.Subscriber(params['fix_topic'], NavSatFix, _cb_fix, queue_size=100)
    if _HAVE_RELPOSNED:
        rospy.Subscriber(params['relpos_topic'], NavRELPOSNED, _cb_relpos, queue_size=50)
    if _HAVE_NAVPVT:
        rospy.Subscriber(params['navpvt_topic'], NavPVT, _cb_navpvt, queue_size=50)

    rospy.loginfo("[tracker_topics] listening: fix=%s relpos=%s(%s) navpvt=%s(%s)",
                  params['fix_topic'],
                  params['relpos_topic'], 'ON' if _HAVE_RELPOSNED else 'OFF',
                  params['navpvt_topic'], 'ON' if _HAVE_NAVPVT else 'OFF')

    # 시각화 루프 (animation 미사용)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dt = 1.0 / max(1.0, float(params['fs']))

    try:
        while not rospy.is_shutdown():
            update_plot_once(ax)
            plt.pause(dt)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
