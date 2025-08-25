ㅇ
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waypoint_tracker_topics.py  (ROS1 Noetic, Ubuntu)

- 직렬(NMEA) 제거: ROS 토픽 구독만 사용
  · /ublox/fix (sensor_msgs/NavSatFix)
  · (옵션) /ublox/navrelposned 또는 /ublox/navpvt 로 RTK 상태 판정
- 웨이포인트 순차 추종 + 시각화(Matplotlib: plt.ion 루프)
  · Heading/Steering 화살표
  · 이동 경로(회색 얇은 선)
  · Target star + Current→Target 점선
  · Info box(좌표/거리/헤딩/조향)
- 퍼블리시: /vehicle/speed_cmd (Float32, m/s), /vehicle/steer_cmd (Float32, deg), /rtk/status (String)
- 기본 경로:
    ~/catkin_ws/src/rtk_waypoint_tracker/config/left_lane.csv  (웨이포인트)
    ~/catkin_ws/src/rtk_waypoint_tracker/config/waypoint_log_YYYYMMDD_HHMMSS.csv (로그)
  ※ 패키지 경로 자동 탐지(rospkg). 파라미터로 언제든 override 가능.

- Matplotlib은 반드시 plt.ion() 루프만 사용 (animation 사용 안함)
"""

import os
import csv
import math
import time
import threading
from collections import deque  # ★ 헤딩 안정화 히스토리

import numpy as np
import pandas as pd

import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

import rospy
import rospkg
from std_msgs.msg import Float32, String
from sensor_msgs.msg import NavSatFix

# ── ublox_msgs (옵션) ─────────────────────────────────
_HAVE_RELPOSNED = False   # NavRELPOSNED 메시지 타입 사용 가능 여부
_HAVE_NAVPVT    = False   # NavPVT 메시지 타입 사용 가능 여부

# ① 최신 버전의 NavRELPOSNED9 메시지 타입 확인
try:
    from ublox_msgs.msg import NavRELPOSNED9 as NavRELPOSNED
    _HAVE_RELPOSNED = True
except Exception:
    # ② 구버전의 NavRELPOSNED 메시지 타입 확인
    try:
        from ublox_msgs.msg import NavRELPOSNED
        _HAVE_RELPOSNED = True
    except Exception:
        # 둘 다 없으면 False 유지
        _HAVE_RELPOSNED = False

# ③ NavPVT 메시지 타입 확인
try:
    from ublox_msgs.msg import NavPVT
    _HAVE_NAVPVT = True
except Exception:
    _HAVE_NAVPVT = False

# ── 전역 기본값 ─────────────────────────────────────
TARGET_RADIUS_DEFAULT = 1.5        # [m] 웨이포인트 도착 반경
MIN_WAYPOINT_DISTANCE_DEFAULT = 0.9 # [m] 웨이포인트 최소 간격 (리샘플링 기준)
FC_DEFAULT = 2.0                   # [Hz] 로패스필터 컷오프 주파수
FS_DEFAULT = 20.0                  # [Hz] 샘플링 주파수 (ROS 루프/plot 주기)
GPS_OUTLIER_THRESHOLD_DEFAULT = 1.0 # [m] GPS 이상치(점프) 허용 거리
STEER_LIMIT_DEG_DEFAULT = 20.0     # [deg] 조향각 제한 (최대 허용치)
CONST_SPEED_DEFAULT = 1.0          # [m/s] 차량 목표 속도 (고정값)
'''
FC_DEFAULT = 2.0 → 2 Hz 저역통과필터(LPF) 컷오프 주파수
→ 초당 2회 이상의 빠른 변동(고주파 잡음)은 걸러지고, 그보다 느린 움직임은 통과.

FS_DEFAULT = 20.0 → 20 Hz 샘플링 주파수
→ 코드 루프(rospy.Rate)와 필터 계산 주기가 초당 20번(= 50ms 간격) 실행.
즉, 20Hz로 GNSS를 처리하며 2Hz 컷오프 필터를 적용해 위치/조향 계산을 안정화.
'''

# 패키지/config 기본 경로 계산
def _default_paths():
    try:
        pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
    cfg = os.path.join(pkg_path, 'config')
    wp  = os.path.join(cfg, 'left_lane.csv')
    log = os.path.join(cfg, f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    return cfg, wp, log

# 전역 변수로 기본 경로 저장
CFG_DIR_DEFAULT, WAYPOINT_CSV_DEFAULT, LOG_CSV_DEFAULT = _default_paths()

# ── 런타임 상태 ─────────────────────────────────────
params = {}
pub_speed = None
pub_steer = None
pub_rtk   = None

current_x, current_y = [], []
waypoints_x = None
waypoints_y = None
waypoint_index = 0

# LPF
alpha = 0.56
_filtered_steering = 0.0

# GPS 필터링 상태
_prev_raw_x = None
_prev_raw_y = None
_prev_f_x = None
_prev_f_y = None

# 헤딩 안정화
_last_lat = None
_last_lon = None
_last_fix_heading_rad = None               # 안정화된 전역 헤딩
heading_hist = deque(maxlen=5)             # (heading_rad, weight=segment_len_m)
rtk_status_txt = "NONE"
_state_lock = threading.Lock()
_last_log_t = 0.0

# ── 유틸 ─────────────────────────────────────────────
def dm_to_dec(dm, direction):
    try:
        d = int(float(dm) / 100)
        m = float(dm) - d * 100
        dec = d + m / 60.0
        return -dec if direction in ['S', 'W'] else dec
    except Exception:
        return None

# 위경도를 Web Mercator 투영 좌표계로 변환 , 단위는 미터
def latlon_to_meters(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def distance_m(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def _ang_wrap(rad):
    return (rad + math.pi) % (2*math.pi) - math.pi

def _ang_diff(a, b):
    return _ang_wrap(a - b)

def _circ_mean_from_hist(hist):
    if not hist:
        return None
    s = 0.0; c = 0.0
    for h, w in hist:
        s += w * math.sin(h)
        c += w * math.cos(h)
    if s == 0.0 and c == 0.0:
        return None
    return math.atan2(s, c)

# 두 벡터의 각도를 계산해서 조향각(deg) 산출
def calculate_steering_angle(v1, v2):
    v1 = np.asarray(v1, dtype=float); v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    TH = 0.05  # 5cm: 거의 정지이면 0°

    if n1 < TH or n2 < TH:
        return 0.0

    dot = float(np.dot(v1, v2))
    c = max(min(dot / (n1 * n2), 1.0), -1.0)
    ang = math.degrees(math.acos(c))

    # 부호(좌/우) 결정
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    if cross < 0:
        ang = -ang

    # 조향각 제한 및 감쇠 (민감도 완화 후 ±20° 제한)
    ang = max(min(ang / 1.3, 20.0), -20.0)
    return ang

def apply_low_pass_filter(current):
    global _filtered_steering, alpha
    _filtered_steering = (1 - alpha) * _filtered_steering + alpha * current
    return _filtered_steering * -1.0  # 부호관례 맞춤

def filter_gps_signal(x, y):
    global _prev_raw_x, _prev_raw_y, _prev_f_x, _prev_f_y, alpha

    # Outlier 제거
    if _prev_raw_x is not None and _prev_raw_y is not None:
        if distance_m(_prev_raw_x, _prev_raw_y, x, y) > float(params['gps_outlier_th']):
            x, y = _prev_raw_x, _prev_raw_y
        else:
            _prev_raw_x, _prev_raw_y = x, y
    else:
        _prev_raw_x, _prev_raw_y = x, y

    # LPF
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
    if pub_speed:
        pub_speed.publish(Float32(data=float(speed)))

def publish_steer_deg(steer_deg):
    sd = clamp(float(steer_deg),
               -float(params['steer_limit_deg']),
                float(params['steer_limit_deg']))
    if pub_steer:
        pub_steer.publish(Float32(data=sd))

def publish_rtk(txt):
    if pub_rtk:
        pub_rtk.publish(String(data=str(txt)))

# ── ROS 콜백 ────────────────────────────────────────
def _cb_fix(msg: NavSatFix):
    global _last_lat, _last_lon, _last_fix_heading_rad, heading_hist

    if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
        return

    _last_lat, _last_lon = float(msg.latitude), float(msg.longitude)
    x, y = latlon_to_meters(_last_lat, _last_lon)
    fx, fy = filter_gps_signal(x, y)

    with _state_lock:
        # 이동 구간 벡터
        if current_x and current_y:
            dx = fx - current_x[-1]
            dy = fy - current_y[-1]
            seg_len = math.hypot(dx, dy)

            # 1) 이동량이 충분할 때만 새 구간 헤딩 계산
            if seg_len > float(params['heading_min_move_m']):
                seg_heading = math.atan2(dy, dx)

                # 2) 안정 헤딩 대비 작은 이동에서의 급격한 점프는 거부
                stable_before = _circ_mean_from_hist(heading_hist)
                accept = True
                if stable_before is not None:
                    jump_deg = abs(math.degrees(_ang_diff(seg_heading, stable_before)))
                    if (jump_deg > float(params['heading_max_jump_deg'])
                        and seg_len < 3.0 * float(params['heading_min_move_m'])):
                        accept = False

                if accept:
                    # 거리 가중 원형평균
                    heading_hist.append((seg_heading, seg_len))

                stable_after = _circ_mean_from_hist(heading_hist)
                if stable_after is not None:
                    _last_fix_heading_rad = stable_after
                else:
                    _last_fix_heading_rad = seg_heading
            # 이동량이 작으면 헤딩 갱신 금지(유지)

        # 좌표 기록은 항상 갱신
        current_x.append(fx)
        current_y.append(fy)

def _cb_relpos(msg):
    global rtk_status_txt
    try:
        mask  = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_MASK'))
        fixed = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_FIXED'))
        flt   = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_FLOAT'))
        bits = int(msg.flags) & mask
        rtk_status_txt = "FIX" if bits == fixed else ("FLOAT" if bits == flt else "NONE")
        publish_rtk(rtk_status_txt)
    except Exception:
        rtk_status_txt = "NONE"

def _cb_navpvt(msg):
    global rtk_status_txt
    try:
        mask  = int(getattr(NavPVT, 'FLAGS_CARRIER_PHASE_MASK'))
        fixed = int(getattr(NavPVT, 'CARRIER_PHASE_FIXED'))
        flt   = int(getattr(NavPVT, 'CARRIER_PHASE_FLOAT'))
        phase = int(msg.flags) & mask
        rtk_status_txt = "FIX" if phase == fixed else ("FLOAT" if phase == flt else "NONE")
        publish_rtk(rtk_status_txt)
    except Exception:
        rtk_status_txt = "NONE"

# ── 시각화(애니메이션 없이 주기 갱신) ─────────────────
def update_plot_once(ax):
    global waypoint_index, _last_log_t
    ax.clear()

    with _state_lock:
        cx = list(current_x); cy = list(current_y)
        heading_rad = _last_fix_heading_rad  # ★ 안정화된 전역 헤딩 사용

    if len(cx) >= 2:
        ax.plot(cx, cy, '-', c='0.6', lw=1.0, label='Route')

    window_size = 20
    start_index = (waypoint_index // window_size) * window_size
    end_index = min(start_index + window_size, len(waypoints_x))

    ax.scatter(waypoints_x[start_index:end_index], waypoints_y[start_index:end_index],
               color='blue', s=10, label='Waypoints')
    for i in range(start_index, end_index):
        c = Circle((waypoints_x[i], waypoints_y[i]), float(params['target_radius']),
                   fill=False, linestyle='--', edgecolor='tab:blue', alpha=0.3)
        ax.add_patch(c)
        ax.text(waypoints_x[i], waypoints_y[i], str(i + 1), fontsize=8, ha='center')

    smooth_deg = 0.0
    info_lines = []

    if cx and cy:
        ax.scatter(cx[-1], cy[-1], color='red', s=50, label='Current')
        tx, ty = waypoints_x[waypoint_index], waypoints_y[waypoint_index]
        ax.plot([cx[-1], tx], [cy[-1], ty], '--', c='cyan', lw=1.0, label='Target Line')
        ax.plot(tx, ty, '*', c='magenta', ms=12, label='Target')

        if len(cx) > 1:
            target_vec = (tx - cx[-1], ty - cy[-1])
            move_vec   = (cx[-1] - cx[-2], cy[-1] - cy[-2])
            angle = calculate_steering_angle(move_vec, target_vec)
            smooth_deg = apply_low_pass_filter(angle)
            # ★ LPF 이후 최종 클램프: 그래프/CSV/퍼블리시 일관성 (기본 ±20°)
            smooth_deg = clamp(
                smooth_deg,
                -float(params['steer_limit_deg']),
                 float(params['steer_limit_deg'])
            )
        else:
            target_vec = ('', '')

        # 헤딩 & 조향 화살표
        L = 2.0
        if heading_rad is not None:
            hx, hy = cx[-1] + L*math.cos(heading_rad), cy[-1] + L*math.sin(heading_rad)
            ax.add_patch(FancyArrowPatch((cx[-1],cy[-1]), (hx,hy),
                                         color='tab:blue', lw=2, arrowstyle='-|>', mutation_scale=15,
                                         label='Heading'))
            steer_rad = math.radians(smooth_deg)
            sx, sy = cx[-1] + L*math.cos(heading_rad + steer_rad), cy[-1] + L*math.sin(heading_rad + steer_rad)
            ax.add_patch(FancyArrowPatch((cx[-1],cy[-1]), (sx,sy),
                                         color='red', lw=2, alpha=0.9, arrowstyle='-|>', mutation_scale=15,
                                         label='Steering'))

        # CSV 로깅
        if params['log_csv']:
            try:
                new = not os.path.exists(params['log_csv'])
                os.makedirs(os.path.dirname(params['log_csv']), exist_ok=True)
                with open(params['log_csv'], 'a', newline='') as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow([
                            'current_x','current_y','prev_x','prev_y',
                            'target_vector_x','target_vector_y',
                            'waypoint_x','waypoint_y',
                            'steer_deg','heading_deg',
                            'speed','dist_to_target','time','rtk_status'
                        ])
                    dist_to_target = distance_m(cx[-1], cy[-1], tx, ty)
                    heading_deg = math.degrees(heading_rad) if heading_rad is not None else ''
                    log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    if len(cx) > 1:
                        w.writerow([
                            cx[-1], cy[-1], cx[-2], cy[-2],
                            target_vec[0], target_vec[1],
                            tx, ty, smooth_deg, heading_deg,
                            params['const_speed'], dist_to_target, log_time, rtk_status_txt
                        ])
                    else:
                        w.writerow([
                            cx[-1], cy[-1], '', '', '', '',
                            tx, ty, smooth_deg, heading_deg,
                            params['const_speed'], dist_to_target, log_time, rtk_status_txt
                        ])
            except Exception as e:
                rospy.logwarn(f"[tracker_topics] log write failed: {e}")

        # 터미널 로그
        now = time.time()
        if now - _last_log_t > 0.5:
            latlon_txt = f"Lat: {_last_lat:.7f}, Lon: {_last_lon:.7f}" if (_last_lat is not None and _last_lon is not None) else "Lat/Lon: (n/a)"
            heading_deg = math.degrees(heading_rad) if heading_rad is not None else 0.0
            dist_to_target = distance_m(cx[-1], cy[-1], tx, ty)
            rospy.loginfo(f"{latlon_txt}, Speed: {params['const_speed']:.2f} m/s, "
                          f"Steering: {smooth_deg:+.2f} deg, Heading: {heading_deg:.2f} deg, "
                          f"Dist→Target: {dist_to_target:.2f} m, RTK: {rtk_status_txt}")
            _last_log_t = now

        # 도착 반경 → 다음 타겟
        if len(cx) > 1 and distance_m(cx[-1], cy[-1], tx, ty) < float(params['target_radius']):
            if waypoint_index < len(waypoints_x) - 1:
                waypoint_index += 1

        # Info Box
        info_lines.append(f"Veh: ({cx[-1]:.1f}, {cy[-1]:.1f}) m")
        d_to_tgt = distance_m(cx[-1], cy[-1], tx, ty)
        info_lines.append(f"Dist→Target: {d_to_tgt:.1f} m")
        if heading_rad is not None:
            info_lines.append(f"Heading: {math.degrees(heading_rad):.1f}°")
        info_lines.append(f"Steering: {smooth_deg:+.1f}°")

    if info_lines:
        ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
                ha='left', va='top', fontsize=9, bbox=dict(fc='white', alpha=0.7))

    ax.set_title(f"ROS GPS Tracker  Steering: {smooth_deg:.2f}°  RTK: {rtk_status_txt}")
    ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)')
    ax.axis('equal'); ax.grid(True, ls=':', alpha=0.5)
    ax.legend(loc='upper right')

# ── 메인 ────────────────────────────────────────────
def main():
    global pub_speed, pub_steer, pub_rtk, waypoints_x, waypoints_y, alpha, params, heading_hist

    rospy.init_node('waypoint_tracker_topics', anonymous=False)

    ublox_ns = rospy.get_param('~ublox_ns', '/ublox')
    params = {
        'fix_topic':        rospy.get_param('~fix_topic',    ublox_ns + '/fix'),
        'relpos_topic':     rospy.get_param('~relpos_topic', ublox_ns + '/navrelposned'),
        'navpvt_topic':     rospy.get_param('~navpvt_topic', ublox_ns + '/navpvt'),
        'waypoint_csv':     rospy.get_param('~waypoint_csv', WAYPOINT_CSV_DEFAULT),
        'target_radius':    float(rospy.get_param('~target_radius', TARGET_RADIUS_DEFAULT)),
        'min_wp_distance':  float(rospy.get_param('~min_wp_distance', MIN_WAYPOINT_DISTANCE_DEFAULT)),
        'fc':               float(rospy.get_param('~fc', FC_DEFAULT)),
        'fs':               float(rospy.get_param('~fs', FS_DEFAULT)),
        'gps_outlier_th':   float(rospy.get_param('~gps_outlier_th', GPS_OUTLIER_THRESHOLD_DEFAULT)),
        'steer_limit_deg':  float(rospy.get_param('~steer_limit_deg', STEER_LIMIT_DEG_DEFAULT)),
        'const_speed':      float(rospy.get_param('~const_speed', CONST_SPEED_DEFAULT)),
        'log_csv':          rospy.get_param('~log_csv', LOG_CSV_DEFAULT),

        # ★ 헤딩 안정화 파라미터
        'heading_min_move_m':   float(rospy.get_param('~heading_min_move_m', 0.07)),
        'heading_hist_len':     int(rospy.get_param('~heading_hist_len', 5)),
        'heading_max_jump_deg': float(rospy.get_param('~heading_max_jump_deg', 35.0)),
    }

    # LPF 계수
    alpha = (2 * math.pi * params['fc']) / (2 * math.pi * params['fc'] + params['fs'])

    # 헤딩 히스토리 maxlen 적용
    heading_hist = deque(list(heading_hist), maxlen=int(params['heading_hist_len']))

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

    # 시각화 루프
    plt.ion()
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111)

    rate = rospy.Rate(params['fs'])
    try:
        while not rospy.is_shutdown():
            update_plot_once(ax)
            plt.pause(0.001)
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        print("csv 저장 되쓰요!")

if __name__ == '__main__':
    main()
