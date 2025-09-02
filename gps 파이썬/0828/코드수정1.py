#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waypoint_tracker_topics.py  (ROS1 Noetic, Ubuntu)

입출력:
- 구독: /gps1/fix (sensor_msgs/NavSatFix)
       (옵션) /gps1/navrelposned, /gps1/navpvt
- 퍼블리시: /vehicle/speed_cmd (Float32), /vehicle/steer_cmd (Float32, deg), /rtk/status (String)

개선(요청 1·2·4 반영) + 속도 명령 고정/상한:
1) 헤딩 안정화 4종 (bootstrap, 임계속도 hold, 원형 EMA, last heading 기반 조향)
2) 동적 이상치 게이팅 + 스텝 클램프
4) 시각화 축 고정
S) 속도 명령 고정/상한:
   - speed_mode: "code"(정수코드) | "const"(m/s)
   - code 모드: speed_code를 0..speed_cap_code(기본 4)로 클램프 → Float32로 퍼블리시(1.0, 2.0, …)
   - const 모드: const_speed를 0..speed_cap_mps(기본 4.0 m/s)로 클램프
   - CSV: speed_cmd(명령), speed_meas_ema(실측)
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
from matplotlib.patches import FancyArrowPatch, Circle

import rospy
import rospkg
from std_msgs.msg import Float32, String
from sensor_msgs.msg import NavSatFix

# ── ublox_msgs (옵션) ─────────────────────────────────
_HAVE_RELPOSNED = False
_HAVE_NAVPVT    = False
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
TARGET_RADIUS_DEFAULT = 1.5        # [m]
MIN_WAYPOINT_DISTANCE_DEFAULT = 0.9
FC_DEFAULT = 2.0                   # [Hz] steer LPF
FS_DEFAULT = 20.0                  # [Hz]
STEER_LIMIT_DEG_DEFAULT = 20.0     # [deg]
CONST_SPEED_DEFAULT = 1.0          # [m/s] (const 모드에서 사용)

# 속도 명령 고정/상한
SPEED_MODE_DEFAULT     = "code"    # "code" | "const"
SPEED_CODE_DEFAULT     = 1         # 정수 코드 기본값
SPEED_CAP_CODE_DEFAULT = 4         # 코드 상한(최대 4)
SPEED_CAP_MPS_DEFAULT  = 4.0       # m/s 상한(최대 4.0 m/s)

# (1) 헤딩 안정화 파라미터
HEADING_MIN_SPEED_DEFAULT = 0.35   # [m/s] 이속 미만이면 헤딩 동결
HEADING_ALPHA_DEFAULT     = 0.25   # 원형 EMA 계수(0~1)
BOOTSTRAP_DIST_DEFAULT    = 2.0    # [m] 시작 직후 전진거리

# (2) 동적 게이팅(+클램프) 파라미터
GATE_FLOOR_M_DEFAULT   = 0.5
GATE_K_VEL_DEFAULT     = 3.0
GATE_K_SIGMA_DEFAULT   = 3.0
GATE_RTK_FIX_DEFAULT   = 0.20
GATE_RTK_FLOAT_DEFAULT = 0.50
GATE_RTK_NONE_DEFAULT  = 1.00

# (4) 시각화
SHOW_ALL_WP_DEFAULT = True
WIN_SIZE_DEFAULT     = 50
AXIS_MARGIN_DEFAULT  = 5.0

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

current_x, current_y, current_t = [], [], []   # 경로 + 타임스탬프
waypoints_x = None
waypoints_y = None
waypoint_index = 0

# steer LPF
alpha = 0.56
_filtered_steering = 0.0

# 동적 게이팅용 상태
_prev_raw_x = None
_prev_raw_y = None
_prev_raw_t = None
_prev_f_x = None
_prev_f_y = None

_last_lat = None
_last_lon = None
rtk_status_txt = "NONE"
_state_lock = threading.Lock()
_last_log_t = 0.0

# 헤딩/속도 상태
_last_fix_heading_rad = None   # 마지막 '확정' 헤딩
_speed_mps = 0.0               # 순간 속도 추정
_speed_ema = 0.0               # 지수평활 속도 추정(표시/로그용)
_speed_alpha = 0.4

# 부트스트랩
_bootstrap_done = False
_boot_start_x = None
_boot_start_y = None

# 시각화 축 고정
AX_MIN_X = AX_MAX_X = AX_MIN_Y = AX_MAX_Y = None

# 최근 퍼블리시 속도 명령(로그/화면 표시용)
_last_speed_cmd = 0.0
_last_speed_mode = SPEED_MODE_DEFAULT

# ── 유틸 ───────────────────────────────────────────
def latlon_to_meters(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def distance_m(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def circ_ema(prev_rad, new_rad, a):
    """원형 EMA: sin/cos 가중 평균 후 atan2 복원"""
    if prev_rad is None:
        return new_rad
    s = (1 - a) * math.sin(prev_rad) + a * math.sin(new_rad)
    c = (1 - a) * math.cos(prev_rad) + a * math.cos(new_rad)
    return math.atan2(s, c)

def apply_low_pass_filter(current):
    """조향 LPF (1차 IIR) + 부호 관례 반전"""
    global _filtered_steering, alpha
    _filtered_steering = (1 - alpha) * _filtered_steering + alpha * current
    return _filtered_steering * -1.0

# ── 동적 게이팅 + 스텝 클램프 + 좌표 LPF ───────────
def filter_gps_signal(x, y, sigma_xy):
    """
    x,y: 새 원시 좌표(웹 메르카토르, m)
    sigma_xy: 수평 표준편차 근사(m). 없으면 0.0
    동적 게이트로 '최대 허용 스텝'을 만들고, 초과분은 방향 유지하며 길이만 클램프.
    이후 좌표 LPF(alpha) 적용.
    """
    global _prev_raw_x, _prev_raw_y, _prev_raw_t, _prev_f_x, _prev_f_y, _speed_ema, rtk_status_txt
    now = time.time()

    # 초기화
    if _prev_raw_x is None:
        _prev_raw_x, _prev_raw_y = x, y
        _prev_raw_t = now
        _prev_f_x, _prev_f_y = x, y
        return x, y

    # 1) dt 계산
    dt = max(1e-3, now - (_prev_raw_t if _prev_raw_t else now))

    # 2) RTK 상태별 여유
    rtk_margin = (params['gate_rtk_fix'] if rtk_status_txt == "FIX"
                  else (params['gate_rtk_float'] if rtk_status_txt == "FLOAT"
                        else params['gate_rtk_none']))

    # 3) 동적 최대 허용 스텝
    max_step = max(params['gate_floor_m'],
                   params['gate_k_vel'] * _speed_ema * dt +
                   params['gate_k_sigma'] * max(0.0, sigma_xy) +
                   rtk_margin)

    # 4) 원시 스텝
    dx = x - _prev_raw_x
    dy = y - _prev_raw_y
    dist = math.hypot(dx, dy)

    # 5) 소프트 게이팅: 스텝 클램프
    if dist > max_step:
        ux, uy = dx / dist, dy / dist
        x = _prev_raw_x + ux * max_step
        y = _prev_raw_y + uy * max_step
        dist = max_step

    # 6) 원시 포인트 갱신
    _prev_raw_x, _prev_raw_y, _prev_raw_t = x, y, now

    # 7) 좌표 LPF
    fx = (1 - alpha) * _prev_f_x + alpha * x
    fy = (1 - alpha) * _prev_f_y + alpha * y
    _prev_f_x, _prev_f_y = fx, fy
    return fx, fy

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

# ── 퍼블리셔 ────────────────────────────────────────
def publish_speed_cmd(v_float32):
    """Float32로 퍼블리시(코드 모드면 1.0/2.0/…로 나감)"""
    if pub_speed:
        pub_speed.publish(Float32(data=float(v_float32)))

def publish_steer_deg(steer_deg):
    sd = clamp(float(steer_deg), -float(params['steer_limit_deg']), float(params['steer_limit_deg']))
    if pub_steer:
        pub_steer.publish(Float32(data=sd))

def publish_rtk(txt):
    if pub_rtk:
        pub_rtk.publish(String(data=str(txt)))

# ── 속도 명령 계산(모드·상한 적용, 런타임 파라미터 반영) ──
def compute_speed_command():
    """
    반환:
      cmd_float  : 퍼블리시에 넣을 Float32 값
      cmd_report : 로그/표시에 쓸 값(코드 모드면 int, const 모드면 float)
      mode_str   : "code" | "const"
    """
    # 런타임에 바뀐 파라미터 반영
    mode = rospy.get_param('~speed_mode', params['speed_mode']).strip().lower()
    if mode == "code":
        code = int(rospy.get_param('~speed_code', params['speed_code']))
        cap  = int(rospy.get_param('~speed_cap_code', params['speed_cap_code']))
        if cap < 0: cap = 0
        code = max(0, min(code, cap))  # 0..cap
        return float(code), int(code), "code"
    else:
        mps  = float(rospy.get_param('~const_speed', params['const_speed']))
        capm = float(rospy.get_param('~speed_cap_mps', params['speed_cap_mps']))
        capm = max(0.0, capm)
        mps  = max(0.0, min(mps, capm))
        return float(mps), float(mps), "const"

# ── ROS 콜백 ────────────────────────────────────────
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
    # 미사용
    pass

def _cb_fix(msg: NavSatFix):
    """GNSS 수신 → 좌표 필터링 → 속도/헤딩 갱신"""
    global _last_lat, _last_lon, _last_fix_heading_rad
    global _bootstrap_done, _boot_start_x, _boot_start_y
    global _speed_mps, _speed_ema

    if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
        return

    _last_lat, _last_lon = float(msg.latitude), float(msg.longitude)
    x, y = latlon_to_meters(_last_lat, _last_lon)

    # 수평 정확도 근사(공분산 있으면 사용)
    sigma_xy = 0.0
    try:
        cov = list(msg.position_covariance)
        if len(cov) >= 5:
            sigma_xy = math.sqrt(max(0.0, cov[0]) + max(0.0, cov[4]))
    except Exception:
        pass

    fx, fy = filter_gps_signal(x, y, sigma_xy)

    now = time.time()
    with _state_lock:
        # 속도 추정
        if current_x and current_y and current_t:
            dt = max(1e-6, now - current_t[-1])
            dist = distance_m(fx, fy, current_x[-1], current_y[-1])
            _speed_mps = dist / dt
            _speed_ema = (1 - _speed_alpha) * _speed_ema + _speed_alpha * _speed_mps
        else:
            _speed_mps = 0.0

        # 부트스트랩
        if _boot_start_x is None:
            _boot_start_x, _boot_start_y = fx, fy
        if not _bootstrap_done:
            if distance_m(_boot_start_x, _boot_start_y, fx, fy) >= float(params['bootstrap_dist']):
                _bootstrap_done = True

        # 헤딩 갱신 조건
        moved = False
        if current_x and current_y:
            dx = fx - current_x[-1]
            dy = fy - current_y[-1]
            moved = (dx*dx + dy*dy) > 1e-8

        if _bootstrap_done and (_speed_ema >= float(params['heading_min_speed'])) and moved:
            raw_heading = math.atan2(dy, dx)
            _last_fix_heading_rad = circ_ema(_last_fix_heading_rad, raw_heading,
                                             float(params['heading_alpha']))
        # push
        current_x.append(fx); current_y.append(fy); current_t.append(now)

# ── 조향각 계산(확정 헤딩 벡터 기반) ────────────────
def steering_from_vectors(heading_rad, cx, cy, tx, ty):
    target_vec = np.array([tx - cx, ty - cy], dtype=float)
    if np.linalg.norm(target_vec) < 1e-6:
        return 0.0
    if heading_rad is None:
        fwd = np.array([1.0, 0.0], dtype=float)
    else:
        fwd = np.array([math.cos(heading_rad), math.sin(heading_rad)], dtype=float)
    det = fwd[0]*target_vec[1] - fwd[1]*target_vec[0]
    dot = fwd[0]*target_vec[0] + fwd[1]*target_vec[1]
    ang = math.degrees(math.atan2(det, dot))
    ang = clamp(ang / 1.3, -float(params['steer_limit_deg']), float(params['steer_limit_deg']))
    return ang

# ── 시각화(plt.ion) ─────────────────────────────────
def update_plot_once(ax):
    global waypoint_index, _last_log_t, _last_speed_cmd, _last_speed_mode
    ax.clear()

    with _state_lock:
        cx = list(current_x); cy = list(current_y)
        heading_rad = _last_fix_heading_rad
        rtk_txt = rtk_status_txt
        v_meas = _speed_ema

    # 경로
    if len(cx) >= 2:
        ax.plot(cx, cy, '-', c='0.6', lw=1.0, label='Route')

    # 웨이포인트
    if bool(params['show_all_waypoints']):
        ax.scatter(waypoints_x, waypoints_y, color='blue', s=10, label='Waypoints')
        for i in range(len(waypoints_x)):
            c = Circle((waypoints_x[i], waypoints_y[i]), float(params['target_radius']),
                       fill=False, linestyle='--', edgecolor='tab:blue', alpha=0.25)
            ax.add_patch(c)
            ax.text(waypoints_x[i], waypoints_y[i], str(i+1), fontsize=7, ha='center')
    else:
        window_size = int(params['win_size'])
        start = max(0, waypoint_index - window_size//2)
        end   = min(len(waypoints_x), start + window_size)
        ax.scatter(waypoints_x[start:end], waypoints_y[start:end], color='blue', s=10, label='Waypoints')
        for i in range(start, end):
            c = Circle((waypoints_x[i], waypoints_y[i]), float(params['target_radius']),
                       fill=False, linestyle='--', edgecolor='tab:blue', alpha=0.25)
            ax.add_patch(c)
            ax.text(waypoints_x[i], waypoints_y[i], str(i+1), fontsize=7, ha='center')

    smooth_deg = 0.0
    info_lines = []
    if cx and cy:
        ax.scatter(cx[-1], cy[-1], color='red', s=50, label='Current')

        tx, ty = waypoints_x[waypoint_index], waypoints_y[waypoint_index]
        ax.plot([cx[-1], tx], [cy[-1], ty], '--', c='cyan', lw=1.0, label='Target Line')
        ax.plot(tx, ty, '*', c='magenta', ms=12, label='Target')

        angle = steering_from_vectors(heading_rad, cx[-1], cy[-1], tx, ty)
        smooth_deg = apply_low_pass_filter(angle)
        smooth_deg = clamp(smooth_deg, -float(params['steer_limit_deg']), float(params['steer_limit_deg']))

        # ── 속도 명령 계산 + 퍼블리시(상한 적용) ──
        cmd_float, cmd_report, mode_str = compute_speed_command()
        publish_speed_cmd(cmd_float)
        _last_speed_cmd = cmd_report
        _last_speed_mode = mode_str

        # 화살표
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

        # ── 로깅/로그 출력 ──
        if params['log_csv']:
            try:
                new = not os.path.exists(params['log_csv'])
                os.makedirs(os.path.dirname(params['log_csv']), exist_ok=True)
                with open(params['log_csv'], 'a', newline='') as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow([
                            'current_x','current_y','prev_x','prev_y',
                            'waypoint_x','waypoint_y',
                            'steer_deg','heading_deg',
                            'speed_cmd','speed_meas_ema',
                            'dist_to_target','time','rtk_status','speed_mode'
                        ])
                    dist_to_target = distance_m(cx[-1], cy[-1], tx, ty)
                    heading_deg = math.degrees(heading_rad) if heading_rad is not None else ''
                    log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    if len(cx) > 1:
                        w.writerow([
                            cx[-1], cy[-1], cx[-2], cy[-2],
                            tx, ty, smooth_deg, heading_deg,
                            _last_speed_cmd, v_meas,
                            dist_to_target, log_time, rtk_txt, _last_speed_mode
                        ])
                    else:
                        w.writerow([
                            cx[-1], cy[-1], '', '',
                            tx, ty, smooth_deg, heading_deg,
                            _last_speed_cmd, v_meas,
                            distance_m(cx[-1], cy[-1], tx, ty), log_time, rtk_txt, _last_speed_mode
                        ])
            except Exception as e:
                rospy.logwarn(f"[tracker_topics] log write failed: {e}")

        now = time.time()
        if now - _last_log_t > 0.5:
            latlon_txt = f"Lat: {_last_lat:.7f}, Lon: {_last_lon:.7f}" if (_last_lat is not None and _last_lon is not None) else "Lat/Lon: (n/a)"
            heading_deg = math.degrees(heading_rad) if heading_rad is not None else 0.0
            dist_to_target = distance_m(cx[-1], cy[-1], tx, ty)
            rospy.loginfo(f"{latlon_txt}, SpeedCmd({mode_str})={_last_speed_cmd}, v(EMA)={v_meas:.2f} m/s, "
                          f"Steer={smooth_deg:+.2f} deg, Heading={heading_deg:.2f} deg, "
                          f"Dist→Target={dist_to_target:.2f} m, RTK={rtk_txt}")
            _last_log_t = now

        # 도착 판정
        if distance_m(cx[-1], cy[-1], tx, ty) < float(params['target_radius']):
            if waypoint_index < len(waypoints_x) - 1:
                waypoint_index += 1

        # Info 박스
        info_lines.append(f"Veh: ({cx[-1]:.1f}, {cy[-1]:.1f}) m")
        info_lines.append(f"Dist→Target: {distance_m(cx[-1], cy[-1], tx, ty):.1f} m")
        if heading_rad is not None:
            info_lines.append(f"Heading: {math.degrees(heading_rad):.1f}°")
        info_lines.append(f"Steering: {smooth_deg:+.1f}°")
        info_lines.append(f"SpeedCmd({mode_str}): {_last_speed_cmd}")

    if info_lines:
        ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
                ha='left', va='top', fontsize=9, bbox=dict(fc='white', alpha=0.7))

    # 축 고정(짤림 방지)
    ax.set_title(f"ROS GPS Tracker  Steering: {smooth_deg:.2f}°  RTK: {rtk_txt}")
    ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)')
    ax.axis('equal'); ax.grid(True, ls=':', alpha=0.5)
    if AX_MIN_X is not None:
        ax.set_xlim(AX_MIN_X, AX_MAX_X); ax.set_ylim(AX_MIN_Y, AX_MAX_Y)
    ax.legend(loc='upper right')

# ── 메인 ────────────────────────────────────────────
def main():
    global pub_speed, pub_steer, pub_rtk, waypoints_x, waypoints_y, alpha, params
    global AX_MIN_X, AX_MAX_X, AX_MIN_Y, AX_MAX_Y

    rospy.init_node('waypoint_tracker_topics', anonymous=False)

    ublox_ns = rospy.get_param('~ublox_ns', '/gps1')
    params = {
        'fix_topic':        rospy.get_param('~fix_topic',    ublox_ns + '/fix'),
        'relpos_topic':     rospy.get_param('~relpos_topic', ublox_ns + '/navrelposned'),
        'navpvt_topic':     rospy.get_param('~navpvt_topic', ublox_ns + '/navpvt'),
        'waypoint_csv':     rospy.get_param('~waypoint_csv', WAYPOINT_CSV_DEFAULT),
        'target_radius':    float(rospy.get_param('~target_radius', TARGET_RADIUS_DEFAULT)),
        'min_wp_distance':  float(rospy.get_param('~min_wp_distance', MIN_WAYPOINT_DISTANCE_DEFAULT)),
        'fc':               float(rospy.get_param('~fc', FC_DEFAULT)),
        'fs':               float(rospy.get_param('~fs', FS_DEFAULT)),
        'steer_limit_deg':  float(rospy.get_param('~steer_limit_deg', STEER_LIMIT_DEG_DEFAULT)),
        'const_speed':      float(rospy.get_param('~const_speed', CONST_SPEED_DEFAULT)),
        'log_csv':          rospy.get_param('~log_csv', LOG_CSV_DEFAULT),

        # 속도 명령 고정/상한
        'speed_mode':       rospy.get_param('~speed_mode', SPEED_MODE_DEFAULT),
        'speed_code':       int(rospy.get_param('~speed_code', SPEED_CODE_DEFAULT)),
        'speed_cap_code':   int(rospy.get_param('~speed_cap_code', SPEED_CAP_CODE_DEFAULT)),
        'speed_cap_mps':    float(rospy.get_param('~speed_cap_mps', SPEED_CAP_MPS_DEFAULT)),

        # 1) 헤딩 안정화
        'heading_min_speed': float(rospy.get_param('~heading_min_speed', HEADING_MIN_SPEED_DEFAULT)),
        'heading_alpha':     float(rospy.get_param('~heading_alpha', HEADING_ALPHA_DEFAULT)),
        'bootstrap_dist':    float(rospy.get_param('~bootstrap_dist', BOOTSTRAP_DIST_DEFAULT)),

        # 2) 동적 게이팅
        'gate_floor_m':      float(rospy.get_param('~gate_floor_m', GATE_FLOOR_M_DEFAULT)),
        'gate_k_vel':        float(rospy.get_param('~gate_k_vel', GATE_K_VEL_DEFAULT)),
        'gate_k_sigma':      float(rospy.get_param('~gate_k_sigma', GATE_K_SIGMA_DEFAULT)),
        'gate_rtk_fix':      float(rospy.get_param('~gate_rtk_fix', GATE_RTK_FIX_DEFAULT)),
        'gate_rtk_float':    float(rospy.get_param('~gate_rtk_float', GATE_RTK_FLOAT_DEFAULT)),
        'gate_rtk_none':     float(rospy.get_param('~gate_rtk_none', GATE_RTK_NONE_DEFAULT)),

        # 4) 시각화
        'show_all_waypoints': bool(rospy.get_param('~show_all_waypoints', SHOW_ALL_WP_DEFAULT)),
        'win_size':           int(rospy.get_param('~win_size', WIN_SIZE_DEFAULT)),
        'axis_margin':        float(rospy.get_param('~axis_margin', AXIS_MARGIN_DEFAULT)),
    }

    # steer LPF 계수
    global alpha
    alpha = (2 * math.pi * params['fc']) / (2 * math.pi * params['fc'] + params['fs'])

    # 퍼블리셔
    global pub_speed, pub_steer, pub_rtk
    pub_speed = rospy.Publisher('/vehicle/speed_cmd', Float32, queue_size=10)
    pub_steer = rospy.Publisher('/vehicle/steer_cmd',  Float32, queue_size=10)
    pub_rtk   = rospy.Publisher('/rtk/status',         String,  queue_size=10)

    # 웨이포인트 + 축 범위 계산
    try:
        os.makedirs(os.path.dirname(params['waypoint_csv']), exist_ok=True)
        global waypoints_x, waypoints_y
        waypoints_x, waypoints_y = load_waypoints(params['waypoint_csv'], params['min_wp_distance'])
        global AX_MIN_X, AX_MAX_X, AX_MIN_Y, AX_MAX_Y
        AX_MIN_X = float(np.min(waypoints_x)) - params['axis_margin']
        AX_MAX_X = float(np.max(waypoints_x)) + params['axis_margin']
        AX_MIN_Y = float(np.min(waypoints_y)) - params['axis_margin']
        AX_MAX_Y = float(np.max(waypoints_y)) + params['axis_margin']
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

    # 루프
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
        print("csv 저장 되었습니다.")

# ── 엔트리 ──────────────────────────────────────────
if __name__ == '__main__':
    main()
