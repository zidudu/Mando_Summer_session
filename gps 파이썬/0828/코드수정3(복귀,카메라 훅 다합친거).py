#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waypoint_tracker_topics.py  (ROS1 Noetic, Ubuntu)

입출력:
- 구독: /gps1/fix (sensor_msgs/NavSatFix)
       (옵션) /gps1/navrelposned, /gps1/navpvt
- 퍼블리시: /vehicle/speed_cmd (Float32), /vehicle/steer_cmd (Float32, deg), /rtk/status (String)

기능 요약(요청 반영):
1) 시작 워밍업: start_warmup_sec 동안 speed=1, steer=0 고정(조향 계산 무시)
2) 정지-재출발 안정화:
   - 저속/무이동 시 헤딩 hold-last + 전/전전 헤딩 후보중 '타겟과 각오차 최소'를 선택
   - 정지 직전 헤딩 스냅샷 → 재출발 시 우선 사용(완만히 전환)
   - 이동 변화율 데드존: Δs < max(ratio*dist_to_target, min_m)면 steer=0
3) 순차 인덱스 고정 + 반경 히스테리시스(radius_hit_count회 연속 진입 시에만 인덱스 증가)
4) 이탈/복귀 모드:
   - dist_to_target > offtrack_trigger_mult*target_radius 이면 복귀 후보 검색
   - 현재 헤딩 대비 ±recovery_heading_limit_deg(기본 30°) 내에서
     recovery_search_radius 내 가장 가까운 '앞쪽 인덱스' 후보를 타겟으로 잠금
   - 복귀 완료(반경 진입) 시 정상 모드로 복귀
5) 전/후 로우패스: 좌표(동적 게이팅+LPF) → 조향 LPF, 헤딩 원형 EMA
6) 속도 명령: "code/const" + 상한 + 하드 고정(SPEED_FORCE)

"""

import os
import csv
import math
import time
import threading
from collections import deque

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
TARGET_RADIUS_DEFAULT = 1.5
MIN_WAYPOINT_DISTANCE_DEFAULT = 0.9
FC_DEFAULT = 2.0                   # steer LPF [Hz]
FS_DEFAULT = 20.0                  # loop [Hz]
STEER_LIMIT_DEG_DEFAULT = 20.0     # 조향 명령 제한 [deg]
CONST_SPEED_DEFAULT = 1.0          # const 모드 기본 속도 [m/s]

# 속도 명령 기본/상한
SPEED_MODE_DEFAULT     = "code"    # "code" | "const"
SPEED_CODE_DEFAULT     = 1
SPEED_CAP_CODE_DEFAULT = 4
SPEED_CAP_MPS_DEFAULT  = 4.0

# ✅ 하드 고정(원하면 사용)
#   ('code', N)  → 정수 코드 N으로 고정 (0~4, 퍼블리시는 1.0/2.0/…)
#   ('const', v) → v m/s로 고정 (0.0~4.0)
#   None         → 비활성(파라미터 사용)
SPEED_FORCE = ('code', 1)
# SPEED_FORCE = None

# (1) 헤딩 안정화 파라미터
HEADING_MIN_SPEED_DEFAULT = 0.35   # [m/s] 이속 미만이면 헤딩 동결
HEADING_ALPHA_DEFAULT     = 0.25   # 원형 EMA 계수(0~1)
BOOTSTRAP_DIST_DEFAULT    = 2.0    # [m] 시작 직후 전진거리

# (A) 시작 워밍업
START_WARMUP_SEC_DEFAULT = 2.0     # [s]

# (B) 정지/재출발 스냅샷
STOP_SPEED_EPS_DEFAULT   = 0.05    # [m/s] 이하면 정지로 간주
STOP_HOLD_SEC_DEFAULT    = 0.5     # [s] 연속 유지 시 정지 스냅샷
RESTART_MIN_SPEED_DEFAULT= 0.30    # [m/s] 재출발 간주

# (C) 조향 데드존(이동 변화율 기반)
STEER_DEADZONE_MOVE_RATIO_DEFAULT = 0.005  # dist_to_target의 1% 미만 이동이면
STEER_DEADZONE_MOVE_MIN_M_DEFAULT = 0.01  # 최소 5cm 기준

# (D) 헤딩 후보 선택(전/전전 포함)
HEAD_BUF_LEN_DEFAULT     = 5       # 최근 확정 헤딩 보관 개수
HEAD_SELECT_MAX_DEG_DEF  = 90.0    # 후보 허용 최대 각오차(절대)
HEAD_GOOD_RATIO_DEFAULT  = 0.75    # raw_error의 75% 이하로 줄이는 후보만 채택

# (E) 반경 히스테리시스
RADIUS_HIT_COUNT_DEFAULT = 2       # 연속 N회 반경 내에 들어와야 인덱스 증가

# (F) 이탈/복귀
OFFTRACK_TRIGGER_MULT_DEF     = 4.0    # dist_to_target > 4*target_radius → 복귀 탐색
RECOVERY_SEARCH_RADIUS_DEF    = 15.0   # [m] 복귀 후보 검색 반경
RECOVERY_LOOKAHEAD_N_DEF      = 30     # 앞쪽 인덱스 범위
RECOVERY_HEADING_LIMIT_DEG_DEF= 30.0   # 현재 헤딩 대비 허용 편차(앞바퀴 각도 고려)

# (G) 시각화
SHOW_ALL_WP_DEFAULT = True
WIN_SIZE_DEFAULT     = 50
AXIS_MARGIN_DEFAULT  = 5.0

# 패키지/config 기본 경로
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

current_x, current_y, current_t = [], [], []   # 필터 좌표 + 타임스탬프
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
_last_fix_heading_rad = None              # 마지막 '확정' 헤딩(EMA 결과)
_speed_mps = 0.0
_speed_ema = 0.0
_speed_alpha = 0.4

# 시작 워밍업
_first_fix_time = None
_warmup_until = None

# 정지/재출발
_stopped_flag = False
_stop_since_t = None
_stop_heading_snapshot = None
_restart_confirm = 0

# 전/전전 포함 헤딩 후보 버퍼
_head_buf = deque(maxlen=HEAD_BUF_LEN_DEFAULT)  # (heading_rad, t, move_dist)

# 반경 히스테리시스
_radius_hit_consec = 0

# 이탈/복귀
_in_recovery = False
_recovery_target_index = None

# 시각화 축 고정
AX_MIN_X = AX_MAX_X = AX_MIN_Y = AX_MAX_Y = None

# 최근 퍼블리시 속도 명령(로그/화면)
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

def ang_norm_rad(a):
    while a > math.pi:  a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def ang_diff_deg(a, b):
    d = math.degrees(ang_norm_rad(a - b))
    return d

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
    동적 게이트로 '최대 허용 스텝' 만들고 초과분은 방향 유지하며 클램프 → 좌표 LPF.
    """
    global _prev_raw_x, _prev_raw_y, _prev_raw_t, _prev_f_x, _prev_f_y, _speed_ema, rtk_status_txt
    now = time.time()

    if _prev_raw_x is None:
        _prev_raw_x, _prev_raw_y = x, y
        _prev_raw_t = now
        _prev_f_x, _prev_f_y = x, y
        return x, y

    dt = max(1e-3, now - (_prev_raw_t if _prev_raw_t else now))

    rtk_margin = (params['gate_rtk_fix'] if rtk_status_txt == "FIX"
                  else (params['gate_rtk_float'] if rtk_status_txt == "FLOAT"
                        else params['gate_rtk_none']))

    max_step = max(params['gate_floor_m'],
                   params['gate_k_vel'] * _speed_ema * dt +
                   params['gate_k_sigma'] * max(0.0, sigma_xy) +
                   rtk_margin)

    dx = x - _prev_raw_x
    dy = y - _prev_raw_y
    dist = math.hypot(dx, dy)

    if dist > max_step:
        ux, uy = dx / dist, dy / dist
        x = _prev_raw_x + ux * max_step
        y = _prev_raw_y + uy * max_step

    _prev_raw_x, _prev_raw_y, _prev_raw_t = x, y, now

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
    if pub_speed:
        pub_speed.publish(Float32(data=float(v_float32)))

def publish_steer_deg(steer_deg):
    sd = clamp(float(steer_deg), -float(params['steer_limit_deg']), float(params['steer_limit_deg']))
    if pub_steer:
        pub_steer.publish(Float32(data=sd))

def publish_rtk(txt):
    if pub_rtk:
        pub_rtk.publish(String(data=str(txt)))

# ── 속도 명령 계산(하드 고정 우선, 그다음 파라미터) ──
def compute_speed_command():
    """
    반환:
      cmd_float  : 퍼블리시에 넣을 Float32 값
      cmd_report : 로그/표시에 쓸 값(코드 모드면 int, const 모드면 float)
      mode_str   : "code" | "const" | "code(force)" | "const(force)"
    """
    if SPEED_FORCE is not None:
        mode, val = SPEED_FORCE
        if str(mode).lower() == 'code':
            cap = int(params.get('speed_cap_code', SPEED_CAP_CODE_DEFAULT))
            code = int(clamp(int(val), 0, cap))
            return float(code), int(code), "code(force)"
        else:
            capm = float(params.get('speed_cap_mps', SPEED_CAP_MPS_DEFAULT))
            mps = float(clamp(float(val), 0.0, capm))
            return mps, mps, "const(force)"

    mode = rospy.get_param('~speed_mode', params['speed_mode']).strip().lower()
    if mode == "code":
        code = int(rospy.get_param('~speed_code', params['speed_code']))
        cap  = int(rospy.get_param('~speed_cap_code', params['speed_cap_code']))
        code = int(clamp(code, 0, cap))
        return float(code), int(code), "code"
    else:
        mps  = float(rospy.get_param('~const_speed', params['const_speed']))
        capm = float(rospy.get_param('~speed_cap_mps', params['speed_cap_mps']))
        mps  = float(clamp(mps, 0.0, capm))
        return mps, mps, "const"

# ── 복귀 후보 선택 ─────────────────────────────────
def pick_recovery_target(cx, cy, heading_rad, start_idx):
    """
    현재 위치/헤딩에서 앞쪽 인덱스 범위 내에서
    - 거리 <= recovery_search_radius
    - 현재 헤딩 대비 각도차 <= recovery_heading_limit_deg
    조건을 만족하는 가장 가까운 인덱스를 반환. 없으면 None.
    """
    lim_deg = float(params['recovery_heading_limit_deg'])
    R = float(params['recovery_search_radius'])
    look = int(params['recovery_lookahead_n'])

    best_i = None
    best_d = 1e18
    if heading_rad is None:
        return None

    end_idx = min(len(waypoints_x)-1, start_idx + look)
    for i in range(start_idx, end_idx+1):
        tx, ty = waypoints_x[i], waypoints_y[i]
        d = distance_m(cx, cy, tx, ty)
        if d > R:
            continue
        bearing = math.atan2(ty - cy, tx - cx)
        diff = abs(ang_diff_deg(bearing, heading_rad))
        if diff <= lim_deg:
            if d < best_d:
                best_d = d
                best_i = i
    return best_i

# ── 헤딩 후보 선택(전/전전 포함) ────────────────────
def select_stable_heading(bearing_to_target, raw_heading):
    """
    저속/무이동 상황에서 사용할 '안정적 헤딩'을 선택.
    - 후보: stop_snapshot, last_fix_heading, 최근 확정 헤딩 버퍼(_head_buf)
    - 정책: 타겟 베어링과의 절대 각오차가 가장 작은 후보 채택.
            단, 후보 각오차 <= head_select_max_deg 이고,
            raw_heading 대비 오차를 head_good_ratio(기본 0.75) 이하로 줄여줄 수 있어야 함.
    """
    cand = []

    # 1) 스냅샷
    if _stop_heading_snapshot is not None:
        cand.append(_stop_heading_snapshot)

    # 2) 마지막 확정 헤딩
    if _last_fix_heading_rad is not None:
        cand.append(_last_fix_heading_rad)

    # 3) 버퍼 후보들(중복 제거)
    seen = set()
    for h, t, md in list(_head_buf):
        key = round(h, 6)
        if key not in seen:
            cand.append(h)
            seen.add(key)

    if not cand:
        return raw_heading  # fallback

    # 점수화
    max_deg = float(params['head_select_max_deg'])
    good_ratio = float(params['head_good_ratio'])
    raw_err = abs(ang_diff_deg(bearing_to_target, raw_heading)) if raw_heading is not None else 180.0

    best_h = None
    best_err = 1e9
    for h in cand:
        err = abs(ang_diff_deg(bearing_to_target, h))
        if err <= max_deg and err <= good_ratio * raw_err:
            if err < best_err:
                best_err = err
                best_h = h

    return best_h if best_h is not None else (_last_fix_heading_rad if _last_fix_heading_rad is not None else raw_heading)

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
    """GNSS 수신 → 좌표 필터링 → 속도/헤딩 갱신 + 시작/정지 상태 머신"""
    global _last_lat, _last_lon, _last_fix_heading_rad
    global _first_fix_time, _warmup_until
    global _speed_mps, _speed_ema
    global _stopped_flag, _stop_since_t, _stop_heading_snapshot, _restart_confirm
    global _head_buf

    if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
        return

    _last_lat, _last_lon = float(msg.latitude), float(msg.longitude)
    x, y = latlon_to_meters(_last_lat, _last_lon)

    # 수평 정확도 근사(공분산 사용 가능 시)
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
        # 첫 fix 시각 & 워밍업 종료시각
        if _first_fix_time is None:
            _first_fix_time = now
            _warmup_until = _first_fix_time + float(params['start_warmup_sec'])

        # 속도 추정
        if current_x and current_y and current_t:
            dt = max(1e-6, now - current_t[-1])
            dist = distance_m(fx, fy, current_x[-1], current_y[-1])
            _speed_mps = dist / dt
            _speed_ema = (1 - _speed_alpha) * _speed_ema + _speed_alpha * _speed_mps
        else:
            _speed_mps = 0.0

        # 부트스트랩
        moved = False
        if current_x and current_y:
            dx = fx - current_x[-1]
            dy = fy - current_y[-1]
            moved = (dx*dx + dy*dy) > 1e-8

        # 헤딩 갱신(부트스트랩 + 임계속도 + 실제 이동)
        # 갱신 시 버퍼에도 보관
        if current_x and current_y:
            boot_dist = distance_m(current_x[0], current_y[0], fx, fy) if current_x and current_y else 0.0
        else:
            boot_dist = 0.0

        if (boot_dist >= float(params['bootstrap_dist'])) and (_speed_ema >= float(params['heading_min_speed'])) and moved:
            raw_heading = math.atan2(dy, dx)
            _last_fix_heading_rad = circ_ema(_last_fix_heading_rad, raw_heading, float(params['heading_alpha']))
            _head_buf.append((_last_fix_heading_rad, now, distance_m(fx, fy, current_x[-1], current_y[-1]) if current_x else 0.0))

        # 정지/재출발 상태
        if _speed_ema <= float(params['stop_speed_eps']):
            # 정지 유지 시간 누적
            if not _stopped_flag:
                if _stop_since_t is None:
                    _stop_since_t = now
                elif (now - _stop_since_t) >= float(params['stop_hold_sec']):
                    _stopped_flag = True
                    _stop_heading_snapshot = _last_fix_heading_rad
                    _restart_confirm = 0
            else:
                # 이미 정지 중
                pass
        else:
            _stop_since_t = None
            if _stopped_flag and (_speed_ema >= float(params['restart_min_speed'])):
                _restart_confirm += 1
                # 몇 번의 유효 이동 샘플 후 정지 해제
                if _restart_confirm >= 2:
                    _stopped_flag = False
                    _restart_confirm = 0  # 재사용

        # push
        current_x.append(fx); current_y.append(fy); current_t.append(now)

# ── 조향각 계산(확정/안정 헤딩 기반) ────────────────
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

# ── 시각화(plt.ion) & 제어 루프 ─────────────────────
def update_plot_once(ax):
    global waypoint_index, _last_log_t, _last_speed_cmd, _last_speed_mode
    global _in_recovery, _recovery_target_index, _radius_hit_consec

    ax.clear()

    with _state_lock:
        cx = list(current_x); cy = list(current_y)
        heading_rad = _last_fix_heading_rad
        rtk_txt = rtk_status_txt
        v_meas = _speed_ema
        warmup_active = (_warmup_until is not None) and (time.time() < _warmup_until)
        stopped_now = _stopped_flag

    # 경로
    if len(cx) >= 2:
        ax.plot(cx, cy, '-', c='0.6', lw=1.0, label='Route')

    # 웨이포인트 표시
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
    flags = []

    if cx and cy:
        ax.scatter(cx[-1], cy[-1], color='red', s=50, label='Current')

        # 현재 사용 타겟 인덱스(복귀 중이면 복귀 타겟, 아니면 정상 인덱스)
        active_index = waypoint_index
        if _in_recovery and (_recovery_target_index is not None):
            active_index = _recovery_target_index

        tx, ty = waypoints_x[active_index], waypoints_y[active_index]
        ax.plot([cx[-1], tx], [cy[-1], ty], '--', c='cyan', lw=1.0, label='Target Line')
        ax.plot(tx, ty, '*', c='magenta', ms=12, label='Target')

        # 타겟 베어링
        bearing_to_target = math.atan2(ty - cy[-1], tx - cx[-1])

        # --- 조향 계산 분기 ---
        # ① 시작 워밍업: speed=1, steer=0 고정
        if warmup_active:
            cmd_float = 1.0
            mode_str = "warmup"
            _last_speed_cmd = 1
            _last_speed_mode = mode_str
            publish_speed_cmd(cmd_float)
            publish_steer_deg(0.0)
            flags.append("WARMUP")
            smooth_deg = 0.0

        else:
            # ② 정지/저이동 데드존: 이동 변화율 기반으로 steer=0
            ds = 0.0
            if len(cx) > 1:
                ds = distance_m(cx[-1], cy[-1], cx[-2], cy[-2])
            dist_to_target = distance_m(cx[-1], cy[-1], tx, ty)
            dz_th = max(float(params['steer_deadzone_move_ratio']) * max(dist_to_target, 1e-3),
                        float(params['steer_deadzone_move_min_m']))

            use_zero_steer = (ds < dz_th)

            # ③ 정지/재출발: 스냅샷/버퍼 기반 안정 헤딩 선택
            raw_heading = heading_rad
            if (raw_heading is None) or use_zero_steer or stopped_now:
                chosen_heading = select_stable_heading(bearing_to_target, raw_heading if raw_heading is not None else bearing_to_target)
                heading_for_ctrl = chosen_heading
                if use_zero_steer:
                    flags.append("DEADZONE")
                if stopped_now:
                    flags.append("STOP_HOLD")
            else:
                heading_for_ctrl = heading_rad

            # ④ 조향 계산
            angle = 0.0
            if not use_zero_steer:
                angle = steering_from_vectors(heading_for_ctrl, cx[-1], cy[-1], tx, ty)
            smooth_deg = apply_low_pass_filter(angle if not use_zero_steer else 0.0)
            smooth_deg = clamp(smooth_deg, -float(params['steer_limit_deg']), float(params['steer_limit_deg']))

            # ⑤ 속도 명령 계산(+상한/하드고정) 및 퍼블리시
            cmd_float, cmd_report, mode_str = compute_speed_command()
            publish_speed_cmd(cmd_float)
            publish_steer_deg(smooth_deg)
            _last_speed_cmd = cmd_report
            _last_speed_mode = mode_str

        # --- 반경 히스테리시스: 순차 인덱스 고정 ---
        dist_to_active = distance_m(cx[-1], cy[-1], tx, ty)
        if dist_to_active <= float(params['target_radius']):
            _radius_hit_consec += 1
        else:
            _radius_hit_consec = 0

        # 정상 모드에서만 인덱스 증가
        if (not _in_recovery) and (_radius_hit_consec >= int(params['radius_hit_count'])):
            if waypoint_index < len(waypoints_x) - 1:
                waypoint_index += 1
            _radius_hit_consec = 0

        # --- 이탈/복귀 판단 ---
        # 정상 모드에서 이탈 조건이면 복귀 타겟 선택
        if (not _in_recovery):
            if dist_to_active > float(params['offtrack_trigger_mult']) * float(params['target_radius']):
                cand = pick_recovery_target(cx[-1], cy[-1], heading_rad, waypoint_index)
                if cand is not None and cand > waypoint_index:
                    _in_recovery = True
                    _recovery_target_index = cand
                    flags.append(f"RECOVERY→{cand+1}")

        # 복귀 중이면 복귀 타겟 달성 시 정상 모드 복귀
        if _in_recovery:
            if dist_to_active <= float(params['target_radius']):
                waypoint_index = _recovery_target_index
                _in_recovery = False
                _recovery_target_index = None
                _radius_hit_consec = 0
                flags.append("RECOVERY_DONE")

        # 화살표 시각화
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

        # ── 로깅 ──
        if params['log_csv']:
            try:
                new = not os.path.exists(params['log_csv'])
                os.makedirs(os.path.dirname(params['log_csv']), exist_ok=True)
                with open(params['log_csv'], 'a', newline='') as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow([
                            'current_x','current_y','prev_x','prev_y',
                            'waypoint_idx','waypoint_x','waypoint_y',
                            'steer_deg','heading_deg',
                            'speed_cmd','speed_meas_ema',
                            'dist_to_target','time','rtk_status','speed_mode','flags'
                        ])
                    heading_deg = math.degrees(heading_rad) if heading_rad is not None else ''
                    log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    px, py = (cx[-2], cy[-2]) if len(cx) > 1 else ('','')
                    w.writerow([
                        cx[-1], cy[-1], px, py,
                        active_index+1, tx, ty,
                        smooth_deg, heading_deg,
                        _last_speed_cmd, v_meas,
                        dist_to_active, log_time, rtk_txt, _last_speed_mode, "|".join(flags)
                    ])
            except Exception as e:
                rospy.logwarn(f"[tracker_topics] log write failed: {e}")

        # ── 로그 출력 ──
        now = time.time()
        if now - _last_log_t > 0.5:
            latlon_txt = f"Lat: {_last_lat:.7f}, Lon: {_last_lon:.7f}" if (_last_lat is not None and _last_lon is not None) else "Lat/Lon: (n/a)"
            heading_deg = math.degrees(heading_rad) if heading_rad is not None else 0.0
            rospy.loginfo(f"{latlon_txt}, SpeedCmd({ _last_speed_mode })={_last_speed_cmd}, v(EMA)={v_meas:.2f} m/s, "
                          f"Steer={smooth_deg:+.2f} deg, Heading={heading_deg:.2f} deg, "
                          f"Dist→Target={dist_to_active:.2f} m, RTK={rtk_txt}, Flags={','.join(flags)}")
            _last_log_t = now

        # Info 박스
        info_lines.append(f"Veh: ({cx[-1]:.1f}, {cy[-1]:.1f}) m")
        info_lines.append(f"Dist→Target: {dist_to_active:.1f} m")
        if heading_rad is not None:
            info_lines.append(f"Heading: {math.degrees(heading_rad):.1f}°")
        info_lines.append(f"Steering: {smooth_deg:+.1f}°")
        info_lines.append(f"SpeedCmd({ _last_speed_mode }): {_last_speed_cmd}")
        if flags:
            info_lines.append(f"Flags: {' | '.join(flags)}")

    if info_lines:
        ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
                ha='left', va='top', fontsize=9, bbox=dict(fc='white', alpha=0.7))

    # 축/스타일
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

        # 속도 명령
        'speed_mode':       rospy.get_param('~speed_mode', SPEED_MODE_DEFAULT),
        'speed_code':       int(rospy.get_param('~speed_code', SPEED_CODE_DEFAULT)),
        'speed_cap_code':   int(rospy.get_param('~speed_cap_code', SPEED_CAP_CODE_DEFAULT)),
        'speed_cap_mps':    float(rospy.get_param('~speed_cap_mps', SPEED_CAP_MPS_DEFAULT)),

        # 헤딩 안정화
        'heading_min_speed': float(rospy.get_param('~heading_min_speed', HEADING_MIN_SPEED_DEFAULT)),
        'heading_alpha':     float(rospy.get_param('~heading_alpha', HEADING_ALPHA_DEFAULT)),
        'bootstrap_dist':    float(rospy.get_param('~bootstrap_dist', BOOTSTRAP_DIST_DEFAULT)),

        # 동적 게이팅
        'gate_floor_m':      float(rospy.get_param('~gate_floor_m', 0.5)),
        'gate_k_vel':        float(rospy.get_param('~gate_k_vel', 3.0)),
        'gate_k_sigma':      float(rospy.get_param('~gate_k_sigma', 3.0)),
        'gate_rtk_fix':      float(rospy.get_param('~gate_rtk_fix', 0.20)),
        'gate_rtk_float':    float(rospy.get_param('~gate_rtk_float', 0.50)),
        'gate_rtk_none':     float(rospy.get_param('~gate_rtk_none', 1.00)),

        # 시작 워밍업
        'start_warmup_sec':  float(rospy.get_param('~start_warmup_sec', START_WARMUP_SEC_DEFAULT)),

        # 정지/재출발
        'stop_speed_eps':    float(rospy.get_param('~stop_speed_eps', STOP_SPEED_EPS_DEFAULT)),
        'stop_hold_sec':     float(rospy.get_param('~stop_hold_sec', STOP_HOLD_SEC_DEFAULT)),
        'restart_min_speed': float(rospy.get_param('~restart_min_speed', RESTART_MIN_SPEED_DEFAULT)),

        # 조향 데드존
        'steer_deadzone_move_ratio': float(rospy.get_param('~steer_deadzone_move_ratio', STEER_DEADZONE_MOVE_RATIO_DEFAULT)),
        'steer_deadzone_move_min_m': float(rospy.get_param('~steer_deadzone_move_min_m', STEER_DEADZONE_MOVE_MIN_M_DEFAULT)),

        # 헤딩 후보 선택
        'head_buf_len':         int(rospy.get_param('~head_buf_len', HEAD_BUF_LEN_DEFAULT)),
        'head_select_max_deg':  float(rospy.get_param('~head_select_max_deg', HEAD_SELECT_MAX_DEG_DEF)),
        'head_good_ratio':      float(rospy.get_param('~head_good_ratio', HEAD_GOOD_RATIO_DEFAULT)),

        # 반경 히스테리시스
        'radius_hit_count':  int(rospy.get_param('~radius_hit_count', RADIUS_HIT_COUNT_DEFAULT)),

        # 이탈/복귀
        'offtrack_trigger_mult':   float(rospy.get_param('~offtrack_trigger_mult', OFFTRACK_TRIGGER_MULT_DEF)),
        'recovery_search_radius':  float(rospy.get_param('~recovery_search_radius', RECOVERY_SEARCH_RADIUS_DEF)),
        'recovery_lookahead_n':    int(rospy.get_param('~recovery_lookahead_n', RECOVERY_LOOKAHEAD_N_DEF)),
        'recovery_heading_limit_deg': float(rospy.get_param('~recovery_heading_limit_deg', RECOVERY_HEADING_LIMIT_DEG_DEF)),

        # 시각화
        'show_all_waypoints': bool(rospy.get_param('~show_all_waypoints', SHOW_ALL_WP_DEFAULT)),
        'win_size':           int(rospy.get_param('~win_size', WIN_SIZE_DEFAULT)),
        'axis_margin':        float(rospy.get_param('~axis_margin', AXIS_MARGIN_DEFAULT)),
    }

    # head buffer 크기 갱신
    global _head_buf
    _head_buf = deque(list(_head_buf), maxlen=int(params['head_buf_len']))

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
