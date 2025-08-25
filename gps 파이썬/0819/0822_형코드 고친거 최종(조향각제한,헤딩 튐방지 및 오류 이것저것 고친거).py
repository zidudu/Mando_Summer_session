#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
좋습니다. 핵심만 딱 정리드릴게요.

알고리즘 4줄 요약

GNSS /ublox/fix를 미터좌표로 변환 → 점프 제거 + LPF → 이동거리 가중 원형평균으로 헤딩 안정화(저속/정지 시 헤딩 프리즈 & 롤인).

웨이포인트는 리샘플링 후 사용, 룩어헤드 Ld=L0+kv·v + 전방 윈도우 + ±30° 각도 게이트로 타깃을 선택, 이탈 시 근방 후보 중 거리+각도 비용 최소로 복귀.

조향각 = (이동벡터, 타깃벡터) 사이각 → LPF → ±steer_limit_deg 클램프, 속도명령 = const_speed - k·|조향| → RTK 상태에 따라 속도 캡 적용.

RTK에 따라 도착반경 스케일링(FIX/FLOAT/NONE) 후 반경 내 진입 시 인덱스 전진, 실시간 시각화/로그.

코스별 “변수 대응” 정리
1) 출발할 때

변수: roll_in_m, v_stop_th, freeze_time_s, heading_min_move_m, heading_max_jump_deg, heading_hist_len

동작: 첫 Fix에서 경로 접선으로 초기 헤딩 설정 → roll_in_m 만큼은 그 헤딩 유지(직진 롤인). 속도 추정이 v_stop_th 이하로 freeze_time_s 지속되면 헤딩 프리즈로 튐 억제.

팁: 정지 후 재출발 튐이 보이면 roll_in_m↑ 또는 heading_min_move_m↑로 보수적으로 잡으시면 됩니다.

2) 직진 코스

변수: lookahead_L0, lookahead_kv, forward_window, steer_limit_deg, fc/fs(=LPF), gps_outlier_th

동작: 직진에선 룩어헤드가 앞쪽 점을 바로 선택 → 조향은 LPF + ±제한으로 미세 진동 억제.

팁: 직선에서 조향이 들쑥이면 fc↓(필터 강하게) 또는 lookahead_L0↑로 더 멀리 보게 하시면 안정적입니다.

3) 곡선 코스

변수: lookahead_L0, lookahead_kv, k_steer_slowdown, speed_min, steer_limit_deg

동작: 속도가 높을수록 Ld가 커져 부드럽게 진입, 조향 절대값이 커질수록 k_steer_slowdown에 의해 자동 감속되어 오버슈트 방지.

팁: 코너에서 언더/오버슈트가 보이면 kv↑(더 멀리)와 k_steer_slowdown↑(더 감속) 조합이 잘 먹힙니다.

4) 교차로 코스

변수: forward_window, rejoin_max_deg(≈30°), rejoin_search_r, rejoin_lambda, rejoin_expand_steps

동작: 타깃 선정 시 전방 윈도우 + ±각도 게이트로 옆 가지 진입 차단. 이탈 감지 시 반경 내 후보 중 거리 + λ·각도 최소비용으로 복귀 타깃 재지정.

팁: 옆길로 새면 rejoin_max_deg↓(예 25°) 또는 forward_window↓로 전방 제약을 더 강하게 주세요.

5) 정지 코스(신호대기 등)

변수: v_stop_th, freeze_time_s, roll_in_m, (RTK 연동) rtk_speed_cap_*, rtk_radius_scale_*

동작: 정지 판정되면 헤딩 프리즈로 드리프트 억제, 재출발 초반엔 롤인 헤딩 유지. RTK가 나빠지면 속도 캡/도착반경 확대로 보수 운행.

팁: 재출발 첫 1~2 m가 흔들리면 freeze_time_s↑와 roll_in_m↑를 함께 조정해 보세요.

제 의견(주관적)

현재 구조는 교차로 이탈·정지 후 튐 같은 실전 이슈를 파라미터로 잘 제어할 수 있게 설계돼서, 튜닝 난이도 대비 안정성이 좋습니다.

곡선 품질은 룩어헤드·감속 만으로도 충분히 좋아지지만, 추후 곡률 기반 속도 계획(lookahead 대신 곡률로 v 프로파일링)을 넣으면 더 매끈해질 겁니다.

교차로에서 아주 근접하게 가지가 겹치면, 여기에 차선 코리도어(횡오차 한계) 추가 게이트까지 넣으면 “옆길 방지”가 한층 단단해집니다.

센서 측면에선 GNSS만으로도 잘 돌아가지만, IMU 요레이트까지 섞으면 저속·정지 구간 헤딩 안정성은 체감적으로 더 올라갑니다.

waypoint_tracker_topics.py  (ROS1 Noetic, Ubuntu)
─────────────────────────────────────────────────────────────────────────────
통합 기능 (A~E, F 제외)
  A. 정지/재출발 헤딩 고정 + 롤인(roll-in) 직진 유지
  B. ±30° 조향 가능 게이트 + 경로 복귀(rejoin)
  C. 룩어헤드(Ld = L0 + k_v·v) 기반 타깃 선정 (전방 윈도우)
  D. 조향 절대값에 따른 자동 감속 (속도 명령 동적 조정)
  E. RTK 상태 연동 안전모드 (FIX/ FLOAT/ NONE에 따라 속도/도착반경 가변)

입력:
  /ublox/fix (sensor_msgs/NavSatFix)
  (옵션) /ublox/navrelposned (ublox_msgs/NavRELPOSNED9 or NavRELPOSNED)
  (옵션) /ublox/navpvt      (ublox_msgs/NavPVT)

출력:
  /vehicle/speed_cmd (std_msgs/Float32)  [m/s]
  /vehicle/steer_cmd (std_msgs/Float32)  [deg] (+좌/-우 or 차량 관례에 맞춰 부호 반전 적용)
  /rtk/status       (std_msgs/String)    ["FIX" | "FLOAT" | "NONE"]

시각화:
  Matplotlib(plt.ion) – 웨이포인트, 도착 반경, 현재 위치, 타깃, 헤딩/조향 화살표, 정보 박스

웨이포인트:
  CSV (Lat, Lon) → Web Mercator(m) 변환, 최소 간격 리샘플링 후 사용
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

# ── ublox_msgs(옵션) ──────────────────────────────────────
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

# ── 기본 파라미터 ─────────────────────────────────────────
TARGET_RADIUS_DEFAULT = 1.5          # [m] 도착 반경(기본)
MIN_WP_DIST_DEFAULT   = 0.9          # [m] 웨이포인트 최소 간격
FC_DEFAULT            = 2.0          # [Hz] 위치/조향 LPF 컷오프
FS_DEFAULT            = 20.0         # [Hz] 루프 주기 (50ms)
GPS_OUTLIER_TH_DEF    = 1.0          # [m] GPS 점프 무시 임계
STEER_LIMIT_DEG_DEF   = 20.0         # [deg] 조향각 제한
CONST_SPEED_DEFAULT   = 1.0          # [m/s] 기준 속도

# (C) 룩어헤드/전방 윈도우/각도 게이트
LOOK_L0_DEFAULT       = 1.2          # [m] L0
LOOK_KV_DEFAULT       = 0.6          # [s] k_v (v[m/s] 곱)
FWD_WINDOW_DEFAULT    = 30           # [pts] 전방 탐색 인덱스 폭
REJOIN_MAX_DEG_DEF    = 20.0         # [deg] 조향 가능 각

# (B) 복귀 탐색
REJOIN_SRCH_R_DEF     = 5.0          # [m] 반경 내 후보 검색
REJOIN_LAMBDA_DEF     = 0.05         # 비용 가중(각도)
REJOIN_EXPAND_STEPS   = 1            # 반경 확장 횟수(간단히 1회)

# (A) 정지/재출발 & 롤인
ROLL_IN_M_DEFAULT     = 1.0          # [m] 롤인 거리(초기 헤딩 유지)
V_STOP_TH_DEFAULT     = 0.08         # [m/s] 정지 판정 속도
FREEZE_TIME_S_DEFAULT = 0.7          # [s] 연속 정지 시간 → 헤딩 프리즈

# (D) 조향 기반 자동 감속
SPEED_MIN_DEFAULT     = 0.25         # [m/s] 최저 속도
K_STEER_SLOW_DEF      = 0.02         # [m/s/deg] |δ|당 감속량

# (E) RTK 안전모드
RTK_CAP_FLOAT_DEF     = 0.6          # [m/s] FLOAT 속도 캡
RTK_CAP_NONE_DEF      = 0.35         # [m/s] NONE  속도 캡
RTK_R_SCALE_FLOAT_DEF = 1.3          # 도착반경 배율(FLOAT)
RTK_R_SCALE_NONE_DEF  = 1.7          # 도착반경 배율(NONE)

# 헤딩 안정화
HEADING_MIN_MOVE_DEF  = 0.07         # [m] 헤딩 갱신 최소 이동
HEADING_HIST_LEN_DEF  = 5            # [seg] 원형평균 히스토리 길이
HEADING_MAX_JUMP_DEF  = 35.0         # [deg] 작은 이동에서 급점프 거부 임계

# ── 경로/로그 기본 경로 ───────────────────────────────────
def _default_paths():
    try:
        pkg = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        pkg = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
    cfg = os.path.join(pkg, 'config')
    wp  = os.path.join(cfg, 'left_lane.csv')
    log = os.path.join(cfg, f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    return cfg, wp, log

CFG_DIR_DEFAULT, WAYPOINT_CSV_DEFAULT, LOG_CSV_DEFAULT = _default_paths()

# ── 런타임 상태 ───────────────────────────────────────────
params = {}
pub_speed = None
pub_steer = None
pub_rtk   = None

current_x, current_y = [], []
waypoints_x = None
waypoints_y = None
waypoint_index = 0

# LPF (좌표/조향)
alpha = 0.56                    # 위치/조향 필터 계수(런타임 재계산)
_filtered_steering = 0.0

# GPS 필터 상태
_prev_raw_x = _prev_raw_y = None
_prev_f_x   = _prev_f_y   = None

# 헤딩/속도 추정
_last_lat = _last_lon = None
_last_fix_heading_rad = None
heading_hist = deque(maxlen=HEADING_HIST_LEN_DEF)
_last_log_t = 0.0

_last_fix_t = None
_est_speed  = 0.0              # [m/s] 추정 속도(간단 LPF)

# A: 정지/재출발 + 롤인
_start_pos = None
_roll_in_done = False
_frozen_heading = None
_moved_since_start = 0.0
_below_v_since_ts = None        # 연속 정지 구간 시작 시각
_heading_frozen = False         # 프리즈 상태

# RTK
rtk_status_txt = "NONE"

_state_lock = threading.Lock()

# ── 유틸 ────────────────────────────────────────────────
def latlon_to_meters(lat, lon):
    """WGS84 → Web Mercator (x,y)[m]"""
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def distance_m(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def _ang_wrap(rad):  return (rad + math.pi) % (2*math.pi) - math.pi
def _ang_diff(a, b): return _ang_wrap(a - b)

def _circ_mean_from_hist(hist):
    if not hist: return None
    s = c = 0.0
    for h, w in hist:
        s += w * math.sin(h); c += w * math.cos(h)
    if s == 0.0 and c == 0.0: return None
    return math.atan2(s, c)

# ── 좌표 필터: 아웃라이어 + 1차 IIR ─────────────────────
def filter_gps_signal(x, y):
    global _prev_raw_x, _prev_raw_y, _prev_f_x, _prev_f_y, alpha
    th = float(params['gps_outlier_th'])

    if _prev_raw_x is not None:
        if distance_m(_prev_raw_x, _prev_raw_y, x, y) > th:
            x, y = _prev_raw_x, _prev_raw_y
        else:
            _prev_raw_x, _prev_raw_y = x, y
    else:
        _prev_raw_x, _prev_raw_y = x, y

    if _prev_f_x is None:
        _prev_f_x, _prev_f_y = x, y

    fx = (1 - alpha) * _prev_f_x + alpha * x
    fy = (1 - alpha) * _prev_f_y + alpha * y
    _prev_f_x, _prev_f_y = fx, fy
    return fx, fy

# ── 조향각 계산 + LPF ──────────────────────────────────
def calculate_steering_angle(v1, v2):
    """두 벡터 사이 각(부호 포함, deg). 좌=+, 우=- (차량 관례에 맞춰 부호 반전은 별도 처리)"""
    v1 = np.asarray(v1, float); v2 = np.asarray(v2, float)
    n1 = np.linalg.norm(v1);    n2 = np.linalg.norm(v2)
    TH = 0.05  # 5 cm
    if n1 < TH or n2 < TH: return 0.0

    c = float(np.dot(v1, v2)) / max(1e-9, n1*n2)
    c = max(min(c, 1.0), -1.0)
    ang = math.degrees(math.acos(c))

    cross = v1[0]*v2[1] - v1[1]*v2[0]
    if cross < 0: ang = -ang

    return max(min(ang / 1.3, STEER_LIMIT_DEG_DEF), -STEER_LIMIT_DEG_DEF)

def apply_low_pass_filter(current_deg):
    """조향각 LPF + 차량 관례 부호 반전"""
    global _filtered_steering, alpha
    _filtered_steering = (1 - alpha) * _filtered_steering + alpha * current_deg
    return _filtered_steering * -1.0

# ── 웨이포인트 로드 ─────────────────────────────────────
def load_waypoints(path_csv, min_wp_dist):
    df = pd.read_csv(path_csv)
    coords = [latlon_to_meters(row['Lat'], row['Lon']) for _, row in df.iterrows()]
    if len(coords) < 1: raise RuntimeError("waypoints csv empty")

    fx = [float(coords[0][0])]; fy = [float(coords[0][1])]
    for xi, yi in coords[1:]:
        if distance_m(fx[-1], fy[-1], xi, yi) >= min_wp_dist:
            fx.append(float(xi)); fy.append(float(yi))
    return np.array(fx), np.array(fy)

# ── C: 룩어헤드 + 전방 윈도우 + 각도 게이트 ──────────────
def pick_target_index(cur_idx, cx, cy, heading_rad, v_cmd):
    """전방 fwd_window 내에서 Ld 이상 첫 점, 각도 게이트 만족점 선택"""
    L0 = float(params.get('lookahead_L0', LOOK_L0_DEFAULT))
    kv = float(params.get('lookahead_kv', LOOK_KV_DEFAULT))
    fwd = int(params.get('forward_window', FWD_WINDOW_DEFAULT))
    max_deg = float(params.get('rejoin_max_deg', REJOIN_MAX_DEG_DEF))

    Ld = L0 + kv * max(0.0, float(v_cmd))
    j0 = cur_idx
    jmax = min(j0 + fwd, len(waypoints_x) - 1)
    for j in range(j0 + 1, jmax + 1):
        dx = waypoints_x[j] - cx; dy = waypoints_y[j] - cy
        d  = math.hypot(dx, dy)
        if d < Ld: 
            continue
        if heading_rad is not None:
            dpsi = abs(_ang_diff(math.atan2(dy, dx), heading_rad))
            if math.degrees(dpsi) > max_deg:
                continue
        return j
    return min(j0 + 1, len(waypoints_x) - 1)

# ── B: 경로 복귀 후보 선택 ──────────────────────────────
def pick_rejoin_waypoint(cx, cy, yaw, search_r=REJOIN_SRCH_R_DEF, max_deg=REJOIN_MAX_DEG_DEF, lam=REJOIN_LAMBDA_DEF):
    cand = []
    cos_y = math.cos(yaw); sin_y = math.sin(yaw)
    for i, (x, y) in enumerate(zip(waypoints_x, waypoints_y)):
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy)
        if d > search_r:
            continue
        # 전방성
        if (cos_y*dx + sin_y*dy) <= 0.0:
            continue
        ang_to = math.atan2(dy, dx)
        ddeg   = abs(math.degrees(_ang_diff(ang_to, yaw)))
        if ddeg <= max_deg:
            cost = d + lam * ddeg
            cand.append((cost, i))
    return min(cand)[1] if cand else None

# ── E: RTK 상태에 따른 도착 반경/속도 캡 ────────────────
def effective_radius(base_r):
    if rtk_status_txt == "FIX":
        return base_r
    if rtk_status_txt == "FLOAT":
        return base_r * float(params.get('rtk_radius_scale_float', RTK_R_SCALE_FLOAT_DEF))
    return base_r * float(params.get('rtk_radius_scale_none', RTK_R_SCALE_NONE_DEF))

def cap_speed_for_rtk(v):
    if rtk_status_txt == "FIX":
        return v
    if rtk_status_txt == "FLOAT":
        return min(v, float(params.get('rtk_speed_cap_float', RTK_CAP_FLOAT_DEF)))
    return min(v, float(params.get('rtk_speed_cap_none', RTK_CAP_NONE_DEF)))

# ── 초기 헤딩(경로 접선) ─────────────────────────────────
def nearest_wp_index(px, py):
    dmin, imin = 1e18, 0
    for i, (x, y) in enumerate(zip(waypoints_x, waypoints_y)):
        d = (x - px)*(x - px) + (y - py)*(y - py)
        if d < dmin:
            dmin, imin = d, i
    return imin

def path_tangent(i):
    j = min(i + 1, len(waypoints_x) - 1)
    dx = waypoints_x[j] - waypoints_x[i]
    dy = waypoints_y[j] - waypoints_y[i]
    if dx == 0.0 and dy == 0.0 and j + 1 < len(waypoints_x):
        dx = waypoints_x[j + 1] - waypoints_x[j]
        dy = waypoints_y[j + 1] - waypoints_y[j]
    return math.atan2(dy, dx)

# ── ROS 퍼블리시 ────────────────────────────────────────
def publish_speed(v_mps):
    if pub_speed: pub_speed.publish(Float32(data=float(v_mps)))

def publish_steer_deg(deg):
    deg = clamp(float(deg), -float(params['steer_limit_deg']), float(params['steer_limit_deg']))
    if pub_steer: pub_steer.publish(Float32(data=deg))

def publish_rtk(txt):
    if pub_rtk: pub_rtk.publish(String(data=str(txt)))

# ── ROS 콜백: NavSatFix ─────────────────────────────────
def _cb_fix(msg: NavSatFix):
    global _last_lat, _last_lon, _last_fix_t, _est_speed
    global _last_fix_heading_rad, heading_hist
    global _start_pos, _roll_in_done, _frozen_heading, _moved_since_start
    global _below_v_since_ts, _heading_frozen

    if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
        return

    lat, lon = float(msg.latitude), float(msg.longitude)
    _last_lat, _last_lon = lat, lon
    x, y = latlon_to_meters(lat, lon)
    fx, fy = filter_gps_signal(x, y)

    now = time.time()
    # 속도 추정
    if _last_fix_t is None:
        _last_fix_t = now

    seg_len = 0.0
    if current_x and current_y:
        seg_len = math.hypot(fx - current_x[-1], fy - current_y[-1])
        dt = max(1e-3, now - _last_fix_t)
        v_inst = seg_len / dt
        # 간단 속도 LPF (알파 고정: 0.2)
        _est_speed = 0.8 * _est_speed + 0.2 * v_inst
        _last_fix_t = now

    # A: 초기화/롤인/정지 프리즈 상태 관리
    if _start_pos is None:
        _start_pos = (fx, fy)
        # 초기 헤딩 = 경로 접선
        i0 = nearest_wp_index(fx, fy)
        _frozen_heading = path_tangent(i0)

    if current_x and current_y:
        _moved_since_start += seg_len

    # 정지 판정 (연속)
    v_stop_th = float(params.get('v_stop_th', V_STOP_TH_DEFAULT))
    freeze_time = float(params.get('freeze_time_s', FREEZE_TIME_S_DEFAULT))
    if _est_speed < v_stop_th:
        if _below_v_since_ts is None:
            _below_v_since_ts = now
        elif (now - _below_v_since_ts) >= freeze_time:
            _heading_frozen = True
    else:
        _below_v_since_ts = None
        # 재출발 직후 일정 거리까지는 초기/마지막 안정 헤딩 유지(롤인)
        if _roll_in_done is False and _moved_since_start >= float(params.get('roll_in_m', ROLL_IN_M_DEFAULT)):
            _roll_in_done = True
        # 정지 프리즈 해제는 롤인 여부와 무관하게, 속도가 충분히 오르면 해제
        if _est_speed >= v_stop_th * 1.2:
            _heading_frozen = False

    # 헤딩 안정화 (세그먼트 길이 가중 원형평균)
    with _state_lock:
        if current_x and current_y:
            dx = fx - current_x[-1]; dy = fy - current_y[-1]
            if seg_len > float(params['heading_min_move_m']):
                seg_heading = math.atan2(dy, dx)

                stable_before = _circ_mean_from_hist(heading_hist)
                accept = True
                if stable_before is not None:
                    jump_deg = abs(math.degrees(_ang_diff(seg_heading, stable_before)))
                    if (jump_deg > float(params['heading_max_jump_deg'])
                        and seg_len < 3.0 * float(params['heading_min_move_m'])):
                        accept = False

                if accept:
                    heading_hist.append((seg_heading, seg_len))

                stable_after = _circ_mean_from_hist(heading_hist)
                _last_fix_heading_rad = stable_after if stable_after is not None else seg_heading
        current_x.append(fx); current_y.append(fy)

# ── ROS 콜백: RTK 상태 ──────────────────────────────────
def _cb_relpos(msg):
    global rtk_status_txt
    try:
        mask  = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_MASK'))
        fixed = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_FIXED'))
        flt   = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_FLOAT'))
        bits  = int(msg.flags) & mask
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

# ── 시각화/메인 루프 ────────────────────────────────────
def update_plot_once(ax):
    """한 프레임 업데이트(플롯/퍼블리시/로깅)"""
    global waypoint_index, _last_log_t

    ax.clear()
    # 스냅샷
    with _state_lock:
        cx = list(current_x); cy = list(current_y)
        stable_heading = _last_fix_heading_rad

    if len(cx) >= 2:
        ax.plot(cx, cy, '-', c='0.6', lw=1.0, label='Route')

    if waypoints_x is None: return

    # A: 롤인/프리즈 적용된 표시 헤딩 결정
    heading_rad = stable_heading
    if (_heading_frozen or not _roll_in_done) and (_frozen_heading is not None):
        heading_rad = _frozen_heading

    # 현재 위치
    smooth_deg = 0.0
    info_lines = []
    eff_radius = effective_radius(float(params['target_radius']))

    # 전방 윈도우 표시
    window_size = int(params.get('forward_window', FWD_WINDOW_DEFAULT))
    start_index = (waypoint_index // window_size) * window_size
    end_index   = min(start_index + window_size, len(waypoints_x))

    ax.scatter(waypoints_x[start_index:end_index], waypoints_y[start_index:end_index],
               color='blue', s=10, label='Waypoints')
    for i in range(start_index, end_index):
        c = Circle((waypoints_x[i], waypoints_y[i]), eff_radius,
                   fill=False, linestyle='--', edgecolor='tab:blue', alpha=0.3)
        ax.add_patch(c)
        ax.text(waypoints_x[i], waypoints_y[i], str(i + 1), fontsize=8, ha='center')

    # 타깃 선정/조향/속도
    if cx and cy:
        ax.scatter(cx[-1], cy[-1], color='red', s=50, label='Current')

        # D: 조향 기반 속도 감속 (기본 속도 기준으로 일단 계산 후 RTK 캡 적용)
        base_v   = float(params['const_speed'])
        k_slow   = float(params.get('k_steer_slowdown', K_STEER_SLOW_DEF))
        v_min    = float(params.get('speed_min', SPEED_MIN_DEFAULT))

        # C: 룩어헤드 타깃 선정 – 예상 v는 직전 추정 속도 사용
        v_for_ld = max(_est_speed, 0.0)

        target_idx = pick_target_index(waypoint_index, cx[-1], cy[-1], heading_rad, v_for_ld)
        tx, ty = waypoints_x[target_idx], waypoints_y[target_idx]

        # 오프코스 탐지 → B: 복귀 타깃
        need_rejoin = False
        if heading_rad is not None:
            ang_to = math.atan2(ty - cy[-1], tx - cx[-1])
            ddeg   = abs(math.degrees(_ang_diff(ang_to, heading_rad)))
            if ddeg > float(params.get('rejoin_max_deg', REJOIN_MAX_DEG_DEF)) + 5.0:
                need_rejoin = True

        if need_rejoin:
            sr = float(params.get('rejoin_search_r', REJOIN_SRCH_R_DEF))
            lam = float(params.get('rejoin_lambda', REJOIN_LAMBDA_DEF))
            cand = pick_rejoin_waypoint(cx[-1], cy[-1], heading_rad, sr, float(params.get('rejoin_max_deg', REJOIN_MAX_DEG_DEF)), lam)
            if cand is None and int(params.get('rejoin_expand_steps', REJOIN_EXPAND_STEPS)) > 0:
                cand = pick_rejoin_waypoint(cx[-1], cy[-1], heading_rad, sr*1.6, float(params.get('rejoin_max_deg', REJOIN_MAX_DEG_DEF)), lam)
            if cand is not None:
                target_idx = cand
                tx, ty = waypoints_x[target_idx], waypoints_y[target_idx]

        # 타깃 시각화
        ax.plot([cx[-1], tx], [cy[-1], ty], '--', c='cyan', lw=1.0, label='Target Line')
        ax.plot(tx, ty, '*', c='magenta', ms=12, label='Target')

        # 조향 계산
        if len(cx) > 1:
            move_vec   = (cx[-1] - cx[-2], cy[-1] - cy[-2])
            target_vec = (tx - cx[-1], ty - cy[-1])
            raw_deg    = calculate_steering_angle(move_vec, target_vec)
            smooth_deg = apply_low_pass_filter(raw_deg)
            smooth_deg = clamp(smooth_deg, -float(params['steer_limit_deg']), float(params['steer_limit_deg']))
        else:
            target_vec = ('', '')

        # 헤딩/조향 화살표
        L = 2.0
        if heading_rad is not None:
            hx, hy = cx[-1] + L*math.cos(heading_rad), cy[-1] + L*math.sin(heading_rad)
            ax.add_patch(FancyArrowPatch((cx[-1],cy[-1]), (hx,hy), color='tab:blue', lw=2,
                                         arrowstyle='-|>', mutation_scale=15, label='Heading'))
            steer_rad = math.radians(smooth_deg)
            sx, sy = cx[-1] + L*math.cos(heading_rad + steer_rad), cy[-1] + L*math.sin(heading_rad + steer_rad)
            ax.add_patch(FancyArrowPatch((cx[-1],cy[-1]), (sx,sy), color='red', lw=2, alpha=0.9,
                                         arrowstyle='-|>', mutation_scale=15, label='Steering'))

        # D: 조향 기반 자동 감속 → E: RTK 캡 → 최종 속도
        v_cmd = max(v_min, base_v - k_slow * abs(smooth_deg))
        v_cmd = cap_speed_for_rtk(v_cmd)

        # 퍼블리시
        publish_steer_deg(smooth_deg)
        publish_speed(v_cmd)

        # 도착 반경(효과 반영) → 인덱스 갱신 (앞으로만)
        if distance_m(cx[-1], cy[-1], tx, ty) < eff_radius:
            waypoint_index = min(max(waypoint_index, target_idx), len(waypoints_x) - 1)

        # 로그/정보
        now = time.time()
        if now - _last_log_t > 0.5:
            latlon_txt = f"Lat: {_last_lat:.7f}, Lon: {_last_lon:.7f}" if (_last_lat is not None and _last_lon is not None) else "Lat/Lon: (n/a)"
            heading_deg = math.degrees(heading_rad) if heading_rad is not None else 0.0
            dist_to_target = distance_m(cx[-1], cy[-1], tx, ty)
            rospy.loginfo(f"{latlon_txt}, v: {_est_speed:.2f} m/s → cmd: {v_cmd:.2f} m/s, "
                          f"Steer: {smooth_deg:+.1f}°, Heading: {heading_deg:.1f}°, "
                          f"Dist→T: {dist_to_target:.2f} m, RTK: {rtk_status_txt}")
            _last_log_t = now

        # CSV 로깅(옵션)
        if params['log_csv']:
            try:
                new = not os.path.exists(params['log_csv'])
                os.makedirs(os.path.dirname(params['log_csv']), exist_ok=True)
                with open(params['log_csv'], 'a', newline='') as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow([
                            'current_x','current_y','prev_x','prev_y',
                            'target_x','target_y','target_idx',
                            'steer_deg','heading_deg','v_est','v_cmd',
                            'dist_to_target','rtk_status','time'
                        ])
                    heading_deg = math.degrees(heading_rad) if heading_rad is not None else ''
                    dist_to_target = distance_m(cx[-1], cy[-1], tx, ty)
                    if len(cx) > 1:
                        w.writerow([cx[-1], cy[-1], cx[-2], cy[-2],
                                    tx, ty, target_idx,
                                    smooth_deg, heading_deg, _est_speed, v_cmd,
                                    dist_to_target, rtk_status_txt,
                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])
                    else:
                        w.writerow([cx[-1], cy[-1], '', '',
                                    tx, ty, target_idx,
                                    smooth_deg, heading_deg, _est_speed, v_cmd,
                                    dist_to_target, rtk_status_txt,
                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())])

        # Info Box
        info_lines.append(f"Veh: ({cx[-1]:.1f}, {cy[-1]:.1f}) m")
        info_lines.append(f"Heading: {math.degrees(heading_rad):.1f}°" if heading_rad is not None else "Heading: n/a")
        info_lines.append(f"Steer: {smooth_deg:+.1f}°")
        info_lines.append(f"v_est→cmd: {_est_speed:.2f}→{v_cmd:.2f} m/s")
        info_lines.append(f"RTK: {rtk_status_txt}")

    if info_lines:
        ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
                ha='left', va='top', fontsize=9, bbox=dict(fc='white', alpha=0.7))

    ax.set_title("RTK Waypoint Tracker (A~E integrated)")
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.axis('equal'); ax.grid(True, ls=':', alpha=0.5)
    ax.legend(loc='upper right')

# ── 메인 ────────────────────────────────────────────────
def main():
    global params, pub_speed, pub_steer, pub_rtk, waypoints_x, waypoints_y, alpha, heading_hist

    rospy.init_node('waypoint_tracker_topics', anonymous=False)

    ublox_ns = rospy.get_param('~ublox_ns', '/ublox')
    params = {
        'fix_topic'       : rospy.get_param('~fix_topic',    ublox_ns + '/fix'),
        'relpos_topic'    : rospy.get_param('~relpos_topic', ublox_ns + '/navrelposned'),
        'navpvt_topic'    : rospy.get_param('~navpvt_topic', ublox_ns + '/navpvt'),
        'waypoint_csv'    : rospy.get_param('~waypoint_csv', WAYPOINT_CSV_DEFAULT),
        'target_radius'   : float(rospy.get_param('~target_radius', TARGET_RADIUS_DEFAULT)),
        'min_wp_distance' : float(rospy.get_param('~min_wp_distance', MIN_WP_DIST_DEFAULT)),
        'fc'              : float(rospy.get_param('~fc', FC_DEFAULT)),
        'fs'              : float(rospy.get_param('~fs', FS_DEFAULT)),
        'gps_outlier_th'  : float(rospy.get_param('~gps_outlier_th', GPS_OUTLIER_TH_DEF)),
        'steer_limit_deg' : float(rospy.get_param('~steer_limit_deg', STEER_LIMIT_DEG_DEF)),
        'const_speed'     : float(rospy.get_param('~const_speed', CONST_SPEED_DEFAULT)),
        'log_csv'         : rospy.get_param('~log_csv', LOG_CSV_DEFAULT),

        # A: 롤인/정지 프리즈
        'roll_in_m'       : float(rospy.get_param('~roll_in_m', ROLL_IN_M_DEFAULT)),
        'v_stop_th'       : float(rospy.get_param('~v_stop_th', V_STOP_TH_DEFAULT)),
        'freeze_time_s'   : float(rospy.get_param('~freeze_time_s', FREEZE_TIME_S_DEFAULT)),

        # C: 룩어헤드/전방 윈도우/각도 게이트
        'lookahead_L0'    : float(rospy.get_param('~lookahead_L0', LOOK_L0_DEFAULT)),
        'lookahead_kv'    : float(rospy.get_param('~lookahead_kv', LOOK_KV_DEFAULT)),
        'forward_window'  : int(rospy.get_param('~forward_window', FWD_WINDOW_DEFAULT)),
        'rejoin_max_deg'  : float(rospy.get_param('~rejoin_max_deg', REJOIN_MAX_DEG_DEF)),

        # B: 복귀 탐색
        'rejoin_search_r' : float(rospy.get_param('~rejoin_search_r', REJOIN_SRCH_R_DEF)),
        'rejoin_lambda'   : float(rospy.get_param('~rejoin_lambda', REJOIN_LAMBDA_DEF)),
        'rejoin_expand_steps': int(rospy.get_param('~rejoin_expand_steps', REJOIN_EXPAND_STEPS)),

        # D: 조향 기반 감속
        'speed_min'       : float(rospy.get_param('~speed_min', SPEED_MIN_DEFAULT)),
        'k_steer_slowdown': float(rospy.get_param('~k_steer_slowdown', K_STEER_SLOW_DEF)),

        # E: RTK 안전모드
        'rtk_speed_cap_float' : float(rospy.get_param('~rtk_speed_cap_float', RTK_CAP_FLOAT_DEF)),
        'rtk_speed_cap_none'  : float(rospy.get_param('~rtk_speed_cap_none', RTK_CAP_NONE_DEF)),
        'rtk_radius_scale_float': float(rospy.get_param('~rtk_radius_scale_float', RTK_R_SCALE_FLOAT_DEF)),
        'rtk_radius_scale_none' : float(rospy.get_param('~rtk_radius_scale_none', RTK_R_SCALE_NONE_DEF)),

        # 헤딩 안정화
        'heading_min_move_m': float(rospy.get_param('~heading_min_move_m', HEADING_MIN_MOVE_DEF)),
        'heading_hist_len'  : int(rospy.get_param('~heading_hist_len', HEADING_HIST_LEN_DEF)),
        'heading_max_jump_deg': float(rospy.get_param('~heading_max_jump_deg', HEADING_MAX_JUMP_DEF)),
    }

    # LPF 계수
    alpha = (2 * math.pi * params['fc']) / (2 * math.pi * params['fc'] + params['fs'])

    # 헤딩 히스토리 길이 적용
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

    # 구독
    rospy.Subscriber(params['fix_topic'], NavSatFix, _cb_fix, queue_size=100)
    if _HAVE_RELPOSNED:
        rospy.Subscriber(params['relpos_topic'], NavRELPOSNED, _cb_relpos, queue_size=50)
    if _HAVE_NAVPVT:
        rospy.Subscriber(params['navpvt_topic'], NavPVT, queue_size=50)

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
