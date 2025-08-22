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
→ 대략 초당 2번 정도 이상의 빠른 변동(고주파 잡음)은 걸러지고, 그보다 느린 움직임은 통과시킨다는 뜻입니다.

FS_DEFAULT = 10.0 → 10 Hz 샘플링 주파수
→ 코드 루프(rospy.Rate(FS_DEFAULT))와 필터 계산 주기가 초당 10번(= 0.1 초 간격) 실행된다는 의미입니다.

즉, 10Hz(100ms)로 GPS 데이터를 받아오면서, 그 신호에 2Hz 컷오프 필터를 적용해서 위치/조향 계산을 안정화시키는 구조입니다.
'''

# 패키지/config 기본 경로 계산
def _default_paths():
    try:
        # 1. rtk_waypoint_tracker 패키지의 절대경로 찾기
        pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        # 2. 실패하면 ~/catkin_ws/src/... 경로로 fallback
        pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')

    # 3. config 폴더 경로
    cfg = os.path.join(pkg_path, 'config')

    # 4. 웨이포인트 기본 CSV (left_lane.csv)
    wp = os.path.join(cfg, 'left_lane.csv')

    # 5. 로그 파일 (waypoint_log_날짜_시간.csv)
    log = os.path.join(cfg, f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    return cfg, wp, log


# 전역 변수로 기본 경로 저장
CFG_DIR_DEFAULT, WAYPOINT_CSV_DEFAULT, LOG_CSV_DEFAULT = _default_paths()

# ── 런타임 상태 ─────────────────────────────────────
params = {}              # 런타임 파라미터(ROS param에서 읽어온 값 저장)
pub_speed = None         # /vehicle/speed_cmd 퍼블리셔 핸들
pub_steer = None         # /vehicle/steer_cmd 퍼블리셔 핸들
pub_rtk   = None         # /rtk/status 퍼블리셔 핸들

current_x, current_y = [], []   # 차량의 이동 경로 (좌표 기록 리스트)
waypoints_x = None              # 웨이포인트 X 좌표 배열
waypoints_y = None              # 웨이포인트 Y 좌표 배열
waypoint_index = 0              # 현재 타겟으로 삼고 있는 웨이포인트 인덱스

# alpha: 필터 계수. 실제로는 아래 공식으로 런타임에서 다시 계산됨.
# alpha = (2π * fc) / (2π * fc + fs)
alpha = 0.56             # LPF 계수 (Low Pass Filter, 런타임에서 계산해서 업데이트됨)
_filtered_steering = 0.0 # 필터링된 조향각 누적값

#GPS 신호를 필터링할 때 쓰는 "이전 상태" 저장 변수.
#_prev_raw_x, _prev_raw_y: 마지막으로 수신한 원시 GPS 좌표.
# _prev_f_x, _prev_f_y: 마지막으로 필터링된 좌표.
_prev_raw_x = None
_prev_raw_y = None
_prev_f_x = None
_prev_f_y = None

_last_lat = None #마지막으로 수신한 위도/경도 값.
_last_lon = None 
rtk_status_txt = "NONE" #현재 RTK 상태 문자열 (FIX / FLOAT / NONE).
_state_lock = threading.Lock() #다중 쓰레드에서 상태를 안전하게 접근하기 위한 락 (ROS 콜백과 메인 루프 동시에 접근할 수 있음).
_last_log_t = 0.0           # 터미널 로그 간격 제어

# ── 유틸, 십진수 위도·경도로 변환하는 함수─────────────────────────────────────────────
def dm_to_dec(dm, direction):
    try:
        d = int(float(dm) / 100)          # 앞의 "도(degree)" 부분
        m = float(dm) - d * 100           # 뒤의 "분(minute)" 부분
        dec = d + m / 60.0                # 도 + (분/60) → 십진수 도(degree)로 변환
        return -dec if direction in ['S', 'W'] else dec  # 남위(S), 서경(W)이면 음수 처리
    except Exception:
        return None

# 위경도를 Web Mercator 투영 좌표계로 변환 , 단위는 미터
def latlon_to_meters(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y
# 단순한 직선 거리 계산 함수, 웨이포인트까지의 거리, GPS 이상치 체크 등에 활용
def distance_m(x1, y1, x2, y2): #두 점 (x1, y1) 과 (x2, y2) (미터 단위)
    return math.hypot(x2 - x1, y2 - y1)# √((Δx)^2 + (Δy)^2) # 두 점 사이의 유클리드 거리 [m]

# 두 벡터의 각도를 계산해서 조향각(deg) 산출
def calculate_steering_angle(v1, v2):
    # v1: 차량의 현재 이동 방향 벡터, v2: 현재 위치 → 타겟 웨이포인트 벡터
    v1 = np.asarray(v1, dtype=float); v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)  # 두 벡터의 길이
    TH = 0.05 # 5cm
    # 벡터가 너무 짧으면 (거의 정지 상태) → 조향각 0
    if n1 < TH or n2 < TH:   
        return 0.0

    dot = float(np.dot(v1, v2))            # 내적 → 각도 계산용
    c = max(min(dot / (n1 * n2), 1.0), -1.0)  # cosθ 값 (범위 보정 -1~1)
    ang = math.degrees(math.acos(c))      # 벡터 간 각도 (0~180°)

    cross = v1[0]*v2[1] - v1[1]*v2[0]     # 외적 (2D에서 방향성 판단)
    if cross < 0: ang = -ang              # 음수 → 오른쪽 조향, 양수 → 왼쪽 조향

    # 조향각 제한 및 감쇠
    ang = max(min(ang / 1.3, 20.0), -20.0)  # 1.3으로 나누어 민감도 완화, ±25° 제한

    # 만약 각도가 크지만 이동이 거의 없는 경우 → 무효화
    if abs(ang) > 20.0 and (n1 < TH or n2 < TH):
        return 0.0

    return ang

def apply_low_pass_filter(current): #새로 계산된 조향각
    #_filtered_steering: 직전 스텝의 필터링된 조향각
    #alpha: LPF 계수 (fc, fs로부터 런타임에서 계산됨)
    # 공식 : yt​=(1−α)yt−1​+αxt​
    # 고주파(튀는 값) 제거, 부드럽게 조향각 변화.
    # 마지막에 * -1.0: 조향 부호 방향을 차량 시스템 관례에 맞게 반전.
    global _filtered_steering, alpha
    # 1차 IIR 저역통과 필터
    _filtered_steering = (1 - alpha) * _filtered_steering + alpha * current
    return _filtered_steering * -1.0  # 부호 반전 (차량 관례 맞추기)

# Outlier 제거: 이전 좌표와 1m 이상 튀면 → 무시하고 이전 값 유지. (0821 문제원인. 내가 손으로 들고 이동하는데 빠르게 이동하다보니 이게 현재값 무시하고 이전값 유지하게 되는 거임)
# LPF 적용: 좌표를 저역통과 필터로 부드럽게 만들어서 노이즈 감소
# 👉 GPS 신호의 순간 점프와 잡음을 동시에 완화.
def filter_gps_signal(x, y): # 새 GPS 좌표 (x, y)
    global _prev_raw_x, _prev_raw_y, _prev_f_x, _prev_f_y, alpha
    
    # 1. 아웃라이어 제거 (Outlier filtering)
    if _prev_raw_x is not None and _prev_raw_y is not None:
        if distance_m(_prev_raw_x, _prev_raw_y, x, y) > float(params['gps_outlier_th']):
            # 이전 점과 1m 이상 튀면 (기본값) → 이상치로 판단 → 무시
            x, y = _prev_raw_x, _prev_raw_y
        else:
            _prev_raw_x, _prev_raw_y = x, y
    else:
        _prev_raw_x, _prev_raw_y = x, y

    # 2. 저역통과 필터 (LPF smoothing)
    if _prev_f_x is None or _prev_f_y is None:
        _prev_f_x, _prev_f_y = x, y  # 초기화
    fx = (1 - alpha) * _prev_f_x + alpha * x
    fy = (1 - alpha) * _prev_f_y + alpha * y

    _prev_f_x, _prev_f_y = fx, fy
    return fx, fy #필터링된 좌표 (fx, fy)

# v 값이 lo보다 작으면 → lo, hi보다 크면 → hi, 범위 안이면 그대로 반환.
def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

# ── 웨이포인트 처리 ─────────────────────────────────
def load_waypoints(path_csv, min_wp_dist):
    # CSV 파일 읽기 (Lat, Lon 컬럼을 기대)
    df = pd.read_csv(path_csv)

    # 위도·경도를 Web Mercator (x,y [m]) 좌표로 변환하여 리스트에 저장
    coords = [latlon_to_meters(row['Lat'], row['Lon']) for _, row in df.iterrows()]

    # 웨이포인트가 하나도 없으면 예외 발생
    if len(coords) < 1:
        raise RuntimeError("waypoints csv empty")

    # 첫 번째 웨이포인트는 무조건 포함
    fx = [float(coords[0][0])]; fy = [float(coords[0][1])]

    # 이전 점과의 거리가 min_wp_dist 이상일 때만 웨이포인트 추가
    for xi, yi in coords[1:]:
        if distance_m(fx[-1], fy[-1], xi, yi) >= min_wp_dist:
            fx.append(float(xi)); fy.append(float(yi))

    # numpy 배열로 반환
    return np.array(fx), np.array(fy)

# ── ROS 퍼블리셔 ────────────────────────────────────
# 속도
def publish_speed(speed):
    # 차량 속도 명령 퍼블리시 (단위: m/s, Float32)
    if pub_speed: 
        pub_speed.publish(Float32(data=float(speed)))
# 조향
def publish_steer_deg(steer_deg):
    # 조향각을 차량 제한값(예: ±20°)으로 클램프
    sd = clamp(float(steer_deg), -float(params['steer_limit_deg']), float(params['steer_limit_deg']))
    
    # 차량 조향 명령 퍼블리시 (단위: deg, Float32)
    if pub_steer: 
        pub_steer.publish(Float32(data=sd))
# rtk 상태
def publish_rtk(txt):
    # RTK 상태 문자열 퍼블리시 ("FIX", "FLOAT", "NONE" 등)
    if pub_rtk: 
        pub_rtk.publish(String(data=str(txt)))

# ── ROS 콜백 ────────────────────────────────────────
_last_fix_heading_rad = None # 마지막으로 계산된 GPS 기반 진행방향(헤딩)**을 라디안 단위로 저장하는 전역 변수

# NavSatFix 콜백: GPS 위도·경도 수신 후 Web Mercator 좌표 변환 및 경로 저장
def _cb_fix(msg: NavSatFix):
    global _last_lat, _last_lon, _last_fix_heading_rad

    # 위도/경도가 유효하지 않으면 무시
    if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
        return

    # 마지막 위도·경도 갱신
    _last_lat, _last_lon = float(msg.latitude), float(msg.longitude)

    # 위도·경도를 Web Mercator (x,y) [m] 좌표로 변환
    x, y = latlon_to_meters(_last_lat, _last_lon)

    # GPS 이상치 필터 적용 (Outlier 제거 + Low-pass)
    fx, fy = filter_gps_signal(x, y)

    # 상태 변수에 경로와 헤딩 저장
    with _state_lock:
        # 이동 벡터 계산 (이전 좌표와 비교)
        if current_x and current_y:
            dx = fx - current_x[-1]
            dy = fy - current_y[-1]
            # 1e-8 이상 이동한 경우에만 헤딩(방향) 갱신
            if dx*dx + dy*dy > 1e-8:
                _last_fix_heading_rad = math.atan2(dy, dx)

        # 현재 좌표를 경로 리스트에 추가
        current_x.append(fx)
        current_y.append(fy)

# RELPOSNED 콜백: RTK 상태 판정 (FIX/FLOAT/NONE)
def _cb_relpos(msg):
    global rtk_status_txt
    try:
        # 비트 마스크 및 상태 플래그 상수
        mask  = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_MASK'))
        fixed = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_FIXED'))
        flt   = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_FLOAT'))

        # 현재 플래그에서 Carrier Solution 상태 추출
        bits = int(msg.flags) & mask

        # RTK 상태 텍스트 판정
        rtk_status_txt = "FIX" if bits == fixed else ("FLOAT" if bits == flt else "NONE")

        # ROS 토픽으로 퍼블리시
        publish_rtk(rtk_status_txt)
    except Exception:
        rtk_status_txt = "NONE"

# NAVPVT 콜백: RTK 상태 판정 (FIX/FLOAT/NONE)
def _cb_navpvt(msg):
    global rtk_status_txt
    try:
        # 비트 마스크 및 상태 플래그 상수
        mask  = int(getattr(NavPVT, 'FLAGS_CARRIER_PHASE_MASK'))
        fixed = int(getattr(NavPVT, 'CARRIER_PHASE_FIXED'))
        flt   = int(getattr(NavPVT, 'CARRIER_PHASE_FLOAT'))

        # 현재 플래그에서 Carrier Phase 상태 추출
        phase = int(msg.flags) & mask

        # RTK 상태 텍스트 판정
        rtk_status_txt = "FIX" if phase == fixed else ("FLOAT" if phase == flt else "NONE")

        # ROS 토픽으로 퍼블리시
        publish_rtk(rtk_status_txt)
    except Exception:
        rtk_status_txt = "NONE"

# ── 시각화(애니메이션 없이 주기 갱신) ─────────────────
def update_plot_once(ax):
    global waypoint_index, _last_log_t
    ax.clear()  # 이전 프레임 지우기

    # 현재 차량 좌표 리스트 복사 (스레드 안전 보장)
    with _state_lock:
        cx = list(current_x); cy = list(current_y)

    # 경로 라인(회색 얇은 선) - 지금까지 이동한 궤적
    if len(cx) >= 2:
        ax.plot(cx, cy, '-', c='0.6', lw=1.0, label='Route')

    # 현재 윈도우에 표시할 웨이포인트 범위 계산 (20개 단위)
    window_size = 20
    start_index = (waypoint_index // window_size) * window_size
    end_index = min(start_index + window_size, len(waypoints_x))

    # 웨이포인트 표시 (파란 점 + 도착 반경 원 + 인덱스 번호)
    ax.scatter(waypoints_x[start_index:end_index], waypoints_y[start_index:end_index],
               color='blue', s=10, label='Waypoints')
    for i in range(start_index, end_index):
        c = Circle((waypoints_x[i], waypoints_y[i]), float(params['target_radius']),
                   fill=False, linestyle='--', edgecolor='tab:blue', alpha=0.3)
        ax.add_patch(c)
        ax.text(waypoints_x[i], waypoints_y[i], str(i + 1), fontsize=8, ha='center')

    # 초기화
    smooth_deg = 0.0
    heading_rad = None
    info_lines = []

    # 현재 위치 및 타겟 웨이포인트가 존재할 경우
    if cx and cy:
        # 현재 차량 위치 (빨간 점)
        ax.scatter(cx[-1], cy[-1], color='red', s=50, label='Current')

        # 현재 타겟 웨이포인트
        tx, ty = waypoints_x[waypoint_index], waypoints_y[waypoint_index]

        # 타겟까지 점선 (청록색) + 타겟 표시 (* 마젠타)
        ax.plot([cx[-1], tx], [cy[-1], ty], '--', c='cyan', lw=1.0, label='Target Line')
        ax.plot(tx, ty, '*', c='magenta', ms=12, label='Target')

        # 최근 두 점으로 차량 헤딩 계산
        if len(cx) > 1:
            dx = cx[-1] - cx[-2]; dy = cy[-1] - cy[-2]
            heading_rad = math.atan2(dy, dx) if (dx*dx + dy*dy) > 1e-9 else None

        # 타겟 방향 벡터와 이동 벡터로 조향각 계산 → 저역통과 필터 적용
        if len(cx) > 1:
            target_vec = (tx - cx[-1], ty - cy[-1])
            move_vec   = (cx[-1] - cx[-2], cy[-1] - cy[-2])
            angle = calculate_steering_angle(move_vec, target_vec)
            smooth_deg = apply_low_pass_filter(angle)
             # 최종 출력 일관성: 로그/화면/퍼블리시 모두 ±steer_limit_deg(기본 20°)로 강제
            smooth_deg = clamp(
                smooth_deg,
                -float(params['steer_limit_deg']),
                float(params['steer_limit_deg'])
            )
        else:
            target_vec = ('', '')

        # 헤딩 화살표(파랑) + 조향 화살표(빨강), 길이 2m
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

        # CSV 로깅: 차량 위치, 타겟, 조향각, 헤딩, 속도, 거리, 시간, RTK 상태 기록
        if params['log_csv']:
            try:
                new = not os.path.exists(params['log_csv'])
                os.makedirs(os.path.dirname(params['log_csv']), exist_ok=True)
                with open(params['log_csv'], 'a', newline='') as f:
                    w = csv.writer(f)
                    # 새 파일일 경우 헤더 작성
                    if new:
                        w.writerow([
                            'current_x','current_y','prev_x','prev_y',
                            'target_vector_x','target_vector_y',
                            'waypoint_x','waypoint_y',
                            'steer_deg','heading_deg',
                            'speed','dist_to_target','time','rtk_status'
                        ])
                    # 거리, 헤딩, 시간 계산 후 저장
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
            
        # 터미널 로그: 0.5초 간격으로 현재 상태 출력
        now = time.time()
        if now - _last_log_t > 0.5:
            latlon_txt = f"Lat: {_last_lat:.7f}, Lon: {_last_lon:.7f}" if (_last_lat is not None and _last_lon is not None) else "Lat/Lon: (n/a)"
            heading_deg = math.degrees(heading_rad) if heading_rad is not None else 0.0
            dist_to_target = distance_m(cx[-1], cy[-1], tx, ty)
            rospy.loginfo(f"{latlon_txt}, Speed: {params['const_speed']:.2f} m/s, "
                          f"Steering: {smooth_deg:+.2f} deg, Heading: {heading_deg:.2f} deg, "
                          f"Dist→Target: {dist_to_target:.2f} m, RTK: {rtk_status_txt}")
            _last_log_t = now

        # 현재 위치가 타겟 도착 반경 안에 들어오면 → 다음 웨이포인트로 이동
        if len(cx) > 1 and distance_m(cx[-1], cy[-1], tx, ty) < float(params['target_radius']):
            if waypoint_index < len(waypoints_x) - 1:
                waypoint_index += 1

        # 화면 우측 상단 Info Box (현재 좌표, 거리, 헤딩, 조향)
        info_lines.append(f"Veh: ({cx[-1]:.1f}, {cy[-1]:.1f}) m")
        d_to_tgt = distance_m(cx[-1], cy[-1], tx, ty)
        info_lines.append(f"Dist→Target: {d_to_tgt:.1f} m")
        if heading_rad is not None:
            info_lines.append(f"Heading: {math.degrees(heading_rad):.1f}°")
        info_lines.append(f"Steering: {smooth_deg:+.1f}°")

    # Info Box 출력
    if info_lines:
        ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
                ha='left', va='top', fontsize=9, bbox=dict(fc='white', alpha=0.7))

    # 그래프 기본 스타일
    ax.set_title(f"ROS GPS Tracker  Steering: {smooth_deg:.2f}°  RTK: {rtk_status_txt}")
    ax.set_xlabel('X (meters)'); ax.set_ylabel('Y (meters)')
    ax.axis('equal'); ax.grid(True, ls=':', alpha=0.5)
    ax.legend(loc='upper right')

# ── 메인 ────────────────────────────────────────────
def main():
    global pub_speed, pub_steer, pub_rtk, waypoints_x, waypoints_y, alpha, params

    # ROS 노드 초기화 (노드 이름: waypoint_tracker_topics)
    rospy.init_node('waypoint_tracker_topics', anonymous=False)

    # ── 파라미터 로드 ─────────────────────────────
    # ROS 파라미터 서버에서 값 가져오고, 없으면 기본값 사용
    ublox_ns = rospy.get_param('~ublox_ns', '/ublox')
    params = {
        'fix_topic':        rospy.get_param('~fix_topic',    ublox_ns + '/fix'),             # GNSS 좌표
        'relpos_topic':     rospy.get_param('~relpos_topic', ublox_ns + '/navrelposned'),    # RTK 상대좌표
        'navpvt_topic':     rospy.get_param('~navpvt_topic', ublox_ns + '/navpvt'),          # RTK PVT 정보
        'waypoint_csv':     rospy.get_param('~waypoint_csv', WAYPOINT_CSV_DEFAULT),          # 웨이포인트 CSV 경로
        'target_radius':    float(rospy.get_param('~target_radius', TARGET_RADIUS_DEFAULT)), # 목표 반경 (도착 판정)
        'min_wp_distance':  float(rospy.get_param('~min_wp_distance', MIN_WAYPOINT_DISTANCE_DEFAULT)), # 웨이포인트 최소 간격
        'fc':               float(rospy.get_param('~fc', FC_DEFAULT)),                       # LPF 차단주파수
        'fs':               float(rospy.get_param('~fs', FS_DEFAULT)),                       # 샘플링 주파수
        'gps_outlier_th':   float(rospy.get_param('~gps_outlier_th', GPS_OUTLIER_THRESHOLD_DEFAULT)), # GPS 이상치 허용범위
        'steer_limit_deg':  float(rospy.get_param('~steer_limit_deg', STEER_LIMIT_DEG_DEFAULT)),       # 조향 제한각
        'const_speed':      float(rospy.get_param('~const_speed', CONST_SPEED_DEFAULT)),     # 고정 속도
        'log_csv':          rospy.get_param('~log_csv', LOG_CSV_DEFAULT),                   # 로그 파일 경로
    }

    # ── 저역통과필터(LPF) 계수 ─────────────────────
    # alpha = (2πfc) / (2πfc + fs)
    alpha = (2 * math.pi * params['fc']) / (2 * math.pi * params['fc'] + params['fs'])

    # ── 퍼블리셔 설정 ──────────────────────────────
    # 차량 속도, 조향각, RTK 상태를 퍼블리시
    pub_speed = rospy.Publisher('/vehicle/speed_cmd', Float32, queue_size=10)
    pub_steer = rospy.Publisher('/vehicle/steer_cmd',  Float32, queue_size=10)
    pub_rtk   = rospy.Publisher('/rtk/status',         String,  queue_size=10)

    # ── 웨이포인트 로드 ───────────────────────────
    try:
        os.makedirs(os.path.dirname(params['waypoint_csv']), exist_ok=True)
        waypoints_x, waypoints_y = load_waypoints(params['waypoint_csv'], params['min_wp_distance'])
    except Exception as e:
        rospy.logerr(f"[tracker_topics] failed to load waypoints: {e}")
        return

    # ── 구독자 설정 ───────────────────────────────
    rospy.Subscriber(params['fix_topic'], NavSatFix, _cb_fix, queue_size=100)    # GNSS 좌표
    if _HAVE_RELPOSNED:  # RTK 상대좌표 콜백 등록 (있을 때만)
        rospy.Subscriber(params['relpos_topic'], NavRELPOSNED, _cb_relpos, queue_size=50)
    if _HAVE_NAVPVT:     # PVT 콜백 등록 (있을 때만)
        rospy.Subscriber(params['navpvt_topic'], NavPVT, _cb_navpvt, queue_size=50)

    # 현재 어떤 토픽이 켜져 있는지 로그 출력
    rospy.loginfo("[tracker_topics] listening: fix=%s relpos=%s(%s) navpvt=%s(%s)",
                  params['fix_topic'],
                  params['relpos_topic'], 'ON' if _HAVE_RELPOSNED else 'OFF',
                  params['navpvt_topic'], 'ON' if _HAVE_NAVPVT else 'OFF')

    # ── 실시간 시각화 설정 ────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111)
    dt = 1.0 / max(1.0, float(params['fs']))  # 주기 계산

    # ROS 루프 주기 설정
    rate = rospy.Rate(params['fs'])
    try:
        while not rospy.is_shutdown():
            # 시각화 업데이트
            update_plot_once(ax)
            plt.pause(0.001)   # GUI 이벤트 플러시 (윈도우 리프레시)
            rate.sleep()       # ROS 주기만큼 슬립
    except KeyboardInterrupt:
        pass
    finally:
        print("csv 저장 되쓰요!")  # 종료 시 CSV 저장 알림

# ── 엔트리 포인트 ───────────────────────────────
if __name__ == '__main__':
    main()
