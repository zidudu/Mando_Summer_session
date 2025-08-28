#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waypoint_tracker_topics_sim.py  (ROS1 Noetic, Ubuntu)

※ GPS/RTK 없이 '알고리즘만' 테스트하는 시뮬레이터 노드입니다.
- GNSS 구독/필터 제거, 내부 시뮬레이션 상태(x,y,heading)를 자전거 모델로 적분
- 입력: 웨이포인트 CSV (Lat, Lon)
- 출력: /vehicle/speed_cmd (m/s), /vehicle/steer_cmd (deg), /rtk/status ("SIM")
- 시각화: 경로/웨이포인트/타깃선/헤딩·조향 화살표 + 정보박스(위경도, 조향각, 헤딩, 타깃 idx, 거리, 변화율)

사용 팁:
rosrun rtk_waypoint_tracker waypoint_tracker_topics_sim.py _waypoint_csv:=/path/to/left_lane.csv \
    _const_speed:=1.2 _steer_limit_deg:=20 _fc:=2.0 _fs:=20 _wheelbase:=2.5 _target_radius:=1.5
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

# ── 기본값 ─────────────────────────────────────────
TARGET_RADIUS_DEFAULT = 1.5        # [m] 도착 반경
MIN_WAYPOINT_DISTANCE_DEFAULT = 0.9 # [m] 웨이포인트 최소 간격(리샘플링)
FC_DEFAULT = 2.0                   # [Hz] 조향 LPF 컷오프
FS_DEFAULT = 20.0                  # [Hz] 루프/플롯 주기
STEER_LIMIT_DEG_DEFAULT = 20.0     # [deg] 조향 제한
CONST_SPEED_DEFAULT = 1.0          # [m/s] 고정 속도
WHEELBASE_DEFAULT = 2.5            # [m] 자전거 모델 휠베이스
AXIS_MARGIN_DEFAULT = 5.0          # [m] 플롯 축 여백
SHOW_ALL_WP_DEFAULT = True         # 전체 웨이포인트 표시

# 패키지/config 기본 경로 계산
def _default_paths():
    try:
        pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
    cfg = os.path.join(pkg_path, 'config')
    wp = os.path.join(cfg, 'left_lane.csv')
    log = os.path.join(cfg, f"sim_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    return cfg, wp, log

CFG_DIR_DEFAULT, WAYPOINT_CSV_DEFAULT, LOG_CSV_DEFAULT = _default_paths()

# ── 전역 상태 ──────────────────────────────────────
params = {}
pub_speed = None
pub_steer = None
pub_rtk   = None

# 시뮬레이션 상태(월드 좌표, Web Mercator m 단위)
state_x = None
state_y = None
heading_rad = None          # 차량 실제(시뮬) 헤딩
heading_disp = None         # 표시용 EMA 헤딩(시각화 안정)
steer_cmd = 0.0
_prev_steer = 0.0

current_x, current_y = [], []   # 주행 경로 기록
waypoints_x = None
waypoints_y = None
waypoint_index = 0

alpha = 0.56                   # 조향 LPF 계수
_filtered_steering = 0.0
_state_lock = threading.Lock()
_last_log_t = 0.0

# ── 유틸 ────────────────────────────────────────────
def latlon_to_meters(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def meters_to_latlon(x, y):
    R = 6378137.0
    lon = math.degrees(x / R)
    lat = math.degrees(2.0 * math.atan(math.exp(y / R)) - math.pi / 2.0)
    return lat, lon

def distance_m(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def circ_ema(prev_rad, new_rad, a):
    if prev_rad is None:
        return new_rad
    s = (1 - a) * math.sin(prev_rad) + a * math.sin(new_rad)
    c = (1 - a) * math.cos(prev_rad) + a * math.cos(new_rad)
    return math.atan2(s, c)

def signed_angle_deg(u, v):
    det = u[0]*v[1] - u[1]*v[0]
    dot = u[0]*v[0] + u[1]*v[1]
    return math.degrees(math.atan2(det, dot))

def apply_low_pass_filter(current):
    """조향 LPF (1차 IIR) + 부호 관례 반전"""
    global _filtered_steering, alpha
    _filtered_steering = (1 - alpha) * _filtered_steering + alpha * current
    return _filtered_steering * -1.0

# ── 웨이포인트 ─────────────────────────────────────
def load_waypoints(path_csv, min_wp_dist):
    df = pd.read_csv(path_csv)
    # 관대한 컬럼 탐색: 'lat'/'lon'이 포함된 첫 컬럼을 사용
    cols = {c.lower(): c for c in df.columns}
    lat_col = next((cols[c] for c in cols if 'lat' in c), list(df.columns)[0])
    lon_col = next((cols[c] for c in cols if 'lon' in c), list(df.columns)[1])
    def _p(v):
        if isinstance(v, str):
            # "Lat : 37.12345" 형태 허용
            import re
            m = re.search(r'[-+]?\d+(\.\d+)?', v.replace(',', ''))
            return float(m.group(0)) if m else float('nan')
        return float(v)
    lats = df[lat_col].apply(_p).to_numpy(dtype=float)
    lons = df[lon_col].apply(_p).to_numpy(dtype=float)

    coords = [latlon_to_meters(la, lo) for la, lo in zip(lats, lons)]
    if len(coords) < 1:
        raise RuntimeError("waypoints csv empty")

    fx = [float(coords[0][0])]; fy = [float(coords[0][1])]
    for xi, yi in coords[1:]:
        if distance_m(fx[-1], fy[-1], xi, yi) >= min_wp_dist:
            fx.append(float(xi)); fy.append(float(yi))
    return np.array(fx), np.array(fy)

# ── 퍼블리셔 ───────────────────────────────────────
def publish_speed(speed):
    if pub_speed:
        pub_speed.publish(Float32(data=float(speed)))

def publish_steer_deg(steer_deg):
    sd = clamp(float(steer_deg), -float(params['steer_limit_deg']), float(params['steer_limit_deg']))
    if pub_steer:
        pub_steer.publish(Float32(data=sd))

def publish_rtk(txt):
    if pub_rtk:
        pub_rtk.publish(String(data=str(txt)))

# ── 컨트롤 + 적분(자전거 모델) ─────────────────────
def controller_and_integrate(dt):
    """현재 상태에서 조향각 계산 → LPF → 자전거 모델로 (x,y,heading) 적분"""
    global state_x, state_y, heading_rad, heading_disp, steer_cmd, _prev_steer, waypoint_index

    # 타깃 웨이포인트
    tx, ty = waypoints_x[waypoint_index], waypoints_y[waypoint_index]

    # 진행/타깃 벡터
    fwd = np.array([math.cos(heading_rad), math.sin(heading_rad)], dtype=float)
    tgt = np.array([tx - state_x, ty - state_y], dtype=float)
    if np.linalg.norm(tgt) < 1e-9:
        raw_deg = 0.0
    else:
        raw_deg = signed_angle_deg(fwd, tgt) / 1.3  # 민감도 완화

    # 조향 LPF + 제한
    raw_deg = clamp(raw_deg, -float(params['steer_limit_deg']), float(params['steer_limit_deg']))
    steer_cmd = apply_low_pass_filter(raw_deg)
    steer_cmd = clamp(steer_cmd, -float(params['steer_limit_deg']), float(params['steer_limit_deg']))

    # 자전거 모델로 상태 적분
    yaw_rate = params['const_speed'] / params['wheelbase'] * math.tan(math.radians(steer_cmd))  # [rad/s]
    heading_rad = heading_rad + yaw_rate * dt
    # 표시용 헤딩(EMA) — 시각화 안정
    heading_disp = circ_ema(heading_disp, heading_rad, 0.25)

    state_x += params['const_speed'] * dt * math.cos(heading_rad)
    state_y += params['const_speed'] * dt * math.sin(heading_rad)

    # 도착 판정 → 다음 웨이포인트
    if distance_m(state_x, state_y, tx, ty) < float(params['target_radius']):
        if waypoint_index < len(waypoints_x) - 1:
            waypoint_index += 1

    # 퍼블리시
    publish_speed(params['const_speed'])
    publish_steer_deg(steer_cmd)

    # 로그/표시용 수치 반환
    steer_rate_dps = (steer_cmd - _prev_steer) / dt
    _prev_steer = steer_cmd
    heading_rate_dps = math.degrees(yaw_rate)
    dist_to_target = distance_m(state_x, state_y, tx, ty)
    return steer_rate_dps, heading_rate_dps, dist_to_target, (tx, ty)

# ── 시각화 ─────────────────────────────────────────
def update_plot_once(ax, steer_rate_dps, heading_rate_dps, dist_to_target, tx, ty):
    global _last_log_t
    ax.clear()

    # 경로 라인
    if len(current_x) >= 2:
        ax.plot(current_x, current_y, '-', lw=1.0, label='Route')

    # 웨이포인트
    if bool(params['show_all_waypoints']):
        ax.scatter(waypoints_x, waypoints_y, s=10, label='Waypoints')
        for i in range(len(waypoints_x)):
            c = Circle((waypoints_x[i], waypoints_y[i]), float(params['target_radius']),
                       fill=False, linestyle='--', alpha=0.25)
            ax.add_patch(c)
            ax.text(waypoints_x[i], waypoints_y[i], str(i+1), fontsize=7, ha='center')
    else:
        window_size = 50
        start = max(0, waypoint_index - window_size//2)
        end   = min(len(waypoints_x), start + window_size)
        ax.scatter(waypoints_x[start:end], waypoints_y[start:end], s=10, label='Waypoints')
        for i in range(start, end):
            c = Circle((waypoints_x[i], waypoints_y[i]), float(params['target_radius']),
                       fill=False, linestyle='--', alpha=0.25)
            ax.add_patch(c)
            ax.text(waypoints_x[i], waypoints_y[i], str(i+1), fontsize=7, ha='center')

    # 현재/타깃
    ax.plot([state_x, tx], [state_y, ty], '--', lw=1.0, label='Target Line')
    ax.plot(tx, ty, '*', ms=12, label='Target')
    ax.scatter(state_x, state_y, s=50, label='Current')

    # 화살표
    L = 2.0
    hx, hy = state_x + L*math.cos(heading_disp), state_y + L*math.sin(heading_disp)
    ax.add_patch(FancyArrowPatch((state_x, state_y), (hx, hy),
                                 lw=2, arrowstyle='-|>', mutation_scale=15, label='Heading'))
    sx, sy = state_x + L*math.cos(heading_disp + math.radians(steer_cmd)), state_y + L*math.sin(heading_disp + math.radians(steer_cmd))
    ax.add_patch(FancyArrowPatch((state_x, state_y), (sx, sy),
                                 lw=2, alpha=0.9, arrowstyle='-|>', mutation_scale=15, label='Steering'))

    # 정보 박스
    lat_now, lon_now = meters_to_latlon(state_x, state_y)
    info_lines = [
        f"Lat: {lat_now:.6f}, Lon: {lon_now:.6f}",
        f"Steer: {steer_cmd:+.2f} deg  (rate: {steer_rate_dps:+.1f} deg/s)",
        f"Heading: {math.degrees(heading_disp):.2f} deg  (rate: {heading_rate_dps:+.2f} deg/s)",
        f"Target idx: {waypoint_index}/{len(waypoints_x)-1}",
        f"Dist→Target: {dist_to_target:.2f} m",
        f"RTK: SIM, Speed: {params['const_speed']:.2f} m/s, Wheelbase: {params['wheelbase']:.2f} m"
    ]
    ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
            ha='left', va='top', fontsize=9, bbox=dict(fc='white', alpha=0.8))

    # 축/스타일
    ax.set_title(f"ROS Waypoint Tracker (SIM)  Steering: {steer_cmd:+.2f}°")
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.axis('equal'); ax.grid(True, ls=':')
    ax.set_xlim(AX_MIN_X, AX_MAX_X); ax.set_ylim(AX_MIN_Y, AX_MAX_Y)
    ax.legend(loc='upper right')

# ── 메인 ───────────────────────────────────────────
def main():
    global pub_speed, pub_steer, pub_rtk, waypoints_x, waypoints_y, alpha, params
    global state_x, state_y, heading_rad, heading_disp
    global AX_MIN_X, AX_MAX_X, AX_MIN_Y, AX_MAX_Y

    rospy.init_node('waypoint_tracker_topics_sim', anonymous=False)

    # 파라미터
    params = {
        'waypoint_csv':     rospy.get_param('~waypoint_csv', WAYPOINT_CSV_DEFAULT),
        'target_radius':    float(rospy.get_param('~target_radius', TARGET_RADIUS_DEFAULT)),
        'min_wp_distance':  float(rospy.get_param('~min_wp_distance', MIN_WAYPOINT_DISTANCE_DEFAULT)),
        'fc':               float(rospy.get_param('~fc', FC_DEFAULT)),
        'fs':               float(rospy.get_param('~fs', FS_DEFAULT)),
        'steer_limit_deg':  float(rospy.get_param('~steer_limit_deg', STEER_LIMIT_DEG_DEFAULT)),
        'const_speed':      float(rospy.get_param('~const_speed', CONST_SPEED_DEFAULT)),
        'wheelbase':        float(rospy.get_param('~wheelbase', WHEELBASE_DEFAULT)),
        'log_csv':          rospy.get_param('~log_csv', LOG_CSV_DEFAULT),
        'axis_margin':      float(rospy.get_param('~axis_margin', AXIS_MARGIN_DEFAULT)),
        'show_all_waypoints': bool(rospy.get_param('~show_all_waypoints', SHOW_ALL_WP_DEFAULT)),
    }

    # 조향 LPF 계수
    alpha = (2 * math.pi * params['fc']) / (2 * math.pi * params['fc'] + params['fs'])

    # 퍼블리셔
    pub_speed = rospy.Publisher('/vehicle/speed_cmd', Float32, queue_size=10)
    pub_steer = rospy.Publisher('/vehicle/steer_cmd',  Float32, queue_size=10)
    pub_rtk   = rospy.Publisher('/rtk/status',         String,  queue_size=10)
    publish_rtk("SIM")

    # 웨이포인트 로드 + 축 범위
    try:
        os.makedirs(os.path.dirname(params['waypoint_csv']), exist_ok=True)
        wpx, wpy = load_waypoints(params['waypoint_csv'], params['min_wp_distance'])
        # 시뮬 시작 상태: 첫 점, 초기 헤딩은 0→1번 점 방향
        if len(wpx) < 2:
            raise RuntimeError("웨이포인트가 2개 이상 필요합니다.")
        state_x, state_y = float(wpx[0]), float(wpy[0])
        heading_rad = math.atan2(wpy[1]-wpy[0], wpx[1]-wpx[0])
        # 표시용 헤딩 초기화
        global heading_disp
        heading_disp = heading_rad

        # 기록 리스트 초기화
        current_x.append(state_x); current_y.append(state_y)

        # 축 범위
        global AX_MIN_X, AX_MAX_X, AX_MIN_Y, AX_MAX_Y, waypoints_x, waypoints_y
        waypoints_x, waypoints_y = wpx, wpy
        AX_MIN_X = float(np.min(waypoints_x)) - params['axis_margin']
        AX_MAX_X = float(np.max(waypoints_x)) + params['axis_margin']
        AX_MIN_Y = float(np.min(waypoints_y)) - params['axis_margin']
        AX_MAX_Y = float(np.max(waypoints_y)) + params['axis_margin']
    except Exception as e:
        rospy.logerr(f"[sim] failed to load waypoints: {e}")
        return

    # 시각화 준비
    plt.ion()
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111)
    rate = rospy.Rate(params['fs'])
    dt = 1.0 / max(1.0, params['fs'])

    # 로그 파일 준비
    log_path = params['log_csv']
    log_new = not os.path.exists(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f_log = open(log_path, 'a', newline='')
    wlog = csv.writer(f_log)
    if log_new:
        wlog.writerow(['time','lat','lon','x','y',
                       'heading_deg','heading_rate_dps',
                       'steer_deg','steer_rate_dps',
                       'waypoint_idx','dist_to_target','speed','rtk'])

    # 메인 루프(자율 주행 시뮬)
    t0 = time.time()
    try:
        while not rospy.is_shutdown():
            # 컨트롤 & 적분
            steer_rate_dps, heading_rate_dps, dist_to_target, (tx, ty) = controller_and_integrate(dt)

            # 경로 기록 push
            current_x.append(state_x); current_y.append(state_y)

            # 로그
            now = time.time() - t0
            lat_now, lon_now = meters_to_latlon(state_x, state_y)
            wlog.writerow([f"{now:.3f}", f"{lat_now:.7f}", f"{lon_now:.7f}",
                           state_x, state_y,
                           math.degrees(heading_disp), heading_rate_dps,
                           steer_cmd, steer_rate_dps,
                           waypoint_index, dist_to_target,
                           params['const_speed'], "SIM"])

            # 시각화
            update_plot_once(ax, steer_rate_dps, heading_rate_dps, dist_to_target, tx, ty)
            plt.pause(0.001)

            # 종료 조건: 마지막 웨이포인트 도달 후 정지
            if waypoint_index >= len(waypoints_x) - 1 and dist_to_target < float(params['target_radius']):
                rospy.loginfo("[sim] path completed.")
                break

            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        f_log.close()
        print("csv 저장 되었습니다.")

# ── 엔트리 ─────────────────────────────────────────
if __name__ == '__main__':
    main()
