#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 + matplotlib · GPS waypoint 추종 & 조향각 퍼블리시 (50ms 타이머 루프)
- 입력: ~fix_topic (sensor_msgs/NavSatFix, default=/gps1/fix)
- 출력(매 50ms 동기 퍼블리시):
    /gps/steer_cmd   (Float32, deg)
    /gps/wp_index    (Int32, 1-based. 반경 밖=0, 겹치면 '가장 큰 인덱스')
    /gps/speed_cmd   (Float32)
    /gps/GRADEUP_ON  (Int32, 0/1)
    /gps/rtk_status  (String, "FIX"/"NONE")
- 특징:
    • 50ms 고정 주기 타이머에서 일괄 계산/퍼블리시
    • 6번 원 최초 진입 시 4초 정지 라치
    • 정지/저속/hold 중엔 '이동 중 얻은 마지막 헤딩 스냅샷' 사용
    • 재출발 시(hold 해제 직후) 2초 동안만 조향 0 강제 (헤딩 계산은 계속 유지)
"""

import os, math, time
import rospy
import rospkg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, Int32, String

# ───────── 퍼블리시 토픽 ─────────
TOPIC_SPEED_CMD     = '/gps/speed_cmd'
TOPIC_STEER_CMD     = '/gps/steer_cmd'
TOPIC_RTK_STATUS    = '/gps/rtk_status'
TOPIC_WP_INDEX      = '/gps/wp_index'
TOPIC_WP_GRADEUP_ON = '/gps/GRADEUP_ON'

# ───────── 상수/파라미터 ─────────
EARTH_RADIUS        = 6378137.0
MIN_DISTANCE        = 2.8
TARGET_RADIUS       = 2.0
ANGLE_LIMIT_DEG     = 27.0

# 경사 구간(1-based)
GRADE_APPROACH_START = 4
GRADE_APPROACH_END   = 5
GRADE_STOP_INDEX     = 6
GRADE_EXIT_START     = 9
GRADE_EXIT_END       = 11
GRADE_HOLD_SEC       = 4.0

# 정지/저속 판정용 최소 이동거리(헤딩 갱신 임계)
HEADING_MOVE_MIN_M   = 0.00  # 1 cm

# 재출발 시 조향 0으로 줄 시간(초)
RESTART_STEER_ZERO_SEC = 2.0

# ───────── 전역 상태 ─────────
DT_TARGET   = 0.05   # 목표 주기(50ms)
TAU         = 0.25   # LPF 시정수(초)
BASE_SPEED  = 5.0    # 기본 속도

# 퍼블리셔
pub_steer = pub_wpidx = pub_speed = pub_grade = pub_rtk = None

# Waypoints / 상태
reduced_waypoints = None
current_x, current_y = [], []        # GPS 업데이트 시에만 append
nav_target_idx0 = 0

# 라치 상태기
hold_active = False
hold_done   = False
hold_end_t  = 0.0

# 재출발 조향 0 강제 구간 종료 시각
restart_steer_zero_until = 0.0

# 헤딩 스냅샷
last_good_heading_rad = None

# 윈도우 최근접 캐시
last_nearest_idx = 0

# 최신 NavSatFix 보관(센서 이벤트 → 타이머 루프)
last_fix = None
last_fix_time = None

# 타이머 루프 시간
last_loop_time = None

# 시각화
enable_plot = False

# 파일 경로(패키지 내부)
try:
    pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
except Exception:
    pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
waypoint_csv_default = os.path.join(pkg_path, 'config', 'raw_track_latlon_18.csv')

# ───────── 유틸 ─────────
def lat_lon_to_meters(lat, lon):
    x = EARTH_RADIUS * lon * math.pi / 180.0
    y = EARTH_RADIUS * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def distance_in_meters(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def _pick_column(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None

def build_reduced_waypoints(csv_path):
    csv_path = os.path.expanduser(csv_path)
    df = pd.read_csv(csv_path, encoding='utf-8-sig', engine='python', sep=None)
    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = set(df.columns)

    lat_col  = _pick_column(cols, ['latitude', 'lat'])
    lon_col  = _pick_column(cols, ['longitude', 'lon'])
    east_col = _pick_column(cols, ['east', 'easting', 'x'])
    north_col= _pick_column(cols, ['north', 'northing', 'y'])

    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors='coerce').to_numpy()
        lon = pd.to_numeric(df[lon_col], errors='coerce').to_numpy()
        pts = [lat_lon_to_meters(a, b) for a, b in zip(lat, lon)]
        wx, wy = zip(*pts)
        rospy.loginfo("Waypoints from lat/lon: (%s, %s)", lat_col, lon_col)
    elif east_col and north_col:
        wx = pd.to_numeric(df[east_col],  errors='coerce').to_numpy()
        wy = pd.to_numeric(df[north_col], errors='coerce').to_numpy()
        rospy.loginfo("Waypoints from east/north: (%s, %s)", east_col, north_col)
    else:
        raise ValueError("CSV 좌표 컬럼을 찾지 못했습니다.")

    reduced_x = [float(wx[0])]; reduced_y = [float(wy[0])]
    for x, y in zip(wx[1:], wy[1:]):
        if distance_in_meters(reduced_x[-1], reduced_y[-1], float(x), float(y)) >= MIN_DISTANCE:
            reduced_x.append(float(x)); reduced_y.append(float(y))
    return np.array([reduced_x, reduced_y])  # (2, N)

def signed_angle_deg(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    dot = float(np.dot(v1, v2)) / (n1 * n2)
    dot = max(min(dot, 1.0), -1.0)
    ang = math.degrees(math.acos(dot))
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return ang if cross >= 0 else -ang

# ───────── 윈도우 기반 wp_index(1-based) ─────────
def find_nearest_index_windowed(x, y, xs, ys, start_idx, window_ahead=80, window_back=15):
    n = len(xs)
    if n == 0: return 0, float('inf')
    i0 = max(0, int(start_idx) - window_back)
    i1 = min(n - 1, int(start_idx) + window_ahead)
    sub_x = np.asarray(xs[i0:i1+1]); sub_y = np.asarray(ys[i0:i1+1])
    d2 = (sub_x - x)**2 + (sub_y - y)**2
    offset = int(np.argmin(d2)); nearest_idx = i0 + offset
    nearest_dist = math.sqrt(float(d2[offset]))
    return nearest_idx, nearest_dist

def expand_indices_within_radius(x, y, xs, ys, idx_center, radius):
    n = len(xs)
    if idx_center < 0 or idx_center >= n: return None, None
    if math.hypot(xs[idx_center] - x, ys[idx_center] - y) > radius: return None, None
    left = idx_center
    while left - 1 >= 0 and math.hypot(xs[left-1]-x, ys[left-1]-y) <= radius: left -= 1
    right = idx_center
    while right + 1 < n and math.hypot(xs[right+1]-x, ys[right+1]-y) <= radius: right += 1
    return left, right

def compute_wp_index_windowed(x, y, xs, ys, start_idx,
                              radius, window_ahead=80, window_back=15,
                              fallback_full_scan_threshold=50.0):
    global last_nearest_idx
    n = len(xs)
    if n == 0: return 0
    nearest_idx, nearest_dist = find_nearest_index_windowed(
        x, y, xs, ys, start_idx, window_ahead=window_ahead, window_back=window_back
    )
    last_nearest_idx = nearest_idx
    if nearest_dist <= radius:
        left, right = expand_indices_within_radius(x, y, xs, ys, nearest_idx, radius)
        if left is not None: return right + 1  # 1-based
    if nearest_dist > fallback_full_scan_threshold:
        xs_a = np.asarray(xs); ys_a = np.asarray(ys)
        d2 = (xs_a - x)**2 + (ys_a - y)**2
        min_i = int(np.argmin(d2))
        if math.hypot(xs_a[min_i]-x, ys_a[min_i]-y) <= radius:
            left, right = expand_indices_within_radius(x, y, xs, ys, min_i, radius)
            if left is not None: return right + 1
        return 0
    return 0

# ───────── RTK/경사 라치 로직 ─────────
def _compute_rtk_status_txt(msg: NavSatFix):
    try:
        s = msg.status.status
        return "FIX" if s is not None and s >= 0 else "NONE"
    except Exception:
        return "NONE"

def _grade_logic(zone_1b, steer_cmd):
    """
    라치/해제 로직 처리. 반환: (steer_out, speed_cmd, grade_val)
    - hold_active 동안: speed=0, steer=0, grade=1
    - hold 해제 순간: restart_steer_zero_until 세팅(2초)
    """
    global hold_active, hold_done, hold_end_t, restart_steer_zero_until
    now = time.time()

    # hold 중: 완전 정지
    if hold_active and (now < hold_end_t):
        return 0.0, 0.0, 1

    # hold 막 끝났을 때: 라치 해제 + 재출발 조향 0 윈도우 시작
    if hold_active and (now >= hold_end_t):
        hold_active = False
        hold_done   = True
        restart_steer_zero_until = now + RESTART_STEER_ZERO_SEC
        rospy.loginfo("[GRADE] Hold done → steer zero for %.1fs", RESTART_STEER_ZERO_SEC)

    # 기본(구간 외) 값
    speed_cmd = BASE_SPEED
    grade_val = 0
    steer_out = steer_cmd

    # 경사 구간에서의 grade on/off
    if zone_1b is not None:
        if GRADE_APPROACH_START <= zone_1b <= GRADE_APPROACH_END:
            grade_val = 1
        elif 7 <= zone_1b <= 8:
            grade_val = 1
        elif GRADE_EXIT_START <= zone_1b <= GRADE_EXIT_END:
            grade_val = 0

    return steer_out, speed_cmd, grade_val

# ───────── 콜백: 최신 GPS만 저장 ─────────
def gps_callback(msg: NavSatFix):
    global last_fix, last_fix_time, current_x, current_y, nav_target_idx0
    global hold_active, hold_done, hold_end_t
 
    last_fix = msg
    last_fix_time = time.time()

    # 좌표 업데이트 및 내부 타깃 진행(새 fix 들어올 때만)
    x, y = lat_lon_to_meters(msg.latitude, msg.longitude)
    current_x.append(x); current_y.append(y)

    tx = reduced_waypoints[0][nav_target_idx0]
    ty = reduced_waypoints[1][nav_target_idx0]
    if distance_in_meters(x, y, tx, ty) < TARGET_RADIUS:
        if nav_target_idx0 < reduced_waypoints.shape[1] - 1:
            nav_target_idx0 += 1

# ───────── 50ms 타이머 루프 ─────────
filtered_steering_angle = 0.0  # LPF 내부 상태

def on_timer(_evt):
    global last_loop_time, last_fix, filtered_steering_angle
    global last_good_heading_rad, nav_target_idx0
    global hold_active, hold_done, hold_end_t, restart_steer_zero_until

    now = time.time()
    dt  = DT_TARGET if last_loop_time is None else max(1e-3, now - last_loop_time)
    last_loop_time = now
    alpha = dt / (TAU + dt)

    # 초기/미수신 처리
    if last_fix is None or len(current_x) == 0:
        _publish_zero_once()
        return

    # 현재 위치(마지막 fix 기준)
    cx = current_x[-1]; cy = current_y[-1]
    tx = reduced_waypoints[0][nav_target_idx0]
    ty = reduced_waypoints[1][nav_target_idx0]

    # wp_index 계산(윈도우 기반)
    xs = reduced_waypoints[0]; ys = reduced_waypoints[1]
    wp_index_1b = compute_wp_index_windowed(
        cx, cy, xs, ys, start_idx=last_nearest_idx,
        radius=TARGET_RADIUS, window_ahead=80, window_back=15,
        fallback_full_scan_threshold=50.0
    )

    # 6번 최초 진입 시 라치 시작
    if (not hold_done) and (not hold_active) and (wp_index_1b == GRADE_STOP_INDEX):
        hold_active = True
        hold_end_t  = now + GRADE_HOLD_SEC
        rospy.loginfo("[GRADE] Entered idx 6 → hold for %.1fs (latched)", GRADE_HOLD_SEC)

    # 헤딩/조향 계산
    if len(current_x) >= 2:
        px = current_x[-2]; py = current_y[-2]
        dxy = np.array([cx - px, cy - py])
        dist = float(np.hypot(dxy[0], dxy[1]))

        # 움직였으면 스냅샷 갱신
        if dist >= HEADING_MOVE_MIN_M:
            last_good_heading_rad = math.atan2(dxy[1], dxy[0])

        # 정지/저속/hold 시엔 스냅샷 사용
        if (dist < HEADING_MOVE_MIN_M) or hold_active:
            if last_good_heading_rad is not None:
                head_vec_use = np.array([math.cos(last_good_heading_rad),
                                         math.sin(last_good_heading_rad)])
            else:
                head_vec_use = np.array([tx - cx, ty - cy])
        else:
            head_vec_use = dxy
    else:
        head_vec_use = np.array([tx - cx, ty - cy])

    tgt_vec  = np.array([tx - cx, ty - cy])
    angle_deg = signed_angle_deg(head_vec_use, tgt_vec)
    angle_deg = max(min(angle_deg, ANGLE_LIMIT_DEG), -ANGLE_LIMIT_DEG)
    # LPF + 부호 보정
    filtered_steering_angle = (1.0 - alpha) * filtered_steering_angle + alpha * angle_deg
    steer_cmd = -filtered_steering_angle

    # 경사 제어 zone (4..11만 의미)
    zone_for_grade = wp_index_1b if (4 <= wp_index_1b <= 11) else None
    steer_out, speed_cmd, grade_val = _grade_logic(zone_for_grade, steer_cmd)

    # ── 재출발 2초 동안은 조향 0 강제(헤딩은 계속 계산됨)
    if now < restart_steer_zero_until:
        steer_out = 0.0

    # 퍼블리시(동일 틱에서 동기)
    #if pub_rtk:   pub_rtk.publish(String(_compute_rtk_status_txt(last_fix)))
    #if pub_wpidx: pub_wpidx.publish(Int32(wp_index_1b))
    if pub_grade: pub_grade.publish(Int32(grade_val))
    if pub_speed: pub_speed.publish(Float32(speed_cmd))
    if pub_steer: pub_steer.publish(Float32(steer_out))

    rospy.loginfo("dt=%.3f | steer=%.2f°, speed=%.2f, grade=%d, wp_idx=%d | hold_active=%s hold_done=%s restart_zero=%s",
                  dt, steer_out, speed_cmd, grade_val, wp_index_1b,
                  hold_active, hold_done,
                  "ON" if now < restart_steer_zero_until else "OFF")

    #### 추가 ### (로깅: 5 Hz)
    _log_to_csv(now=now,
                lat=last_fix.latitude if last_fix else float('nan'),
                lon=last_fix.longitude if last_fix else float('nan'),
                x=cx, y=cy,
                speed_pub=speed_cmd,
                steer_pub_deg=steer_out,
                heading_deg=_compute_heading_deg_for_log(),
                grade_pub=grade_val,
                wp_index_1b=wp_index_1b,
                target_idx_1b=(nav_target_idx0 + 1),
                rtk=_compute_rtk_status_txt(last_fix),
                hold=int(1 if hold_active else 0))

# ───────── 공통: 0 한번 퍼블리시 ─────────
def _publish_zero_once():
    if pub_speed: pub_speed.publish(Float32(0.0))
    if pub_steer: pub_steer.publish(Float32(0.0))
    if pub_rtk:   pub_rtk.publish(String("NONE"))
    if pub_wpidx: pub_wpidx.publish(Int32(0))
    if pub_grade: pub_grade.publish(Int32(0))

# ───────── 시각화(옵션) ─────────
def update_plot(_):
    ax = plt.gca(); ax.clear()
    ax.scatter(reduced_waypoints[0], reduced_waypoints[1], s=10, marker='o', label='Waypoints')
    for i, (wx, wy) in enumerate(zip(reduced_waypoints[0], reduced_waypoints[1]), 1):
        ax.add_patch(plt.Circle((wx, wy), TARGET_RADIUS, fill=False, linestyle='--', alpha=0.5))
        ax.text(wx, wy, str(i), fontsize=8, ha='center', va='center')

    if current_x and current_y:
        cx, cy = current_x[-1], current_y[-1]
        ax.scatter(cx, cy, s=50, marker='x', label='Current')
        tx = reduced_waypoints[0][nav_target_idx0]
        ty = reduced_waypoints[1][nav_target_idx0]
        ax.arrow(cx, cy, tx - cx, ty - cy, head_width=0.5, head_length=0.5)

        k = max(0, len(current_x) - 200)
        for i in range(k + 1, len(current_x)):
            ax.arrow(current_x[i-1], current_y[i-1],
                     current_x[i]-current_x[i-1], current_y[i]-current_y[i-1],
                     head_width=0.2, head_length=0.2, length_includes_head=True)

    ax.set_aspect('equal', 'box'); ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.grid(True); ax.legend(loc='best')
    ax.set_title('50 ms timer · τ=0.25 s; limit ±30°')

# ───────── 셧다운 ─────────
def _on_shutdown():
    try:
        for _ in range(2):
            _publish_zero_once()
            rospy.sleep(0.03)   # 30ms 전송 여유
        rospy.loginfo("[tracker] shutdown zeros published")
    except Exception as e:
        rospy.logwarn(f"[tracker] shutdown failed: {e}")

# ───────── 메인 ─────────
def main():
    global pub_steer, pub_wpidx, pub_speed, pub_grade, pub_rtk
    global reduced_waypoints, TAU, BASE_SPEED, enable_plot

    rospy.init_node('gps_waypoint_tracker_50ms', anonymous=True)

    # 파라미터
    fix_topic   = rospy.get_param('~fix_topic',   '/gps1/fix')
    enable_plot = bool(rospy.get_param('~enable_plot', False))
    TAU         = float(rospy.get_param('~tau', 0.25))
    BASE_SPEED  = float(rospy.get_param('~base_speed', 5.0))

    # 경로
    csv_path = rospy.get_param('~csv_path', waypoint_csv_default)

    rospy.loginfo("Timer loop: 50 ms (20 Hz) | tau=%.3fs | base_speed=%.2f", TAU, BASE_SPEED)
    rospy.loginfo("Waypoints CSV: %s", csv_path)

    # 퍼블리셔
    pub_steer = rospy.Publisher(TOPIC_STEER_CMD, Float32, queue_size=10)
    pub_wpidx = rospy.Publisher(TOPIC_WP_INDEX,  Int32,   queue_size=10)
    pub_speed = rospy.Publisher(TOPIC_SPEED_CMD, Float32, queue_size=10)
    pub_grade = rospy.Publisher(TOPIC_WP_GRADEUP_ON, Int32, queue_size=10)
    pub_rtk   = rospy.Publisher(TOPIC_RTK_STATUS, String, queue_size=10)

    # 서브스크라이브
    rospy.Subscriber(fix_topic, NavSatFix, gps_callback)
    rospy.loginfo("Subscribed NavSatFix: %s", fix_topic)

    # 웨이포인트 구성
    global reduced_waypoints
    reduced_waypoints = build_reduced_waypoints(csv_path)
    rospy.loginfo("Waypoints loaded: %d (reduced spacing >= %.1fm)", reduced_waypoints.shape[1], MIN_DISTANCE)

    # 타이머 루프 50ms
    rospy.Timer(rospy.Duration.from_sec(DT_TARGET), on_timer)

    # 셧다운 훅
    rospy.on_shutdown(_on_shutdown)

    #### 추가 ### (로깅 초기화: 파일 생성 및 헤더/CSV 이름 1회 기록)
    _init_logging(csv_path)

    if enable_plot:
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, update_plot, interval=300)
        plt.show()
    else:
        rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        _on_shutdown()
        pass


#### 추가 ### ------------------------------------------------------
# 추가 임포트
import csv

# 로깅 전역
log_csv_path = None
_last_log_time = 0.0
LOG_INTERVAL_SEC = 0.2  # 5 Hz
_loaded_csv_name = None

def _init_logging(csv_path):
    """pkg_path/logs/wp_log_YYYYMMDD_HHMMSS.csv 생성 + CSV 이름 1회 + 헤더 작성"""
    global log_csv_path, _loaded_csv_name
    try:
        logs_dir = os.path.join(pkg_path, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        fname = f"wp_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        log_csv_path = os.path.join(logs_dir, fname)
        _loaded_csv_name = os.path.basename(os.path.expanduser(csv_path))

        with open(log_csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            # 첫 줄: 매핑한 CSV 이름(1회만)
            w.writerow(['csv_name', _loaded_csv_name])
            # 두 번째 줄: 데이터 헤더
            w.writerow([
                'time', 'time_epoch',        # ← 현재 시간(문자열) + epoch 초(소수)
                'lat','lon','x','y',
                'speed_pub','steer_pub_deg','heading_deg',
                'grade_pub','wp_index_1b','target_idx_1b',
                'rtk','hold_active'
            ])
        rospy.loginfo(f"[log] CSV initialized: {log_csv_path} (csv_name={_loaded_csv_name})")
    except Exception as e:
        rospy.logwarn(f"[log] init failed: {e}")

def _compute_heading_deg_for_log():
    """로그용 헤딩(deg). 이동량 있으면 최근 2점, 아니면 스냅샷, 없으면 NaN"""
    try:
        if len(current_x) >= 2:
            dx = current_x[-1] - current_x[-2]
            dy = current_y[-1] - current_y[-2]
            if math.hypot(dx, dy) >= HEADING_MOVE_MIN_M:
                return math.degrees(math.atan2(dy, dx))
        if last_good_heading_rad is not None:
            return math.degrees(last_good_heading_rad)
    except Exception:
        pass
    return float('nan')

def _log_to_csv(now, lat, lon, x, y,
                speed_pub, steer_pub_deg, heading_deg,
                grade_pub, wp_index_1b, target_idx_1b, rtk, hold):
    """주기적으로 한 줄 로깅 (과도한 파일 커짐 방지 위해 5 Hz 제한)"""
    global _last_log_time
    try:
        if not log_csv_path:
            return
        if (now - _last_log_time) < LOG_INTERVAL_SEC:
            return
        _last_log_time = now

        with open(log_csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),  # time
                f"{now:.3f}",                                            # time_epoch (초)
                f"{lat:.7f}", f"{lon:.7f}",
                f"{x:.3f}", f"{y:.3f}",
                f"{float(speed_pub):.2f}",
                f"{float(steer_pub_deg):.2f}",
                f"{float(heading_deg):.2f}" if math.isfinite(heading_deg) else "",
                int(grade_pub),
                int(wp_index_1b),
                int(target_idx_1b),
                rtk,
                int(hold)
            ])
    except Exception as e:
        rospy.logwarn(f"[log] write failed: {e}")
#### 추가 ### ------------------------------------------------------
