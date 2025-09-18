#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 + matplotlib · GPS waypoint 추종 & 조향각 퍼블리시
(타이머 50ms 루프 · 경사 로직 · 항상 퍼블리시 · 중앙 퍼블리시 · 안전 셧다운
 · 윈도우 기반 wp_index · 헤딩 스냅샷 Δs<0.05m 일 때 유지)

출력(루프당 각 1회):
  /gps/steer_cmd      (Float32, deg)
  /gps/wp_index       (Int32, 1-based. 반경 밖=0)  ← 전체 1..N에서 겹치면 "가장 큰 인덱스"
  /gps/speed_cmd      (Float32)                    : 기본 ~base_speed
  /gps/GRADEUP_ON     (Int32, 0/1)                 : 기본 0
  /gps/rtk_status     (String)                     : "FIX"/"NONE"(간단 매핑)
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

# ───────── 토픽 이름(합의 반영) ─────────
TOPIC_SPEED_CMD     = '/gps/speed_cmd'
TOPIC_STEER_CMD     = '/gps/steer_cmd'
TOPIC_RTK_STATUS    = '/gps/rtk_status'
TOPIC_WP_INDEX      = '/gps/wp_index'
TOPIC_WP_GRADEUP_ON = '/gps/GRADEUP_ON'

# ───────── 상수/파라미터 기본 ─────────
EARTH_RADIUS       = 6378137.0
MIN_DISTANCE       = 2.8
TARGET_RADIUS      = 2.8
ANGLE_LIMIT_DEG    = 30.0
HEADING_MOVE_MIN_M = 0.05     # ← Δs < 0.05 m 이면 헤딩 스냅샷 유지

# 경사 구간(1-based)
GRADE_APPROACH_START = 4
GRADE_APPROACH_END   = 5
GRADE_STOP_INDEX     = 6
GRADE_EXIT_START     = 9
GRADE_EXIT_END       = 11
GRADE_HOLD_SEC       = 4.0

# ───────── 전역 상태 ─────────
# 필터 파라미터(파라미터 서버에서 로드)
DT   = 0.05
TAU  = 0.25
ALPHA = None
BASE_SPEED = 5.0

# 퍼블리셔
pub_steer = pub_wpidx = pub_speed = pub_grade = pub_rtk = None

# 셧다운/타이머
shutting_down = False
timer_50ms = None

# 웨이포인트/네비
reduced_waypoints = None          # (2, N)
nav_target_idx0   = 0             # 내부 네비용(0-based)
last_nearest_idx  = 0             # 윈도우 검색 캐시(0-based)

# 헤딩/좌표 히스토리
current_x, current_y = [], []
filtered_steering_angle = 0.0
last_good_heading_rad = None      # Δs 충분할 때 갱신하는 신뢰 가능한 헤딩(rad)

# 마지막 GPS 샘플(타이머에서 사용)
have_fix = False
last_fix_x = last_fix_y = 0.0
last_fix_status = -1
last_fix_time = 0.0

# 파일 경로(패키지 내부)
try:
    pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
except Exception:
    pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
waypoint_csv_default = os.path.join(pkg_path, 'config', 'raw_track_latlon_18.csv')
logs_dir_default     = os.path.join(pkg_path, 'logs')
os.makedirs(logs_dir_default, exist_ok=True)
log_csv_default      = os.path.join(logs_dir_default, f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")

# ───────── 유틸 ─────────
def lat_lon_to_meters(lat, lon):
    x = EARTH_RADIUS * lon * math.pi / 180.0
    y = EARTH_RADIUS * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def distance_in_meters(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def _pick_column(cols, candidates):
    for c in candidates:
        if c in cols: return c
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
        wx, wy = zip(*[lat_lon_to_meters(a, b) for a, b in zip(lat, lon)])
        rospy.loginfo("Waypoints from lat/lon: (%s,%s)", lat_col, lon_col)
    elif east_col and north_col:
        wx = pd.to_numeric(df[east_col],  errors='coerce').to_numpy()
        wy = pd.to_numeric(df[north_col], errors='coerce').to_numpy()
        rospy.loginfo("Waypoints from east/north: (%s,%s)", east_col, north_col)
    else:
        raise ValueError("CSV 좌표 컬럼을 찾지 못했습니다.")

    rx, ry = [float(wx[0])], [float(wy[0])]
    for x, y in zip(wx[1:], wy[1:]):
        if distance_in_meters(rx[-1], ry[-1], float(x), float(y)) >= MIN_DISTANCE:
            rx.append(float(x)); ry.append(float(y))
    return np.array([rx, ry])

def signed_angle_deg(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    dot = float(np.dot(v1, v2)) / (n1 * n2)
    dot = max(min(dot, 1.0), -1.0)
    ang = math.degrees(math.acos(dot))
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return ang if cross >= 0 else -ang

def lpf_and_invert(current_angle_deg):
    global filtered_steering_angle
    filtered_steering_angle = (1.0 - ALPHA) * filtered_steering_angle + ALPHA * current_angle_deg
    return -filtered_steering_angle

# ───────── 공통 0 퍼블리시 ─────────
def _publish_zero_once():
    if pub_speed: pub_speed.publish(Float32(0.0))
    if pub_steer: pub_steer.publish(Float32(0.0))
    if pub_rtk:   pub_rtk.publish(String("NONE"))
    if pub_wpidx: pub_wpidx.publish(Int32(0))
    if pub_grade: pub_grade.publish(Int32(0))

# ───────── wp_index (윈도우 + 겹치면 최대 인덱스) ─────────
def find_nearest_index_windowed(x, y, xs, ys, start_idx, window_ahead=80, window_back=15):
    n = len(xs)
    if n == 0: return 0, float('inf')
    i0 = max(0, int(start_idx) - window_back)
    i1 = min(n - 1, int(start_idx) + window_ahead)
    sub_x = np.asarray(xs[i0:i1+1]); sub_y = np.asarray(ys[i0:i1+1])
    d2 = (sub_x - x)**2 + (sub_y - y)**2
    k = int(np.argmin(d2))
    return i0 + k, math.sqrt(float(d2[k]))

def expand_indices_within_radius(x, y, xs, ys, idx_center, radius):
    n = len(xs)
    if idx_center < 0 or idx_center >= n: return None, None
    if distance_in_meters(xs[idx_center], ys[idx_center], x, y) > radius:
        return None, None
    left = idx_center
    while left-1 >= 0 and distance_in_meters(xs[left-1], ys[left-1], x, y) <= radius:
        left -= 1
    right = idx_center
    while right+1 < n and distance_in_meters(xs[right+1], ys[right+1], x, y) <= radius:
        right += 1
    return left, right

def compute_wp_index_windowed(x, y, xs, ys, start_idx,
                              radius,
                              window_ahead=80, window_back=15,
                              fallback_full_scan_threshold=50.0):
    global last_nearest_idx
    n = len(xs)
    if n == 0: return 0
    nearest_idx, nearest_dist = find_nearest_index_windowed(
        x, y, xs, ys, start_idx, window_ahead, window_back
    )
    last_nearest_idx = nearest_idx
    if nearest_dist <= radius:
        lr = expand_indices_within_radius(x, y, xs, ys, nearest_idx, radius)
        if lr[0] is not None:
            return lr[1] + 1  # 1-based(가장 큰 인덱스)
    if nearest_dist > fallback_full_scan_threshold:
        xs_a, ys_a = np.asarray(xs), np.asarray(ys)
        d2_all = (xs_a - x)**2 + (ys_a - y)**2
        i = int(np.argmin(d2_all))
        if distance_in_meters(xs_a[i], ys_a[i], x, y) <= radius:
            lr = expand_indices_within_radius(x, y, xs, ys, i, radius)
            if lr[0] is not None:
                return lr[1] + 1
    return 0

# ───────── RTK/경사 로직 ─────────
def _compute_rtk_status_txt_from_status(s):
    try:
        return "FIX" if (s is not None and s >= 0) else "NONE"
    except Exception:
        return "NONE"

hold_active = False
hold_done   = False
hold_end_t  = 0.0

def _compute_grade_logic_with_zone(zone_1b, steer_cmd):
    """zone_1b: 4..11에서만 제어. 반환 (steer_out, zone_1b, speed_cmd, grade_val)."""
    global hold_active, hold_done, hold_end_t
    if shutting_down:
        return 0.0, None, 0.0, 0
    now = time.time()
    speed_cmd = BASE_SPEED
    grade_val = 0
    steer_out = steer_cmd

    if zone_1b is not None:
        if GRADE_APPROACH_START <= zone_1b <= GRADE_APPROACH_END:   # 4~5
            grade_val = 1
        elif zone_1b == GRADE_STOP_INDEX:                          # 6
            grade_val = 1
            if not hold_done:
                if not hold_active:
                    hold_active = True
                    hold_end_t  = now + GRADE_HOLD_SEC
                    rospy.loginfo("[GRADE] Entered idx 6 → hold for %.1fs", GRADE_HOLD_SEC)
                if now < hold_end_t:
                    speed_cmd = 0.0
                    steer_out = 0.0
                else:
                    hold_active = False
                    hold_done   = True
            # hold_done이면 그냥 BASE_SPEED 유지 (재정지 금지)
        elif 7 <= zone_1b <= 8:
            grade_val = 1
        elif GRADE_EXIT_START <= zone_1b <= GRADE_EXIT_END:        # 9~11
            grade_val = 0

    return steer_out, zone_1b, speed_cmd, grade_val

# ───────── GPS 콜백: 샘플만 갱신(퍼블리시 없음) ─────────
def gps_callback(msg: NavSatFix):
    global have_fix, last_fix_x, last_fix_y, last_fix_status, last_fix_time
    try:
        x, y = lat_lon_to_meters(msg.latitude, msg.longitude)
        last_fix_x, last_fix_y = x, y
        last_fix_status = msg.status.status if msg.status is not None else -1
        last_fix_time = rospy.get_time()
        have_fix = True
    except Exception as e:
        rospy.logwarn_throttle(2.0, f"[gps_callback] bad fix: {e}")

# ───────── 타이머 50ms: 항상 퍼블리시 1회 ─────────
def timer_cb(_event):
    global nav_target_idx0, last_good_heading_rad

    # 셧다운 가드
    if shutting_down:
        _publish_zero_once()
        return

    # FIX 없으면 안전 모드로 0들 송신
    if not have_fix:
        if pub_rtk:   pub_rtk.publish(String("NONE"))
        if pub_wpidx: pub_wpidx.publish(Int32(0))
        if pub_grade: pub_grade.publish(Int32(0))
        if pub_speed: pub_speed.publish(Float32(0.0))
        if pub_steer: pub_steer.publish(Float32(0.0))
        return

    # 최신 좌표를 작업 버퍼로
    x, y = last_fix_x, last_fix_y
    current_x.append(x); current_y.append(y)

    # 내부 네비 타깃(0-based) — waypoint 도달 시 진행
    tx = reduced_waypoints[0][nav_target_idx0]
    ty = reduced_waypoints[1][nav_target_idx0]
    if distance_in_meters(x, y, tx, ty) < TARGET_RADIUS:
        if nav_target_idx0 < reduced_waypoints.shape[1] - 1:
            nav_target_idx0 += 1
            tx = reduced_waypoints[0][nav_target_idx0]
            ty = reduced_waypoints[1][nav_target_idx0]

    # RTK 텍스트
    rtk_txt = _compute_rtk_status_txt_from_status(last_fix_status)

    # ── 헤딩 계산 (Δs < 0.05 m → 스냅샷 유지)
    if len(current_x) >= 2:
        prev = np.array([current_x[-2], current_y[-2]])
        curr = np.array([current_x[-1], current_y[-1]])
        move_vec = curr - prev
        move_dist = float(np.linalg.norm(move_vec))

        # 타깃 벡터
        tgt_vec = np.array([tx, ty]) - curr

        if move_dist >= HEADING_MOVE_MIN_M:
            # 충분히 움직였을 때: 실제 헤딩 갱신 + 스냅샷 갱신
            head_vec = move_vec
            last_good_heading_rad = math.atan2(head_vec[1], head_vec[0])
        else:
            # 정지/저속: 스냅샷 헤딩 사용(없으면 타깃 방향 fallback)
            if last_good_heading_rad is not None:
                head_vec = np.array([math.cos(last_good_heading_rad), math.sin(last_good_heading_rad)])
            else:
                head_vec = tgt_vec  # 초기 fallback

        angle_deg = signed_angle_deg(head_vec, tgt_vec)
        angle_deg = max(min(angle_deg, ANGLE_LIMIT_DEG), -ANGLE_LIMIT_DEG)
        smooth_inv = lpf_and_invert(angle_deg)
    else:
        # 초기 프레임: 0도를 필터로 안정화
        smooth_inv = lpf_and_invert(0.0)

    # ── wp_index(1-based) 계산
    xs = reduced_waypoints[0]; ys = reduced_waypoints[1]
    wp_index_1b = compute_wp_index_windowed(
        x, y, xs, ys,
        start_idx=last_nearest_idx,
        radius=TARGET_RADIUS,
        window_ahead=80, window_back=15,
        fallback_full_scan_threshold=50.0
    )

    # 경사 제어 zone(4..11에서만 동작)
    zone_for_grade = wp_index_1b if (4 <= wp_index_1b <= 11) else None

    # 경사 로직
    steer_out, _zone_used, speed_cmd, grade_val = _compute_grade_logic_with_zone(zone_for_grade, smooth_inv)

    # ── 퍼블리시(루프당 1회)
    if pub_rtk:   pub_rtk.publish(String(rtk_txt))
    if pub_wpidx: pub_wpidx.publish(Int32(wp_index_1b))
    if pub_grade: pub_grade.publish(Int32(grade_val))
    if pub_speed: pub_speed.publish(Float32(speed_cmd))
    if pub_steer: pub_steer.publish(Float32(steer_out))

    # 로그
    rospy.loginfo("steer=%.2f°, speed=%.2f, grade=%d, wp_idx=%d | alpha=%.3f | move_snap=%s",
                  steer_out, speed_cmd, grade_val, wp_index_1b, ALPHA,
                  "ON" if (len(current_x)>=2 and distance_in_meters(current_x[-1],current_y[-1],current_x[-2],current_y[-2])<HEADING_MOVE_MIN_M) else "OFF")

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
    ax.set_title('Timer 50ms · Always publish · Heading snapshot Δs<0.05m')

# ───────── 셧다운 ─────────
def _on_shutdown():
    global shutting_down, timer_50ms
    shutting_down = True
    try:
        if timer_50ms is not None:
            try: timer_50ms.shutdown()
            except Exception: pass
        for _ in range(2):
            _publish_zero_once()
            rospy.sleep(0.03)
        rospy.loginfo("[tracker] shutdown zeros published")
    except Exception as e:
        rospy.logwarn(f"[tracker] shutdown failed: {e}")

# ───────── 메인 ─────────
def main():
    global pub_steer, pub_wpidx, pub_speed, pub_grade, pub_rtk
    global reduced_waypoints, DT, TAU, ALPHA, BASE_SPEED, timer_50ms

    rospy.init_node('gps_waypoint_tracker', anonymous=True)

    # 파라미터
    fix_topic   = rospy.get_param('~fix_topic',   '/gps1/fix')
    enable_plot = rospy.get_param('~enable_plot', True)
    DT  = float(rospy.get_param('~dt',  0.05))
    TAU = float(rospy.get_param('~tau', 0.25))
    ALPHA = DT / (TAU + DT)
    BASE_SPEED = float(rospy.get_param('~base_speed', 5.0))

    # 경로/로그(패키지 내부 기본)
    csv_path = rospy.get_param('~csv_path', waypoint_csv_default)
    _logs_dir = rospy.get_param('~logs_dir', logs_dir_default)
    _log_csv  = rospy.get_param('~log_csv',  log_csv_default)

    rospy.loginfo("LPF: dt=%.3fs, tau=%.3fs, alpha=%.3f | base_speed=%.2f", DT, TAU, ALPHA, BASE_SPEED)
    rospy.loginfo("Waypoints CSV: %s", csv_path)

    # 퍼블리셔
    pub_steer = rospy.Publisher(TOPIC_STEER_CMD, Float32, queue_size=10)
    pub_wpidx = rospy.Publisher(TOPIC_WP_INDEX,  Int32,   queue_size=10)
    pub_speed = rospy.Publisher(TOPIC_SPEED_CMD, Float32, queue_size=10)
    pub_grade = rospy.Publisher(TOPIC_WP_GRADEUP_ON, Int32, queue_size=10)
    pub_rtk   = rospy.Publisher(TOPIC_RTK_STATUS, String, queue_size=10)

    # 구독 (샘플 저장 전용)
    rospy.Subscriber(fix_topic, NavSatFix, gps_callback)
    rospy.loginfo("Subscribed NavSatFix: %s", fix_topic)

    # 웨이포인트
    global reduced_waypoints
    reduced_waypoints = build_reduced_waypoints(csv_path)
    rospy.loginfo("Waypoints loaded: %d (spacing >= %.1fm)", reduced_waypoints.shape[1], MIN_DISTANCE)

    # 타이머 50ms 루프 시작
    timer_50ms = rospy.Timer(rospy.Duration(0.05), timer_cb)

    # 셧다운 훅
    rospy.on_shutdown(_on_shutdown)

    if enable_plot:
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, update_plot, interval=300)
        plt.show()   # GUI 루프
    else:
        rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        _on_shutdown()
        pass
