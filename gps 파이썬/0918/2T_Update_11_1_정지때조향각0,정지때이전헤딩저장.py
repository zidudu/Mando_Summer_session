#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 + matplotlib · GPS waypoint 추종 & 조향각 퍼블리시
(경사 로직 + 항상 퍼블리시 + 퍼블리시 중앙화 + 안전 셧다운 + 윈도우 기반 wp_index + 헤딩 스냅샷)

- 입력: (param) ~fix_topic  (sensor_msgs/NavSatFix, default=/gps1/fix)
- 출력(항상 퍼블리시, 루프당 각 1회):
  * /gps/steer_cmd      (std_msgs/Float32, deg)
  * /gps/wp_index       (std_msgs/Int32, 1-based. 반경 밖=0) ← 전체 1..N 범위에서 겹침 시 "가장 큰 인덱스"
  * /gps/speed_cmd      (std_msgs/Float32)      : 기본 ~base_speed
  * /gps/GRADEUP_ON     (std_msgs/Int32, 0/1)   : 기본 0
  * /gps/rtk_status     (std_msgs/String)       : "FIX"/"NONE" (간단 매핑)
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

# ───────── 퍼블리시 토픽(정식 이름) ─────────
TOPIC_SPEED_CMD     = '/gps/speed_cmd'     # Float32
TOPIC_STEER_CMD     = '/gps/steer_cmd'     # Float32 (deg)
TOPIC_RTK_STATUS    = '/gps/rtk_status'    # String ("FIX"/"FLOAT"/"NONE")
TOPIC_WP_INDEX      = '/gps/wp_index'      # Int32 (1-based, 반경 밖=0)
TOPIC_WP_GRADEUP_ON = '/gps/GRADEUP_ON'    # Int32 (0/1)

# ───────── 상수/파라미터 기본 ─────────
EARTH_RADIUS     = 6378137.0     # [m]
MIN_DISTANCE     = 2.8           # [m] 웨이포인트 간 최소 간격
TARGET_RADIUS    = 2.8           # [m] 원 반경(도달/구간 판정)
ANGLE_LIMIT_DEG  = 30.0          # [deg] 조향 제한

# 경사 구간(1-based index)
GRADE_APPROACH_START = 4
GRADE_APPROACH_END   = 5
GRADE_STOP_INDEX     = 6
GRADE_EXIT_START     = 9
GRADE_EXIT_END       = 11
GRADE_HOLD_SEC       = 4.0

# ───────── 전역 상태 ─────────
filtered_steering_angle = 0.0
current_x, current_y = [], []
nav_target_idx0 = 0    # 내부 네비게이션용(0-based)

# 헤딩 스냅샷(정지/저속 시 사용)
last_good_heading_rad = None      # 마지막 '신뢰 가능한' 헤딩(rad)
snap_frames_left = 0              # hold 해제 직후 강제 스냅샷 프레임 수

# 필터/속도 파라미터(파라미터 서버에서 로드)
DT   = 0.05   # [s]
TAU  = 0.25   # [s]
ALPHA = None  # dt/(tau+dt)
BASE_SPEED = 5.0  # 항상 퍼블리시용 기본 속도

# 헤딩 스냅샷 파라미터
HEADING_MOVE_MIN_M = 0.05         # ★ Δs 임계값(기본 0.05 m)
HEADING_SNAP_HOLD_FRAMES = 2      # hold 해제 후 강제 스냅샷 프레임 수

# 퍼블리셔
pub_steer = None
pub_wpidx = None
pub_speed = None
pub_grade = None
pub_rtk   = None

# 셧다운/타이머
shutting_down = False
pub_timer = None

# 경사 상태기
hold_active = False  # 현재 4초 정지 중인지
hold_done   = False  # 정지를 이미 한 번 수행했는지(재정지 방지)
hold_end_t  = 0.0    # hold 종료 시각

# 윈도우 기반 최근접 검색용 캐시
last_nearest_idx = 0

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
        available = ", ".join(sorted(cols))
        raise ValueError("CSV 좌표 컬럼을 찾지 못했습니다. 현재 컬럼: " + available)

    reduced_x = [float(wx[0])]
    reduced_y = [float(wy[0])]
    for x, y in zip(wx[1:], wy[1:]):
        if distance_in_meters(reduced_x[-1], reduced_y[-1], float(x), float(y)) >= MIN_DISTANCE:
            reduced_x.append(float(x))
            reduced_y.append(float(y))

    return np.array([reduced_x, reduced_y])  # (2, N)

def signed_angle_deg(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = float(np.dot(v1, v2)) / (n1 * n2)
    dot = max(min(dot, 1.0), -1.0)
    ang = math.degrees(math.acos(dot))
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return ang if cross >= 0 else -ang

def lpf_and_invert(current_angle_deg):
    global filtered_steering_angle, ALPHA
    filtered_steering_angle = (1.0 - ALPHA) * filtered_steering_angle + ALPHA * current_angle_deg
    return -filtered_steering_angle

# ───────── 공통: 0 한 번 퍼블리시 헬퍼 ─────────
def _publish_zero_once():
    if pub_speed: pub_speed.publish(Float32(0.0))
    if pub_steer: pub_steer.publish(Float32(0.0))
    if pub_rtk:   pub_rtk.publish(String("NONE"))
    if pub_wpidx: pub_wpidx.publish(Int32(0))
    if pub_grade: pub_grade.publish(Int32(0))

# ───────── 윈도우 기반 wp_index(1-based) 계산 ─────────
reduced_waypoints = None  # main에서 로드

def find_nearest_index_windowed(x, y, xs, ys, start_idx, window_ahead=80, window_back=15):
    """start_idx 주변의 작은 윈도우만 검사하여 최근접 인덱스와 거리(미터)를 반환."""
    n = len(xs)
    if n == 0:
        return 0, float('inf')
    i0 = max(0, int(start_idx) - window_back)
    i1 = min(n - 1, int(start_idx) + window_ahead)
    sub_x = np.asarray(xs[i0:i1+1]); sub_y = np.asarray(ys[i0:i1+1])
    d2 = (sub_x - x)**2 + (sub_y - y)**2
    offset = int(np.argmin(d2))
    nearest_idx = i0 + offset
    nearest_dist = math.sqrt(float(d2[offset]))
    return nearest_idx, nearest_dist

def expand_indices_within_radius(x, y, xs, ys, idx_center, radius):
    """idx_center를 중심으로 좌우 확장하여 radius 이내인 인덱스 범위를 반환. 없으면 (None, None)."""
    n = len(xs)
    if idx_center < 0 or idx_center >= n:
        return None, None
    if math.hypot(xs[idx_center] - x, ys[idx_center] - y) > radius:
        return None, None
    left = idx_center
    while left - 1 >= 0 and math.hypot(xs[left-1] - x, ys[left-1] - y) <= radius:
        left -= 1
    right = idx_center
    while right + 1 < n and math.hypot(xs[right+1] - x, ys[right+1] - y) <= radius:
        right += 1
    return left, right

def compute_wp_index_windowed(x, y, xs, ys, start_idx,
                              radius,
                              window_ahead=80, window_back=15,
                              fallback_full_scan_threshold=50.0):
    """
    윈도우에서 최근접을 찾고, 반경 내 연속 구간을 좌우 확장해 '가장 큰 인덱스(1-based)'를 반환.
    윈도우 근처에 없고 최근접 거리가 너무 크면(이탈 추정) 드물게 전체 스캔 폴백.
    """
    global last_nearest_idx
    n = len(xs)
    if n == 0:
        return 0

    nearest_idx, nearest_dist = find_nearest_index_windowed(
        x, y, xs, ys, start_idx, window_ahead=window_ahead, window_back=window_back
    )
    last_nearest_idx = nearest_idx

    if nearest_dist <= radius:
        left, right = expand_indices_within_radius(x, y, xs, ys, nearest_idx, radius)
        if left is not None:
            return right + 1  # 1-based

    if nearest_dist > fallback_full_scan_threshold:
        xs_a = np.asarray(xs); ys_a = np.asarray(ys)
        d2 = (xs_a - x)**2 + (ys_a - y)**2
        min_i = int(np.argmin(d2))
        if math.hypot(xs_a[min_i]-x, ys_a[min_i]-y) <= radius:
            left, right = expand_indices_within_radius(x, y, xs, ys, min_i, radius)
            if left is not None:
                return right + 1
        return 0

    return 0

# ───────── RTK/경사 로직 유틸 ─────────
def _compute_rtk_status_txt(msg: NavSatFix):
    try:
        s = msg.status.status
        return "FIX" if s is not None and s >= 0 else "NONE"
    except Exception:
        return "NONE"

def _compute_grade_logic_with_zone(zone_1b, steer_cmd):
    """
    zone_1b: 1-based 또는 None (경사 제어는 4..11에서만 동작)
    반환: (steer_out, zone_1b, speed_cmd, grade_val)
    """
    global hold_active, hold_done, hold_end_t, shutting_down, snap_frames_left
    if shutting_down:
        return 0.0, None, 0.0, 0

    now = time.time()
    speed_cmd = BASE_SPEED
    grade_val = 0
    steer_out = steer_cmd

    if zone_1b is not None:
        if GRADE_APPROACH_START <= zone_1b <= GRADE_APPROACH_END:
            grade_val = 1

        elif zone_1b == GRADE_STOP_INDEX:
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
                    speed_cmd   = BASE_SPEED
                    # ★ hold 해제 직후 N프레임 동안 스냅샷 헤딩 강제 사용
                    snap_frames_left = max(snap_frames_left, HEADING_SNAP_HOLD_FRAMES)

        elif 7 <= zone_1b <= 8:
            grade_val = 1

        elif GRADE_EXIT_START <= zone_1b <= GRADE_EXIT_END:
            grade_val = 0

    return steer_out, zone_1b, speed_cmd, grade_val

# ───────── 콜백 (퍼블리시 중앙화: 여기서만 publish 호출) ─────────
def gps_callback(msg: NavSatFix):
    global nav_target_idx0, reduced_waypoints, last_nearest_idx
    global last_good_heading_rad, snap_frames_left

    # 셧다운 시: 0만 퍼블리시하고 종료
    if shutting_down:
        _publish_zero_once()
        return

    # 현재 위치(m)
    x, y = lat_lon_to_meters(msg.latitude, msg.longitude)
    current_x.append(x); current_y.append(y)

    # 내부 네비게이션 타깃(0-based)
    tx = reduced_waypoints[0][nav_target_idx0]
    ty = reduced_waypoints[1][nav_target_idx0]

    # 도달 판정(내부 네비 인덱스만 증가)
    if distance_in_meters(x, y, tx, ty) < TARGET_RADIUS:
        if nav_target_idx0 < reduced_waypoints.shape[1] - 1:
            nav_target_idx0 += 1

    # RTK 상태 텍스트 계산 (항상)
    rtk_txt = _compute_rtk_status_txt(msg)

    # ── 조향각 계산(헤딩 스냅샷 사용)
    if len(current_x) >= 2:
        prev = np.array([current_x[-2], current_y[-2]])
        curr = np.array([current_x[-1], current_y[-1]])
        head_vec_raw = curr - prev
        ds = float(np.linalg.norm(head_vec_raw))

        use_snapshot = False
        # ① 이동이 너무 작을 때(정지/저속 노이즈) → 스냅샷 사용 (임계 0.05 m)
        if ds < HEADING_MOVE_MIN_M:
            use_snapshot = True
        # ② hold 해제 직후 N프레임 강제 스냅샷
        if snap_frames_left > 0:
            use_snapshot = True
            snap_frames_left -= 1

        if not use_snapshot:
            # 새로 얻은 헤딩을 스냅샷으로 갱신
            last_good_heading_rad = math.atan2(head_vec_raw[1], head_vec_raw[0])
            head_vec_used = head_vec_raw
        else:
            # 스냅샷이 아직 없으면 타깃 방향을 임시 헤딩으로 사용
            if last_good_heading_rad is None:
                tgt_vec_fallback = np.array([tx, ty]) - curr
                if np.linalg.norm(tgt_vec_fallback) < 1e-6:
                    tgt_vec_fallback = np.array([1.0, 0.0])
                last_good_heading_rad = math.atan2(tgt_vec_fallback[1], tgt_vec_fallback[0])
            head_vec_used = np.array([math.cos(last_good_heading_rad),
                                      math.sin(last_good_heading_rad)], dtype=float)

        # 타깃 벡터 및 각도
        tgt_vec = np.array([tx, ty]) - curr
        angle_deg = signed_angle_deg(head_vec_used, tgt_vec)
        angle_deg = max(min(angle_deg, ANGLE_LIMIT_DEG), -ANGLE_LIMIT_DEG)
        smooth_inv = lpf_and_invert(angle_deg)
    else:
        smooth_inv = lpf_and_invert(0.0)

    # ── wp_index(1-based) 계산 (룩어헤드 없음, 윈도우 + 겹침 시 최대 인덱스)
    xs = reduced_waypoints[0]; ys = reduced_waypoints[1]
    wp_index_1b = compute_wp_index_windowed(
        x, y, xs, ys,
        start_idx=last_nearest_idx,  # 초기 0
        radius=TARGET_RADIUS,
        window_ahead=80, window_back=15,
        fallback_full_scan_threshold=50.0
    )

    # 경사 제어용 zone (4..11에서만 동작)
    zone_for_grade = wp_index_1b if (4 <= wp_index_1b <= 11) else None

    # ── 경사 로직(값만 계산)
    steer_out, zone_used, speed_cmd, grade_val = _compute_grade_logic_with_zone(zone_for_grade, smooth_inv)

    # ── 퍼블리시: 루프당 각 토픽 1회
    if pub_rtk:   pub_rtk.publish(String(rtk_txt))
    if pub_wpidx: pub_wpidx.publish(Int32(wp_index_1b))     # 전체 1..N, 반경 밖=0
    if pub_grade: pub_grade.publish(Int32(grade_val))
    if pub_speed: pub_speed.publish(Float32(speed_cmd))
    if pub_steer: pub_steer.publish(Float32(steer_out))

    rospy.loginfo("steer=%.2f°, speed=%.2f, grade=%d, wp_idx=%d (dt=%.3f, tau=%.3f, alpha=%.3f)",
                  steer_out, speed_cmd, grade_val, wp_index_1b, DT, TAU, ALPHA)

# ───────── 시각화 ─────────
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
    ax.set_title('Always publish (centralized) · dt=50 ms, tau=0.25 s; limit ±30°')

# ───────── 셧다운 핸들러 ─────────
def _on_shutdown():
    """종료 시 0 한번 깔끔하게 송신."""
    global shutting_down, pub_timer
    shutting_down = True
    try:
        if pub_timer is not None:
            try:
                pub_timer.shutdown()
            except Exception:
                pass
        for _ in range(2):
            _publish_zero_once()
            rospy.sleep(0.03)   # 30ms 전송 여유
        rospy.loginfo("[tracker] shutdown zeros published")
    except Exception as e:
        rospy.logwarn(f"[tracker] shutdown failed: {e}")

# ───────── 메인 ─────────
def main():
    global pub_steer, pub_wpidx, pub_speed, pub_grade, pub_rtk
    global reduced_waypoints, DT, TAU, ALPHA, BASE_SPEED
    global HEADING_MOVE_MIN_M, HEADING_SNAP_HOLD_FRAMES

    rospy.init_node('gps_waypoint_tracker', anonymous=True)

    # 파라미터
    fix_topic   = rospy.get_param('~fix_topic',   '/gps1/fix')
    enable_plot = rospy.get_param('~enable_plot', True)
    DT  = float(rospy.get_param('~dt',  0.05))   # 50ms
    TAU = float(rospy.get_param('~tau', 0.25))   # 0.25s
    ALPHA = DT / (TAU + DT)
    BASE_SPEED = float(rospy.get_param('~base_speed', 5.0))
    # 헤딩 스냅샷 파라미터(튜닝 가능)
    HEADING_MOVE_MIN_M = float(rospy.get_param('~heading_min_move', 0.05))
    HEADING_SNAP_HOLD_FRAMES = int(rospy.get_param('~heading_snap_hold_frames', 2))

    # 경로/로그 기본값(패키지 내부)
    csv_path = rospy.get_param('~csv_path', waypoint_csv_default)
    _logs_dir = rospy.get_param('~logs_dir', logs_dir_default)
    _log_csv  = rospy.get_param('~log_csv',  log_csv_default)  # 현재는 경로만 준비

    rospy.loginfo("LPF: dt=%.3fs, tau=%.3fs, alpha=%.3f | base_speed=%.2f | heading_min_move=%.3f m | snap_frames=%d",
                  DT, TAU, ALPHA, BASE_SPEED, HEADING_MOVE_MIN_M, HEADING_SNAP_HOLD_FRAMES)
    rospy.loginfo("Waypoints CSV: %s", csv_path)
    rospy.loginfo("Logs dir: %s", _logs_dir)

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

    # 셧다운 훅 등록
    rospy.on_shutdown(_on_shutdown)

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
