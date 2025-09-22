#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 · GPS waypoint 추종 (콜백 기반, 시간기반 라치 복구) + CSV 로깅
- 콜백에서만 처리합니다.
- 라치/재출발은 wall-time 기반 (time.time())으로 원상복귀했습니다.
- LPF ALPHA는 콜백 간 실제 dt로 ALPHA=dt/(TAU+dt) 적응.
- CSV 로깅: 위도,경도,헤딩,퍼블리시 조향각,퍼블리시 속도,퍼블리시 grade_up,현재 시간,퍼블리시 타겟 인덱스,rtk 상태
"""
import os, math, time, csv
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
TOPIC_WP_INDEX      = '/gps/wp_index'        # 항상 '타겟 인덱스(1-based)'
TOPIC_WP_GRADEUP_ON = '/gps/GRADE_ON' if False else '/gps/GRADEUP_ON'  # 고정

# ───────── 상수/파라미터 ─────────
EARTH_RADIUS        = 6378137.0
MIN_DISTANCE        = 2.8
TARGET_RADIUS       = 2.0
ANGLE_LIMIT_DEG     = 27.0

# Grade 구간 정의(모두 1-based, '타겟 인덱스' 기준)
GRADE_APPROACH_START = 5
GRADE_APPROACH_END   = 6
GRADE_STOP_INDEX     = 7
GRADE_EXIT_START     = 9
GRADE_EXIT_END       = 10
# 완전 정지 인덱스
STOP = 18
GRADE_HOLD_SEC       = 4.0

# 정지/저속 판정용 최소 이동거리(헤딩 갱신 임계)
HEADING_MOVE_MIN_M   = 0.00  # 1 cm

# 재출발 시 조향 0으로 둘 시간(초)
RESTART_STEER_ZERO_SEC = 2.0

# ───────── 전역 상태 ─────────
TAU          = 0.25   # LPF 시간상수(초)
BASE_SPEED   = 5.0    # 기본 속도
DT_FALLBACK  = 0.05   # dt 실측 실패 시 사용할 기본값

# Waypoints / 위치/헤딩 상태
reduced_waypoints = None
current_x, current_y = [], []
nav_target_idx0 = 0                  # 0-based 타겟 인덱스

# 라치 상태기 (time-based)
hold_active = False
hold_done   = False
hold_end_t  = 0.0
restart_steer_zero_until = 0.0

# 헤딩 스냅샷/시간
last_good_heading_rad = None
_last_fix_walltime    = None  # dt 실측용(이전 콜백 시간)

# LPF 내부 상태
filtered_steering_angle = 0.0  # y[k]
ALPHA = 0.0                    # dt/(TAU+dt) – 콜백마다 갱신

# 퍼블리셔
pub_steer = pub_wpidx = pub_speed = pub_grade = pub_rtk = None

# 파일 경로(패키지 내부)
try:
    pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
except Exception:
    pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
waypoint_csv_default = os.path.join(pkg_path, 'config', 'raw_track_latlon_22.csv')

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

# === 시간상수 기반 LPF + 부호 반전 유지 ===
def lpf_and_invert(current_angle_deg):
    global filtered_steering_angle, ALPHA
    filtered_steering_angle = (1.0 - ALPHA) * filtered_steering_angle + ALPHA * current_angle_deg
    return -filtered_steering_angle  # 부호 반전 유지

# ───────── RTK 상태 텍스트 ─────────
def _compute_rtk_status_txt(msg: NavSatFix):
    try:
        s = msg.status.status
        if s is not None and s >= 2:  # status 2+: RTK FIX
            return "FIX"
        elif s is not None and s == 1:  # 1: FLOAT
            return "FLOAT"
        else:
            return "NONE"
    except Exception:
        return "NONE"

# ───────── Grade/Speed 결정 ─────────
def _grade_logic_by_target(target_1b, steer_cmd):
    global hold_active, hold_done, hold_end_t, restart_steer_zero_until
    now = time.time()

    # hold 중: 완전 정지
    if hold_active and (now < hold_end_t):
        return 0.0, 0.0, 1

    # hold 막 끝: 라치 해제 + 재출발 조향 0 윈도우 시작
    if hold_active and (now >= hold_end_t):
        hold_active = False
        hold_done   = True
        restart_steer_zero_until = now + RESTART_STEER_ZERO_SEC
        rospy.loginfo("[GRADE] Hold done → steer zero for %.1fs", RESTART_STEER_ZERO_SEC)

    speed_cmd = BASE_SPEED
    grade_val = 0
    steer_out = steer_cmd

    # 접근/탈출 동안 grade=1 유지
    if target_1b is not None:
        if GRADE_APPROACH_START <= target_1b < GRADE_EXIT_START:
            grade_val = 1
        elif GRADE_EXIT_START <= target_1b <= GRADE_EXIT_END:
            grade_val = 0
        else:
            grade_val = 0

    # 완전정지 구간(필요 시)
    if target_1b is not None and target_1b == STOP:
        steer_out = 0.0
        speed_cmd = 0.0
        grade_val = 0
    return steer_out, speed_cmd, grade_val

# ───────── CSV 로깅 유틸 ─────────
_log_fh = None
_log_writer = None
_log_count = 0
_log_autoflush_every = 10  # 10줄마다 flush

def _default_log_path():
    logs_dir = os.path.join(pkg_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")

def init_csv_logger(log_csv_path=None):
    """CSV 로거 초기화: 헤더 1회 기록."""
    global _log_fh, _log_writer, _log_count
    if _log_fh is not None:
        return
    path = log_csv_path or _default_log_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _log_fh = open(path, 'a', newline='', encoding='utf-8')
    _log_writer = csv.writer(_log_fh)
    if os.stat(path).st_size == 0:
        _log_writer.writerow([
            'time_str',           # 현재 시간(로컬)
            'lat', 'lon',         # 위도, 경도
            'heading_deg',        # 헤딩(도)
            'steer_pub_deg',      # 퍼블리시 조향각(도)
            'speed_pub',          # 퍼블리시 속도
            'grade_pub',          # 퍼블리시 grade_up(0/1)
            'target_idx_pub',     # 퍼블리시 타겟 인덱스(1-based)
            'rtk_status'          # RTK 상태 문자열
        ])
    _log_count = 0
    rospy.loginfo("[log] CSV logging to: %s", path)

def log_csv_row(lat, lon, heading_deg, steer_pub_deg, speed_pub, grade_pub, tgt_idx_pub, rtk_txt):
    """요청 필드만 정확히 한 줄 기록."""
    global _log_fh, _log_writer, _log_count
    if _log_writer is None:
        return
    _log_writer.writerow([
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        f"{lat:.7f}", f"{lon:.7f}",
        (f"{heading_deg:.2f}" if math.isfinite(heading_deg) else ""),
        f"{steer_pub_deg:.2f}",
        f"{speed_pub:.2f}",
        int(grade_pub),
        int(tgt_idx_pub),
        rtk_txt
    ])
    _log_count += 1
    if (_log_count % _log_autoflush_every) == 0:
        try:
            _log_fh.flush()
        except Exception:
            pass

# ───────── NavSatFix 콜백(유일한 계산 루프) ─────────
def gps_callback(msg: NavSatFix):
    global _last_fix_walltime, ALPHA
    global last_good_heading_rad, nav_target_idx0
    global hold_active, hold_done, hold_end_t, restart_steer_zero_until

    now = time.time()

    # dt 실측 → ALPHA 갱신(실패 시 fallback)
    if _last_fix_walltime is None:
        dt = DT_FALLBACK
    else:
        dt = max(1e-3, now - _last_fix_walltime)
    _last_fix_walltime = now
    ALPHA = dt / (TAU + dt)

    # 위치 갱신
    x, y = lat_lon_to_meters(msg.latitude, msg.longitude)
    current_x.append(x); current_y.append(y)

    # 현 타겟
    tx = reduced_waypoints[0][nav_target_idx0]
    ty = reduced_waypoints[1][nav_target_idx0]

    # 타겟 도달 → 다음 인덱스
    if distance_in_meters(x, y, tx, ty) < TARGET_RADIUS:
        if nav_target_idx0 < reduced_waypoints.shape[1] - 1:
            nav_target_idx0 += 1
            tx = reduced_waypoints[0][nav_target_idx0]
            ty = reduced_waypoints[1][nav_target_idx0]

    # 타겟 인덱스(1-based)
    target_1b = int(nav_target_idx0 + 1)

    # STOP_INDEX '최초 진입' 시 정지 라치
    if (not hold_done) and (not hold_active) and (target_1b == GRADE_STOP_INDEX):
        hold_active = True
        hold_end_t  = now + GRADE_HOLD_SEC
        rospy.loginfo("[GRADE] Entered STOP_INDEX=%d → hold for %.1fs (latched)",
                      GRADE_STOP_INDEX, GRADE_HOLD_SEC)

    # 헤딩/조향 계산
    if len(current_x) >= 2:
        px = current_x[-2]; py = current_y[-2]
        dxy = np.array([x - px, y - py])
        dist = float(np.hypot(dxy[0], dxy[1]))

        if dist >= HEADING_MOVE_MIN_M:
            last_good_heading_rad = math.atan2(dxy[1], dxy[0])

        if (dist < HEADING_MOVE_MIN_M) or hold_active:
            if last_good_heading_rad is not None:
                head_vec_use = np.array([math.cos(last_good_heading_rad),
                                         math.sin(last_good_heading_rad)])
            else:
                head_vec_use = np.array([tx - x, ty - y])
        else:
            head_vec_use = dxy
    else:
        head_vec_use = np.array([tx - x, ty - y])

    tgt_vec  = np.array([tx - x, ty - y])
    angle_deg = signed_angle_deg(head_vec_use, tgt_vec)
    angle_deg = max(min(angle_deg, ANGLE_LIMIT_DEG), -ANGLE_LIMIT_DEG)

    # LPF + 부호반전
    steer_cmd = lpf_and_invert(angle_deg)

    # Grade 로직(타겟 인덱스 기반)
    steer_out, speed_cmd, grade_val = _grade_logic_by_target(target_1b, steer_cmd)

    # 재출발 2초 조향 0 강제
    if now < restart_steer_zero_until:
        steer_out = 0.0

    # RTK 텍스트
    rtk_txt = _compute_rtk_status_txt(msg)

    # 퍼블리시
    if pub_rtk:   pub_rtk.publish(String(rtk_txt))
    if pub_wpidx: pub_wpidx.publish(Int32(target_1b))
    if pub_grade: pub_grade.publish(Int32(grade_val))
    if pub_speed: pub_speed.publish(Float32(speed_cmd))
    if pub_steer: pub_steer.publish(Float32(steer_out))

    # 헤딩(도) 계산(로그용)
    if head_vec_use is not None and np.linalg.norm(head_vec_use) > 0:
        heading_deg = math.degrees(math.atan2(head_vec_use[1], head_vec_use[0]))
    else:
        heading_deg = float('nan')

    # ── CSV 로깅(요청 필드만) ──
    log_csv_row(
        lat=float(msg.latitude),
        lon=float(msg.longitude),
        heading_deg=heading_deg,
        steer_pub_deg=float(steer_out),
        speed_pub=float(speed_cmd),
        grade_pub=int(grade_val),
        tgt_idx_pub=int(target_1b),
        rtk_txt=str(rtk_txt)
    )

    rospy.loginfo(
        "dt=%.3f | steer=%.2f°, speed=%.2f, grade=%d, tgt_idx=%d, rtk=%s | hold_active=%s hold_done=%s restart_zero=%s",
        dt, steer_out, speed_cmd, grade_val, target_1b, rtk_txt,
        hold_active, hold_done,
        "ON" if now < restart_steer_zero_until else "OFF"
    )

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
    ax.set_title('Callback-only · adaptive LPF (ALPHA=dt/(TAU+dt)) · limit ±27°')

# ───────── 셧다운 ─────────
def _publish_zero_once():
    if pub_speed: pub_speed.publish(Float32(0.0))
    if pub_steer: pub_steer.publish(Float32(0.0))
    if pub_rtk:   pub_rtk.publish(String("NONE"))
    if pub_wpidx: pub_wpidx.publish(Int32(0))
    if pub_grade: pub_grade.publish(Int32(0))

def _on_shutdown():
    try:
        for _ in range(2):
            _publish_zero_once()
            rospy.sleep(0.03)
        rospy.loginfo("[tracker] shutdown zeros published")
    except Exception as e:
        rospy.logwarn(f"[tracker] shutdown failed: {e}")
    # CSV 파일 닫기
    global _log_fh
    try:
        if _log_fh:
            _log_fh.flush()
            _log_fh.close()
            _log_fh = None
            rospy.loginfo("[log] CSV closed.")
    except Exception:
        pass

# ───────── 메인 ─────────
def main():
    global pub_steer, pub_wpidx, pub_speed, pub_grade, pub_rtk
    global reduced_waypoints, TAU, BASE_SPEED, DT_FALLBACK

    rospy.init_node('gps_waypoint_tracker_cb_time', anonymous=True)

    # 파라미터
    fix_topic    = rospy.get_param('~fix_topic',   '/gps1/fix')
    csv_path     = rospy.get_param('~csv_path',    waypoint_csv_default)
    log_csv_path = rospy.get_param('~log_csv',     '')  # 비어있으면 자동 경로
    enable_plot  = bool(rospy.get_param('~enable_plot', False))
    TAU          = float(rospy.get_param('~tau', 0.25))
    BASE_SPEED   = float(rospy.get_param('~base_speed', 5.0))
    DT_FALLBACK  = float(rospy.get_param('~dt_fallback', 0.05))

    rospy.loginfo("Mode: Callback-only (time-based latches) | tau=%.3fs | base_speed=%.2f", TAU, BASE_SPEED)
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
    rospy.loginfo("Waypoints loaded: %d (reduced spacing >= %.1fm)",
                  reduced_waypoints.shape[1], MIN_DISTANCE)

    # CSV 로거 준비
    init_csv_logger(log_csv_path or None)

    # 셧다운 훅
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
