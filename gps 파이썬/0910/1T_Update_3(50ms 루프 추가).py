#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 + matplotlib · GPS waypoint 추종 & 조향각 퍼블리시 (정리판 · 경로해결 rospkg 방식)
- 변경점: 메인 처리 루프를 rospy.Timer 기준으로 ~dt (기본 0.05s = 50ms) 마다 실행하도록 변경.
- 입력: (param) ~fix_topic  (sensor_msgs/NavSatFix, default=/gps1/fix)
- 출력:
  * /vehicle/steer_cmd (std_msgs/Float32)  → LPF + "부호 반전 유지" (원래 /filtered_steering_angle 대신 통신 맞춤)
  * /current_waypoint_index (std_msgs/Int32) → 현재 목표 웨이포인트 인덱스 (0-based)
  * /vehicle/speed_cmd (std_msgs/Float32)   → 고정 속도 값 퍼블리시
  * /rtk/status (std_msgs/String)            → "FIXED"/"FLOAT"/"NONE"
- 파라미터:
  * ~csv_path, ~fix_topic, ~relpos_topic, ~enable_plot, ~dt, ~tau, ~log_csv
"""

import os
import math
import time
import csv
import rospy
import rospkg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, Int32, String

# ====== 상수 ======
EARTH_RADIUS    = 6378137.0     # [m]
MIN_DISTANCE    = 2.8           # [m] 웨이포인트 간 최소 간격
TARGET_RADIUS   = 2.8           # [m] 목표 웨이포인트 도달 반경
ANGLE_LIMIT_DEG = 27.0          # [deg] 조향 제한

# ---- 고정 속도 (범례/로그/퍼블리시 모두 이 값 사용) ----
FIXED_SPEED = 2.0  # [m/s]

# ====== 전역 상태 ======
filtered_steering_angle = 0.0
current_x, current_y = [], []
current_t = []                  # 시간(초) 저장 → 속도 계산용
current_waypoint_index = 0

# 필터 파라미터 (파라미터 서버에서 로드)
DT   = 0.05   # [s]
TAU  = 0.25   # [s]
ALPHA = None

# 퍼블리셔
steering_pub = None
waypoint_index_pub = None
speed_pub = None
rtk_pub = None

# 로깅
log_csv_path = None
_last_log_wall = 0.0

# RTK 상태
rtk_status_txt = "NONE"

# 플롯 객체 (전역으로 보관)
fig = None
ani = None

# ====== 경로/파일 경로 (패키지 기준) ======
def _default_paths():
    try:
        pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
    waypoint_csv = os.path.join(pkg_path, 'config', 'raw_track_latlon_6.csv')
    logs_dir     = os.path.join(pkg_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_csv      = os.path.join(logs_dir, time.strftime('waypoint_log_%Y%m%d_%H%M%S.csv', time.localtime()))
    return waypoint_csv, log_csv

WAYPOINT_CSV_DEFAULT, LOG_CSV_DEFAULT = _default_paths()

# ====== 유틸 함수들 ======
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
        rospy.loginfo("Waypoints from lat/lon columns: (%s, %s)", lat_col, lon_col)
    elif east_col and north_col:
        wx = pd.to_numeric(df[east_col],  errors='coerce').to_numpy()
        wy = pd.to_numeric(df[north_col], errors='coerce').to_numpy()
        rospy.loginfo("Waypoints from east/north columns: (%s, %s)", east_col, north_col)
    else:
        available = ", ".join(sorted(cols))
        raise ValueError("CSV에서 사용할 좌표 컬럼을 찾지 못했습니다. 현재 컬럼: " + available)

    reduced_x = [float(wx[0])]
    reduced_y = [float(wy[0])]
    for x, y in zip(wx[1:], wy[1:]):
        if distance_in_meters(reduced_x[-1], reduced_y[-1], float(x), float(y)) >= MIN_DISTANCE:
            reduced_x.append(float(x))
            reduced_y.append(float(y))

    return np.array([reduced_x, reduced_y])

def signed_angle_deg(v1, v2):
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = float(np.dot(v1, v2)) / (n1 * n2)
    dot = max(min(dot, 1.0), -1.0)
    ang = math.degrees(math.acos(dot))
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return ang if cross >= 0 else -ang

def nearest_in_radius_index(x, y, xs, ys, radius):
    xs = np.asarray(xs); ys = np.asarray(ys)
    d2 = (xs - x)**2 + (ys - y)**2
    i = int(np.argmin(d2))
    if math.hypot(xs[i]-x, ys[i]-y) <= radius:
        return i
    return -1

# LPF (원본)
def lpf_and_invert(current_angle_deg):
    global filtered_steering_angle, ALPHA
    filtered_steering_angle = (1.0 - ALPHA) * filtered_steering_angle + ALPHA * current_angle_deg
    return -filtered_steering_angle

# ====== RTK RELPOSNED 콜백(옵션) ======
_HAVE_RELPOSNED = False
try:
    from ublox_msgs.msg import NavRELPOSNED9 as NavRELPOSNED
    _HAVE_RELPOSNED = True
except Exception:
    try:
        from ublox_msgs.msg import NavRELPOSNED
        _HAVE_RELPOSNED = True
    except Exception:
        _HAVE_RELPOSNED = False

def _cb_relpos(msg):
    global rtk_status_txt
    try:
        carr_soln = int((int(msg.flags) >> 3) & 0x3)
        if carr_soln == 2:   rtk_status_txt = "FIXED"
        elif carr_soln == 1: rtk_status_txt = "FLOAT"
        else:                rtk_status_txt = "NONE"
    except Exception:
        rtk_status_txt = "NONE"

# ====== GPS 콜백: 큐에 적재만 함 ======
from queue import Queue
gps_queue = Queue()
last_fix_time = 0.0

def gps_callback(msg: NavSatFix):
    global last_fix_time
    # 필터: invalid status 처리(있으면)
    if hasattr(msg, "status") and getattr(msg.status, "status", 0) < 0:
        return
    lat, lon = float(msg.latitude), float(msg.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return
    stamp = msg.header.stamp.to_sec() if msg.header and msg.header.stamp else rospy.Time.now().to_sec()
    gps_queue.put((lat, lon, stamp, msg))
    last_fix_time = time.time()

# ====== 큐 처리 / 제어 로직 (타이머 콜백으로 주기 실행) ======
# reduced_waypoints는 main에서 로드
reduced_waypoints = None

def process_loop(event):
    """
    Timer 콜백: DT 주기로 호출됩니다. (원래 main 루프 내부 로직을 옮긴 것)
    - gps_queue를 비우며 위치/조향/퍼블리시/로깅/플롯 갱신 수행
    """
    global current_waypoint_index, reduced_waypoints, _last_log_wall, filtered_steering_angle

    updated = False
    # 한 주기에서 처리할 최대 메시지 수 (너무 오래 걸리지 않게 제한)
    MAX_PER_INVOC = 200

    processed = 0
    while not gps_queue.empty() and processed < MAX_PER_INVOC:
        lat, lon, tsec, raw_msg = gps_queue.get()
        processed += 1
        updated = True

        # 현재 위치(m) 누적
        x, y = lat_lon_to_meters(lat, lon)
        current_x.append(x); current_y.append(y)
        current_t.append(tsec)

        # 현재 목표 웨이포인트
        tx = reduced_waypoints[0][current_waypoint_index]
        ty = reduced_waypoints[1][current_waypoint_index]

        # 도달 판정
        if distance_in_meters(x, y, tx, ty) < TARGET_RADIUS:
            if current_waypoint_index < reduced_waypoints.shape[1] - 1:
                current_waypoint_index += 1
                try:
                    waypoint_index_pub.publish(Int32(current_waypoint_index))
                except Exception:
                    pass

        # 조향각 계산 & 퍼블리시 (직전 위치가 있어야 진행)
        if len(current_x) >= 2 and len(current_t) >= 2:
            prev = np.array([current_x[-2], current_y[-2]])
            curr = np.array([current_x[-1], current_y[-1]])
            head_vec = curr - prev
            tgt_vec  = np.array([tx, ty]) - curr

            heading_deg = math.degrees(math.atan2(head_vec[1], head_vec[0]))
            dt_local = max(1e-3, current_t[-1] - current_t[-2])
            speed_mps = float(np.hypot(head_vec[0], head_vec[1]) / dt_local)

            angle_deg = signed_angle_deg(head_vec, tgt_vec)
            angle_deg = max(min(angle_deg, ANGLE_LIMIT_DEG), -ANGLE_LIMIT_DEG)
            smooth_inv = lpf_and_invert(angle_deg)

            # 퍼블리시
            try:
                if steering_pub: steering_pub.publish(Float32(smooth_inv))
            except Exception:
                pass
            try:
                if speed_pub: speed_pub.publish(Float32(FIXED_SPEED))
            except Exception:
                pass
            try:
                if rtk_pub: rtk_pub.publish(String(rtk_status_txt))
            except Exception:
                pass

            # 로깅 (0.5s 주기)
            noww = time.time()
            if log_csv_path and (noww - _last_log_wall > 0.5):
                try:
                    new = not os.path.exists(log_csv_path)
                    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
                    with open(log_csv_path, 'a', newline='') as f:
                        w = csv.writer(f)
                        if new:
                            w.writerow(['time','lat','lon','speed_mps','heading_deg','steer_deg',
                                        'wp_index(1based)','target_index(1based)','rtk','pub_speed'])
                        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                    f"{lat:.7f}", f"{lon:.7f}",
                                    f"{speed_mps:.2f}", f"{heading_deg:.1f}",
                                    f"{smooth_inv:.2f}",
                                    _wp_in_radius_1based(x, y),
                                    current_waypoint_index + 1,
                                    rtk_status_txt,
                                    f"{FIXED_SPEED:.2f}"])
                    _last_log_wall = noww
                except Exception as e:
                    rospy.logwarn(f"[tracker] log write failed: {e}")

            # 터미널 INFO (throttle)
            rospy.loginfo_throttle(
                0.5,
                f"lat={lat:.7f}, lon={lon:.7f} | v={speed_mps:.2f} m/s | "
                f"heading={heading_deg:.1f}° | steer={smooth_inv:+.2f}° | "
                f"WP(in-radius)={_wp_in_radius_1based(x, y)} | TGT={current_waypoint_index+1} | RTK={rtk_status_txt} | PUB_SPEED={FIXED_SPEED:.2f}"
            )

    # 플롯 갱신 (GUI 스레드에서 안전하게 호출)
    if updated and fig is not None:
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass

def _wp_in_radius_1based(x, y):
    if reduced_waypoints is None: return 0
    idx0 = nearest_in_radius_index(x, y, reduced_waypoints[0], reduced_waypoints[1], TARGET_RADIUS)
    return (idx0 + 1) if idx0 >= 0 else 0

# ====== 시각화 콜백 ======
def update_plot(_):
    ax = plt.gca()
    ax.clear()

    # 웨이포인트 + 반경
    ax.scatter(reduced_waypoints[0], reduced_waypoints[1], s=10, marker='o', label='Waypoints')
    for i, (wx, wy) in enumerate(zip(reduced_waypoints[0], reduced_waypoints[1]), 1):
        ax.add_patch(plt.Circle((wx, wy), TARGET_RADIUS, fill=False, linestyle='--'))
        ax.text(wx, wy, str(i), fontsize=8, ha='center', va='center')

    # 현재 위치 & 타겟(★) & 경로
    if current_x and current_y:
        cx, cy = current_x[-1], current_y[-1]
        ax.scatter(cx, cy, s=50, marker='x', label='Current')

        tx = reduced_waypoints[0][current_waypoint_index]
        ty = reduced_waypoints[1][current_waypoint_index]
        ax.plot([tx], [ty], marker='*', color='k', markersize=12, linestyle='None', label='Target *')
        ax.arrow(cx, cy, tx - cx, ty - cy, head_width=0.5, head_length=0.5)

        k = max(0, len(current_x) - 200)
        for i in range(k + 1, len(current_x)):
            ax.arrow(current_x[i-1], current_y[i-1],
                     current_x[i]-current_x[i-1], current_y[i]-current_y[i-1],
                     head_width=0.2, head_length=0.2, length_includes_head=True)

        # 범례 데이터 계산 (속도는 고정값 FIXED_SPEED)
        speed_mps = 0.0
        heading_deg = float('nan')
        if len(current_x) >= 2 and len(current_t) >= 2:
            dx = current_x[-1] - current_x[-2]
            dy = current_y[-1] - current_y[-2]
            heading_deg = math.degrees(math.atan2(dy, dx))
            dt_local = max(1e-3, current_t[-1] - current_t[-2])
            speed_mps = math.hypot(dx, dy) / dt_local

        wp_in = _wp_in_radius_1based(cx, cy)
        h_speed  = plt.Line2D([], [], linestyle='None', label=f"Speed: {FIXED_SPEED:.2f} m/s")
        h_head   = plt.Line2D([], [], linestyle='None', label=f"Heading: {heading_deg:.1f}°")
        h_steer  = plt.Line2D([], [], linestyle='None', label=f"Steer: {filtered_steering_angle:+.1f}°")
        h_rtk    = plt.Line2D([], [], linestyle='None', label=f"RTK: {rtk_status_txt}")
        h_idx    = plt.Line2D([], [], linestyle='None', label=f"WP/TGT: {wp_in}/{current_waypoint_index+1}")

        ax.legend(handles=[h_speed, h_head, h_steer, h_rtk, h_idx], loc='best')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True)
    ax.set_title('Filtered steering (inverted) · target starred; legend shows speed/heading/steer/RTK/WP·TGT')

# ====== 종료시 0 퍼블리시 핸들러 ======
def _on_shutdown():
    rospy.loginfo("[tracker] shutdown: publishing zeros to speed and steer")
    try:
        if speed_pub is not None:
            speed_pub.publish(Float32(0.0))
        if steering_pub is not None:
            steering_pub.publish(Float32(0.0))
    except Exception as e:
        rospy.logwarn(f"[tracker] shutdown publish failed: {e}")
    time.sleep(0.1)

# ====== 메인 ======
def main():
    global steering_pub, waypoint_index_pub, speed_pub, rtk_pub
    global reduced_waypoints, DT, TAU, ALPHA, log_csv_path
    global fig, ani

    rospy.init_node('gps_waypoint_tracker', anonymous=True)
    rospy.on_shutdown(_on_shutdown)

    # 파라미터
    csv_path    = rospy.get_param('~csv_path', WAYPOINT_CSV_DEFAULT)
    fix_topic   = rospy.get_param('~fix_topic',   '/gps1/fix')
    relpos_topic= rospy.get_param('~relpos_topic','/gps1/navrelposned')
    enable_plot = rospy.get_param('~enable_plot', True)
    DT  = float(rospy.get_param('~dt',  0.05))
    TAU = float(rospy.get_param('~tau', 0.25))
    ALPHA = DT / (TAU + DT)

    log_csv_path = rospy.get_param('~log_csv', LOG_CSV_DEFAULT)

    rospy.loginfo("LPF params: dt=%.3fs, tau=%.3fs, alpha=%.3f", DT, TAU, ALPHA)
    rospy.loginfo("CSV: %s | LOG: %s", csv_path, log_csv_path)
    rospy.loginfo("Using FIXED_SPEED = %.2f m/s for legend/info/publish", FIXED_SPEED)

    # 퍼블리셔
    steering_pub = rospy.Publisher('/vehicle/steer_cmd', Float32, queue_size=10)
    waypoint_index_pub = rospy.Publisher('/current_waypoint_index', Int32, queue_size=10)
    speed_pub = rospy.Publisher('/vehicle/speed_cmd', Float32, queue_size=10)
    rtk_pub   = rospy.Publisher('/rtk/status',      String,  queue_size=10)

    # 서브스크라이브
    rospy.Subscriber(fix_topic, NavSatFix, gps_callback)
    if _HAVE_RELPOSNED:
        try:
            rospy.Subscriber(relpos_topic, NavRELPOSNED, _cb_relpos)
            rospy.loginfo("Subscribed: %s (RTK status ON)", relpos_topic)
        except Exception:
            rospy.logwarn("RELPOSNED subscribe failed; RTK status disabled.")
    rospy.loginfo("Subscribed NavSatFix: %s", fix_topic)

    # 웨이포인트 로드 (원본 방식)
    global reduced_waypoints
    reduced_waypoints = build_reduced_waypoints(csv_path)
    rospy.loginfo("Waypoints loaded: %d (reduced spacing >= %.1fm)", reduced_waypoints.shape[1], MIN_DISTANCE)

    # 플롯 준비 (옵션)
    if enable_plot:
        fig = plt.figure()
        # animation의 interval을 DT에 맞춤(밀리초)
        try:
            ani = animation.FuncAnimation(fig, update_plot, interval=max(1, int(DT * 1000)))
        except Exception:
            # 폴백: 300ms
            ani = animation.FuncAnimation(fig, update_plot, interval=300)
        plt.show(block=False)
    else:
        fig = None

    # Timer로 주기적 처리 등록 (DT 초 마다)
    rospy.Timer(rospy.Duration(DT), process_loop)

    # ROS spin: Timer와 콜백은 별도 스레드에서 동작하므로 여기서는 기다리기만 하면 됩니다.
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
