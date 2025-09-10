#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 + matplotlib · GPS waypoint 추종 & 조향각 퍼블리시 (정리판 · 알고리즘 그대로)
- 입력: (param) ~fix_topic  (sensor_msgs/NavSatFix, default=/gps1/fix)
- 출력:
  * /filtered_steering_angle (std_msgs/Float32)  → LPF + "부호 반전 유지"
  * /current_waypoint_index (std_msgs/Int32)     → 현재 목표 웨이포인트 인덱스(0-based, 원본 그대로)
  * /vehicle/speed_cmd       (std_msgs/Float32)  → 고정 속도 값(요청 추가)
  * /rtk/status              (std_msgs/String)   → "FIXED"/"FLOAT"/"NONE"(요청 추가)
- 파라미터:
  * ~csv_path     (str)   : 웨이포인트 CSV 경로 (기본: ~/rtk_waypoint_tracker/config/left_lane.csv)
  * ~fix_topic    (str)   : NavSatFix 토픽      (기본: /gps1/fix)
  * ~relpos_topic (str)   : u-blox RELPOSNED    (기본: /gps1/navrelposned)  # RTK 상태용(있을 때만 사용)
  * ~enable_plot  (bool)  : matplotlib 표시     (기본: true)
  * ~dt           (float) : 필터 샘플 주기 [s]  (기본: 0.05 = 50ms)
  * ~tau          (float) : 1차 LPF 시간상수[s] (기본: 0.25)
  * ~fixed_speed  (float) : 퍼블리시할 고정 속도 값 [m/s] (기본: 2.0)
  * ~log_csv      (str)   : 로그 CSV 경로 (기본: ~/rtk_waypoint_tracker/logs/waypoint_log_YYYYmmdd_HHMMSS.csv)
"""

import os, math, time, csv
import rospy
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

# ====== 전역 상태 ======
filtered_steering_angle = 0.0
current_x, current_y = [], []
current_t = []                  # 시간(초) 저장 → 속도 계산용
current_waypoint_index = 0

# 필터 파라미터(파라미터 서버에서 로드)
DT   = 0.05   # [s]   기본 50ms
TAU  = 0.25   # [s]
ALPHA = None  # dt/(tau+dt)

# 퍼블리셔
steering_pub = None
waypoint_index_pub = None
speed_pub = None                # 추가: 고정 속도 퍼블리시
rtk_pub = None                  # 추가: RTK 상태 퍼블리시

# 로깅
log_csv_path = None
_last_log_wall = 0.0

# RTK 상태
rtk_status_txt = "NONE"

# ====== 유틸 ======
def lat_lon_to_meters(lat, lon):
    """(deg, deg) -> (m, m) Web Mercator 근사 (소구역에서 오차 작음)"""
    x = EARTH_RADIUS * lon * math.pi / 180.0
    y = EARTH_RADIUS * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def distance_in_meters(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def _pick_column(df_cols, candidates):
    """df 컬럼 목록(df_cols)에서 candidates 중 첫 번째로 일치하는 이름을 반환 (소문자 매칭). 없으면 None."""
    for c in candidates:
        if c in df_cols:
            return c
    return None

def build_reduced_waypoints(csv_path):
    """
    CSV 컬럼 자동 인식:
      - 위/경도(도): (latitude|lat), (longitude|lon)
      - 동/북   (m): (east|easting|x), (north|northing|y)
    어떤 모드를 썼는지 INFO 로깅.
    """
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
        raise ValueError(
            "CSV에서 사용할 좌표 컬럼을 찾지 못했습니다.\n"
            f"- 허용되는 조합 1) lat/lon  2) east/north (또는 x/y)\n"
            f"- 현재 컬럼: {available}"
        )

    # 간격 축소(>= MIN_DISTANCE)
    reduced_x = [float(wx[0])]
    reduced_y = [float(wy[0])]
    for x, y in zip(wx[1:], wy[1:]):
        if distance_in_meters(reduced_x[-1], reduced_y[-1], float(x), float(y)) >= MIN_DISTANCE:
            reduced_x.append(float(x))
            reduced_y.append(float(y))

    return np.array([reduced_x, reduced_y])  # shape: (2, N)

def signed_angle_deg(v1, v2):
    """두 벡터 사이 부호 있는 각도(도). 반시계 양수, 시계 음수."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = float(np.dot(v1, v2)) / (n1 * n2)
    dot = max(min(dot, 1.0), -1.0)
    ang = math.degrees(math.acos(dot))
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return ang if cross >= 0 else -ang

def nearest_in_radius_index(x, y, xs, ys, radius):
    """반경 radius 안에 들어오면 0-based 인덱스, 아니면 -1"""
    xs = np.asarray(xs); ys = np.asarray(ys)
    d2 = (xs - x)**2 + (ys - y)**2
    i = int(np.argmin(d2))
    if math.hypot(xs[i]-x, ys[i]-y) <= radius:
        return i
    return -1

# === 시간상수 기반 LPF (원본) + 부호 반전 유지 ===
def lpf_and_invert(current_angle_deg):
    """
    1차 저역필터:
      y[k] = (1-ALPHA)*y[k-1] + ALPHA*x[k],  where ALPHA = DT/(TAU+DT)
    이후 '부호 반전 유지'를 적용해 반환.
    """
    global filtered_steering_angle, ALPHA
    filtered_steering_angle = (1.0 - ALPHA) * filtered_steering_angle + ALPHA * current_angle_deg
    return -filtered_steering_angle  # 부호 반전 유지(원본 동작)

# ====== RTK 상태(옵션: u-blox RELPOSNED 있을 때만 사용) ======
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
        # UBX NAV-RELPOSNED flags.carrSoln: 0=NONE, 1=FLOAT, 2=FIXED
        carr_soln = int((int(msg.flags) >> 3) & 0x3)
        if carr_soln == 2:   rtk_status_txt = "FIXED"
        elif carr_soln == 1: rtk_status_txt = "FLOAT"
        else:                rtk_status_txt = "NONE"
    except Exception:
        rtk_status_txt = "NONE"

# ====== 콜백 ======
reduced_waypoints = None  # main에서 로드
def gps_callback(msg: NavSatFix):
    global current_waypoint_index, reduced_waypoints, _last_log_wall

    # 현재 위치(m) + 시간 누적
    x, y = lat_lon_to_meters(msg.latitude, msg.longitude)
    current_x.append(x); current_y.append(y)
    tsec = msg.header.stamp.to_sec() if msg.header and msg.header.stamp else rospy.Time.now().to_sec()
    current_t.append(tsec)

    # 현재 목표 웨이포인트(원본 알고리즘: 인덱스 그대로)
    tx = reduced_waypoints[0][current_waypoint_index]
    ty = reduced_waypoints[1][current_waypoint_index]

    # 도달 판정(원본 그대로)
    if distance_in_meters(x, y, tx, ty) < TARGET_RADIUS:
        if current_waypoint_index < reduced_waypoints.shape[1] - 1:
            current_waypoint_index += 1
            waypoint_index_pub.publish(Int32(current_waypoint_index))

    # 조향각 계산 & 퍼블리시 (직전 위치가 있어야 진행)
    if len(current_x) >= 2:
        prev = np.array([current_x[-2], current_y[-2]])
        curr = np.array([current_x[-1], current_y[-1]])
        head_vec = curr - prev
        tgt_vec  = np.array([tx, ty]) - curr

        # 헤딩/속도 계산(로깅/범례용)
        heading_deg = math.degrees(math.atan2(head_vec[1], head_vec[0]))
        dt = max(1e-3, current_t[-1] - current_t[-2])
        speed_mps = float(np.hypot(head_vec[0], head_vec[1]) / dt)

        angle_deg = signed_angle_deg(head_vec, tgt_vec)
        # 제한
        angle_deg = max(min(angle_deg, ANGLE_LIMIT_DEG), -ANGLE_LIMIT_DEG)
        # 시간상수 기반 LPF + 부호 반전
        smooth_inv = lpf_and_invert(angle_deg)

        # 퍼블리시(원본 + 추가: 속도, RTK)
        steering_pub.publish(Float32(smooth_inv))
        if speed_pub:
            speed_pub.publish(Float32(rospy.get_param('~fixed_speed', 2.0)))  # 고정 속도 값 퍼블리시
        if rtk_pub:
            rtk_pub.publish(String(rtk_status_txt))

        # ── 로깅 (0.5s 주기) ──
        noww = time.time()
        if log_csv_path and (noww - _last_log_wall > 0.5):
            try:
                new = not os.path.exists(log_csv_path)
                os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
                with open(log_csv_path, 'a', newline='') as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(['time','lat','lon','speed_mps','heading_deg','steer_deg',
                                    'wp_index(1based)','target_index(1based)','rtk'])
                    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                f"{msg.latitude:.7f}", f"{msg.longitude:.7f}",
                                f"{speed_mps:.2f}", f"{heading_deg:.1f}",
                                f"{smooth_inv:.2f}",
                                # 현재 내가 '반경 내'에 있는 인덱스
                                _wp_in_radius_1based(x, y),
                                current_waypoint_index + 1,
                                rtk_status_txt])
                _last_log_wall = noww
            except Exception as e:
                rospy.logwarn(f"[tracker] log write failed: {e}")

        # ── 터미널 INFO (0.5s) ──
        rospy.loginfo_throttle(
            0.5,
            f"lat={msg.latitude:.7f}, lon={msg.longitude:.7f} | v={speed_mps:.2f} m/s | "
            f"heading={heading_deg:.1f}° | steer={smooth_inv:+.2f}° | "
            f"WP(in-radius)={_wp_in_radius_1based(x, y)} | TGT={current_waypoint_index+1} | RTK={rtk_status_txt}"
        )

def _wp_in_radius_1based(x, y):
    """현재 위치가 반경 내에 있으면 1-based 인덱스, 아니면 0"""
    if reduced_waypoints is None: return 0
    idx0 = nearest_in_radius_index(x, y, reduced_waypoints[0], reduced_waypoints[1], TARGET_RADIUS)
    return (idx0 + 1) if idx0 >= 0 else 0

# ====== 시각화 ======
def update_plot(_):
    ax = plt.gca()
    ax.clear()

    # 웨이포인트 + 반경
    ax.scatter(reduced_waypoints[0], reduced_waypoints[1], s=10, marker='o', label='Waypoints')
    for i, (wx, wy) in enumerate(zip(reduced_waypoints[0], reduced_waypoints[1]), 1):
        ax.add_patch(plt.Circle((wx, wy), TARGET_RADIUS, fill=False, linestyle='--'))
        ax.text(wx, wy, str(i), fontsize=8, ha='center', va='center')

    # 현재 위치 & 타겟(★ 별표) & 벡터
    if current_x and current_y:
        cx, cy = current_x[-1], current_y[-1]
        ax.scatter(cx, cy, s=50, marker='x', label='Current')

        tx = reduced_waypoints[0][current_waypoint_index]
        ty = reduced_waypoints[1][current_waypoint_index]
        ax.plot([tx], [ty], marker='*', color='k', markersize=12, linestyle='None', label='Target *')  # ★ 별표
        ax.arrow(cx, cy, tx - cx, ty - cy, head_width=0.5, head_length=0.5)

        # 이동 경로(최근 200점만)
        k = max(0, len(current_x) - 200)
        for i in range(k + 1, len(current_x)):
            ax.arrow(current_x[i-1], current_y[i-1],
                     current_x[i]-current_x[i-1], current_y[i]-current_y[i-1],
                     head_width=0.2, head_length=0.2, length_includes_head=True)

        # 동적 범례용 값 계산
        speed_mps = 0.0
        heading_deg = float('nan')
        if len(current_x) >= 2 and len(current_t) >= 2:
            dx = current_x[-1] - current_x[-2]
            dy = current_y[-1] - current_y[-2]
            heading_deg = math.degrees(math.atan2(dy, dx))
            dt = max(1e-3, current_t[-1] - current_t[-2])
            speed_mps = math.hypot(dx, dy) / dt

        wp_in = _wp_in_radius_1based(cx, cy)
        # 동적 텍스트만을 위한 더미 라인 핸들들
        h_speed  = plt.Line2D([], [], linestyle='None', label=f"Speed: {speed_mps:.2f} m/s")
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

# ====== 메인 ======
def main():
    global steering_pub, waypoint_index_pub, speed_pub, rtk_pub
    global reduced_waypoints, DT, TAU, ALPHA, log_csv_path

    rospy.init_node('gps_waypoint_tracker', anonymous=True)

    # 파라미터
    csv_path    = rospy.get_param('~csv_path',    os.path.expanduser('~/rtk_waypoint_tracker/config/left_lane.csv'))
    fix_topic   = rospy.get_param('~fix_topic',   '/gps1/fix')
    relpos_topic= rospy.get_param('~relpos_topic','/gps1/navrelposned')  # RTK 상태용(옵션)
    enable_plot = rospy.get_param('~enable_plot', True)
    DT  = float(rospy.get_param('~dt',  0.05))   # 50ms 기본
    TAU = float(rospy.get_param('~tau', 0.25))   # 0.25s 기본
    ALPHA = DT / (TAU + DT)

    # 로그 파일 경로
    default_log_dir = os.path.expanduser('~/rtk_waypoint_tracker/logs')
    os.makedirs(default_log_dir, exist_ok=True)
    default_log_name = time.strftime('waypoint_log_%Y%m%d_%H%M%S.csv', time.localtime())
    log_csv_path = os.path.expanduser(rospy.get_param('~log_csv', os.path.join(default_log_dir, default_log_name)))

    rospy.loginfo("LPF params: dt=%.3fs, tau=%.3fs, alpha=%.3f", DT, TAU, ALPHA)
    rospy.loginfo("CSV: %s | LOG: %s", csv_path, log_csv_path)

    # 퍼블리셔 생성(원본 + 추가)
    steering_pub = rospy.Publisher('/filtered_steering_angle', Float32, queue_size=10)
    waypoint_index_pub = rospy.Publisher('/current_waypoint_index', Int32, queue_size=10)
    speed_pub = rospy.Publisher('/vehicle/speed_cmd', Float32, queue_size=10)  # 추가
    rtk_pub   = rospy.Publisher('/rtk/status',      String,  queue_size=10)    # 추가

    # 서브스크라이브
    rospy.Subscriber(fix_topic, NavSatFix, gps_callback)
    if _HAVE_RELPOSNED:
        try:
            rospy.Subscriber(relpos_topic, NavRELPOSNED, _cb_relpos)
            rospy.loginfo("Subscribed: %s (RTK status ON)", relpos_topic)
        except Exception:
            rospy.logwarn("RELPOSNED subscribe failed; RTK status disabled.")
    rospy.loginfo("Subscribed NavSatFix: %s", fix_topic)

    # 웨이포인트 구성(원본)
    global reduced_waypoints
    reduced_waypoints = build_reduced_waypoints(csv_path)
    rospy.loginfo("Waypoints loaded: %d (reduced spacing >= %.1fm)", reduced_waypoints.shape[1], MIN_DISTANCE)

    if enable_plot:
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, update_plot, interval=300)  # 300ms 주기 업데이트
        plt.show()
    else:
        rospy.loginfo("Headless mode: plotting disabled (~enable_plot=false).")
        rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
