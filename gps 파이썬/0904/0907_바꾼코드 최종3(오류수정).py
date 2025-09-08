#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import time
from collections import deque

import rospy
import rospkg
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, String, Int32

import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import geopy.distance
import pandas as pd
import numpy as np
from queue import Queue

# ──────────────────────────────────────────────────────────────────
# 기본 파라미터 (필요시 rosparam으로 덮어쓰기)
# ──────────────────────────────────────────────────────────────────
WAYPOINT_SPACING   = 2.5         # m
TARGET_RADIUS_END  = 2.0         # m  (반경 안이면 '그 WP에 있다'로 간주)
MAX_STEER_DEG      = 20.0        # deg
SIGN_CONVENTION    = -1.0

LOOKAHEAD_MIN      = 3.2         # m
LOOKAHEAD_MAX      = 4.0         # m
LOOKAHEAD_K        = 0.2         # m per (m/s)

LPF_FC_HZ          = 0.8         # Hz (조향 LPF)
SPEED_BUF_LEN      = 10
MAX_JITTER_SPEED   = 4.0         # m/s
MIN_MOVE_FOR_HEADING = 0.05      # m

FS_DEFAULT         = 20.0        # 퍼블리시/루프 Hz
GPS_TIMEOUT_SEC    = 1.0         # 최근 fix 없을 때 안전정지

# ── 속도 명령 상수(코드에서 바로 고정하고 싶을 때) ──
#  - 0~4 사이 정수. None이면 rosparam(~speed_code) 사용.
SPEED_FORCE_CODE = 2          # 예: 1 또는 2로 고정. rosparam 무시
# SPEED_FORCE_CODE = None     # ← 이렇게 두면 rosparam(~speed_code) 사용
SPEED_CAP_CODE_DEFAULT = 4    # 코드 상한(0~4)

# 시각화 옵션
ANNOTATE_WAYPOINT_INDEX = True
DRAW_WAYPOINT_CIRCLES   = True

# 퍼블리시 토픽
TOPIC_SPEED_CMD    = '/vehicle/speed_cmd'     # Float32 (코드 0~4를 그대로 실수로 전송)
TOPIC_STEER_CMD    = '/vehicle/steer_cmd'     # Float32 (deg)
TOPIC_RTK_STATUS   = '/rtk/status'            # String ("FIX"/"FLOAT"/"NONE")
TOPIC_WP_INDEX     = '/tracker/wp_index'      # Int32 (반경 밖=0, 안=해당 인덱스 1-based)

# u-blox 옵셔널 의존성 (RTK 상태)
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

# ──────────────────────────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────────────────────────
gps_queue = Queue()
latest_filtered_angle = 0.0
pos_buf   = deque(maxlen=20)
speed_buf = deque(maxlen=SPEED_BUF_LEN)

pub_speed = None
pub_steer = None
pub_rtk   = None
pub_wpidx = None

rtk_status_txt = "NONE"
last_fix_time  = 0.0
# 퍼블리시/표시용 WP 인덱스: 반경 밖이면 -1, 안이면 0-based 인덱스
wp_index_active = -1

# 로그
log_csv_path = None
_last_log_wall = 0.0

# ──────────────────────────────────────────────────────────────────
# 경로/파일 경로 (내 코드 방식)
# ──────────────────────────────────────────────────────────────────
def _default_paths():
    try:
        pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')

    # 패키지 내부 config/left_lane.csv를 기본으로 사용
    waypoint_csv = os.path.join(pkg_path, 'config', 'left_lane.csv')
    logs_dir     = os.path.join(pkg_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_csv      = os.path.join(logs_dir, f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    return waypoint_csv, log_csv

WAYPOINT_CSV_DEFAULT, LOG_CSV_DEFAULT = _default_paths()

# ──────────────────────────────────────────────────────────────────
# 좌표/웨이포인트 유틸
# ──────────────────────────────────────────────────────────────────
def latlon_to_xy_fn(ref_lat, ref_lon):
    def _to_xy(lat, lon):
        northing = geopy.distance.geodesic((ref_lat, ref_lon), (lat, ref_lon)).meters
        easting  = geopy.distance.geodesic((ref_lat, ref_lon), (ref_lat, lon)).meters
        if lat < ref_lat: northing *= -1
        if lon < ref_lon: easting  *= -1
        return easting, northing
    return _to_xy

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def generate_waypoints_along_path(path, spacing=3.0):
    if len(path) < 2:
        return path
    new_points = [path[0]]
    dist_accum = 0.0
    last_point = np.array(path[0], dtype=float)
    for i in range(1, len(path)):
        current_point = np.array(path[i], dtype=float)
        segment = current_point - last_point
        seg_len = np.linalg.norm(segment)
        while dist_accum + seg_len >= spacing:
            remain = spacing - dist_accum
            direction = segment / seg_len
            new_point = last_point + direction * remain
            new_points.append(tuple(new_point))
            last_point = new_point
            segment = current_point - last_point
            seg_len = np.linalg.norm(segment)
            dist_accum = 0.0
        dist_accum += seg_len
        last_point = current_point
    if euclidean_dist(new_points[-1], path[-1]) > 1e-6:
        new_points.append(path[-1])
    return new_points

def wrap_deg(a): return (a + 180.0) % 360.0 - 180.0

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float); v2 = np.array(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0: return 0.0
    dot = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    ang = math.degrees(math.acos(dot))
    if v1[0]*v2[1] - v1[1]*v2[0] < 0: ang = -ang
    return ang

class AngleLPF:
    def __init__(self, fc_hz=3.0, init_deg=0.0):
        self.fc = fc_hz
        self.y = init_deg
        self.t_last = None
    def update(self, target_deg, t_sec):
        if self.t_last is None:
            self.t_last = t_sec
            self.y = target_deg
            return self.y
        dt = max(1e-3, t_sec - self.t_last)
        tau = 1.0 / (2.0 * math.pi * self.fc)
        alpha = dt / (tau + dt)
        err = wrap_deg(target_deg - self.y)
        self.y = wrap_deg(self.y + alpha * err)
        self.t_last = t_sec
        return self.y

def find_nearest_index(x, y, xs, ys, start_idx, window_ahead=60, window_back=10):
    n = len(xs)
    i0, i1 = max(0, start_idx - window_back), min(n - 1, start_idx + window_ahead)
    sub_x, sub_y = np.array(xs[i0:i1+1]), np.array(ys[i0:i1+1])
    d2 = (sub_x - x)**2 + (sub_y - y)**2
    return i0 + int(np.argmin(d2))

def target_index_from_lookahead(nearest_idx, Ld, spacing, n):
    steps = max(1, int(math.ceil(Ld / max(1e-6, spacing))))
    return min(n - 1, nearest_idx + steps)

def nearest_in_radius_index(x, y, xs, ys, radius):
    """반경 radius 안에 들어오면 0-based 인덱스, 아니면 -1"""
    xs = np.asarray(xs); ys = np.asarray(ys)
    d2 = (xs - x)**2 + (ys - y)**2
    i = int(np.argmin(d2))
    if math.hypot(xs[i]-x, ys[i]-y) <= radius:
        return i
    return -1

# ──────────────────────────────────────────────────────────────────
# ROS 콜백/퍼블리시
# ──────────────────────────────────────────────────────────────────
def gps_callback(data: NavSatFix):
    global last_fix_time
    if hasattr(data, "status") and getattr(data.status, "status", 0) < 0:
        return
    lat, lon = float(data.latitude), float(data.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return
    stamp = data.header.stamp.to_sec() if data.header and data.header.stamp else rospy.Time.now().to_sec()
    gps_queue.put((lat, lon, stamp))
    last_fix_time = time.time()

def _cb_relpos(msg):
    # UBX NAV-RELPOSNED flags.carrSoln: 0=NONE, 1=FLOAT, 2=FIXED
    global rtk_status_txt
    try:
        carr_soln = int((int(msg.flags) >> 3) & 0x3)  # 메시지 정의에 따라 달라질 수 있음
        if carr_soln == 2:   rtk_status_txt = "FIX"
        elif carr_soln == 1: rtk_status_txt = "FLOAT"
        else:                rtk_status_txt = "NONE"
    except Exception:
        rtk_status_txt = "NONE"

def publish_all(event, speed_code_default=1, one_based=True):
    """주기 퍼블리시(속도/조향/RTK/인덱스) + 타임아웃 안전정지"""
    now = time.time()
    no_gps = (now - last_fix_time) > rospy.get_param('~gps_timeout_sec', GPS_TIMEOUT_SEC)

    # 속도 결정
    if SPEED_FORCE_CODE is not None:
        code = int(SPEED_FORCE_CODE); cap = int(SPEED_CAP_CODE_DEFAULT)
    else:
        code = int(rospy.get_param('~speed_code', speed_code_default))
        cap  = int(rospy.get_param('~speed_cap_code', SPEED_CAP_CODE_DEFAULT))
    code = max(0, min(code, cap))

    v_out = 0.0 if no_gps else float(code)
    steer_out = 0.0 if no_gps else float(latest_filtered_angle)

    if pub_speed: pub_speed.publish(Float32(v_out))
    if pub_steer: pub_steer.publish(Float32(steer_out))
    if pub_rtk:   pub_rtk.publish(String(rtk_status_txt))
    if pub_wpidx:
        # 반경 밖이면 0, 안이면 1-based 인덱스 퍼블리시
        if wp_index_active >= 0:
            idx_pub = (wp_index_active + 1) if one_based else wp_index_active
        else:
            idx_pub = 0
        pub_wpidx.publish(Int32(int(idx_pub)))

def _on_shutdown():
    try:
        if pub_speed: pub_speed.publish(Float32(0.0))
        if pub_steer: pub_steer.publish(Float32(0.0))
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────
def main():
    global pub_speed, pub_steer, pub_rtk, pub_wpidx
    global latest_filtered_angle, log_csv_path, wp_index_active
    global _last_log_wall

    rospy.init_node('rtk_waypoint_tracker', anonymous=False)
    rospy.on_shutdown(_on_shutdown)

    # 파라미터
    waypoint_csv = rospy.get_param('~waypoint_csv', WAYPOINT_CSV_DEFAULT)
    log_csv_path = rospy.get_param('~log_csv',      LOG_CSV_DEFAULT)
    fs           = float(rospy.get_param('~fs',     FS_DEFAULT))
    one_based    = bool(rospy.get_param('~wp_index_one_based', True))
    ublox_ns     = rospy.get_param('~ublox_ns', '/gps1')
    fix_topic    = rospy.get_param('~fix_topic',    ublox_ns + '/fix')
    relpos_topic = rospy.get_param('~relpos_topic', ublox_ns + '/navrelposned')

    # 퍼블리셔/구독자
    pub_speed = rospy.Publisher(TOPIC_SPEED_CMD, Float32, queue_size=10)
    pub_steer = rospy.Publisher(TOPIC_STEER_CMD, Float32, queue_size=10)
    pub_rtk   = rospy.Publisher(TOPIC_RTK_STATUS, String,  queue_size=10)
    pub_wpidx = rospy.Publisher(TOPIC_WP_INDEX,   Int32,   queue_size=10)

    rospy.Subscriber(fix_topic, NavSatFix, gps_callback, queue_size=100)
    if _HAVE_RELPOSNED:
        rospy.Subscriber(relpos_topic, NavRELPOSNED, _cb_relpos, queue_size=50)
    rospy.loginfo("[tracker] subscribe: fix=%s, relpos=%s(%s)", fix_topic, relpos_topic, "ON" if _HAVE_RELPOSNED else "OFF")

    # CSV 로드
    if not os.path.exists(waypoint_csv):
        rospy.logerr("[tracker] waypoint csv not found: %s", waypoint_csv)
        return

    df = pd.read_csv(waypoint_csv)
    ref_lat = float(df['Lat'][0]); ref_lon = float(df['Lon'][0])
    to_xy = latlon_to_xy_fn(ref_lat, ref_lon)

    csv_coords = [to_xy(row['Lat'], row['Lon']) for _, row in df.iterrows()]
    original_path = list(csv_coords)
    spaced_waypoints = generate_waypoints_along_path(original_path, spacing=WAYPOINT_SPACING)
    spaced_x, spaced_y = zip(*spaced_waypoints)

    # ── 플롯: 상단 경로 + 하단 상태패널 ──
    plt.ion()
    fig = plt.figure(figsize=(7.5, 8.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
    ax  = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[1, 0]); ax_info.axis('off')

    ax.plot([p[0] for p in csv_coords], [p[1] for p in csv_coords], 'g-', label='CSV Path')
    ax.plot(spaced_x, spaced_y, 'b.-', markersize=3, label=f'{WAYPOINT_SPACING:.0f}m Waypoints')
    live_line,   = ax.plot([], [], 'r-', linewidth=1, label='Live GPS')
    current_pt,  = ax.plot([], [], 'ro', label='Current')
    target_line, = ax.plot([], [], 'g--', linewidth=1, label='Target Line')
    ax.axis('equal'); ax.grid(True); ax.legend()

    minx, maxx = min(min([p[0] for p in csv_coords]), min(spaced_x))-10, max(max([p[0] for p in csv_coords]), max(spaced_x))+10
    miny, maxy = min(min([p[1] for p in csv_coords]), min(spaced_y))-10, max(max([p[1] for p in csv_coords]), max(spaced_y))+10
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

    if ANNOTATE_WAYPOINT_INDEX or DRAW_WAYPOINT_CIRCLES:
        for idx, (xw, yw) in enumerate(zip(spaced_x, spaced_y), 1):
            if ANNOTATE_WAYPOINT_INDEX:
                ax.text(xw, yw, str(idx), fontsize=8, ha='center', va='center', color='black')
            if DRAW_WAYPOINT_CIRCLES:
                ax.add_patch(Circle((xw, yw), TARGET_RADIUS_END, color='blue', fill=False, linestyle='--', alpha=0.5))

    # 조향 필터/상태
    lpf = AngleLPF(fc_hz=LPF_FC_HZ)
    prev_Ld = LOOKAHEAD_MIN
    nearest_idx_prev = 0
    last_heading_vec = None

    # 퍼블리시 타이머
    rospy.Timer(rospy.Duration(1.0/max(1.0, fs)), lambda e: publish_all(e, speed_code_default=1, one_based=one_based))

    rate = rospy.Rate(fs)
    try:
        while not rospy.is_shutdown():
            updated = False
            while not gps_queue.empty():
                lat, lon, tsec = gps_queue.get()
                x, y = to_xy(lat, lon)
                updated = True

                # 속도 추정(스파이크 제거)
                if len(pos_buf) > 0:
                    t_prev, x_prev, y_prev = pos_buf[-1]
                    dt = max(1e-3, tsec - t_prev)
                    d  = math.hypot(x - x_prev, y - y_prev)
                    inst_v = d / dt
                    if inst_v > MAX_JITTER_SPEED:
                        continue
                    speed_buf.append(inst_v)
                pos_buf.append((tsec, x, y))
                speed_mps = float(np.median(speed_buf)) if speed_buf else 0.0

                # 제어용 타겟 인덱스(룩어헤드)
                nearest_idx = find_nearest_index(x, y, spaced_x, spaced_y, nearest_idx_prev, 80, 15)
                nearest_idx_prev = nearest_idx

                Ld_target = max(LOOKAHEAD_MIN, min(LOOKAHEAD_MAX, LOOKAHEAD_MIN + LOOKAHEAD_K * speed_mps))
                Ld = prev_Ld + 0.2 * (Ld_target - prev_Ld)
                prev_Ld = Ld

                tgt_idx = target_index_from_lookahead(nearest_idx, Ld, WAYPOINT_SPACING, len(spaced_x))
                tx, ty = spaced_x[tgt_idx], spaced_y[tgt_idx]
                target_line.set_data([x, tx], [y, ty])

                # 퍼블리시/표시용 인덱스: 반경 안이면 0-based, 아니면 -1
                wp_index_active = nearest_in_radius_index(x, y, spaced_x, spaced_y, TARGET_RADIUS_END)

                # 헤딩 추정(최근 이동)
                heading_vec = None
                for k in range(2, min(len(pos_buf), 5)+1):
                    t0, x0, y0 = pos_buf[-k]
                    if math.hypot(x - x0, y - y0) >= MIN_MOVE_FOR_HEADING:
                        heading_vec = (x - x0, y - y0)
                        break
                if heading_vec is not None:
                    last_heading_vec = heading_vec
                elif last_heading_vec is None:
                    # 아직 헤딩 부트스트랩 안됨
                    continue

                # 조향 계산 + 동적 LPF
                target_vec = (tx - x, ty - y)
                raw_angle = angle_between(last_heading_vec, target_vec)
                base_fc = LPF_FC_HZ
                lpf.fc = min(2.0, base_fc + 0.5) if abs(raw_angle) > 10 else base_fc
                filt_angle = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, lpf.update(raw_angle, tsec)))
                latest_filtered_angle = SIGN_CONVENTION * filt_angle

                # 시각화 갱신
                live_line.set_data([p[1] for p in pos_buf], [p[2] for p in pos_buf])
                current_pt.set_data([x], [y])

                # ── 하단 상태 패널 ──
                ax_info.clear(); ax_info.axis('off')
                heading_deg = (math.degrees(math.atan2(last_heading_vec[1], last_heading_vec[0]))
                               if last_heading_vec is not None else float('nan'))
                wp_display = (wp_index_active + 1) if (wp_index_active >= 0) else 0
                status = [
                    f"Speed(EMA): {speed_mps:.2f} m/s",
                    f"Heading: {heading_deg:.1f}°",
                    f"Steer: {latest_filtered_angle:+.1f}°",
                    f"WP(in-radius): {wp_display}",
                    f"RTK: {rtk_status_txt}"
                ]
                ax_info.text(0.02, 0.5, " | ".join(status), fontsize=11, va='center')

                # 로그 (0.5s에 한번)
                noww = time.time()
                if log_csv_path and (noww - _last_log_wall > 0.5):
                    try:
                        new = not os.path.exists(log_csv_path)
                        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
                        with open(log_csv_path, 'a', newline='') as f:
                            w = csv.writer(f)
                            if new:
                                w.writerow(['time','lat','lon','x','y',
                                            'wp_in_radius(1based)','tgt_idx(1based)',
                                            'tx','ty',
                                            'steer_deg','speed_mps','Ld','rtk'])
                            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                        f"{lat:.7f}", f"{lon:.7f}", f"{x:.3f}", f"{y:.3f}",
                                        wp_display, tgt_idx+1, f"{tx:.3f}", f"{ty:.3f}",
                                        f"{latest_filtered_angle:.2f}", f"{speed_mps:.2f}",
                                        f"{Ld:.2f}", rtk_status_txt])
                        _last_log_wall = noww
                    except Exception as e:
                        rospy.logwarn(f"[tracker] log write failed: {e}")

                # 콘솔 디버그
                rospy.loginfo_throttle(
                    0.5,
                    f"WP(in-radius)={wp_display} / tgt={tgt_idx+1}/{len(spaced_x)} | "
                    f"v={speed_mps:.2f} m/s | Ld={Ld:.2f} | steer={latest_filtered_angle:+.2f} deg | RTK={rtk_status_txt}"
                )

            if updated:
                fig.canvas.draw_idle()
            plt.pause(0.001)
            rate.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        _on_shutdown()

if __name__ == '__main__':
    main()
