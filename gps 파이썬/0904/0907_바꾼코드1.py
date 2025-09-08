#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import time
from collections import deque
from queue import Queue

import rospy
import rospkg
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32

import numpy as np
import pandas as pd
import geopy.distance

import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ───────────────── 설정(기본값) ─────────────────
WAYPOINT_SPACING = 2.5          # m (간격 리샘플)
TARGET_RADIUS_END = 2.0         # 최종 도달 반경
MAX_STEER_DEG = 20.0            # 조향 제한 [deg]
SIGN_CONVENTION = -1.0          # 하드웨어 부호 관례

# Lookahead
LOOKAHEAD_MIN = 3.2             # m
LOOKAHEAD_MAX = 4.0             # m
LOOKAHEAD_K   = 0.2             # m per (m/s)

# 각도 저역필터 컷오프
LPF_FC_HZ = 0.8                 # Hz

# 속도/스파이크 처리
SPEED_BUF_LEN = 10
MAX_JITTER_SPEED = 4.0          # m/s
MIN_MOVE_FOR_HEADING = 0.05     # m

# 시각화
ANNOTATE_WAYPOINT_INDEX = True
DRAW_WAYPOINT_CIRCLES = True
AXIS_MARGIN_DEFAULT = 5.0

# ─────────────── 패키지 경로/기본 CSV/로그 ───────────────
def _default_paths():
    """topics 코드 방식: 패키지 기준 기본 경로 계산"""
    try:
        pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
    cfg = os.path.join(pkg_path, 'config')
    wp  = os.path.join(cfg, 'left_lane.csv')  # 필요시 rosparam ~waypoint_csv로 교체
    log = os.path.join(cfg, f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    return cfg, wp, log

CFG_DIR_DEFAULT, WAYPOINT_CSV_DEFAULT, LOG_CSV_DEFAULT = _default_paths()

# ─────────────── 전역 상태 (ROS/플롯/데이터) ───────────────
gps_queue = Queue()
steering_pub = None
latest_filtered_angle = 0.0

# 위치/속도/헤딩
pos_buf = deque(maxlen=20)        # (t, x, y)
speed_buf = deque(maxlen=SPEED_BUF_LEN)
speed_ema = 0.0
speed_alpha = 0.4
last_heading_vec = None  # (dx, dy)

# 웨이포인트(전역: 콜백/그리기에서 사용)
spaced_x, spaced_y = [], []

# 로깅 주기
_last_log_t = 0.0

# ─────────────── 유틸 ───────────────
def latlon_to_xy(ref_lat, ref_lon, lat, lon):
    northing = geopy.distance.geodesic((ref_lat, ref_lon), (lat, ref_lon)).meters
    easting  = geopy.distance.geodesic((ref_lat, ref_lon), (ref_lat, lon)).meters
    if lat < ref_lat: northing *= -1
    if lon < ref_lon: easting  *= -1
    return float(easting), float(northing)

def euclidean_dist(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))

def generate_waypoints_along_path(path, spacing=3.0):
    """선분 따라 spacing 간격으로 리샘플"""
    if len(path) < 2:
        return path
    new_points = [tuple(map(float, path[0]))]
    dist_accum = 0.0
    last_point = np.array(path[0], dtype=float)
    for i in range(1, len(path)):
        current_point = np.array(path[i], dtype=float)
        segment = current_point - last_point
        seg_len = float(np.linalg.norm(segment))
        while dist_accum + seg_len >= spacing:
            remain = spacing - dist_accum
            direction = segment / seg_len
            new_point = last_point + direction * remain
            new_points.append((float(new_point[0]), float(new_point[1])))
            last_point = new_point
            segment = current_point - last_point
            seg_len = float(np.linalg.norm(segment))
            dist_accum = 0.0
        dist_accum += seg_len
        last_point = current_point
    if euclidean_dist(new_points[-1], path[-1]) > 1e-6:
        new_points.append((float(path[-1][0]), float(path[-1][1])))
    return new_points

def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float); v2 = np.array(v2, dtype=float)
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    dot = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    ang = math.degrees(math.acos(dot))
    if v1[0]*v2[1] - v1[1]*v2[0] < 0:
        ang = -ang
    return float(ang)

class AngleLPF:
    def __init__(self, fc_hz=3.0, init_deg=0.0):
        self.fc = float(fc_hz)
        self.y = float(init_deg)
        self.t_last = None
    def reset(self, value_deg=0.0, t_sec=None):
        self.y = float(value_deg); self.t_last = t_sec
    def update(self, target_deg, t_sec):
        if self.t_last is None:
            self.t_last = t_sec; self.y = float(target_deg); return self.y
        dt = max(1e-3, float(t_sec) - float(self.t_last))
        tau = 1.0 / (2.0 * math.pi * self.fc)
        alpha = dt / (tau + dt)
        err = wrap_deg(float(target_deg) - self.y)
        self.y = wrap_deg(self.y + alpha * err)
        self.t_last = t_sec
        return self.y

# ─────────────── Pure pursuit 보조 ───────────────
def find_nearest_index(x, y, xs, ys, start_idx, window_ahead=60, window_back=10):
    n = len(xs)
    i0, i1 = max(0, start_idx - window_back), min(n - 1, start_idx + window_ahead)
    sub_x, sub_y = np.array(xs[i0:i1+1]), np.array(ys[i0:i1+1])
    d2 = (sub_x - x)**2 + (sub_y - y)**2
    return i0 + int(np.argmin(d2))

def target_index_from_lookahead(nearest_idx, Ld, spacing, n):
    steps = max(1, int(math.ceil(Ld / max(1e-6, spacing))))
    return min(n - 1, nearest_idx + steps)

# ─────────────── ROS 콜백/퍼블리셔 ───────────────
def gps_callback(data: NavSatFix):
    if hasattr(data, "status") and getattr(data.status, "status", 0) < 0:
        return
    lat, lon = float(data.latitude), float(data.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return
    stamp = data.header.stamp.to_sec() if (hasattr(data, "header") and data.header and data.header.stamp) \
            else rospy.Time.now().to_sec()
    gps_queue.put((lat, lon, stamp))

def publish_steering(event):
    if steering_pub is not None:
        steering_pub.publish(Float32(float(latest_filtered_angle)))

def _on_shutdown():
    """노드 종료 시 조향 0° 퍼블리시(안전)"""
    try:
        if steering_pub is not None:
            steering_pub.publish(Float32(0.0))
            rospy.sleep(0.05)
    except Exception:
        pass

# ─────────────── 메인 ───────────────
def main():
    global steering_pub, latest_filtered_angle
    global spaced_x, spaced_y, last_heading_vec, speed_ema, _last_log_t

    rospy.init_node('waypoint_tracker_node', anonymous=False)

    # ----- 파라미터(패키지 기본 + rosparam) -----
    cfg_dir, csv_default, log_default = _default_paths()
    waypoint_csv = rospy.get_param('~waypoint_csv', csv_default)
    log_csv_path = rospy.get_param('~log_csv', log_default)
    axis_margin  = float(rospy.get_param('~axis_margin', AXIS_MARGIN_DEFAULT))

    # CSV 적재(위치 ref 세팅 → XY 변환 → 간격 리샘플)
    if not os.path.exists(waypoint_csv):
        rospy.logerr(f"[waypoint_tracker_node] waypoint CSV not found: {waypoint_csv}")
        return
    df = pd.read_csv(waypoint_csv)
    if 'Lat' not in df.columns or 'Lon' not in df.columns or len(df) < 2:
        rospy.logerr("[waypoint_tracker_node] CSV must have columns: Lat, Lon (>=2 rows)")
        return
    ref_lat = float(df['Lat'].iloc[0]); ref_lon = float(df['Lon'].iloc[0])
    csv_coords = [latlon_to_xy(ref_lat, ref_lon, float(r['Lat']), float(r['Lon'])) for _, r in df.iterrows()]
    csv_x, csv_y = zip(*csv_coords)
    original_path = list(zip(csv_x, csv_y))
    spaced_waypoints = generate_waypoints_along_path(original_path, spacing=WAYPOINT_SPACING)
    spaced_x, spaced_y = list(zip(*spaced_waypoints))

    # 퍼블리셔/타이머/종료훅
    rospy.on_shutdown(_on_shutdown)  # 종료시 0° 퍼블리시(예: Timer 사용 예시와 함께 자주 쓰임). :contentReference[oaicite:2]{index=2}
    steering_pub = rospy.Publisher("/filtered_steering_angle", Float32, queue_size=10)
    rospy.Subscriber("/gps1/fix", NavSatFix, gps_callback)
    rospy.Timer(rospy.Duration(0.05), publish_steering)  # 20Hz 송출(타이머는 주기 콜백 실행). :contentReference[oaicite:3]{index=3}

    # 시각화 준비(topics 스타일)
    plt.ion()
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(csv_x, csv_y, 'g-', label='CSV Path')
    ax.plot(spaced_x, spaced_y, 'b.-', markersize=3, label=f'{WAYPOINT_SPACING:.0f}m Waypoints')
    live_line, = ax.plot([], [], 'r-', linewidth=1, label='Route')
    current_scatter, = ax.plot([], [], 'ro', label='Current')
    target_line, = ax.plot([], [], 'c--', linewidth=1, label='Target Line')
    ax.axis('equal'); ax.grid(True, ls=':', alpha=0.5); ax.legend(loc='upper right')

    minx, maxx = min(min(csv_x), min(spaced_x))-axis_margin, max(max(csv_x), max(spaced_x))+axis_margin
    miny, maxy = min(min(csv_y), min(spaced_y))-axis_margin, max(max(csv_y), max(spaced_y))+axis_margin
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

    if ANNOTATE_WAYPOINT_INDEX or DRAW_WAYPOINT_CIRCLES:
        for idx, (x, y) in enumerate(zip(spaced_x, spaced_y), 1):
            if ANNOTATE_WAYPOINT_INDEX:
                ax.text(x, y, str(idx), fontsize=7, ha='center', va='center', color='black')
            if DRAW_WAYPOINT_CIRCLES:
                ax.add_patch(Circle((x, y), TARGET_RADIUS_END, fill=False,
                                    linestyle='--', edgecolor='tab:blue', alpha=0.25))

    # 로그 준비(topics 스타일)
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'current_x','current_y','prev_x','prev_y',
                'waypoint_idx','waypoint_x','waypoint_y',
                'steer_deg','heading_deg',
                'speed_cmd','speed_meas_ema',
                'dist_to_target','time','rtk_status','speed_mode','flags'
            ])

    # Pure pursuit 실행 상태
    prev_Ld = LOOKAHEAD_MIN
    nearest_idx_prev = 0
    lpf = AngleLPF(fc_hz=LPF_FC_HZ)

    try:
        while not rospy.is_shutdown():
            updated = False
            while not gps_queue.empty():
                lat, lon, tsec = gps_queue.get()
                # XY 변환 (동일 ref)
                x, y = latlon_to_xy(ref_lat, ref_lon, lat, lon)
                updated = True

                # 속도 추정
                if pos_buf:
                    t_prev, x_prev, y_prev = pos_buf[-1]
                    dt = max(1e-3, tsec - t_prev)
                    d  = math.hypot(x - x_prev, y - y_prev)
                    inst_v = d / dt
                    if inst_v > MAX_JITTER_SPEED:
                        continue  # 점프 제거
                    speed_buf.append(inst_v)
                    speed_ema = (1 - speed_alpha) * speed_ema + speed_alpha * inst_v
                pos_buf.append((tsec, x, y))

                # 경로 인덱스(근접점→lookahead)
                nearest_idx = find_nearest_index(x, y, spaced_x, spaced_y, nearest_idx_prev, 80, 15)
                nearest_idx_prev = nearest_idx
                speed_mps = float(np.median(speed_buf)) if speed_buf else 0.0
                Ld_target = max(LOOKAHEAD_MIN, min(LOOKAHEAD_MAX, LOOKAHEAD_MIN + LOOKAHEAD_K * speed_mps))
                Ld = prev_Ld + 0.2 * (Ld_target - prev_Ld); prev_Ld = Ld

                tgt_idx = target_index_from_lookahead(nearest_idx, Ld, WAYPOINT_SPACING, len(spaced_x))
                tx, ty = spaced_x[tgt_idx], spaced_y[tgt_idx]
                target_line.set_data([x, tx], [y, ty])

                # 헤딩 추정(최근 이동 벡터)
                heading_vec = None
                for k in range(2, min(len(pos_buf), 5)+1):
                    t0, x0, y0 = pos_buf[-k]
                    if math.hypot(x - x0, y - y0) >= MIN_MOVE_FOR_HEADING:
                        heading_vec = (x - x0, y - y0); break
                if heading_vec is not None:
                    last_heading_vec = heading_vec
                elif last_heading_vec is None:
                    continue

                # 조향 계산 + 동적 LPF(부호관례/제한)
                target_vec = (tx - x, ty - y)
                raw_angle = angle_between(last_heading_vec, target_vec)
                lpf.fc = min(2.0, LPF_FC_HZ + 0.5) if abs(raw_angle) > 10 else LPF_FC_HZ
                filt_angle = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, lpf.update(raw_angle, tsec)))
                latest_filtered_angle = SIGN_CONVENTION * filt_angle

                # 시각화 갱신
                live_line.set_data([p[1] for p in pos_buf], [p[2] for p in pos_buf])
                current_scatter.set_data([x], [y])

                # 로깅(topics 형식 맞춤: 일부 필드는 공란)
                dist_to_target = euclidean_dist((x, y), (tx, ty))
                heading_deg = 0.0
                if last_heading_vec is not None:
                    heading_deg = math.degrees(math.atan2(last_heading_vec[1], last_heading_vec[0]))

                # 로그 쓰기
                try:
                    with open(log_csv_path, 'a', newline='') as f:
                        w = csv.writer(f)
                        px, py = (pos_buf[-2][1], pos_buf[-2][2]) if len(pos_buf) > 1 else ('','')
                        w.writerow([
                            x, y, px, py,
                            tgt_idx+1, tx, ty,
                            latest_filtered_angle, heading_deg,
                            '', speed_ema,         # speed_cmd 없음 → 공란, 측정 EMA 기록
                            dist_to_target, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                            '', '', ''             # rtk_status, speed_mode, flags 미사용
                        ])
                except Exception as e:
                    rospy.logwarn(f"[waypoint_tracker_node] log write failed: {e}")

                # 콘솔 요약(0.5s 간격)
                now = time.time()
                if now - _last_log_t > 0.5:
                    rospy.loginfo(
                        f"v(EMA)={speed_ema:.2f} m/s, "
                        f"Steer={latest_filtered_angle:+.2f} deg, "
                        f"Heading={heading_deg:.2f} deg, "
                        f"Dist→Target={dist_to_target:.2f} m, "
                        f"Idx={nearest_idx}->{tgt_idx+1}"
                    )
                    _last_log_t = now

                # 도착 판정
                if tgt_idx >= len(spaced_x) - 1 and dist_to_target <= TARGET_RADIUS_END:
                    rospy.loginfo("✅ All waypoints reached.")
                    rospy.signal_shutdown("mission complete")

            if updated:
                # Info 박스(topics 느낌)
                ax.collections = [c for c in ax.collections if not isinstance(c, matplotlib.collections.PathCollection)]
                info_lines = []
                if pos_buf:
                    x, y = pos_buf[-1][1], pos_buf[-1][2]
                    info_lines.append(f"Veh: ({x:.1f}, {y:.1f}) m")
                if last_heading_vec is not None:
                    info_lines.append(f"Heading: {math.degrees(math.atan2(last_heading_vec[1], last_heading_vec[0])):.1f}°")
                info_lines.append(f"Steering: {latest_filtered_angle:+.1f}°")
                ax.text(0.02, 0.98, "\n".join(info_lines), transform=ax.transAxes,
                        ha='left', va='top', fontsize=9, bbox=dict(fc='white', alpha=0.7))
                fig.canvas.draw_idle()
            plt.pause(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        # 안전 정지 시도
        _on_shutdown()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
