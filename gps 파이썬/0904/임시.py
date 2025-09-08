#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
from queue import Queue
from collections import deque
import geopy.distance
import pandas as pd
import numpy as np
import csv
import math
import time

# ── 경로/토픽 (우리 환경) ─────────────────────────────
try:
    import rospkg
    _pkg = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    CSV_PATH = f"{_pkg}/config/left_lane.csv"   # ← 우리 CSV
except Exception:
    # rospkg 없거나 패키지 못 찾으면 기존 경로로 폴백
    CSV_PATH = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/left_lane.csv"

GPS_FIX_TOPIC = "/gps1/fix"              # ← 우리 구독 토픽
STEER_CMD_TOPIC = "/vehicle/steer_cmd"   # ← 우리 퍼블리시 토픽(단위: deg)

# ── 설정 ─────────────────────────────────────────────
WAYPOINT_SPACING = 2.5          # m
TARGET_RADIUS_END = 2.0         # 최종 종착지 도달 반경
MAX_STEER_DEG = 30.0            # 조향 제한 (deg)
SIGN_CONVENTION = -1.0          # 차체/액추에이터 부호 관례

# Lookahead
LOOKAHEAD_MIN = 3.2             # m
LOOKAHEAD_MAX = 4.0             # m
LOOKAHEAD_K   = 0.2             # m per (m/s)

# 각도 저역필터 컷오프
LPF_FC_HZ = 0.8                 # Hz

# 속도/스파이크 처리
SPEED_BUF_LEN = 10
MAX_JITTER_SPEED = 4.0          # m/s

# 플롯 옵션
ANNOTATE_WAYPOINT_INDEX = True
DRAW_WAYPOINT_CIRCLES = True

# ── 데이터 적재 및 좌표 유틸 ─────────────────────────
df = pd.read_csv(CSV_PATH)
ref_lat = df['Lat'][0]
ref_lon = df['Lon'][0]

def latlon_to_xy(lat, lon):
    northing = geopy.distance.geodesic((ref_lat, ref_lon), (lat, ref_lon)).meters
    easting  = geopy.distance.geodesic((ref_lat, ref_lon), (ref_lat, lon)).meters
    if lat < ref_lat: northing *= -1
    if lon < ref_lon: easting  *= -1
    return easting, northing

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

# CSV -> XY
csv_coords = [latlon_to_xy(row['Lat'], row['Lon']) for _, row in df.iterrows()]
csv_x, csv_y = zip(*csv_coords)
original_path = list(zip(csv_x, csv_y))
spaced_waypoints = generate_waypoints_along_path(original_path, spacing=WAYPOINT_SPACING)
spaced_x, spaced_y = zip(*spaced_waypoints)

# ── 각도/필터 유틸 ────────────────────────────────────
def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    dot = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    ang = math.degrees(math.acos(dot))
    if v1[0]*v2[1] - v1[1]*v2[0] < 0:
        ang = -ang
    return ang

class AngleLPF:
    def __init__(self, fc_hz=3.0, init_deg=0.0):
        self.fc = fc_hz
        self.y = init_deg
        self.t_last = None

    def reset(self, value_deg=0.0, t_sec=None):
        self.y = value_deg
        self.t_last = t_sec

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

# ── ROS 변수 ─────────────────────────────────────────
gps_queue = Queue()
steering_pub = None
latest_filtered_angle = 0.0
pos_buf = deque(maxlen=20)
speed_buf = deque(maxlen=SPEED_BUF_LEN)
MIN_MOVE_FOR_HEADING = 0.05

# Pure pursuit 보조
def find_nearest_index(x, y, xs, ys, start_idx, window_ahead=60, window_back=10):
    n = len(xs)
    i0, i1 = max(0, start_idx - window_back), min(n - 1, start_idx + window_ahead)
    sub_x, sub_y = np.array(xs[i0:i1+1]), np.array(ys[i0:i1+1])
    d2 = (sub_x - x)**2 + (sub_y - y)**2
    return i0 + int(np.argmin(d2))

def target_index_from_lookahead(nearest_idx, Ld, spacing, n):
    steps = max(1, int(math.ceil(Ld / max(1e-6, spacing))))
    return min(n - 1, nearest_idx + steps)

# 콜백
def gps_callback(data: NavSatFix):
    if hasattr(data, "status") and getattr(data.status, "status", 0) < 0:
        return
    lat, lon = float(data.latitude), float(data.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return
    stamp = data.header.stamp.to_sec() if data.header and data.header.stamp else rospy.Time.now().to_sec()
    gps_queue.put((lat, lon, stamp))

def publish_steering(event):
    global latest_filtered_angle
    if steering_pub is not None:
        steering_pub.publish(Float32(latest_filtered_angle))

# ── 메인 ──────────────────────────────────────────────
def main():
    global steering_pub, latest_filtered_angle
    prev_Ld = LOOKAHEAD_MIN

    rospy.init_node('gps_listener', anonymous=True)
    rospy.Subscriber(GPS_FIX_TOPIC, NavSatFix, gps_callback)
    steering_pub = rospy.Publisher(STEER_CMD_TOPIC, Float32, queue_size=10)
    rospy.Timer(rospy.Duration(0.05), publish_steering)

    # 플롯
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(csv_x, csv_y, 'g-', label='CSV Path')
    ax.plot(spaced_x, spaced_y, 'b.-', markersize=3, label=f'{WAYPOINT_SPACING:.0f}m Waypoints')
    live_line, = ax.plot([], [], 'r-', linewidth=1, label='Live GPS')
    current_scatter, = ax.plot([], [], 'ro', label='Current')
    target_line, = ax.plot([], [], 'g--', linewidth=1, label='Target Line')
    ax.axis('equal'); ax.grid(True); ax.legend()

    minx, maxx = min(min(csv_x), min(spaced_x))-10, max(max(csv_x), max(spaced_x))+10
    miny, maxy = min(min(csv_y), min(spaced_y))-10, max(max(csv_y), max(spaced_y))+10
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

    if ANNOTATE_WAYPOINT_INDEX or DRAW_WAYPOINT_CIRCLES:
        for idx, (x, y) in enumerate(zip(spaced_x, spaced_y), 1):
            if ANNOTATE_WAYPOINT_INDEX:
                ax.text(x, y, str(idx), fontsize=8, ha='center', va='center', color='black')
            if DRAW_WAYPOINT_CIRCLES:
                ax.add_patch(plt.Circle((x, y), TARGET_RADIUS_END, color='blue', fill=False, linestyle='--', alpha=0.5))

    with open('/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/left_lane.csv', 'w', newline='') as f:
        csv.writer(f).writerows(spaced_waypoints)

    nearest_idx_prev = 0
    lpf = AngleLPF(fc_hz=LPF_FC_HZ)
    last_heading_vec = None

    try:
        while not rospy.is_shutdown():
            updated = False
            while not gps_queue.empty():
                lat, lon, tsec = gps_queue.get()
                x, y = latlon_to_xy(lat, lon)
                updated = True

                # 속도 처리
                if pos_buf:
                    t_prev, x_prev, y_prev = pos_buf[-1]
                    dt = max(1e-3, tsec - t_prev)
                    d = math.hypot(x - x_prev, y - y_prev)
                    inst_v = d / dt
                    if inst_v > MAX_JITTER_SPEED:
                        continue
                    speed_buf.append(inst_v)
                pos_buf.append((tsec, x, y))

                # 경로 인덱스
                nearest_idx = find_nearest_index(x, y, spaced_x, spaced_y, nearest_idx_prev, 80, 15)
                nearest_idx_prev = nearest_idx

                # Lookahead 스무딩
                speed_mps = float(np.median(speed_buf)) if speed_buf else 0.0
                Ld_target = max(LOOKAHEAD_MIN, min(LOOKAHEAD_MAX, LOOKAHEAD_MIN + LOOKAHEAD_K * speed_mps))
                alpha_ld = 0.2
                Ld = prev_Ld + alpha_ld * (Ld_target - prev_Ld)
                prev_Ld = Ld

                tgt_idx = target_index_from_lookahead(nearest_idx, Ld, WAYPOINT_SPACING, len(spaced_x))
                tx, ty = spaced_x[tgt_idx], spaced_y[tgt_idx]
                target_line.set_data([x, tx], [y, ty])

                # 헤딩
                heading_vec = None
                for k in range(2, min(len(pos_buf), 5)+1):
                    t0, x0, y0 = pos_buf[-k]
                    if math.hypot(x - x0, y - y0) >= MIN_MOVE_FOR_HEADING:
                        heading_vec = (x - x0, y - y0)
                        break
                if heading_vec is not None:
                    last_heading_vec = heading_vec
                elif last_heading_vec is None:
                    continue

                # 조향각 계산 + 동적 LPF
                target_vec = (tx - x, ty - y)
                raw_angle = angle_between(last_heading_vec, target_vec)
                base_fc = LPF_FC_HZ
                lpf.fc = min(2.0, base_fc + 0.5) if abs(raw_angle) > 10 else base_fc
                filt_angle = max(-MAX_STEER_DEG, min(MAX_STEER_DEG, lpf.update(raw_angle, tsec)))

                latest_filtered_angle = SIGN_CONVENTION * filt_angle
                live_line.set_data([p[1] for p in pos_buf], [p[2] for p in pos_buf])
                current_scatter.set_data([x], [y])

                print(f"v={speed_mps:.1f} m/s | Ld={Ld:.2f} m | idx={nearest_idx}->{tgt_idx} | Raw={raw_angle:.2f}° | Filt={filt_angle:.2f}° | Pub={latest_filtered_angle:.2f}°")

                # 종료 조건(최종 반경 도달)
                if tgt_idx >= len(spaced_x) - 1 and euclidean_dist((x, y), (spaced_x[-1], spaced_y[-1])) <= TARGET_RADIUS_END:
                    print("✅ All waypoints reached.")
                    raise KeyboardInterrupt

            if updated:
                fig.canvas.draw_idle()
            plt.pause(0.01)

    except KeyboardInterrupt:
        print("종료됨.")
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
