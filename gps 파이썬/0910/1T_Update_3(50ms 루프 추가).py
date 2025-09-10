#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 + matplotlib · GPS waypoint 추종 & 조향각 퍼블리시 (Timer 기반 50ms 루프)
- /vehicle/steer_cmd, /vehicle/speed_cmd, /rtk/status, /current_waypoint_index 퍼블리시
- matplotlib 창이 검게만 뜨는 문제: facecolor 강제, 전역 ax 사용, GUI 펌프 타이머 추가로 해결
"""

import os, math, time, csv
import rospy, rospkg
import numpy as np
import pandas as pd

# ── matplotlib 백엔드 & 스타일(흰 배경 강제) ─────────────────────────
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

matplotlib.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.facecolor':'white',
    'axes.edgecolor':   'black',
    'text.color':       'black',
    'axes.labelcolor':  'black',
    'xtick.color':      'black',
    'ytick.color':      'black',
})

from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, Int32, String
from queue import Queue

# ====== 상수 ======
EARTH_RADIUS    = 6378137.0
MIN_DISTANCE    = 2.8
TARGET_RADIUS   = 2.8
ANGLE_LIMIT_DEG = 27.0
FIXED_SPEED     = 2.0          # 고정 퍼블리시 속도[m/s]

# ====== 전역 상태 ======
filtered_steering_angle = 0.0
current_x, current_y, current_t = [], [], []
current_waypoint_index = 0

DT, TAU, ALPHA = 0.05, 0.25, None

steering_pub = waypoint_index_pub = speed_pub = rtk_pub = None
log_csv_path = None; _last_log_wall = 0.0
rtk_status_txt = "NONE"

# 플롯 객체(전역 축 사용)
fig = None
ax  = None
ani = None

# ====== 경로/파일 경로 ======
def _default_paths():
    try:
        pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
    waypoint_csv = os.path.join(pkg_path, 'config', 'raw_track_latlon_6.csv')
    logs_dir     = os.path.join(pkg_path, 'logs'); os.makedirs(logs_dir, exist_ok=True)
    log_csv      = os.path.join(logs_dir, time.strftime('waypoint_log_%Y%m%d_%H%M%S.csv', time.localtime()))
    return waypoint_csv, log_csv

WAYPOINT_CSV_DEFAULT, LOG_CSV_DEFAULT = _default_paths()

# ====== 유틸 ======
def lat_lon_to_meters(lat, lon):
    x = EARTH_RADIUS * lon * math.pi / 180.0
    y = EARTH_RADIUS * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def distance_in_meters(x1, y1, x2, y2): return math.hypot(x2 - x1, y2 - y1)

def _pick_column(cols, cands):
    for c in cands:
        if c in cols: return c
    return None

def build_reduced_waypoints(csv_path):
    csv_path = os.path.expanduser(csv_path)
    df = pd.read_csv(csv_path, encoding='utf-8-sig', engine='python', sep=None)
    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = set(df.columns)
    lat_col  = _pick_column(cols, ['latitude','lat'])
    lon_col  = _pick_column(cols, ['longitude','lon'])
    east_col = _pick_column(cols, ['east','easting','x'])
    north_col= _pick_column(cols, ['north','northing','y'])

    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors='coerce').to_numpy()
        lon = pd.to_numeric(df[lon_col], errors='coerce').to_numpy()
        wx, wy = zip(*[lat_lon_to_meters(a,b) for a,b in zip(lat,lon)])
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

def nearest_in_radius_index(x, y, xs, ys, r):
    xs = np.asarray(xs); ys = np.asarray(ys)
    d2 = (xs - x)**2 + (ys - y)**2; i = int(np.argmin(d2))
    return i if math.hypot(xs[i]-x, ys[i]-y) <= r else -1

def lpf_and_invert(cur_deg):
    global filtered_steering_angle, ALPHA
    filtered_steering_angle = (1.0 - ALPHA)*filtered_steering_angle + ALPHA*cur_deg
    return -filtered_steering_angle

# ====== RTK 상태 (옵션) ======
_HAVE_RELPOSNED = False
try:
    from ublox_msgs.msg import NavRELPOSNED9 as NavRELPOSNED; _HAVE_RELPOSNED = True
except Exception:
    try:
        from ublox_msgs.msg import NavRELPOSNED; _HAVE_RELPOSNED = True
    except Exception:
        _HAVE_RELPOSNED = False

def _cb_relpos(msg):
    global rtk_status_txt
    try:
        carr_soln = int((int(msg.flags) >> 3) & 0x3)
        rtk_status_txt = "FIXED" if carr_soln==2 else ("FLOAT" if carr_soln==1 else "NONE")
    except Exception:
        rtk_status_txt = "NONE"

# ====== GPS 콜백 → 큐 적재 ======
gps_queue = Queue(); last_fix_time = 0.0
def gps_callback(msg: NavSatFix):
    global last_fix_time
    if hasattr(msg,"status") and getattr(msg.status,"status",0) < 0: return
    lat, lon = float(msg.latitude), float(msg.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)): return
    t = msg.header.stamp.to_sec() if msg.header and msg.header.stamp else rospy.Time.now().to_sec()
    gps_queue.put((lat, lon, t)); last_fix_time = time.time()

# ====== 보조 ======
reduced_waypoints = None

def _wp_in_radius_1based(x,y):
    if reduced_waypoints is None: return 0
    idx0 = nearest_in_radius_index(x, y, reduced_waypoints[0], reduced_waypoints[1], TARGET_RADIUS)
    return (idx0+1) if idx0>=0 else 0

# ====== 주기 처리(50ms) ======
def process_loop(_event):
    global current_waypoint_index, _last_log_wall
    updated = False; MAX_PER = 200
    cnt = 0
    while not gps_queue.empty() and cnt < MAX_PER:
        lat, lon, tsec = gps_queue.get(); cnt += 1; updated = True
        x, y = lat_lon_to_meters(lat, lon)
        current_x.append(x); current_y.append(y); current_t.append(tsec)

        tx = reduced_waypoints[0][current_waypoint_index]
        ty = reduced_waypoints[1][current_waypoint_index]

        if distance_in_meters(x, y, tx, ty) < TARGET_RADIUS:
            if current_waypoint_index < reduced_waypoints.shape[1]-1:
                current_waypoint_index += 1
                if waypoint_index_pub: waypoint_index_pub.publish(Int32(current_waypoint_index))

        if len(current_x) >= 2:
            prev = np.array([current_x[-2], current_y[-2]])
            curr = np.array([current_x[-1], current_y[-1]])
            head_vec = curr - prev; tgt_vec = np.array([tx, ty]) - curr
            heading_deg = math.degrees(math.atan2(head_vec[1], head_vec[0]))
            dt_local = max(1e-3, current_t[-1]-current_t[-2])
            speed_mps = float(np.hypot(head_vec[0], head_vec[1]) / dt_local)

            ang = max(-ANGLE_LIMIT_DEG, min(ANGLE_LIMIT_DEG, signed_angle_deg(head_vec, tgt_vec)))
            steer_cmd = lpf_and_invert(ang)

            if steering_pub: steering_pub.publish(Float32(steer_cmd))
            if speed_pub:    speed_pub.publish(Float32(FIXED_SPEED))
            if rtk_pub:      rtk_pub.publish(String(rtk_status_txt))

            noww = time.time()
            if log_csv_path and (noww - _last_log_wall > 0.5):
                try:
                    new = not os.path.exists(log_csv_path)
                    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
                    with open(log_csv_path,'a',newline='') as f:
                        w = csv.writer(f)
                        if new:
                            w.writerow(['time','lat','lon','speed_mps','heading_deg','steer_deg',
                                        'wp_index(1based)','target_index(1based)','rtk','pub_speed'])
                        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                    f"{lat:.7f}", f"{lon:.7f}",
                                    f"{speed_mps:.2f}", f"{heading_deg:.1f}",
                                    f"{steer_cmd:.2f}",
                                    _wp_in_radius_1based(x,y),
                                    current_waypoint_index+1,
                                    rtk_status_txt,
                                    f"{FIXED_SPEED:.2f}"])
                    _last_log_wall = noww
                except Exception as e:
                    rospy.logwarn(f"[tracker] log write failed: {e}")

        rospy.loginfo_throttle(
            0.5,
            f"TGT={current_waypoint_index+1} | steer={filtered_steering_angle:+.2f}° | RTK={rtk_status_txt} | PUB_SPEED={FIXED_SPEED:.2f}"
        )

    if updated and fig is not None:
        try: fig.canvas.draw_idle()
        except Exception: pass

# ====== 플롯 갱신 ======
def update_plot(_frame=None):
    global ax
    if ax is None or reduced_waypoints is None: return
    ax.cla()  # 기존 그려진 것 지우기
    ax.set_facecolor('white')

    # 웨이포인트 + 반경
    ax.scatter(reduced_waypoints[0], reduced_waypoints[1], s=10, marker='o', label='Waypoints', color='tab:blue')
    for i,(wx,wy) in enumerate(zip(reduced_waypoints[0], reduced_waypoints[1]), 1):
        ax.add_patch(Circle((wx,wy), TARGET_RADIUS, fill=False, linestyle='--', edgecolor='tab:blue', alpha=0.7))
        ax.text(wx, wy, str(i), fontsize=8, ha='center', va='center', color='black')

    # 현재 위치 & 타겟
    if current_x and current_y:
        cx, cy = current_x[-1], current_y[-1]
        ax.scatter(cx, cy, s=50, marker='x', label='Current', color='tab:red')

        tx = reduced_waypoints[0][current_waypoint_index]
        ty = reduced_waypoints[1][current_waypoint_index]
        ax.plot([tx],[ty], marker='*', markersize=12, linestyle='None', color='k', label='Target ★')
        ax.arrow(cx, cy, tx-cx, ty-cy, head_width=0.5, head_length=0.5, color='k')

        k = max(0, len(current_x)-200)
        for i in range(k+1, len(current_x)):
            ax.arrow(current_x[i-1], current_y[i-1],
                     current_x[i]-current_x[i-1], current_y[i]-current_y[i-1],
                     head_width=0.2, head_length=0.2, length_includes_head=True, color='tab:red', alpha=0.6)

        # 범례 텍스트
        speed_mps = 0.0; heading_deg = float('nan')
        if len(current_x)>=2 and len(current_t)>=2:
            dx = current_x[-1]-current_x[-2]; dy = current_y[-1]-current_y[-2]
            heading_deg = math.degrees(math.atan2(dy, dx))
            dt_local = max(1e-3, current_t[-1]-current_t[-2])
            speed_mps = math.hypot(dx,dy)/dt_local

        wp_in = _wp_in_radius_1based(cx, cy)
        h_speed  = plt.Line2D([], [], linestyle='None', label=f"Speed: {FIXED_SPEED:.2f} m/s")
        h_head   = plt.Line2D([], [], linestyle='None', label=f"Heading: {heading_deg:.1f}°")
        h_steer  = plt.Line2D([], [], linestyle='None', label=f"Steer: {filtered_steering_angle:+.1f}°")
        h_rtk    = plt.Line2D([], [], linestyle='None', label=f"RTK: {rtk_status_txt}")
        h_idx    = plt.Line2D([], [], linestyle='None', label=f"WP/TGT: {wp_in}/{current_waypoint_index+1}")
        ax.legend(handles=[h_speed,h_head,h_steer,h_rtk,h_idx], loc='best')

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.grid(True, color='#cccccc')
    ax.set_title('Filtered steering (inverted) · target starred; legend shows speed/heading/steer/RTK/WP·TGT')

# ====== 종료시 0 퍼블리시 ======
def _on_shutdown():
    rospy.loginfo("[tracker] shutdown: publishing zeros to speed and steer")
    try:
        if speed_pub:    speed_pub.publish(Float32(0.0))
        if steering_pub: steering_pub.publish(Float32(0.0))
    except Exception as e:
        rospy.logwarn(f"[tracker] shutdown publish failed: {e}")
    time.sleep(0.1)

# GUI 이벤트 펌프(ROS 타이머로 돌림)
def _pump_plt(_):
    try: plt.pause(0.001)
    except Exception: pass

# ====== 메인 ======
def main():
    global steering_pub, waypoint_index_pub, speed_pub, rtk_pub
    global reduced_waypoints, DT, TAU, ALPHA, log_csv_path
    global fig, ax, ani

    rospy.init_node('gps_waypoint_tracker', anonymous=True)
    rospy.on_shutdown(_on_shutdown)

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
    rospy.loginfo("Using FIXED_SPEED = %.2f m/s", FIXED_SPEED)

    steering_pub = rospy.Publisher('/vehicle/steer_cmd', Float32, queue_size=10)
    waypoint_index_pub = rospy.Publisher('/current_waypoint_index', Int32, queue_size=10)
    speed_pub = rospy.Publisher('/vehicle/speed_cmd', Float32, queue_size=10)
    rtk_pub   = rospy.Publisher('/rtk/status',      String,  queue_size=10)

    rospy.Subscriber(fix_topic, NavSatFix, gps_callback)
    if _HAVE_RELPOSNED:
        try:
            rospy.Subscriber(relpos_topic, NavRELPOSNED, _cb_relpos)
            rospy.loginfo("Subscribed RELPOSNED: %s (RTK ON)", relpos_topic)
        except Exception:
            rospy.logwarn("RELPOSNED subscribe failed; RTK disabled.")
    rospy.loginfo("Subscribed NavSatFix: %s", fix_topic)

    # 웨이포인트 로드
    global reduced_waypoints
    reduced_waypoints = build_reduced_waypoints(csv_path)
    rospy.loginfo("Waypoints loaded: %d (spacing >= %.1fm)", reduced_waypoints.shape[1], MIN_DISTANCE)

    # 플롯 준비
    if enable_plot:
        plt.ion()
        fig, ax = plt.subplots(num="Waypoint Tracker")
        fig.patch.set_facecolor('white'); ax.set_facecolor('white')
        ani = animation.FuncAnimation(fig, update_plot, interval=max(1, int(DT*1000)))
        plt.show(block=False)
        rospy.Timer(rospy.Duration(max(0.05, DT)), _pump_plt)

    # 처리 타이머(제어 50ms)
    rospy.Timer(rospy.Duration(DT), process_loop)
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
