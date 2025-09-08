#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, csv, math, time
from collections import deque

import rospy, rospkg
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

# ───────────────────────────────
# 기본 파라미터
# ───────────────────────────────
WAYPOINT_SPACING   = 2.5
TARGET_RADIUS_END  = 2.0
MAX_STEER_DEG      = 20.0
SIGN_CONVENTION    = -1.0

LOOKAHEAD_MIN      = 3.2
LOOKAHEAD_MAX      = 4.0
LOOKAHEAD_K        = 0.2

LPF_FC_HZ          = 0.8
SPEED_BUF_LEN      = 10
MAX_JITTER_SPEED   = 4.0
MIN_MOVE_FOR_HEADING = 0.05

FS_DEFAULT         = 20.0
GPS_TIMEOUT_SEC    = 1.0

# 속도 상수 (dSPACE 호환: 1,2,3,4,5,6 등 정수 코드)
SPEED_FORCE_CODE = 2
SPEED_CAP_CODE_DEFAULT = 6

# 퍼블리시 토픽
TOPIC_SPEED_CMD    = '/vehicle/speed_cmd'
TOPIC_STEER_CMD    = '/vehicle/steer_cmd'
TOPIC_RTK_STATUS   = '/rtk/status'
TOPIC_WP_INDEX     = '/tracker/wp_index'

# u-blox RTK 상태 메시지 (옵션)
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

# ───────────────────────────────
# 전역 상태
# ───────────────────────────────
gps_queue = Queue()
latest_filtered_angle = 0.0
pos_buf   = deque(maxlen=20)
speed_buf = deque(maxlen=SPEED_BUF_LEN)

pub_speed = pub_steer = pub_rtk = pub_wpidx = None
rtk_status_txt = "NONE"
last_fix_time  = 0.0
wp_index_active = 0
log_csv_path = None
_last_log_wall = 0.0

# 상태 텍스트 핸들
status_text = None
last_pub_v  = 0.0
last_meas_v = 0.0

# ───────────────────────────────
# 경로 기본
# ───────────────────────────────
def _default_paths():
    try:
        pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
    waypoint_csv = os.path.join(pkg_path, 'config', 'left_lane.csv')
    logs_dir     = os.path.join(pkg_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_csv      = os.path.join(logs_dir, f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    return waypoint_csv, log_csv

WAYPOINT_CSV_DEFAULT, LOG_CSV_DEFAULT = _default_paths()

# ───────────────────────────────
# 유틸
# ───────────────────────────────
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
    if len(path) < 2: return path
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
    v1, v2 = np.array(v1, float), np.array(v2, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    dot = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
    ang = math.degrees(math.acos(dot))
    if v1[0]*v2[1] - v1[1]*v2[0] < 0: ang = -ang
    return ang

class AngleLPF:
    def __init__(self, fc_hz=3.0, init_deg=0.0):
        self.fc, self.y, self.t_last = fc_hz, init_deg, None
    def update(self, target_deg, t_sec):
        if self.t_last is None:
            self.t_last, self.y = t_sec, target_deg
            return self.y
        dt = max(1e-3, t_sec - self.t_last)
        tau = 1/(2*math.pi*self.fc)
        alpha = dt/(tau+dt)
        err = wrap_deg(target_deg - self.y)
        self.y = wrap_deg(self.y + alpha*err)
        self.t_last = t_sec
        return self.y

def find_nearest_index(x, y, xs, ys, start_idx, window_ahead=60, window_back=10):
    n = len(xs)
    i0, i1 = max(0, start_idx-window_back), min(n-1, start_idx+window_ahead)
    sub_x, sub_y = np.array(xs[i0:i1+1]), np.array(ys[i0:i1+1])
    d2 = (sub_x-x)**2 + (sub_y-y)**2
    return i0 + int(np.argmin(d2))

def target_index_from_lookahead(nearest_idx, Ld, spacing, n):
    steps = max(1, int(math.ceil(Ld/max(1e-6, spacing))))
    return min(n-1, nearest_idx+steps)

# ───────────────────────────────
# ROS 콜백/퍼블리시
# ───────────────────────────────
def gps_callback(data: NavSatFix):
    global last_fix_time
    lat, lon = float(data.latitude), float(data.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)): return
    stamp = data.header.stamp.to_sec() if data.header and data.header.stamp else rospy.Time.now().to_sec()
    gps_queue.put((lat, lon, stamp))
    last_fix_time = time.time()

def _cb_relpos(msg):
    global rtk_status_txt
    try:
        carr_soln = int((int(msg.flags)>>3) & 0x3)
        rtk_status_txt = "FIX" if carr_soln==2 else ("FLOAT" if carr_soln==1 else "NONE")
    except Exception:
        rtk_status_txt = "NONE"

def publish_all(event, speed_code_default=1, one_based=True):
    global last_pub_v
    now = time.time()
    no_gps = (now-last_fix_time) > rospy.get_param('~gps_timeout_sec', GPS_TIMEOUT_SEC)
    if SPEED_FORCE_CODE is not None:
        code = int(SPEED_FORCE_CODE); cap=int(SPEED_CAP_CODE_DEFAULT)
    else:
        code = int(rospy.get_param('~speed_code', speed_code_default))
        cap  = int(rospy.get_param('~speed_cap_code', SPEED_CAP_CODE_DEFAULT))
    code = max(0, min(code, cap))
    v_out = 0.0 if no_gps else float(code)
    steer_out = 0.0 if no_gps else float(latest_filtered_angle)
    last_pub_v = v_out
    if pub_speed: pub_speed.publish(Float32(v_out))
    if pub_steer: pub_steer.publish(Float32(steer_out))
    if pub_rtk:   pub_rtk.publish(String(rtk_status_txt))
    if pub_wpidx:
        idx_pub = (wp_index_active+1) if one_based else wp_index_active
        pub_wpidx.publish(Int32(int(idx_pub)))

def _on_shutdown():
    try:
        if pub_speed: pub_speed.publish(Float32(0.0))
        if pub_steer: pub_steer.publish(Float32(0.0))
    except: pass

# ───────────────────────────────
# 메인
# ───────────────────────────────
def main():
    global pub_speed, pub_steer, pub_rtk, pub_wpidx
    global latest_filtered_angle, log_csv_path, wp_index_active
    global _last_log_wall, last_meas_v, status_text

    rospy.init_node('rtk_waypoint_tracker', anonymous=False)
    rospy.on_shutdown(_on_shutdown)

    waypoint_csv, log_csv_path = rospy.get_param('~waypoint_csv', WAYPOINT_CSV_DEFAULT), rospy.get_param('~log_csv', LOG_CSV_DEFAULT)
    fs = float(rospy.get_param('~fs', FS_DEFAULT))
    one_based = bool(rospy.get_param('~wp_index_one_based', True))

    ublox_ns = rospy.get_param('~ublox_ns','/gps1')
    fix_topic, relpos_topic = ublox_ns+'/fix', ublox_ns+'/navrelposned'

    pub_speed = rospy.Publisher(TOPIC_SPEED_CMD, Float32, queue_size=10)
    pub_steer = rospy.Publisher(TOPIC_STEER_CMD, Float32, queue_size=10)
    pub_rtk   = rospy.Publisher(TOPIC_RTK_STATUS, String, queue_size=10)
    pub_wpidx = rospy.Publisher(TOPIC_WP_INDEX, Int32, queue_size=10)

    rospy.Subscriber(fix_topic, NavSatFix, gps_callback, queue_size=100)
    if _HAVE_RELPOSNED: rospy.Subscriber(relpos_topic, NavRELPOSNED, _cb_relpos, queue_size=50)

    if not os.path.exists(waypoint_csv):
        rospy.logerr("[tracker] waypoint csv not found: %s", waypoint_csv); return

    df = pd.read_csv(waypoint_csv)
    ref_lat, ref_lon = float(df['Lat'][0]), float(df['Lon'][0])
    to_xy = latlon_to_xy_fn(ref_lat, ref_lon)
    csv_coords = [to_xy(row['Lat'], row['Lon']) for _,row in df.iterrows()]
    spaced_waypoints = generate_waypoints_along_path(csv_coords, spacing=WAYPOINT_SPACING)
    spaced_x, spaced_y = zip(*spaced_waypoints)

    plt.ion(); fig, ax = plt.subplots(figsize=(7.5,7.5))
    ax.plot([p[0] for p in csv_coords],[p[1] for p in csv_coords],'g-',label='CSV Path')
    ax.plot(spaced_x,spaced_y,'b.-',ms=3,label=f'{WAYPOINT_SPACING:.0f}m WPs')
    live_line,=ax.plot([],[],'r-',lw=1,label='Live GPS')
    current_pt,=ax.plot([],[],'ro',label='Current')
    target_line,=ax.plot([],[],'g--',lw=1,label='Target Line')
    ax.axis('equal'); ax.grid(True); ax.legend()
    ax.set_xlim(min(min([p[0] for p in csv_coords]),min(spaced_x))-10,
                max(max([p[0] for p in csv_coords]),max(spaced_x))+10)
    ax.set_ylim(min(min([p[1] for p in csv_coords]),min(spaced_y))-10,
                max(max([p[1] for p in csv_coords]),max(spaced_y))+10)

    status_text = fig.text(0.02,0.02,"Init...",fontsize=10,ha='left')

    lpf=AngleLPF(fc_hz=LPF_FC_HZ); prev_Ld=LOOKAHEAD_MIN; nearest_idx_prev=0; last_heading_vec=None
    rospy.Timer(rospy.Duration(1.0/max(1.0,fs)),lambda e: publish_all(e,one_based=one_based))
    rate=rospy.Rate(fs)

    try:
        while not rospy.is_shutdown():
            updated=False
            while not gps_queue.empty():
                lat,lon,tsec=gps_queue.get(); x,y=to_xy(lat,lon); updated=True
                if pos_buf:
                    t_prev,x_prev,y_prev=pos_buf[-1]
                    dt=max(1e-3,tsec-t_prev); d=math.hypot(x-x_prev,y-y_prev)
                    inst_v=d/dt
                    if inst_v>MAX_JITTER_SPEED: continue
                    speed_buf.append(inst_v)
                pos_buf.append((tsec,x,y))
                last_meas_v=float(np.median(speed_buf)) if speed_buf else 0.0

                nearest_idx=find_nearest_index(x,y,spaced_x,spaced_y,nearest_idx_prev,80,15)
                nearest_idx_prev=nearest_idx
                Ld_target=max(LOOKAHEAD_MIN,min(LOOKAHEAD_MAX,LOOKAHEAD_MIN+LOOKAHEAD_K*last_meas_v))
                Ld=prev_Ld+0.2*(Ld_target-prev_Ld); prev_Ld=Ld

                tgt_idx=target_index_from_lookahead(nearest_idx,Ld,WAYPOINT_SPACING,len(spaced_x))
                tx,ty=spaced_x[tgt_idx],spaced_y[tgt_idx]; target_line.set_data([x,tx],[y,ty])
                wp_index_active=int(tgt_idx)

                heading_vec=None
                for k in range(2,min(len(pos_buf),5)+1):
                    t0,x0,y0=pos_buf[-k]
                    if math.hypot(x-x0,y-y0)>=MIN_MOVE_FOR_HEADING: heading_vec=(x-x0,y-y0); break
                if heading_vec is not None: last_heading_vec=heading_vec
                elif last_heading_vec is None: continue

                raw_angle=angle_between(last_heading_vec,(tx-x,ty-y))
                lpf.fc=min(2.0,LPF_FC_HZ+0.5) if abs(raw_angle)>10 else LPF_FC_HZ
                filt_angle=max(-MAX_STEER_DEG,min(MAX_STEER_DEG,lpf.update(raw_angle,tsec)))
                latest_filtered_angle=SIGN_CONVENTION*filt_angle

                live_line.set_data([p[1] for p in pos_buf],[p[2] for p in pos_buf])
                current_pt.set_data([x],[y])

                # 상태 텍스트 갱신
                status_text.set_text(
                    f"pub_v={last_pub_v:.1f} | meas_v={last_meas_v:.2f} m/s | "
                    f"steer={latest_filtered_angle:+.1f}° | WP={wp_index_active+1}/{len(spaced_x)} | RTK={rtk_status_txt}"
                )

                rospy.loginfo_throttle(0.5,
                    f"WP(in-radius)={0 if euclidean_dist((x,y),(spaced_x[wp_index_active],spaced_y[wp_index_active]))>TARGET_RADIUS_END else wp_index_active+1} "
                    f"/ tgt={wp_index_active+1}/{len(spaced_x)} | meas_v={last_meas_v:.2f} m/s | pub_v={last_pub_v:.1f} | "
                    f"Ld={Ld:.2f} | steer={latest_filtered_angle:+.2f} deg | RTK={rtk_status_txt}"
                )

            if updated: fig.canvas.draw_idle()
            plt.pause(0.001); rate.sleep()
    finally: _on_shutdown()

if __name__=='__main__': main()
