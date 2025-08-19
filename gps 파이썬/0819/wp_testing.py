#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waypoints_live_gngga_viewer_seq_tx_enu_ros.py
• 기존 ENU 기반 시각화/제어 로직 유지
• 계산 결과를 ROS 토픽(/vehicle/speed_cmd, /vehicle/steer_cmd, /rtk/status)로 퍼블리시 추가
"""

import math
import time
import csv
import datetime
from collections import deque

import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# ───────── ROS 추가 ─────────
import rospy
from std_msgs.msg import Float32, String

# ─────────────────────────────────────────────────────────
# 설정값 (기본값)  — ROS 파라미터로 덮어쓰기 됩니다.
# ─────────────────────────────────────────────────────────
CONFIG = {
    'csv_path': r"C:\Users\dlwlr\OneDrive - 한라대학교\바탕 화면\GPS_purse_Python\bocsa\waypoints_2m_1.csv",
    'gnss_port': 'COM10',
    'ardu_port': 'COM7',
    'baud': 115200,
    'timeout': 0.0,
    'auto_mode': False,
    'wheelbase': 0.28,
    'look_dist0': 1.2,
    'k_ld': 0.3,
    'k_v': 0.5,
    'arrive_r': 0.5,
    'exit_r': 0.8,
    'tx_period': 0.2,
    'tx_on': True,
    'const_speed': 1.0,
    'route_maxlen': 10000,
    'log_csv': 'gps_wp_log 1.csv',
    'v_th': 0.1,
}
# 조향 제한(기존 코드 ±8 deg 유지; 30으로 바꾸시려면 아래를 30.0으로 조정)
STEER_LIMIT_DEG = 8.0

# WGS84
_a  = 6378137.0
_f  = 1 / 298.257223563
_e2 = _f * (2 - _f)

stop_flag = False

# ─────────────────────────────────────────────────────────
# NMEA / 좌표 변환
# ─────────────────────────────────────────────────────────
def ddmm2deg(val: float) -> float:
    d = int(val // 100)
    m = val - 100 * d
    return d + m / 60.0

def parse_gngga(line: str):
    """
    $GNGGA 의 (위도, 경도, FixQuality) 반환.
    반환: (lat, lon, fixq) 또는 None
    """
    parts = line.split(',')
    if not line.startswith('$GNGGA') or len(parts) < 7:
        return None
    try:
        lat = ddmm2deg(float(parts[2]))
        lon = ddmm2deg(float(parts[4]))
        if parts[3] == 'S': lat = -lat
        if parts[5] == 'W': lon = -lon
        fixq = int(parts[6])  # 0=invalid,1=GPS(SPS),2=DGPS,4=RTK_FIX,5=RTK_FLOAT 등
        return lat, lon, fixq
    except Exception:
        return None

def fix_quality_to_text(fixq: int) -> str:
    if fixq == 0: return "NONE"
    if fixq == 1: return "SPS"
    if fixq == 2: return "DGPS"
    if fixq == 4: return "RTK_FIX"
    if fixq == 5: return "RTK_FLOAT"
    return f"Q{fixq}"

def lla_to_ecef_xy(lat, lon, h=0.0):
    φ = math.radians(lat); λ = math.radians(lon)
    N = _a / math.sqrt(1 - _e2 * math.sin(φ)**2)
    x = (N + h) * math.cos(φ) * math.cos(λ)
    y = (N + h) * math.cos(φ) * math.sin(λ)
    return x, y

def ecef_xy_to_enu(x, y, lat0, lon0, h0=0.0):
    x0, y0 = lla_to_ecef_xy(lat0, lon0, h0)
    dx, dy = x - x0, y - y0
    φ0 = math.radians(lat0); λ0 = math.radians(lon0)
    t = np.array([
        [-math.sin(λ0),               math.cos(λ0)],
        [-math.sin(φ0)*math.cos(λ0), -math.sin(φ0)*math.sin(λ0)]
    ])
    enu = t.dot(np.array([dx, dy]))
    return enu[0], enu[1]

# ─────────────────────────────────────────────────────────
# 웨이포인트
# ─────────────────────────────────────────────────────────
def load_waypoints(path: str, interval: float = 2.0):
    df = pd.read_csv(path)
    lat0, lon0 = df.loc[0, ['Lat', 'Lon']]
    pts = []
    for _, row in df.iterrows():
        x, y = lla_to_ecef_xy(row['Lat'], row['Lon'], 0.0)
        e, n = ecef_xy_to_enu(x, y, lat0, lon0, 0.0)
        pts.append((e, n))
    arr = np.array(pts)
    df['x'], df['y'] = arr[:,0], arr[:,1]
    dist = np.hypot(df['x'].diff().fillna(0), df['y'].diff().fillna(0)).cumsum()
    wps = []
    for d in np.arange(0, dist.iloc[-1], interval):
        idx = dist.searchsorted(d)
        if idx == 0:
            wps.append((df.at[0,'x'], df.at[0,'y']))
        else:
            s0, s1 = dist.iloc[idx-1], dist.iloc[idx]
            t = (d - s0)/(s1 - s0)
            x = df.at[idx-1,'x'] + t*(df.at[idx,'x'] - df.at[idx-1,'x'])
            y = df.at[idx-1,'y'] + t*(df.at[idx,'y'] - df.at[idx-1,'y'])
            wps.append((x, y))
    wx, wy = np.array(wps).T
    return df, lat0, lon0, wx, wy

# ─────────────────────────────────────────────────────────
# 시리얼/플롯
# ─────────────────────────────────────────────────────────
def open_serial(port, baud, timeout):
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        print(f"[INFO] Opened {port}")
        return ser
    except Exception as e:
        print(f"[ERROR] {port}: {e}")
        return None

def init_plot(df, wx, wy, arrive_r):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(df['x'], df['y'], color='gray', lw=1, label='Path')
    for i,(x,y) in enumerate(zip(wx, wy), 1):
        ax.scatter(x, y, c='red', s=18, label='Waypoints' if i==1 else '')
        ax.text(x, y, str(i), fontsize=6, ha='right', va='bottom')
        ax.add_patch(Circle((x,y), arrive_r, ec='blue', ls='--', fc='none',
                            label='Arrive Radius' if i==1 else ''))
    gps_pt, = ax.plot([], [], 'o', c='gold', ms=8, label='GPS')
    wp_pt,  = ax.plot([], [], '*', c='magenta', ms=12, label='Target WP')
    route_ln,= ax.plot([], [], '-', c='saddlebrown', lw=1, label='Route')
    tgt_ln,  = ax.plot([], [], '--', c='cyan', lw=1, label='TargetLine')
    info_txt = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       ha='left', va='top',
                       bbox=dict(fc='white', alpha=0.7), fontsize=9)
    ax.legend(loc='upper right')
    fig.canvas.draw()
    return fig, ax, gps_pt, wp_pt, route_ln, tgt_ln, info_txt, None, None

def find_rejoin_wp(cx, cy, wx, wy, yaw, delta_deg):
    theta = yaw + math.radians(delta_deg)
    v = np.array([math.cos(theta), math.sin(theta)])
    vecs = np.vstack([wx-cx, wy-cy]).T
    projs = vecs.dot(v)
    ahead = np.nonzero(projs > 0)[0]
    if len(ahead):
        return int(ahead.max())
    return int(np.hypot(wx-cx, wy-cy).argmin())

def update_visualization(cx, cy, tx, ty, fig_ax, route, st):
    fig, ax, gps_pt, wp_pt, route_ln, tgt_ln, info_txt, h_arrow, s_arrow = fig_ax
    gps_pt.set_data([cx],[cy])
    wp_pt.set_data([tx],[ty])
    route_ln.set_data(route[0],route[1])
    tgt_ln.set_data([cx,tx],[cy,ty])

    if h_arrow: h_arrow.remove()
    if s_arrow: s_arrow.remove()

    hx, hy = cx + math.cos(st['yaw']), cy + math.sin(st['yaw'])
    h_arrow = FancyArrowPatch((cx,cy),(hx,hy), color='blue', lw=2, arrowstyle='-|>', mutation_scale=15)
    ax.add_patch(h_arrow)
    sx, sy = cx + math.cos(st['yaw']+math.radians(st['delta'])), cy + math.sin(st['yaw']+math.radians(st['delta']))
    s_arrow = FancyArrowPatch((cx,cy),(sx,sy), color='red', lw=2, arrowstyle='-|>', mutation_scale=15)
    ax.add_patch(s_arrow)

    dist = math.hypot(tx-cx, ty-cy)
    time_str = datetime.datetime.fromtimestamp(st['t']).strftime('%H:%M:%S')
    info_txt.set_text(
        f"Time: {time_str}\n"
        f"Heading: {math.degrees(st['yaw']):.1f}°\n"
        f"Steering: {st['delta']:+.1f}°\n"
        f"Pos: ({cx:.1f}, {cy:.1f}) m\n"
        f"Dist→Target: {dist:.1f} m\n"
        f"Target WP: {st['target']+1}\n"
        f"Current WP: {st.get('last_wp',1)}"
    )

    m = 10
    ax.set_xlim(cx-m, cx+m)
    ax.set_ylim(cy-m, cy+m)
    fig.canvas.draw_idle()
    plt.pause(0.001)
    return fig, ax, gps_pt, wp_pt, route_ln, tgt_ln, info_txt, h_arrow, s_arrow

def save_logs(logs, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time','lat','lon','heading','delta','wp'])
        w.writerows(logs)

# ─────────────────────────────────────────────────────────
# 메인 실행 (ROS 퍼블리시 추가)
# ─────────────────────────────────────────────────────────
def run_loop_ros():
    rospy.init_node('gngga_waypoint_tracker', anonymous=False)

    # ROS 파라미터로 CONFIG 덮어쓰기
    CONFIG['csv_path']   = rospy.get_param('~csv_path',   CONFIG['csv_path'])
    CONFIG['gnss_port']  = rospy.get_param('~gnss_port',  CONFIG['gnss_port'])
    CONFIG['ardu_port']  = rospy.get_param('~ardu_port',  CONFIG['ardu_port'])
    CONFIG['baud']       = rospy.get_param('~baud',       CONFIG['baud'])
    CONFIG['timeout']    = rospy.get_param('~timeout',    CONFIG['timeout'])
    CONFIG['wheelbase']  = float(rospy.get_param('~wheelbase',  CONFIG['wheelbase']))
    CONFIG['look_dist0'] = float(rospy.get_param('~look_dist0', CONFIG['look_dist0']))
    CONFIG['k_ld']       = float(rospy.get_param('~k_ld',       CONFIG['k_ld']))
    CONFIG['k_v']        = float(rospy.get_param('~k_v',        CONFIG['k_v']))
    CONFIG['arrive_r']   = float(rospy.get_param('~arrive_r',   CONFIG['arrive_r']))
    CONFIG['exit_r']     = float(rospy.get_param('~exit_r',     CONFIG['exit_r']))
    CONFIG['tx_period']  = float(rospy.get_param('~tx_period',  CONFIG['tx_period']))
    CONFIG['tx_on']      = bool(rospy.get_param('~tx_on',       CONFIG['tx_on']))
    CONFIG['const_speed']= float(rospy.get_param('~const_speed',CONFIG['const_speed']))
    CONFIG['route_maxlen']=int(rospy.get_param('~route_maxlen', CONFIG['route_maxlen']))
    CONFIG['log_csv']    = rospy.get_param('~log_csv',    CONFIG['log_csv'])

    # 퍼블리시 토픽 이름
    speed_topic  = rospy.get_param('~speed_topic',  '/vehicle/speed_cmd')
    steer_topic  = rospy.get_param('~steer_topic',  '/vehicle/steer_cmd')
    status_topic = rospy.get_param('~status_topic', '/rtk/status')

    pub_speed  = rospy.Publisher(speed_topic,  Float32, queue_size=10)
    pub_steer  = rospy.Publisher(steer_topic,  Float32, queue_size=10)
    pub_status = rospy.Publisher(status_topic, String,  queue_size=10)

    df, lat0, lon0, wx, wy = load_waypoints(CONFIG['csv_path'], interval=2.0)
    ser_gnss = open_serial(CONFIG['gnss_port'], CONFIG['baud'], CONFIG['timeout'])
    ser_ardu = open_serial(CONFIG['ardu_port'], CONFIG['baud'], CONFIG['timeout'])

    fig_ax = init_plot(df, wx, wy, CONFIG['arrive_r'])

    st = {
        'state':'INIT', 'first':False,
        'prev_x':None, 'prev_y':None, 'prev_t':None,
        'yaw':0.0, 'v':0.0, 'delta':0.0,
        'target':0, 'wp_reached':False,
        'last_tx':0.0, 'lat':None, 'lon':None,
        'last_wp':1, 't':time.time()
    }
    route = [deque(maxlen=CONFIG['route_maxlen']), deque(maxlen=CONFIG['route_maxlen'])]
    logs = []

    rate = rospy.Rate(50)  # 시각화 갱신은 plt.pause가 처리

    rospy.loginfo("[viewer] start loop")
    while not rospy.is_shutdown():
        if ser_gnss and ser_gnss.in_waiting:
            raw = ser_gnss.readline().decode('ascii','ignore').strip()
            p = parse_gngga(raw)
            if not p:
                rate.sleep(); continue

            st['lat'], st['lon'], fixq = p
            x_ecef, y_ecef = lla_to_ecef_xy(st['lat'], st['lon'], 0.0)
            cx, cy = ecef_xy_to_enu(x_ecef, y_ecef, lat0, lon0, 0.0)
            now = time.time()
            route[0].append(cx); route[1].append(cy)

            # 상태 업데이트
            if st['state'] == 'INIT':
                st['t'] = now
                if not st['first']:
                    st['prev_x'], st['prev_y'], st['prev_t'] = cx, cy, now
                    st['first'] = True
                else:
                    st['state'] = 'TRACK'
                tx, ty = wx[0], wy[0]
            else:
                st['t'] = now
                dx, dy = cx - st['prev_x'], cy - st['prev_y']
                dt = now - st['prev_t']
                if dt>0 and math.hypot(dx,dy)>1e-3:
                    st['yaw'] = math.atan2(dy,dx)
                    st['v']   = math.hypot(dx,dy)/dt
                st['prev_x'], st['prev_y'], st['prev_t'] = cx, cy, now

                tx, ty = wx[st['target']], wy[st['target']]
                d2 = (tx-cx)**2 + (ty-cy)**2
                in2, out2 = CONFIG['arrive_r']**2, CONFIG['exit_r']**2

                if not st['wp_reached'] and d2 < in2:
                    st['wp_reached'], st['last_wp'] = True, st['target']+1
                    st['target'] = min(st['target']+1, len(wx)-1)
                elif st['wp_reached'] and d2 > out2:
                    st['wp_reached'] = False

                if d2 > out2:
                    new_tgt = find_rejoin_wp(cx, cy, wx, wy, st['yaw'], st['delta'])
                    st['target'], st['wp_reached'] = new_tgt, False
                tx, ty = wx[st['target']], wy[st['target']]

                # Pure-Pursuit
                Ld = CONFIG['look_dist0'] + CONFIG['k_ld'] + CONFIG['k_v'] * st['v']
                alpha = math.atan2(ty-cy, tx-cx) - st['yaw']
                alpha = math.atan2(math.sin(alpha), math.cos(alpha))
                raw_delta_deg = math.degrees(math.atan2(2*CONFIG['wheelbase']*math.sin(alpha), Ld))
                st['delta'] = max(-STEER_LIMIT_DEG, min(STEER_LIMIT_DEG, raw_delta_deg))

                # Arduino로도 송신(옵션)
                if CONFIG['tx_on'] and ser_ardu and ser_ardu.is_open and now - st['last_tx'] >= CONFIG['tx_period']:
                    ser_ardu.write(f"{CONFIG['const_speed']},{st['delta']:.2f}\n".encode())
                    st['last_tx'] = now

                logs.append([now, st['lat'], st['lon'], math.degrees(st['yaw']), st['delta'], st['target']+1])

            # ── ROS 퍼블리시 ──
            pub_speed.publish(Float32(CONFIG['const_speed']))  # 현재 로직은 고정 속도
            pub_steer.publish(Float32(st['delta']))            # deg
            pub_status.publish(String(fix_quality_to_text(fixq)))

            # 콘솔 로그(ROS)
            rospy.loginfo_throttle(0.5,
                f"Lat:{st['lat']:.7f}, Lon:{st['lon']:.7f}, "
                f"RTK:{fix_quality_to_text(fixq)}, Speed:{CONFIG['const_speed']:.2f} m/s, Steer:{st['delta']:+.2f} deg")

            # 시각화
            fig_ax = update_visualization(cx, cy, tx, ty, fig_ax, route, st)

        rate.sleep()

    save_logs(logs, CONFIG['log_csv'])
    rospy.loginfo("[viewer] end loop")
    plt.close('all')

if __name__ == '__main__':
    try:
        run_loop_ros()
    except rospy.ROSInterruptException:
        pass
