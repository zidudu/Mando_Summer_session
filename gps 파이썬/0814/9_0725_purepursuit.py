#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waypoints_live_gngga_viewer_seq_tx.py  (로그 저장 수정판)
• INIT→TRACK 상태머신 + Pure-Pursuit + 이중 반경 히스테리시스
• 매 프레임 로그를 run_loop에서 수집하여 종료 시 CSV로 저장
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

# ────────────────────────────────────────────────────────────────
# 설정값
# ────────────────────────────────────────────────────────────────
CONFIG = {
    'csv_path':     r"C:\Users\dlwlr\OneDrive - 한라대학교\바탕 화면\GPS_purse_Python\bocsa\waypoints_2m_1.csv",
    'gnss_port':    'COM10',
    'ardu_port':    'COM7',
    'baud':         115200,
    'timeout':      0.0,
    'auto_mode':    False,
    'wheelbase':    0.28,
    'look_dist0':   1.2,
    'k_ld':         0.3,
    'k_v':          0.5,
    'arrive_r':     0.5,
    'exit_r':       0.8,
    'tx_period':    0.2,
    'tx_on':        True,
    'const_speed':  1,
    'route_maxlen': 10000,
    'log_csv':      'gps_wp_log4.csv',
    'v_th':         0.1,
}
CONST = {'deg2rad': math.pi/180, 'R': 6_378_137.0}
stop_flag = False

def ddmm2deg(val: float) -> float:
    d = int(val // 100); m = val - 100*d
    return d + m/60.0

def parse_gngga(line: str):
    parts = line.split(',')
    if not line.startswith('$GNGGA') or len(parts) < 6:
        return None
    lat = ddmm2deg(float(parts[2])); lon = ddmm2deg(float(parts[4]))
    if parts[3] == 'S': lat = -lat
    if parts[5] == 'W': lon = -lon
    return lat, lon

def load_waypoints(path: str, interval: float = 2.0):
    df = pd.read_csv(path)
    lat0, lon0 = df.loc[0, ['Lat','Lon']]
    df['x'] = (df['Lon']-lon0)*np.cos(lat0*CONST['deg2rad'])*CONST['deg2rad']*CONST['R']
    df['y'] = (df['Lat']-lat0)*CONST['deg2rad']*CONST['R']
    dist = np.hypot(df['x'].diff().fillna(0), df['y'].diff().fillna(0)).cumsum()
    wps = []
    for d in np.arange(0, dist.iloc[-1], interval):
        idx = dist.searchsorted(d)
        if idx == 0:
            wps.append((df.at[0,'x'], df.at[0,'y']))
        else:
            s0, s1 = dist.iloc[idx-1], dist.iloc[idx]
            t = (d - s0) / (s1 - s0)
            x = df.at[idx-1,'x'] + t*(df.at[idx,'x']-df.at[idx-1,'x'])
            y = df.at[idx-1,'y'] + t*(df.at[idx,'y']-df.at[idx-1,'y'])
            wps.append((x, y))
    wx, wy = np.array(wps).T
    return df, lat0, lon0, wx, wy

def open_serial(port, baud, timeout):
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        print(f"[INFO] Opened {port}")
        return ser
    except Exception as e:
        print(f"[ERROR] {port}: {e}")
        return None

def init_plot(df, wx, wy):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(df['x'], df['y'], color='gray', lw=1, label='Path')
    for i,(x,y) in enumerate(zip(wx,wy),1):
        ax.scatter(x,y,c='red',s=18, label='Waypoints' if i==1 else "")
        ax.text(x,y,str(i),fontsize=6,ha='right',va='bottom')
        ax.add_patch(Circle((x,y), CONFIG['arrive_r'], ec='blue', ls='--',
                            fc='none', label='Arrive Radius' if i==1 else ""))
    gps_pt,   = ax.plot([], [], 'o', c='gold', ms=8, label='GPS')
    wp_pt,    = ax.plot([], [], '*', c='magenta', ms=12, label='Target WP')
    route_ln, = ax.plot([], [], '-', c='saddlebrown', lw=1, label='Route')
    tgt_ln,   = ax.plot([], [], '--', c='cyan', lw=1, label='TargetLine')
    info_txt  = ax.text(0.02,0.98,'', transform=ax.transAxes,
                        ha='left', va='top',
                        bbox=dict(fc='white', alpha=0.7),
                        fontsize=9)
    ax.legend(loc='upper right')
    def on_key(evt):
        global stop_flag
        if evt.key in ['q','escape']:
            stop_flag = True
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.draw()
    return fig, ax, gps_pt, wp_pt, route_ln, tgt_ln, info_txt, None, None

def handle_init(cx, cy, now, st, ser):
    st['t'] = now
    if not st['first']:
        st['prev_x'], st['prev_y'], st['prev_t'] = cx, cy, now
        st['first'] = True
    else:
        st['state'] = 'TRACK'
    # 로그: INIT 단계에서도 매 프레임 기록
    st['log_buffer'].append([now, st['lat'], st['lon'], 0.0, 0.0, 1])
    if CONFIG['tx_on'] and ser and ser.is_open and now - st['last_tx'] >= CONFIG['tx_period']:
        ser.write(f"{CONFIG['const_speed']},0\n".encode())
        st['last_tx'] = now

def handle_track(cx, cy, now, wx, wy, st, ser, logs):
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
        st['wp_reached'] = True
        st['last_wp'] = st['target']+1
        st['target'] = min(st['target']+1, len(wx)-1)
    elif st['wp_reached'] and d2 > out2:
        st['wp_reached'] = False

    # Pure-Pursuit
    Ld = CONFIG['look_dist0'] + CONFIG['k_ld'] + CONFIG['k_v'] * st['v']
    alpha = math.atan2(ty-cy, tx-cx) - st['yaw']
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    raw_delta = math.degrees(math.atan2(2*CONFIG['wheelbase']*math.sin(alpha), Ld))
    st['delta'] = max(-8, min(8, raw_delta))

    # 로그: TRACK 매 프레임
    logs.append([now, st['lat'], st['lon'],
                 math.degrees(st['yaw']), st['delta'],
                 st['target']+1])

    if CONFIG['tx_on'] and ser and ser.is_open and now - st['last_tx'] >= CONFIG['tx_period']:
        ser.write(f"{CONFIG['const_speed']},{st['delta']:.2f}\n".encode())
        st['last_tx'] = now

    return tx, ty

def update_visualization(cx, cy, tx, ty, fig_ax, route, st):
    fig, ax, gps_pt, wp_pt, route_ln, tgt_ln, info_txt, h_arrow, s_arrow = fig_ax
    gps_pt.set_data([cx],[cy])
    wp_pt.set_data([tx],[ty])
    route_ln.set_data(route[0], route[1])
    tgt_ln.set_data([cx,tx], [cy,ty])
    if h_arrow: h_arrow.remove()
    if s_arrow: s_arrow.remove()

    hx, hy = cx+math.cos(st['yaw']), cy+math.sin(st['yaw'])
    h_arrow = FancyArrowPatch((cx,cy),(hx,hy), color='blue', lw=2,
                              arrowstyle='-|>', mutation_scale=15)
    ax.add_patch(h_arrow)
    sx = cx+math.cos(st['yaw']+math.radians(st['delta']))
    sy = cy+math.sin(st['yaw']+math.radians(st['delta']))
    s_arrow = FancyArrowPatch((cx,cy),(sx,sy), color='red', lw=2,
                              arrowstyle='-|>', mutation_scale=15)
    ax.add_patch(s_arrow)

    dist = math.hypot(tx-cx, ty-cy)
    time_str = datetime.datetime.fromtimestamp(st['t']).strftime('%H:%M:%S')
    info_txt.set_text(
        f"Time: {time_str}\n"
        f"Heading: {math.degrees(st['yaw']):.1f}°\n"
        f"Steering: {st['delta']:+.1f}°\n"
        f"Position: ({cx:.1f},{cy:.1f}) m\n"
        f"Dist→Target: {dist:.1f} m\n"
        f"Target WP: {st['target']+1}\n"
        f"Current WP: {st['last_wp']}"
    )

    m = 10
    ax.set_xlim(cx-m, cx+m)
    ax.set_ylim(cy-m, cy+m)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return fig, ax, gps_pt, wp_pt, route_ln, tgt_ln, info_txt, h_arrow, s_arrow

def save_logs(logs):
    with open(CONFIG['log_csv'],'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['time','lat','lon','heading','delta','wp'])
        w.writerows(logs)

def run_loop():
    df,lat0,lon0,wx,wy = load_waypoints(CONFIG['csv_path'])
    ser_gnss = open_serial(CONFIG['gnss_port'], CONFIG['baud'], CONFIG['timeout'])
    ser_ardu = open_serial(CONFIG['ardu_port'], CONFIG['baud'], CONFIG['timeout'])
    fig_ax   = init_plot(df,wx,wy)

    st = {
        'state':'INIT','first':False,'prev_x':None,'prev_y':None,'prev_t':None,
        'yaw':0,'v':0,'delta':0,'target':0,'wp_reached':False,
        'last_tx':0,'lat':None,'lon':None,'last_wp':1,
        't':time.time(),
        'log_buffer': []
    }
    route = [deque(maxlen=CONFIG['route_maxlen']), deque(maxlen=CONFIG['route_maxlen'])]
    logs  = []

    print('[INFO] Loop start')
    while not stop_flag:
        if ser_gnss and ser_gnss.in_waiting:
            raw = ser_gnss.readline().decode('ascii','ignore').strip()
            p = parse_gngga(raw)
            if not p: continue

            st['lat'], st['lon'] = p
            cx = (st['lon']-lon0)*math.cos(lat0*CONST['deg2rad'])*CONST['deg2rad']*CONST['R']
            cy = (st['lat']-lat0)*CONST['deg2rad']*CONST['R']
            now = time.time()
            route[0].append(cx); route[1].append(cy)

            # INIT 단계 로그 버퍼를 실제 logs로 옮기기
            if st['state']=='INIT' and st['log_buffer']:
                logs.extend(st['log_buffer'])
                st['log_buffer'].clear()

            if st['state']=='INIT':
                handle_init(cx, cy, now, st, ser_ardu)
                # **INIT 시에도 tx, ty 정의**
                tx, ty = wx[0], wy[0]
            else:
                tx, ty = handle_track(cx, cy, now, wx, wy, st, ser_ardu, logs)

            fig_ax = update_visualization(cx, cy, tx, ty, fig_ax, route, st)

        time.sleep(0.001)

    save_logs(logs)
    print('[INFO] Loop end')

if __name__ == '__main__':
    try:
        run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
