#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waypoints_live_gngga_viewer_seq_tx_enu.py (ENU 변환판, Z축 완전 미포함)
• INIT→TRACK 상태머신 + Pure-Pursuit + 이중 반경 히스테리시스 + 경로 복귀
• LLA→ECEF→ENU 변환 적용 (East, North 좌표만 사용)
• 매 프레임 데이터 수집 후 종료 시 CSV 저장
"""

import math
import time
import csv
import datetime
from collections import deque

import serial            # 시리얼 통신
import numpy as np       # 수치 계산
import pandas as pd      # 데이터 처리
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# ─────────────────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────────────────
CONFIG = {
    'csv_path': r"C:\Users\dlwlr\OneDrive - 한라대학교\바탕 화면\GPS_purse_Python\bocsa\waypoints_2m_1.csv",
    'gnss_port': 'COM10',      # GNSS 수신 포트
    'ardu_port': 'COM7',       # Arduino 제어 포트
    'baud': 115200,            # 통신 속도
    'timeout': 0.0,            # 시리얼 타임아웃
    'auto_mode': False,        # 자동 추종 모드 여부
    'wheelbase': 0.28,         # 차량 축간 거리 [m]
    'look_dist0': 1.2,         # 기본 Look-ahead 거리 [m]
    'k_ld': 0.3,               # Look-ahead 고정 계수
    'k_v': 0.5,                # 속도 가변 계수
    'arrive_r': 0.5,           # 도착 반경 [m]
    'exit_r': 0.8,             # 이탈 반경 [m]
    'tx_period': 0.2,          # Arduino 전송 주기 [s]
    'tx_on': True,             # 제어 신호 전송 여부
    'const_speed': 1,          # 고정 속도 명령 [m/s]
    'route_maxlen': 10000,     # 경로 기록 최대 길이
    'log_csv': 'gps_wp_log 1.csv',# 로그 저장 파일명
    'v_th': 0.1,               # 속도 임계값 (정지 판단)
}

# WGS84 지구 타원체 상수
_a  = 6378137.0               # 장반경 [m]
_f  = 1 / 298.257223563       # 편평률
_e2 = _f * (2 - _f)           # 제2 이심률

stop_flag = False  # 루프 종료 플래그

# ─────────────────────────────────────────────────────────
# NMEA 파싱 및 좌표 변환 함수
# ─────────────────────────────────────────────────────────
def ddmm2deg(val: float) -> float:
    """NMEA ddmm.mmmm 형식을 십진도(decimal degrees)로 변환"""
    d = int(val // 100)
    m = val - 100 * d
    return d + m / 60.0

def parse_gngga(line: str):
    """$GNGGA 문자열에서 위도·경도 파싱, 실패 시 None 반환"""
    parts = line.split(',')
    if not line.startswith('$GNGGA') or len(parts) < 6:
        return None
    lat = ddmm2deg(float(parts[2]))
    lon = ddmm2deg(float(parts[4]))
    if parts[3] == 'S': lat = -lat
    if parts[5] == 'W': lon = -lon
    return lat, lon

# ─────────────────────────────────────────────────────────
# ECEF 변환: X, Y 계산
# ─────────────────────────────────────────────────────────
def lla_to_ecef_xy(lat, lon, h=0.0):
    """위경도(lat,lon) → ECEF(x,y) 변환 """
    φ = math.radians(lat)
    λ = math.radians(lon)
    N = _a / math.sqrt(1 - _e2 * math.sin(φ)**2)
    x = (N + h) * math.cos(φ) * math.cos(λ)
    y = (N + h) * math.cos(φ) * math.sin(λ)
    return x, y

# ─────────────────────────────────────────────────────────
# ENU 변환:  East/North 반환
# ─────────────────────────────────────────────────────────
def ecef_xy_to_enu(x, y, lat0, lon0, h0=0.0):
    """ECEF(x,y) → ENU(east,north) 변환"""
    x0, y0 = lla_to_ecef_xy(lat0, lon0, h0)
    dx, dy = x - x0, y - y0
    φ0 = math.radians(lat0)
    λ0 = math.radians(lon0)
    # 2×2 변환 행렬
    t = np.array([
        [-math.sin(λ0),                math.cos(λ0)],
        [-math.sin(φ0)*math.cos(λ0),  -math.sin(φ0)*math.sin(λ0)]
    ])
    enu = t.dot(np.array([dx, dy]))
    return enu[0], enu[1]

# ─────────────────────────────────────────────────────────
# 웨이포인트 로드 및 보간
# ─────────────────────────────────────────────────────────
def load_waypoints(path: str, interval: float = 2.0):
    """
    CSV에서 Lat/Lon 읽어와 ECEF→ENU 변환 후
    일정 간격(interval)마다 보간된 Waypoint 생성
    """
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
# 시리얼 포트 오픈
# ─────────────────────────────────────────────────────────
def open_serial(port, baud, timeout):
    """시리얼 포트 열기, 실패 시 None 리턴"""
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        print(f"[INFO] Opened {port}")
        return ser
    except Exception as e:
        print(f"[ERROR] {port}: {e}")
        return None

# ─────────────────────────────────────────────────────────
# 초기 그래프 설정
# ─────────────────────────────────────────────────────────
def init_plot(df, wx, wy):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(df['x'], df['y'], color='gray', lw=1, label='Path')
    for i,(x,y) in enumerate(zip(wx, wy), 1):
        ax.scatter(x, y, c='red', s=18, label='Waypoints' if i==1 else '')
        ax.text(x, y, str(i), fontsize=6, ha='right', va='bottom')
        ax.add_patch(Circle((x,y), CONFIG['arrive_r'],
                            ec='blue', ls='--', fc='none',
                            label='Arrive Radius' if i==1 else ''))
    gps_pt, = ax.plot([], [], 'o', c='gold', ms=8, label='GPS')
    wp_pt,  = ax.plot([], [], '*', c='magenta', ms=12, label='Target WP')
    route_ln,= ax.plot([], [], '-', c='saddlebrown', lw=1, label='Route')
    tgt_ln,  = ax.plot([], [], '--', c='cyan', lw=1, label='TargetLine')
    info_txt = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       ha='left', va='top',
                       bbox=dict(fc='white', alpha=0.7), fontsize=9)
    ax.legend(loc='upper right')

    def on_key(evt):
        global stop_flag
        if evt.key in ['q', 'escape']:
            stop_flag = True

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.draw()
    return fig, ax, gps_pt, wp_pt, route_ln, tgt_ln, info_txt, None, None

# ─────────────────────────────────────────────────────────
# 경로 이탈 시 재진입 포인트 계산
# ─────────────────────────────────────────────────────────
def find_rejoin_wp(cx, cy, wx, wy, yaw, delta):
    theta = yaw + math.radians(delta)
    v = np.array([math.cos(theta), math.sin(theta)])
    vecs = np.vstack([wx-cx, wy-cy]).T
    projs = vecs.dot(v)
    ahead = np.nonzero(projs > 0)[0]
    if len(ahead):
        return int(ahead.max())
    return int(np.hypot(wx-cx, wy-cy).argmin())

# ─────────────────────────────────────────────────────────
# INIT 상태 처리
# ─────────────────────────────────────────────────────────
def handle_init(cx, cy, now, st, ser_ardu):
    st['t'] = now
    if not st['first']:
        st['prev_x'], st['prev_y'], st['prev_t'] = cx, cy, now
        st['first'] = True
    else:
        st['state'] = 'TRACK'
    if CONFIG['tx_on'] and ser_ardu and ser_ardu.is_open and now - st['last_tx'] >= CONFIG['tx_period']:
        ser_ardu.write(f"{CONFIG['const_speed']},0\n".encode())
        st['last_tx'] = now

# ─────────────────────────────────────────────────────────
# TRACK 상태 처리 및 Pure-Pursuit 연산
# ─────────────────────────────────────────────────────────
def handle_track(cx, cy, now, wx, wy, st, ser_ardu, logs):
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

    # 도착/이탈 히스테리시스
    if not st['wp_reached'] and d2 < in2:
        st['wp_reached'], st['last_wp'] = True, st['target']+1
        st['target'] = min(st['target']+1, len(wx)-1)
    elif st['wp_reached'] and d2 > out2:
        st['wp_reached'] = False

    # 이탈 시 재진입 WP 결정
    if d2 > out2:
        new_tgt = find_rejoin_wp(cx, cy, wx, wy, st['yaw'], st['delta'])
        st['target'], st['wp_reached'] = new_tgt, False
    tx, ty = wx[st['target']], wy[st['target']]

    # Pure-Pursuit 조향각 계산
    Ld = CONFIG['look_dist0'] + CONFIG['k_ld'] + CONFIG['k_v'] * st['v']
    alpha = math.atan2(ty-cy, tx-cx) - st['yaw']
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    st['delta'] = max(-8, min(8, math.degrees(math.atan2(2*CONFIG['wheelbase']*math.sin(alpha), Ld))))

    # 제어 신호 전송
    if CONFIG['tx_on'] and ser_ardu and ser_ardu.is_open and now - st['last_tx'] >= CONFIG['tx_period']:
        ser_ardu.write(f"{CONFIG['const_speed']},{st['delta']:.2f}\n".encode())
        st['last_tx'] = now

    logs.append([now, st['lat'], st['lon'], math.degrees(st['yaw']), st['delta'], st['target']+1])
    return tx, ty

# ─────────────────────────────────────────────────────────
# 시각화 업데이트
# ─────────────────────────────────────────────────────────
def update_visualization(cx, cy, tx, ty, fig_ax, route, st):
    fig, ax, gps_pt, wp_pt, route_ln, tgt_ln, info_txt, h_arrow, s_arrow = fig_ax
    gps_pt.set_data([cx],[cy])
    wp_pt.set_data([tx],[ty])
    route_ln.set_data(route[0],route[1])
    tgt_ln.set_data([cx,tx],[cy,ty])

    if h_arrow: h_arrow.remove()
    if s_arrow: s_arrow.remove()

    # Heading arrow
    hx, hy = cx + math.cos(st['yaw']), cy + math.sin(st['yaw'])
    h_arrow = FancyArrowPatch((cx,cy),(hx,hy), color='blue', lw=2, arrowstyle='-|>', mutation_scale=15)
    ax.add_patch(h_arrow)
    # Steering arrow
    sx, sy = cx + math.cos(st['yaw']+math.radians(st['delta'])), cy + math.sin(st['yaw']+math.radians(st['delta']))
    s_arrow = FancyArrowPatch((cx,cy),(sx,sy), color='red', lw=2, arrowstyle='-|>', mutation_scale=15)
    ax.add_patch(s_arrow)

    # Info box
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

    # Fixed view around vehicle
    m = 10
    ax.set_xlim(cx-m, cx+m)
    ax.set_ylim(cy-m, cy+m)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    return fig, ax, gps_pt, wp_pt, route_ln, tgt_ln, info_txt, h_arrow, s_arrow

# ─────────────────────────────────────────────────────────
# 로그 저장
# ─────────────────────────────────────────────────────────
def save_logs(logs):
    """로그 리스트를 CSV 파일로 저장"""
    with open(CONFIG['log_csv'], 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time','lat','lon','heading','delta','wp'])
        w.writerows(logs)

# ─────────────────────────────────────────────────────────
# 메인 실행 루프
# ─────────────────────────────────────────────────────────
def run_loop():
    df, lat0, lon0, wx, wy = load_waypoints(CONFIG['csv_path'])
    ser_gnss = open_serial(CONFIG['gnss_port'], CONFIG['baud'], CONFIG['timeout'])
    ser_ardu = open_serial(CONFIG['ardu_port'], CONFIG['baud'], CONFIG['timeout'])
    fig_ax = init_plot(df, wx, wy)

    st = {
        'state':'INIT', 'first':False,
        'prev_x':None, 'prev_y':None, 'prev_t':None,
        'yaw':0, 'v':0, 'delta':0,
        'target':0, 'wp_reached':False,
        'last_tx':0, 'lat':None, 'lon':None,
        'last_wp':1, 't':time.time()
    }
    route = [deque(maxlen=CONFIG['route_maxlen']), deque(maxlen=CONFIG['route_maxlen'])]
    logs = []

    print('[INFO] Loop start')
    while not stop_flag:
        if ser_gnss and ser_gnss.in_waiting:
            raw = ser_gnss.readline().decode('ascii','ignore').strip()
            p = parse_gngga(raw)
            if not p: continue
            st['lat'], st['lon'] = p
            # ENU 변환
            x_ecef, y_ecef = lla_to_ecef_xy(st['lat'], st['lon'], 0.0)
            cx, cy = ecef_xy_to_enu(x_ecef, y_ecef, lat0, lon0, 0.0)
            now = time.time()
            route[0].append(cx)
            route[1].append(cy)

            if st['state'] == 'INIT':
                handle_init(cx, cy, now, st, ser_ardu)
                tx, ty = wx[0], wy[0]
            else:
                tx, ty = handle_track(cx, cy, now, wx, wy, st, ser_ardu, logs)

            fig_ax = update_visualization(cx, cy, tx, ty, fig_ax, route, st)

        time.sleep(0.001)

    save_logs(logs)
    print('[INFO] Loop end')
    plt.close('all')

if __name__ == '__main__':
    try:
        run_loop()
    except KeyboardInterrupt:
        pass
