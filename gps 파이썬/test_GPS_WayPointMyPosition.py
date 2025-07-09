# Display the full contents of the GPS live map with waypoints script
file_path = "/mnt/data/gps_live_map_with_waypoints.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()
print(content)
STDOUT/STDERR

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 RTK‑GPS 경로 + 현재 차량 위치(빨강 점) + 웨이포인트(초록 점) OSM 지도 시각화
---------------------------------------------------------------
• EVK‑F9P → $GNGGA(NMEA) 수신
• 이동평균 + 정지 탐지(윈도우 평균) 후 표시
• Web Mercator(EPSG:3857) 좌표로 변환해 contextily 타일과 합성
• 실시간으로 궤적, 현재 위치, 웨이포인트를 갱신
• Ctrl+C 로 종료 시 Raw/Filtered CSV 및 Waypoints CSV 저장
© 2025-07-09
"""

import os, time, math, serial
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx

# ── 시리얼 / 지오 설정 ─────────────────────────────
PORT, BAUD, TIMEOUT = "COM10", 115200, 0.1  # u-center와 동일하게
R_WGS84 = 6_378_137                         # WGS‑84 지구 반경(m)

# ── 필터 파라미터 ────────────────────────────────
WIN_SMA, WIN_STILL, EPS_M = 20, 30, 0.03

# ── 웨이포인트 간격 (m) ───────────────────────────
STATIC_SPACING = 1.0

# ── 결과 파일 경로 ────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_CSV     = os.path.join(BASE_DIR, "raw_track.csv")
FLT_CSV     = os.path.join(BASE_DIR, "filtered_track.csv")
WPT_CSV     = os.path.join(BASE_DIR, "waypoints.csv")

# ── 유틸리티 함수 ──────────────────────────────────
def nmea2deg(val: str) -> float:
    """ddmm.mmmm → 십진도"""
    try: v = float(val)
    except: return float('nan')
    d = int(v // 100); m = v - d*100
    return d + m/60

def mercator_xy(lat_deg: float, lon_deg: float):
    x = R_WGS84 * math.radians(lon_deg)
    y = R_WGS84 * math.log(math.tan(math.radians(90+lat_deg)/2))
    return x, y

def open_port():
    try: return serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    except Exception as e:
        print("포트 열기 실패:", e)
        return None

def save_csv(fname: str, arr: np.ndarray, header="X_m,Y_m"):
    np.savetxt(fname, arr, delimiter=',', header=header, comments='', fmt='%.6f')

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def make_waypoints(xs, ys, spacing_m=STATIC_SPACING):
    """일정 간격(static)으로 Waypoint 생성"""
    if len(xs) == 0: return np.empty((0,2))
    wpts = [(xs[0], ys[0])]
    acc = 0.0
    for i in range(1, len(xs)):
        seg = dist((xs[i-1], ys[i-1]), (xs[i], ys[i]))
        acc += seg
        while acc >= spacing_m:
            ratio = (seg - (acc - spacing_m)) / seg
            x_wp = xs[i-1] + (xs[i]-xs[i-1])*ratio
            y_wp = ys[i-1] + (ys[i]-ys[i-1])*ratio
            wpts.append((x_wp, y_wp))
            acc -= spacing_m
    if (xs[-1], ys[-1]) != wpts[-1]:
        wpts.append((xs[-1], ys[-1]))
    return np.array(wpts)

# ── 인터랙티브 플롯 설정 ─────────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("실시간 GPS + Waypoints (OSM)")
ax.grid(True)

path_line,   = ax.plot([], [], lw=1.5, label="Filtered Path", color="blue")
curr_scatter = ax.scatter([], [], s=50, color="red", label="Current")
wp_scatter   = ax.scatter([], [], s=20, color="green", label="Waypoints")
ax.legend(loc='upper left')

# 버퍼 초기화
xs_raw, ys_raw = [], []
xs_f,   ys_f   = [], []
sma_win = deque(maxlen=WIN_SMA)
still_win = deque(maxlen=WIN_STILL)

basemap_added = False
ser = open_port()
print("GNGGA 수신 시작 – Ctrl+C 로 종료")

try:
    while True:
        if not ser or not ser.is_open:
            time.sleep(1)
            ser = open_port()
            continue

        line = ser.readline().decode(errors='ignore').strip()
        if not line.startswith("$GNGGA"): continue
        parts = line.split(",")
        lat_r, ns, lon_r, ew = parts[2:6]
        lat = nmea2deg(lat_r); lon = nmea2deg(lon_r)
        if ns=='S': lat=-lat
        if ew=='W': lon=-lon

        x, y = mercator_xy(lat, lon)

        # 이동평균
        sma_win.append((x,y))
        x_sma, y_sma = np.mean(sma_win, axis=0)

        # 정지 탐지
        still_win.append((x_sma, y_sma))
        if len(still_win)>1:
            sw = np.array(still_win)
            dev = np.max(np.linalg.norm(sw - sw.mean(axis=0), axis=1))
        else:
            dev = EPS_M + 1
        x_use, y_use = (sw.mean(axis=0) if dev< EPS_M else (x_sma,y_sma))

        xs_raw.append(x); ys_raw.append(y)
        xs_f.append(x_use); ys_f.append(y_use)

        # Waypoints 생성
        wpts = make_waypoints(xs_f, ys_f, spacing_m=STATIC_SPACING)

        # 플롯 업데이트
        path_line.set_data(xs_f, ys_f)
        curr_scatter.set_offsets([[x_use, y_use]])
        wp_scatter.set_offsets(wpts)

        # 배경 타일 추가 (한 번만)
        if not basemap_added and len(xs_f)>=10:
            try:
                ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.OpenStreetMap.Mapnik)
                basemap_added = True
            except Exception as e:
                print("타일 로드 오류:", e)

        ax.relim(); ax.autoscale_view()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("\n수집 종료 – 저장 중…")

finally:
    if ser and ser.is_open: ser.close()
    plt.ioff(); plt.show()

    # CSV 저장
    save_csv(RAW_CSV, np.column_stack([xs_raw, ys_raw]))
    save_csv(FLT_CSV, np.column_stack([xs_f,   ys_f]))
    # Waypoints 저장
    save_csv(WPT_CSV, wpts, header="X_m,Y_m")
    print(f"저장 완료: raw->{RAW_CSV}, filtered->{FLT_CSV}, waypoints->{WPT_CSV}")
