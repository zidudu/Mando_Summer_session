#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTK-GPS 실시간 로우 데이터 수집 + 방위각(heading) 기반 동적 웨이포인트
  - $GNGGA → 위‧경도 → Web Mercator(X,Y)
  - 실시간 궤적 플롯
  - 종료 시점: 연속 샘플 heading 변화 |Δψ|에 따라
      |Δψ|<1°  → 0.55 m  (직선)
      1°≤|Δψ|<5°→ 0.40 m  (완만)
      ≥5°      → 0.28 m  (급곡선)
    간격으로 웨이포인트 생성
"""
import os, serial, time, math
import numpy as np
import matplotlib.pyplot as plt

# ── 유틸 ──────────────────────────────────────────
def unique_filepath(dirpath, basename, ext=".csv"):
    path = os.path.join(dirpath, basename + ext)
    if not os.path.exists(path): return path
    idx = 1
    while True:
        path = os.path.join(dirpath, f"{basename}_{idx}{ext}")
        if not os.path.exists(path): return path
        idx += 1

def save_csv(fname, arr, header):
    np.savetxt(fname, arr, delimiter=",",
               header=",".join(header), comments="", fmt="%.6f")

def nmea2deg(v):
    try:
        v = float(v); d = int(v//100); m = v - d*100
        return d + m/60
    except: return float('nan')

def mercator_xy(lat, lon):
    R = 6_378_137.0
    x = R*math.radians(lon)
    y = R*math.log(math.tan(math.radians(90+lat)/2))
    return x, y

# ── Heading-기반 웨이포인트 ───────────────────────
def heading_waypoints(xs, ys,
                      D_STRAIGHT=0.55, D_MID=0.40, D_CURVE=0.28):
    # bearing 배열
    dx = np.diff(xs); dy = np.diff(ys)
    bearings = np.degrees(np.arctan2(dy, dx))          # −180~180°
    bearings = (bearings + 360) % 360                  # 0~360
    dists = np.hypot(dx, dy)

    wpts = [(xs[0], ys[0])]
    acc  = 0.0
    for i in range(1, len(xs)):
        if i == 1:
            dpsi = 0.0
        else:
            dpsi = abs(bearings[i-1] - bearings[i-2])
            dpsi = 360 - dpsi if dpsi > 180 else dpsi  # 최소 회전각

        # 구간별 목표 간격
        if dpsi < 1.0:
            d_target = D_STRAIGHT
        elif dpsi < 5.0:
            d_target = D_MID
        else:
            d_target = D_CURVE

        seg = dists[i-1]
        acc += seg
        while acc >= d_target and seg > 0:
            overshoot = acc - d_target
            ratio = (seg - overshoot) / seg
            x_wp = xs[i-1] + dx[i-1]*ratio
            y_wp = ys[i-1] + dy[i-1]*ratio
            wpts.append((x_wp, y_wp))
            acc -= d_target
    wpts.append((xs[-1], ys[-1]))
    return wpts

# ── 설정 ──────────────────────────────────────────
PORT, BAUD, TIMEOUT = "COM10", 115200, 0.1
BASE = os.path.dirname(os.path.abspath(__file__))
LOG_RAW = unique_filepath(BASE, "Hraw_track_xy")
LOG_WPT = unique_filepath(BASE, "Hwaypoints_heading")

# ── 플롯 ──────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots()
ax.set_title("RTK-GPS 실시간 궤적 (Heading 기반)")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.grid(True)
line_raw, = ax.plot([], [], '.-', alpha=0.6)
xs, ys = [], []

def open_serial():
    try:
        return serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    except Exception as e:
        print("Serial open err:", e); return None

ser = open_serial()
print("GNGGA 수신…  Ctrl+C 종료")

# ── 메인 루프 ─────────────────────────────────────
try:
    while True:
        if ser is None or not ser.is_open:
            time.sleep(1); ser = open_serial(); continue
        s = ser.readline().decode(errors="ignore").strip()
        if not s.startswith("$GNGGA"): continue
        p = s.split(",");  # lat N/S lon E/W
        if len(p) < 6 or not p[2] or not p[4]: continue

        lat = nmea2deg(p[2]); lon = nmea2deg(p[4])
        lat *= -1 if p[3]=="S" else 1
        lon *= -1 if p[5]=="W" else 1
        x,y = mercator_xy(lat, lon)

        xs.append(x); ys.append(y)
        line_raw.set_data(xs, ys)
        ax.relim(); ax.autoscale_view()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("\n종료 → 웨이포인트 생성")

finally:
    if ser and ser.is_open: ser.close()

    # ① raw 저장
    save_csv(LOG_RAW, np.column_stack([xs,ys]), ["X","Y"])
    # ② heading 기반 웨이포인트
    wpts = heading_waypoints(xs, ys)
    save_csv(LOG_WPT, np.array(wpts), ["X_wp","Y_wp"])
    print("저장:", LOG_RAW, LOG_WPT)

    plt.ioff(); plt.show()
