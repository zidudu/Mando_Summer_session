#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTK-GPS 실시간 수집 + 이동평균·정지 탐지 + 번호 라벨
+ CSV 저장 + 웨이포인트
  · static 1 m 간격          : make_waypoints()
  · 곡률 기반(adaptive)      : resample_by_curvature()
  · Pure-Pursuit Look-ahead  : wp_pure_pursuit()   ← NEW
© 2025-07-09
"""

import os, time, math, serial, sys
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# ───────────────── 경로 설정 ───────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "raw_track_pp.csv")
FLT_FILE = os.path.join(BASE_DIR, "filtered_track_pp.csv")
RAW_WPT_FILE  = os.path.join(BASE_DIR, "raw_wp_static.csv")
WPT_STATIC    = os.path.join(BASE_DIR, "filt_wp_static.csv")
WPT_CURV_FILE = os.path.join(BASE_DIR, "filt_wp_curv.csv")
WPT_PP_FILE   = os.path.join(BASE_DIR, "filt_wp_purepursuit.csv")

# ───────────────── 시리얼 설정 ────────────────
PORT, BAUD, TIMEOUT = "COM10", 115200, 0.1
R_WGS84 = 6_378_137               # WGS-84 반경(m)

# ───────────────── 필터 파라미터 ───────────────
WIN_SMA, WIN_STILL, EPS_M = 20, 30, 0.03

# ───────────────── NMEA 헬퍼 ──────────────────
def nmea2deg(v: str) -> float:
    try: v = float(v)
    except: return float('nan')
    d = int(v // 100); m = v - d*100
    return d + m / 60

def mercator_xy(lat_deg, lon_deg):
    x = R_WGS84 * math.radians(lon_deg)
    y = R_WGS84 * math.log(math.tan(math.radians(90 + lat_deg) / 2))
    return x, y

def open_port():
    try: return serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    except Exception as e:
        print("포트 오류:", e); return None

# ───────────────── 곡률 기반 함수 ─────────────
D_STRAIGHT, D_MID, D_CURVE = 1.0, 0.40, 0.28
K_TH1, K_TH2 = 0.10, 0.20

def curvature(x, y):
    x, y = np.asarray(x), np.asarray(y)
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = dx*ddy - dy*ddx
    den = (dx*dx + dy*dy)**1.5
    κ = np.zeros_like(x)
    mask = den > 1e-6
    κ[mask] = np.abs(num[mask] / den[mask])
    return κ

def resample_by_curvature(xs, ys):
    pts = np.vstack([xs, ys]).T
    pts = pts[~np.isnan(pts).any(axis=1)]
    pts = np.unique(pts, axis=0)
    if len(pts) < 4:
        return pts
    x, y = pts[:,0], pts[:,1]
    tck, _ = splprep([x, y], s=0, k=min(3, len(x)-1))
    u = np.linspace(0, 1, max(2*len(x), 500))
    xd, yd = splev(u, tck)
    κ = curvature(xd, yd)
    d_target = np.where(κ < K_TH1, D_STRAIGHT,
                np.where(κ < K_TH2, D_MID, D_CURVE))
    dist = np.hstack(([0], np.cumsum(np.hypot(np.diff(xd), np.diff(yd)))))
    new_d, acc = [0.0], 0.0
    for i in range(1, len(dist)):
        acc += dist[i] - dist[i-1]
        if acc >= d_target[i]:
            new_d.append(dist[i]); acc = 0.0
    new_d.append(dist[-1])
    x_out = np.interp(new_d, dist, xd)
    y_out = np.interp(new_d, dist, yd)
    return np.column_stack([x_out, y_out])

# ───────────────── 일정 간격 함수 ─────────────
STATIC_SPACING = 1.0
def dist(p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def make_waypoints(xs, ys, spacing=STATIC_SPACING):
    if not xs: return np.empty((0,2))
    wpts, acc = [(xs[0], ys[0])], 0.0
    for i in range(1, len(xs)):
        seg = dist((xs[i-1], ys[i-1]), (xs[i], ys[i]))
        acc += seg
        while acc >= spacing:
            r = (seg - (acc-spacing)) / seg
            wpts.append((xs[i-1] + (xs[i]-xs[i-1])*r,
                         ys[i-1] + (ys[i]-ys[i-1])*r))
            acc -= spacing
    if wpts[-1] != (xs[-1], ys[-1]):
        wpts.append((xs[-1], ys[-1]))
    return np.array(wpts)

# ───────────────── Pure-Pursuit 기반 ──────────
def wp_pure_pursuit(xs, ys, v=None, k0=1.0, k1=0.3,
                    d_min=0.3, d_max=3.0):
    if len(xs) < 3: return np.column_stack([xs, ys])
    xs, ys = np.asarray(xs), np.asarray(ys)
    L = k0 + (k1*np.asarray(v[:len(xs)-1]) if v is not None and len(v)>=len(xs)-1
              else k0*np.ones(len(xs)-1))
    psi = np.arctan2(np.diff(ys), np.diff(xs))
    dpsi = (np.diff(psi) + np.pi) % (2*np.pi) - np.pi
    d_seg = 2 * L * np.sin(np.abs(dpsi)/2)
    d_seg = np.clip(d_seg, d_min, d_max)
    wpts, acc = [(xs[0], ys[0])], 0.0
    for i in range(1, len(xs)):
        seg = dist((xs[i-1], ys[i-1]), (xs[i], ys[i]))
        acc += seg
        target = d_seg[i-1] if i-1 < len(d_seg) else d_seg[-1]
        while acc >= target:
            r = (seg - (acc - target)) / seg
            wpts.append((xs[i-1] + (xs[i]-xs[i-1])*r,
                         ys[i-1] + (ys[i]-ys[i-1])*r))
            acc -= target
    if wpts[-1] != (xs[-1], ys[-1]):
        wpts.append((xs[-1], ys[-1]))
    return np.array(wpts)

# ───────────────── CSV 저장 ──────────────────
def save_csv(path, arr, header="X_m,Y_m"):
    np.savetxt(path, arr, delimiter=',',
               header=header, comments='', fmt='%.6f')

# ───────────────── 실시간 수집 루프 ──────────
xs_raw, ys_raw, xs_f, ys_f = [], [], [], []
sma_win, still_win = deque(maxlen=WIN_SMA), deque(maxlen=WIN_STILL)

plt.ion()
fig, ax = plt.subplots()
raw_line,  = ax.plot([], [], 'o', ms=2, alpha=0.3, label='Raw')
filt_line, = ax.plot([], [], '.-', lw=1.2, label='Filtered')
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_title("RTK-GPS 실시간 + PP Waypoints"); ax.legend()

ser = open_port()
print("GNGGA 수신 시작 – Ctrl+C 종료")
try:
    while True:
        if not ser or not ser.is_open:
            time.sleep(1); ser = open_port(); continue
        line = ser.readline().decode(errors='ignore').strip()
        if not line.startswith("$GNGGA"): continue
        lat_r, ns, lon_r, ew = line.split(',')[2:6]
        lat = nmea2deg(lat_r); lon = nmea2deg(lon_r)
        if ns == 'S': lat = -lat
        if ew == 'W': lon = -lon
        x, y = mercator_xy(lat, lon)

        sma_win.append((x, y))
        x_sma, y_sma = np.mean(sma_win, axis=0)

        still_win.append((x_sma, y_sma))
        if len(still_win) > 1:
            sw = np.array(still_win)
            dev = np.max(np.linalg.norm(sw - sw.mean(0), axis=1))
        else:
            dev = EPS_M + 1
        x_use, y_use = (sw.mean(0) if dev < EPS_M else (x_sma, y_sma))

        xs_raw.append(x); ys_raw.append(y)
        xs_f.append(x_use); ys_f.append(y_use)

        raw_line.set_data(xs_raw, ys_raw)
        filt_line.set_data(xs_f, ys_f)
        ax.relim(); ax.autoscale_view()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("\n종료 – CSV 저장 중…")

finally:
    if ser and ser.is_open: ser.close()
    plt.ioff(); plt.show()

    save_csv(LOG_FILE, np.column_stack([xs_raw, ys_raw]))
    save_csv(FLT_FILE, np.column_stack([xs_f,  ys_f]))
    save_csv(RAW_WPT_FILE, make_waypoints(xs_raw, ys_raw), header="X_m,Y_m")
    save_csv(WPT_STATIC,   make_waypoints(xs_f, ys_f),    header="X_m,Y_m")
    save_csv(WPT_CURV_FILE, resample_by_curvature(xs_f, ys_f), header="X_m,Y_m")
    save_csv(WPT_PP_FILE,  wp_pure_pursuit(xs_f, ys_f),   header="X_m,Y_m")

    print("CSV 저장 완료:")
    print("  Raw Track          →", LOG_FILE)
    print("  Filtered Track     →", FLT_FILE)
    print("  Raw WP (1 m)       →", RAW_WPT_FILE)
    print("  Static WP (1 m)    →", WPT_STATIC)
    print("  Curvature WP       →", WPT_CURV_FILE)
    print("  Pure-Pursuit WP    →", WPT_PP_FILE)
