# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# RTK-GPS 실시간 수집 + 이동평균·정지 탐지 + CSV 저장 + 실시간 시각화
# © 2025-07-08
# """
# import serial, time, math
# from collections import deque
# import numpy as np
# import matplotlib.pyplot as plt

# # ── 기본 설정 ─────────────────────────────
# PORT, BAUD, TIMEOUT = "COM7", 115200, 0.1
# R_WGS84 = 6_378_137

# # 이동평균·정지 탐지 파라미터
# WIN_SMA   = 20
# WIN_STILL = 30
# EPS_M     = 0.03

# # ── CSV 저장 함수 ─────────────────────────
# def save_csv(fname: str, arr: np.ndarray):
#     """2D 좌표 배열 → CSV(X_m,Y_m) 저장"""
#     np.savetxt(fname, arr, delimiter=',',
#                header='X_m,Y_m', comments='', fmt='%.6f')

# # ── NMEA 보조 함수 ────────────────────────
# def nmea2deg(v):
#     try: v = float(v)
#     except: return float('nan')
#     d = int(v // 100); m = v - d * 100
#     return d + m / 60

# def mercator_xy(lat_deg, lon_deg):
#     x = R_WGS84 * math.radians(lon_deg)
#     y = R_WGS84 * math.log(math.tan(math.radians(90 + lat_deg) / 2))
#     return x, y

# # ── 실시간 수집·시각화 ────────────────────
# xs_raw, ys_raw, xs_f, ys_f = [], [], [], []
# sma_win   = deque(maxlen=WIN_SMA)
# still_win = deque(maxlen=WIN_STILL)

# plt.ion()
# fig, ax = plt.subplots()
# line_raw,  = ax.plot([], [], 'o', ms=2, alpha=0.3, label="Raw")
# line_filt, = ax.plot([], [], '.-', lw=1.2, label="Filtered")
# ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
# ax.set_title("RTK-GPS 이동평균·정지 탐지 효과"); ax.legend()

# def open_port():
#     import serial
#     try: return serial.Serial(PORT, BAUD, timeout=TIMEOUT)
#     except Exception as e:
#         print("포트 오류:", e); return None

# ser = open_port()
# print("GNGGA 수신 시작 – Ctrl+C 로 종료")
# try:
#     while True:
#         if not ser or not ser.is_open:
#             time.sleep(1); ser = open_port(); continue
#         raw = ser.readline().decode(errors="ignore").strip()
#         if not raw.startswith("$GNGGA"): continue
#         p = raw.split(","); lat_r, ns, lon_r, ew = p[2:6]
#         lat = nmea2deg(lat_r); lon = nmea2deg(lon_r)
#         if ns == "S": lat = -lat
#         if ew == "W": lon = -lon
#         x, y = mercator_xy(lat, lon)

#         # ─ 이동평균 ─
#         sma_win.append((x, y))
#         x_sma, y_sma = np.mean(sma_win, axis=0)

#         # ─ 정지 탐지 ─
#         still_win.append((x_sma, y_sma))
#         if len(still_win) > 1:
#             sw = np.array(still_win)
#             max_dev = np.max(np.linalg.norm(sw - sw.mean(0), axis=1))
#         else:
#             max_dev = EPS_M + 1

#         x_use, y_use = (sw.mean(0) if max_dev < EPS_M else (x_sma, y_sma))

#         # ─ 버퍼 및 그래프 갱신 ─
#         xs_raw.append(x);  ys_raw.append(y)
#         xs_f.append(x_use); ys_f.append(y_use)

#         line_raw.set_data(xs_raw, ys_raw)
#         line_filt.set_data(xs_f, ys_f)
#         ax.relim(); ax.autoscale_view()
#         plt.pause(0.01)

# except KeyboardInterrupt:
#     print("\n수집 종료 – CSV 저장 중…")
# finally:
#     if ser and ser.is_open: ser.close()
#     plt.ioff(); plt.show()

#     # ─ CSV 저장 ─
#     save_csv("raw_track_35.csv",  np.column_stack([xs_raw, ys_raw]))
#     save_csv("filtered_track35.csv", np.column_stack([xs_f,  ys_f]))
#     print("CSV 저장 완료: raw_track.csv / filtered_track.csv")

# # 가만히 서서 1분 동안 측정 5번 하기








#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# RTK‑GPS 실시간 수집 + 이동평균·정지 탐지 + 번호 라벨 + CSV 저장
# 2025‑07‑08
# """
# import os, serial, time, math,  serial , sys
# from collections import deque
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import splprep, splev     # 곡선 보간·재샘플링 에 사용 (SciPy 설치 필요)

# # ── 경로 설정 ──────────────────────────────
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 현재 스크립트 위치
# LOG_FILE = os.path.join(BASE_DIR, "로우 데이터20_3.csv")     # 원시 좌표
# FLT_FILE = os.path.join(BASE_DIR, "이동 필터 데이터20_3.csv")# 필터 좌표
# WPT_FILE = os.path.join(BASE_DIR, "필터 웨이포인트.csv")# (미사용·예비)

# # ── 수집·필터 파라미터 ────────────────────
# PORT, BAUD, TIMEOUT = "COM10", 115200, 0.1 # COM7 이 원래꺼
# R_WGS84   = 6_378_137
# # 이동 평균 윈도우 크기
# WIN_SMA   = 20
# #정지 탐지 윈도우 크기
# WIN_STILL = 30
# #허용 편차(미터)
# EPS_M     = 0.03

# # 간격 파라미터
# D_STRAIGHT, D_MID, D_CURVE = 1, 0.40, 0.28 #0.55, 0.40, 0.28    # 웨이포인트 목표 간격 – 직선 0.55 m, 완만곡 0.40 m, 급곡 0.28 m
# K_TH1, K_TH2 = 0.10, 0.20                        # 곡률 임계값(1/m) – κ < 0.10: 직선, 0.10 ≤ κ < 0.20: 완만곡, κ ≥ 0.20: 급곡



# # ── 유틸리티 ──────────────────────────────

# # np.gradient로 1차·2차 도함수를 구해 
# # 곡률 κ = |(x′y″ − y′x″)/(x′²+y′²)^(3/2)| 을 계산
# def curvature(x, y):                             # 중앙차분 1·2차 미분으로 곡률계산
#     x, y = np.asarray(x), np.asarray(y)
#     dx  = np.gradient(x);  dy  = np.gradient(y)
#     ddx = np.gradient(dx); ddy = np.gradient(dy)
#     num = dx*ddy - dy*ddx
#     den = (dx*dx + dy*dy)**1.5
#     kappa = np.zeros_like(x); kappa[:] = np.nan
#     mask = den > 1e-6
#     kappa[mask] = np.abs(num[mask]/den[mask])
#     return kappa
# # 곡률 기반 리샘플링 (후처리)
# def resample_by_curvature(x, y):
#     # ─ 0) 배열화 & 필터 ──────────────────────
#     pts = np.vstack([x, y]).T
#     pts = pts[~np.isnan(pts).any(axis=1)]   # NaN 제거
#     pts = np.unique(pts, axis=0)            # 완전 중복 제거
#     if len(pts) < 4:        # 포인트 너무 적으면 그대로 반환
#         return pts
#     x, y = pts[:,0], pts[:,1]

#     # ─ 1) 스플라인 (점수에 맞춰 k 자동 설정) ─
#     k_spline = min(3, len(x) - 1)           # 3 → 2 → 1 자동 감소
#     tck, u = splprep([x, y], s=0, k=k_spline, per=False)

#     # ─ 2) 고해상도 샘플 & 곡률 계산 ─
#     u_dense = np.linspace(0, 1, max(2*len(x), 500))
#     x_d, y_d = splev(u_dense, tck)
#     kappa = curvature(x_d, y_d)

#     # ─ 3) 구간별 목표 간격 배열 ──────────────
#     d_target = np.where(kappa < K_TH1, D_STRAIGHT,
#                  np.where(kappa < K_TH2, D_MID, D_CURVE))

#     # ─ 4) 누적 길이 기반 리샘플 ─────────────
#     dist = np.hstack(([0], np.cumsum(np.hypot(np.diff(x_d), np.diff(y_d)))))
#     new_pts, acc = [0.0], 0.0
#     for i in range(1, len(dist)):
#         acc += dist[i] - dist[i-1]
#         if acc >= d_target[i]:
#             new_pts.append(dist[i]); acc = 0.0
#     new_pts.append(dist[-1])
#     x_out = np.interp(new_pts, dist, x_d)
#     y_out = np.interp(new_pts, dist, y_d)
#     return np.vstack([x_out, y_out]).T



# #2D 배열을 X_m,Y_m 헤더로 CSV 저장
# def save_csv(fname, arr):
#     np.savetxt(fname, arr, delimiter=',',
#                header='X_m,Y_m', comments='', fmt='%.6f')  # :contentReference[oaicite:3]{index=3}
# #NMEA 형식(ddmm.mmmm) 문자열을 십진도(°)로 변환
# def nmea2deg(v):
#     try: v = float(v)
#     except: return float('nan')
#     d = int(v // 100); m = v - d * 100
#     return d + m / 60
# #위경도를 Web Mercator 좌표계의 X, Y (미터)로 변환
# def mercator_xy(lat, lon):
#     x = R_WGS84 * math.radians(lon)
#     y = R_WGS84 * math.log(math.tan(math.radians(90 + lat) / 2))
#     return x, y

# # ── 실시간 플롯 준비 ───────────────────────
# xs_raw, ys_raw, xs_f, ys_f = [], [], [], []
# sma_win = deque(maxlen=WIN_SMA)
# still_win = deque(maxlen=WIN_STILL)

# plt.ion()
# fig, ax = plt.subplots()
# # 왼쪽 상단 raw filterd 설명
# line_raw,  = ax.plot([], [], 'o', ms=2, alpha=0.3, label="Raw")
# line_filt, = ax.plot([], [], '.-', lw=1.2, label="Filtered")
# # 표시
# ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
# ax.set_title("RTK‑GPS 이동평균·정지 탐지 효과")
# # 그리기
# ax.legend()

# #포인트에 인덱스 번호를 레이블로 붙이기
# def annotate_point(x, y, idx):
#     ax.annotate(str(idx), xy=(x, y), xytext=(3, 3),
#                 textcoords="offset points", fontsize=7,
#                 color='gray')
# #시리얼 포트 오픈, 실패 시 None 반환
# def open_port():
#     try: return serial.Serial(PORT, BAUD, timeout=TIMEOUT)
#     except Exception as e:
#         print("포트 오류:", e); return None
    



# ## 추가
# # ──────────────────────────────────────────────────────────
# # ①  두 점 사이 거리 계산용 함수
# def dist(p1, p2):
#     return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# # ②  일정 거리 간격으로 웨이포인트 생성
# def make_waypoints(xs, ys, spacing_m=1.0):
#     """필터 좌표 → 일정 간격 웨이포인트 리스트[(x,y)…]"""
#     if not xs:     # 좌표가 없으면 빈 리스트 반환
#         return []
#     wpts = [(xs[0], ys[0])]            # 첫 점은 무조건 포함
#     acc   = 0.0                        # 지난 세그먼트 누적 거리
#     for i in range(1, len(xs)):
#         seg_d = dist((xs[i-1], ys[i-1]), (xs[i], ys[i]))
#         acc  += seg_d
#         # 누적 거리가 간격 이상이면 웨이포인트 채택
#         while acc >= spacing_m:
#             # 선형 보간으로 정확히 spacing_m 떨어진 점 계산
#             ratio = (seg_d - (acc - spacing_m)) / seg_d
#             x_wp  = xs[i-1] + (xs[i]-xs[i-1]) * ratio
#             y_wp  = ys[i-1] + (ys[i]-ys[i-1]) * ratio
#             wpts.append((x_wp, y_wp))
#             acc -= spacing_m
#     if wpts[-1] != (xs[-1], ys[-1]):   # 마지막 점이 포함 안 됐으면 추가
#         wpts.append((xs[-1], ys[-1]))
#     return wpts

# # ③  웨이포인트 저장 함수
# def save_waypoints_csv(fname, wpts):
#     arr = np.column_stack([
#         np.arange(1, len(wpts)+1),              # WP_ID
#         np.array(wpts)                          # X_m, Y_m
#     ])
#     np.savetxt(fname, arr, delimiter=',',
#                header='WP_ID,X_m,Y_m', comments='', fmt=['%d','%.6f','%.6f'])    
# # ─────────────────────────────────────────────────────────────────────
# # ── 추가 : 타이머 시작 ───────────────────────────
# start_time = time.time()
# # ── 데이터 수집 루프 ───────────────────────
# ser = open_port()
# print("GNGGA 수신 시작 – Ctrl+C 종료")
# try:
#     while True:
#         # ── 추가 : 경과 시간 계산 및 표시───────────────────────
#         # elapsed = time.time() - start_time
#         # ax.set_title(f"RTK-GPS 이동평균·정지 탐지 효과 ({elapsed:.1f}s)")
#         # if elapsed >= 60.0:
#         #     print("1분(60초) 경과 – 데이터 저장 및 종료합니다.")
#         #     break
#         # ─────────────────────────

#         if not ser or not ser.is_open:
#             time.sleep(1); ser = open_port(); continue
        
#         raw = ser.readline().decode(errors="ignore").strip()
#         if not raw.startswith("$GNGGA"): continue
        
#         lat_r, ns, lon_r, ew = raw.split(",")[2:6]
#         lat, lon = nmea2deg(lat_r), nmea2deg(lon_r)
#         if ns == "S": lat = -lat
#         if ew == "W": lon = -lon
#         x, y = mercator_xy(lat, lon)

#         # 이동평균
#         sma_win.append((x, y))
#         x_sma, y_sma = np.mean(sma_win, axis=0)

#         # 정지 탐지
#         still_win.append((x_sma, y_sma))
#         if len(still_win) > 1:
#             sw = np.array(still_win)
#             dev = np.max(np.linalg.norm(sw - sw.mean(0), axis=1))
#         else:
#             dev = EPS_M + 1
#         x_use, y_use = (sw.mean(0) if dev < EPS_M else (x_sma, y_sma))

#         # 버퍼·플롯 갱신
#         idx = len(xs_raw) + 1
#         xs_raw.append(x);  ys_raw.append(y)
#         xs_f.append(x_use); ys_f.append(y_use)

#         line_raw.set_data(xs_raw, ys_raw)
#         line_filt.set_data(xs_f, ys_f)
#         annotate_point(x, y, idx)          # ← Raw 포인트 번호 매기기
#         ax.relim(); ax.autoscale_view()
#         plt.pause(0.01)

# except KeyboardInterrupt:
#     print("\n수집 종료 – CSV 저장 중…")
# finally:
#     if ser and ser.is_open: ser.close()
#     plt.ioff(); plt.show()

#     save_csv(LOG_FILE, np.column_stack([xs_raw, ys_raw]))
#     save_csv(FLT_FILE, np.column_stack([xs_f,  ys_f]))
#     print(f"CSV 저장 완료:\n  Raw → {LOG_FILE}\n  Filt → {FLT_FILE}")
#     # ─ 웨이포인트 생성 & 저장 ─
#     waypoints = make_waypoints(xs_f, ys_f, spacing_m=1.0)  # 1 m 간격 (필요 시 변경)
#     save_waypoints_csv(WPT_FILE, waypoints)
#     print(f"Waypoint CSV 저장 완료 → {WPT_FILE}")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTK‑GPS 실시간 수집 + 이동평균·정지 탐지 + 번호 라벨 + CSV 저장 + 웨이포인트 (일정 간격 및 곡률 기반)
2025‑07‑08
"""
import os
import sys
import serial
import time
import math
import csv
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# ── 경로 설정 ───────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
LOG_FILE      = os.path.join(BASE_DIR, "로우 데이터2.csv")
FLT_FILE      = os.path.join(BASE_DIR, "이동 필터 데이터2.csv")
# ── 추가 : 원시(raw) 웨이포인트 파일 경로 ───────────────────────────
RAW_WPT_FILE = os.path.join(BASE_DIR, "로우_웨이포인트2.csv")
WPT_FILE      = os.path.join(BASE_DIR, "필터 웨이포인트_정적2.csv")
WPT_CURV_FILE = os.path.join(BASE_DIR, "필터 웨이포인트_곡률2.csv")

# ── 시리얼 & 지구 상수 ────────────────────
PORT, BAUD, TIMEOUT = "COM10", 115200, 0.1
R_WGS84             = 6_378_137

# ── 필터 파라미터 ─────────────────────────
WIN_SMA, WIN_STILL, EPS_M = 20, 30, 0.03

# ── 웨이포인트 파라미터 ──────────────────
# 일정 간격
STATIC_SPACING = 1.0  # m
# 곡률 기반
D_STRAIGHT, D_MID, D_CURVE = 1.0, 0.40, 0.28
K_TH1, K_TH2             = 0.10, 0.20

# ── NMEA → 십진도 도우미 ─────────────────
def nmea2deg(v):
    try:    v = float(v)
    except: return float('nan')
    d = int(v // 100)
    m = v - d * 100
    return d + m / 60

# 위경도 → Web Mercator
def mercator_xy(lat, lon):
    x = R_WGS84 * math.radians(lon)
    y = R_WGS84 * math.log(math.tan(math.radians(90 + lat) / 2))
    return x, y

# ── 이동평균·정지 탐지용 함수 ───────────────
def open_port():
    try:
        return serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    except Exception as e:
        print("포트 오류:", e)
        return None

# ── 곡률 계산 ──────────────────────────────
def curvature(x, y):
    x, y   = np.asarray(x), np.asarray(y)
    dx, dy = np.gradient(x), np.gradient(y)
    ddx    = np.gradient(dx)
    ddy    = np.gradient(dy)
    num    = dx * ddy - dy * ddx
    den    = (dx*dx + dy*dy)**1.5
    κ      = np.zeros_like(x)
    mask   = den > 1e-6
    κ[mask] = np.abs(num[mask] / den[mask])
    return κ

# ── 곡률 기반 리샘플링 ─────────────────────
def resample_by_curvature(xs, ys):
    pts = np.vstack([xs, ys]).T
    pts = pts[~np.isnan(pts).any(axis=1)]
    pts = np.unique(pts, axis=0)
    if len(pts) < 4:
        return pts
    x, y       = pts[:,0], pts[:,1]
    k_spline   = min(3, len(x) - 1)
    tck, _     = splprep([x, y], s=0, k=k_spline)
    u_dense    = np.linspace(0, 1, max(2*len(x), 500))
    x_d, y_d   = splev(u_dense, tck)
    κ          = curvature(x_d, y_d)
    d_target   = np.where(κ < K_TH1, D_STRAIGHT,
                  np.where(κ < K_TH2, D_MID, D_CURVE))
    dist_acc   = np.hstack(([0], np.cumsum(np.hypot(np.diff(x_d), np.diff(y_d)))))
    new_d, acc = [0.0], 0.0
    for i in range(1, len(dist_acc)):
        seg = dist_acc[i] - dist_acc[i-1]
        acc += seg
        if acc >= d_target[i]:
            new_d.append(dist_acc[i])
            acc = 0.0
    new_d.append(dist_acc[-1])
    x_out = np.interp(new_d, dist_acc, x_d)
    y_out = np.interp(new_d, dist_acc, y_d)
    return np.vstack([x_out, y_out]).T

# ── 일정 간격 리샘플링 ─────────────────────
def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def make_waypoints(xs, ys, spacing_m=STATIC_SPACING):
    if not xs:
        return []
    wpts = [(xs[0], ys[0])]
    acc  = 0.0
    for i in range(1, len(xs)):
        seg_d = dist((xs[i-1], ys[i-1]), (xs[i], ys[i]))
        acc  += seg_d
        while acc >= spacing_m:
            ratio = (seg_d - (acc - spacing_m)) / seg_d
            x_wp  = xs[i-1] + (xs[i]-xs[i-1]) * ratio
            y_wp  = ys[i-1] + (ys[i]-ys[i-1]) * ratio
            wpts.append((x_wp, y_wp))
            acc -= spacing_m
    if wpts[-1] != (xs[-1], ys[-1]):
        wpts.append((xs[-1], ys[-1]))
    return wpts

# ── CSV 저장 함수 ───────────────────────────
def save_csv(fname, arr, header=["X_m","Y_m"]):
    np.savetxt(fname, arr, delimiter=',', header=','.join(header), comments='', fmt='%.6f')

def save_waypoints_csv(fname, wpts):
    arr = np.column_stack([np.arange(1, len(wpts)+1), np.array(wpts)])
    np.savetxt(fname, arr, delimiter=',', header='WP_ID,X_m,Y_m', comments='', fmt=['%d','%.6f','%.6f'])

# ── 실시간 수집 & 필터링 ──────────────────
xs_raw, ys_raw, xs_f, ys_f = [], [], [], []
sma_win   = deque(maxlen=WIN_SMA)
still_win = deque(maxlen=WIN_STILL)

plt.ion()
fig, ax = plt.subplots()
raw_line , = ax.plot([], [], 'o', ms=2, alpha=0.3, label='Raw')
filt_line, = ax.plot([], [], '.-', lw=1.2, label='Filtered')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('RTK‑GPS 이동평균·정지 탐지')
ax.legend()

def annotate_point(x, y, idx):
    ax.annotate(str(idx), xy=(x, y), xytext=(3, 3), textcoords='offset points', fontsize=7, color='gray')

ser = open_port()
print('GNGGA 수신 시작 – Ctrl+C 종료')
try:
    while True:
        if not ser or not ser.is_open:
            time.sleep(1)
            ser = open_port()
            continue
        raw = ser.readline().decode(errors='ignore').strip()
        if not raw.startswith('$GNGGA'):
            continue
        parts    = raw.split(',')
        lat_r, ns, lon_r, ew = parts[2:6]
        lat = nmea2deg(lat_r)
        lon = nmea2deg(lon_r)
        if ns == 'S': lat = -lat
        if ew == 'W': lon = -lon
        x, y = mercator_xy(lat, lon)

        sma_win.append((x, y))
        x_sma, y_sma = np.mean(sma_win, axis=0)

        still_win.append((x_sma, y_sma))
        if len(still_win) > 1:
            sw  = np.array(still_win)
            dev = np.max(np.linalg.norm(sw - sw.mean(0), axis=1))
        else:
            dev = EPS_M + 1
        x_use, y_use = (sw.mean(0) if dev < EPS_M else (x_sma, y_sma))

        idx = len(xs_raw) + 1
        xs_raw.append(x)
        ys_raw.append(y)
        xs_f.append(x_use)
        ys_f.append(y_use)

        raw_line.set_data(xs_raw, ys_raw)
        filt_line.set_data(xs_f, ys_f)
        annotate_point(x, y, idx)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

except KeyboardInterrupt:
    print('\n수집 종료, CSV 저장 중…')

finally:
    if ser and ser.is_open:
        ser.close()
    plt.ioff()
    plt.show()

    # ① 원본(raw) / 필터(filtered) 저장
    save_csv(LOG_FILE, np.column_stack([xs_raw, ys_raw]))
    save_csv(FLT_FILE, np.column_stack([xs_f,  ys_f]))

    # ② 원시(raw) 일정 간격 웨이포인트
    raw_wpts = make_waypoints(xs_raw, ys_raw, spacing_m=STATIC_SPACING)
    save_waypoints_csv(RAW_WPT_FILE, raw_wpts)
    print(f"원시 웨이포인트 {len(raw_wpts)}개 저장 → {RAW_WPT_FILE}")

    # ③ 필터된(정적) 웨이포인트
    static_wpts = make_waypoints(xs_f, ys_f, spacing_m=STATIC_SPACING)
    save_waypoints_csv(WPT_FILE, static_wpts)
    print(f"정적 웨이포인트 {len(static_wpts)}개 저장 → {WPT_FILE}")

    # ④ 곡률 기반 웨이포인트
    curv_wpts = resample_by_curvature(xs_f, ys_f)
    save_csv(WPT_CURV_FILE, curv_wpts)
    print(f"곡률 기반 웨이포인트 {len(curv_wpts)}개 저장 → {WPT_CURV_FILE}")
