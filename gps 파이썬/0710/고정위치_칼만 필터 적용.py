#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTK-GPS 실시간 로우 데이터 수집 및 시각화 + 칼만 필터 스무딩
  - NMEA GNGGA 메시지로부터 위도·경도 파싱
  - Web Mercator (X, Y) 계산
  - 칼만 필터 적용 (상태: [x, y, vx, vy], 관측: [x, y])
  - 실시간 플롯
  - finally에서 한 번에 CSV 저장 (파일명 중복 시 _1, _2 … 자동 인덱싱)
"""
import os
import serial
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# ── 중복 방지용 파일명 생성 함수 ─────────────────────────
def unique_filepath(dirpath: str, basename: str, ext: str = ".csv") -> str:
    candidate = os.path.join(dirpath, f"{basename}{ext}")
    if not os.path.exists(candidate):
        return candidate
    idx = 1
    while True:
        candidate = os.path.join(dirpath, f"{basename}_{idx}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1

# ── numpy 기반 CSV 저장 함수 ────────────────────────────
def save_csv(fname: str, arr: np.ndarray, header=["X_m","Y_m"]):
    np.savetxt(fname,
               arr,
               delimiter=',',
               header=','.join(header),
               comments='',
               fmt='%.6f')

# ── 설정 값 ─────────────────────────────────────────────
PORT, BAUD, TIMEOUT = "COM10", 115200, 0.1
RADIUS_WGS84       = 6_378_137.0     # 지구 반지름 (m)

# ── 로그 파일 경로 (중복 시 _1, _2 … 자동 인덱싱) ──────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = unique_filepath(BASE_DIR, "raw_track_xy")

# ── NMEA ddmm.mmmm → 십진도 변환 함수 ─────────────────────
def nmea2deg(val_str: str) -> float:
    try:
        v = float(val_str)
        deg = int(v // 100)
        minute = v - deg * 100
        return deg + minute / 60.0
    except:
        return float('nan')

# ── 위/경도 → Web Mercator(X, Y) 변환 함수 ────────────────
def mercator_xy(lat: float, lon: float):
    x = RADIUS_WGS84 * math.radians(lon)
    y = RADIUS_WGS84 * math.log(math.tan(math.radians(90 + lat) / 2))
    return x, y

# ── 실시간 플롯 초기화 ───────────────────────────────────
plt.ion()
fig, ax = plt.subplots()
ax.set_title("RTK-GPS Raw Data Track (X, Y) with Kalman Smoothing")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.grid(True)
line_raw, = ax.plot([], [], marker='.', linestyle='', alpha=0.3, label="Raw")
line_filt, = ax.plot([], [], marker='.', linestyle='-', label="Kalman")
ax.legend()
xs_raw, ys_raw = [], []
xs_filt, ys_filt = [], []

# ── 칼만 필터 초기화 ───────────────────────────────────
dt = 1.0  # 예상 측정 주기(초)
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1,  0],
                 [0, 0, 0,  1]])
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])
# 초기 상태는 나중에 첫 측정으로 설정
kf.x = np.array([0., 0., 0., 0.])
kf.P *= 10.0
q = 0.1
kf.Q = q * np.eye(4)
r = 0.5
kf.R = r * np.eye(2)

# ── 시리얼 포트 열기 ────────────────────────────────────
def open_serial():
    try:
        s = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        print(f"{PORT} 포트가 열렸습니다. 로그 파일 → {LOG_FILE}")
        return s
    except Exception as e:
        print("시리얼 포트 열기 실패:", e)
        return None

ser = open_serial()
print("GNGGA 수신 대기 중... (Ctrl+C 로 종료)")

initialized = False

try:
    while True:
        if ser is None or not ser.is_open:
            time.sleep(1)
            ser = open_serial()
            continue

        raw = ser.readline().decode("ascii", errors="ignore").strip()
        if not raw.startswith("$GNGGA"):
            continue
        parts = raw.split(",")
        if len(parts) < 6:
            continue

        lat = nmea2deg(parts[2])
        lon = nmea2deg(parts[4])
        if parts[3] == "S": lat = -lat
        if parts[5] == "W": lon = -lon

        x, y = mercator_xy(lat, lon)

        # 첫 측정 시 칼만 필터 초기 상태 설정
        if not initialized:
            kf.x[:2] = np.array([x, y])
            initialized = True

        # 원본 데이터 저장/플롯
        xs_raw.append(x)
        ys_raw.append(y)

        # 칼만 필터 예측 및 갱신
        kf.predict()
        kf.update(np.array([x, y]))
        x_f, y_f = kf.x[0], kf.x[1]
        xs_filt.append(x_f)
        ys_filt.append(y_f)

        # 플롯 갱신
        line_raw.set_data(xs_raw, ys_raw)
        line_filt.set_data(xs_filt, ys_filt)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

        # 콘솔 출력
        print(f"{time.strftime('%H:%M:%S')}  Raw X={x:.2f}, Y={y:.2f}  |  KF X={x_f:.2f}, Y={y_f:.2f}")

except KeyboardInterrupt:
    print("\n데이터 수집을 종료합니다.")

finally:
    if ser and ser.is_open:
        ser.close()

    # 최종 수집된 raw+filt 데이터를 각각 CSV 저장
    raw_arr  = np.column_stack([xs_raw, ys_raw])
    filt_arr = np.column_stack([xs_filt, ys_filt])
    save_csv(unique_filepath(BASE_DIR, "raw_track_xy"), raw_arr,  header=["X_raw","Y_raw"])
    save_csv(unique_filepath(BASE_DIR, "kf_track_xy"),  filt_arr, header=["X_filt","Y_filt"])
    print(f"원본 데이터 저장 → raw_track_xy.csv, 필터링 데이터 저장 → kf_track_xy.csv")

    plt.ioff()
    plt.show()
