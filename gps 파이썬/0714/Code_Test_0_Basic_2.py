#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTK-GPS 실시간 로우 데이터 수집 및 시각화
  - NMEA GNGGA 메시지로부터 위도·경도 파싱
  - Web Mercator (X, Y) 계산 → 실시간 플롯
  - 종료 시 CSV 저장 (Lat, Lon 저장, left_lane.csv와 같은 형식)
"""
import os
import serial
import time
import math
import numpy as np
import matplotlib.pyplot as plt

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
def save_csv(fname: str, arr: np.ndarray, header=["Lat","Lon"]):
    np.savetxt(fname,
               arr,
               delimiter=',',
               header=','.join(header),
               comments='',
               fmt='%.8f')

# ── 설정 값 ─────────────────────────────────────────────
PORT, BAUD, TIMEOUT = "COM10", 115200, 0.1
RADIUS_WGS84       = 6_378_137.0     # 지구 반지름 (m)

# ── 로그 파일 경로 ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = unique_filepath(BASE_DIR, "raw_track_latlon")

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
ax.set_title("RTK-GPS 로우 데이터 트랙 (Mercator X, Y)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.grid(True)
line, = ax.plot([], [], marker='.', linestyle='')
xs, ys = [], []         # Mercator 좌표 (플롯용)
lats, lons = [], []     # 위경도 (저장용)

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

        # 위도·경도 파싱
        lat = nmea2deg(parts[2])
        lon = nmea2deg(parts[4])
        if parts[3] == "S": lat = -lat
        if parts[5] == "W": lon = -lon

        # 저장용 (Lat,Lon)
        lats.append(lat)
        lons.append(lon)

        # 플롯용 (Mercator 변환)
        x, y = mercator_xy(lat, lon)
        xs.append(x)
        ys.append(y)

        # 플롯 업데이트
        line.set_data(xs, ys)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

        # 콘솔 출력
        print(f"{time.strftime('%H:%M:%S')}  Lat={lat:.8f}, Lon={lon:.8f}")

except KeyboardInterrupt:
    print("\n데이터 수집을 종료합니다.")

finally:
    if ser and ser.is_open:
        ser.close()

    # 최종 수집된 Lat, Lon 배열을 numpy로 저장
    data = np.column_stack([lats, lons])
    save_csv(LOG_FILE, data, header=["Lat","Lon"])
    print(f"수집된 위경도 데이터가 저장되었습니다 → {LOG_FILE}")

    plt.ioff()
    plt.show()
