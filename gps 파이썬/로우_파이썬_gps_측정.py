#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTK-GPS 실시간 로우 데이터 수집 및 시각화
  - NMEA GNGGA 메시지로부터 위도·경도 파싱
  - Web Mercator (X, Y) 계산
  - 실시간 플롯만 수행
  - finally에서 한 번에 CSV 저장 (파일명 중복 시 _1, _2 … 자동 인덱싱)
"""
import os
import serial
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# ── 중복 방지용 파일명 생성 함수 ─────────────────────────
def unique_filepath(dirpath: str, basename: str, ext: str = ".csv") -> str:
    """dirpath에 basename+ext가 있으면 basename_1.ext, basename_2.ext … 로 반환"""
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
ax.set_title("RTK-GPS 로우 데이터 트랙 (X, Y)")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.grid(True)
line, = ax.plot([], [], marker='.', linestyle='')
xs, ys = [], []

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
        # 포트가 닫히면 재시도
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

        # Web Mercator 변환
        x, y = mercator_xy(lat, lon)

        # 플롯 업데이트
        xs.append(x)
        ys.append(y)
        line.set_data(xs, ys)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

        # 콘솔 출력
        print(f"{time.strftime('%H:%M:%S')}  X={x:.2f} m, Y={y:.2f} m")

except KeyboardInterrupt:
    print("\n데이터 수집을 종료합니다.")

finally:
    # 포트 닫기
    if ser and ser.is_open:
        ser.close()

    # 최종 수집된 X, Y 배열을 numpy로 저장
    data = np.column_stack([xs, ys])
    save_csv(LOG_FILE, data)
    print(f"수집된 로우 데이터가 저장되었습니다 → {LOG_FILE}")

    plt.ioff()
    plt.show()
