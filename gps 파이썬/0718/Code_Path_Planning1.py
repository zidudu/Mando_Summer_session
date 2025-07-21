#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waypoint_and_live_pos_viewer.py
───────────────────────────────────────────────────────────
• CSV 웨이포인트 → 파란 점 표시
• NMEA $GNGGA 메시지로부터 위도·경도 파싱
• Web Mercator → (X, Y) 변환
• 실시간 내 위치(빨간 점) 업데이트
───────────────────────────────────────────────────────────
"""

import os
import math
import time
import serial
import numpy as np
import matplotlib.pyplot as plt

# ────── 사용자 설정 ──────────────────────────────────────
CSV_PATH   = "waypoints_dynamic_12.csv"   # 웨이포인트 CSV 파일
PORT, BAUD, TIMEOUT = "COM10", 115200, 0.1
POINT_SIZE = 12                            # 웨이포인트 점 크기
# ────────────────────────────────────────────────────────

# ────── NMEA ddmm.mmmm → 십진도 변환 ──────────────────────
def nmea2deg(val_str: str) -> float:
    try:
        v = float(val_str)
        deg = int(v // 100)
        minute = v - deg * 100
        return deg + minute / 60.0
    except:
        return float("nan")

# ────── 위/경도 → Web Mercator(X, Y) 변환 ────────────────
def mercator_xy(lat: float, lon: float):
    R = 6_378_137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.radians(90 + lat) / 2))
    return x, y

# ────── CSV 웨이포인트 로드 ───────────────────────────────
def load_waypoints(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] CSV 미존재: {path}")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    print(f"[INFO] 웨이포인트 {len(data)}개 로드 → {path}")
    return data[:,0], data[:,1]

# ────── 시리얼 포트 열기 ─────────────────────────────────
def open_serial():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        print(f"[INFO] Serial opened → {PORT}")
        return ser
    except Exception as e:
        print("[ERROR] Serial open failed:", e)
        return None

# ────── 메인 ────────────────────────────────────────────
def main():
    # 1) 웨이포인트 읽기
    wp_x, wp_y = load_waypoints(CSV_PATH)

    # 2) 플롯 준비
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.grid(True, ls="--", alpha=0.6)
    ax.set_title("Waypoints (blue) & Live Position (red)")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.scatter(wp_x, wp_y, s=POINT_SIZE, c="blue", label="Waypoints")
    live_pt, = ax.plot([], [], "ro", ms=8, label="Live Position", zorder=5)
    ax.legend(loc="upper right")

    # 3) 시리얼 연결
    ser = open_serial()
    if ser is None:
        return

    print("GNGGA 수신 대기 중... (Ctrl+C 로 종료)")
    xs, ys = [], []

    try:
        while True:
            raw = ser.readline().decode("ascii", errors="ignore").strip()
            # GNGGA만 선택
            if not raw.startswith("$GNGGA"):
                continue
            parts = raw.split(",")
            if len(parts) < 6:
                continue

            # 위도·경도 파싱 (참고 코드 그대로)
            lat = nmea2deg(parts[2])
            lon = nmea2deg(parts[4])
            if parts[3] == "S": lat = -lat
            if parts[5] == "W": lon = -lon

            # Web Mercator 변환
            x, y = mercator_xy(lat, lon)
            xs.append(x); ys.append(y)

            # 실시간 표시
            live_pt.set_data(x, y)
            ax.relim(); ax.autoscale_view()
            plt.pause(0.01)

            # (선택) 콘솔 출력
            print(f"{time.strftime('%H:%M:%S')}  X={x:.2f} m, Y={y:.2f} m")

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중지 요청")

    finally:
        if ser and ser.is_open:
            ser.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
