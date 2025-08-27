#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latlon_logger_node.py — ROS1 (rospy) 경로 로깅 + 실시간 시각화
────────────────────────────────────────────────────────────
• /ublox/fix (sensor_msgs/NavSatFix) 구독
• CSV 실시간 append 저장 (헤더: Lat,Lon)
  경로: /home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/raw_track_latlon.csv
        (중복 시 _1, _2 … 자동 인덱싱)
• 시각화(plt.ion 루프): Mercator (X,Y)로 궤적 라인 + 현재 위치 + Info box
    - waypoint_tracker_topics.py 의 plt.ion 스타일만 가져와 적용
"""

import os
import csv
import math
import threading
from typing import List, Tuple

import rospy
from sensor_msgs.msg import NavSatFix

# ── Matplotlib 백엔드 설정 (Qt5Agg → 실패 시 TkAgg)
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ───────────────── 저장 경로/파일 (하드코딩 + 자동 인덱스) ─────────────────
SAVE_DIR   = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config"
BASE_NAME  = "raw_track_latlon"        # 예: raw_track_latlon.csv, raw_track_latlon_1.csv ...
FIX_TOPIC  = "/ublox/fix"
PLOT_HZ    = 10.0                      # 시각화 주기(Hz). 너무 높이면 GUI가 버벅일 수 있습니다.

def unique_filepath(dirpath: str, basename: str, ext: str = ".csv") -> str:
    os.makedirs(dirpath, exist_ok=True)
    candidate = os.path.join(dirpath, f"{basename}{ext}")
    if not os.path.exists(candidate):
        return candidate
    idx = 1
    while True:
        candidate = os.path.join(dirpath, f"{basename}_{idx}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1

# ───────────────── 좌표/유틸 ─────────────────
RADIUS_WGS84 = 6_378_137.0
def latlon_to_mercator_xy(lat: float, lon: float) -> Tuple[float, float]:
    x = RADIUS_WGS84 * math.radians(lon)
    y = RADIUS_WGS84 * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

# ───────────────── 노드 구현 ─────────────────
class LatLonLoggerNode:
    def __init__(self):
        # 토픽은 필요 시 파라미터로만 바꿀 수 있게(기본 하드코딩 유지)
        self.fix_topic = rospy.get_param("~fix_topic", FIX_TOPIC)

        # 저장 파일 생성(+헤더 보장)
        self.csv_path = unique_filepath(SAVE_DIR, BASE_NAME, ".csv")
        self._lock: threading.Lock = threading.Lock()
        self._ensure_header()

        # 데이터 버퍼(시각화용)
        self._lats: List[float] = []
        self._lons: List[float] = []
        self._xs:   List[float] = []
        self._ys:   List[float] = []

        # 시각화 초기화 (waypoint_tracker_topics.py의 ion 루프 스타일)
        self._init_plot()

        # 구독/종료 훅
        rospy.Subscriber(self.fix_topic, NavSatFix, self._cb_fix, queue_size=100)
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(f"[latlon_logger] subscribe: {self.fix_topic}")
        rospy.loginfo(f"[latlon_logger] CSV: {self.csv_path} (Lat,Lon)")

    # ── 파일 헤더 보장 ──
    def _ensure_header(self):
        need_header = (not os.path.exists(self.csv_path)) or (os.path.getsize(self.csv_path) == 0)
        if need_header:
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Lat", "Lon"])

    # ── 콜백: NavSatFix ──
    def _cb_fix(self, msg: NavSatFix):
        if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
            return

        lat = float(msg.latitude)
        lon = float(msg.longitude)

        # 1) 실시간 append 저장
        try:
            with self._lock, open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([lat, lon])
        except Exception as e:
            rospy.logwarn(f"[latlon_logger] CSV append 실패: {e}")

        # 2) 시각화 버퍼 업데이트
        x, y = latlon_to_mercator_xy(lat, lon)
        with self._lock:
            self._lats.append(lat); self._lons.append(lon)
            self._xs.append(x);     self._ys.append(y)

        # 진행 상태 로그(스로틀)
        rospy.loginfo_throttle(1.0, f"[latlon_logger] N={len(self._lats)}  Lat={lat:.7f}, Lon={lon:.7f}")

    # ── 시각화 초기화 ──
    def _init_plot(self):
        plt.ion()
        self.fig = plt.figure(figsize=(7.5, 7.5))
        self.ax  = self.fig.add_subplot(111)
        self.line_route, = self.ax.plot([], [], '-', c='0.6', lw=1.0, label='Route')
        self.pt_cur,      = self.ax.plot([], [], 'o', c='red', ms=6, label='Current')
        self.info_txt = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                     ha='left', va='top', fontsize=9,
                                     bbox=dict(fc='white', alpha=0.7))

        self.ax.set_title("ROS GPS Logger (Mercator X,Y)")
        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)')
        self.ax.grid(True, ls=':', alpha=0.5)
        self.ax.axis('equal')
        self.ax.legend(loc='upper right')
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    # ── 시각화 1스텝 업데이트 (waypoint_tracker_topics.py 스타일) ──
    def update_plot_once(self):
        with self._lock:
            xs = list(self._xs); ys = list(self._ys)
            lats = list(self._lats); lons = list(self._lons)

        self.ax.clear()

        # 궤적 라인
        if len(xs) >= 2:
            self.ax.plot(xs, ys, '-', c='0.6', lw=1.0, label='Route')

        # 현재 위치
        if xs and ys:
            self.ax.plot(xs[-1], ys[-1], 'o', c='red', ms=6, label='Current')

        # Info box (샘플 수, 최근 Lat/Lon)
        lines = [f"Samples: {len(xs)}"]
        if lats and lons:
            lines.append(f"Lat: {lats[-1]:.7f}")
            lines.append(f"Lon: {lons[-1]:.7f}")
        self.ax.text(0.02, 0.98, "\n".join(lines), transform=self.ax.transAxes,
                     ha='left', va='top', fontsize=9, bbox=dict(fc='white', alpha=0.7))

        # 스타일 재설정 (clear() 했으니 다시)
        self.ax.set_title("ROS GPS Logger (Mercator X,Y)")
        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)')
        self.ax.grid(True, ls=':', alpha=0.5)
        self.ax.axis('equal')
        self.ax.legend(loc='upper right')

        self.fig.canvas.draw_idle()

    # ── 종료 처리 ──
    def _on_shutdown(self):
        rospy.loginfo(f"[latlon_logger] 종료. 최종 파일: {self.csv_path}")
        try:
            plt.ioff()
            plt.show()
        except Exception:
            pass

def main():
    rospy.init_node("latlon_logger", anonymous=False)

    node = LatLonLoggerNode()

    # waypoint_tracker_topics.py 처럼 plt.ion 루프 사용
    rate_hz = max(1.0, float(rospy.get_param("~plot_hz", PLOT_HZ)))
    rate = rospy.Rate(rate_hz)
    try:
        while not rospy.is_shutdown():
            node.update_plot_once()
            plt.pause(0.001)   # GUI 이벤트 플러시
            rate.sleep()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
