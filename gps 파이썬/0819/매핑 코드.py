#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latlon_logger_node.py
────────────────────────────────────────────────────────────
ROS1에서 /ublox/fix (sensor_msgs/NavSatFix) 구독
→ (Lat, Lon) CSV 저장 (중복 시 _1, _2 … 자동 인덱싱)
→ 옵션: 실시간 시각화 (ENU 좌표)
────────────────────────────────────────────────────────────
"""

import os
import csv
import math
import time
import threading
from typing import Optional, Tuple, List

import rospy
from sensor_msgs.msg import NavSatFix

# ───────── 파일명 유틸 ─────────
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

# ───────── 좌표 유틸 (ENU용) ─────────
_R = 6_378_137.0
def _deg2rad(x: float) -> float: return x * math.pi / 180.0
def lla2enu(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    dlat = _deg2rad(lat - lat0)
    dlon = _deg2rad(lon - lon0)
    lat0r = _deg2rad(lat0)
    x = _R * dlon * math.cos(lat0r)
    y = _R * dlat
    return x, y

class LatLonLoggerNode:
    def __init__(self):
        self.fix_topic     = rospy.get_param('~fix_topic', '/ublox/fix')
        self.save_dir      = os.path.expanduser(rospy.get_param('~save_dir', '~/.ros/latlon_logs'))
        self.base_name     = rospy.get_param('~base_name', 'raw_track_latlon')
        self.viz_enable    = bool(rospy.get_param('~viz_enable', False))
        self.viz_backend   = rospy.get_param('~viz_backend', 'TkAgg')

        # CSV 준비
        self.csv_path = unique_filepath(self.save_dir, self.base_name, '.csv')
        self._rows: List[Tuple[float,float]] = []  # (lat, lon)
        self._lock = threading.Lock()

        # Viz 준비(옵션)
        self._viz_ok = False
        self._lat0: Optional[float] = None
        self._lon0: Optional[float] = None
        self._route_xy: List[Tuple[float,float]] = []
        if self.viz_enable:
            try:
                import matplotlib
                matplotlib.use(self.viz_backend)
                import matplotlib.pyplot as plt
                self.plt = plt
                self._viz_ok = True
                self._init_plot()
            except Exception as e:
                rospy.logwarn(f"[latlon_logger] Matplotlib 사용 불가: {e}. 시각화 비활성화")
                self._viz_ok = False

        rospy.Subscriber(self.fix_topic, NavSatFix, self._cb_fix, queue_size=50)
        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo(f"[latlon_logger] logging to: {self.csv_path}")
        rospy.loginfo(f"[latlon_logger] fix topic: {self.fix_topic}, viz: {self.viz_enable}")

    def _init_plot(self):
        self.plt.ion()
        self.fig, self.ax = self.plt.subplots(figsize=(7,7))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, ls=':')
        self.ax.set_title("Lat/Lon Logger — ENU View (relative to first fix)")
        (self.line_route,) = self.ax.plot([], [], '-', lw=1.2, label='Route')
        (self.pt_cur,)     = self.ax.plot([], [], 'o', ms=8, label='Current')
        self.ax.legend(loc='upper right')
        self.txt = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                ha='left', va='top',
                                bbox=dict(fc='white', alpha=0.7), fontsize=9)

    def _cb_fix(self, msg: NavSatFix):
        if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
            return

        with self._lock:
            self._rows.append((float(msg.latitude), float(msg.longitude)))

        rospy.loginfo_throttle(1.0, f"Lat={msg.latitude:.8f}, Lon={msg.longitude:.8f}")

        if self._viz_ok:
            if self._lat0 is None:
                self._lat0, self._lon0 = float(msg.latitude), float(msg.longitude)
            x, y = lla2enu(msg.latitude, msg.longitude, self._lat0, self._lon0)
            self._route_xy.append((x, y))
            self._update_plot(x, y)

    def _update_plot(self, x: float, y: float):
        rx = [p[0] for p in self._route_xy]
        ry = [p[1] for p in self._route_xy]
        self.line_route.set_data(rx, ry)
        self.pt_cur.set_data([x],[y])
        self.ax.set_xlim(x-12, x+12)
        self.ax.set_ylim(y-12, y+12)
        self.txt.set_text(f"Samples: {len(self._route_xy)}\nPos ENU: ({x:.1f}, {y:.1f}) m")
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def _on_shutdown(self):
        try:
            with self._lock:
                rows = list(self._rows)
            if not rows:
                rospy.logwarn("[latlon_logger] 저장할 데이터가 없습니다.")
                return

            save_tmp = self.csv_path + ".tmp"
            with open(save_tmp, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["Lat", "Lon"])
                w.writerows(rows)
            os.replace(save_tmp, self.csv_path)
            rospy.loginfo(f"[latlon_logger] CSV 저장 완료 → {self.csv_path}")
        except Exception as e:
            rospy.logerr(f"[latlon_logger] CSV 저장 실패: {e}")

def main():
    rospy.init_node('latlon_logger')
    node = LatLonLoggerNode()
    rospy.spin()

if __name__ == "__main__":
    main()
