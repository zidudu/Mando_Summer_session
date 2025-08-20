#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latlon_logger_node.py (RViz + Matplotlib 통합판)
────────────────────────────────────────────────────────────
ROS1에서 /ublox/fix (sensor_msgs/NavSatFix) 구독
→ (Lat, Lon) CSV 저장 (중복 시 _1, _2 … 자동 인덱싱)
→ 실시간 시각화:
   • Matplotlib: ENU 상대좌표(첫 측정 기준) 라이브 뷰
   • RViz: Marker(LINE_STRIP, SPHERE) + nav_msgs/Path 퍼블리시
────────────────────────────────────────────────────────────
파라미터
  ~fix_topic    : str   (default "/ublox/fix")
  ~save_dir     : str   (default "~/.ros/latlon_logs")
  ~base_name    : str   (default "raw_track_latlon")
  ~viz_enable   : bool  (default False)  # Matplotlib
  ~viz_backend  : str   (default "TkAgg")
  ~frame_id     : str   (default "map")  # RViz용 frame
  ~marker_scale : float (default 0.15)   # LINE_STRIP 두께[m]
  ~point_scale  : float (default 0.6)    # Current SPHERE 지름[m]
"""

import os
import csv
import math
import time
import threading
from typing import Optional, Tuple, List

import rospy
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from std_msgs.msg import Header, ColorRGBA

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
    x = _R * dlon * math.cos(lat0r)  # East
    y = _R * dlat                    # North
    return x, y

class LatLonLoggerNode:
    def __init__(self):
        # 기본 파라미터
        self.fix_topic     = rospy.get_param('~fix_topic', '/ublox/fix')
        self.save_dir      = os.path.expanduser(rospy.get_param('~save_dir', '~/.ros/latlon_logs'))
        self.base_name     = rospy.get_param('~base_name', 'raw_track_latlon')
        self.viz_enable    = bool(rospy.get_param('~viz_enable', False))
        self.viz_backend   = rospy.get_param('~viz_backend', 'TkAgg')
        self.frame_id      = rospy.get_param('~frame_id', 'map')  # RViz Fixed Frame과 맞추세요
        self.marker_scale  = float(rospy.get_param('~marker_scale', 0.15))
        self.point_scale   = float(rospy.get_param('~point_scale', 0.6))

        # CSV 준비
        self.csv_path = unique_filepath(self.save_dir, self.base_name, '.csv')
        self._rows: List[Tuple[float,float]] = []  # (lat, lon)
        self._lock = threading.Lock()

        # ENU 기준점 및 경로 누적
        self._lat0: Optional[float] = None
        self._lon0: Optional[float] = None
        self._route_xy: List[Tuple[float,float]] = []

        # Matplotlib 준비(옵션)
        self._viz_ok = False
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

        # RViz 퍼블리셔
        ns = "latlon_logger"
        self.pub_route_marker  = rospy.Publisher(f"/{ns}/route_marker",   Marker, queue_size=1)
        self.pub_current_marker= rospy.Publisher(f"/{ns}/current_marker", Marker, queue_size=1)
        self.pub_path          = rospy.Publisher(f"/{ns}/path",           Path,   queue_size=1)

        # 미리 Marker/Path 템플릿 생성
        self.route_marker  = self._make_route_marker()
        self.current_marker= self._make_current_marker()
        self.path_msg      = self._make_path_msg()

        # 구독 & 종료 훅
        rospy.Subscriber(self.fix_topic, NavSatFix, self._cb_fix, queue_size=50)
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(f"[latlon_logger] logging to: {self.csv_path}")
        rospy.loginfo(f"[latlon_logger] fix topic: {self.fix_topic}, viz(MPL): {self.viz_enable}, frame_id: {self.frame_id}")

    # ────────── Matplotlib ──────────
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

    # ────────── RViz 메세지 템플릿 ──────────
    def _std_header(self) -> Header:
        return Header(stamp=rospy.Time.now(), frame_id=self.frame_id)

    def _make_route_marker(self) -> Marker:
        m = Marker()
        m.header = self._std_header()
        m.ns = "route"
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = self.marker_scale         # 선 두께
        m.color = ColorRGBA(0.1, 0.5, 1.0, 1.0)  # 파란 계열
        m.pose.orientation.w = 1.0
        return m

    def _make_current_marker(self) -> Marker:
        m = Marker()
        m.header = self._std_header()
        m.ns = "current"
        m.id = 2
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.scale.x = self.point_scale
        m.scale.y = self.point_scale
        m.scale.z = self.point_scale * 0.5
        m.color = ColorRGBA(1.0, 0.2, 0.2, 1.0)  # 빨간 점
        m.pose.orientation.w = 1.0
        return m

    def _make_path_msg(self) -> Path:
        p = Path()
        p.header = self._std_header()
        return p

    # ────────── 콜백 ──────────
    def _cb_fix(self, msg: NavSatFix):
        if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
            return

        # CSV 누적
        with self._lock:
            self._rows.append((float(msg.latitude), float(msg.longitude)))

        # 기준점 설정 및 ENU 변환
        if self._lat0 is None:
            self._lat0, self._lon0 = float(msg.latitude), float(msg.longitude)

        x, y = lla2enu(msg.latitude, msg.longitude, self._lat0, self._lon0)
        self._route_xy.append((x, y))

        # Matplotlib 업데이트
        rospy.loginfo_throttle(1.0, f"Lat={msg.latitude:.8f}, Lon={msg.longitude:.8f} | ENU=({x:.2f},{y:.2f}) m")
        if self._viz_ok:
            self._update_plot(x, y)

        # RViz: Marker & Path 업데이트/퍼블리시
        self._publish_rviz(x, y)

    # ────────── Matplotlib 업데이트 ──────────
    def _update_plot(self, x: float, y: float):
        rx = [p[0] for p in self._route_xy]
        ry = [p[1] for p in self._route_xy]
        self.line_route.set_data(rx, ry)
        self.pt_cur.set_data([x],[y])
        # 팔로우 뷰
        self.ax.set_xlim(x-12, x+12)
        self.ax.set_ylim(y-12, y+12)
        self.txt.set_text(f"Samples: {len(self._route_xy)}\nPos ENU: ({x:.1f}, {y:.1f}) m")
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    # ────────── RViz 퍼블리시 ──────────
    def _publish_rviz(self, x: float, y: float):
        now = rospy.Time.now()

        # Route LINE_STRIP
        self.route_marker.header.stamp = now
        self.route_marker.points = [Point(px, py, 0.0) for (px, py) in self._route_xy]
        self.pub_route_marker.publish(self.route_marker)

        # Current SPHERE
        self.current_marker.header.stamp = now
        self.current_marker.pose.position.x = x
        self.current_marker.pose.position.y = y
        self.current_marker.pose.position.z = 0.0
        self.pub_current_marker.publish(self.current_marker)

        # Path
        self.path_msg.header.stamp = now
        ps = PoseStamped()
        ps.header = self._std_header()
        ps.header.stamp = now
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        self.path_msg.poses.append(ps)
        self.pub_path.publish(self.path_msg)

    # ────────── 종료 처리 ──────────
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
