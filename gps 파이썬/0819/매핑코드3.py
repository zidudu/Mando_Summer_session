#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latlon_logger_node.py (RViz + Matplotlib 통합판, lat/lon CSV 실시간 저장)
────────────────────────────────────────────────────────────
ROS1에서 /ublox/fix (sensor_msgs/NavSatFix) 구독
→ ENU(X,Y) 실시간 시각화(MPL 선택) + RViz(Path/Marker)
→ CSV 로깅: 지정 파일로 실시간 append (처음에 헤더 'lat,lon' 자동 기록)
────────────────────────────────────────────────────────────
파라미터
  ~fix_topic     : str   (default "/ublox/fix")
  ~viz_enable    : bool  (default False)      # Matplotlib on/off
  ~viz_backend   : str   (default "TkAgg")
  ~frame_id      : str   (default "map")      # RViz Fixed Frame
  ~marker_scale  : float (default 0.15)       # LINE_STRIP 두께[m]
  ~point_scale   : float (default 0.6)        # Current SPHERE 지름[m]
  ~save_dir      : str   (default "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config")
  ~filename      : str   (default "left_lane.csv")  # 저장 파일명
"""

import os
import csv
import math
import threading
from typing import Optional, Tuple, List

import rospy
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from std_msgs.msg import Header, ColorRGBA

# ───────── 좌표 유틸 (ENU용 시각화) ─────────
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
        # ── 파라미터
        self.fix_topic    = rospy.get_param('~fix_topic', '/ublox/fix')
        self.viz_enable   = bool(rospy.get_param('~viz_enable', False))
        self.viz_backend  = rospy.get_param('~viz_backend', 'TkAgg')
        self.frame_id     = rospy.get_param('~frame_id', 'map')
        self.marker_scale = float(rospy.get_param('~marker_scale', 0.15))
        self.point_scale  = float(rospy.get_param('~point_scale', 0.6))

        # 저장 경로/파일
        self.save_dir     = os.path.expanduser(
            rospy.get_param('~save_dir', '/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config')
        )
        self.filename     = rospy.get_param('~filename', 'left_lane.csv')
        os.makedirs(self.save_dir, exist_ok=True)
        self.csv_path     = os.path.join(self.save_dir, self.filename)

        # 파일 락
        self._lock = threading.Lock()

        # ENU 기준점 및 경로(시각화용)
        self._lat0: Optional[float] = None
        self._lon0: Optional[float] = None
        self._route_xy: List[Tuple[float,float]] = []

        # ── Matplotlib (옵션)
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

        # ── RViz 퍼블리셔
        ns = "latlon_logger"
        self.pub_route_marker   = rospy.Publisher(f"/{ns}/route_marker",   Marker, queue_size=1)
        self.pub_current_marker = rospy.Publisher(f"/{ns}/current_marker", Marker, queue_size=1)
        self.pub_path           = rospy.Publisher(f"/{ns}/path",           Path,   queue_size=1)

        # 메시지 템플릿
        self.route_marker   = self._make_route_marker()
        self.current_marker = self._make_current_marker()
        self.path_msg       = self._make_path_msg()

        # 구독/종료 훅
        rospy.Subscriber(self.fix_topic, NavSatFix, self._cb_fix, queue_size=50)
        rospy.on_shutdown(self._on_shutdown)

        # 파일이 없거나 비어있으면 헤더 기록
        self._ensure_header()

        rospy.loginfo(f"[latlon_logger] CSV: {self.csv_path}")
        rospy.loginfo(f"[latlon_logger] fix topic: {self.fix_topic}, viz(MPL): {self.viz_enable}, frame_id: {self.frame_id}")

    # ───────── 파일 헤더 보장 ─────────
    def _ensure_header(self):
        try:
            need_header = (not os.path.exists(self.csv_path)) or (os.path.getsize(self.csv_path) == 0)
            if need_header:
                with self._lock, open(self.csv_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['lat', 'lon'])
        except Exception as e:
            rospy.logwarn(f"[latlon_logger] 헤더 기록 실패: {e}")

    # ───────── Matplotlib ─────────
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
        self.plt.show(block=False)
        try:
            self.fig.canvas.manager.window.raise_()
        except Exception:
            pass

    # ───────── RViz 템플릿 ─────────
    def _std_header(self) -> Header:
        return Header(stamp=rospy.Time.now(), frame_id=self.frame_id)

    def _make_route_marker(self) -> Marker:
        m = Marker()
        m.header = self._std_header()
        m.ns = "route"
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = self.marker_scale
        m.color = ColorRGBA(0.1, 0.5, 1.0, 1.0)  # 파란 선
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

    # ───────── 콜백 ─────────
    def _cb_fix(self, msg: NavSatFix):
        if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
            return

        # ENU 시각화를 위한 기준점 설정 및 변환
        if self._lat0 is None:
            self._lat0, self._lon0 = float(msg.latitude), float(msg.longitude)

        x, y = lla2enu(msg.latitude, msg.longitude, self._lat0, self._lon0)
        self._route_xy.append((x, y))
        rospy.loginfo_throttle(1.0, f"Fix lat/lon=({msg.latitude:.7f},{msg.longitude:.7f}) | ENU=({x:.2f},{y:.2f}) m")

        # Matplotlib 업데이트
        if self._viz_ok:
            self._update_plot(x, y)

        # RViz 퍼블리시
        self._publish_rviz(x, y)

        # CSV 실시간 append: 헤더는 이미 보장됨
        try:
            with self._lock, open(self.csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                # 형식: lat,lon  (소수점 10자리)
                w.writerow([f"{msg.latitude:.10f}", f"{msg.longitude:.10f}"])
        except Exception as e:
            rospy.logwarn(f"[latlon_logger] CSV append 실패: {e}")

    # ───────── Matplotlib 업데이트 ─────────
    def _update_plot(self, x: float, y: float):
        rx = [p[0] for p in self._route_xy]
        ry = [p[1] for p in self._route_xy]
        self.line_route.set_data(rx, ry)
        self.pt_cur.set_data([x],[y])
        # 팔로우 뷰
        R = 12.0
        self.ax.set_xlim(x - R, x + R)
        self.ax.set_ylim(y - R, y + R)
        self.txt.set_text(f"Samples: {len(self._route_xy)}\nPos ENU: ({x:.1f}, {y:.1f}) m")
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    # ───────── RViz 퍼블리시 ─────────
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

    # ───────── 종료 처리 ─────────
    def _on_shutdown(self):
        try:
            if self._viz_ok:
                self.plt.ioff()
        except Exception:
            pass

def main():
    rospy.init_node('latlon_logger')
    _ = LatLonLoggerNode()
    rospy.spin()

if __name__ == "__main__":
    main()
