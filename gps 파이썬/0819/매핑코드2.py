#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latlon_logger_node.py (RViz + Matplotlib 통합판, log_csv 실시간 저장)
────────────────────────────────────────────────────────────
ROS1에서 /ublox/fix (sensor_msgs/NavSatFix) 구독
→ ENU(X,Y) 실시간 시각화(MPL 선택) + RViz(Path/Marker)
→ CSV 로깅: ~log_csv 경로로 실시간 append (처음에 헤더 자동 기록)
   헤더: ['current_x','current_y','prev_x','prev_y',
          'target_vector_x','target_vector_y','waypoint_x','waypoint_y','steer_deg']
   * 본 노드는 타겟/웨이포인트/조향을 계산하지 않으므로 해당 칼럼은 빈 칸으로 기록합니다.
────────────────────────────────────────────────────────────
파라미터
  ~fix_topic    : str   (default "/ublox/fix")
  ~viz_enable   : bool  (default False)  # Matplotlib
  ~viz_backend  : str   (default "Qt5Agg")
  ~frame_id     : str   (default "map")  # RViz용 frame
  ~marker_scale : float (default 0.15)   # LINE_STRIP 두께[m]
  ~point_scale  : float (default 0.6)    # Current SPHERE 지름[m]
  ~mpl_window_r : float (default 12.0)   # MPL 팔로우뷰 반경[m]
  ~log_csv      : str   (default "")     # 예: "~/.ros/latlon_logs/heading_vectors.csv"
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
        self.viz_enable    = bool(rospy.get_param('~viz_enable', False))
        self.viz_backend   = rospy.get_param('~viz_backend', 'Qt5Agg')
        self.frame_id      = rospy.get_param('~frame_id', 'map')  # RViz Fixed Frame과 맞추세요
        self.marker_scale  = float(rospy.get_param('~marker_scale', 0.15))
        self.point_scale   = float(rospy.get_param('~point_scale', 0.6))
        self.mpl_window_r  = float(rospy.get_param('~mpl_window_r', 12.0))
        self.log_csv       = os.path.expanduser(rospy.get_param('~log_csv', ''))  # ← 새 저장 방식

        # ENU 기준점 및 경로 누적
        self._lat0: Optional[float] = None
        self._lon0: Optional[float] = None
        self._route_xy: List[Tuple[float,float]] = []

        # 이전 ENU(직전 값) — CSV 'prev_x/prev_y' 용
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None

        # 파일 접근 락
        self._lock = threading.Lock()

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

        if self.log_csv:
            rospy.loginfo(f"[latlon_logger] log_csv: {self.log_csv}")
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
        # 창 표시
        self.plt.show(block=False)
        try:
            self.fig.canvas.manager.window.raise_()
        except Exception:
            pass

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

        # 기준점 설정 및 ENU 변환
        if self._lat0 is None:
            self._lat0, self._lon0 = float(msg.latitude), float(msg.longitude)

        x, y = lla2enu(msg.latitude, msg.longitude, self._lat0, self._lon0)
        self._route_xy.append((x, y))

        rospy.loginfo_throttle(1.0, f"ENU=({x:.2f},{y:.2f}) m")

        # Matplotlib 업데이트
        if self._viz_ok:
            self._update_plot(x, y)

        # RViz 업데이트/퍼블리시
        self._publish_rviz(x, y)

        # CSV 로깅 (요청 방식: ~log_csv 경로로 실시간 append, 첫 행 헤더)
        if self.log_csv:
            try:
                with self._lock:
                    new_file = not os.path.exists(self.log_csv)
                    dirpath = os.path.dirname(self.log_csv)
                    if dirpath:
                        os.makedirs(dirpath, exist_ok=True)
                    with open(self.log_csv, 'a', newline='') as f:
                        w = csv.writer(f)
                        if new_file:
                            w.writerow(['current_x','current_y','prev_x','prev_y',
                                        'target_vector_x','target_vector_y','waypoint_x','waypoint_y','steer_deg'])
                        if self._prev_x is not None and self._prev_y is not None:
                            # 본 노드는 타겟/웨이포인트/조향을 계산하지 않으므로 빈 칸으로 남김
                            w.writerow([x, y, self._prev_x, self._prev_y, '', '', '', '', ''])
                        else:
                            w.writerow([x, y, '', '', '', '', '', '', ''])
            except Exception as e:
                rospy.logwarn(f"[latlon_logger] log write failed: {e}")

        # prev 업데이트
        self._prev_x, self._prev_y = x, y

    # ────────── Matplotlib 업데이트 ──────────
    def _update_plot(self, x: float, y: float):
        rx = [p[0] for p in self._route_xy]
        ry = [p[1] for p in self._route_xy]
        self.line_route.set_data(rx, ry)
        self.pt_cur.set_data([x],[y])
        # 팔로우 뷰
        R = self.mpl_window_r
        self.ax.set_xlim(x - R, x + R)
        self.ax.set_ylim(y - R, y + R)
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
        # 추가 파일 저장 없음(실시간 append 방식)
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
