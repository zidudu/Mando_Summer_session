#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latlon_logger_node.py — ROS1 (rospy) 경로 로깅 노드 (실시간 append)
────────────────────────────────────────────────────────────
• /ublox/fix (sensor_msgs/NavSatFix) 구독 → (Lat, Lon) 실시간 CSV 저장
• 저장 경로: /home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/<auto_name>.csv
    - 기본 파일명: raw_track_latlon.csv
    - 중복 시: raw_track_latlon_1.csv, raw_track_latlon_2.csv ...
• CSV 헤더: Lat,Lon
"""

import os
import csv
import math
import threading
from typing import Optional

import rospy
from sensor_msgs.msg import NavSatFix

# ───────────────── 고정 저장 디렉터리 ─────────────────
SAVE_DIR = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config"
BASE_NAME = "raw_track_latlon"   # 기본 베이스 이름 (left_lane.csv는 예시였음)

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

class LatLonLoggerNode:
    def __init__(self):
        # 토픽은 필요하면 launch에서 바꿀 수 있도록 파라미터만 허용 (기본 /ublox/fix)
        self.fix_topic = rospy.get_param("~fix_topic", "/ublox/fix")

        # 파일 경로 결정(중복 방지)
        self.csv_path: str = unique_filepath(SAVE_DIR, BASE_NAME, ".csv")

        # 파일 락 & 헤더 보장
        self._lock = threading.Lock()
        self._ensure_header()

        # 구독/종료 훅
        rospy.Subscriber(self.fix_topic, NavSatFix, self._cb_fix, queue_size=50)
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(f"[latlon_logger] 구독: {self.fix_topic}")
        rospy.loginfo(f"[latlon_logger] CSV: {self.csv_path}  (형식: Lat,Lon)")

    def _ensure_header(self):
        """파일이 없거나 비어 있으면 헤더 Lat,Lon 기록"""
        try:
            need_header = (not os.path.exists(self.csv_path)) or (os.path.getsize(self.csv_path) == 0)
            if need_header:
                with self._lock, open(self.csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["Lat", "Lon"])
        except Exception as e:
            rospy.logerr(f"[latlon_logger] 헤더 쓰기 실패: {e}")

    def _cb_fix(self, msg: NavSatFix):
        # 유효성 체크
        if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
            return

        lat = float(msg.latitude)
        lon = float(msg.longitude)

        # 실시간 append 저장
        try:
            with self._lock, open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                # 소수점 자릿수 고정 원하시면 f"{lat:.8f}" 형태로 바꾸세요.
                w.writerow([lat, lon])
        except Exception as e:
            rospy.logwarn(f"[latlon_logger] CSV append 실패: {e}")

        # 진행 상황 로그(1초 스로틀)
        rospy.loginfo_throttle(1.0, f"[latlon_logger] Lat={lat:.8f}, Lon={lon:.8f}")

    def _on_shutdown(self):
        # 실시간으로 이미 디스크에 쓰고 있으므로 여기서는 안내만
        rospy.loginfo(f"[latlon_logger] 종료. 최종 파일: {self.csv_path}")

def main():
    rospy.init_node("latlon_logger")
    _ = LatLonLoggerNode()
    rospy.spin()

if __name__ == "__main__":
    main()
