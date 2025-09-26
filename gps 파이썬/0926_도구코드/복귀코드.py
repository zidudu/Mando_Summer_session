#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 + Matplotlib: GPS 상태갱신 + 조향각 퍼블리시 포함 (Waypoints 시각화)
- CSV(Lat,Lon) → (East, North) 웨이포인트 로드/표시
- /fix(NavSatFix)로 현재 위치 상태 갱신
- 진행방향 vs 목표 웨이포인트 방향으로 조향각 계산 → LPF → 퍼블리시
"""

import os
import math
import rospy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, Int32
from typing import Optional

# ─── 설정(기본값) ───────────────────────────────────────────────────────
MIN_WP_DIST   = 2.8                    # 웨이포인트 간 최소 간격(m)
LEFT_LANE_CSV = 'left_lane.csv'
MAX_STEER_DEG = 30.0                   # ±30° 제한

# (복귀 각도 LPF: RC 파라미터에서 계산된 alpha 사용)
RC_FC    = 5.0
RC_FS    = 10.0
RC_ALPHA = (2 * math.pi * RC_FC) / (2 * math.pi * RC_FC + RC_FS)

# ─── 전역 상태 ─────────────────────────────────────────────────────────
waypoints = None               # shape (N,2)
car_pos   = None               # np.array([x,y]) 최근 위치
prev_pos  = None               # 이전 위치 (헤딩 계산용)
rec_filtered_angle = 0.0       # LPF 내부 상태
cur_wp_idx = 0                 # 현재 목표 웨이포인트 인덱스

# ROS 퍼블리셔 (init 이후 바인딩)
steer_pub = None               # /filtered_steering_angle
wpidx_pub = None               # /current_waypoint_index

# ─── 유틸 ───────────────────────────────────────────────────────────────
def latlon_to_meters(lat: float, lon: float):
    """WGS84 근사: 작은 지역 기준 East/North(m) 좌표로 변환"""
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90 + lat) * math.pi / 360.0))
    return x, y

def distance_m(a, b):
    """두 점(2D)의 거리"""
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))

def load_waypoints(csv_path: str, min_spacing_m: float) -> np.ndarray:
    """CSV의 Lat/Lon을 (x,y)로 변환하고 최소 간격으로 축소"""
    df = pd.read_csv(csv_path)
    if not {"Lat", "Lon"}.issubset(df.columns):
        raise ValueError("CSV에 'Lat', 'Lon' 컬럼이 필요합니다.")
    pts = [latlon_to_meters(r["Lat"], r["Lon"]) for _, r in df.iterrows()]
    reduced = []
    for x, y in pts:
        if not reduced or distance_m(reduced[-1], (x, y)) >= min_spacing_m:
            reduced.append((float(x), float(y)))
    if not reduced:
        raise RuntimeError("웨이포인트가 비었습니다. CSV 경로/내용을 확인하세요.")
    return np.array(reduced, dtype=np.float64)

def select_recovery_wp(car_pos: np.ndarray,
                       car_heading_rad: float,
                       wps: np.ndarray,
                       Ld: float = 0.5,
                       max_ang: float = math.radians(100)) -> Optional[int]:
    """차량 전방 각도 범위 내에서 스코어(각+거리) 최소 WP 선택"""
    fwd = np.array([math.cos(car_heading_rad), math.sin(car_heading_rad)], dtype=np.float64)
    best_idx, best_score = None, float("inf")
    for i, wp in enumerate(wps):
        v = wp - car_pos
        dist = np.linalg.norm(v)
        if dist < Ld:
            continue
        dir_wp = v / (dist + 1e-6)
        ang = math.acos(np.clip(np.dot(fwd, dir_wp), -1.0, 1.0))
        if ang > max_ang:
            continue
        score = ang + 0.01 * dist
        if score < best_score:
            best_idx, best_score = i, score
    return best_idx

def calculate_steering_angle(v_forward: np.ndarray, v_target: np.ndarray) -> float:
    """조향 각도(deg): 좌회전 +, 우회전 -"""
    n1, n2 = np.linalg.norm(v_forward), np.linalg.norm(v_target)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_th = np.clip(np.dot(v_forward, v_target) / (n1 * n2), -1.0, 1.0)
    theta = math.degrees(math.acos(cos_th))
    cross = v_forward[0] * v_target[1] - v_forward[1] * v_target[0]
    return -theta if cross < 0 else theta

def lowpass(prev_val: float, new_val: float, alpha: float) -> float:
    return (1 - alpha) * prev_val + alpha * new_val

# ─── ROS 콜백: 상태 갱신 + 조향 퍼블리시 ────────────────────────────────
def gps_callback(msg: NavSatFix):
    global car_pos, prev_pos, cur_wp_idx, rec_filtered_angle

    # RTK 고정만 쓰려면: if msg.status.status != 2: return

    # 현재 위치 갱신
    xy = np.array(latlon_to_meters(msg.latitude, msg.longitude), dtype=np.float64)
    if car_pos is None:
        car_pos = xy
        prev_pos = xy
        return
    prev_pos, car_pos = car_pos, xy

    # 헤딩(라디안): 이전→현재
    delta = car_pos - prev_pos
    if np.linalg.norm(delta) < 1e-6:
        return
    heading = math.atan2(delta[1], delta[0])

    # 목표 WP 선택(전방 범위)
    idx = select_recovery_wp(car_pos, heading, waypoints)
    if idx is not None:
        cur_wp_idx = idx

    target = waypoints[cur_wp_idx]
    tgt_vec = target - car_pos
    fwd_vec = delta

    # 조향각 계산 → 제한 → LPF
    steer_raw = calculate_steering_angle(fwd_vec, tgt_vec)
    steer_raw = float(np.clip(steer_raw, -MAX_STEER_DEG, +MAX_STEER_DEG))
    rec_filtered_angle = lowpass(rec_filtered_angle, steer_raw, RC_ALPHA)

    # 퍼블리시
    steer_pub.publish(Float32(rec_filtered_angle))
    wpidx_pub.publish(Int32(cur_wp_idx))

    # 디버그 로그(필요시 주석)
    rospy.loginfo("GPS: (%.2f, %.2f)  WP[%d]=(%.2f, %.2f)  steer raw=%.1f filt=%.1f",
                  car_pos[0], car_pos[1], cur_wp_idx, target[0], target[1],
                  steer_raw, rec_filtered_angle)

# ─── Matplotlib 업데이트(옵션) ──────────────────────────────────────────
def make_anim():
    fig, ax = plt.subplots(figsize=(8, 6))
    def update(_):
        ax.clear()
        # 웨이포인트 및 인덱스
        ax.plot(waypoints[:, 0], waypoints[:, 1], 'k--', label='Waypoints')
        ax.scatter(waypoints[:, 0], waypoints[:, 1], c='gray', s=20)
        for i, (wx, wy) in enumerate(waypoints, 0):
            ax.text(wx, wy, str(i), fontsize=7, ha='center', va='center')
        # 현재/목표
        if car_pos is not None:
            ax.plot(car_pos[0], car_pos[1], 'ro', label='Car')
            tgt = waypoints[cur_wp_idx]
            ax.plot(tgt[0], tgt[1], 'go', label='Target')
            ax.arrow(car_pos[0], car_pos[1], tgt[0]-car_pos[0], tgt[1]-car_pos[1],
                     head_width=0.5, head_length=0.5, length_includes_head=True, color='g', linestyle='--')
            ax.text(0.02, 0.98, f"steer={rec_filtered_angle:+.1f}°",
                    transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', fc='w', alpha=0.8))
        ax.set_title('GPS Waypoints · steering publish')
        ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)')
        ax.axis('equal'); ax.grid(True); ax.legend(loc='best')
    ani = animation.FuncAnimation(fig, update, interval=300)
    return ani

# ─── main ───────────────────────────────────────────────────────────────
def main():
    global waypoints, steer_pub, wpidx_pub

    rospy.init_node('wp_tracker_with_steering_pub', anonymous=True)

    # 파라미터
    csv_path    = rospy.get_param('~csv_path', os.path.expanduser('~/다운로드/raw_track_latlon_17.csv'))
    fix_topic   = rospy.get_param('~fix_topic', '/gps1/fix')
    steer_topic = rospy.get_param('~steer_topic', '/filtered_steering_angle')
    wpidx_topic = rospy.get_param('~wp_index_topic', '/current_waypoint_index')
    enable_plot = rospy.get_param('~enable_plot', True)

    # 웨이포인트 로드
    waypoints = load_waypoints(csv_path, MIN_WP_DIST)  # (N,2)
    rospy.loginfo("Waypoints loaded: %d points (>= %.1fm spacing)", waypoints.shape[0], MIN_WP_DIST)

    # 퍼블리셔/구독 설정
    steer_pub = rospy.Publisher(steer_topic, Float32, queue_size=10)
    wpidx_pub = rospy.Publisher(wpidx_topic, Int32, queue_size=10)
    rospy.Subscriber(fix_topic, NavSatFix, gps_callback)
    rospy.loginfo("Subscribed: %s  |  Publishing: %s, %s", fix_topic, steer_topic, wpidx_topic)

    if enable_plot:
        ani = make_anim()
        plt.show()
    else:
        rospy.loginfo("Headless mode (no plotting).")
        rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
