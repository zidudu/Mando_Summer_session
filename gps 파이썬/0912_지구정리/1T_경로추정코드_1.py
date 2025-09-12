#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 + matplotlib · GPS waypoint 추종 & 조향각 퍼블리시 (정리판)
- 입력: (param) ~fix_topic  (sensor_msgs/NavSatFix, default=/gps1/fix)
- 출력:
  * /filtered_steering_angle (std_msgs/Float32)  → LPF + "부호 반전 유지"
  * /current_waypoint_index (std_msgs/Int32)     → 현재 목표 웨이포인트 인덱스
- 파라미터:
  * ~csv_path     (str)   : 웨이포인트 CSV 경로 (기본: ~/다운로드/left_final.csv)
  * ~fix_topic    (str)   : NavSatFix 토픽      (기본: /gps1/fix)
  * ~enable_plot  (bool)  : matplotlib 표시     (기본: true)
  * ~dt           (float) : 필터 샘플 주기 [s]  (기본: 0.05 = 50ms)
  * ~tau          (float) : 1차 LPF 시간상수[s] (기본: 0.25)
"""

import os, math
import rospy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, Int32

# ====== 상수 ======
EARTH_RADIUS   = 6378137.0     # [m]
MIN_DISTANCE   = 2.8           # [m] 웨이포인트 간 최소 간격
TARGET_RADIUS  = 2.8           # [m] 목표 웨이포인트 도달 반경
ANGLE_LIMIT_DEG = 30.0         # [deg] 조향 제한

# ====== 전역 상태 ======
filtered_steering_angle = 0.0
current_x, current_y = [], []
current_waypoint_index = 0

# 필터 파라미터(파라미터 서버에서 로드)
DT   = 0.05   # [s]   기본 50ms
TAU  = 0.25   # [s]
ALPHA = None  # dt/(tau+dt)로 계산

# 퍼블리셔 (init_node 이후에 실제 인스턴스 생성)
steering_pub = None
waypoint_index_pub = None

# ====== 유틸 ======
def lat_lon_to_meters(lat, lon):
    """(deg, deg) -> (m, m) Web Mercator 근사 (소구역에서 오차 작음)"""
    x = EARTH_RADIUS * lon * math.pi / 180.0
    y = EARTH_RADIUS * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def distance_in_meters(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def _pick_column(df_cols, candidates):
    """df 컬럼 목록(df_cols)에서 candidates 중 첫 번째로 일치하는 이름을 반환 (소문자 매칭). 없으면 None."""
    for c in candidates:
        if c in df_cols:
            return c
    return None

def build_reduced_waypoints(csv_path):
    """
    CSV 컬럼 자동 인식:
      - 위/경도(도): (latitude|lat), (longitude|lon)
      - 동/북   (m): (east|easting|x), (north|northing|y)
    어떤 모드를 썼는지 INFO 로깅.
    """
    # 경로 정규화 + 확장
    csv_path = os.path.expanduser(csv_path)

    # 구분자/BOM 자동 처리
    df = pd.read_csv(csv_path, encoding='utf-8-sig', engine='python', sep=None)
    # 컬럼 전처리: 소문자 + 좌우공백 제거
    df.columns = [str(c).strip().lower() for c in df.columns]

    cols = set(df.columns)
    # 후보군 정의
    lat_cands  = ['latitude', 'lat']
    lon_cands  = ['longitude', 'lon']
    east_cands = ['east', 'easting', 'x']
    north_cands= ['north', 'northing', 'y']

    lat_col  = _pick_column(cols, lat_cands)
    lon_col  = _pick_column(cols, lon_cands)
    east_col = _pick_column(cols, east_cands)
    north_col= _pick_column(cols, north_cands)

    if lat_col and lon_col:
        # 위/경도(도) → m 변환
        lat = pd.to_numeric(df[lat_col], errors='coerce').to_numpy()
        lon = pd.to_numeric(df[lon_col], errors='coerce').to_numpy()
        pts = [lat_lon_to_meters(a, b) for a, b in zip(lat, lon)]
        wx, wy = zip(*pts)
        rospy.loginfo("Waypoints from lat/lon columns: (%s, %s)", lat_col, lon_col)
    elif east_col and north_col:
        # 이미 m 좌표
        wx = pd.to_numeric(df[east_col],  errors='coerce').to_numpy()
        wy = pd.to_numeric(df[north_col], errors='coerce').to_numpy()
        rospy.loginfo("Waypoints from east/north columns: (%s, %s)", east_col, north_col)
    else:
        # 사용 가능한 컬럼 보여주고 실패
        available = ", ".join(sorted(cols))
        raise ValueError(
            "CSV에서 사용할 좌표 컬럼을 찾지 못했습니다.\n"
            f"- 허용되는 조합 1) lat/lon  2) east/north (또는 x/y)\n"
            f"- 현재 컬럼: {available}"
        )

    # 간격 축소(>= MIN_DISTANCE)
    reduced_x = [float(wx[0])]
    reduced_y = [float(wy[0])]
    for x, y in zip(wx[1:], wy[1:]):
        if distance_in_meters(reduced_x[-1], reduced_y[-1], float(x), float(y)) >= MIN_DISTANCE:
            reduced_x.append(float(x))
            reduced_y.append(float(y))

    return np.array([reduced_x, reduced_y])  # shape: (2, N)

def signed_angle_deg(v1, v2):
    """두 벡터 사이 부호 있는 각도(도). 반시계 양수, 시계 음수."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = float(np.dot(v1, v2)) / (n1 * n2)
    dot = max(min(dot, 1.0), -1.0)
    ang = math.degrees(math.acos(dot))
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return ang if cross >= 0 else -ang

# === 시간상수 기반 LPF (첫 번째 코드 방식) + 부호 반전 유지 ===
def lpf_and_invert(current_angle_deg):
    """
    1차 저역필터:
      y[k] = (1-ALPHA)*y[k-1] + ALPHA*x[k],  where ALPHA = DT/(TAU+DT)
    이후 '부호 반전 유지'를 적용해 반환.
    """
    global filtered_steering_angle, ALPHA
    filtered_steering_angle = (1.0 - ALPHA) * filtered_steering_angle + ALPHA * current_angle_deg
    return -filtered_steering_angle  # 부호 반전 유지

# ====== 콜백 ======
reduced_waypoints = None  # main에서 로드
def gps_callback(msg: NavSatFix):
    global current_waypoint_index, reduced_waypoints

    # 현재 위치(m) 누적
    x, y = lat_lon_to_meters(msg.latitude, msg.longitude)
    current_x.append(x)
    current_y.append(y)

    # 현재 목표 웨이포인트
    tx = reduced_waypoints[0][current_waypoint_index]
    ty = reduced_waypoints[1][current_waypoint_index]

    # 도달 판정
    if distance_in_meters(x, y, tx, ty) < TARGET_RADIUS:
        if current_waypoint_index < reduced_waypoints.shape[1] - 1:
            current_waypoint_index += 1
            waypoint_index_pub.publish(Int32(current_waypoint_index))

    # 조향각 계산 & 퍼블리시 (직전 위치가 있어야 진행)
    if len(current_x) >= 2:
        prev = np.array([current_x[-2], current_y[-2]])
        curr = np.array([current_x[-1], current_y[-1]])
        head_vec = curr - prev
        tgt_vec  = np.array([tx, ty]) - curr

        angle_deg = signed_angle_deg(head_vec, tgt_vec)
        # 제한
        angle_deg = max(min(angle_deg, ANGLE_LIMIT_DEG), -ANGLE_LIMIT_DEG)
        # 시간상수 기반 LPF + 부호 반전
        smooth_inv = lpf_and_invert(angle_deg)

        rospy.loginfo("Raw=%.2f°, Filtered(Inverted)=%.2f° (dt=%.3fs, tau=%.3fs, alpha=%.3f)",
                      angle_deg, smooth_inv, DT, TAU, ALPHA)
        steering_pub.publish(Float32(smooth_inv))

# ====== 시각화 ======
def update_plot(_):
    ax = plt.gca()
    ax.clear()

    # 웨이포인트 + 반경
    ax.scatter(reduced_waypoints[0], reduced_waypoints[1], s=10, marker='o', label='Waypoints')
    for i, (wx, wy) in enumerate(zip(reduced_waypoints[0], reduced_waypoints[1]), 1):
        ax.add_patch(plt.Circle((wx, wy), TARGET_RADIUS, fill=False, linestyle='--'))
        ax.text(wx, wy, str(i), fontsize=8, ha='center', va='center')

    # 현재 위치 & 벡터
    if current_x and current_y:
        cx, cy = current_x[-1], current_y[-1]
        ax.scatter(cx, cy, s=50, marker='x', label='Current')

        tx = reduced_waypoints[0][current_waypoint_index]
        ty = reduced_waypoints[1][current_waypoint_index]
        ax.arrow(cx, cy, tx - cx, ty - cy, head_width=0.5, head_length=0.5)

        # 이동 경로(최근 200점만)
        k = max(0, len(current_x) - 200)
        for i in range(k + 1, len(current_x)):
            ax.arrow(current_x[i-1], current_y[i-1],
                     current_x[i]-current_x[i-1], current_y[i]-current_y[i-1],
                     head_width=0.2, head_length=0.2, length_includes_head=True)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.grid(True)
    ax.legend(loc='best')
    ax.set_title('Filtered steering (inverted) · dt=50 ms, tau=0.25 s; limit ±30°')

# ====== 메인 ======
def main():
    global steering_pub, waypoint_index_pub, reduced_waypoints, DT, TAU, ALPHA

    rospy.init_node('gps_waypoint_tracker', anonymous=True)

    # 파라미터
    csv_path    = rospy.get_param('~csv_path',    os.path.expanduser('~/다운로드/raw_track_latlon_16.csv'))
    fix_topic   = rospy.get_param('~fix_topic',   '/gps1/fix')
    enable_plot = rospy.get_param('~enable_plot', True)
    DT  = float(rospy.get_param('~dt',  0.05))   # 50ms 기본
    TAU = float(rospy.get_param('~tau', 0.25))   # 0.25s 기본
    ALPHA = DT / (TAU + DT)

    rospy.loginfo("LPF params: dt=%.3fs, tau=%.3fs, alpha=%.3f", DT, TAU, ALPHA)

    # 퍼블리셔는 init_node 이후 생성
    steering_pub = rospy.Publisher('/filtered_steering_angle', Float32, queue_size=10)
    waypoint_index_pub = rospy.Publisher('/current_waypoint_index', Int32, queue_size=10)

    # 서브스크라이브
    rospy.Subscriber(fix_topic, NavSatFix, gps_callback)
    rospy.loginfo("Subscribed NavSatFix: %s", fix_topic)

    # 웨이포인트 구성
    reduced_waypoints = build_reduced_waypoints(csv_path)
    rospy.loginfo("Waypoints loaded: %d (reduced spacing >= %.1fm)", reduced_waypoints.shape[1], MIN_DISTANCE)

    if enable_plot:
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, update_plot, interval=300)  # 300ms 주기 업데이트
        plt.show()
    else:
        # 헤드리스 모드: ROS 스핀만
        rospy.loginfo("Headless mode: plotting disabled (~enable_plot=false).")
        rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
