#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 + Matplotlib: GPS 상태갱신 + 조향각 퍼블리시 포함 (Waypoints 시각화)
- CSV(Lat,Lon) → (East, North) 웨이포인트 로드/표시
- /fix(NavSatFix)로 현재 위치 상태 갱신
- 진행방향 vs 목표 웨이포인트 방향으로 조향각 계산 → LPF → 퍼블리시
- (추가) 고정 속도 퍼블리시(/vehicle/speed_cmd), RTK 상태 퍼블리시(/rtk/status, 옵션)
- (추가) CSV 로깅(주기: ~0.5s)
"""

import os
import math
import time
import rospy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, Int32, String
from typing import Optional

# rospkg(패키지 경로 탐색)
try:
    import rospkg
except Exception:
    rospkg = None

# u-blox RTK 상태(옵션) — 없으면 자동 비활성
_HAVE_RELPOS = False
try:
    from ublox_msgs.msg import NavRELPOSNED9 as NavRELPOSNED
    _HAVE_RELPOS = True
except Exception:
    try:
        from ublox_msgs.msg import NavRELPOSNED
        _HAVE_RELPOS = True
    except Exception:
        _HAVE_RELPOS = False

# ─── 설정(기본값) ───────────────────────────────────────────────────────
MIN_WP_DIST   = 2.8                    # 웨이포인트 간 최소 간격(m)
DEFAULT_CSV   = 'left_lane.csv'        # 패키지/config 내 기본 파일명
MAX_STEER_DEG = 27.0                   # ±27° 제한

# (복귀 각도 LPF: RC 파라미터에서 계산된 alpha 사용 — 알고리즘 유지)
RC_FC    = 5.0
RC_FS    = 10.0
RC_ALPHA = (2 * math.pi * RC_FC) / (2 * math.pi * RC_FC + RC_FS)

# 시각화
INDEX_STEP = 1          # 인덱스 라벨 표시 간격
SHOW_START_END = True   # 시작/끝 강조

# 고정 속도 퍼블리시
ENABLE_SPEED_PUB = True
SPEED_TOPIC      = '/vehicle/speed_cmd'
SPEED_VALUE      = 3.0     # 시스템 규약에 맞춰 해석(코드/속도)
SPEED_PUB_HZ     = 10.0

# 로깅
LOG_ENABLE = True
LOG_DIR_DEFAULT = None  # 패키지/logs 자동
LOG_CSV_PATH = None     # 직접 지정 시 사용
_LOG_LAST_WALL = 0.0

# ─── 전역 상태 ─────────────────────────────────────────────────────────
waypoints = None               # shape (N,2)
car_pos   = None               # np.array([x,y]) 최근 위치
prev_pos  = None               # 이전 위치 (헤딩 계산용)
rec_filtered_angle = 0.0       # LPF 내부 상태
cur_wp_idx = 0                 # 현재 목표 웨이포인트 인덱스

# 최근 GPS 원시 위경도(로그용)
_last_lat = None
_last_lon = None

# ROS 퍼블리셔/구독자
steer_pub = None               # /filtered_steering_angle
wpidx_pub = None               # /current_waypoint_index
speed_pub = None               # /vehicle/speed_cmd
rtk_pub   = None               # /rtk/status

# RTK 상태 문자열
rtk_status_txt = "NONE"

# ─── 경로 유틸 ─────────────────────────────────────────────────────────
def resolve_csv_path(path_param: Optional[str], default_name: str = DEFAULT_CSV) -> str:
    """
    CSV 경로 해석:
      1) 절대경로 존재 → 그대로
      2) CWD 기준 상대경로 존재 → 그대로
      3) rtk_waypoint_tracker/config/<파일명> → 존재 시 사용
      4) ~ 확장 후 사용
    """
    if path_param:
        p = os.path.expanduser(path_param)
        if os.path.isabs(p) and os.path.exists(p):
            return p
        if os.path.exists(p):
            return p

    pkg_path = None
    if rospkg is not None:
        try:
            pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
        except Exception:
            pkg_path = None

    if pkg_path is not None:
        base = os.path.basename(path_param) if path_param else default_name
        candidate = os.path.join(pkg_path, 'config', base)
        if os.path.exists(candidate):
            return candidate

    return os.path.expanduser(path_param or default_name)

def resolve_log_path(log_csv_param: Optional[str], log_dir_param: Optional[str]) -> str:
    """
    로그 파일 경로 생성:
      - log_csv 지정 시 그 경로 사용
      - 아니면: <pkg>/logs/waypoint_log_YYYYmmdd_HHMMSS.csv
      - 패키지 없으면: ~/catkin_ws/src/rtk_waypoint_tracker/logs/...
      - 위도/경도/조향/속도를 주기적으로 append
    """
    if log_csv_param:
        path = os.path.expanduser(log_csv_param)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    # 디렉토리 우선순위
    if log_dir_param:
        log_dir = os.path.expanduser(log_dir_param)
    else:
        log_dir = None
        if rospkg is not None:
            try:
                pkg = rospkg.RosPack().get_path('rtk_waypoint_tracker')
                log_dir = os.path.join(pkg, 'logs')
            except Exception:
                pass
        if not log_dir:
            log_dir = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker/logs')

    os.makedirs(log_dir, exist_ok=True)
    fname = f"waypoint_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return os.path.join(log_dir, fname)

# ─── 유틸 ───────────────────────────────────────────────────────────────
def latlon_to_meters(lat: float, lon: float):
    """WGS84 근사: 작은 지역 기준 East/North(m) 좌표로 변환 (Web Mercator 유사)"""
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
    """차량 전방 각도 범위 내에서 스코어(각+거리) 최소 WP 선택 — 알고리즘 유지"""
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

# ─── RTK 상태(옵션) ────────────────────────────────────────────────────
def relpos_cb(msg: 'NavRELPOSNED'):
    """u-blox RELPOSNED flags 기반 RTK 상태 표시: 2=FIX, 1=FLOAT, 그 외 NONE"""
    global rtk_status_txt
    try:
        carr_soln = (int(msg.flags) >> 3) & 0x3  # 0=none,1=float,2=fix
        if carr_soln == 2:   rtk_status_txt = "FIX"
        elif carr_soln == 1: rtk_status_txt = "FLOAT"
        else:                rtk_status_txt = "NONE"
    except Exception:
        rtk_status_txt = "NONE"

# ─── 로깅 ───────────────────────────────────────────────────────────────
def log_append_if_due(path, lat, lon, x, y, wp_idx, steer_raw, steer_filt, speed_val, rtk_txt, period=0.5):
    global _LOG_LAST_WALL
    noww = time.time()
    if (noww - _LOG_LAST_WALL) < period:
        return
    try:
        new_file = not os.path.exists(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a') as f:
            if new_file:
                f.write("time,lat,lon,x,y,wp_idx,steer_raw_deg,steer_filt_deg,speed_value,rtk\n")
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},"
                    f"{lat:.7f},{lon:.7f},{x:.3f},{y:.3f},{int(wp_idx)},"
                    f"{steer_raw:.2f},{steer_filt:.2f},{speed_val:.2f},{rtk_txt}\n")
        _LOG_LAST_WALL = noww
    except Exception as e:
        rospy.logwarn_throttle(2.0, f"[log] write failed: {e}")

# ─── ROS 콜백: 상태 갱신 + 조향 퍼블리시 ────────────────────────────────
def gps_callback(msg: NavSatFix):
    global car_pos, prev_pos, cur_wp_idx, rec_filtered_angle, _last_lat, _last_lon

    # 현재 위치 갱신
    _last_lat, _last_lon = float(msg.latitude), float(msg.longitude)
    xy = np.array(latlon_to_meters(_last_lat, _last_lon), dtype=np.float64)
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

    # 목표 WP 선택(전방 범위) — 알고리즘 유지
    idx = select_recovery_wp(car_pos, heading, waypoints)
    if idx is not None:
        cur_wp_idx = idx

    target = waypoints[cur_wp_idx]
    tgt_vec = target - car_pos
    fwd_vec = delta

    # 조향각 계산 → 제한 → LPF (알고리즘 그대로)
    steer_raw = calculate_steering_angle(fwd_vec, tgt_vec)
    steer_raw = float(np.clip(steer_raw, -MAX_STEER_DEG, +MAX_STEER_DEG))
    rec_filtered_angle = lowpass(rec_filtered_angle, steer_raw, RC_ALPHA)

    # 퍼블리시(조향, 인덱스, RTK)
    if steer_pub: steer_pub.publish(Float32(rec_filtered_angle))
    if wpidx_pub: wpidx_pub.publish(Int32(cur_wp_idx))
    if rtk_pub:   rtk_pub.publish(String(rtk_status_txt))

    # 로깅(0.5s 주기)
    if LOG_ENABLE and LOG_CSV_PATH:
        log_append_if_due(
            LOG_CSV_PATH, _last_lat, _last_lon,
            car_pos[0], car_pos[1], cur_wp_idx,
            steer_raw, rec_filtered_angle,
            SPEED_VALUE if ENABLE_SPEED_PUB else 0.0,
            rtk_status_txt
        )

    # 디버그 로그
    rospy.loginfo_throttle(0.5,
        "GPS: (%.2f, %.2f)  WP[%d]=(%.2f, %.2f)  steer raw=%.1f filt=%.1f  RTK=%s",
        car_pos[0], car_pos[1], cur_wp_idx, target[0], target[1],
        steer_raw, rec_filtered_angle, rtk_status_txt
    )

# ─── 고정 속도 퍼블리시 타이머 ──────────────────────────────────────────
def speed_timer_cb(event):
    if speed_pub is not None:
        speed_pub.publish(Float32(SPEED_VALUE))

# ─── Matplotlib 업데이트(옵션) ──────────────────────────────────────────
def make_anim():
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(_):
        ax.clear()
        # 웨이포인트 및 인덱스
        ax.plot(waypoints[:, 0], waypoints[:, 1], 'k--', label='Waypoints (path)')
        ax.scatter(waypoints[:, 0], waypoints[:, 1], c='gray', s=18, label='Waypoints (nodes)')
        if INDEX_STEP >= 1:
            for i, (wx, wy) in enumerate(waypoints, 0):
                if i % INDEX_STEP == 0:
                    ax.text(wx, wy, str(i), fontsize=7, ha='center', va='center', color='dimgray')
        if SHOW_START_END and waypoints.shape[0] >= 2:
            sx, sy = waypoints[0]
            ex, ey = waypoints[-1]
            ax.scatter([sx], [sy], c='blue', s=45, label='Start')
            ax.scatter([ex], [ey], c='purple', s=45, label='End')

        # 현재/목표 + HUD
        if car_pos is not None:
            ax.plot(car_pos[0], car_pos[1], 'ro', label='Car')
            tgt = waypoints[cur_wp_idx]
            ax.plot(tgt[0], tgt[1], 'go', label='Target')
            ax.arrow(car_pos[0], car_pos[1], tgt[0]-car_pos[0], tgt[1]-car_pos[1],
                     head_width=0.5, head_length=0.5, length_includes_head=True,
                     color='g', linestyle='--', linewidth=1.0)

            # HUD(좌상단): 조향/인덱스/속도/RTK
            hud = (f"steer={rec_filtered_angle:+.1f}°  |  idx={cur_wp_idx}  |  "
                   f"v={SPEED_VALUE:.1f}  |  RTK={rtk_status_txt}")
            ax.text(0.02, 0.98, hud,
                    transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', fc='w', ec='0.6', alpha=0.85))

        ax.set_title('GPS Waypoints · Steering / Speed / RTK')
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.axis('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', framealpha=0.88)

    ani = animation.FuncAnimation(fig, update, interval=300)
    return ani

# ─── main ───────────────────────────────────────────────────────────────
def main():
    global waypoints, steer_pub, wpidx_pub, speed_pub, rtk_pub
    global ENABLE_SPEED_PUB, SPEED_TOPIC, SPEED_VALUE, SPEED_PUB_HZ
    global LOG_ENABLE, LOG_CSV_PATH

    rospy.init_node('wp_tracker_with_steering_pub', anonymous=True)

    # 파라미터
    csv_param    = rospy.get_param('~csv_path', DEFAULT_CSV)         # 문자열(절대/상대/파일명)
    fix_topic    = rospy.get_param('~fix_topic', '/gps1/fix')
    steer_topic  = rospy.get_param('~steer_topic', '/filtered_steering_angle')
    wpidx_topic  = rospy.get_param('~wp_index_topic', '/current_waypoint_index')
    enable_plot  = rospy.get_param('~enable_plot', True)

    # 고정 속도 퍼블리시 파라미터
    ENABLE_SPEED_PUB = bool(rospy.get_param('~enable_speed_pub', ENABLE_SPEED_PUB))
    SPEED_TOPIC      = rospy.get_param('~speed_topic', SPEED_TOPIC)
    SPEED_VALUE      = float(rospy.get_param('~speed_value', SPEED_VALUE))
    SPEED_PUB_HZ     = float(rospy.get_param('~speed_pub_hz', SPEED_PUB_HZ))

    # 로깅 파라미터
    LOG_ENABLE   = bool(rospy.get_param('~log_enable', LOG_ENABLE))
    LOG_CSV_PATH = resolve_log_path(
        rospy.get_param('~log_csv', LOG_CSV_PATH),
        rospy.get_param('~log_dir', LOG_DIR_DEFAULT)
    )
    if LOG_ENABLE:
        rospy.loginfo("Logging enabled → %s", LOG_CSV_PATH)
    else:
        rospy.loginfo("Logging disabled")

    # CSV 경로 해석 및 웨이포인트 로드
    csv_path = resolve_csv_path(csv_param, DEFAULT_CSV)
    rospy.loginfo("CSV path resolved: %s", csv_path)
    waypoints = load_waypoints(csv_path, MIN_WP_DIST)  # (N,2)
    rospy.loginfo("Waypoints loaded: %d points (>= %.1fm spacing)", waypoints.shape[0], MIN_WP_DIST)

    # 퍼블리셔/구독 설정
    steer_pub = rospy.Publisher(steer_topic, Float32, queue_size=10)
    wpidx_pub = rospy.Publisher(wpidx_topic, Int32, queue_size=10)
    if ENABLE_SPEED_PUB:
        speed_pub = rospy.Publisher(SPEED_TOPIC, Float32, queue_size=10)
        rospy.Timer(rospy.Duration(1.0 / max(1e-3, SPEED_PUB_HZ)), speed_timer_cb)
        rospy.loginfo("Speed publish ON → topic=%s, value=%.2f @ %.1f Hz", SPEED_TOPIC, SPEED_VALUE, SPEED_PUB_HZ)
    else:
        rospy.loginfo("Speed publish OFF")

    # RTK 상태 퍼블리셔 + (옵션) ublox 구독
    rtk_pub = rospy.Publisher('/rtk/status', String, queue_size=10)
    if _HAVE_RELPOS:
        relpos_topic = rospy.get_param('~relpos_topic', '/gps1/navrelposned')
        rospy.Subscriber(relpos_topic, NavRELPOSNED, relpos_cb, queue_size=50)
        rospy.loginfo("RTK status source: %s (ublox RELPOSNED)", relpos_topic)
    else:
        rospy.loginfo("RTK status: ublox RELPOSNED 메시지 미탑재(기본 NONE)")

    rospy.Subscriber(fix_topic, NavSatFix, gps_callback)
    rospy.loginfo("Subscribed: %s  |  Publishing: %s, %s, %s, /rtk/status",
                  fix_topic, steer_topic, wpidx_topic, SPEED_TOPIC if ENABLE_SPEED_PUB else "(speed OFF)")

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
