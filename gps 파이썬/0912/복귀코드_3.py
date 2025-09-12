#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
복귀코드.py  ─ Pure Pursuit 복귀 전용
- 순차 인덱스 방식 사용하지 않습니다.
- GRADE_UP_ON 토픽은 항상 0으로 퍼블리시합니다.
- 퍼블리시 토픽: /gps/speed_cmd, /gps/steer_cmd, /gps/status, /gps/wp_index, /gps/GRADEUP_ON
- 로깅 컬럼은 경로추종.py와 동일하며 flag_name은 빈 문자열로 기록합니다.
"""

import os, csv, math, time, signal
from collections import deque
import rospy
import numpy as np
import pandas as pd
import geopy.distance
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, String, Int32

# ────────────────────────────────────────────────────────────────────
# 상수/파라미터
# ────────────────────────────────────────────────────────────────────
WAYPOINT_SPACING       = 2.5
TARGET_RADIUS_END      = 2.0
MAX_STEER_DEG          = 27.0
SIGN_CONVENTION        = -1.0

FS_DEFAULT             = 20.0
GPS_TIMEOUT_SEC        = 1.0
SPEED_CODE_DEFAULT     = 3
SPEED_CAP_DEFAULT      = 6

LOOKAHEAD_MIN          = 3.2
LOOKAHEAD_MAX          = 4.0
LOOKAHEAD_K            = 0.2   # v에 따른 가감
WHEELBASE              = 0.28  # m

SPEED_BUF_LEN          = 10
MAX_JITTER_SPEED       = 4.0
MIN_MOVE_FOR_HEADING   = 0.05

# 퍼블리시 토픽(통일)
TOPIC_SPEED_CMD   = '/gps/speed_cmd'
TOPIC_STEER_CMD   = '/gps/steer_cmd'
TOPIC_RTK_STATUS  = '/gps/status'
TOPIC_WP_INDEX    = '/gps/wp_index'
TOPIC_GRADE_UP_ON = '/gps/GRADEUP_ON'   # 항상 0

# ────────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────────
def latlon_to_xy_fn(ref_lat, ref_lon):
    def _to_xy(lat, lon):
        north = geopy.distance.geodesic((ref_lat, ref_lon), (lat, ref_lon)).meters
        east  = geopy.distance.geodesic((ref_lat, ref_lon), (ref_lat, lon)).meters
        if lat < ref_lat: north *= -1
        if lon < ref_lon: east  *= -1
        return east, north
    return _to_xy

def nearest_idx_window(x, y, xs, ys, start, fwd=80, back=15):
    n=len(xs); i0=max(0,start-back); i1=min(n-1,start+fwd)
    subx=np.array(xs[i0:i1+1]); suby=np.array(ys[i0:i1+1])
    d2=(subx-x)**2 + (suby-y)**2
    return i0 + int(np.argmin(d2))

def target_idx_from_Ld(nearest, Ld, spacing, n):
    step = max(1, int(math.ceil(Ld/max(1e-6, spacing))))
    return min(n-1, nearest+step)

def pure_pursuit_delta_rad(dx, dy, yaw_rad, L, Ld):
    alpha = math.atan2(dy, dx) - yaw_rad
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    return math.atan2(2.0*L*math.sin(alpha), max(1e-6, Ld))

# ────────────────────────────────────────────────────────────────────
# 전역
# ────────────────────────────────────────────────────────────────────
gps_q = deque(maxlen=100)
pos_buf = deque(maxlen=SPEED_BUF_LEN*2)
speed_buf = deque(maxlen=SPEED_BUF_LEN)

pub_speed = pub_steer = pub_rtk = pub_wpidx = pub_grade = None
rtk_txt = "NONE"
last_fix_wall = 0.0

latest_steer_deg = 0.0
_state = dict(speed_code=SPEED_CODE_DEFAULT, wp_in_radius=-1, last_pub_speed=0.0)

log_csv_path = None
_last_log_wall = 0.0

# ────────────────────────────────────────────────────────────────────
# 콜백/퍼블리시
# ────────────────────────────────────────────────────────────────────
def cb_fix(msg: NavSatFix):
    global last_fix_wall
    if hasattr(msg,'status') and getattr(msg.status,'status',0) < 0: return
    lat, lon = float(msg.latitude), float(msg.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)): return
    t = msg.header.stamp.to_sec() if (msg.header and msg.header.stamp) else rospy.Time.now().to_sec()
    gps_q.append((lat,lon,t))
    last_fix_wall = time.time()

def cb_relpos(msg):
    global rtk_txt
    try:
        carr = int((int(msg.flags) >> 3) & 0x3)
        rtk_txt = "FIXED" if carr==2 else ("FLOAT" if carr==1 else "NONE")
    except Exception:
        rtk_txt = "NONE"

def publish_all(evt):
    now=time.time()
    no_gps = (now - last_fix_wall) > rospy.get_param('~gps_timeout_sec', GPS_TIMEOUT_SEC)
    v_out  = 0.0 if no_gps else float(_state.get('speed_code', 0))
    s_out  = 0.0 if no_gps else float(latest_steer_deg)
    if pub_speed: pub_speed.publish(Float32(v_out))
    if pub_steer: pub_steer.publish(Float32(s_out))
    if pub_rtk:   pub_rtk.publish(String(rtk_txt))
    wp_pub = (_state['wp_in_radius']+1) if _state['wp_in_radius']>=0 else 0
    if pub_wpidx: pub_wpidx.publish(Int32(int(wp_pub)))
    if pub_grade: pub_grade.publish(Int32(0))  # 항상 0
    _state['last_pub_speed'] = v_out

# ────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────
def main():
    global pub_speed, pub_steer, pub_rtk, pub_wpidx, pub_grade
    global latest_steer_deg, log_csv_path

    rospy.init_node('path_return_pp', anonymous=False)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    fs        = float(rospy.get_param('~fs', FS_DEFAULT))
    base_code = int(rospy.get_param('~speed_code', SPEED_CODE_DEFAULT))
    cap_code  = int(rospy.get_param('~speed_cap_code', SPEED_CAP_DEFAULT))
    ublox_ns  = rospy.get_param('~ublox_ns', '/gps1')
    fix_topic = rospy.get_param('~fix_topic', ublox_ns + '/fix')
    rel_topic = rospy.get_param('~relpos_topic', ublox_ns + '/navrelposned')

    pkg_guess = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
    default_csv = os.path.join(pkg_guess, 'config', 'raw_track_latlon_18.csv')
    csv_path = rospy.get_param('~csv_path', rospy.get_param('~waypoint_csv', default_csv))

    # 퍼블리셔/구독
    pub_speed = rospy.Publisher(TOPIC_SPEED_CMD,   Float32, queue_size=10)
    pub_steer = rospy.Publisher(TOPIC_STEER_CMD,   Float32, queue_size=10)
    pub_rtk   = rospy.Publisher(TOPIC_RTK_STATUS,  String,  queue_size=10)
    pub_wpidx = rospy.Publisher(TOPIC_WP_INDEX,    Int32,   queue_size=10)
    pub_grade = rospy.Publisher(TOPIC_GRADE_UP_ON, Int32,   queue_size=10)

    rospy.Subscriber(fix_topic, NavSatFix, cb_fix, queue_size=100)
    try:
        from ublox_msgs.msg import NavRELPOSNED9 as NavRELPOSNED
        rospy.Subscriber(rel_topic, NavRELPOSNED, cb_relpos, queue_size=50)
    except Exception:
        try:
            from ublox_msgs.msg import NavRELPOSNED
            rospy.Subscriber(rel_topic, NavRELPOSNED, cb_relpos, queue_size=50)
        except Exception:
            pass

    rospy.Timer(rospy.Duration(1.0/max(1.0,fs)), publish_all)

    # 웨이포인트 로드
    if not os.path.exists(csv_path):
        rospy.logerr("waypoint csv not found: %s", csv_path); return
    df = pd.read_csv(csv_path)
    assert {'Lat','Lon'}.issubset(df.columns), "CSV에 Lat,Lon 컬럼이 필요합니다."
    ref_lat, ref_lon = float(df['Lat'][0]), float(df['Lon'][0])
    to_xy = latlon_to_xy_fn(ref_lat, ref_lon)
    orig_xy = [to_xy(r['Lat'], r['Lon']) for _, r in df.iterrows()]

    # 간격 보정
    spaced=[orig_xy[0]]; acc=0.0
    for i in range(1,len(orig_xy)):
        p0=np.array(spaced[-1]); p1=np.array(orig_xy[i])
        seg=p1-p0; L=np.linalg.norm(seg)
        while acc+L>=WAYPOINT_SPACING:
            rem=WAYPOINT_SPACING-acc
            spaced.append(tuple(p0+(seg/L)*rem))
            p0=np.array(spaced[-1]); seg=p1-p0; L=np.linalg.norm(seg); acc=0.0
        acc+=L
    if spaced[-1]!=orig_xy[-1]: spaced.append(orig_xy[-1])
    xs, ys = map(np.asarray, zip(*spaced))

    _state['speed_code'] = max(0, min(cap_code, base_code))

    # 로깅 파일
    logs_dir = os.path.join(pkg_guess, 'logs'); os.makedirs(logs_dir, exist_ok=True)
    log_csv_path = rospy.get_param('~log_csv', os.path.join(logs_dir, f"return_pp_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"))

    nearest_prev = 0
    last_heading = None

    rate = rospy.Rate(fs)
    try:
        while not rospy.is_shutdown():
            while gps_q:
                lat,lon,tsec = gps_q.popleft()
                x,y = to_xy(lat,lon)

                # 속도 추정
                if pos_buf:
                    t0,x0,y0 = pos_buf[-1]
                    dt=max(1e-3, tsec-t0); d=math.hypot(x-x0, y-y0)
                    v=d/dt
                    if v<=MAX_JITTER_SPEED: speed_buf.append(v)
                pos_buf.append((tsec,x,y))
                v_mps = float(np.median(speed_buf)) if speed_buf else 0.0

                # 최근접 + 룩어헤드
                nearest = nearest_idx_window(x,y,xs,ys,nearest_prev,80,15)
                nearest_prev = nearest
                Ld = max(LOOKAHEAD_MIN, min(LOOKAHEAD_MAX, LOOKAHEAD_MIN + LOOKAHEAD_K*v_mps))
                tgt_idx = target_idx_from_Ld(nearest, Ld, WAYPOINT_SPACING, len(xs))
                tx,ty = xs[tgt_idx], ys[tgt_idx]

                # 반경 내 인덱스(퍼블리시용)
                d2 = (xs-x)**2 + (ys-y)**2
                i_min=int(np.argmin(d2))
                wp_in = i_min if math.hypot(xs[i_min]-x, ys[i_min]-y) <= TARGET_RADIUS_END else -1
                _state['wp_in_radius']=wp_in

                # 헤딩(최근 이동 벡터)
                heading=None
                for k in range(2, min(len(pos_buf), 6)):
                    t0,x0,y0 = pos_buf[-k]
                    if math.hypot(x-x0,y-y0) >= MIN_MOVE_FOR_HEADING:
                        heading=(x-x0, y-y0); break
                if heading is not None: last_heading=heading
                if last_heading is None:
                    latest_steer_deg = 0.0
                else:
                    yaw = math.atan2(last_heading[1], last_heading[0])
                    delta = pure_pursuit_delta_rad(tx-x, ty-y, yaw, WHEELBASE, Ld)
                    latest_steer_deg = SIGN_CONVENTION*float(np.degrees(delta))
                    latest_steer_deg = float(np.clip(latest_steer_deg, -MAX_STEER_DEG, +MAX_STEER_DEG))

                # ── 로깅 (0.5s) ──────────────────────────
                noww=time.time()
                if log_csv_path and (noww - _last_log_wall > 0.5):
                    try:
                        is_new = not os.path.exists(log_csv_path)
                        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
                        with open(log_csv_path,'a',newline='') as f:
                            w=csv.writer(f)
                            if is_new:
                                w.writerow(['time','lat','lon',
                                            'wp_in_idx(1based)','tgt_idx(1based)',
                                            'steer_deg','meas_speed_mps',
                                            'speed_cur_code','speed_des_code',
                                            'rtk_status','flag_name'])
                            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                        f"{lat:.7f}", f"{lon:.7f}",
                                        (wp_in+1) if wp_in>=0 else 0,
                                        (tgt_idx+1),
                                        f"{latest_steer_deg:.2f}", f"{v_mps:.2f}",
                                        int(_state['speed_code']), int(_state['speed_code']),
                                        rtk_txt, "" ])
                        globals()['_last_log_wall'] = noww
                    except Exception as e:
                        rospy.logwarn(f"[return] log write failed: {e}")

                rospy.loginfo_throttle(
                    0.5,
                    f"[PP] tgt={tgt_idx+1} wp_in={(wp_in+1) if wp_in>=0 else 0} | "
                    f"v={v_mps:.2f} m/s | code={_state['speed_code']} | "
                    f"Ld={Ld:.2f} | steer={latest_steer_deg:+.1f}° | RTK={rtk_txt}"
                )

            rate.sleep()
    finally:
        try:
            for _ in range(3):
                if pub_speed: pub_speed.publish(Float32(0.0))
                if pub_steer: pub_steer.publish(Float32(0.0))
                if pub_grade: pub_grade.publish(Int32(0))
                time.sleep(0.02)
        except Exception:
            pass

if __name__ == '__main__':
    main()
