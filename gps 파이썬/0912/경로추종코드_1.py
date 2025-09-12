#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
경로추종.py  — 순차 인덱스 + 조향 LPF + (내장) 플래그 알고리즘
- Pure Pursuit는 사용하지 않습니다(복귀 전용 노드에서만 사용).
- 플래그 알고리즘은 복귀코드_플래그.py의 구조/키를 반영하여 이 파일에 통합했습니다.
- 공통 퍼블리시 토픽:
    /gps/speed_cmd (Float32, code)
    /gps/steer_cmd (Float32, deg)
    /gps/status    (String: "FIXED"|"FLOAT"|"NONE")
    /gps/wp_index  (Int32: 원 안이면 1-based, 아니면 0)
    /gps/GRADEUP_ON (Int32: 플래그 값 0/1)

- 로깅 컬럼 동일:
  time, lat, lon, wp_in_idx(1based), tgt_idx(1based), steer_deg,
  meas_speed_mps, speed_cur_code, speed_des_code, rtk_status, flag_name
"""

import os, csv, math, time, signal
from collections import deque

import rospy
import numpy as np
import pandas as pd
import geopy.distance

from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, String, Int32

# ───────────────────────────────────────────────────────────────────
# 토픽 (통일)
# ───────────────────────────────────────────────────────────────────
TOPIC_SPEED_CMD   = '/gps/speed_cmd'
TOPIC_STEER_CMD   = '/gps/steer_cmd'
TOPIC_RTK_STATUS  = '/gps/status'     # 중요: '/gps/rtk_status' 아님
TOPIC_WP_INDEX    = '/gps/wp_index'
TOPIC_GRADE_UP_ON = '/gps/GRADEUP_ON'

# ───────────────────────────────────────────────────────────────────
# 경로/제어 기본 파라미터
# ───────────────────────────────────────────────────────────────────
WAYPOINT_SPACING       = 2.5   # m (보간 간격)
TARGET_RADIUS_END      = 2.0   # m (도달 판정 반경)
MAX_STEER_DEG          = 27.0  # deg (조향 제한)
SIGN_CONVENTION        = -1.0  # 좌/우 반전 필요 시 -1.0

LPF_FC_HZ              = 0.8   # 조향 LPF 차단주파수[Hz]
SPEED_BUF_LEN          = 10
MAX_JITTER_SPEED       = 4.0   # m/s (속도 스파이크 제거)
MIN_MOVE_FOR_HEADING   = 0.05  # m (헤딩 추정 최소 이동)
FS_DEFAULT             = 20.0  # Hz
GPS_TIMEOUT_SEC        = 1.0   # 최근 fix 타임아웃 → 안전정지

BASE_SPEED_CODE_DEFAULT  = 5
GLOBAL_SPEED_CAP_DEFAULT = 10
STEP_PER_LOOP_DEFAULT    = 2    # 기본 램프 rate (code/loop)

# ───────────────────────────────────────────────────────────────────
# (내장) 플래그 알고리즘 파라미터/정의
#   - 복귀코드_플래그.py의 키/의미를 그대로 사용
#   - radius_scale/grade_topic/speed_code/step_per_loop 등 반영
#   - stop_on_hit + stop_duration_sec 간단 구현(구간 진입 시 정지 유지)
# ───────────────────────────────────────────────────────────────────
STOP_FLAG_STAY_SEC = 3.0

FLAG_DEFS = [
    {'name': 'GRADE_START', 'start': 3, 'end': 4,
     'radius_scale': 1.0, 'lookahead_scale': 0.95,
     'speed_code': 5, 'speed_cap': 7, 'step_per_loop': 2,
     'stop_on_hit': False, 'stop_duration_sec': None, 'sequential': False,
     'grade_topic': 1},

    # 정지 구간 (홀드)
    {'name': 'GRADE_UP', 'start': 5, 'end': 5,
     'radius_scale': 0.3, 'lookahead_scale': 0.95,
     'speed_code': 0, 'speed_cap': 0, 'step_per_loop': 2,
     'stop_on_hit': True, 'stop_duration_sec': 3, 'sequential': False,
     'grade_topic': 1},

    {'name': 'GRADE_GO', 'start': 6, 'end': 8,
     'radius_scale': 1.0, 'lookahead_scale': 0.95,
     'speed_code': 5, 'speed_cap': 10, 'step_per_loop': 2,
     'stop_on_hit': False, 'stop_duration_sec': None, 'sequential': False,
     'grade_topic': 1},

    {'name': 'GRADE_END', 'start': 9, 'end': 10,
     'radius_scale': 1.0, 'lookahead_scale': 0.95,
     'speed_code': 5, 'speed_cap': 10, 'step_per_loop': 2,
     'stop_on_hit': False, 'stop_duration_sec': None, 'sequential': False,
     'grade_topic': 0},
]

def build_flag_zones(flag_defs):
    zones = []
    for fd in flag_defs:
        s0 = int(min(fd['start'], fd['end'])) - 1
        e0 = int(max(fd['start'], fd['end'])) - 1
        zones.append({
            'name': fd['name'],
            'start0': s0,
            'end0': e0,
            'radius_scale': float(fd.get('radius_scale', 1.0)),
            'lookahead_scale': float(fd.get('lookahead_scale', 1.0)),
            'speed_code': fd.get('speed_code', None),
            'speed_cap': fd.get('speed_cap', None),
            'step_per_loop': int(fd.get('step_per_loop', STEP_PER_LOOP_DEFAULT)),
            'stop_on_hit': bool(fd.get('stop_on_hit', False)),
            'stop_duration_sec': float(fd.get('stop_duration_sec', STOP_FLAG_STAY_SEC)
                                       if fd.get('stop_duration_sec', None) is not None
                                       else STOP_FLAG_STAY_SEC),
            'sequential': bool(fd.get('sequential', False)),
            'grade_topic': fd.get('grade_topic', None),
            'disp': f"{fd['start']}–{fd['end']} (1-based)"
        })
    return zones

def in_radius_idx(x, y, idx0, xs, ys, radius):
    tx, ty = xs[idx0], ys[idx0]
    return (math.hypot(x - tx, y - ty) <= radius)

# 플래그 상태 전역 (간단화)
flag_zones      = build_flag_zones(FLAG_DEFS)
active_flag     = None
just_entered    = False
just_exited     = False
hold_active     = False
hold_until      = 0.0
hold_reason     = ""

def flag_enter(zone):  rospy.loginfo(f"[flag] ENTER {zone['name']} {zone['disp']}")
def flag_exit(zone):   rospy.loginfo(f"[flag] EXIT  {zone['name']} {zone['disp']}")

def flag_update_state(x, y, nearest_idx, xs, ys, base_radius):
    """
    - 시작점 근방(반경 base_radius)에 들어오면 그 구간을 활성화
    - 활성 구간의 end0를 지나 멀어지면 비활성화
    - eff_radius = base_radius * radius_scale
    - (선택) stop_on_hit: 구간 최초 진입 시 stop_duration_sec 동안 정지 홀드
    """
    global active_flag, just_entered, just_exited, hold_active, hold_until, hold_reason

    just_entered = False
    just_exited  = False

    # 들어간 플래그가 없다면: 시작점 반경으로 진입 판단
    if active_flag is None:
        for z in flag_zones:
            if in_radius_idx(x, y, z['start0'], xs, ys, base_radius):
                active_flag = z
                just_entered = True
                flag_enter(z)
                # 진입 시 홀드가 필요한 구간이면 설정
                if z.get('stop_on_hit', False):
                    hold_active = True
                    hold_until  = time.time() + float(z.get('stop_duration_sec', STOP_FLAG_STAY_SEC))
                    hold_reason = z['name']
                break
    else:
        z = active_flag
        # 구간 종료 조건: end0까지 진입했다가 벗어나면 종료로 간주(단순화)
        #   - end0 반경 밖이고, 현재 최근접 인덱스가 end0를 지나면 종료
        end_hit = in_radius_idx(x, y, z['end0'], xs, ys, base_radius)
        if (not end_hit) and (nearest_idx >= z['end0']):
            just_exited = True
            flag_exit(z)
            active_flag = None

    eff_radius = base_radius
    look_scl   = 1.0
    if active_flag is not None:
        z = active_flag
        eff_radius = base_radius * float(z.get('radius_scale', 1.0))
        look_scl   = float(z.get('lookahead_scale', 1.0))

    return eff_radius, look_scl, active_flag, just_entered, just_exited

# ───────────────────────────────────────────────────────────────────
# 유틸/보조
# ───────────────────────────────────────────────────────────────────
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

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float); v2 = np.array(v2, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1==0.0 or n2==0.0: return 0.0
    dot = np.clip(np.dot(v1,v2)/(n1*n2), -1.0, 1.0)
    ang = math.degrees(math.acos(dot))
    # 좌(+)/우(-) 부호
    if v1[0]*v2[1] - v1[1]*v2[0] < 0: ang = -ang
    return ang

class AngleLPF:
    def __init__(self, fc_hz=1.0):
        self.fc = float(fc_hz); self.y = 0.0; self.t_last=None
    def update(self, target_deg, t):
        if self.t_last is None:
            self.t_last=t; self.y=float(target_deg); return self.y
        dt=max(1e-3, t-self.t_last)
        tau=1.0/(2.0*math.pi*self.fc)
        alpha=dt/(tau+dt)
        err=((target_deg - self.y + 180.0) % 360.0) - 180.0
        self.y=((self.y + alpha*err + 180.0) % 360.0) - 180.0
        self.t_last=t
        return self.y

# ───────────────────────────────────────────────────────────────────
# 전역 상태
# ───────────────────────────────────────────────────────────────────
gps_q = deque(maxlen=100)
pos_buf = deque(maxlen=SPEED_BUF_LEN*2)
speed_buf = deque(maxlen=SPEED_BUF_LEN)

pub_speed = pub_steer = pub_rtk = pub_wpidx = pub_grade = None
rtk_txt = "NONE"
last_fix_wall = 0.0

seq_idx = 0
latest_steer_deg = 0.0

log_csv_path = None
_last_log_wall = 0.0

_state = dict(wp_in_radius=-1, last_pub_speed=0.0, grade_val=0, speed_cur=0, speed_des=0)

# ───────────────────────────────────────────────────────────────────
# 콜백/퍼블리시
# ───────────────────────────────────────────────────────────────────
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

    v_out  = 0.0 if no_gps else float(_state.get('speed_cur', 0.0))
    s_out  = 0.0 if no_gps else float(latest_steer_deg)

    if pub_speed: pub_speed.publish(Float32(v_out))
    if pub_steer: pub_steer.publish(Float32(s_out))
    if pub_rtk:   pub_rtk.publish(String(rtk_txt))

    wp_pub = (_state['wp_in_radius']+1) if _state['wp_in_radius']>=0 else 0
    if pub_wpidx: pub_wpidx.publish(Int32(int(wp_pub)))

    if pub_grade: pub_grade.publish(Int32(int(_state.get('grade_val', 0))))

    _state['last_pub_speed'] = v_out

# ───────────────────────────────────────────────────────────────────
# 메인
# ───────────────────────────────────────────────────────────────────
def main():
    global pub_speed, pub_steer, pub_rtk, pub_wpidx, pub_grade
    global latest_steer_deg, log_csv_path, seq_idx
    global hold_active, hold_until, hold_reason

    rospy.init_node('path_follower_seq', anonymous=False)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # 파라미터
    fs        = float(rospy.get_param('~fs', FS_DEFAULT))
    base_code = int(rospy.get_param('~speed_code', BASE_SPEED_CODE_DEFAULT))
    cap_code  = int(rospy.get_param('~speed_cap_code', GLOBAL_SPEED_CAP_DEFAULT))
    step_glob = int(rospy.get_param('~step_per_loop', STEP_PER_LOOP_DEFAULT))

    ublox_ns  = rospy.get_param('~ublox_ns', '/gps1')
    fix_topic = rospy.get_param('~fix_topic', ublox_ns + '/fix')
    rel_topic = rospy.get_param('~relpos_topic', ublox_ns + '/navrelposned')

    # CSV 경로 (래퍼가 ~csv_path 또는 ~waypoint_csv로 주입 / 하드코딩 가능)
    pkg_guess   = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
    default_csv = os.path.join(pkg_guess, 'config', 'raw_track_latlon_18.csv')
    csv_path    = rospy.get_param('~csv_path', rospy.get_param('~waypoint_csv', default_csv))

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

    # 웨이포인트 로드/보간
    if not os.path.exists(csv_path):
        rospy.logerr("waypoint csv not found: %s", csv_path); return
    df = pd.read_csv(csv_path)
    assert {'Lat','Lon'}.issubset(df.columns), "CSV에 Lat,Lon 컬럼이 필요합니다."

    ref_lat, ref_lon = float(df['Lat'][0]), float(df['Lon'][0])
    to_xy = latlon_to_xy_fn(ref_lat, ref_lon)
    orig_xy = [to_xy(r['Lat'], r['Lon']) for _, r in df.iterrows()]

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

    rospy.loginfo("[flag] zones: " + ", ".join([f"{z['name']}({z['disp']})" for z in flag_zones]))

    # 초기값
    speed_cur = max(0, min(cap_code, base_code))
    _state['speed_cur'] = speed_cur
    _state['speed_des'] = speed_cur

    lpf = AngleLPF(fc_hz=LPF_FC_HZ)
    nearest_prev = 0
    last_heading = None

    # 로깅 파일
    logs_dir = os.path.join(pkg_guess, 'logs'); os.makedirs(logs_dir, exist_ok=True)
    log_csv_path = rospy.get_param('~log_csv', os.path.join(logs_dir, f"path_seq_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"))

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
                    if v <= MAX_JITTER_SPEED: speed_buf.append(v)
                pos_buf.append((tsec,x,y))
                v_mps = float(np.median(speed_buf)) if speed_buf else 0.0

                # 최근접 인덱스 (창 제한)
                nearest = nearest_idx_window(x,y,xs,ys,nearest_prev,80,15)
                nearest_prev = nearest

                # ── 플래그 상태 갱신 (반경/토픽/정지홀드) ──────────────────
                eff_radius, _, z_active, just_in, just_out = flag_update_state(
                    x, y, nearest, xs, ys, base_radius=TARGET_RADIUS_END
                )

                # 순차 인덱스: 목표 idx는 seq_idx. 도달 시 다음으로
                if seq_idx < 0: seq_idx = 0
                if math.hypot(xs[seq_idx]-x, ys[seq_idx]-y) <= eff_radius:
                    seq_idx = min(seq_idx+1, len(xs)-1)
                tgt_idx = seq_idx

                # GRADE_UP_ON 값
                if z_active is not None and (z_active.get('grade_topic') is not None):
                    _state['grade_val'] = 1 if int(z_active['grade_topic'])!=0 else 0
                else:
                    _state['grade_val'] = 0

                # 속도 코드 목표/램핑 (홀드 중이면 0)
                if hold_active and time.time() < hold_until:
                    des  = 0
                    cap  = 0
                    step = STEP_PER_LOOP_DEFAULT
                else:
                    # 홀드 해제
                    if hold_active and time.time() >= hold_until:
                        hold_active = False
                        hold_reason = ""
                    if z_active is not None and (z_active.get('speed_code') is not None):
                        des  = int(z_active['speed_code'])
                        cap  = int(z_active.get('speed_cap', cap_code if cap_code is not None else GLOBAL_SPEED_CAP_DEFAULT))
                        step = int(z_active.get('step_per_loop', step_glob))
                    else:
                        des, cap, step = int(base_code), int(cap_code), int(step_glob)

                des = max(0, min(cap if cap is not None else GLOBAL_SPEED_CAP_DEFAULT, des))
                if _state['speed_cur'] < des:
                    _state['speed_cur'] = min(des, _state['speed_cur'] + max(1, step))
                elif _state['speed_cur'] > des:
                    _state['speed_cur'] = max(des, _state['speed_cur'] - max(1, step))
                _state['speed_des'] = des

                # 반경 내 인덱스(퍼블리시용)
                wp_in = (tgt_idx if math.hypot(xs[tgt_idx]-x, ys[tgt_idx]-y) <= eff_radius else -1)
                _state['wp_in_radius'] = wp_in

                # 진행 헤딩(최근 이동 벡터) → 조향 (LPF, Pure Pursuit 금지)
                heading=None
                for k in range(2, min(len(pos_buf), 6)):
                    t0,x0,y0 = pos_buf[-k]
                    if math.hypot(x-x0,y-y0) >= MIN_MOVE_FOR_HEADING:
                        heading=(x-x0, y-y0); break
                if heading is None:
                    latest_steer_deg = 0.0
                else:
                    tx,ty = xs[tgt_idx], ys[tgt_idx]
                    raw = angle_between(heading, (tx-x, ty-y))
                    raw = float(np.clip(raw, -MAX_STEER_DEG, +MAX_STEER_DEG))
                    filt = lpf.update(raw, tsec)
                    latest_steer_deg = SIGN_CONVENTION * float(np.clip(filt, -MAX_STEER_DEG, +MAX_STEER_DEG))

                # ── 로깅 (0.5s) ───────────────────────────────
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
                                        int(_state['speed_cur']), int(_state['speed_des']),
                                        rtk_txt, (z_active['name'] if z_active else "")])
                        globals()['_last_log_wall'] = noww
                    except Exception as e:
                        rospy.logwarn(f"[path] log write failed: {e}")

                # 콘솔 상태
                rospy.loginfo_throttle(
                    0.5,
                    f"[SEQ] tgt={tgt_idx+1} wp_in={(wp_in+1) if wp_in>=0 else 0} | "
                    f"v={v_mps:.2f} m/s | code {_state['speed_cur']}->{_state['speed_des']} | "
                    f"steer={latest_steer_deg:+.1f}° | RTK={rtk_txt} | "
                    f"flag={(z_active['name'] if z_active else '')} | GUP={_state['grade_val']} "
                    f"{'(HOLD:'+hold_reason+')' if hold_active else ''}"
                )

            rate.sleep()
    finally:
        # 안전 종료
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
