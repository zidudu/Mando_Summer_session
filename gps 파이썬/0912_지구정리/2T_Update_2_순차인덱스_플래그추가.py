#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
경로추종(정리판+플래그+시각화/로깅)
- 알고리즘: 진행벡터 vs 타깃벡터 각도 → 1차 LPF(dt,tau) → '부호 반전 유지' → ±제한
- 인덱스: 순차 인덱스 (도달반경 내 진입 시 +1), 플래그 활성 시 반경/속도/grade 제어
- 토픽:
  /gps/speed_cmd   (std_msgs/Float32) : 속도 코드(램핑)
  /gps/steer_cmd   (std_msgs/Float32) : 조향 각도(deg, LPF+부호반전 후)
  /gps/status      (std_msgs/String)  : RTK 상태(FIXED/FLOAT/NONE)
  /gps/wp_index    (std_msgs/Int32)   : 현재 위치가 반경 안에 들어간 WP (1-based, 없으면 0)
  /gps/GRADEUP_ON  (std_msgs/Int32)   : 플래그 기반 0/1
- 로깅(csv): time, lat, lon, wp_in_idx(1based), tgt_idx(1based), steer_deg,
             meas_speed_mps, speed_cur_code, speed_des_code, rtk_status, flag_name, grade_topic
- 시각화: CSV/웨이포인트(간격축소), 현재 궤적, 현재점, 타깃 라인, HUD/범례
"""

import os, math, time, signal, csv, queue
from collections import deque

import rospy
import numpy as np
import pandas as pd

from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, String, Int32

# ──────────────────────────────────────────────────────────────────
# Matplotlib (옵션)
# ──────────────────────────────────────────────────────────────────
_HAVE_MPL = False
try:
    import matplotlib
    # GUI 백엔드 (환경에 맞게 변경 가능)
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

# ──────────────────────────────────────────────────────────────────
# u-blox RELPOSNED(선택)
# ──────────────────────────────────────────────────────────────────
_HAVE_RELPOSNED = False
try:
    from ublox_msgs.msg import NavRELPOSNED9 as NavRELPOSNED
    _HAVE_RELPOSNED = True
except Exception:
    try:
        from ublox_msgs.msg import NavRELPOSNED
        _HAVE_RELPOSNED = True
    except Exception:
        _HAVE_RELPOSNED = False

# ──────────────────────────────────────────────────────────────────
# 상수/파라미터 기본값
# ──────────────────────────────────────────────────────────────────
# 좌표/웨이포인트
EARTH_RADIUS      = 6378137.0   # Web Mercator 근사(소구역용)
MIN_WP_SPACING    = 2.8         # 웨이포인트 최소 간격 축소(m)
TARGET_RADIUS_END = 2.0         # 도달 판정 반경(플래그로 가변)

# 조향/필터
ANGLE_LIMIT_DEG   = 27.0        # 조향 제한(±)
DT_DEFAULT        = 0.05        # [s] 50ms
TAU_DEFAULT       = 0.25        # [s] 250ms
INVERT_AFTER_LPF  = True        # '부호 반전 유지' 기본 ON
SIGN_CONVENTION   = 1.0         # 환경에 따라 ±1 조정

# 속도/헤딩/루프
SPEED_BUF_LEN        = 10
MAX_JITTER_SPEED     = 4.0
MIN_MOVE_FOR_HEADING = 0.05
FS_DEFAULT           = 20.0
GPS_TIMEOUT_SEC      = 1.0

# 속도 코드(램핑)
BASE_SPEED_CODE_DEFAULT  = 5
GLOBAL_SPEED_CAP_DEFAULT = 10
STEP_PER_LOOP_DEFAULT    = 2

# 토픽
TOPIC_SPEED_CMD   = '/gps/speed_cmd'
TOPIC_STEER_CMD   = '/gps/steer_cmd'
TOPIC_RTK_STATUS  = '/gps/status'
TOPIC_WP_INDEX    = '/gps/wp_index'
TOPIC_GRADE_UP_ON = '/gps/GRADEUP_ON'

# ──────────────────────────────────────────────────────────────────
# 플래그 FSM 정의 (필요시 수정 가능)
# ──────────────────────────────────────────────────────────────────
STOP_FLAG_STAY_SEC = 3.0
FLAG_DEFS = [
    {'name':'GRADE_START','start':3,'end':4,'radius_scale':1.0,'speed_code':5,'speed_cap':7,'step_per_loop':2,'stop_on_hit':False,'stop_duration_sec':None,'grade_topic':1},
    {'name':'GRADE_UP',   'start':5,'end':5,'radius_scale':0.3,'speed_code':0,'speed_cap':0,'step_per_loop':2,'stop_on_hit':True, 'stop_duration_sec':3,  'grade_topic':1},
    {'name':'GRADE_GO',   'start':6,'end':8,'radius_scale':1.0,'speed_code':5,'speed_cap':10,'step_per_loop':2,'stop_on_hit':False,'stop_duration_sec':None,'grade_topic':1},
    {'name':'GRADE_END',  'start':9,'end':10,'radius_scale':1.0,'speed_code':5,'speed_cap':10,'step_per_loop':2,'stop_on_hit':False,'stop_duration_sec':None,'grade_topic':0},
]
def _build_flag_zones(defs):
    zs=[]
    for d in defs:
        s0=min(d['start'],d['end'])-1; e0=max(d['start'],d['end'])-1
        zs.append(dict(
            name=d['name'], start0=s0, end0=e0, disp=f"{d['start']}–{d['end']}",
            radius_scale=float(d.get('radius_scale',1.0)),
            speed_code=d.get('speed_code',None),
            speed_cap=d.get('speed_cap',None),
            step_per_loop=int(d.get('step_per_loop',STEP_PER_LOOP_DEFAULT)),
            stop_on_hit=bool(d.get('stop_on_hit',False)),
            stop_duration_sec=float(d.get('stop_duration_sec',STOP_FLAG_STAY_SEC)) if d.get('stop_duration_sec',None) is not None else STOP_FLAG_STAY_SEC,
            grade_topic=d.get('grade_topic',None)
        ))
    return zs
FLAG_ZONES   = _build_flag_zones(FLAG_DEFS)
_active_flag = None
_hold_active = False
_hold_until  = 0.0
_hold_reason = ""

def _in_radius_idx(x,y,idx,xs,ys,r):
    return (math.hypot(xs[idx]-x, ys[idx]-y) <= r)

def flag_update_state(x,y,nearest,xs,ys,base_r):
    """
    진입: start0 반경, 종료: end0를 지나 반경 벗어나면 off.
    활성 중에는 eff_radius=base_r*radius_scale, grade_topic/stop_on_hit/속도 파라미터 반영.
    """
    global _active_flag,_hold_active,_hold_until,_hold_reason
    just_in=False; just_out=False
    if _active_flag is None:
        for z in FLAG_ZONES:
            if _in_radius_idx(x,y,z['start0'],xs,ys,base_r):
                _active_flag=z; just_in=True; rospy.loginfo(f"[flag] ENTER {z['name']} {z['disp']}")
                if z['stop_on_hit']:
                    _hold_active=True; _hold_until=time.time()+z['stop_duration_sec']; _hold_reason=z['name']
                break
    else:
        z=_active_flag
        at_end=_in_radius_idx(x,y,z['end0'],xs,ys,base_r)
        if (not at_end) and (nearest>=z['end0']):
            rospy.loginfo(f"[flag] EXIT {_active_flag['name']} {_active_flag['disp']}")
            _active_flag=None; just_out=True
    eff=base_r
    if _active_flag is not None:
        eff=base_r*float(_active_flag.get('radius_scale',1.0))
    return eff,_active_flag,just_in,just_out

# ──────────────────────────────────────────────────────────────────
# 좌표/웨이포인트 유틸 (정리판: 컬럼 자동인식 + 간격축소)
# ──────────────────────────────────────────────────────────────────
def lat_lon_to_meters(lat, lon):
    x = EARTH_RADIUS * math.radians(lon)
    y = EARTH_RADIUS * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    return x, y

def _pick_column(cols, cands):
    for c in cands:
        if c in cols: return c
    return None

def build_reduced_waypoints(csv_path):
    csv_path = os.path.expanduser(csv_path)
    df = pd.read_csv(csv_path, encoding='utf-8-sig', engine='python', sep=None)
    df.columns = [str(c).strip().lower() for c in df.columns]
    cols = set(df.columns)

    lat_col  = _pick_column(cols, ['latitude','lat'])
    lon_col  = _pick_column(cols, ['longitude','lon'])
    east_col = _pick_column(cols, ['east','easting','x'])
    north_col= _pick_column(cols, ['north','northing','y'])

    if lat_col and lon_col:
        lat = pd.to_numeric(df[lat_col], errors='coerce').to_numpy()
        lon = pd.to_numeric(df[lon_col], errors='coerce').to_numpy()
        pts = [lat_lon_to_meters(a, b) for a, b in zip(lat, lon)]
        xs, ys = list(zip(*pts))
        rospy.loginfo("Waypoints from lat/lon columns: (%s, %s)", lat_col, lon_col)
    elif east_col and north_col:
        xs = pd.to_numeric(df[east_col],  errors='coerce').to_numpy().tolist()
        ys = pd.to_numeric(df[north_col], errors='coerce').to_numpy().tolist()
        rospy.loginfo("Waypoints from east/north columns: (%s, %s)", east_col, north_col)
    else:
        available = ", ".join(sorted(cols))
        raise ValueError("CSV 좌표 컬럼을 찾지 못했습니다. 현재 컬럼: " + available)

    rx=[float(xs[0])]; ry=[float(ys[0])]
    for x,y in zip(xs[1:], ys[1:]):
        if math.hypot(float(x)-rx[-1], float(y)-ry[-1]) >= MIN_WP_SPACING:
            rx.append(float(x)); ry.append(float(y))
    return np.array(rx), np.array(ry)  # shape=(N,)

# 근접/반경 인덱스
def find_nearest_index(x, y, xs, ys, start, fwd=80, back=15):
    i0=max(0,start-back); i1=min(len(xs)-1,start+fwd)
    subx=np.array(xs[i0:i1+1]); suby=np.array(ys[i0:i1+1])
    d2=(subx-x)**2+(suby-y)**2
    return i0+int(np.argmin(d2))

def nearest_in_radius_index(x, y, xs, ys, r):
    d2=(xs-x)**2+(ys-y)**2
    i=int(np.argmin(d2))
    return i if math.hypot(xs[i]-x, ys[i]-y)<=r else -1

# 각도
def signed_angle_deg(v1, v2):
    n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
    if n1==0 or n2==0: return 0.0
    dot=float(np.dot(v1,v2))/(n1*n2); dot=max(min(dot,1.0),-1.0)
    ang=math.degrees(math.acos(dot))
    cross=v1[0]*v2[1]-v1[1]*v2[0]
    return ang if cross>=0 else -ang

# ──────────────────────────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────────────────────────
gps_queue = queue.Queue()
pos_buf   = deque(maxlen=SPEED_BUF_LEN*2)
speed_buf = deque(maxlen=SPEED_BUF_LEN)

pub_speed = pub_steer = pub_rtk = pub_wpidx = pub_grade = None
rtk_status_txt = "NONE"
_last_fix_wall = 0.0

# 시각화 공유 상태
latest_filtered_angle = 0.0   # 퍼블리시 직전 조향각
_filtered_internal    = 0.0   # LPF 내부 상태(부호반전 전)
wp_index_active       = -1
last_pub_speed_code   = 0.0

# 경로/제어
xs_way = ys_way = None
seq_idx = 0
_nearest_idx_prev = 0
_last_heading_vec = None

# 필터 파라미터
DT   = DT_DEFAULT
TAU  = TAU_DEFAULT
ALPHA = DT/(TAU+DT)

# 로깅
log_csv_path = None
_last_log_wall = 0.0

# 플래그 상태
grade_val = 0
des_cached = BASE_SPEED_CODE_DEFAULT  # 로깅용(des)

# ──────────────────────────────────────────────────────────────────
# ROS 콜백/퍼블리셔
# ──────────────────────────────────────────────────────────────────
def gps_callback(msg: NavSatFix):
    global _last_fix_wall
    if hasattr(msg,'status') and getattr(msg.status,'status',0)<0:
        return
    lat,lon=float(msg.latitude),float(msg.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)): return
    t = msg.header.stamp.to_sec() if (msg.header and msg.header.stamp) else rospy.Time.now().to_sec()
    gps_queue.put((lat,lon,t))
    _last_fix_wall=time.time()

def _cb_relpos(msg):
    global rtk_status_txt
    try:
        carr=int((int(msg.flags)>>3)&0x3)
        rtk_status_txt = "FIXED" if carr==2 else ("FLOAT" if carr==1 else "NONE")
    except Exception:
        rtk_status_txt="NONE"

def publish_all(_evt):
    """퍼블리시 주기 타이머 → 마지막 계산값을 퍼블리시"""
    global last_pub_speed_code
    now=time.time()
    no_gps=(now-_last_fix_wall)>GPS_TIMEOUT_SEC
    v_out=0.0 if no_gps else float(last_pub_speed_code)  # speed_cmd_cur를 last_pub_speed_code로 보관
    s_out=0.0 if no_gps else float(latest_filtered_angle)
    if pub_speed: pub_speed.publish(Float32(v_out))
    if pub_steer: pub_steer.publish(Float32(s_out))
    if pub_rtk:   pub_rtk.publish(String(rtk_status_txt))
    if pub_wpidx: pub_wpidx.publish(Int32(int((wp_index_active+1) if wp_index_active>=0 else 0)))
    if pub_grade: pub_grade.publish(Int32(int(grade_val)))

# ──────────────────────────────────────────────────────────────────
# LPF + '부호 반전 유지' + SIGN_CONVENTION + 제한
# ──────────────────────────────────────────────────────────────────
def lpf_and_invert(angle_deg):
    global _filtered_internal
    _filtered_internal = (1.0-ALPHA)*_filtered_internal + ALPHA*float(angle_deg)
    val = -_filtered_internal if INVERT_AFTER_LPF else _filtered_internal
    val = SIGN_CONVENTION * val
    return max(-ANGLE_LIMIT_DEG, min(ANGLE_LIMIT_DEG, val))

# ──────────────────────────────────────────────────────────────────
# 메인 제어루프(시각화 애니메이션 타이머에서 호출)
# ──────────────────────────────────────────────────────────────────
def _anim_update(_frame):
    """GPS 큐 처리 → 순차 인덱스 전진 → 플래그/속도 램핑 → 조향 계산/LPF → 시각화/로깅"""
    global latest_filtered_angle, wp_index_active, seq_idx, grade_val, last_pub_speed_code, des_cached
    global _nearest_idx_prev, _last_heading_vec

    updated=False
    speed_mps=0.0
    lat=lon=x=y=float('nan')

    # 큐 비우면서 최신 상태 반영
    while not gps_queue.empty():
        lat,lon,tsec=gps_queue.get()
        x,y = lat_lon_to_meters(lat, lon)
        updated=True

        # 속도 추정
        if len(pos_buf)>0:
            t_prev,x_prev,y_prev=pos_buf[-1]
            dt=max(1e-3,tsec-t_prev); d=math.hypot(x-x_prev,y-y_prev)
            inst_v=d/dt
            if inst_v>MAX_JITTER_SPEED:
                continue
            speed_buf.append(inst_v)
        pos_buf.append((tsec,x,y))
        speed_mps=float(np.median(speed_buf)) if speed_buf else 0.0

        # 최근접 인덱스(플래그 종료 조건에서 사용)
        nearest_idx = find_nearest_index(x,y,xs_way,ys_way,_nearest_idx_prev,80,15)
        _nearest_idx_prev = nearest_idx

        # ── 플래그 상태/효과 ──
        eff_radius, z_active, _, _ = flag_update_state(x,y,nearest_idx,xs_way,ys_way,TARGET_RADIUS_END)

        # 순차 인덱스 전진(도달 시 +1)
        if seq_idx<0: seq_idx=0
        if math.hypot(xs_way[seq_idx]-x, ys_way[seq_idx]-y) <= eff_radius:
            seq_idx = min(seq_idx+1, len(xs_way)-1)

        tgt_idx = seq_idx
        tx,ty = xs_way[tgt_idx], ys_way[tgt_idx]

        # 반경 내 인덱스(퍼블리시용 1-based, 없으면 0)
        wp_index_active = nearest_in_radius_index(x,y,xs_way,ys_way,eff_radius)

        # ── 속도 코드 램핑 ──
        base_code = int(rospy.get_param('~speed_code', BASE_SPEED_CODE_DEFAULT))
        cap_code  = int(rospy.get_param('~speed_cap_code', GLOBAL_SPEED_CAP_DEFAULT))
        step_glob = int(rospy.get_param('~step_per_loop', STEP_PER_LOOP_DEFAULT))

        if _hold_active and time.time()<_hold_until:
            des,cap,step = 0,0,STEP_PER_LOOP_DEFAULT
        else:
            if _hold_active and time.time()>=_hold_until:
                globals()['_hold_active']=False; globals()['_hold_reason']=""
            if (z_active is not None) and (z_active.get('speed_code') is not None):
                des  = int(z_active['speed_code'])
                cap  = int(z_active.get('speed_cap', cap_code))
                step = int(z_active.get('step_per_loop', step_glob))
            else:
                des,cap,step = int(base_code), int(cap_code), int(step_glob)
        des = max(0, min(cap, des))

        # 현재 퍼블리시 값(last_pub_speed_code)을 des로 부드럽게 이동
        cur = int(last_pub_speed_code)
        if cur < des:  cur = min(des, cur + max(1, step))
        elif cur > des: cur = max(des, cur - max(1, step))
        last_pub_speed_code = float(cur)  # Timer에서 퍼블

        des_cached = des  # 로깅용(des)

        # ── grade 토픽 ──
        if (z_active is not None) and (z_active.get('grade_topic') is not None):
            grade_val = 1 if int(z_active['grade_topic'])!=0 else 0
        else:
            grade_val = 0

        # ── 조향 계산(정리판) ──
        heading_vec=None
        for k in range(2, min(len(pos_buf),5)+1):
            t0,x0,y0=pos_buf[-k]
            if math.hypot(x-x0,y-y0) >= MIN_MOVE_FOR_HEADING:
                heading_vec=(x-x0, y-y0); break
        if heading_vec is None:
            continue

        raw_angle = signed_angle_deg(heading_vec, (tx-x, ty-y))
        raw_angle = max(-ANGLE_LIMIT_DEG, min(ANGLE_LIMIT_DEG, raw_angle))
        latest_filtered_angle = lpf_and_invert(raw_angle)

        # ── 시각화 갱신 ──
        if _ENABLE_GUI and _HAVE_MPL:
            _live_line.set_data([p[1] for p in pos_buf], [p[2] for p in pos_buf])
            _current_pt.set_data([x],[y])
            _target_line.set_data([x,tx],[y,ty])

            heading_deg = math.degrees(math.atan2(heading_vec[1], heading_vec[0])) if heading_vec else float('nan')
            wp_disp = (wp_index_active+1) if wp_index_active>=0 else 0

            _ax_info.clear(); _ax_info.axis('off')
            status = [
                f"Meas v: {speed_mps:.2f} m/s",
                f"Pub v(code): {last_pub_speed_code:.1f}→des:{des_cached}",
                f"Steer: {latest_filtered_angle:+.1f}°",
                f"Heading: {heading_deg:.1f}°",
                f"WP(in-radius): {wp_disp}",
                f"RTK: {rtk_status_txt}",
                f"FLAG: {(z_active['name'] if z_active else '')} / grade:{grade_val}"
            ]
            _ax_info.text(0.02, 0.5, " | ".join(status), fontsize=11, va='center')

            _hud_text.set_text(
                f"pub_v={last_pub_speed_code:.1f} | meas_v={speed_mps:.2f} m/s | "
                f"steer={latest_filtered_angle:+.1f}° | WP={wp_disp}"
            )

        # ── 로깅(0.5s) ──
        noww=time.time()
        if log_csv_path and (noww-_last_log_wall>0.5):
            try:
                new=not os.path.exists(log_csv_path)
                os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
                with open(log_csv_path,'a',newline='') as f:
                    w=csv.writer(f)
                    if new:
                        w.writerow(['time','lat','lon',
                                    'wp_in_idx(1based)','tgt_idx(1based)',
                                    'steer_deg','meas_speed_mps',
                                    'speed_cur_code','speed_des_code',
                                    'rtk_status','flag_name','grade_topic'])
                    wp_disp=(wp_index_active+1) if wp_index_active>=0 else 0
                    w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                f"{lat:.7f}", f"{lon:.7f}",
                                wp_disp, (tgt_idx+1),
                                f"{latest_filtered_angle:.2f}", f"{speed_mps:.2f}",
                                int(last_pub_speed_code), int(des_cached),
                                rtk_status_txt, (z_active['name'] if z_active else ""), int(grade_val)])
                globals()['_last_log_wall']=noww
            except Exception as e:
                rospy.logwarn(f"[seq] log write failed: {e}")

        rospy.loginfo_throttle(0.5,
            f"[SEQ] WP={(wp_index_active+1) if wp_index_active>=0 else 0} | "
            f"v={speed_mps:.2f} | pub={last_pub_speed_code}→{des_cached} | "
            f"steer={latest_filtered_angle:+.1f}° | RTK={rtk_status_txt} | FLAG={(z_active['name'] if z_active else '')} | GUP={grade_val}"
        )
    return

# ──────────────────────────────────────────────────────────────────
# 창/종료 핸들러
# ──────────────────────────────────────────────────────────────────
def _on_close(_evt):
    rospy.signal_shutdown("Figure closed by user")

def _on_shutdown():
    try:
        for _ in range(3):
            if pub_speed: pub_speed.publish(Float32(0.0))
            if pub_steer: pub_steer.publish(Float32(0.0))
            if pub_grade: pub_grade.publish(Int32(0))
            time.sleep(0.02)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────
def main():
    global pub_speed, pub_steer, pub_rtk, pub_wpidx, pub_grade
    global xs_way, ys_way, seq_idx
    global _fig, _ax, _ax_info, _live_line, _current_pt, _target_line, _hud_text
    global _ENABLE_GUI, log_csv_path, DT, TAU, ALPHA, INVERT_AFTER_LPF, SIGN_CONVENTION

    rospy.init_node('path_follower_seq', anonymous=False)
    rospy.on_shutdown(_on_shutdown)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # 파라미터
    pkg_guess   = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
    default_csv = os.path.join(pkg_guess, 'config', 'raw_track_latlon_18.csv')
    waypoint_csv= rospy.get_param('~csv_path', rospy.get_param('~waypoint_csv', default_csv))
    logs_dir    = os.path.join(pkg_guess, 'logs'); os.makedirs(logs_dir, exist_ok=True)
    log_csv_path= rospy.get_param('~log_csv', os.path.join(logs_dir, f"path_seq_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"))

    fs          = float(rospy.get_param('~fs', FS_DEFAULT))
    ublox_ns    = rospy.get_param('~ublox_ns', '/gps1')
    fix_topic   = rospy.get_param('~fix_topic',    ublox_ns + '/fix')
    rel_topic   = rospy.get_param('~relpos_topic', ublox_ns + '/navrelposned')

    _ENABLE_GUI = bool(rospy.get_param('~enable_gui', True))
    annotate_wp = bool(rospy.get_param('~annotate_wp_index', False))
    draw_circ   = bool(rospy.get_param('~draw_wp_circles', True))

    # LPF/부호 파라미터
    DT   = float(rospy.get_param('~dt',  DT_DEFAULT))
    TAU  = float(rospy.get_param('~tau', TAU_DEFAULT))
    ALPHA = DT/(TAU+DT)
    INVERT_AFTER_LPF = bool(rospy.get_param('~invert_after_lpf', INVERT_AFTER_LPF))
    SIGN_CONVENTION  = float(rospy.get_param('~sign_convention', SIGN_CONVENTION))
    rospy.loginfo("LPF params: dt=%.3fs, tau=%.3fs, alpha=%.3f (invert_after_lpf=%s, sign=%.1f)",
                  DT, TAU, ALPHA, INVERT_AFTER_LPF, SIGN_CONVENTION)

    # 퍼블리셔/구독
    pub_speed = rospy.Publisher(TOPIC_SPEED_CMD,   Float32, queue_size=10)
    pub_steer = rospy.Publisher(TOPIC_STEER_CMD,   Float32, queue_size=10)
    pub_rtk   = rospy.Publisher(TOPIC_RTK_STATUS,  String,  queue_size=10)
    pub_wpidx = rospy.Publisher(TOPIC_WP_INDEX,    Int32,   queue_size=10)
    pub_grade = rospy.Publisher(TOPIC_GRADE_UP_ON, Int32,   queue_size=10)

    rospy.Subscriber(fix_topic, NavSatFix, gps_callback, queue_size=100)
    if _HAVE_RELPOSNED:
        rospy.Subscriber(rel_topic, NavRELPOSNED, _cb_relpos, queue_size=50)

    # 웨이포인트 로드(정리판: 간격축소)
    if not os.path.exists(waypoint_csv):
        rospy.logerr("[seq] waypoint csv not found: %s", waypoint_csv); return
    try:
        xs_way, ys_way = build_reduced_waypoints(waypoint_csv)
    except Exception as e:
        rospy.logerr("[seq] waypoint csv parse failed: %s", e); return
    seq_idx = 0

    rospy.loginfo("Waypoints loaded(reduced): %d (>= %.1fm spacing)", len(xs_way), MIN_WP_SPACING)

    # ── 플롯 구성 ──
    if _ENABLE_GUI and _HAVE_MPL:
        global _fig,_ax,_ax_info,_live_line,_current_pt,_target_line,_hud_text
        _fig = plt.figure(figsize=(7.8, 9.0))
        gs = _fig.add_gridspec(2,1,height_ratios=[4,1])
        _ax = _fig.add_subplot(gs[0,0])
        _ax_info = _fig.add_subplot(gs[1,0]); _ax_info.axis('off')

        _ax.plot(xs_way, ys_way, 'b.-', markersize=3, label=f'>={MIN_WP_SPACING:.1f}m Waypoints')
        _live_line,  = _ax.plot([], [], 'r-', linewidth=1, label='Live GPS')
        _current_pt, = _ax.plot([], [], 'ro',              label='Current')
        _target_line,= _ax.plot([], [], 'g--', linewidth=1, label='Target Line')
        _ax.axis('equal'); _ax.grid(True); _ax.legend()

        _hud_text = _ax.text(0.98, 0.02, "",
                             transform=_ax.transAxes, ha='right', va='bottom',
                             fontsize=9, bbox=dict(fc='white', alpha=0.75, ec='0.5'))

        minx=min(xs_way)-10; maxx=max(xs_way)+10
        miny=min(ys_way)-10; maxy=max(ys_way)+10
        _ax.set_xlim(minx,maxx); _ax.set_ylim(miny,maxy)

        if annotate_wp or draw_circ:
            for i,(xw,yw) in enumerate(zip(xs_way, ys_way), 1):
                if annotate_wp:
                    _ax.text(xw,yw,str(i),fontsize=7,ha='center',va='center',color='black')
                if draw_circ:
                    _ax.add_patch(Circle((xw,yw), TARGET_RADIUS_END, color='blue', fill=False, linestyle='--', alpha=0.35))

        _fig.canvas.mpl_connect('close_event', _on_close)

    # 퍼블리시 타이머 (fs Hz)
    rospy.Timer(rospy.Duration(1.0/max(1.0,fs)), publish_all)

    # 애니메이션/헤드리스 루프
    if _ENABLE_GUI and _HAVE_MPL:
        interval_ms = int(1000.0/max(1.0,fs))
        animation.FuncAnimation(_fig, _anim_update, interval=interval_ms, blit=False)
        plt.show()
    else:
        rospy.loginfo("[seq] Headless mode")
        rospy.Timer(rospy.Duration(1.0/max(1.0,fs)), lambda e: _anim_update(None))
        rospy.spin()

if __name__ == '__main__':
    main()
