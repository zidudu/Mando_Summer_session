#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
경로추종.py — 순차 인덱스 + 조향 LPF + (내장) 플래그 + 실시간 시각화/범례
- Pure Pursuit 사용 금지 (복귀 전용 노드에서만 사용)
- 토픽: /gps/speed_cmd, /gps/steer_cmd, /gps/status, /gps/wp_index, /gps/GRADEUP_ON
- 로그: time, lat, lon, wp_in_idx(1based), tgt_idx(1based), steer_deg,
        meas_speed_mps, speed_cur_code, speed_des_code, rtk_status, flag_name, grade_topic
"""

import os, csv, math, time, signal, queue
from collections import deque

import rospy
import numpy as np
import pandas as pd
import geopy.distance

from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, String, Int32

# ── optional plotting (참고 코드 스타일) ─────────────────────────────
_HAVE_MPL = False
try:
    import matplotlib
    matplotlib.use('TkAgg')  # GUI 백엔드 (환경에 따라 변경 가능)
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

# ──────────────────────────────────────────────────────────────────
# 공통 상수/토픽
# ──────────────────────────────────────────────────────────────────
TOPIC_SPEED_CMD   = '/gps/speed_cmd'
TOPIC_STEER_CMD   = '/gps/steer_cmd'
TOPIC_RTK_STATUS  = '/gps/status'
TOPIC_WP_INDEX    = '/gps/wp_index'
TOPIC_GRADE_UP_ON = '/gps/GRADEUP_ON'

WAYPOINT_SPACING       = 2.5   # m 보간 간격
TARGET_RADIUS_END      = 2.0   # m 도달 판정 반경 (플래그로 가변 eff_radius)
MAX_STEER_DEG          = 27.0  # deg
SIGN_CONVENTION        = -1.0  # 좌/우 반전 필요 시 -1.0

LPF_FC_HZ              = 0.8   # 조향 LPF 차단 주파수
SPEED_BUF_LEN          = 10
MAX_JITTER_SPEED       = 4.0
MIN_MOVE_FOR_HEADING   = 0.05
FS_DEFAULT             = 20.0
GPS_TIMEOUT_SEC        = 1.0

BASE_SPEED_CODE_DEFAULT  = 5
GLOBAL_SPEED_CAP_DEFAULT = 10
STEP_PER_LOOP_DEFAULT    = 2

# ──────────────────────────────────────────────────────────────────
# (내장) 플래그 알고리즘 (복귀코드_플래그.py 참조하여 통합)
# ──────────────────────────────────────────────────────────────────
STOP_FLAG_STAY_SEC = 3.0
FLAG_DEFS = [
    {'name':'GRADE_START','start':3,'end':4,'radius_scale':1.0,'lookahead_scale':0.95,'speed_code':5,'speed_cap':7,'step_per_loop':2,'stop_on_hit':False,'stop_duration_sec':None,'grade_topic':1},
    {'name':'GRADE_UP',   'start':5,'end':5,'radius_scale':0.3,'lookahead_scale':0.95,'speed_code':0,'speed_cap':0,'step_per_loop':2,'stop_on_hit':True,'stop_duration_sec':3,'grade_topic':1},
    {'name':'GRADE_GO',   'start':6,'end':8,'radius_scale':1.0,'lookahead_scale':0.95,'speed_code':5,'speed_cap':10,'step_per_loop':2,'stop_on_hit':False,'stop_duration_sec':None,'grade_topic':1},
    {'name':'GRADE_END',  'start':9,'end':10,'radius_scale':1.0,'lookahead_scale':0.95,'speed_code':5,'speed_cap':10,'step_per_loop':2,'stop_on_hit':False,'stop_duration_sec':None,'grade_topic':0},
]
def _build_flag_zones(defs):
    zs=[]
    for d in defs:
        s0=min(d['start'],d['end'])-1; e0=max(d['start'],d['end'])-1
        zs.append(dict(name=d['name'],start0=s0,end0=e0,
                       radius_scale=float(d.get('radius_scale',1.0)),
                       lookahead_scale=float(d.get('lookahead_scale',1.0)),
                       speed_code=d.get('speed_code',None),
                       speed_cap=d.get('speed_cap',None),
                       step_per_loop=int(d.get('step_per_loop',STEP_PER_LOOP_DEFAULT)),
                       stop_on_hit=bool(d.get('stop_on_hit',False)),
                       stop_duration_sec=float(d.get('stop_duration_sec',STOP_FLAG_STAY_SEC)) if d.get('stop_duration_sec',None) is not None else STOP_FLAG_STAY_SEC,
                       grade_topic=d.get('grade_topic',None),
                       disp=f"{d['start']}–{d['end']}"))
    return zs
FLAG_ZONES   = _build_flag_zones(FLAG_DEFS)
_active_flag = None
_hold_active = False
_hold_until  = 0.0
_hold_reason = ""

def _in_radius_idx(x,y,idx,xs,ys,r):
    return (math.hypot(xs[idx]-x, ys[idx]-y) <= r)

def flag_update_state(x,y,nearest,xs,ys,base_r):
    """진입: start0 반경, 종료: end0 지나 벗어나면 off. eff_radius/grade 토픽/홀드 제어."""
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
    eff=base_r; look=1.0
    if _active_flag is not None:
        eff=base_r*float(_active_flag.get('radius_scale',1.0))
        look=float(_active_flag.get('lookahead_scale',1.0))
    return eff,look,_active_flag,just_in,just_out

# ──────────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────────
def latlon_to_xy_fn(ref_lat, ref_lon):
    def _to_xy(lat,lon):
        north=geopy.distance.geodesic((ref_lat,ref_lon),(lat,ref_lon)).meters
        east =geopy.distance.geodesic((ref_lat,ref_lon),(ref_lat,lon)).meters
        if lat<ref_lat: north*=-1
        if lon<ref_lon: east*=-1
        return east,north
    return _to_xy

def generate_waypoints_along_path(path_xy, spacing=WAYPOINT_SPACING):
    out=[path_xy[0]]; acc=0.0
    for i in range(1,len(path_xy)):
        p0=np.array(out[-1]); p1=np.array(path_xy[i])
        seg=p1-p0; L=np.linalg.norm(seg)
        while acc+L>=spacing:
            rem=spacing-acc
            out.append(tuple(p0+(seg/L)*rem))
            p0=np.array(out[-1]); seg=p1-p0; L=np.linalg.norm(seg); acc=0.0
        acc+=L
    if out[-1]!=path_xy[-1]: out.append(path_xy[-1])
    return out

def find_nearest_index(x,y,xs,ys,start,fwd=80,back=15):
    i0=max(0,start-back); i1=min(len(xs)-1,start+fwd)
    subx=np.array(xs[i0:i1+1]); suby=np.array(ys[i0:i1+1])
    d2=(subx-x)**2+(suby-y)**2
    return i0+int(np.argmin(d2))

def nearest_in_radius_index(x,y,xs,ys,r):
    d2=(xs-x)**2+(ys-y)**2
    i=int(np.argmin(d2))
    return i if math.hypot(xs[i]-x,ys[i]-y)<=r else -1

def angle_between(v1,v2):
    v1=np.array(v1,float); v2=np.array(v2,float)
    n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
    if n1==0 or n2==0: return 0.0
    dot=np.clip(np.dot(v1,v2)/(n1*n2),-1.0,1.0)
    ang=math.degrees(math.acos(dot))
    if v1[0]*v2[1]-v1[1]*v2[0]<0: ang=-ang
    return ang

class AngleLPF:
    def __init__(self, fc_hz=LPF_FC_HZ): self.fc=float(fc_hz); self.y=0.0; self.t_last=None
    def update(self, target_deg, t):
        if self.t_last is None: self.t_last=t; self.y=float(target_deg); return self.y
        dt=max(1e-3, t-self.t_last); tau=1.0/(2.0*math.pi*self.fc); a=dt/(tau+dt)
        err=((target_deg-self.y+180)%360)-180
        self.y=((self.y+a*err+180)%360)-180; self.t_last=t; return self.y

# ──────────────────────────────────────────────────────────────────
# ROS 전역/상태
# ──────────────────────────────────────────────────────────────────
gps_queue = queue.Queue()
pos_buf   = deque(maxlen=SPEED_BUF_LEN*2)
speed_buf = deque(maxlen=SPEED_BUF_LEN)

pub_speed = pub_steer = pub_rtk = pub_wpidx = pub_grade = None
rtk_status_txt = "NONE"
_last_fix_wall = 0.0

# 시각화 공유 전역 (참고 코드와 동일 네이밍)
latest_filtered_angle = 0.0
wp_index_active       = -1
last_pub_speed_code   = 0.0

# 경로/좌표
_spaced_x=_spaced_y=None
_to_xy=None
_df=None

# 애니메이션용 상태
_prev_Ld = 3.5                # (경로추종에서는 사용 안 하지만 유지)
_nearest_idx_prev = 0
_last_heading_vec = None

# 플롯 객체
_ENABLE_GUI=True
_HAVE_RELPOSNED=False
_fig=_ax=_ax_info=_live_line=_current_pt=_target_line=_hud_text=None

# 로깅
log_csv_path=None
_last_log_wall=0.0

# 제어/플래그 상태
seq_idx=0
speed_cmd_cur=0
grade_val=0

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
    global last_pub_speed_code
    now=time.time()
    no_gps=(now-_last_fix_wall)>GPS_TIMEOUT_SEC
    v_out=0.0 if no_gps else float(speed_cmd_cur)
    s_out=0.0 if no_gps else float(latest_filtered_angle)
    if pub_speed: pub_speed.publish(Float32(v_out))
    if pub_steer: pub_steer.publish(Float32(s_out))
    if pub_rtk:   pub_rtk.publish(String(rtk_status_txt))
    if pub_wpidx: pub_wpidx.publish(Int32(int((wp_index_active+1) if wp_index_active>=0 else 0)))
    if pub_grade: pub_grade.publish(Int32(int(grade_val)))
    last_pub_speed_code=v_out

# ──────────────────────────────────────────────────────────────────
# 애니메이션 업데이트 함수 (참고 코드 스타일)
# ──────────────────────────────────────────────────────────────────
def _anim_update(_frame):
    """GUI 타이머마다 호출되어 화면 갱신 + 내부 상태 업데이트"""
    global latest_filtered_angle, wp_index_active, seq_idx, speed_cmd_cur, grade_val
    global _prev_Ld, _nearest_idx_prev, _last_heading_vec, _spaced_x, _spaced_y, _to_xy

    updated=False
    speed_mps=0.0
    lat=lon=x=y=float('nan')
    lpf=_anim_update.lpf

    # 큐에서 최신 GPS 모두 소비
    while not gps_queue.empty():
        lat,lon,tsec=gps_queue.get()
        x,y=_to_xy(lat,lon)
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

        # 최근접 인덱스
        nearest_idx=find_nearest_index(x,y,_spaced_x,_spaced_y,_nearest_idx_prev,80,15)
        _nearest_idx_prev=nearest_idx

        # ── 플래그 상태/효과 ──
        eff_radius, _, z_active, _, _ = flag_update_state(x,y,nearest_idx,_spaced_x,_spaced_y,TARGET_RADIUS_END)
        # 순차 인덱스 진전
        if seq_idx<0: seq_idx=0
        if math.hypot(_spaced_x[seq_idx]-x,_spaced_y[seq_idx]-y)<=eff_radius:
            seq_idx=min(seq_idx+1,len(_spaced_x)-1)
        tgt_idx=seq_idx
        tx,ty=_spaced_x[tgt_idx],_spaced_y[tgt_idx]

        # 반경 내 인덱스
        wp_index_active = nearest_in_radius_index(x,y,_spaced_x,_spaced_y,eff_radius)

        # 속도 코드(플래그/램프)
        base_code = int(rospy.get_param('~speed_code', BASE_SPEED_CODE_DEFAULT))
        cap_code  = int(rospy.get_param('~speed_cap_code', GLOBAL_SPEED_CAP_DEFAULT))
        step_glob = int(rospy.get_param('~step_per_loop', STEP_PER_LOOP_DEFAULT))

        if _hold_active and time.time()<_hold_until:
            des,cap,step=0,0,STEP_PER_LOOP_DEFAULT
        else:
            if _hold_active and time.time()>=_hold_until:
                globals()['_hold_active']=False; globals()['_hold_reason']=""
            if (z_active is not None) and (z_active.get('speed_code') is not None):
                des  = int(z_active['speed_code'])
                cap  = int(z_active.get('speed_cap', cap_code))
                step = int(z_active.get('step_per_loop', step_glob))
            else:
                des,cap,step=int(base_code),int(cap_code),int(step_glob)
        des=max(0,min(cap,des))
        if speed_cmd_cur<des:  speed_cmd_cur=min(des, speed_cmd_cur+max(1,step))
        elif speed_cmd_cur>des: speed_cmd_cur=max(des, speed_cmd_cur-max(1,step))

        # grade 토픽
        if (z_active is not None) and (z_active.get('grade_topic') is not None):
            grade_val = 1 if int(z_active['grade_topic'])!=0 else 0
        else:
            grade_val = 0

        # 헤딩/조향(LPF + 동적 차단주파수)
        heading_vec=None
        for k in range(2, min(len(pos_buf),5)+1):
            t0,x0,y0=pos_buf[-k]
            if math.hypot(x-x0,y-y0)>=MIN_MOVE_FOR_HEADING:
                heading_vec=(x-x0,y-y0); break
        if heading_vec is not None: _last_heading_vec=heading_vec
        if _last_heading_vec is None:
            continue
        raw_angle=angle_between(_last_heading_vec,(tx-x,ty-y))
        base_fc=LPF_FC_HZ
        lpf.fc=min(2.0, base_fc+0.5) if abs(raw_angle)>10 else base_fc
        filt_angle=max(-MAX_STEER_DEG, min(MAX_STEER_DEG, lpf.update(raw_angle, tsec)))
        latest_filtered_angle = SIGN_CONVENTION * filt_angle

        # ── 시각화 갱신 ──
        if _ENABLE_GUI and _HAVE_MPL:
            _live_line.set_data([p[1] for p in pos_buf], [p[2] for p in pos_buf])
            _current_pt.set_data([x],[y])
            _target_line.set_data([x,tx],[y,ty])

            heading_deg = math.degrees(math.atan2(_last_heading_vec[1], _last_heading_vec[0])) if _last_heading_vec else float('nan')
            wp_disp = (wp_index_active+1) if wp_index_active>=0 else 0

            _ax_info.clear(); _ax_info.axis('off')
            status = [
                f"Meas v: {speed_mps:.2f} m/s",
                f"Pub v(code): {last_pub_speed_code:.1f}",
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
                                int(speed_cmd_cur), int(des),
                                rtk_status_txt, (z_active['name'] if z_active else ""), int(grade_val)])
                globals()['_last_log_wall']=noww
            except Exception as e:
                rospy.logwarn(f"[seq] log write failed: {e}")

        rospy.loginfo_throttle(0.5,
            f"[SEQ] WP={(wp_index_active+1) if wp_index_active>=0 else 0} | "
            f"v={speed_mps:.2f} | pub={speed_cmd_cur}→{des} | "
            f"steer={latest_filtered_angle:+.1f}° | RTK={rtk_status_txt} | FLAG={(z_active['name'] if z_active else '')} | GUP={grade_val}"
        )
    return

_anim_update.lpf = AngleLPF(fc_hz=LPF_FC_HZ)

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
    global log_csv_path, _spaced_x, _spaced_y, _to_xy, _df
    global _fig, _ax, _ax_info, _live_line, _current_pt, _target_line, _hud_text
    global _ENABLE_GUI, _HAVE_RELPOSNED, seq_idx, speed_cmd_cur

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

    base_code   = int(rospy.get_param('~speed_code', BASE_SPEED_CODE_DEFAULT))
    cap_code    = int(rospy.get_param('~speed_cap_code', GLOBAL_SPEED_CAP_DEFAULT))
    speed_cmd_cur = max(0, min(cap_code, base_code))

    # 퍼블리셔/구독
    pub_speed = rospy.Publisher(TOPIC_SPEED_CMD,   Float32, queue_size=10)
    pub_steer = rospy.Publisher(TOPIC_STEER_CMD,   Float32, queue_size=10)
    pub_rtk   = rospy.Publisher(TOPIC_RTK_STATUS,  String,  queue_size=10)
    pub_wpidx = rospy.Publisher(TOPIC_WP_INDEX,    Int32,   queue_size=10)
    pub_grade = rospy.Publisher(TOPIC_GRADE_UP_ON, Int32,   queue_size=10)

    rospy.Subscriber(fix_topic, NavSatFix, gps_callback, queue_size=100)
    try:
        from ublox_msgs.msg import NavRELPOSNED9 as NavRELPOSNED
        _HAVE_RELPOSNED=True
    except Exception:
        try:
            from ublox_msgs.msg import NavRELPOSNED
            _HAVE_RELPOSNED=True
        except Exception:
            _HAVE_RELPOSNED=False
    if _HAVE_RELPOSNED:
        from ublox_msgs.msg import NavRELPOSNED as _RELP
        rospy.Subscriber(rel_topic, _RELP, _cb_relpos, queue_size=50)

    # CSV 로드
    if not os.path.exists(waypoint_csv):
        rospy.logerr("[seq] waypoint csv not found: %s", waypoint_csv); return
    _df = pd.read_csv(waypoint_csv)
    ref_lat=float(_df['Lat'][0]); ref_lon=float(_df['Lon'][0])
    global _to_xy; _to_xy=latlon_to_xy_fn(ref_lat,ref_lon)
    csv_xy=[_to_xy(r['Lat'],r['Lon']) for _,r in _df.iterrows()]
    spaced=generate_waypoints_along_path(csv_xy, spacing=WAYPOINT_SPACING)
    global _spaced_x,_spaced_y; _spaced_x,_spaced_y=tuple(zip(*spaced))

    # ── 플롯 구성 (참고 코드 방식) ──
    if _ENABLE_GUI and _HAVE_MPL:
        global _fig,_ax,_ax_info,_live_line,_current_pt,_target_line,_hud_text
        _fig=plt.figure(figsize=(7.8,9.0))
        gs=_fig.add_gridspec(2,1,height_ratios=[4,1])
        _ax=_fig.add_subplot(gs[0,0])
        _ax_info=_fig.add_subplot(gs[1,0]); _ax_info.axis('off')

        _ax.plot([p[0] for p in csv_xy], [p[1] for p in csv_xy], 'g-',  label='CSV Path')
        _ax.plot(_spaced_x,_spaced_y,'b.-',markersize=3,           label=f'{WAYPOINT_SPACING:.0f}m Waypoints')
        _live_line,  = _ax.plot([],[],'r-', linewidth=1, label='Live GPS')
        _current_pt, = _ax.plot([],[],'ro',              label='Current')
        _target_line,= _ax.plot([],[],'g--', linewidth=1, label='Target Line')
        _ax.axis('equal'); _ax.grid(True); _ax.legend()

        _hud_text = _ax.text(0.98,0.02,"", transform=_ax.transAxes, ha='right', va='bottom',
                             fontsize=9, bbox=dict(fc='white', alpha=0.75, ec='0.5'))

        minx=min(min([p[0] for p in csv_xy]), min(_spaced_x))-10
        maxx=max(max([p[0] for p in csv_xy]), max(_spaced_x))+10
        miny=min(min([p[1] for p in csv_xy]), min(_spaced_y))-10
        maxy=max(max([p[1] for p in csv_xy]), max(_spaced_y))+10
        _ax.set_xlim(minx,maxx); _ax.set_ylim(miny,maxy)

        if annotate_wp or draw_circ:
            for i,(xw,yw) in enumerate(zip(_spaced_x,_spaced_y),1):
                if annotate_wp:
                    _ax.text(xw,yw,str(i),fontsize=7,ha='center',va='center',color='black')
                if draw_circ:
                    _ax.add_patch(Circle((xw,yw), TARGET_RADIUS_END, color='blue', fill=False, linestyle='--', alpha=0.35))

        _fig.canvas.mpl_connect('close_event', _on_close)

    # 퍼블리시 타이머
    rospy.Timer(rospy.Duration(1.0/max(1.0,fs)), publish_all)

    # 애니메이션/헤드리스 루프
    if _ENABLE_GUI and _HAVE_MPL:
        interval_ms=int(1000.0/max(1.0,fs))
        animation.FuncAnimation(_fig, _anim_update, interval=interval_ms, blit=False)
        plt.show()
    else:
        rospy.loginfo("[seq] Headless mode")
        rospy.Timer(rospy.Duration(1.0/max(1.0,fs)), lambda e: _anim_update(None))
        rospy.spin()

if __name__ == '__main__':
    main()
