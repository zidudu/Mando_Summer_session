#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
복귀코드.py — Pure Pursuit 복귀 + 실시간 시각화/범례
- 순차 인덱스 미사용, GRADE_UP_ON=0 고정
- 토픽: /gps/speed_cmd, /gps/steer_cmd, /gps/status, /gps/wp_index, /gps/GRADEUP_ON(=0)
- 로그: time, lat, lon, wp_in_idx(1based), tgt_idx(1based), steer_deg,
        meas_speed_mps, speed_cur_code, speed_des_code, rtk_status, flag_name, grade_topic(=0)
"""

import os, csv, math, time, signal, queue
from collections import deque

import rospy
import numpy as np
import pandas as pd
import geopy.distance

from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, String, Int32

# plotting
_HAVE_MPL=False
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Circle
    _HAVEmpl=True
    _HAVE_MPL=True
except Exception:
    _HAVE_MPL=False

# 공통 토픽/상수
TOPIC_SPEED_CMD   = '/gps/speed_cmd'
TOPIC_STEER_CMD   = '/gps/steer_cmd'
TOPIC_RTK_STATUS  = '/gps/status'
TOPIC_WP_INDEX    = '/gps/wp_index'
TOPIC_GRADE_UP_ON = '/gps/GRADEUP_ON'  # 항상 0

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
LOOKAHEAD_K            = 0.2
WHEELBASE              = 0.28

SPEED_BUF_LEN          = 10
MAX_JITTER_SPEED       = 4.0
MIN_MOVE_FOR_HEADING   = 0.05
LPF_FC_HZ              = 0.8

# 유틸
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

def target_index_from_lookahead(nearest, Ld, spacing, n):
    step=max(1,int(math.ceil(Ld/max(1e-6,spacing))))
    return min(n-1, nearest+step)

def nearest_in_radius_index(x,y,xs,ys,r):
    d2=(xs-x)**2+(ys-y)**2
    i=int(np.argmin(d2))
    return i if math.hypot(xs[i]-x,ys[i]-y)<=r else -1

def pure_pursuit_delta_rad(dx,dy,yaw_rad,L,Ld):
    alpha=math.atan2(dy,dx)-yaw_rad
    alpha=math.atan2(math.sin(alpha), math.cos(alpha))
    return math.atan2(2.0*L*math.sin(alpha), max(1e-6,Ld))

class AngleLPF:
    def __init__(self, fc_hz=LPF_FC_HZ): self.fc=float(fc_hz); self.y=0.0; self.t_last=None
    def update(self, target_deg, t):
        if self.t_last is None: self.t_last=t; self.y=float(target_deg); return self.y
        dt=max(1e-3,t-self.t_last); tau=1.0/(2.0*math.pi*self.fc); a=dt/(tau+dt)
        err=((target_deg-self.y+180)%360)-180
        self.y=((self.y+a*err+180)%360)-180; self.t_last=t; return self.y

# ROS 전역
gps_queue=queue.Queue()
pos_buf=deque(maxlen=SPEED_BUF_LEN*2)
speed_buf=deque(maxlen=SPEED_BUF_LEN)

pub_speed=pub_steer=pub_rtk=pub_wpidx=pub_grade=None
rtk_status_txt="NONE"
_last_fix_wall=0.0

# 시각화 전역
latest_filtered_angle=0.0
wp_index_active=-1
last_pub_speed_code=0.0

_spaced_x=_spaced_y=None
_to_xy=None
_df=None

_prev_Ld=3.5
_nearest_idx_prev=0
_last_heading_vec=None

_ENABLE_GUI=True
_HAVE_RELPOSNED=False
_fig=_ax=_ax_info=_live_line=_current_pt=_target_line=_hud_text=None

log_csv_path=None
_last_log_wall=0.0

# 속도 코드(복귀는 단일 코드)
speed_code_cur=0

# 콜백/퍼블리셔
def gps_callback(msg: NavSatFix):
    global _last_fix_wall
    if hasattr(msg,'status') and getattr(msg.status,'status',0)<0: return
    lat,lon=float(msg.latitude),float(msg.longitude)
    if not (math.isfinite(lat) and math.isfinite(lon)): return
    t= msg.header.stamp.to_sec() if (msg.header and msg.header.stamp) else rospy.Time.now().to_sec()
    gps_queue.put((lat,lon,t))
    _last_fix_wall=time.time()

def _cb_relpos(msg):
    global rtk_status_txt
    try:
        carr=int((int(msg.flags)>>3)&0x3)
        rtk_status_txt="FIXED" if carr==2 else ("FLOAT" if carr==1 else "NONE")
    except Exception:
        rtk_status_txt="NONE"

def publish_all(_evt):
    global last_pub_speed_code
    now=time.time()
    no_gps=(now-_last_fix_wall)>GPS_TIMEOUT_SEC
    v_out=0.0 if no_gps else float(speed_code_cur)
    s_out=0.0 if no_gps else float(latest_filtered_angle)
    if pub_speed: pub_speed.publish(Float32(v_out))
    if pub_steer: pub_steer.publish(Float32(s_out))
    if pub_rtk:   pub_rtk.publish(String(rtk_status_txt))
    if pub_wpidx: pub_wpidx.publish(Int32(int((wp_index_active+1) if wp_index_active>=0 else 0)))
    if pub_grade: pub_grade.publish(Int32(0))
    last_pub_speed_code=v_out

# 애니메이션 업데이트
def _anim_update(_frame):
    """GUI 타이머마다 호출: Pure Pursuit 복귀"""
    global latest_filtered_angle, wp_index_active
    global _prev_Ld, _nearest_idx_prev, _last_heading_vec, _spaced_x, _spaced_y, _to_xy

    updated=False
    speed_mps=0.0
    lat=lon=x=y=float('nan')
    lpf=_anim_update.lpf

    while not gps_queue.empty():
        lat,lon,tsec=gps_queue.get()
        x,y=_to_xy(lat,lon)
        updated=True

        # 속도 추정
        if len(pos_buf)>0:
            t_prev,x_prev,y_prev=pos_buf[-1]
            dt=max(1e-3,tsec-t_prev); d=math.hypot(x-x_prev,y-y_prev)
            inst_v=d/dt
            if inst_v>MAX_JITTER_SPEED: continue
            speed_buf.append(inst_v)
        pos_buf.append((tsec,x,y))
        speed_mps=float(np.median(speed_buf)) if speed_buf else 0.0

        # 최근접 & 룩어헤드
        nearest_idx=find_nearest_index(x,y,_spaced_x,_spaced_y,_nearest_idx_prev,80,15)
        _nearest_idx_prev=nearest_idx
        Ld_target=max(LOOKAHEAD_MIN, min(LOOKAHEAD_MAX, LOOKAHEAD_MIN + LOOKAHEAD_K*speed_mps))
        Ld=_prev_Ld+0.2*(Ld_target-_prev_Ld); _prev_Ld=Ld

        tgt_idx=target_index_from_lookahead(nearest_idx, Ld, WAYPOINT_SPACING, len(_spaced_x))
        tx,ty=_spaced_x[tgt_idx], _spaced_y[tgt_idx]
        wp_index_active=nearest_in_radius_index(x,y,_spaced_x,_spaced_y,TARGET_RADIUS_END)

        # 헤딩/조향(PP + LPF)
        heading=None
        for k in range(2, min(len(pos_buf),5)+1):
            t0,x0,y0=pos_buf[-k]
            if math.hypot(x-x0,y-y0)>=MIN_MOVE_FOR_HEADING:
                heading=(x-x0,y-y0); break
        if heading is not None: _last_heading_vec=heading
        if _last_heading_vec is None:
            continue
        yaw=math.atan2(_last_heading_vec[1], _last_heading_vec[0])
        delta=pure_pursuit_delta_rad(tx-x,ty-y,yaw,WHEELBASE,Ld)
        raw_deg=SIGN_CONVENTION*math.degrees(delta)
        raw_deg=max(-MAX_STEER_DEG,min(MAX_STEER_DEG,raw_deg))
        # LPF 약간 적용
        lpf.fc = 1.1 if abs(raw_deg)>10 else LPF_FC_HZ
        latest_filtered_angle = lpf.update(raw_deg, tsec)

        # 시각화
        if _ENABLE_GUI and _HAVE_MPL:
            _live_line.set_data([p[1] for p in pos_buf], [p[2] for p in pos_buf])
            _current_pt.set_data([x],[y])
            _target_line.set_data([x,tx],[y,ty])

            wp_disp=(wp_index_active+1) if wp_index_active>=0 else 0
            _ax_info.clear(); _ax_info.axis('off')
            _ax_info.text(0.02,0.5,
                f"Meas v: {speed_mps:.2f} m/s | Pub v(code): {last_pub_speed_code:.1f} | "
                f"Ld: {Ld:.2f} m | Steer: {latest_filtered_angle:+.1f}° | WP(in-radius): {wp_disp} | RTK: {rtk_status_txt}",
                fontsize=11, va='center')

            _hud_text.set_text(
                f"pub_v={last_pub_speed_code:.1f} | meas_v={speed_mps:.2f} m/s | "
                f"Ld={Ld:.2f} | steer={latest_filtered_angle:+.1f}° | WP={wp_disp}"
            )

        # 로그(0.5s)
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
                                wp_disp,(tgt_idx+1),
                                f"{latest_filtered_angle:.2f}", f"{speed_mps:.2f}",
                                int(speed_code_cur), int(speed_code_cur),
                                rtk_status_txt, "", 0])
                globals()['_last_log_wall']=noww
            except Exception as e:
                rospy.logwarn(f"[return] log write failed: {e}")

        rospy.loginfo_throttle(0.5,
            f"[PP] WP={(wp_index_active+1) if wp_index_active>=0 else 0} | v={speed_mps:.2f} | "
            f"code={speed_code_cur} | Ld={_prev_Ld:.2f} | steer={latest_filtered_angle:+.1f}° | RTK={rtk_status_txt}"
        )
    return

_anim_update.lpf = AngleLPF(fc_hz=LPF_FC_HZ)

def _on_close(_evt): rospy.signal_shutdown("Figure closed by user")

def _on_shutdown():
    try:
        for _ in range(3):
            if pub_speed: pub_speed.publish(Float32(0.0))
            if pub_steer: pub_steer.publish(Float32(0.0))
            if pub_grade: pub_grade.publish(Int32(0))
            time.sleep(0.02)
    except Exception: pass

# 메인
def main():
    global pub_speed, pub_steer, pub_rtk, pub_wpidx, pub_grade
    global log_csv_path, _spaced_x, _spaced_y, _to_xy, _df
    global _fig,_ax,_ax_info,_live_line,_current_pt,_target_line,_hud_text
    global _ENABLE_GUI, _HAVE_RELPOSNED, speed_code_cur

    rospy.init_node('path_return_pp', anonymous=False)
    rospy.on_shutdown(_on_shutdown)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    pkg_guess   = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')
    default_csv = os.path.join(pkg_guess, 'config', 'raw_track_latlon_18.csv')
    waypoint_csv= rospy.get_param('~csv_path', rospy.get_param('~waypoint_csv', default_csv))
    logs_dir    = os.path.join(pkg_guess, 'logs'); os.makedirs(logs_dir, exist_ok=True)
    log_csv_path= rospy.get_param('~log_csv', os.path.join(logs_dir, f"return_pp_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"))

    fs          = float(rospy.get_param('~fs', FS_DEFAULT))
    ublox_ns    = rospy.get_param('~ublox_ns', '/gps1')
    fix_topic   = rospy.get_param('~fix_topic',    ublox_ns + '/fix')
    rel_topic   = rospy.get_param('~relpos_topic', ublox_ns + '/navrelposned')
    _ENABLE_GUI = bool(rospy.get_param('~enable_gui', True))
    annotate_wp = bool(rospy.get_param('~annotate_wp_index', False))
    draw_circ   = bool(rospy.get_param('~draw_wp_circles', True))

    base_code   = int(rospy.get_param('~speed_code', SPEED_CODE_DEFAULT))
    cap_code    = int(rospy.get_param('~speed_cap_code', SPEED_CAP_DEFAULT))
    speed_code_cur = max(0, min(cap_code, base_code))

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
        rospy.logerr("[return] waypoint csv not found: %s", waypoint_csv); return
    _df=pd.read_csv(waypoint_csv)
    ref_lat=float(_df['Lat'][0]); ref_lon=float(_df['Lon'][0])
    global _to_xy; _to_xy=latlon_to_xy_fn(ref_lat,ref_lon)
    csv_xy=[_to_xy(r['Lat'],r['Lon']) for _,r in _df.iterrows()]
    spaced=generate_waypoints_along_path(csv_xy, spacing=WAYPOINT_SPACING)
    global _spaced_x,_spaced_y; _spaced_x,_spaced_y=tuple(zip(*spaced))

    # 플롯 구성
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
        _target_line,= _ax.plot([],[],'m--', linewidth=1, label='PP Target')
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

    # 애니메이션/헤드리스
    if _ENABLE_GUI and _HAVE_MPL:
        interval_ms=int(1000.0/max(1.0,fs))
        animation.FuncAnimation(_fig, _anim_update, interval=interval_ms, blit=False)
        plt.show()
    else:
        rospy.loginfo("[return] Headless mode")
        rospy.Timer(rospy.Duration(1.0/max(1.0,fs)), lambda e: _anim_update(None))
        rospy.spin()

if __name__=='__main__':
    main()
