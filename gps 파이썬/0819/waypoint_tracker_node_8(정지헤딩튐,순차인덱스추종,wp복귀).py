#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waypoint_tracker_node.py

— 순차 인덱스 추종(타깃 고정, 반경 진입 시에만 다음 인덱스) + 오프트랙 시 복귀 탐색
— 시작 부트(시간/거리 하이브리드) + 정지 헤딩 고정(게이트 + 원형평균)
— RViz/Matplotlib 시각화, 2 m 리샘플, 도착 반경 표시
"""

import math
import csv
import time
import os
import threading
from collections import deque
from typing import List, Tuple, Optional

import rospy
from std_msgs.msg import Float32, String
from sensor_msgs.msg import NavSatFix
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# ---------------- ublox_msgs 임포트 가드 ----------------
_HAVE_RELPOSNED = False
_HAVE_NAVPVT = False
try:
    from ublox_msgs.msg import NavRELPOSNED9 as NavRELPOSNED
    _HAVE_RELPOSNED = True
except Exception:
    try:
        from ublox_msgs.msg import NavRELPOSNED
        _HAVE_RELPOSNED = True
    except Exception:
        _HAVE_RELPOSNED = False

try:
    from ublox_msgs.msg import NavPVT
    _HAVE_NAVPVT = True
except Exception:
    _HAVE_NAVPVT = False

_ROS_OK = True

# ================== 좌표/각 유틸 ==================
_R_EARTH = 6_378_137.0  # WGS84 radius (m)

def deg2rad(x: float) -> float:
    return x * math.pi / 180.0

def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi

def wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

def mercator_xy(lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    lat = deg2rad(lat_deg); lon = deg2rad(lon_deg)
    x = _R_EARTH * lon
    y = _R_EARTH * math.log(max(1e-12, math.tan(math.pi/4.0 + lat/2.0)))
    return x, y

def enu_xy(lat_deg: float, lon_deg: float, lat0_deg: float, lon0_deg: float) -> Tuple[float, float]:
    dlat = deg2rad(lat_deg - lat0_deg)
    dlon = deg2rad(lon_deg - lon0_deg)
    lat0 = deg2rad(lat0_deg)
    x = _R_EARTH * dlon * math.cos(lat0)
    y = _R_EARTH * dlat
    return x, y

def slerp_angle(a: float, b: float, t: float) -> float:
    d = wrap_pi(b - a)
    return wrap_pi(a + t * d)

def circ_mean(angles_rad: List[float]) -> float:
    if not angles_rad:
        return 0.0
    cx = sum(math.cos(a) for a in angles_rad)
    sy = sum(math.sin(a) for a in angles_rad)
    if abs(cx) < 1e-12 and abs(sy) < 1e-12:
        return wrap_pi(angles_rad[-1])
    return math.atan2(sy, cx)

# ================== Matplotlib Live Visualizer (옵션) ==================
try:
    import matplotlib
    _HAVE_MPL_BASE = True
except Exception as _e:
    try:
        rospy.logwarn(f"[tracker_ros] Matplotlib base import 실패: {_e}")
    except Exception:
        print(f"[tracker_ros] Matplotlib base import 실패: {_e}")
    _HAVE_MPL_BASE = False

class MatplotViz:
    def __init__(self, backend='Qt5Agg', follow=True, range_m=12.0, rate_hz=8.0,
                 draw_wp_rings=True, wp_ring_step=1, wp_label_step=5, route_maxlen=10000):
        self._ok = False
        if not _HAVE_MPL_BASE:
            return
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyArrowPatch, Circle
            self.plt = plt
            self.FancyArrowPatch = FancyArrowPatch
            self.Circle = Circle
        except Exception as e:
            try: rospy.logwarn(f"[tracker_ros] Matplotlib 로드 실패: {e}")
            except Exception: print(f"[tracker_ros] Matplotlib 로드 실패: {e}")
            return

        self._ok = True
        self.follow = follow
        self.range_m = float(range_m)
        self.min_dt = 1.0/float(rate_hz)
        self._last_draw = 0.0
        self._lock = threading.Lock()
        self._snap = None

        self.draw_wp_rings = bool(draw_wp_rings)
        self.wp_ring_step  = max(1, int(wp_ring_step))
        self.wp_label_step = max(1, int(wp_label_step))
        self.route_maxlen = int(route_maxlen)
        self._route = []

        self.plt.ion()
        self.fig, self.ax = self.plt.subplots(figsize=(8,8))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, ls=':', alpha=0.5)
        self.ax.set_title("Waypoint Tracker — Live View")

        (self.line_path,)  = self.ax.plot([], [], '-',  c='0.6', lw=1.2, label='Path')
        (self.sc_wp,)      = self.ax.plot([], [], 'o',  c='red', ms=3.5,  label='Waypoints')
        (self.pt_car,)     = self.ax.plot([], [], 'o',  c='tab:blue',  ms=8,  label='GPS')
        (self.pt_tgt,)     = self.ax.plot([], [], '*',  c='magenta',   ms=12, label='Target WP')
        (self.line_tgt,)   = self.ax.plot([], [], '--', c='cyan',  lw=1.0, label='TargetLine')
        (self.line_route,) = self.ax.plot([], [], '-',  c='saddlebrown', lw=1.0, label='Route')

        self.txt = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                ha='left', va='top',
                                bbox=dict(fc='white', alpha=0.7), fontsize=9)
        self.arr_head = None
        self.arr_steer = None
        self.arrive_ring = None

        self._wp_ring_patches = []
        self._wp_labels = []
        self._last_ring_key = None

        self.ax.legend(loc='upper right')

    def _safe_del(self, artist):
        try:
            if artist is not None:
                artist.remove()
        except Exception:
            pass

    def _clear_labels(self):
        for t in self._wp_labels:
            self._safe_del(t)
        self._wp_labels = []

    def ingest(self, core, veh_heading_rad=None, steer_rad=None, arrive_r=1.0):
        if not self._ok:
            return
        wps = core.viz_waypoints
        car = core.viz_vehicle_xy
        tgt = core.viz_target_xy
        snap = {
            'wps': list(wps) if wps else None,
            'car': tuple(car) if car else None,
            'tgt': tuple(tgt) if tgt else None,
            'heading': veh_heading_rad,
            'steer': steer_rad,
            'arrive_r': float(arrive_r),
        }
        with self._lock:
            self._snap = snap

    def _ensure_wp_rings_and_labels(self, wps, arrive_r):
        key = (len(wps), float(arrive_r), int(self.wp_ring_step), int(self.wp_label_step))
        if self._last_ring_key == key:
            return
        for p in self._wp_ring_patches:
            self._safe_del(p)
        self._wp_ring_patches = []
        self._clear_labels()

        if self.draw_wp_rings:
            for i, (x, y) in enumerate(wps):
                if i % self.wp_ring_step != 0:
                    continue
                ring = self.Circle((x, y), arrive_r, ec='tab:blue', ls='--', fc='none', alpha=0.25)
                self.ax.add_patch(ring)
                self._wp_ring_patches.append(ring)

        for i, (x, y) in enumerate(wps, start=1):
            if i % self.wp_label_step != 0 and i != 1:
                continue
            t = self.ax.text(x, y, str(i), fontsize=6, ha='right', va='bottom')
            self._wp_labels.append(t)

        self._last_ring_key = key

    def render(self):
        if not self._ok:
            return
        now = time.time()
        if (now - self._last_draw) < self.min_dt:
            return
        with self._lock:
            snap = self._snap
        if snap is None:
            return
        self._last_draw = now

        wps = snap['wps']; car = snap['car']; tgt = snap['tgt']
        heading = snap['heading']; steer = snap['steer']; arrive_r = snap['arrive_r']

        if wps:
            xs = [p[0] for p in wps]; ys = [p[1] for p in wps]
            self.line_path.set_data(xs, ys)
            self.sc_wp.set_data(xs, ys)
            self._ensure_wp_rings_and_labels(wps, arrive_r)

        if car:
            self.pt_car.set_data([car[0]],[car[1]])
            self._route.append(car)
            if len(self._route) > 10000:
                self._route = self._route[-10000:]
            rx = [p[0] for p in self._route]; ry = [p[1] for p in self._route]
            self.line_route.set_data(rx, ry)

        if tgt:
            self.pt_tgt.set_data([tgt[0]],[tgt[1]])
            if car:
                self.line_tgt.set_data([car[0], tgt[0]],[car[1], tgt[1]])

        self._safe_del(self.arrive_ring)
        if tgt:
            self.arrive_ring = self.Circle((tgt[0], tgt[1]), arrive_r, ec='tab:blue', ls='--', fc='none', alpha=0.7)
            self.ax.add_patch(self.arrive_ring)

        self._safe_del(self.arr_head); self._safe_del(self.arr_steer)
        if car and heading is not None:
            hx, hy = car[0] + math.cos(heading), car[1] + math.sin(heading)
            self.arr_head = self.FancyArrowPatch((car[0],car[1]), (hx,hy), color='tab:blue',
                                                 lw=2, arrowstyle='-|>', mutation_scale=15)
            self.ax.add_patch(self.arr_head)
        if car and (heading is not None) and (steer is not None):
            sx, sy = car[0] + math.cos(heading+steer), car[1] + math.sin(heading+steer)
            self.arr_steer = self.FancyArrowPatch((car[0],car[1]), (sx,sy), color='red',
                                                  lw=2, arrowstyle='-|>', mutation_scale=15, alpha=0.9)
            self.ax.add_patch(self.arr_steer)

        info = []
        if car: info.append(f"Veh: ({car[0]:.1f}, {car[1]:.1f}) m")
        if tgt and car:
            d = math.hypot(tgt[0]-car[0], tgt[1]-car[1])
            info.append(f"Dist→Target: {d:.1f} m")
        if heading is not None: info.append(f"Heading: {math.degrees(heading):.1f}°")
        if steer   is not None: info.append(f"Steering: {math.degrees(steer):+.1f}°")
        self.txt.set_text("\n".join(info))

        if self.follow and car:
            r = self.range_m
            self.ax.set_xlim(car[0]-r, car[0]+r)
            self.ax.set_ylim(car[1]-r, car[1]+r)
        elif wps:
            self.ax.relim(); self.ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

# ================== Polyline 유틸 ==================
class Polyline:
    def __init__(self, pts: List[Tuple[float, float]]):
        if len(pts) < 2:
            raise ValueError("Polyline needs >= 2 points")
        self.pts = pts
        self.segs = [(pts[i], pts[i+1]) for i in range(len(pts)-1)]
        self.seg_vec = [(b[0]-a[0], b[1]-a[1]) for a,b in self.segs]
        self.seg_len = [math.hypot(v[0], v[1]) for v in self.seg_vec]
        self.cum_s = [0.0]; s = 0.0
        for L in self.seg_len:
            s += L; self.cum_s.append(s)
        self.total_len = s

    def project(self, p: Tuple[float, float]) -> Tuple[float, Tuple[float, float], int, float, float]:
        px, py = p
        best = (1e18, (0.0, 0.0), 0, 0.0, 0.0)
        for i, ((ax, ay), (bx, by)) in enumerate(self.segs):
            vx, vy = self.seg_vec[i]
            L2 = vx*vx + vy*vy
            if L2 <= 1e-12: continue
            t = ((px-ax)*vx + (py-ay)*vy) / L2
            t = max(0.0, min(1.0, t))
            qx = ax + t*vx; qy = ay + t*vy
            d = math.hypot(px-qx, py-qy)
            if d < best[0]:
                s_at_q = self.cum_s[i] + t*self.seg_len[i]
                best = (d, (qx, qy), i, t, s_at_q)
        return best

    def heading_at(self, seg_idx: int) -> float:
        vx, vy = self.seg_vec[seg_idx]
        return math.atan2(vy, vx)

    def s_of_index(self, idx: int) -> float:
        return self.cum_s[max(0, min(idx, len(self.cum_s)-1))]

# ================== TrackerCore ==================
class TrackerCore:
    def __init__(self,
                 csv_path: str,
                 coord_mode: str,
                 anchor_source: str,
                 anchor_lat: float,
                 anchor_lon: float,
                 wheelbase: float,
                 look_dist0: float,
                 k_ld: float,
                 k_v: float,
                 arrive_r: float,
                 approach_r: float,
                 exit_r: float,
                 const_speed: float,
                 accel_limit: float,
                 decel_limit: float,
                 # 부트(시간/거리)
                 boot_t_sec: float,
                 boot_d_m: float,
                 boot_speed: float,
                 # 헤딩 게이트/히스토리
                 heading_update_d_min: float,
                 heading_update_v_min: float,
                 heading_history_len: int,
                 # 시작 Ld 클램프
                 startup_ld_sec: float,
                 startup_ld_min: float,
                 startup_ld_max: float,
                 # 순차 추종 & 복귀
                 sequential_follow: bool,
                 start_index: int,
                 recover_alpha_deg: float,
                 offtrack_k: float,
                 # 복귀/검색(반경, 히스테리시스 시간 등은 기존 값 활용)
                 theta_th_deg: float,
                 win_ahead_min: int,
                 win_ahead_max: int,
                 search_r0: float,
                 search_dr: float,
                 search_rmax: float,
                 lock_time: float,
                 # 저속 헤딩 보정 & α 클램프
                 v_heading_blend_v: float,
                 alpha_clamp_deg: float,
                 alpha_clamp_v_thresh: float,
                 alpha_clamp_enable: bool,
                 log_csv: str = "",
                 resample_interval_m: float = 2.0):

        # 좌표/앵커
        self.coord_mode = coord_mode.lower()
        self.anchor_source = anchor_source.lower()
        self.anchor_lat = anchor_lat
        self.anchor_lon = anchor_lon

        # 주행/제어
        self.wheelbase = wheelbase
        self.look_dist0 = look_dist0
        self.k_ld = k_ld
        self.k_v = k_v
        self.arrive_r = arrive_r
        self.approach_r = max(approach_r, arrive_r)
        self.exit_r = exit_r
        self.const_speed = const_speed
        self.accel_limit = max(0.0, accel_limit)
        self.decel_limit = max(0.0, decel_limit)

        # 부트
        self.boot_t_sec  = max(0.0, boot_t_sec)
        self.boot_d_m    = max(0.0, boot_d_m)
        self.boot_speed  = max(0.0, boot_speed)

        # 헤딩 게이트/히스토리
        self.heading_update_d_min = max(0.0, heading_update_d_min)
        self.heading_update_v_min = max(0.0, heading_update_v_min)
        self.heading_history_len  = max(1, int(heading_history_len))

        # 시작 Ld 클램프
        self.startup_ld_sec = max(0.0, startup_ld_sec)
        self.startup_ld_min = max(0.05, startup_ld_min)
        self.startup_ld_max = max(self.startup_ld_min, startup_ld_max)

        # 순차 추종 & 복귀
        self.sequential_follow = bool(sequential_follow)
        self.recover_alpha_rad = deg2rad(recover_alpha_deg)
        self.offtrack_k = max(1.0, offtrack_k)

        self.theta_th = deg2rad(theta_th_deg)
        self.win_ahead_min = win_ahead_min
        self.win_ahead_max = win_ahead_max
        self.search_r0 = search_r0
        self.search_dr = search_dr
        self.search_rmax = search_rmax
        self.lock_time = lock_time

        # 저속 헤딩 보정 & α 클램프
        self.v_heading_blend_v = max(0.0, v_heading_blend_v)
        self.alpha_clamp_rad = deg2rad(alpha_clamp_deg)
        self.alpha_clamp_v_thresh = max(0.0, alpha_clamp_v_thresh)
        self.alpha_clamp_enable = bool(alpha_clamp_enable)

        self.log_csv = log_csv
        self.resample_interval_m = float(resample_interval_m)

        # 상태
        self._lat0 = None; self._lon0 = None; self._anchor_ready = False
        self._prev_time = None; self._prev_speed = 0.0
        self._prev_xy: Optional[Tuple[float,float]] = None

        self._veh_heading = 0.0
        self._heading_hold = 0.0
        self._heading_hist = deque(maxlen=self.heading_history_len)

        self._poly: Optional[Polyline] = None
        self._wps_raw = self._load_waypoints(csv_path)
        self._wps_xy: Optional[List[Tuple[float,float]]] = None

        # 순차 추종 상태
        self._cur_idx = max(0, int(start_index))
        self._completed = False

        # 타깃 락(호환용)
        self._last_target_idx: Optional[int] = None
        self._last_target_s: Optional[float] = None
        self._last_lock_time: float = -1e9

        # 부트 상태
        self._boot_active = False
        self._boot_start_ts = 0.0
        self._boot_dist_acc = 0.0
        self._startup_ld_until = 0.0
        self._first_update_done = False

        # 시각화
        self._viz_xy: Optional[Tuple[float,float]] = None
        self._viz_target: Optional[Tuple[float,float]] = None

        # 초기 좌표계 준비
        if self.coord_mode == 'wgs84':
            if not math.isnan(self.anchor_lat) and not math.isnan(self.anchor_lon):
                self._lat0, self._lon0 = float(self.anchor_lat), float(self.anchor_lon)
                self._prepare_wp_xy_wgs84()
            elif self.anchor_source == 'waypoint':
                self._lat0, self._lon0 = float(self._wps_raw[0][0]), float(self._wps_raw[0][1])
                self._prepare_wp_xy_wgs84()
            else:
                self._anchor_ready = False
        elif self.coord_mode == 'mercator':
            self._prepare_wp_xy_mercator()
            self._anchor_ready = True
        else:
            raise ValueError("coord_mode must be 'wgs84' or 'mercator'")

    # ---------- Waypoints ----------
    def _load_waypoints(self, path: str) -> List[Tuple[float,float]]:
        rows = []
        with open(path, 'r') as f:
            r = csv.reader(f)
            rows = [row for row in r if row and any(c.strip() for c in row)]
        if not rows:
            raise RuntimeError("Empty waypoint CSV")

        lat_idx, lon_idx, start_i = 0, 1, 0
        header = [c.strip().lower() for c in rows[0]]

        def _is_float(s):
            try: float(s); return True
            except Exception: return False

        if not all(_is_float(c) for c in rows[0][:2]):
            maybe_lat = None; maybe_lon = None
            for i, name in enumerate(header):
                if name in ('lat','latitude'): maybe_lat = i
                if name in ('lon','lng','longitude','long'): maybe_lon = i
            if maybe_lat is not None and maybe_lon is not None:
                lat_idx, lon_idx, start_i = maybe_lat, maybe_lon, 1
            else:
                start_i = 1

        data = []
        for row in rows[start_i:]:
            if len(row) <= max(lat_idx, lon_idx): continue
            try:
                lat = float(row[lat_idx]); lon = float(row[lon_idx])
                data.append((lat, lon))
            except Exception: continue

        if len(data) < 2:
            raise RuntimeError("Need >=2 waypoints in CSV")
        return data

    def _resample_xy(self, pts: List[Tuple[float,float]], interval: float) -> List[Tuple[float,float]]:
        if interval <= 0.0 or len(pts) < 2:
            return pts[:]
        seg_len = []
        for i in range(len(pts)-1):
            ax, ay = pts[i]; bx, by = pts[i+1]
            seg_len.append(math.hypot(bx-ax, by-ay))
        cum = [0.0]; s = 0.0
        for L in seg_len:
            s += L; cum.append(s)
        total = s
        if total <= 1e-6:
            return pts[:]
        samples = [0.0]; k = 1
        while k*interval < total:
            samples.append(k*interval); k += 1
        samples.append(total)
        out = []; i = 0
        for ss in samples:
            while i+1 < len(cum) and cum[i+1] < ss - 1e-9:
                i += 1
            if i+1 >= len(cum):
                out.append(pts[-1]); break
            s0, s1 = cum[i], cum[i+1]
            ax, ay = pts[i]; bx, by = pts[i+1]
            t = 0.0 if s1<=s0 else (ss - s0)/(s1 - s0)
            x = ax + t*(bx - ax)
            y = ay + t*(by - ay)
            out.append((x, y))
        return out

    def _prepare_wp_xy_wgs84(self):
        xy = [enu_xy(lat, lon, self._lat0, self._lon0) for (lat,lon) in self._wps_raw]
        self._wps_xy = self._resample_xy(xy, self.resample_interval_m)
        self._poly = Polyline(self._wps_xy)
        self._anchor_ready = True

    def _prepare_wp_xy_mercator(self):
        xy = [tuple(w) for w in self._wps_raw]
        self._wps_xy = self._resample_xy(xy, self.resample_interval_m)
        self._poly = Polyline(self._wps_xy)

    # ---------- 시각화 조회 ----------
    @property
    def viz_waypoints(self) -> Optional[List[Tuple[float,float]]]:
        return self._wps_xy
    @property
    def viz_vehicle_xy(self) -> Optional[Tuple[float,float]]:
        return self._viz_xy
    @property
    def viz_target_xy(self) -> Optional[Tuple[float,float]]:
        return self._viz_target
    @property
    def veh_heading(self) -> float:
        return self._veh_heading
    @property
    def boot_active(self) -> bool:
        return self._boot_active

    # ---------- 핵심 업데이트 ----------
    def update(self, lat: float, lon: float, now: float, rtk_code: int) -> Tuple[float, float, str, float]:
        # 좌표 변환
        if self.coord_mode == 'wgs84':
            if not self._anchor_ready:
                self._lat0, self._lon0 = float(lat), float(lon)
                self._prepare_wp_xy_wgs84()
            x, y = enu_xy(lat, lon, self._lat0, self._lon0)
        else:
            x, y = mercator_xy(lat, lon)

        # 이동량/속도 추정
        d_move = 0.0; v_meas = 0.0
        dt = 0.0 if self._prev_time is None else max(1e-3, now - self._prev_time)
        if self._prev_xy is not None:
            dx = x - self._prev_xy[0]; dy = y - self._prev_xy[1]
            d_move = math.hypot(dx, dy); v_meas = d_move / dt
        self._prev_xy = (x, y)

        # 경로 투영(복귀/품질 판정에 사용)
        dist_to_path, q, seg_idx, t_on_seg, s_at_q = self._poly.project((x, y))
        path_heading = self._poly.heading_at(seg_idx)

        # 첫 FIX → 부트 시작
        if not self._first_update_done:
            self._first_update_done = True
            self._boot_active = True
            self._boot_start_ts = now
            self._boot_dist_acc = 0.0
            self._startup_ld_until = now + self.startup_ld_sec
            # 초기 헤딩 홀드는 경로 접선으로 시딩
            self._veh_heading = path_heading
            self._heading_hold = path_heading
            self._heading_hist.clear()
            self._heading_hist.append(path_heading)

        # 부트 상태 갱신
        if self._boot_active:
            self._boot_dist_acc += d_move
            if (now - self._boot_start_ts) >= self.boot_t_sec or self._boot_dist_acc >= self.boot_d_m:
                self._boot_active = False

        # 헤딩 갱신 게이트
        heading_update_ok = (d_move >= self.heading_update_d_min) and (v_meas >= self.heading_update_v_min)
        if heading_update_ok:
            new_h = math.atan2(y - (self._prev_xy[1] - (y - self._prev_xy[1])), x - (self._prev_xy[0] - (x - self._prev_xy[0])))
            # 위 식은 결국 dy, dx 사용과 동일
            new_h = math.atan2(y - self._prev_xy[1], x - self._prev_xy[0])
            self._heading_hist.append(new_h)
            self._veh_heading = circ_mean(list(self._heading_hist))
            self._heading_hold = self._veh_heading
        else:
            self._veh_heading = self._heading_hold

        # Ld 계산(속도 기반) + 시작 초기 클램프
        v_prev_cmd = self._prev_speed
        Ld = self.look_dist0 + self.k_ld + self.k_v * max(0.0, v_prev_cmd)
        if now < self._startup_ld_until:
            Ld = min(self.startup_ld_max, max(self.startup_ld_min, Ld))

        # ───── 순차 인덱스 추종 ─────
        if self.sequential_follow and not self._completed:
            # 타깃 고정: 현재 인덱스
            cur_idx = min(self._cur_idx, len(self._poly.pts)-1)
            target_xy = self._poly.pts[cur_idx]
            d_to_target = math.hypot(target_xy[0]-x, target_xy[1]-y)

            # 도착 판정 → 다음 인덱스
            if d_to_target <= self.arrive_r and cur_idx < len(self._poly.pts)-1:
                self._cur_idx = cur_idx + 1
                cur_idx = self._cur_idx
                target_xy = self._poly.pts[cur_idx]
                d_to_target = math.hypot(target_xy[0]-x, target_xy[1]-y)
            elif d_to_target <= self.arrive_r and cur_idx == len(self._poly.pts)-1:
                self._completed = True

            # 복귀 조건 체크 (오프트랙/비현실 조향)
            # 1) 타깃이 헤딩 뒤쪽/측면, 2) 요구 각이 너무 큼, 3) 경로 이탈거리 큼
            # 타깃 방향
            tgt_bearing = math.atan2(target_xy[1]-y, target_xy[0]-x)
            alpha = wrap_pi(tgt_bearing - self._veh_heading)
            forward_dot = math.cos(alpha)  # >0이면 전방

            need_recover = (abs(alpha) > self.recover_alpha_rad) or (forward_dot <= 0.0) or (dist_to_path > self.offtrack_k * self.arrive_r)

            if need_recover:
                # 현재 인덱스 이후에서 복귀 후보 탐색 (조향 가용범위 내)
                new_idx = self._search_recover_candidate(x, y, self._veh_heading, start_idx=cur_idx)
                if new_idx is not None:
                    self._cur_idx = new_idx
                    cur_idx = new_idx
                    target_xy = self._poly.pts[cur_idx]
                    d_to_target = math.hypot(target_xy[0]-x, target_xy[1]-y)
                # 실패 시에는 현 타깃 유지(무한 루프 방지용 Ld_eff로 완만히 접근)

            # 조향 계산
            veh_heading_eff = self._veh_heading
            if heading_update_ok and self.v_heading_blend_v > 1e-6 and v_meas < self.v_heading_blend_v:
                t = 1.0 - (v_meas / self.v_heading_blend_v)
                veh_heading_eff = slerp_angle(self._veh_heading, path_heading, t)

            alpha = wrap_pi(math.atan2(target_xy[1]-y, target_xy[0]-x) - veh_heading_eff)

            if self._boot_active:
                steer_rad = 0.0
            else:
                if self.alpha_clamp_enable and v_meas < self.alpha_clamp_v_thresh:
                    cap = self.alpha_clamp_rad
                    alpha = min(max(alpha, -cap), cap)
                # 순차 모드에선 타깃까지 실제 거리로 Ld를 제한
                Ld_eff = max(0.1, min(Ld, d_to_target if d_to_target > 1e-6 else Ld))
                steer_rad = math.atan2(2.0*self.wheelbase*math.sin(alpha), Ld_eff)

            # 속도 명령 (부트 시 강제, 이외는 기존 로직)
            v_target = (self.boot_speed if self._boot_active else
                        (0.0 if d_to_target <= self.arrive_r else
                         (self.const_speed if d_to_target >= self.approach_r else
                          self.const_speed * (d_to_target - self.arrive_r)/(self.approach_r - self.arrive_r))))
            v_cmd = self._apply_slew(self._prev_speed, v_target, dt)
            self._prev_speed = v_cmd; self._prev_time = now

            # RTK 문자열
            rtk_txt = 'FIX' if rtk_code == 2 else ('FLOAT' if rtk_code == 1 else 'NONE')

            # 시각화
            self._viz_xy = (x, y); self._viz_target = (target_xy[0], target_xy[1])

            return v_cmd, steer_rad, rtk_txt, d_to_target

        # ───── (옵션) 비-순차 모드: 기존 복귀/룩어헤드 방식 (호환용) ─────
        # 필요 시 남겨둔 경로 투영 기반 타깃 선택 로직을 사용할 수 있습니다.
        # 여기서는 간결히 유지: s_at_q + Ld
        target_s = min(self._poly.total_len, s_at_q + Ld)
        target_idx = self._index_near_s(target_s)
        target_xy = self._poly.pts[target_idx]
        d_to_target = math.hypot(target_xy[0]-x, target_xy[1]-y)

        veh_heading_eff = self._veh_heading
        if heading_update_ok and self.v_heading_blend_v > 1e-6 and v_meas < self.v_heading_blend_v:
            t = 1.0 - (v_meas / self.v_heading_blend_v)
            veh_heading_eff = slerp_angle(self._veh_heading, path_heading, t)
        alpha = wrap_pi(math.atan2(target_xy[1]-y, target_xy[0]-x) - veh_heading_eff)

        if self._boot_active:
            steer_rad = 0.0
        else:
            if self.alpha_clamp_enable and v_meas < self.alpha_clamp_v_thresh:
                cap = self.alpha_clamp_rad
                alpha = min(max(alpha, -cap), cap)
            steer_rad = math.atan2(2.0*self.wheelbase*math.sin(alpha), max(0.1, Ld))

        v_target = (self.boot_speed if self._boot_active else
                    (0.0 if d_to_target <= self.arrive_r else
                     (self.const_speed if d_to_target >= self.approach_r else
                      self.const_speed * (d_to_target - self.arrive_r)/(self.approach_r - self.arrive_r))))
        v_cmd = self._apply_slew(self._prev_speed, v_target, dt)
        self._prev_speed = v_cmd; self._prev_time = now

        rtk_txt = 'FIX' if rtk_code == 2 else ('FLOAT' if rtk_code == 1 else 'NONE')
        self._viz_xy = (x, y); self._viz_target = (target_xy[0], target_xy[1])
        return v_cmd, steer_rad, rtk_txt, d_to_target

    # ---------- 순차 복귀 후보 탐색 ----------
    def _search_recover_candidate(self, x: float, y: float, heading: float, start_idx: int) -> Optional[int]:
        """현재 인덱스 이후에서, 가용 조향 범위(α≤recover_alpha) 안의 가장 가까운/앞쪽 후보를 선택."""
        best = None; bestJ = 1e18
        N = len(self._poly.pts)
        Rmax = self.search_rmax
        for idx in range(start_idx, N):
            px, py = self._poly.pts[idx]
            dx, dy = px - x, py - y
            dist = math.hypot(dx, dy)
            if dist > Rmax:  # 탐색 반경 밖
                continue
            alpha = abs(wrap_pi(math.atan2(dy, dx) - heading))
            if alpha > self.recover_alpha_rad:
                continue
            # 앞쪽 선호(간단 점수: 거리 + 각도 패널티)
            J = dist + (alpha / self.recover_alpha_rad)
            if J < bestJ:
                bestJ = J; best = idx
            # 아주 가까운(도착반경~2*도착반경) 후보는 즉시 채택해도 무방
            if dist < 2.0 * self.arrive_r and alpha <= self.recover_alpha_rad:
                return idx
        return best

    # ---------- 보조 ----------
    def _index_near_s(self, s: float) -> int:
        cs = self._poly.cum_s
        n = len(cs)
        if s <= cs[0]: return 0
        if s >= cs[-1]: return len(self._poly.pts)-1
        lo, hi = 0, n-1
        while lo+1 < hi:
            mid = (lo+hi)//2
            if cs[mid] <= s: lo = mid
            else: hi = mid
        return lo if (s - cs[lo]) <= (cs[hi]-s) else hi

    def _apply_slew(self, prev_v: float, tgt_v: float, dt: float) -> float:
        if dt <= 0.0: return tgt_v
        if tgt_v > prev_v:
            dv = min(tgt_v - prev_v, self.accel_limit * dt); return prev_v + dv
        else:
            dv = min(prev_v - tgt_v, self.decel_limit * dt); return prev_v - dv

    def save_logs(self): pass

# ================== TrackerRosNode ==================
class TrackerRosNode:
    _STEER_IN_RAD = False
    def __init__(self):
        # 파라미터
        csv_path     = rospy.get_param('~csv_path', '/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/left_lane.csv')
        coord_mode   = rospy.get_param('~coord_mode', 'wgs84')
        anchor_src   = rospy.get_param('~anchor_source', 'waypoint')
        anchor_lat   = float(rospy.get_param('~anchor_lat', float('nan')))
        anchor_lon   = float(rospy.get_param('~anchor_lon', float('nan')))

        wheelbase    = float(rospy.get_param('~wheelbase', 0.28))
        look_dist0   = float(rospy.get_param('~look_dist0', 1.0))
        k_ld         = float(rospy.get_param('~k_ld', 0.0))
        k_v          = float(rospy.get_param('~k_v', 0.5))
        arrive_r     = float(rospy.get_param('~arrive_r', 1.0))
        approach_r   = float(rospy.get_param('~approach_r', 3.0))
        exit_r       = float(rospy.get_param('~exit_r', 0.8))
        const_speed  = float(rospy.get_param('~const_speed', 1.0))
        accel_limit  = float(rospy.get_param('~accel_limit', 0.5))
        decel_limit  = float(rospy.get_param('~decel_limit', 0.8))

        # 부트/헤딩
        boot_t_sec   = float(rospy.get_param('~boot_t_sec', 2.0))
        boot_d_m     = float(rospy.get_param('~boot_d_m', 0.8))
        boot_speed   = float(rospy.get_param('~boot_speed', const_speed))
        heading_update_d_min = float(rospy.get_param('~heading_update_d_min', 0.05))
        heading_update_v_min = float(rospy.get_param('~heading_update_v_min', 0.25))
        heading_history_len  = int(rospy.get_param('~heading_history_len', 5))

        # 시작 Ld 클램프
        startup_ld_sec  = float(rospy.get_param('~startup_ld_sec', 2.0))
        startup_ld_min  = float(rospy.get_param('~startup_ld_min', 0.8))
        startup_ld_max  = float(rospy.get_param('~startup_ld_max', 2.2))

        # 순차 추종 & 복귀
        sequential_follow = bool(rospy.get_param('~sequential_follow', True))
        start_index       = int(rospy.get_param('~start_index', 0))
        recover_alpha_deg = float(rospy.get_param('~recover_alpha_deg', 30.0))
        offtrack_k        = float(rospy.get_param('~offtrack_k', 2.0))

        # 복귀/검색(기존 파라미터 재사용)
        theta_th_deg = float(rospy.get_param('~theta_th_deg', 30.0))
        win_ahead_min= int(rospy.get_param('~win_ahead_min', 3))
        win_ahead_max= int(rospy.get_param('~win_ahead_max', 40))
        search_r0    = float(rospy.get_param('~search_r0', 1.0))
        search_dr    = float(rospy.get_param('~search_dr', 0.5))
        search_rmax  = float(rospy.get_param('~search_rmax', 10.0))
        lock_time    = float(rospy.get_param('~lock_time', 0.7))

        # 저속 헤딩 보정 & α 클램프
        v_heading_blend_v     = float(rospy.get_param('~v_heading_blend_v', 0.5))
        alpha_clamp_deg       = float(rospy.get_param('~alpha_clamp_deg', 60.0))
        alpha_clamp_v_thresh  = float(rospy.get_param('~alpha_clamp_v_thresh', 0.5))
        alpha_clamp_enable    = bool(rospy.get_param('~alpha_clamp_enable', True))

        # 리샘플
        resample_interval_m   = float(rospy.get_param('~resample_interval_m', 2.0))

        # 출력/단위
        self.do_publish = bool(rospy.get_param('~do_publish', True))
        TrackerRosNode._STEER_IN_RAD = bool(rospy.get_param('~steer_in_rad', False))
        self.rate_hz = float(rospy.get_param('~publish_rate', 50.0))

        self.speed_min = float(rospy.get_param('~speed_min', 0.0))
        self.speed_max = float(rospy.get_param('~speed_max', const_speed))
        self.steer_limit_deg = float(rospy.get_param('~steer_limit_deg', 30.0))

        # 속도 고정(부트 중엔 boot_speed 우선)
        self.force_const_speed = bool(rospy.get_param('~force_const_speed', True))

        # RViz frame
        self.frame_id = rospy.get_param('~frame_id', 'map')

        # ── Matplotlib Viz 옵션 ──
        self.viz_enable       = bool(rospy.get_param('~viz_enable', True))
        self.viz_backend      = rospy.get_param('~viz_backend', 'Qt5Agg')
        self.viz_follow       = bool(rospy.get_param('~viz_follow', True))
        self.viz_range_m      = float(rospy.get_param('~viz_range_m', 12.0))
        self.viz_rate_hz      = float(rospy.get_param('~viz_rate_hz', 8.0))
        self.viz_wp_rings     = bool(rospy.get_param('~viz_wp_rings', True))
        self.viz_wp_ring_step = int(rospy.get_param('~viz_wp_ring_step', 1))
        self.viz_wp_label_step= int(rospy.get_param('~viz_wp_label_step', 5))
        self.viz_route_maxlen = int(rospy.get_param('~viz_route_maxlen', 10000))

        self.viz = MatplotViz(backend=self.viz_backend,
                              follow=self.viz_follow,
                              range_m=self.viz_range_m,
                              rate_hz=self.viz_rate_hz,
                              draw_wp_rings=self.viz_wp_rings,
                              wp_ring_step=self.viz_wp_ring_step,
                              wp_label_step=self.viz_wp_label_step,
                              route_maxlen=self.viz_route_maxlen) if self.viz_enable else None

        # RViz 웨이포인트 원 표시
        self.rviz_wp_rings     = bool(rospy.get_param('~rviz_wp_rings', True))
        self.rviz_ring_step    = int(rospy.get_param('~rviz_ring_step', 1))

        # 토픽
        ublox_ns = rospy.get_param('~ublox_ns', '/ublox')
        self.fix_topic    = rospy.get_param('~fix_topic',    ublox_ns + '/fix')
        self.relpos_topic = rospy.get_param('~relpos_topic', ublox_ns + '/navrelposned')
        self.navpvt_topic = rospy.get_param('~navpvt_topic', ublox_ns + '/navpvt')

        # 코어 생성
        self.core = TrackerCore(csv_path=csv_path,
                                coord_mode=coord_mode,
                                anchor_source=anchor_src,
                                anchor_lat=anchor_lat,
                                anchor_lon=anchor_lon,
                                wheelbase=wheelbase,
                                look_dist0=look_dist0,
                                k_ld=k_ld,
                                k_v=k_v,
                                arrive_r=arrive_r,
                                approach_r=approach_r,
                                exit_r=exit_r,
                                const_speed=const_speed,
                                accel_limit=accel_limit,
                                decel_limit=decel_limit,
                                boot_t_sec=boot_t_sec,
                                boot_d_m=boot_d_m,
                                boot_speed=boot_speed,
                                heading_update_d_min=heading_update_d_min,
                                heading_update_v_min=heading_update_v_min,
                                heading_history_len=heading_history_len,
                                startup_ld_sec=startup_ld_sec,
                                startup_ld_min=startup_ld_min,
                                startup_ld_max=startup_ld_max,
                                sequential_follow=sequential_follow,
                                start_index=start_index,
                                recover_alpha_deg=recover_alpha_deg,
                                offtrack_k=offtrack_k,
                                theta_th_deg=theta_th_deg,
                                win_ahead_min=win_ahead_min,
                                win_ahead_max=win_ahead_max,
                                search_r0=search_r0,
                                search_dr=search_dr,
                                search_rmax=search_rmax,
                                lock_time=lock_time,
                                v_heading_blend_v=v_heading_blend_v,
                                alpha_clamp_deg=alpha_clamp_deg,
                                alpha_clamp_v_thresh=alpha_clamp_v_thresh,
                                alpha_clamp_enable=alpha_clamp_enable,
                                log_csv=rospy.get_param('~log_csv', ''),
                                resample_interval_m=resample_interval_m)

        # 퍼블리셔
        self.pub_speed  = rospy.Publisher('/vehicle/speed_cmd', Float32, queue_size=10)
        self.pub_steer  = rospy.Publisher('/vehicle/steer_cmd', Float32, queue_size=10)
        self.pub_rtk    = rospy.Publisher('/rtk/status', String, queue_size=10)

        # RViz 마커 퍼블리셔
        self.pub_marker = rospy.Publisher('~markers', Marker, queue_size=10, latch=True)

        # 구독자
        self._last_carr = 0  # 0 NONE, 1 FLOAT, 2 FIX
        rospy.Subscriber(self.fix_topic, NavSatFix, self._cb_fix, queue_size=100)
        if _HAVE_RELPOSNED:
            rospy.Subscriber(self.relpos_topic, NavRELPOSNED, self._cb_relpos, queue_size=50)
        if _HAVE_NAVPVT:
            rospy.Subscriber(self.navpvt_topic, NavPVT, self._cb_navpvt, queue_size=50)

        rospy.loginfo("[tracker_ros] ready: fix=%s relpos=%s(%s) navpvt=%s(%s)",
                      self.fix_topic,
                      self.relpos_topic, 'ON' if _HAVE_RELPOSNED else 'OFF',
                      self.navpvt_topic, 'ON' if _HAVE_NAVPVT else 'OFF')

        # 초기 표시
        self._publish_waypoints_marker()
        self._publish_waypoint_rings_marker()

        # CSV 로깅
        self._log_fp = None; self._log_writer = None
        try:
            log_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else "."
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._log_path = os.path.join(log_dir, f"waypoint_log_{ts}.csv")
            self._log_fp = open(self._log_path, "w", newline="")
            self._log_writer = csv.writer(self._log_fp)
            self._log_writer.writerow(["time", "lat", "lon", "speed", "steer", "rtk", "dist", "boot"])
            self._log_fp.flush()
            rospy.loginfo(f"[tracker_ros] logging to: {self._log_path}")
        except Exception as e:
            rospy.logwarn(f"[tracker_ros] log open failed: {e}")
            self._log_fp = None; self._log_writer = None

        rospy.on_shutdown(self._close_log)

    # RTK 상태 콜백
    def _cb_relpos(self, msg):
        try:
            mask = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_MASK'))
            fixed = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_FIXED'))
            flt   = int(getattr(NavRELPOSNED, 'FLAGS_CARR_SOLN_FLOAT'))
            bits = int(msg.flags) & mask
            self._last_carr = 2 if bits == fixed else (1 if bits == flt else 0)
        except Exception:
            self._last_carr = 0

    def _cb_navpvt(self, msg):
        try:
            mask = int(getattr(NavPVT, 'FLAGS_CARRIER_PHASE_MASK'))
            fixed = int(getattr(NavPVT, 'CARRIER_PHASE_FIXED'))
            flt   = int(getattr(NavPVT, 'CARRIER_PHASE_FLOAT'))
            phase = int(msg.flags) & mask
            self._last_carr = 2 if phase == fixed else (1 if phase == flt else 0)
        except Exception:
            self._last_carr = 0

    # 메인 콜백
    def _cb_fix(self, msg: 'NavSatFix'):
        if not (math.isfinite(msg.latitude) and math.isfinite(msg.longitude)):
            return
        now = rospy.Time.now().to_sec()
        v_cmd, steer_rad, rtk_txt, d_to_tgt = self.core.update(msg.latitude, msg.longitude, now, self._last_carr)

        # 속도 퍼블리시
        if self.core.boot_active:
            v_pub = float(self.core.boot_speed)
        else:
            if self.force_const_speed:
                v_pub = float(self.core.const_speed)
            else:
                v_pub = max(self.speed_min, min(self.speed_max, v_cmd))

        # 조향 한계/단위
        steer_out = steer_rad if TrackerRosNode._STEER_IN_RAD else rad2deg(steer_rad)
        steer_lim = (self.steer_limit_deg if not TrackerRosNode._STEER_IN_RAD else deg2rad(self.steer_limit_deg))
        steer_out = max(-steer_lim, min(steer_lim, steer_out))
        unit = 'rad' if TrackerRosNode._STEER_IN_RAD else 'deg'

        rospy.loginfo(f"Lat: {msg.latitude:.7f}, Lon: {msg.longitude:.7f}, Dist: {d_to_tgt:.2f} m, RTK: {rtk_txt}, Boot: {int(self.core.boot_active)}")
        rospy.loginfo(f"Speed: {v_pub:.2f} m/s, Steering: {steer_out:.2f} {unit}")

        if self.do_publish:
            self.pub_speed.publish(Float32(data=float(v_pub)))
            self.pub_steer.publish(Float32(data=float(steer_out)))
            self.pub_rtk.publish(String(data=rtk_txt))

        # CSV 로그
        try:
            if self._log_writer is not None:
                self._log_writer.writerow([
                    f"{now:.3f}",
                    f"{msg.latitude:.8f}",
                    f"{msg.longitude:.8f}",
                    f"{float(v_pub):.3f}",
                    f"{float(steer_out):.6f}",
                    rtk_txt,
                    f"{float(d_to_tgt):.3f}",
                    int(self.core.boot_active),
                ])
                if self._log_fp is not None:
                    self._log_fp.flush()
        except Exception as e:
            rospy.logwarn(f"[tracker_ros] log write failed: {e}")

        # RViz
        self._publish_vehicle_marker()
        self._publish_target_marker()
        self._publish_waypoints_marker()
        self._publish_waypoint_rings_marker()

        # Matplotlib
        if self.viz is not None:
            self.viz.ingest(self.core,
                            veh_heading_rad=self.core.veh_heading,
                            steer_rad=steer_rad,
                            arrive_r=self.core.arrive_r)

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.viz is not None:
                self.viz.render()
            rate.sleep()
        self.core.save_logs()

    def _close_log(self):
        try:
            if self._log_fp is not None:
                self._log_fp.flush(); self._log_fp.close()
                rospy.loginfo(f"[tracker_ros] log saved: {getattr(self, '_log_path', '(unknown)')}")
        except Exception as e:
            rospy.logwarn(f"[tracker_ros] log close failed: {e}")

    # ================== RViz Marker ==================
    def _publish_waypoints_marker(self):
        pts = self.core.viz_waypoints
        if not pts: return
        mk = Marker()
        mk.header.frame_id = self.frame_id; mk.header.stamp = rospy.Time.now()
        mk.ns = "tracker"; mk.id = 1
        mk.type = Marker.LINE_STRIP; mk.action = Marker.ADD
        mk.scale.x = 0.1
        mk.color.r = 0.0; mk.color.g = 1.0; mk.color.b = 0.0; mk.color.a = 1.0
        mk.pose.orientation.w = 1.0
        mk.points = [Point(x=p[0], y=p[1], z=0.0) for p in pts]
        self.pub_marker.publish(mk)

    def _publish_vehicle_marker(self):
        p = self.core.viz_vehicle_xy
        if not p: return
        mk = Marker()
        mk.header.frame_id = self.frame_id; mk.header.stamp = rospy.Time.now()
        mk.ns = "tracker"; mk.id = 2
        mk.type = Marker.SPHERE; mk.action = Marker.ADD
        mk.scale.x = 0.4; mk.scale.y = 0.4; mk.scale.z = 0.1
        mk.color.r = 0.1; mk.color.g = 0.4; mk.color.b = 1.0; mk.color.a = 1.0
        mk.pose.position.x = p[0]; mk.pose.position.y = p[1]; mk.pose.position.z = 0.0
        mk.pose.orientation.w = 1.0
        self.pub_marker.publish(mk)

    def _publish_target_marker(self):
        p = self.core.viz_target_xy
        if not p: return
        mk = Marker()
        mk.header.frame_id = self.frame_id; mk.header.stamp = rospy.Time.now()
        mk.ns = "tracker"; mk.id = 3
        mk.type = Marker.SPHERE; mk.action = Marker.ADD
        r = float(self.core.arrive_r)
        mk.scale.x = 2*r; mk.scale.y = 2*r; mk.scale.z = 0.05
        mk.color.r = 1.0; mk.color.g = 0.2; mk.color.b = 0.2; mk.color.a = 0.6
        mk.pose.position.x = p[0]; mk.pose.position.y = p[1]; mk.pose.position.z = 0.0
        mk.pose.orientation.w = 1.0
        self.pub_marker.publish(mk)

    def _publish_waypoint_rings_marker(self):
        if not self.rviz_wp_rings: return
        pts = self.core.viz_waypoints
        if not pts: return
        mk = Marker()
        mk.header.frame_id = self.frame_id; mk.header.stamp = rospy.Time.now()
        mk.ns = "tracker"; mk.id = 4
        mk.type = Marker.SPHERE_LIST; mk.action = Marker.ADD
        r = float(self.core.arrive_r)
        mk.scale.x = 2*r; mk.scale.y = 2*r; mk.scale.z = 0.05
        mk.color.r = 0.0; mk.color.g = 0.4; mk.color.b = 1.0; mk.color.a = 0.18
        mk.pose.orientation.w = 1.0
        step = max(1, int(self.rviz_ring_step))
        mk.points = [Point(x=p[0], y=p[1], z=0.0) for i, p in enumerate(pts) if i % step == 0]
        self.pub_marker.publish(mk)

# ================== entry ==================
def _run_ros_node():
    rospy.init_node('waypoint_tracker_returnable')
    node = TrackerRosNode()
    try:
        node.spin()
    finally:
        node.core.save_logs()

if __name__ == '__main__':
    if not _ROS_OK:
        raise RuntimeError('ROS dependencies missing. Install ros-noetic-ublox / ros-noetic-ublox-msgs and source your setup.bash')
    _run_ros_node()
