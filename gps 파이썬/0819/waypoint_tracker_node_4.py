#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waypoint_tracker_node.py — 복귀 로직 + 저속 헤딩 보정 + 한계 클램프 + RViz 시각화 + Matplotlib 라이브 뷰(옵션)
"""

import math
import csv
import time
import rospy
from typing import List, Tuple, Optional
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

# ================== 좌표 변환 유틸 ==================
_R_EARTH = 6_378_137.0  # WGS84 radius (m)

def deg2rad(x: float) -> float:
    return x * math.pi / 180.0

def rad2deg(x: float) -> float:
    return x * 180.0 / math.pi

def wrap_pi(a: float) -> float:
    """[-pi, pi]로 wrap"""
    return math.atan2(math.sin(a), math.cos(a))

def mercator_xy(lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
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
    """각도 a→b로 t(0..1) 보간"""
    d = wrap_pi(b - a)
    return wrap_pi(a + t * d)

# ================== Matplotlib Live Visualizer (옵션) ==================
try:
    import matplotlib  # backend은 생성자에서 설정
    _HAVE_MPL_BASE = True
except Exception as _e:
    rospy.logwarn(f"[tracker_ros] Matplotlib base import 실패: {_e}")
    _HAVE_MPL_BASE = False

class MatplotViz:
    """RViz와 별개로 2D 실시간 궤적/타깃/헤딩을 Matplotlib 창에 표시"""
    def __init__(self, backend='TkAgg', follow=True, range_m=12.0, rate_hz=8.0):
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
            rospy.logwarn(f"[tracker_ros] Matplotlib backend/pyplot 로드 실패: {e}")
            return

        self._ok = True
        self.follow = follow
        self.range_m = float(range_m)
        self.min_dt = 1.0/float(rate_hz)
        self._last_draw = 0.0

        self.plt.ion()
        self.fig, self.ax = self.plt.subplots(figsize=(7,7))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid(True, ls=':', alpha=0.5)
        self.ax.set_title("Waypoint Tracker — Live View")

        # 기본 아티팩트
        (self.line_wp,)  = self.ax.plot([], [], '-',  c='0.6', lw=1.2, label='Waypoints')
        (self.pt_car,)   = self.ax.plot([], [], 'o',  c='tab:blue',  ms=8, label='Vehicle')
        (self.pt_tgt,)   = self.ax.plot([], [], '*',  c='tab:red',   ms=12, label='Target')
        (self.line_tgt,) = self.ax.plot([], [], '--', c='tab:cyan',  lw=1.0, label='Target Line')

        self.txt = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                ha='left', va='top',
                                bbox=dict(fc='white', alpha=0.7), fontsize=9)

        self.arr_head = None
        self.arr_steer = None
        self.arrive_ring = None

        self.ax.legend(loc='upper right')

    def _safe_del(self, artist):
        try:
            if artist is not None:
                artist.remove()
        except Exception:
            pass

    def update(self, core, veh_heading_rad=None, steer_rad=None, arrive_r=1.0):
        if not self._ok:
            return
        now = time.time()
        if (now - self._last_draw) < self.min_dt:
            return
        self._last_draw = now

        wps = core.viz_waypoints
        car = core.viz_vehicle_xy
        tgt = core.viz_target_xy

        if wps:
            xs = [p[0] for p in wps]
            ys = [p[1] for p in wps]
            self.line_wp.set_data(xs, ys)
        if car:
            self.pt_car.set_data([car[0]],[car[1]])
        if tgt:
            self.pt_tgt.set_data([tgt[0]],[tgt[1]])
            if car:
                self.line_tgt.set_data([car[0], tgt[0]],[car[1], tgt[1]])

        # 도착 반경 링
        self._safe_del(self.arrive_ring)
        if tgt:
            self.arrive_ring = self.Circle((tgt[0], tgt[1]), arrive_r, ec='tab:blue', ls='--', fc='none', alpha=0.7)
            self.ax.add_patch(self.arrive_ring)

        # 헤딩/조향 화살표
        self._safe_del(self.arr_head)
        self._safe_del(self.arr_steer)
        if car and veh_heading_rad is not None:
            hx, hy = car[0] + math.cos(veh_heading_rad), car[1] + math.sin(veh_heading_rad)
            self.arr_head = self.FancyArrowPatch((car[0],car[1]), (hx,hy), color='tab:blue',
                                                 lw=2, arrowstyle='-|>', mutation_scale=14)
            self.ax.add_patch(self.arr_head)
        if car and (veh_heading_rad is not None) and (steer_rad is not None):
            sx, sy = car[0] + math.cos(veh_heading_rad+steer_rad), car[1] + math.sin(veh_heading_rad+steer_rad)
            self.arr_steer = self.FancyArrowPatch((car[0],car[1]), (sx,sy), color='tab:red',
                                                  lw=2, arrowstyle='-|>', mutation_scale=14, alpha=0.8)
            self.ax.add_patch(self.arr_steer)

        # 정보 박스
        info_lines = []
        if car: info_lines.append(f"Veh: ({car[0]:.1f},{car[1]:.1f}) m")
        if tgt and car:
            d = math.hypot(tgt[0]-car[0], tgt[1]-car[1])
            info_lines.append(f"→ Target: {d:.1f} m")
        if veh_heading_rad is not None:
            info_lines.append(f"Heading: {math.degrees(veh_heading_rad):.1f}°")
        if steer_rad is not None:
            info_lines.append(f"Steer: {math.degrees(steer_rad):+.1f}°")
        self.txt.set_text("\n".join(info_lines))

        # 뷰 제어
        if self.follow and car:
            r = self.range_m
            self.ax.set_xlim(car[0]-r, car[0]+r)
            self.ax.set_ylim(car[1]-r, car[1]+r)
        elif wps:
            self.ax.relim(); self.ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

# ================== Polyline 유틸 ==================
class Polyline:
    def __init__(self, pts: List[Tuple[float, float]]):
        if len(pts) < 2:
            raise ValueError("Polyline needs >= 2 points")
        self.pts = pts
        self.segs = [(pts[i], pts[i+1]) for i in range(len(pts)-1)]
        self.seg_vec = [(b[0]-a[0], b[1]-a[1]) for a,b in self.segs]
        self.seg_len = [math.hypot(v[0], v[1]) for v in self.seg_vec]
        self.cum_s = [0.0]
        s = 0.0
        for L in self.seg_len:
            s += L
            self.cum_s.append(s)   # 길이: N (pts와 동일 길이)
        self.total_len = s

    def project(self, p: Tuple[float, float]) -> Tuple[float, Tuple[float, float], int, float, float]:
        """점 p를 폴리라인에 직교 투영. 반환: (dist, q, seg_idx, t, s_at_q)"""
        px, py = p
        best = (1e18, (0.0, 0.0), 0, 0.0, 0.0)
        for i, ((ax, ay), (bx, by)) in enumerate(self.segs):
            vx, vy = self.seg_vec[i]
            L2 = vx*vx + vy*vy
            if L2 <= 1e-12:
                continue
            t = ((px-ax)*vx + (py-ay)*vy) / L2
            t = max(0.0, min(1.0, t))
            qx = ax + t*vx
            qy = ay + t*vy
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

    def index_window(self, center_seg: int, dmin: int, dmax: int) -> range:
        i0 = max(0, center_seg + dmin)
        i1 = min(len(self.pts)-1, center_seg + dmax)
        return range(i0, i1+1)

    def ray_intersections(self, p: Tuple[float,float], heading: float):
        """차량 위치 p에서 heading 방향 반직선과 경로 세그먼트 교차점 목록 반환."""
        px, py = p
        vx = math.cos(heading)
        vy = math.sin(heading)
        out = []
        for i, ((ax, ay), (bx, by)) in enumerate(self.segs):
            sx, sy = bx-ax, by-ay
            denom = vx*(-sy) + vy*(sx)
            if abs(denom) < 1e-12:
                continue  # 평행/거의 평행
            dx, dy = ax - px, ay - py
            u = (dx*(-sy) + dy*(sx)) / denom
            t = (vx*dy - vy*dx) / denom
            if u >= 0.0 and 0.0 <= t <= 1.0:
                qx = ax + t*sx
                qy = ay + t*sy
                s_at_q = self.cum_s[i] + t*self.seg_len[i]
                out.append((u, (qx,qy), s_at_q, i))
        out.sort(key=lambda x: x[0])
        return out

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
                 # 복귀/검색 파라미터
                 theta_th_deg: float,
                 win_ahead_min: int,
                 win_ahead_max: int,
                 search_r0: float,
                 search_dr: float,
                 search_rmax: float,
                 lock_time: float,
                 # 저속 헤딩 보정 & 알파 클램프
                 v_heading_blend_v: float,
                 alpha_clamp_deg: float,
                 alpha_clamp_v_thresh: float,
                 alpha_clamp_enable: bool,
                 log_csv: str = ""):
        # 기본 설정
        self.coord_mode = coord_mode.lower()
        self.anchor_source = anchor_source.lower()
        self.anchor_lat = anchor_lat
        self.anchor_lon = anchor_lon

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

        self.theta_th = deg2rad(theta_th_deg)
        self.win_ahead_min = win_ahead_min
        self.win_ahead_max = win_ahead_max
        self.search_r0 = search_r0
        self.search_dr = search_dr
        self.search_rmax = search_rmax
        self.lock_time = lock_time

        # 저속 헤딩 보정 & 알파 클램프
        self.v_heading_blend_v = max(0.0, v_heading_blend_v)    # 이 속도 미만에서 경로헤딩 가중
        self.alpha_clamp_rad = deg2rad(alpha_clamp_deg)
        self.alpha_clamp_v_thresh = max(0.0, alpha_clamp_v_thresh)
        self.alpha_clamp_enable = bool(alpha_clamp_enable)

        self.log_csv = log_csv

        # 상태 변수
        self._lat0 = None
        thelon = None
        self._lon0 = None
        self._anchor_ready = False

        self._prev_time = None
        self._prev_speed = 0.0
        self._prev_xy: Optional[Tuple[float,float]] = None
        self._veh_heading = 0.0

        self._poly: Optional[Polyline] = None
        self._wps_raw = self._load_waypoints(csv_path)
        self._wps_xy: Optional[List[Tuple[float,float]]] = None

        self._last_target_idx: Optional[int] = None
        self._last_target_s: Optional[float] = None
        self._last_lock_time: float = -1e9

        # 시각화용 최신 상태
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
            else:  # fix 수신 후 결정
                self._anchor_ready = False
        elif self.coord_mode == 'mercator':
            self._prepare_wp_xy_mercator()
            self._anchor_ready = True
        else:
            raise ValueError("coord_mode must be 'wgs84' or 'mercator'")

    # ---------- Waypoints ----------
    def _load_waypoints(self, path: str) -> List[Tuple[float,float]]:
        data = []
        with open(path, 'r') as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) < 2:
                    continue
                data.append((float(row[0]), float(row[1])))
        if len(data) < 2:
            raise RuntimeError("Need >=2 waypoints in CSV")
        return data

    def _prepare_wp_xy_wgs84(self):
        self._wps_xy = [enu_xy(lat, lon, self._lat0, self._lon0) for (lat,lon) in self._wps_raw]
        self._poly = Polyline(self._wps_xy)
        self._anchor_ready = True

    def _prepare_wp_xy_mercator(self):
        self._wps_xy = [tuple(w) for w in self._wps_raw]
        self._poly = Polyline(self._wps_xy)

    # ---------- 업데이트 ----------
    def update(self, lat: float, lon: float, now: float, rtk_code: int) -> Tuple[float, float, str, float]:
        # 좌표 변환
        if self.coord_mode == 'wgs84':
            if not self._anchor_ready:
                self._lat0, self._lon0 = float(lat), float(lon)
                self._prepare_wp_xy_wgs84()
            x, y = enu_xy(lat, lon, self._lat0, self._lon0)
        else:
            x, y = mercator_xy(lat, lon)

        # 차량 헤딩(전-현 좌표) 업데이트
        if self._prev_xy is not None:
            dx_h = x - self._prev_xy[0]
            dy_h = y - self._prev_xy[1]
            if dx_h*dx_h + dy_h*dy_h > 1e-4:  # 1 cm^2 이상 이동
                self._veh_heading = math.atan2(dy_h, dx_h)
        self._prev_xy = (x, y)

        # 경로 투영
        dist_to_path, q, seg_idx, t_on_seg, s_at_q = self._poly.project((x, y))
        path_heading = self._poly.heading_at(seg_idx)

        # lookahead 거리 (속도 기반)
        v_prev = self._prev_speed
        Ld = max(0.1, self.look_dist0 + self.k_v * max(0.0, v_prev))

        # 정상 추종 조건
        heading_err = abs(wrap_pi(self._veh_heading - path_heading))
        on_track = (dist_to_path <= self.arrive_r) and (heading_err <= self.theta_th)

        # 타깃 결정
        target_xy, target_s, target_idx = None, None, None

        if on_track:
            # 정상: s*+Ld 지점 타깃
            target_s = min(self._poly.total_len, s_at_q + Ld)
            target_idx = self._index_near_s(target_s)
            target_xy = self._poly.pts[target_idx]
        else:
            # 복귀: (A) 전방 윈도우 → (B) 레이 교차 → (C) 반경 확장
            found = False
            cand = self._search_ahead_window(x, y, s_at_q)
            if cand is not None:
                target_idx, target_xy, target_s = cand
                found = True
            if not found:
                cand2 = self._search_ray_intersection((x, y), s_at_q, self._veh_heading)
                if cand2 is not None:
                    target_xy, target_s, target_idx = cand2
                    found = True
            if not found:
                cand3 = self._search_expand_radius((x, y), s_at_q)
                if cand3 is not None:
                    target_idx, target_xy, target_s = cand3
                    found = True
            if not found:
                target_s = min(self._poly.total_len, s_at_q + max(0.5, Ld*0.5))
                target_idx = self._index_near_s(target_s)
                target_xy = self._poly.pts[target_idx]

        # 타깃 락(토글 방지)
        if self._last_target_s is not None and (now - self._last_lock_time) < self.lock_time:
            keep = False
            if self._last_target_idx is not None:
                last_xy = self._poly.pts[self._last_target_idx]
                ang = abs(wrap_pi(self._veh_heading - math.atan2(last_xy[1]-y, last_xy[0]-x)))
                if ang <= 1.5*self.theta_th and self._last_target_s >= s_at_q:
                    keep = True
            if keep:
                target_idx = self._last_target_idx
                target_s = self._last_target_s
                target_xy = self._poly.pts[target_idx]

        # 타깃 확정
        self._last_target_idx = target_idx
        self._last_target_s = target_s
        self._last_lock_time = now

        # ─── 저속 헤딩 보정 + α 클램프 ───
        # 1) 저속에서는 경로 헤딩으로 보정(블렌딩)
        veh_heading_eff = self._veh_heading
        if self.v_heading_blend_v > 1e-6 and v_prev < self.v_heading_blend_v:
            # v=0 → 경로헤딩 100% / v=blend_v → 보정 0%
            t = 1.0 - (v_prev / self.v_heading_blend_v)
            veh_heading_eff = slerp_angle(self._veh_heading, path_heading, t)

        # 2) α 계산
        alpha = wrap_pi(math.atan2(target_xy[1]-y, target_xy[0]-x) - veh_heading_eff)

        # 3) 저속에서는 α 클램프(예: ±60°)
        if self.alpha_clamp_enable and v_prev < self.alpha_clamp_v_thresh:
            cap = self.alpha_clamp_rad
            if alpha > cap:
                alpha = cap
            elif alpha < -cap:
                alpha = -cap

        # 조향각: Pure-Pursuit
        steer_rad = math.atan2(2.0*self.wheelbase*math.sin(alpha), max(0.1, Ld))

        # 속도 명령: 거리 기반 램프 + 슬루율
        d_to_target = math.hypot(target_xy[0]-x, target_xy[1]-y)
        tgt_speed = 0.0 if d_to_target <= self.arrive_r else (
            self.const_speed if d_to_target >= self.approach_r else
            self.const_speed * (d_to_target - self.arrive_r)/(self.approach_r - self.arrive_r)
        )
        dt = 0.0 if self._prev_time is None else max(0.0, now - self._prev_time)
        v_cmd = self._apply_slew(self._prev_speed, tgt_speed, dt)
        self._prev_speed = v_cmd
        self._prev_time = now

        # RTK 문자열
        rtk_txt = 'FIX' if rtk_code == 2 else ('FLOAT' if rtk_code == 1 else 'NONE')

        # 시각화 상태 저장
        self._viz_xy = (x, y)
        self._viz_target = (target_xy[0], target_xy[1])

        return v_cmd, steer_rad, rtk_txt, d_to_target

    # ---------- 복귀 전략 구현 ----------
    def _search_ahead_window(self, x: float, y: float, s_curr: float):
        _, _, seg_idx, _, _ = self._poly.project((x, y))
        best = None
        bestJ = 1e18
        for idx in self._poly.index_window(seg_idx, self.win_ahead_min, self.win_ahead_max):
            px, py = self._poly.pts[idx]
            vx, vy = px - x, py - y
            dist = math.hypot(vx, vy)
            if dist < 1e-6:
                continue
            ang_err = abs(wrap_pi(math.atan2(vy, vx) - self._veh_heading))
            s_i = self._poly.s_of_index(idx)
            if s_i < s_curr:
                continue
            if ang_err > self.theta_th:
                continue
            J = dist + (ang_err / self.theta_th)
            if J < bestJ:
                bestJ = J
                best = (idx, (px, py), s_i)
        return best

    def _search_ray_intersection(self, p: Tuple[float,float], s_curr: float, heading: float):
        inters = self._poly.ray_intersections(p, heading)
        for u_ray, q, s_q, seg_idx in inters:
            if s_q >= s_curr + 0.01:
                idx = self._index_near_s(s_q)
                return (q, s_q, idx)
        return None

    def _search_expand_radius(self, p: Tuple[float,float], s_curr: float):
        x, y = p
        R = self.search_r0
        while R <= self.search_rmax + 1e-9:
            best = None
            bestJ = 1e18
            for idx, (px, py) in enumerate(self._poly.pts):
                s_i = self._poly.s_of_index(idx)
                if s_i < s_curr:
                    continue
                dx, dy = px-x, py-y
                dist = math.hypot(dx, dy)
                if dist > R:
                    continue
                ang_err = abs(wrap_pi(math.atan2(dy, dx) - self._veh_heading))
                if ang_err > self.theta_th:
                    continue
                J = dist + (ang_err / self.theta_th)
                if J < bestJ:
                    bestJ = J
                    best = (idx, (px, py), s_i)
            if best is not None:
                return best
            R += self.search_dr
        return None

    # ---------- 보조 ----------
    def _index_near_s(self, s: float) -> int:
        cs = self._poly.cum_s
        n = len(cs)
        if s <= cs[0]:
            return 0
        if s >= cs[-1]:
            return len(self._poly.pts)-1
        lo, hi = 0, n-1
        while lo+1 < hi:
            mid = (lo+hi)//2
            if cs[mid] <= s:
                lo = mid
            else:
                hi = mid
        return lo if (s - cs[lo]) <= (cs[hi]-s) else hi

    def _apply_slew(self, prev_v: float, tgt_v: float, dt: float) -> float:
        if dt <= 0.0:
            return tgt_v
        if tgt_v > prev_v:
            dv = min(tgt_v - prev_v, self.accel_limit * dt)
            return prev_v + dv
        else:
            dv = min(prev_v - tgt_v, self.decel_limit * dt)
            return prev_v - dv

    # ---------- 시각화 상태 조회(노드에서 접근) ----------
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

    def save_logs(self):
        pass

# ================== TrackerRosNode ==================
class TrackerRosNode:
    _STEER_IN_RAD = False

    def __init__(self):
        # 파라미터 로드
        csv_path     = rospy.get_param('~csv_path', '/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/waypoints_example.csv')
        coord_mode   = rospy.get_param('~coord_mode', 'wgs84')            # 'wgs84' | 'mercator'
        anchor_src   = rospy.get_param('~anchor_source', 'waypoint')      # 'waypoint' | 'fix'
        anchor_lat   = float(rospy.get_param('~anchor_lat', float('nan')))
        anchor_lon   = float(rospy.get_param('~anchor_lon', float('nan')))

        wheelbase    = float(rospy.get_param('~wheelbase', 0.28))
        look_dist0   = float(rospy.get_param('~look_dist0', 1.0))
        k_ld         = float(rospy.get_param('~k_ld', 0.0))               # (미사용, 인터페이스 유지)
        k_v          = float(rospy.get_param('~k_v', 0.5))                # Ld = look_dist0 + k_v*v
        arrive_r     = float(rospy.get_param('~arrive_r', 1.0))
        approach_r   = float(rospy.get_param('~approach_r', 3.0))
        exit_r       = float(rospy.get_param('~exit_r', 0.8))
        const_speed  = float(rospy.get_param('~const_speed', 1.0))
        accel_limit  = float(rospy.get_param('~accel_limit', 0.5))        # m/s^2
        decel_limit  = float(rospy.get_param('~decel_limit', 0.8))        # m/s^2

        # 복귀/검색 파라미터
        theta_th_deg = float(rospy.get_param('~theta_th_deg', 30.0))
        win_ahead_min= int(rospy.get_param('~win_ahead_min', 3))
        win_ahead_max= int(rospy.get_param('~win_ahead_max', 40))
        search_r0    = float(rospy.get_param('~search_r0', 1.0))
        search_dr    = float(rospy.get_param('~search_dr', 0.5))
        search_rmax  = float(rospy.get_param('~search_rmax', 10.0))
        lock_time    = float(rospy.get_param('~lock_time', 0.7))

        # 저속 헤딩 보정 & α 클램프
        v_heading_blend_v     = float(rospy.get_param('~v_heading_blend_v', 0.3))  # m/s
        alpha_clamp_deg       = float(rospy.get_param('~alpha_clamp_deg', 60.0))
        alpha_clamp_v_thresh  = float(rospy.get_param('~alpha_clamp_v_thresh', 0.5))
        alpha_clamp_enable    = bool(rospy.get_param('~alpha_clamp_enable', True))

        self.do_publish = bool(rospy.get_param('~do_publish', True))
        TrackerRosNode._STEER_IN_RAD = bool(rospy.get_param('~steer_in_rad', False))
        self.rate_hz = float(rospy.get_param('~publish_rate', 50.0))

        # 하위 액추에이터 한계/단위
        self.speed_min = float(rospy.get_param('~speed_min', 0.0))
        self.speed_max = float(rospy.get_param('~speed_max', const_speed))
        self.steer_limit_deg = float(rospy.get_param('~steer_limit_deg', 30.0))  # ±limit

        # RViz frame
        self.frame_id = rospy.get_param('~frame_id', 'map')

        # ── Matplotlib Viz 옵션 ──
        self.viz_enable  = bool(rospy.get_param('~viz_enable', True))   # 기본 on
        self.viz_backend = rospy.get_param('~viz_backend', 'TkAgg')     # Qt5Agg, TkAgg 등
        self.viz_follow  = bool(rospy.get_param('~viz_follow', True))
        self.viz_range_m = float(rospy.get_param('~viz_range_m', 12.0))
        self.viz_rate_hz = float(rospy.get_param('~viz_rate_hz', 8.0))
        self.viz = MatplotViz(backend=self.viz_backend,
                              follow=self.viz_follow,
                              range_m=self.viz_range_m,
                              rate_hz=self.viz_rate_hz) if self.viz_enable else None

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
                                log_csv=rospy.get_param('~log_csv', ''))

        # 퍼블리셔
        self.pub_speed = rospy.Publisher('/vehicle/speed_cmd', Float32, queue_size=10)
        self.pub_steer = rospy.Publisher('/vehicle/steer_cmd', Float32, queue_size=10)
        self.pub_rtk   = rospy.Publisher('/rtk/status', String, queue_size=10)

        # RViz 마커 퍼블리셔 (latch=True: RViz가 나중에 켜져도 보임)
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

        # 웨이포인트 라인 초기 1회 송출(앵커/변환 준비되면 갱신)
        self._publish_waypoints_marker()

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

        # ── 한계/단위 클램프 ──
        # 속도 제한
        v_pub = max(self.speed_min, min(self.speed_max, v_cmd))

        # 조향 단위/한계
        steer_out = steer_rad if TrackerRosNode._STEER_IN_RAD else rad2deg(steer_rad)
        steer_lim = (self.steer_limit_deg if not TrackerRosNode._STEER_IN_RAD
                     else deg2rad(self.steer_limit_deg))
        if steer_out >  steer_lim: steer_out =  steer_lim
        if steer_out < -steer_lim: steer_out = -steer_lim
        unit = 'rad' if TrackerRosNode._STEER_IN_RAD else 'deg'

        rospy.loginfo(f"Lat: {msg.latitude:.7f}, Lon: {msg.longitude:.7f}, Dist: {d_to_tgt:.2f} m, RTK: {rtk_txt}")
        rospy.loginfo(f"Speed: {v_pub:.2f} m/s, Steering: {steer_out:.2f} {unit}")

        if self.do_publish:
            self.pub_speed.publish(Float32(data=float(v_pub)))
            self.pub_steer.publish(Float32(data=float(steer_out)))
            self.pub_rtk.publish(String(data=rtk_txt))

        # RViz 시각화 갱신
        self._publish_vehicle_marker()
        self._publish_target_marker()
        self._publish_waypoints_marker()

        # Matplotlib 실시간 시각화 갱신(옵션)
        if self.viz is not None:
            self.viz.update(self.core,
                            veh_heading_rad=self.core.veh_heading,
                            steer_rad=steer_rad,
                            arrive_r=self.core.arrive_r)

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            rate.sleep()
        self.core.save_logs()

    # ================== RViz Marker ==================
    def _publish_waypoints_marker(self):
        pts = self.core.viz_waypoints
        if not pts:
            return
        mk = Marker()
        mk.header.frame_id = self.frame_id
        mk.header.stamp = rospy.Time.now()
        mk.ns = "tracker"
        mk.id = 1
        mk.type = Marker.LINE_STRIP
        mk.action = Marker.ADD
        mk.scale.x = 0.1  # 선 두께 [m]
        mk.color.r = 0.0; mk.color.g = 1.0; mk.color.b = 0.0; mk.color.a = 1.0
        mk.pose.orientation.w = 1.0
        mk.points = [Point(x=p[0], y=p[1], z=0.0) for p in pts]
        self.pub_marker.publish(mk)

    def _publish_vehicle_marker(self):
        p = self.core.viz_vehicle_xy
        if not p:
            return
        mk = Marker()
        mk.header.frame_id = self.frame_id
        mk.header.stamp = rospy.Time.now()
        mk.ns = "tracker"
        mk.id = 2
        mk.type = Marker.SPHERE
        mk.action = Marker.ADD
        mk.scale.x = 0.4; mk.scale.y = 0.4; mk.scale.z = 0.1
        mk.color.r = 0.1; mk.color.g = 0.4; mk.color.b = 1.0; mk.color.a = 1.0  # 파랑
        mk.pose.position.x = p[0]; mk.pose.position.y = p[1]; mk.pose.position.z = 0.0
        mk.pose.orientation.w = 1.0
        self.pub_marker.publish(mk)

    def _publish_target_marker(self):
        p = self.core.viz_target_xy
        if not p:
            return
        mk = Marker()
        mk.header.frame_id = self.frame_id
        mk.header.stamp = rospy.Time.now()
        mk.ns = "tracker"
        mk.id = 3
        mk.type = Marker.SPHERE
        mk.action = Marker.ADD
        mk.scale.x = 0.3; mk.scale.y = 0.3; mk.scale.z = 0.1
        mk.color.r = 1.0; mk.color.g = 0.2; mk.color.b = 0.2; mk.color.a = 1.0  # 빨강
        mk.pose.position.x = p[0]; mk.pose.position.y = p[1]; mk.pose.position.z = 0.0
        mk.pose.orientation.w = 1.0
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
