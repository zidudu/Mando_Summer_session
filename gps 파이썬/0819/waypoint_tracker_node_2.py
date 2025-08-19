#!/usr/bin/env python3
"""
waypoint_tracker_node.py — 복귀 로직 통합 완전체
─────────────────────────────────────────────────────────────────────────────
핵심 기능
  • GNSS(NavSatFix) 수신 → 좌표계 통일(ENU or Mercator) → 경로 투영/진행도(s) 계산
  • 정상 추종: lookahead Ld = look_dist0 + k_v·v 앞의 경로점을 타깃
  • 복귀 모드: (A) 전방 윈도우 각도 게이트 + 비용 최소화 → (B) 헤딩 레이–경로 교차점 → (C) 반경 확장
  • 진행도(s) 단조 증가(뒤로 점프 금지), 타깃 락(T_lock)으로 토글링 방지
  • Pure-Pursuit 조향각(δ) 계산: δ = atan2(2L·sin(α), Ld)
  • 속도 램프 + 가감속 제한(슬루율)

주의/전제
  • CSV 좌표가 wgs84(lat,lon)인지, mercator(m)인지 파라미터로 맞춰야 합니다.
  • ENU는 좁은 영역에서 근사(수 km 규모 적합). Mercator는 전역 OK(극지 제외).
  • steer_cmd 단위는 ~steer_in_rad로 선택(기본 deg).

권장 파라미터(초기값)
  • theta_th_deg=30, win_ahead_min=3, win_ahead_max=40, search_r0=1.0, search_dr=0.5, search_rmax=10.0,
    lock_time=0.7, look_dist0=1.0, k_v=0.5, arrive_r=1.0, approach_r=3.0, accel_limit=0.5, decel_limit=0.8
─────────────────────────────────────────────────────────────────────────────
"""

import math
import csv
import rospy
from typing import List, Tuple, Optional
from std_msgs.msg import Float32, String
from sensor_msgs.msg import NavSatFix

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
        """점 p를 폴리라인에 직교 투영.
        반환: (dist, q, seg_idx, t, s_at_q)
          • dist: p와 투영점 q의 거리
          • q: 투영점 좌표
          • seg_idx: q가 속한 세그먼트 인덱스 i (구간 i→i+1)
          • t: 세그먼트 내 보간 파라미터 [0,1]
          • s_at_q: 경로 누적거리 s (q까지)
        """
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

    def ray_intersections(self, p: Tuple[float,float], heading: float) -> List[Tuple[float, Tuple[float,float], float, int]]:
        """차량 위치 p에서 heading 방향의 반직선과 경로 세그먼트 교차점 목록 반환.
        반환 요소: (u_ray, q, s_at_q, seg_idx) — u_ray는 레이 파라미터(작을수록 가까움, u_ray>=0)
        """
        px, py = p
        vx = math.cos(heading)
        vy = math.sin(heading)
        out = []
        for i, ((ax, ay), (bx, by)) in enumerate(self.segs):
            sx, sy = bx-ax, by-ay
            denom = vx*(-sy) + vy*(sx)
            if abs(denom) < 1e-12:
                continue  # 평행/거의 평행
            # 교차: p + u*[vx,vy] = a + t*[sx,sy]
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

        self.log_csv = log_csv

        # 상태 변수
        self._lat0 = None
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
            # 복귀: 순차 전략 (A) 후보 탐색 → (B) 레이 교차 → (C) 반경 확장
            found = False
            # (A) 전방 윈도우 + 각도 게이트 + 비용
            cand = self._search_ahead_window(x, y, s_at_q)
            if cand is not None:
                target_idx, target_xy, target_s = cand
                found = True
            if not found:
                # (B) 레이–경로 교차
                cand2 = self._search_ray_intersection((x, y), s_at_q, self._veh_heading)
                if cand2 is not None:
                    target_xy, target_s, target_idx = cand2
                    found = True
            if not found:
                # (C) 반경 확장
                cand3 = self._search_expand_radius((x, y), s_at_q)
                if cand3 is not None:
                    target_idx, target_xy, target_s = cand3
                    found = True
            if not found:
                # 최악의 경우: 투영점 근방으로 안전 fallback
                target_s = min(self._poly.total_len, s_at_q + max(0.5, Ld*0.5))
                target_idx = self._index_near_s(target_s)
                target_xy = self._poly.pts[target_idx]

        # 히스테리시스(락)
        if self._last_target_s is not None and (now - self._last_lock_time) < self.lock_time:
            # 기존 타깃 유지 조건(헤딩 게이트 1.5배 허용 & 진행도 증가)
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

        # 조향각: Pure-Pursuit
        alpha = wrap_pi(math.atan2(target_xy[1]-y, target_xy[0]-x) - self._veh_heading)
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

        return v_cmd, steer_rad, rtk_txt, d_to_target

    # ---------- 복귀 전략 구현 ----------
    def _search_ahead_window(self, x: float, y: float, s_curr: float):
        # 투영 세그먼트 찾기
        _, _, seg_idx, _, _ = self._poly.project((x, y))
        best = None
        bestJ = 1e18
        for idx in self._poly.index_window(seg_idx, self.win_ahead_min, self.win_ahead_max):
            px, py = self._poly.pts[idx]
            vx = px - x
            vy = py - y
            dist = math.hypot(vx, vy)
            if dist < 1e-6:
                continue
            ang_err = abs(wrap_pi(math.atan2(vy, vx) - self._veh_heading))
            s_i = self._poly.s_of_index(idx)
            if s_i < s_curr:  # 뒤로 금지
                continue
            if ang_err > self.theta_th:
                continue
            # 비용: 거리 + 각도
            J = dist + (ang_err / self.theta_th)  # 간단 가중
            if J < bestJ:
                bestJ = J
                best = (idx, (px, py), s_i)
        return best

    def _search_ray_intersection(self, p: Tuple[float,float], s_curr: float, heading: float):
        inters = self._poly.ray_intersections(p, heading)
        for u_ray, q, s_q, seg_idx in inters:
            if s_q >= s_curr + 0.01:  # 약간 앞쪽
                idx = self._index_near_s(s_q)
                return (q, s_q, idx)
        return None

    def _search_expand_radius(self, p: Tuple[float,float], s_curr: float):
        x, y = p
        R = self.search_r0
        while R <= self.search_rmax + 1e-9:
            best = None
            bestJ = 1e18
            # 전 구간 스캔(최적화 여지 있음)
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
        # cum_s는 단조 증가이므로 선형/이분탐색 가능. 간단히 선형(윈도우 작으면 OK)
        cs = self._poly.cum_s
        n = len(cs)
        if s <= cs[0]:
            return 0
        if s >= cs[-1]:
            return len(self._poly.pts)-1
        # 이분 탐색
        lo, hi = 0, n-1
        while lo+1 < hi:
            mid = (lo+hi)//2
            if cs[mid] <= s:
                lo = mid
            else:
                hi = mid
        # s는 [cs[lo], cs[hi]] 사이
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

        self.do_publish = bool(rospy.get_param('~do_publish', True))
        TrackerRosNode._STEER_IN_RAD = bool(rospy.get_param('~steer_in_rad', False))
        self.rate_hz = float(rospy.get_param('~publish_rate', 50.0))

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
                                log_csv=rospy.get_param('~log_csv', ''))

        # 퍼블리셔
        self.pub_speed = rospy.Publisher('/vehicle/speed_cmd', Float32, queue_size=10)
        self.pub_steer = rospy.Publisher('/vehicle/steer_cmd', Float32, queue_size=10)
        self.pub_rtk   = rospy.Publisher('/rtk/status', String, queue_size=10)

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

        steer_out = steer_rad if TrackerRosNode._STEER_IN_RAD else rad2deg(steer_rad)
        unit = 'rad' if TrackerRosNode._STEER_IN_RAD else 'deg'

        rospy.loginfo(f"Lat: {msg.latitude:.7f}, Lon: {msg.longitude:.7f}, Dist: {d_to_tgt:.2f} m, RTK: {rtk_txt}")
        rospy.loginfo(f"Speed: {v_cmd:.2f} m/s, Steering: {steer_out:.2f} {unit}")

        if self.do_publish:
            self.pub_speed.publish(Float32(data=float(v_cmd)))
            self.pub_steer.publish(Float32(data=float(steer_out)))
            self.pub_rtk.publish(String(data=rtk_txt))

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            rate.sleep()
        self.core.save_logs()


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
