#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_current_vs_waypoints_flags_with_names.py
- 로그 CSV (lat,lon) 과 웨이포인트 CSV (Lat,Lon or x,y)를 비교/시각화
- 플래그 구간별로 다른 waypoint spacing / radius_scale 지원
- 범례 및 통계 박스에 로그 / wpt CSV 파일명 표시
- 각 웨이포인트에 대해 구간별 원 반경 시각화
- 로그에 기록된 타깃 인덱스(tgt_idx) 시각화:
    - 진한 갈색 빈 원(edge only)
    - 진한 갈색 굵은 번호 (1,2,3... 등장순서)
    - 같은 위치에 여러 번호가 있을 때 연속 범위는 'a~b'로 축약
"""
import os
import re
import math
import json
import ast
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import geopy.distance

# ----------------- 기본 경로 (필요 시 수정) -----------------
DEFAULT_LOG_CSV = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/logs/waypoint_log_20250911_025832.csv"
DEFAULT_WPT_CSV = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/config/raw_track_latlon_17_교차.csv"
DEFAULT_TRACKER_PY = "/home/jigu/catkin_ws/src/rtk_waypoint_tracker/src/waypoint_tracker_node_Flag.py"
OUT_PNG = "waypoint_vs_log_flags_with_names.png"

# 전역 기본값
GLOBAL_WAYPOINT_SPACING = 2.0   # [m]
TARGET_RADIUS_END = 2.5         # [m]
ANNOTATE_WP_INDEX = True
DRAW_WP_CIRCLES = True

# 타깃 색상 (살짝 진한 갈색)
TARGET_COLOR = "#5C4033"  # 필요하면 다른 hex로 변경 가능
# ---------------------------------------------------------

def latlon_to_xy_fn(ref_lat, ref_lon):
    def _to_xy(lat, lon):
        northing = geopy.distance.geodesic((ref_lat, ref_lon), (lat, ref_lon)).meters
        easting  = geopy.distance.geodesic((ref_lat, ref_lon), (ref_lat, lon)).meters
        if lat < ref_lat: northing *= -1
        if lon < ref_lon: easting  *= -1
        return float(easting), float(northing)
    return _to_xy

def find_col(df, key):
    keys = {c.lower(): c for c in df.columns}
    return keys.get(key.lower(), None)

def load_waypoints(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    wdf = pd.read_csv(path)
    lat_col = find_col(wdf, 'lat'); lon_col = find_col(wdf, 'lon')
    if lat_col and lon_col:
        lats = wdf[lat_col].astype(float).values
        lons = wdf[lon_col].astype(float).values
        return ('latlon', lats, lons, wdf)
    x_col = find_col(wdf, 'x'); y_col = find_col(wdf, 'y')
    if x_col and y_col:
        xs = wdf[x_col].astype(float).values
        ys = wdf[y_col].astype(float).values
        return ('xy', xs, ys, wdf)
    raise RuntimeError("웨이포인트 CSV에 'Lat/Lon' 또는 'x,y' 컬럼이 필요합니다.")

def load_log_latlon(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    lat_col = find_col(df, 'lat'); lon_col = find_col(df, 'lon')
    if lat_col is None or lon_col is None:
        raise RuntimeError("로그 CSV에 'lat','lon' 컬럼이 필요합니다.")
    lats = df[lat_col].astype(float).values
    lons = df[lon_col].astype(float).values
    return lats, lons, df

def generate_waypoints_along_path(path_pts, spacing=3.0):
    if spacing is None or spacing <= 0 or len(path_pts) < 2:
        return np.array(path_pts)
    pts = [tuple(p) for p in path_pts]
    new_points = [pts[0]]
    last_point = np.array(pts[0], dtype=float)
    dist_accum = 0.0
    for i in range(1, len(pts)):
        current_point = np.array(pts[i], dtype=float)
        segment = current_point - last_point
        seg_len = np.linalg.norm(segment)
        while dist_accum + seg_len >= spacing:
            remain = spacing - dist_accum
            if seg_len == 0:
                break
            direction = segment / seg_len
            new_point = last_point + direction * remain
            new_points.append(tuple(new_point))
            last_point = new_point
            segment = current_point - last_point
            seg_len = np.linalg.norm(segment)
            dist_accum = 0.0
        dist_accum += seg_len
        last_point = current_point
    if np.linalg.norm(np.array(new_points[-1]) - np.array(pts[-1])) > 1e-6:
        new_points.append(tuple(pts[-1]))
    return np.array(new_points)

def compute_nearest_dists(log_x, log_y, wpt_x, wpt_y):
    wx = np.asarray(wpt_x); wy = np.asarray(wpt_y)
    n_log = len(log_x)
    dists = np.zeros(n_log)
    idxs  = np.zeros(n_log, dtype=int)
    for i in range(n_log):
        dx = wx - log_x[i]
        dy = wy - log_y[i]
        d2 = dx*dx + dy*dy
        j = int(np.argmin(d2))
        dists[i] = math.hypot(wx[j]-log_x[i], wy[j]-log_y[i])
        idxs[i] = j
    return dists, idxs

def compute_stats(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    mean = float(np.mean(arr))
    std  = float(np.std(arr))
    rmse = float(np.sqrt(np.mean(arr**2)))
    mn   = float(np.min(arr))
    mx   = float(np.max(arr))
    return mean, std, rmse, mn, mx

def try_parse_FLAG_DEFS_from_py(py_path):
    if not os.path.exists(py_path):
        return []
    txt = open(py_path, 'r', encoding='utf-8', errors='ignore').read()
    m = re.search(r'FLAG_DEFS\s*=\s*(\[[\s\S]*?\])', txt)
    if not m:
        return []
    list_text = m.group(1)
    try:
        parsed = ast.literal_eval(list_text)
        return parsed
    except Exception:
        return []

def build_flag_zones(flag_defs):
    zones = []
    for fd in flag_defs:
        try:
            s0 = int(min(fd['start'], fd['end'])) - 1
            e0 = int(max(fd['start'], fd['end'])) - 1
        except Exception:
            continue
        zones.append({
            'name': fd.get('name', 'ZONE'),
            'start0': s0,
            'end0': e0,
            'radius_scale': float(fd.get('radius_scale', 1.0)),
            'lookahead_scale': float(fd.get('lookahead_scale', 1.0)),
            'speed_code': fd.get('speed_code', None),
            'speed_cap': fd.get('speed_cap', None),
            'step_per_loop': int(fd.get('step_per_loop', 1)),
            'stop_on_hit': bool(fd.get('stop_on_hit', False)),
            'stop_duration_sec': float(fd.get('stop_duration_sec', 0.0) if fd.get('stop_duration_sec', None) is not None else 0.0),
            'spacing': float(fd.get('spacing', GLOBAL_WAYPOINT_SPACING)),
            'disp_range': f"{fd.get('start')}–{fd.get('end')} (1-based)"
        })
    return zones

def find_tgt_col(df):
    # 'tgt_idx' 또는 'tgt'와 'idx'를 포함한 컬럼명을 찾아 반환
    for c in df.columns:
        lc = c.lower()
        if 'tgt' in lc and 'idx' in lc:
            return c
    for c in df.columns:
        if 'tgt' in c.lower():
            return c
    return None

def collapse_numbers(nums):
    """정렬된 정수 리스트 nums -> 연속이면 'a~b', 아니면 'a,b,c' 형태로 축약 반환"""
    if not nums:
        return ""
    nums = sorted(nums)
    runs = []
    start = nums[0]; prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        else:
            if start == prev:
                runs.append(str(start))
            else:
                runs.append(f"{start}~{prev}")
            start = n; prev = n
    # 마지막 run
    if start == prev:
        runs.append(str(start))
    else:
        runs.append(f"{start}~{prev}")
    return ",".join(runs)

def main(args):
    # file basenames for legend
    log_basename = os.path.splitext(os.path.basename(args.log))[0]
    wpt_basename = os.path.splitext(os.path.basename(args.wpt))[0]

    # load waypoints
    kind, a1, a2, wdf = load_waypoints(args.wpt)
    if kind == 'latlon':
        ref_lat = float(a1[0]); ref_lon = float(a2[0])
        to_xy = latlon_to_xy_fn(ref_lat, ref_lon)
        wpts_xy = np.array([to_xy(lat, lon) for lat, lon in zip(a1, a2)])
    else:
        wpts_xy = np.column_stack((a1, a2)).astype(float)

    # load logs
    log_lats, log_lons, logdf = load_log_latlon(args.log)
    if kind == 'latlon':
        to_xy = latlon_to_xy_fn(ref_lat, ref_lon)
        log_xy = np.array([to_xy(lat, lon) for lat, lon in zip(log_lats, log_lons)])
    else:
        # fallback: use first log point as ref
        ref_lat = float(log_lats[0]); ref_lon = float(log_lons[0])
        to_xy = latlon_to_xy_fn(ref_lat, ref_lon)
        log_xy = np.array([to_xy(lat, lon) for lat, lon in zip(log_lats, log_lons)])
        wpts_xy = wpts_xy - wpts_xy[0]  # shift

    # shift origin to log first point
    origin = log_xy[0]
    log_xy = log_xy - origin
    wpts_xy = wpts_xy - origin

    # parse flags from tracker py if provided
    flag_defs = []
    if args.tracker_py and os.path.exists(args.tracker_py):
        parsed = try_parse_FLAG_DEFS_from_py(args.tracker_py)
        if parsed:
            flag_defs = parsed

    if args.flags_json and os.path.exists(args.flags_json):
        try:
            with open(args.flags_json, 'r', encoding='utf-8') as f:
                j = json.load(f)
            if isinstance(j, list):
                flag_defs = j
        except Exception:
            pass

    flag_zones = build_flag_zones(flag_defs)

    base_spacing = args.spacing if args.spacing is not None else GLOBAL_WAYPOINT_SPACING
    spaced_all = generate_waypoints_along_path(wpts_xy, spacing=base_spacing)
    spaced_x_all = spaced_all[:,0]; spaced_y_all = spaced_all[:,1]

    # per-zone resampling
    zone_segments = []
    for z in flag_zones:
        s = z['start0']; e = z['end0']
        s = max(0, s); e = min(len(wpts_xy)-1, e)
        seg_pts = np.array(wpts_xy[s:e+1])
        seg_spacing = float(z.get('spacing', base_spacing) or base_spacing)
        seg_spaced = generate_waypoints_along_path(seg_pts, spacing=seg_spacing)
        zone_segments.append((z, seg_spaced))

    dists, nearest_idxs = compute_nearest_dists(log_xy[:,0], log_xy[:,1], spaced_x_all, spaced_y_all)
    mean, std, rmse, mn, mx = compute_stats(dists)

    # map spaced indices to zones
    idx_to_zone = np.full(len(spaced_x_all), -1, dtype=int)
    for zid, z in enumerate(flag_zones):
        s0, e0 = z['start0'], z['end0']
        if s0 < 0 or e0 >= len(wpts_xy) or s0>e0: continue
        start_pt = wpts_xy[s0]; end_pt = wpts_xy[e0]
        dist_start = (spaced_x_all - start_pt[0])**2 + (spaced_y_all - start_pt[1])**2
        dist_end   = (spaced_x_all - end_pt[0])**2 + (spaced_y_all - end_pt[1])**2
        i_start = int(np.argmin(dist_start))
        i_end   = int(np.argmin(dist_end))
        if i_start > i_end: i_start, i_end = i_end, i_start
        idx_to_zone[i_start:i_end+1] = zid

    zone_dists = defaultdict(list)
    for i, ndx in enumerate(nearest_idxs):
        zid = int(idx_to_zone[ndx])
        zone_dists[zid].append(dists[i])

    zone_stats = {}
    for zid, z in enumerate(flag_zones):
        arr = np.array(zone_dists.get(zid, []))
        zone_stats[zid] = compute_stats(arr)

    # visited detection
    visited = np.zeros(len(spaced_x_all), dtype=bool)
    for i, ndx in enumerate(nearest_idxs):
        zid = int(idx_to_zone[ndx])
        if zid >= 0:
            eff_r = TARGET_RADIUS_END * flag_zones[zid].get('radius_scale', 1.0)
        else:
            eff_r = TARGET_RADIUS_END
        if dists[i] <= eff_r:
            visited[ndx] = True

    # plotting
    fig, ax = plt.subplots(figsize=(10,10))
    # labels include basenames
    ax.plot(wpts_xy[:,0], wpts_xy[:,1], 'g-', linewidth=1.0, label=f'CSV Path ({wpt_basename})')
    ax.plot(spaced_x_all, spaced_y_all, 'b.-', markersize=3, label=f'Global spaced {base_spacing:.1f} m ({wpt_basename})')
    ax.plot(log_xy[:,0], log_xy[:,1], 'r.-', linewidth=1.0, label=f'Log Path ({log_basename})')
    ax.scatter(log_xy[0,0], log_xy[0,1], c='green', s=80, marker='o', label=f'Log Start (0) [{log_basename}]')
    ax.scatter(log_xy[-1,0], log_xy[-1,1], c='black', s=80, marker='s', label=f'Log End (1) [{log_basename}]')

    cmap = plt.get_cmap('tab10')
    for zid, (z, seg) in enumerate(zone_segments):
        color = cmap(zid % 10)
        segx, segy = seg[:,0], seg[:,1]
        ax.plot(segx, segy, '-', color=color, linewidth=2.0, label=f"{z['name']} (spacing={z.get('spacing'):.1f} m)")
        r = TARGET_RADIUS_END * z.get('radius_scale', 1.0)
        if DRAW_WP_CIRCLES:
            for (sx, sy) in zip(segx, segy):
                ax.add_patch(Circle((sx, sy), r, color=color, fill=False, linestyle='--', alpha=0.25, linewidth=0.8))
        mid_idx = len(seg)//2
        if mid_idx>0:
            ax.text(segx[mid_idx], segy[mid_idx], z['name'], fontsize=9, fontweight='bold', color=color,
                    bbox=dict(fc='white', alpha=0.7, edgecolor='none'))

    if ANNOTATE_WP_INDEX:
        step_idx = max(1, int(len(spaced_x_all)//200))
        for k, (xw, yw) in enumerate(zip(spaced_x_all, spaced_y_all), start=1):
            if k % step_idx == 0:
                ax.text(xw, yw, str(k), fontsize=7, ha='center', va='center', color='black', alpha=0.6)

    ax.scatter(spaced_x_all[~visited], spaced_y_all[~visited], c='navy', s=18, marker='.', label='WP (not visited)')
    ax.scatter(spaced_x_all[visited], spaced_y_all[visited], c='limegreen', s=46, marker='o', label='WP (visited)')

    # --- 모든 웨이포인트에 대해 '유효 반경(eff_r)' 원을 그림 (구간별 scale 적용) ---
    label_added_for_zone = {}  # zid -> bool (범례용 라벨 1회만 추가)
    label_added_for_zone[-1] = False  # 전역 반경(구간에 속하지 않는 WP)
    for i, (xw, yw) in enumerate(zip(spaced_x_all, spaced_y_all)):
        zid = int(idx_to_zone[i])
        if zid >= 0:
            z = flag_zones[zid]
            color = cmap(zid % 10)
            eff_r = TARGET_RADIUS_END * z.get('radius_scale', 1.0)
            label = None
            if not label_added_for_zone.get(zid, False):
                label = f"{z['name']} radius x{z.get('radius_scale',1.0):.2f}"
                label_added_for_zone[zid] = True
        else:
            color = 'gray'
            eff_r = TARGET_RADIUS_END
            label = None
            if not label_added_for_zone.get(-1, False):
                label = f"Global radius ({TARGET_RADIUS_END:.2f} m)"
                label_added_for_zone[-1] = True
        c = Circle((xw, yw), eff_r, color=color, fill=False, linestyle='-', alpha=0.25, linewidth=0.8, label=label)
        ax.add_patch(c)
    # ------------------------------------------------------------------------------

    # --- 로그에서 tgt_idx 컬럼 찾아서 '타깃 잡힌 순서'로 번호 매겨 시각화 ---
    tgt_col = find_tgt_col(logdf)
    if tgt_col is not None:
        # 지도상의 텍스트 오프셋 (지도 span 기준)
        span_x = max(spaced_x_all) - min(spaced_x_all) if len(spaced_x_all)>0 else 1.0
        span_y = max(spaced_y_all) - min(spaced_y_all) if len(spaced_y_all)>0 else 1.0
        tx_off = span_x * 0.006
        ty_off = span_y * 0.006

        tgt_series = pd.to_numeric(logdf[tgt_col], errors='coerce')
        seen = set()
        ordered_targets = []
        # 로그 행 순서대로 등장한 서로 다른 tgt_idx들을 수집 (양수만)
        for v in tgt_series:
            if np.isnan(v):
                continue
            vi = int(round(float(v)))
            if vi <= 0:
                continue
            if vi not in seen:
                seen.add(vi)
                ordered_targets.append(vi)

        # ordered_targets -> (order_number, idx0) 목록 생성
        targets_info = []
        for order_num, tgt_one_based in enumerate(ordered_targets, start=1):
            idx0 = int(tgt_one_based) - 1
            if idx0 < 0 or idx0 >= len(spaced_x_all):
                continue
            targets_info.append((order_num, idx0))

        # 같은 웨이포인트 인덱스(idx0)에 속한 order_num들을 그룹화
        idx_to_orders = {}
        for order_num, idx0 in targets_info:
            idx_to_orders.setdefault(idx0, []).append(order_num)

        # 그룹별로 (연속 여부 검사) -> 라벨 문자열 생성 및 플롯
        label_used = False
        for idx0, orders in idx_to_orders.items():
            xw = spaced_x_all[idx0]; yw = spaced_y_all[idx0]
            zid = int(idx_to_zone[idx0]) if len(idx_to_zone)>0 else -1
            if zid >= 0:
                eff_r = TARGET_RADIUS_END * flag_zones[zid].get('radius_scale', 1.0)
            else:
                eff_r = TARGET_RADIUS_END

            # 진한 갈색 빈 원 (edge only), 범례 라벨은 한 번만 추가
            circ = Circle((xw, yw), eff_r, edgecolor=TARGET_COLOR, facecolor='none',
                          linewidth=1.6, linestyle='-', alpha=0.95,
                          label=('Targets (order)' if not label_used else None))
            ax.add_patch(circ)
            label_used = True

            # 라벨 텍스트: 연속이면 "a~b", 아니면 "a,b,..." — 진한 갈색, 굵게 표시
            label_txt = collapse_numbers(orders)
            ax.text(xw + tx_off, yw + ty_off, label_txt,
                    color=TARGET_COLOR, fontsize=12, fontweight='bold', ha='center', va='center', zorder=10)
    # ------------------------------------------------------------------------------

    # stats text (include basenames)
    stats_lines = []
    stats_lines.append(f"Log: {log_basename}   Waypoint: {wpt_basename}")
    stats_lines.append(f"Overall errors (n={len(dists)}): mean={mean:.3f} m  std={std:.3f} m  rmse={rmse:.3f} m  max={mx:.3f} m")
    for zid, z in enumerate(flag_zones):
        m, s, r, mnz, mxz = zone_stats.get(zid, (np.nan,)*5)
        stats_lines.append(f"[{zid}] {z['name']} {z['disp_range']}: n={len(zone_dists.get(zid,[]))} mean={m:.3f} std={s:.3f} rmse={r:.3f} max={mxz:.3f}")
    ax.text(0.98, 0.98, "\n".join(stats_lines), transform=ax.transAxes, fontsize=9,
            va='top', ha='right', bbox=dict(fc='white', alpha=0.9))

    ax.set_xlabel("X [m] (local)")
    ax.set_ylabel("Y [m] (local)")
    ax.set_title("Waypoint vs Log (flag-aware) — names & per-WP radius & ordered targets (brown)")
    ax.axis('equal'); ax.grid(True)
    ax.legend(loc='upper left', fontsize=9)

    out_path = args.out
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Saved: {out_path}")
    print(f"[INFO] overall mean={mean:.3f} std={std:.3f} rmse={rmse:.3f} max={mx:.3f}")

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log",  default=DEFAULT_LOG_CSV, help="Log CSV (lat,lon columns)")
    p.add_argument("--wpt",  default=DEFAULT_WPT_CSV, help="Waypoints CSV (Lat,Lon or x,y)")
    p.add_argument("--tracker_py", default=DEFAULT_TRACKER_PY, help="(옵션) 추종 코드 파일 경로 — FLAG_DEFS 리터럴 자동 파싱 시도")
    p.add_argument("--flags_json", default=None, help="(옵션) flag 정의 JSON 파일 (리스트 형태)")
    p.add_argument("--spacing", type=float, default=None, help="(옵션) 전역 재샘플링 spacing [m]")
    p.add_argument("--out", default=OUT_PNG, help="출력 PNG 파일명")
    args = p.parse_args()
    main(args)
