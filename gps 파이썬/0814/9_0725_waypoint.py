#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_waypoints.py

원본 Lat/Lon CSV를 읽어 일정 간격(INTERVAL_M)으로 보간된 웨이포인트를 생성하여
Index,Lat,Lon 구조의 CSV로 저장합니다.
"""

import math
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────
INPUT_CSV  = r"C:\Users\dlwlr\OneDrive - 한라대학교\바탕 화면\GPS_purse_Python\bocsa\raw_track_latlon_1.csv"
OUTPUT_CSV = "waypoints_2m_1.csv"
INTERVAL_M = 2.0  # 보간 간격 [m]
# ──────────────────────────────────────────────────────────────

def latlon_to_enu(lat, lon, lat_ref, lon_ref):
    """
    위경도(lat, lon) → ENU(x, y) 변환
    lat_ref, lon_ref을 기준점으로 사용
    """
    R = 6_378_137.0
    deg2rad = math.pi / 180.0
    x = (lon - lon_ref) * math.cos(lat_ref * deg2rad) * deg2rad * R
    y = (lat - lat_ref) * deg2rad * R
    return x, y

def enu_to_latlon(x, y, lat_ref, lon_ref):
    """
    ENU(x, y) → 위경도(lat, lon) 변환
    lat_ref, lon_ref을 기준점으로 사용
    """
    R = 6_378_137.0
    deg2rad = math.pi / 180.0
    rad2deg = 180.0 / math.pi
    lat = lat_ref + (y / R) * rad2deg
    lon = lon_ref + (x / (R * math.cos(lat_ref * deg2rad))) * rad2deg
    return lat, lon

def generate_waypoints(df, interval):
    """
    DataFrame(df['Lat'], df['Lon'])로부터
    ENU 좌표를 계산하고, interval[m] 간격으로 보간된
    (x, y) 리스트와 기준점(lat0, lon0)을 반환합니다.
    """
    # 기준점: 첫 행의 위경도
    lat0 = df.at[0, 'Lat']
    lon0 = df.at[0, 'Lon']

    # ENU 변환
    xs = []
    ys = []
    for _, row in df.iterrows():
        x, y = latlon_to_enu(row['Lat'], row['Lon'], lat0, lon0)
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)

    # 누적 거리 계산
    dx = np.diff(xs, prepend=xs[0])
    dy = np.diff(ys, prepend=ys[0])
    dists = np.hypot(dx, dy).cumsum()  # numpy array
    max_d = dists[-1]

    # 보간 거리값
    sample_ds = np.arange(0, max_d, interval)

    # 보간된 ENU 좌표 리스트 생성
    wx = []
    wy = []
    for d in sample_ds:
        idx = np.searchsorted(dists, d)
        if idx == 0:
            wx.append(xs[0])
            wy.append(ys[0])
        else:
            d0, d1 = dists[idx-1], dists[idx]
            t = (d - d0) / (d1 - d0)
            x_interp = xs[idx-1] + t * (xs[idx] - xs[idx-1])
            y_interp = ys[idx-1] + t * (ys[idx] - ys[idx-1])
            wx.append(x_interp)
            wy.append(y_interp)

    return wx, wy, lat0, lon0

def main():
    # 1) 원본 CSV에서 Lat, Lon만 읽기
    df = pd.read_csv(INPUT_CSV, usecols=['Lat', 'Lon'])

    # 2) 보간된 ENU 좌표 및 기준점 생성
    wx, wy, lat0, lon0 = generate_waypoints(df, INTERVAL_M)

    # 3) ENU → 위경도 변환
    lats = []
    lons = []
    for x, y in zip(wx, wy):
        lat, lon = enu_to_latlon(x, y, lat0, lon0)
        lats.append(lat)
        lons.append(lon)

    # 4) DataFrame 구성 (Index: 1-base)
    wdf = pd.DataFrame({
        'Index': np.arange(1, len(lats) + 1),
        'Lat':   lats,
        'Lon':   lons
    })

    # 5) CSV로 저장
    wdf.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"[INFO] '{OUTPUT_CSV}' 저장 완료: 총 {len(wdf)}개 웨이포인트")

if __name__ == "__main__":
    main()
