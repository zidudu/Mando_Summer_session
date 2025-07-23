#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
left_lane.csv → ENU 변환 + 2m 간격 웨이포인트 생성/시각화
  - 각 웨이포인트에 반경 0.5m 원 표시
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# WGS84 타원체 상수
_a  = 6378137.0
_f  = 1 / 298.257223563
_e2 = _f * (2 - _f)

def lla_to_ecef(lat, lon):
    """위경도(lat, lon) → ECEF(X, Y, Z) 변환 (고도=0)"""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    N = _a / np.sqrt(1 - _e2 * np.sin(lat_rad)**2)
    X = N * np.cos(lat_rad) * np.cos(lon_rad)
    Y = N * np.cos(lat_rad) * np.sin(lon_rad)
    Z = (N * (1 - _e2)) * np.sin(lat_rad)
    return X, Y, Z

def ecef_to_enu(X, Y, Z, lat0, lon0):
    """ECEF → ENU 변환 (E, N만 계산)"""
    x0, y0, z0 = lla_to_ecef(np.array([lat0]), np.array([lon0]))
    dx, dy, dz = X - x0, Y - y0, Z - z0
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)
    R = np.array([
        [-np.sin(lon0_rad),               np.cos(lon0_rad),              0],
        [-np.sin(lat0_rad)*np.cos(lon0_rad), -np.sin(lat0_rad)*np.sin(lon0_rad), np.cos(lat0_rad)]
    ])
    enu = R @ np.vstack((dx, dy, dz))
    return enu[0], enu[1]

def load_latlon(csv_path):
    """Pandas로 CSV 읽어서 Lat, Lon 반환"""
    df = pd.read_csv(csv_path)
    return df['Lat'].values, df['Lon'].values

def generate_waypoints(E, N, spacing=2.0):
    """ENU 궤적을 따라 일정 간격으로 웨이포인트 생성"""
    seg_d = np.hypot(np.diff(E), np.diff(N))
    s = np.hstack(([0], np.cumsum(seg_d)))
    s_des = np.arange(0, s[-1], spacing)
    E_wp = np.interp(s_des, s, E)
    N_wp = np.interp(s_des, s, N)
    return E_wp, N_wp

def plot_with_circles(E, N, E_wp, N_wp, radius=0.5):
    """ENU 궤적, 웨이포인트, 순번, 반경 원 모두 시각화"""
    fig, ax = plt.subplots(figsize=(8,6))
    # 1) 궤적
    ax.plot(E, N, '.-', color='gray', alpha=0.6, label='Trajectory', zorder=1)
    # 2) 웨이포인트
    ax.scatter(E_wp, N_wp, c='orange', s=25, label='Waypoints (2 m)', zorder=2)
    # 3) 웨이포인트 순번 + 반경 원
    for idx, (x, y) in enumerate(zip(E_wp, N_wp)):
        # 순번
        ax.text(x, y, str(idx),
                fontsize=6, ha='right', va='bottom',
                zorder=4)
        # 반경 원
        circ = Circle((x, y), radius=radius,
                      edgecolor='blue', facecolor='none',
                      lw=1, zorder=3)
        ax.add_patch(circ)
    # 4) 시작/끝점 강조
    ax.scatter(E[0],  N[0],  c='green', s=60, label='Start', zorder=5)
    ax.scatter(E[-1], N[-1], c='red',   s=60, label='End',   zorder=5)

    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_title(f'ENU 궤적 + 2m 웨이포인트 (반경 {radius}m)')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    plt.show()

def main():
    csv_file = 'left_lane.csv'
    lat, lon = load_latlon(csv_file)
    lat0, lon0 = lat[0], lon[0]

    X, Y, Z = lla_to_ecef(lat, lon)
    E, N    = ecef_to_enu(X, Y, Z, lat0, lon0)

    E_wp, N_wp = generate_waypoints(E, N, spacing=2.0)
    plot_with_circles(E, N, E_wp, N_wp, radius=0.5)

if __name__ == '__main__':
    main()
