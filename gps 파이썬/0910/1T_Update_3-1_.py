#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import time
from collections import deque

import rospy
import rospkg
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, String, Int32

# ── matplotlib 백엔드 & 스타일(흰 배경 강제) ─────────────────────
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
matplotlib.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.facecolor':'white',
    'axes.edgecolor':   'black',
    'text.color':       'black',
    'axes.labelcolor':  'black',
    'xtick.color':      'black',
    'ytick.color':      'black',
})

import geopy.distance
import pandas as pd
import numpy as np
from queue import Queue

# ──────────────────────────────────────────────────────────────────
# 기본 파라미터 (필요시 rosparam으로 덮어쓰기)
# ──────────────────────────────────────────────────────────────────
WAYPOINT_SPACING   = 2.5         # m
TARGET_RADIUS_END  = 2.0         # m  (반경 안이면 '그 WP에 있다'로 간주)
MAX_STEER_DEG      = 20.0        # deg
SIGN_CONVENTION    = -1.0

LOOKAHEAD_MIN      = 3.2         # m
LOOKAHEAD_MAX      = 4.0         # m
LOOKAHEAD_K        = 0.2         # m per (m/s)

LPF_FC_HZ          = 0.8         # Hz (조향 LPF)
SPEED_BUF_LEN      = 10
MAX_JITTER_SPEED   = 4.0         # m/s
MIN_MOVE_FOR_HEADING = 0.05      # m

FS_DEFAULT         = 20.0        # 퍼블리시/루프 Hz (⇒ 50ms)
GPS_TIMEOUT_SEC    = 1.0         # 최근 fix 없을 때 안전정지

# ── 속도 명령 상수(코드에서 바로 고정하고 싶을 때) ───────────────
#  - dSPACE 요구: 1..6 등 정수 코드. None이면 rosparam(~speed_code) 사용.
SPEED_FORCE_CODE = 3          # 예: 1 또는 2/3로 고정. rosparam 무시
# SPEED_FORCE_CODE = None     # ← 이렇게 두면 rosparam(~speed_code) 사용
SPEED_CAP_CODE_DEFAULT = 6    # 코드 상한(예: 6)

# 시각화 옵션
ANNOTATE_WAYPOINT_INDEX = True
DRAW_WAYPOINT_CIRCLES   = True

# 퍼블리시 토픽
TOPIC_SPEED_CMD    = '/vehicle/speed_cmd'     # Float32 (코드 0~6를 그대로 실수로 전송)
TOPIC_STEER_CMD    = '/vehicle/steer_cmd'     # Float32 (deg)
TOPIC_RTK_STATUS   = '/rtk/status'            # String ("FIX"/"FLOAT"/"NONE")
TOPIC_WP_INDEX     = '/tracker/wp_index'      # Int32 (반경 밖=0, 안=해당 인덱스 1-based)

# u-blox 옵셔널 의존성 (RTK 상태)
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
# 전역 상태
# ──────────────────────────────────────────────────────────────────
gps_queue = Queue()
latest_filtered_angle = 0.0
pos_buf   = deque(maxlen=SPEED_BUF_LEN*2)
speed_buf = deque(maxlen=SPEED_BUF_LEN)

pub_speed = None
pub_steer = None
pub_rtk   = None
pub_wpidx = None

rtk_status_txt = "NONE"
last_fix_time  = 0.0

# 퍼블리시/표시용 WP 인덱스: 반경 밖이면 -1, 안이면 0-based 인덱스
wp_index_active = -1

# 퍼블리시된 속도(코드)를 저장해서 시각화/로그에 쓰기
last_pub_speed_code = 0.0

# 로그
log_csv_path = None
_last_log_wall = 0.0

# 경로/좌표 관련 전역
to_xy = None                 # (lat,lon)→(x,y) 함수
csv_coords = []              # 원본 CSV 경로
spaced_x, spaced_y = [], []  # 간격 보정된 웨이포인트
nearest_idx_prev = 0
prev_Ld = LOOKAHEAD_MIN
last_heading_vec = None
last_tgt_idx = 0
last_tx = 0.0
last_ty = 0.0

# 플롯 전역
fig = None
ax_top = None
ax_info = None
_need_redraw = False

# ──────────────────────────────────────────────────────────────────
# 경로/파일 경로 (내 코드 방식)
# ──────────────────────────────────────────────────────────────────
def _default_paths():
    try:
        pkg_path = rospkg.RosPack().get_path('rtk_waypoint_tracker')
    except Exception:
        pkg_path = os.path.expanduser('~/catkin_ws/src/rtk_waypoint_tracker')

    waypoint_csv = os.path.join(pkg_path, 'config', 'raw_track_latlon_6.csv')
    logs_dir     = os.path.join(pkg_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_csv      = os.path.join(logs_dir, f"waypoint_log_{time.strftime('%Y%m
