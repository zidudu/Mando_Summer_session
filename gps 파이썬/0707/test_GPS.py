# # GPS 미터 변환 & 로깅

# import serial, time, math, csv, os
# import matplotlib.pyplot as plt

# # ───────────────── 설정 ───────────────────
# PORT, BAUD, TIMEOUT = "COM7", 115200, 0.1
# R = 6_378_137                         # WGS-84 반경(m)
# LOG_FILE = "waypoint_log.csv"
# # ─────────────────────────────────────────

# def nmea2deg(val_str: str) -> float:
#     """ddmm.mmmm / dddmm.mmmm → 십진수도(°)"""
#     try:
#         v = float(val_str)
#     except ValueError:
#         return float('nan')
#     deg = int(v // 100)
#     minutes = v - deg * 100
#     return deg + minutes / 60

# def reopen_serial():
#     try:
#         s = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
#         print(f" {PORT} 포트 열림")
#         return s
#     except Exception as e:
#         print(f" 포트 오류: {e}")
#         return None

# # ─── CSV 준비 ───────────────────────────
# need_header = not (os.path.isfile(LOG_FILE) and os.path.getsize(LOG_FILE) > 0)
# csv_f = open(LOG_FILE, "a", newline="", encoding="utf-8")
# writer = csv.writer(csv_f)
# if need_header:
#     writer.writerow(["X_m","Y_m"])

# # ─── 플롯 준비 ───────────────────────────
# plt.ion()                       # 인터랙티브 모드
# fig, ax = plt.subplots()
# ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
# ax.set_title("실시간 경로 (Web Mercator)")
# ax.grid(True)
# line, = ax.plot([], [], marker='.', linestyle='')  # 빈 시리즈
# xs, ys = [], []            # 누적 데이터

# ser = reopen_serial()
# print("GNGGA 수신 시작 (Ctrl+C 로 종료)")

# try:
#     while True:
#         if not ser or not ser.is_open:
#             ser = reopen_serial(); time.sleep(1); continue

#         raw = ser.readline().decode("ascii", errors="ignore").strip()
#         if not raw.startswith("$GNGGA"):
#             continue

#         p = raw.split(",")
#         if len(p) < 6:
#             continue

#         lat_r, ns = p[2], p[3]
#         lon_r, ew = p[4], p[5]

#         lat = nmea2deg(lat_r); lon = nmea2deg(lon_r)
#         if ns == "S": lat = -lat
#         if ew == "W": lon = -lon

#         x = R * math.radians(lon)
#         y = R * math.log(math.tan(math.radians(90 + lat) / 2))

#         # ─  업데이트 ─
#         xs.append(x); ys.append(y)
#         line.set_data(xs, ys)
#         ax.relim(); ax.autoscale_view()   # 축 범위 자동 조정
#         plt.pause(0.01)                   # GUI 이벤트 처리

#         # ─ 화면 출력 ─
#         print(f"{lat_r},{lon_r}  →  X={x:.2f} m,  Y={y:.2f} m")

#         # ─ CSV 기록 ─
#         now = time.strftime(" %H:%M:%S")
#         writer.writerow([f"{x:.2f}", f"{y:.2f}"])

#         csv_f.flush()

# except KeyboardInterrupt:
#     print("\n 수신 종료")
# finally:
#     if ser and ser.is_open: ser.close()
#     csv_f.close()
#     plt.ioff(); plt.show()

import serial, time, math, csv, os, sys          # 시리얼 통신·시간 지연·삼각함수·CSV 입출력·파일 검사 용
import matplotlib.pyplot as plt                  # 실시간 좌표 시각화 (interactive plot)
import numpy as np
from scipy.interpolate import splprep, splev     # 곡선 보간·재샘플링 에 사용 (SciPy 설치 필요)

# ───────────── 기본 설정 ──────────────
PORT, BAUD, TIMEOUT = "COM7", 115200, 0.1        # 연결할 GPS‑USB 포트, 보드레이트, 타임아웃
R_WGS84 = 6_378_137                              # 지구 반지름(m)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "raw_track3.csv")                       # 원시 경로
WPT_FILE = os.path.join(BASE_DIR, "waypoints3_curv.csv")                  # 최종 웨이포인트
# 간격 파라미터
D_STRAIGHT, D_MID, D_CURVE = 1, 0.40, 0.28 #0.55, 0.40, 0.28    # 웨이포인트 목표 간격 – 직선 0.55 m, 완만곡 0.40 m, 급곡 0.28 m
K_TH1, K_TH2 = 0.10, 0.20                        # 곡률 임계값(1/m) – κ < 0.10: 직선, 0.10 ≤ κ < 0.20: 완만곡, κ ≥ 0.20: 급곡


# ──────── 십진수 위·경도 변환 ───────────
def nmea2deg(val):                               
    try: v=float(val)
    except: return float('nan')
    d=int(v//100); m=v-d*100
    return d+m/60

def mercator_xy(lat_deg, lon_deg):               # 평면좌표 변환
    x = R_WGS84 * math.radians(lon_deg)
    y = R_WGS84 * math.log(math.tan(math.radians(90+lat_deg)/2))
    return x, y

def curvature(x, y):                             # 중앙차분 1·2차 미분으로 곡률계산
    x, y = np.asarray(x), np.asarray(y)
    dx  = np.gradient(x);  dy  = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    num = dx*ddy - dy*ddx
    den = (dx*dx + dy*dy)**1.5
    kappa = np.zeros_like(x); kappa[:] = np.nan
    mask = den > 1e-6
    kappa[mask] = np.abs(num[mask]/den[mask])
    return kappa

def resample_by_curvature(x, y):
    # ─ 0) 배열화 & 필터 ──────────────────────
    pts = np.vstack([x, y]).T
    pts = pts[~np.isnan(pts).any(axis=1)]   # NaN 제거
    pts = np.unique(pts, axis=0)            # 완전 중복 제거
    if len(pts) < 4:        # 포인트 너무 적으면 그대로 반환
        return pts
    x, y = pts[:,0], pts[:,1]

    # ─ 1) 스플라인 (점수에 맞춰 k 자동 설정) ─
    k_spline = min(3, len(x) - 1)           # 3 → 2 → 1 자동 감소
    tck, u = splprep([x, y], s=0, k=k_spline, per=False)

    # ─ 2) 고해상도 샘플 & 곡률 계산 ─
    u_dense = np.linspace(0, 1, max(2*len(x), 500))
    x_d, y_d = splev(u_dense, tck)
    kappa = curvature(x_d, y_d)

    # ─ 3) 구간별 목표 간격 배열 ──────────────
    d_target = np.where(kappa < K_TH1, D_STRAIGHT,
                 np.where(kappa < K_TH2, D_MID, D_CURVE))

    # ─ 4) 누적 길이 기반 리샘플 ─────────────
    dist = np.hstack(([0], np.cumsum(np.hypot(np.diff(x_d), np.diff(y_d)))))
    new_pts, acc = [0.0], 0.0
    for i in range(1, len(dist)):
        acc += dist[i] - dist[i-1]
        if acc >= d_target[i]:
            new_pts.append(dist[i]); acc = 0.0
    new_pts.append(dist[-1])
    x_out = np.interp(new_pts, dist, x_d)
    y_out = np.interp(new_pts, dist, y_d)
    return np.vstack([x_out, y_out]).T

def save_csv(fname, arr):                      # CSV 파일(헤더 X_m, Y_m) 저장
    with open(fname,"w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["X_m","Y_m"]); w.writerows(arr)

# ────────────── 실시간 수집 ──────────────
xs, ys = [], []
plt.ion(); fig, ax = plt.subplots()
line, = ax.plot([], [], '.-'); ax.set_xlabel("X"); ax.set_ylabel("Y")

def open_port():
    try:
        return serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    except Exception as e:
        print("포트 오류:",e); return None

ser = open_port(); print("GNGGA 수신 시작 (Ctrl+C 종료)")
try:
    while True:
        if not ser or not ser.is_open:
            time.sleep(1); ser=open_port(); continue
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw.startswith("$GNGGA"): continue
        p=raw.split(",");  lat_r, ns, lon_r, ew = p[2:6]
        lat=nmea2deg(lat_r); lon=nmea2deg(lon_r)
        if ns=="S": lat=-lat  
        if ew=="W": lon=-lon
        x,y = mercator_xy(lat, lon)
        xs.append(x); ys.append(y)
        line.set_data(xs, ys); ax.relim(); ax.autoscale_view()
        plt.pause(0.01)
except KeyboardInterrupt:
    print("\n수집 종료, 웨이포인트 생성 중…")
finally:
    if ser and ser.is_open: ser.close()
    plt.ioff(); plt.show()

# ────────────── 후처리 (웨이포인트) ─────────
raw = np.vstack([xs, ys]).T
save_csv(LOG_FILE, raw)  # 원본 백업
wp = resample_by_curvature(xs, ys)
save_csv(WPT_FILE, wp) 
print(f"웨이포인트 {len(wp)}개를 {WPT_FILE}에 저장했습니다.")
