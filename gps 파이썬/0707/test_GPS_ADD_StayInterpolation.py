
import serial, time, math, csv, os, sys          # 시리얼 통신·시간 지연·삼각함수·CSV 입출력·파일 검사 용
import matplotlib.pyplot as plt                  # 실시간 좌표 시각화 (interactive plot)
import numpy as np
from scipy.interpolate import splprep, splev     # 곡선 보간·재샘플링 에 사용 (SciPy 설치 필요)

# ───────────── 기본 설정 ──────────────
PORT, BAUD, TIMEOUT = "COM7", 115200, 0.1        # 연결할 GPS‑USB 포트, 보드레이트, 타임아웃
R_WGS84 = 6_378_137                              # 지구 반지름(m)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 원시 로그(raw_track3.csv)와 
# 최종 웨이포인트(waypoints3_curv.csv) 파일 경로를 지정
LOG_FILE = os.path.join(BASE_DIR, "raw_track3.csv")                       # 원시 경로
WPT_FILE = os.path.join(BASE_DIR, "waypoints3_curv.csv")                  # 최종 웨이포인트
# 간격 파라미터
D_STRAIGHT, D_MID, D_CURVE = 1, 0.40, 0.28 #0.55, 0.40, 0.28    # 웨이포인트 목표 간격 – 직선 0.55 m, 완만곡 0.40 m, 급곡 0.28 m
K_TH1, K_TH2 = 0.10, 0.20                        # 곡률 임계값(1/m) – κ < 0.10: 직선, 0.10 ≤ κ < 0.20: 완만곡, κ ≥ 0.20: 급곡


# ──────── 십진수 위·경도 변환 ───────────
#NMEA 형식 (d)ddmm.mmmm에서 deg = dd, min = mm.mmmm을 분/60으로 
# 바꾼 뒤 합산하는 전형적인 변환 방식
def nmea2deg(val):                               
    try: v=float(val)
    except: return float('nan') # 오류날때 0
    d=int(v//100); m=v-d*100 # v에 100 나눔. 그리고 m은 v - d * 100. v = 37.3, d = 3, m = 37.3 - 300 = -262.7 
    return d+m/60 # 3 + 4 = 7
# EPSG:3857 Web Mercator 공식대로 위·경도를 미터 단위 XY로 투영
def mercator_xy(lat_deg, lon_deg):               # 평면좌표 변환
    x = R_WGS84 * math.radians(lon_deg)
    y = R_WGS84 * math.log(math.tan(math.radians(90+lat_deg)/2))
    return x, y
# np.gradient로 1차·2차 도함수를 구해 
# 곡률 κ = |(x′y″ − y′x″)/(x′²+y′²)^(3/2)| 을 계산
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
# 곡률 기반 리샘플링 (후처리)
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

# 앞 좌표와 거리 차이 계산
def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


# ────────────── 실시간 수집 ──────────────
xs, ys = [], []
plt.ion(); fig, ax = plt.subplots()
line, = ax.plot([], [], '.-'); ax.set_xlabel("X"); ax.set_ylabel("Y")
 # ★ 이전 좌표 저장용 변수
prev_x = prev_y = None  

def open_port():
    try:
        return serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    except Exception as e:
        print("포트 오류:",e); return None
# 포트 열기
ser = open_port(); print("GNGGA 수신 시작 (Ctrl+C 종료)")
try:
    while True:
        if not ser or not ser.is_open:
            time.sleep(1); ser=open_port(); continue
        # (1) NMEA 읽기
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw.startswith("$GNGGA"): continue
        p=raw.split(",");  
        
         # ── ★ ① HDOP 필터 ─────────────────
        if len(p) < 9:        # 필드 개수 점검. HDOP 필드가 없는 불완전·손상 NMEA 문장은 무시
            continue
        try:
            hdop = float(p[8])
        except ValueError:
            continue
        if hdop > 1.5:        # HDOP가 1.5보다 크면 무시
            continue
        # ──────────────────────────────────
        # (2) 위·경도 → 평면 좌표
        lat_r, ns, lon_r, ew = p[2:6]
        lat=nmea2deg(lat_r); lon=nmea2deg(lon_r)
        if ns=="S": lat=-lat  
        if ew=="W": lon=-lon
        x,y = mercator_xy(lat, lon)

        # ── ★ ② 최소 이동 거리 필터 ──────
        if prev_x is not None:
            if math.hypot(x - prev_x, y - prev_y) < 0.05:  # 5 cm 미만 변화는 건너뜀
                continue
        # ──────────────────────────────────
        prev_x, prev_y = x, y   # 다음 루프를 위해 저장

        # (3) 좌표 누적·플롯 업데이트
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
