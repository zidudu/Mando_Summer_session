#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTK-GPS 실시간 로우 데이터 수집 및 시각화 + 곡률 기반 동적 웨이포인트 생성
  - NMEA GNGGA 메시지에서 위도·경도 파싱
  - Web Mercator (X, Y) 계산
  - 실시간 플롯 (Raw 궤적)
  - 종료 시점에 곡률에 따라 D_STRAIGHT, D_MID, D_CURVE 간격으로 웨이포인트 생성
  - finally에서 CSV 저장 (raw 및 waypoints, 중복 시 _1,_2 자동 인덱싱)
"""
import os, serial, time, math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ── 파일명 중복 방지 ─────────────────────────────
def unique_filepath(dirpath: str, basename: str, ext: str = ".csv") -> str:
    path = os.path.join(dirpath, basename + ext)
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        path = os.path.join(dirpath, f"{basename}_{idx}{ext}")
        if not os.path.exists(path):
            return path
        idx += 1

# ── CSV 저장 ──────────────────────────────────────
def save_csv(fname: str, arr: np.ndarray, header: list):
    np.savetxt(fname, arr, delimiter=",",
               header=",".join(header), comments="", fmt="%.6f")

# ── NMEA → 십진도 변환 ─────────────────────────────
def nmea2deg(val_str: str) -> float:
    try:
        v = float(val_str)
        d = int(v // 100)
        m = v - 100*d
        return d + m/60.0
    except:
        return float('nan')

# ── 위/경도 → Web Mercator 변환 ─────────────────────
def mercator_xy(lat: float, lon: float):
    R = 6_378_137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.radians(90+lat)/2))
    return x, y

# ── 곡률 계산 ───────────────────────────────────────
def curvature(xs, ys):
    x = np.asarray(xs); y = np.asarray(ys)
    dx = np.gradient(x); dy = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    num = dx*ddy - dy*ddx
    den = (dx*dx + dy*dy)**1.5
    k = np.zeros_like(x)
    mask = den>1e-6
    k[mask] = np.abs(num[mask]/den[mask])
    return k

# ── 동적 웨이포인트 생성 ────────────────────────────
def dynamic_waypoints(xs, ys, K_TH1, K_TH2, D_STRAIGHT, D_MID, D_CURVE):
    kappa = curvature(xs, ys)
    wpts = [(xs[0], ys[0])]
    acc = 0.0
    for i in range(1, len(xs)):
        seg = math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
        acc += seg
        # 곡률에 따라 간격 결정
        κ = kappa[i]
        if κ < K_TH1:
            d_target = D_STRAIGHT
        elif κ < K_TH2:
            d_target = D_MID
        else:
            d_target = D_CURVE
        # 목표 간격마다 웨이포인트 추가
        while acc >= d_target:
            overshoot = acc - d_target
            ratio = (seg - overshoot)/seg
            x_wp = xs[i-1] + (xs[i]-xs[i-1])*ratio
            y_wp = ys[i-1] + (ys[i]-ys[i-1])*ratio
            wpts.append((x_wp, y_wp))
            acc -= d_target
    # 마지막 점 추가
    wpts.append((xs[-1], ys[-1]))
    return wpts

# ── 설정 ───────────────────────────────────────────
PORT, BAUD, TIMEOUT = "COM10", 115200, 0.1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_RAW = unique_filepath(BASE_DIR, "raw_track_xy")
LOG_WPT = unique_filepath(BASE_DIR, "waypoints_dynamic")

# 곡률 임계 및 간격 (m)
K_TH1, K_TH2        = 0.10, 0.20
D_STRAIGHT, D_MID, D_CURVE = 0.55, 0.40, 0.28

# ── 실시간 플롯 초기화 ─────────────────────────────
plt.ion()
fig, ax = plt.subplots()
ax.set_title("RTK-GPS 실시간 궤적")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.grid(True)
line_raw, = ax.plot([], [], '.-', alpha=0.6, label="Raw")
ax.legend()
xs, ys = [], []

# ── 시리얼 연결 ─────────────────────────────────────
def open_serial():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        print(f"Serial open: {PORT}")
        return ser
    except Exception as e:
        print("Serial open failed:", e)
        return None

ser = open_serial()
print("GNGGA 수신 대기 (Ctrl+C 종료)")

# ── 메인 루프 ───────────────────────────────────────
try:
    while True:
        if ser is None or not ser.is_open:
            time.sleep(1)
            ser = open_serial()
            continue
        line = ser.readline().decode(errors="ignore").strip()
        if not line.startswith("$GNGGA"): continue
        parts = line.split(",")
        if len(parts) < 6: continue

        # 위도·경도 파싱
        lat = nmea2deg(parts[2]); lon = nmea2deg(parts[4])
        if parts[3]=="S": lat=-lat
        if parts[5]=="W": lon=-lon
        x,y = mercator_xy(lat, lon)

        # 저장 및 플롯 업데이트
        xs.append(x); ys.append(y)
        line_raw.set_data(xs, ys)
        ax.relim(); ax.autoscale_view()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("\n수집 종료, 웨이포인트 생성 중…")

finally:
    if ser and ser.is_open: ser.close()
    # raw 저장
    raw_arr = np.column_stack([xs, ys])
    save_csv(LOG_RAW, raw_arr, ["X","Y"])
    # 동적 웨이포인트 생성 및 저장
    wpts = dynamic_waypoints(xs, ys, K_TH1, K_TH2, D_STRAIGHT, D_MID, D_CURVE)
    w_arr = np.array(wpts)
    save_csv(LOG_WPT, w_arr, ["X_m","Y_m"])
    print("저장 완료:", LOG_RAW, LOG_WPT)

    plt.ioff()
    plt.show()







# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 직선 및 곡선 기준 경로 추종 오차 분석 및 시각화
#   - CSV에서 X, Y 데이터 로드
#   - (A) np.polyfit 선형 회귀: 회귀선 수직 편차 계산
#   - (B) SciPy splprep/splev 스플라인: 곡선 수직 편차 계산
#   - 통계(카디널리티, 평균·표준편차·RMSE·범위) 출력
#   - (1) 원본 데이터 & 회귀선/스플라인 플롯
#   - (2) 편차 vs 샘플 인덱스 플롯
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import splprep, splev
# from scipy.optimize import fminbound


# def load_data(fn):
#     """CSV 파일에서 X, Y 데이터 로드 (헤더 1줄 건너뜀)"""
#     data = np.loadtxt(fn, delimiter=",", skiprows=1)
#     return data[:,0], data[:,1]

# # ── (A) 선형 회귀 기반 편차 계산 ──────────────────────────────
# def linear_dev(x, y):
#     """1차 회귀 계수를 사용해 각 점의 수직 편차를 계산"""
#     m, b = np.polyfit(x, y, 1)      # 회귀선 계수 m, b 계산
#     den = np.hypot(m, 1)            # 거리 분모 계산
#     dist = np.abs(m*x - y + b) / den  # 수직 거리 계산
#     return dist, m, b

# # (B) 스플라인 근사 시 smoothing 값을 조정
# def fit_spline(x, y, smoothing=1.0, degree=3):
#     """
#     B-스플라인 근사를 통해 곡선 모델 생성
#       - smoothing>0: 데이터 포인트마다 정확히 통과하지 않음
#       - degree=3: 3차 B-spline
#     """
#     tck, u = splprep([x, y], s=smoothing, k=degree)
#     return tck


# def dist_to_curve(u0, xi, yi, tck):
#     """파라미터 u0 위치에서 스플라인 곡선까지의 유클리드 거리 계산"""
#     xs, ys = splev(u0, tck)
#     return np.hypot(xs - xi, ys - yi)


# def spline_dev(x, y, tck):
#     """모든 점에 대해 스플라인 곡선까지의 최소 수직 거리를 탐색"""
#     dists = []
#     for xi, yi in zip(x, y):
#         u_opt = fminbound(lambda u: dist_to_curve(u, xi, yi, tck), 0.0, 1.0, disp=0)
#         dists.append(dist_to_curve(u_opt, xi, yi, tck))
#     return np.array(dists)

# # ── 통계 출력 함수 ─────────────────────────────────────
# def print_stats(name, dist):
#     """카디널리티, 평균, 표준편차, RMSE, 범위를 출력 (한글/영어 혼용)"""
#     n = len(dist)
#     mean = dist.mean()               # 평균 편차
#     std  = dist.std()                # 표준편차
#     rmse = np.sqrt(np.mean(dist**2)) # RMSE 계산
#     rng  = dist.max() - dist.min()   # 편차 범위
#     print(f"[{name}] Cardinality | 카디널리티: {n}")
#     print(f"[{name}] Mean 편차: {mean:.3f} m, Avg deviation: {mean:.3f} m")
#     print(f"[{name}] Std 표준편차: {std:.3f} m, Std deviation: {std:.3f} m")
#     print(f"[{name}] RMSE: {rmse:.3f} m")
#     print(f"[{name}] Range 편차 범위: {rng:.3f} m")

# # ── 시각화 함수 ─────────────────────────────────────────────
# def plot_path(x, y, m, b, tck):
#     """원본 데이터, 회귀선, 스플라인 곡선을 한 그래프에 표시"""
#     plt.figure(figsize=(6,5))
#     plt.scatter(x, y, s=10, alpha=0.6, label="Data Points")  # 원본 점 표시
#     # 회귀선 그리기
#     xl = np.linspace(x.min(), x.max(), 100)
#     plt.plot(xl, m*xl + b, 'g--', lw=1.5, label="Regression Line")  # 회귀선 표시
#     # 스플라인 곡선 그리기
#     u_fine = np.linspace(0, 1, 200)
#     xs, ys = splev(u_fine, tck)
#     plt.plot(xs, ys, 'r-', lw=1.5, label="Spline Curve")  # 스플라인 표시
#     plt.xlabel("X [m]")    # X축 레이블
#     plt.ylabel("Y [m]")    # Y축 레이블
#     plt.title("Data & Regression/Spline")  # 제목
#     plt.legend()             # 범례
#     plt.grid(True)           # 그리드 표시


# def plot_devs(line_dev_vals, spline_dev_vals):
#     """샘플 인덱스별 선형 오차와 스플라인 오차 비교 플롯"""
#     idx = np.arange(len(line_dev_vals))
#     plt.figure(figsize=(6,4))
#     plt.plot(idx, line_dev_vals, 'g-o', label="Linear Deviation")   # 선형 편차 표시
#     plt.plot(idx, spline_dev_vals, 'r--s', label="Spline Deviation")  # 스플라인 편차 표시
#     plt.xlabel("Sample Index")            # X축 레이블
#     plt.ylabel("Deviation [m]")           # Y축 레이블
#     plt.title("Linear vs. Spline Deviation Comparison")  # 제목
#     plt.legend()                        # 범례
#     plt.grid(True)                      # 그리드 표시
#     plt.tight_layout()                  # 레이아웃 조정
#     plt.show()                          # 화면 출력


# def main():
#     fn = "waypoints_dynamic_3.csv"  # 실제 CSV 파일명으로 수정해주세요
#     x, y = load_data(fn)              # 데이터 로드

#     # 선형 편차 계산
#     lin_dev, m, b = linear_dev(x, y)
#     print_stats("Linear Regression", lin_dev)  # 통계 출력

#     # 스플라인 편차 계산
#     tck = fit_spline(x, y)              # 스플라인 적합
#     spl_dev = spline_dev(x, y, tck)     # 편차 계산
#     print_stats("Spline", spl_dev)    # 통계 출력

#     # 결과 시각화
#     plot_path(x, y, m, b, tck)          # 경로 플롯
#     plot_devs(lin_dev, spl_dev)         # 편차 비교 플롯

# if __name__ == "__main__":
#     main()
