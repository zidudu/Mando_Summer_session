"""
RTK‑GPS 실시간 수집 + 이동평균·정지 탐지 + 번호 라벨 + CSV 저장
2025‑07‑08
"""
import os, serial, time, math
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# ── 경로 설정 ──────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 현재 스크립트 위치
LOG_FILE = os.path.join(BASE_DIR, "로우 데이터20_3.csv")     # 원시 좌표
FLT_FILE = os.path.join(BASE_DIR, "이동 필터 데이터20_3.csv")# 필터 좌표
#WPT_FILE = os.path.join(BASE_DIR, "필터 웨이포인트.csv")# (미사용·예비)

# ── 수집·필터 파라미터 ────────────────────
PORT, BAUD, TIMEOUT = "COM10", 115200, 0.1 # COM7 이 원래꺼
R_WGS84   = 6_378_137
# 이동 평균 값
WIN_SMA   = 20
WIN_STILL = 30
EPS_M     = 0.03

# ── 유틸리티 ──────────────────────────────
def save_csv(fname, arr):
    np.savetxt(fname, arr, delimiter=',',
               header='X_m,Y_m', comments='', fmt='%.6f')  # :contentReference[oaicite:3]{index=3}

def nmea2deg(v):
    try: v = float(v)
    except: return float('nan')
    d = int(v // 100); m = v - d * 100
    return d + m / 60

def mercator_xy(lat, lon):
    x = R_WGS84 * math.radians(lon)
    y = R_WGS84 * math.log(math.tan(math.radians(90 + lat) / 2))
    return x, y

# ── 실시간 플롯 준비 ───────────────────────
xs_raw, ys_raw, xs_f, ys_f = [], [], [], []
sma_win = deque(maxlen=WIN_SMA)
still_win = deque(maxlen=WIN_STILL)

plt.ion()
fig, ax = plt.subplots()
line_raw,  = ax.plot([], [], 'o', ms=2, alpha=0.3, label="Raw")
line_filt, = ax.plot([], [], '.-', lw=1.2, label="Filtered")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_title("RTK‑GPS 이동평균·정지 탐지 효과")
ax.legend()

def annotate_point(x, y, idx):
    ax.annotate(str(idx), xy=(x, y), xytext=(3, 3),
                textcoords="offset points", fontsize=7,
                color='gray')

def open_port():
    try: return serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    except Exception as e:
        print("포트 오류:", e); return None

# ── 데이터 수집 루프 ───────────────────────
ser = open_port()
print("GNGGA 수신 시작 – Ctrl+C 종료")
try:
    while True:
  

        if not ser or not ser.is_open:
            time.sleep(1); ser = open_port(); continue
        
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw.startswith("$GNGGA"): continue
        
        lat_r, ns, lon_r, ew = raw.split(",")[2:6]
        lat, lon = nmea2deg(lat_r), nmea2deg(lon_r)
        if ns == "S": lat = -lat
        if ew == "W": lon = -lon
        x, y = mercator_xy(lat, lon)

        # 이동평균
        sma_win.append((x, y))
        x_sma, y_sma = np.mean(sma_win, axis=0)

        # 정지 탐지
        still_win.append((x_sma, y_sma))
        if len(still_win) > 1:
            sw = np.array(still_win)
            dev = np.max(np.linalg.norm(sw - sw.mean(0), axis=1))
        else:
            dev = EPS_M + 1
        x_use, y_use = (sw.mean(0) if dev < EPS_M else (x_sma, y_sma))

        # 버퍼·플롯 갱신
        idx = len(xs_raw) + 1
        xs_raw.append(x);  ys_raw.append(y)
        xs_f.append(x_use); ys_f.append(y_use)

        line_raw.set_data(xs_raw, ys_raw)
        line_filt.set_data(xs_f, ys_f)
        annotate_point(x, y, idx)          # ← Raw 포인트 번호 매기기
        ax.relim(); ax.autoscale_view()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("\n수집 종료 – CSV 저장 중…")
finally:
    if ser and ser.is_open: ser.close()
    plt.ioff(); plt.show()

    save_csv(LOG_FILE, np.column_stack([xs_raw, ys_raw]))
    save_csv(FLT_FILE, np.column_stack([xs_f,  ys_f]))
    print(f"CSV 저장 완료:\n  Raw → {LOG_FILE}\n  Filt → {FLT_FILE}")
