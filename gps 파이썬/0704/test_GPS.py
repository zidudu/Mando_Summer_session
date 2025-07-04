import serial, time, math, csv, os
import matplotlib.pyplot as plt

# ───────────────── 설정 ───────────────────
PORT, BAUD, TIMEOUT = "COM7", 115200, 0.1
R = 6_378_137                         # WGS-84 반경(m)
LOG_FILE = "waypoint_log.csv"
# ─────────────────────────────────────────

def nmea2deg(val_str: str) -> float:
    """ddmm.mmmm / dddmm.mmmm → 십진수도(°)"""
    try:
        v = float(val_str)
    except ValueError:
        return float('nan')
    deg = int(v // 100)
    minutes = v - deg * 100
    return deg + minutes / 60

def reopen_serial():
    try:
        s = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        print(f" {PORT} 포트 열림")
        return s
    except Exception as e:
        print(f" 포트 오류: {e}")
        return None

# ─── CSV 준비 ───────────────────────────
need_header = not (os.path.isfile(LOG_FILE) and os.path.getsize(LOG_FILE) > 0)
csv_f = open(LOG_FILE, "a", newline="", encoding="utf-8")
writer = csv.writer(csv_f)
if need_header:
    writer.writerow(["X_m","Y_m"])

# ─── 플롯 준비 ───────────────────────────
plt.ion()                       # 인터랙티브 모드
fig, ax = plt.subplots()
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.set_title("실시간 경로 (Web Mercator)")
ax.grid(True)
line, = ax.plot([], [], marker='.', linestyle='')  # 빈 시리즈
xs, ys = [], []            # 누적 데이터

ser = reopen_serial()
print("GNGGA 수신 시작 (Ctrl+C 로 종료)")

try:
    while True:
        if not ser or not ser.is_open:
            ser = reopen_serial(); time.sleep(1); continue

        raw = ser.readline().decode("ascii", errors="ignore").strip()
        if not raw.startswith("$GNGGA"):
            continue

        p = raw.split(",")
        if len(p) < 6:
            continue

        lat_r, ns = p[2], p[3]
        lon_r, ew = p[4], p[5]

        lat = nmea2deg(lat_r); lon = nmea2deg(lon_r)
        if ns == "S": lat = -lat
        if ew == "W": lon = -lon

        x = R * math.radians(lon)
        y = R * math.log(math.tan(math.radians(90 + lat) / 2))

        # ─ 프로ット 업데이트 ─
        xs.append(x); ys.append(y)
        line.set_data(xs, ys)
        ax.relim(); ax.autoscale_view()   # 축 범위 자동 조정
        plt.pause(0.01)                   # GUI 이벤트 처리

        # ─ 화면 출력 ─
        print(f"{lat_r},{lon_r}  →  X={x:.2f} m,  Y={y:.2f} m")

        # ─ CSV 기록 ─
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([f"{x:.2f}", f"{y:.2f}"])

        csv_f.flush()

except KeyboardInterrupt:
    print("\n 수신 종료")
finally:
    if ser and ser.is_open: ser.close()
    csv_f.close()
    plt.ioff(); plt.show()
