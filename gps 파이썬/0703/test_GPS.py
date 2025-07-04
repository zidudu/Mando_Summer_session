# rtk_live_plot_v2.py  – COM·BAUD를 u-center와 동일하게
PORT, BAUD = "COM7", 115200

import serial, math, pynmea2, matplotlib
matplotlib.use("TkAgg")        # VS Code GUI 백엔드 강제
import matplotlib.pyplot as plt

# ── 1. 시리얼 포트 열기 ────────────────────────────────────────────
ser = serial.Serial(PORT, BAUD, timeout=0.2)
ser.flushInput()
print("GGA ΔX·ΔY 수신 중… Ctrl+C 로 중단\n")

# ── 2. 그래프 준비 ────────────────────────────────────────────────
plt.ion(); fig, ax = plt.subplots()
ax.set_title("Web Mercator Δ궤적")
ax.set_xlabel("ΔX (m)"); ax.set_ylabel("ΔY (m)")
ax.grid(True)

USE_SCATTER = False   # True → scatter, False → line+점
if USE_SCATTER:
    ln = ax.scatter([], [])        # 빈 scatter
else:
    ln, = ax.plot([], [], '.-', color='#1f77b4', linewidth=0.8,
                  markersize=4)    # 점+선 (.-)

# ±0.20 m로 축 고정
ax.set_xlim(-0.20, 0.20)
ax.set_ylim(-0.20, 0.20)

xs, ys = [], []
x0 = y0 = None                    # 기준점
R, D2R = 6_378_137.0, math.pi/180

def ddmm2deg(v: str) -> float:
    d, m = divmod(float(v), 100)
    return d + m/60

# ── 3. 실시간 루프 ────────────────────────────────────────────────
try:
    while True:
        raw = ser.readline()
        if raw[:1] in (b'\xd3', b'\xb5'):        # RTCM/UBX 건너뜀
            continue
        if not (raw.startswith(b"$") and b"GGA" in raw):
            continue

        try:
            msg = pynmea2.parse(raw.decode("ascii", errors="ignore"))
        except pynmea2.ParseError:
            continue
        if not (msg.lat and msg.lon):
            continue

        # 3-1. 위·경도 → Mercator
        lat = ddmm2deg(msg.lat) * (1 if msg.lat_dir == 'N' else -1)
        lon = ddmm2deg(msg.lon) * (1 if msg.lon_dir == 'E' else -1)
        x = R * lon * D2R
        y = R * math.log(math.tan((lat*D2R + math.pi/2)/2))

        # 기준점 보정
        if x0 is None:
            x0, y0 = x, y
        dx, dy = x - x0, y - y0

        # 3-2. 콘솔 출력 (UTC·고도 포함)
        print(f"[{msg.timestamp}]  ΔX {dx:7.2f} m   ΔY {dy:7.2f} m   "
              f"ALT {msg.altitude:6.1f} m")

        # 3-3. 그래프 갱신
        xs.append(dx); ys.append(dy)
        if USE_SCATTER:
            ln.set_offsets(list(zip(xs, ys)))
        else:
            ln.set_data(xs, ys)
        plt.pause(0.03)
except KeyboardInterrupt:
    print("\n사용자 중단 – 시리얼 포트 닫고 종료합니다.")
finally:
    ser.close(); plt.ioff(); plt.show()
