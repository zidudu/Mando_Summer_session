# # rtk_live_plot_clean.py  – 포트·보율을 u-center와 동일하게
# PORT, BAUD = "COM7", 115200

# import serial, math, pynmea2, matplotlib
# matplotlib.use("TkAgg")        # VS Code에서 빈 창 방지:contentReference[oaicite:4]{index=4}
# import matplotlib.pyplot as plt

# ser = serial.Serial(PORT, BAUD, timeout=0.2); ser.flushInput()
# print("GGA 위·경도·XY 수신 중… Ctrl+C 로 중단\n")

# plt.ion(); fig, ax = plt.subplots()
# ax.set_title("Web Mercator 경로"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
# ax.grid(True); ln, = ax.plot([], [], '.', color='#1f77b4')

# xs, ys = [], [];  R, D2R = 6_378_137.0, math.pi/180  # WGS-84

# def ddmm2deg(v): d,m = divmod(v,100); return d + m/60     #:contentReference[oaicite:5]{index=5}

# try:
#     while True:
#         raw = ser.readline()
#         # ─ 바이너리(RTCM 0xD3, UBX 0xB562) 차단
#         if raw[:1] in (b'\xd3', b'\xb5'):   # 0xB5=UBX:contentReference[oaicite:6]{index=6}
#             continue
#         # ─ GGA 문장만 처리
#         if not (raw.startswith(b"$") and b"GGA" in raw):
#             continue
#         try:
#             msg = pynmea2.parse(raw.decode("ascii", errors="ignore"))  #:contentReference[oaicite:7]{index=7}
#         except pynmea2.ParseError:
#             continue
#         if not (msg.lat and msg.lon):      # No Fix
#             continue

#         lat = ddmm2deg(float(msg.lat)) * (1 if msg.lat_dir=="N" else -1)
#         lon = ddmm2deg(float(msg.lon)) * (1 if msg.lon_dir=="E" else -1)
#         x = R * lon * D2R
#         y = R * math.log(math.tan((lat*D2R + math.pi/2)/2))

#         print(f"{lat:.8f}, {lon:.8f}  →  X {x:.1f} m   Y {y:.1f} m")
#         xs.append(x); ys.append(y)
#         ln.set_data(xs, ys); ax.relim(); ax.autoscale_view()
#         plt.pause(0.05)
# except KeyboardInterrupt:
#     print("\n사용자 중단 – 시리얼 포트 닫고 종료합니다.")
# finally:
#     ser.close(); plt.ioff(); plt.show()
# rtk_live_plot_clean.py – Δ좌표 + 점만 표시
PORT, BAUD = "COM7", 115200

import serial, math, pynmea2, matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

ser = serial.Serial(PORT, BAUD, timeout=0.2);  ser.flushInput()
print("GGA ΔX·ΔY 수신 중… Ctrl+C 로 중단\n")

plt.ion(); fig, ax = plt.subplots()
ax.set_title("Web Mercator Δ경로")
ax.set_xlabel("ΔX (m)"); ax.set_ylabel("ΔY (m)")
ax.grid(True)
ln, = ax.plot([], [], '.', linestyle='None', color='#1f77b4')  # 점만

xs, ys = [], []
x0 = y0 = None                 # 최초 기준점
R, D2R = 6_378_137.0, math.pi/180

def ddmm2deg(v): d, m = divmod(v, 100); return d + m/60

try:
    while True:
        raw = ser.readline()
        if raw[:1] in (b'\xd3', b'\xb5'):    # RTCM/UBX 차단
            continue
        if not (raw.startswith(b"$") and b"GGA" in raw):
            continue
        try:
            msg = pynmea2.parse(raw.decode("ascii", errors="ignore"))
        except pynmea2.ParseError:
            continue
        if not (msg.lat and msg.lon):        # No-Fix
            continue

        lat = ddmm2deg(float(msg.lat)) * (1 if msg.lat_dir == "N" else -1)
        lon = ddmm2deg(float(msg.lon)) * (1 if msg.lon_dir == "E" else -1)
        x = R * lon * D2R
        y = R * math.log(math.tan((lat*D2R + math.pi/2)/2))

        if x0 is None:          # 첫 점을 원점으로
            x0, y0 = x, y
        dx, dy = x - x0, y - y0

        print(f"ΔX {dx:7.2f} m   ΔY {dy:7.2f} m   (lat {lat:.8f}, lon {lon:.8f})")
        xs.append(dx); ys.append(dy)
        ln.set_data(xs, ys)
        ax.relim(); ax.autoscale_view()
        plt.pause(0.05)
except KeyboardInterrupt:
    print("\n사용자 중단 – 시리얼 포트 닫고 종료합니다.")
finally:
    ser.close(); plt.ioff(); plt.show()
