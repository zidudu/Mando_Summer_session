import serial
import time
import csv
import math
import pandas as pd
import numpy as np
import os
import threading
import struct  # 1-byte binary transmission
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Settings ───────────────────────────────
PORT = 'COM11'
BAUD = 460800
ARDUINO_PORT = 'COM22' 
ARDUINO_BAUD = 9600
TARGET_RADIUS = 1.5
MIN_WAYPOINT_DISTANCE = 0.9
GPS_LOG_CSV = 'heading_vectors.csv'
WAYPOINT_CSV = 'waypoints.csv'

# Low pass filter settings
f_c = 2.0
f_s = 10.0
alpha = (2 * math.pi * f_c) / (2 * math.pi * f_c + f_s)
filtered_steering_angle = 0.0

# Outlier filter settings
GPS_OUTLIER_THRESHOLD = 1.0  # meter 단위 큰 변화 거름

# ── Arduino serial initialization ────────────
try:
    arduino_ser = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.1)
    time.sleep(2)
    print(f"Arduino connected: {ARDUINO_PORT}@{ARDUINO_BAUD}")
except Exception as e:
    print(f"Arduino serial connection failed: {e}")
    arduino_ser = None

# ── Function definitions ─────────────────────
def dm_to_dec(dm, direction):
    try:
        d = int(float(dm) / 100)
        m = float(dm) - d * 100
        dec = d + m / 60
        return -dec if direction in ['S', 'W'] else dec
    except:
        return None

def latlon_to_meters(lat, lon):
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan((90 + lat) * math.pi / 360.0))
    return x, y

def distance_m(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def calculate_steering_angle(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    THRESHOLD = 0.05
    if norm1 < THRESHOLD or norm2 < THRESHOLD:
        return 0.0
    dot = np.dot(v1, v2)
    cos_theta = max(min(dot / (norm1 * norm2), 1.0), -1.0)
    angle = math.degrees(math.acos(cos_theta))
    # FIX: cross product typo
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross < 0:
        angle = -angle
    angle = max(min(angle / 1.3, 25), -25)
    if abs(angle) > 20 and (norm1 < THRESHOLD or norm2 < THRESHOLD):
        return 0.0
    return angle

def apply_low_pass_filter(current):
    global filtered_steering_angle
    filtered_steering_angle = (1 - alpha) * filtered_steering_angle + alpha * current
    return filtered_steering_angle * -1

# 혼합 GPS 필터링 (아웃라이어 제거 + 저역통과)
prev_x, prev_y = None, None
prev_filtered_x, prev_filtered_y = None, None
def filter_gps_signal(x, y):
    global prev_x, prev_y, prev_filtered_x, prev_filtered_y
    if prev_x is not None and prev_y is not None:
        if distance_m(prev_x, prev_y, x, y) > GPS_OUTLIER_THRESHOLD:
            x, y = prev_x, prev_y
        else:
            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = x, y
    if prev_filtered_x is None or prev_filtered_y is None:
        prev_filtered_x, prev_filtered_y = x, y
    filtered_x = (1 - alpha) * prev_filtered_x + alpha * x
    filtered_y = (1 - alpha) * prev_filtered_y + alpha * y
    prev_filtered_x, prev_filtered_y = filtered_x, filtered_y
    return filtered_x, filtered_y

# ── Waypoints processing ─────────────────────
df = pd.read_csv(WAYPOINT_CSV)
coords = [latlon_to_meters(row['Lat'], row['Lon']) for _, row in df.iterrows()]  # [(x, y)]

# 반드시 float만 저장 (에러 발생 부분 FIX)
filtered_x = [coords[0][0]]
filtered_y = [coords[0][1]]
for xi, yi in coords[1:]:
    print('filtered_x[-1]:', filtered_x[-1], type(filtered_x[-1]))
    print('filtered_y[-1]:', filtered_y[-1], type(filtered_y[-1]))
    print('xi:', xi, type(xi))
    print('yi:', yi, type(yi))
    if distance_m(filtered_x[-1], filtered_y[-1], xi, yi) >= MIN_WAYPOINT_DISTANCE:
        filtered_x.append(xi)
        filtered_y.append(yi)

waypoints_x = np.array(filtered_x)
waypoints_y = np.array(filtered_y)

# ── Initialization ──────────────────────────
current_x, current_y = [], []
heading_vectors = []
waypoint_index = 0

if not os.path.exists(GPS_LOG_CSV):
    with open(GPS_LOG_CSV, 'w', newline='') as f:
        csv.writer(f).writerow([
            'current_x', 'current_y', 'prev_x', 'prev_y',
            'target_vector_x', 'target_vector_y', 'waypoint_x', 'waypoint_y', 'angle'
        ])

# ── Visualization setup ─────────────────────
fig = plt.figure()

def update_plot(_):
    global waypoint_index
    ax = plt.gca()
    ax.clear()

    window_size = 20
    start_index = (waypoint_index // window_size) * window_size
    end_index = min(start_index + window_size, len(waypoints_x))

    ax.scatter(waypoints_x[start_index:end_index], waypoints_y[start_index:end_index], color='blue', s=10, label='Waypoints')
    for i in range(start_index, end_index):
        circle = plt.Circle((waypoints_x[i], waypoints_y[i]), TARGET_RADIUS,
                            fill=False, linestyle='--', color='blue')
        ax.add_patch(circle)
        ax.text(waypoints_x[i], waypoints_y[i], str(i + 1), fontsize=8, ha='center')

    if current_x and current_y:
        ax.scatter(current_x[-1], current_y[-1], color='red', s=50, label='Current')
        tx, ty = waypoints_x[waypoint_index], waypoints_y[waypoint_index]
        ax.arrow(current_x[-1], current_y[-1], tx - current_x[-1], ty - current_y[-1],
                 head_width=0.5, head_length=0.5, color='green')

        for i in range(1, len(current_x)):
            dx = current_x[i] - current_x[i - 1]
            dy = current_y[i] - current_y[i - 1]
            ax.arrow(current_x[i - 1], current_y[i - 1], dx, dy,
                     head_width=0.2, head_length=0.2, color='orange')

        if len(current_x) > 1:
            dx = current_x[-1] - current_x[-2]
            dy = current_y[-1] - current_y[-2]
            heading_vectors.append((dx, dy))
            target_vec = (tx - current_x[-1], ty - current_y[-1])
            angle = calculate_steering_angle((dx, dy), target_vec)
            smooth = apply_low_pass_filter(angle)
        else:
            angle = 0.0
            smooth = 0.0
            target_vec = ('', '')

        print(f"Raw: {angle:.2f}°, Filtered: {smooth:.2f}°")

        with open(GPS_LOG_CSV, 'a', newline='') as f:
            if len(current_x) > 1:
                csv.writer(f).writerow([
                    current_x[-1], current_y[-1], current_x[-2], current_y[-2],
                    target_vec[0], target_vec[1], tx, ty, smooth
                ])
            else:
                csv.writer(f).writerow([
                    current_x[-1], current_y[-1], '', '',
                    '', '', tx, ty, smooth
                ])

        if arduino_ser:
            try:
                send_angle = max(min(smooth, 20), -20)
                angle_byte = int(round(send_angle))
                arduino_ser.write(b'G')
                arduino_ser.write(struct.pack('b', angle_byte))
                print(f"Sent: G {angle_byte}")
            except Exception as e:
                print(f"Arduino transmission error: {e}")

        if len(current_x) > 1 and distance_m(current_x[-1], current_y[-1], tx, ty) < TARGET_RADIUS and waypoint_index < len(waypoints_x) - 1:
            waypoint_index += 1

    if len(current_x) > 1:
        ax.set_title(f"Windows GPS Logger  Steering Angle: {smooth:.2f}°")
    else:
        ax.set_title('Windows GPS Logger')

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

# ── GPS reception thread ─────────────────────
def read_gps():
    global waypoint_index
    with serial.Serial(PORT, BAUD, timeout=0.1) as ser:
        while True:
            try:
                line = ser.readline().decode('ascii', errors='ignore').strip()
                if line.startswith('$GNGGA') or line.startswith('$GPGGA'):
                    parts = line.split(',')
                    if len(parts) > 5 and parts[2] and parts[3] and parts[4] and parts[5]:
                        lat = dm_to_dec(parts[2], parts[3])
                        lon = dm_to_dec(parts[4], parts[5])
                        if lat is not None and lon is not None:
                            x, y = latlon_to_meters(lat, lon)
                            filtered_x_val, filtered_y_val = filter_gps_signal(x, y)
                            if len(current_x) == 0 or distance_m(current_x[-1], current_y[-1], filtered_x_val, filtered_y_val) >= 0.001:
                                current_x.append(filtered_x_val)
                                current_y.append(filtered_y_val)
            except Exception as e:
                print(f"Error: {e}")
            time.sleep(0.001)

# ── Run ─────────────────────
threading.Thread(target=read_gps, daemon=True).start()
ani = animation.FuncAnimation(fig, update_plot, interval=100)
plt.show()
