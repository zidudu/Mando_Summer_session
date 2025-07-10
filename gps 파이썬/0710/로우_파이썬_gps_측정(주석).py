#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTK-GPS 실시간 로우 데이터 수집 및 시각화
  - NMEA GNGGA 메시지로부터 위도·경도 파싱
  - Web Mercator (X, Y) 계산
  - 실시간 플롯만 수행
  - finally에서 한 번에 CSV 저장 (파일명 중복 시 _1, _2 … 자동 인덱싱)
  - 1초에 한번 GNGGA 문장 보냄(u-blox F9P EVK 같은 모듈을 특별히 설정하지 않으면 기본 1 Hz, 즉 1초에 한 번 GNGGA 문장을 내보냅니다.)
    (만약 u-center에서 GNGGA 출력 주기를 5 Hz나 10 Hz로 바꾸면, 그에 맞춰 0.2 초(5 Hz)나 0.1 초(10 Hz)마다 좌표를 받게 됩니다)
  - 결론적으로, 특별히 장비 설정을 바꾸지 않으셨다면 초당 1회(1 Hz) 로 새로운 X,Y 값을 받습니다.
"""
import os
import serial
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# ── 중복 방지용 파일명 생성 함수 ─────────────────────────
def unique_filepath(dirpath: str, basename: str, ext: str = ".csv") -> str:
    """dirpath에 basename+ext가 있으면 basename_1.ext, basename_2.ext … 로 반환"""
    candidate = os.path.join(dirpath, f"{basename}{ext}")
    if not os.path.exists(candidate):
        return candidate
    idx = 1
    while True:
        candidate = os.path.join(dirpath, f"{basename}_{idx}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1

# ── numpy 기반 CSV 저장 함수 ────────────────────────────
def save_csv(fname: str, arr: np.ndarray, header=["X_m","Y_m"]):
    np.savetxt(fname,
               arr,
               delimiter=',',
               header=','.join(header),
               comments='',
               fmt='%.6f')

# ── 설정 값 ─────────────────────────────────────────────
PORT, BAUD, TIMEOUT = "COM10", 115200, 0.1
RADIUS_WGS84       = 6_378_137.0     # 지구 반지름 (m)

# ── 로그 파일 경로 (중복 시 _1, _2 … 자동 인덱싱) ──────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 스크립트 위치
LOG_FILE = unique_filepath(BASE_DIR, "raw_track_xy")  # 중복 없는 CSV 파일명을 LOG_FILE에 설정

# ── NMEA ddmm.mmmm → 십진도 변환 함수 ─────────────────────

#val_str 매개변수는 NMEA 형식의 위도·경도 문자열(예: "3723.2475")을 의미, 반환값 타입 float(십진도)
def nmea2deg(val_str: str) -> float: # NMEA 형식 ddmm.mmmm 또는 dddmm.mmmm 문자열을 float 로 바꿈
    try: # 예외 처리 블록의 시작
        v = float(val_str) # 입력된 문자열 val_str을 부동소수점 숫자로 변환
        deg = int(v // 100) # 도 # v // 100 은 소수점 이하를 버리고 100으로 나눈 몫을 계산
        minute = v - deg * 100 # 전체 값 v에서 deg * 100을 빼면 “분(′)” 부분만 남습니다.
        return deg + minute / 60.0 # 분”을 60으로 나누어 십진도로 환산,  도”에 더해 최종 십진도(°) 값을 리턴
    except:
        return float('nan') # 어떠한 예외(예: 빈 문자열, 숫자로 변환 불가 등)가 발생하면 nan 리턴

# ── 위/경도 → Web Mercator(X, Y) 변환 함수 ────────────────
def mercator_xy(lat: float, lon: float): # lat(위도)와 lon(경도)을 float(십진도(°)) 단위로 받음
    x = RADIUS_WGS84 * math.radians(lon) # 경도(degree)를 라디안(radian) 으로 변환
                                        # 이를 지구 반지름 RADIUS_WGS84(미터)와 곱해 X 좌표(미터 단위)를 계산
                                        # → 경도가 동쪽으로 1° 이동할 때 대략 111 km 이동하는 거리를 반영한 값
    y = RADIUS_WGS84 * math.log(math.tan(math.radians(90 + lat) / 2)) # 먼저 (90 + lat)를 라디안 단위로 바꾼 뒤 / 2를 해 줍니다:
                                                                      # 이는 위도 lat를 Web Mercator 공식의 π/4 + φ/2 형태로 변환한 것과 동일
                                                                      # math.tan(θ)로 탄젠트를 구하고,math.log(...)로 자연로그를 취해 Y 방향 좌표(미터 단위) 값을 얻음
                                                                      # → 위도가 북쪽으로 높아질수록 지구 표면이 수직 방향으로 무한히 늘어나는 투영 특성을 반영 


    return x, y # 계산된 X, Y 값을 튜플로 반환
 
# ── 실시간 플롯 초기화 ───────────────────────────────────
plt.ion() # 인터랙티브 모드(on) 로 전환 # plt.pause()나 plt.draw() 등을 호출할 때마다 화면이 자동으로 갱신되어, 루프 안에서 실시간으로 플롯을 업데이트할 수 있습니다.
fig, ax = plt.subplots()  # 새로운 Figure 객체(fig)와 그 위에 그려질 Axes 객체(ax)를 생성합니다.
                          # 하나의 창(window)에 하나의 그래프 축을 만들 때 자주 쓰는 패턴입니다.
ax.set_title("RTK-GPS Raw Data Track (X, Y)") # 래프 상단에 표시될 제목(title) 을 설정
ax.set_xlabel("X [m]") # X축 레이블(label) 을 “X [m]” 로 지정
ax.set_ylabel("Y [m]") # Y축 레이블(label) 을 “Y [m]” 로 지정
ax.grid(True) # 그래프 배경에 격자선(grid) 을 켭니다.
line, = ax.plot([], [], marker='.', linestyle='-') # 빈 데이터 ([], []) 로 시작하는 선 그래프 객체를 하나 만듭니다.
                                                  # 반환값이 튜플 형태라 line, = ... 처럼 쉼표를 붙여 첫 번째 요소만 꺼냅니다.
                                                  # 나중에 line.set_data(xs, ys) 로 xs, ys 리스트를 넣어주면, 이 객체가 갱신

xs, ys = [], [] # 실시간으로 수집되는 X 좌표 리스트(xs) 와 Y 좌표 리스트(ys) 를 빈 상태로 초기화
                # 루프 안에서 각각 append(x) / append(y) 로 값을 추가해 주면, 플롯이 점점 그려집니다.
# ── 시리얼 포트 열기 ────────────────────────────────────
def open_serial(): # 시리얼 포트를 열고, 성공하면 그 Serial 객체를 반환하며, 실패하면 None을 반환하도록 정의
    try:
        s = serial.Serial(PORT, BAUD, timeout=TIMEOUT) # pyserial 라이브러리의 Serial 생성자를 호출해 실제로 시리얼 포트를 엽니다.
                                                       # PORT: COM 포트 이름 (예: "COM10")
                                                       # BAUD: 통신 속도 (예: 115200 bps)
                                                       # timeout: 읽기 대기 시간(초 단위, 예: 0.1)
                                                       # 성공하면 s 변수에 Serial 객체가 저장
        print(f"{PORT} 포트가 열렸습니다. 로그 파일 → {LOG_FILE}")
        return s # 열린 시리얼 포트 객체 s를 호출자에게 반환, 이후 이 객체를 통해 readline(), write() 등의 입출력 메서드를 사용할 수 있습니다.
    except Exception as e:
        print("시리얼 포트 열기 실패:", e)
        return None

ser = open_serial() # 앞서 정의한 open_serial() 함수를 호출해 시리얼 포트를 열고, 성공하면 그 객체를 ser에 저장
print("GNGGA 수신 대기 중... (Ctrl+C 로 종료)")

try: # 이후 키보드 인터럽트(Ctrl+C) 등으로 루프를 빠져나올 때 finally 블록으로 정상 진입시키기 위한 예외 처리 블록을 시작합니다.
    while True:
        # 포트가 닫히면 재시도
        if ser is None or not ser.is_open: # 시리얼 객체 ser가 없거나 포트가 닫힌 상태인지 확인
            time.sleep(1) # 닫힌 상태라면 1초 대기해 잠시 휴지기를 줍니다.
            ser = open_serial() # 다시 open_serial()을 호출해 포트를 재연결
            continue # 루프의 맨 처음으로 되돌아가, 이후 코드 실행을 건너뜁니다.

        raw = ser.readline().decode("ascii", errors="ignore").strip() # 시리얼 포트에서 한 줄(\n까지) 읽은 바이트를 ASCII 문자열로 디코딩하고,
                                                                      # 앞뒤 공백(\r\n 등)을 strip()으로 제거해 raw에 저장합니다.
        if not raw.startswith("$GNGGA"): # 읽어들인 문자열이 $GNGGA로 시작하지 않으면
            continue                     #루프 시작으로 돌아가 다음 줄을 읽습니다. 

        parts = raw.split(",") # 쉼표(,) 기준으로 문자열을 분해해 리스트 parts에 저장합니다.
        if len(parts) < 6: # GGA 메시지는 최소 6개 필드를 가져야 위도·경도 정보가 있으므로,
            continue       # 필드가 부족하면 유효하지 않은 메시지로 간주하고 루프 재시작합니다.

        # 위도·경도 파싱
        lat = nmea2deg(parts[2]) # 리스트의 3번째 요소(parts[2])에 담긴 위도(ddmm.mmmm)를 nmea2deg로 십진도로 변환해 lat에 저장
        lon = nmea2deg(parts[4]) # 5번째 요소(parts[4])에 담긴 경도(ddmm.mmmm)를 변환해 lon에 저장
        if parts[3] == "S": lat = -lat # 위도 뒤의 방향 인디케이터(parts[3])가 "S"(남위)이면 부호를 음수로 바꿉니다.
        if parts[5] == "W": lon = -lon # 경도 뒤의 방향(parts[5])이 "W"(서경)이면 lon을 음수로 만듭니다.

        # Web Mercator 변환
        x, y = mercator_xy(lat, lon) # 십진도로 바뀐 lat, lon을 mercator_xy 함수에 넘겨 Web Mercator X,Y 좌표(미터 단위)를 계산해 받습니다.

        # 플롯 업데이트 
        xs.append(x) # 실시간 추적용 리스트 xs, ys에 새 좌표를 덧붙입니다.
        ys.append(y) # 실시간 추적용 리스트 xs, ys에 새 좌표를 덧붙입니다.
        line.set_data(xs, ys) # 빈 plot 객체 line에 업데이트된 전체 좌표 리스트를 다시 설정
        ax.relim() # 축(Axes)의 데이터 한계를 재계산하도록 표시
        ax.autoscale_view() # 새로 계산된 한계에 맞춰 보기 영역(view limits)을 자동으로 조정
        plt.pause(0.01) # 0.01초 동안 대기하며, 이 동안 화면이 갱신(인터랙티브 모드라면 이 호출만으로도 플롯이 실시간으로 그려집니다.)

        # 콘솔 출력(현재시간,위도,경도)
        print(f"{time.strftime('%H:%M:%S')}  X={x:.2f} m, Y={y:.2f} m")

except KeyboardInterrupt: # 키보드 인터럽트( Ctrl+C ) 예외를 잡습니다.
    print("\n데이터 수집을 종료합니다.") 

finally:
    # 포트 닫기
    if ser and ser.is_open: # ser 객체가 존재하고(ser가 None이 아니며),실제로 포트가 열려 있는 상태(ser.is_open==True)인지를 확인합니다.
        ser.close() # 조건이 참이라면 Serial 객체의 close() 메서드를 호출해 포트 연결을 해제

    # 최종 수집된 X, Y 배열을 numpy로 저장
    data = np.column_stack([xs, ys]) # 실시간으로 모은 리스트 xs, ys를 열(column) 방향으로 합쳐 (N, 2) 형태의 2차원 NumPy 배열로 만듭니다.
    save_csv(LOG_FILE, data) # 앞서 정의한 save_csv() 함수를 호출해, LOG_FILE 경로에 data 배열을 한 번에 CSV로 저장
    print(f"수집된 로우 데이터가 저장되었습니다 → {LOG_FILE}") # 저장 완료 메시지를 콘솔에 출력

    plt.ioff() # 인터랙티브 모드(off) 로 전환 이제부터는 plt.pause() 없이도 plt.show()만으로 창을 띄웁니다.
    plt.show() # 최종 플롯 창을 블로킹 모드로 띄웁니다.사용자가 창을 닫기 전까지 스크립트가 종료되지 않습니다.
