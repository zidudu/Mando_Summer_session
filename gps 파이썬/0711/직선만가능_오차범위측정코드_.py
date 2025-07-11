#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
선형 회귀 기반 직선 추종 오차 분석 및 시각화
  - CSV에서 X, Y 데이터 로드
  - np.polyfit을 이용해 선형 회귀 계수 계산 (y = m x + b)
  - 각 점의 회귀선 수직 편차 계산
  - 통계치 출력 (평균, 표준편차, 범위, RMSE)
  - (1) 원본 데이터 & 회귀선 플롯
  - (2) 샘플 인덱스 대비 편차 플롯 (평균, 평균+2σ, 이상치)
"""

import numpy as np                # 수치 계산을 위한 NumPy 모듈 불러오기
import matplotlib.pyplot as plt   # 그래프 출력을 위한 Matplotlib 모듈 불러오기

# 1) CSV 불러오기 (헤더 스킵)
filename = "raw_track_xy_4.csv"  # 분석할 CSV 파일명 지정 (실제 파일명으로 교체)
# np.loadtxt: 텍스트 파일에서 숫자를 배열로 읽어옴
# delimiter: 구분자, skiprows: 건너뛸 행 수
data = np.loadtxt(filename, delimiter=",", skiprows=1)
# x_coords, y_coords에 각각 첫 번째 열과 두 번째 열을 할당
x_coords = data[:, 0]
y_coords = data[:, 1]

# 2) 회귀 계수 계산
# np.polyfit(x, y, 1): 1차 다항식(직선) 회귀계수 [slope, intercept] 반환
slope, intercept = np.polyfit(x_coords, y_coords, 1)

# 3) 수직 편차 계산
# 회귀선 y = slope * x + intercept
# 한 점 (x_i, y_i)의 직선까지 수직거리 = |m x_i - y_i + b| / sqrt(m^2 + 1)
denominator = np.hypot(slope, 1)  # sqrt(slope^2 + 1)
distances = np.abs(slope * x_coords - y_coords + intercept) / denominator

# 4) 통계치 계산
mean_distance  = distances.mean()                     # 평균 편차
std_distance   = distances.std()                      # 표준편차
range_distance = distances.max() - distances.min()    # 편차 범위(최대-최소)
rmse_distance  = np.sqrt(np.mean(distances**2))       # RMSE(루트 평균 제곱 오차)

# 통계치 출력
print(f"Mean deviation        : {mean_distance:.3f} m")
print(f"Standard deviation     : {std_distance:.3f} m")
print(f"Deviation range        : {range_distance:.3f} m")
print(f"RMSE deviation         : {rmse_distance:.3f} m")

# 이상치 인덱스 식별 (편차 > 평균 + 2 * 표준편차)
threshold = mean_distance + 2 * std_distance
outlier_indices = np.where(distances > threshold)[0]
print("Outlier indices:", outlier_indices.tolist())

# 5) 원본 데이터 및 회귀선 플롯
plt.figure(figsize=(6, 5))                                # 그림 크기 설정
plt.scatter(x_coords, y_coords, s=10, alpha=0.6,
            label="Original Data")                        # 산점도: 원본 데이터
# 회귀선을 그리기 위한 x 값 범위 생성
x_line = np.linspace(x_coords.min(), x_coords.max(), 100)
# 해당 x_line에 대응하는 y 값 계산
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'r-', lw=1.5,
         label=f"Regression line: y = {slope:.3e} x + {intercept:.3f}")  # 회귀선 그리기
plt.xlabel("X [m]")                      # x축 레이블
plt.ylabel("Y [m]")                      # y축 레이블
plt.title("Original Data and Regression Line")  # 그래프 제목
plt.legend()                             # 범례 표시
plt.grid(True)                           # 격자 표시

# 6) 샘플 인덱스 대비 편차 플롯
plt.figure(figsize=(6, 4))                # 새로운 그림 생성
plt.plot(distances, marker='o', linestyle='-',
         label="Distance")                # 편차 선 그래프
# 평균 편차 선
plt.axhline(mean_distance, color='orange', lw=1,
            label=f"Mean = {mean_distance:.3f} m")
# 평균 + 2σ 선
plt.axhline(threshold, color='red', ls='--', lw=1,
            label=f"Mean + 2σ = {threshold:.3f} m")
# 이상치 강조
plt.scatter(outlier_indices, distances[outlier_indices],
            color='red', zorder=5, label="Outliers")
plt.xlabel("Sample Index")               # x축 레이블
plt.ylabel("Perpendicular Distance [m]")  # y축 레이블
plt.title("Distance from Regression Line")  # 그래프 제목
plt.legend()                             # 범례 표시
plt.grid(True)                           # 격자 표시

plt.tight_layout()                       # 레이아웃 자동 조정
plt.show()                               # 그래프 출력
