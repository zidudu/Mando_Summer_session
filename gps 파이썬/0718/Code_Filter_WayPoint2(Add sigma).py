#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_curve_and_make_waypoints_onefig.py (주석 추가 버전)
────────────────────────────────────────────────────────
■ 기능 요약
    • 원본 CSV 궤적(x, y) → (곡선: B‑스플라인 / 직선: 선형 회귀)로 스무딩 필터링
    • 곡률(κ)에 따라 동적 간격으로 웨이포인트 생성
    • 1×3 Figure 출력
        ① Raw vs Filtered 트랙 비교
        ② Waypoints 분포
        ③ Raw/Filtered 오차 통계 (Mean, RMSE, Range, Std)
    • 웨이포인트를 CSV(중복 방지 자동 인덱싱)로 저장

※ 직선 구간은 선형 회귀(1차 다항식)로 평활화, 곡선 구간은 B‑스플라인으로 스무딩 처리
※ 모든 변수와 함수에 한국어 설명 주석 추가
"""

import sys
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev

# ───────────────────────────────────────────────────────
# ★★★ 사용자 조정 파라미터 ★★★
#    • CSV_IN  : 입력 CSV 파일명 (헤더: X,Y) ─ 기본값 "raw_track_xy_18.csv"
#    • CURV_TH : 직선/곡선 판별 곡률 임계값
#    • MIN_LEN : 세그먼트 최소 길이(샘플 수) ─ 너무 짧은 세그먼트는 무시
#    • SMOOTH_S: 스플라인 스무딩 강도(클수록 부드러움)
#    • K1, K2  : 동적 웨이포인트용 곡률 구간 경계
#    • D_S, D_M, D_C : 직선·중간·급곡선 구간에서의 웨이포인트 간격[m]
#    • VIEW_R  : Figure 좌우/상하 시야 범위[m]
# ───────────────────────────────────────────────────────
CSV_IN    = sys.argv[1] if len(sys.argv) > 1 else "raw_track_xy_18.csv"
CURV_TH   = 0.015
MIN_LEN   = 5
SMOOTH_S  = 0.6
K1, K2    = 0.10, 0.20
D_S, D_M, D_C = 0.55, 0.40, 0.28
VIEW_R    = 5.0

# ───────────────────────────────────────────────────────
# 1. 유틸리티 함수들
# ───────────────────────────────────────────────────────

def load_xy(fname: str):
    """CSV 파일에서 X, Y 열 로드 (헤더 1줄 스킵)"""
    d = np.loadtxt(fname, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1]


def curvature(x: np.ndarray, y: np.ndarray):
    """평면 궤적의 곡률 κ 계산

    κ = |x'·y'' − y'·x''| / (x'^2 + y'^2)^(3/2)
    (x', y'): 1차 미분, (x'', y''): 2차 미분
    """
    dx, dy = np.gradient(x), np.gradient(y)          # 1차 미분
    ddx, ddy = np.gradient(dx), np.gradient(dy)     # 2차 미분
    num = dx * ddy - dy * ddx                       # 분자
    den = (dx ** 2 + dy ** 2) ** 1.5               # 분모
    kappa = np.zeros_like(x)
    mask = den > 1e-12                              # 0‑division 보호
    kappa[mask] = np.abs(num[mask] / den[mask])
    return kappa


def segments(kappa: np.ndarray):
    """곡률 배열을 이용해 [직선 / 곡선] 세그먼트 분할

    반환값:
        segs : [(i0, i1), ...]  # 각 세그먼트의 [시작, 끝) 인덱스
        ty   : ["line" | "curve", ...]  # 세그먼트 타입
    """
    segs, ty = [], []
    i, N = 0, len(kappa)
    while i < N:
        mode = "line" if kappa[i] < CURV_TH else "curve"
        j = i + 1
        # 동일 모드가 이어지는 구간 끝까지 탐색
        while j < N and ((mode == "line" and kappa[j] < CURV_TH) or
                          (mode == "curve" and kappa[j] >= CURV_TH)):
            j += 1
        # 충분히 길면 세그먼트로 채택
        if j - i >= MIN_LEN:
            segs.append((i, j))
            ty.append(mode)
        i = j
    return segs, ty


def smooth_curve(x: np.ndarray, y: np.ndarray):
    """곡선 세그먼트(B‑스플라인) 스무딩"""
    # 중복 좌표 제거 후 스플라인(최대 차수 k=3 사용)
    pts = list(dict.fromkeys(zip(x, y)))            # 순서 유지 중복 제거
    ux, uy = zip(*pts)
    n = len(ux)
    if n < 2:
        return x, y
    k = min(3, n - 1)
    tck, _ = splprep([ux, uy], s=SMOOTH_S, k=k)
    xs_s, ys_s = splev(np.linspace(0, 1, len(x)), tck)
    return xs_s, ys_s


def smooth_line_regress(x: np.ndarray, y: np.ndarray):
    """직선 세그먼트(선형 회귀) 스무딩"""
    if len(x) < 2:
        return x, y
    m, b = np.polyfit(x, y, 1)      # y = m·x + b
    y_fit = m * x + b
    return x, y_fit


def filter_track(xr: np.ndarray, yr: np.ndarray):
    """원본(Raw) 궤적을 직선/곡선별로 필터링하여 평활화된 궤적 반환"""
    kappa = curvature(xr, yr)
    segs, ty = segments(kappa)

    xf, yf = xr.copy(), yr.copy()   # 결과 배열 (in‑place 갱신)
    for (i0, i1), mode in zip(segs, ty):
        if mode == "curve":
            xs_s, ys_s = smooth_curve(xr[i0:i1], yr[i0:i1])
        else:  # "line"
            xs_s, ys_s = smooth_line_regress(xr[i0:i1], yr[i0:i1])
        xf[i0:i1], yf[i0:i1] = xs_s, ys_s
    return xf, yf


def dyn_wp(x: np.ndarray, y: np.ndarray):
    """곡률 기반 동적 웨이포인트 생성"""
    kap = curvature(x, y)
    wpts = [(x[0], y[0])]  # 첫 점 고정
    acc = 0.0              # 누적 거리
    for i in range(1, len(x)):
        seg = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        acc += seg
        # 현재 샘플의 목표 간격 결정
        κ = kap[i]
        d_t = D_S if κ < K1 else (D_M if κ < K2 else D_C)
        # 누적 거리가 목표 간격 이상이면 웨이포인트 추가
        while acc >= d_t:
            r = (seg - (acc - d_t)) / seg  # 마지막 구간 상 보간 비율
            wpts.append((x[i - 1] + (x[i] - x[i - 1]) * r,
                         y[i - 1] + (y[i] - y[i - 1]) * r))
            acc -= d_t
    wpts.append((x[-1], y[-1]))           # 마지막 점 추가
    return np.array(wpts)


def seg_errors(x: np.ndarray, y: np.ndarray):
    """세그먼트별 모델 대비 편차(Residual) 계산"""
    kappa = curvature(x, y)
    segs, ty = segments(kappa)
    dev = []
    for (i0, i1), mode in zip(segs, ty):
        xs, ys = x[i0:i1], y[i0:i1]
        if mode == "line":
            m, b = np.polyfit(xs, ys, 1)
            dev.append(np.abs(m * xs - ys + b) / math.hypot(m, -1))
        else:  # curve
            xs_s, ys_s = smooth_curve(xs, ys)
            dev.append(np.hypot(xs - xs_s, ys - ys_s))
    return np.hstack(dev) if dev else np.array([])


def stats(arr: np.ndarray):
    """편차 배열로부터 (평균, RMSE, Range, 표준편차) 계산"""
    return arr.mean(), math.sqrt((arr ** 2).mean()), np.ptp(arr), arr.std()


def save_wp(arr: np.ndarray, stem: str = "waypoints_dynamic") -> Path:
    """웨이포인트 CSV 저장 (중복 시 _1, _2 … 자동 인덱싱)"""
    p = Path(f"{stem}.csv")
    for idx in range(1000):
        if not p.exists():
            np.savetxt(p, arr, delimiter=",", header="X_m,Y_m", comments="", fmt="%.6f")
            return p
        p = Path(f"{stem}_{idx}.csv")

# ───────────────────────────────────────────────────────
# 2. 메인 실행부
# ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # ① CSV 로드 및 원점(첫 샘플) 기준 상대 좌표화
    X, Y = load_xy(CSV_IN)
    xr, yr = X - X[0], Y - Y[0]          # ΔX, ΔY (m)

    # ② 궤적 필터링
    xf, yf = filter_track(xr, yr)

    # ③ 동적 웨이포인트 생성 & 저장
    WPT = dyn_wp(xf, yf)
    out_csv = save_wp(WPT)

    # ④ 오차 통계 계산 (Raw vs Filtered)
    err_raw, err_flt = seg_errors(xr, yr), seg_errors(xf, yf)
    m_r, rmse_r, rng_r, std_r = stats(err_raw)
    m_f, rmse_f, rng_f, std_f = stats(err_flt)

    # ⑤ Figure(1×3) 생성
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Raw vs Filtered
    ax[0].plot(xr, yr, "0.6", lw=1, label="Raw")
    ax[0].plot(xf, yf, "r-", lw=1.5, label="Filtered")
    ax[0].set_title("Raw vs Filtered")
    ax[0].axis("equal")
    ax[0].set_xlim(-VIEW_R, VIEW_R)
    ax[0].set_ylim(-VIEW_R, VIEW_R)
    ax[0].set_xlabel("ΔX [m]")
    ax[0].set_ylabel("ΔY [m]")
    ax[0].legend()
    ax[0].grid()

    # (2) Waypoints
    ax[1].plot(xf, yf, "k-", lw=1, alpha=0.4, label="Filtered")
    ax[1].scatter(WPT[:, 0], WPT[:, 1], c="orange", marker="*", s=40,
                  label=f"Waypoints ({len(WPT)})")
    ax[1].set_title("Dynamic Waypoints")
    ax[1].axis("equal")
    ax[1].set_xlim(-VIEW_R, VIEW_R)
    ax[1].set_ylim(-VIEW_R, VIEW_R)
    ax[1].set_xlabel("ΔX [m]")
    ax[1].set_ylabel("ΔY [m]")
    ax[1].legend()
    ax[1].grid()

    # (3) Error Statistics 막대그래프
    labels = ["Mean", "RMSE", "Range", "Std(σ)"]
    raw_vals = [m_r, rmse_r, rng_r, std_r]
    flt_vals = [m_f, rmse_f, rng_f, std_f]
    x_pos = np.arange(len(labels))
    w = 0.35  # bar width

    ax[2].bar(x_pos - w / 2, raw_vals, width=w, label="Raw", color="skyblue")
    ax[2].bar(x_pos + w / 2, flt_vals, width=w, label="Filtered", color="salmon")

    # 막대 위에 값 표시
    for i, v in enumerate(raw_vals):
        ax[2].text(i - w / 2, v + 1e-4, f"{v:.3f}", ha="center", va="bottom")
    for i, v in enumerate(flt_vals):
        ax[2].text(i + w / 2, v + 1e-4, f"{v:.3f}", ha="center", va="bottom")

    ax[2].set_xticks(x_pos)
    ax[2].set_xticklabels(labels)
    ax[2].set_ylabel("[m]")
    ax[2].set_title("Error Statistics")
    ax[2].legend()
    ax[2].grid(axis="y", ls="--")

    plt.tight_layout()
    plt.show()

    # ⑥ 콘솔 요약 출력
    print("\n─── 오차 통계 (단위:m) ───")
    print(f"[Raw]      mean={m_r:.4f}, RMSE={rmse_r:.4f}, range={rng_r:.4f}, std={std_r:.4f}")
    print(f"[Filtered] mean={m_f:.4f}, RMSE={rmse_f:.4f}, range={rng_f:.4f}, std={std_f:.4f}")
    print(f"\n[Waypoints] 저장 완료 → {out_csv} (총 {len(WPT)}개)")
