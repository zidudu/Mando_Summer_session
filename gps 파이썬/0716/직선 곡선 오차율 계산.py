# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# gps_traj_error_simple.py

# • RDP로 웨이포인트 단순화
# • 곡률 기반으로 직선/곡선 세그먼트 분리
#   - 직선: 선형 회귀 → Cross-Track Error 계산
#   - 곡선: B-스플라인 보간 → 수직 편차 계산
# • 각 세그먼트별 및 전체 오차 통계 출력
# • 궤적·편차 히스토그램·통계 막대그래프 시각화
# """

# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from rdp import rdp
# from scipy.interpolate import splprep, splev

# # ─── 사용자 설정 ─────────────────────────────────────────
# CSV_FILE     = "raw_track_xy_20.csv"  # 헤더: X,Y
# EPS_RDP      = 0.1     # RDP 허용 오차 [m]
# CURV_THRESH  = 0.02    # 곡률 임계값 (이 값 이하 → 직선)
# MIN_SEG_LEN  = 5       # 최소 세그먼트 길이 (샘플 수)
# SMOOTH_PARAM = 0.0     # splprep 스무딩 파라미터
# # ─────────────────────────────────────────────────────────


# def load_xy(path):
#     """CSV에서 X,Y 로드 (헤더 1줄 건너뜀)"""
#     data = np.loadtxt(path, delimiter=',', skiprows=1)
#     return data[:,0], data[:,1]


# def compute_curvature(xs, ys):
#     """수치미분으로 곡률 κ 계산"""
#     dx = np.gradient(xs)
#     dy = np.gradient(ys)
#     ddx = np.gradient(dx)
#     ddy = np.gradient(dy)
#     num = dx*ddy - dy*ddx
#     den = (dx*dx + dy*dy)**1.5
#     κ = np.zeros_like(xs)
#     mask = den > 1e-12
#     κ[mask] = np.abs(num[mask] / den[mask])
#     return κ


# def segment_by_curvature(κ):
#     """
#     κ < CURV_THRESH: 직선 세그먼트
#     κ ≥ CURV_THRESH: 곡선 세그먼트
#     """
#     segs, types = [], []
#     i = 0
#     N = len(κ)
#     while i < N:
#         mode = "line" if κ[i] < CURV_THRESH else "curve"
#         j = i+1
#         while j < N and ((mode=="line" and κ[j]<CURV_THRESH)
#                         or (mode=="curve" and κ[j]>=CURV_THRESH)):
#             j += 1
#         if j - i >= MIN_SEG_LEN:
#             segs.append((i, j))
#             types.append(mode)
#         i = j
#     return segs, types


# def fit_line_and_error(xs, ys):
#     """1차 polyfit + Cross-Track Error 계산"""
#     m, b = np.polyfit(xs, ys, 1)
#     # ax + by + c = 0 형태: a=m, b=-1, c=b
#     a, bb, c = m, -1.0, b
#     den = math.hypot(a, bb)
#     errs = np.abs(a*xs + bb*ys + c) / den
#     return errs, (m, b)


# def fit_spline_and_error(xs, ys):
#     """B-스플라인 보간 + 수직 거리 계산"""
#     k = max(1, min(3, len(xs)-1))
#     tck, _ = splprep([xs, ys], s=SMOOTH_PARAM, k=k)
#     u = np.linspace(0,1,len(xs))
#     xs_s, ys_s = splev(u, tck)
#     errs = np.hypot(xs - xs_s, ys - ys_s)
#     return errs, tck


# def seg_stats(deviations):
#     """[평균, 표준편차, RMSE, 최소, 최대, 범위] 반환"""
#     arr = np.asarray(deviations)
#     mean = arr.mean()
#     std  = arr.std()
#     rmse = math.sqrt((arr**2).mean())
#     mn, mx = arr.min(), arr.max()
#     rng = mx - mn
#     return mean, std, rmse, mn, mx, rng


# if __name__ == "__main__":
#     # 1) 로우 XY 로드 & RDP 단순화
#     xs, ys = load_xy(CSV_FILE)
#     simp = rdp(np.column_stack([xs, ys]), epsilon=EPS_RDP)
#     simp_x, simp_y = simp[:,0], simp[:,1]
#     idx_map = [np.argmin((xs-x)**2 + (ys-y)**2)
#                for x,y in zip(simp_x, simp_y)]

#     # 2) 전체 곡률 프로파일
#     κ = compute_curvature(xs, ys)

#     # 3) 곡률로 세그먼트 분리
#     segs, types = segment_by_curvature(κ)

#     # 4) 각 세그먼트별 오차 계산
#     dev_line, dev_curve = [], []
#     models = []
#     for (i0,i1), t in zip(segs, types):
#         seg_x = xs[i0:i1]
#         seg_y = ys[i0:i1]

#         if t == "line":
#             errs, line_model = fit_line_and_error(seg_x, seg_y)
#             dev_line.append(errs)
#             models.append(("line", line_model, (i0,i1)))
#         else:
#             errs, spline_model = fit_spline_and_error(seg_x, seg_y)
#             dev_curve.append(errs)
#             models.append(("curve", spline_model, (i0,i1)))

#     # 5) 전체 오차 통계
#     all_dev = np.hstack(dev_line + dev_curve) if (dev_line+dev_curve) else np.array([])
#     stats_all = seg_stats(all_dev)
#     stats_line = seg_stats(np.hstack(dev_line))   if dev_line else None
#     stats_curve= seg_stats(np.hstack(dev_curve))  if dev_curve else None

#     # 6) 시각화
#     fig, axes = plt.subplots(1, 3, figsize=(18,5))

#     # (1) Raw XY와 세그먼트 모델
#     ax = axes[0]
#     ax.plot(xs, ys, 'k.', ms=4, label="Raw")
#     for kind, mdl, (i0,i1) in models:
#         if kind=="line":
#             m,b = mdl
#             xx = [xs[i0], xs[i1-1]]
#             yy = [m*xx[0]+b, m*xx[1]+b]
#             ax.plot(xx, yy, 'g--')
#         else:
#             tck = mdl
#             u = np.linspace(0,1,200)
#             xs_s, ys_s = splev(u, tck)
#             ax.plot(xs_s, ys_s, 'b-')
#     ax.set_title("Raw & Segment Fits")
#     ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.axis("equal"); ax.grid()

#     # (2) 편차 히스토그램
#     ax = axes[1]
#     ax.hist(all_dev, bins=30, color='tab:purple', alpha=0.7)
#     ax.set_title("Deviation Histogram")
#     ax.set_xlabel("Error [m]"); ax.set_ylabel("Count"); ax.grid(axis='y', ls='--')

#     # (3) 통계 막대그래프
#     ax = axes[2]
#     labels, values = [], []
#     if stats_line:
#         labels.append("Line")
#         values.append(stats_line)
#     if stats_curve:
#         labels.append("Curve")
#         values.append(stats_curve)
#     metrics = ["mean","std","rmse","min","max","range"]
#     x = np.arange(len(metrics))
#     w = 0.8/len(values) if values else 0.8
#     for idx, (lab, vals) in enumerate(zip(labels, values)):
#         ax.bar(x+idx*w, vals, w, label=lab)
#         for xi, vv in zip(x+idx*w, vals):
#             ax.text(xi, vv+1e-4, f"{vv:.3f}", ha='center', va='bottom', fontsize=7)
#     ax.set_xticks(x + w*(len(values)-1)/2)
#     ax.set_xticklabels(metrics)
#     ax.set_title("Segment Error Stats")
#     ax.set_ylabel("Error [m]"); ax.legend(); ax.grid(axis='y', ls='--')

#     plt.tight_layout()
#     plt.show()

#     # 7) 세그먼트별 편차 개별 플롯 (선택)
#     if dev_line:
#         plt.figure(figsize=(6,3))
#         plt.bar(range(sum(len(d) for d in dev_line)), np.hstack(dev_line), color='g')
#         plt.title("Line Segment Errors"); plt.xlabel("Sample Index"); plt.ylabel("Error [m]"); plt.grid(True)
#         plt.tight_layout(); plt.show()
#     if dev_curve:
#         plt.figure(figsize=(6,3))
#         plt.bar(range(sum(len(d) for d in dev_curve)), np.hstack(dev_curve), color='b')
#         plt.title("Curve Segment Errors"); plt.xlabel("Sample Index"); plt.ylabel("Error [m]"); plt.grid(True)
#         plt.tight_layout(); plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gps_traj_error_ransac.py

• RDP 단순화 → 곡률 기반 세그먼트 분리
  - 곡률 < CURV_THRESH: 직선 구간 → RANSAC으로 피팅
  - 곡률 ≥ CURV_THRESH: 곡선 구간 → B-스플라인으로 피팅
• Cross-Track Error / Spline Error 계산
• 세그먼트별·전체 오차 통계 + 시각화
"""

import math, sys
import numpy as np
import matplotlib.pyplot as plt
from rdp import rdp
from scipy.interpolate import splprep, splev
from sklearn.linear_model import RANSACRegressor

# ─── 파라미터 ───────────────────────────────────────────────
CSV_FILE     = sys.argv[1] if len(sys.argv)>1 else "raw_track_xy_17.csv"
EPS_RDP      = 0.1      # RDP 단순화 오차 [m]
CURV_THRESH  = 0.02     # 곡률 임계값
MIN_SEG_LEN  = 5        # 최소 세그먼트 길이 [샘플 수]
LINE_THR     = 0.05     # RANSAC 잔차 한계 [m]
SMOOTH_PARAM = 0.0      # splprep 스무딩 파라미터
# ────────────────────────────────────────────────────────────

def load_xy(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return data[:,0], data[:,1]

def compute_curvature(xs, ys):
    dx = np.gradient(xs); dy = np.gradient(ys)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    num = dx*ddy - dy*ddx
    den = (dx*dx + dy*dy)**1.5
    κ = np.zeros_like(xs)
    mask = den > 1e-12
    κ[mask] = np.abs(num[mask] / den[mask])
    return κ

def segment_by_curvature(κ):
    segs, types = [], []
    i = 0; N = len(κ)
    while i < N:
        mode = "line" if κ[i] < CURV_THRESH else "curve"
        j = i+1
        while j<N and ((mode=="line" and κ[j]<CURV_THRESH) or
                       (mode=="curve" and κ[j]>=CURV_THRESH)):
            j += 1
        if j-i >= MIN_SEG_LEN:
            segs.append((i,j))
            types.append(mode)
        i = j
    return segs, types

def fit_line_ransac_and_error(xs, ys):
    """
    RANSAC으로 직선 피팅 후 inlier만 Cross-Track Error 계산
    반환: (errors, (slope, intercept), inlier_mask)
    """
    model = RANSACRegressor(residual_threshold=LINE_THR)
    model.fit(xs.reshape(-1,1), ys)
    m = model.estimator_.coef_[0]
    b = model.estimator_.intercept_
    mask = model.inlier_mask_
    # ax + by + c = 0 형태: a=m, b=-1, c=b
    a, bb, c = m, -1.0, b
    den = math.hypot(a, bb)
    errs = np.abs(a*xs + bb*ys + c) / den
    return errs[mask], (m, b), mask

def fit_spline_and_error(xs, ys):
    """
    B-스플라인 보간 → 각 점의 수직 편차 계산
    반환: (errors, spline_tck)
    """
    k = max(1, min(3, len(xs)-1))
    tck, _ = splprep([xs, ys], s=SMOOTH_PARAM, k=k)
    u = np.linspace(0,1,len(xs))
    xs_s, ys_s = splev(u, tck)
    errs = np.hypot(xs - xs_s, ys - ys_s)
    return errs, tck

def seg_stats(arr):
    """[mean, std, rmse, min, max, range]"""
    a = np.asarray(arr)
    mean = a.mean()
    std  = a.std()
    rmse = math.sqrt((a**2).mean())
    mn, mx = a.min(), a.max()
    return mean, std, rmse, mn, mx, (mx-mn)

if __name__=="__main__":
    # 1) 데이터 로드 + RDP 단순화
    xs, ys = load_xy(CSV_FILE)
    simp = rdp(np.column_stack([xs, ys]), epsilon=EPS_RDP)
    # (RDP 상의 점 위치 → 원본 인덱스 맵)
    idx_map = [np.argmin((xs-x)**2 + (ys-y)**2)
               for x,y in zip(simp[:,0], simp[:,1])]

    # 2) 곡률 계산 + 세그먼트 분리
    κ = compute_curvature(xs, ys)
    segs, types = segment_by_curvature(κ)

    # 3) 각 세그먼트별 에러 계산
    dev_line, dev_curve = [], []
    models = []
    for (i0,i1), t in zip(segs, types):
        seg_x = xs[i0:i1]; seg_y = ys[i0:i1]
        if t=="line":
            errs, lm, mask = fit_line_ransac_and_error(seg_x, seg_y)
            dev_line.append(errs)
            models.append(("line", lm, (i0,i1), mask))
        else:
            errs, tck = fit_spline_and_error(seg_x, seg_y)
            dev_curve.append(errs)
            models.append(("curve", tck, (i0,i1), None))

    # 4) 전체·세그먼트 통계
    all_err = np.hstack(dev_line + dev_curve) if (dev_line+dev_curve) else np.array([])
    stats_all   = seg_stats(all_err)   if all_err.size   else None
    stats_line  = seg_stats(np.hstack(dev_line))  if dev_line   else None
    stats_curve = seg_stats(np.hstack(dev_curve)) if dev_curve  else None

    # 5) 시각화
    fig, axes = plt.subplots(1,3,figsize=(18,5))
    # (1) 궤적 + 세그먼트 모델
    ax = axes[0]
    ax.plot(xs, ys, 'k.', ms=4)
    for kind, mdl, (i0,i1), mask in models:
        if kind=="line":
            m,b = mdl
            x0, x1 = xs[i0], xs[i1-1]
            y0, y1 = m*x0+b, m*x1+b
            ax.plot([x0,x1],[y0,y1],'g--')
        else:
            tck = mdl
            u = np.linspace(0,1,200)
            xs_s, ys_s = splev(u, tck)
            ax.plot(xs_s, ys_s, 'b-')
    ax.set_title("Raw & Segment Fits"); ax.axis("equal"); ax.grid()

    # (2) 에러 히스토그램
    ax = axes[1]
    ax.hist(all_err, bins=30, color='tab:purple', alpha=0.7)
    ax.set_title("Deviation Histogram"); ax.set_xlabel("Error [m]"); ax.set_ylabel("Count"); ax.grid(axis='y', ls='--')

    # (3) 통계 막대그래프
    ax = axes[2]
    labels, vals = [], []
    if stats_line:  labels.append("Line");  vals.append(stats_line)
    if stats_curve: labels.append("Curve"); vals.append(stats_curve)
    metrics = ["mean","std","rmse","min","max","range"]
    x = np.arange(len(metrics)); w = 0.8/max(1,len(vals))
    for i,(lab,v) in enumerate(zip(labels, vals)):
        ax.bar(x+i*w, v, w, label=lab)
        for xi, vv in zip(x+i*w, v):
            ax.text(xi, vv+1e-4, f"{vv:.3f}", ha='center', va='bottom')
    ax.set_xticks(x + w*(len(vals)-1)/2)
    ax.set_xticklabels(metrics)
    ax.set_title("Segment Error Stats"); ax.set_ylabel("Error [m]"); ax.legend(); ax.grid(axis='y', ls='--')

    plt.tight_layout()
    plt.show()
