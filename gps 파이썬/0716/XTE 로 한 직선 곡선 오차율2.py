#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mixed Trajectory Error Analysis
 - path_csv : raw trajectory CSV (X,Y)
 - K_TH     : curvature threshold (curve if κ>=K_TH)
 - SMOOTH   : B-spline smoothing param

Metrics:
  Straight segments → RANSAC RMSE, XTE mean, XTE max
  Curve    segments → B-spline RMSE, discrete Fréchet

Visualization:
  1) trajectory colored by straight/curve
  2) bar chart of the 5 metrics
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import splprep, splev

def load_xy(fname):
    data = np.loadtxt(fname, delimiter=',', skiprows=1)
    return data[:,0], data[:,1]

def curvature(xs, ys):
    dx, dy = np.gradient(xs), np.gradient(ys)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = dx*ddy - dy*ddx
    den = (dx*dx + dy*dy)**1.5
    k = np.zeros_like(xs)
    mask = den > 1e-6
    k[mask] = np.abs(num[mask]/den[mask])
    return k

def rmse(err):
    return math.sqrt(np.mean(err**2))

def discrete_frechet(P, Q):
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)
    def c(i, j):
        if ca[i,j] > -0.5:
            return ca[i,j]
        d = np.linalg.norm(P[i] - Q[j])
        if i==0 and j==0:
            ca[i,j] = d
        elif i>0 and j==0:
            ca[i,j] = max(c(i-1,0), d)
        elif i==0 and j>0:
            ca[i,j] = max(c(0,j-1), d)
        else:
            ca[i,j] = max(min(c(i-1,j), c(i-1,j-1), c(i,j-1)), d)
        return ca[i,j]
    return c(n-1, m-1)

def analyze_mixed(path_csv, K_TH=0.02, SMOOTH=1.0):
    xs, ys = load_xy(path_csv)
    kappa = curvature(xs, ys)
    is_curve    = kappa >= K_TH
    is_straight = ~is_curve
    idx_s = np.where(is_straight)[0]
    idx_c = np.where(is_curve)[0]

    # Straight metrics
    if len(idx_s) >= 2:
        Xs = xs[idx_s].reshape(-1,1)
        Ys = ys[idx_s]
        model = RANSACRegressor().fit(Xs, Ys)
        m, b = model.estimator_.coef_[0], model.estimator_.intercept_
        den = math.hypot(m,1)
        errs = np.abs(m*Xs.squeeze() - Ys + b) / den
        rmse_s, xte_m, xte_M = rmse(errs), errs.mean(), errs.max()
    else:
        rmse_s, xte_m, xte_M = np.nan, np.nan, np.nan

    # Curve metrics
    if len(idx_c) >= 3:
        xc, yc = xs[idx_c], ys[idx_c]
        n = len(xc); k = min(3, n-1)
        tck, _ = splprep([xc, yc], s=SMOOTH, k=k)
        u = np.linspace(0,1,n)
        xf, yf = splev(u, tck)
        P = np.column_stack([xc, yc])
        Q = np.column_stack([xf, yf])
        devs = np.linalg.norm(P - Q, axis=1)
        rmse_c = rmse(devs)
        fre_c  = discrete_frechet(P, Q)
    else:
        rmse_c, fre_c = np.nan, np.nan

    return xs, ys, is_curve, is_straight, (rmse_s, xte_m, xte_M), (rmse_c, fre_c)

if __name__ == "__main__":
    # — 파일 및 파라미터 설정 —  
    path_csv = "raw_track_xy_17.csv"
    K_TH      = 0.02
    SMOOTH    = 1.0
    # ————————————————————

    xs, ys, is_curve, is_straight, stra_metrics, curve_metrics = analyze_mixed(path_csv, K_TH, SMOOTH)
    rmse_s, xte_m, xte_M = stra_metrics
    rmse_c, fre_c        = curve_metrics

    # 1) Trajectory segmentation plot
    plt.figure(figsize=(6,6))
    plt.scatter(xs[is_straight], ys[is_straight], c='blue',  s=5, label='Straight')
    plt.scatter(xs[is_curve],    ys[is_curve],    c='orange', s=5, label='Curve')
    plt.title("Trajectory Segmentation")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.legend(); plt.axis('equal'); plt.grid(True)
    plt.show()

    # 2) Bar chart of metrics
    labels = ["RMSE\n(Straight)", "XTE mean\n(Straight)", "XTE max\n(Straight)",
              "RMSE\n(Curve)",   "Fréchet\n(Curve)"]
    values = [rmse_s, xte_m, xte_M, rmse_c, fre_c]
    plt.figure(figsize=(7,4))
    bars = plt.bar(labels, values, color=['#4C78A8']*3 + ['#F58518']*2)
    plt.title("Straight vs Curve Metrics")
    plt.ylabel("Error [m]")
    plt.ylim(0, np.nanmax(values)*1.2)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}", ha='center')
    plt.tight_layout()
    plt.show()
