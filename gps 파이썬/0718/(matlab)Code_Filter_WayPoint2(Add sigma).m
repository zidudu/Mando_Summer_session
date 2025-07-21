%% filter_curve_and_make_waypoints_onefig.m
% -------------------------------------------------------------------------
% ● 원본 CSV 궤적을 (곡선: B‑스플라인 / 직선: 선형 회귀)로 스무딩합니다.
% ● 곡률 기반으로 동적 간격 웨이포인트를 생성합니다.
% ● Figure(1×3)을 그려서 ① Raw vs Filtered ② Waypoints ③ Mean/RMSE/Range/Std
%   막대그래프를 표시합니다.
% ● 최종 웨이포인트를 CSV로 저장합니다.
% -------------------------------------------------------------------------
% 작성일 : 2025‑07‑21
% 작성자 : ChatGPT (MATLAB 변환)
% -------------------------------------------------------------------------
% 사용 방법
%   ▶ MATLAB에서 본 스크립트를 실행하십시오. CSV_IN 변수를 수정하시면 됩니다.
%   ▶ 기타 파라미터는 필요에 따라 조정하십시오.
% -------------------------------------------------------------------------

clear; close all; clc;

%% ── 파라미터 설정 ─────────────────────────────────────────────
CSV_IN  = "raw_track_xy_18.csv"; % 기본 CSV (첫 열 X[m] / 두 번째 열 Y[m])
CURV_TH = 0.015;   % 곡률 임계값 (직선/곡선 구분)
MIN_LEN = 5;       % 세그먼트 최소 샘플 개수
SMOOTH_S = 0.6;    % 곡선 스플라인 스무딩 인자 (0~1, 1=보간, 0=평균)

% 곡률 기반 웨이포인트 간격 파라미터
K1 = 0.10; K2 = 0.20;           % 곡률 임계값 2단계
D_S = 0.55; D_M = 0.40; D_C = 0.28;  % 직선/중간/곡선 웨이포인트 간 거리 [m]

VIEW_R = 5.0;      % 시각화 한계 [m]
%% ── 데이터 로드 및 상대 좌표 변환 ─────────────────────────────
try
    data = readmatrix(CSV_IN);
catch
    error("CSV 파일을 찾을 수 없습니다: %s", CSV_IN);
end
X = data(:,1); Y = data(:,2);

xr = X - X(1); % 첫 점을 원점으로 이동
yr = Y - Y(1);

%% ── 트랙 스무딩 (곡선/직선 분리) ───────────────────────────────
[xf, yf] = filter_track(xr, yr, CURV_TH, MIN_LEN, SMOOTH_S);

%% ── 동적 웨이포인트 생성 ─────────────────────────────────────
WPT = dyn_wp(xf, yf, K1, K2, D_S, D_M, D_C);

%% ── 오차 계산 (세그먼트별) ───────────────────────────────────
err_raw = seg_errors(xr, yr, CURV_TH, MIN_LEN, SMOOTH_S);
err_flt = seg_errors(xf, yf, CURV_TH, MIN_LEN, SMOOTH_S);

[m_r, rmse_r, rng_r, std_r] = stats(err_raw);
[m_f, rmse_f, rng_f, std_f] = stats(err_flt);

%% ── 결과 Figure(1×3) 그리기 ─────────────────────────────────
figure('Name', 'Filter & Waypoints', 'Units', 'pixels', 'Position', [100 100 1400 450]);

% (1) Raw vs Filtered
subplot(1,3,1);
plot(xr, yr, '.', 'Color', [0.4 0.4 0.4], 'DisplayName', 'Raw'); hold on;
plot(xf, yf, 'r-', 'LineWidth', 1.2, 'DisplayName', 'Filtered');
axis equal; xlim([-VIEW_R VIEW_R]); ylim([-VIEW_R VIEW_R]);
xlabel('\DeltaX [m]'); ylabel('\DeltaY [m]'); title('Raw vs Filtered'); grid on;
legend('Location','best');

% (2) Dynamic Waypoints
subplot(1,3,2);
plot(xf, yf, 'k-', 'LineWidth',0.8, 'Color',[0 0 0 0.3]); hold on;
scatter(WPT(:,1), WPT(:,2), 36, 'MarkerEdgeColor','k', ...
        'MarkerFaceColor',[1 0.5 0], 'DisplayName', sprintf('Waypoints (%d)', size(WPT,1)));
axis equal; xlim([-VIEW_R VIEW_R]); ylim([-VIEW_R VIEW_R]);
xlabel('\DeltaX [m]'); ylabel('\DeltaY [m]'); title('Dynamic Waypoints'); grid on;
legend('Location','best');

% (3) Error Statistics
subplot(1,3,3);
labels = {'Mean','RMSE','Range','Std(\sigma)'}; x = 1:numel(labels);
barWidth = 0.35;
bar(x - barWidth/2, [m_r, rmse_r, rng_r, std_r], barWidth, 'FaceColor',[0.35 0.7 0.9], 'DisplayName','Raw'); hold on;
bar(x + barWidth/2, [m_f, rmse_f, rng_f, std_f], barWidth, 'FaceColor',[1 0.5 0.5], 'DisplayName','Filtered');
set(gca, 'XTick', x, 'XTickLabel', labels);
ylabel('[m]'); title('Error Statistics'); grid on; legend('Location','best');

% 값 라벨 표시
text(x - barWidth/2, [m_r, rmse_r, rng_r, std_r] + 1e-4, composed(num2str([m_r; rmse_r; rng_r; std_r]', '%.3f\n')), 'HorizontalAlignment','center');
text(x + barWidth/2, [m_f, rmse_f, rng_f, std_f] + 1e-4, composed(num2str([m_f; rmse_f; rng_f; std_f]', '%.3f\n')), 'HorizontalAlignment','center');

sgtitle('Track Filtering & Waypoint Generation');

%% ── 웨이포인트 CSV 저장 ─────────────────────────────────────
wp_file = unique_filename('waypoints_dynamic', '.csv');
writematrix(WPT, wp_file, 'Delimiter', ',');

%% ── 콘솔 출력 ------------------------------------------------
fprintf('\n─── 오차 통계 ───\n');
fprintf('[Raw]      mean = %.4f m, RMSE = %.4f m, range = %.4f m, std = %.4f m\n', m_r, rmse_r, rng_r, std_r);
fprintf('[Filtered] mean = %.4f m, RMSE = %.4f m, range = %.4f m, std = %.4f m\n', m_f, rmse_f, rng_f, std_f);
fprintf('\n[Waypoints] 저장 완료 → %s (총 %d개)\n\n', wp_file, size(WPT,1));

%% ====================== 함수 정의 ==========================
function kappa = curvature(x, y)
    % 곡률 κ 계산 (2차 미분 기반)
    dx  = gradient(x); dy  = gradient(y);
    ddx = gradient(dx); ddy = gradient(dy);
    num = dx .* ddy - dy .* ddx;
    den = (dx.^2 + dy.^2).^1.5;
    kappa = zeros(size(x));
    mask = den > 1e-12;
    kappa(mask) = abs(num(mask) ./ den(mask));
end

function [segs, types] = segments(kappa, CURV_TH, MIN_LEN)
    % 곡률을 이용해 직선/곡선 세그먼트 구분
    segs  = []; types = {};
    i = 1; N = numel(kappa);
    while i <= N
        mode = "line";
        if kappa(i) >= CURV_TH
            mode = "curve";
        end
        j = i + 1;
        while j <= N && ((mode == "line" && kappa(j) < CURV_TH) || (mode == "curve" && kappa(j) >= CURV_TH))
            j = j + 1;
        end
        if (j - i) >= MIN_LEN
            segs  = [segs; i, j-1]; %#ok<AGROW>
            types = [types; {mode}]; %#ok<AGROW>
        end
        i = j;
    end
end

function [xs_s, ys_s] = smooth_curve(xs, ys, SMOOTH_S)
    % 곡선 영역 스플라인 스무딩 → 동일 길이로 반환
    n = numel(xs);
    if n < 2
        xs_s = xs; ys_s = ys; return;
    end
    % 매개변수 u를 0~1로 정규화 (호 길이 기준)
    d = hypot(diff(xs), diff(ys));
    u = [0; cumsum(d)] / sum(d);
    % csaps (Cubic smoothing spline)
    pp_x = csaps(u, xs, SMOOTH_S);
    pp_y = csaps(u, ys, SMOOTH_S);
    uu = linspace(0, 1, n);
    xs_s = fnval(pp_x, uu)';
    ys_s = fnval(pp_y, uu)';
end

function [xs_s, ys_s] = smooth_line_regress(xs, ys)
    % 직선 영역 선형 회귀로 스무딩
    if numel(xs) < 2
        xs_s = xs; ys_s = ys; return;
    end
    p = polyfit(xs, ys, 1);
    ys_s = polyval(p, xs);
    xs_s = xs;
end

function [xf, yf] = filter_track(xr, yr, CURV_TH, MIN_LEN, SMOOTH_S)
    % 전체 트랙을 세그먼트별로 필터링 (곡선/직선)
    kappa = curvature(xr, yr);
    [segs, types] = segments(kappa, CURV_TH, MIN_LEN);
    xf = xr; yf = yr;
    for s = 1:size(segs,1)
        i0 = segs(s,1); i1 = segs(s,2);
        idx = i0:i1;
        if types{s} == "curve"
            [xs_s, ys_s] = smooth_curve(xr(idx), yr(idx), SMOOTH_S);
        else
            [xs_s, ys_s] = smooth_line_regress(xr(idx), yr(idx));
        end
        xf(idx) = xs_s; yf(idx) = ys_s;
    end
end

function WPT = dyn_wp(x, y, K1, K2, D_S, D_M, D_C)
    % 곡률에 따라 동적 간격으로 웨이포인트 생성
    kap = curvature(x, y);
    WPT = [x(1), y(1)];
    acc = 0.0;
    for i = 2:numel(x)
        seg = hypot(x(i) - x(i-1), y(i) - y(i-1));
        acc = acc + seg;
        if kap(i) < K1
            d_t = D_S;
        elseif kap(i) < K2
            d_t = D_M;
        else
            d_t = D_C;
        end
        while acc >= d_t
            r = (seg - (acc - d_t)) / seg;
            new_pt = [x(i-1) + (x(i) - x(i-1))*r, y(i-1) + (y(i) - y(i-1))*r];
            WPT = [WPT; new_pt]; %#ok<AGROW>
            acc = acc - d_t;
        end
    end
    WPT = [WPT; x(end), y(end)];
end

function err = seg_errors(x, y, CURV_TH, MIN_LEN, SMOOTH_S)
    % 세그먼트별 편차 계산 (직선: 수직거리, 곡선: 절대거리)
    kappa = curvature(x, y);
    [segs, types] = segments(kappa, CURV_TH, MIN_LEN);
    err = [];
    for s = 1:size(segs,1)
        i0 = segs(s,1); i1 = segs(s,2);
        idx = i0:i1;
        xs = x(idx); ys = y(idx);
        if types{s} == "line"
            p = polyfit(xs, ys, 1);
            % 수직거리 공식 |ax+by+c| / sqrt(a^2+b^2)
            a = p(1); b = -1; c = p(2);
            d = abs(a*xs + b*ys + c) ./ hypot(a, b);
        else
            [xs_s, ys_s] = smooth_curve(xs, ys, SMOOTH_S);
            d = hypot(xs - xs_s, ys - ys_s);
        end
        err = [err; d]; %#ok<AGROW>
    end
end

function [m, rmse_val, rng, sd] = stats(arr)
    % 기본 통계 계산
    if isempty(arr)
        m = NaN; rmse_val = NaN; rng = NaN; sd = NaN; return;
    end
    m = mean(arr);
    rmse_val = sqrt(mean(arr.^2));
    rng = max(arr) - min(arr);
    sd = std(arr);
end

function fname = unique_filename(base, ext)
    % 같은 이름이 있으면 _1, _2 … 자동 인덱싱
    idx = 0;
    while true
        if idx == 0
            candidate = sprintf('%s%s', base, ext);
        else
            candidate = sprintf('%s_%d%s', base, idx, ext);
        end
        if ~isfile(candidate)
            fname = candidate; return;
        end
        idx = idx + 1;
    end
end
