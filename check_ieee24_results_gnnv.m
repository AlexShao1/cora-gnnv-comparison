function check_ieee24_results_gnnv(model_folder)
%CHECK_IEEE24_RESULTS_GNNV  Analyze and visualize IEEE24 verification results
%   Similar to check_cora_results_gnnv but adapted for graph classification
%
% Expects .mat files from reach_ieee24_gnnv containing:
%   - Rcell: cell array of ImageStar (2-class logits per graph)
%   - labels: 1-based true labels (1=stable, 2=unstable)
%   - preds: predicted labels at ε=0
%   - isRobust: 1 if verified, 0 otherwise
%   - tGraph: per-graph runtime [s]
%   - eps: perturbation radius

%% Locate result files
resDir = fullfile(model_folder, 'results', 'ieee24_gnnv');
files  = dir(fullfile(resDir, 'graph_eps*.mat'));
if isempty(files)
    fprintf('No reach sets found in %s\n', resDir);
    return
end

fprintf('\n=== IEEE24 VERIFICATION SUMMARY ===\n');

%% Containers for plotting
epsVals = [];
robPct  = [];
accPct  = [];
timeCDF = {};

%% Loop over epsilon files
for f = files'
    D = load(fullfile(f.folder, f.name));
    
    Rcell    = D.Rcell;
    labels   = D.labels(:);
    preds    = D.preds(:);
    isRobust = D.isRobust(:);
    eps_val  = D.eps;
    n_graphs = numel(labels);
    
    % Accuracy at ε=0
    accuracy = mean(preds == labels) * 100;
    
    % Certified percentage
    certified = mean(isRobust) * 100;
    
    % Per-class breakdown
    stable_idx = (labels == 1);
    unstable_idx = (labels == 2);
    certified_stable = mean(isRobust(stable_idx)) * 100;
    certified_unstable = mean(isRobust(unstable_idx)) * 100;
    
    % Timing
    if isfield(D, 'tGraph')
        total_time = sum(D.tGraph);
        avg_time = mean(D.tGraph);
        
        % CDF data
        [tSorted, idx] = sort(D.tGraph(:));
        rSorted = isRobust(idx);
        cumTime = cumsum(tSorted);
        cumVerified = cumsum(rSorted) / n_graphs * 100;
        thisCDF = struct('eps', eps_val, 'cumTime', cumTime, 'cumVerified', cumVerified);
        timeCDF{end+1} = thisCDF; %#ok<AGROW>
    else
        total_time = NaN;
        avg_time = NaN;
    end
    
    % Console report
    fprintf('\n[%s]  ε = %.4f\n', f.name, eps_val);
    fprintf('  Accuracy (ε=0)  : %.2f%%\n', accuracy);
    fprintf('  Certified total : %4d / %4d  (%.2f%%)\n', ...
            sum(isRobust), n_graphs, certified);
    fprintf('    ├─ stable     : %.2f%%  (%d / %d)\n', ...
            certified_stable, sum(isRobust(stable_idx)), sum(stable_idx));
    fprintf('    └─ unstable   : %.2f%%  (%d / %d)\n', ...
            certified_unstable, sum(isRobust(unstable_idx)), sum(unstable_idx));
    if ~isnan(total_time)
        fprintf('  Time            : %.1f s  (avg %.3f s/graph)\n', ...
                total_time, avg_time);
    end
    
    % Store for plots
    epsVals(end+1) = eps_val; %#ok<AGROW>
    robPct(end+1)  = certified; %#ok<AGROW>
    accPct(end+1)  = accuracy; %#ok<AGROW>
end

%% PLOT 1: Certified accuracy vs epsilon
[epsVals, ord] = sort(epsVals);
robPct = robPct(ord);
accPct = accPct(ord);

figure('Name', 'IEEE24 Certification vs Epsilon'); clf; hold on
plot(epsVals, robPct, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
     'Color', [0.2 0.4 0.8], 'MarkerFaceColor', [0.2 0.4 0.8], ...
     'DisplayName', 'Certified');
plot(epsVals, accPct, '--s', 'LineWidth', 1.5, 'MarkerSize', 8, ...
     'Color', [0.8 0.4 0.2], 'MarkerFaceColor', [0.8 0.4 0.2], ...
     'DisplayName', 'Clean Accuracy');
xlabel('\epsilon (L_\infty radius)', 'FontSize', 12);
ylabel('Percentage [%]', 'FontSize', 12);
title('IEEE24 Cascading Failure - Certified Robustness', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 11);
grid on; box on;
ylim([0 105]);

%% PLOT 2: Time-coverage CDF
if ~isempty(timeCDF)
    figure('Name', 'IEEE24 Cost-Coverage Tradeoff'); clf; hold on
    cmap = lines(numel(timeCDF));
    for k = 1:numel(timeCDF)
        C = timeCDF{k};
        if isempty(C.cumTime), continue; end
        stairs(C.cumTime, C.cumVerified, ...
               'Color', cmap(k,:), 'LineWidth', 1.8, ...
               'DisplayName', sprintf('\\epsilon = %.3g', C.eps));
    end
    xlabel('Cumulative Time [s]', 'FontSize', 12);
    ylabel('Verified Graphs [%]', 'FontSize', 12);
    title('Cost-Coverage Tradeoff', 'FontSize', 14);
    legend('Location', 'southeast', 'FontSize', 11);
    grid on; box on;
end

%% PLOT 3: Robustness by class
figure('Name', 'IEEE24 Per-Class Certification'); clf;
class_names = {'Stable', 'Unstable'};
n_eps = numel(epsVals);
stable_cert = zeros(1, n_eps);
unstable_cert = zeros(1, n_eps);

for i = 1:n_eps
    fname = fullfile(resDir, sprintf('graph_eps%.4f.mat', epsVals(i)));
    if ~exist(fname, 'file'), continue; end
    D = load(fname);
    stable_idx = (D.labels == 1);
    unstable_idx = (D.labels == 2);
    stable_cert(i) = mean(D.isRobust(stable_idx)) * 100;
    unstable_cert(i) = mean(D.isRobust(unstable_idx)) * 100;
end

plot(epsVals, stable_cert, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
     'Color', [0.2 0.7 0.3], 'DisplayName', 'Stable');
hold on;
plot(epsVals, unstable_cert, '-s', 'LineWidth', 2, 'MarkerSize', 8, ...
     'Color', [0.9 0.3 0.3], 'DisplayName', 'Unstable');
xlabel('\epsilon', 'FontSize', 12);
ylabel('Certified [%]', 'FontSize', 12);
title('Per-Class Certification Rate', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 11);
grid on; box on;

fprintf('\n=== ANALYSIS COMPLETE ===\n');
end