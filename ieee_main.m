function completed = ieee_main()
%IEEE_MAIN  Main verification pipeline for IEEE24 cascading failure GNN
%   Runs GNNV-based verification on IEEE24 dataset and generates plots
%
% Pipeline:
%   1. Load IEEE24 GCN model + data (24 nodes, 3 features, binary classification)
%   2. Run reach_ieee24_gnnv for multiple ε perturbations
%   3. Analyze results with check_ieee24_results_gnnv
%   4. Save all outputs to timestamped results folder
%
% Outputs:
%   results/<timestamp>/
%       ├── ieee24_gnnv/graph_epsXXXX.mat  (reach sets)
%       ├── plots/*.fig, *.png              (certification curves)
%       └── results.txt                     (console log)

%% 0) Setup paths
NNV_ROOT = '/Users/alexshao/Desktop/verification/nnv';  % ← adjust to your NNV path
addpath(genpath(NNV_ROOT), '-begin');
addpath(genpath(pwd));

% Start parallel pool if not running (optional, can speed up large ε sweeps)
if isempty(gcp('nocreate'))
    parpool('local', 4);  % adjust worker count as needed
end

%% 1) Paths
basepath   = '.';
model_dir  = '/Users/alexshao/Desktop/Cora Verify/IEEE24';  % ← your IEEE24 export folder
resultpath = sprintf('%s/results/%s', basepath, datestr(datetime,'yymmdd-hhMMss'));
mkdir(resultpath);
evalpath   = sprintf('%s/evaluation', resultpath); mkdir(evalpath);
plotpath   = sprintf('%s/plots', resultpath);      mkdir(plotpath);

set(0, 'defaultFigureRenderer', 'painters');
resultstxt = sprintf('%s/results.txt', resultpath);
if exist(resultstxt,'file'), delete(resultstxt); end
diary(resultstxt);

%% 2) Header
disp('========================================================');
disp('IEEE24 Cascading Failure GNN Verification (GNNV/NNV)');
fprintf('Date: %s\n', datestr(datetime()));
try
    disp(CORAVERSION);  % if you have CORA loaded
catch
    fprintf('NNV version: %s\n', nnv_version());  % or NNV version check
end
disp('========================================================');
disp(' ');

%% 3) Verification parameters
epsList = [0, 0.005];  % ← adjust as needed
fprintf('Running verification for ε = [%s]\n', num2str(epsList));
fprintf('Model folder: %s\n\n', model_dir);

%% 4) Run verification
try
    fprintf('--- Step 1: Computing reach sets ---\n');
    %reach_ieee24_gnnv(model_dir, epsList);
    % Test on first 10 graphs only
    reach_ieee24_gnnv(model_dir, epsList, 'num_graphs', 20);
    fprintf('✅ Reach sets computed successfully!\n\n');
catch ME
    fprintf('❌ ERROR during reach computation:\n');
    disp(ME.getReport());
    diary off;
    completed = 0;
    return;
end

%% 5) Analyze results
try
    fprintf('--- Step 2: Analyzing results ---\n');
    check_ieee24_results_gnnv(model_dir);
    fprintf('✅ Analysis complete!\n\n');
catch ME
    fprintf('❌ ERROR during analysis:\n');
    disp(ME.getReport());
    diary off;
    completed = 0;
    return;
end

%% 6) Save plots
fprintf('--- Step 3: Saving plots ---\n');
h = findobj('type', 'figure');
for j = 1:length(h)
    figname = sprintf('ieee24_verification_%d', j);
    savefig(h(j), fullfile(plotpath, [figname '.fig']));
    saveas(h(j), fullfile(plotpath, [figname '.png']));
    fprintf('  Saved: %s\n', figname);
end
close all;

%% 7) Move reach set results to organized location
src_dir = fullfile(model_dir, 'results', 'ieee24_gnnv');
if exist(src_dir, 'dir')
    copyfile(src_dir, fullfile(evalpath, 'ieee24_gnnv'));
    fprintf('  Copied reach sets to: %s\n', evalpath);
end

%% 8) Summary
disp(' ');
disp('========================================================');
disp('Verification Complete!');
fprintf('Date: %s\n', datestr(datetime()));
fprintf('Results saved to: %s\n', resultpath);
fprintf('  - Reach sets:  %s/evaluation/ieee24_gnnv/\n', resultpath);
fprintf('  - Plots:       %s/plots/\n', resultpath);
fprintf('  - Console log: %s/results.txt\n', resultpath);
disp('========================================================');

diary off;
completed = 1;
end