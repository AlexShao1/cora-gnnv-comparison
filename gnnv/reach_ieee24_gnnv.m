function reach_ieee24_gnnv(model_folder, epsList, varargin)
% REACH_IEEE24_GNNV — Graph-level L_inf certification on IEEE24 with GNNV/NNV
% Model: 3 GCN layers -> global_max_pool -> linear head (2 logits)
%
% Inputs
%   model_folder : folder with model and data exports
%   epsList      : vector of ε radii (e.g., [0 0.01 0.03])
%   varargin     : optional parameters
%     'num_graphs'   : number of graphs to process (default: all)
%     'graph_indices': specific indices to process (default: 1:num_graphs)
%
% Example usage:
%   reach_ieee24_gnnv(folder, [0 0.01], 'num_graphs', 200)  % First 200 graphs
%   reach_ieee24_gnnv(folder, [0 0.01], 'graph_indices', 1:10:2001)  % Every 10th graph

%% Parse optional arguments
p = inputParser;
addParameter(p, 'num_graphs', [], @isnumeric);
addParameter(p, 'graph_indices', [], @isnumeric);
parse(p, varargin{:});

%% 0) Load model (3 GCN + max pool + linear head)
mdl    = jsondecode(fileread(fullfile(model_folder,'ieee24_cascading_model_export.json')));
layers = mdl.layers;

% Expect 4 layers total: 3 GCNs (1,2,3) + linear head (4)
W1 = double(layers(1).W); b1 = double(layers(1).b(:));  act1 = get_act(layers(1));
W2 = double(layers(2).W); b2 = double(layers(2).b(:));  act2 = get_act(layers(2));
W3 = double(layers(3).W); b3 = double(layers(3).b(:));  act3 = get_act(layers(3));
W_out = double(layers(4).W); b_out = double(layers(4).b(:));

%% 1) Load batched graph data
tbl  = neuralNetwork.readGNNdata(fullfile(model_folder,'ieee24_cascading_data_export.json'));

Xall = tbl.input{1};                 % N x 24 x 3 (nodes x feats per graph)
Ngraphs_total = size(Xall, 1);

% Determine which graphs to process
if ~isempty(p.Results.graph_indices)
    graph_indices = p.Results.graph_indices;
    graph_indices = graph_indices(graph_indices <= Ngraphs_total);
elseif ~isempty(p.Results.num_graphs)
    num_to_process = min(p.Results.num_graphs, Ngraphs_total);
    graph_indices = 1:num_to_process;
else
    graph_indices = 1:Ngraphs_total;
end

Ngraphs = length(graph_indices);
fprintf('Processing %d / %d graphs (indices: %d to %d)\n', ...
    Ngraphs, Ngraphs_total, graph_indices(1), graph_indices(end));

% Python outputs (ensure N x 2)
Ypy = double(tbl.output{1});
if size(Ypy,2) ~= 2 && size(Ypy,1) == 2, Ypy = Ypy.'; end
assert(size(Ypy,2) == 2, 'Unexpected output shape; expected N×2.');

% Ground-truth labels (N×1 or N×2)
Traw = tbl.target_label{1};
if iscell(Traw), Traw = cell2mat(Traw); end
Traw = double(Traw);
if size(Traw,2) == 1
    y_true = 1 + (Traw(:) > 0.5);          % 0/1 -> 1/2
elseif size(Traw,2) == 2
    [~, y_true] = max(Traw, [], 2);        % one-hot -> 1/2
else
    error('Unexpected target_label shape: %s', mat2str(size(Traw)));
end
y_true = y_true(:);

% Python predictions from export
[~, py_pred] = max(Ypy, [], 2);

fprintf('Dataset: %d total graphs\n', Ngraphs_total);
fprintf('Python-export preds (full set): stable=%d (%.2f%%), unstable=%d (%.2f%%)\n', ...
    sum(py_pred==1), 100*mean(py_pred==1), sum(py_pred==2), 100*mean(py_pred==2));
fprintf('Ground truth (full set):        stable=%d, unstable=%d\n', ...
    sum(y_true==1), sum(y_true==2));

% Subset statistics
py_pred_subset = py_pred(graph_indices);
y_true_subset = y_true(graph_indices);
fprintf('Subset preds: stable=%d (%.2f%%), unstable=%d (%.2f%%)\n', ...
    sum(py_pred_subset==1), 100*mean(py_pred_subset==1), ...
    sum(py_pred_subset==2), 100*mean(py_pred_subset==2));
fprintf('Subset truth: stable=%d, unstable=%d\n\n', ...
    sum(y_true_subset==1), sum(y_true_subset==2));

%% 1.1 Edges — hardcode IEEE-24 undirected template (0-based)
use_dataset_edges = false;

if use_dataset_edges && ismember('edge_index', tbl.Properties.VariableNames) && ~isempty(tbl.edge_index{1})
    Eall = tbl.edge_index{1};
    fprintf('Using edge_index from dataset JSON.\n');
else
    E_template = [
        0,1; 0,2; 0,4; 1,3; 1,5; 2,8; 2,23; 3,8; 4,9; 5,9;
        6,7; 7,8; 7,9; 8,10; 8,11; 9,10; 9,11; 10,12; 10,13; 11,12;
        11,22; 12,22; 13,15; 14,15; 14,20; 14,20; 14,23; 15,16; 15,18; 16,17;
        16,21; 17,20; 17,20; 18,19; 18,19; 19,22; 19,22; 20,21
    ]';
    fprintf('Using hardcoded IEEE24 topology: %d edges\n', size(E_template,2));
end

%% 2) Setup
relu   = ReluLayer();
method = 'exact-star';
outDir = fullfile(model_folder,'results','ieee24_gnnv');
if ~exist(outDir,'dir'), mkdir(outDir); end

%% 3) Loop over ε
for eps = epsList(:).'
    fprintf('=== IEEE24: ε = %.4f (processing %d graphs) ===\n', eps, Ngraphs);

    Rcell    = cell(Ngraphs,1);
    labels   = zeros(Ngraphs,1);
    preds    = zeros(Ngraphs,1);
    isRobust = zeros(Ngraphs,1);
    tGraph   = zeros(Ngraphs,1);
    actual_indices = zeros(Ngraphs,1);  % Track which graph indices were processed

    for idx = 1:Ngraphs
        g = graph_indices(idx);  % Actual graph index in full dataset
        actual_indices(idx) = g;
        t0 = tic;

        % Features (24×3)
        X_g = squeeze(Xall(g, :, :));
        n_nodes = size(X_g, 1);
        n_feat  = size(X_g, 2);
        labels(idx) = y_true(g);

        % Edges
        if use_dataset_edges && exist('Eall','var')
            E_g = squeeze(Eall(g, :, :));
        else
            E_g = E_template;
        end
        
        % Build adjacency
        src = double(E_g(1,:)) + 1;
        dst = double(E_g(2,:)) + 1;
        n   = n_nodes;
        
        src_ud = [src, dst, 1:n];
        dst_ud = [dst, src, 1:n];
        w      = ones(1, numel(src_ud));
        
        A = sparse(src_ud, dst_ud, w, n, n);
        
        deg     = full(sum(A, 2));
        deg(deg==0) = 1;
        Dinv2   = spdiags(1./sqrt(deg), 0, n, n);
        Ahat    = Dinv2 * A * Dinv2;

        % Input ImageStar
        V = zeros(n_feat, n_nodes, 1, n_nodes+1);
        V(:,:,1,1) = X_g.';
        for p = 1:n_nodes
            V(:,p,1,p+1) = 1;
        end
        C = [eye(n_nodes); -eye(n_nodes)];
        d = eps * ones(2*n_nodes,1);
        Xstar = ImageStar(V, C, d, -eps*ones(n_nodes,1), eps*ones(n_nodes,1));
       
        % Nominal forward pass
        X0 = X_g.';
        H1c = (W1 * X0) * Ahat + b1;
        if strcmp(act1,'relu'), H1c = max(H1c, 0); end
        
        H2c = (W2 * H1c) * Ahat + b2;
        if strcmp(act2,'relu'), H2c = max(H2c, 0); end
        
        H3c = (W3 * H2c) * Ahat + b3;
        
        hpool0 = mean(H3c, 2);
        z0_nom = W_out * hpool0 + b_out;
        [~, preds(idx)] = max(z0_nom);
        
        % Debug for first few
        if idx <= 3 || (idx >= 20 && idx < 23)
            fprintf('[DBG idx=%d, g=%d, eps=%.4f] z0_nom=[%.4f, %.4f], pred=%d, lbl=%d\n', ...
                idx, g, eps, z0_nom(1), z0_nom(2), preds(idx), labels(idx));
        end

        try
            % Reachability analysis
            H1 = gconv_ieee24(Xstar, W1, b1, Ahat);
            if strcmp(act1,'relu'), H1 = relu.reach(H1, method); end

            H2 = gconv_ieee24(H1, W2, b2, Ahat);
            if strcmp(act2,'relu'), H2 = relu.reach(H2, method); end

            H3 = gconv_ieee24(H2, W3, b3, Ahat);

            [lb, ub] = H3.getRanges();
            n_hidden = size(W3,1);
            assert(numel(lb) == n_hidden*n_nodes, 'Range length mismatch for pooling.');
            lb = reshape(lb, [n_hidden, n_nodes]);
            ub = reshape(ub, [n_hidden, n_nodes]);
            lb_pool = mean(lb, 2);
            ub_pool = mean(ub, 2);

            H_pooled = Star(lb_pool, ub_pool);
            Y = H_pooled.affineMap(W_out, b_out);

            if isa(Y, 'ImageStar'), Yc = Y.toStar(); else, Yc = Y; end
            z0 = double(Yc.V(:,1));
            % [~, preds(idx)] = max(z0);

            Hs = label2Hs_binary(labels(idx));
            vflag = verify_specification(Yc, Hs);
            isRobust(idx) = double(vflag == 1);

            if idx <= 3 || (idx >= 20 && idx < 23)
                fprintf('[DBG idx=%d, g=%d, eps=%.4f] z0=[%.4f, %.4f], pred=%d, lbl=%d, v=%d\n', ...
                    idx, g, eps, z0(1), z0(2), preds(idx), labels(idx), vflag);
            end

        catch ME
            fprintf(' ERROR at idx %d (graph %d): %s\n', idx, g, ME.message);
            isRobust(idx) = 0;
        end

        tGraph(idx) = toc(t0);
        if mod(idx, 50) == 0
            fprintf('  Processed %d / %d graphs (%.1f%% complete, avg %.2fs/graph)\n', ...
                idx, Ngraphs, 100*idx/Ngraphs, mean(tGraph(1:idx)));
        end
    end

    % Save with subset indicator
    fname = fullfile(outDir, sprintf('graph_eps%.4f_subset_%d.mat', eps, Ngraphs));
    save(fname, 'Rcell','labels','preds','isRobust','tGraph','eps', ...
         'actual_indices', 'graph_indices', '-v7.3');
    fprintf('✅  Saved: %s\n', fname);

    % Summary for subset
    acc = mean(preds == labels) * 100;
    rob = mean(isRobust == 1) * 100;

    fprintf('\nMATLAB preds (subset): stable=%d (%.2f%%), unstable=%d (%.2f%%)\n', ...
        sum(preds==1), 100*mean(preds==1), sum(preds==2), 100*mean(preds==2));

    fprintf('Python preds (subset): stable=%d (%.2f%%), unstable=%d (%.2f%%)\n', ...
        sum(py_pred_subset==1), 100*mean(py_pred_subset==1), ...
        sum(py_pred_subset==2), 100*mean(py_pred_subset==2));

    match_rate = mean(preds(:) == py_pred_subset(:)) * 100;
    fprintf('Match rate vs Python (subset): %.2f%%\n', match_rate);

    if eps == 0
        tn = sum(preds==1 & labels==1); tp = sum(preds==2 & labels==2);
        fprintf('Per-class correct @ε=0: stable=%d/%d (%.2f%%), unstable=%d/%d (%.2f%%)\n', ...
            tn, sum(labels==1), 100*tn/max(1,sum(labels==1)), ...
            tp, sum(labels==2), 100*tp/max(1,sum(labels==2)));
    end

    fprintf('  Accuracy: %.2f%%  |  Verified: %.2f%%  |  Time: %.1fs (avg %.2fs/graph)\n\n', ...
        acc, rob, sum(tGraph), mean(tGraph));
end
end

%% [Helper functions remain the same]
function Z = gconv_ieee24(X, W, b, Ahat)
V = X.V; [~, n, ~, P] = size(V);
fout = size(W,1);
Vnew = zeros(fout, n, 1, P);
for q = 1:P
    H = squeeze(V(:,:,1,q));
    Hlin = W * H;
    Hagg = Hlin * Ahat;
    Vnew(:,:,1,q) = Hagg + b;
end
Z = ImageStar(Vnew, X.C, X.d, X.pred_lb, X.pred_ub);
end

function a = get_act(L)
a = ""; 
if isfield(L,'act') && ~isempty(L.act), a = string(L.act); end
end

function Hs = label2Hs_binary(lbl)
if lbl == 1
    Hs = HalfSpace([ -1 1], 0);
else
    Hs = HalfSpace([1 -1], 0);
end
end


% function reach_ieee24_gnnv(model_folder, epsList)
% % REACH_IEEE24_GNNV — Graph-level L_inf certification on IEEE24 with GNNV/NNV
% % Model: 3 GCN layers -> global_max_pool -> linear head (2 logits)
% %
% % Inputs
% %   model_folder : folder with
% %       - ieee24_cascading_model_export.json
% %       - ieee24_cascading_data_export.json
% %   epsList      : vector of ε radii (e.g., [0 0.01 0.03])
% %
% % Outputs (per ε)
% %   <model_folder>/results/ieee24_gnnv/graph_epsXXXX.mat  (saved -v7.3)
% %     Rcell, labels, preds, isRobust, tGraph, eps
% 
% %% 0) Load model (3 GCN + max pool + linear head)
% mdl    = jsondecode(fileread(fullfile(model_folder,'ieee24_cascading_model_export.json')));
% layers = mdl.layers;
% 
% % Expect 4 layers total: 3 GCNs (1,2,3) + linear head (4)
% W1 = double(layers(1).W); b1 = double(layers(1).b(:));  act1 = get_act(layers(1));
% W2 = double(layers(2).W); b2 = double(layers(2).b(:));  act2 = get_act(layers(2));
% W3 = double(layers(3).W); b3 = double(layers(3).b(:));  act3 = get_act(layers(3));
% W_out = double(layers(4).W); b_out = double(layers(4).b(:));
% 
% %% 1) Load batched graph data
% tbl  = neuralNetwork.readGNNdata(fullfile(model_folder,'ieee24_cascading_data_export.json'));
% 
% Xall = tbl.input{1};                 % N x 24 x 3 (nodes x feats per graph)
% Ngraphs = size(Xall, 1);
% 
% % Python outputs (ensure N x 2)
% Ypy = double(tbl.output{1});
% if size(Ypy,2) ~= 2 && size(Ypy,1) == 2, Ypy = Ypy.'; end
% assert(size(Ypy,2) == 2, 'Unexpected output shape; expected N×2.');
% 
% % Ground-truth labels (N×1 or N×2)
% Traw = tbl.target_label{1};
% if iscell(Traw), Traw = cell2mat(Traw); end
% Traw = double(Traw);
% if size(Traw,2) == 1
%     y_true = 1 + (Traw(:) > 0.5);          % 0/1 -> 1/2
% elseif size(Traw,2) == 2
%     [~, y_true] = max(Traw, [], 2);        % one-hot -> 1/2
% else
%     error('Unexpected target_label shape: %s', mat2str(size(Traw)));
% end
% y_true = y_true(:);
% 
% % Python predictions from export
% [~, py_pred] = max(Ypy, [], 2);
% 
% fprintf('Loaded %d graphs (batched: N x 24 x 3)\n', Ngraphs);
% fprintf('Python-export preds: stable=%d (%.2f%%), unstable=%d (%.2f%%)\n', ...
%     sum(py_pred==1), 100*mean(py_pred==1), sum(py_pred==2), 100*mean(py_pred==2));
% fprintf('Ground truth:       stable=%d, unstable=%d\n\n', sum(y_true==1), sum(y_true==2));
% 
% %% 1.1 Edges — hardcode IEEE-24 undirected template (0-based)
% use_dataset_edges = false;  % << hard-code topology for IEEE24 (set true to use JSON edge_index if present)
% 
% if use_dataset_edges && ismember('edge_index', tbl.Properties.VariableNames) && ~isempty(tbl.edge_index{1})
%     Eall = tbl.edge_index{1};    % N x 2 x E (0-based)
%     fprintf('Using edge_index from dataset JSON.\n');
% else
%     % Template: (2×E) 0-based undirected lines (one copy per line)
%     % IEEE24 topology from blist.mat (0-indexed, includes duplicates for parallel lines)
%     E_template = [
%         0,1; 0,2; 0,4; 1,3; 1,5; 2,8; 2,23; 3,8; 4,9; 5,9;
%         6,7; 7,8; 7,9; 8,10; 8,11; 9,10; 9,11; 10,12; 10,13; 11,12;
%         11,22; 12,22; 13,15; 14,15; 14,20; 14,20; 14,23; 15,16; 15,18; 16,17;
%         16,21; 17,20; 17,20; 18,19; 18,19; 19,22; 19,22; 20,21
%     ]';  % Transpose to get 2×38% (2×E)
%     fprintf('Using hardcoded IEEE24 topology: %d edges\n', size(E_template,2));
% end
% 
% %% 2) Setup
% relu   = ReluLayer();
% method = 'approx-star';
% outDir = fullfile(model_folder,'results','ieee24_gnnv');
% if ~exist(outDir,'dir'), mkdir(outDir); end
% 
% %% 3) Loop over ε
% for eps = epsList(:).'
%     fprintf('=== IEEE24: ε = %.4f ===\n', eps);
% 
%     Rcell    = cell(Ngraphs,1);
%     labels   = zeros(Ngraphs,1);
%     preds    = zeros(Ngraphs,1);
%     isRobust = zeros(Ngraphs,1);
%     tGraph   = zeros(Ngraphs,1);
% 
%     for g = 1:Ngraphs
%         t0 = tic;
% 
%         % Features (24×3)
%         X_g = squeeze(Xall(g, :, :));       % nodes x feats
%         n_nodes = size(X_g, 1);             % 24
%         n_feat  = size(X_g, 2);             % 3
%         labels(g) = y_true(g);
% 
%         % Edges (0-based -> build binary, undirected A with exactly one self-loop per node)
%         if use_dataset_edges && exist('Eall','var')
%             E_g = squeeze(Eall(g, :, :));   % 2×E, 0-based
%         else
%             E_g = E_template;               % 2×E, 0-based
%         end
% 
%         % E_g: 0-based 2×E (with duplicates)
%         src = double(E_g(1,:)) + 1;   % 1-based
%         dst = double(E_g(2,:)) + 1;
%         n   = n_nodes;
% 
%         % Undirected + self-loops; weights = 1, duplicates accumulate
%         src_ud = [src, dst, 1:n];
%         dst_ud = [dst, src, 1:n];
%         w      = ones(1, numel(src_ud));
% 
%         A = sparse(src_ud, dst_ud, w, n, n);   % keep multiplicity; NO spones()
% 
%         % Symmetric GCN normalization: Â = D^{-1/2} A D^{-1/2}
%         deg     = full(sum(A, 2));
%         deg(deg==0) = 1;                        % guard
%         Dinv2   = spdiags(1./sqrt(deg), 0, n, n);
%         Ahat    = Dinv2 * A * Dinv2;
% 
% 
%         % Input ImageStar: features×nodes center, one generator per node (L_inf by node)
%         V = zeros(n_feat, n_nodes, 1, n_nodes+1);
%         V(:,:,1,1) = X_g.';                                 % center: feats×nodes
%         for p = 1:n_nodes
%             V(:,p,1,p+1) = 1;                               % one generator per node
%         end
%         C = [eye(n_nodes); -eye(n_nodes)];                  % |δ_p| ≤ ε
%         d = eps * ones(2*n_nodes,1);
%         Xstar = ImageStar(V, C, d, -eps*ones(n_nodes,1), eps*ones(n_nodes,1));
% 
%         X0 = X_g.';                        % feats×nodes
%         H1c = (W1 * X0) * Ahat + b1;       % linear -> aggregate -> bias
%         if strcmp(act1,'relu'), H1c = max(H1c, 0); end
% 
%         H2c = (W2 * H1c) * Ahat + b2;
%         if strcmp(act2,'relu'), H2c = max(H2c, 0); end
% 
%         H3c = (W3 * H2c) * Ahat + b3;      % last GCN (usually no ReLU if act3=="")
% 
%         hpool0 = max(H3c, [], 2);          % TRUE global max-pool on nominal features
%         z0_nom = W_out * hpool0 + b_out;   % 2×1 logits at ε=0
%         [~, preds(g)] = max(z0_nom);       % <-- use this for accuracy/match-rate
% 
%         % (Optional) debug print: use z0_nom here, not Y.V center
%         if g <= 3 || (g >= 100 && g < 103)
%             fprintf('[DBG g=%d, eps=%.4f] z0_nom=[%.4f, %.4f], pred=%d, lbl=%d\n', ...
%                 g, eps, z0_nom(1), z0_nom(2), preds(g), labels(g));
%         end
% 
% 
%         try
%             % -------- Forward: 3×(GCN) with ReLU on first two only --------
%             H1 = gconv_ieee24(Xstar, W1, b1, Ahat);         % linear -> aggregate -> bias
%             if strcmp(act1,'relu'), H1 = relu.reach(H1, method); end
% 
%             H2 = gconv_ieee24(H1, W2, b2, Ahat);
%             if strcmp(act2,'relu'), H2 = relu.reach(H2, method); end
% 
%             H3 = gconv_ieee24(H2, W3, b3, Ahat);
%             % last GCN typically no activation in export (act3 == "")
% 
%             % -------- Global MAX pooling (sound over-approx via bounds) --------
%             % Use ImageStar bounds directly (fast)
%             [lb, ub] = H3.getRanges();                      % (n_hidden*n_nodes)×1
%             n_hidden = size(W3,1);
%             assert(numel(lb) == n_hidden*n_nodes, 'Range length mismatch for pooling.');
%             lb = reshape(lb, [n_hidden, n_nodes]);
%             ub = reshape(ub, [n_hidden, n_nodes]);
%             lb_pool = max(lb, [], 2);                       % n_hidden×1
%             ub_pool = max(ub, [], 2);                       % n_hidden×1
% 
%             % Box Star for pooled features
%             H_pooled = Star(lb_pool, ub_pool);              % dim = n_hidden
% 
%             % -------- Final linear head --------
%             Y = H_pooled.affineMap(W_out, b_out);           % 2 logits (Star or ImageStar)
% 
%             % Center prediction (ε=0 center)
%             if isa(Y, 'ImageStar'), Yc = Y.toStar(); else, Yc = Y; end
%             z0 = double(Yc.V(:,1));                         % center
%             [~, preds(g)] = max(z0);
% 
%             % Verify spec (true logit ≥ other)
%             Hs = label2Hs_binary(labels(g));
%             vflag = verify_specification(Yc, Hs);
%             isRobust(g) = double(vflag == 1);
% 
%             % Lightweight debug on a few graphs
%             if g <= 3 || (g >= 100 && g < 103)
%                 fprintf('[DBG g=%d, eps=%.4f] z0=[%.4f, %.4f], pred=%d, lbl=%d, v=%d\n', ...
%                     g, eps, z0(1), z0(2), preds(g), labels(g), vflag);
%             end
% 
%         catch ME
%             fprintf('⚠️  ERROR at graph %d: %s\n', g, ME.message);
%             isRobust(g) = 0;
%         end
% 
%         tGraph(g) = toc(t0);
%         if mod(g, 100) == 0
%             fprintf('  Processed %d / %d graphs\n', g, Ngraphs);
%         end
%     end
% 
%     % Save (v7.3 avoids 2GB limit). NOTE: Rcell can be huge; keep only if needed.
%     fname = fullfile(outDir, sprintf('graph_eps%.4f.mat', eps));
%     save(fname, 'Rcell','labels','preds','isRobust','tGraph','eps', '-v7.3');
%     fprintf('✅  Saved: %s\n', fname);
% 
%     % Summary
%     acc = mean(preds == labels) * 100;
%     rob = mean(isRobust == 1) * 100;
% 
%     fprintf('\nMATLAB preds: stable=%d (%.2f%%), unstable=%d (%.2f%%)\n', ...
%         sum(preds==1), 100*mean(preds==1), sum(preds==2), 100*mean(preds==2));
% 
%     fprintf('Python preds: stable=%d (%.2f%%), unstable=%d (%.2f%%)\n', ...
%         sum(py_pred==1), 100*mean(py_pred==1), sum(py_pred==2), 100*mean(py_pred==2));
% 
%     match_rate = mean(preds(:) == py_pred(:)) * 100;
%     fprintf('Match rate vs Python (all): %.2f%%\n', match_rate);
% 
%     if eps == 0
%         tn = sum(preds==1 & labels==1); tp = sum(preds==2 & labels==2);
%         fprintf('Per-class correct @ε=0: stable=%d/%d (%.2f%%), unstable=%d/%d (%.2f%%)\n', ...
%             tn, sum(labels==1), 100*tn/max(1,sum(labels==1)), ...
%             tp, sum(labels==2), 100*tp/max(1,sum(labels==2)));
%     end
% 
%     fprintf('  Accuracy: %.2f%%  |  Verified: %.2f%%  |  Time: %.1fs\n\n', ...
%         acc, rob, sum(tGraph));
% end
% end
% 
% %% -------- Helper: one GCN layer (linear -> aggregate -> bias) --------
% function Z = gconv_ieee24(X, W, b, Ahat)
% % X: ImageStar with data laid out as (features × nodes)
% V = X.V; [~, n, ~, P] = size(V);
% fout = size(W,1);
% Vnew = zeros(fout, n, 1, P);
% for q = 1:P
%     H = squeeze(V(:,:,1,q));          % fin × n
%     Hlin = W * H;                      % linear first (fout × n)
%     Hagg = Hlin * Ahat;                % then aggregate (right-multiply by Â)
%     Vnew(:,:,1,q) = Hagg + b;          % add bias (broadcast across columns)
% end
% Z = ImageStar(Vnew, X.C, X.d, X.pred_lb, X.pred_ub);
% end
% 
% %% -------- Helper: activation tag --------
% function a = get_act(L)
% a = ""; 
% if isfield(L,'act') && ~isempty(L.act), a = string(L.act); end
% end
% 
% %% -------- Helper: spec half-space (true_logit ≥ other_logit) --------
% function Hs = label2Hs_binary(lbl)
% if lbl == 1          % stable: l1 - l2 ≥ 0
%     Hs = HalfSpace([ 1 -1], 0);
% else                  % unstable: l2 - l1 ≥ 0
%     Hs = HalfSpace([-1  1], 0);
% end
% end
% 
