function [res, evResult] = cora_ieee_eval(varargin)
% -------------------------------------------------------------------------
% defaults
sanity_tol = 5e-2;   % tolerance for numeric forward vs exported logits
[NUM_CHECKS, MODEL, DELTA, PERT_EDGES, DO_ENUM, VERBOSE, GRAPH_INDICES] = setDefaultValues({ ...
    100, './experiments/models/IEEE24_CASCADING', 0.01, 0.00, false, false,[]}, varargin);

dateStr = datestr(datetime());
seed0 = randi(10^6, 1); 
rng(seed0);
if VERBOSE, fprintf('Seed: %d\n', seed0); end

% clear any leftovers
if exist('margins_at_zero','var'); clear margins_at_zero; end

% -------------------------------------------------------------------------
% load model + data (no edge_index used anywhere)
[nn, data, meta] = aux_load_ieee_flexible(MODEL);

% normalize labels to numeric columns
for col = ["output_label","target_label"]
    v = data.(col);
    if iscell(v), v = cellfun(@(x) double(x(1)), v); else, v = double(v); end
    data.(col) = v(:);
end

MAX_RUNS   = height(data);
NUM_CHECKS = min(NUM_CHECKS, MAX_RUNS);
idx        = 1:NUM_CHECKS;
data = data(idx, :);

% feature scale — see absolute size of DELTA
feat_min = zeros(NUM_CHECKS,1);
feat_max = zeros(NUM_CHECKS,1);
feat_std = zeros(NUM_CHECKS,1);
for ii = 1:NUM_CHECKS
    Xi = data.input{ii};
    feat_min(ii) = min(Xi(:));
    feat_max(ii) = max(Xi(:));
    feat_std(ii) = std(Xi(:));
end
fprintf('[scale] %s | nodes=%d, feat/node=%d\n', meta.dataset_tag, size(data.input{1},1), size(data.input{1},2));
fprintf('[scale] DELTA=%.4g | feature: min=%.3g max=%.3g median(std)=%.3g\n', ...
    DELTA, min(feat_min), max(feat_max), median(feat_std));

% -------------------------------------------------------------------------
% result storage
resvec            = false(1, NUM_CHECKS);
resVerifiedSet    = false(1, NUM_CHECKS);
resViolated       = false(1, NUM_CHECKS);
timeSet           = nan(1, NUM_CHECKS);
numNodes          = nan(1, NUM_CHECKS);
numEdges          = nan(1, NUM_CHECKS);
margins_at_zero   = [];
counterSuccessful = 1;
counterFailed     = 0;
MEs               = {};
failedIdx         = []; 

% -------------------------------------------------------------------------
% main loop
for k = 1:NUM_CHECKS
    try
        [nn_red, G, x_vec, y_ref, target_label] = aux_constructGraph_ieee(data, k, nn);
        Xi = data.input{k};  % [nodes x feats]
        [nodes, feats] = size(Xi);

       % === DEBUG: Graph structure check (only first iteration) ===
        if k == 1
            fprintf('\n=== GRAPH STRUCTURE DEBUG ===\n');
            fprintf('MATLAB graph() reports: %d nodes, %d edges (undirected count)\n', ...
                G.numnodes, G.numedges);
            
            % Check the actual adjacency matrix CORA uses
            A = G.adjacency;
            fprintf('Adjacency matrix A: %d×%d, nnz=%d\n', size(A,1), size(A,2), nnz(A));
            
            % Extract diagonal for sparse matrix
            A_diag = full(spdiags(A, 0));
            fprintf('Self-loops present: %d/%d nodes\n', sum(A_diag>0), G.numnodes);
            fprintf('A is symmetric: %s\n', string(issymmetric(A)));
            
            % Count actual edges (excluding self-loops)
            non_self_loop_edges = (nnz(A) - sum(A_diag>0)) / 2;  % Divide by 2 for undirected
            fprintf('Non-self-loop edges: %.0f, Self-loops: %d, Total: %d\n', ...
                non_self_loop_edges, sum(A_diag>0), nnz(A));
            
            % Sample some degrees
            deg = full(sum(A, 2));
            fprintf('Node degrees: min=%d, max=%d, mean=%.1f\n', ...
                min(deg), max(deg), mean(deg));
            
            fprintf('============================\n\n');
            
            nn_red.resetGNN();
            options.nn = struct('graph', G);
            
            L = nn_red.layers;
            
            % Current order: Linear, GCN, ReLU
            lin1 = L{1};
            gcn1 = L{2};    
            relu1 = L{3};  
            
            % Test current order
            nn_A = neuralNetwork({lin1, gcn1, relu1});
            nn_A.resetGNN();
            zA = nn_A.evaluate(x_vec, options);
            
            % Test old wrong order
            nn_B = neuralNetwork({gcn1, lin1, relu1});
            nn_B.resetGNN();
            zB = nn_B.evaluate(x_vec, options);
            
            d = zA - zB;
            fprintf('[ORDER DEBUG] after 1st block:\n');
            fprintf('  Current (Lin→GCN→ReLU):  [% .3f, % .3f, ...]\n', zA(1), zA(2));
            fprintf('  Old Order (GCN→Lin→ReLU): [% .3f, % .3f, ...]\n', zB(1), zB(2));
            fprintf('  max|Δ|=%.3e,  ‖Δ‖₂=%.3e\n', max(abs(d)), norm(d));
            
            A = spones(G.adjacency);
            if nnz(diag(A)) == 0, A = A + speye(G.numnodes); end  % safety: ensure I is present
            deg = full(sum(A,2)); deg(deg==0) = 1;
            Dinv2 = spdiags(1./sqrt(deg), 0, G.numnodes, G.numnodes);
            Ahat  = Dinv2 * A * Dinv2;
            
            % --- Pull weights/biases: OPTION A (safer): from the CORA nn you already loaded ---
            % Lnn  = nn.layers;                     % your CORA network (already parsed)
            % W1   = double(Lnn{2}.W);  b1 = double(Lnn{2}.b(:));   % after first nnGCNLayer
            % W2   = double(Lnn{5}.W);  b2 = double(Lnn{5}.b(:));
            % W3   = double(Lnn{8}.W);  b3 = double(Lnn{8}.b(:));
            % Wout = double(Lnn{10}.W); bout = double(Lnn{10}.b(:));
            Lnn = nn.layers;
            W1 = double(Lnn{1}.W);   b1 = double(Lnn{1}.b(:));    % ← Layer 1 now (was 2)
            W2 = double(Lnn{4}.W);   b2 = double(Lnn{4}.b(:));    % ← Layer 4 now (was 5)
            W3 = double(Lnn{7}.W);   b3 = double(Lnn{7}.b(:));    % ← Layer 7 now (was 8)
            Wout = double(Lnn{10}.W); bout = double(Lnn{10}.b(:));

            X = data.input{k};                     % [N x F]  (F=3)
            N = size(X,1);
            
            H1 = Ahat * (X * W1.') + repmat(b1.', N, 1);  H1 = max(H1, 0);
            H2 = Ahat * (H1 * W2.') + repmat(b2.', N, 1); H2 = max(H2, 0);
            H3 = Ahat * (H2 * W3.') + repmat(b3.', N, 1); % last GCN: no ReLU
            
            h  = mean(H3, 1).';                    % global_mean_pool over nodes -> [H x 1]
            z  = Wout * h + bout;                  % [2 x 1] logits
            
            fprintf('[GOLD] PyG-manual: [%.3f, %.3f]\n', z(1), z(2));
            fprintf('[REF ] PyG-export: [%.3f, %.3f]\n', y_ref(1), y_ref(2));

            % Test against PyTorch export
            y_pred_test = nn_red.evaluate(x_vec, options);
            fprintf('\n[MATCH TEST] CORA vs PyTorch export:\n');
            fprintf('  CORA y_pred: [%.3f, %.3f]\n', y_pred_test(1), y_pred_test(2));
            fprintf('  PyG y_ref:   [%.3f, %.3f]\n', y_ref(1), y_ref(2));
            fprintf('  Difference:  [%.3f, %.3f]\n', ...
                y_pred_test(1)-y_ref(1), y_pred_test(2)-y_ref(2));
        end

        % if k <= 3
        %     fprintf('  Sample %d input stats: min=%.3f, max=%.3f, mean=%.3f\n', ...
        %             k, min(x_vec), max(x_vec), mean(x_vec));
        % end
        nn_red.resetGNN();
        options.nn = struct('graph', G);
        y_pred = nn_red.evaluate(x_vec, options);
        numNodes(k) = G.numnodes; 
        numEdges(k) = G.numedges;

        % numeric forward (sanity vs exported logits)
        resvec(k) = compareMatrices(y_pred, y_ref, sanity_tol);
        if k <= 3
            fprintf('  [Sample %d] y_pred: [%.3f, %.3f], y_ref: [%.3f, %.3f], match: %d\n', ...
                     counterSuccessful, y_pred(1), y_pred(2), y_ref(1), y_ref(2), resvec(k));
        end

        % collect margin at Δ=0
        if DELTA == 0
            cls = numel(y_pred); 
            others = setdiff(1:cls, target_label);
            margins_at_zero(end+1,1) = y_pred(target_label) - max(y_pred(others)); %#ok<AGROW>
        end

        % % decide # perturbed edges
        % if floor(PERT_EDGES) == PERT_EDGES
        %     numPertEdges = PERT_EDGES;
        % else
        %     numPertEdges = ceil(G.numedges * PERT_EDGES);
        % end
        % 
        % [Gpert, idxPertEdges] = aux_perturb_graph_rmedge(G, 0);
        % if numel(idxPertEdges) < numPertEdges, continue; end
        % 
        % nn_red.resetGNN();
        % options.nn.graph          = Gpert;
        % options.nn.idx_pert_edges = idxPertEdges;
        % options.nn.num_generators = 10000;
        % 
        % if k == 1
        %     fprintf('\n=== INPUT SHAPE CHECK ===\n');
        %     fprintf('Input Xi shape: [%d x %d] (should be [nodes x features] = [24 x 3])\n', size(Xi, 1), size(Xi, 2));
        %     st = dbstack(1);
        %     caller = '';
        %     if ~isempty(st), caller = sprintf(' @%s:%d', st(1).name, st(1).line); end
        %     fprintf('[k=%d]%s x_vec shape: [%d x %d]\n', k, caller, size(x_vec,1), size(x_vec,2));
        %     fprintf('=========================\n\n');
        % end
        
        % numeric forward sanity (same x_vec)
        % nn_red.resetGNN();
        % options.nn = struct('graph', G);
        % y_pred = nn_red.evaluate(x_vec, options);
        % resvec(k) = compareMatrices(y_pred, y_ref, sanity_tol);
        
        % node-tied L_inf box: one generator per NODE (shared across all its features)
       if DELTA == 0
            X = polyZonotope(x_vec);
        else
            % Node-tied L_inf box: one generator per NODE
            Ggen = zeros(nodes*feats, nodes);  % DENSE
            for p = 1:nodes
                rows = (p-1)*feats + (1:feats);
                Ggen(rows, p) = 1;
            end
            X = compact(polyZonotope(x_vec, DELTA * Ggen));
        end
        
        nn_red.resetGNN();
        options.nn = struct();
        options.nn.graph = G;

        t0 = tic;
        Y  = nn_red.evaluate(X, options);
        resVerifiedSet(k) = aux_isVerified(Y, target_label);
        timeSet(k)        = toc(t0);

        counterSuccessful = counterSuccessful + 1;
    catch ME
        counterFailed = counterFailed + 1;
        MEs{end+1}    = ME;
        failedIdx(end+1) = k;
        fprintf('  [Sample %d] FAILED: %s\n', k, ME.message);
    end
end

% -------------------------------------------------------------------------
% diagnostics
ok_sanity = mean(resvec(~isnan(resvec)));
fprintf('Sanity check (forward==export, tol=%.1e): %.2f\n', sanity_tol, ok_sanity);

fprintf('Graph sizes: nodes min/med/max = %d / %d / %d, edges min/med/max = %d / %d / %d\n', ...
    min(numNodes), median(numNodes), max(numNodes), ...
    min(numEdges), median(numEdges), max(numEdges));

if ~isempty(margins_at_zero)
    q = quantile(margins_at_zero, [0 .05 .25 .5 .75 .95 1]);
    fprintf('[margins Δ=0] min=%.4g  p05=%.4g  p25=%.4g  p50=%.4g  p75=%.4g  p95=%.4g  max=%.4g\n', q);
    fprintf('[margins Δ=0] nonpositive = %d / %d\n', sum(margins_at_zero<=0), numel(margins_at_zero));
end

% summary
fprintf('\nSummary [%s]: perturb %.4g | success=%d | skipped=%d | fail=%d | verified=%.2f | violated=%.2f\n', ...
    meta.dataset_tag, DELTA, ...
    sum(~isnan(resvec)), 0, 0, ...
    mean(resVerifiedSet(~isnan(resVerifiedSet))), ...
    mean(resViolated(~isnan(resViolated))));

% gather results
evResult = struct;
evResult.NUM_CHECKS      = NUM_CHECKS;
evResult.MODEL           = MODEL;
evResult.DELTA           = DELTA;
evResult.NUM_PERT_EDGES  = PERT_EDGES;
evResult.date            = dateStr;
evResult.seed            = seed0;
evResult.MEs             = MEs;
evResult.failedIdx       = failedIdx;
evResult.res             = all(resvec);
evResult.resvec          = resvec;
evResult.resVerifiedSet  = resVerifiedSet;
evResult.timeSet         = timeSet;
evResult.numNodes        = numNodes;
evResult.numEdges        = numEdges;

res = all(resvec);

% =========================================================================
% GNNV-STYLE OUTPUT FORMATTING
% =========================================================================

% After the main loop and before gathering results, add:

fprintf('\n========================================================\n');
fprintf('IEEE24 VERIFICATION SUMMARY (CORA)\n');
fprintf('========================================================\n\n');

fprintf('[Δ=%.4f]  ε = %.4f\n', DELTA, DELTA);

% true labels (0/1 in your table → +1 for 1-based)
y_true = data.target_label(:) + 1;

% predicted labels from CORA numeric forward
y_pred_cls = zeros(NUM_CHECKS,1);
for k = 1:NUM_CHECKS
    [nn_red, G, x_vec, ~, ~] = aux_constructGraph_ieee(data, k, nn);
    nn_red.resetGNN();
    options.nn = struct('graph', G);
    yk = nn_red.evaluate(x_vec, options);
    [~, y_pred_cls(k)] = max(yk);
end

acc_eps0 = mean(y_pred_cls(:) == y_true(1:NUM_CHECKS)) * 100;
fprintf('  Accuracy (ε=0)  : %.2f%%\n', acc_eps0);


% Certified total
num_certified = sum(resVerifiedSet(~isnan(resVerifiedSet)));
fprintf('  Certified total : %4d / %4d  (%.2f%%)\n', ...
    num_certified, NUM_CHECKS, mean(resVerifiedSet(~isnan(resVerifiedSet)))*100);

% Per-class breakdown
if DELTA == 0
    % Get actual predictions (convert from sanity check)
    % Assume resvec indicates correct predictions
    stable_indices = find(data.target_label == 0);
    unstable_indices = find(data.target_label == 1);
    
    % Certified per class
    certified_stable = sum(resVerifiedSet(stable_indices));
    total_stable = length(stable_indices);
    certified_unstable = sum(resVerifiedSet(unstable_indices));
    total_unstable = length(unstable_indices);
    
    fprintf('    ├─ stable     : %.2f%%  (%d / %d)\n', ...
        100*certified_stable/max(1,total_stable), certified_stable, total_stable);
    fprintf('    └─ unstable   : %.2f%%  (%d / %d)\n', ...
        100*certified_unstable/max(1,total_unstable), certified_unstable, total_unstable);
end

% Time statistics
fprintf('  Time            : %.1f s  (avg %.3f s/graph)\n', ...
    sum(timeSet(~isnan(timeSet))), mean(timeSet(~isnan(timeSet))));

fprintf('\n========================================================\n');

end

% =========================================================================
% HELPERS
% =========================================================================
function [nn, data, meta] = aux_load_ieee_flexible(modelFolder)
    % ----- MODEL -----
    mCands = { fullfile(modelFolder,'model_export.json'), ...
               fullfile(modelFolder,'cora_ready','model_export.json') };
    mGlob  = dir(fullfile(modelFolder,'*_cora_cascading_model_export.json'));
    model_json = pick_first_existing(mCands, mGlob);
    fprintf('Reading network from: %s\n', model_json);

    % If layers accidentally stored as a cell-array, convert to struct-array
    % Read JSON
    raw_json = jsondecode(fileread(model_json));
    
    % If layers is a cell array, pad fields and convert to struct array
    if iscell(raw_json.layers)
        fprintf('  Converting cell array layers to struct array (with field padding)...\n');
    
        L = raw_json.layers;
        nL = numel(L);
    
        % Union of fields we care about (pad missing ones)
        want = {'type','W','b','act'};
    
        % Preallocate struct array with those fields
        emptyTemplate = struct('type',"", 'W', [], 'b', [], 'act', "");
        layers_struct = repmat(emptyTemplate, 1, nL);
    
        for i = 1:nL
            s = L{i};
    
            % Coerce to struct if needed (jsondecode usually already is)
            if ~isstruct(s)
                error('Layer %d is not a struct after jsondecode.', i);
            end
    
            % Pad missing fields
            if ~isfield(s,'type'), s.type = ""; end
            if ~isfield(s,'W'),    s.W    = []; end
            if ~isfield(s,'b'),    s.b    = []; end
            if ~isfield(s,'act'),  s.act  = ""; end
    
            % Ensure field order & types match the template
            layers_struct(i).type = string(s.type);  % string scalar
            layers_struct(i).W    = s.W;             % numeric or []
            layers_struct(i).b    = s.b;             % numeric column or []
            layers_struct(i).act  = string(s.act);   % string scalar
        end
    
        raw_json.layers = layers_struct;
    
        % Write to a temp file and let CORA read it
        temp_json = fullfile(tempdir, 'temp_cora_model.json');
        fid = fopen(temp_json, 'w'); fprintf(fid, '%s', jsonencode(raw_json)); fclose(fid);
        nn = neuralNetwork.readGNNetwork(temp_json);
        if exist(temp_json,'file'), delete(temp_json); end
    else
        % Already a struct array
        nn = neuralNetwork.readGNNetwork(model_json);
        pool_idx = find(cellfun(@(L) isa(L,'nnGNNGlobalPoolingLayer'), nn.layers));
        assert(numel(pool_idx)==1, 'Expected exactly one global pooling layer');
        
        PL = nn.layers{pool_idx};
        is_mean = false;
        if isprop(PL,'poolType'), is_mean = strcmp(PL.poolType,'mean'); end
        if isprop(PL,'type'),     is_mean = is_mean || strcmp(PL.type,'mean'); end
        if isprop(PL,'mode'),     is_mean = is_mean || strcmp(PL.mode,'mean'); end
        assert(is_mean, 'Global pooling is not MEAN');
        
        % 2) Pooling sits right before the classifier
        assert(pool_idx == numel(nn.layers)-1, 'Pooling should be penultimate layer');
        assert(isa(nn.layers{pool_idx+1}, 'nnGNNLinearLayer'), 'Final layer should be linear');
        
        % 3) No activation after pooling
        if pool_idx < numel(nn.layers) && isa(nn.layers{pool_idx+1}, 'nnActivationLayer')
            error('Found activation after pooling; PyG export uses no activation there.');
        end
    end


    fprintf('Model layers: %d\n', length(nn.layers));
    for i = 1:length(nn.layers)
        fprintf('  Layer %d: %s\n', i, class(nn.layers{i}));
    end

    fprintf('\n=== WEIGHT SHAPE CHECK ===\n');
    if isa(nn.layers{1}, 'nnGNNLinearLayer')
        W1 = nn.layers{1}.W;
        fprintf('CORA received W1 shape: [%d x %d]\n', size(W1, 1), size(W1, 2));
        fprintf('PyTorch exported: [32 x 3] (out x in)\n');
        if size(W1, 1) == 3 && size(W1, 2) == 32
            fprintf('→ Weights are TRANSPOSED (need to fix export)\n');
        elseif size(W1, 1) == 32 && size(W1, 2) == 3
            fprintf('→ Weights match PyTorch (correct)\n');
        else
            fprintf('→ Unexpected shape!\n');
        end
    else
        fprintf('Layer 1 is not nnGNNLinearLayer (it is %s)\n', class(nn.layers{1}));
    end
    fprintf('==========================\n\n');

    % ----- DATA -----
    dCands = { fullfile(modelFolder,'data_export.json'), ...
               fullfile(modelFolder,'cora_ready','data_export.json') };
    dGlob  = dir(fullfile(modelFolder,'*_cora_cascading_data_export.json'));
    data_json = pick_first_existing(dCands, dGlob);
    fprintf('Reading data from: %s\n', data_json);

    [~, nm] = fileparts(data_json);
    low = lower(nm);
    if contains(low,'ieee39'),   tag='IEEE39';
    elseif contains(low,'ieee118'), tag='IEEE118';
    elseif contains(low,'ieee24'),  tag='IEEE24';
    else, tag='IEEE-UNKNOWN';
    end

    data = aux_convert_matrix_to_table(data_json);
    fprintf('Converted to %d graphs\n', height(data));
    meta = struct('dataset_tag', tag, 'data_json', data_json, 'model_json', model_json);
end

function data = aux_convert_matrix_to_table(data_json)
    jd  = jsondecode(fileread(data_json));
    headers = jd{1}; columns = jd{2};
    H = containers.Map(headers, 1:numel(headers));

    inputs        = columns{H('input')};
    outputs       = columns{H('output')};
    output_labels = columns{H('output_label')};
    target_labels = columns{H('target_label')};

    % Shapes
    if ndims(inputs) == 3
        N      = size(inputs, 1);
        Nnodes = size(inputs, 2);
        Nfeat  = size(inputs, 3);
        fprintf('  Input in 3D array format\n');
    elseif ismatrix(inputs) && ~isvector(inputs)
        N   = size(inputs, 1);
        tot = size(inputs, 2);
        if mod(tot, 3) ~= 0
            error('Unexpected feature count: %d (cannot deduce nodes×features)', tot);
        end
        Nfeat  = 3;
        Nnodes = tot / 3;
        fprintf('  Input in 2D array format\n');
    else
        error('Unexpected "input" format');
    end
    fprintf('  Detected: %d samples, %d nodes, %d features/node\n', N, Nnodes, Nfeat);

    % Ensure outputs are N×CLS
    if isnumeric(outputs) && size(outputs,1) < size(outputs,2)
        outputs = outputs.';
        fprintf('  Transposed outputs\n');
    end

    % Build table (no edge_index column)
    data = table();
    for i = 1:N
        if ndims(inputs) == 3
            Xi = squeeze(inputs(i,:,:));      % [nodes x feat]
        else
            Xi = reshape(inputs(i,:), Nfeat, Nnodes).';
        end
        data.input{i}        = Xi;
        data.output{i}       = outputs(i,:)';
        data.output_label(i) = double(output_labels(i));
        data.target_label(i) = double(target_labels(i));
    end
end

function E0 = aux_create_default_edges(N)
    % IEEE24 FIXED edge list (34 unique undirected edges, 0-based)
    % This replaces the function in your gnn_eval_ieee script
    
    if N == 24
        E0 = [
            0,1; 0,2; 0,4; 1,3; 1,5; 2,8; 2,23; 3,8; 4,9; 5,9;
            6,7; 7,8; 7,9; 8,10; 8,11; 9,10; 9,11; 10,12; 10,13; 11,12;
            11,22; 12,22; 13,15; 14,15; 14,20; 14,20; 14,23; 15,16; 15,18; 16,17;
            16,21; 17,20; 17,20; 18,19; 18,19; 19,22; 19,22; 20,21
        ]'; 
    else
        % Simple line as fallback (undirected 0-based)
        E0 = zeros(2, N-1);
        for u = 0:N-2, v = u+1; E0(:,u+1) = [u; v]; end
    end
end

function G = aux_create_graph(numNodes, adj_0based)
    % --- GNNV-style adjacency (keep duplicate edge weights; add self-loops) ---
    E = adj_0based + 1;              % [2 x E], 1-based
    n = numNodes;

    if isempty(E), G = graph([], [], n); return; end

    src = E(1,:); 
    dst = E(2,:);

    % undirected + self-loops (exactly like your GNNV code)
    src_ud = [src, dst, 1:n];
    dst_ud = [dst, src, 1:n];
    w      = ones(1, numel(src_ud));

    % NOTE: DO NOT binarize with spones(A) here – keep duplicate weights
    A = sparse(src_ud, dst_ud, w, n, n);

    % keep weights inside the MATLAB graph
    G = graph(A);
end





function res = aux_isVerified(Y, target_label)
    W = eye(length(Y.c)); W(:,target_label) = W(:,target_label)-1;
    I = interval(W*Y);
    res = all(I.sup <= 0);
end

function [Gpert, idxPertEdges] = aux_perturb_graph_rmedge(G, numPertEdges)
    if numPertEdges<=0, idxPertEdges=[]; Gpert=G; return; end
    [~, r] = max(degree(G));
    T = G.minspantree('Root',r,'Type','forest');
    Gpert = G.rmedge(T.Edges.EndNodes(:,1), T.Edges.EndNodes(:,2));
    if Gpert.numedges==0, idxPertEdges=[]; return; end
    pick = randsample(Gpert.numedges, min(numPertEdges, Gpert.numedges));
    E = Gpert.Edges.EndNodes(pick,:);
    idxPertEdges = G.findedge(E(:,1),E(:,2));
end

function [nn_red, G, x_vec, y_ref, target_label] = aux_constructGraph_ieee(data, i, nn)
    Xorg = data.input{i};                    % [nodes x feat]
    y_ref = data.output{i};                  % [2 x 1]
    target_label = data.target_label(i) + 1; % 1-based class index

    % Always use the hardcoded IEEE-24 template (same as your GNNV intent)
    E0 = aux_create_default_edges(size(Xorg,1));   % 0-based 2×E
    G  = aux_create_graph(size(Xorg,1), E0);       % simple, undirected, no self-loops

    x_vec = Xorg(:);   % (feats x nodes)(:)
    nn_red = nn;
end

function path = pick_first_existing(candidates, varargin)
    if ~iscell(candidates), error('first arg must be cellstr'); end
    allPaths = candidates(:);
    if ~isempty(varargin)
        ds = varargin{1};
        if isstruct(ds)
            for k=1:numel(ds), allPaths{end+1,1}=fullfile(ds(k).folder, ds(k).name); end
        end
    end
    for i=1:numel(allPaths)
        p = allPaths{i};
        if ~isempty(p) && exist(p,'file')==2, path=p; return; end
    end
    error('pick_first_existing: none exist:\n%s', strjoin(allPaths,newline));
end

