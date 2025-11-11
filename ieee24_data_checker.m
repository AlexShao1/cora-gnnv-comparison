function ieee24_data_checker(model_folder, num_rows)
% IEEE24_DATA_CHECKER  Inspect ieee24_cascading_data_export.json, handling batched (N×...) and per-row formats.
%
% Usage:
%   ieee24_data_checker(model_folder)         % sample first 5 graphs
%   ieee24_data_checker(model_folder, 10)     % sample first 10 graphs
%
% Prints:
%   • Variable names & classes
%   • Detects format: BATCHED (one row, first dim = N) or PER-ROW (N rows)
%   • For sampled graphs: shapes, axis guesses, feature stats
%   • Edge index orientation (2×E vs E×2), index base (0/1), range vs 24 nodes
%   • Label preview parsed to {1=stable, 2=unstable}
%   • Verdict per graph: can standardize to (24×3, 2×E, scalar class)?
%
% No modifications to your data; print-only.

if nargin < 2, num_rows = 5; end
data_path = fullfile(model_folder, 'ieee24_cascading_data_export.json');
assert(exist(data_path, 'file')==2, 'Not found: %s', data_path);

tbl = neuralNetwork.readGNNdata(data_path);
fprintf('=== IEEE24 DATA CHECK ===\n');
fprintf('File: %s\nRows in table: %d\n\n', data_path, height(tbl));

% Header
disp('Table variables:');
disp(tbl.Properties.VariableNames);
for vn = tbl.Properties.VariableNames
    v = vn{1};
    fprintf('  %-16s class=%s\n', v, class(tbl.(v)));
end
fprintf('\n');

% Require expected variables
needVars = {'input','edge_index','target_label'};
for k = 1:numel(needVars)
    assert(any(strcmp(tbl.Properties.VariableNames, needVars{k})), ...
        'Missing table variable: %s', needVars{k});
end

% Determine data layout
isBatched = false;
N = height(tbl);
if N == 1
    % One row; check if contents’ first dim is Ngraphs
    if iscell(tbl.input) && ~isempty(tbl.input{1}) && isnumeric(tbl.input{1})
        Xall = tbl.input{1};
        if ndims(Xall) >= 3
            N = size(Xall,1);
            isBatched = true;
        end
    end
end

if isBatched
    fprintf('DETECTED FORMAT: BATCHED (one table row with arrays sized N×...)\n');
else
    fprintf('DETECTED FORMAT: PER-ROW (one graph per table row)\n');
end

M = min(N, num_rows);
fprintf('Sampling %d/%d graphs: %s\n\n', M, N, mat2str(1:M));

okCount = 0;
for k = 1:M
    fprintf('--- Graph %d ---\n', k);

    % ---------- INPUT ----------
    if isBatched
        Xraw = tbl.input{1};
        assert(size(Xraw,1) >= k, 'input batch too small');
        Xk = squeeze(double(Xraw(k,:,:)));    % (24×3) or (3×24)
    else
        Xk = squeeze(double(tbl.input{k}));
    end
    fprintf('input:         class=%s, size=%s, ndims=%d\n', class(Xk), mat2str(size(Xk)), ndims(Xk));

    % Axis guess
    szX = size(Xk);
    ix24 = find(szX==24, 1);
    ix3  = find(szX==3,  1, 'last');
    canInput = false;
    if ~isempty(ix24) && ~isempty(ix3)
        fprintf('  axes guess:  24@dim%d, 3@dim%d\n', ix24, ix3);
        if isequal(szX, [24 3])
            fprintf('  layout:      OK (24x3)\n'); canInput = true;
        elseif isequal(szX, [3 24])
            fprintf('  layout:      transposed (3x24) — would need transpose in pipeline\n'); canInput = true;
        else
            fprintf('  note:        would need permute/reshape to (24x3)\n'); canInput = true;
        end
    else
        fprintf('  WARNING:     could not locate dims 24 and 3 in input\n');
    end
    Xvec = Xk(:);
    fprintf('  feat stats:  min=%g, max=%g, NaN=%d, Inf=%d\n', ...
        min(Xvec), max(Xvec), sum(isnan(Xvec)), sum(isinf(Xvec)));

    % ---------- EDGE_INDEX ----------
    if isBatched
        Eraw = tbl.edge_index{1};
        assert(size(Eraw,1) >= k, 'edge_index batch too small');
        Ek = squeeze(double(Eraw(k,:,:)));    % (2×E) or (E×2)
    else
        Ek = squeeze(double(tbl.edge_index{k}));
    end
    fprintf('edge_index:    class=%s, size=%s, ndims=%d\n', class(Ek), mat2str(size(Ek)), ndims(Ek));

    % Orient to 2×E *for analysis only* (no transpose of N-D; Ek is 2-D now)
    canEdges = false; orient = 'unknown';
    if ismatrix(Ek)
        if size(Ek,1) == 2
            E2 = Ek; orient = '2xE'; canEdges = true;
        elseif size(Ek,2) == 2
            E2 = Ek.'; orient = 'Ex2 (T -> 2xE)'; canEdges = true;
        else
            fprintf('  WARNING:     edge_index is not 2xE or Ex2\n');
        end
    else
        fprintf('  WARNING:     edge_index not 2-D after squeeze\n');
    end
    fprintf('  orient:      %s\n', orient);
    if canEdges
        Emin = min(E2, [], 'all');
        Emax = max(E2, [], 'all');
        Ecnt = size(E2,2);
        fprintf('  stats:       E=%d, min=%g, max=%g\n', Ecnt, Emin, Emax);
        if Emin == 0
            fprintf('  index base:  likely 0-based\n');
        elseif Emin >= 1
            fprintf('  index base:  likely 1-based\n');
        else
            fprintf('  index base:  mixed/unknown\n');
        end
        if (Emax <= 23 && Emin >= 0) || (Emax <= 24 && Emin >= 1)
            fprintf('  node range:  consistent with 24 nodes\n');
        else
            fprintf('  node range:  UNUSUAL for 24 nodes\n');
        end
    end

    % ---------- LABEL ----------
    if isBatched
        Lraw = tbl.target_label{1};
        % Lraw could be: N×1 (0/1), N×2 (prob/one-hot), N×K (indicators)
        lbl = try_parse_label_batched(Lraw, k);
    else
        Lraw = tbl.target_label{k};
        lbl  = try_parse_label_scalarish(Lraw);
    end
    canLabel = ~isnan(lbl);
    if canLabel
        fprintf('label:         parsed -> %d (1=stable, 2=unstable)\n', lbl);
    else
        fprintf('label:         FAILED TO PARSE\n');
    end

    verdict = canInput && canEdges && canLabel;
    if verdict, okCount = okCount + 1; end
    fprintf('VERDICT graph %d: input:%d edges:%d label:%d  ==> CAN STANDARDIZE? %s\n\n', ...
        k, canInput, canEdges, canLabel, tern(verdict,'YES','NO'));
end

fprintf('=== SUMMARY on first %d graphs ===\n', M);
fprintf('Graphs fully standardizable now (dry-run): %d / %d\n', okCount, M);
fprintf('===============================\n');

end

% ----------------- helpers -----------------

function lbl = try_parse_label_batched(Lraw, k)
% Return 1 or 2, or NaN if not parseable (batched arrays).
    lbl = NaN;
    if isempty(Lraw), return; end
    if iscell(Lraw), Lraw = Lraw{1}; end

    if ~isnumeric(Lraw) && ~islogical(Lraw)
        % Strings/categoricals unlikely in batched export; bail gracefully.
        return;
    end

    sz = size(Lraw);
    % Case A: N×1 (0/1)
    if numel(sz) == 2 && sz(1) >= k && sz(2) == 1
        v = double(Lraw(k,1));
        if isfinite(v)
            lbl = 1 + (v > 0.5);
        end
        return;
    end
    % Case B: N×2 (logits/prob/one-hot)
    if numel(sz) == 2 && sz(1) >= k && sz(2) == 2
        r = double(Lraw(k,:));
        [~, idx] = max(r(:));
        lbl = idx; return;
    end
    % Case C: N×K, K>2 (binary indicators; any 1 ⇒ unstable)
    if numel(sz) == 2 && sz(1) >= k && sz(2) > 2
        r = double(Lraw(k,:));
        if all(isfinite(r))
            lbl = 1 + double(any(r > 0.5));
        end
        return;
    end
    % Case D: Higher-dim — try to squeeze down to a vector for k
    if numel(sz) >= 3 && sz(1) >= k
        r = squeeze(double(Lraw(k,:,:)));
        if isvector(r)
            r = r(:).';
            if numel(r) == 1
                lbl = 1 + double(r > 0.5);
            elseif numel(r) == 2
                [~, idx] = max(r(:)); lbl = idx;
            elseif numel(r) > 2
                lbl = 1 + double(any(r > 0.5));
            end
        end
    end
end

function lbl = try_parse_label_scalarish(raw)
% Non-batched: try to get 1/2, else NaN (no throwing).
    lbl = NaN;
    if iscell(raw) && ~isempty(raw), raw = raw{1}; end
    if isempty(raw), return; end

    if (isnumeric(raw)||islogical(raw)) && isscalar(raw)
        lbl = 1 + double(raw > 0.5); return;
    end
    if isnumeric(raw) && isvector(raw) && numel(raw)==2
        [~,idx]=max(double(raw(:))); lbl=idx; return;
    end
    if isnumeric(raw) && isvector(raw) && numel(raw)>2
        r = double(raw(:));
        if all(isfinite(r)), lbl = 1 + double(any(r > 0.5)); end
        return;
    end
    if isstring(raw) || ischar(raw)
        s = lower(strtrim(string(raw)));
        if s=="stable",   lbl=1; return; end
        if s=="unstable", lbl=2; return; end
        v = str2double(s); if ~isnan(v), lbl = 1 + double(v > 0.5); end
        return;
    end
    if iscategorical(raw)
        s = lower(string(raw));
        if s=="stable", lbl=1; return; end
        if s=="unstable", lbl=2; return; end
    end
end

function out = tern(cond, a, b)
    if cond, out=a; else, out=b; end
end
