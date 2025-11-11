function completed = cora_ieee_main()
CORA_ROOT = '/Users/alexshao/Desktop/Cora Verify';   % <- your CORA repo root
addpath(genpath(CORA_ROOT), '-begin');  
rng(1); warning off

% --- paths ---------------------------------------------------------------
basepath = '.';
datapath = sprintf("%s/experiments", basepath);
resultpath = sprintf("%s/results/%s", basepath, datestr(datetime,'yymmdd-hhMMss'));
mkdir(resultpath)
evalpath = sprintf("%s/evaluation", resultpath); mkdir(evalpath)
plotpath = sprintf("%s/plots", resultpath);     mkdir(plotpath)

set(0, 'defaultFigureRenderer', 'painters')
resultstxt = sprintf("%s/results.txt", resultpath);
if exist(resultstxt,'file'), delete(resultstxt); end
diary(resultstxt)

disp("--------------------------------------------------------")
disp('IEEE24 Verification (CORA pipeline)')
fprintf("Date: %s\n", datestr(datetime()))
disp(CORAVERSION)
disp("--------------------------------------------------------")
disp(" ")

% --- RUN SCRIPTS: evaluation + plots (IEEE24 only) -----------------------
scrips = {; ...
    @() aux_evaluate_ieee(datapath,evalpath), "evaluation_ieee24";
    %@() gnn_eval_plots(evalpath),              "plots";
    };

fprintf("Running %d scripts.. \n\n", size(scrips,1));
for i = 1:size(scrips,1)
    disp("--------------------------------------------------------")
    script = scrips{i, 1};
    name   = scrips{i, 2};
    try
        fprintf("Running '%s' ...\n", name)
        script();

        disp(" ")
        fprintf("'%s' was run successfully!\n", name)
        fprintf("Saving plots to '%s'..\n", plotpath)

        h = findobj('type', 'figure');
        for j = 1:length(h)
            savefig(sprintf("%s/%s_%d.fig", plotpath, name, j));
            saveas(gcf, sprintf("%s/%s_%d.png", plotpath, name, j));
            close(gcf)
        end
    catch ME
        disp(" ")
        fprintf("An ERROR occured during execution of '%s':\n", name);
        disp(ME.getReport())
        disp("Continuing with next script..")
    end
    disp(" ")
end

disp("--------------------------------------------------------")
disp(" ")
completed = 1;
disp("Completed!")
fprintf("Date: %s\n", datestr(datetime()))
diary off
end

% -------------------------------------------------------------------------
function aux_evaluate_ieee(datapath, evalpath)
    % List one or more model folders. Each folder must contain the exported
    % model/data JSONs (either standard names or the ieee24_* alternates).
    models = { ...
        '/Users/alexshao/Desktop/Cora Verify/IEEE24', ...
        % '/path/to/IEEE39', ...
        % '/path/to/IEEE118',
    };

    % Verification settings
    deltas     = [0, 0.01, 0.02, 0.03, 0.04, 0.05];   % feature perturbations (L_inf on node features)
    pert_edges = 0;                       % no edge removals in this phase
    do_enum    = false;                   % no enumeration

    for mi = 1:numel(models)
        model = models{mi};
        if isfolder(model)
            modelpath = model;
        else
            modelpath = sprintf('%s/%s', datapath, model);
        end

        try
            [~, data_tbl] = aux_load_ieee_flexible(modelpath);
        catch ME
            fprintf(2, 'ERROR: could not load data from "%s"\n', modelpath);
            rethrow(ME);
        end

        total_rows = height(data_tbl);
        mask_ok = data_tbl.output_label == data_tbl.target_label;
        data_tbl = data_tbl(mask_ok, :);
        NUM_CHECKS = height(data_tbl);

        fprintf('\n=== DATASET SUMMARY ===\n');
        fprintf('Path: %s\n', modelpath);
        fprintf('Total exported rows: %d\n', total_rows);
        fprintf('Correct-at-baseline rows: %d\n', NUM_CHECKS);
        if NUM_CHECKS == 0
            fprintf(2, ['WARNING: No correctly-classified rows; skipping this dataset. ', ...
                        'Either relax the filter or check your export.\n']);
            continue;
        end

        aux_evalute_gnn_details_ieee(datapath, evalpath, NUM_CHECKS, {modelpath}, ...
                                     deltas, pert_edges, do_enum);
    end
end


% -------------------------------------------------------------------------
function aux_evalute_gnn_details_ieee(datapath, evalpath, NUM_CHECKS, models, deltas, pert_edges, do_enumeration)
    for model_i = 1:numel(models)
        for delta = deltas
            for pert_edge = (numel(pert_edges)==1 && pert_edges==0) * 0 + (numel(pert_edges)>1) .* pert_edges
                model = models{model_i};
                disp('---')
                fprintf('pert_edge: %.3f, delta=%.3f, model=%s, do_enumeration=%i\n', ...
                        pert_edge, delta, model, do_enumeration)

                if isfolder(model)
                    modelpath = model;                 % absolute or already-correct path
                else
                    modelpath = sprintf('%s/%s', datapath, model);
                end



                % IEEE-friendly evaluator (flexible file names)
                [~, evResult] = cora_ieee_eval( ...
                    100, modelpath, delta, pert_edge, do_enumeration);
                
                % After aux_evalute_gnn_details_ieee runs, check errors
                if isfield(evResult, 'MEs') && ~isempty(evResult.MEs)
                    fprintf('\n=== ERRORS ENCOUNTERED ===\n');
                    for k = 1:min(3, length(evResult.MEs))
                        fprintf('Error %d:\n', k);
                        disp(evResult.MEs{k}.message);
                        disp(evResult.MEs{k}.stack(1));
                    end
                end

                % save
                evalmodelpath = sprintf('%s/%s', evalpath, model);
                mkdir(evalmodelpath)
                save( ...
                    sprintf('%s/evResult-edge%.3f-d%.3f-enum%i.mat', ...
                            evalmodelpath, pert_edge, delta, do_enumeration), ...
                    'evResult');
                disp(' ')
            end
        end
    end
end

function [nn_stub, data_tbl] = aux_load_ieee_flexible(modelFolder)

nn_stub = [];  % not used by the caller in this peek

%% CHANGE: helper for first existing path
pick_first_existing = @(cands) local_pick_first_existing(cands);

% ----- resolve DATA json (prefer *_cascading_*, fall back to data_export) -----
cands_data = { ...
    fullfile(modelFolder, 'ieee39__cora_cascading_data_export.json')    % <-- prefer 39
    fullfile(modelFolder, 'cora_ready', 'ieee39__cora_cascading_data_export.json')
    fullfile(modelFolder, 'ieee24_cora_cascading_data_export.json')
    fullfile(modelFolder, 'cora_ready', 'ieee24_cora_cascading_data_export.json')
    fullfile(modelFolder, 'ieee118_cora_cascading_data_export.json')
    fullfile(modelFolder, 'cora_ready', 'ieee118_cora_cascading_data_export.json')
    fullfile(modelFolder, 'data_export.json')
    fullfile(modelFolder, 'cora_ready','data_export.json')
    };
data_json = pick_first_existing(cands_data);
fprintf('aux_load_ieee_flexible: reading data from: %s\n', data_json);

% ----- read JSON -----
str = fileread(data_json);
jd  = jsondecode(str);

% Accept column-oriented JSON: {headers, columns}
if iscell(jd)
    headers = jd{1};
    columns = jd{2};

    % build a map header -> index
    hmap = containers.Map;
    for i = 1:numel(headers)
        hmap(headers{i}) = i;
    end

    olab = columns{hmap('output_label')};
    tlab = columns{hmap('target_label')};

    % Normalize to numeric column vectors
    olab = aux_to_column_double(olab);
    tlab = aux_to_column_double(tlab);

    data_tbl = table(olab, tlab, 'VariableNames', {'output_label','target_label'});
else
    % STRUCT format (rare)
    if isfield(jd,'output_label') && isfield(jd,'target_label')
        olab = aux_to_column_double(jd.output_label);
        tlab = aux_to_column_double(jd.target_label);
        data_tbl = table(olab, tlab, 'VariableNames', {'output_label','target_label'});
    else
        error('Unexpected JSON structure. Missing output_label/target_label.');
    end
end
end


% --------- helpers ---------
function col = aux_to_column_double(x)
% Accepts numeric arrays or cells and returns a double column vector.
if iscell(x)
    % flatten cell to numeric
    try
        x = cellfun(@double, x);
    catch
        % if nested cells, try to pull out scalars
        x = cellfun(@(t) double(t(1)), x);
    end
end
if ~isvector(x)
    % allow row vector
    x = x(:);
end
col = double(x(:));
end

function path = local_pick_first_existing(cands)
for ii = 1:numel(cands)
    p = cands{ii};
    if ~isempty(p) && exist(p,'file') == 2
        path = p; return
    end
end
% Also allow wildcard as a last resort (e.g., *_cascading_data_export.json)
for ii = 1:numel(cands)
    p = cands{ii};
    if contains(p,'*')
        d = dir(p);
        if ~isempty(d)
            path = fullfile(d(1).folder, d(1).name);
            return
        end
    end
end
error('No data JSON found. Tried:\n%s', strjoin(cands,newline));
end