function demo(stop_time)
if nargin < 1
    stop_time = [];
end

mat_file = 'data/ems_schedule.mat';
if ~isfile(mat_file)
    error('%s not found. Run python precompute.py first.', mat_file);
end

fprintf('==> [1/5] Loading parameters & schedule...\n');
evcs_params;

% Define locals AFTER evcs_params (which has 'clear' inside)
model_name = 'EVCS_model_v2';

fprintf('==> [2/5] Opening model...\n');
if ~bdIsLoaded(model_name)
    open_system(model_name);
end

if isempty(stop_time)
    stop_time = evalin('base', 'double(P_load.time(end))');
end
fprintf('==> [3/5] StopTime: %.0f s\n', stop_time);
set_param(model_name, 'StopTime', num2str(stop_time));

fprintf('==> [4/5] Opening scopes (errors here are non-fatal)...\n');
sub = [model_name '/EMS_References'];
try, open_system([sub '/Scope_Mode']);     catch err, fprintf('  Scope_Mode: %s\n', err.message); end
try, open_system([sub '/Scope_LoadRef']);  catch err, fprintf('  Scope_LoadRef: %s\n', err.message); end
try
    sup_scopes = find_system([model_name '/supervisor'], 'BlockType', 'Scope');
    for i = 1:numel(sup_scopes)
        try, open_system(sup_scopes{i}); catch, end
    end
catch err
    fprintf('  supervisor scopes: %s\n', err.message);
end

fprintf('==> [5/5] Running sim() now...\n');

sig_names = {'P_load','P_solar_set','P_batt_set','P_h2_set','P_grid_set', ...
'en_solar','en_batt','en_h2','en_grid','MODE_CMD','P_EV'};
for i = 1:numel(sig_names)
v = evalin('base', sig_names{i});
v.signals.dimensions = 1;
assignin('base', sig_names{i}, v);
end
set_param('EVCS_model_v2', 'StopTime', '210')
tic; sim('EVCS_model_v2'); 
fprintf('==> Done in %.1f s.\n', toc);

% --- save the run ---
ts = datestr(now, 'yyyymmdd_HHMMSS');
out_dir = fullfile('data', 'sim_runs');
if ~isfolder(out_dir), mkdir(out_dir); end
out_file = fullfile(out_dir, sprintf('sim_%s_t%d.mat', ts, round(stop_time)));

% Pull workspace variables we want to preserve alongside the sim output
schedule_snapshot = struct();
for v = {'P_load', 'MODE_CMD', 'P_EV', 'P_solar_set', 'P_batt_set', ...
         'P_h2_set', 'P_grid_set', 'en_solar', 'en_batt', 'en_h2', 'en_grid'}
    if evalin('base', sprintf('exist(''%s'', ''var'')', v{1}))
        schedule_snapshot.(v{1}) = evalin('base', v{1});
    end
end

save(out_file, 'sim_out', 'schedule_snapshot', 'stop_time', 'elapsed', '-v7.3');
fprintf('==> Saved results to %s (%.1f MB)\n', out_file, ...
        dir(out_file).bytes / 1e6);
end