function run_simulation(model_name, mat_file)
% RUN_SIMULATION  End-to-end demo runner for the v2 EVCS model with RL.
%
%   run_simulation()                                   uses defaults
%   run_simulation('EVCS_model_v2', 'data/ems_schedule.mat')
%
% Sequence:
%   1. Load EVCS parameters into base WS (P struct + flat names + .mat)
%   2. Open the model
%   3. Set StopTime to the schedule duration
%   4. Open the EMS_References scopes so they pop up automatically
%   5. Run

if nargin < 1, model_name = 'EVCS_model_v2';            end
if nargin < 2, mat_file   = 'data/ems_schedule.mat';    end

assert(isfile(mat_file), ...
    '%s not found.  Run python precompute.py first.', mat_file);

% --- 1. Parameters & schedule ---
fprintf('Loading parameters & schedule...\n');
evcs_params;     % populates P, flat names, AND loads the .mat

% --- 2. Model ---
if ~bdIsLoaded(model_name)
    open_system(model_name);
end

% --- 3. Stop time ---
stop_seconds = evalin('base', 'double(P_load.time(end))');
fprintf('StopTime: %.0f s (%.1f h, %.1f days)\n', ...
        stop_seconds, stop_seconds/3600, stop_seconds/86400);
set_param(model_name, 'StopTime', num2str(stop_seconds));

% --- 4. Auto-open the EMS scopes for the demo ---
sub = [model_name '/EMS_References'];
if getSimulinkBlockHandle(sub) ~= -1
    try
        open_system([sub '/Scope_LoadRef']);
        open_system([sub '/Scope_Setpoints']);
        open_system([sub '/Scope_Enables']);
        open_system([sub '/Scope_Mode']);
    catch
        % scopes don't exist yet — user hasn't run build_ems_references
        warning(['EMS_References subsystem not found.\n' ...
                 'Run >> build_ems_references(''%s'')  first.'], model_name);
    end
end

% --- 5. Simulate ---
fprintf('Running %s ...\n', model_name);
sim(model_name);
fprintf('Done.\n');
end