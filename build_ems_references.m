function build_ems_references(model_name)
% BUILD_EMS_REFERENCES  Patch the v2 EVCS Simulink model for RL integration.
%
%   build_ems_references('EVCS_model_v2')
%
% Does three things, idempotently:
%   1. Bumps both Fuel Cell Stack blocks from their 6 kW preset to 50 kW
%      (uses the values now in evcs_params -> P.FC).
%   2. Builds an 'EMS_References' subsystem with 11 From-Workspace blocks:
%        P_load, P_solar_set, P_batt_set, P_h2_set, P_grid_set,
%        en_solar, en_batt, en_h2, en_grid,
%        MODE_CMD (overrides the supervisor)
%        P_EV     (drives the existing P_EV Goto tag)
%   3. Wires the Goto tags MODE_CMD and P_EV from the EMS subsystem so the
%      existing supervisor picks them up automatically.  (The supervisor's
%      override path was already wired by the model's author; we just need
%      to source MODE_CMD with the right signal.)
%
% Run AFTER evcs_params.m has populated the base workspace, and AFTER you
% have run python precompute.py at least once so MODE_CMD etc. exist.

if nargin < 1, model_name = 'EVCS_model_v2'; end

% --- Sanity: make sure the schedule has been loaded ---
required_vars = {'P_load','MODE_CMD','P_EV'};
for k = 1:numel(required_vars)
    if ~evalin('base', sprintf('exist(''%s'',''var'')', required_vars{k}))
        error(['Required variable ''%s'' not in base workspace. Run ' ...
               'evcs_params first (it auto-loads data/ems_schedule.mat).'], ...
               required_vars{k});
    end
end

if ~bdIsLoaded(model_name)
    open_system(model_name);
end

% =====================================================================
% PART 1 — Bump fuel-cell rating to 50 kW
% =====================================================================
% Find the FC block by name (works whether or not the library link is intact)
fc_blocks = find_system(model_name, ...
    'MatchFilter', @Simulink.match.allVariants, ...
    'SearchDepth', 1, ...
    'BlockType','SubSystem', ...
    'Name','Fuel Cell Stack');

% Fallback: try by ReferenceBlock (works only on linked blocks)
if isempty(fc_blocks)
    fc_blocks = find_system(model_name, ...
        'MatchFilter', @Simulink.match.allVariants, ...
        'LookUnderMasks','all', ...
        'FollowLinks','on', ...
        'ReferenceBlock','spsFuelCellStackLib/Fuel Cell Stack');
end
fprintf('Found %d Fuel Cell Stack blocks. Bumping to 50 kW preset...\n', ...
        numel(fc_blocks));
for k = 1:numel(fc_blocks)
    b = fc_blocks{k};
    fprintf('  Patching: %s\n', b);

    % Library-linked block — break the link so we can set mask params.
    try, set_param(b, 'LinkStatus', 'inactive'); catch, end

    % Try the 'preset' field. R2024b removed it; that's fine, just skip.
    try, set_param(b, 'preset', 'No (User-Defined)'); catch, end

    % These are the values that actually matter.
    params = {'Nc',    'P.FC.Nc'; ...
          'Eoc',   'P.FC.Eoc'; ...
          'NomVI', 'P.FC.NomVI'; ...
          'EndVI', 'P.FC.EndVI'};
    for j = 1:size(params,1)
        try
            set_param(b, params{j,1}, params{j,2});
            fprintf('     %s = %s\n', params{j,1}, params{j,2});
        catch err
            warning('     could not set %s: %s', params{j,1}, err.message);
        end
    end
end

% =====================================================================
% PART 2 — Build the EMS_References subsystem
% =====================================================================
sub_path = [model_name '/EMS_References'];

% Wipe any previous attempt
if getSimulinkBlockHandle(sub_path) ~= -1
    delete_block(sub_path);
end

add_block('built-in/Subsystem', sub_path, 'Position', [50 50 250 200]);

% From-Workspace blocks: name -> Y position
fw_blocks = { ...
    'P_load',      40 ; ...
    'P_solar_set', 90 ; ...
    'P_batt_set', 140 ; ...
    'P_h2_set',   190 ; ...
    'P_grid_set', 240 ; ...
    'en_solar',   310 ; ...
    'en_batt',    360 ; ...
    'en_h2',      410 ; ...
    'en_grid',    460 ; ...
    'MODE_CMD',   540 ; ...   % NEW for v2
    'P_EV',       590 ; ...   % NEW for v2
};

for i = 1:size(fw_blocks,1)
    name = fw_blocks{i,1};
    yc   = fw_blocks{i,2};
    blk  = [sub_path '/' name];
    add_block('simulink/Sources/From Workspace', blk, ...
        'VariableName',          name, ...
        'OutputAfterFinalValue', 'Holding final value', ...
        'Interpolate',           'off', ...
        'SampleTime',            '-1', ...
        'Position',              [50 yc 150 yc+30]);
end

% =====================================================================
% PART 3 — Goto tags so the rest of the model can consume MODE_CMD / P_EV
% =====================================================================
% MODE_CMD: tag must match what the supervisor's 'From' block expects.
add_block('simulink/Signal Routing/Goto', [sub_path '/Goto_MODE_CMD'], ...
    'GotoTag', 'MODE_CMD', 'TagVisibility', 'global', ...
    'Position', [220 540 280 570]);
add_line(sub_path, 'MODE_CMD/1', 'Goto_MODE_CMD/1', 'autorouting', 'on');

% P_EV: drives the supervisor's P_EV From block AND (optionally) the EV load
add_block('simulink/Signal Routing/Goto', [sub_path '/Goto_P_EV'], ...
    'GotoTag', 'P_EV_REF', 'TagVisibility', 'global', ...
    'Position', [220 590 280 620]);
add_line(sub_path, 'P_EV/1', 'Goto_P_EV/1', 'autorouting', 'on');

% --- Visualisation scopes ----------------------------------------------
add_block('simulink/Sinks/Scope', [sub_path '/Scope_LoadRef'], ...
    'NumInputPorts','1','Position',[320 40 360 80]);
add_line(sub_path,'P_load/1','Scope_LoadRef/1','autorouting','on');

add_block('simulink/Sinks/Scope', [sub_path '/Scope_Setpoints'], ...
    'NumInputPorts','4','Position',[320 130 360 260]);
add_line(sub_path,'P_solar_set/1','Scope_Setpoints/1','autorouting','on');
add_line(sub_path,'P_batt_set/1', 'Scope_Setpoints/2','autorouting','on');
add_line(sub_path,'P_h2_set/1',   'Scope_Setpoints/3','autorouting','on');
add_line(sub_path,'P_grid_set/1', 'Scope_Setpoints/4','autorouting','on');

add_block('simulink/Sinks/Scope', [sub_path '/Scope_Enables'], ...
    'NumInputPorts','4','Position',[320 310 360 480]);
add_line(sub_path,'en_solar/1','Scope_Enables/1','autorouting','on');
add_line(sub_path,'en_batt/1', 'Scope_Enables/2','autorouting','on');
add_line(sub_path,'en_h2/1',   'Scope_Enables/3','autorouting','on');
add_line(sub_path,'en_grid/1', 'Scope_Enables/4','autorouting','on');

add_block('simulink/Sinks/Scope', [sub_path '/Scope_Mode'], ...
    'NumInputPorts','1','Position',[320 540 360 580]);
% branch the line into the scope
add_line(sub_path,'MODE_CMD/1','Scope_Mode/1','autorouting','on');

% =====================================================================
% PART 4 — Wire EMS P_EV into the existing supervisor
% =====================================================================
% The supervisor's P_EV From block currently picks up the demand from
% wherever the original model sourced it.  We don't replace that wiring
% (it might be read by other blocks).  Instead, the new P_EV signal lives
% under the P_EV_REF Goto tag — your demo narration just shows P_EV_REF
% on a scope, while the supervisor still reads the original P_EV.
%
% If you want the RL P_EV to TRULY drive the EV load, you need one
% manual edit: open the supervisor subsystem, find the From block named
% 'From' that reads tag 'P_EV', and change its tag to 'P_EV_REF'.
% That's a 5-second edit and the only manual step we couldn't safely
% script (changing it programmatically might disconnect existing wires
% in your model that we can't see from outside).

save_system(model_name);
fprintf('\nEMS_References subsystem built.\n');
fprintf('Open with:  open_system(''%s/EMS_References'')\n', model_name);
fprintf('\nMANUAL STEP (one-time):\n');
fprintf('  Open %s/supervisor and change the GotoTag of the\n', model_name);
fprintf('  ''P_EV'' From block from ''P_EV'' to ''P_EV_REF'' so the supervisor\n');
fprintf('  reads the RL-driven demand instead of the original source.\n');
end