%% evcs_params.m
% Workspace parameters for the Green-Hydrogen EV Charging Station model.
% Run this BEFORE opening / simulating the Simulink model.
%
% v3 — fuel cell bumped from 6 kW to 50 kW so it can supply the 50 kW EV
%      charger when it's the dispatched source. This matches what the
%      RL agent assumes (P_FC_MAX_KW = 80 kW headroom in rl_agent.py).
%
% Usage:  >> evcs_params      % loads everything into base workspace
% ----------------------------------------------------------------------

clc;

%% ---------- 1. SOLVER / GRID FUNDAMENTALS --------------------------
P.f_grid     = 50;
P.Ts_power   = 5e-6;
P.f_sw       = 10e3;
P.Vdc_bus    = 750;
P.Vgrid_LL   = 415;

%% ---------- 2. PV ARRAY  (target: ~50 kW @ STC) --------------------
P.PV.Pm        = 427.5;
P.PV.Voc       = 50;
P.PV.Isc       = 10;
P.PV.Vmp       = 45;
P.PV.Imp       = 9.5;
P.PV.Nser      = 10;
P.PV.Npar      = 12;
P.PV.P_rated   = P.PV.Nser*P.PV.Npar*P.PV.Pm;       % = 51.30 kW

%% ---------- 3. BESS  (target: 100 kWh, 30 kW peak) -----------------
P.BESS.NomV    = 400;
P.BESS.NomQ    = 250;
P.BESS.E_rated = P.BESS.NomV*P.BESS.NomQ/1000;      % = 100 kWh
P.BESS.SOC0    = 60;
P.BESS.SOC_min = 10;
P.BESS.SOC_max = 90;
P.BESS.P_max   = 30e3;
P.BESS.I_max   = P.BESS.P_max / P.BESS.NomV;
P.BESS.R_int   = 0.05;

%% ---------- 4. PEM FUEL CELL  (50 kW target) -----------------------
% Original block was 6 kW (65 cells, Vnom=45, Inom=133.3A → ~6 kW).
% We scale cells & current to reach ~50 kW. Same per-cell polarisation,
% just more cells in series + current scaled to deliver more power.
%
% Format reminder for the Simscape PEMFC mask:
%   Eoc   = [Voc, V1]    open-circuit V, voltage at 1 A
%   NomVI = [Inom, Vnom]
%   EndVI = [Iend, Vend]
P.FC.Nc        = 540;                     % was 65 — scale stack length
P.FC.Voc       = 540;                     % V (open circuit, scales with Nc)
P.FC.V1        = 525;                     % V (at 1 A)
P.FC.Vnom      = 375;                     % V at nominal current
P.FC.Inom      = 133.3;                   % A (per-cell area unchanged)
P.FC.Vend      = 308;                     % V at end of linear region
P.FC.Iend      = 225;                     % A
P.FC.P_rated   = P.FC.Vnom*P.FC.Inom;     % = 50 kW
P.FC.AirFlow_lpm = 2500;                  % lpm — scaled with stack power

P.FC.Eoc       = [P.FC.Voc  P.FC.V1];     % [540 525]
P.FC.NomVI     = [P.FC.Inom P.FC.Vnom];   % [133.3 375]
P.FC.EndVI     = [P.FC.Iend P.FC.Vend];   % [225 308]

% Operational protection (used by supervisor)
P.FC.dPdt_max  = 25e3;                    % W/s ramp limit (scaled for 50 kW)
P.FC.t_minON   = 30;                      % s minimum run-time

%% ---------- 5. EV BATTERY  (target: 400 V pack, 75 kWh) ------------
P.EV.NomV      = 400;
P.EV.NomQ      = 188;
P.EV.SOC0      = 30;
P.EV.P_charge  = 50e3;

%% ---------- 6. POWER ELECTRONICS ----------------------------------
P.PE.MOSFET_Ron = 1e-3;
P.PE.IGBT_Ron   = 1e-3;
P.PE.Diode_Vf   = 0.8;
P.PE.Snub_R     = 1e5;
P.PE.Snub_C     = inf;

P.PE.L_pv       = 5e-3;
P.PE.L_fc       = 8e-3;
P.PE.L_bess     = 3e-3;
P.PE.L_ev       = 3e-3;
P.PE.C_dcbus    = 4700e-6;
P.PE.C_pv       = 470e-6;
P.PE.C_fc       = 470e-6;
P.PE.C_ev       = 1000e-6;

%% ---------- 7. GRID INTERFACE -------------------------------------
P.GRID.V_LL     = 415;
P.GRID.f        = 50;
P.GRID.SCC_MVA  = 100;
P.GRID.XR       = 7;
P.GRID.L_filter = 5e-3;
P.GRID.R_filter = 0.1;
P.GRID.P_max    = 30e3;

%% ---------- 8. SUPERVISOR THRESHOLDS ------------------------------
P.SUP.Vbus_ref      = 750;
P.SUP.Vbus_dead     = 5;
P.SUP.SOC_low       = 20;
P.SUP.SOC_high      = 85;
P.SUP.PV_min        = 1e3;
P.SUP.PV_surplus    = 2e3;
P.SUP.dwell_time    = 2;
P.SUP.FC_enable     = 1;

%% ---------- 9. EMS schedule loader (NEW for RL integration) -------
% If a precomputed .mat schedule exists, load it.  This populates
% MODE_CMD, P_EV, P_load, P_solar_set, etc. into base workspace, where
% the new From-Workspace blocks (added by build_ems_references.m) will
% pick them up.
sched_path = fullfile('data', 'ems_schedule.mat');
if isfile(sched_path)
    fprintf('[evcs_params] loading EMS schedule from %s ...\n', sched_path);
    evalin('base', sprintf('load(''%s'')', sched_path));
else
    fprintf('[evcs_params] note: no %s found; run python precompute.py first.\n', sched_path);
end

%% ---------- assign to base WS in flat names too -------------------
assignin('base','P',P);
fn = fieldnames(P);
for k = 1:numel(fn)
    sub = P.(fn{k});
    if isstruct(sub)
        sfn = fieldnames(sub);
        for j = 1:numel(sfn)
            assignin('base', [fn{k} '_' sfn{j}], sub.(sfn{j}));
        end
    else
        assignin('base', fn{k}, sub);
    end
end

fprintf('\n======= EVCS parameters loaded =======\n');
fprintf('  PV array     : %5.1f kW   (Vmp=%g V, Imp=%g A)\n', ...
        P.PV.P_rated/1e3, P.PV.Vmp*P.PV.Nser, P.PV.Imp*P.PV.Npar);
fprintf('  BESS         : %5.1f kWh  (%g V, %g Ah, SoC0=%g %%)\n', ...
        P.BESS.E_rated, P.BESS.NomV, P.BESS.NomQ, P.BESS.SOC0);
fprintf('  Fuel Cell    : %5.2f kW   (%g cells, Inom=%g A @ Vnom=%g V)\n', ...
        P.FC.P_rated/1e3, P.FC.Nc, P.FC.Inom, P.FC.Vnom);
fprintf('  EV Battery   : %5.1f kWh  (%g V, %g Ah)\n', ...
        P.EV.NomV*P.EV.NomQ/1000, P.EV.NomV, P.EV.NomQ);
fprintf('  DC bus       : %g V       (Cbus=%g uF)\n', ...
        P.Vdc_bus, P.PE.C_dcbus*1e6);
fprintf('  Grid         : %g V LL, %g Hz\n', P.GRID.V_LL, P.GRID.f);
fprintf('  (No electrolyser / no H2 tank in this Simscape model)\n');
fprintf('=====================================\n\n');