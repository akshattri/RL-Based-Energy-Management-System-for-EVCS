"""
simulink_bridge.py
==================
Converts per-slot RL decisions (`rollout_df` from rl_agent.rollout) into the
time-domain reference signals expected by the Simulink model, writes them
into a .mat file, and (optionally) drives the simulation through the
MATLAB Engine for Python.

Signals exported to MATLAB base workspace (each is a Simulink.SimulationData
.Dataset / `timeseries`-compatible struct so 'From Workspace' blocks accept
it directly):

    P_load        load reference                              [kW]
    P_solar_set   how much solar should deliver to the bus    [kW]
    P_batt_set    battery setpoint (+discharge / -charge)     [kW]
    P_h2_set      fuel-cell setpoint (+) / electrolyser (-)   [kW]
    P_grid_set    grid setpoint                               [kW]
    en_solar      0/1 enable for the solar converter
    en_batt       0/1 enable for the battery converter
    en_h2         0/1 enable for the fuel-cell converter
    en_grid       0/1 enable for the grid breaker
    schedule      raw rollout dataframe (as struct), for logging

The .mat file can also be opened standalone — see matlab/run_simulation.m.

USAGE
-----
    from simulink_bridge import build_signals, save_mat, run_simulink

    sigs = build_signals(rollout_df, slot_hours=4, sample_dt=1.0)
    save_mat(sigs, "data/ems_schedule.mat")

    # if MATLAB engine is available:
    run_simulink("EVCS_model__1", "data/ems_schedule.mat",
                 stop_time=len(rollout_df) * 4 * 3600)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.io import savemat


# ---------------------------------------------------------------------------
# 1. timeseries construction
# ---------------------------------------------------------------------------
@dataclass
class Signal:
    """Plain (time, value) pair that Simulink 'From Workspace' can read."""
    time: np.ndarray            # shape (N,)  in seconds
    data: np.ndarray            # shape (N,)  in engineering units

    def as_struct(self) -> dict:
        return {
            "time":    self.time.reshape(-1, 1),
            "signals": {"values":     self.data.reshape(-1, 1),
                        "dimensions": np.array([[1]], dtype=np.int32)},
        }


def _stairstep(values: np.ndarray, slot_seconds: float,
               sample_dt: float = 1.0) -> Signal:
    """
    Repeat each slot value at `sample_dt`-second cadence so a Simulink
    'From Workspace' block (with ZOH interpolation) reproduces the staircase.
    """
    samples_per_slot = max(1, int(round(slot_seconds / sample_dt)))
    data = np.repeat(values.astype(float), samples_per_slot)
    time = np.arange(len(data)) * sample_dt
    return Signal(time=time, data=data)


def build_signals(rollout_df: pd.DataFrame,
                  slot_hours: float = 4.0,
                  sample_dt:  float = 1.0,
                  power_unit_kw: bool = True) -> Dict[str, Signal]:
    """
    Build all Simulink reference signals from the per-slot rollout.

    rollout_df columns expected (kWh per slot):
        demand, solar_used, battery, hydrogen, grid,
        batt_charge, h2_charge, curtailed
    """
    needed = {"demand", "solar_used", "battery", "hydrogen", "grid"}
    missing = needed - set(rollout_df.columns)
    if missing:
        raise ValueError(f"rollout_df missing columns: {missing}")

    # slot_seconds = slot_hours * 3600.0
    slot_seconds = 30.0    # demo compression: 4h schedule slot -> 30s sim time
    # convert kWh per slot → average kW during the slot (×1 if user prefers kW already)
    k = (1.0 / slot_hours) if power_unit_kw else 1.0

    P_load   = rollout_df["demand"].values     * k
    P_solar  = rollout_df["solar_used"].values * k                # to load
    # battery: +discharge to load, -charge from solar surplus
    P_batt   = (rollout_df["battery"].values
                - rollout_df.get("batt_charge", 0).values) * k
    # H2: +discharge to load, -charge (electrolyser pulling power)
    P_h2     = (rollout_df["hydrogen"].values
                - rollout_df.get("h2_charge", 0).values / 0.7) * k    # /eta_ely → elec input
    P_grid   = rollout_df["grid"].values * k

    # enable flags: a source is "on" whenever it contributes meaningfully
    eps = 1e-3
    en_solar = (np.abs(P_solar) > eps).astype(float)
    en_batt  = (np.abs(P_batt)  > eps).astype(float)
    en_h2    = (np.abs(P_h2)    > eps).astype(float)
    en_grid  = (np.abs(P_grid)  > eps).astype(float)

    sigs = {
        "P_load":      _stairstep(P_load,   slot_seconds, sample_dt),
        "P_solar_set": _stairstep(P_solar,  slot_seconds, sample_dt),
        "P_batt_set":  _stairstep(P_batt,   slot_seconds, sample_dt),
        "P_h2_set":    _stairstep(P_h2,     slot_seconds, sample_dt),
        "P_grid_set":  _stairstep(P_grid,   slot_seconds, sample_dt),
        "en_solar":    _stairstep(en_solar, slot_seconds, sample_dt),
        "en_batt":     _stairstep(en_batt,  slot_seconds, sample_dt),
        "en_h2":       _stairstep(en_h2,    slot_seconds, sample_dt),
        "en_grid":     _stairstep(en_grid,  slot_seconds, sample_dt),
    }
    return sigs


# ---------------------------------------------------------------------------
# 2. .mat file output
# ---------------------------------------------------------------------------
def save_mat(signals: Dict[str, Signal], path: str,
             rollout_df: Optional[pd.DataFrame] = None) -> str:
    """Persist all signals + the raw schedule to a .mat the model can read."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    payload = {name: sig.as_struct() for name, sig in signals.items()}

    if rollout_df is not None:
        # store schedule as a struct of arrays so MATLAB sees it as a struct
        sched = {c: rollout_df[c].values
                 for c in rollout_df.columns
                 if rollout_df[c].dtype.kind in "fiub"}
        payload["schedule"] = sched

    # also write meta-info that the dashboard / MATLAB script may need
    payload["meta"] = {
        "n_slots":  np.array([[len(next(iter(signals.values())).time)
                               // (signals[next(iter(signals))].data.size
                                   // len(signals[next(iter(signals))].data))]]),
        "duration_s": np.array([[float(next(iter(signals.values())).time[-1])]]),
    }

    savemat(path, payload, do_compression=True, oned_as="column")
    print(f"[bridge] wrote {path}  (signals: {list(signals)})")
    return path


# ---------------------------------------------------------------------------
# 3. (optional) drive Simulink directly from Python
# ---------------------------------------------------------------------------
def run_simulink(model_name: str,
                 mat_file: str,
                 stop_time: Optional[float] = None,
                 simulink_dir: Optional[str] = None,
                 logged_signals: tuple = ("P_load", "P_solar_set", "P_batt_set",
                                          "P_h2_set", "P_grid_set"),
                 ) -> Dict[str, np.ndarray]:
    """
    Start MATLAB, load the .mat into base workspace, run the Simulink model,
    return logged signals as {name: array}.

    Requires:  pip install matlabengine        (matched to your MATLAB version)
    """
    try:
        import matlab.engine
    except ImportError as e:
        raise RuntimeError(
            "matlab.engine not installed. Run:\n"
            "    cd <MATLAB>/extern/engines/python && python setup.py install\n"
            "or:  pip install matlabengine"
        ) from e

    print("[bridge] starting MATLAB engine ...")
    eng = matlab.engine.start_matlab()
    try:
        if simulink_dir:
            eng.cd(simulink_dir, nargout=0)

        eng.evalc(f"load('{mat_file}');")
        eng.evalc(f"open_system('{model_name}');")
        if stop_time is not None:
            eng.set_param(model_name, "StopTime", str(stop_time), nargout=0)

        print(f"[bridge] running sim '{model_name}' ...")
        eng.set_param(model_name, "SimulationCommand", "start", nargout=0)
        # block until done
        while eng.get_param(model_name, "SimulationStatus") != "stopped":
            pass

        out = {}
        for s in logged_signals:
            try:
                out[s] = np.asarray(eng.evalin("base", f"{s}.signals.values"))
            except Exception:
                pass
        print("[bridge] simulation finished.")
        return out
    finally:
        eng.quit()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # generate a synthetic schedule so this script can be tested without the RL
    n = 18
    df = pd.DataFrame({
        "demand":      np.linspace(40, 80, n),
        "solar_used":  np.maximum(0, np.sin(np.linspace(0, 6, n)) * 60),
        "battery":     np.where(np.arange(n) % 2, 20, 0),
        "hydrogen":    np.where(np.arange(n) % 3 == 0, 15, 0),
        "grid":        np.where(np.arange(n) % 2 == 0, 0, 10),
        "batt_charge": np.where(np.arange(n) % 4 == 0, 25, 0),
        "h2_charge":   np.where(np.arange(n) % 4 == 0, 10, 0),
        "curtailed":   np.zeros(n),
    })
    sigs = build_signals(df, slot_hours=4.0, sample_dt=10.0)
    save_mat(sigs, "data/ems_schedule.mat", rollout_df=df)
    print("OK")