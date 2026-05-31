"""
bridge_to_supervisor.py
=======================
Convert the per-slot RL rollout into the time-domain signals the
existing v2 Simulink supervisor expects:

    MODE_CMD   (int32 timeseries) — overrides the rule-based logic
                 0 = AUTO     1 = PV         2 = BESS      3 = FC
                 4 = GRID     5 = PV->BESS

    P_EV       (double timeseries, watts)

This version implements TIME-MULTIPLEXED dispatch within each slot:
if a slot's RL decision allocates 83 kWh to solar and 95 kWh to grid,
the supervisor runs solar for the first ~47% of the slot then grid for
the remaining ~53%.  Energy delivered per source matches the dashboard
exactly, under the assumption that demand is uniform across the slot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from simulink_bridge import Signal, _stairstep


# Map from rollout column name to supervisor mode integer
COLUMN_TO_MODE = {
    "solar_used": 1,   # PV
    "battery":    2,   # BESS discharge
    "hydrogen":   3,   # Fuel cell
    "grid":       4,   # Grid
}

MODE_BATT_CHARGE_ONLY = 5     # special "PV -> BESS" mode (low-demand slots)


def _slot_mode_segments(row: pd.Series,
                        demand_threshold: float = 5.0):
    """
    Return a list of (mode, time_fraction) pairs describing how the slot
    should be partitioned in time among the contributing sources.

    Sum of fractions = 1.0.  Single-source slots produce one segment.
    """
    demand = float(row.get("demand", 0.0))

    # Special low-demand "battery charging" slot
    if demand < demand_threshold and float(row.get("batt_charge", 0.0)) > 1.0:
        return [(MODE_BATT_CHARGE_ONLY, 1.0)]

    contribs = []
    for col, mode in COLUMN_TO_MODE.items():
        kwh = float(row.get(col, 0.0))
        if kwh > 1e-3:
            contribs.append((mode, kwh))

    if not contribs:
        return [(0, 1.0)]   # AUTO; supervisor decides

    total = sum(k for _, k in contribs)
    # Sort biggest first so the staircase looks ordered on the scope
    contribs.sort(key=lambda x: -x[1])
    return [(m, k / total) for m, k in contribs]


def build_supervisor_signals(rollout_df: pd.DataFrame,
                             slot_hours: float = 4.0,
                             sample_dt:  float = 10.0,
                             ev_charger_kw: float = 50.0,
                             ev_demand_unit: str = "watts",
                             slot_seconds_override: float = None,
                             ):
    """
    Build the time-multiplexed MODE_CMD and the (constant-per-slot) P_EV.

    `slot_seconds_override` — if set, each slot occupies that many seconds
    of simulated time (the demo compression).  Otherwise computed from
    `slot_hours` (real-world 4 h = 14400 s).
    """
    needed = {"demand", "solar_used", "battery", "hydrogen", "grid"}
    missing = needed - set(rollout_df.columns)
    if missing:
        raise ValueError(f"rollout_df missing columns: {missing}")

    slot_seconds = (slot_seconds_override if slot_seconds_override is not None
                    else slot_hours * 3600.0)

    # Build piecewise time-multiplexed MODE_CMD
    times = []
    modes = []
    t_cursor = 0.0
    for _, row in rollout_df.iterrows():
        for mode, frac in _slot_mode_segments(row):
            seg_dur = frac * slot_seconds
            n_samples = max(1, int(round(seg_dur / sample_dt)))
            for i in range(n_samples):
                times.append(t_cursor + i * sample_dt)
                modes.append(float(mode))
            t_cursor += n_samples * sample_dt

    mode_signal = Signal(time=np.array(times),
                         data=np.array(modes, dtype=float))

    # P_EV: average power over the whole slot, constant within
    p_ev_kw = (rollout_df["demand"].values / slot_hours).clip(0, ev_charger_kw)
    p_ev_w = p_ev_kw * 1000.0 if ev_demand_unit == "watts" else p_ev_kw
    p_ev_signal = _stairstep(p_ev_w, slot_seconds, sample_dt)

    return {
        "MODE_CMD": mode_signal,
        "P_EV":     p_ev_signal,
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.DataFrame({
        "demand":      [180,    88,   113,   77,    163],
        "solar_used":  [ 84,    88,   113,   26,      0],
        "battery":     [  0,     0,     0,    0,      0],
        "hydrogen":    [  0,     0,     0,   51,    160],
        "grid":        [ 96,     0,     0,    0,      0],
        "batt_charge": [  0,     0,     0,    0,      0],
    })
    sigs = build_supervisor_signals(df, slot_hours=4.0,
                                    sample_dt=1.0,
                                    slot_seconds_override=30.0)
    print(f"total samples: {len(sigs['MODE_CMD'].time)}")
    print("\nSlot 0 (split solar / grid 84/96 = ~47%/53%):")
    for t, m in zip(sigs['MODE_CMD'].time[:32], sigs['MODE_CMD'].data[:32]):
        print(f"  t={t:5.1f}s  mode={int(m)}")
    print("\nSlot 3 (split solar / hydrogen 26/51 = ~34%/66%):")
    for t, m in zip(sigs['MODE_CMD'].time[90:122], sigs['MODE_CMD'].data[90:122]):
        print(f"  t={t:5.1f}s  mode={int(m)}")