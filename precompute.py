"""
precompute.py
=============
Run forecast + RL rollout once and dump everything Streamlit AND the
v2 Simulink supervisor need to disk:

  data/precomputed.pkl   for the dashboard
  data/ems_schedule.mat  for Simulink (now includes MODE_CMD and P_EV)

Usage:
    python precompute.py "Roorkee, India" 3 2025-05-01
"""
import os, sys, pickle
import pandas as pd

from forecasting          import run_forecast
from rl_agent             import rollout
from simulink_bridge      import build_signals, save_mat
from bridge_to_supervisor import build_supervisor_signals


def precompute(location, days, start_date,
               models_dir="models",
               out_dir="data",
               slot_hours=4.0,
               sample_dt=10.0,
               ev_charger_kw=50.0):
    os.makedirs(out_dir, exist_ok=True)

    print(f"==> forecast {location} / {days} days / {start_date}")
    forecast_df = run_forecast(
        location, days, start_date,
        models_dir=models_dir,
        slot_demand_csv=os.path.join(models_dir, "slot_demand1.csv"))

    print("==> RL rollout")
    rollout_df = rollout(forecast_df,
                         model_path=os.path.join(models_dir, "evcs_dqn"))

    # Make pandas datetimes pickle-portable across versions
    for df in (forecast_df, rollout_df):
        for col in list(df.columns):
            s = df[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                try:
                    if pd.api.types.is_datetime64tz_dtype(s):
                        s = s.dt.tz_localize(None)
                    df[col] = s.astype("datetime64[ns]")
                except Exception:
                    df[col] = df[col].astype(str)

    # ---- Simulink reference signals (existing) ---------------------------
    print("==> Simulink reference signals")
    signals = build_signals(rollout_df, slot_hours=slot_hours,
                            sample_dt=sample_dt)

    # ---- Supervisor override signals (NEW for v2 model) ------------------
    print("==> supervisor MODE_CMD + P_EV")
    sup_signals = build_supervisor_signals(rollout_df,
                                       slot_hours=slot_hours,
                                       sample_dt=sample_dt,
                                       ev_charger_kw=ev_charger_kw,
                                       slot_seconds_override=30.0)  # match bridge
    signals.update(sup_signals)

    # ---- Write .mat ------------------------------------------------------
    mat_path = save_mat(signals,
                        os.path.join(out_dir, "ems_schedule.mat"),
                        rollout_df=rollout_df)

    # ---- Write pickle for Streamlit --------------------------------------
    bundle = {
        "forecast_df": forecast_df,
        "rollout_df":  rollout_df,
        "signals":     signals,
        "mat_path":    mat_path,
        "inputs":      {"location": location, "days": days,
                        "start_date": start_date},
    }
    out = os.path.join(out_dir, "precomputed.pkl")
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"==> wrote {out}")
    return out


if __name__ == "__main__":
    location   = sys.argv[1] if len(sys.argv) > 1 else "Roorkee, India"
    days       = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    start_date = sys.argv[3] if len(sys.argv) > 3 else "2025-05-01"
    precompute(location, days, start_date)