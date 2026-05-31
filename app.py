"""
app.py
======
Streamlit demo dashboard.  Drives the whole pipeline:
    location, days, start_date  →  forecast  →  RL rollout
    →  Simulink reference signals  →  (optional) run Simulink

Run:
    streamlit run app.py
"""
from __future__ import annotations

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import os
import sys
import json
import traceback
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# local imports
sys.path.insert(0, os.path.dirname(__file__))
from forecasting import run_forecast
from rl_agent    import rollout, train
from simulink_bridge import build_signals, save_mat, run_simulink


# ---------------------------------------------------------------------------
# page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RL-EMS Demo", layout="wide")
st.title("RL-based Energy Management — Software Demo")
st.caption("Forecast → RL strategy → Simulink reference signals")

# ---------------------------------------------------------------------------
# sidebar inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1.  Inputs")
    location  = st.text_input("Location",  value="Roorkee, India")
    n_days    = st.number_input("Number of days", min_value=1, max_value=14, value=3)
    start_dt  = st.date_input("Start date", value=date(2025, 5, 1))

    st.divider()
    st.header("2.  Models")
    model_dir = st.text_input("Models folder", value="models")
    rl_path   = st.text_input("RL policy",
                              value=os.path.join("models", "evcs_dqn"))

    st.divider()
    st.header("3.  Simulink")
    sim_stop_time = st.number_input(
        "MATLAB sim StopTime (s)",
        min_value=10.0, max_value=10000.0, value=540.0, step=10.0,
        help="540 s = full 3-day schedule at compressed timebase. "
             "Try 90 for a quick demo run.",
    )

    run_btn   = st.button("▶  Run pipeline", type="primary")
    sim_btn   = st.button("🔌  Run Simulink in MATLAB",
                          help="Launches MATLAB on macOS and runs demo() "
                               "against the latest schedule.")


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------
ss = st.session_state
for k in ("forecast_df", "rollout_df", "signals", "mat_path"):
    ss.setdefault(k, None)


# ---------------------------------------------------------------------------
# pipeline execution
# ---------------------------------------------------------------------------
if run_btn:
    try:
        import subprocess, pickle, sys
        sd_str = start_dt.strftime("%Y-%m-%d")
        venv_python = "/Users/akshat/Desktop/ems_demo/.venv/bin/python"
        cmd = [venv_python, "precompute.py", location, str(int(n_days)), sd_str]

        st.write(f"Running: `{' '.join(cmd)}`")
        with st.spinner(f"Running pipeline for {location} / {n_days} days / {sd_str} ..."):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        # Always show the logs so we can see what happened
        with st.expander("pipeline log (stdout)", expanded=(result.returncode != 0)):
            st.code(result.stdout or "(no stdout)")
        if result.stderr:
            with st.expander("pipeline log (stderr)", expanded=(result.returncode != 0)):
                st.code(result.stderr)

        if result.returncode != 0:
            st.error(f"Pipeline failed with exit code {result.returncode}")
        elif not os.path.exists("data/precomputed.pkl"):
            st.error("Pipeline reported success but data/precomputed.pkl was not written.")
        else:
            with open("data/precomputed.pkl", "rb") as f:
                bundle = pickle.load(f)
            ss.forecast_df = bundle["forecast_df"]
            ss.rollout_df  = bundle["rollout_df"]
            ss.signals     = bundle["signals"]
            ss.mat_path    = bundle["mat_path"]
            st.success(f"{location} / {n_days} days / {sd_str} → {len(ss.forecast_df)} slots")
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.code(traceback.format_exc())
# if run_btn:
#     try:
#         with st.spinner("1/3  Running forecast pipeline ..."):
#             ss.forecast_df = run_forecast(
#                 location=location,
#                 number_of_days=int(n_days),
#                 start_date=start_dt.strftime("%Y-%m-%d"),
#                 models_dir=model_dir,
#                 slot_demand_csv=os.path.join(model_dir, "slot_demand1.csv"),
#             )
#         st.success(f"Forecast OK — {len(ss.forecast_df)} slots")

#         with st.spinner("2/3  RL rollout ..."):
#             ss.rollout_df = rollout(ss.forecast_df, model_path=rl_path)
#         st.success("RL strategy ready")

#         with st.spinner("3/3  Building Simulink reference signals ..."):
#             ss.signals = build_signals(ss.rollout_df, slot_hours=4.0,
#                                        sample_dt=float(sample_dt))
#             ss.mat_path = save_mat(ss.signals,
#                                    os.path.join("data", "ems_schedule.mat"),
#                                    rollout_df=ss.rollout_df)
#         st.success(f"Schedule written → {ss.mat_path}")

#     except FileNotFoundError as e:
#         st.error(f"Missing model file: {e}\n"
#                  f"Expected files in **{model_dir}/**: ffnn_model.h5, "
#                  f"X_scaler.save, y_scaler.save, linear_model_mt.pkl, "
#                  f"slot_demand1.csv, evcs_dqn.zip")
#     except Exception as e:
#         st.error(f"Pipeline failed: {e}")
#         st.code(traceback.format_exc())


if sim_btn:
    try:
        from matlab_runner import trigger_demo
        proj = os.path.dirname(os.path.abspath(__file__))
        with st.spinner("Launching MATLAB ..."):
            result = trigger_demo(proj, stop_time=float(sim_stop_time))
        if result["ok"]:
            st.success(result["message"])
            st.caption(f"Trigger script: `{result['script_path']}`")
        else:
            st.error(result["message"])
    except Exception as e:
        st.error(f"Failed to launch MATLAB: {e}")
        st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Visual layout: 3 tabs (forecast / RL strategy / Simulink references)
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📈  Forecast", "🤖  RL Strategy", "🔌  Simulink Signals"])

with tab1:
    if ss.forecast_df is not None:
        st.subheader("Slot-level forecast")
        st.dataframe(ss.forecast_df.round(2), use_container_width=True)

        fig, ax = plt.subplots(figsize=(11, 4))
        ix = np.arange(len(ss.forecast_df))
        ax.bar(ix - 0.18, ss.forecast_df["cum_energy"],     width=0.36, label="Solar (kWh)")
        ax.bar(ix + 0.18, ss.forecast_df["Forecast_Demand"], width=0.36, label="Demand (kWh)")
        ax.set_xticks(ix)
        ax.set_xticklabels(ss.forecast_df["slot_label"], rotation=45, ha="right",
                           fontsize=8)
        ax.set_ylabel("kWh")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        st.pyplot(fig)
    else:
        st.info("Click **Run pipeline** to generate a forecast.")

with tab2:
    if ss.rollout_df is not None:
        st.subheader("Per-slot dispatch decided by the RL agent")
        st.dataframe(ss.rollout_df.round(2), use_container_width=True)

        # stacked-bar energy flow
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ix = np.arange(len(ss.rollout_df))
        s = ss.rollout_df
        ax.bar(ix, s["solar_used"],                                     label="Solar",  color="#FFD700")
        ax.bar(ix, s["battery"],   bottom=s["solar_used"],              label="Battery", color="#32CD32")
        ax.bar(ix, s["hydrogen"],  bottom=s["solar_used"] + s["battery"],
                                                                        label="Hydrogen", color="#87CEEB")
        ax.bar(ix, s["grid"], bottom=s["solar_used"] + s["battery"] + s["hydrogen"],
                                                                        label="Grid", color="#FF6347")
        ax.plot(ix, s["demand"], "k--o", linewidth=1.5, label="Demand")
        ax.set_xlabel("slot"); ax.set_ylabel("kWh"); ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        st.pyplot(fig)

        # quick KPI strip
        c1, c2, c3, c4 = st.columns(4)
        total = max(s["demand"].sum(), 1e-9)
        renew = (s["solar_used"] + s["battery"] + s["hydrogen"]).sum()
        c1.metric("Total demand (kWh)", f"{s['demand'].sum():.1f}")
        c2.metric("Renewable share",     f"{100 * renew / total:.1f}%")
        c3.metric("Grid imported (kWh)", f"{s['grid'].sum():.1f}")
        c4.metric("Operating cost (₹)", f"{s['cost'].sum():.0f}")
    else:
        st.info("RL rollout will appear here after **Run pipeline**.")

with tab3:
    if ss.signals is not None:
        st.subheader("Reference signals exported to Simulink")
        # show first few seconds of every signal
        cols = list(ss.signals)
        N = min(len(ss.signals[cols[0]].time), 1500)
        df_view = pd.DataFrame({"t [s]": ss.signals[cols[0]].time[:N]})
        for c in cols:
            df_view[c] = ss.signals[c].data[:N]

        fig, axes = plt.subplots(len(cols), 1, sharex=True,
                                 figsize=(11, 1.4 * len(cols)))
        for ax, c in zip(axes, cols):
            ax.plot(df_view["t [s]"], df_view[c])
            ax.set_ylabel(c, fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("time [s] from sim start")
        st.pyplot(fig)

        st.caption(f".mat file: `{ss.mat_path}`")
        with open(ss.mat_path, "rb") as f:
            st.download_button("Download ems_schedule.mat",
                               data=f.read(),
                               file_name="ems_schedule.mat")
    else:
        st.info("Simulink signals will appear after **Run pipeline**.")