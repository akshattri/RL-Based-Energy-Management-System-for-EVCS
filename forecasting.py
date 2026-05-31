"""
forecasting.py
==============
Forecasting pipeline for the EMS demo.

Produces a 4-hour-slot DataFrame with columns:
    slot_label       human-readable slot, e.g. "2025-05-01 04:00 - 08:00"
    slot_start       pd.Timestamp at slot start
    cum_energy       solar generation forecast for the slot (kWh, normalised)
    Forecast_Demand  demand forecast for the slot (kWh)
    real_solar       jittered "actual" solar (kWh) used by the RL env
    real_demand      jittered "actual" demand (kWh) used by the RL env

This version uses subprocess curl + on-disk JSON cache for HTTP calls,
because requests/urllib hang silently on some macOS+Anaconda setups even
when curl works.  Cached responses live in data/ and are reused on every
subsequent run with the same (lat, lon, start, end) — perfect for offline
demos.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.parse
import warnings
from datetime import datetime, timedelta
from math import ceil
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

LOCAL_TZ = "Asia/Kolkata"
FEATURES = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
CACHE_DIR = "data"


# ---------------------------------------------------------------------------
# 0. shared HTTP helper (uses curl, caches to disk)
# ---------------------------------------------------------------------------
def _http_get_json(url: str, params: dict, *,
                   cache_path: Optional[str] = None,
                   timeout: int = 30,
                   retries: int = 3,
                   user_agent: Optional[str] = None) -> dict | list:
    """
    GET → JSON via subprocess curl (Python's requests/urllib hang silently
    on some macOS arm64 + Anaconda setups; curl always works on those).

    If `cache_path` is given and the file exists, returns the cached payload
    immediately. On a successful fetch, writes the response to cache.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[http] cache hit: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    qs = urllib.parse.urlencode(params)
    full_url = f"{url}?{qs}" if qs else url
    cmd = ["curl", "-s", "--max-time", str(timeout)]
    if user_agent:
        cmd += ["-A", user_agent]
    cmd += [full_url]

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[http] attempt {attempt}/{retries}: {url}")
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=timeout + 5)
            if result.returncode != 0:
                raise RuntimeError(f"curl exit {result.returncode}: "
                                   f"{result.stderr.strip()}")
            if not result.stdout.strip():
                raise RuntimeError("empty response body")
            data = json.loads(result.stdout)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(data, f)
                print(f"[http] cached → {cache_path}")
            return data
        except Exception as e:
            last_err = e
            print(f"[http] {type(e).__name__}: {e}")
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(f"GET {url} failed after {retries} tries. "
                       f"Last error: {last_err}")


# ---------------------------------------------------------------------------
# 1. helpers
# ---------------------------------------------------------------------------
def get_end_date(start_date_str: str, number_of_days: int) -> str:
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    return (start_date + timedelta(days=number_of_days)).strftime("%Y-%m-%d")


def get_coordinates(place_name: str) -> Tuple[Optional[float], Optional[float]]:
    """Lat/lon from OpenStreetMap Nominatim. Returns (None, None) on failure."""
    safe_name = place_name.replace(" ", "_").replace(",", "").replace("/", "_")
    cache = os.path.join(CACHE_DIR, f"geocode_{safe_name}.json")
    try:
        payload = _http_get_json(
            "https://nominatim.openstreetmap.org/search",
            {"q": place_name, "format": "json", "limit": 1},
            cache_path=cache,
            timeout=15,
            user_agent="EMS-Demo/1.0",
        )
        if isinstance(payload, list) and len(payload) > 0:
            d = payload[0]
            return float(d["lat"]), float(d["lon"])
    except Exception as e:
        print(f"[geocode] failed: {e}")
    return None, None


# ---------------------------------------------------------------------------
# 2. demand forecast
# ---------------------------------------------------------------------------
def build_demand_slots(
    slot_demand_csv: str,
    start_date: str,
    number_of_days: int,
    slot_hours: int = 4,
    fill: str = "repeat",
) -> pd.DataFrame:
    """Tile / slice the historical-average demand to cover the requested window."""
    demand_df = pd.read_csv(slot_demand_csv)
    if "Avg_kWh" not in demand_df.columns:
        raise ValueError("slot_demand_csv must contain a column 'Avg_kWh'")

    vals = demand_df["Avg_kWh"].dropna().reset_index(drop=True).astype(float)

    slots_per_day = 24 // slot_hours
    steps = number_of_days * slots_per_day

    if len(vals) >= steps:
        selected = vals.iloc[:steps].values
    elif fill == "repeat":
        reps = ceil(steps / max(1, len(vals)))
        selected = np.tile(vals.values, reps)[:steps]
    elif fill == "pad":
        pad_len = steps - len(vals)
        selected = np.concatenate([vals.values, np.repeat(vals.values[-1], pad_len)])
    else:
        raise ValueError(f"unknown fill mode: {fill}")

    slot_starts = pd.date_range(start=pd.Timestamp(start_date), periods=steps,
                                freq=f"{slot_hours}h")
    slot_ends = slot_starts + pd.Timedelta(hours=slot_hours)
    out = pd.DataFrame({
        "Slot_Number": np.arange(1, steps + 1),
        "Slot_Start":  slot_starts,
        "Slot_End":    slot_ends,
        "Slot_Label":  slot_starts.strftime("%Y-%m-%d %H:%M") + " - " +
                       slot_ends.strftime("%H:%M"),
        "Forecast_kWh": selected,
    })
    out["Forecast_kWh_scaled"] = np.ceil(out["Forecast_kWh"] * 10)
    return out


# ---------------------------------------------------------------------------
# 3. weather + solar forecast
# ---------------------------------------------------------------------------
def fetch_weather(lat: float, lon: float, start: str, end: str,
                  timezone: str = LOCAL_TZ,
                  retries: int = 3,
                  timeout: int = 30) -> pd.DataFrame:
    """Open-Meteo archive API → 15-min DataFrame of temp + irradiance."""
    cache = os.path.join(
        CACHE_DIR,
        f"weather_{lat:.4f}_{lon:.4f}_{start}_{end}.json",
    )
    data = _http_get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": "temperature_2m,shortwave_radiation",
            "timezone": "UTC",
        },
        cache_path=cache,
        timeout=timeout,
        retries=retries,
    )
    if "hourly" not in data:
        raise RuntimeError(f"no hourly data: {data}")

    times = pd.to_datetime(data["hourly"]["time"]).tz_localize("UTC")
    df = pd.DataFrame(
        {"temperature_2m":  data["hourly"]["temperature_2m"],
         "irradiance_Wm2":  data["hourly"]["shortwave_radiation"]},
        index=times,
    )

    idx = pd.date_range(start=times.min(), end=times.max(), freq="15min", tz="UTC")
    df = df.reindex(idx.union(df.index)).interpolate("time").reindex(idx)
    df["irradiance_kWm2"] = df["irradiance_Wm2"] / 1000.0
    df.index = df.index.tz_convert(pytz.timezone(timezone))
    df.index.name = "timestamp"
    return df


def predict_solar(
    df_weather: pd.DataFrame,
    ffnn_model_path: str = "ffnn_model.h5",
    x_scaler_path:   str = "X_scaler.save",
    y_scaler_path:   str = "y_scaler.save",
    mt_model_path:   str = "linear_model_mt.pkl",
) -> pd.DataFrame:
    """Use the saved FFNN + linear module-temp model to estimate kWh from weather."""
    print("[solar] importing tensorflow ...")
    import joblib
    from tensorflow.keras.models import load_model
    print("[solar] tensorflow imported")

    print(f"[solar] loading FFNN from {ffnn_model_path} ...")
    ffnn = load_model(ffnn_model_path, compile=False)
    print("[solar] FFNN loaded")

    print("[solar] loading scalers + module-temp model ...")
    x_scaler  = joblib.load(x_scaler_path)
    y_scaler  = joblib.load(y_scaler_path)
    mt_model  = joblib.load(mt_model_path)
    print("[solar] all artifacts loaded")

    df = df_weather.rename(columns={
        "temperature_2m":  "AMBIENT_TEMPERATURE",
        "irradiance_kWm2": "IRRADIATION",
    }).copy()
    df["MODULE_TEMPERATURE"] = mt_model.predict(
        df[["AMBIENT_TEMPERATURE", "IRRADIATION"]]
    )

    print("[solar] running FFNN inference ...")
    Xs = x_scaler.transform(df[FEATURES]).astype("float32")

    # Bypass model.predict() — it triggers a slow graph compile on macOS arm64.
    # Calling the model directly stays in eager mode and returns instantly.
    import tensorflow as tf
    yhat = ffnn(tf.constant(Xs), training=False).numpy().flatten()

    yhat = y_scaler.inverse_transform(yhat.reshape(-1, 1)).flatten()
    yhat = yhat - np.min(yhat)
    print(f"[solar] inference done, {len(yhat)} predictions")

    df["kWh"] = pd.Series(yhat, index=df.index)
    return df[FEATURES + ["kWh"]]


def aggregate_to_slots(df_results: pd.DataFrame, slot_hours: int = 4,
                       solar_capacity_factor: float = 400.0) -> pd.DataFrame:
    """Resample 15-min predictions to slot-level, force kWh=0 when no irradiance."""
    df_4h = pd.DataFrame()
    df_4h["avg_temp"]        = df_results["AMBIENT_TEMPERATURE"].resample(f"{slot_hours}h").mean()
    df_4h["avg_module_temp"] = df_results["MODULE_TEMPERATURE"].resample(f"{slot_hours}h").mean()
    df_4h["avg_irradiance"]  = df_results["IRRADIATION"].resample(f"{slot_hours}h").mean()
    df_4h["cum_energy"]      = df_results["kWh"].resample(f"{slot_hours}h").sum()

    df_4h = df_4h.reset_index().rename(columns={"timestamp": "slot_start"})
    df_4h["slot_end"] = df_4h["slot_start"] + pd.Timedelta(hours=slot_hours)
    df_4h["slot_label"] = (df_4h["slot_start"].dt.strftime("%Y-%m-%d %H:%M")
                           + "–" + df_4h["slot_end"].dt.strftime("%H:%M"))

    df_4h.loc[df_4h["avg_irradiance"] == 0, "cum_energy"] = 0.0
    df_4h["cum_energy"] = df_4h["cum_energy"] / solar_capacity_factor

    cols = ["slot_label", "slot_start", "avg_temp", "avg_module_temp",
            "avg_irradiance", "cum_energy"]
    return df_4h[cols]


# ---------------------------------------------------------------------------
# 4. top-level: combine solar + demand
# ---------------------------------------------------------------------------
def run_forecast(
    location: str,
    number_of_days: int,
    start_date: str,
    *,
    models_dir: str = "models",
    slot_demand_csv: str = "models/slot_demand1.csv",
    slot_hours: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """One-shot forecast that returns the slot DataFrame the RL agent expects."""
    print(f"[forecast] location={location!r} days={number_of_days} start={start_date}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    end_date = get_end_date(start_date, number_of_days)
    lat, lon = get_coordinates(location)
    if lat is None:
        raise RuntimeError(f"could not geocode {location!r}")
    print(f"[forecast] {location}: ({lat:.4f}, {lon:.4f})")

    print("[forecast] step 1/3: demand slots")
    forecast_demand = build_demand_slots(slot_demand_csv, start_date,
                                         number_of_days, slot_hours)

    print("[forecast] step 2/3: weather")
    weather = fetch_weather(lat, lon, start_date, end_date)

    print("[forecast] step 3/3: solar prediction")
    solar = predict_solar(
        weather,
        ffnn_model_path=os.path.join(models_dir, "ffnn_model.h5"),
        x_scaler_path  =os.path.join(models_dir, "X_scaler.save"),
        y_scaler_path  =os.path.join(models_dir, "y_scaler.save"),
        mt_model_path  =os.path.join(models_dir, "linear_model_mt.pkl"),
    )
    df_4h = aggregate_to_slots(solar, slot_hours)

    df_4h = df_4h.iloc[:len(forecast_demand)].copy()
    df_4h["Forecast_Demand"] = forecast_demand["Forecast_kWh_scaled"].values[:len(df_4h)]
    df_4h = df_4h.dropna(subset=["Forecast_Demand"]).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    df_4h["real_solar"]  = (df_4h["cum_energy"] *
                            rng.normal(1.0, 0.15, len(df_4h))).clip(lower=0)
    df_4h["real_demand"] = (df_4h["Forecast_Demand"] *
                            rng.normal(1.0, 0.10, len(df_4h))).clip(lower=0)

    return df_4h


if __name__ == "__main__":
    df = run_forecast("Roorkee, India", 3, "2025-05-01")
    print(df.to_string())