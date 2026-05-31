"""
rl_agent.py
===========
Green-H2 RL EMS — updated to:
  * start the day with empty battery AND empty H2 (initial SOC = 0)
  * always allocate a portion of surplus solar to electrolyser (not battery-first)
  * discharge H2 BEFORE battery whenever the agent requests fuel-cell action
  * reward FC electrical output strongly so the agent actually uses stored H2

Public API unchanged:
    rollout(df_4h, model_path)  →  per-slot decisions DataFrame
    train(df_4h, total_timesteps, save_path)
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# System parameters (must match notebook + Simulink ratings)
# ---------------------------------------------------------------------------
SLOT_HOURS = 4.0          # one row of df_4h represents 4 hours

# Battery
BATT_CAP        = 400.0           # kWh
BATT_MAX_KW     = 200.0           # kW
ETA_BATT_CH     = 0.90
ETA_BATT_DCH    = 0.90
BATT_DEG_COST   = 0.5             # ₹/kWh throughput

# Hydrogen (energy-equivalent kWh)
H2_CAP          = 500.0
P_ELY_MAX_KW    = 80.0
P_FC_MAX_KW     = 80.0
ETA_ELY         = 0.70
ETA_FC          = 0.70
H2_USE_COST     = 0.6
GRID_PRICE      = 8.0             # ₹/kWh

# per-slot energy caps (kWh)
BATT_MAX_EN  = BATT_MAX_KW  * SLOT_HOURS
P_ELY_MAX_EN = P_ELY_MAX_KW * SLOT_HOURS
P_FC_MAX_EN  = P_FC_MAX_KW  * SLOT_HOURS

# Default split of surplus solar between electrolyser and battery.
# Hydrogen-first: 60% to H2, 40% to battery.  Agent can push more to ely
# via actions 1-3, but never less than this floor.
SURPLUS_TO_H2_FRACTION  = 0.6


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
def storage_update_slot(demand, solar, batt_soc, h2_available, pending_h2_prev,
                        ely_elec_kwh=0.0, fc_request_kwh=0.0,
                        allow_ely_grid=False):
    """
    Order of operations:
      1) solar → load (direct)
      2) IF fc_request > 0  : FC → load BEFORE battery   (green-H2 preference)
         ELSE              : battery → load
                              FC      → load (whatever fc_request says)
      3) grid covers any leftover deficit
      4) Surplus solar (after step 1):
         → electrolyser first (max of agent's request and default share),
           capped by H2 capacity & ELY power limit
         → battery (whatever solar is left)
         → curtailment (last resort)
    """
    h2_avail = h2_available + (pending_h2_prev or 0.0)

    # ---- 1. solar direct ----
    solar_used = min(solar, demand)
    remaining  = demand - solar_used

    # ---- 2. discharge: FC first if requested, else battery first ----
    batt_discharge = 0.0
    fc_actual_elec = 0.0
    h2_discharge   = 0.0

    if fc_request_kwh > 0.0:
        # green-H2 preference: fuel cell BEFORE battery
        fc_req = min(fc_request_kwh, P_FC_MAX_EN, max(0.0, remaining))
        h2_needed_for_fc = min(h2_avail, fc_req / ETA_FC)
        fc_actual_elec   = h2_needed_for_fc * ETA_FC
        h2_discharge     = h2_needed_for_fc
        remaining       -= fc_actual_elec

        # battery covers anything still missing
        batt_discharge   = min(remaining, batt_soc, BATT_MAX_EN)
        remaining       -= batt_discharge
    else:
        # default order: battery → FC (FC only used if agent explicitly asks,
        # which is action 4 or 5 above; if action 0-3 then fc_request==0 and
        # this branch leaves H2 alone)
        batt_discharge   = min(remaining, batt_soc, BATT_MAX_EN)
        remaining       -= batt_discharge

        fc_req = min(fc_request_kwh, P_FC_MAX_EN, max(0.0, remaining))
        h2_needed_for_fc = min(h2_avail, fc_req / ETA_FC)
        fc_actual_elec   = h2_needed_for_fc * ETA_FC
        h2_discharge     = h2_needed_for_fc
        remaining       -= fc_actual_elec

    # ---- 3. grid as last resort ----
    grid_used = max(0.0, remaining)

    # ---- 4. surplus solar handling: ELY share first, then battery ----
    surplus_solar = max(0.0, solar - solar_used)

    ely_from_solar   = 0.0
    batt_charge_elec = 0.0
    h2_charge        = 0.0
    curtailed        = 0.0
    ely_from_grid    = 0.0

    if surplus_solar > 1e-9:
        # ELY allocation: agent's request OR default share, whichever is bigger,
        # capped by H2 capacity (in chemical kWh) and ELY power limit.
        h2_space        = max(0.0, H2_CAP - h2_avail)
        ely_room_kwh    = h2_space / ETA_ELY      # input kWh that fits H2 tank
        desired_ely     = max(ely_elec_kwh, surplus_solar * SURPLUS_TO_H2_FRACTION)
        ely_from_solar  = min(surplus_solar, desired_ely, P_ELY_MAX_EN, ely_room_kwh)

        remaining_solar  = surplus_solar - ely_from_solar
        # battery soaks up what's left, capped by capacity & power
        batt_charge_elec = min(BATT_CAP - batt_soc, remaining_solar, BATT_MAX_EN)
        leftover         = remaining_solar - batt_charge_elec

        # optional: ely from grid (only if explicitly allowed; never used by RL agent)
        if allow_ely_grid and ely_elec_kwh > ely_from_solar:
            extra_ely_room = max(0.0, P_ELY_MAX_EN - ely_from_solar)
            ely_from_grid  = min(ely_elec_kwh - ely_from_solar, extra_ely_room,
                                 ely_room_kwh - ely_from_solar)

        h2_charge = (ely_from_solar + ely_from_grid) * ETA_ELY
        curtailed = max(0.0, leftover)

    # ---- 5. SOC update & cost ----
    new_batt_soc = max(0.0, min(BATT_CAP,
                                batt_soc - batt_discharge
                                + batt_charge_elec * ETA_BATT_CH))
    new_h2_available = h2_avail - h2_discharge

    cost = (grid_used      * GRID_PRICE
            + batt_discharge * BATT_DEG_COST
            + fc_actual_elec * H2_USE_COST
            + ely_from_grid  * GRID_PRICE)

    return {
        "solar_used":        solar_used,
        "batt_discharge":    batt_discharge,
        "h2_discharge":      h2_discharge,
        "fc_actual_elec":    fc_actual_elec,
        "grid_used":         grid_used,
        "batt_charge":       batt_charge_elec,
        "h2_charge":         h2_charge,
        "ely_from_grid":     ely_from_grid,
        "ely_from_solar":    ely_from_solar,
        "curtailed":         curtailed,
        "new_batt_soc":      new_batt_soc,
        "new_h2_available":  new_h2_available,
        "cost":              cost,
    }


# ---------------------------------------------------------------------------
# Gym env
# ---------------------------------------------------------------------------
def _make_env_class():
    """Lazy import of gymnasium (used by SB3 >= 2.3)."""
    import gymnasium as gym
    from gymnasium import spaces

    class EVChargingEnv(gym.Env):
        def __init__(self, df, train_mode=False):
            super().__init__()
            self.df = df.reset_index(drop=True)
            self.N  = len(self.df)
            self.train_mode = train_mode

            self.action_space      = spaces.Discrete(6)
            self.observation_space = spaces.Box(0.0, 1.0, (7,), dtype=np.float32)

            # initial conditions: empty battery AND empty H2.
            self.batt_soc     = 0.0
            self.h2_available = 0.0
            self.pending_h2   = 0.0
            self.t            = 0
            self.max_steps    = self.N - 1
            self.last_info    = {}

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = (np.random.randint(0, max(1, self.N - 6))
                      if self.train_mode else 0)
            self.batt_soc     = 0.0       # ← was 0.5 * BATT_CAP
            self.h2_available = 0.0
            self.pending_h2   = 0.0
            return self._get_state(), {}

        def _get_state(self):
            row = self.df.iloc[self.t]
            future = self.df.iloc[self.t: min(self.t + 4, self.N)]
            future_deficit = (future["Forecast_Demand"] - future["cum_energy"]).mean()
            max_p = max(BATT_CAP, H2_CAP,
                        self.df["cum_energy"].max(),
                        self.df["Forecast_Demand"].max(), 1.0)
            return np.array([
                float(row["real_solar"])  / max_p,
                float(row["real_demand"]) / max_p,
                self.batt_soc      / BATT_CAP,
                self.h2_available  / H2_CAP,
                self.pending_h2    / H2_CAP,
                max(0.0, future_deficit) / max_p,
                (self.t % 24) / 24.0,
            ], dtype=np.float32)

        def step(self, action):
            row    = self.df.iloc[self.t]
            demand = float(row["real_demand"])
            solar  = float(row["real_solar"])

            ely_kwh, fc_kwh = {
                0: (0.0,                 0.0),
                1: (P_ELY_MAX_EN * 0.25, 0.0),
                2: (P_ELY_MAX_EN * 0.5,  0.0),
                3: (P_ELY_MAX_EN * 1.0,  0.0),
                4: (0.0, P_FC_MAX_EN * 0.25),
                5: (0.0, P_FC_MAX_EN * 0.5),
            }[int(action)]

            info = storage_update_slot(
                demand, solar, self.batt_soc, self.h2_available, self.pending_h2,
                ely_elec_kwh   = min(ely_kwh, P_ELY_MAX_EN),
                fc_request_kwh = min(fc_kwh,  P_FC_MAX_EN),
                allow_ely_grid = False,
            )
            self.batt_soc     = info["new_batt_soc"]
            self.h2_available = info["new_h2_available"]

            self.last_info = {
                "slot_idx":     self.t,
                "demand":       demand,
                "solar_used":   info["solar_used"],
                "battery":      info["batt_discharge"],
                "hydrogen":     info["fc_actual_elec"],   # electrical kWh delivered
                "grid":         info["grid_used"],
                "batt_charge":  info["batt_charge"],
                "h2_charge":    info["h2_charge"],
                "curtailed":    info["curtailed"],
                "cost":         info["cost"],
                "batt_soc":     self.batt_soc,
                "h2_soc":       self.h2_available,
            }
            reward = compute_reward(info)
            self.pending_h2 = info["h2_charge"]
            self.t += 1
            done = bool(self.t >= self.max_steps)
            obs = (self._get_state() if not done
                   else np.zeros(self.observation_space.shape, dtype=np.float32))
            return obs, float(reward), done, False, self.last_info

    return EVChargingEnv


def compute_reward(info):
    """
    Reward shaping aligned with the user's goals:
      • use solar directly  →  reward
      • PRODUCE H2 from surplus solar (good — but moderate, not runaway)
      • USE H2 to serve load (large reward — this is what was missing)
      • discourage battery cycling (degradation, lifecycle concern)
      • avoid grid (large penalty)
      • avoid curtailment (waste)
      • forbid grid-fed electrolysis (not green hydrogen)
    """
    return (
        + 2.0 * info["solar_used"]
        + 1.5 * info["h2_charge"]            # ↓ from 4.0 — was causing runaway production
        + 5.0 * info["fc_actual_elec"]       # ↑↑ reward USING H2 (per electrical kWh)
        - 2.5 * info["batt_discharge"]       # discourage battery discharge specifically
        - 0.5 * info["batt_charge"]          # mild penalty on battery charging
        - 8.0 * info["grid_used"]
        - 5.0 * info["curtailed"]
        - 10.0 * info.get("ely_from_grid", 0.0)
    )


# ---------------------------------------------------------------------------
# Train + rollout API
# ---------------------------------------------------------------------------
def train(df_4h: pd.DataFrame, total_timesteps: int = 50_000,
          save_path: str = "models/evcs_dqn") -> str:
    """Train a DQN on a representative dataset. Saves the SB3 model zip."""
    from stable_baselines3 import DQN
    Env = _make_env_class()
    env = Env(df_4h, train_mode=True)
    model = DQN("MlpPolicy", env,
                learning_rate=5e-4, buffer_size=50_000, batch_size=64,
                gamma=0.99, learning_starts=1000, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    print(f"[rl] saved model → {save_path}.zip")
    return save_path


def rollout(df_4h: pd.DataFrame,
            model_path: str = "models/evcs_dqn") -> pd.DataFrame:
    """Greedy rollout. Returns one row per slot with the EMS decisions."""
    from stable_baselines3 import DQN
    Env = _make_env_class()
    env = Env(df_4h, train_mode=False)
    model = DQN.load(model_path)

    obs, _ = env.reset()
    rows = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        rows.append(info)
    out = pd.DataFrame(rows)

    if "slot_label" in df_4h.columns:
        out["slot_label"] = df_4h["slot_label"].iloc[: len(out)].values
    if "slot_start" in df_4h.columns:
        out["slot_start"] = df_4h["slot_start"].iloc[: len(out)].values
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        from forecasting import run_forecast
        df = run_forecast("Roorkee, India", 7, "2025-04-01")
        train(df, total_timesteps=50_000)
    else:
        print("usage:  python rl_agent.py train")