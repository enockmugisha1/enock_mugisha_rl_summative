"""
AgriGuard: a sequential decision environment for precision irrigation and
crop-health management on a Rwandan smallholder farm.

The agent controls a single autonomous farm rover that drives across a 4x4 field
of crop plots over a 60-day growing season. Each step is one ~3-hour work slot.
Yield is only realised when a plot is harvested, so the irrigation, fertiliser
and pest decisions taken early in the season are credited only ~200 steps later.

See README.md for the full observation-provenance table.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Action semantics (discrete, 9 actions, each maps to a real rover command)
# ---------------------------------------------------------------------------
MOVE_N, MOVE_S, MOVE_E, MOVE_W = 0, 1, 2, 3
IRRIGATE, FERTILIZE, TREAT_PEST, HARVEST, WAIT = 4, 5, 6, 7, 8

ACTION_NAMES = [
    "MOVE_NORTH",    # drive rover one plot north   (RTK waypoint command)
    "MOVE_SOUTH",    # drive rover one plot south
    "MOVE_EAST",     # drive rover one plot east
    "MOVE_WEST",     # drive rover one plot west
    "IRRIGATE",      # open drip valve on current plot, draw from tank
    "FERTILIZE",     # dose current plot with urea/NPK from hopper
    "TREAT_PEST",    # spray biopesticide on current plot
    "HARVEST",       # harvest current plot, bank the yield
    "WAIT",          # idle this slot, conserve water/fuel (e.g. rain forecast)
]


class AgriGuardEnv(gym.Env):
    """Precision farm management as a finite-horizon stochastic MDP."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 8}

    # --- field / season geometry -------------------------------------------
    GRID = 4                    # 4x4 = 16 plots
    SEASON_DAYS = 60            # maize short-season cycle
    SLOTS_PER_DAY = 4           # four ~3h work slots per working day

    # --- agronomic constants ----------------------------------------------
    HEALTHY_LO, HEALTHY_HI = 0.35, 0.85   # ideal volumetric soil-water band
    WILT_POINT = 0.15                     # below this the crop takes drought damage
    WATERLOG_POINT = 0.93                 # above this roots suffocate
    FIELD_CAPACITY = 0.80                 # water held against gravity; excess percolates
    DRAIN_RATE = 0.55                     # fraction of the excess that drains per slot
    WATERLOG_SEVERITY = 0.5               # saturation is less lethal than wilting
    MATURE_GROWTH = 0.60                  # minimum growth for a non-premature harvest

    IRRIGATE_GAIN = 0.26        # moisture added by one irrigation pass
    IRRIGATE_WATER = 0.045      # fraction of full tank consumed per pass
    FERT_GAIN = 0.40            # nutrient index added per fertiliser dose
    FERT_RUNOFF = 0.90          # above this nutrient level the excess leaches away
    PEST_KNOCKDOWN = 0.20       # pest level multiplier after treatment

    GROWTH_RATE = 0.0135        # growth per slot under ideal conditions
    NUTRIENT_UPTAKE = 0.006     # nutrient consumed per slot of growth
    PEST_SEED_PROB = 0.020      # daily chance a clean plot becomes infested
    PEST_GROWTH = 0.026         # logistic infestation growth rate per slot
    PEST_SPREAD = 0.009         # infestation transmitted to each neighbour per slot
    STRESS_DAMAGE = 0.018       # health lost per slot of drought/waterlog stress

    FERT_UNITS = 12             # fertiliser doses available for the whole season
    PEST_UNITS = 8              # biopesticide doses available for the whole season

    # --- reward weights ----------------------------------------------------
    R_YIELD = 14.0              # terminal-ish payoff per unit of harvested yield
    R_HEALTH_SHAPE = 0.12       # dense shaping: fraction of plots in healthy band
    R_IDLE_COST = 0.02          # opportunity cost of idling a work slot
    R_WATER_COST = 0.035        # economic cost of pumping water
    R_FERT_COST = 0.09          # cost of a fertiliser dose
    R_PEST_COST = 0.09          # cost of a biopesticide dose
    R_MOVE_COST = 0.012         # fuel/time cost of driving between plots
    R_WASTE = 0.45              # penalty for an infeasible action (empty tank etc.)
    R_WATERLOG = 0.70           # penalty for over-irrigating into waterlogging
    R_RUNOFF = 0.55             # penalty for nitrate runoff (environmental harm)
    R_DEATH = 3.0               # penalty per plot lost
    R_PREMATURE = 2.5           # penalty for harvesting an immature plot
    R_SEASON_BONUS = 18.0       # end-of-season bonus scaled by harvest completion

    def __init__(self, render_mode: str | None = None, seed: int | None = None,
                 max_slots: int | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.n_plots = self.GRID * self.GRID
        self.max_slots = max_slots or self.SEASON_DAYS * self.SLOTS_PER_DAY

        self.action_space = spaces.Discrete(len(ACTION_NAMES))

        # obs = 16*(moisture, growth, nutrient, pest, health) + 16*2 status flags
        #       + 16 rover one-hot + 7 globals
        self.obs_dim = self.n_plots * 5 + self.n_plots * 2 + self.n_plots + 7
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self._np_random, _ = gym.utils.seeding.np_random(seed)

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
        rng = self._np_random
        n = self.n_plots

        # Start state: crop just transplanted, soil near field capacity.
        self.moisture = rng.uniform(0.42, 0.58, size=n)
        self.growth = np.full(n, 0.05)
        self.nutrient = rng.uniform(0.50, 0.70, size=n)
        self.pest = np.zeros(n)
        self.health = np.ones(n)
        self.alive = np.ones(n, dtype=bool)
        self.harvested = np.zeros(n, dtype=bool)

        self.tank = 1.0                                  # full reservoir
        self.fert_stock = float(self.FERT_UNITS)
        self.pest_stock = float(self.PEST_UNITS)

        self.rover = 0                                   # rover parked at plot (0,0)
        self.day = 0
        self.slot = 0
        self.steps = 0
        self.banked_yield = 0.0
        self.matured_harvests = 0
        self.premature_harvests = 0
        self.rained_today = False
        self._roll_weather()

        self.episode_log: list[dict] = []
        return self._obs(), self._info()

    # ------------------------------------------------------------------- step
    def step(self, action: int):
        action = int(action)
        reward = 0.0
        p = self.rover
        event = ""

        # ---------------- 1. apply the chosen action ----------------------
        if action in (MOVE_N, MOVE_S, MOVE_E, MOVE_W):
            r, c = divmod(p, self.GRID)
            if action == MOVE_N:
                r -= 1
            elif action == MOVE_S:
                r += 1
            elif action == MOVE_E:
                c += 1
            else:
                c -= 1
            if 0 <= r < self.GRID and 0 <= c < self.GRID:
                self.rover = r * self.GRID + c
                reward -= self.R_MOVE_COST
            else:
                reward -= self.R_WASTE          # drove into the fence
                event = "blocked"

        elif action == IRRIGATE:
            if self.tank < self.IRRIGATE_WATER:
                reward -= self.R_WASTE          # valve opened on an empty tank
                event = "dry_tank"
            elif self.harvested[p] or not self.alive[p]:
                reward -= self.R_WASTE          # watering bare soil
                event = "idle_plot"
            else:
                self.tank -= self.IRRIGATE_WATER
                self.moisture[p] = min(1.0, self.moisture[p] + self.IRRIGATE_GAIN)
                reward -= self.R_WATER_COST
                if self.moisture[p] > self.WATERLOG_POINT:
                    reward -= self.R_WATERLOG   # over-irrigation
                    event = "waterlogged"

        elif action == FERTILIZE:
            if self.fert_stock < 1.0:
                reward -= self.R_WASTE
                event = "no_fertilizer"
            elif self.harvested[p] or not self.alive[p]:
                reward -= self.R_WASTE
                event = "idle_plot"
            else:
                self.fert_stock -= 1.0
                before = self.nutrient[p]
                self.nutrient[p] = min(1.0, before + self.FERT_GAIN)
                reward -= self.R_FERT_COST
                if before > self.FERT_RUNOFF:
                    reward -= self.R_RUNOFF     # nitrate leaches into groundwater
                    event = "runoff"

        elif action == TREAT_PEST:
            if self.pest_stock < 1.0:
                reward -= self.R_WASTE
                event = "no_pesticide"
            elif self.harvested[p] or not self.alive[p]:
                reward -= self.R_WASTE
                event = "idle_plot"
            else:
                self.pest_stock -= 1.0
                self.pest[p] *= self.PEST_KNOCKDOWN
                reward -= self.R_PEST_COST

        elif action == HARVEST:
            if self.harvested[p] or not self.alive[p]:
                reward -= self.R_WASTE
                event = "idle_plot"
            elif self.growth[p] < self.MATURE_GROWTH:
                # Premature harvest: you get the grain that formed, and a penalty.
                y = 0.25 * self.growth[p] * self.health[p]
                self.banked_yield += y
                reward += self.R_YIELD * y - self.R_PREMATURE
                self.harvested[p] = True
                self.premature_harvests += 1
                event = "premature_harvest"
            else:
                y = self.growth[p] * self.health[p] * (1.0 - 0.75 * self.pest[p])
                self.banked_yield += y
                reward += self.R_YIELD * y
                self.harvested[p] = True
                self.matured_harvests += 1
                event = "harvest"

        elif action == WAIT:
            # Idling is a legitimate move (e.g. hold off irrigating when rain is
            # forecast) but it is not free: a work slot spent doing nothing has a
            # labour opportunity cost. Without this, WAIT is the only zero-cost
            # action and becomes a do-nothing local optimum that value-based
            # methods fall into long before the delayed harvest reward propagates.
            reward -= self.R_IDLE_COST

        # ---------------- 2. advance the world one slot -------------------
        reward += self._advance_world()

        # ---------------- 3. dense shaping on field condition -------------
        active = self.alive & ~self.harvested
        if active.any():
            in_band = ((self.moisture >= self.HEALTHY_LO) &
                       (self.moisture <= self.HEALTHY_HI) & active).sum()
            reward += self.R_HEALTH_SHAPE * in_band / self.n_plots

        # ---------------- 4. termination ----------------------------------
        self.steps += 1
        newly_dead = int(self._reap())
        reward -= self.R_DEATH * newly_dead

        season_over = self.day >= self.SEASON_DAYS
        field_closed = not (self.alive & ~self.harvested).any()
        terminated = bool(season_over or field_closed)
        truncated = bool(self.steps >= self.max_slots and not terminated)

        if terminated or truncated:
            # Completion bonus counts ONLY plots brought in at maturity. Paying it
            # for any harvest would make rush-harvesting 16 immature plots on day 1
            # a degenerate optimum, so premature cuts earn no share of the bonus.
            completion = self.matured_harvests / self.n_plots
            reward += self.R_SEASON_BONUS * completion

        self.episode_log.append({
            "step": self.steps, "day": self.day, "action": ACTION_NAMES[action],
            "reward": round(reward, 3), "event": event,
            "tank": round(self.tank, 3), "yield": round(self.banked_yield, 3),
        })
        return self._obs(), float(reward), terminated, truncated, self._info()

    # --------------------------------------------------------------- dynamics
    def _advance_world(self) -> float:
        """Exogenous + endogenous dynamics for one work slot."""
        rng = self._np_random
        pen = 0.0
        active = self.alive & ~self.harvested

        # (a) soil water balance: free water percolates below the root zone,
        #     then evapotranspiration removes more, scaled by the day's temperature.
        #     Without drainage the field saturates after any rain and never recovers,
        #     which would make waterlogging a weather artefact instead of a
        #     consequence of the agent over-irrigating.
        excess = np.maximum(0.0, self.moisture - self.FIELD_CAPACITY)
        self.moisture -= self.DRAIN_RATE * excess
        evap = 0.006 + 0.013 * self.temp
        self.moisture = np.clip(self.moisture - evap, 0.0, 1.0)

        # (b) crop growth, gated by water, nutrients and pest pressure
        ok = (active & (self.moisture >= self.HEALTHY_LO)
              & (self.moisture <= self.HEALTHY_HI) & (self.nutrient > 0.18))
        gain = self.GROWTH_RATE * (1.0 - 0.80 * self.pest) * self.health
        self.growth = np.where(ok, np.clip(self.growth + gain, 0.0, 1.0), self.growth)
        self.nutrient = np.where(ok, np.clip(self.nutrient - self.NUTRIENT_UPTAKE, 0.0, 1.0),
                                 self.nutrient)

        # (c) abiotic stress erodes plot health irreversibly
        drought = active & (self.moisture < self.WILT_POINT)
        soaked = active & (self.moisture > self.WATERLOG_POINT)
        self.health = np.clip(
            self.health - self.STRESS_DAMAGE * (drought + self.WATERLOG_SEVERITY * soaked),
            0.0, 1.0)

        # (d) pest contagion: self-amplification + spread to 4-neighbours.
        #     This is the main source of delayed cost in the environment.
        infested = self.pest.copy()
        # Logistic rather than exponential growth: an untreated outbreak still
        # saturates, but slowly enough that the 8 biopesticide doses can contain it.
        self.pest = np.clip(
            self.pest + self.PEST_GROWTH * infested * (1.0 - infested) * active, 0.0, 1.0)
        spread = np.zeros_like(self.pest)
        for i in range(self.n_plots):
            r, c = divmod(i, self.GRID)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.GRID and 0 <= cc < self.GRID:
                    spread[i] += infested[rr * self.GRID + cc]
        self.pest = np.clip(self.pest + self.PEST_SPREAD * spread * active, 0.0, 1.0)
        self.health = np.clip(self.health - 0.007 * self.pest * active, 0.0, 1.0)

        # (e) end of the working day: rain, new infestations, fresh forecast
        self.slot += 1
        if self.slot >= self.SLOTS_PER_DAY:
            self.slot = 0
            self.day += 1
            if rng.random() < self.rain_prob:
                depth = rng.uniform(0.10, 0.28)
                self.moisture = np.clip(self.moisture + depth, 0.0, 1.0)
                self.tank = min(1.0, self.tank + 0.60 * depth)   # roof/pond capture
                self.rained_today = True
            else:
                self.rained_today = False
            fresh = active & (self.pest < 0.02) & (rng.random(self.n_plots) < self.PEST_SEED_PROB)
            self.pest = np.where(fresh, rng.uniform(0.10, 0.25, self.n_plots), self.pest)
            self._roll_weather()
        return pen

    def _roll_weather(self):
        """Bimodal Rwandan rainfall: wet onset, dry mid-season, wet tail."""
        phase = 2.0 * np.pi * self.day / self.SEASON_DAYS
        self.rain_prob = float(np.clip(0.34 + 0.22 * np.cos(phase)
                                      + self._np_random.normal(0, 0.05), 0.10, 0.85))
        self.temp = float(np.clip(0.45 + 0.22 * np.sin(phase)
                                 + self._np_random.normal(0, 0.06), 0.0, 1.0))

    def _reap(self) -> int:
        """Kill plots whose health has collapsed; return how many died this slot."""
        doomed = self.alive & ~self.harvested & (self.health <= 0.02)
        self.alive = self.alive & ~doomed
        return int(doomed.sum())

    # ------------------------------------------------------------ observation
    def _obs(self) -> np.ndarray:
        rover = np.zeros(self.n_plots, dtype=np.float32)
        rover[self.rover] = 1.0
        return np.concatenate([
            self.moisture, self.growth, self.nutrient, self.pest, self.health,
            self.alive.astype(np.float32), self.harvested.astype(np.float32),
            rover,
            [self.tank,
             self.fert_stock / self.FERT_UNITS,
             self.pest_stock / self.PEST_UNITS,
             self.day / self.SEASON_DAYS,
             self.rain_prob,
             self.temp,
             float(self.rained_today)],
        ]).astype(np.float32)

    def _info(self) -> dict:
        return {
            "day": self.day, "slot": self.slot, "rover": self.rover,
            "tank": self.tank, "banked_yield": self.banked_yield,
            "alive": int(self.alive.sum()), "harvested": int(self.harvested.sum()),
            "mean_growth": float(self.growth.mean()),
            "mean_pest": float(self.pest.mean()),
            "matured": self.matured_harvests, "premature": self.premature_harvests,
            "fert_stock": self.fert_stock, "pest_stock": self.pest_stock,
        }

    # ---------------------------------------------------------------- render
    def render(self):
        if self.render_mode == "ansi":
            return self.to_ascii()
        print(self.to_ascii())

    def to_ascii(self) -> str:
        """Terminal view used for the verbose output in the demo video."""
        rows = []
        rows.append(f"Day {self.day:2d}/{self.SEASON_DAYS}  slot {self.slot+1}/{self.SLOTS_PER_DAY}"
                    f"   tank={self.tank:4.2f}  fert={self.fert_stock:.0f}"
                    f"  pest-kit={self.pest_stock:.0f}  yield={self.banked_yield:5.2f}")
        rows.append(f"rain P={self.rain_prob:.2f}  temp={self.temp:.2f}"
                    f"  {'RAIN' if self.rained_today else 'dry '}")
        for r in range(self.GRID):
            line = []
            for c in range(self.GRID):
                i = r * self.GRID + c
                if self.harvested[i]:
                    cell = " HARV "
                elif not self.alive[i]:
                    cell = " DEAD "
                else:
                    cell = f"{self.moisture[i]:.2f}/{self.growth[i]:.2f}"
                mark = "[" if i == self.rover else " "
                end = "]" if i == self.rover else " "
                line.append(f"{mark}{cell}{end}")
            rows.append(" ".join(line))
        return "\n".join(rows)

    def snapshot(self) -> dict:
        """Serialisable frame consumed by the Three.js renderer and any
        downstream API / mobile client (see README integration section)."""
        return {
            "day": int(self.day), "slot": int(self.slot), "step": int(self.steps),
            "tank": float(self.tank), "fert": float(self.fert_stock),
            "pestkit": float(self.pest_stock), "rover": int(self.rover),
            "yield": float(self.banked_yield), "rain_prob": float(self.rain_prob),
            "matured": int(self.matured_harvests),
            "temp": float(self.temp), "rained": bool(self.rained_today),
            "grid": self.GRID,
            "plots": [
                {"moisture": float(self.moisture[i]), "growth": float(self.growth[i]),
                 "nutrient": float(self.nutrient[i]), "pest": float(self.pest[i]),
                 "health": float(self.health[i]), "alive": bool(self.alive[i]),
                 "harvested": bool(self.harvested[i])}
                for i in range(self.n_plots)
            ],
        }
