"""
Environment contract tests.

These lock down the properties the report claims about AgriGuard: a valid
Gymnasium API, bounded observations, action-dependent state transitions, working
terminal conditions, and -- importantly -- that the degenerate "rush-harvest"
strategy is not profitable.
"""

from __future__ import annotations

import numpy as np
import pytest

from environment import ACTION_NAMES, AgriGuardEnv
from environment.custom_env import (FERTILIZE, HARVEST, IRRIGATE, MOVE_E, MOVE_N,
                                    MOVE_S, MOVE_W, TREAT_PEST, WAIT)


@pytest.fixture
def env():
    e = AgriGuardEnv(seed=0)
    e.reset(seed=0)
    return e


# ----------------------------------------------------------------- spaces
def test_action_space_is_exhaustive_and_named(env):
    assert env.action_space.n == len(ACTION_NAMES) == 9


def test_observation_within_declared_space_for_whole_episode(env):
    obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    rng = np.random.default_rng(3)
    for _ in range(600):
        obs, _, term, trunc, _ = env.step(int(rng.integers(0, 9)))
        assert env.observation_space.contains(obs), "observation left its Box"
        assert np.all(np.isfinite(obs))
        if term or trunc:
            obs, _ = env.reset()


def test_observation_dimension_matches_documentation(env):
    # 16 plots x 7 per-plot values + 16 rover one-hot + 7 globals
    assert env.observation_space.shape == (16 * 7 + 16 + 7,) == (135,)


# ----------------------------------------------------------- start state
def test_start_state_is_freshly_planted(env):
    assert np.allclose(env.growth, 0.05)
    assert env.tank == 1.0
    assert env.pest.sum() == 0
    assert env.alive.all() and not env.harvested.any()
    assert env.rover == 0 and env.day == 0


def test_reset_is_reproducible_under_a_seed():
    a, _ = AgriGuardEnv().reset(seed=42)
    b, _ = AgriGuardEnv().reset(seed=42)
    assert np.allclose(a, b)


# ------------------------------------------------- action-dependent state
def test_irrigation_raises_moisture_and_drains_the_tank(env):
    before_m, before_t = env.moisture[env.rover], env.tank
    env.step(IRRIGATE)
    assert env.moisture[env.rover] > before_m
    assert env.tank < before_t


def test_fertiliser_raises_nutrient_and_consumes_a_dose(env):
    before_n, before_s = env.nutrient[env.rover], env.fert_stock
    env.step(FERTILIZE)
    assert env.nutrient[env.rover] > before_n
    assert env.fert_stock == before_s - 1


def test_treatment_knocks_pest_down(env):
    env.pest[env.rover] = 0.8
    env.step(TREAT_PEST)
    assert env.pest[env.rover] < 0.8
    assert env.pest_stock == env.PEST_UNITS - 1


def test_movement_changes_rover_position_and_respects_the_fence(env):
    env.step(MOVE_E)
    assert env.rover == 1
    env.step(MOVE_W)
    assert env.rover == 0
    _, r, _, _, _ = env.step(MOVE_N)      # into the fence from row 0
    assert env.rover == 0
    assert r < 0, "driving into the boundary should be penalised"


def test_moisture_decays_without_irrigation(env):
    env.rain_prob = 0.0                   # suppress rain for a clean measurement
    start = env.moisture.copy()
    for _ in range(3):
        env._advance_world()
    assert (env.moisture < start).all()


def test_pest_spreads_to_neighbours(env):
    env.pest[:] = 0.0
    env.pest[5] = 0.6                     # plot (1,1): has four neighbours
    for _ in range(12):
        env._advance_world()
    for nb in (1, 4, 6, 9):
        assert env.pest[nb] > 0.0, f"infestation did not reach neighbour {nb}"


def test_waterlogging_is_penalised(env):
    env.moisture[env.rover] = 0.92        # one pass pushes it past saturation
    _, r_wet, _, _, _ = env.step(IRRIGATE)
    env.reset(seed=0)
    env.moisture[env.rover] = 0.30        # a genuinely dry plot
    _, r_dry, _, _, _ = env.step(IRRIGATE)
    assert r_dry > r_wet, "over-irrigating should score worse than irrigating dry soil"


def test_soil_drains_towards_field_capacity(env):
    env.moisture[:] = 1.0
    env.rain_prob = 0.0
    for _ in range(10):
        env._advance_world()
    assert env.moisture.max() < env.WATERLOG_POINT, "saturated soil must percolate"


# -------------------------------------------------- rewards & termination
def test_mature_harvest_beats_premature_harvest(env):
    env.growth[env.rover] = 0.95
    _, r_ripe, _, _, _ = env.step(HARVEST)
    env.reset(seed=0)
    env.growth[env.rover] = 0.10
    _, r_green, _, _, _ = env.step(HARVEST)
    assert r_ripe > r_green
    assert r_green < 0, "cutting an immature plot should be a net loss"


def test_rush_harvest_exploit_is_not_profitable():
    """Harvesting everything immediately must score far below a patient policy.

    The completion bonus deliberately counts only plots brought in at maturity;
    without that, cutting all 16 plots on day 1 would be a degenerate optimum.
    """
    returns = []
    for seed in range(5):
        env = AgriGuardEnv(seed=seed)
        env.reset(seed=seed)
        total = 0.0
        while True:
            if not env.harvested[env.rover] and env.alive[env.rover]:
                a = HARVEST
            else:
                r, c = divmod(env.rover, env.GRID)
                a = MOVE_E if c < env.GRID - 1 else (MOVE_S if r < env.GRID - 1 else WAIT)
            _, rew, term, trunc, info = env.step(a)
            total += rew
            if term or trunc:
                break
        returns.append(total)
        assert info["matured"] == 0
    assert np.mean(returns) < 0, "rush-harvesting must not be a winning strategy"


def test_episode_terminates_and_respects_the_slot_budget(env):
    obs, _ = env.reset(seed=11)
    steps = 0
    while steps < 10_000:
        _, _, term, trunc, info = env.step(WAIT)
        steps += 1
        if term or trunc:
            break
    assert term or trunc, "episode never finished"
    assert steps <= env.max_slots
    assert info["day"] <= env.SEASON_DAYS


def test_terminates_when_every_plot_is_closed(env):
    env.alive[:] = False                  # whole field lost
    _, _, term, _, _ = env.step(WAIT)
    assert term


def test_infeasible_actions_are_penalised(env):
    env.tank = 0.0
    _, r, _, _, _ = env.step(IRRIGATE)
    assert r < 0
    env.fert_stock = 0.0
    _, r2, _, _, _ = env.step(FERTILIZE)
    assert r2 < 0


# -------------------------------------------------------- serialisation
def test_snapshot_is_json_serialisable_for_the_renderer(env):
    import json
    env.step(IRRIGATE)
    snap = env.snapshot()
    json.dumps(snap)                       # must not raise
    assert len(snap["plots"]) == 16
    assert set(snap["plots"][0]) >= {"moisture", "growth", "pest", "health",
                                     "alive", "harvested"}
    assert snap["grid"] == 4


def test_ascii_view_renders(env):
    text = env.to_ascii()
    assert "Day" in text and len(text.splitlines()) >= 6
