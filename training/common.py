"""
Shared training infrastructure for the four AgriGuard agents.

Everything that must be identical across DQN / REINFORCE / A2C / PPO lives here:
the environment factory, the evaluation protocol, the CSV logger and the network
sizing. Only the algorithm-specific loss, data flow and hyperparameters differ,
which is what makes the comparison in the report a fair one.
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import warnings

import numpy as np
import torch

# TorchRL 0.15 emits a FutureWarning about auto-registering policy transforms and
# a benign requires_grad->scalar warning when logging losses. Neither affects
# results; silencing them keeps the demo terminal output readable.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*requires_grad=True to a scalar.*")
from torchrl.envs import (TransformedEnv, StepCounter, Compose, DoubleToFloat,
                          RewardSum, SerialEnv)
from torchrl.envs.libs.gym import GymWrapper

from environment import AgriGuardEnv

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
FIGS = ROOT / "figures"
for _d in (LOGS, MODELS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")          # the env is a cheap numpy sim; CPU is faster

# Single-threaded torch is ~11x faster here than the 12-thread default: the
# policy network is tiny (135 -> 256 -> 256 -> 9), so BLAS thread
# synchronisation costs far more than the matmuls themselves. Measured on this
# machine: 100 fps with 12 threads vs 1097 fps with 1. Parallelism is taken at
# the level of whole sweep runs instead (see training/sweep.py).
torch.set_num_threads(1)


# --------------------------------------------------------------------- seeding
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------- environment setup
def _single_env(seed: int | None = None, device=DEVICE) -> TransformedEnv:
    """One AgriGuard instance wrapped for TorchRL."""
    base = GymWrapper(
        AgriGuardEnv(seed=seed),
        categorical_action_encoding=True,   # integer actions: one shared convention
        device=device,
    )
    # RewardSum maintains a running per-episode return inside the env, so
    # completed-episode returns can be read off exactly even when several farms
    # are stepped in parallel (flattening a [num_envs, T] batch and slicing on
    # `done` splices rewards across different farms and is simply wrong).
    env = TransformedEnv(base, Compose(DoubleToFloat(), StepCounter(), RewardSum()))
    if seed is not None:
        env.set_seed(seed)
    return env


def make_env(seed: int | None = None, device=DEVICE, num_envs: int = 1):
    """Training environment. `num_envs > 1` returns a batched SerialEnv, which
    amortises TorchRL's per-step overhead across several simultaneous farms.

    Every algorithm calls this same factory, so all four agents are trained and
    scored on an identical environment definition.
    """
    if num_envs <= 1:
        return _single_env(seed, device)
    return SerialEnv(num_envs, [
        (lambda s=(None if seed is None else seed + 1000 * i): _single_env(s, device))
        for i in range(num_envs)
    ])


def make_eval_env(seed: int | None = None, device=DEVICE) -> TransformedEnv:
    """Always a single un-batched env, so per-episode agronomic metrics
    (matured / premature / dead plots) can be read straight off the simulator."""
    return _single_env(seed, device)


def env_dims(env) -> tuple[int, int]:
    return env.observation_spec["observation"].shape[-1], int(env.action_spec.space.n)



def completed_episode_returns(batch) -> list[float]:
    """Returns of every episode that terminated inside this collector batch.

    Relies on the RewardSum transform: ("next", "episode_reward") holds the
    running return, so the value at a `done` step *is* that episode's return.
    Correct for any collector batch shape, batched envs included.
    """
    done = batch["next", "done"].reshape(-1)
    if not bool(done.any()):
        return []
    ep_r = batch["next", "episode_reward"].reshape(-1)
    return ep_r[done].tolist()


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(policy, episodes: int = 12, seed0: int = 10_000, device=DEVICE) -> dict:
    """Greedy/stochastic-policy rollout on held-out seeds.

    Evaluation seeds start at 10_000 so they never overlap the training seeds,
    which is the generalisation evidence reported in the discussion section.
    """
    rets, yields, matured, premature, dead, lens = [], [], [], [], [], []
    for k in range(episodes):
        env = make_eval_env(seed=seed0 + k, device=device)
        td = env.reset()
        total, steps = 0.0, 0
        while True:
            td = policy(td)
            td = env.step(td)
            total += float(td["next", "reward"].sum())
            steps += 1
            done = bool(td["next", "done"].any()) or bool(td["next", "terminated"].any())
            trunc = bool(td["next"].get("truncated", torch.zeros(1)).any())
            if done or trunc:
                break
            td = td["next"]
        info = env.base_env._env.unwrapped._info()
        rets.append(total); lens.append(steps)
        yields.append(info["banked_yield"]); matured.append(info["matured"])
        premature.append(info["premature"]); dead.append(16 - info["alive"])
        env.close()
    return {
        "eval_return": float(np.mean(rets)), "eval_return_std": float(np.std(rets)),
        "eval_yield": float(np.mean(yields)), "eval_matured": float(np.mean(matured)),
        "eval_premature": float(np.mean(premature)), "eval_dead": float(np.mean(dead)),
        "eval_len": float(np.mean(lens)),
    }


# -------------------------------------------------------------------- logging
@dataclass
class RunLogger:
    """Writes one CSV of learning-curve rows plus one JSON summary per run."""
    algo: str
    run_id: str
    hparams: dict
    rows: list = field(default_factory=list)
    t0: float = field(default_factory=time.time)

    def log(self, **kw):
        self.rows.append(kw)

    def save(self, summary: dict) -> Path:
        d = LOGS / self.algo
        d.mkdir(parents=True, exist_ok=True)
        if self.rows:
            keys = sorted({k for r in self.rows for k in r})
            with open(d / f"{self.run_id}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.rows)
        summary = {"algo": self.algo, "run_id": self.run_id,
                   "wall_time_s": round(time.time() - self.t0, 1),
                   **{f"hp_{k}": v for k, v in self.hparams.items()}, **summary}
        (d / f"{self.run_id}.json").write_text(json.dumps(summary, indent=2))
        return d / f"{self.run_id}.json"


def moving_average(x, w: int = 10):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return x
    w = max(1, min(w, len(x)))
    return np.convolve(x, np.ones(w) / w, mode="valid")
