"""
Reference policies and checkpoint loading.

The expert heuristic is not an RL agent: it is a hand-written agronomic
controller used as an upper reference point in the report, in the same way the
random policy is the lower reference point. Reporting both makes the learning
curves interpretable -- "+90" means nothing on its own, but it is informative
against random (+13) and expert (+184).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from environment.custom_env import (FERTILIZE, HARVEST, IRRIGATE, MOVE_E, MOVE_N,
                                    MOVE_S, MOVE_W, TREAT_PEST, WAIT)
from training.common import LOGS, MODELS


# ------------------------------------------------------------ expert policy
class ExpertHeuristic:
    """Triage controller that commits to one plot and one job until it is done.

    Committing matters: re-planning every slot makes the rover oscillate between
    two equally urgent plots and never actually act, which scores worse than
    random. This is itself a useful illustration of why the task needs a policy
    rather than a greedy rule.
    """

    def __init__(self):
        self.target = None
        self.job = None

    def reset(self):
        self.target = None
        self.job = None

    @staticmethod
    def _step_towards(env, target):
        g = env.GRID
        r, c = divmod(env.rover, g)
        tr, tc = divmod(target, g)
        if r < tr:
            return MOVE_S
        if r > tr:
            return MOVE_N
        if c < tc:
            return MOVE_E
        if c > tc:
            return MOVE_W
        return None

    def _job_for(self, env, i):
        """Highest-priority job for plot i, with a score for triage."""
        if env.growth[i] >= env.MATURE_GROWTH:
            return HARVEST, 100 + env.growth[i]
        if env.pest[i] > 0.20 and env.pest_stock >= 1:
            return TREAT_PEST, 50 + env.pest[i] * 10
        if env.moisture[i] < env.HEALTHY_LO and env.tank > env.IRRIGATE_WATER:
            return IRRIGATE, 40 + (env.HEALTHY_LO - env.moisture[i]) * 20
        if env.nutrient[i] < 0.22 and env.fert_stock >= 1:
            return FERTILIZE, 20
        return None, -1

    def __call__(self, env):
        active = env.alive & ~env.harvested
        if not active.any():
            return WAIT
        # stay on the committed job while it is still the right one
        if self.target is not None and active[self.target]:
            job, _ = self._job_for(env, self.target)
            if job == self.job:
                mv = self._step_towards(env, self.target)
                return mv if mv is not None else self.job
        # otherwise re-plan once, preferring nearby work
        best, best_job, best_score = None, None, -1.0
        g = env.GRID
        r, c = divmod(env.rover, g)
        for i in range(env.n_plots):
            if not active[i]:
                continue
            job, score = self._job_for(env, i)
            if job is None:
                continue
            tr, tc = divmod(i, g)
            score -= 0.6 * (abs(r - tr) + abs(c - tc))
            if score > best_score:
                best, best_job, best_score = i, job, score
        if best is None:
            return WAIT
        self.target, self.job = best, best_job
        mv = self._step_towards(env, best)
        return mv if mv is not None else best_job


# ------------------------------------------------------- checkpoint loading
def _resolve_checkpoint(name: str) -> Path:
    """Accept a path, a run_id, or 'best' (highest held-out eval return)."""
    p = Path(name)
    if p.suffix == ".pt" and p.exists():
        return p
    if name == "best":
        f = LOGS / "sweep_results.csv"
        if f.exists():
            import pandas as pd
            df = pd.read_csv(f)
            if "error" in df:
                df = df[df["error"].isna()]
            if len(df):
                row = df.sort_values("eval_return", ascending=False).iloc[0]
                name = str(row["run_id"])
                print(f"[baselines] best agent = {name} "
                      f"({row['algo']}, eval_return={row['eval_return']:.2f})")
    for sub in ("dqn", "pg"):
        c = MODELS / sub / f"{name}.pt"
        if c.exists():
            return c
    found = sorted(MODELS.glob("*/*.pt"))
    raise FileNotFoundError(
        f"no checkpoint for {name!r}. Available: {[f.stem for f in found]}")


def load_trained_policy(name: str):
    """Return `(obs_array -> int action, label)` for a saved checkpoint."""
    ckpt_path = _resolve_checkpoint(name)
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg = ckpt["cfg"]
    obs_dim, n_act = ckpt["obs_dim"], ckpt["n_act"]
    is_dqn = ckpt_path.parent.name == "dqn"

    from training.common import make_eval_env
    proto = make_eval_env(seed=0)
    spec = proto.action_spec
    proto.close()

    if is_dqn:
        from training.dqn_training import DQNConfig, build_agent
        keep = {k: v for k, v in cfg.items() if k in DQNConfig.__dataclass_fields__}
        actor = build_agent(obs_dim, n_act, DQNConfig(**keep), spec)
        actor.load_state_dict(ckpt["state_dict"])
        actor.eval()

        @torch.no_grad()
        def act(obs):
            td = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            return int(td[1].argmax()) if isinstance(td, tuple) else int(td["action"])
    else:
        from training.pg_training import PGConfig, build_actor_critic
        keep = {k: v for k, v in cfg.items() if k in PGConfig.__dataclass_fields__}
        actor, _ = build_actor_critic(obs_dim, n_act, PGConfig(**keep), spec)
        actor.load_state_dict(ckpt["state_dict"])
        actor.eval()

        @torch.no_grad()
        def act(obs):
            from tensordict import TensorDict
            td = TensorDict({"observation":
                             torch.as_tensor(obs, dtype=torch.float32)}, [])
            # Sample from the policy, exactly as evaluate() does. These agents
            # are trained with default_interaction_type="random", so the greedy
            # argmax is a *different* policy from the one the reported scores
            # were measured on -- on this task it collapses onto a single action.
            return int(actor(td)["action"])

    return act, f"{ckpt_path.parent.name}/{ckpt_path.stem}"


# ----------------------------------------------------------------- report
def report_baselines(episodes: int = 25):
    """Print the random and expert reference scores quoted in the report."""
    from environment import AgriGuardEnv

    rows = {}
    rng = np.random.default_rng(0)
    for label, make in (("random", None), ("expert heuristic", ExpertHeuristic)):
        res = []
        for ep in range(episodes):
            env = AgriGuardEnv(seed=ep)
            env.reset(seed=ep)
            pol = make() if make else None
            total = 0.0
            while True:
                a = pol(env) if pol else int(rng.integers(0, 9))
                _, r, term, trunc, info = env.step(a)
                total += r
                if term or trunc:
                    break
            res.append((total, info["banked_yield"], info["matured"],
                        info["premature"], 16 - info["alive"]))
        arr = np.array(res)
        rows[label] = dict(ret=arr[:, 0].mean(), sd=arr[:, 0].std(),
                           yld=arr[:, 1].mean(), matured=arr[:, 2].mean(),
                           premature=arr[:, 3].mean(), dead=arr[:, 4].mean())
        r = rows[label]
        print(f"{label:18s} return={r['ret']:8.2f} ±{r['sd']:5.1f}  "
              f"yield={r['yld']:5.2f}  matured={r['matured']:4.1f}/16  "
              f"premature={r['premature']:4.1f}  lost={r['dead']:4.1f}")
    (LOGS / "baselines.json").write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    report_baselines()
