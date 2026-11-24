"""
Hyperparameter sweeps: 10 meaningfully different runs for each of the four
algorithms, all on the identical AgriGuard environment and the same frame budget.

Every run varies controls that matter for its own algorithm family:

  DQN        learning rate, discount, replay capacity, batch size, epsilon floor
             and annealing fraction, target soft-update rate, Double-DQN on/off,
             update-to-data ratio, network width/depth.
  REINFORCE  rollout length, learning rate, entropy coefficient, discount,
             advantage normalisation, critic weight, network size.
  A2C        rollout length, entropy coefficient, GAE lambda, critic weight,
             learning rate, discount, gradient clipping.
  PPO        clip epsilon, epochs per batch, minibatch size, GAE lambda,
             entropy coefficient, learning rate, rollout length.

Runs execute in parallel worker processes because torch is pinned to one thread
per process (see training/common.py), so the 12 cores are used across runs
rather than inside one matmul.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import pandas as pd

from training.common import LOGS

FRAMES = 150_000          # identical budget for every run, so the sweep is fair


# ---------------------------------------------------------------- DQN grid
def dqn_grid(frames: int = FRAMES) -> list[dict]:
    """Centred on the regime found by the diagnostic study in the report
    (Huber objective, high update-to-data ratio, sustained exploration), then
    varying one control at a time so each row is attributable.

    Spans every DQN-specific control the brief asks for: exploration schedule,
    replay capacity, target-network update rate, update-to-data ratio,
    n-step credit assignment, Double-DQN and the regression objective.
    """
    base = dict(total_frames=frames, frames_per_batch=1_000, eval_episodes=8,
                init_random_frames=10_000, gamma=0.99, lr=5e-5, eps_init=1.0,
                eps_end=0.25, eps_anneal_frac=0.8, n_step=3, target_tau=0.005,
                buffer_size=100_000, batch_size=128, utd=64, double_dqn=True,
                loss_fn="smooth_l1")
    return [
        {**base, "run_id": "dqn01_tuned_best",    "utd": 128},
        {**base, "run_id": "dqn02_utd_starved",   "utd": 8},
        {**base, "run_id": "dqn03_utd_mid",       "utd": 64},
        {**base, "run_id": "dqn04_lr_high",       "lr": 2.5e-4},
        {**base, "run_id": "dqn05_explore_greedy","eps_end": 0.02, "eps_anneal_frac": 0.3},
        {**base, "run_id": "dqn06_explore_max",   "eps_end": 0.40, "eps_anneal_frac": 0.9},
        {**base, "run_id": "dqn07_target_fast",   "target_tau": 0.05},
        {**base, "run_id": "dqn08_target_slow",   "target_tau": 0.001},
        {**base, "run_id": "dqn09_buffer_small",  "buffer_size": 5_000},
        {**base, "run_id": "dqn10_naive_l2_1step","n_step": 1, "double_dqn": False,
                                                  "loss_fn": "l2", "utd": 8, "lr": 2.5e-4,
                                                  "eps_end": 0.05, "eps_anneal_frac": 0.5},
    ]


# ---------------------------------------------------------- REINFORCE grid
def reinforce_grid(frames: int = FRAMES) -> list[dict]:
    base = dict(algo="reinforce", total_frames=frames, eval_episodes=8)
    return [
        {**base, "run_id": "rf01_baseline",     "frames_per_batch": 4_000, "lr": 3e-4, "entropy_coeff": 0.01,  "gamma": 0.99, "normalize_advantage": True,  "critic_coeff": 0.5},
        {**base, "run_id": "rf02_rollout_long", "frames_per_batch": 8_000, "lr": 3e-4, "entropy_coeff": 0.01,  "gamma": 0.99, "normalize_advantage": True,  "critic_coeff": 0.5},
        {**base, "run_id": "rf03_rollout_short","frames_per_batch": 2_000, "lr": 3e-4, "entropy_coeff": 0.01,  "gamma": 0.99, "normalize_advantage": True,  "critic_coeff": 0.5},
        {**base, "run_id": "rf04_lr_high",      "frames_per_batch": 4_000, "lr": 1e-3, "entropy_coeff": 0.01,  "gamma": 0.99, "normalize_advantage": True,  "critic_coeff": 0.5},
        {**base, "run_id": "rf05_lr_low",       "frames_per_batch": 4_000, "lr": 1e-4, "entropy_coeff": 0.01,  "gamma": 0.99, "normalize_advantage": True,  "critic_coeff": 0.5},
        {**base, "run_id": "rf06_entropy_high", "frames_per_batch": 4_000, "lr": 3e-4, "entropy_coeff": 0.05,  "gamma": 0.99, "normalize_advantage": True,  "critic_coeff": 0.5},
        {**base, "run_id": "rf07_entropy_zero", "frames_per_batch": 4_000, "lr": 3e-4, "entropy_coeff": 0.0,   "gamma": 0.99, "normalize_advantage": True,  "critic_coeff": 0.5},
        {**base, "run_id": "rf08_no_advnorm",   "frames_per_batch": 4_000, "lr": 3e-4, "entropy_coeff": 0.01,  "gamma": 0.99, "normalize_advantage": False, "critic_coeff": 0.5},
        {**base, "run_id": "rf09_gamma_low",    "frames_per_batch": 4_000, "lr": 3e-4, "entropy_coeff": 0.01,  "gamma": 0.95, "normalize_advantage": True,  "critic_coeff": 0.5},
        {**base, "run_id": "rf10_critic_heavy", "frames_per_batch": 4_000, "lr": 3e-4, "entropy_coeff": 0.01,  "gamma": 0.99, "normalize_advantage": True,  "critic_coeff": 1.0, "hidden": 512},
    ]


# ---------------------------------------------------------------- A2C grid
def a2c_grid(frames: int = FRAMES) -> list[dict]:
    base = dict(algo="a2c", total_frames=frames, eval_episodes=8)
    return [
        {**base, "run_id": "a2c01_baseline",      "frames_per_batch": 2_000, "lr": 3e-4, "entropy_coeff": 0.01, "lmbda": 0.95, "critic_coeff": 0.5, "gamma": 0.99},
        {**base, "run_id": "a2c02_rollout_short", "frames_per_batch": 1_000, "lr": 3e-4, "entropy_coeff": 0.01, "lmbda": 0.95, "critic_coeff": 0.5, "gamma": 0.99},
        {**base, "run_id": "a2c03_rollout_long",  "frames_per_batch": 8_000, "lr": 3e-4, "entropy_coeff": 0.01, "lmbda": 0.95, "critic_coeff": 0.5, "gamma": 0.99},
        {**base, "run_id": "a2c04_lr_high",       "frames_per_batch": 2_000, "lr": 1e-3, "entropy_coeff": 0.01, "lmbda": 0.95, "critic_coeff": 0.5, "gamma": 0.99},
        {**base, "run_id": "a2c05_lr_low",        "frames_per_batch": 2_000, "lr": 1e-4, "entropy_coeff": 0.01, "lmbda": 0.95, "critic_coeff": 0.5, "gamma": 0.99},
        {**base, "run_id": "a2c06_entropy_high",  "frames_per_batch": 2_000, "lr": 3e-4, "entropy_coeff": 0.05, "lmbda": 0.95, "critic_coeff": 0.5, "gamma": 0.99},
        {**base, "run_id": "a2c07_entropy_zero",  "frames_per_batch": 2_000, "lr": 3e-4, "entropy_coeff": 0.0,  "lmbda": 0.95, "critic_coeff": 0.5, "gamma": 0.99},
        {**base, "run_id": "a2c08_lambda_low",    "frames_per_batch": 2_000, "lr": 3e-4, "entropy_coeff": 0.01, "lmbda": 0.70, "critic_coeff": 0.5, "gamma": 0.99},
        {**base, "run_id": "a2c09_lambda_one",    "frames_per_batch": 2_000, "lr": 3e-4, "entropy_coeff": 0.01, "lmbda": 1.00, "critic_coeff": 0.5, "gamma": 0.99},
        {**base, "run_id": "a2c10_critic_heavy",  "frames_per_batch": 2_000, "lr": 3e-4, "entropy_coeff": 0.01, "lmbda": 0.95, "critic_coeff": 1.0, "gamma": 0.995, "max_grad_norm": 0.5},
    ]


# ---------------------------------------------------------------- PPO grid
def ppo_grid(frames: int = FRAMES) -> list[dict]:
    base = dict(algo="ppo", total_frames=frames, eval_episodes=8)
    return [
        {**base, "run_id": "ppo01_baseline",     "frames_per_batch": 4_000, "lr": 3e-4, "clip_epsilon": 0.2, "ppo_epochs": 6,  "minibatch_size": 256, "entropy_coeff": 0.01, "lmbda": 0.95},
        {**base, "run_id": "ppo02_clip_tight",   "frames_per_batch": 4_000, "lr": 3e-4, "clip_epsilon": 0.05,"ppo_epochs": 6,  "minibatch_size": 256, "entropy_coeff": 0.01, "lmbda": 0.95},
        {**base, "run_id": "ppo03_clip_loose",   "frames_per_batch": 4_000, "lr": 3e-4, "clip_epsilon": 0.4, "ppo_epochs": 6,  "minibatch_size": 256, "entropy_coeff": 0.01, "lmbda": 0.95},
        {**base, "run_id": "ppo04_epochs_many",  "frames_per_batch": 4_000, "lr": 3e-4, "clip_epsilon": 0.2, "ppo_epochs": 16, "minibatch_size": 256, "entropy_coeff": 0.01, "lmbda": 0.95},
        {**base, "run_id": "ppo05_epochs_one",   "frames_per_batch": 4_000, "lr": 3e-4, "clip_epsilon": 0.2, "ppo_epochs": 1,  "minibatch_size": 256, "entropy_coeff": 0.01, "lmbda": 0.95},
        {**base, "run_id": "ppo06_lr_high",      "frames_per_batch": 4_000, "lr": 1e-3, "clip_epsilon": 0.2, "ppo_epochs": 6,  "minibatch_size": 256, "entropy_coeff": 0.01, "lmbda": 0.95},
        {**base, "run_id": "ppo07_entropy_high", "frames_per_batch": 4_000, "lr": 3e-4, "clip_epsilon": 0.2, "ppo_epochs": 6,  "minibatch_size": 256, "entropy_coeff": 0.05, "lmbda": 0.95},
        {**base, "run_id": "ppo08_entropy_zero", "frames_per_batch": 4_000, "lr": 3e-4, "clip_epsilon": 0.2, "ppo_epochs": 6,  "minibatch_size": 256, "entropy_coeff": 0.0,  "lmbda": 0.95},
        {**base, "run_id": "ppo09_lambda_low",   "frames_per_batch": 4_000, "lr": 3e-4, "clip_epsilon": 0.2, "ppo_epochs": 6,  "minibatch_size": 256, "entropy_coeff": 0.01, "lmbda": 0.80},
        {**base, "run_id": "ppo10_big_batch",    "frames_per_batch": 8_000, "lr": 3e-4, "clip_epsilon": 0.2, "ppo_epochs": 10, "minibatch_size": 512, "entropy_coeff": 0.01, "lmbda": 0.95, "hidden": 512},
    ]


GRIDS = {"dqn": dqn_grid, "reinforce": reinforce_grid, "a2c": a2c_grid, "ppo": ppo_grid}


# --------------------------------------------------------------- execution
def _run_one(spec: dict) -> dict:
    """Worker entry point. Imported lazily so each process builds its own torch."""
    algo = spec.pop("algo", "dqn") if "algo" in spec else "dqn"
    # Resume support: a finished run leaves logs/<algo>/<run_id>.json behind, so
    # an interrupted sweep can be relaunched without repeating completed work.
    done_file = LOGS / algo / f"{spec['run_id']}.json"
    if done_file.exists():
        import json as _json
        prev = _json.loads(done_file.read_text())
        print(f"  skip {algo:9s} {spec['run_id']:22s} (already complete, "
              f"eval_return={prev.get('eval_return', float('nan')):.2f})", flush=True)
        return prev
    try:
        if algo == "dqn":
            from training.dqn_training import DQNConfig, train_dqn
            m = train_dqn(DQNConfig(**spec), verbose=False)
        else:
            from training.pg_training import PGConfig, train_pg
            m = train_pg(PGConfig(algo=algo, **spec), verbose=False)
        print(f"  done {algo:9s} {spec['run_id']:22s} "
              f"eval={m['eval_return']:8.2f} matured={m['eval_matured']:4.1f}", flush=True)
        return {"algo": algo, **spec, **m}
    except Exception as e:  # keep the sweep alive if one configuration fails
        print(f"  FAIL {algo} {spec.get('run_id')}: {type(e).__name__}: {e}", flush=True)
        return {"algo": algo, **spec, "error": f"{type(e).__name__}: {e}"}


def run_sweep(which: str = "all", workers: int = 5, frames: int = FRAMES) -> Path:
    algos = list(GRIDS) if which == "all" else [which]
    specs: list[dict] = []
    for a in algos:
        for s in GRIDS[a](frames):
            s = dict(s)
            s.setdefault("algo", a)
            specs.append(s)

    print(f"[sweep] {len(specs)} runs across {algos} | {frames:,} frames each "
          f"| {workers} parallel workers")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(_run_one, specs)
    dt = time.time() - t0

    df = pd.DataFrame(results)
    out = LOGS / "sweep_results.csv"
    if out.exists():                       # merge with any earlier partial sweep
        old = pd.read_csv(out)
        df = pd.concat([old[~old["run_id"].isin(df["run_id"])], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"[sweep] finished in {dt/60:.1f} min -> {out}")

    ok = df[df.get("error").isna()] if "error" in df else df
    if len(ok):
        best = ok.sort_values("eval_return", ascending=False)
        print("\n[sweep] best run per algorithm:")
        for a, g in best.groupby("algo"):
            r = g.iloc[0]
            print(f"  {a:9s} {r['run_id']:22s} eval_return={r['eval_return']:8.2f} "
                  f"matured={r['eval_matured']:.1f}/16 dead={r['eval_dead']:.1f}")
        (LOGS / "best_runs.json").write_text(json.dumps(
            {a: g.iloc[0]["run_id"] for a, g in best.groupby("algo")}, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="all", choices=["all", *GRIDS])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--frames", type=int, default=FRAMES)
    a = ap.parse_args()
    run_sweep(a.algo, a.workers, a.frames)
