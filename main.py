"""
AgriGuard RL summative -- single entry point.

    uv run main.py                          random-policy demo + 3D viewer
    uv run main.py --mode demo --agent best run the best trained agent instead
    uv run main.py --mode train --algo ppo  train one algorithm
    uv run main.py --mode sweep --algo all  full 10-runs-per-algorithm sweep
    uv run main.py --mode plots             rebuild every report figure
    uv run main.py --mode baselines         random vs expert-heuristic reference
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


# --------------------------------------------------------------------- demo
def run_demo(agent: str = "random", episodes: int = 1, render: bool = True,
             delay: float = 0.06, verbose: bool = True):
    """Roll out a policy, printing the terminal view and driving the 3D viewer."""
    from environment import AgriGuardEnv, ACTION_NAMES
    from environment.rendering import ThreeJSRenderer

    env = AgriGuardEnv(seed=7)
    policy, label = _load_policy(agent, env)

    renderer = ThreeJSRenderer(open_browser=render) if render else None
    if renderer:
        print("[demo] leave the browser tab open; the field updates live\n")
        time.sleep(2.0)   # give the browser a moment to connect

    for ep in range(episodes):
        obs, _ = env.reset(seed=100 + ep)
        total, step = 0.0, 0
        while True:
            action = policy(obs, env)
            obs, reward, term, trunc, info = env.step(action)
            total += reward
            step += 1
            if renderer:
                renderer.update(env, {"action": ACTION_NAMES[action]})
                time.sleep(delay)
            if verbose and step % 8 == 0:
                print(f"\n--- {label} | step {step} | action {ACTION_NAMES[action]} "
                      f"| return {total:7.2f} ---")
                print(env.to_ascii())
            if term or trunc:
                break
        print(f"\n[demo] episode {ep+1}: return={total:.2f} "
              f"yield={info['banked_yield']:.2f} matured={info['matured']}/16 "
              f"premature={info['premature']} lost={16-info['alive']} "
              f"days={info['day']} steps={step}")

    if renderer:
        print("\n[demo] viewer stays live for 30s so the final field can be inspected")
        time.sleep(30)
        renderer.close()


def _load_policy(agent: str, env):
    """Return a `(obs, env) -> action` callable for 'random', 'expert' or a model."""
    if agent == "random":
        rng = np.random.default_rng(0)
        return (lambda o, e: int(rng.integers(0, e.action_space.n))), "random policy"
    if agent == "expert":
        from training.baselines import ExpertHeuristic
        exp = ExpertHeuristic()
        return (lambda o, e: exp(e)), "expert heuristic"

    # otherwise: a trained checkpoint
    import torch
    from training.baselines import load_trained_policy
    fn, label = load_trained_policy(agent)
    return (lambda o, e, _fn=fn: _fn(o)), label


# -------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="AgriGuard RL summative")
    ap.add_argument("--mode", default="demo",
                    choices=["demo", "train", "sweep", "plots", "baselines"])
    ap.add_argument("--algo", default="all",
                    choices=["all", "dqn", "reinforce", "a2c", "ppo"])
    ap.add_argument("--agent", default="random",
                    help="'random', 'expert', 'best', or a run_id / .pt path")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--frames", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--delay", type=float, default=0.06)
    a = ap.parse_args(argv)

    if a.mode == "demo":
        run_demo(a.agent, a.episodes, render=not a.no_render, delay=a.delay)

    elif a.mode == "train":
        if a.algo in ("all", "dqn"):
            from training.dqn_training import DQNConfig, train_dqn
            train_dqn(DQNConfig(run_id="dqn_default", total_frames=a.frames))
        for alg in (["reinforce", "a2c", "ppo"] if a.algo == "all"
                    else ([a.algo] if a.algo != "dqn" else [])):
            from training.pg_training import PGConfig, train_pg
            train_pg(PGConfig(algo=alg, run_id=f"{alg}_default", total_frames=a.frames))

    elif a.mode == "sweep":
        from training.sweep import run_sweep
        run_sweep(a.algo, a.workers, a.frames)
        from training.plotting import make_all
        make_all()

    elif a.mode == "plots":
        from training.plotting import make_all
        make_all()

    elif a.mode == "baselines":
        from training.baselines import report_baselines
        report_baselines()

    return 0


if __name__ == "__main__":
    sys.exit(main())
