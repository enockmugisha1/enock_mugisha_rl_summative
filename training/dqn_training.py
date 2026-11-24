"""
DQN (value-based) agent for AgriGuard, implemented with TorchRL.

Exposes the controls the report's DQN table sweeps: learning rate, discount,
network width/depth, replay-buffer capacity, batch size, epsilon schedule
(exploration), target-network update rate, Double-DQN on/off and the
update-to-data ratio.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.postprocs import MultiStep
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate

from training.common import (DEVICE, MODELS, RunLogger, completed_episode_returns,
                             env_dims, evaluate, make_env, set_seed)


@dataclass
class DQNConfig:
    run_id: str = "dqn_default"
    seed: int = 0
    total_frames: int = 60_000
    frames_per_batch: int = 1_000
    init_random_frames: int = 2_000
    # --- shared controls
    lr: float = 2.5e-4
    gamma: float = 0.99
    hidden: int = 256
    depth: int = 2
    # --- replay
    buffer_size: int = 100_000
    batch_size: int = 128
    utd: int = 8                  # gradient updates per collected batch
    # --- exploration
    eps_init: float = 1.0
    eps_end: float = 0.05
    eps_anneal_frac: float = 0.5  # fraction of training spent annealing epsilon
    # --- credit assignment
    n_step: int = 5               # n-step returns; 1 = classic one-step TD
    # --- target network / stabilisation
    target_tau: float = 0.005     # soft-update rate
    double_dqn: bool = True
    # Huber ("smooth_l1") bounds the gradient of rare large TD errors. Harvest
    # (+yield) and plot-death (-3) are two orders of magnitude bigger than the
    # routine per-step costs (~0.02-0.09), so an l2 objective lets a handful of
    # transitions dominate the batch gradient and the value head diverges.
    loss_fn: str = "smooth_l1"
    max_grad_norm: float = 10.0
    num_envs: int = 8            # batched farms per collector step
    eval_episodes: int = 10


def build_agent(obs_dim: int, n_act: int, cfg: DQNConfig, action_spec):
    net = MLP(in_features=obs_dim, out_features=n_act,
              num_cells=[cfg.hidden] * cfg.depth,
              activation_class=torch.nn.ReLU).to(DEVICE)
    # small final layer so initial Q-values are near zero -> stable early targets
    with torch.no_grad():
        last = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.mul_(0.1); last.bias.zero_()
    actor = QValueActor(net, in_keys=["observation"], spec=action_spec,
                        action_space="categorical").to(DEVICE)
    return actor


def train_dqn(cfg: DQNConfig, verbose: bool = True) -> dict:
    set_seed(cfg.seed)
    proto = make_env(seed=cfg.seed)
    obs_dim, n_act = env_dims(proto)
    action_spec = proto.action_spec
    proto.close()

    actor = build_agent(obs_dim, n_act, cfg, action_spec)
    explorer = EGreedyModule(
        spec=action_spec, eps_init=cfg.eps_init, eps_end=cfg.eps_end,
        annealing_num_steps=max(1, int(cfg.total_frames * cfg.eps_anneal_frac)),
    ).to(DEVICE)
    policy_explore = TensorDictSequential(actor, explorer)

    loss_mod = DQNLoss(actor, loss_function=cfg.loss_fn, delay_value=True,
                       double_dqn=cfg.double_dqn, action_space="categorical")
    loss_mod.make_value_estimator(gamma=cfg.gamma)
    target_updater = SoftUpdate(loss_mod, eps=1.0 - cfg.target_tau)
    optim = torch.optim.Adam(loss_mod.parameters(), lr=cfg.lr)

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.buffer_size, device=DEVICE),
        batch_size=cfg.batch_size,
    )
    # n-step returns. With one-step TD the harvest payoff sits ~100 slots beyond
    # the irrigation decisions that earn it, and WAIT -- the only zero-cost
    # action -- dominates long before that reward propagates back. Bootstrapping
    # over n transitions moves the signal back n times faster and is what makes
    # DQN competitive on this task.
    postproc = MultiStep(gamma=cfg.gamma, n_steps=cfg.n_step) if cfg.n_step > 1 else None
    collector = Collector(
        create_env_fn=lambda: make_env(seed=cfg.seed, num_envs=cfg.num_envs),
        policy=policy_explore,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        postproc=postproc,
        device=DEVICE,
    )

    log = RunLogger("dqn", cfg.run_id, asdict(cfg))
    frames = 0
    ep_returns: list[float] = []
    best_eval = -1e9

    for batch in collector:
        frames += batch.numel()
        buffer.extend(batch.reshape(-1).cpu())

        # completed-episode returns inside this batch (for the learning curve)
        ep_returns.extend(completed_episode_returns(batch))

        losses, qvals = [], []
        if frames > cfg.init_random_frames:
            for _ in range(cfg.utd):
                sample = buffer.sample().to(DEVICE)
                ld = loss_mod(sample)
                loss = ld["loss"]
                optim.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), cfg.max_grad_norm)
                optim.step()
                target_updater.step()
                losses.append(float(loss.detach()))
                qvals.append(float(sample["action_value"].max(-1).values.mean())
                             if "action_value" in sample.keys() else float("nan"))
            explorer.step(cfg.frames_per_batch)

        log.log(frames=frames,
                train_return=float(np.mean(ep_returns[-10:])) if ep_returns else float("nan"),
                loss=float(np.mean(losses)) if losses else float("nan"),
                q_value=float(np.nanmean(qvals)) if qvals else float("nan"),
                epsilon=float(explorer.eps),
                grad_norm=float(gn) if losses else float("nan"))

        if verbose and (frames // cfg.frames_per_batch) % 10 == 0:
            print(f"  [dqn {cfg.run_id}] {frames:>6}/{cfg.total_frames} "
                  f"ret={log.rows[-1]['train_return']:7.2f} "
                  f"loss={log.rows[-1]['loss']:8.4f} eps={explorer.eps:.3f}")

    collector.shutdown()

    # greedy evaluation on held-out seeds
    metrics = evaluate(actor, episodes=cfg.eval_episodes)
    metrics["final_train_return"] = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
    metrics["episodes_seen"] = len(ep_returns)
    out = MODELS / "dqn"; out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": actor.state_dict(), "cfg": asdict(cfg),
                "obs_dim": obs_dim, "n_act": n_act}, out / f"{cfg.run_id}.pt")
    log.save(metrics)
    if verbose:
        print(f"  [dqn {cfg.run_id}] eval_return={metrics['eval_return']:.2f} "
              f"matured={metrics['eval_matured']:.1f} dead={metrics['eval_dead']:.1f}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-frames", type=int, default=60_000)
    ap.add_argument("--run-id", default="dqn_default")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train_dqn(DQNConfig(run_id=a.run_id, seed=a.seed, total_frames=a.total_frames))
