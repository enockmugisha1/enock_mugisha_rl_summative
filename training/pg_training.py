"""
Policy-gradient agents for AgriGuard, implemented with TorchRL:

  * REINFORCE  -- Monte-Carlo policy gradient with a learned value baseline
                  (GAE with lambda = 1.0 recovers the full-return estimator)
  * A2C        -- synchronous advantage actor-critic, n-step bootstrapped
  * PPO        -- clipped-surrogate actor-critic with multiple epochs per batch

All three share the same actor and critic architecture and the same environment
factory as DQN, so differences in the report come from the algorithms rather
than from the networks or the task.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer, SamplerWithoutReplacement
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

from training.common import (completed_episode_returns, DEVICE, MODELS, RunLogger, env_dims, evaluate,
                             make_env, set_seed)

ALGOS = ("reinforce", "a2c", "ppo")


@dataclass
class PGConfig:
    algo: str = "ppo"
    run_id: str = "ppo_default"
    seed: int = 0
    total_frames: int = 60_000
    frames_per_batch: int = 2_000      # rollout length per policy update
    num_envs: int = 8
    # --- shared controls
    lr: float = 3e-4
    gamma: float = 0.99
    hidden: int = 256
    depth: int = 2
    max_grad_norm: float = 1.0
    # --- advantage estimation
    lmbda: float = 0.95               # GAE lambda (forced to 1.0 for REINFORCE)
    normalize_advantage: bool = True
    # --- policy-gradient specifics
    entropy_coeff: float = 0.01       # exploration pressure
    critic_coeff: float = 0.5         # value-loss weight
    clip_epsilon: float = 0.2         # PPO only
    ppo_epochs: int = 6               # PPO only
    minibatch_size: int = 256         # PPO only
    eval_episodes: int = 10


def build_actor_critic(obs_dim: int, n_act: int, cfg: PGConfig, action_spec):
    """Separate policy and value heads, matching the DQN network budget."""
    policy_net = MLP(in_features=obs_dim, out_features=n_act,
                     num_cells=[cfg.hidden] * cfg.depth,
                     activation_class=torch.nn.Tanh).to(DEVICE)
    # Near-uniform initial policy: small final weights keep early entropy high
    # so none of the nine farm actions is starved before learning begins.
    with torch.no_grad():
        last = [m for m in policy_net.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.mul_(0.01)
        last.bias.zero_()

    policy_module = TensorDictModule(policy_net, in_keys=["observation"],
                                     out_keys=["logits"])
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["logits"],
        out_keys=["action"],
        spec=action_spec,
        # Integer-valued categorical over the 9 farm actions, matching the
        # categorical action spec used by every algorithm here.
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
        default_interaction_type="random",
    ).to(DEVICE)

    value_net = MLP(in_features=obs_dim, out_features=1,
                    num_cells=[cfg.hidden] * cfg.depth,
                    activation_class=torch.nn.Tanh).to(DEVICE)
    critic = ValueOperator(value_net, in_keys=["observation"]).to(DEVICE)
    return actor, critic


def _make_loss(cfg: PGConfig, actor, critic):
    """Build the algorithm-specific loss plus its advantage estimator."""
    if cfg.algo == "ppo":
        loss = ClipPPOLoss(
            actor_network=actor, critic_network=critic,
            clip_epsilon=cfg.clip_epsilon, entropy_bonus=cfg.entropy_coeff > 0,
            entropy_coeff=cfg.entropy_coeff, critic_coeff=cfg.critic_coeff,
            normalize_advantage=cfg.normalize_advantage, loss_critic_type="l2",
        )
        lmbda = cfg.lmbda
    elif cfg.algo == "a2c":
        loss = A2CLoss(
            actor_network=actor, critic_network=critic,
            entropy_bonus=cfg.entropy_coeff > 0, entropy_coeff=cfg.entropy_coeff,
            critic_coeff=cfg.critic_coeff, loss_critic_type="l2",
        )
        lmbda = cfg.lmbda
    elif cfg.algo == "reinforce":
        loss = ReinforceLoss(actor_network=actor, critic_network=critic,
                             loss_critic_type="l2")
        # lambda = 1 makes the advantage the full discounted return minus the
        # baseline, i.e. the Monte-Carlo REINFORCE estimator.
        lmbda = 1.0
    else:
        raise ValueError(f"unknown algo {cfg.algo!r}")

    adv = GAE(gamma=cfg.gamma, lmbda=lmbda, value_network=critic,
              average_gae=cfg.normalize_advantage and cfg.algo != "ppo")
    loss.make_value_estimator(adv)
    return loss, adv


def _entropy_of(td) -> float:
    """Mean policy entropy over a batch, used for the PG entropy figure."""
    logits = td["logits"].reshape(-1, td["logits"].shape[-1])
    logp = torch.log_softmax(logits, dim=-1)
    return float(-(logp.exp() * logp).sum(-1).mean())


def train_pg(cfg: PGConfig, verbose: bool = True) -> dict:
    assert cfg.algo in ALGOS, cfg.algo
    set_seed(cfg.seed)
    proto = make_env(seed=cfg.seed)
    obs_dim, n_act = env_dims(proto)
    action_spec = proto.action_spec
    proto.close()

    actor, critic = build_actor_critic(obs_dim, n_act, cfg, action_spec)
    loss_mod, adv_mod = _make_loss(cfg, actor, critic)
    optim = torch.optim.Adam(loss_mod.parameters(), lr=cfg.lr)

    collector = Collector(
        create_env_fn=lambda: make_env(seed=cfg.seed, num_envs=cfg.num_envs),
        policy=actor,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=DEVICE,
    )
    # PPO reuses each batch for several epochs; REINFORCE/A2C use it once.
    replay = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.frames_per_batch, device=DEVICE),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.minibatch_size,
    )

    log = RunLogger(cfg.algo, cfg.run_id, asdict(cfg))
    frames = 0
    ep_returns: list[float] = []

    for batch in collector:
        frames += batch.numel()

        ep_returns.extend(completed_episode_returns(batch))

        with torch.no_grad():
            adv_mod(batch)

        ent = _entropy_of(batch)
        objective_losses, critic_losses, grad_norms = [], [], []

        if cfg.algo == "ppo":
            flat = batch.reshape(-1)
            for _ in range(cfg.ppo_epochs):
                replay.empty()
                replay.extend(flat)
                n_mb = max(1, cfg.frames_per_batch // cfg.minibatch_size)
                for _ in range(n_mb):
                    mb = replay.sample()
                    ld = loss_mod(mb)
                    total = sum(v for k, v in ld.items() if k.startswith("loss_"))
                    optim.zero_grad(set_to_none=True)
                    total.backward()
                    gn = torch.nn.utils.clip_grad_norm_(loss_mod.parameters(),
                                                        cfg.max_grad_norm)
                    optim.step()
                    objective_losses.append(float(ld.get("loss_objective", torch.tensor(0.)).detach()))
                    critic_losses.append(float(ld.get("loss_critic", torch.tensor(0.)).detach()))
                    grad_norms.append(float(gn))
        else:
            # One gradient step on the whole on-policy batch (A2C / REINFORCE).
            flat = batch.reshape(-1)
            ld = loss_mod(flat)
            if cfg.algo == "reinforce":
                # TorchRL's ReinforceLoss exposes neither an entropy bonus nor a
                # critic weight, so both are applied explicitly here. Without this
                # the entropy_coeff / critic_coeff columns of the REINFORCE sweep
                # would be inert and those rows would be silent duplicates.
                actor_key = "loss_actor" if "loss_actor" in ld.keys() else "loss_objective"
                value_key = "loss_value" if "loss_value" in ld.keys() else "loss_critic"
                total = ld[actor_key] + cfg.critic_coeff * ld[value_key]
                if cfg.entropy_coeff > 0:
                    dist = actor.get_dist(flat.select(*actor.in_keys, strict=False))
                    total = total - cfg.entropy_coeff * dist.entropy().mean()
            else:
                total = sum(v for k, v in ld.items() if k.startswith("loss_"))
            optim.zero_grad(set_to_none=True)
            total.backward()
            gn = torch.nn.utils.clip_grad_norm_(loss_mod.parameters(), cfg.max_grad_norm)
            optim.step()
            key = "loss_objective" if "loss_objective" in ld.keys() else "loss_actor"
            objective_losses.append(float(ld.get(key, torch.tensor(0.)).detach()))
            critic_losses.append(float(ld.get("loss_critic",
                                              ld.get("loss_value", torch.tensor(0.))).detach()))
            grad_norms.append(float(gn))

        log.log(frames=frames,
                train_return=float(np.mean(ep_returns[-10:])) if ep_returns else float("nan"),
                entropy=ent,
                loss_objective=float(np.mean(objective_losses)),
                loss_critic=float(np.mean(critic_losses)),
                grad_norm=float(np.mean(grad_norms)))

        if verbose and (frames // cfg.frames_per_batch) % 5 == 0:
            print(f"  [{cfg.algo} {cfg.run_id}] {frames:>6}/{cfg.total_frames} "
                  f"ret={log.rows[-1]['train_return']:7.2f} H={ent:.3f} "
                  f"Lc={log.rows[-1]['loss_critic']:8.3f}")

    collector.shutdown()

    metrics = evaluate(actor, episodes=cfg.eval_episodes)
    metrics["final_train_return"] = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
    metrics["final_entropy"] = float(np.mean([r["entropy"] for r in log.rows[-5:]]))
    metrics["episodes_seen"] = len(ep_returns)

    out = MODELS / "pg"; out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": actor.state_dict(), "cfg": asdict(cfg),
                "obs_dim": obs_dim, "n_act": n_act}, out / f"{cfg.run_id}.pt")
    log.save(metrics)
    if verbose:
        print(f"  [{cfg.algo} {cfg.run_id}] eval_return={metrics['eval_return']:.2f} "
              f"matured={metrics['eval_matured']:.1f} dead={metrics['eval_dead']:.1f}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS, default="ppo")
    ap.add_argument("--total-frames", type=int, default=60_000)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train_pg(PGConfig(algo=a.algo, run_id=a.run_id or f"{a.algo}_default",
                      seed=a.seed, total_frames=a.total_frames))
