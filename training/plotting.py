"""
Every figure used in the report, generated from the CSV/JSON files written by
the training runs. Nothing here re-trains anything, so figures can be rebuilt
cheaply with `uv run main.py --mode plots`.

Figures produced:
  fig1_comparative_rewards.png   four algorithms' learning curves on one axis
  fig2_dqn_objective.png         DQN TD loss + mean Q (the value objective)
  fig3_pg_entropy.png            policy entropy for REINFORCE / A2C / PPO
  fig4_convergence.png           frames needed to reach return thresholds
  fig5_generalization.png        held-out evaluation vs training return
  fig6_sweep_sensitivity.png     per-algorithm hyperparameter sensitivity
  fig7_agronomic_outcomes.png    what the policies actually do on the farm
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.common import FIGS, LOGS, moving_average

ALGOS = ["dqn", "reinforce", "a2c", "ppo"]
LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
COLORS = {"dqn": "#2f6f9f", "reinforce": "#c4762a", "a2c": "#4f8f4a", "ppo": "#9a4f9a"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})


# ------------------------------------------------------------------ helpers
def _baselines() -> tuple[float, float]:
    """Random and expert reference returns, written by training/baselines.py."""
    f = LOGS / "baselines.json"
    if f.exists():
        b = json.loads(f.read_text())
        return b["random"]["ret"], b["expert heuristic"]["ret"]
    return float("nan"), float("nan")


RAND_REF, EXPERT_REF = _baselines()


def _sweep() -> pd.DataFrame:
    f = LOGS / "sweep_results.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if "error" in df:
        df = df[df["error"].isna()]
    return df


def _best_run_ids() -> dict:
    f = LOGS / "best_runs.json"
    if f.exists():
        return json.loads(f.read_text())
    df = _sweep()
    if df.empty:
        return {}
    return {a: g.sort_values("eval_return", ascending=False).iloc[0]["run_id"]
            for a, g in df.groupby("algo")}


def _curve(algo: str, run_id: str) -> pd.DataFrame | None:
    f = LOGS / algo / f"{run_id}.csv"
    return pd.read_csv(f) if f.exists() else None


def _all_curves(algo: str) -> list[tuple[str, pd.DataFrame]]:
    d = LOGS / algo
    if not d.exists():
        return []
    return [(f.stem, pd.read_csv(f)) for f in sorted(d.glob("*.csv"))]


# ------------------------------------------------------------------ figure 1
def fig_comparative_rewards(best: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for a in ALGOS:
        rid = best.get(a)
        if not rid:
            continue
        df = _curve(a, rid)
        if df is None or "train_return" not in df:
            continue
        y = df["train_return"].to_numpy(dtype=float)
        x = df["frames"].to_numpy(dtype=float)
        m = ~np.isnan(y)
        if m.sum() < 3:
            continue
        sm = moving_average(y[m], 8)
        ax1.plot(x[m][len(x[m]) - len(sm):], sm, color=COLORS[a],
                 label=f"{LABELS[a]} ({rid})", lw=1.7)
    ax1.axhline(RAND_REF, ls=":", c="grey", lw=1.2,
                label=f"random policy ({RAND_REF:+.0f})")
    ax1.axhline(EXPERT_REF, ls="--", c="black", lw=1.2,
                label=f"expert heuristic ({EXPERT_REF:+.0f})")
    ax1.set_xlabel("environment frames")
    ax1.set_ylabel("episode return (moving avg, 8)")
    ax1.set_title("(a) Training return, best run per algorithm")
    ax1.legend(fontsize=6.5, loc="upper left")

    df = _sweep()
    if not df.empty:
        order = [a for a in ALGOS if a in set(df["algo"])]
        data = [df[df["algo"] == a]["eval_return"].dropna().to_numpy() for a in order]
        bp = ax2.boxplot(data, patch_artist=True, widths=0.6,
                         tick_labels=[LABELS[a] for a in order])
        for patch, a in zip(bp["boxes"], order):
            patch.set_facecolor(COLORS[a]); patch.set_alpha(0.55)
        for a, d in zip(order, data):
            ax2.scatter(np.full(len(d), order.index(a) + 1)
                        + np.random.uniform(-.08, .08, len(d)), d,
                        s=10, color="k", zorder=3, alpha=.6)
        ax2.axhline(RAND_REF, ls=":", c="grey", lw=1.2)
        ax2.set_ylabel("held-out evaluation return")
        ax2.set_title("(b) Spread across all 10 runs per algorithm")
    fig.savefig(FIGS / "fig1_comparative_rewards.png")
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig_dqn_objective(best: dict):
    rid = best.get("dqn")
    df = _curve("dqn", rid) if rid else None
    if df is None:
        return
    # label the objective with the loss this run actually used
    sw = _sweep()
    row = sw[sw["run_id"] == rid]
    lfn = str(row.iloc[0]["loss_fn"]) if ("loss_fn" in sw.columns and len(row)) else "Huber"
    lname = {"smooth_l1": "Huber", "l2": "L2"}.get(lfn, lfn)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].plot(df["frames"], df["loss"], color=COLORS["dqn"], lw=1.2)
    ax[0].set_yscale("log"); ax[0].set_xlabel("frames")
    ax[0].set_ylabel(f"TD loss ({lname}, log scale)")
    ax[0].set_title("(a) DQN objective: temporal-difference loss")

    if "q_value" in df:
        ax[1].plot(df["frames"], df["q_value"], color="#9f4f2f", lw=1.2)
        ax[1].axhline(0, color="#999", lw=0.8, ls=":")
        ax[1].set_xlabel("frames"); ax[1].set_ylabel("mean max-Q on replay batch")
        ax[1].set_title("(b) Value estimate drift")

    ax[2].plot(df["frames"], df["epsilon"], color="#555", lw=1.4)
    ax[2].set_xlabel("frames"); ax[2].set_ylabel("epsilon")
    ax[2].set_title("(c) Exploration schedule")
    fig.savefig(FIGS / "fig2_dqn_objective.png")
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig_pg_entropy(best: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for a in ("reinforce", "a2c", "ppo"):
        rid = best.get(a)
        df = _curve(a, rid) if rid else None
        if df is None or "entropy" not in df:
            continue
        ax1.plot(df["frames"], df["entropy"], color=COLORS[a], lw=1.5, label=LABELS[a])
        if "loss_critic" in df:
            ax2.plot(df["frames"], df["loss_critic"].abs().clip(lower=1e-6),
                     color=COLORS[a], lw=1.3, label=LABELS[a])
    ax1.axhline(np.log(9), ls="--", c="grey", lw=1.1, label="uniform policy ln(9)=2.20")
    ax1.set_xlabel("frames"); ax1.set_ylabel("policy entropy (nats)")
    ax1.set_title("(a) Policy entropy: exploration collapse rate")
    ax1.legend(fontsize=7)
    ax2.set_yscale("log"); ax2.set_xlabel("frames")
    ax2.set_ylabel("critic loss (log scale)")
    ax2.set_title("(b) Value-function fit")
    ax2.legend(fontsize=7)
    fig.savefig(FIGS / "fig3_pg_entropy.png")
    plt.close(fig)


# ------------------------------------------------------------------ figure 4
def fig_convergence(best: dict):
    thresholds = [0, 40, 90, 130]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.7))
    width = 0.2
    for i, a in enumerate(ALGOS):
        rid = best.get(a)
        df = _curve(a, rid) if rid else None
        need = []
        for th in thresholds:
            if df is None or "train_return" not in df:
                need.append(np.nan); continue
            y = df["train_return"].to_numpy(dtype=float)
            x = df["frames"].to_numpy(dtype=float)
            hit = np.where(~np.isnan(y) & (y >= th))[0]
            need.append(x[hit[0]] if len(hit) else np.nan)
        ax1.bar(np.arange(len(thresholds)) + i * width, need, width,
                color=COLORS[a], label=LABELS[a], alpha=.85)
    ax1.set_xticks(np.arange(len(thresholds)) + 1.5 * width)
    ax1.set_xticklabels([f"≥ {t}" for t in thresholds])
    ax1.set_xlabel("return threshold"); ax1.set_ylabel("frames to first reach")
    ax1.set_title("(a) Sample efficiency (missing bar = never reached)")
    ax1.legend(fontsize=7)

    for a in ALGOS:
        rid = best.get(a)
        df = _curve(a, rid) if rid else None
        if df is None or "train_return" not in df:
            continue
        y = pd.Series(df["train_return"]).rolling(10).std().to_numpy()
        ax2.plot(df["frames"], y, color=COLORS[a], lw=1.3, label=LABELS[a])
    ax2.set_xlabel("frames"); ax2.set_ylabel("rolling std of return (10)")
    ax2.set_title("(b) Training stability (lower = steadier)")
    ax2.legend(fontsize=7)
    fig.savefig(FIGS / "fig4_convergence.png")
    plt.close(fig)


# ------------------------------------------------------------------ figure 5
def fig_generalization():
    df = _sweep()
    if df.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for a in ALGOS:
        g = df[df["algo"] == a]
        if g.empty:
            continue
        ax1.scatter(g["final_train_return"], g["eval_return"], s=36,
                    color=COLORS[a], label=LABELS[a], alpha=.85, edgecolor="w")
    lim = [min(df["final_train_return"].min(), df["eval_return"].min()) - 10,
           max(df["final_train_return"].max(), df["eval_return"].max()) + 10]
    ax1.plot(lim, lim, ls="--", c="grey", lw=1)
    ax1.set_xlabel("final training return (seeds 0-…)")
    ax1.set_ylabel("held-out evaluation return (seeds 10000+)")
    ax1.set_title("(a) Generalisation: train vs unseen-season evaluation")
    ax1.legend(fontsize=7)

    order = [a for a in ALGOS if a in set(df["algo"])]
    best_rows = [df[df["algo"] == a].sort_values("eval_return").iloc[-1] for a in order]
    x = np.arange(len(order))
    ax2.bar(x - .18, [r["eval_return"] for r in best_rows], .36,
            yerr=[r.get("eval_return_std", 0) for r in best_rows],
            capsize=3, color=[COLORS[a] for a in order], label="best run")
    ax2.bar(x + .18, [df[df["algo"] == a]["eval_return"].mean() for a in order], .36,
            color=[COLORS[a] for a in order], alpha=.4, label="sweep mean")
    ax2.axhline(RAND_REF, ls=":", c="grey", lw=1.2)
    ax2.axhline(EXPERT_REF, ls="--", c="black", lw=1.1)
    ax2.set_xticks(x); ax2.set_xticklabels([LABELS[a] for a in order])
    ax2.set_ylabel("evaluation return"); ax2.legend(fontsize=7)
    ax2.set_title("(b) Best vs mean (dotted random, dashed expert)")
    fig.savefig(FIGS / "fig5_generalization.png")
    plt.close(fig)


# ------------------------------------------------------------------ figure 6
def fig_sweep_sensitivity():
    df = _sweep()
    if df.empty:
        return
    order = [a for a in ALGOS if a in set(df["algo"])]
    fig, axes = plt.subplots(1, len(order), figsize=(3.0 * len(order), 4.3))
    if len(order) == 1:
        axes = [axes]
    for ax, a in zip(axes, order):
        g = df[df["algo"] == a].sort_values("eval_return")
        labels = [str(r).split("_", 1)[-1][:18] for r in g["run_id"]]
        ax.barh(range(len(g)), g["eval_return"], color=COLORS[a], alpha=.85)
        ax.set_yticks(range(len(g))); ax.set_yticklabels(labels, fontsize=6.5)
        ax.axvline(RAND_REF, ls=":", c="grey", lw=1)
        ax.set_xlabel("eval return"); ax.set_title(LABELS[a], fontsize=9)
    fig.suptitle("Hyperparameter sensitivity: all 10 runs per algorithm", fontsize=10)
    fig.savefig(FIGS / "fig6_sweep_sensitivity.png")
    plt.close(fig)


# ------------------------------------------------------------------ figure 7
def fig_agronomic_outcomes():
    df = _sweep()
    if df.empty:
        return
    order = [a for a in ALGOS if a in set(df["algo"])]
    best = [df[df["algo"] == a].sort_values("eval_return").iloc[-1] for a in order]
    metrics = [("eval_matured", "plots matured /16", 16),
               ("eval_premature", "premature harvests", None),
               ("eval_dead", "plots lost", None),
               ("eval_yield", "banked yield", None)]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    for ax, (key, title, cap) in zip(axes, metrics):
        vals = [r.get(key, np.nan) for r in best]
        ax.bar([LABELS[a] for a in order], vals,
               color=[COLORS[a] for a in order], alpha=.85)
        if cap:
            ax.axhline(cap, ls="--", c="grey", lw=1)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelrotation=20, labelsize=7.5)
    fig.suptitle("What the best policy of each algorithm actually does on the farm",
                 fontsize=10)
    fig.savefig(FIGS / "fig7_agronomic_outcomes.png")
    plt.close(fig)


# ------------------------------------------------------------------- tables
def write_tables():
    df = _sweep()
    if df.empty:
        return
    out = []
    cols = {
        "dqn": ["run_id", "lr", "gamma", "batch_size", "buffer_size", "eps_end",
                "eps_anneal_frac", "n_step", "loss_fn", "target_tau", "double_dqn", "utd", "hidden",
                "eval_return", "eval_return_std", "eval_matured", "eval_dead"],
        "reinforce": ["run_id", "lr", "gamma", "frames_per_batch", "entropy_coeff",
                      "normalize_advantage", "critic_coeff", "hidden",
                      "eval_return", "eval_return_std", "eval_matured", "eval_dead"],
        "a2c": ["run_id", "lr", "gamma", "frames_per_batch", "entropy_coeff",
                "lmbda", "critic_coeff", "max_grad_norm",
                "eval_return", "eval_return_std", "eval_matured", "eval_dead"],
        "ppo": ["run_id", "lr", "frames_per_batch", "clip_epsilon", "ppo_epochs",
                "minibatch_size", "entropy_coeff", "lmbda",
                "eval_return", "eval_return_std", "eval_matured", "eval_dead"],
    }
    for a in ALGOS:
        g = df[df["algo"] == a]
        if g.empty:
            continue
        # The sweep CSV carries each run's true configuration under an `hp_`
        # prefix (written by RunLogger). Bare columns exist only for some rows,
        # so always prefer `hp_<name>` and fall back to the bare name.
        keep, rename = [], {}
        for c in cols[a]:
            src = f"hp_{c}" if f"hp_{c}" in g.columns and g[f"hp_{c}"].notna().any() else c
            if src in g.columns:
                keep.append(src); rename[src] = c
        t = (g[keep].rename(columns=rename)
                    .sort_values("eval_return", ascending=False))
        # Small-magnitude controls (lr, target_tau) must not be rounded to 0.
        for c in ("lr", "target_tau"):
            if c in t.columns:
                t[c] = t[c].map(lambda v: f"{v:.1e}" if pd.notna(v) else "")
        for c in t.columns:
            if c not in ("lr", "target_tau") and t[c].dtype.kind == "f":
                t[c] = t[c].round(2)
        t.to_csv(LOGS / f"table_{a}.csv", index=False)
        out.append(f"\n### {LABELS[a]}\n\n" + t.to_markdown(index=False))
    (LOGS / "hyperparameter_tables.md").write_text(
        "# Hyperparameter experiment tables\n" + "\n".join(out))
    print(f"[plots] wrote per-algorithm tables -> {LOGS/'hyperparameter_tables.md'}")


def make_all():
    best = _best_run_ids()
    if not best:
        print("[plots] no sweep results yet; falling back to any available runs")
        best = {a: (_all_curves(a)[0][0] if _all_curves(a) else None) for a in ALGOS}
        best = {k: v for k, v in best.items() if v}
    fig_comparative_rewards(best)
    fig_dqn_objective(best)
    fig_pg_entropy(best)
    fig_convergence(best)
    fig_generalization()
    fig_sweep_sensitivity()
    fig_agronomic_outcomes()
    write_tables()
    made = sorted(p.name for p in FIGS.glob("*.png"))
    print(f"[plots] {len(made)} figures in {FIGS}:")
    for m in made:
        print("   ", m)


if __name__ == "__main__":
    make_all()
