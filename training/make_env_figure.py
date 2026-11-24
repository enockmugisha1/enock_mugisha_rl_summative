"""
Produce the annotated environment figure used in the report.

Runs the expert controller to a representative mid-season field, stages the
states that need to be visible (a pest outbreak, a lost plot, a dry plot, some
harvested stubble), asks the Three.js viewer to render and upload a PNG together
with the screen position of every plot, then labels that PNG with matplotlib.

    uv run python -m training.make_env_figure            # needs a browser once
    uv run python -m training.make_env_figure --annotate-only

The browser step is what renders WebGL, so the first form opens the viewer and
waits for it; `--annotate-only` re-labels an already captured PNG.
"""

from __future__ import annotations

import argparse
import json
import time
import webbrowser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from environment import ACTION_NAMES, AgriGuardEnv
from environment.rendering import ThreeJSRenderer
from training.baselines import ExpertHeuristic
from training.common import FIGS

ROOT = Path(__file__).resolve().parent.parent
PNG = ROOT / "assets" / "environment_view.png"
POINTS = ROOT / "assets" / "environment_view.points.json"


def stage_field(steps: int = 46):
    """Advance to mid-season, then stage every visual state the legend claims."""
    env = AgriGuardEnv(seed=7)
    env.reset(seed=7)
    expert = ExpertHeuristic()
    for _ in range(steps):
        a = expert(env)
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    env.pest[10] = 0.85          # a severe outbreak
    env.pest[11] = 0.45          # spreading to the neighbour
    env.health[3] = 0.0          # a plot lost to stress
    env.alive[3] = False
    env.harvested[12] = True     # stubble
    env.moisture[1] = 0.12       # visibly dry (pale) soil
    return env


def capture(timeout_s: int = 180, open_browser: bool = True) -> bool:
    env = stage_field()
    r = ThreeJSRenderer(open_browser=False, capture_path=PNG, title="AgriGuard capture")
    r.update(env, {"action": "TREAT_PEST"})
    url = f"{r.url}?capture=1&w=1200&h=720"
    print(f"[figure] open this to render the scene: {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if r.captured:
            print(f"[figure] captured -> {r.captured}")
            r.close()
            return True
        time.sleep(0.5)
    r.close()
    print("[figure] timed out waiting for the browser to upload the render")
    return False


def annotate():
    """Label the captured render using the renderer's own projected positions."""
    if not PNG.exists():
        raise SystemExit(f"{PNG} missing -- run without --annotate-only first")
    img = mpimg.imread(PNG)
    h, w = img.shape[:2]
    pts = json.loads(POINTS.read_text()) if POINTS.exists() else None

    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    ax.imshow(img)
    ax.axis("off")

    if pts:
        sx, sy = w / pts.get("w", w), h / pts.get("h", h)
        plots = [(x * sx, y * sy) for x, y in pts["plots"]]
        rover = (pts["rover"][0] * sx, pts["rover"][1] * sy)
        state = pts.get("state", {}).get("plots", [])

        def call(target, text, dx, dy, color="#ffd24a"):
            ax.annotate(
                text, xy=target, xytext=(target[0] + dx, target[1] + dy),
                color="white", fontsize=9.5, fontweight="medium",
                ha="left" if dx > 0 else "right", va="center",
                bbox=dict(boxstyle="round,pad=0.42", fc="#101c14",
                          ec=color, lw=1.2, alpha=0.93),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                shrinkA=0, shrinkB=4,
                                connectionstyle="arc3,rad=0.12"))

        call(rover, "Rover: acts on the plot\nit is standing on", -150, -95, "#6fd3ff")
        call(plots[10], "Pest outbreak (0.85)\nspreads to 4-neighbours", 175, 70, "#ff6b5e")
        call(plots[3], "Plot lost to\ncumulative stress", 150, -95, "#b9b9b9")
        call(plots[12], "Harvested: yield banked,\nplot closed for the season", -165, 70, "#ffd24a")
        call(plots[1], "Dry soil (0.12):\npale bed, below wilting point", 120, -115, "#ffa94a")
        if state:
            tall = max(range(16), key=lambda i: state[i]["growth"]
                       if state[i]["alive"] and not state[i]["harvested"] else -1)
            call(plots[tall], "Crop height and colour\ntrack growth stage", 175, 25, "#7fe08a")

    ax.set_title("AgriGuard environment: 4×4 smallholder field, mid-season (day 11 of 60)",
                 fontsize=12, color="#1b1b1b", pad=12)
    fig.text(0.5, 0.022,
             "Rendered live in Three.js (WebGL). The rover occupies one plot at a time and "
             "chooses among 9 actions per work slot;\nsoil brightness encodes moisture, stalk "
             "height and colour encode growth, red haze encodes infestation severity.",
             ha="center", fontsize=8.6, color="#3a3a3a")
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "fig0_environment.png"
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figure] annotated environment figure -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotate-only", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.annotate_only:
        capture(open_browser=not a.no_browser)
    annotate()
