"""Fill the official assignment template with the AgriGuard results.

Run:  uv run --with python-docx python report/build_template_report.py
Keeps the template's own styles, headings and table formatting; only the
bracketed placeholder text is replaced and the tables are populated.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
TPL = Path("/home/enock/Downloads/"
           "Machine_Learning_Techniques_II - Summative_Assignment - Report Template.docx")
OUT = ROOT / "report" / "Enock_Mugisha_RL_Summative_Report.docx"
FIGS = ROOT / "figures"
LOGS = ROOT / "logs"

GITHUB = "https://github.com/enockmugisha1/enock_mugisha_rl_summative"
VIDEO = "https://youtu.be/DHY4afuGcKY"


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}w"


def sanitize_widths(doc):
    """The template stores fractional table widths (w="1504.83..."), which Word
    tolerates but python-docx rejects when it parses them as Twips ints."""
    fixed = 0
    for el in doc.element.body.iter():
        v = el.get(W)
        if v is not None:
            try:
                int(v)
            except ValueError:
                el.set(W, str(int(round(float(v)))))
                fixed += 1
    return fixed


# ---------------------------------------------------------------- helpers
def paras(doc):
    return doc.paragraphs


def set_text(p, text, *, bold=False, italic=False, size=None):
    """Replace a paragraph's text, keeping its paragraph style."""
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    return p


DOC = None  # set in main(); new paragraphs need a parent exposing .part

STYLE_IDS = {"normal": "Normal", "Heading 1": "Heading1",
             "Heading 2": "Heading2", "Heading 3": "Heading3"}


def _new_p(style="normal"):
    """A genuinely new, empty paragraph with its style set via pStyle.

    Never a deep copy of an existing paragraph: copying a heading inherits its
    Heading style and silently turns body text into a heading.
    """
    from docx.text.paragraph import Paragraph
    el = OxmlElement("w:p")
    sid = STYLE_IDS.get(style)
    if sid and sid != "Normal":
        pPr = OxmlElement("w:pPr")
        st = OxmlElement("w:pStyle")
        st.set(qn("w:val"), sid)
        pPr.append(st)
        el.append(pPr)
    return Paragraph(el, DOC)


def insert_after(anchor_el, text="", style="normal"):
    """Insert a fresh paragraph immediately after an arbitrary XML element
    (a paragraph *or* a table) and return it."""
    para = _new_p(style)
    anchor_el.addnext(para._element)
    if text:
        set_text(para, text)
    return para


def para_after(p, text="", style="normal"):
    return insert_after(p._element, text, style)


def figure_after(p, png: Path, caption: str, width=5.2):
    """Insert a picture + italic caption after paragraph p. Returns last para."""
    holder = para_after(p, style="normal")
    holder.alignment = WD_ALIGN_PARAGRAPH.CENTER
    holder.add_run().add_picture(str(png), width=Inches(width))
    cap = para_after(holder, style="normal")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.font.size = Pt(9)
    return cap


def fill_table(tbl, header, rows, *, font=7.0):
    """Write header + rows into an existing template table, resizing as needed."""
    while len(tbl.columns) < len(header):
        tbl.add_column(Inches(0.6))
    for i, h in enumerate(header):
        c = tbl.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(str(h))
        run.bold = True
        run.font.size = Pt(font)
    while len(tbl.rows) > 1:
        tbl._tbl.remove(tbl.rows[-1]._tr)
    for row in rows:
        cells = tbl.add_row().cells
        for i, v in enumerate(row[:len(header)]):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run("" if v is None else str(v))
            run.font.size = Pt(font)


def tidy(v):
    """Compact numeric formatting for table cells."""
    if isinstance(v, float):
        if v != v:
            return "-"
        if v == int(v) and abs(v) >= 1:
            return str(int(v))
        return f"{v:g}"
    return str(v)


def table_rows(csv_name, cols, labels):
    df = pd.read_csv(LOGS / csv_name)
    rows = []
    for _, r in df.iterrows():
        rows.append([tidy(r[c]) for c in cols])
    return labels, rows



def add_hyperlink(para, url, text):
    """A real clickable Word hyperlink (plain text alone is not clickable)."""
    part = para.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single")
    rPr.append(u)
    run.append(rPr)
    t = OxmlElement("w:t"); t.text = text
    run.append(t)
    link.append(run)
    para._p.append(link)
    return para


def set_body(p, text, space_after=5):
    """Write a multi-paragraph block.

    The content strings use \n to separate paragraphs; writing them into a
    single Word paragraph produces a wall of text, so each chunk becomes its own
    paragraph with space between them.
    """
    chunks = [c.strip() for c in text.split("\n") if c.strip()]
    if not chunks:
        return p
    set_text(p, chunks[0])
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.06
    last = p
    for c in chunks[1:]:
        last = para_after(last, c)
        last.paragraph_format.space_after = Pt(space_after)
        last.paragraph_format.line_spacing = 1.06
    return last


# ---------------------------------------------------------------- content
OVERVIEW = (
    "AgriGuard models precision irrigation and crop-health management on a Rwandan "
    "smallholder maize farm as a sequential decision problem. A single autonomous rover "
    "works a 4x4 field of 16 plots across a 60-day season, choosing one of nine actions "
    "per three-hour work slot (240 decisions per episode) while water, fertiliser and "
    "biopesticide are strictly limited and pests spread between neighbouring plots. "
    "Yield is paid only at harvest, roughly 74 slots after the irrigation that earns it, "
    "so the task demands long-horizon credit assignment rather than myopic prediction. "
    "Four agents - DQN, REINFORCE, A2C and PPO - were implemented in TorchRL and trained "
    "on this identical environment under a matched 150,000-frame budget, with ten "
    "hyperparameter configurations each (40 runs) and evaluation on held-out weather seeds."
)

ENVIRONMENT = (
    "The environment is a finite-horizon stochastic MDP implemented as a Gymnasium "
    "environment (environment/custom_env.py). The field is a 4x4 grid of maize plots; "
    "the season is 60 days of 4 work slots, giving a 240-step horizon. Each plot carries "
    "persistent state - soil moisture, growth stage, nutrient level, pest infestation and "
    "cumulative health - that evolves every slot and responds to what the rover does.\n"
    "Per slot, evapotranspiration removes 0.011 + 0.020 x temperature of soil moisture; a "
    "plot grows by 0.0135 x (1 - 0.8 x pest) x health only while moisture lies inside the "
    "healthy band [0.35, 0.85] and nutrient exceeds 0.18. Drought (moisture < 0.15) or "
    "waterlogging (moisture > 0.93) permanently erodes health, and a plot whose health "
    "reaches zero is lost for the season. Infestation amplifies itself and transmits to "
    "each 4-neighbour every slot, so an untreated outbreak compounds spatially. Per day, "
    "rain falls with a seasonal probability, adding moisture and partially refilling the "
    "tank, new infestations seed with probability 0.03, and a fresh forecast is drawn.\n"
    "Start state: day 0 immediately after transplanting - growth 0.05 on all plots, "
    "moisture ~ U(0.42, 0.58), nutrient ~ U(0.50, 0.70), zero pests, full health, a full "
    "water tank, 12 fertiliser doses, 8 biopesticide doses, rover parked at plot (0,0).\n"
    "Terminal conditions: the episode ends when the season closes (day 60 = 240 slots) or "
    "when every plot is either harvested or dead. The end-of-season completion bonus is "
    "paid on termination.\n"
    "Why reinforcement learning is required: the reward is delayed by ~74 steps, the water "
    "budget couples decisions across the whole season, and pest contagion makes the cost of "
    "inaction compound over space and time. No labelled dataset of correct actions exists, "
    "and no fixed irrigation calendar can be optimal because the optimal action is "
    "conditional on a stochastic weather and infestation state. The problem is therefore "
    "sequential decision-making under uncertainty, not supervised prediction."
)

AGENT = (
    "The agent is a single autonomous farm rover - equivalently, an advisory system telling "
    "a farmer what to do in the next work slot. It occupies exactly one plot at a time and "
    "acts only on the plot beneath it, so moving is itself a decision that costs a work "
    "slot. Its capabilities are deliberately those a real machine would have: drive between "
    "adjacent plots, open a drip valve, dose fertiliser, spray biopesticide, harvest, or "
    "idle. It cannot act on two plots at once, cannot see the future weather beyond a "
    "forecast probability, and cannot exceed its seasonal input budgets. Its objective is to "
    "maximise the grain actually brought in at maturity by the end of the season."
)

ACTION_SPACE = (
    "The action space is discrete: Discrete(9). Every action maps to a command a real rover "
    "or a farmer could execute, and the list is exhaustive - exactly one is taken per slot.\n"
    "0 MOVE_NORTH, 1 MOVE_SOUTH, 2 MOVE_EAST, 3 MOVE_WEST - drive one plot in that direction "
    "(RTK waypoint command); cost -0.012 fuel/time. Driving into the field boundary is "
    "infeasible and costs -0.45.\n"
    "4 IRRIGATE - open the drip valve on the current plot: moisture +0.26, tank -0.055, cost "
    "-0.035. Irrigating from an empty tank, or watering bare soil, is infeasible (-0.45).\n"
    "5 FERTILIZE - dose urea/NPK from the hopper: nutrient +0.40, cost -0.09, one of only 12 "
    "doses per season. Over-application above a nutrient level of 0.90 causes nitrate runoff "
    "(-0.55).\n"
    "6 TREAT_PEST - spray biopesticide: pest x 0.20, cost -0.09, one of only 8 doses.\n"
    "7 HARVEST - cut and bank the current plot. Pays +14.0 x yield if mature; a premature cut "
    "pays the partial yield minus a 2.5 penalty and forfeits any share of the completion bonus.\n"
    "8 WAIT - idle the slot, conserving water and fuel ahead of forecast rain; cost -0.02. "
    "The idle cost is deliberate: without it, doing nothing would be free and a value-based "
    "agent converges on a do-nothing policy."
)

OBSERVATION = (
    "The agent observes a 135-dimensional vector, Box(0, 1, shape=(135,)) - not the full "
    "simulator state. It comprises 16 plots x 7 per-plot features (112 values), a 16-dimensional "
    "one-hot encoding of the rover's position, and 7 global values. Every entry is natively "
    "scaled to [0, 1], so the policy network consumes the raw vector with no further "
    "normalisation. The agent does not observe the hidden parameters that drive the simulator "
    "(the exact growth coefficient, the pest seeding draw, or tomorrow's realised rainfall); it "
    "sees only the measurable quantities below, each mapped to how it would be obtained on a "
    "real farm."
)

OBS_PATH = {
    "moisture[16]": "LoRaWAN -> gateway -> MQTT broker -> state estimator -> obs[0:16]",
    "growth[16]": "Raster tile -> per-plot zonal mean -> normalised -> obs[16:32]",
    "nutrient[16]": "Sensor log -> rolling weekly mean -> obs[32:48]",
    "pest[16]": "Phone upload -> CNN severity score -> obs[48:64]",
    "health[16]": "Computed in the state estimator -> obs[64:80]",
    "alive[16]": "Computed from health -> obs[80:96]",
    "harvested[16]": "Farm database flag -> obs[96:112]",
    "rover[16]": "GPS fix -> nearest-plot index -> one-hot -> obs[112:128]",
    "tank": "Telemetry -> obs[128]",
    "fert_stock": "Inventory app -> /12 -> obs[129]",
    "pest_stock": "Inventory app -> /8 -> obs[130]",
    "day_frac": "Clock -> /60 -> obs[131]",
    "rain_prob": "Forecast API -> obs[132]",
    "temp": "Weather station -> normalised -> obs[133]",
    "rained_today": "Rain gauge -> obs[134]",
}

OBS_ROWS = [
    ["moisture[16]", "Volumetric soil water content per plot", "Capacitive soil-moisture probe buried per plot (LoRaWAN -> MQTT)", "float32, continuous", "0.0 - 1.0 (0 = oven-dry, 1 = saturated); updated every 15 min"],
    ["growth[16]", "Crop development stage per plot", "Sentinel-2 NDVI, refined by drone multispectral imagery", "float32, continuous", "0.0 - 1.0 (1.0 = mature); satellite every 5 days, drone on demand"],
    ["nutrient[16]", "Soil nitrogen index per plot", "In-situ NPK sensor calibrated by lab soil test", "float32, continuous", "0.0 - 1.0; updated weekly"],
    ["pest[16]", "Infestation severity per plot", "CNN classifier on sticky-trap / leaf photographs from the farmer's phone", "float32, continuous", "0.0 - 1.0; updated daily"],
    ["health[16]", "Cumulative, irreversible stress damage", "Derived in the state estimator from drought/waterlog/pest exposure", "float32, continuous", "0.0 - 1.0; recomputed every slot"],
    ["alive[16]", "Whether the plot is still productive", "Derived: health <= 0.02", "float32 binary", "0 or 1; every slot"],
    ["harvested[16]", "Whether the plot has been cut", "Rover harvest log / farm database", "float32 binary", "0 or 1; on event"],
    ["rover[16]", "Rover position, one-hot over the 16 plots", "RTK-GPS telemetry on the rover", "float32 one-hot", "exactly one entry = 1; 1 Hz"],
    ["tank", "Reservoir level", "Ultrasonic level sensor in the water tank", "float32, continuous", "0.0 - 1.0; every minute"],
    ["fert_stock", "Fertiliser doses remaining / 12", "Hopper load cell + inventory app", "float32, continuous", "0.0 - 1.0; on use"],
    ["pest_stock", "Biopesticide doses remaining / 8", "Inventory app", "float32, continuous", "0.0 - 1.0; on use"],
    ["day_frac", "Season progress", "System clock against the recorded planting date", "float32, continuous", "0.0 - 1.0; continuous"],
    ["rain_prob", "Probability of rain tomorrow", "Rwanda Meteorology Agency / OpenWeather forecast API", "float32, continuous", "0.0 - 1.0; refreshed every 6 h"],
    ["temp", "Normalised air temperature", "On-farm weather station", "float32, continuous", "0.0 - 1.0; hourly"],
    ["rained_today", "Whether rain fell today", "Tipping-bucket rain gauge", "float32 binary", "0 or 1; on event"],
]

REWARD = (
    "The reward is the sum of a sparse harvest payoff, a small dense shaping term, and the "
    "running costs of acting. Formally, for a step taking action a in state s:\n"
    "R = R_harvest + R_shape - R_costs - R_penalties, with the terminal bonus added on "
    "termination.\n"
    "Rewarded: harvesting a mature plot pays +14.0 x growth x health x (1 - 0.75 x pest), so "
    "yield is scaled by how well the plot was actually looked after; the end-of-season bonus "
    "pays +18.0 x (plots matured / 16) and counts only plots brought in at maturity, which "
    "removes the degenerate optimum of rush-harvesting the field on day 1; and a dense shaping "
    "term pays +0.12 x (plots inside the healthy moisture band / 16) each slot, which makes the "
    "240-step horizon learnable while staying small enough that passivity never pays.\n"
    "Penalised: losing a plot costs -3.0 (the dominant risk: 16 lost plots = -48); a premature "
    "harvest costs -2.5; waterlogging costs -0.70 and nitrate runoff -0.55, so over-application "
    "is punished as an environmental externality; an infeasible action costs -0.45; and routine "
    "operation costs -0.035 per irrigation, -0.09 per fertiliser or pesticide dose, -0.012 per "
    "move and -0.02 per idle slot.\n"
    "The resulting trade-off is the heart of the task: every action costs money, but inaction "
    "kills the crop, and harvesting early forfeits both the yield multiplier and the bonus. "
    "For reference, a uniformly random policy scores -8.7 and a hand-written expert heuristic "
    "scores +171.0; these bracket all learned results reported below."
)

SYSTEM_DESIGN = (
    "All four agents share one environment factory (training/common.py: make_env), one frame "
    "budget (150,000 environment frames) and one evaluation protocol, so the comparison is "
    "objective. Training collects from 8 environments stepped in parallel. Evaluation runs on "
    "held-out seeds starting at 10,000, which never overlap the training seeds, and reports the "
    "mean return plus agronomic outcomes (plots matured, cut prematurely, and lost). Both "
    "network families are multilayer perceptrons over the 135-dimensional observation with two "
    "hidden layers of 256 ReLU units unless a sweep row states otherwise. Trained policies are "
    "serialised as TorchRL state dicts (~0.4 MB) under models/, and play.py loads a checkpoint "
    "and drives the live Three.js viewer through the same make_env interface used in training, "
    "so the deployment contract is a single observation -> action call."
)

DQN_DESIGN = (
    "DQN (training/dqn_training.py) uses TorchRL's DQNLoss over a QValueActor wrapping a "
    "135 -> 256 -> 256 -> 9 MLP. The final layer is initialised small so early Q-values start "
    "near zero and the bootstrap targets stay stable.\n"
    "Experience is stored in a "
    "LazyTensorStorage replay buffer (100,000 transitions by default) and sampled in batches of "
    "128. A delayed target network is tracked by SoftUpdate at rate tau. Exploration uses an "
    "EGreedyModule annealed from eps=1.0 to a floor over a configurable fraction of training. "
    "\nThree modifications proved necessary on this task. First, n-step returns (n=3): with "
    "one-step TD the harvest payoff sits ~74 steps beyond the irrigation that earns it and the "
    "signal propagates far too slowly.\n"
    "Second, a Huber (smooth L1) objective rather than L2: "
    "harvest (+14) and plot death (-3) are two orders of magnitude larger than routine step "
    "costs (0.02-0.09), and under L2 a handful of transitions dominated each batch gradient, "
    "driving pre-clip gradient norms above 120 and diverging the value head.\n"
    "Third, a high "
    "update-to-data ratio: at 8 gradient updates per 1,000 collected frames the network "
    "received only ~1,200 updates in total and never fit, so the ratio is swept from 8 to 128. "
    "Double-DQN is enabled by default and disabled in one control row."
)

PG_DESIGN = (
    "The three policy-gradient agents share one implementation (training/pg_training.py) built "
    "on a ProbabilisticActor with a Categorical distribution over the 9 actions and a separate "
    "ValueOperator critic, both 135 -> 256 -> 256 MLPs. Advantages come from a generalised "
    "advantage estimator (GAE).\n"
    "REINFORCE uses TorchRL's ReinforceLoss with lambda = 1.0, making the advantage the full "
    "discounted Monte-Carlo return minus a learned baseline. ReinforceLoss exposes neither an "
    "entropy bonus nor a critic weight, so both are applied explicitly in the training loop; "
    "without that the entropy and critic columns of its sweep would have been inert and three "
    "rows would have been silent duplicates.\n"
    "A2C uses A2CLoss with GAE(lambda), a value baseline and an entropy bonus, taking one "
    "gradient step per collected on-policy batch.\n"
    "PPO uses ClipPPOLoss: the clipped surrogate objective bounds each policy update inside a "
    "trust region and the same batch is reused for several epochs over minibatches, which is "
    "what converts the noisy advantages of this delayed-reward task into stable progress. "
    "Entropy regularisation and advantage normalisation are configurable and both are swept."
)


INTEGRATION = (
    "Integration path. A trained policy is a single TorchRL state dict of about 0.4 MB under "
    "models/pg/ or models/dqn/, saved together with its configuration, observation dimension "
    "and action count, so a checkpoint is self-describing and can be reloaded without the "
    "training code. training/baselines.py: load_trained_policy(name) resolves a checkpoint and "
    "returns a plain callable with the signature act(obs: np.ndarray[135]) -> int, which is the "
    "entire inference contract - no TorchRL objects cross that boundary.\n"
    "play.py consumes exactly that callable and drives the live Three.js viewer, which is the "
    "same interface a deployment would use. For a field system the policy would be exported to "
    "TorchScript or ONNX and served two ways: on-device inside the rover controller, or behind "
    "a small HTTP endpoint for an advisory mobile app, where the 135-dimensional observation is "
    "assembled from the sensor sources in the table above and the returned integer maps straight "
    "back to one of the nine rover commands. Because the observation is already scaled to [0, 1] "
    "and the action space is a 9-way discrete choice, the serving payload is a 135-float vector "
    "in and one integer out.\n"
    "What is implemented today is the decision layer and its simulator; the sensor integration "
    "described above is a deployment design, not a live data pipeline."
)

IMPLEMENTATION_INTRO = (
    "Every row below is a separate training run on the identical environment with the identical "
    "150,000-frame budget; the only differences are the hyperparameters shown. Mean reward is the "
    "mean return over held-out evaluation seeds (10,000+). Rows are ordered best to worst."
)

DQN_ANALYSIS = (
    "Three controls dominate the DQN table. Learning rate: dqn04_lr_high (2.5e-4) is the only "
    "configuration to clear the random floor, at +7.17; every 5e-5 run lands between -19 and -76, "
    "because at this budget the smaller step size never moves the value head far enough. "
    "\nTarget-update rate: tau = 0.05 (dqn07_target_fast, -75.97) is the worst run in the entire "
    "study - a fast-tracking target destroys the stationarity the TD regression depends on - while "
    "tau = 0.001 recovers to -36.58. Replay capacity: shrinking the buffer to 5,000 (-59.57) costs "
    "about 11 points against the 100,000 baseline (-48.82), since with a 240-step horizon a small "
    "buffer holds only ~20 episodes and samples stay strongly correlated.\n"
    "Exploration behaves as "
    "theory predicts but cannot rescue the method: a near-greedy floor (eps_end = 0.02) matures "
    "zero plots, and widening to 0.40 only reaches -19.31, because uniform random actions almost "
    "never produce the long coordinated sequence - irrigate repeatedly, then harvest at maturity - "
    "that the task rewards. The deliberately naive control dqn10 (one-step TD, no Double-DQN, L2 "
    "objective, starved update ratio) scores -62.20, confirming the combined value of the n-step, "
    "Huber and Double-DQN machinery."
)

RF_ANALYSIS = (
    "REINFORCE is dominated by the learning rate: rf04_lr_high (1e-3) reaches +15.41 while "
    "rf05_lr_low (1e-4) collapses to -21.38, a 37-point swing on one control. Removing advantage "
    "normalisation (rf08, -4.37) costs about 11 points against the baseline (+6.37), which is the "
    "expected result for a Monte-Carlo estimator whose advantage scale varies by an order of "
    "magnitude across a season. Rollout length matters in the opposite direction to A2C: the long "
    "8,000-frame rollout (-4.45) is worse than the 4,000-frame baseline, because at a fixed frame "
    "budget longer rollouts mean fewer gradient updates. Entropy has only a mild effect here "
    "(+6.37 at 0.01 against +2.58 at 0 and +0.94 at 0.05), consistent with a policy whose entropy "
    "barely falls during training."
)

A2C_ANALYSIS = (
    "Rollout length is the strongest A2C control: 1,000 frames per batch (+17.64) beats 2,000 "
    "(-3.75) and 8,000 (+2.39), because A2C takes one gradient step per batch and at a fixed frame "
    "budget the number of updates is the binding constraint. GAE lambda follows the expected "
    "bias-variance pattern: lambda = 0.70 (+13.46) beats lambda = 0.95 (-3.75) and lambda = 1.00 "
    "(+1.15), trading a little bias for a large variance reduction over a 240-step horizon. "
    "Removing the entropy bonus helps (+16.88 against -3.75), indicating that on this task the "
    "value baseline already supplies adequate exploration and the bonus mostly adds noise."
)

PPO_ANALYSIS = (
    "PPO is the most stable method: every one of its ten configurations is positive, and its worst "
    "run (+28.50) still beats the best run of all three other algorithms. Both of its headline "
    "sweeps are non-monotonic. Clipping: eps = 0.05 scores +28.50, eps = 0.2 scores +78.83 and "
    "eps = 0.4 scores +73.57 - tight clipping costs 50 points while a loose trust region costs only "
    "5, so on this task PPO is far more sensitive to under-exploration than to a permissive update "
    "bound. Epochs: 1 epoch (+32.49) wastes the collected batch, 6 epochs (+78.83) is optimal, and "
    "16 epochs (+63.09) over-fits the policy to stale data and drifts outside the trust region the "
    "clipping is meant to protect. Notably ppo08_entropy_zero (+67.64) beats ppo07_entropy_high "
    "(+60.08): the clipped surrogate already supplies enough exploration pressure, so an added "
    "entropy bonus mostly costs return."
)

CUM_REWARDS = (
    "Figure 2(a) plots the training return of each algorithm's best configuration against the "
    "random floor (-8.7) and the expert ceiling (+171.0); Figure 2(b) shows the held-out evaluation "
    "return of all ten runs per algorithm. The separation is categorical rather than marginal. PPO "
    "reaches +78.83 - about 46 per cent of the expert heuristic - while DQN's best reaches +7.17, "
    "roughly 4 per cent.\n"
    "Median performance tells the same story more sharply: PPO +61.58, A2C "
    "+2.73, REINFORCE +1.04, DQN -39.50. Every algorithm's best run clears the random floor, so all "
    "four learn something, but only PPO learns a policy a farmer would deploy.\n"
    "The spread within "
    "each algorithm is itself a result: DQN spans 83.1 points across its ten configurations while "
    "PPO spans 50.3 points entirely above zero, so a practitioner deploying DQN here would have to "
    "tune carefully to avoid a catastrophic configuration whereas PPO's worst case remains usable."
)

STABILITY = (
    "Figure 3 shows the DQN value objective. The TD loss rises through training while the mean "
    "max-Q drifts steadily negative to about -3: the value head is not converging, because it "
    "chases a target that keeps moving under a behaviour policy that is still exploring at eps = "
    "0.25. That DQN scores +7.17 at all is because the relative ordering of actions survives a poor "
    "absolute fit, not because the value function is accurate.\n"
    "Figure 4(a) shows policy entropy for the three policy-gradient methods against the "
    "uniform-policy maximum of ln(9) = 2.20. PPO commits hardest, falling from 2.20 to about 1.60; "
    "A2C plateaus near 1.85; REINFORCE stalls around 1.92, barely distinguishable from a uniform "
    "policy after 150,000 frames. Figure 4(b) explains why: REINFORCE's critic loss sits an order "
    "of magnitude above A2C's and PPO's (about 130 against about 10). Its Monte-Carlo baseline "
    "never fits, the advantages stay noisy, and the policy cannot commit to anything."
)

CONVERGENCE = (
    "Figure 5 reports the frames each method needed to first reach a set of return thresholds. "
    "Within the shared 150,000-frame budget - about 625 episodes of 240 steps - PPO is the only "
    "method to pass +50, crossing it at roughly 90,000 frames (~375 episodes) and still improving "
    "when the budget ended. A2C and REINFORCE reach their plateaus near +15 at around 60,000-80,000 "
    "frames and then flatten, so their limitation is the quality of the gradient estimate rather "
    "than the budget. DQN never reaches a stable level at all: its best run oscillates between +25 "
    "and +38 in training while its greedy evaluation sits at +7.17, which is the signature of a "
    "value function that has not converged rather than a policy that has. None of the four had "
    "fully converged at 150,000 frames, so these are rankings under a fixed budget, not asymptotic "
    "results."
)

GENERALIZATION = (
    "Every number reported in this study is measured on held-out initial states: evaluation seeds "
    "start at 10,000 and never overlap the seeds used for training, so each evaluation episode "
    "presents unseen weather, unseen pest seeding and unseen initial soil moisture. Figure 6 plots "
    "final training return against held-out evaluation return, and the gap between them is the "
    "clearest single statistic in the study:\n"
    "PPO +62.32 train / +58.67 eval (gap 3.66); A2C +11.46 / +6.82 (gap 4.64); REINFORCE +8.22 / "
    "+0.10 (gap 8.13); DQN -11.89 / -39.37 (gap 27.48).\n"
    "PPO generalises almost perfectly - a 3.66-point gap on a 62-point return. DQN's 27.48-point "
    "gap is a different phenomenon from over-fitting: its training returns are collected under an "
    "eps-greedy behaviour policy still exploring at eps = 0.25, and removing that exploration at "
    "evaluation exposes a value function that has not converged. The agronomic outcomes confirm "
    "the ranking is real and not an artefact of reward shaping: on held-out seasons PPO's best "
    "policy brings in 12.6 of 16 plots at maturity with 1.2 premature cuts and 2.1 plots lost, "
    "against 7.8 matured for A2C and REINFORCE (but 5.5 and 4.4 premature cuts respectively - they "
    "learned irrigation but not patience) and 4.0 matured for DQN with 9.9 plots lost."
)

CONCLUSION = (
    "Four reinforcement-learning algorithms were implemented in TorchRL and compared on a custom "
    "240-step farm-management environment under an identical 150,000-frame budget with ten "
    "hyperparameter configurations each. PPO performed best by a wide margin (+78.83; 12.6 of 16 "
    "plots matured; 46 per cent of the expert ceiling), followed by A2C (+17.64) and REINFORCE "
    "(+15.41), with DQN last (+7.17).\n"
    "The ordering follows from the structure of the environment. Its reward is paid once per plot, "
    "roughly 74 steps after the decisions that earn it, and the dense shaping term is deliberately "
    "small. A value-based method must propagate that signal backwards through a long bootstrap "
    "chain while its target network keeps moving, and the sweep shows exactly this failing: rising "
    "TD loss, drifting Q-values, an 83-point spread across configurations and a 27-point "
    "generalisation gap. Policy-gradient methods optimise whole-trajectory returns directly and "
    "need only the sign of the advantage to be right, not the absolute value estimate. PPO adds "
    "the decisive ingredient - a clipped surrogate that bounds each update so a noisy advantage "
    "cannot destroy a working policy - which makes it simultaneously the best and the most stable "
    "method here.\n"
    "Strengths and weaknesses. DQN is the most sample-hungry and the most hyperparameter-sensitive, "
    "but it is off-policy and would benefit most from a larger budget; its best run was still "
    "improving when training stopped. REINFORCE is the simplest but its Monte-Carlo baseline fits "
    "poorly over a 240-step horizon, leaving high-variance gradients. A2C's learned baseline fixes "
    "much of that variance at low cost and is competitive at a fraction of PPO's compute. PPO is "
    "the most expensive per frame yet the most reliable, and is the only method whose worst "
    "configuration remains deployable.\n"
    "Limitations. The 150,000-frame budget is modest and penalises DQN most. Each configuration was "
    "run on a single seed, so differences of a few points - +78.83 against +78.78, for instance - "
    "are not separable from noise, and only the large effects discussed above are safely "
    "interpretable. The simulator's growth, pest and rainfall dynamics are parameterised from "
    "agronomic rules of thumb rather than calibrated against field data from a specific Rwandan "
    "site, and the observation pipeline described in the table above is a deployment design rather "
    "than an implemented sensor integration. Finally, the expert heuristic (+171.0) still comfortably "
    "exceeds PPO (+78.83), so none of these agents is yet competitive with a well-designed "
    "rule-based schedule - closing that gap is the natural next step."
)


def add_page_numbers(doc):
    """Centred 'Page N of M' in the footer of every section."""
    for section in doc.sections:
        footer = section.footer
        para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        for r in list(para.runs):
            r._element.getparent().remove(r._element)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER

        def field(instr):
            r = OxmlElement("w:r")
            f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
            it.text = instr
            f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
            r.append(f1); r.append(it); r.append(f2)
            return r

        para.add_run("Page ")
        para._p.append(field(" PAGE "))
        para.add_run(" of ")
        para._p.append(field(" NUMPAGES "))
        for r in para.runs:
            r.font.size = Pt(9)


# ---------------------------------------------------------------- assembly
def main():
    global DOC
    doc = Document(str(TPL))
    DOC = doc
    print('sanitised widths:', sanitize_widths(doc))
    P = doc.paragraphs

    set_text(P[1], "Student Name: Enock Mugisha")
    set_text(P[2], "Video Recording: ")
    add_hyperlink(P[2], VIDEO, VIDEO)
    gh = para_after(P[2], "GitHub Repository: ")
    add_hyperlink(gh, GITHUB, GITHUB)

    set_body(P[4], OVERVIEW)

    # Environment Description (P5 heading) -> body + the environment visual
    env_body = set_body(para_after(P[5]), ENVIRONMENT)
    figure_after(env_body, FIGS / "fig0_environment.png",
                 "Figure 1 - The AgriGuard environment rendered live in Three.js (WebGL). Soil "
                 "brightness encodes moisture, stalk height and colour encode growth stage, red "
                 "haze encodes infestation severity; the rover acts on the plot beneath it.")

    set_body(P[7], AGENT)
    set_text(P[8], "Action Space", bold=True)
    set_body(P[9], ACTION_SPACE)
    set_body(P[11], OBSERVATION)
    set_body(P[15], REWARD)

    sysd = set_body(para_after(P[16]), SYSTEM_DESIGN)
    set_body(para_after(sysd), INTEGRATION)
    set_body(P[18], DQN_DESIGN)
    set_text(P[19], "Policy Gradient Methods (REINFORCE, A2C and PPO)")
    set_body(P[20], PG_DESIGN)

    set_body(para_after(P[21]), IMPLEMENTATION_INTRO)
    set_text(P[22], "DQN")
    P[22].style = doc.styles["Heading 3"]
    set_text(P[24], "REINFORCE")
    set_text(P[28], "PPO")
    P[28].style = doc.styles["Heading 3"]

    # ---- tables -------------------------------------------------------
    # Template order: 'DQN' label, T1, H3 REINFORCE, T2, blanks, 'PPO', T3, H3 ''
    # Each analysis is inserted directly after the table it describes.
    t = doc.tables

    # Table 0 in the template is the observation-provenance table.
    fill_table(t[0],
               ["Observation", "Description", "Source (Sensor/Camera/API/Data)",
                "Encoding / Data Type", "Range & Update Cadence",
                "Path into the agent"],
               [r + [OBS_PATH.get(r[0], "")] for r in OBS_ROWS], font=6.5)

    hdr, rows = table_rows(
        "table_dqn.csv",
        ["lr", "gamma", "buffer_size", "batch_size", "eps_end", "n_step",
         "target_tau", "utd", "loss_fn", "eval_return"],
        ["Learning Rate", "Gamma", "Replay Buffer", "Batch Size", "Epsilon Floor",
         "n-step", "Target Tau", "Updates/Batch", "Objective", "Mean Reward"])
    fill_table(t[1], hdr, rows)
    set_body(insert_after(t[1]._tbl), DQN_ANALYSIS)

    hdr, rows = table_rows(
        "table_reinforce.csv",
        ["lr", "gamma", "frames_per_batch", "entropy_coeff", "normalize_advantage",
         "critic_coeff", "hidden", "eval_return"],
        ["Learning Rate", "Gamma", "Rollout (frames)", "Entropy Coeff",
         "Advantage Norm", "Critic Coeff", "Hidden", "Mean Reward"])
    fill_table(t[2], hdr, rows)
    set_body(insert_after(t[2]._tbl), RF_ANALYSIS)

    hdr, rows = table_rows(
        "table_ppo.csv",
        ["lr", "frames_per_batch", "clip_epsilon", "ppo_epochs", "minibatch_size",
         "entropy_coeff", "lmbda", "eval_return"],
        ["Learning Rate", "Rollout (frames)", "Clip Epsilon", "Epochs",
         "Minibatch", "Entropy Coeff", "GAE Lambda", "Mean Reward"])
    fill_table(t[3], hdr, rows)
    ppo_an = set_body(insert_after(t[3]._tbl), PPO_ANALYSIS)

    # The template ships only three hyperparameter tables; A2C is required too.
    a2c_head = insert_after(ppo_an._element, "A2C", style="Heading 3")
    a2c_tbl = doc.add_table(rows=1, cols=8)
    a2c_tbl.style = t[3].style
    a2c_head._element.addnext(a2c_tbl._tbl)
    hdr, rows = table_rows(
        "table_a2c.csv",
        ["lr", "gamma", "frames_per_batch", "entropy_coeff", "lmbda",
         "critic_coeff", "max_grad_norm", "eval_return"],
        ["Learning Rate", "Gamma", "Rollout (frames)", "Entropy Coeff",
         "GAE Lambda", "Critic Coeff", "Grad Clip", "Mean Reward"])
    fill_table(a2c_tbl, hdr, rows)
    set_body(insert_after(a2c_tbl._tbl), A2C_ANALYSIS)

    # ---- results discussion -------------------------------------------
    set_text(P[32], "Each figure below is referenced and interpreted in the text that follows it.")
    set_text(P[34], "Cumulative Rewards", bold=True)
    set_body(P[35], CUM_REWARDS)
    figure_after(P[35], FIGS / "fig1_comparative_rewards.png",
                 "Figure 2 - (a) Training return of each algorithm's best configuration against the "
                 "random floor and expert ceiling. (b) Held-out evaluation return of all ten runs "
                 "per algorithm.")

    set_text(P[36], "Training Stability")
    stab = set_body(para_after(P[36]), STABILITY)
    f3 = figure_after(stab, FIGS / "fig2_dqn_objective.png",
                      "Figure 3 - DQN value objective for the best run: TD loss, mean max-Q on "
                      "replay batches, and the epsilon schedule.")
    figure_after(f3, FIGS / "fig3_pg_entropy.png",
                 "Figure 4 - (a) Policy entropy for the three policy-gradient methods against the "
                 "uniform maximum ln(9) = 2.20. (b) Critic loss, log scale.")

    set_text(P[37], "Episodes To Converge", bold=True)
    set_body(P[38], CONVERGENCE)
    set_text(P[39], "")
    figure_after(P[39], FIGS / "fig4_convergence.png",
                 "Figure 5 - Frames required to first reach each return threshold, per algorithm.")

    set_text(P[41], "Generalization", bold=True)
    set_body(P[42], GENERALIZATION)
    figure_after(P[42], FIGS / "fig5_generalization.png",
                 "Figure 6 - Final training return against held-out evaluation return. Points on "
                 "the diagonal generalise; points below it are optimistic on the training "
                 "distribution.")

    set_body(P[44], CONCLUSION)

    add_page_numbers(doc)

    # drop the template's trailing empty Heading 3 and any stray empty paragraphs
    for para in list(doc.paragraphs):
        if not para.text.strip() and para.style.name.startswith("Heading"):
            para._element.getparent().remove(para._element)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print("wrote", OUT)


def Paragraph_of(tbl, doc):
    return None


if __name__ == "__main__":
    main()
