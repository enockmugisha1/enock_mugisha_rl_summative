# AgriGuard — Precision Irrigation & Crop-Health RL Agent

**Enock Mugisha — Machine Learning Techniques II, Summative (Mission-Based RL)**

A custom Gymnasium environment in which an autonomous farm rover manages a 4×4 field
of maize plots across a 60-day growing season, trained with four RL algorithms
implemented in **TorchRL**: DQN, REINFORCE, A2C and PPO.

---

## Quick start

```bash
git clone <repo-url> && cd enock_mugisha_rl_summative
uv sync
uv run main.py
```

`uv run main.py` runs a random-policy demo, prints the terminal view and opens the
3D Three.js visualisation. Other entry points:

| Command | What it does |
|---|---|
| `uv run main.py --mode demo` | Random policy rollout + 3D viewer |
| `uv run main.py --mode train --algo dqn` | Train one algorithm with default hyperparameters |
| `uv run main.py --mode sweep --algo all` | Run the full 10-run-per-algorithm hyperparameter sweep |
| `uv run main.py --mode plots` | Regenerate every figure used in the report |
| `uv run play.py` | Load the best saved agent and watch it act in the 3D viewer |
| `uv run pytest -q` | Run the environment test suite |

---

## 1. The mission

Smallholder farms in Rwanda lose yield to two coupled problems: irrigation water is
scarce and arrives unpredictably as rain, and pest outbreaks spread between adjacent
plots faster than a part-time farmer can respond. AgriGuard models the decision
problem faced by a low-cost autonomous rover (or an advisory app telling a farmer
what to do next) that must allocate **water, fertiliser, biopesticide and labour
time** across a field to maximise the harvest actually brought in at the end of the
season.

### Why this needs reinforcement learning

* **Delayed credit assignment.** Yield is only paid out when a plot is harvested —
  up to 240 slots after the irrigation decisions that determined it. There is no
  immediate signal telling the agent that watering plot 7 on day 3 was correct.
* **Non-myopic budgeting under exogenous uncertainty.** The tank refills only when
  it rains, and Rwandan rainfall is bimodal (wet onset, dry mid-season, wet tail).
  Irrigating greedily in week 2 empties the reservoir before the flowering window
  in week 6, when water matters most. The optimal policy is *state-conditional*,
  so no fixed irrigation calendar can be optimal.
* **A contagion process creates compounding future cost.** Infestation amplifies
  itself and spreads to 4-neighbours every slot. Treating one cheap plot now versus
  containing an outbreak requires lookahead over a spatial process.
* **It is not supervised learning.** No dataset of labelled "correct actions"
  exists; the objective is cumulative season return under stochastic weather, which
  is exactly a sequential decision problem.

---

## 2. Environment specification

### 2.1 Action space — `Discrete(9)`

Every action is a command a real rover (or advisory app) can execute.

| # | Action | Real-world mapping | Effect in the simulator |
|---|---|---|---|
| 0 | `MOVE_NORTH` | RTK waypoint drive | rover moves one plot north; `-0.012` fuel cost |
| 1 | `MOVE_SOUTH` | RTK waypoint drive | rover moves one plot south |
| 2 | `MOVE_EAST` | RTK waypoint drive | rover moves one plot east |
| 3 | `MOVE_WEST` | RTK waypoint drive | rover moves one plot west |
| 4 | `IRRIGATE` | open drip valve on current plot | `moisture += 0.26`, `tank -= 0.055` |
| 5 | `FERTILIZE` | dose urea/NPK from hopper | `nutrient += 0.40`, one of 12 doses |
| 6 | `TREAT_PEST` | spray biopesticide | `pest *= 0.20`, one of 8 doses |
| 7 | `HARVEST` | cut and bank the plot | banks yield, closes the plot |
| 8 | `WAIT` | idle the slot | conserves water and fuel ahead of forecast rain; costs `-0.02` labour time |

Driving into the field boundary, irrigating from an empty tank, dosing with no stock
left, or acting on an already-harvested plot are all *infeasible* and cost `-0.45`.

### 2.2 Observation space — `Box(0, 1, shape=(135,))`

Each of the 16 plots contributes 7 values (112 total), plus a 16-dim one-hot rover
position and 7 global values = **135 dimensions**. **Observation provenance** — how each number would be obtained on a real farm:

| Observation | Meaning | Encoding / range | Real-world source | Update cadence | Path to the agent |
|---|---|---|---|---|---|
| `moisture[16]` | volumetric soil water content | float `[0,1]`, 0 = oven-dry, 1 = saturated | capacitive soil-moisture probe buried per plot | every 15 min | LoRaWAN → gateway → MQTT → state vector |
| `growth[16]` | crop development stage | float `[0,1]`, 0.6 = mature | NDVI from Sentinel-2 tiles, refined by drone multispectral | 5 days (satellite), on demand (drone) | raster → per-plot zonal mean → normalised |
| `nutrient[16]` | soil nitrogen index | float `[0,1]` | in-situ NPK sensor, calibrated by lab soil test | weekly | sensor log → rolling mean |
| `pest[16]` | infestation severity | float `[0,1]` | CNN classifier on sticky-trap / leaf photos from the farmer's phone | daily | image → classifier → severity score |
| `health[16]` | cumulative stress damage | float `[0,1]`, irreversible | derived: integral of drought/waterlog/pest exposure | every slot | computed in the state estimator |
| `alive[16]` | plot still productive | binary | derived from `health <= 0.02` | every slot | computed |
| `harvested[16]` | plot already cut | binary | rover harvest log | on event | farm database |
| `rover[16]` | rover position, one-hot | binary | RTK-GPS on the rover | 1 Hz | telemetry |
| `tank` | reservoir level | float `[0,1]` | ultrasonic level sensor in the tank | 1 min | telemetry |
| `fert_stock` | fertiliser doses left / 12 | float `[0,1]` | hopper load cell + inventory app | on use | farm database |
| `pest_stock` | biopesticide doses left / 8 | float `[0,1]` | inventory app | on use | farm database |
| `day_frac` | season progress | float `[0,1]` | system clock vs planting date | continuous | computed |
| `rain_prob` | P(rain tomorrow) | float `[0,1]` | Rwanda Meteorology Agency / OpenWeather forecast API | every 6 h | API → forecast feature |
| `temp` | normalised air temperature | float `[0,1]` | on-farm weather station | hourly | telemetry |
| `rained_today` | rain occurred today | binary | tipping-bucket rain gauge | on event | telemetry |

All values are already in `[0,1]`, so the policy network consumes the raw vector
with no further scaling.

### 2.3 Start state

Day 0, immediately after transplanting: `growth = 0.05` on all 16 plots,
`moisture ~ U(0.42, 0.58)` (soil near field capacity), `nutrient ~ U(0.50, 0.70)`,
`pest = 0`, `health = 1`, full tank (`1.0`), 12 fertiliser doses, 8 biopesticide
doses, rover parked at plot (0,0).

### 2.4 Transition dynamics

Per slot: evapotranspiration removes `0.011 + 0.020 × temp` moisture; crops grow by
`0.0135 × (1 − 0.8 × pest) × health` **only** while moisture is inside the healthy
band `[0.35, 0.85]` and `nutrient > 0.18`; drought (`moisture < 0.15`) or
waterlogging (`moisture > 0.93`) permanently erode `health` by `0.030`; infestation
self-amplifies by `0.055` and transmits `0.022` to each 4-neighbour. Per day: rain
occurs with the seasonal probability, adding `U(0.10, 0.34)` moisture and refilling
the tank by 55% of that depth; clean plots become newly infested with probability
`0.030`; a new forecast is drawn.

### 2.5 Reward structure

| Component | Value | Purpose |
|---|---|---|
| Harvest yield | `+14.0 × growth × health × (1 − 0.75 × pest)` | the real objective, paid only at harvest |
| Season completion bonus | `+18.0 × (plots harvested **at maturity** / 16)` | rewards bringing the crop in; premature cuts earn no share |
| Healthy-field shaping | `+0.12 × (plots in moisture band / 16)` per slot | dense signal that makes the 240-step horizon learnable, kept small enough that passivity does not pay |
| Water cost | `−0.035` per irrigation | pumping is not free |
| Fertiliser / pesticide cost | `−0.09` per dose | input costs |
| Movement cost | `−0.012` per move | fuel and time |
| Idle cost | `−0.02` per `WAIT` | a work slot spent idle still costs labour time |
| Infeasible action | `−0.45` | discourages flailing |
| Waterlogging | `−0.70` | penalises over-irrigation |
| Nitrate runoff | `−0.55` | environmental externality of over-fertilising |
| Plot death | `−3.0` | losing a plot is the worst local outcome |
| Premature harvest | `−2.5` | discourages cashing in before maturity |

### 2.6 Terminal conditions

1. **Season end** — `day >= 60`: the season settles and the completion bonus is paid.
2. **Field closed** — every plot is either harvested or dead; nothing is left to manage.
3. **Truncation** — the 240-slot budget is exhausted without either of the above.

---

## 3. Repository layout

```
.
├── pyproject.toml              # uv-managed, CPU-pinned torch index
├── uv.lock
├── main.py                     # single entry point: demo / train / sweep / plots
├── play.py                     # replay the best agent in the 3D viewer
├── environment/
│   ├── custom_env.py           # AgriGuardEnv (Gymnasium API)
│   └── rendering.py            # Three.js 3D renderer + live state server
├── training/
│   ├── dqn_training.py         # DQN (value-based) in TorchRL
│   ├── pg_training.py          # REINFORCE, A2C, PPO in TorchRL
│   ├── sweep.py                # 10 hyperparameter runs per algorithm
│   └── plotting.py             # all report figures
├── models/{dqn,pg}/            # saved policy weights
├── logs/                       # per-run CSV metrics + sweep summaries
├── figures/                    # generated report figures
├── assets/viewer.html          # Three.js scene
└── tests/                      # pytest environment contract tests
```

## 4. Integration path

`AgriGuardEnv.snapshot()` serialises the full field state to plain JSON. The
renderer's state server exposes it at `GET /state`, which is exactly the contract a
mobile client or farm dashboard would consume: the trained policy is loaded with
`torch.load`, served behind the same process, and returns the next recommended
action for the farmer. The 3D viewer in `assets/viewer.html` is a pure browser
client of that endpoint, which demonstrates the serialisation path end to end.
