# AgriGuard: Comparing Value-Based and Policy-Gradient Reinforcement Learning for Precision Irrigation and Crop-Health Management

**Enock Mugisha** — Machine Learning Techniques II, Mission-Based RL Summative
Repository: `https://github.com/enockmugisha1/enock_mugisha_rl_summative`

---

## 1. Mission and problem statement

Smallholder maize farms in Rwanda lose yield to two coupled problems: irrigation water is
scarce and arrives unpredictably as rain, and pest outbreaks spread between adjacent plots
faster than a part-time farmer can respond. **AgriGuard** models the decision problem faced
by a low-cost autonomous farm rover — or equivalently an advisory app telling a farmer what
to do in the next work slot — that must allocate **water, fertiliser, biopesticide and
labour time** across a 4×4 field to maximise the harvest actually brought in at the end of
a 60-day season.

The agent controls one rover on a 16-plot field. A day contains four ~3-hour work slots, so
an episode is **240 sequential decisions**. In each slot the rover may perform exactly one
action on the plot it currently occupies. With one rover, sixteen plots and hard input
budgets, every slot spent irrigating one plot is a slot not spent treating an outbreak
elsewhere.

### 1.1 Why this requires reinforcement learning

* **Delayed credit assignment.** Yield is realised only when a plot is harvested. A plot
  needs roughly **74 consecutive slots** of healthy growth to reach maturity, so the `+14`
  harvest payout is credited long after the irrigation decisions that earned it. No
  immediate signal says that watering plot 7 on day 3 was correct.
* **Non-myopic budgeting under exogenous uncertainty.** The tank refills only when it
  rains, and rainfall is bimodal (wet onset, dry mid-season, wet tail). Irrigating greedily
  in week 2 empties the reservoir before the flowering window in week 6. The optimal policy
  is *state-conditional*, so no fixed irrigation calendar is optimal.
* **A contagion process creates compounding future cost.** Infestation self-amplifies and
  transmits to 4-neighbours every slot, so containing an outbreak requires lookahead over a
  spatial process.
* **It is not supervised learning.** No dataset of labelled "correct actions" exists. The
  objective is cumulative season return under stochastic weather — a sequential decision
  problem by construction.

---

## 2. Environment specification

[[FIG:fig0_environment.png|Figure 1 — The AgriGuard environment rendered live in Three.js (WebGL). Soil brightness encodes moisture, stalk height and colour encode growth stage, red haze encodes infestation severity; the rover occupies one plot at a time and acts on the plot beneath it.]]

### 2.1 Action space — `Discrete(9)`

| # | Action | Real-world mapping | Effect in the simulator |
|---|---|---|---|
| 0–3 | `MOVE_N/S/E/W` | RTK waypoint drive | rover moves one plot; `−0.012` fuel cost |
| 4 | `IRRIGATE` | open drip valve on current plot | `moisture += 0.26`, `tank −= 0.055` |
| 5 | `FERTILIZE` | dose urea/NPK from hopper | `nutrient += 0.40`, one of 12 season doses |
| 6 | `TREAT_PEST` | spray biopesticide | `pest ×= 0.20`, one of 8 season doses |
| 7 | `HARVEST` | cut and bank the plot | banks yield, closes the plot |
| 8 | `WAIT` | idle the slot | conserves water/fuel before forecast rain; `−0.02` labour |

Driving into the field boundary, irrigating from an empty tank, dosing with no stock left,
or acting on an already-harvested plot are *infeasible* and cost `−0.45`.

### 2.2 Observation space — `Box(0, 1, shape=(135,))`

16 plots × 7 per-plot features (112) + 16-dim one-hot rover position + 7 globals = **135
dimensions**, all natively in `[0,1]` so the network consumes the raw vector unscaled.
The full observation-provenance table — meaning, encoding/range, real-world sensor source,
update cadence and path to the agent for every dimension — is reproduced in the repository
README (§2.2) and summarised here:

| Observation | Meaning | Range | Real-world source | Cadence |
|---|---|---|---|---|
| `moisture[16]` | volumetric soil water | `[0,1]` | capacitive probe per plot | 15 min |
| `growth[16]` | development stage | `[0,1]` | Sentinel-2 NDVI + drone multispectral | 5 days |
| `nutrient[16]` | soil nitrogen index | `[0,1]` | in-situ NPK sensor | weekly |
| `pest[16]` | infestation severity | `[0,1]` | CNN on sticky-trap/leaf photos | daily |
| `health[16]` | cumulative stress damage | `[0,1]` | derived state estimator | per slot |
| `alive[16]`, `harvested[16]` | plot status flags | binary | derived / harvest log | per slot |
| `rover[16]` | rover position one-hot | binary | RTK-GPS telemetry | 1 Hz |
| `tank`, `fert_stock`, `pest_stock` | resource levels | `[0,1]` | level sensor, load cell, inventory | on use |
| `day_frac`, `rain_prob`, `temp`, `rained_today` | season & weather | `[0,1]` | clock, Rwanda Met API, station, gauge | 6 h |

### 2.3 Start state

Day 0, immediately after transplanting: `growth = 0.05` on all plots,
`moisture ~ U(0.42, 0.58)`, `nutrient ~ U(0.50, 0.70)`, `pest = 0`, `health = 1`, full tank,
12 fertiliser doses, 8 biopesticide doses, rover parked at plot (0,0).

### 2.4 Transition dynamics

Per slot, evapotranspiration removes `0.011 + 0.020 × temp` moisture; crops grow by
`0.0135 × (1 − 0.8 × pest) × health` **only** while moisture lies in the healthy band
`[0.35, 0.85]` and `nutrient > 0.18`; drought (`< 0.15`) or waterlogging (`> 0.93`)
permanently erode `health`; infestation self-amplifies and transmits to 4-neighbours.
Per day, rain occurs with seasonal probability, adding moisture and refilling the tank.

### 2.5 Reward structure

| Component | Value | Purpose |
|---|---|---|
| Harvest yield | `+14.0 × growth × health × (1 − 0.75 × pest)` | the real objective, paid at harvest |
| Season completion bonus | `+18.0 × (matured harvests / 16)` | premature cuts earn no share |
| Healthy-field shaping | `+0.12 × (plots in band / 16)` per slot | dense signal over a 240-step horizon |
| Plot death | `−3.0` each | the dominant risk |
| Premature harvest | `−2.5` | prevents rush-harvesting |
| Waterlogging / nitrate runoff | `−0.70` / `−0.55` | over-application penalties |
| Water / inputs / movement / idle | `−0.035` / `−0.09` / `−0.012` / `−0.02` | running costs |
| Infeasible action | `−0.45` | discourages flailing |

### 2.6 Terminal conditions

An episode ends when **(a)** the season closes (day 60 = 240 slots), or **(b)** every plot is
either dead or harvested. The completion bonus is paid on termination.

### 2.7 Reference policies

To make returns interpretable, two non-learned references bracket the task:

| Policy | Return | Yield | Matured | Dead |
|---|---|---|---|---|
| Random | −8.7 ± 26.4 | 3.05 | 5.8 | 5.3 |
| Expert heuristic (ceiling) | **171.0 ± 25.9** | 10.65 | 15.7 | 0.3 |

---

## 3. Implementation

All four agents are implemented with **TorchRL** and call the *same* environment factory
(`training/common.py: make_env`), so the comparison is objective. Every run uses an
identical budget of **150,000 environment frames**, collected from 8 parallel farms, and is
evaluated greedily (DQN) or stochastically (policy gradients) on **held-out seeds starting
at 10,000** that never overlap training seeds.

| Algorithm | Family | TorchRL loss module | Key machinery |
|---|---|---|---|
| DQN | Value-based | `DQNLoss` | replay buffer, target net (soft update), ε-greedy, n-step, Double-DQN |
| REINFORCE | Policy gradient (Monte-Carlo) | `ReinforceLoss` | full-return advantage, entropy bonus |
| A2C | Actor–critic | `A2CLoss` | GAE(λ) advantage, value baseline, entropy bonus |
| PPO | Actor–critic (trust region) | `ClipPPOLoss` | clipped surrogate, multi-epoch reuse, GAE(λ) |

*Results, hyperparameter tables and discussion follow in §4–§6.*

---

## 4. Results

All 40 runs (10 configurations × 4 algorithms) used the same environment, the same
150,000-frame budget and the same held-out evaluation seeds.

[[FIG:fig1_comparative_rewards.png|Figure 2 — (a) Training return of the best configuration of each algorithm against the random floor (−8.7) and expert ceiling (+171.0). (b) Held-out evaluation return of all 10 runs per algorithm; every PPO configuration outperforms every configuration of the other three algorithms.]]

### 4.1 Comparative performance

| Algorithm | Best run | Best | Median | Worst | Spread |
|---|---|---|---|---|---|
| **PPO** | `ppo01_baseline` | **+78.83** | +61.58 | +28.50 | 50.3 |
| A2C | `a2c02_rollout_short` | +17.64 | +2.73 | −3.75 | 21.4 |
| REINFORCE | `rf04_lr_high` | +15.41 | +1.04 | −21.38 | 36.8 |
| DQN | `dqn04_lr_high` | +7.17 | −39.50 | −75.97 | 83.1 |
| *Random* | — | −8.7 | | | |
| *Expert heuristic* | — | +171.0 | | | |

PPO recovers **46% of the expert heuristic's return**; DQN recovers 4%. The separation is
categorical rather than marginal: PPO's *worst* configuration (+28.50) beats the *best*
configuration of every other algorithm. Each algorithm's best run clears the random floor,
so all four learn something, but only PPO learns a policy a farmer would deploy.

### 4.2 What the policies actually do

[[FIG:fig7_agronomic_outcomes.png|Figure 3 — Agronomic outcome of each algorithm's best policy: plots brought in at maturity, harvested prematurely, and lost to stress.]]

| Best policy | Matured /16 | Premature | Dead | Yield |
|---|---|---|---|---|
| PPO | **12.6** | 1.2 | 2.1 | 6.65 |
| A2C | 7.8 | 5.5 | 2.6 | 3.92 |
| REINFORCE | 7.8 | 4.4 | 3.1 | 4.21 |
| DQN | 4.0 | 1.5 | **9.9** | 2.69 |

These columns explain the return gap mechanically. PPO keeps the field alive *and* waits
for maturity. A2C and REINFORCE keep plots alive (2.6 and 3.1 dead) but cut 5.5 and 4.4
plots early, forfeiting both the yield multiplier and the completion bonus — they learned
irrigation but not patience. DQN loses **9.9 of 16 plots**: its greedy policy still neglects
large parts of the field.

### 4.3 Hyperparameter experiments

The four tables below are reproduced from `logs/hyperparameter_tables.md`.

**Table 1 — DQN: 10 configurations**

| run_id               |      lr |   gamma |   batch_size |   buffer_size |   eps_end |   eps_anneal_frac |   n_step | loss_fn   |   target_tau | double_dqn   |   utd |   hidden |   eval_return |   eval_return_std |   eval_matured |   eval_dead |
|:---------------------|--------:|--------:|-------------:|--------------:|----------:|------------------:|---------:|:----------|-------------:|:-------------|------:|---------:|--------------:|------------------:|---------------:|------------:|
| dqn04_lr_high        | 0.00025 |    0.99 |          128 |        100000 |      0.25 |               0.8 |        3 | smooth_l1 |        0.005 | True         |    64 |      256 |          7.17 |             27.19 |           4    |        9.88 |
| dqn06_explore_max    | 5e-05   |    0.99 |          128 |        100000 |      0.4  |               0.9 |        3 | smooth_l1 |        0.005 | True         |    64 |      256 |        -19.31 |             23.21 |           2.88 |       12.75 |
| dqn01_tuned_best     | 5e-05   |    0.99 |          128 |        100000 |      0.25 |               0.8 |        3 | smooth_l1 |        0.005 | True         |   128 |      256 |        -19.4  |             12.17 |           1.25 |       14.5  |
| dqn08_target_slow    | 5e-05   |    0.99 |          128 |        100000 |      0.25 |               0.8 |        3 | smooth_l1 |        0.001 | True         |    64 |      256 |        -36.58 |              4.5  |           0    |       15.88 |
| dqn05_explore_greedy | 5e-05   |    0.99 |          128 |        100000 |      0.02 |               0.3 |        3 | smooth_l1 |        0.005 | True         |    64 |      256 |        -38.98 |              6.32 |           0    |       15.88 |
| dqn02_utd_starved    | 5e-05   |    0.99 |          128 |        100000 |      0.25 |               0.8 |        3 | smooth_l1 |        0.005 | True         |     8 |      256 |        -40.03 |              8.28 |           1    |       14.88 |
| dqn03_utd_mid        | 5e-05   |    0.99 |          128 |        100000 |      0.25 |               0.8 |        3 | smooth_l1 |        0.005 | True         |    64 |      256 |        -48.82 |              7.35 |           1.38 |       14.38 |
| dqn09_buffer_small   | 5e-05   |    0.99 |          128 |          5000 |      0.25 |               0.8 |        3 | smooth_l1 |        0.005 | True         |    64 |      256 |        -59.57 |              8.86 |           1    |       14.88 |
| dqn10_naive_l2_1step | 0.00025 |    0.99 |          128 |        100000 |      0.05 |               0.5 |        1 | l2        |        0.005 | False        |     8 |      256 |        -62.2  |              9.69 |           0.62 |       14.75 |
| dqn07_target_fast    | 5e-05   |    0.99 |          128 |        100000 |      0.25 |               0.8 |        3 | smooth_l1 |        0.05  | True         |    64 |      256 |        -75.97 |              8.14 |           0    |       15.88 |

**Table 2 — REINFORCE: 10 configurations**

| run_id             |     lr |   gamma |   frames_per_batch |   entropy_coeff |   normalize_advantage |   critic_coeff |   hidden |   eval_return |   eval_return_std |   eval_matured |   eval_dead |
|:-------------------|-------:|--------:|-------------------:|----------------:|----------------------:|---------------:|---------:|--------------:|------------------:|---------------:|------------:|
| rf04_lr_high       | 0.001  |    0.99 |               4000 |            0.01 |                     1 |            0.5 |      256 |         15.41 |             35.69 |           7.75 |        3.12 |
| rf01_baseline      | 0.0003 |    0.99 |               4000 |            0.01 |                     1 |            0.5 |      256 |          6.37 |             16.88 |           6.88 |        4.75 |
| nan                |        |  nan    |                nan |          nan    |                   nan |          nan   |      nan |          4.66 |             24.24 |           7.12 |        5.25 |
| rf07_entropy_zero  | 0.0003 |    0.99 |               4000 |            0    |                     1 |            0.5 |      256 |          2.58 |             30.21 |           6.12 |        4.5  |
| rf03_rollout_short | 0.0003 |    0.99 |               2000 |            0.01 |                     1 |            0.5 |      256 |          1.15 |             18.9  |           6.12 |        5.12 |
| rf06_entropy_high  | 0.0003 |    0.99 |               4000 |            0.05 |                     1 |            0.5 |      256 |          0.94 |             26.6  |           6.38 |        5.62 |
| rf09_gamma_low     | 0.0003 |    0.95 |               4000 |            0.01 |                     1 |            0.5 |      256 |          0.08 |             19.61 |           6.12 |        4.5  |
| rf08_no_advnorm    | 0.0003 |    0.99 |               4000 |            0.01 |                     0 |            0.5 |      256 |         -4.37 |             21.24 |           5.88 |        4.88 |
| rf02_rollout_long  | 0.0003 |    0.99 |               8000 |            0.01 |                     1 |            0.5 |      256 |         -4.45 |             30.45 |           6.25 |        4.75 |
| rf05_lr_low        | 0.0001 |    0.99 |               4000 |            0.01 |                     1 |            0.5 |      256 |        -21.38 |             22.42 |           5.25 |        7    |

**Table 3 — A2C: 10 configurations**

| run_id              |     lr |   gamma |   frames_per_batch |   entropy_coeff |   lmbda |   critic_coeff |   max_grad_norm |   eval_return |   eval_return_std |   eval_matured |   eval_dead |
|:--------------------|-------:|--------:|-------------------:|----------------:|--------:|---------------:|----------------:|--------------:|------------------:|---------------:|------------:|
| a2c02_rollout_short | 0.0003 |    0.99 |               1000 |            0.01 |    0.95 |            0.5 |           nan   |         17.64 |             19.18 |           7.75 |        2.62 |
| a2c07_entropy_zero  | 0.0003 |    0.99 |               2000 |            0    |    0.95 |            0.5 |           nan   |         16.88 |             20.08 |           7.75 |        2.88 |
| a2c08_lambda_low    | 0.0003 |    0.99 |               2000 |            0.01 |    0.7  |            0.5 |           nan   |         13.46 |             35.91 |           7.12 |        2.88 |
| a2c04_lr_high       | 0.001  |    0.99 |               2000 |            0.01 |    0.95 |            0.5 |           nan   |         13.36 |             19.44 |           8.12 |        4.38 |
| a2c06_entropy_high  | 0.0003 |    0.99 |               2000 |            0.05 |    0.95 |            0.5 |           nan   |          2.79 |             18.58 |           7.12 |        4.38 |
| a2c10_critic_heavy  | 0.0003 |    1    |               2000 |            0.01 |    0.95 |            1   |             0.5 |          2.67 |             26.01 |           7    |        3.5  |
| a2c03_rollout_long  | 0.0003 |    0.99 |               8000 |            0.01 |    0.95 |            0.5 |           nan   |          2.39 |             22.51 |           7    |        3.88 |
| a2c05_lr_low        | 0.0001 |    0.99 |               2000 |            0.01 |    0.95 |            0.5 |           nan   |          1.57 |             20.06 |           6.75 |        4    |
| a2c09_lambda_one    | 0.0003 |    0.99 |               2000 |            0.01 |    1    |            0.5 |           nan   |          1.15 |             18.9  |           6.12 |        5.12 |
| a2c01_baseline      | 0.0003 |    0.99 |               2000 |            0.01 |    0.95 |            0.5 |           nan   |         -3.75 |             12.52 |           6.25 |        4.38 |

**Table 4 — PPO: 10 configurations**

| run_id             |     lr |   frames_per_batch |   clip_epsilon |   ppo_epochs |   minibatch_size |   entropy_coeff |   lmbda |   eval_return |   eval_return_std |   eval_matured |   eval_dead |
|:-------------------|-------:|-------------------:|---------------:|-------------:|-----------------:|----------------:|--------:|--------------:|------------------:|---------------:|------------:|
| ppo01_baseline     | 0.0003 |               4000 |           0.2  |            6 |              256 |            0.01 |    0.95 |         78.83 |             19.98 |          12.62 |        2.12 |
| ppo09_lambda_low   | 0.0003 |               4000 |           0.2  |            6 |              256 |            0.01 |    0.8  |         78.78 |              9.07 |          13    |        2.5  |
| ppo03_clip_loose   | 0.0003 |               4000 |           0.4  |            6 |              256 |            0.01 |    0.95 |         73.57 |             32.81 |          12.38 |        2.25 |
| ppo08_entropy_zero | 0.0003 |               4000 |           0.2  |            6 |              256 |            0    |    0.95 |         67.64 |             25.95 |          11.88 |        3.38 |
| ppo04_epochs_many  | 0.0003 |               4000 |           0.2  |           16 |              256 |            0.01 |    0.95 |         63.09 |             33.21 |          10.38 |        4.75 |
| ppo07_entropy_high | 0.0003 |               4000 |           0.2  |            6 |              256 |            0.05 |    0.95 |         60.08 |             18.37 |          11    |        3.88 |
| ppo10_big_batch    | 0.0003 |               8000 |           0.2  |           10 |              512 |            0.01 |    0.95 |         55.69 |             17.97 |          11    |        3.75 |
| ppo06_lr_high      | 0.001  |               4000 |           0.2  |            6 |              256 |            0.01 |    0.95 |         48    |             21.27 |          10.88 |        4.12 |
| ppo05_epochs_one   | 0.0003 |               4000 |           0.2  |            1 |              256 |            0.01 |    0.95 |         32.49 |             33.02 |          10.12 |        2.88 |
| ppo02_clip_tight   | 0.0003 |               4000 |           0.05 |            6 |              256 |            0.01 |    0.95 |         28.5  |             15.2  |           9.5  |        4.38 |


**DQN** — exploration, replay, target updates, n-step, Double-DQN and the objective were
varied. The ranking is dominated by three controls:

* **Learning rate.** `dqn04_lr_high` (2.5e-4) is the only configuration to clear the random
  floor at **+7.17**; every 5e-5 run lands between −19 and −76. At 150k frames the low
  learning rate simply does not move the value head far enough.
* **Target-network rate.** τ=0.05 (`dqn07_target_fast`, **−75.97**) is the worst run in the
  sweep — a fast-tracking target destroys the stationarity the TD regression depends on.
  τ=0.001 (−36.58) is the best of the 5e-5 group.
* **Replay capacity.** Shrinking the buffer to 5,000 (`dqn09_buffer_small`, −59.57) costs
  ~11 points against the 100,000 baseline (−48.82): with a 240-step horizon a small buffer
  holds only ~20 episodes and the samples stay strongly correlated.
* The deliberately naive control `dqn10_naive_l2_1step` (one-step TD, no Double-DQN, L2
  objective, starved update ratio) scores **−62.20**, confirming the combined value of the
  n-step, Huber and Double-DQN machinery.

**REINFORCE** — `rf04_lr_high` (1e-3) reaches **+15.41**, while `rf05_lr_low` (1e-4)
collapses to **−21.38**, a 37-point swing from the learning rate alone. Removing advantage
normalisation (`rf08_no_advnorm`, −4.37) costs ~11 points against the baseline (+6.37),
which is expected for a Monte-Carlo estimator whose advantage scale varies by an order of
magnitude across the season.

**A2C** — the rollout length matters most: 1,000 frames per batch (**+17.64**) beats 2,000
(−3.75) and 8,000 (+2.39). Shorter rollouts mean more gradient updates per frame, and with
only 150k frames the update count is the binding constraint. GAE λ=0.70 (+13.46) beats
λ=0.95 (−3.75) and λ=1.00 (+1.15) — lower λ trades a little bias for a large variance
reduction over a 240-step horizon.

**PPO** — the most stable of the four, with every run positive. The clipping and epoch
sweeps are both non-monotonic:

| ε (clip) | 0.05 | 0.2 | 0.4 |
|---|---|---|---|
| eval | +28.50 | **+78.83** | +73.57 |

| epochs | 1 | 6 | 16 |
|---|---|---|---|
| eval | +32.49 | **+78.83** | +63.09 |

Tight clipping costs 50 points while loose clipping costs only 5, so on this task PPO is far
more sensitive to *under*-exploration than to a permissive trust region. One epoch wastes
the collected batch; sixteen over-fit the policy to stale data. Notably
`ppo08_entropy_zero` (+67.64) **beats** `ppo07_entropy_high` (+60.08): the clipped surrogate
already supplies enough exploration pressure, so an added entropy bonus mostly costs return.

### 4.4 Optimisation behaviour

[[FIG:fig2_dqn_objective.png|Figure 4 — DQN value objective for the best run: (a) TD loss, (b) mean max-Q on replay batches, (c) epsilon schedule.]]

DQN's TD loss **rises** through training while its mean max-Q **drifts negative to ≈ −3**.
The value head is not converging: it is tracking a moving target under a behaviour policy
that keeps changing, and it still ranks actions well enough to score +7.17 only because the
*relative* ordering of actions survives the poor absolute fit.

[[FIG:fig3_pg_entropy.png|Figure 5 — (a) Policy entropy for the three policy-gradient methods against the uniform-policy maximum ln(9)=2.20. (b) Critic loss on a log scale.]]

The entropy curves separate the three policy-gradient methods cleanly. PPO commits hardest
(2.20 → 1.60), A2C plateaus at 1.85, and REINFORCE stalls near 1.92 — barely distinguishable
from the uniform policy after 150k frames. Panel (b) gives the reason: REINFORCE's critic
loss sits an **order of magnitude above** A2C's and PPO's (≈130 versus ≈10). Its Monte-Carlo
baseline never fits, the advantages stay noisy, and the policy cannot commit.

### 4.5 Convergence and generalisation

[[FIG:fig4_convergence.png|Figure 6 — Frames required to first reach each return threshold, per algorithm.]]

[[FIG:fig5_generalization.png|Figure 7 — Final training return against held-out evaluation return. Points on the diagonal generalise; points below it are optimistic on the training distribution.]]

| Algorithm | Mean train | Mean eval | Gap |
|---|---|---|---|
| PPO | +62.32 | +58.67 | **3.66** |
| A2C | +11.46 | +6.82 | 4.64 |
| REINFORCE | +8.22 | +0.10 | 8.13 |
| DQN | −11.89 | −39.37 | **27.48** |

PPO generalises almost perfectly (3.66 points on a 62-point return). DQN's 27-point gap is
the clearest single statistic in the study: its training returns are collected under an
ε-greedy behaviour policy that still explores at ε=0.25, whereas evaluation is fully greedy.
Removing exploration exposes a value function that has not converged, and performance
collapses on unseen weather seeds.

---

## 5. Discussion

**Why policy gradients beat value iteration here.** AgriGuard pays its real reward once per
plot, roughly 74 slots after the irrigation decisions that earn it, and the dense shaping
term is deliberately small (max 0.12/slot). A value-based agent must propagate that signal
backwards through a 240-step bootstrap chain while the target network keeps moving; the
sweep shows this is exactly where DQN fails (rising TD loss, drifting Q, 83-point spread
across configurations). Policy-gradient methods optimise the return of whole trajectories
directly and never need the intermediate value estimates to be accurate — only the
*advantage sign* has to be right. PPO's clipped surrogate adds the missing ingredient,
bounding each update so a noisy advantage cannot destroy a working policy, which is why it
is both the best (+78.83) and the most stable (every run positive, 3.66-point eval gap).

**Exploration.** The three algorithms explore by different mechanisms and the sweep
separates them. DQN's ε-greedy is state-independent: `dqn05_explore_greedy` (ε→0.02)
matures zero plots, but raising ε to 0.40 only reaches −19.31, because uniform random
actions almost never produce the *sequence* — irrigate repeatedly, then harvest at maturity —
that the task rewards. Entropy-regularised policy gradients explore in policy space
instead, perturbing a behaviour that is already partly correct, which is a far better match
for a task whose reward requires long coordinated action chains.

**Stability.** Spread across the 10 runs is itself a result: DQN 83.1, PPO 50.3 but entirely
above zero. A practitioner deploying DQN here would have to tune carefully to avoid a
catastrophic configuration; PPO's worst case is still a usable policy.

### 5.1 Limitations

* **Budget.** 150k frames is modest. DQN is the most sample-hungry method and is likely the
  most penalised by this ceiling; its best run was still improving when training stopped.
* **Single seed per configuration.** Each cell is one seed, so differences of a few points
  (e.g. +78.83 vs +78.78) are not separable from noise. Only the large effects discussed
  above are safely interpretable.
* **Simulator fidelity.** Growth, pest spread and rainfall are parameterised from
  agronomic rules of thumb, not calibrated against field data from a specific Rwandan site.
* **Ceiling.** The expert heuristic (+171.0) remains well above PPO (+78.83), so none of
  these agents is yet competitive with a well-designed rule-based schedule.

### 5.2 Deployment path

The trained policy is a 135→9 MLP saved as a TorchRL state dict (`models/pg/*.pt`), ~0.4 MB.
It is served by `play.py` through the same `make_env` interface used in training, so the
integration contract is a single `observation → action` call. Exporting to TorchScript or
ONNX would let the same policy run on-device inside the rover controller or behind a small
HTTP endpoint for an advisory mobile app, with the observation vector assembled from the
sensor sources listed in §2.2.

---

## 6. Conclusion

Four reinforcement-learning algorithms were implemented in TorchRL and compared on a custom
240-step sequential farm-management environment under an identical 150k-frame budget, with
10 hyperparameter configurations each. **PPO is the clear winner (+78.83, 12.6 of 16 plots
matured, 46% of the expert ceiling)**, followed by A2C (+17.64) and REINFORCE (+15.41), with
DQN last (+7.17). The ordering follows directly from the environment's delayed-reward
structure: methods that optimise trajectory returns handle ~74-step credit assignment far
better than one-step value bootstrapping, and PPO's trust region converts that advantage
into the most stable results in the study.
