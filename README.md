# Multi-Objective Reinforcement Learning for Tetris (PPO)

Does replacing the usual "lines cleared" reward with a multi-objective reward change how a
Tetris agent plays, and what is the trade-off between raw score and long-term survival?

**Authors:** Aswin Bhat, Harsh Ganatra

---

## Idea in short

Most Tetris RL work rewards only lines cleared, which tends to produce risky play that scores
briefly and then dies early. We train a **Proximal Policy Optimization (PPO)** agent with a
weighted four-part reward and compare it against a lines-only baseline.

| Term | Meaning | Effect |
|------|---------|--------|
| r1 | line-clear score (more lines at once pays more) | reward |
| r2 | newly created holes | penalty |
| r3 | taller / bumpier stack | penalty |
| r4 | small bonus per piece placed without dying | reward |

```
R = w1*r1 - w2*r2 - w3*r3 + w4*r4        (baseline: R = r1)
```

The weights w1..w4 are **fixed hyperparameters** chosen before each run. PPO trains the
neural networks, not these weights. The experiment is an ablation over the weights, with several
random seeds each. We measure what changes; we do not assume the shaped reward wins.

## Problem formulation

- **State:** binary board grid (20 x 10), board features (column heights, aggregate height,
  holes, bumpiness), and the current piece type.
- **Action:** a `(rotation, column)` placement of the current piece, encoded as
  `rotation * 10 + column` (40 slots). Illegal placements are removed by an action mask. A
  placement is legal if the piece can be dropped straight down from above the board.
- **Reward:** the weighted reward above, computed from the board before and after each move.
- **Episode:** ends when the next piece has no legal placement, or after 500 pieces during
  training (time limit; the critic bootstraps through it). Next pieces are drawn at random.
- **Algorithm:** PPO with a categorical policy over the placements, GAE advantages, and
  action masking.

## Network (afterstate actor-critic)

The first version of this project used an actor that read the current board and produced 40
separate output scores. It learned slowly (about 55 pieces survived after 2 million
placements). The final design follows the afterstate-evaluating actor used by recent PPO
Tetris agents:

- **Actor:** for every legal placement, compute the features of the board that would result
  (after placing the piece and clearing lines), and score it with ONE small MLP shared by all
  placements. The scores of illegal placements are masked, and a softmax over the scores gives
  the policy. The features are the abstract's column heights, aggregate height, holes and
  bumpiness, plus standard Dellacherie-Thiery features (lines cleared, landing height, eroded
  cells, row and column transitions, wells, hole depth, rows with holes), see `features.py`.
- **Critic:** a small conv encoder on the current board, an MLP on the 13 board features and a
  piece embedding, giving V(s).
- Actor and critic are separate networks; their gradients are clipped separately.

## Files

| File | What it does |
|------|--------------|
| `reward.py` | Board features (heights, holes, bumpiness) and the four reward terms |
| `features.py` | Vectorized afterstate features for all candidate placements |
| `tetris_env.py` | Placement-based Tetris environment with action masking and afterstate features |
| `network.py` | Afterstate actor + conv critic |
| `ppo.py` | PPO training: rollout buffer, GAE, clipped updates, CSV logging, checkpoints |
| `run_experiments.py` | Launches all runs (reward setting x seed); skips finished runs, resumes interrupted ones |
| `evaluate.py` | Every agent plays the same fixed games; neutral metrics; random and greedy reference agents; checkpoint sweep |
| `plot.py` | Training curves, trade-off plot, final comparison, checkpoint progress |
| `tetris_demo.py` | Streamlit live demo (falling pieces, line-clear animation, reward panel) |
| `tetris_viz.py` | Entry point for the deployed Streamlit app (runs the demo) |
| `requirements-train.txt` | Dependencies for training / evaluation / plots |

## Setup

```bash
pip install -r requirements-train.txt     # training, evaluation, plots
pip install streamlit                      # only for the demo
```

Python 3.10 or newer. Do not put `torch` in `requirements.txt` (Streamlit Cloud installs that
file); that is why training dependencies live in `requirements-train.txt`.

## Quick checks

```bash
python reward.py
python features.py
python tetris_env.py
python network.py
python ppo.py --self-test
streamlit run tetris_demo.py
```

## Running the experiments

```bash
# 1) First run: the multi-objective agent, seed 0
python run_experiments.py --settings multi --seeds 0 --total-steps 2000000 --device cuda

# 2) The baseline (lines only), seed 0
python run_experiments.py --settings baseline --seeds 0 --total-steps 2000000 --device cuda

# 3) Full set: all settings, 3 seeds (finished runs are skipped)
python run_experiments.py --settings all --seeds 0,1,2 --total-steps 2000000 --parallel 3

# 4) Evaluate all trained agents on the same fixed games (+ checkpoint progress)
python evaluate.py --sweep

# 5) Draw the graphs into figures/
python plot.py
```

- The network is tiny, so `--device cpu` can be as fast as or faster than `--device cuda`.
  Time 20,000 steps both ways and pick the faster one.
- `python run_experiments.py --list` shows the reward settings (edit `SETTINGS` in that file).
- Every run lives in `runs/<setting>_s<seed>/` with `config.json`, `metrics.csv`, `train.log`
  and `checkpoints/`.
- Checkpoints are saved at PPO iterations 1, 10, 50 and then every 100th (one iteration is
  2048 placements), plus `latest.pt` and `final.pt`:

```python
from ppo import load_checkpoint
model, meta = load_checkpoint("runs/multi_s0/checkpoints/iter_000100.pt")
```

Checkpoints from the first (slot-based) design cannot be loaded; `load_checkpoint` says so
with a clear error. Keep old `runs/` folders out of the new `runs/` directory.

Use the same training budget and number of seeds for every setting. Evaluation uses seeds
that training never sees, and every metric is counted from the game itself (lines, pieces
survived, holes, height, bumpiness), never from an agent's own reward. Evaluation caps games at
1000 pieces (`--max-pieces`); well-trained agents reach the cap, so also compare lines per game
and board quality, and raise the cap if survival saturates.

## Graphs

| Figure | Source |
|--------|--------|
| fig1 lines cleared vs training steps | training logs |
| fig2 pieces survived vs training steps | training logs |
| fig3 trade-off: lines vs survival per setting | evaluate.py |
| fig4 final comparison bars (lines, pieces, holes, height, bumpiness) | evaluate.py |
| fig5 holes and height during training | training logs |
| fig6 policy entropy | training logs |
| fig7 policy and value loss | training logs |
| fig8 agent quality across saved checkpoints | evaluate.py --sweep |

## Design notes

- The environment is written directly in NumPy for speed and testability instead of wrapping
  `tetris-gymnasium` (named in the abstract). Team decision pending.
- Board height is not stated in the abstract. We use the standard 20 rows.
- r2 and r3 are measured as the change caused by the move (after line clearing minus before).
  r3 is the change in bumpiness plus the change in maximum height.
- The weights in `SETTINGS` are starting points, not tuned values.
- Because the actor sees the same board features that define r2 and r3, the lines-only baseline
  can also learn to avoid holes through the value signal. The gap between baseline and
  multi-objective agents may therefore be smaller than expected; that is a valid result.

## Current status

Environment, reward, features, network, PPO, launcher, evaluation, plotting and the demo are
written and self-tested. The full experiments have not been run yet, so there are no results.

## References

1. Schulman et al., Proximal policy optimization algorithms, arXiv:1707.06347, 2017.
2. Schulman et al., High-dimensional continuous control using generalized advantage estimation, ICLR 2016.
3. Huang and Ontanon, A closer look at invalid action masking in policy gradient algorithms, FLAIRS 2022.
4. Weichart and Hartl, Tetris Gymnasium: A configurable reinforcement learning environment, 2024.
5. Thiery and Scherrer, Building controllers for Tetris, ICGA Journal 32(1), 2009.
6. Roijers et al., A survey of multi-objective sequential decision-making, JAIR 48, 2013.
