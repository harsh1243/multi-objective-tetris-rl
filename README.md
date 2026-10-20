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
neural network, not these weights. The experiment is an ablation over the weights, with several
random seeds each. We measure what changes; we do not assume the shaped reward wins.

## Problem formulation

- **State:** binary board grid (20 x 10), a feature vector (column heights, aggregate height,
  holes, bumpiness), and the current piece type.
- **Action:** a `(rotation, column)` placement of the current piece, encoded as
  `rotation * 10 + column` (40 slots). Illegal placements are removed by an action mask.
- **Reward:** the weighted reward above, computed from the board before and after each move.
- **Episode:** ends when the next piece has no legal placement. Next pieces are drawn at random.
- **Algorithm:** PPO with a categorical policy, GAE advantages, and a shared conv encoder for
  the board plus a small MLP for the features.

## Files

| File | What it does |
|------|--------------|
| `reward.py` | Board features (heights, holes, bumpiness) and the four reward terms |
| `tetris_env.py` | Placement-based Tetris environment with action masking (`reset` / `step`) |
| `network.py` | Actor-critic network: conv board encoder + feature MLP + piece embedding, masked policy |
| `ppo.py` | PPO training: rollout buffer, GAE, clipped updates, CSV logging, checkpoints |
| `run_experiments.py` | Launches all runs (reward setting x seed), in parallel if wanted; skips finished runs, resumes interrupted ones |
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

Python 3.10 or newer. `streamlit` 1.37 or newer for the demo. Do not put `torch` in
`requirements.txt` (Streamlit Cloud installs that file); that is why training dependencies
live in `requirements-train.txt`.

## Quick checks

Each module has a built-in self-test:

```bash
python reward.py
python tetris_env.py
python network.py
python ppo.py --self-test
streamlit run tetris_demo.py
```

## Running the experiments

```bash
# 1) Pilot: one baseline and one multi-objective run. Check steps/s and that curves improve.
python run_experiments.py --settings baseline,multi --seeds 0 --total-steps 2000000 --device cuda

# 2) Full ablation: all settings, 3 seeds, a few runs at once (about one per CPU core)
python run_experiments.py --settings all --seeds 0,1,2 --total-steps 2000000 --parallel 3

# 3) Evaluate all trained agents on the same fixed games (+ checkpoint progress)
python evaluate.py --sweep

# 4) Draw the graphs into figures/
python plot.py
```

- `python run_experiments.py --list` shows the reward settings (edit `SETTINGS` in that file).
- Re-running the same command is safe: finished runs are skipped, interrupted runs resume.
- Every run lives in `runs/<setting>_s<seed>/` with `config.json`, `metrics.csv`, `train.log`
  and `checkpoints/`.
- Checkpoints are saved at PPO iterations 1, 10, 50 and then every 100th (one iteration is
  2048 placements), plus `latest.pt` and `final.pt`. Load one with:

```python
from ppo import load_checkpoint
model, meta = load_checkpoint("runs/multi_s0/checkpoints/iter_000100.pt")
```

Use the same training budget and number of seeds for every setting. Evaluation uses seeds
that training never sees, and every metric is counted from the game itself (lines, pieces
survived, holes, height, bumpiness), never from an agent's own reward.

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

## Design notes and open decisions

- The environment is written directly in NumPy for speed and testability instead of wrapping
  `tetris-gymnasium` (named in the abstract). Team decision pending.
- Board height is not stated in the abstract. We use the standard 20 rows.
- r2 and r3 are measured as the change caused by the move (after line clearing minus before).
  r3 is the change in bumpiness plus the change in maximum height.
- Reward terms are not normalized to a common scale and the weights in `SETTINGS` are starting
  points, not tuned values.
- No episode-length cap during training by default. Evaluation caps games at 500 pieces.
- Early test runs (250k placements, single seed) improved over random play but flattened at
  roughly 40 to 45 pieces; longer runs are needed to know where training saturates. Run the
  pilot first and choose the training budget from its curves.

## Current status

Environment, reward, network, PPO, launcher, evaluation, plotting and the demo are written and
self-tested. The full experiments have not been run yet, so there are no results.

## References

1. Schulman et al., Proximal policy optimization algorithms, arXiv:1707.06347, 2017.
2. Schulman et al., High-dimensional continuous control using generalized advantage estimation, ICLR 2016.
3. Huang and Ontanon, A closer look at invalid action masking in policy gradient algorithms, FLAIRS 2022.
4. Weichart and Hartl, Tetris Gymnasium: A configurable reinforcement learning environment, 2024.
5. Thiery and Scherrer, Building controllers for Tetris, ICGA Journal 32(1), 2009.
6. Roijers et al., A survey of multi-objective sequential decision-making, JAIR 48, 2013.
