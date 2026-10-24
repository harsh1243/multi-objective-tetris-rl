"""
evaluate.py - compare trained agents on the SAME fixed games, with neutral metrics.

Every agent plays the same evaluation seeds (never used in training), so each one sees the
same piece sequences. Metrics are counted from the game itself, not from each agent's
training reward, so agents trained with different rewards can be compared fairly:

    lines          lines cleared per game
    pieces         pieces survived per game (capped at --max-pieces)
    avg_holes      average number of holes over the game
    avg_height     average max stack height over the game
    avg_bumpiness  average bumpiness over the game
    hit_cap        fraction of games that reached the piece cap (survived the whole time)

Usage:
    python evaluate.py                          # final.pt of every run in runs/, plus random + greedy reference agents
    python evaluate.py --checkpoint iter_000100 # evaluate a specific saved iteration instead
    python evaluate.py --sweep                  # also evaluate every saved checkpoint (for the progress-over-training graph)
    python evaluate.py --games 200 --max-pieces 2000 --refs none

Outputs (in results/):  eval_games.csv  summary.csv  [sweep.csv]
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch

from network import to_tensors
from ppo import load_checkpoint
from reward import RewardWeights
from tetris_env import TetrisPlacementEnv

# Greedy reference agent lives in the demo module (it only imports numpy and our own files).
from tetris_demo import greedy_action

METRICS = ["lines", "pieces", "avg_holes", "avg_height", "avg_bumpiness"]
EVAL_SEED_START = 1_000_000  # far from the training seeds (seed * 10000 + env index)


def _board_stats(state, rows, cols):
    f = state["features"]
    return (
        float(f[cols + 1]) * rows * cols,  # holes
        float(f[:cols].max()) * rows,  # max height
        float(f[cols + 2]) * rows * (cols - 1),  # bumpiness
    )


def _result(seed, env, sums, steps, terminated, truncated):
    n = max(steps, 1)
    return {
        "game_seed": seed,
        "lines": env.total_lines,
        "pieces": env.pieces_placed,
        "avg_holes": sums[0] / n,
        "avg_height": sums[1] / n,
        "avg_bumpiness": sums[2] / n,
        "hit_cap": float(truncated and not terminated),
    }


def eval_model(model, seeds, max_pieces, deterministic=True, device="cpu", chunk=50):
    """Play one game per seed with a trained network (batched across games)."""
    torch.manual_seed(0)
    model.to(device).eval()
    results = {}
    for start in range(0, len(seeds), chunk):
        chunk_seeds = list(seeds[start : start + chunk])
        envs = [TetrisPlacementEnv(max_pieces=max_pieces) for _ in chunk_seeds]
        states, masks = [], []
        for env, s in zip(envs, chunk_seeds):
            st, mk = env.reset(seed=s)
            states.append(st)
            masks.append(mk)
        sums = [np.zeros(3) for _ in envs]
        steps = [0] * len(envs)
        active = list(range(len(envs)))
        while active:
            obs = to_tensors([states[i] for i in active], [masks[i] for i in active], device)
            actions, _, _ = model.act(obs, deterministic=deterministic)
            actions = actions.cpu().numpy()
            still = []
            for k, i in enumerate(active):
                env = envs[i]
                st, _, done, info, mk = env.step(int(actions[k]))
                sums[i] += _board_stats(st, env.rows, env.cols)
                steps[i] += 1
                states[i], masks[i] = st, mk
                if done:
                    results[chunk_seeds[i]] = _result(
                        chunk_seeds[i], env, sums[i], steps[i], info["terminated"], info["truncated"]
                    )
                else:
                    still.append(i)
            active = still
    return [results[s] for s in seeds]


def eval_reference(kind, seeds, max_pieces):
    """Random or greedy (best immediate multi-objective reward) reference agents."""
    rng = np.random.default_rng(0)
    weights = RewardWeights()
    out = []
    for s in seeds:
        env = TetrisPlacementEnv(max_pieces=max_pieces)
        state, mask = env.reset(seed=s)
        sums, steps, done, info = np.zeros(3), 0, False, {}
        while not done:
            if kind == "random":
                a = int(rng.choice(np.flatnonzero(env.mask)))
            else:
                a = greedy_action(env, weights, rng)
            state, _, done, info, mask = env.step(a)
            sums += _board_stats(state, env.rows, env.cols)
            steps += 1
        out.append(_result(s, env, sums, steps, info["terminated"], info["truncated"]))
    return out


def find_runs(runs_dir):
    runs = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        cfg_path = os.path.join(d, "config.json")
        if os.path.isdir(os.path.join(d, "checkpoints")) and os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            name = os.path.basename(d)
            runs.append({"dir": d, "name": name, "setting": re.sub(r"_s\d+$", "", name), "seed": cfg["seed"]})
    return runs


def summarize(df):
    """Mean +- std across runs (seeds) for each setting; for single-run agents, std across games."""
    rows = []
    for setting, g in df.groupby("setting", sort=False):
        per_run = g.groupby(["agent"])[METRICS + ["hit_cap"]].mean()
        row = {"setting": setting, "n_runs": len(per_run)}
        for m in METRICS + ["hit_cap"]:
            row[f"{m}_mean"] = per_run[m].mean()
            row[f"{m}_std"] = per_run[m].std() if len(per_run) > 1 else g[m].std()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--checkpoint", default="final", help="final, latest, or e.g. iter_000100")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--max-pieces", type=int, default=1000, help="cap per game so strong agents do not play forever")
    ap.add_argument("--sample", action="store_true", help="sample actions instead of taking the most likely one")
    ap.add_argument("--refs", default="random,greedy", help="reference agents, or 'none'")
    ap.add_argument("--sweep", action="store_true", help="also evaluate every saved checkpoint of every run")
    ap.add_argument("--sweep-games", type=int, default=20)
    ap.add_argument("--sweep-max-pieces", type=int, default=500, help="piece cap used for the checkpoint sweep")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    seeds = list(range(EVAL_SEED_START, EVAL_SEED_START + args.games))
    runs = find_runs(args.runs_dir)
    if not runs and args.refs == "none":
        raise SystemExit(f"no runs found in {args.runs_dir}/")

    rows = []
    for r in runs:
        path = os.path.join(r["dir"], "checkpoints", f"{args.checkpoint}.pt")
        if not os.path.exists(path):
            print(f"skip {r['name']}: {path} not found")
            continue
        model, meta = load_checkpoint(path, args.device)
        print(f"evaluating {r['name']} ({args.checkpoint}, iteration {meta['iteration']}) on {args.games} games ...", flush=True)
        for res in eval_model(model, seeds, args.max_pieces, not args.sample, args.device):
            rows.append({"agent": r["name"], "setting": r["setting"], "seed": r["seed"], **res})

    if args.refs != "none":
        for kind in [k.strip() for k in args.refs.split(",")]:
            print(f"evaluating reference agent '{kind}' on {args.games} games ...", flush=True)
            for res in eval_reference(kind, seeds, args.max_pieces):
                rows.append({"agent": f"ref_{kind}", "setting": f"ref_{kind}", "seed": -1, **res})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, "eval_games.csv"), index=False)
    summary = summarize(df)
    summary.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)

    show = summary[["setting", "n_runs"]].copy()
    for m in METRICS:
        show[m] = [f"{a:.1f} ± {b:.1f}" if not np.isnan(b) else f"{a:.1f}" for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    show["hit_cap"] = [f"{a:.2f}" for a in summary["hit_cap_mean"]]
    print("\n" + show.to_string(index=False))
    print(f"\nsaved {args.results_dir}/eval_games.csv and summary.csv")

    if args.sweep:
        sweep_seeds = seeds[: args.sweep_games]
        srows = []
        for r in runs:
            ckpts = sorted(glob.glob(os.path.join(r["dir"], "checkpoints", "iter_*.pt")))
            final = os.path.join(r["dir"], "checkpoints", "final.pt")
            if os.path.exists(final):
                ckpts.append(final)
            for path in ckpts:
                model, meta = load_checkpoint(path, args.device)
                res = pd.DataFrame(eval_model(model, sweep_seeds, args.sweep_max_pieces, not args.sample, args.device))
                srows.append({
                    "agent": r["name"], "setting": r["setting"], "seed": r["seed"],
                    "checkpoint": os.path.basename(path)[:-3], "iteration": meta["iteration"],
                    "total_steps": meta["total_steps"], **res[METRICS].mean().to_dict(),
                })
                print(f"sweep {r['name']} {os.path.basename(path)}: lines {res['lines'].mean():.1f} pieces {res['pieces'].mean():.1f}", flush=True)
        pd.DataFrame(srows).to_csv(os.path.join(args.results_dir, "sweep.csv"), index=False)
        print(f"saved {args.results_dir}/sweep.csv")


if __name__ == "__main__":
    main()
