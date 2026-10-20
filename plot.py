"""
plot.py - draw the graphs for the report / presentation.

From the training logs (runs/<setting>_s<seed>/metrics.csv):
    fig1_lines_vs_steps.png       lines cleared per game during training
    fig2_pieces_vs_steps.png      pieces survived per game during training
    fig5_board_quality.png        average holes and height during training
    fig6_entropy.png              policy entropy (exploring -> committing)
    fig7_losses.png               policy loss and value loss
From evaluate.py (results/summary.csv, results/sweep.csv):
    fig3_tradeoff.png             lines vs survival, one point per setting
    fig4_final_comparison.png     bars: lines, pieces, holes, height, bumpiness
    fig8_checkpoint_progress.png  how agents improve across saved checkpoints

Curves show the mean over seeds as a line and +-1 std as a shaded band.
Every metric is counted from the game itself, so settings with different rewards are comparable.

Usage:  python plot.py [--runs-dir runs] [--results-dir results] [--fig-dir figures] [--smooth 10]
"""

import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#222222", "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]


def load_runs(runs_dir):
    """setting -> list of metrics DataFrames (one per seed)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        csv = os.path.join(d, "metrics.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        if df.empty:
            continue
        name = os.path.basename(d)
        out.setdefault(re.sub(r"_s\d+$", "", name), []).append(df)
    return out


def setting_colors(settings):
    ordered = sorted(settings, key=lambda s: (s != "baseline", s))  # baseline first (black)
    return {s: COLORS[i % len(COLORS)] for i, s in enumerate(ordered)}, ordered


def mean_band(dfs, col, smooth):
    """Mean and std across seeds of a (smoothed) column, aligned on iteration."""
    series = []
    for df in dfs:
        s = df.set_index("iteration")[col].interpolate(limit_direction="both")
        series.append(s.rolling(smooth, min_periods=1).mean())
    both = pd.concat(series, axis=1)
    steps = pd.concat([df.set_index("iteration")["total_steps"] for df in dfs], axis=1).mean(axis=1)
    return steps.loc[both.index].values, both.mean(axis=1).values, both.std(axis=1).fillna(0).values


def curve_panel(ax, runs, cols_colors, col, ylabel, smooth):
    colors, ordered = cols_colors
    for s in ordered:
        x, m, sd = mean_band(runs[s], col, smooth)
        ax.plot(x, m, color=colors[s], label=f"{s} (n={len(runs[s])})", lw=2 if s == "baseline" else 1.6)
        if len(runs[s]) > 1:
            ax.fill_between(x, m - sd, m + sd, color=colors[s], alpha=0.18, lw=0)
    ax.set_xlabel("training steps (placements)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("saved", path)


def training_figures(runs, fig_dir, smooth):
    cc = setting_colors(list(runs))
    singles = [
        ("fig1_lines_vs_steps.png", "ep_lines_mean", "lines cleared per game", "Lines cleared during training"),
        ("fig2_pieces_vs_steps.png", "ep_pieces_mean", "pieces survived per game", "Survival during training"),
        ("fig6_entropy.png", "entropy", "policy entropy", "Policy entropy"),
    ]
    for fname, col, ylabel, title in singles:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        curve_panel(ax, runs, cc, col, ylabel, smooth)
        ax.set_title(title)
        ax.legend(fontsize=8)
        save(fig, os.path.join(fig_dir, fname))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    curve_panel(axes[0], runs, cc, "avg_holes", "average holes on the board", smooth)
    curve_panel(axes[1], runs, cc, "avg_height", "average max height", smooth)
    axes[0].set_title("Holes during training")
    axes[1].set_title("Stack height during training")
    axes[0].legend(fontsize=8)
    save(fig, os.path.join(fig_dir, "fig5_board_quality.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    curve_panel(axes[0], runs, cc, "policy_loss", "policy loss", smooth)
    curve_panel(axes[1], runs, cc, "value_loss", "value loss", smooth)
    axes[1].set_yscale("log")
    axes[0].set_title("Policy loss (stays near zero in PPO)")
    axes[1].set_title("Value loss (log scale)")
    axes[0].legend(fontsize=8)
    save(fig, os.path.join(fig_dir, "fig7_losses.png"))


def eval_figures(summary_path, fig_dir):
    s = pd.read_csv(summary_path)
    cc = setting_colors(list(s["setting"]))
    colors = cc[0]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for _, r in s.iterrows():
        ref = r["setting"].startswith("ref_")
        ax.errorbar(r["lines_mean"], r["pieces_mean"], xerr=r["lines_std"], yerr=r["pieces_std"],
                    fmt="s" if ref else "o", ms=9, capsize=3, color=colors[r["setting"]], alpha=0.9)
        ax.annotate(r["setting"], (r["lines_mean"], r["pieces_mean"]), textcoords="offset points",
                    xytext=(7, 6), fontsize=9)
    ax.set_xlabel("lines cleared per game")
    ax.set_ylabel("pieces survived per game")
    ax.set_title("Trade-off: score vs survival (mean ± std over seeds)")
    ax.grid(alpha=0.3)
    save(fig, os.path.join(fig_dir, "fig3_tradeoff.png"))

    panels = [("lines", "lines cleared"), ("pieces", "pieces survived"), ("avg_holes", "avg holes"),
              ("avg_height", "avg max height"), ("avg_bumpiness", "avg bumpiness")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (m, title) in zip(axes.ravel(), panels):
        ax.bar(range(len(s)), s[f"{m}_mean"], yerr=s[f"{m}_std"].fillna(0), capsize=3,
               color=[colors[x] for x in s["setting"]])
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels(s["setting"], rotation=40, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Final comparison on the same fixed evaluation games")
    save(fig, os.path.join(fig_dir, "fig4_final_comparison.png"))


def sweep_figure(sweep_path, fig_dir):
    d = pd.read_csv(sweep_path)
    cc = setting_colors(list(d["setting"]))
    colors, ordered = cc
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (m, ylabel) in zip(axes, [("lines", "lines cleared per game"), ("pieces", "pieces survived per game")]):
        for s in ordered:
            g = d[d["setting"] == s].groupby("iteration")[m].agg(["mean", "std"]).fillna(0)
            ax.plot(g.index, g["mean"], "o-", color=colors[s], label=s, lw=2 if s == "baseline" else 1.6)
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=colors[s], alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("PPO iteration of the saved checkpoint")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Agent quality at saved checkpoints (same evaluation games)")
    save(fig, os.path.join(fig_dir, "fig8_checkpoint_progress.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--smooth", type=int, default=10, help="moving-average window in iterations")
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)

    runs = load_runs(a.runs_dir)
    if runs:
        training_figures(runs, a.fig_dir, a.smooth)
    else:
        print(f"no training logs found in {a.runs_dir}/ (skipping figures 1, 2, 5, 6, 7)")

    summary = os.path.join(a.results_dir, "summary.csv")
    if os.path.exists(summary):
        eval_figures(summary, a.fig_dir)
    else:
        print("no results/summary.csv yet: run evaluate.py for figures 3 and 4")

    sweep = os.path.join(a.results_dir, "sweep.csv")
    if os.path.exists(sweep):
        sweep_figure(sweep, a.fig_dir)
    else:
        print("no results/sweep.csv yet: run 'python evaluate.py --sweep' for figure 8")


if __name__ == "__main__":
    main()
