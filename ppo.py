"""
ppo.py - PPO training for the placement-based Tetris env.

One ITERATION = collect n_envs * n_steps placements (default 8 * 256 = 2048), compute GAE
advantages, run a few epochs of clipped PPO updates on that data, then throw the data away.

Everything for a run goes to  runs/<run_name>/ :
    config.json              hyperparameters + reward weights + seed
    metrics.csv              one row per iteration (losses, entropy, KL, clip fraction,
                             episode lines / pieces survived, holes, height, bumpiness)
    checkpoints/iter_000001.pt, iter_000010.pt, iter_000050.pt, iter_000100.pt, iter_000200.pt, ...
    checkpoints/latest.pt    refreshed every few iterations (use it to resume)
    checkpoints/final.pt     end of training

Examples:
    python ppo.py --run-name baseline_s0 --weights baseline --seed 0
    python ppo.py --run-name multi_s0 --weights 1,0.5,0.2,0.1 --seed 0 --total-steps 3000000
    python ppo.py --run-name multi_s0 --resume runs/multi_s0/checkpoints/latest.pt
    python ppo.py --self-test

Load a saved agent later:
    from ppo import load_checkpoint
    model, meta = load_checkpoint("runs/multi_s0/checkpoints/iter_000100.pt")
"""

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from network import TetrisActorCritic, to_tensors
from reward import RewardWeights
from tetris_env import TetrisPlacementEnv


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
@dataclass
class PPOConfig:
    run_name: str = "run"
    out_dir: str = "runs"
    seed: int = 0
    weights: tuple = (1.0, 0.5, 0.2, 0.1)  # (w1, w2, w3, w4); baseline = (1, 0, 0, 0)
    max_pieces: int | None = None  # episode cap; None = play until the board overflows
    total_steps: int = 2_000_000  # total placements to train for
    n_envs: int = 8
    n_steps: int = 256  # per env; buffer = n_envs * n_steps = 2048
    n_epochs: int = 4
    minibatch_size: int = 256
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    ckpt_early: tuple = (1, 10, 50)  # always saved
    ckpt_every: int = 100  # then every 100th iteration: 100, 200, 300, ...
    latest_every: int = 10  # refresh latest.pt (for resuming)
    log_every: int = 10  # console print frequency


def parse_weights(s: str) -> tuple:
    s = s.strip().lower()
    if s == "baseline":
        return (1.0, 0.0, 0.0, 0.0)
    if s in ("default", "multi"):
        return (1.0, 0.5, 0.2, 0.1)
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("weights must be 'baseline', 'default' or 'w1,w2,w3,w4'")
    return tuple(parts)


def should_checkpoint(it: int, cfg: PPOConfig) -> bool:
    return it in cfg.ckpt_early or (cfg.ckpt_every > 0 and it % cfg.ckpt_every == 0)


# ----------------------------------------------------------------------------
# GAE
# ----------------------------------------------------------------------------
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """
    rewards, values, dones: arrays (T, E). dones[t] = 1 if the episode ended at step t.
    last_value: (E,) value of the observation after the final step.
    Returns advantages and returns, both (T, E).
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros_like(last_value)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


# ----------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------
def load_checkpoint(path: str, device: str = "cpu"):
    """Returns (model in eval mode, meta dict with iteration, total_steps, config, metrics)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TetrisActorCritic().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {k: ckpt[k] for k in ("iteration", "total_steps", "config", "metrics")}
    return model, meta


METRIC_COLUMNS = [
    "iteration", "total_steps", "elapsed_sec", "steps_per_sec",
    "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance",
    "episodes", "ep_lines_mean", "ep_pieces_mean", "ep_train_reward_mean",
    "avg_holes", "avg_height", "avg_bumpiness",
]


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------
class PPOTrainer:
    def __init__(self, cfg: PPOConfig, device: str | None = None, resume: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        w = RewardWeights(*cfg.weights)
        self.envs = [
            TetrisPlacementEnv(weights=w, max_pieces=cfg.max_pieces, seed=cfg.seed * 10_000 + i)
            for i in range(cfg.n_envs)
        ]
        self.rows, self.cols = self.envs[0].rows, self.envs[0].cols
        self.num_actions = self.envs[0].num_actions

        self.model = TetrisActorCritic(self.rows, self.cols, 7, self.num_actions).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)

        self.iteration = 0
        self.total_steps = 0
        self.run_dir = os.path.join(cfg.out_dir, cfg.run_name)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        if resume:
            ckpt = torch.load(resume, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.opt.load_state_dict(ckpt["optimizer"])
            self.iteration, self.total_steps = ckpt["iteration"], ckpt["total_steps"]
            print(f"resumed from {resume} at iteration {self.iteration}")
        else:
            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                json.dump(asdict(cfg), f, indent=2)
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(METRIC_COLUMNS)

        self.states, self.masks = [], []
        for env in self.envs:
            s, m = env.reset()
            self.states.append(s)
            self.masks.append(m)
        self.last_metrics = {}

    # ------------------------------------------------------------------
    def _value_of(self, states, masks):
        b, f, p, m = to_tensors(states, masks, self.device)
        with torch.no_grad():
            _, v = self.model(b, f, p, m)
        return v.cpu().numpy()

    def collect_rollout(self):
        cfg, T, E = self.cfg, self.cfg.n_steps, self.cfg.n_envs
        R, C, A, F = self.rows, self.cols, self.num_actions, self.cols + 3
        dev = self.device

        buf_board = torch.zeros((T, E, 1, R, C), device=dev)
        buf_feat = torch.zeros((T, E, F), device=dev)
        buf_piece = torch.zeros((T, E), dtype=torch.long, device=dev)
        buf_mask = torch.zeros((T, E, A), dtype=torch.bool, device=dev)
        buf_action = torch.zeros((T, E), dtype=torch.long, device=dev)
        buf_logp = torch.zeros((T, E), device=dev)
        values = np.zeros((T, E), dtype=np.float32)
        rewards = np.zeros((T, E), dtype=np.float32)
        dones = np.zeros((T, E), dtype=np.float32)

        episodes = []  # (lines, pieces, train_reward) of games that ended in this rollout
        sum_holes = sum_height = sum_bump = 0.0

        for t in range(T):
            b, f, p, m = to_tensors(self.states, self.masks, dev)
            action, logp, value = self.model.act(b, f, p, m)
            buf_board[t], buf_feat[t], buf_piece[t], buf_mask[t] = b, f, p, m
            buf_action[t], buf_logp[t] = action, logp
            values[t] = value.cpu().numpy()
            acts = action.cpu().numpy()

            trunc_idx, trunc_states, trunc_masks = [], [], []
            for i, env in enumerate(self.envs):
                state, r, done, info, mask = env.step(int(acts[i]))
                feats = state["features"]
                sum_height += float(feats[: self.cols].max()) * R
                sum_holes += float(feats[self.cols + 1]) * R * C
                sum_bump += float(feats[self.cols + 2]) * R * (C - 1)
                rewards[t, i] = r
                dones[t, i] = float(done)
                if done:
                    if info["truncated"] and not info["terminated"]:
                        trunc_idx.append(i)
                        trunc_states.append(state)
                        trunc_masks.append(mask)
                    episodes.append((env.total_lines, env.pieces_placed, env.total_reward))
                    state, mask = env.reset()
                self.states[i], self.masks[i] = state, mask

            if trunc_idx:  # time-limit ending, not a real death: bootstrap with the critic
                rewards[t, trunc_idx] += cfg.gamma * self._value_of(trunc_states, trunc_masks)

        last_value = self._value_of(self.states, self.masks)
        adv, ret = compute_gae(rewards, values, dones, last_value, cfg.gamma, cfg.gae_lambda)

        n = T * E
        rollout = {
            "board": buf_board.reshape(n, 1, R, C),
            "features": buf_feat.reshape(n, F),
            "piece": buf_piece.reshape(n),
            "mask": buf_mask.reshape(n, A),
            "action": buf_action.reshape(n),
            "logp": buf_logp.reshape(n),
            "adv": torch.as_tensor(adv.reshape(n), device=dev),
            "ret": torch.as_tensor(ret.reshape(n), device=dev),
        }
        var_ret = ret.var()
        explained = float("nan") if var_ret < 1e-12 else float(1 - (ret - values).var() / var_ret)
        stats = {
            "episodes": len(episodes),
            "ep_lines_mean": float(np.mean([e[0] for e in episodes])) if episodes else float("nan"),
            "ep_pieces_mean": float(np.mean([e[1] for e in episodes])) if episodes else float("nan"),
            "ep_train_reward_mean": float(np.mean([e[2] for e in episodes])) if episodes else float("nan"),
            "avg_holes": sum_holes / n,
            "avg_height": sum_height / n,
            "avg_bumpiness": sum_bump / n,
            "explained_variance": explained,
        }
        return rollout, stats

    def update(self, rollout):
        cfg = self.cfg
        n = rollout["action"].shape[0]
        pol, val, ent, kl, clipf, count = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for _ in range(cfg.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, cfg.minibatch_size):
                idx = perm[start : start + cfg.minibatch_size]
                logp, entropy, value = self.model.evaluate(
                    rollout["board"][idx], rollout["features"][idx], rollout["piece"][idx],
                    rollout["mask"][idx], rollout["action"][idx],
                )
                adv = rollout["adv"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                log_ratio = logp - rollout["logp"][idx]
                ratio = log_ratio.exp()
                pg_loss = torch.max(
                    -adv * ratio, -adv * torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
                ).mean()
                v_loss = ((value - rollout["ret"][idx]) ** 2).mean()
                ent_mean = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_mean

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    pol += pg_loss.item(); val += v_loss.item(); ent += ent_mean.item()
                    kl += ((ratio - 1) - log_ratio).mean().item()
                    clipf += ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()
                    count += 1
        return {
            "policy_loss": pol / count, "value_loss": val / count, "entropy": ent / count,
            "approx_kl": kl / count, "clip_fraction": clipf / count,
        }

    # ------------------------------------------------------------------
    def save(self, name: str):
        path = os.path.join(self.ckpt_dir, name)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.opt.state_dict(),
                "iteration": self.iteration,
                "total_steps": self.total_steps,
                "config": asdict(self.cfg),
                "metrics": self.last_metrics,
            },
            path,
        )
        return path

    def train(self):
        cfg = self.cfg
        per_iter = cfg.n_steps * cfg.n_envs
        total_iters = math.ceil(cfg.total_steps / per_iter)
        t0 = time.time()
        start_steps = self.total_steps
        while self.iteration < total_iters:
            if cfg.anneal_lr:
                frac = 1.0 - self.iteration / total_iters
                for g in self.opt.param_groups:
                    g["lr"] = cfg.lr * frac

            rollout, stats = self.collect_rollout()
            losses = self.update(rollout)
            self.iteration += 1
            self.total_steps += per_iter
            elapsed = time.time() - t0
            row = {
                "iteration": self.iteration, "total_steps": self.total_steps,
                "elapsed_sec": elapsed, "steps_per_sec": (self.total_steps - start_steps) / max(elapsed, 1e-9),
                **losses, **stats,
            }
            self.last_metrics = row
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([row[c] for c in METRIC_COLUMNS])

            if should_checkpoint(self.iteration, cfg):
                self.save(f"iter_{self.iteration:06d}.pt")
            if self.iteration % cfg.latest_every == 0:
                self.save("latest.pt")
            if self.iteration % cfg.log_every == 0 or self.iteration == 1 or self.iteration == total_iters:
                print(
                    f"it {self.iteration:5d}/{total_iters} steps {self.total_steps:9d} | "
                    f"lines {row['ep_lines_mean']:7.1f} pieces {row['ep_pieces_mean']:7.1f} | "
                    f"pol {row['policy_loss']:+.3f} val {row['value_loss']:.3f} ent {row['entropy']:.2f} "
                    f"kl {row['approx_kl']:.4f} clip {row['clip_fraction']:.2f} | {row['steps_per_sec']:.0f} steps/s",
                    flush=True,
                )
        self.save("final.pt")
        print(f"done. checkpoints in {self.ckpt_dir}")


# ----------------------------------------------------------------------------
# Self-test:  python ppo.py --self-test
# ----------------------------------------------------------------------------
def _self_test():
    import shutil
    import tempfile

    # 1) GAE with lambda = 1 and no episode ends equals discounted return + bootstrap
    rng = np.random.default_rng(0)
    r = rng.normal(size=(5, 2)).astype(np.float32)
    v = rng.normal(size=(5, 2)).astype(np.float32)
    last_v = rng.normal(size=(2,)).astype(np.float32)
    g = 0.9
    _, ret = compute_gae(r, v, np.zeros_like(r), last_v, g, 1.0)
    for t in range(5):
        expect = sum(g**k * r[t + k] for k in range(5 - t)) + g ** (5 - t) * last_v
        assert np.allclose(ret[t], expect, atol=1e-5)

    # 2) an episode end blocks reward/bootstrap from flowing back across it
    d = np.zeros((5, 2), dtype=np.float32)
    d[1] = 1.0
    _, ret = compute_gae(r, v, d, last_v, g, 1.0)
    assert np.allclose(ret[0], r[0] + g * r[1], atol=1e-5)
    print("GAE ok")

    # 3) checkpoint schedule: 1, 10, 50, then every 100
    cfg = PPOConfig()
    its = [i for i in range(1, 501) if should_checkpoint(i, cfg)]
    assert its == [1, 10, 50, 100, 200, 300, 400, 500], its
    print("checkpoint iterations:", its)

    # 4) tiny training run: files, CSV rows, save/load round trip, resume
    tmp = tempfile.mkdtemp()
    try:
        cfg = PPOConfig(run_name="t", out_dir=tmp, n_envs=2, n_steps=32, minibatch_size=32,
                        total_steps=3 * 64, ckpt_early=(1, 2), ckpt_every=0, latest_every=1, log_every=1)
        tr = PPOTrainer(cfg, device="cpu")
        tr.train()
        files = sorted(os.listdir(os.path.join(tmp, "t", "checkpoints")))
        assert files == ["final.pt", "iter_000001.pt", "iter_000002.pt", "latest.pt"], files
        with open(os.path.join(tmp, "t", "metrics.csv")) as f:
            assert len(list(csv.reader(f))) == 1 + 3
        model, meta = load_checkpoint(os.path.join(tmp, "t", "checkpoints", "final.pt"))
        assert meta["iteration"] == 3 and meta["total_steps"] == 3 * 64
        s, m = tr.envs[0].reset()
        b, f_, p, mk = to_tensors(s, m)
        with torch.no_grad():
            l1, _ = tr.model.eval()(b, f_, p, mk)
            l2, _ = model(b, f_, p, mk)
        assert torch.allclose(l1, l2, atol=1e-6)
        cfg2 = PPOConfig(**{**asdict(cfg), "total_steps": 4 * 64, "weights": cfg.weights,
                            "ckpt_early": (1, 2), })
        tr2 = PPOTrainer(cfg2, device="cpu", resume=os.path.join(tmp, "t", "checkpoints", "final.pt"))
        tr2.train()
        assert tr2.iteration == 4
        print("tiny run, checkpoints, load and resume ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ppo.py: all self-tests passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--weights", default="default", help="baseline | default | w1,w2,w3,w4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--max-pieces", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n-epochs", type=int, default=4)
    ap.add_argument("--minibatch-size", type=int, default=256)
    ap.add_argument("--resume", default=None, help="path to a checkpoint (e.g. latest.pt)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return
    cfg = PPOConfig(
        run_name=a.run_name, out_dir=a.out_dir, seed=a.seed, weights=parse_weights(a.weights),
        total_steps=a.total_steps, n_envs=a.n_envs, n_steps=a.n_steps, max_pieces=a.max_pieces, lr=a.lr,
        ent_coef=a.ent_coef, gamma=a.gamma, n_epochs=a.n_epochs, minibatch_size=a.minibatch_size,
    )
    PPOTrainer(cfg, device=a.device, resume=a.resume).train()


if __name__ == "__main__":
    main()
