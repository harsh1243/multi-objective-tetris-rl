"""
network.py - actor-critic network for placement-based Tetris (afterstate actor).

ACTOR   one small scoring network applied to EVERY legal placement:
            afterstate features of the placement (22 numbers) -> MLP -> one score
        scores of illegal placements are set to -1e9, then softmax -> probabilities.
        The same weights score all 40 slots, so "holes are bad" is learned once for every
        piece and every position (this is the afterstate-evaluating actor used by the
        state-of-the-art PPO Tetris agents).

CRITIC  conv encoder on the current board + MLP on the 13 board features + piece embedding
        -> one number, V(s) = how good the current position is.

Inputs (obs dict of batched tensors, built by to_tensors):
    board    (B, 1, rows, cols)  binary board
    features (B, cols + 3)       heights, aggregate height, holes, bumpiness
    piece    (B,)                current piece type, 0..6
    cand     (B, 40, cols + 12)  afterstate features per placement slot
    mask     (B, 40) bool        True = legal placement

PPO (ppo.py) trains this network. Actor and critic are separate networks (no shared layers).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

NEG_INF = -1e9
ARCH = "afterstate-v1"  # stored in checkpoints so old-design checkpoints are rejected cleanly


def _init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    """Orthogonal init, the usual choice for PPO."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    return layer


class TetrisActorCritic(nn.Module):
    def __init__(self, rows: int = 20, cols: int = 10, num_pieces: int = 7,
                 num_actions: int = 40, hidden: int = 64):
        super().__init__()
        self.rows, self.cols, self.num_actions = rows, cols, num_actions
        cand_dim, feature_dim = cols + 12, cols + 3

        # ---- actor: one shared scorer for every candidate placement
        self.scorer = nn.Sequential(
            _init(nn.Linear(cand_dim, hidden)), nn.ReLU(),
            _init(nn.Linear(hidden, hidden)), nn.ReLU(),
            _init(nn.Linear(hidden, 1), gain=0.01),  # small init -> near-uniform policy at the start
        )

        # ---- critic: board conv + features + piece embedding
        self.conv = nn.Sequential(
            _init(nn.Conv2d(1, 16, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(16, 32, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, 1, rows, cols)).shape[1]
        self.board_fc = nn.Sequential(_init(nn.Linear(conv_out, 64)), nn.ReLU())
        self.feat_mlp = nn.Sequential(_init(nn.Linear(feature_dim, 64)), nn.ReLU())
        self.piece_emb = nn.Embedding(num_pieces, 8)
        self.trunk = nn.Sequential(_init(nn.Linear(64 + 64 + 8, 128)), nn.ReLU())
        self.value_head = _init(nn.Linear(128, 1), gain=1.0)

    # ------------------------------------------------------------------
    def forward(self, obs):
        """Returns (masked logits (B, 40), value (B,))."""
        scores = self.scorer(obs["cand"]).squeeze(-1)
        logits = scores.masked_fill(~obs["mask"], NEG_INF)
        b = self.board_fc(self.conv(obs["board"]))
        f = self.feat_mlp(obs["features"])
        p = self.piece_emb(obs["piece"])
        value = self.value_head(self.trunk(torch.cat([b, f, p], dim=1))).squeeze(-1)
        return logits, value

    # ---- helpers used by the PPO loop ---------------------------------
    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        """Pick an action while collecting experience. Returns (action, log_prob, value)."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, action):
        """Re-score stored actions during the PPO update. Returns (log_prob, entropy, value)."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value


# ----------------------------------------------------------------------
# Converting env output (numpy) into tensors
# ----------------------------------------------------------------------
def to_tensors(states, masks, device="cpu"):
    """
    states: a single state dict from the env, or a list of them
    masks : a single (num_actions,) bool array, or a list of them
    returns an obs dict of tensors with a batch dimension
    """
    if isinstance(states, dict):
        states, masks = [states], [masks]
    return {
        "board": torch.as_tensor(np.stack([s["board"] for s in states]), dtype=torch.float32, device=device),
        "features": torch.as_tensor(np.stack([s["features"] for s in states]), dtype=torch.float32, device=device),
        "piece": torch.as_tensor([s["piece"] for s in states], dtype=torch.long, device=device),
        "cand": torch.as_tensor(np.stack([s["cand"] for s in states]), dtype=torch.float32, device=device),
        "mask": torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device),
    }


def index_obs(obs, idx):
    """Select a minibatch from a dict of tensors."""
    return {k: v[idx] for k, v in obs.items()}


# ----------------------------------------------------------------------
# Self-test:  python network.py
# ----------------------------------------------------------------------
def _self_test():
    from tetris_env import TetrisPlacementEnv

    torch.manual_seed(0)
    net = TetrisActorCritic()
    actor_params = sum(p.numel() for p in net.scorer.parameters())
    total = sum(p.numel() for p in net.parameters())
    print(f"parameters: actor {actor_params:,}, total {total:,}")

    env = TetrisPlacementEnv(seed=0)
    state, mask = env.reset()

    # 1) shapes
    obs = to_tensors(state, mask)
    logits, value = net(obs)
    assert logits.shape == (1, 40) and value.shape == (1,)

    # 2) illegal actions get exactly zero probability
    probs = torch.softmax(logits, dim=-1)[0].detach()
    assert float(probs[~obs["mask"][0]].sum()) == 0.0
    assert abs(float(probs.sum()) - 1.0) < 1e-5

    # 3) the actor uses ONE shared scoring function: shuffling the candidate slots shuffles
    #    the logits in exactly the same way
    perm = torch.randperm(40)
    obs2 = dict(obs)
    obs2["cand"], obs2["mask"] = obs["cand"][:, perm], obs["mask"][:, perm]
    logits2, _ = net(obs2)
    assert torch.allclose(logits[:, perm], logits2, atol=1e-6)

    # 4) sampling always returns a legal action (many draws, many boards)
    for _ in range(300):
        obs = to_tensors(state, mask)
        action, logp, _ = net.act(obs)
        assert mask[int(action)], "sampled an illegal action"
        state, _, done, _, mask = env.step(int(action))
        if done:
            state, mask = env.reset()

    # 5) batched evaluate + backward pass (what PPO will do)
    states, masks = [], []
    state, mask = env.reset()
    for _ in range(16):
        states.append(state); masks.append(mask)
        a = int(np.random.choice(np.flatnonzero(mask)))
        state, _, done, _, mask = env.step(a)
        if done:
            state, mask = env.reset()
    obs = to_tensors(states, masks)
    actions = torch.as_tensor([int(np.random.choice(np.flatnonzero(mk))) for mk in masks])
    logp, entropy, value = net.evaluate(obs, actions)
    loss = -logp.mean() + value.pow(2).mean() - 0.01 * entropy.mean()
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())
    print(f"batched evaluate ok, mean entropy {float(entropy.detach().mean()):.2f}")

    # 6) an untrained agent plays full games
    lengths = []
    for g in range(20):
        state, mask = env.reset()
        done = False
        while not done:
            action, _, _ = net.act(to_tensors(state, mask))
            state, _, done, info, mask = env.step(int(action))
        lengths.append(info["pieces_placed"])
    print(f"untrained network: mean game length {np.mean(lengths):.1f} pieces")
    print("network.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
