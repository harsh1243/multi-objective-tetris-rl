"""
network.py - the "brain": an actor-critic network for placement-based Tetris.

Inputs (batched):
    board    (B, 1, rows, cols) float   binary grid, 1 = filled
    features (B, cols + 3)      float   heights, aggregate height, holes, bumpiness
    piece    (B,)               long    current piece type, 0..6
    mask     (B, num_actions)   bool    True = legal placement

Architecture (shared encoder, two heads):
    board    -> conv layers -> flatten -> linear  --+
    features -> small MLP                           +--> concat -> shared trunk --> actor head  (logits over 40 placements)
    piece    -> embedding                         --+                          +-> critic head (state value V(s))

Illegal actions get a very large negative logit, so their probability is 0
(invalid action masking, Huang & Ontanon 2022). PPO (ppo.py) trains this network.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

NEG_INF = -1e9


def _init(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    """Orthogonal init, the usual choice for PPO."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    return layer


class TetrisActorCritic(nn.Module):
    def __init__(self, rows: int = 20, cols: int = 10, num_pieces: int = 7, num_actions: int = 40):
        super().__init__()
        self.rows, self.cols, self.num_actions = rows, cols, num_actions
        feature_dim = cols + 3

        # board encoder (shared by actor and critic)
        self.conv = nn.Sequential(
            _init(nn.Conv2d(1, 32, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 3, padding=1)), nn.ReLU(),
            nn.MaxPool2d(2),  # (64, rows/2, cols/2)
            _init(nn.Conv2d(64, 64, 3, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = 64 * (rows // 2) * (cols // 2)
        self.board_fc = nn.Sequential(_init(nn.Linear(conv_out, 128)), nn.ReLU())

        # hand-crafted feature MLP and piece embedding
        self.feat_mlp = nn.Sequential(_init(nn.Linear(feature_dim, 64)), nn.ReLU())
        self.piece_emb = nn.Embedding(num_pieces, 16)

        # shared trunk, then the two heads
        self.trunk = nn.Sequential(_init(nn.Linear(128 + 64 + 16, 256)), nn.ReLU())
        self.actor = _init(nn.Linear(256, num_actions), gain=0.01)  # small init -> near-uniform start
        self.critic = _init(nn.Linear(256, 1), gain=1.0)

    # ------------------------------------------------------------------
    def _encode(self, board, features, piece):
        b = self.board_fc(self.conv(board))
        f = self.feat_mlp(features)
        p = self.piece_emb(piece)
        return self.trunk(torch.cat([b, f, p], dim=1))

    def forward(self, board, features, piece, mask):
        """Returns (masked logits, value)."""
        h = self._encode(board, features, piece)
        logits = self.actor(h).masked_fill(~mask, NEG_INF)
        value = self.critic(h).squeeze(-1)
        return logits, value

    # ---- helpers used by the PPO loop ---------------------------------
    @torch.no_grad()
    def act(self, board, features, piece, mask, deterministic: bool = False):
        """Pick an action while collecting experience. Returns (action, log_prob, value)."""
        logits, value = self.forward(board, features, piece, mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, board, features, piece, mask, action):
        """Re-score stored actions during the PPO update. Returns (log_prob, entropy, value)."""
        logits, value = self.forward(board, features, piece, mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy(), value


# ----------------------------------------------------------------------
# Converting env output (numpy) into tensors
# ----------------------------------------------------------------------
def to_tensors(states, masks, device="cpu"):
    """
    states: a single state dict from the env, or a list of them
    masks : a single (num_actions,) bool array, or a list of them
    returns (board, features, piece, mask) tensors with a batch dimension
    """
    if isinstance(states, dict):
        states, masks = [states], [masks]
    board = torch.as_tensor(np.stack([s["board"] for s in states]), dtype=torch.float32, device=device)
    features = torch.as_tensor(np.stack([s["features"] for s in states]), dtype=torch.float32, device=device)
    piece = torch.as_tensor([s["piece"] for s in states], dtype=torch.long, device=device)
    mask = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)
    return board, features, piece, mask


# ----------------------------------------------------------------------
# Self-test:  python network.py
# ----------------------------------------------------------------------
def _self_test():
    from tetris_env import TetrisPlacementEnv

    torch.manual_seed(0)
    net = TetrisActorCritic()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"parameters: {n_params:,}")

    env = TetrisPlacementEnv(seed=0)
    state, mask = env.reset()

    # 1) shapes
    board, features, piece, m = to_tensors(state, mask)
    logits, value = net(board, features, piece, m)
    assert logits.shape == (1, 40) and value.shape == (1,)

    # 2) illegal actions get exactly zero probability
    probs = torch.softmax(logits, dim=-1)[0].detach()
    assert float(probs[~m[0]].sum()) == 0.0
    assert abs(float(probs.sum()) - 1.0) < 1e-5

    # 3) sampling always returns a legal action (over many draws, many boards)
    for _ in range(200):
        board, features, piece, m = to_tensors(state, mask)
        action, logp, _ = net.act(board, features, piece, m)
        assert mask[int(action)], "sampled an illegal action"
        state, _, done, _, mask = env.step(int(action))
        if done:
            state, mask = env.reset()

    # 4) batched evaluate + backward pass (what PPO will do)
    states, masks = [], []
    state, mask = env.reset()
    for _ in range(16):
        states.append(state); masks.append(mask)
        a = int(np.random.choice(np.flatnonzero(mask)))
        state, _, done, _, mask = env.step(a)
        if done:
            state, mask = env.reset()
    board, features, piece, m = to_tensors(states, masks)
    actions = torch.as_tensor([int(np.random.choice(np.flatnonzero(mk))) for mk in masks])
    logp, entropy, value = net.evaluate(board, features, piece, m, actions)
    loss = -logp.mean() + value.pow(2).mean() - 0.01 * entropy.mean()
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())
    print(f"batched evaluate ok, mean entropy {float(entropy.detach().mean()):.2f} (max possible about {np.log(34):.2f})")

    # 5) an untrained agent plays full games
    lengths = []
    for g in range(20):
        state, mask = env.reset()
        done = False
        while not done:
            board, features, piece, m = to_tensors(state, mask)
            action, _, _ = net.act(board, features, piece, m)
            state, _, done, info, mask = env.step(int(action))
        lengths.append(info["pieces_placed"])
    print(f"untrained network: mean game length {np.mean(lengths):.1f} pieces")
    print("network.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
