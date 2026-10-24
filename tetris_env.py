"""
tetris_env.py - placement-based Tetris environment (the "wrapper").

One env.step(action) = one piece placed.
    action = rotation * cols + column      (4 rotations x 10 columns = 40 slots)
    column = leftmost board column the piece's shape occupies

For every state the env also prepares what the agent needs to decide:
    mask   (40,)    True = legal placement (illegal slots: rotations the piece does not have,
                    columns where the shape does not fit)
    cand   (40, F)  features of the AFTERSTATE of every placement (the board after placing the
                    piece and clearing lines). The actor scores each legal candidate from these
                    features with one shared function. Illegal slots are zeros.

Board convention: (rows, cols) int8 array, row 0 is the TOP, 0 = empty,
1..7 = id of the piece that filled the cell (useful for colors in the visualizer).

Reward comes from reward.py (R = w1*r1 - w2*r2 - w3*r3 + w4*r4).
"""

import numpy as np

from features import afterstate_features
from reward import RewardWeights, board_features, compute_reward

PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]

_BASE_SHAPES = {
    "I": [[1, 1, 1, 1]],
    "O": [[1, 1], [1, 1]],
    "T": [[0, 1, 0], [1, 1, 1]],
    "S": [[0, 1, 1], [1, 1, 0]],
    "Z": [[1, 1, 0], [0, 1, 1]],
    "J": [[1, 0, 0], [1, 1, 1]],
    "L": [[0, 0, 1], [1, 1, 1]],
}


def _unique_rotations(shape: np.ndarray):
    """All distinct rotations of a shape (O has 1, I/S/Z have 2, T/J/L have 4)."""
    rotations, seen, cur = [], set(), shape
    for _ in range(4):
        cur = np.ascontiguousarray(cur)
        key = (cur.shape, cur.tobytes())
        if key not in seen:
            seen.add(key)
            rotations.append(cur.copy())
        cur = np.rot90(cur)
    return rotations


# ROTATIONS[piece_index] -> list of 2D int8 arrays (one per distinct rotation)
ROTATIONS = [
    _unique_rotations(np.array(_BASE_SHAPES[name], dtype=np.int8)) for name in PIECE_NAMES
]


def _shape_info(shape: np.ndarray) -> dict:
    h, w = shape.shape
    bottom = np.array([max(i for i in range(h) if shape[i, j]) for j in range(w)])
    ci, cj = np.nonzero(shape)
    return {"h": h, "w": w, "bottom": bottom, "ci": ci, "cj": cj}


SHAPE_INFO = [[_shape_info(s) for s in rots] for rots in ROTATIONS]


class TetrisPlacementEnv:
    def __init__(
        self,
        rows: int = 20,
        cols: int = 10,
        weights: RewardWeights = RewardWeights(),
        max_pieces: int | None = None,
        seed: int | None = None,
    ):
        self.rows, self.cols = rows, cols
        self.weights = weights
        self.max_pieces = max_pieces  # episode cap (truncation), None = play until the board overflows
        self.num_actions = 4 * cols
        self.feature_dim = cols + 3  # board features: heights + aggregate height + holes + bumpiness
        self.cand_dim = cols + 12  # afterstate features per candidate (see features.py)
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((rows, cols), dtype=np.int8)

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.pieces_placed = 0
        self.total_lines = 0
        self.total_reward = 0.0
        self.piece = int(self.rng.integers(len(PIECE_NAMES)))
        self.mask, self.cand = self._candidates(self.piece)
        return self._state(), self.mask.copy()

    def step(self, action: int):
        action = int(action)
        if not (0 <= action < self.num_actions) or not self.mask[action]:
            raise ValueError(f"illegal action {action} for piece {PIECE_NAMES[self.piece]}")

        rotation, col = divmod(action, self.cols)
        shape = ROTATIONS[self.piece][rotation]
        h, w = shape.shape
        board_before = self.board.copy()

        # 1) drop the piece to its landing row and lock it in
        row = self._landing_row(shape, col)
        region = self.board[row : row + h, col : col + w]
        region[shape > 0] = self.piece + 1

        # 2) clear full lines (rows above fall down)
        full = (self.board > 0).all(axis=1)
        lines = int(full.sum())
        if lines:
            kept = self.board[~full]
            self.board = np.vstack([np.zeros((lines, self.cols), dtype=np.int8), kept])

        placed_piece = self.piece
        self.pieces_placed += 1
        self.total_lines += lines

        # 3) draw next piece; game over if it has no legal placement
        self.piece = int(self.rng.integers(len(PIECE_NAMES)))
        self.mask, self.cand = self._candidates(self.piece)
        terminated = not self.mask.any()
        truncated = self.max_pieces is not None and self.pieces_placed >= self.max_pieces

        # 4) reward
        r = compute_reward(board_before, self.board, lines, terminated, self.weights)
        self.total_reward += r.total

        info = {
            "piece": PIECE_NAMES[placed_piece],  # the piece that was just placed
            "rotation": rotation,
            "column": col,
            "landing_row": row,
            "lines_cleared": lines,
            "pieces_placed": self.pieces_placed,
            "total_lines": self.total_lines,
            "terminated": terminated,
            "truncated": truncated,
            "reward_components": (r.r1, r.r2, r.r3, r.r4),
        }
        done = terminated or truncated
        return self._state(), r.total, done, info, self.mask.copy()

    def grid(self) -> np.ndarray:
        """Current board for the visualizer (copy)."""
        return self.board.copy()

    # ------------------------------------------------------------- internals
    def _fits(self, shape: np.ndarray, row: int, col: int) -> bool:
        h, w = shape.shape
        if row + h > self.rows or col + w > self.cols:
            return False
        region = self.board[row : row + h, col : col + w]
        return not np.any((region > 0) & (shape > 0))

    def _landing_row(self, shape: np.ndarray, col: int) -> int:
        row = 0
        while self._fits(shape, row + 1, col):
            row += 1
        return row

    def _landings(self, piece: int):
        """Vectorized landing search. For each rotation: (rotation, legal columns, landing rows).
        A piece slides straight down from the top, so in each column it stops at the first filled cell:
        landing_row = min over the shape's columns of (first filled row - lowest shape cell - 1)."""
        filled = self.board > 0
        top = np.where(filled.any(axis=0), filled.argmax(axis=0), self.rows)
        out = []
        for rot, info in enumerate(SHAPE_INFO[piece]):
            w = info["w"]
            n = self.cols - w + 1
            idx = np.arange(n)[:, None] + np.arange(w)[None, :]
            land = (top[idx] - info["bottom"][None, :] - 1).min(axis=1)
            cols = np.flatnonzero(land >= 0)  # must fit at the top row to be placeable
            out.append((rot, cols, land[cols]))
        return out

    def _compute_mask(self, piece: int) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=bool)
        for rot, cols, _ in self._landings(piece):
            mask[rot * self.cols + cols] = True
        return mask

    def _afterstates(self, piece: int):
        """Boards after every legal placement (piece locked, full lines cleared).
        Returns dict: actions (N,), boards (N,R,C), lines (N,), eroded (N,), landing_height (N,)."""
        R, C = self.rows, self.cols
        acts, boards, lines_l, eroded_l, landing_l = [], [], [], [], []
        for rot, cols, rows in self._landings(piece):
            k = len(cols)
            if k == 0:
                continue
            info = SHAPE_INFO[piece][rot]
            b = np.broadcast_to(self.board, (k, R, C)).copy()
            rr = rows[:, None] + info["ci"][None, :]
            cc = cols[:, None] + info["cj"][None, :]
            ar = np.arange(k)[:, None]
            b[ar, rr, cc] = piece + 1
            full = (b > 0).all(axis=2)  # (k, R)
            lines = full.sum(axis=1)
            eroded = lines * full[ar, rr].sum(axis=1)
            landing = (R - rows - info["h"]) + (info["h"] - 1) / 2.0
            for n in np.flatnonzero(lines > 0):  # only the few candidates that clear lines
                kept = b[n][~full[n]]
                b[n] = np.vstack([np.zeros((int(lines[n]), C), dtype=np.int8), kept])
            acts.append(rot * C + cols)
            boards.append(b)
            lines_l.append(lines)
            eroded_l.append(eroded)
            landing_l.append(landing)
        if not acts:
            return {"actions": np.zeros(0, dtype=int), "boards": np.zeros((0, R, C), dtype=np.int8),
                    "lines": np.zeros(0), "eroded": np.zeros(0), "landing_height": np.zeros(0)}
        return {
            "actions": np.concatenate(acts), "boards": np.concatenate(boards),
            "lines": np.concatenate(lines_l), "eroded": np.concatenate(eroded_l),
            "landing_height": np.concatenate(landing_l),
        }

    def _candidates(self, piece: int):
        """(mask, cand): legal-placement mask (A,) and afterstate features (A, cand_dim)."""
        mask = np.zeros(self.num_actions, dtype=bool)
        cand = np.zeros((self.num_actions, self.cand_dim), dtype=np.float32)
        a = self._afterstates(piece)
        if len(a["actions"]):
            mask[a["actions"]] = True
            cand[a["actions"]] = afterstate_features(
                a["boards"], a["lines"], a["landing_height"], a["eroded"]
            )
        return mask, cand

    def _state(self) -> dict:
        f = board_features(self.board)
        features = np.concatenate(
            [
                f["heights"] / self.rows,
                [
                    f["aggregate_height"] / (self.rows * self.cols),
                    f["holes"] / (self.rows * self.cols),
                    f["bumpiness"] / (self.rows * (self.cols - 1)),
                ],
            ]
        ).astype(np.float32)
        return {
            "board": (self.board > 0).astype(np.float32)[None],  # (1, rows, cols)
            "features": features,  # (cols + 3,)
            "piece": self.piece,  # int 0..6
            "cand": self.cand,  # (num_actions, cand_dim) afterstate features per placement
        }


# ----------------------------------------------------------------------------
# Self-test + benchmark:  python tetris_env.py
# ----------------------------------------------------------------------------
def _self_test():
    import time

    env = TetrisPlacementEnv(seed=0)
    env.reset()

    # 1) legal-action counts on an empty 10-wide board must match the abstract
    expected = {"O": 9, "I": 17, "S": 17, "Z": 17, "T": 34, "L": 34, "J": 34}
    for idx, name in enumerate(PIECE_NAMES):
        n = int(env._compute_mask(idx).sum())
        assert n == expected[name], (name, n)
    print("legal actions on empty board:", {n: int(env._compute_mask(i).sum()) for i, n in enumerate(PIECE_NAMES)})

    # 2) hand-made line clear: bottom row filled in cols 4..9, horizontal I at col 0
    env.reset()
    env.board[19, 4:] = 1
    env.piece = 0  # I
    env.mask, env.cand = env._candidates(0)
    _, reward, done, info, _ = env.step(0 * env.cols + 0)  # rotation 0 (horizontal), col 0
    assert info["lines_cleared"] == 1 and info["landing_row"] == 19, info
    assert (env.board[19] == 0).all()
    print("line clear test ok, info:", {k: info[k] for k in ("piece", "landing_row", "lines_cleared")})

    # 3) afterstate features: O piece completing the bottom row clears 1 line and erodes 2 piece cells
    env.reset()
    env.board[19, :6] = 1; env.board[19, 8:] = 1
    env.piece = 1  # O
    mask, cand = env._candidates(1)
    a = env._afterstates(1)
    k = int(np.flatnonzero(a["actions"] == 6)[0])  # O at column 6
    assert a["lines"][k] == 1 and a["eroded"][k] == 2 and abs(a["landing_height"][k] - 0.5) < 1e-9
    assert cand[6, 13] == 1 / 2 and abs(cand[6, 15] - 2 / 8) < 1e-6  # lines_cleared, eroded (scaled)
    assert not cand[~mask].any()  # illegal slots are all zeros
    print("afterstate features ok")

    # 4) env.step() lands exactly where the afterstate preview said, for many random moves
    rng = np.random.default_rng(5)
    for g in range(30):
        env = TetrisPlacementEnv(seed=g); env.reset()
        done = False
        while not done:
            a = env._afterstates(env.piece)
            act = int(rng.choice(np.flatnonzero(env.mask)))
            k = int(np.flatnonzero(a["actions"] == act)[0])
            _, _, done, _, _ = env.step(act)
            assert (env.board == a["boards"][k]).all()
    print("step() matches the afterstate preview")

    # 5) determinism with the same seed
    x, y = TetrisPlacementEnv(seed=1), TetrisPlacementEnv(seed=1)
    x.reset(), y.reset()
    for _ in range(30):
        act = int(rng.choice(np.flatnonzero(x.mask)))
        sx, rx, dx, _, _ = x.step(act)
        sy, ry, dy, _, _ = y.step(act)
        assert rx == ry and dx == dy and (x.board == y.board).all() and (sx["cand"] == sy["cand"]).all()
        if dx:
            break
    print("determinism ok")

    # 6) random agent: game lengths and speed (including the afterstate features)
    env = TetrisPlacementEnv(seed=2)
    lengths, steps, t0 = [], 0, time.time()
    for g in range(200):
        env.reset()
        done = False
        while not done:
            act = int(rng.choice(np.flatnonzero(env.mask)))
            _, _, done, info, _ = env.step(act)
            steps += 1
        lengths.append(info["pieces_placed"])
    dt = time.time() - t0
    print(f"random agent: mean length {np.mean(lengths):.1f} pieces, {steps / dt:.0f} placements/sec")
    print("tetris_env.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
