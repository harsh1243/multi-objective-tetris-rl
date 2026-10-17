"""
tetris_env.py - placement-based Tetris environment (the "wrapper").

One env.step(action) = one piece placed.
    action = rotation * cols + column      (4 rotations x 10 columns = 40 slots)
    column = leftmost board column the piece's shape occupies

The env also returns an action mask (True = legal) so the policy never picks an
illegal placement. Illegal slots are: rotations a piece does not have, and
columns where the shape does not fit.

Board convention: (rows, cols) int8 array, row 0 is the TOP, 0 = empty,
1..7 = id of the piece that filled the cell (useful for colors in the visualizer).

Reward comes from reward.py (R = w1*r1 - w2*r2 - w3*r3 + w4*r4).
"""

import numpy as np

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
        self.max_pieces = max_pieces  # episode cap (truncation), None = play until death
        self.num_actions = 4 * cols
        self.feature_dim = cols + 3  # heights + aggregate height + holes + bumpiness
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
        self.mask = self._compute_mask(self.piece)
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
        self.mask = self._compute_mask(self.piece)
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

    def _compute_mask(self, piece: int) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=bool)
        for rot, shape in enumerate(ROTATIONS[piece]):
            w = shape.shape[1]
            for col in range(self.cols - w + 1):
                if self._fits(shape, 0, col):  # must fit at the top to be placeable
                    mask[rot * self.cols + col] = True
        return mask

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
    env.mask = env._compute_mask(0)
    _, reward, done, info, _ = env.step(0 * env.cols + 0)  # rotation 0 (horizontal), col 0
    assert info["lines_cleared"] == 1 and info["landing_row"] == 19, info
    assert (env.board[19] == 0).all()
    print("line clear test ok, info:", {k: info[k] for k in ("piece", "landing_row", "lines_cleared")})

    # 3) determinism with the same seed
    a, b = TetrisPlacementEnv(seed=1), TetrisPlacementEnv(seed=1)
    a.reset(), b.reset()
    rng = np.random.default_rng(5)
    for _ in range(30):
        act = int(rng.choice(np.flatnonzero(a.mask)))
        sa, ra, da, _, _ = a.step(act)
        sb, rb, db, _, _ = b.step(act)
        assert ra == rb and da == db and (a.board == b.board).all()
        if da:
            break
    print("determinism ok")

    # 4) random agent: game lengths and speed
    env = TetrisPlacementEnv(seed=2)
    rng = np.random.default_rng(3)
    lengths, steps, t0 = [], 0, time.time()
    for g in range(300):
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
