"""
reward.py - board features and the multi-objective reward for Tetris.

R = w1*r1 - w2*r2 - w3*r3 + w4*r4

r1: line-clear score for this move (more lines at once pays more)
r2: change in number of holes caused by this move   (positive = worse)
r3: change in bumpiness + change in max height       (positive = worse)
r4: survival bonus, 1 unit per piece placed without ending the game

Board convention: 2D numpy array of shape (rows, cols), row 0 is the TOP.
Any non-zero cell is filled. This matches the visualizer.

The baseline agent uses only r1 (see baseline_weights()).
"""

from dataclasses import dataclass

import numpy as np

# Reward for clearing 0, 1, 2, 3, 4 lines in one move.
LINE_CLEAR_SCORES = (0.0, 1.0, 3.0, 5.0, 8.0)


# ----------------------------------------------------------------------------
# Board features (also used for the agent's state and for the visualizer)
# ----------------------------------------------------------------------------
def column_heights(board: np.ndarray) -> np.ndarray:
    """Height of each column (0 if empty)."""
    filled = board > 0
    rows = board.shape[0]
    return np.where(filled.any(axis=0), rows - filled.argmax(axis=0), 0)


def count_holes(board: np.ndarray) -> int:
    """Empty cells that have at least one filled cell above them in the same column."""
    filled = board > 0
    holes = 0
    for c in range(board.shape[1]):
        col = filled[:, c]
        if col.any():
            top = col.argmax()
            holes += int((~col[top:]).sum())
    return holes


def bumpiness(heights: np.ndarray) -> int:
    """Sum of absolute height differences between adjacent columns."""
    return int(np.abs(np.diff(heights)).sum())


def board_features(board: np.ndarray) -> dict:
    """All hand-crafted features in one place."""
    h = column_heights(board)
    return {
        "heights": h,
        "aggregate_height": int(h.sum()),
        "max_height": int(h.max()),
        "holes": count_holes(board),
        "bumpiness": bumpiness(h),
    }


# ----------------------------------------------------------------------------
# Reward
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardWeights:
    w1: float = 1.0  # line clears
    w2: float = 0.5  # new holes
    w3: float = 0.2  # height / bumpiness
    w4: float = 0.1  # survival


def baseline_weights() -> RewardWeights:
    """Single-objective baseline: R = r1."""
    return RewardWeights(w1=1.0, w2=0.0, w3=0.0, w4=0.0)


@dataclass(frozen=True)
class RewardInfo:
    """Raw components (before weights) plus the final scalar. Handy for logging."""

    r1: float
    r2: float
    r3: float
    r4: float
    total: float


def compute_reward(
    board_before: np.ndarray,
    board_after: np.ndarray,
    lines_cleared: int,
    game_over: bool,
    weights: RewardWeights = RewardWeights(),
) -> RewardInfo:
    """
    board_before : settled board BEFORE the piece was placed
    board_after  : settled board AFTER placing the piece AND clearing full lines
    lines_cleared: number of full rows removed by this move (0 to 4)
    game_over    : True if this move ended the game
    """
    fb = board_features(board_before)
    fa = board_features(board_after)

    r1 = LINE_CLEAR_SCORES[min(int(lines_cleared), 4)]
    r2 = float(fa["holes"] - fb["holes"])
    r3 = float((fa["bumpiness"] - fb["bumpiness"]) + (fa["max_height"] - fb["max_height"]))
    r4 = 0.0 if game_over else 1.0

    total = weights.w1 * r1 - weights.w2 * r2 - weights.w3 * r3 + weights.w4 * r4
    return RewardInfo(r1=r1, r2=r2, r3=r3, r4=r4, total=float(total))


# ----------------------------------------------------------------------------
# Quick self-test on hand-made boards:  python reward.py
# ----------------------------------------------------------------------------
def _empty(rows=20, cols=10):
    return np.zeros((rows, cols), dtype=int)


def _self_test():
    # 1) nothing changes -> only the survival bonus
    e = _empty()
    info = compute_reward(e, e, 0, False, RewardWeights(1, 0.5, 0.2, 0.1))
    assert (info.r1, info.r2, info.r3, info.r4) == (0, 0, 0, 1), info
    assert abs(info.total - 0.1) < 1e-9

    # 2) a block floating over an empty cell creates exactly 1 hole
    after = _empty()
    after[18, 0] = 1  # filled, with row 19 col 0 empty underneath
    assert count_holes(after) == 1
    info = compute_reward(_empty(), after, 0, False, RewardWeights(1, 1, 0, 0))
    assert info.r2 == 1 and info.total == -1.0, info

    # 3) clearing one line: bottom row had 9 cells, piece fills the 10th, row removed
    before = _empty()
    before[19, :9] = 1
    info = compute_reward(before, _empty(), 1, False, baseline_weights())
    assert info.r1 == 1.0 and info.total == 1.0, info
    assert info.r3 < 0  # the stack got lower and flatter -> negative penalty = reward

    # 4) game over -> no survival bonus
    info = compute_reward(e, e, 0, True, RewardWeights(0, 0, 0, 1))
    assert info.r4 == 0 and info.total == 0

    # 5) baseline ignores everything except line clears
    info = compute_reward(e, after, 2, False, baseline_weights())
    assert info.total == 3.0

    print("reward.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
