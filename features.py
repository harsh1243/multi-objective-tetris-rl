"""
features.py - board features of an AFTERSTATE (the board right after a piece is placed
and full lines are cleared), computed for many candidate placements at once.

These are the standard Dellacherie-Thiery style features (Thiery and Scherrer 2009, the
paper cited as [5] in the abstract) plus the abstract's own four (column heights,
aggregate height, holes, bumpiness). The actor scores each candidate placement from these
numbers with ONE shared function, instead of learning 40 separate output slots.

Board convention: (N, rows, cols) arrays, row 0 is the TOP, non-zero = filled.
Every feature is divided by a fixed constant so typical values are around 0..1
(fixed constants, not running statistics, so PPO's stored log-probs stay valid).
"""

import numpy as np

FEATURE_NAMES = (
    [f"height_{i}" for i in range(10)]
    + [
        "aggregate_height", "max_height", "bumpiness", "lines_cleared",
        "landing_height", "eroded_cells", "row_transitions", "col_transitions",
        "holes", "wells", "hole_depth", "rows_with_holes",
    ]
)
NUM_CAND_FEATURES = len(FEATURE_NAMES)  # 22 for a 10-wide board

# Fixed divisors (rows=20, cols=10), chosen from typical values seen during play.
_SCALES = dict(
    height=20.0, aggregate=100.0, max_height=20.0, bumpiness=30.0, lines=2.0,
    landing=20.0, eroded=8.0, row_trans=60.0, col_trans=60.0,
    holes=20.0, wells=20.0, hole_depth=60.0, rows_holes=8.0,
)


def column_heights_batch(filled: np.ndarray) -> np.ndarray:
    """filled: (N, R, C) bool -> heights (N, C)."""
    R = filled.shape[1]
    has = filled.any(axis=1)
    first = filled.argmax(axis=1)
    return np.where(has, R - first, 0)


def afterstate_features(boards: np.ndarray, lines: np.ndarray, landing_height: np.ndarray,
                        eroded: np.ndarray) -> np.ndarray:
    """
    boards        (N, R, C) board AFTER line clearing
    lines         (N,) lines cleared by the move
    landing_height(N,) height of the placed piece's centre (before clearing)
    eroded        (N,) lines cleared x piece cells that were in the cleared lines
    returns       (N, 12 + C) float32 feature matrix
    """
    filled = boards > 0
    N, R, C = filled.shape
    h = column_heights_batch(filled)
    max_h = h.max(axis=1)

    # holes: empty cells that have a filled cell somewhere above them
    cum = np.cumsum(filled, axis=1)
    hole_mask = (cum > 0) & ~filled
    holes = hole_mask.sum(axis=(1, 2))
    rows_with_holes = hole_mask.any(axis=2).sum(axis=1)
    hole_depth = (cum * hole_mask).sum(axis=(1, 2))  # filled cells stacked above each hole

    bump = np.abs(np.diff(h, axis=1)).sum(axis=1)

    # row transitions (walls count as filled), only for rows inside the stack
    pad = np.ones((N, R, 1), dtype=bool)
    p = np.concatenate([pad, filled, pad], axis=2)
    row_tr = (p[:, :, 1:] != p[:, :, :-1]).sum(axis=2)  # (N, R)
    in_stack = np.arange(R)[None, :] >= (R - max_h)[:, None]
    row_transitions = (row_tr * in_stack).sum(axis=1)

    # column transitions (floor counts as filled, space above the board is empty)
    floor = np.ones((N, 1, C), dtype=bool)
    q = np.concatenate([filled, floor], axis=1)
    col_transitions = (q[:, 1:, :] != q[:, :-1, :]).sum(axis=(1, 2))

    # wells: empty cells with filled (or wall) left and right neighbours; a vertical run of
    # d such cells adds 1 + 2 + ... + d
    left = np.concatenate([pad, filled[:, :, :-1]], axis=2)
    right = np.concatenate([filled[:, :, 1:], pad], axis=2)
    well_cells = ~filled & left & right
    depth = np.zeros((N, C), dtype=np.int32)
    wells = np.zeros(N, dtype=np.int64)
    for r in range(R):
        depth = np.where(well_cells[:, r, :], depth + 1, 0)
        wells += depth.sum(axis=1)

    s = _SCALES
    cols = [
        h / s["height"],
        (h.sum(axis=1) / s["aggregate"])[:, None],
        (max_h / s["max_height"])[:, None],
        (bump / s["bumpiness"])[:, None],
        (np.asarray(lines) / s["lines"])[:, None],
        (np.asarray(landing_height) / s["landing"])[:, None],
        (np.asarray(eroded) / s["eroded"])[:, None],
        (row_transitions / s["row_trans"])[:, None],
        (col_transitions / s["col_trans"])[:, None],
        (holes / s["holes"])[:, None],
        (wells / s["wells"])[:, None],
        (hole_depth / s["hole_depth"])[:, None],
        (rows_with_holes / s["rows_holes"])[:, None],
    ]
    return np.concatenate(cols, axis=1).astype(np.float32)


# ----------------------------------------------------------------------------
# Self-test with hand-calculated boards:  python features.py
# ----------------------------------------------------------------------------
def _feat(board, lines=0, landing=0.0, eroded=0.0):
    f = afterstate_features(np.asarray(board)[None], np.array([lines]), np.array([landing]),
                            np.array([eroded]))[0]
    s = _SCALES
    return {
        "agg": round(f[10] * s["aggregate"]), "max": round(f[11] * s["max_height"]),
        "bump": round(f[12] * s["bumpiness"]), "row_tr": round(f[16] * s["row_trans"]),
        "col_tr": round(f[17] * s["col_trans"]), "holes": round(f[18] * s["holes"]),
        "wells": round(f[19] * s["wells"]), "hole_depth": round(f[20] * s["hole_depth"]),
        "rows_holes": round(f[21] * s["rows_holes"]),
    }


def _self_test():
    R, C = 20, 10
    # 1) empty board: only the 10 floor transitions (one per column)
    e = np.zeros((R, C), dtype=np.int8)
    f = _feat(e)
    assert f == {"agg": 0, "max": 0, "bump": 0, "row_tr": 0, "col_tr": 10, "holes": 0,
                 "wells": 0, "hole_depth": 0, "rows_holes": 0}, f

    # 2) a single block in the bottom-left corner
    b = e.copy(); b[19, 0] = 1
    f = _feat(b)
    assert (f["agg"], f["max"], f["bump"], f["holes"]) == (1, 1, 1, 0), f
    assert f["row_tr"] == 2 and f["col_tr"] == 10, f  # one filled->empty and one empty->wall in row 19

    # 3) a block floating over an empty cell: exactly one hole, one block above it
    b = e.copy(); b[18, 3] = 1
    f = _feat(b)
    assert (f["holes"], f["rows_holes"], f["hole_depth"]) == (1, 1, 1), f

    # 4) a 3-deep well between two towers: depth sum 1 + 2 + 3 = 6
    b = e.copy(); b[17:, 0] = 1; b[17:, 2] = 1
    assert _feat(b)["wells"] == 6, _feat(b)

    # 5) a 2-deep well next to the wall counts the wall as filled: 1 + 2 = 3
    b = e.copy(); b[18:, 1:] = 1
    assert _feat(b)["wells"] == 3 + 0, _feat(b)

    # 6) bumpiness of a staircase 1,2,3: |1-2| + |2-3| + |3-0| (+ zeros) = 5
    b = e.copy(); b[19, 0] = 1; b[18:, 1] = 1; b[17:, 2] = 1
    assert _feat(b)["bump"] == 1 + 1 + 3 + 0, _feat(b)

    # 7) feature matrix shape and scaling stay in a sane range
    out = afterstate_features(np.stack([e, b]), np.array([0, 2]), np.array([0.5, 3.0]), np.array([0, 8]))
    assert out.shape == (2, NUM_CAND_FEATURES) and out.dtype == np.float32
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() < 3
    print("features.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
