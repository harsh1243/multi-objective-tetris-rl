"""
tetris_demo.py - live demo of an agent playing Tetris in Streamlit.

Run:  streamlit run tetris_demo.py
Needs tetris_env.py and reward.py in the same folder (numpy + streamlit only, no torch).

What you see:
  - the next piece in the NEXT box, which then falls into the spot the agent chose
  - when a line is completed: it flashes, the row disappears, and the stack drops down
  - live stats (pieces, lines, reward, holes, height, bumpiness) and the last move's r1..r4

The demo agent is Random or Greedy (picks the placement with the best immediate reward).
Pick the reward weights in the sidebar to see how the reward changes the play style.
A trained PPO agent can replace choose_action() later.
"""

import copy
import time

import numpy as np

from reward import RewardWeights, baseline_weights, board_features, compute_reward
from tetris_env import PIECE_NAMES, ROTATIONS, TetrisPlacementEnv

# board id 1..7 = I, O, T, S, Z, J, L  ->  (light, dark) gradient colors
PIECE_COLORS = {
    1: ("#3ddbc4", "#12877a"),
    2: ("#ffc15e", "#d9822b"),
    3: ("#d55bff", "#8a1fb3"),
    4: ("#8be28b", "#2e9b4a"),
    5: ("#ff6b81", "#c9283f"),
    6: ("#6d8bff", "#2f45c9"),
    7: ("#ff9f5a", "#d4621a"),
}

CSS = """
<style>
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stDecoration"], footer, #MainMenu { display:none !important; }
.block-container { padding-top:0.6rem !important; padding-bottom:0 !important; max-width:900px; }
[data-testid="stVerticalBlock"] { gap:0.3rem; }
[data-testid="stSidebar"] { background:#0f1326; }
[data-testid="stSidebar"] * { color:#dfe3ff; }
:root { --cell: min(34px, calc((100vh - 150px) / 20)); --mini: calc(var(--cell) * 0.6); }
.stApp { background: radial-gradient(circle at 50% 0%, #1a1f3a 0%, #0b0e1a 60%); }
.viz-title { text-align:center; font-size:1.8rem; font-weight:800; letter-spacing:1px; line-height:1.2;
  background:linear-gradient(90deg,#6d8bff,#d55bff,#ff6b81); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; margin:0; }
.viz-sub { text-align:center; color:#7c86b8; margin-bottom:0.6rem; font-size:0.85rem; }
.viz-wrap { display:flex; gap:24px; justify-content:center; align-items:flex-start; }
.board-box { position:relative; }
.board { display:grid; padding:10px; background:#0a0d18; border:2px solid #2b3260;
  border-radius:14px; box-shadow:0 0 30px rgba(109,139,255,0.18), inset 0 0 20px rgba(0,0,0,0.6);
  grid-template-columns:repeat(10, var(--cell)); }
.cell { width:var(--cell); height:var(--cell); }
.cell-empty { background:#141829; box-shadow: inset 0 0 0 1px #1d2340; border-radius:3px; }
.cell-fill { border-radius:5px; }
.cell-flash { border-radius:5px; background:#ffffff; box-shadow:0 0 16px #ffffff, 0 0 30px #9fb4ff; }
.banner { position:absolute; top:42%; left:0; right:0; text-align:center; font-size:1.6rem; font-weight:800;
  color:#fff; background:rgba(10,13,24,0.85); padding:14px 0; letter-spacing:2px; }
.panel { width:210px; display:flex; flex-direction:column; gap:10px; }
.stats { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.span2 { grid-column: span 2; }
.card { background:#10142a; border:1px solid #262d57; border-radius:12px; padding:9px 12px; }
.card-label { color:#7c86b8; font-size:0.68rem; letter-spacing:1.5px; text-transform:uppercase; }
.card-value { color:#f2f4ff; font-size:1.3rem; font-weight:700; margin-top:1px; }
.card-small { color:#c9d0ff; font-size:0.85rem; margin-top:3px; }
.next-grid { display:grid; grid-template-columns:repeat(4, var(--mini)); justify-content:center; margin-top:8px; }
.mini { width:var(--mini); height:var(--mini); }
.pos { color:#7be0a0; } .neg { color:#ff8a9b; }
</style>
"""


# ----------------------------------------------------------------------------
# Game logic helpers (no Streamlit here, so they can be tested on their own)
# ----------------------------------------------------------------------------
def peek_next(env) -> int:
    """The piece the env will draw after the current one (peeks at a copy of the RNG)."""
    rng = copy.deepcopy(env.rng)
    return int(rng.integers(len(PIECE_NAMES)))


def simulate_placement(env, action):
    """What would happen if we played `action`? Does not change the env.
    Returns (locked_board, board_after_clear, lines, landing_row, full_row_indices)."""
    rot, col = divmod(int(action), env.cols)
    shape = ROTATIONS[env.piece][rot]
    h, w = shape.shape
    row = env._landing_row(shape, col)
    locked = env.board.copy()
    locked[row : row + h, col : col + w][shape > 0] = env.piece + 1
    full = (locked > 0).all(axis=1)
    lines = int(full.sum())
    after = locked
    if lines:
        after = np.vstack([np.zeros((lines, env.cols), dtype=np.int8), locked[~full]])
    return locked, after, lines, row, np.flatnonzero(full)


def greedy_action(env, weights: RewardWeights, rng) -> int:
    """Pick the legal placement with the best immediate reward (ties broken randomly)."""
    best, best_actions = -np.inf, []
    for a in np.flatnonzero(env.mask):
        _, after, lines, _, _ = simulate_placement(env, a)
        r = compute_reward(env.board, after, lines, False, weights).total
        if r > best + 1e-9:
            best, best_actions = r, [int(a)]
        elif abs(r - best) <= 1e-9:
            best_actions.append(int(a))
    return int(rng.choice(best_actions))


def play_one_piece(env, action, show, delay, last_prev=None):
    """
    Animate one placement and apply it to the env.
    show(grid, overlay=None, flash_rows=(), flash_on=True, blank_rows=(), phase="fall", last=None)
    draws one frame. Returns (last_move_info, done).
    """
    rot, col = divmod(int(action), env.cols)
    shape = ROTATIONS[env.piece][rot]
    pid = env.piece + 1
    before = env.grid()
    locked, _, lines, land_row, full_rows = simulate_placement(env, action)

    # 1) piece falls from the top to its landing row
    n_frames = min(land_row + 1, 12)
    for r in sorted(set(np.linspace(0, land_row, n_frames).astype(int).tolist())):
        show(before, overlay=(shape, r, col, pid), phase="fall", last=last_prev)
        time.sleep(delay)

    # 2) apply the move in the env (this is the real game step)
    _, reward, done, info, _ = env.step(action)
    last = {"piece": info["piece"], "components": info["reward_components"], "total": reward}

    # 3) line clear: flash, remove the rows, then the stack drops
    if lines:
        for k in range(3):
            show(locked, flash_rows=full_rows, flash_on=(k % 2 == 0), phase="landed", last=last)
            time.sleep(delay * 1.5)
        show(locked, blank_rows=full_rows, phase="landed", last=last)
        time.sleep(delay * 1.5)

    show(env.grid(), phase="landed", last=last)
    time.sleep(delay)
    return last, done


# ----------------------------------------------------------------------------
# HTML rendering
# ----------------------------------------------------------------------------
def _cell(v: int, flash: bool = False) -> str:
    if flash:
        return '<div class="cell cell-flash"></div>'
    if v == 0:
        return '<div class="cell cell-empty"></div>'
    light, dark = PIECE_COLORS[v]
    return (
        f'<div class="cell cell-fill" style="background:linear-gradient(145deg,{light},{dark});'
        f"box-shadow:inset 2px 2px 3px rgba(255,255,255,0.45),"
        f'inset -3px -3px 4px rgba(0,0,0,0.35),0 0 8px {dark}88"></div>'
    )


def render_board(grid, overlay=None, flash_rows=(), flash_on=True, blank_rows=(), banner=None) -> str:
    g = np.array(grid, dtype=int)
    if overlay is not None:
        shape, row, col, pid = overlay
        h, w = shape.shape
        g[row : row + h, col : col + w][shape > 0] = pid
    for r in blank_rows:
        g[r, :] = 0
    flash_set = set(int(r) for r in flash_rows) if flash_on else set()
    cells = "".join(
        _cell(int(g[r, c]), flash=(r in flash_set and g[r, c] > 0))
        for r in range(g.shape[0])
        for c in range(g.shape[1])
    )
    banner_html = f'<div class="banner">{banner}</div>' if banner else ""
    return f'<div class="board-box"><div class="board">{cells}</div>{banner_html}</div>'


def render_next(piece_idx: int) -> str:
    shape = ROTATIONS[piece_idx][0]
    h, w = shape.shape
    r0, c0 = (4 - h) // 2, (4 - w) // 2
    cells = []
    for r in range(4):
        for c in range(4):
            inside = r0 <= r < r0 + h and c0 <= c < c0 + w and shape[r - r0, c - c0] > 0
            if inside:
                light, dark = PIECE_COLORS[piece_idx + 1]
                cells.append(
                    f'<div class="mini cell-fill" style="background:linear-gradient(145deg,{light},{dark});'
                    f'box-shadow:inset 1px 1px 2px rgba(255,255,255,0.45),0 0 6px {dark}88"></div>'
                )
            else:
                cells.append('<div class="mini"></div>')
    return (
        f'<div class="card"><div class="card-label">Next piece ({PIECE_NAMES[piece_idx]})</div>'
        f'<div class="next-grid">{"".join(cells)}</div></div>'
    )


def _stat(label, value, extra=""):
    return (
        f'<div class="card {extra}"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div></div>'
    )


def _signed(x: float) -> str:
    cls = "pos" if x > 0 else ("neg" if x < 0 else "")
    return f'<span class="{cls}">{x:+.2f}</span>'


def render_last_move(last) -> str:
    if not last:
        body = '<div class="card-small">waiting for the first move</div>'
    else:
        r1, r2, r3, r4 = last["components"]
        body = (
            f'<div class="card-small">piece {last["piece"]} &nbsp; reward {_signed(last["total"])}</div>'
            f'<div class="card-small">r1 {r1:+.1f} · r2 {r2:+.0f} · r3 {r3:+.0f} · r4 {r4:+.0f}</div>'
        )
    return f'<div class="card"><div class="card-label">Last move</div>{body}</div>'


def render_scene(env, grid, history, last, phase="fall", banner=None, **board_kwargs) -> str:
    f = board_features(env.board)
    # NEXT box: while a piece is falling it shows the piece after it; once it landed,
    # the piece that is about to drop next.
    upcoming = peek_next(env) if phase == "fall" else env.piece
    stats = (
        '<div class="stats">'
        + _stat("Pieces", env.pieces_placed)
        + _stat("Lines", env.total_lines)
        + _stat("Reward", f"{env.total_reward:.1f}")
        + _stat("Holes", f["holes"])
        + _stat("Height", f["max_height"])
        + _stat("Bumpiness", f["bumpiness"])
        + "</div>"
    )
    if history:
        games = len(history)
        avg_p = np.mean([h[0] for h in history])
        avg_l = np.mean([h[1] for h in history])
        hist_body = f'<div class="card-small">{games} games · avg {avg_p:.0f} pieces · avg {avg_l:.1f} lines</div>'
    else:
        hist_body = '<div class="card-small">no finished games yet</div>'
    hist = f'<div class="card"><div class="card-label">History</div>{hist_body}</div>'
    panel = f'<div class="panel">{render_next(upcoming)}{stats}{render_last_move(last)}{hist}</div>'
    board = render_board(grid, banner=banner, **board_kwargs)
    return f'<div class="viz-wrap">{board}{panel}</div>'


# ----------------------------------------------------------------------------
# Streamlit app
# ----------------------------------------------------------------------------
def main():
    import streamlit as st

    st.set_page_config(
        page_title="Tetris RL Demo", page_icon="🧱", layout="centered", initial_sidebar_state="expanded"
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="viz-title">Tetris RL Demo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="viz-sub">Live simulation · pieces fall, lines clear, reward updates</div>',
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("run", True)
    with st.sidebar:
        st.header("Demo controls")
        st.toggle("Run simulation", key="run")
        agent = st.radio("Agent", ["Greedy (best immediate reward)", "Random"])
        preset = st.radio("Reward", ["Baseline (R = r1)", "Multi-objective (default)", "Custom"])
        if preset.startswith("Baseline"):
            weights = baseline_weights()
        elif preset.startswith("Multi"):
            weights = RewardWeights()
        else:
            weights = RewardWeights(
                w1=st.slider("w1 line clears", 0.0, 5.0, 1.0, 0.1),
                w2=st.slider("w2 new holes", 0.0, 3.0, 0.5, 0.1),
                w3=st.slider("w3 height / bumpiness", 0.0, 3.0, 0.2, 0.1),
                w4=st.slider("w4 survival", 0.0, 2.0, 0.1, 0.05),
            )
        speed = st.slider("Speed", 1, 10, 5)
        restart = st.button("New game")

    cfg = (agent, (weights.w1, weights.w2, weights.w3, weights.w4))
    if restart or st.session_state.get("cfg") != cfg or "env" not in st.session_state:
        st.session_state.cfg = cfg
        env = TetrisPlacementEnv(weights=weights)
        env.reset()
        st.session_state.env = env
        st.session_state.history = []
        st.session_state.last = None
        st.session_state.rng = np.random.default_rng()

    env = st.session_state.env
    rng = st.session_state.rng
    box = st.empty()

    def show(grid, phase="fall", banner=None, last=None, **kw):
        html = render_scene(env, grid, st.session_state.history, last, phase=phase, banner=banner, **kw)
        box.markdown(html, unsafe_allow_html=True)

    if not st.session_state.run:
        show(env.grid(), phase="landed", last=st.session_state.last)
        st.caption("Paused. Turn on 'Run simulation' in the sidebar.")
        return

    delay = 0.30 / speed
    while True:  # a widget change in the sidebar stops this and reruns the script
        if agent.startswith("Greedy"):
            action = greedy_action(env, env.weights, rng)
        else:
            action = int(rng.choice(np.flatnonzero(env.mask)))
        last, done = play_one_piece(env, action, show, delay, st.session_state.last)
        st.session_state.last = last
        if done:
            show(env.grid(), phase="landed", banner="GAME OVER", last=last)
            time.sleep(1.5)
            st.session_state.history.append((env.pieces_placed, env.total_lines))
            env.reset()
            st.session_state.last = None


if __name__ == "__main__":
    main()
