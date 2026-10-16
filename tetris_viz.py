import numpy as np
import streamlit as st

st.set_page_config(page_title="Tetris RL Visualizer", page_icon="🧱", layout="centered")

ROWS, COLS = 20, 10
CELL = 32  # pixel size of one board cell

# color id -> (light, dark) used for the block gradient. 0 = empty. Extend to 7 later.
COLORS = {
    1: ("#ff6b81", "#c9283f"),  # red
    2: ("#3ddbc4", "#12877a"),  # teal
    3: ("#ffc15e", "#d9822b"),  # amber
    4: ("#6d8bff", "#2f45c9"),  # blue
    5: ("#d55bff", "#8a1fb3"),  # purple
}

CSS = """
<style>
.stApp { background: radial-gradient(circle at 50% 0%, #1a1f3a 0%, #0b0e1a 60%); }
.viz-title { text-align:center; font-size:2.4rem; font-weight:800; letter-spacing:1px;
  background:linear-gradient(90deg,#6d8bff,#d55bff,#ff6b81); -webkit-background-clip:text;
  -webkit-text-fill-color:transparent; margin:0.2rem 0 0.2rem 0; }
.viz-sub { text-align:center; color:#7c86b8; margin-bottom:1.4rem; font-size:0.95rem; }
.viz-wrap { display:flex; gap:28px; justify-content:center; align-items:flex-start; }
.board { display:grid; padding:10px; background:#0a0d18; border:2px solid #2b3260;
  border-radius:14px; box-shadow:0 0 30px rgba(109,139,255,0.18), inset 0 0 20px rgba(0,0,0,0.6); }
.cell-empty { background:#141829; box-shadow: inset 0 0 0 1px #1d2340; border-radius:3px; }
.cell-fill { border-radius:5px; }
.panel { width:190px; display:flex; flex-direction:column; gap:14px; }
.card { background:#10142a; border:1px solid #262d57; border-radius:12px; padding:12px 14px; }
.card-label { color:#7c86b8; font-size:0.72rem; letter-spacing:1.5px; text-transform:uppercase; }
.card-value { color:#f2f4ff; font-size:1.6rem; font-weight:700; margin-top:2px; }
.next-grid { display:grid; grid-template-columns:repeat(4,22px); gap:0; justify-content:center; margin-top:8px; }
</style>
"""


def cell_html(v: int) -> str:
    if v == 0:
        return f'<div class="cell-empty" style="width:{CELL}px;height:{CELL}px"></div>'
    light, dark = COLORS[v]
    return (
        f'<div class="cell-fill" style="width:{CELL}px;height:{CELL}px;'
        f'background:linear-gradient(145deg,{light},{dark});'
        f'box-shadow:inset 2px 2px 3px rgba(255,255,255,0.45),'
        f'inset -3px -3px 4px rgba(0,0,0,0.35),0 0 8px {dark}88"></div>'
    )


def render_board(grid: np.ndarray) -> str:
    """Pure display: takes a (ROWS, COLS) int grid, returns HTML. No game logic here."""
    cells = "".join(cell_html(int(v)) for v in grid.flatten())
    return (
        f'<div class="board" style="grid-template-columns:repeat({grid.shape[1]},{CELL}px)">'
        f"{cells}</div>"
    )


def board_metrics(grid: np.ndarray) -> dict:
    """Same features the agent will use: heights, holes, bumpiness (computed from the grid)."""
    filled = grid > 0
    heights = np.where(filled.any(axis=0), ROWS - filled.argmax(axis=0), 0)
    holes = 0
    for c in range(grid.shape[1]):
        col = filled[:, c]
        if col.any():
            top = col.argmax()
            holes += int((~col[top:]).sum())
    bump = int(np.abs(np.diff(heights)).sum())
    return {"max_height": int(heights.max()), "holes": holes, "bumpiness": bump}


def render_next_piece(piece_id) -> str:
    """Placeholder 4x4 box. Later: draw the real next piece here."""
    cells = "".join(
        '<div class="cell-empty" style="width:22px;height:22px"></div>' for _ in range(16)
    )
    return (
        '<div class="card"><div class="card-label">Next piece</div>'
        f'<div class="next-grid">{cells}</div></div>'
    )


def stat_card(label: str, value) -> str:
    return (
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div></div>'
    )


def on_new_piece(state):
    """Hook called once per tick. Later: step env, apply agent action, update state['grid']."""
    pass


def make_demo_grid():
    rng = np.random.default_rng(0)
    grid = np.zeros((ROWS, COLS), dtype=int)
    for r in range(ROWS - 6, ROWS):  # fake settled blocks at the bottom
        for c in range(COLS):
            if rng.random() < 0.75:
                grid[r, c] = rng.integers(1, 6)
    return grid


if "grid" not in st.session_state:
    st.session_state.grid = make_demo_grid()
    st.session_state.score = 0
    st.session_state.lines = 0

st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="viz-title">Tetris RL Visualizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="viz-sub">Multi-objective PPO agent · live board view</div>',
    unsafe_allow_html=True,
)


@st.fragment(run_every=1)  # re-runs only this block every second
def board_area():
    on_new_piece(st.session_state)
    grid = st.session_state.grid
    m = board_metrics(grid)
    panel = (
        '<div class="panel">'
        + render_next_piece(None)
        + stat_card("Score", st.session_state.score)
        + stat_card("Lines", st.session_state.lines)
        + stat_card("Holes", m["holes"])
        + stat_card("Max height", m["max_height"])
        + stat_card("Bumpiness", m["bumpiness"])
        + "</div>"
    )
    st.markdown(
        f'<div class="viz-wrap">{render_board(grid)}{panel}</div>',
        unsafe_allow_html=True,
    )


board_area()
