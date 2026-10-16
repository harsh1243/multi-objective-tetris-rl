import numpy as np
import streamlit as st

ROWS, COLS = 20, 10
CELL = 28  # pixel size of one cell

# 0 = empty, 1..5 = piece color ids (extend to 7 later)
COLORS = {
    0: "#1e1e2e",
    1: "#e63946",
    2: "#2a9d8f",
    3: "#f4a261",
    4: "#4361ee",
    5: "#b5179e",
}


def render_board(grid: np.ndarray) -> str:
    """Pure display: takes a (ROWS, COLS) int grid, returns HTML. No game logic here."""
    cells = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = COLORS[int(grid[r, c])]
            cells.append(
                f'<div style="width:{CELL}px;height:{CELL}px;background:{color};'
                f'border:1px solid #11111b;box-sizing:border-box;border-radius:3px"></div>'
            )
    return (
        f'<div style="display:grid;grid-template-columns:repeat({grid.shape[1]},{CELL}px);'
        f'width:{CELL * grid.shape[1]}px;background:#11111b;padding:4px;border-radius:6px">'
        + "".join(cells)
        + "</div>"
    )


def render_next_piece(piece_id):
    """Placeholder: fill this in later when the next-piece panel is added."""
    return '<div style="color:#888">next piece (coming soon)</div>'


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

st.title("Tetris RL Visualizer")
left, right = st.columns([1, 1])


@st.fragment(run_every=1)  # re-runs only this block every second
def board_area():
    on_new_piece(st.session_state)
    with left:
        st.markdown(render_board(st.session_state.grid), unsafe_allow_html=True)
    with right:
        st.markdown(render_next_piece(None), unsafe_allow_html=True)


board_area()
