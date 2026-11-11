# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pretty-printing for :class:`pymosaic.layout.Layout`.

Renders a layout as a CuTe-style 2D grid of *offsets*: cell ``(i, j)`` is
the offset the layout maps that coordinate to. This is the most intuitive
way to read a layout — you see the actual memory pattern, not just the
``shape:stride`` signature.

Two renderers:

- **flat grid** (default for layouts with no nested mode): a plain ASCII
  ``+---+`` grid. Rank handling:
    - rank 0 (scalar): the single offset.
    - rank 1 ``(N,)``: one row of ``N`` offsets.
    - rank 2 ``(M, N)``: an ``M x N`` grid.
    - rank >= 3: first mode as rows, the rest flattened (column-major)
      across columns; column headers show the remaining coordinate tuple.

- **hierarchical grid** (used automatically for a rank-2 layout where a
  top-level mode is itself a tuple, e.g. ``((2, 2), 3)``): a box-drawing
  grid whose axes are labelled by the *nested coordinate* and whose
  sub-block boundaries are drawn with a heavier (double) line — so the
  tile structure is visible, the way ``basic_tiling`` shows tiles. Force
  on/off with ``nested=True`` / ``nested=False``.

Everything here is pure Python and operates on the public ``Layout``
surface (``shape``/``stride``/``len``/``cosize``/indexing), so it carries
no extra dependencies and triggers exactly the same JIT compile a normal
evaluation would.
"""

from __future__ import annotations

import sys
from typing import Any

from . import _render

__all__ = ["pretty", "pretty_print"]

# Default truncation guards so ``pretty(Layout.row_major(4096, 4096))``
# doesn't try to format 16M cells. Override per call.
_MAX_ROWS = 64
_MAX_COLS = 64


def _colex_decompose(flat: int, sizes: list[int]) -> list[int]:
    """Decompose ``flat`` into per-mode coordinates, column-major
    (leftmost mode varies fastest) — matching mosaic's flat linearisation.
    """
    coords = []
    for s in sizes:
        if s <= 0:
            coords.append(0)
            continue
        coords.append(flat % s)
        flat //= s
    return coords


# ---------------------------------------------------------------------------
# Hierarchical axis decomposition
# ---------------------------------------------------------------------------


def _build_axis(mode: Any) -> tuple[list[Any], list[int]]:
    """Decompose one axis ``mode`` (a possibly-nested int-tuple) into the
    ordered list of per-position coordinate labels plus the "block level"
    of each boundary between consecutive positions.

    Ordering is mosaic's column-major (leftmost sub-mode varies fastest).

    Returns ``(labels, seps)`` where:
      - ``labels[k]`` is the coordinate of position ``k`` (an ``int`` for a
        flat mode, a nested tuple mirroring ``mode`` for a hierarchical one).
      - ``seps[k]`` (length ``len(labels) - 1``) is an integer; ``0`` is the
        innermost (minor) boundary, larger values mark more major block
        boundaries (where an outer sub-mode rolls over).
    """
    if not isinstance(mode, tuple):
        n = int(mode)
        return list(range(n)), [0] * max(0, n - 1)

    children = [_build_axis(m) for m in mode]
    child_labels = [c[0] for c in children]
    child_seps = [c[1] for c in children]
    sizes = [len(cl) for cl in child_labels]

    total = 1
    for s in sizes:
        total *= s

    labels: list[Any] = []
    seps: list[int] = []
    prev: list[int] | None = None
    for flat in range(total):
        idx = []
        f = flat
        for s in sizes:
            idx.append(f % s if s else 0)
            f //= s if s else 1
        labels.append(tuple(child_labels[d][idx[d]] for d in range(len(sizes))))
        if prev is not None:
            # Find the carry point: the lowest dimension that advanced
            # (all dimensions below it rolled max -> 0).
            d = 0
            while d < len(sizes) and prev[d] == sizes[d] - 1 and idx[d] == 0:
                d += 1
            if d == 0:
                # Only the fastest child advanced one internal step.
                seps.append(child_seps[0][idx[0] - 1] if sizes[0] > 1 else 0)
            else:
                # Children 0..d-1 rolled over and child d advanced: this is a
                # boundary "above" everything inside the rolled children.
                inner_max = 0
                for dd in range(d + 1):
                    if child_seps[dd]:
                        inner_max = max(inner_max, max(child_seps[dd]))
                seps.append(inner_max + 1)
        prev = idx
    return labels, seps


def _fmt_label(x: Any) -> str:
    """Render a coordinate label: ``int`` as-is, nested tuple as ``(a,b)``."""
    if isinstance(x, tuple):
        return "(" + ",".join(_fmt_label(e) for e in x) + ")"
    return str(x)


# Box-drawing junction glyphs keyed by (north, south, east, west) line
# weights, where 0 = absent, 1 = single, 2 = double. A vertical line passing
# through a vertex has equal north/south weight (or one side absent at a
# border); a horizontal line has equal east/west weight.
_JUNC = {
    # corners (outer border is always single)
    (0, 1, 1, 0): "\u250c",
    (0, 1, 0, 1): "\u2510",
    (1, 0, 1, 0): "\u2514",
    (1, 0, 0, 1): "\u2518",
    # top edge T-junctions  (north absent)
    (0, 1, 1, 1): "\u252c",
    (0, 2, 1, 1): "\u2565",
    # bottom edge T-junctions (south absent)
    (1, 0, 1, 1): "\u2534",
    (2, 0, 1, 1): "\u2568",
    # left edge T-junctions  (west absent)
    (1, 1, 1, 0): "\u251c",
    (1, 1, 2, 0): "\u255e",
    # right edge T-junctions (east absent)
    (1, 1, 0, 1): "\u2524",
    (1, 1, 0, 2): "\u2561",
    # interior crossings
    (1, 1, 1, 1): "\u253c",
    (1, 1, 2, 2): "\u256a",
    (2, 2, 1, 1): "\u256b",
    (2, 2, 2, 2): "\u256c",
}


def _hier_grid(
    layout: Any,
    m0: Any,
    m1: Any,
    title: str,
    max_rows: int,
    max_cols: int,
) -> str:
    """Render a rank-2 layout as a box-drawing grid with nested-coordinate
    axis labels and heavier lines at sub-block boundaries."""
    row_labels, row_seps = _build_axis(m0)
    col_labels, col_seps = _build_axis(m1)

    rows_total, cols_total = len(row_labels), len(col_labels)
    show_rows = min(rows_total, max_rows)
    show_cols = min(cols_total, max_cols)
    truncated = show_rows < rows_total or show_cols < cols_total

    row_labels = row_labels[:show_rows]
    col_labels = col_labels[:show_cols]
    row_seps = row_seps[: show_rows - 1]
    col_seps = col_seps[: show_cols - 1]

    cells = [
        [int(layout[(row_labels[r], col_labels[c])]) for c in range(show_cols)]
        for r in range(show_rows)
    ]

    col_hdr = [_fmt_label(c) for c in col_labels]
    row_hdr = [_fmt_label(r) for r in row_labels]

    w = 1
    for r in range(show_rows):
        for c in range(show_cols):
            w = max(w, len(str(cells[r][c])))
    for h in col_hdr:
        w = max(w, len(h))
    lab_w = max((len(h) for h in row_hdr), default=0)

    # Boundary weights: outer border single; interior double iff block level > 0.
    col_w = [1] + [2 if s > 0 else 1 for s in col_seps] + [1]
    row_w = [1] + [2 if s > 0 else 1 for s in row_seps] + [1]

    def line(prefix: str, units: list[str], seps: list[str]) -> str:
        s = prefix
        for i in range(len(units)):
            s += seps[i] + units[i]
        return s + seps[len(units)]

    def sep_row(ri: int) -> str:
        juncs = []
        for c in range(show_cols + 1):
            v = col_w[c]
            n = v if ri > 0 else 0
            s_ = v if ri < show_rows else 0
            h = row_w[ri]
            e = h if c < show_cols else 0
            west = h if c > 0 else 0
            juncs.append(_JUNC.get((n, s_, e, west), "\u253c"))
        fill = ("\u2550" if row_w[ri] == 2 else "\u2500") * (w + 2)
        return line(f"{'':>{lab_w}} ", [fill] * show_cols, juncs)

    lines = [title]
    lines.append(
        line(
            f"{'':>{lab_w}} ",
            [h.center(w + 2) for h in col_hdr],
            [" "] * (show_cols + 1),
        )
    )
    lines.append(sep_row(0))
    data_seps = ["\u2551" if cw == 2 else "\u2502" for cw in col_w]
    for r in range(show_rows):
        units = [f" {cells[r][c]:>{w}} " for c in range(show_cols)]
        lines.append(line(f"{row_hdr[r]:>{lab_w}} ", units, data_seps))
        lines.append(sep_row(r + 1))

    if truncated:
        lines.append(
            f"  (showing {show_rows}x{show_cols} of {rows_total}x{cols_total})"
        )

    return "\n".join(line_.rstrip() for line_ in lines)


def pretty(
    layout: Any,
    *,
    max_rows: int = _MAX_ROWS,
    max_cols: int = _MAX_COLS,
    nested: bool | None = None,
) -> str:
    """Render ``layout`` as a header line plus an offset grid; return the string.

    Args:
        layout: a :class:`pymosaic.layout.Layout` (or anything exposing the
            same ``shape`` / ``__len__`` / ``cosize`` / ``__getitem__`` API).
        max_rows: cap on rows rendered (excess is elided with a note).
        max_cols: cap on columns rendered (excess is elided with a note).
        nested: hierarchical (block) rendering for rank-2 layouts. ``None``
            (default) auto-enables it when a top-level mode is itself a tuple;
            ``True`` forces it; ``False`` forces the flat grid.
    """
    shape = layout.shape
    size = len(layout)
    cosize = layout.cosize
    title = f"{layout!s}   size={size}  cosize={cosize}"

    # Rank 0 — scalar layout (shape is not a tuple).
    if not isinstance(shape, tuple):
        return f"{title}\n(scalar layout)"

    rank = len(shape)
    if rank == 0 or size == 0:
        return f"{title}\n(empty layout)"

    # Hierarchical (block) rendering for rank-2 layouts with a nested mode.
    if rank == 2 and nested is not False:
        m0, m1 = shape
        if nested is True or isinstance(m0, tuple) or isinstance(m1, tuple):
            return _hier_grid(layout, m0, m1, title, max_rows, max_cols)

    # Per top-level-mode domain sizes.
    mode_sizes = [_render.size(m) for m in shape]

    # Decide the row/column split for the flat grid.
    if rank == 1:
        # A vector reads best as a single row.
        n = mode_sizes[0]
        rows, cols = 1, n
        row_header_for = lambda i: ""  # noqa: E731 — no row labels for a 1-row grid
        col_header_for = lambda j: str(j)  # noqa: E731
        coord_for = lambda i, j: (j,)  # noqa: E731
    else:
        rows = mode_sizes[0]
        rest_sizes = mode_sizes[1:]
        cols = 1
        for s in rest_sizes:
            cols *= s
        row_header_for = lambda i: str(i)  # noqa: E731
        if rank == 2:
            col_header_for = lambda j: str(j)  # noqa: E731
        else:
            col_header_for = (
                lambda j: "("
                + ",".join(  # noqa: E731
                    str(c) for c in _colex_decompose(j, rest_sizes)
                )
                + ")"
            )
        coord_for = lambda i, j: (i, *_colex_decompose(j, rest_sizes))  # noqa: E731

    show_rows = min(rows, max_rows)
    show_cols = min(cols, max_cols)
    trunc_rows = rows > show_rows
    trunc_cols = cols > show_cols

    # Evaluate the offsets we'll display.
    matrix: list[list[str]] = []
    for i in range(show_rows):
        row_cells: list[str] = []
        for j in range(show_cols):
            off = int(layout[coord_for(i, j)])
            row_cells.append(str(off))
        matrix.append(row_cells)

    headers = [col_header_for(j) for j in range(show_cols)]
    row_labels = [row_header_for(i) for i in range(show_rows)]

    # Column inner width: widest of any cell or header.
    w = 1
    for h in headers:
        w = max(w, len(h))
    for row_cells in matrix:
        for c in row_cells:
            w = max(w, len(c))

    lab_w = max((len(x) for x in row_labels), default=0)

    def row(label: str, cells: list[str], pipe: str) -> str:
        seg = pipe.join(f" {c:>{w}} " for c in cells)
        return f"{label:>{lab_w}} {pipe}{seg}{pipe}"

    def sep() -> str:
        seg = "+".join("-" * (w + 2) for _ in range(show_cols))
        return f"{'':>{lab_w}} +{seg}+"

    lines = [title]

    header_line = row("", headers, " ")
    if trunc_cols:
        header_line += "  ..."
    lines.append(header_line)

    lines.append(sep())
    for i in range(show_rows):
        data_line = row(row_labels[i], matrix[i], "|")
        if trunc_cols:
            data_line += "  ..."
        lines.append(data_line)
        lines.append(sep())

    if trunc_rows:
        lines.append(f"{'...':>{lab_w}}")

    if trunc_rows or trunc_cols:
        lines.append(f"  (showing {show_rows}x{show_cols} of {rows}x{cols})")

    return "\n".join(line.rstrip() for line in lines)


def pretty_print(
    layout: Any,
    *,
    file: Any = None,
    max_rows: int = _MAX_ROWS,
    max_cols: int = _MAX_COLS,
    nested: bool | None = None,
) -> None:
    """Render ``layout`` with :func:`pretty` and write it to ``file``
    (``sys.stdout`` by default), followed by a newline."""
    out = file or sys.stdout
    out.write(
        pretty(layout, max_rows=max_rows, max_cols=max_cols, nested=nested) + "\n"
    )
