# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python port of examples/cpp/mfma/mfma_layout.cpp.

Encodes AMD MFMA layouts as CuTe-style thread-value (TV) layouts using
mosaic's hierarchical (shape, stride) tuples, then verifies the
encoding three ways:

  1. Bijectivity — every operand cell is touched exactly once by the
     64 lanes x VALS_PER_LANE VGPRs of the wavefront.
  2. A worked-example walk: where does ``D_tv(lane=5, vid=2)`` land in
     the 32x32 D tile?
  3. Ground-truth check against AMD's matrix instruction calculator.
     When the CSV exports are present under
     ``examples/cpp/mfma/matrix_calculator_layouts/CDNA3_v_mfma_f32_32x32x8_f16/``
     every (lane, vid) -> operand cell mapping is compared with the
     layout's output. Those dumps are not committed; the check is
     skipped when the directory is absent.

This is a 1:1 port of the cpp logic — same TV layouts, same
bijectivity / ground-truth checks, same B-axis swap caveat.
"""

from __future__ import annotations

import pathlib
import re

from pymosaic.layout import Layout, idx2crd

WAVE = 64

# The cpp example reads the CSVs from a path templated against
# ``MFMA_DATA_DIR``. The Python port locates them by walking up from
# this script to the repo root and into ``examples/cpp/mfma/``.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_CSV_DIR = (
    _REPO_ROOT
    / "examples"
    / "cpp"
    / "mfma"
    / "matrix_calculator_layouts"
    / "CDNA3_v_mfma_f32_32x32x8_f16"
)


_CELL_RE = re.compile(r"^[ABCDK]\[(\d+)\]\[(\d+)\](?:\.B(\d+))?$", re.IGNORECASE)


def parse_calc_csv(path: pathlib.Path) -> list[tuple[int, int, int, int, int]]:
    """Parse one of AMD's ``--matrix-layout --csv`` outputs.

    Returns a list of ``(tid, vid, i, j, block)`` ground-truth entries.

    Header is three lines (Architecture / Instruction / column header).
    Body is ``lane,cell_v0,cell_v1,...`` where each cell is ``X[i][j]``
    or ``X[i][j].B<n>`` (``X`` in {A, B, C, D, K}). Empty / non-matching
    cells are skipped (the calculator emits ``""`` for inactive vid
    slots).
    """
    out: list[tuple[int, int, int, int, int]] = []
    with path.open() as f:
        lines = f.read().splitlines()
    body = lines[3:]  # skip the three header lines
    for line in body:
        if not line.strip():
            continue
        fields = [tok.strip() for tok in line.split(",")]
        if not fields:
            continue
        try:
            tid = int(fields[0])
        except ValueError:
            continue
        for vid, cell in enumerate(fields[1:]):
            m = _CELL_RE.match(cell)
            if not m:
                continue
            i = int(m.group(1))
            j = int(m.group(2))
            block = int(m.group(3)) if m.group(3) else 0
            out.append((tid, vid, i, j, block))
    return out


def check_bijection(tv: Layout, rows: int, cols: int, vals: int, name: str) -> None:
    """Assert every matrix cell is hit exactly once."""
    hit = [0] * (rows * cols)
    for tid in range(WAVE):
        for vid in range(vals):
            f = tv[tid, vid]
            assert 0 <= f < rows * cols, f"{name}: out-of-range flat={f}"
            hit[f] += 1
    for f, count in enumerate(hit):
        assert count == 1, f"{name}: cell flat={f} hit {count}x (expected 1)"
    print(
        f"  {name}: {rows * cols} cells covered exactly once by {WAVE} lanes x {vals} VGPRs."
    )


def verify_against_calc(
    tv: Layout, rows: int, cols: int, csv_path: pathlib.Path, swap: bool, name: str
) -> None:
    """Check that TV's (tid, vid) -> operand cell matches AMD's calculator."""
    cells = parse_calc_csv(csv_path)
    assert cells, f"{name}: calculator CSV produced no cells ({csv_path})"
    mismatches = 0
    shown = 0
    for tid, vid, calc_i, calc_j, block in cells:
        # B is canonicalised N-major in our TV layout but k-major in the
        # calculator. swap=True flips the indices for B only.
        our_i = calc_j if swap else calc_i
        our_j = calc_i if swap else calc_j
        expected = block * (rows * cols) + our_j * rows + our_i
        got = tv[tid, vid]
        if got != expected:
            mismatches += 1
            if shown < 5:
                print(
                    f"  MISMATCH {name}(tid={tid}, vid={vid}): TV gave {got}, "
                    f"calc says {name}[{calc_i}][{calc_j}].B{block} (expected {expected})"
                )
                shown += 1
    suffix = "" if mismatches == 0 else f" ({mismatches} mismatches)"
    note = " (all match)" if mismatches == 0 else ""
    print(f"  {name}: {len(cells)} calculator entries verified{note}{suffix}")
    assert mismatches == 0, f"{name}: TV layout disagrees with calculator"


def main() -> int:
    # -------- A: 32 (M) x 8 (K), col-major flat --------
    #   tid: (32 lanes per col, 2 col-groups), strides (1, 128)
    #   vid: 4 VGPRs along K,                  stride 32
    A_tv = Layout(shape=((32, 2), 4), stride=((1, 128), 32))
    A_rows, A_cols, A_vals = 32, 8, 4

    # -------- B: 32 (N) x 8 (K), col-major flat --------
    B_tv = Layout(shape=((32, 2), 4), stride=((1, 128), 32))
    B_rows, B_cols, B_vals = 32, 8, 4

    # -------- D: 32 (M) x 32 (N), col-major flat --------
    #   tid: (32 lanes per col-block, 2 col-blocks), strides (32, 4)
    #   vid: (4 row-groups, 4 rows per group),       strides (1, 8)
    D_tv = Layout(shape=((32, 2), (4, 4)), stride=((32, 4), (1, 8)))
    D_rows, D_cols, D_vals = 32, 32, 16

    print("v_mfma_f32_32x32x8_f16 — TV layouts (tid, vid) -> col-major flat")
    print(f"  A_tv = {A_tv}   (rows={A_rows}, cols={A_cols})")
    print(f"  B_tv = {B_tv}   (rows={B_rows}, cols={B_cols})")
    print(f"  D_tv = {D_tv}   (rows={D_rows}, cols={D_cols})\n")

    # -------- Worked example: where does (lane=5, vid=2) land in D? --------
    # L[tid, vid] accepts scalar tid/vid; mosaic internally idx2crd's
    # each scalar against its sub-shape, then inner-products with the
    # sub-stride. See include/mosaic/stride.hpp:79-107.
    lane, vid = 5, 2
    flat = D_tv[lane, vid]
    # Decode the col-major flat into (row, col). Wrap the (rows, cols)
    # in a tiny Layout solely so we can reuse the JIT-bound idx2crd.
    decoder = Layout(shape=(D_rows, D_cols), stride=(1, D_rows))
    rc = idx2crd(flat, decoder)
    print(
        f"Worked example: D_tv[lane={lane}, vid={vid}] = {flat}  ->  "
        f"(row={rc[0]}, col={rc[1]})\n"
    )

    # -------- Bijectivity check --------
    print("Bijectivity:")
    check_bijection(A_tv, A_rows, A_cols, A_vals, "A")
    check_bijection(B_tv, B_rows, B_cols, B_vals, "B")
    check_bijection(D_tv, D_rows, D_cols, D_vals, "D")

    # -------- Ground-truth check vs. AMD's matrix calculator --------
    if not _CSV_DIR.is_dir():
        print(f"\n(skip) calculator CSV directory not found at {_CSV_DIR}")
        return 0
    print("\nGround-truth check vs. AMD matrix calculator:")
    verify_against_calc(
        A_tv,
        A_rows,
        A_cols,
        _CSV_DIR / "CDNA3_v_mfma_f32_32x32x8_f16_A_matrix_layout.csv",
        swap=False,
        name="A",
    )
    verify_against_calc(
        B_tv,
        B_rows,
        B_cols,
        _CSV_DIR / "CDNA3_v_mfma_f32_32x32x8_f16_B_matrix_layout.csv",
        swap=True,
        name="B",
    )
    verify_against_calc(
        D_tv,
        D_rows,
        D_cols,
        _CSV_DIR / "CDNA3_v_mfma_f32_32x32x8_f16_D_matrix_layout.csv",
        swap=False,
        name="D",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
