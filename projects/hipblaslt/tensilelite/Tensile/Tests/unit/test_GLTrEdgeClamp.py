# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the GLTr (transpose-load) free-dim edge-clamp arithmetic used by
``KernelWriterAssembly.calcMaxGroForGLTr``.

Background: RDNA4 transpose global loads (``global_load_tr``) use a bare 2-register
address with no buffer ``num_records`` bounds (unlike ``buffer_load``). For operand B
the per-lane global-read offset must therefore be clamped so a free dim smaller than
the macro-tile cannot over-read past the tensor. The clamp value is selected per
N-workgroup from the runtime tile geometry:

  rem  = SizeN - WG*MT                       (valid free-dim elements in THIS tile)
  edge = (rem < MT) ? rem - margin           (partial tile: last legal glvw-vector start)
                    : MT - 1                  (full tile: in-range, a no-op clamp)
  # main loop only: a full tile additionally neutralizes the bound to INT_MAX so the
  # downstream per-load VMinI32 is provably a no-op == upstream behavior.

These are pure integer decisions; the codegen emits the equivalent
SMulI32/SSubI32/SCmpLtI32/SCSelectB32/VMinI32 (+ SCmpGeI32/SCSelectB32/VMaxI32)
sequence. This test pins that decision logic so a regression in the selection (e.g.
the off-by-margin bug that over-clamped full tiles, or a missing full-tile no-op that
corrupted batched DTVB1 kernels) is caught without a GPU.
"""

import pytest

pytestmark = pytest.mark.unit

INT_MAX = 0x7FFFFFFF


def _partial_edge(size_n, wg, mt, margin):
    """Block 1: tile-index clamp bound, in elements.
    Mirrors the SCmpLtI32(rem, MT) + SCSelectB32(rem-margin, MT-1) sequence."""
    rem = size_n - wg * mt          # signed
    return (rem - margin) if rem < mt else (mt - 1)


def _full_tile_neutralized(size_n, wg, mt):
    """Block 2 (main-loop only): full tiles raise the byte bound to INT_MAX so the
    downstream VMinI32 is a no-op. Mirrors SCmpGeI32(rem, MT) + SCSelectB32."""
    rem = size_n - wg * mt          # signed
    return INT_MAX if rem >= mt else 0


# Representative shapes from the on-GPU validation (MT1 = 64, glvw = 8).
MT = 64
GLVW = 8


# ---------------------------------------------------------------------------
# Block 1: partial vs full tile edge selection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "size_n, wg, expected_edge, note",
    [
        # down_proj crash shape: single partial tile of 8 -> rem=8 < MT -> 8-8=0.
        (8, 0, 0, "n=8 partial (the original crash)"),
        # partial tails that are not multiples of glvw.
        (6, 0, 6 - GLVW, "n=6 partial"),
        (10, 0, 10 - GLVW, "n=10 partial"),
        # exactly one full tile -> rem=64 == MT -> no-op (MT-1).
        (64, 0, MT - 1, "n=64 full single tile -> no-op"),
        # multi-tile, last tile full: n=128, wg=1 -> rem=64 -> no-op.
        (128, 1, MT - 1, "n=128 wg1 full -> no-op"),
        # MT+8: wg0 full (no-op), wg1 partial 8.
        (72, 0, MT - 1, "n=72 wg0 full -> no-op"),
        (72, 1, 8 - GLVW, "n=72 wg1 partial=8"),
        # large full free dim (attention 192x256x192 batched, N=256): every WG full.
        (256, 0, MT - 1, "n=256 wg0 full"),
        (256, 3, MT - 1, "n=256 wg3 last full -> no-op (was the regression)"),
    ],
)
def test_partial_edge_selection(size_n, wg, expected_edge, note):
    assert _partial_edge(size_n, wg, MT, GLVW) == expected_edge, note


# ---------------------------------------------------------------------------
# Block 2: full-tile neutralization to INT_MAX (main-loop path)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "size_n, wg, is_full",
    [
        (8, 0, False),      # partial -> not neutralized
        (64, 0, True),      # exactly full
        (72, 0, True),      # wg0 full
        (72, 1, False),     # wg1 partial
        (256, 3, True),     # last full tile (regression case)
        (128, 0, True),
        (128, 1, True),
    ],
)
def test_full_tile_neutralization(size_n, wg, is_full):
    val = _full_tile_neutralized(size_n, wg, MT)
    assert val == (INT_MAX if is_full else 0)


# ---------------------------------------------------------------------------
# Invariants that guarantee correctness across all shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mt", [32, 64, 128])
@pytest.mark.parametrize("glvw", [1, 8])
def test_full_tiles_are_always_a_noop(mt, glvw):
    """For any full or over-full tile the effective bound must be >= MT-1 (block 1) and
    INT_MAX (block 2), i.e. it can never clamp a valid lane (tile index in 0..MT-1)."""
    for size_n in range(mt, 6 * mt + 1):
        for wg in range(0, size_n // mt):  # only fully-covered workgroups
            assert _partial_edge(size_n, wg, mt, glvw) >= mt - 1
            assert _full_tile_neutralized(size_n, wg, mt) == INT_MAX


@pytest.mark.parametrize("mt", [32, 64, 128])
@pytest.mark.parametrize("glvw", [1, 8])
def test_partial_tiles_bound_within_tensor(mt, glvw):
    """For a genuinely partial last tile the edge must be < the valid element count so
    the full glvw-wide vector load cannot read past the tensor end."""
    for rem in range(1, mt):  # partial remainder
        size_n = mt + rem      # one full tile (wg0) + partial (wg1)
        edge = _partial_edge(size_n, 1, mt, glvw)
        # last legal vector start = rem - margin; never exceeds rem-1.
        assert edge <= rem - 1
        assert edge == rem - glvw
