# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from collections import Counter
import operator

import pytest
from rocke.helpers.loads import CoalescedTileLoader


class IntegerBuilder:
    """Interpret the visitor's index operations, not its geometry checks."""

    const_i32 = staticmethod(int)
    add = staticmethod(operator.add)
    mul = staticmethod(operator.mul)
    div = staticmethod(operator.floordiv)
    mod = staticmethod(operator.mod)


@pytest.mark.parametrize("grid", [None, (3, 4, 1)], ids=["default", "prebound"])
@pytest.mark.parametrize(
    "rows,cols,width,threads,axis,diagnostic",
    [
        (4, 6, 4, 3, "col", "must divide the vector axis"),
        (6, 4, 4, 3, "row", "must divide the vector axis"),
        (4, 2, 4, 2, "col", "must divide the vector axis"),
        (8, 8, 0, 4, "col", "load_vec and block_size must be positive"),
        (8, 8, 2, 0, "col", "load_vec and block_size must be positive"),
        (8, 0, 2, 4, "col", "tile dimensions must be positive"),
        (-8, 8, 2, 4, "row", "tile dimensions must be positive"),
        (8, 8, 2, 3, "col", "not divisible by block_size"),
        (65536, 65536, 1, 4, "col", "chunk geometry exceeds i32"),
    ],
)
def test_invalid_geometry_rejects_before_builder_or_callbacks(
    rows, cols, width, threads, axis, diagnostic, grid
):
    loader = CoalescedTileLoader(rows, cols, threads, width, vector_axis=axis)

    def unexpected(*_):
        pytest.fail("invalid geometry invoked a traversal callback")

    # None cannot emit even a constant; a premature builder call fails the test.
    with pytest.raises(ValueError, match=diagnostic):
        loader.visit_chunks(
            None, tid=0, emit=unexpected, grid=grid, before_decode=unexpected
        )


@pytest.mark.parametrize("method", ["load", "load_global", "plan_slots", "fetch"])
def test_other_loader_entry_points_validate_before_building_grid(method):
    loader = CoalescedTileLoader(8, 8, 4, 0)
    kwargs = {"tid": 0}
    if method != "plan_slots":
        kwargs.update(rsrc=object(), descriptor=None)
    if method == "load":
        kwargs["smem_dst"] = None
    with pytest.raises(ValueError, match="load_vec and block_size must be positive"):
        getattr(loader, method)(None, **kwargs)


@pytest.mark.parametrize("rows,cols,axis", [(5, 8, "col"), (8, 5, "row")])
def test_aligned_vector_axis_covers_each_element_once(rows, cols, axis):
    width, threads = 4, 5
    loader = CoalescedTileLoader(rows, cols, threads, width, vector_axis=axis)
    coords = Counter()

    def emit(row, col, _):
        for lane in range(width):
            coords[
                row + (lane if axis == "row" else 0),
                col + (lane if axis == "col" else 0),
            ] += 1

    for tid in range(threads):
        loader.visit_chunks(IntegerBuilder(), tid=tid, emit=emit)
    assert coords == Counter((row, col) for row in range(rows) for col in range(cols))
