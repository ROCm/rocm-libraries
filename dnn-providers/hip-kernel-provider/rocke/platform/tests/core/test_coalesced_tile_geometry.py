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


DIVISOR_ERROR = "CoalescedTileLoader: load_vec and block_size must be positive"
EXTENT_ERROR = "CoalescedTileLoader: tile dimensions must be positive"
VECTOR_ERROR = "CoalescedTileLoader: load_vec must divide the vector axis"
OVERFLOW_ERROR = "CoalescedTileLoader: chunk geometry exceeds i32"


@pytest.mark.parametrize("grid", [None, (3, 4, 1)], ids=["default", "prebound"])
@pytest.mark.parametrize(
    "rows,cols,width,threads,axis,diagnostic",
    [
        (4, 6, 4, 3, "col", VECTOR_ERROR),
        (6, 4, 4, 3, "row", VECTOR_ERROR),
        (4, 2, 4, 2, "col", VECTOR_ERROR),
        (8, 8, 0, 4, "col", DIVISOR_ERROR),
        (8, 8, 2, 0, "col", DIVISOR_ERROR),
        (0, 0, 0, 0, "col", DIVISOR_ERROR),
        (0, 2, 4, 2, "col", EXTENT_ERROR),
        (-8, 8, 2, 4, "row", EXTENT_ERROR),
        (8, 8, 2, 3, "col", "tile 8x8 / 2 = 32 not divisible by block_size 3"),
        (65536, 65536, 1, 3, "col", OVERFLOW_ERROR),
    ],
)
def test_invalid_geometry_rejects_before_builder_or_callbacks(
    rows, cols, width, threads, axis, diagnostic, grid
):
    loader = CoalescedTileLoader(rows, cols, threads, width, vector_axis=axis)

    def unexpected(*_):
        pytest.fail("invalid geometry invoked a traversal callback")

    # None cannot emit even a constant; a premature builder call fails the test.
    with pytest.raises(ValueError) as error:
        loader.visit_chunks(
            None, tid=0, emit=unexpected, grid=grid, before_decode=unexpected
        )
    assert str(error.value) == diagnostic


@pytest.mark.parametrize("method", ["load", "load_global", "plan_slots", "fetch"])
def test_other_loader_entry_points_validate_before_building_grid(method):
    loader = CoalescedTileLoader(8, 8, 4, 0)
    kwargs = {"tid": 0}
    if method != "plan_slots":
        kwargs.update(rsrc=object(), descriptor=None)
    if method == "load":
        kwargs["smem_dst"] = None
    with pytest.raises(ValueError) as error:
        getattr(loader, method)(None, **kwargs)
    assert str(error.value) == DIVISOR_ERROR


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
