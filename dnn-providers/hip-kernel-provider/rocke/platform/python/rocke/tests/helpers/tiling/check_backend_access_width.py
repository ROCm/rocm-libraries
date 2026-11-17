# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The backend merges each lane's global run to the width the tiling layer plans for it.

``load_fragment`` and ``store_fragment`` move a lane's run of a window either in one wide access
or one storage unit per access, addressed so the backend can merge the run back
(:func:`~rocke.helpers.tiling.emit.planned_global_access_units`). Whether the backend does is a
property of the compiler: the per-element loads once compiled to byte, short, dword and dwordx2
pieces of a 16-byte run, with different failures per LLVM version, and every correctness test
still passed. This check compiles a few one-wave copy kernels and requires every global load and
store in the code object to move exactly the planned width.

The cases cover the classes that fragmented: a run the layer keeps scalar because of its dtype
(u8) with a block origin and with a parameter origin, one and four row blocks per lane; an f16
run kept scalar by an odd row stride; and, as the control, an f16 run the layer loads wide.
Every case's stores are per element. A tile that may overhang its tensor masks each load on its
own and is not covered here.

It needs COMGR, llvm-objdump and llvm-readelf (it skips without them) but no GPU. It compiles,
so it is not part of ordinary test discovery: its installed CTest entry names this file
(``rocke_tiling_backend_width_pytest``, standard tier). It checks the LLVM version of the COMGR
it runs with; the failures it guards against differ by version.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rocke.analysis.isa import analyze_hsaco
from rocke.core.dtypes import dtype_info
from rocke.core.ir import F16, U8
from rocke.helpers.compile import compile_kernel
from rocke.helpers.tiling.coalescing_probe import achieved_widths
from rocke.helpers.tiling.emit import planned_global_access_units
from rocke.tests.comgr import require_comgr_target
from rocke.tests.llc import find_llvm_tool

from ._copy_kernels import CopyCase, copy_kernel

_ARCH = "gfx942"
_TENSOR_EXTENT = 4096

_CASES = [
    CopyCase(
        "u8_block_r4",
        U8,
        run=16,
        rows_per_lane=4,
        rows=_TENSOR_EXTENT,
        columns=_TENSOR_EXTENT,
    ),
    CopyCase(
        "u8_param_r4",
        U8,
        run=16,
        rows_per_lane=4,
        rows=_TENSOR_EXTENT,
        columns=_TENSOR_EXTENT,
        column_origin="param",
    ),
    CopyCase(
        "u8_param_r1",
        U8,
        run=16,
        rows_per_lane=1,
        rows=_TENSOR_EXTENT,
        columns=_TENSOR_EXTENT,
        column_origin="param",
    ),
    CopyCase(
        "f16_odd_row_stride_r4",
        F16,
        run=8,
        rows_per_lane=4,
        rows=_TENSOR_EXTENT,
        columns=_TENSOR_EXTENT,
        row_stride=_TENSOR_EXTENT + 1,
    ),
    CopyCase(
        "f16_wide_load_r4",
        F16,
        run=8,
        rows_per_lane=4,
        rows=_TENSOR_EXTENT,
        columns=_TENSOR_EXTENT,
    ),
]


@pytest.fixture(scope="module")
def llvm_tools() -> tuple[str, str]:
    """llvm-objdump and llvm-readelf, checked once with COMGR's gfx942 target; skips without
    them."""
    require_comgr_target(_ARCH)
    objdump, readelf = find_llvm_tool("llvm-objdump"), find_llvm_tool("llvm-readelf")
    if objdump is None or readelf is None:
        pytest.skip("llvm-objdump or llvm-readelf is not available")
    return objdump, readelf


def _disassembly(hsaco: bytes, tools: tuple[str, str]) -> str:
    """The disassembly of a code object."""
    objdump, readelf = tools
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "copy.hsaco"
        path.write_bytes(hsaco)
        return analyze_hsaco(
            path, objdump=objdump, readelf=readelf, keep_text=True
        ).objdump_text


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_every_global_access_reaches_the_planned_width(
    case: CopyCase, llvm_tools: tuple[str, str]
) -> None:
    kernel, tile, window = copy_kernel(case)
    disassembly = _disassembly(compile_kernel(kernel, arch=_ARCH).hsaco, llvm_tools)
    planned = planned_global_access_units(tile, window)
    expected = {planned: tile.register_count // planned}
    bits = dtype_info(case.dtype.name).encoded_bits
    for direction in ("load", "store"):
        achieved = achieved_widths(
            disassembly, direction=direction, space="global", dtype_bits=bits
        )
        assert achieved == expected, (
            f"{case.name} on {_ARCH}: global {direction}s reach {achieved} "
            f"({{elements per access: accesses}}), planned {expected} -- the backend did not "
            "merge the lane's run to the planned width"
        )
