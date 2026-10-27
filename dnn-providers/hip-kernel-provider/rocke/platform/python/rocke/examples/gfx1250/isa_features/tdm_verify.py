# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate a gfx1250 rank-2 TDM tile round trip: global -> LDS -> global."""

from __future__ import annotations

import numpy as np

from rocke.core.ir import I32, IRBuilder, KernelDef, PtrType

try:
    from .common import (
        DeviceArena,
        Reporter,
        Runtime,
        ValidatedArtifact,
        launch,
        make_parser,
        record_compile_check,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        DeviceArena,
        Reporter,
        Runtime,
        ValidatedArtifact,
        launch,
        make_parser,
        record_compile_check,
    )

# One wave per workgroup; the wave issues both transfers.
_THREADS = 32
_TILE_COLS = 64
_TILE_ROWS = 16
# A 32 x 128 i32 tensor stored with a wider row pitch, so row_stride differs
# from the tensor extent and a wrong stride shows up as shifted rows.
_ROWS = 32
_COLS = 128
_PITCH = 160
_GRID = (_COLS // _TILE_COLS, _ROWS // _TILE_ROWS, 1)
# Words each lane reads back from LDS per tile row.
_LANE_WORDS = _TILE_COLS // _THREADS


def build_kernel() -> KernelDef:
    """Build a kernel that moves one tile per workgroup through LDS.

    Workgroup ``(x, y)`` owns the tile whose top-left element is
    ``(y * 16, x * 64)``. Its descriptors point at that element, with the
    extents left from there, so the hardware sees an in-bounds tile. The TDM
    load fills LDS; the TDM store writes the same LDS tile to ``dst``. Each
    lane also reads LDS directly into ``staged``, which checks the packed
    row-major LDS layout independently of the store.
    """
    builder = IRBuilder("gfx1250_tdm_verify")
    builder.kernel.attrs["max_workgroup_size"] = _THREADS
    src = builder.param(
        "src", PtrType(I32, "global"), readonly=True, noalias=True, align=16
    )
    dst = builder.param("dst", PtrType(I32, "global"), noalias=True, align=16)
    staged = builder.param(
        "staged", PtrType(I32, "global"), writeonly=True, noalias=True, align=16
    )
    rows = builder.param("rows", I32)
    cols = builder.param("cols", I32)
    pitch = builder.param("pitch", I32)
    lane = builder.thread_id_x()
    tile_x = builder.block_id_x()
    tile_y = builder.block_id_y()

    row0 = builder.mul(tile_y, builder.const_i32(_TILE_ROWS))
    col0 = builder.mul(tile_x, builder.const_i32(_TILE_COLS))
    element = builder.add(builder.mul(row0, pitch), col0)
    offset = builder.mul(element, builder.const_i32(4))
    extent0 = builder.sub(cols, col0)
    extent1 = builder.sub(rows, row0)

    shared = builder.smem_alloc(I32, [_TILE_ROWS, _TILE_COLS], name_hint="tile")
    lds = builder.smem_addr_of(shared)
    shape = dict(
        elem_bytes=4,
        tensor_dim0=extent0,
        tensor_dim1=extent1,
        row_stride=pitch,
        tile_dim0=_TILE_COLS,
        tile_dim1=_TILE_ROWS,
    )
    builder.tensor_load_to_lds(
        *builder.tdm_descriptor_2d(builder.global_ptr_add(src, offset), lds, **shape)
    )
    # The DMA writes LDS behind the wave's back; drain it before reading.
    builder.s_wait_tensorcnt(0)

    block = builder.add(tile_x, builder.mul(tile_y, builder.const_i32(_GRID[0])))
    tile_base = builder.mul(block, builder.const_i32(_TILE_ROWS * _TILE_COLS))
    col = builder.mul(lane, builder.const_i32(_LANE_WORDS))
    for row in range(_TILE_ROWS):
        row_value = builder.const_i32(row)
        words = builder.smem_load_vN(shared, row_value, col, dtype=I32, n=_LANE_WORDS)
        slot = builder.add(
            tile_base, builder.add(builder.const_i32(row * _TILE_COLS), col)
        )
        for index in range(_LANE_WORDS):
            builder.global_store(
                staged,
                builder.add(slot, builder.const_i32(index)),
                builder.vec_extract(words, index),
                align=4,
            )

    builder.tensor_store_from_lds(
        *builder.tdm_descriptor_2d(builder.global_ptr_add(dst, offset), lds, **shape)
    )
    builder.s_wait_tensorcnt(0)
    builder.ret()
    return builder.kernel


def _tiles(tensor: np.ndarray) -> np.ndarray:
    """``[grid_y * grid_x, 16, 64]`` tiles in workgroup order."""
    return (
        tensor.reshape(_GRID[1], _TILE_ROWS, _GRID[0], _TILE_COLS)
        .transpose(0, 2, 1, 3)
        .reshape(_GRID[0] * _GRID[1], _TILE_ROWS, _TILE_COLS)
    )


def _run_functional(validated: ValidatedArtifact) -> tuple[bool, str]:
    source = (
        np.arange(_ROWS * _PITCH, dtype=np.uint32) * np.uint32(0x01020409)
        + np.uint32(0x11223344)
    ).astype(np.int32)
    tensor = source.reshape(_ROWS, _PITCH)[:, :_COLS]
    # The store must write the tensor window and leave the pitch gap alone.
    fill = np.int32(-0x5A5A5A5B)
    expected_dst = np.full((_ROWS, _PITCH), fill, dtype=np.int32)
    expected_dst[:, :_COLS] = tensor
    expected_staged = _tiles(tensor)
    runtime = Runtime()
    with DeviceArena(runtime) as device:
        src_dev = device.input(source)
        dst_dev = device.output(expected_dst.nbytes, fill=0xA5)
        staged_dev = device.output(expected_staged.nbytes, fill=0xA5)
        launch(
            runtime,
            validated,
            grid=_GRID,
            block=(_THREADS, 1, 1),
            pack_format="<QQQIII",
            pack_values=(src_dev, dst_dev, staged_dev, _ROWS, _COLS, _PITCH),
        )
        dst = device.read(dst_dev, dtype=np.dtype(np.int32), shape=expected_dst.shape)
        staged = device.read(
            staged_dev, dtype=np.dtype(np.int32), shape=expected_staged.shape
        )
    details = []
    ok = True
    for label, actual, expected in (
        ("lds", staged, expected_staged),
        ("dst", dst, expected_dst),
    ):
        bad = np.argwhere(actual != expected)
        detail = f"{label}: mismatches={len(bad)}"
        if len(bad):
            ok = False
            first = tuple(int(v) for v in bad[0])
            detail += (
                f", first at {first}: got {int(actual[first]):#x}, "
                f"expected {int(expected[first]):#x}"
            )
        details.append(detail)
    shape = f"tensor {_ROWS}x{_COLS} pitch {_PITCH}, tile {_TILE_ROWS}x{_TILE_COLS}"
    return ok, f"{shape}; " + "; ".join(details)


_LLVM_REQUIRED = (
    "= ptrtoint ptr addrspace(1) ",
    "@llvm.amdgcn.readfirstlane.i32(i32 ",
    "call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> ",
    "call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> ",
    "call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)",
    # Group 1 dword 0: data_size 2 (4-byte elements), no padding, no mask.
    "(i32 131072)",
)

_ISA_REQUIRED = (
    r"\btensor_load_to_lds\b",
    r"\btensor_store_from_lds\b",
    r"\bs_wait_tensorcnt\s+0x0\b",
    # The direct LDS read-back (the backend may pair the row loads).
    r"\bds_load_\w*b64\b",
)


def main(argv: list[str] | None = None) -> int:
    args = make_parser(__doc__).parse_args(argv)
    reporter = Reporter(args.arch)
    validated = record_compile_check(
        reporter,
        "tdm.compile",
        build_kernel(),
        arch=args.arch,
        llvm_required=_LLVM_REQUIRED,
        isa_required=_ISA_REQUIRED,
    )
    name = "tdm.functional.round_trip"
    if validated is None:
        reporter.skipped(name, "compile validation failed")
    elif args.compile_only:
        reporter.skipped(name, "--compile-only requested")
    else:
        try:
            ok, detail = _run_functional(validated)
        except Exception as exc:  # noqa: BLE001
            reporter.failed(name, f"{type(exc).__name__}: {exc}")
        else:
            if ok:
                reporter.passed(name, detail)
            else:
                reporter.failed(name, detail)
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
