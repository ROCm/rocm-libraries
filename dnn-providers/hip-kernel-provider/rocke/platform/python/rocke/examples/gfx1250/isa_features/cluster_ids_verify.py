# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate the gfx1250 workgroup-cluster id reads, the cluster barrier, and
cluster launch."""

from __future__ import annotations

import numpy as np

from rocke.core.ir import I32, IRBuilder, KernelDef, PtrType

try:
    from .common import (
        DeviceArena,
        Reporter,
        Runtime,
        launch,
        make_parser,
        record_compile_check,
    )
except ImportError:
    from common import (  # type: ignore[no-redef]
        DeviceArena,
        Reporter,
        Runtime,
        launch,
        make_parser,
        record_compile_check,
    )

_THREADS = 64
_AXES = ("x", "y", "z")
# Three per-axis reads on three axes, flat id, max flat id, cluster_size("x").
_SLOTS = 12
# The clustered launch: a 2x2x1 cluster tiled over a 4x4x1 grid.
_CLUSTER = (2, 2, 1)
_GRID = (4, 4, 1)


def build_kernel(cluster: tuple[int, int, int] | None = None) -> KernelDef:
    """Build a kernel that records every cluster read, then crosses one cluster
    barrier.

    Each workgroup writes ``_SLOTS`` i32 values to ``out[linear * _SLOTS +
    slot]``, where ``linear = bx + grid_x * (by + grid_y * bz)``, so a launch
    check can compare them against the launch shape. ``cluster`` declares the
    kernel's compiled ``cluster_dims``.
    """
    suffix = "" if cluster is None else "_{}x{}x{}".format(*cluster)
    builder = IRBuilder(f"gfx1250_cluster_ids_verify{suffix}")
    builder.kernel.attrs["max_workgroup_size"] = _THREADS
    if cluster is not None:
        builder.set_cluster_dims(*cluster)
    out = builder.param("out", PtrType(I32, "global"), writeonly=True, noalias=True, align=16)
    grid_x = builder.param("grid_x", I32)
    grid_y = builder.param("grid_y", I32)
    row = builder.add(
        builder.block_id_y(), builder.mul(grid_y, builder.block_id_z())
    )
    linear = builder.add(builder.block_id_x(), builder.mul(grid_x, row))
    base = builder.mul(linear, builder.const_i32(_SLOTS))
    slot = 0

    def record(value) -> None:
        nonlocal slot
        index = builder.add(base, builder.const_i32(slot))
        builder.global_store(out, index, value, align=4)
        slot += 1

    for axis in _AXES:
        record(builder.cluster_id(axis))
        record(builder.cluster_workgroup_id(axis))
        record(builder.cluster_workgroup_max_id(axis))
    record(builder.cluster_workgroup_flat_id())
    record(builder.cluster_workgroup_max_flat_id())
    record(builder.cluster_size("x"))
    assert slot == _SLOTS
    builder.cluster_barrier()
    builder.ret()
    return builder.kernel


def expected_ids(
    grid: tuple[int, int, int], cluster: tuple[int, int, int]
) -> np.ndarray:
    """Host reference: the ``_SLOTS`` values each workgroup should record.

    A launch without a cluster shape is a 1x1x1 cluster. Inside a cluster the
    flat id runs x fastest, then y, then z.
    """
    gx, gy, gz = grid
    cx, cy, cz = cluster
    rows = []
    for bz in range(gz):
        for by in range(gy):
            for bx in range(gx):
                block = (bx, by, bz)
                values = []
                for b, c in zip(block, cluster):
                    values += [b // c, b % c, c - 1]
                wx, wy, wz = (b % c for b, c in zip(block, cluster))
                values += [wx + cx * (wy + cy * wz), cx * cy * cz - 1, cx]
                rows.append(values)
    return np.asarray(rows, dtype=np.int32)


def _run_functional(
    validated: object,
    grid: tuple[int, int, int],
    cluster: tuple[int, int, int] | None,
) -> tuple[bool, str]:
    expected = expected_ids(grid, cluster or (1, 1, 1))
    runtime = Runtime()
    with DeviceArena(runtime) as device:
        out = device.output(expected.nbytes, fill=0xFF)
        launch(
            runtime,
            validated,
            grid=grid,
            block=(_THREADS, 1, 1),
            pack_format="<QII",
            pack_values=(out, grid[0], grid[1]),
            cluster=cluster,
        )
        actual = device.read(out, dtype=np.dtype(np.int32), shape=expected.shape)
    bad = np.argwhere(actual != expected)
    shape = "no cluster" if cluster is None else "cluster {}x{}x{}".format(*cluster)
    detail = f"grid {grid}, {shape}: mismatches={len(bad)}"
    if len(bad):
        workgroup, slot = (int(v) for v in bad[0])
        detail += (
            f", first at workgroup {workgroup} slot {slot}: "
            f"got {int(actual[workgroup, slot])}, "
            f"expected {int(expected[workgroup, slot])}"
        )
    return len(bad) == 0, detail


def _record_functional(
    reporter: Reporter,
    name: str,
    validated: object,
    compile_only: bool,
    grid: tuple[int, int, int],
    cluster: tuple[int, int, int] | None,
) -> None:
    if validated is None:
        reporter.skipped(name, "compile validation failed")
        return
    if compile_only:
        reporter.skipped(name, "--compile-only requested")
        return
    try:
        ok, detail = _run_functional(validated, grid, cluster)
    except Exception as exc:  # noqa: BLE001
        reporter.failed(name, f"{type(exc).__name__}: {exc}")
        return
    if ok:
        reporter.passed(name, detail)
    else:
        reporter.failed(name, detail)


_LLVM_REQUIRED = tuple(
    f"call i32 @llvm.amdgcn.{stem}.{axis}()"
    for stem in ("cluster.id", "cluster.workgroup.id", "cluster.workgroup.max.id")
    for axis in _AXES
) + (
    "call i32 @llvm.amdgcn.cluster.workgroup.flat.id()",
    "call i32 @llvm.amdgcn.cluster.workgroup.max.flat.id()",
    '  fence syncscope("cluster") release\n'
    "  call void @llvm.amdgcn.s.cluster.barrier()\n"
    '  fence syncscope("cluster") acquire\n',
)

_ISA_REQUIRED = (
    # Cluster ids come from the trap temporaries, packed 4 bits per field in
    # ttmp6: wg id x/y/z at 0/4/8, max id x/y/z at 12/16/20, max flat id at 24.
    r"\bttmp9\b",
    r"\bttmp7\b",
    r"\bs_and_b32\b[^\n]*\bttmp6\b",
    r"\bs_bfe_u32\b[^\n]*\bttmp6\b[^\n]*0x40004\b",
    r"\bs_bfe_u32\b[^\n]*\bttmp6\b[^\n]*0x4000c\b",
    r"\bs_bfe_u32\b[^\n]*\bttmp6\b[^\n]*0x40018\b",
    r"\bs_getreg_b32\b[^\n]*HW_REG_IB_STS2",
    # One wave per workgroup joins the cluster barrier (-3) after the
    # workgroup barrier (-1); every wave waits on it. llvm-objdump prints the
    # wait immediate as unsigned 16-bit, so -3 is 0xfffd.
    r"\bs_barrier_signal_isfirst\s+-1\b",
    r"\bs_barrier_signal\s+-3\b",
    r"\bs_barrier_wait\s+(?:-3|0xfffd)\b",
    # Cluster-scope release drains stores; acquire invalidates at SE scope.
    r"\bs_wait_storecnt\s+0x0\b",
    r"\bglobal_inv\b[^\n]*scope:SCOPE_SE\b",
)

# With the shape compiled in, the backend folds the per-axis max ids (and so
# cluster_size) to constants, which removes their ttmp6 extracts. The max flat
# id is still read.
_CLUSTERED_ISA_REQUIRED = tuple(
    pattern for pattern in _ISA_REQUIRED if "0x4000c" not in pattern
)


def main(argv: list[str] | None = None) -> int:
    args = make_parser(__doc__).parse_args(argv)
    reporter = Reporter(args.arch)
    plain = record_compile_check(
        reporter,
        "cluster-ids.compile",
        build_kernel(),
        arch=args.arch,
        llvm_required=_LLVM_REQUIRED,
        isa_required=_ISA_REQUIRED,
    )
    cx, cy, cz = _CLUSTER
    clustered = record_compile_check(
        reporter,
        "cluster-launch.compile",
        build_kernel(_CLUSTER),
        arch=args.arch,
        llvm_required=_LLVM_REQUIRED + (f'"amdgpu-cluster-dims"="{cx},{cy},{cz}"',),
        isa_required=_CLUSTERED_ISA_REQUIRED,
        # The backend copies the attribute into the kernel metadata, which the
        # loader reads; llvm-readelf prints the list flow- or block-style.
        notes_required=(
            rf"\.cluster_dims:\s*(?:\[\s*{cx},\s*{cy},\s*{cz}\s*\]"
            rf"|-\s*{cx}\s+-\s*{cy}\s+-\s*{cz}\b)",
        ),
    )
    # A launch without a cluster shape is a 1x1x1 cluster.
    _record_functional(
        reporter, "cluster-ids.functional", plain, args.compile_only, _GRID, None
    )
    _record_functional(
        reporter,
        "cluster-launch.functional",
        clustered,
        args.compile_only,
        _GRID,
        _CLUSTER,
    )
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
