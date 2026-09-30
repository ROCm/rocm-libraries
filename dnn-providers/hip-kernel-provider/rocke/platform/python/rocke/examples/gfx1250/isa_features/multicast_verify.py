# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate the gfx1250 cluster multicast loads, synchronous and async-to-LDS."""

from __future__ import annotations

import numpy as np

from rocke.core.ir import I32, I64, IRBuilder, KernelDef, PtrType

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

_THREADS = 64
# One 16-byte source record per lane per group.
_RECORD_WORDS = 4
# b32 (1) + b64 (2) + b128 (4) synchronous, then async b8 (1) + b32 (1) +
# b64 (2) + b128 (4) read back from LDS.
_SLOTS = 15
# LDS words per lane: b8 at 0, b32 at 1, b64 at 2..3, b128 at 4..7.
_LDS_WORDS = 8
_ASYNC = ((1, 0), (4, 1), (8, 2), (16, 4))
_CLUSTER = (4, 1, 1)
_GRID = (8, 1, 1)
# Workgroups per multicast group: each alone, pairs, then the whole cluster.
_SPANS = (("one", 1), ("subset", 2), ("all", 4))


def build_kernel() -> KernelDef:
    """Build a kernel where each multicast group loads one shared record.

    ``span`` workgroups form a group: the workgroups with cluster flat ids
    ``first .. first + span - 1``, where ``first`` rounds the flat id down to a
    multiple of ``span``. Every member issues the same address with the
    group's mask, ``span_mask << first``, so the loads may be merged. The
    record comes from the group's first workgroup, so a workgroup that
    received a neighbouring group's data is caught.
    """
    builder = IRBuilder("gfx1250_multicast_verify")
    builder.kernel.attrs["max_workgroup_size"] = _THREADS
    builder.set_cluster_dims(*_CLUSTER)
    source = builder.param(
        "source", PtrType(I32, "global"), readonly=True, noalias=True, align=16
    )
    out = builder.param(
        "out", PtrType(I32, "global"), writeonly=True, noalias=True, align=16
    )
    span = builder.param("span", I32)
    span_mask = builder.param("span_mask", I32)
    lane = builder.thread_id_x()
    block = builder.block_id_x()

    flat = builder.cluster_workgroup_flat_id()
    position = builder.mod(flat, span)
    mask = builder.shl(span_mask, builder.sub(flat, position))
    group_block = builder.sub(block, position)

    record = builder.mul(
        builder.add(builder.mul(group_block, builder.const_i32(_THREADS)), lane),
        builder.const_i32(_RECORD_WORDS * 4),
    )
    src = builder.global_ptr_add(source, record)

    base = builder.mul(
        builder.add(builder.mul(block, builder.const_i32(_THREADS)), lane),
        builder.const_i32(_SLOTS),
    )
    slot = 0

    def store(value) -> None:
        nonlocal slot
        builder.global_store(out, builder.add(base, builder.const_i32(slot)), value, align=4)
        slot += 1

    store(builder.cluster_load(src, mask))
    for width, lanes in ((8, 2), (16, 4)):
        loaded = builder.cluster_load(src, mask, width_bytes=width)
        for index in range(lanes):
            store(builder.vec_extract(loaded, index))

    shared = builder.smem_alloc(I32, [_THREADS, _LDS_WORDS], name_hint="stage")
    lane_lds = builder.smem_ptr_add(
        builder.smem_addr_of(shared),
        builder.zext(builder.mul(lane, builder.const_i32(_LDS_WORDS * 4)), I64),
    )
    for width, word in _ASYNC:
        destination = builder.smem_ptr_add(lane_lds, builder.const_i64(word * 4))
        builder.cluster_load_async_to_lds(src, destination, mask, width_bytes=width)
    # The DMA writes LDS behind the wave's back; drain it before reading.
    builder.s_wait_asynccnt(0)

    zero = builder.const_i32(0)
    staged = builder.smem_load_vN(shared, lane, zero, dtype=I32, n=8)
    byte_shift = builder.const_i32(24)
    store(builder.lshr(builder.shl(builder.vec_extract(staged, 0), byte_shift), byte_shift))
    for index in range(1, 8):
        store(builder.vec_extract(staged, index))
    assert slot == _SLOTS
    builder.ret()
    return builder.kernel


def expected_output(source: np.ndarray, span: int) -> np.ndarray:
    """Host reference: the ``_SLOTS`` values each lane of each workgroup stores."""
    records = source.reshape(_GRID[0], _THREADS, _RECORD_WORDS)
    rows = np.empty((_GRID[0], _THREADS, _SLOTS), dtype=np.int32)
    for block in range(_GRID[0]):
        flat = block % _CLUSTER[0]
        record = records[block - flat % span]
        rows[block] = np.concatenate(
            [
                record[:, :1],
                record[:, :2],
                record[:, :4],
                record[:, :1] & 0xFF,
                record[:, :1],
                record[:, :2],
                record[:, :4],
            ],
            axis=1,
        )
    return rows


def _run_functional(validated: ValidatedArtifact, span: int) -> tuple[bool, str]:
    words = _GRID[0] * _THREADS * _RECORD_WORDS
    source = (
        np.arange(words, dtype=np.uint32) * np.uint32(0x01020409) + np.uint32(0x11223344)
    ).astype(np.int32)
    expected = expected_output(source, span)
    runtime = Runtime()
    with DeviceArena(runtime) as device:
        source_dev = device.input(source)
        out_dev = device.output(expected.nbytes, fill=0xA5)
        launch(
            runtime,
            validated,
            grid=_GRID,
            block=(_THREADS, 1, 1),
            pack_format="<QQII",
            pack_values=(source_dev, out_dev, span, (1 << span) - 1),
            cluster=_CLUSTER,
        )
        actual = device.read(out_dev, dtype=np.dtype(np.int32), shape=expected.shape)
    bad = np.argwhere(actual != expected)
    detail = f"span {span}, cluster {_CLUSTER}, grid {_GRID}: mismatches={len(bad)}"
    if len(bad):
        block, lane, slot = (int(v) for v in bad[0])
        detail += (
            f", first at workgroup {block} lane {lane} slot {slot}: "
            f"got {int(actual[block, lane, slot]):#x}, "
            f"expected {int(expected[block, lane, slot]):#x}"
        )
    return len(bad) == 0, detail


_LLVM_REQUIRED = (
    "= call i32 @llvm.amdgcn.cluster.load.b32.i32(ptr addrspace(1) ",
    "= call <2 x i32> @llvm.amdgcn.cluster.load.b64.v2i32(ptr addrspace(1) ",
    "= call <4 x i32> @llvm.amdgcn.cluster.load.b128.v4i32(ptr addrspace(1) ",
    "call void @llvm.amdgcn.cluster.load.async.to.lds.b8(ptr addrspace(1) ",
    "call void @llvm.amdgcn.cluster.load.async.to.lds.b32(ptr addrspace(1) ",
    "call void @llvm.amdgcn.cluster.load.async.to.lds.b64(ptr addrspace(1) ",
    "call void @llvm.amdgcn.cluster.load.async.to.lds.b128(ptr addrspace(1) ",
    "call void @llvm.amdgcn.s.wait.asynccnt(i16 0)",
)

_ISA_REQUIRED = (
    # The participation mask travels in M0; the backend may compute it there
    # directly (e.g. ``s_lshl_b32 m0, ...``) rather than via ``s_mov_b32``.
    r"\bs_\w+\s+m0,",
    r"\bcluster_load_b32\b",
    r"\bcluster_load_b64\b",
    r"\bcluster_load_b128\b",
    r"\bcluster_load_async_to_lds_b8\b",
    r"\bcluster_load_async_to_lds_b32\b",
    r"\bcluster_load_async_to_lds_b64\b",
    r"\bcluster_load_async_to_lds_b128\b",
    r"\bs_wait_asynccnt\s+0x0\b",
)


def main(argv: list[str] | None = None) -> int:
    args = make_parser(__doc__).parse_args(argv)
    reporter = Reporter(args.arch)
    validated = record_compile_check(
        reporter,
        "multicast.compile",
        build_kernel(),
        arch=args.arch,
        llvm_required=_LLVM_REQUIRED,
        isa_required=_ISA_REQUIRED,
    )
    for label, span in _SPANS:
        name = f"multicast.functional.{label}"
        if validated is None:
            reporter.skipped(name, "compile validation failed")
            continue
        if args.compile_only:
            reporter.skipped(name, "--compile-only requested")
            continue
        try:
            ok, detail = _run_functional(validated, span)
        except Exception as exc:  # noqa: BLE001
            reporter.failed(name, f"{type(exc).__name__}: {exc}")
            continue
        if ok:
            reporter.passed(name, detail)
        else:
            reporter.failed(name, detail)
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
