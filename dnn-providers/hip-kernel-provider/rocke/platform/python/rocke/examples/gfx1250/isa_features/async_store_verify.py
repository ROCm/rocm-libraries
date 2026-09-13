# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate all gfx1250 asynchronous LDS-to-global store widths."""

from __future__ import annotations

import numpy as np

from rocke.core.ir import I8, I32, I64, IRBuilder, KernelDef, PtrType

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
_SOURCE_BYTES = 16
_OUTPUT_BYTES = 32
_TRANSFERS = ((0, 1), (4, 4), (8, 8), (16, 16))


def build_kernel() -> KernelDef:
    """Stage one 16-byte record per lane and issue all four DMA widths."""
    builder = IRBuilder("gfx1250_async_store_verify")
    builder.kernel.attrs["max_workgroup_size"] = _THREADS
    source = builder.param(
        "source", PtrType(I32, "global"), readonly=True, noalias=True, align=16
    )
    output = builder.param(
        "output", PtrType(I8, "global"), writeonly=True, noalias=True, align=16
    )
    shared = builder.smem_alloc(I32, [_THREADS, 4], name_hint="staged")
    lane = builder.thread_id_x()
    zero = builder.const_i32(0)

    source_index = builder.mul(lane, builder.const_i32(4))
    record = builder.global_load_vN(source, source_index, I32, 4, align=16)
    builder.smem_store_vN(shared, [lane, zero], record, 4)
    builder.inline_asm("s_wait_dscnt 0", "~{memory}", sideeffect=True, convergent=True)

    shared_base = builder.smem_addr_of(shared)
    shared_offset = builder.zext(
        builder.mul(lane, builder.const_i32(_SOURCE_BYTES)), I64
    )
    shared_record = builder.smem_ptr_add(shared_base, shared_offset)
    output_lane_base = builder.mul(lane, builder.const_i32(_OUTPUT_BYTES))
    for output_offset, width in _TRANSFERS:
        destination = builder.global_ptr_add(
            output, builder.add(output_lane_base, builder.const_i32(output_offset))
        )
        builder.global_store_async_from_lds(
            destination,
            shared_record,
            width_bytes=width,
            offset_bytes=0,
            cachepolicy=0,
        )
    builder.s_wait_asynccnt(0)
    builder.ret()
    return builder.kernel


def _run_functional(validated: ValidatedArtifact) -> tuple[bool, str]:
    source = (
        np.arange(_THREADS * 4, dtype=np.uint32) * np.uint32(0x01020409)
        + np.uint32(0x11223344)
    ).astype(np.int32)
    source_bytes = source.astype("<i4", copy=False).view(np.uint8).reshape(_THREADS, 16)
    expected = np.full((_THREADS, _OUTPUT_BYTES), 0xA5, dtype=np.uint8)
    for output_offset, width in _TRANSFERS:
        expected[:, output_offset : output_offset + width] = source_bytes[:, :width]

    runtime = Runtime()
    with DeviceArena(runtime) as device:
        source_dev = device.input(source)
        output_dev = device.output(expected.nbytes, fill=0xA5)
        launch(
            runtime,
            validated,
            grid=(1, 1, 1),
            block=(_THREADS, 1, 1),
            pack_format="<QQ",
            pack_values=(source_dev, output_dev),
        )
        actual = device.read(output_dev, dtype=np.dtype(np.uint8), shape=expected.shape)
    mismatch = int(np.count_nonzero(actual != expected))
    return mismatch == 0, f"widths=1/4/8/16, byte mismatches={mismatch}"


def main(argv: list[str] | None = None) -> int:
    args = make_parser(__doc__).parse_args(argv)
    reporter = Reporter(args.arch)
    validated = record_compile_check(
        reporter,
        "async-store.compile",
        build_kernel(),
        arch=args.arch,
        llvm_required=(
            "@llvm.amdgcn.global.store.async.from.lds.b8(",
            "@llvm.amdgcn.global.store.async.from.lds.b32(",
            "@llvm.amdgcn.global.store.async.from.lds.b64(",
            "@llvm.amdgcn.global.store.async.from.lds.b128(",
            "call void @llvm.amdgcn.s.wait.asynccnt(i16 0)",
        ),
        isa_required=(
            r"\bglobal_store_async_from_lds_b8\b",
            r"\bglobal_store_async_from_lds_b32\b",
            r"\bglobal_store_async_from_lds_b64\b",
            r"\bglobal_store_async_from_lds_b128\b",
            r"\bs_wait_asynccnt\b",
        ),
    )
    if validated is None:
        reporter.skipped("async-store.functional", "compile validation failed")
    elif args.compile_only:
        reporter.skipped("async-store.functional", "--compile-only requested")
    else:
        try:
            ok, detail = _run_functional(validated)
        except Exception as exc:  # noqa: BLE001
            reporter.failed("async-store.functional", f"{type(exc).__name__}: {exc}")
        else:
            if ok:
                reporter.passed("async-store.functional", detail)
            else:
                reporter.failed("async-store.functional", detail)
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
