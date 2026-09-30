# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate the gfx1250 data-prefetch operations with an i32 transform."""

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
_DELTA = 23
_LINES = 4
_BUFFER_OFFSET = 256


def build_kernel() -> KernelDef:
    """Build a copy/transform kernel that issues every data prefetch first.

    ``table`` and ``flat`` view the same device buffer as ``source`` through
    the constant and flat address spaces, so each ``s_prefetch_data`` pointer
    overload is exercised against real memory.
    """
    builder = IRBuilder("gfx1250_data_prefetch_verify")
    builder.kernel.attrs["max_workgroup_size"] = _THREADS
    source = builder.param(
        "source", PtrType(I32, "global"), readonly=True, noalias=True, align=16
    )
    table = builder.param("table", PtrType(I32, "global"), addr_space="constant")
    flat = builder.param("flat", PtrType(I32, "private"))
    output = builder.param(
        "output", PtrType(I32, "global"), writeonly=True, noalias=True, align=16
    )
    nbytes = builder.param("nbytes", I32)
    lane = builder.thread_id_x()
    lines = builder.const_i32(_LINES)

    builder.enable_scalar_prefetch()
    builder.s_prefetch_data(source, lines)
    builder.s_prefetch_data(table, lines)
    builder.s_prefetch_data(flat, lines)
    rsrc = builder.buffer_rsrc(source, nbytes)
    builder.s_buffer_prefetch_data(rsrc, lines, offset=_BUFFER_OFFSET)
    builder.global_prefetch(source)
    builder.flat_prefetch(flat)

    value = builder.global_load_i32(source, lane, align=4)
    transformed = builder.add(value, builder.const_i32(_DELTA))
    builder.global_store(output, lane, transformed, align=4)
    builder.ret()
    return builder.kernel


def _run_functional(validated: object) -> tuple[bool, str]:
    source = (np.arange(_THREADS, dtype=np.int32) * 7) - 101
    expected = source + _DELTA
    runtime = Runtime()
    with DeviceArena(runtime) as device:
        source_dev = device.input(source)
        output_dev = device.output(expected.nbytes)
        launch(
            runtime,
            validated,
            grid=(1, 1, 1),
            block=(_THREADS, 1, 1),
            pack_format="<QQQQI",
            pack_values=(
                source_dev,
                source_dev,
                source_dev,
                output_dev,
                source.nbytes,
            ),
        )
        actual = device.read(output_dev, dtype=np.dtype(np.int32), shape=expected.shape)
    mismatch = int(np.count_nonzero(actual != expected))
    return mismatch == 0, f"prefetch leaves the i32 transform exact, mismatches={mismatch}"


def main(argv: list[str] | None = None) -> int:
    args = make_parser(__doc__).parse_args(argv)
    reporter = Reporter(args.arch)
    validated = record_compile_check(
        reporter,
        "data-prefetch.compile",
        build_kernel(),
        arch=args.arch,
        llvm_required=(
            "call void @llvm.amdgcn.s.setreg(i32 1537, i32 1)",
            "call void @llvm.amdgcn.s.prefetch.data.p1(",
            "call void @llvm.amdgcn.s.prefetch.data.p4(",
            "call void @llvm.amdgcn.s.prefetch.data.p0(",
            "call void @llvm.amdgcn.s.buffer.prefetch.data(",
            "call void @llvm.amdgcn.global.prefetch(",
            "call void @llvm.amdgcn.flat.prefetch(",
        ),
        isa_required=(
            r"\bs_setreg",
            r"\bs_prefetch_data\b",
            r"\bs_buffer_prefetch_data\b",
            r"\bglobal_prefetch_b8\b",
            r"\bflat_prefetch_b8\b",
        ),
    )
    if validated is None:
        reporter.skipped("data-prefetch.functional", "compile validation failed")
    elif args.compile_only:
        reporter.skipped("data-prefetch.functional", "--compile-only requested")
    else:
        try:
            ok, detail = _run_functional(validated)
        except Exception as exc:  # noqa: BLE001
            reporter.failed("data-prefetch.functional", f"{type(exc).__name__}: {exc}")
        else:
            if ok:
                reporter.passed("data-prefetch.functional", detail)
            else:
                reporter.failed("data-prefetch.functional", detail)
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
