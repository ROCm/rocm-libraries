# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate the four gfx1250 scalar-control instructions with an i32 transform."""

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
_DELTA = 17


def build_kernel() -> KernelDef:
    """Build a copy/transform kernel containing all four scalar controls."""
    builder = IRBuilder("gfx1250_scalar_controls_verify")
    builder.kernel.attrs["max_workgroup_size"] = _THREADS
    source = builder.param(
        "source", PtrType(I32, "global"), readonly=True, noalias=True, align=16
    )
    output = builder.param(
        "output", PtrType(I32, "global"), writeonly=True, noalias=True, align=16
    )
    lane = builder.thread_id_x()

    builder.s_clause(0)
    value = builder.global_load_i32(source, lane, align=4)
    builder.s_delay_alu(0)
    transformed = builder.add(value, builder.const_i32(_DELTA))
    builder.s_wait_alu(0)
    builder.s_wait_xcnt(0)
    builder.global_store(output, lane, transformed, align=4)
    builder.ret()
    return builder.kernel


def _run_functional(validated: object) -> tuple[bool, str]:
    source = (np.arange(_THREADS, dtype=np.int32) * 13) - 29
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
            pack_format="<QQ",
            pack_values=(source_dev, output_dev),
        )
        actual = device.read(output_dev, dtype=np.dtype(np.int32), shape=expected.shape)
    mismatch = int(np.count_nonzero(actual != expected))
    return mismatch == 0, f"exact i32 transform, mismatches={mismatch}"


def main(argv: list[str] | None = None) -> int:
    args = make_parser(__doc__).parse_args(argv)
    reporter = Reporter(args.arch)
    validated = record_compile_check(
        reporter,
        "scalar-controls.compile",
        build_kernel(),
        arch=args.arch,
        llvm_required=(
            'asm sideeffect "s_delay_alu 0"',
            'asm sideeffect "s_wait_alu 0"',
            'asm sideeffect "s_clause 0"',
            'asm sideeffect "s_wait_xcnt 0"',
        ),
        isa_required=(
            r"\bs_delay_alu\b",
            r"\bs_wait_alu\b",
            r"\bs_clause\b",
            r"\bs_wait_xcnt\b",
        ),
    )
    if validated is None:
        reporter.skipped("scalar-controls.functional", "compile validation failed")
    elif args.compile_only:
        reporter.skipped("scalar-controls.functional", "--compile-only requested")
    else:
        try:
            ok, detail = _run_functional(validated)
        except Exception as exc:  # noqa: BLE001
            reporter.failed(
                "scalar-controls.functional", f"{type(exc).__name__}: {exc}"
            )
        else:
            if ok:
                reporter.passed("scalar-controls.functional", detail)
            else:
                reporter.failed("scalar-controls.functional", detail)
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
