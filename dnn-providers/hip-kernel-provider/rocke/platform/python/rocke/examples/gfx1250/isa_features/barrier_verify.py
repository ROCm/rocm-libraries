# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate gfx1250 split barriers and a cross-wave LDS handoff."""

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
_WORKGROUP_SIGNAL = 0xFFFFFFFF
_WORKGROUP_WAIT = 0xFFFF


def _wait_for_lds(builder: IRBuilder) -> None:
    """Drain LDS traffic and act as a compiler memory fence."""
    builder.inline_asm("s_wait_dscnt 0", "~{memory}", sideeffect=True, convergent=True)


def build_workgroup_kernel() -> KernelDef:
    """Build a two-wave producer/consumer using the workgroup split barrier."""
    builder = IRBuilder("gfx1250_split_barrier_verify")
    builder.kernel.attrs["max_workgroup_size"] = _THREADS
    source = builder.param(
        "source", PtrType(I32, "global"), readonly=True, noalias=True, align=16
    )
    output = builder.param(
        "output", PtrType(I32, "global"), writeonly=True, noalias=True, align=16
    )
    shared = builder.smem_alloc(I32, [_THREADS], name_hint="handoff")
    lane = builder.thread_id_x()

    value = builder.global_load_i32(source, lane, align=4)
    produced = builder.add(
        builder.mul(value, builder.const_i32(3)), builder.const_i32(5)
    )
    builder.smem_store_vN(shared, [lane], produced, 1)
    _wait_for_lds(builder)

    builder.s_barrier_signal(_WORKGROUP_SIGNAL)
    peer = builder.mod(
        builder.add(lane, builder.const_i32(_THREADS // 2)),
        builder.const_i32(_THREADS),
    )
    builder.s_barrier_wait(_WORKGROUP_WAIT)

    _wait_for_lds(builder)
    consumed = builder.vec_extract(
        builder.smem_load_vN(shared, peer, dtype=I32, n=1), 0
    )
    builder.global_store(output, lane, consumed, align=4)
    builder.ret()
    return builder.kernel


def build_named_lifecycle_kernel() -> KernelDef:
    """Build, but never launch, the named-barrier lifecycle bridge."""
    builder = IRBuilder("gfx1250_named_barrier_compile_only")
    builder.kernel.attrs["max_workgroup_size"] = 32
    barrier = builder.smem_alloc(I64, [1], name_hint="named_barrier")
    barrier_pointer = builder.smem_addr_of(barrier)
    members = builder.const_i32(2)
    builder.s_barrier_init(barrier_pointer, members)
    builder.s_barrier_signal_var(barrier_pointer, members)
    builder.s_barrier_join(barrier_pointer)
    builder.s_wakeup_barrier(barrier_pointer)
    builder.s_barrier_leave(0)
    builder.ret()
    return builder.kernel


def _run_functional(validated: ValidatedArtifact) -> tuple[bool, str]:
    source = (np.arange(_THREADS, dtype=np.int32) * 11) - 7
    produced = source * 3 + 5
    expected = np.roll(produced, _THREADS // 2)
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
    return mismatch == 0, f"cross-wave LDS handoff, mismatches={mismatch}"


def main(argv: list[str] | None = None) -> int:
    args = make_parser(__doc__).parse_args(argv)
    reporter = Reporter(args.arch)
    workgroup = record_compile_check(
        reporter,
        "split-barrier.compile",
        build_workgroup_kernel(),
        arch=args.arch,
        llvm_required=(
            "call void @llvm.amdgcn.s.barrier.signal(i32 4294967295)",
            "call void @llvm.amdgcn.s.barrier.wait(i16 65535)",
            'asm sideeffect "s_wait_dscnt 0"',
        ),
        isa_required=(r"\bs_barrier_signal\b", r"\bs_barrier_wait\b"),
    )
    record_compile_check(
        reporter,
        "named-barrier.compile",
        build_named_lifecycle_kernel(),
        arch=args.arch,
        llvm_required=(
            "call void @llvm.amdgcn.s.barrier.init(",
            "call void @llvm.amdgcn.s.barrier.signal.var(",
            "call void @llvm.amdgcn.s.barrier.join(",
            "call void @llvm.amdgcn.s.wakeup.barrier(",
            "call void @llvm.amdgcn.s.barrier.leave(i16 0)",
        ),
        isa_required=(
            r"\bs_barrier_init\b",
            r"\bs_barrier_signal\s+m0\b",
            r"\bs_barrier_join\b",
            r"\bs_wakeup_barrier\b",
            r"\bs_barrier_leave\b",
        ),
    )
    reporter.skipped(
        "named-barrier.functional",
        "lifecycle/member-count semantics are not exposed by ROCKE",
    )

    if workgroup is None:
        reporter.skipped("split-barrier.functional", "compile validation failed")
    elif args.compile_only:
        reporter.skipped("split-barrier.functional", "--compile-only requested")
    else:
        try:
            ok, detail = _run_functional(workgroup)
        except Exception as exc:  # noqa: BLE001
            reporter.failed("split-barrier.functional", f"{type(exc).__name__}: {exc}")
        else:
            if ok:
                reporter.passed("split-barrier.functional", detail)
            else:
                reporter.failed("split-barrier.functional", detail)
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
