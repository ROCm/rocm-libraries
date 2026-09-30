# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate the gfx1250 workgroup-cluster id reads and the cluster barrier."""

from __future__ import annotations

from rocke.core.ir import I32, IRBuilder, KernelDef, PtrType

try:
    from .common import Reporter, make_parser, record_compile_check
except ImportError:
    from common import Reporter, make_parser, record_compile_check  # type: ignore[no-redef]

_THREADS = 64
_AXES = ("x", "y", "z")
# Three per-axis reads on three axes, flat id, max flat id, cluster_size("x").
_SLOTS = 12


def build_kernel() -> KernelDef:
    """Build a kernel that records every cluster read, then crosses one cluster
    barrier.

    Each workgroup writes ``_SLOTS`` i32 values to ``out[block_id_x * _SLOTS +
    slot]``, so a later cluster-launch check can compare them against the
    launch shape.
    """
    builder = IRBuilder("gfx1250_cluster_ids_verify")
    builder.kernel.attrs["max_workgroup_size"] = _THREADS
    out = builder.param("out", PtrType(I32, "global"), writeonly=True, noalias=True, align=16)
    base = builder.mul(builder.block_id_x(), builder.const_i32(_SLOTS))
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


def main(argv: list[str] | None = None) -> int:
    args = make_parser(__doc__).parse_args(argv)
    reporter = Reporter(args.arch)
    validated = record_compile_check(
        reporter,
        "cluster-ids.compile",
        build_kernel(),
        arch=args.arch,
        llvm_required=tuple(
            f"call i32 @llvm.amdgcn.{stem}.{axis}()"
            for stem in ("cluster.id", "cluster.workgroup.id", "cluster.workgroup.max.id")
            for axis in _AXES
        )
        + (
            "call i32 @llvm.amdgcn.cluster.workgroup.flat.id()",
            "call i32 @llvm.amdgcn.cluster.workgroup.max.flat.id()",
            '  fence syncscope("cluster") release\n'
            "  call void @llvm.amdgcn.s.cluster.barrier()\n"
            '  fence syncscope("cluster") acquire\n',
        ),
        isa_required=(
            # Cluster ids come from the trap temporaries, packed 4 bits per
            # field in ttmp6: wg id x/y/z at 0/4/8, max id x/y/z at 12/16/20,
            # max flat id at 24.
            r"\bttmp9\b",
            r"\bttmp7\b",
            r"\bs_and_b32\b[^\n]*\bttmp6\b",
            r"\bs_bfe_u32\b[^\n]*\bttmp6\b[^\n]*0x40004\b",
            r"\bs_bfe_u32\b[^\n]*\bttmp6\b[^\n]*0x4000c\b",
            r"\bs_bfe_u32\b[^\n]*\bttmp6\b[^\n]*0x40018\b",
            r"\bs_getreg_b32\b[^\n]*HW_REG_IB_STS2",
            # One wave per workgroup joins the cluster barrier (-3) after the
            # workgroup barrier (-1); every wave waits on it. llvm-objdump
            # prints the wait immediate as unsigned 16-bit, so -3 is 0xfffd.
            r"\bs_barrier_signal_isfirst\s+-1\b",
            r"\bs_barrier_signal\s+-3\b",
            r"\bs_barrier_wait\s+(?:-3|0xfffd)\b",
            # Cluster-scope release drains stores; acquire invalidates at SE scope.
            r"\bs_wait_storecnt\s+0x0\b",
            r"\bglobal_inv\b[^\n]*scope:SCOPE_SE\b",
        ),
    )
    if validated is None:
        reporter.skipped("cluster-ids.functional", "compile validation failed")
    else:
        reporter.skipped(
            "cluster-ids.functional",
            "needs a cluster-shaped launch, which ROCKE does not expose yet",
        )
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
