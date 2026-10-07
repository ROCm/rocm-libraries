# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate gfx1250 tensor-DMA intrinsics without launching invalid descriptors."""

from __future__ import annotations

from rocke.core.ir import I32, IRBuilder, KernelDef

try:
    from .common import Reporter, make_parser, record_compile_check
except ImportError:
    from common import Reporter, make_parser, record_compile_check  # type: ignore[no-redef]


def build_kernel() -> KernelDef:
    """Build a compile-only probe with correctly typed descriptor groups."""
    builder = IRBuilder("gfx1250_tdm_compile_only")
    builder.kernel.attrs["max_workgroup_size"] = 32
    descriptor_4 = builder.zero_vec(I32, 4)
    descriptor_8 = builder.zero_vec(I32, 8)
    builder.tensor_load_to_lds(
        descriptor_4,
        descriptor_8,
        descriptor_4,
        descriptor_4,
        descriptor_8,
        cachepolicy=0,
    )
    builder.tensor_store_from_lds(
        descriptor_4,
        descriptor_8,
        descriptor_4,
        descriptor_4,
        descriptor_8,
        cachepolicy=0,
    )
    builder.s_wait_tensorcnt(0)
    builder.ret()
    return builder.kernel


def main(argv: list[str] | None = None) -> int:
    args = make_parser(__doc__).parse_args(argv)
    reporter = Reporter(args.arch)
    record_compile_check(
        reporter,
        "tdm.compile",
        build_kernel(),
        arch=args.arch,
        llvm_required=(
            "call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>",
            "call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32>",
            "call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)",
            "<8 x i32>",
        ),
        isa_required=(
            r"\btensor_load_to_lds\b",
            r"\btensor_store_from_lds\b",
            r"\bs_wait_tensorcnt\b",
        ),
    )
    reporter.skipped(
        "tdm.functional",
        "ROCKE does not expose construction of a valid D# memory/LDS descriptor",
    )
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
