# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate gfx1250 tensor-DMA intrinsics without launching invalid descriptors."""

from __future__ import annotations

from rocke.core.ir import BF16, I32, IRBuilder, KernelDef, PtrType
from rocke.core.tdm import build_tdm_descriptor_2d, tdm_padding_for_tile

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


def build_descriptor_kernel() -> KernelDef:
    """Load one padded ``128x32`` bf16 tile of a row-major ``MxK`` tensor.

    Exercises the real descriptor path: a runtime global address, a runtime
    LDS address, runtime tensor extents, and hardware row padding.
    """
    builder = IRBuilder("gfx1250_tdm_descriptor_2d")
    builder.kernel.attrs["max_workgroup_size"] = 256
    block_m, block_k, pad = 128, 32, 8
    a_ptr = builder.param(
        "A", PtrType(BF16, "global"), noalias=True, readonly=True, align=16
    )
    k_dim = builder.param("K", I32)
    smem = builder.smem_alloc(BF16, [block_m, block_k + pad], name_hint="A_smem")
    pad_enable, pad_interval, pad_amount = tdm_padding_for_tile(2, block_k, pad)
    groups = build_tdm_descriptor_2d(
        builder,
        global_addr=builder.global_addr_of(a_ptr, builder.const_i32(0)),
        lds_addr=builder.smem_addr_of(smem),
        elem_bytes=2,
        tensor_dim0=k_dim,
        tensor_dim1=builder.const_i32(block_m),
        tile_dim0=block_k,
        tile_dim1=block_m,
        dim0_stride=1,
        dim1_stride=4096,
        pad_enable=pad_enable,
        pad_interval=pad_interval,
        pad_amount=pad_amount,
    )
    builder.tensor_load_to_lds(*groups, cachepolicy=0)
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
    record_compile_check(
        reporter,
        "tdm.descriptor_2d",
        build_descriptor_kernel(),
        arch=args.arch,
        llvm_required=(
            "call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>",
            "ptrtoint ptr addrspace(1)",
        ),
        isa_required=(
            r"\btensor_load_to_lds\b",
            r"\bs_wait_tensorcnt\b",
        ),
    )
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
