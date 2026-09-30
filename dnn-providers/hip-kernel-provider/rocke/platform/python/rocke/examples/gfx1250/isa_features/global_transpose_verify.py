# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Validate gfx1250 global_load_tr16_b128 lowering and final ISA."""

from __future__ import annotations

from rocke.core.ir import BF16, F16, I16, IRBuilder, KernelDef, PtrType, Type

try:
    from .common import Reporter, make_parser, record_compile_check
except ImportError:
    from common import Reporter, make_parser, record_compile_check  # type: ignore[no-redef]

_THREADS = 32
_DTYPES = (
    ("f16", F16, "v8f16", "<8 x half>"),
    ("bf16", BF16, "v8bf16", "<8 x bfloat>"),
    ("i16", I16, "v8i16", "<8 x i16>"),
)


def build_kernel(label: str, dtype: Type) -> KernelDef:
    """Build a wave32 probe that consumes the returned eight-element vector."""
    builder = IRBuilder(f"gfx1250_global_tr16_{label}_verify")
    builder.kernel.attrs["max_workgroup_size"] = _THREADS
    source = builder.param(
        "source", PtrType(dtype, "global"), readonly=True, noalias=True, align=16
    )
    output = builder.param(
        "output", PtrType(dtype, "global"), writeonly=True, noalias=True, align=16
    )
    lane = builder.thread_id_x()
    lane_source = builder.global_ptr_add(
        source, builder.mul(lane, builder.const_i32(16))
    )
    fragment = builder.global_load_tr16_b128(lane_source, dtype=dtype)
    builder.global_store(
        output, builder.mul(lane, builder.const_i32(8)), fragment, align=16
    )
    builder.ret()
    return builder.kernel


def main(argv: list[str] | None = None) -> int:
    args = make_parser(__doc__).parse_args(argv)
    reporter = Reporter(args.arch)
    for label, dtype, suffix, llvm_type in _DTYPES:
        record_compile_check(
            reporter,
            f"global-tr16-{label}.compile",
            build_kernel(label, dtype),
            arch=args.arch,
            llvm_required=(
                f"call {llvm_type} @llvm.amdgcn.global.load.tr.b128.{suffix}(",
            ),
            isa_required=(r"\bglobal_load_tr16_b128\b",),
        )

    reporter.skipped(
        "global-tr16.functional",
        "exact wave32 lane permutation is not documented by the exposed ROCKE API",
    )
    return reporter.finish()


if __name__ == "__main__":
    raise SystemExit(main())
