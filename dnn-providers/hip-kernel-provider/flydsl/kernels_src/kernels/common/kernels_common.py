# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Common helpers shared by kernel modules.

Keep helper naming consistent with other kernel helpers (e.g. `mfma_preshuffle_pipeline.py`),
but this module is intentionally small and MLIR-dialect facing.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import builtin
from flydsl._mlir.dialects import gpu as _gpu
from flydsl.runtime.device import get_rocm_arch
from flydsl.runtime.device import get_warp_size as get_warp_size

# Memory/atomic primitives now live in mem_ops; re-exported here for back-compat.
from kernels.common.mem_ops import _create_llvm_ptr
from kernels.common.mem_ops import atomic_add as atomic_add
from kernels.common.mem_ops import get_llvm_ptr as get_llvm_ptr

LOG2E = 1.4426950408889634

_VALID_A_DTYPES = frozenset(("fp8", "fp16", "int8", "fp4"))
_VALID_B_DTYPES = frozenset(("fp8", "fp16", "int8", "int4", "fp4"))


def validate_moe_dtypes(a_dtype: str, b_dtype: str) -> None:
    """Validate a_dtype/b_dtype strings for mixed MoE kernels."""
    if a_dtype not in _VALID_A_DTYPES:
        raise ValueError(
            f"a_dtype must be one of {tuple(sorted(_VALID_A_DTYPES))}, got {a_dtype!r}"
        )
    if b_dtype not in _VALID_B_DTYPES:
        raise ValueError(
            f"b_dtype must be one of {tuple(sorted(_VALID_B_DTYPES))}, got {b_dtype!r}"
        )


def dtype_to_elem_type(dtype_str: str):
    """Map a dtype string to its FlyDSL numeric type.

    Supported: 'f32', 'f16', 'bf16', 'fp8' (OCP e4m3fn, not the fnuz variant).
    """
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    if dtype_str == "fp8":
        return fx.Float8E4M3FN
    raise ValueError(
        f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', 'bf16', or 'fp8')"
    )


def cvt_sr_f32_to_bf16(x, rand):
    """Stochastically convert f32 to bf16 using raw random bits.

    The add-then-truncate conversion matches AMD's non-saturating behavior.
    """
    bits = fx.Float32(x).bitcast(fx.Uint32) + (fx.Uint32(rand) & fx.Uint32(0xFFFF))
    return fx.Uint16(bits >> fx.Uint32(16)).bitcast(fx.BFloat16)


def default_f8_type() -> ir.Type:
    """Select E4M3 f8 type compatible with the current GPU arch.

    - gfx95* (MI350): FP8 E4M3FN (OCP)
    - gfx12*: FP8 E4M3FN (OCP)
    - gfx94* (MI300): FP8 E4M3FNUZ

    Raises ``RuntimeError`` on gfx11* (RDNA3/RDNA3.5): these chips have no
    native FP8 instructions, so FP8 compute would surface as a late LLVM
    "cannot select" error. Fail early with a clear message instead.
    """
    arch = ""
    try:
        arch = str(get_rocm_arch())
    except Exception:
        arch = ""
    if "gfx95" in arch or "gfx12" in arch:
        return fx.Float8E4M3FN.ir_type
    if arch.startswith("gfx11"):
        raise RuntimeError(
            f"default_f8_type(): no native FP8 support on {arch}; "
            "FP8 instructions are available on gfx94*, gfx95*, and gfx12*. "
            "Use bf16/f16 GEMM via "
            "`rdna3_f16_gemm.create_wmma_gemm_module` on gfx11* targets."
        )
    return fx.Float8E4M3FNUZ.ir_type


def stream_ptr_to_async_token(stream_ptr_value, loc=None, ip=None):
    stream_llvm_ptr = _create_llvm_ptr(stream_ptr_value)

    async_token_type = _gpu.AsyncTokenType.get()
    cast_op = builtin.UnrealizedConversionCastOp(
        [async_token_type], [stream_llvm_ptr], loc=loc, ip=ip
    )
    return cast_op.results[0]
