# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A flat elementwise narrowing cast: ``D[i] = cast_to_dtype_d(A[i])``.

``direct_wgrad`` (:class:`kernels.common.conv_direct_grouped.DirectConvWgradSpec`)
and ``direct_wgrad_tiled``
(:class:`kernels.common.conv_direct_wgrad_tiled.DirectWgradTiledSpec`)
atomic-add their fp32 partial sums straight into their "D" kernarg, with an
offset formula that already matches the real dW tensor's final KYXC
(k-major, then (y, x), then channel-in-group) layout exactly -- no
permutation. So once "D" is pointed at an fp32 *scratch* buffer the same
shape as the real dW instead of the real (fp16/bf16) dW tensor, that
scratch is already a byte-faithful fp32 copy of the final answer, in the
same linear order as dW: the only remaining step is a narrowing cast, with
no index permutation at all. That is what this kernel does.

Unlike :class:`kernels.common.conv_direct_dw_wgrad.DwWgradFoldSpec` (which
performs a genuine reduction across partitions *and* a channel/tap axis
transpose as part of folding a tap-major scratch into dW), this kernel does
no reduction and no transpose -- reusing the fold kernel here would scramble
already-final-order data.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import F32, IRBuilder, KernelDef, Value
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import _io_type


def _cast_dtype_ir(dtype_d: str):
    """The IR element type ``dtype_d`` (``"fp16"`` or ``"bf16"``) stores as."""
    return _io_type(dtype_d)


def _cast_store_align(dtype_d: str) -> int:
    return 2


def _cast_value(b, v_f32: Value, dtype_d: str) -> Value:
    if dtype_d == "bf16":
        return b.trunc_f32_to_bf16(v_f32)
    if dtype_d == "fp16":
        return b.trunc_f32_to_f16(v_f32)
    raise ValueError(f"unsupported direct_wgrad_cast dtype_d: {dtype_d!r}")


@dataclass(frozen=True)
class DirectWgradCastSpec:
    """``D[i] = cast(A[i])`` for ``i`` in ``[0, count)``, one pass, no reduction.

    ``dtype_d`` is the real dW element type (``"fp16"`` or ``"bf16"``); ``A``
    is always the fp32 scratch the atomic producer wrote. ``count`` (the
    number of elements) is a runtime kernarg, not baked, so one compiled
    binary serves every problem shape.
    """

    dtype_d: str
    name: str = "direct_wgrad_cast"
    threads: int = 256

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(self.name, self.dtype_d, f"t{self.threads}")

    def launch_geometry(self, count: int):
        return (ceil_div(count, self.threads), 1, 1), (self.threads, 1, 1)


def is_valid_direct_wgrad_cast_spec(spec: DirectWgradCastSpec):
    if spec.dtype_d not in ("fp16", "bf16"):
        return False, f"unsupported dtype_d {spec.dtype_d!r}; expected 'fp16' or 'bf16'"
    if spec.threads < 1 or spec.threads > 1024:
        return False, "threads must be in [1, 1024]"
    return True, "ok"


# Kernargs after the A/D pointers and byte sizes, in order.
DIRECT_WGRAD_CAST_ARGS = ("count",)


def direct_wgrad_cast_signature(spec: DirectWgradCastSpec) -> list:
    """A = the fp32 scratch (final dW order), D = dW (``spec.dtype_d``)."""
    from rocke.helpers.spec import SignatureBuilder

    sb = (
        SignatureBuilder()
        .ptr("A", "fp32")
        .ptr("D", spec.dtype_d)
        .scalar("A_bytes", "i32")
        .scalar("D_bytes", "i32")
    )
    for name in DIRECT_WGRAD_CAST_ARGS:
        sb = sb.scalar(name, "i32")
    return sb.build()


def build_direct_wgrad_cast(spec: DirectWgradCastSpec, arch: str = "gfx950") -> KernelDef:
    ok, why = is_valid_direct_wgrad_cast_spec(spec)
    if not ok:
        raise ValueError(f"invalid DirectWgradCastSpec: {why}")
    from rocke.core.ir import I32, PtrType

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads
    a = b.param("A", PtrType(F32, "global"), noalias=True, readonly=True, align=4)
    dw = b.param(
        "D",
        PtrType(_cast_dtype_ir(spec.dtype_d), "global"),
        noalias=True,
        writeonly=True,
        align=_cast_store_align(spec.dtype_d),
    )
    b.param("A_bytes", I32)
    b.param("D_bytes", I32)
    count = b.param("count", I32)

    def ci(v):
        return b.const_i32(v)

    i = b.add(b.mul(b.block_id_x(), ci(spec.threads)), b.thread_id_x())
    ok_i = b.cmp_lt(i, count)
    i_safe = b.select(ok_i, i, ci(0))
    v_f32 = b.global_load_f32(a, i_safe, align=4)
    cast = _cast_value(b, v_f32, spec.dtype_d)
    with b.scf_if(ok_i):
        b.global_store(dw, i, cast, align=_cast_store_align(spec.dtype_d))
    return b.kernel
