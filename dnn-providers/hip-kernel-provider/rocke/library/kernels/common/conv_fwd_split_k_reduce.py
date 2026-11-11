# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Stage 2 of the split-K forward convolution: fold the slabs and cast.

Stage 1 (``ImplicitGemmConvSpec`` with ``split_k > 1``) stores each K slice's
fp32 partial output into its own slab of a ``[ks_count][M][K]`` workspace --
plain stores, no atomics, every element written exactly once. This kernel sums
the ``ks_count`` slabs element-wise in a fixed order and converts the result to
``dtype_d`` into the output tensor ``D`` (N[D]HWK, the same packed layout as one
slab). The result is deterministic.

The degree is a kernarg, so one binary per (``dtype_d``, ``vec``) serves every
split. Each thread owns ``vec`` consecutive output elements: one vector load per
slab, ``vec`` adds, one vector store. ``vec`` must divide the element count
(``M * K``), which a K divisible by ``vec`` guarantees.

Kernel signature::

    ws_ptr   : f32 global ptr, readonly    -- workspace [ks_count * elems]
    d_ptr    : dtype_d global ptr          -- output [elems]
    elems    : i32 -- elements per slab (M * K)
    ws_bytes : i32 -- workspace byte size (ABI boundary)
    d_bytes  : i32 -- output byte size (ABI boundary)
    ks_count : i32 -- number of slabs (the split-K degree)

Grid: ``(ceil(elems / (vec * block_size)), 1, 1)``; block ``(block_size, 1, 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.io import io_ir_type, pack_f32_to, store_scalar_from_f32
from rocke.helpers.spec import SignatureBuilder, ceil_div, kernel_name_join


@dataclass(frozen=True)
class FwdSplitKReduceSpec:
    """One Stage 2 binary: the output dtype and elements per thread.

    ``degree`` > 0 specialises the fold for exactly that many slabs: the slab
    loop is unrolled, so all of a thread's loads issue back to back instead of
    one per dependent loop trip (``ks_count`` must then equal it). 0 keeps the
    runtime loop over ``ks_count``.
    """

    dtype_d: str = "bf16"
    vec: int = 4
    block_size: int = 256
    degree: int = 0
    name: str = "conv_fwd_splitk_reduce"

    def kernel_name(self) -> str:
        parts = [f"v{self.vec}", f"b{self.block_size}"]
        if self.degree:
            parts.append(f"s{self.degree}")
        return kernel_name_join(self.name, *parts, self.dtype_d)

    def validate(self) -> None:
        if self.vec not in (1, 2, 4):
            raise ValueError(f"vec must be 1, 2 or 4 (got {self.vec})")
        if self.dtype_d not in ("fp16", "bf16"):
            raise ValueError(f"dtype_d must be fp16 or bf16 (got {self.dtype_d})")
        if self.degree < 0:
            raise ValueError(f"degree must be >= 0 (got {self.degree})")


def build_conv_fwd_split_k_reduce(
    spec: FwdSplitKReduceSpec, *, arch: str = "gfx950"  # noqa: ARG001
) -> KernelDef:
    spec.validate()
    vec = spec.vec
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    ws_ptr = b.param(
        "ws_ptr", PtrType(F32, "global"), noalias=True, readonly=True, align=16
    )
    d_ptr = b.param(
        "d_ptr",
        PtrType(io_ir_type(spec.dtype_d), "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    elems = b.param("elems", I32)
    _ws_bytes = b.param("ws_bytes", I32)  # noqa: F841 -- ABI boundary
    _d_bytes = b.param("d_bytes", I32)  # noqa: F841 -- ABI boundary
    ks_count = b.param("ks_count", I32)

    gid = b.add(b.mul(b.block_id_x(), b.const_i32(spec.block_size)), b.thread_id_x())
    idx = b.mul(gid, b.const_i32(vec))

    def _load(off):
        if vec == 1:
            return [b.global_load_f32(ws_ptr, off)]
        v = b.global_load_vN(ws_ptr, off, F32, vec)
        return [b.vec_extract(v, i) for i in range(vec)]

    with b.scf_if(b.cmp_lt(idx, elems)):
        if spec.degree:
            # Unrolled: every slab's load is issued before the first add.
            slabs = [
                _load(idx if s == 0 else b.add(b.mul(b.const_i32(s), elems), idx))
                for s in range(spec.degree)
            ]
            total = slabs[0]
            for part in slabs[1:]:
                total = [b.fadd(a, p) for a, p in zip(total, part)]
        else:
            zero = b.const_f32(0.0)
            init = [(f"acc{i}", zero) for i in range(vec)]
            loop = b.scf_for_iter(
                b.const_i32(0), ks_count, b.const_i32(1), init, iv_name="slab"
            )
            with loop as (s, accs):
                parts = _load(b.add(b.mul(s, elems), idx))
                b.scf_yield(*[b.fadd(a, p) for a, p in zip(accs, parts)])
            total = list(loop.results)
        if vec == 1:
            store_scalar_from_f32(b, d_ptr, idx, total[0], dtype=spec.dtype_d)
        else:
            b.global_store_vN(
                d_ptr, idx, pack_f32_to(b, total, dtype=spec.dtype_d), vec
            )
    return b.kernel


def fwd_split_k_reduce_grid(
    spec: FwdSplitKReduceSpec, elems: int
) -> Tuple[int, int, int]:
    return (max(1, ceil_div(elems, spec.vec * spec.block_size)), 1, 1)


def fwd_split_k_reduce_signature(spec: FwdSplitKReduceSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("ws_ptr", "fp32")
        .ptr("d_ptr", spec.dtype_d)
        .scalar("elems", "i32")
        .scalar("ws_bytes", "i32")
        .scalar("d_bytes", "i32")
        .scalar("ks_count", "i32")
        .build()
    )


def fwd_split_k_reduce_vec(elems: int) -> int:
    """The widest ``vec`` that divides ``elems``."""
    return next(v for v in (4, 2, 1) if elems % v == 0)


def fwd_split_k_workspace_nbytes(M: int, K: int, ks_count: int) -> int:
    """Bytes of the ``[ks_count][M][K]`` fp32 workspace."""
    return 4 * ks_count * M * K
