# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 token sampler (greedy argmax).

The end-of-step op: pick the next token id per sequence from the LM-head logits.
``out[t] = argmax_v logits[t, v]`` with deterministic lowest-index tie-break
(matching ``numpy.argmax``), which is the ``temperature == 0`` greedy path the
serving engines use by default.

One workgroup owns one token row: each thread scans a strided slice of the vocab
tracking a local ``(max, idx)``, then an LDS index-reduction
(:func:`helpers.reduction.block_lds_reduce_with_index`) collapses the block to
the row argmax; lane 0 writes the id. Arch-neutral (reduction + scalar ops, no
WMMA), so it builds/runs on gfx1250 and CDNA wave64.

Stochastic sampling (temperature / top-k / top-p with an RNG) is a documented
follow-on; greedy is the decode-critical path and what ``temperature=0`` selects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Type
from ...helpers.reduction import block_lds_reduce_with_index

__all__ = [
    "Qwen3GreedySamplerSpec",
    "build_qwen3_greedy_sampler",
    "qwen3_greedy_sampler_grid",
    "qwen3_greedy_sampler_signature",
]


def _dtype_ir(dtype: str) -> Type:
    if dtype in ("f32", "fp32"):
        return F32
    if dtype == "bf16":
        from ...core.ir import BF16

        return BF16
    if dtype in ("fp16", "f16"):
        from ...core.ir import F16

        return F16
    raise ValueError(f"unsupported logits dtype {dtype!r}")


def _require_supported(arch: str) -> Tuple[bool, str]:
    from ...core.arch import ArchTarget

    try:
        ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    return True, "supported"


@dataclass(frozen=True)
class Qwen3GreedySamplerSpec:
    """Greedy argmax sampler over the vocab. ``block_size`` must be a power of 2."""

    logits_dtype: str = "f32"
    block_size: int = 256
    name: str = "rocke_gfx1250_qwen3_greedy_sampler"

    def __post_init__(self) -> None:
        if self.logits_dtype not in ("f32", "fp32", "bf16", "fp16", "f16"):
            raise ValueError("logits_dtype must be f32/bf16/fp16")
        if self.block_size & (self.block_size - 1):
            raise ValueError("block_size must be a power of two")

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        return kernel_name_join(self.name, self.logits_dtype, f"bs{self.block_size}")


def qwen3_greedy_sampler_grid(
    num_tokens: int, spec: Qwen3GreedySamplerSpec
) -> Tuple[int, int, int]:
    """One workgroup per token row."""
    return (num_tokens, 1, 1)


def qwen3_greedy_sampler_signature(spec: Qwen3GreedySamplerSpec):
    from ...helpers.spec import SignatureBuilder

    dt = "fp16" if spec.logits_dtype in ("fp16", "f16") else spec.logits_dtype
    dt = "f32" if dt == "fp32" else dt
    return (
        SignatureBuilder()
        .ptr("logits", dt)
        .ptr("out_ids", "i32")
        .scalar("vocab", "i32")
        .build()
    )


def build_qwen3_greedy_sampler(
    spec: Qwen3GreedySamplerSpec, *, arch: str = "gfx1250"
) -> KernelDef:
    ok, reason = _require_supported(arch)
    if not ok:
        raise NotImplementedError(reason)

    dt = _dtype_ir(spec.logits_dtype)
    bs = spec.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = bs

    logits = b.param(
        "logits", PtrType(dt, "global"), noalias=True, readonly=True, align=16
    )
    out_ids = b.param(
        "out_ids", PtrType(I32, "global"), noalias=True, writeonly=True, align=16
    )
    vocab = b.param("vocab", I32)

    lds_val = b.smem_alloc(F32, [bs], name_hint="samp_val")
    lds_idx = b.smem_alloc(F32, [bs], name_hint="samp_idx")

    token = b.block_id_x()
    tid = b.thread_id_x()
    c_bs = b.const_i32(bs)
    row_base = b.mul(token, vocab)
    neg_inf = b.const_f32(-3.0e38)

    # ---- per-thread strided argmax over the vocab ----
    for_op = b.scf_for_iter(
        tid, vocab, c_bs, [("m", neg_inf), ("i", b.const_i32(0))], iv_name="v"
    )
    with for_op as (v, iter_vars):
        m, i = iter_vars[0], iter_vars[1]
        lg = b.global_load(logits, b.add(row_base, v), dt, align=2)
        lg_f = lg if dt == F32 else b.cast_to_f32(lg)
        better = b.fcmp("olt", m, lg_f)
        b.scf_yield(b.select(better, lg_f, m), b.select(better, v, i))
    local_m = for_op.results[0]
    local_i = for_op.results[1]

    # ---- block argmax (deterministic lowest-index tie-break) ----
    _, arg = block_lds_reduce_with_index(
        b, local_m, local_i, lds_val, lds_idx, tid, block_size=bs, combine="argmax"
    )

    with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
        b.global_store(out_ids, token, arg, align=4)

    return b.kernel
