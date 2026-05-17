# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Top-K + softmax kernel (CK Tile ``09_topk_softmax`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/09_topk_softmax``. Given
``X`` of shape ``(M, N)`` (typically MoE router logits where ``M`` is
tokens and ``N`` is the number of experts), the kernel produces:

* ``Y`` : ``(M, K)`` — softmax probabilities over the K selected entries.
* ``Idx`` : ``(M, K)`` — i32 indices of the K selected entries.

Algorithm (per row, K iterations of *find + mask + softmax*):

  1. Each lane scans its slice of the row, tracking ``(local_max_val,
     local_max_idx)``.
  2. Block-LDS reduce ``max(local_max_val)`` to get the global max for
     the row.
  3. The lane(s) whose ``local_max_val`` matches the global max write
     their ``local_max_idx`` into an LDS slot; on tie, the last writer
     wins (any of the matching indices is a correct argmax).
  4. Thread 0 records ``(global_max_val, winning_idx)`` as the
     ``pick_k``-th entry.
  5. Each lane that owns ``winning_idx`` masks it out in its local
     cache (set to ``-inf``) so the next iteration skips it.
  6. After K iterations, softmax over the K picked values is computed
     in a single block-LDS reduce: ``Y[m, k] = exp(picked[k] - vmax) / sum_j(exp(picked[j] - vmax))``.

The cache lives in per-thread f32 registers (size
``ceil(N / block_size)``); the kernel reads the entire row from HBM
exactly once into the cache and never touches it again.

What we cover today:

* Input dtype : ``f16`` / ``bf16`` / ``f32`` (auto-promoted to f32
  for the compute pipeline).
* Output dtype : ``f16`` / ``bf16`` / ``f32``.
* ``K <= 32`` and ``K <= N`` (the kernel raises otherwise).
* ``block_size in {32, 64, 128, 256}`` with the constraint
  ``N <= block_size * 64`` (so the per-thread cache stays bounded).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import F32, I32, IRBuilder, KernelDef, PtrType
from ..helpers.io import io_ir_type
from ..helpers.reduction import block_lds_reduce
from ..helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join
from ..helpers.tensor_view import make_lds_view


DType = Literal["f16", "bf16", "f32"]


_NEG_INF_F32 = -3.4028234663852886e38


@dataclass(frozen=True)
class TopkSoftmaxSpec:
    """One concrete topk-softmax kernel configuration."""

    n_per_row: int  # N — entries per row (experts for MoE)
    k: int  # K — top-k count
    dtype: DType = "f32"  # input X dtype
    out_dtype: DType = "f32"  # output Y dtype
    block_size: int = 64
    name: str = "ck_dsl_topk_softmax"

    @property
    def elems_per_thread(self) -> int:
        # We round up so the kernel handles N values that don't divide
        # block_size by zero-padding (the masked-out tail elements
        # carry ``-inf`` and never win an argmax).
        return (self.n_per_row + self.block_size - 1) // self.block_size

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            self.out_dtype,
            f"N{self.n_per_row}",
            f"K{self.k}",
            f"b{self.block_size}",
        )


def is_valid_spec(spec: TopkSoftmaxSpec) -> Tuple[bool, str]:
    if spec.dtype not in ("f16", "bf16", "f32"):
        return False, f"unsupported dtype {spec.dtype!r}"
    if spec.out_dtype not in ("f16", "bf16", "f32"):
        return False, f"unsupported out_dtype {spec.out_dtype!r}"
    if spec.k <= 0:
        return False, f"k must be > 0 (got {spec.k})"
    if spec.k > 32:
        return False, f"k must be <= 32 (got {spec.k})"
    if spec.k > spec.n_per_row:
        return False, f"k ({spec.k}) > n_per_row ({spec.n_per_row})"
    if spec.block_size not in (32, 64, 128, 256):
        return False, (f"block_size {spec.block_size} not in {{32, 64, 128, 256}}")
    if spec.elems_per_thread > 64:
        return False, (
            f"elems_per_thread {spec.elems_per_thread} > 64; pick a larger block_size"
        )
    return True, "ok"


def _scalar_load_f32(b: IRBuilder, ptr, idx, *, dtype: str):
    """Promote a typed scalar load to f32 regardless of source dtype."""
    if dtype == "f32":
        return b.global_load_f32(ptr, idx)
    if dtype == "f16":
        return b.cast_to_f32(b.global_load_f16(ptr, idx))
    if dtype == "bf16":
        return b.cast_to_f32(b.global_load_bf16(ptr, idx))
    raise ValueError(f"unsupported dtype {dtype!r}")


def _scalar_store_from_f32(b: IRBuilder, ptr, idx, value_f32, *, dtype: str):
    """Demote an f32 to the output dtype + scalar store."""
    if dtype == "f32":
        b.global_store(ptr, idx, value_f32, align=4)
        return
    if dtype in ("f16", "bf16"):
        from ..helpers.io import io_ir_type as _io_ty

        b.global_store(ptr, idx, b.cast_f32_to(value_f32, _io_ty(dtype)))
        return
    raise ValueError(f"unsupported dtype {dtype!r}")


def build_topk_softmax(spec: TopkSoftmaxSpec) -> KernelDef:
    """Build the IR for one topk-softmax instance.

    Kernel signature::

        (X: ptr<dtype, global>,         # (M, N) input logits
         Y: ptr<out_dtype, global>,     # (M, K) softmax-of-top-k values
         Idx: ptr<i32, global>,         # (M, K) indices
         M: i32, N: i32)

    Grid: ``(M, 1, 1)`` — one CTA per row.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid topk_softmax spec: {why}")

    N = spec.n_per_row
    K = spec.k
    BS = spec.block_size
    epot = spec.elems_per_thread

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param(
        "X",
        PtrType(io_ir_type(spec.dtype) if spec.dtype != "f32" else F32, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    Y = b.param(
        "Y",
        PtrType(
            io_ir_type(spec.out_dtype) if spec.out_dtype != "f32" else F32, "global"
        ),
        noalias=True,
        writeonly=True,
        align=16,
    )
    Idx = b.param("Idx", PtrType(I32, "global"), noalias=True, writeonly=True, align=4)
    M = b.param("M", I32)  # noqa: F841 — ABI symmetry with CK Tile
    _ = b.param("N", I32)  # noqa: F841 — validated by caller, equals n_per_row

    tid = b.thread_id_x()
    row = b.block_id_x()

    # Compute the base offset for this row's slice of X: ``row * N``.
    c_N = b.const_i32(N)
    row_base = b.mul(row, c_N)
    c_BS = b.const_i32(BS)
    c_neg_inf = b.const_f32(_NEG_INF_F32)

    # Per-thread cache: load this thread's slice of the row into f32
    # registers (one register per element). We pre-fill the tail with
    # ``-inf`` so any out-of-bounds index never wins an argmax, even
    # before the explicit mask sequence below kicks in.
    cache: list = []  # f32 values
    cache_idx: list = []  # i32 indices into the row (for argmax write-back)
    for e in range(epot):
        local_idx = b.add(tid, b.mul(c_BS, b.const_i32(e)))
        in_bounds = b.cmp_lt(local_idx, c_N)
        # ``masked_global_load`` clamps the index when the mask is
        # false and substitutes ``-inf`` for the result. Avoids a
        # spurious OOB load for rows where ``N`` isn't a clean
        # multiple of ``block_size``.
        loaded = b.masked_global_load(
            X,
            b.add(row_base, local_idx),
            in_bounds,
            c_neg_inf
            if spec.dtype == "f32"
            else b.cast_f32_to(c_neg_inf, io_ir_type(spec.dtype)),
            io_ir_type(spec.dtype) if spec.dtype != "f32" else F32,
        )
        if spec.dtype == "f32":
            v_f32 = loaded
        else:
            v_f32 = b.cast_to_f32(loaded)
        cache.append(v_f32)
        cache_idx.append(local_idx)

    # LDS scratch: one ``BS``-sized f32 buffer for the block reductions
    # plus one i32 slot for the per-iteration argmax winner.
    lds_red = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_red").base
    # ``lds_winner`` holds the argmax idx for the current iteration.
    # We allocate it as f32 (one slot) so we can use the existing
    # ``smem_*_vN_f32`` ops; the value is a bitcast of an i32 and
    # the AMDGPU backend emits the same ``ds_write_b32`` /
    # ``ds_read_b32`` it would for a typed i32 LDS.
    lds_winner = make_lds_view(b, dtype=F32, shape=(1,), name_hint="lds_winner").base

    # K iterations of pick-max + mask. ``picks_val[k]`` / ``picks_idx[k]``
    # are the SSA values we'll feed into the softmax at the end.
    picks_val: list = []
    picks_idx: list = []

    for pick_k in range(K):
        # 1) Per-thread local max + arg.
        local_max = c_neg_inf
        local_arg = b.const_i32(-1)
        for e in range(epot):
            is_greater = b.fcmp("ogt", cache[e], local_max)
            local_max = b.select(is_greater, cache[e], local_max)
            local_arg = b.select(is_greater, cache_idx[e], local_arg)

        # 2) Block-LDS reduce the value.
        global_max = block_lds_reduce(
            b, local_max, lds_red, tid, block_size=BS, combine="max"
        )

        # 3) Argmax write-back: matching threads race-write into the
        # winner LDS slot. The race is benign (any matching idx is a
        # valid argmax); the barrier below makes the winning value
        # globally visible. The IR's ``smem_*_vN`` path is wired for
        # f16/bf16/f32 only, so we route i32 through the f32 LDS path
        # via bitcast -- exactly the same 32-bit ``ds_write_b32`` /
        # ``ds_read_b32`` on the backend.
        matches = b.fcmp("oeq", local_max, global_max)
        with b.scf_if(matches):
            arg_as_f32 = b.bitcast(local_arg, F32)
            b.smem_store_vN_f32(lds_winner, [b.const_i32(0)], arg_as_f32, 1)
        b.sync()
        winner_vec_f32 = b.smem_load_vN_f32(lds_winner, b.const_i32(0), n=1)
        winner_idx = b.bitcast(b.vec_extract(winner_vec_f32, 0), I32)

        picks_val.append(global_max)
        picks_idx.append(winner_idx)

        # 4) Mask out the winning element for the next iteration:
        # every lane checks each of its cached indices against the
        # winning idx and overwrites the matching slot with ``-inf``.
        for e in range(epot):
            owns = b.cmp_eq(cache_idx[e], winner_idx)
            cache[e] = b.select(owns, c_neg_inf, cache[e])

    # 5) Softmax over the K picked values. ``picks_val[0]`` is already
    # the largest (we picked it first), so we use it as the numeric-
    # stability max; the subsequent picks_val[k] are subtracted before
    # the exp.
    vmax = picks_val[0]
    LN2_E = b.const_f32(1.4426950408889634)  # log2(e)
    exps = []
    for k in range(K):
        s = b.fmul(b.fsub(picks_val[k], vmax), LN2_E)
        exps.append(b.exp2(s))
    s_sum = exps[0]
    for k in range(1, K):
        s_sum = b.fadd(s_sum, exps[k])
    inv_sum = b.rcp(s_sum)

    # 6) Per-row write of the softmaxed values + indices. Only thread
    # 0 emits the writes — the K-wide store fans out to ``K`` scalar
    # writes (K <= 32 by spec), which is cheap enough that we don't
    # bother distributing across lanes.
    with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
        c_K = b.const_i32(K)
        row_out_base = b.mul(row, c_K)
        for k in range(K):
            y_f32 = b.fmul(exps[k], inv_sum)
            out_off = b.add(row_out_base, b.const_i32(k))
            _scalar_store_from_f32(b, Y, out_off, y_f32, dtype=spec.out_dtype)
            b.global_store(Idx, out_off, picks_idx[k], align=4)

    # ``_scalar_load_f32`` is currently unused (we route through
    # ``masked_global_load`` above); keep the import alive so the
    # helper stays in the public API for future variants that don't
    # need OOB masking.
    _ = _scalar_load_f32  # noqa: F841

    return b.kernel


def topk_softmax_grid(m: int, spec: TopkSoftmaxSpec) -> Tuple[int, int, int]:
    """Return the launch grid: one CTA per row."""
    return ceil_div_grid((m, 1))


def topk_softmax_signature(spec: TopkSoftmaxSpec):
    return (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Y", spec.out_dtype)
        .ptr("Idx", "i32")
        .scalar("M", "i32")
        .scalar("N", "i32")
        .build()
    )


__all__ = [
    "TopkSoftmaxSpec",
    "build_topk_softmax",
    "is_valid_spec",
    "topk_softmax_grid",
    "topk_softmax_signature",
]
