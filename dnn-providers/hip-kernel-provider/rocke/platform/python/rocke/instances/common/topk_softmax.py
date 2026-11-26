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
  2. Each lane packs ``(local_max_val, local_max_idx)`` into one i64
     **argmax key** whose signed order is the argmax order: a larger value
     wins and a tie goes to the smaller index. A **wave-XOR butterfly**
     takes the largest key across the wave (``log2(min(BS, wave_size))``
     stages; the in-row stages on DPP, the rest on ``ds_swizzle`` /
     ``ds_bpermute``). Mirrors CK Tile's ``BlockTopkStream2D``
     ``ArgmaxPacket`` reduce and AITER's ``multithread_reduce(arg_max,
     THREADS_PER_ROW)``. In a block of more than one wave each wave's
     leader stores its key to LDS and every lane takes the largest of the
     waves' keys.
  3. Each lane that owns ``winning_idx`` masks it out in its local
     cache (set to ``-inf``) so the next iteration skips it.
  4. After K iterations, softmax over the K picked values is computed
     per-thread in registers (the picks are wave-broadcast):
     ``Y[m, k] = exp(picked[k] - vmax) / sum_j(exp(picked[j] - vmax))``.
  5. Output writes are **distributed across K lanes**: lane ``k``
     stores ``(Y[m, k], Idx[m, k])`` instead of serialising K writes
     on thread 0.

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
import functools
from typing import Literal, Tuple

from ...core.ir import (
    F32,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
)
from ...helpers.distribution import (
    TileDistribution,
    TileDistributionEncoding,
    make_static_tile_distribution,
)
from ...helpers.io import io_ir_type
from ...helpers.reduction import butterfly_allreduce
from ...helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join
from ...helpers.tensor_view import make_lds_view


def _ilog2(n: int) -> int:
    """Floor log2 for power-of-two ``n``; raises if ``n`` is not a power of 2."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"_ilog2 expects a positive power of 2, got {n}")
    out = 0
    while (1 << out) < n:
        out += 1
    return out


# An argmax key packs a candidate ``(val, idx)`` into one i64 whose signed
# order is the argmax order the kernel documents: a larger value wins and,
# among equal values, the smaller index wins. Taking the largest key is then
# one integer compare, the same in every lane and every reduction order.
#
#   key = (ordered(val) << 32) | (idx ^ 0x7FFFFFFF)
#
# ``ordered`` reads the f32 bits as a two's-complement integer that orders
# like the value: a non-negative float keeps its bits, a negative one becomes
# ``INT32_MIN - bits`` (minus its magnitude), so ``-0.0`` and ``+0.0`` both
# map to 0 and tie, as they do under ``fcmp oeq``. This matches the local
# scan's ``fcmp`` order because the kernel keeps f32 subnormals (IEEE mode):
# under flush-to-zero ``fcmp`` would tie a subnormal with zero, the key
# would not. The low word ``idx ^
# 0x7FFFFFFF``, read unsigned, reverses the signed order of ``idx`` over the
# whole i32 range (the local scan's ``-1`` sentinel included).
_INT32_MIN = -(1 << 31)
_INDEX_REVERSE_MASK = (1 << 31) - 1


def _argmax_key(b: IRBuilder, val: Value, idx: Value) -> Value:
    """The i64 argmax key of f32 ``val`` and i32 ``idx``. ``val`` must not be
    NaN: the local scan's ``ogt`` never selects one."""
    bits = b.bitcast(val, I32)
    is_negative = b.cmp_lt(bits, b.const_i32(0))
    ordered = b.select(is_negative, b.sub(b.const_i32(_INT32_MIN), bits), bits)
    high = b.shl(b.sext(ordered, I64), b.const_i64(32))
    low = b.zext(b.xor(idx, b.const_i32(_INDEX_REVERSE_MASK)), I64)
    return b.lor(high, low)


def _argmax_key_unpack(b: IRBuilder, key: Value) -> tuple[Value, Value]:
    """``(val, idx)`` of an argmax key, inverting :func:`_argmax_key`; a
    ``-0.0`` value comes back as ``+0.0``."""
    ordered = b.trunc(b.lshr(key, b.const_i64(32)), I32)
    is_negative = b.cmp_lt(ordered, b.const_i32(0))
    bits = b.select(is_negative, b.sub(b.const_i32(_INT32_MIN), ordered), ordered)
    idx = b.xor(b.trunc(key, I32), b.const_i32(_INDEX_REVERSE_MASK))
    return b.bitcast(bits, F32), idx


def _larger_key(b: IRBuilder, key: Value, other: Value) -> Value:
    """The larger of two argmax keys (signed compare)."""
    return b.select(b.cmp_gt(other, key), other, key)


def _lds_argmax_key_fold(b: IRBuilder, lds_keys: Value, count: int) -> Value:
    """The largest of ``count`` argmax keys in LDS. Every lane reads every
    slot, so every lane holds the result."""

    def load_key(slot: int) -> Value:
        stored = b.smem_load_vN(lds_keys, b.const_i32(slot), dtype=I64, n=1)
        return b.vec_extract(stored, 0)

    best = load_key(0)
    for slot in range(1, count):
        best = _larger_key(b, best, load_key(slot))
    return best


def make_input_distribution(
    *, block_size: int, elems_per_thread: int
) -> TileDistribution:
    """Warp-per-row input tile distribution (CK Tile topk_softmax
    ``MakeInputDistribution``).

    The row's ``N`` logits are decomposed ``(IssuesPerThread, block_size)``
    over a single X dim: the lane id (P0) owns the ``block_size``
    (ThreadPerWarp + WarpPerBlock collapsed) level and the per-thread
    issue count lives in the Y tile. :meth:`TileDistribution.calculate_x`
    then reconstructs ``local_idx = e * block_size + tid`` -- exactly the
    legacy hand-rolled ``tid + BS*e`` lane decode -- so the cache element
    order and the global addresses (still gated by the explicit
    :func:`IRBuilder.masked_global_load` OOB mask for ``N`` not a clean
    multiple of ``block_size``) are unchanged.
    """
    encoding = TileDistributionEncoding(
        # X0 = N = (IssuesPerThread, block_size).
        Hs=((elems_per_thread, block_size),),
        # P0 = lane id -> X0 level 1 (the block_size dim).
        Ps2RHs_major=((1,),),
        Ps2RHs_minor=((1,),),
        # Y0 -> X0 level 0 (the per-thread issue / Repeat dim).
        Ys2RHs_major=(1,),
        Ys2RHs_minor=(0,),
    )
    return make_static_tile_distribution(encoding)


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
    name: str = "rocke_topk_softmax"

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


def is_valid_spec(spec: TopkSoftmaxSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    from ...core.arch import ArchTarget

    try:
        ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
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
        from ...helpers.io import io_ir_type as _io_ty

        b.global_store(ptr, idx, b.cast_f32_to(value_f32, _io_ty(dtype)))
        return
    raise ValueError(f"unsupported dtype {dtype!r}")


def build_topk_softmax(spec: TopkSoftmaxSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one topk-softmax instance.

    Kernel signature::

        (X: ptr<dtype, global>,         # (M, N) input logits
         Y: ptr<out_dtype, global>,     # (M, K) softmax-of-top-k values
         Idx: ptr<i32, global>,         # (M, K) indices
         M: i32, N: i32)

    Grid: ``(M, 1, 1)`` — one CTA per row.

    ``arch`` selects the target's :class:`ArchTarget` (wave size). Each
    wave reduces its lanes' argmax keys with the wave-XOR butterfly; when
    the row spans more than one wave (``block_size > wave_size``) the
    waves' keys meet in LDS. On wave32 targets (e.g. gfx1151) a
    ``block_size == 64`` row therefore spans two waves and takes the LDS
    step: the butterfly never reaches across waves.
    """
    from ...core.arch import ArchTarget

    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid topk_softmax spec: {why}")

    wave_size = ArchTarget.from_gfx(arch).wave_size

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
    c_neg_inf = b.const_f32(_NEG_INF_F32)

    # Per-thread cache: load this thread's slice of the row into f32
    # registers (one register per element). We pre-fill the tail with
    # ``-inf`` so any out-of-bounds index never wins an argmax, even
    # before the explicit mask sequence below kicks in.
    #
    # CK-Tile adoption: the per-element column index ``local_idx`` is
    # reconstructed from the warp-per-row input distribution
    # (:func:`make_input_distribution`) instead of the hand-rolled
    # ``tid + BS*e`` lane decode. ``calculate_x`` emits
    # ``e*block_size + tid`` (integer add is commutative, so the address
    # is identical); the explicit OOB mask below is retained because
    # ``load_tile`` has no masked path and ``N`` need not divide
    # ``block_size``.
    in_dist = make_input_distribution(block_size=BS, elems_per_thread=epot)
    cache: list = []  # f32 values
    cache_idx: list = []  # i32 indices into the row (for argmax write-back)
    for e in range(epot):
        (local_idx,) = in_dist.calculate_x(b, ys=[b.const_i32(e)], ps=[[tid]])
        in_bounds = b.cmp_lt(local_idx, c_N)
        # ``masked_global_load`` clamps the index when the mask is
        # false and substitutes ``-inf`` for the result. Avoids a
        # spurious OOB load for rows where ``N`` isn't a clean
        # multiple of ``block_size``.
        loaded = b.masked_global_load(
            X,
            b.add(row_base, local_idx),
            in_bounds,
            (
                c_neg_inf
                if spec.dtype == "f32"
                else b.cast_f32_to(c_neg_inf, io_ir_type(spec.dtype))
            ),
            io_ir_type(spec.dtype) if spec.dtype != "f32" else F32,
        )
        if spec.dtype == "f32":
            v_f32 = loaded
        else:
            v_f32 = b.cast_to_f32(loaded)
        cache.append(v_f32)
        cache_idx.append(local_idx)

    # Each wave takes its lanes' largest argmax key with the wave-XOR
    # butterfly: registers only, no LDS round trip and no barrier. A row of
    # more than one wave then has each wave's leader lane store the wave's key
    # to LDS, and after a barrier every lane takes the largest of the
    # ``BS / wave_size`` keys. A second barrier keeps the next pick's stores
    # off the slots other waves may still be reading. On wave32 a ``BS == 64``
    # row is two waves, so it takes the LDS step (a ``ds_bpermute`` for the
    # cross-wave XOR mask only addresses one wave).
    num_waves = max(BS // wave_size, 1)
    intra_stages = _ilog2(min(BS, wave_size))
    if num_waves > 1:
        c_wave_size = b.const_i32(wave_size)
        wave = b.div(tid, c_wave_size)
        is_wave_leader = b.cmp_eq(b.mod(tid, c_wave_size), b.const_i32(0))
        lds_wave_keys = make_lds_view(
            b, dtype=I64, shape=(num_waves,), name_hint="lds_wave_keys"
        ).base

    # K iterations of pick-max + mask. ``picks_val[k]`` /
    # ``picks_idx[k]`` are the wave-broadcast SSA values we feed
    # into the softmax at the end. After the reduction every lane
    # holds the same (val, idx) so the K output writes can fan out
    # across K lanes (step 6) regardless of which path we took.
    picks_val: list = []
    picks_idx: list = []

    for pick_k in range(K):
        # 1) Per-thread local argmax over this lane's cache slice.
        # The cache loop is iterated in ascending ``local_idx``
        # order, so ``ogt`` (strict) naturally keeps the smaller
        # idx on ties — no explicit tie-break here. Across lanes the
        # argmax key carries the tie-break (smaller idx wins).
        local_max = c_neg_inf
        local_arg = b.const_i32(-1)
        for e in range(epot):
            is_greater = b.fcmp("ogt", cache[e], local_max)
            local_max = b.select(is_greater, cache[e], local_max)
            local_arg = b.select(is_greater, cache_idx[e], local_arg)

        # 2) The block's largest argmax key, in every lane.
        key = butterfly_allreduce(
            b,
            _argmax_key(b, local_max, local_arg),
            functools.partial(_larger_key, b),
            stages=intra_stages,
        )
        if num_waves > 1:
            with b.scf_if(is_wave_leader):
                b.smem_store_vN(lds_wave_keys, [wave], key, 1)
            b.sync()
            key = _lds_argmax_key_fold(b, lds_wave_keys, num_waves)
            b.sync()
        global_max, winner_idx = _argmax_key_unpack(b, key)

        picks_val.append(global_max)
        picks_idx.append(winner_idx)

        # 3) Mask out the winning element for the next iteration:
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

    # 6) Per-row write of the softmaxed values + indices. We
    # **distribute the K stores across K lanes**: lane ``k`` writes
    # ``(Y[m, k], Idx[m, k])``. Since ``picks_val`` / ``picks_idx``
    # are block-broadcast (every lane holds the same key after step 2),
    # each lane's k-th SSA value is correct. ``K <= 32 <= BS`` by spec,
    # so the K writes always fit in distinct lanes.
    c_K = b.const_i32(K)
    row_out_base = b.mul(row, c_K)
    for k in range(K):
        with b.scf_if(b.cmp_eq(tid, b.const_i32(k))):
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
