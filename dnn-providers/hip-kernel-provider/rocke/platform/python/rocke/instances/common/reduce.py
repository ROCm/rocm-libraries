# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Row-wise reduction kernel instance builder (CK Tile ``05_reduce`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/05_reduce``. For each row
of an ``(M, N)`` tensor produces one scalar per row by applying:

    sum :  Y[m] = sum_n(X[m,n])
    max :  Y[m] = max_n(X[m,n])
    mean:  Y[m] = sum_n(X[m,n]) / N

Compute is in f32 internally; the output dtype matches the input dtype
(f16 in / f16 out, etc.).

CK Tile parity shape (``BlockReduce2d{Sync, CrossWarpSync}`` in
:file:`include/ck_tile/ops/reduce/block/block_reduce2d.hpp`):

* **Stage 1** -- thread-level: each thread sweeps its ``vec``-wide
  chunks and folds them into one f32 register. The per-chunk fold uses
  a balanced binary tree (depth ``log2(vec)``) instead of the natural
  left-fold chain so the back-to-back ``fadd`` / ``fmax`` ops can
  pipeline.
* **Stage 2** -- warp-level: an XOR butterfly via ``ds_swizzle_xor`` /
  ``ds_bpermute`` (six stages for wave64) combines partials within one
  wave. This is CK Tile's ``block_tile_reduce_xor_sync``: every lane
  in the wave ends up holding the wave-local reduction.
* **Stage 3** -- cross-warp: each wave's lane 0 writes its partial to
  a ``num_warps``-slot LDS scratch, one ``sync``, then every thread
  loads all ``num_warps`` entries and tree-combines them. For BS=256
  / wave64 this replaces an 8-round LDS tree (8 syncs) with a single
  ``sync`` + 4-way combine.

For block sizes that don't fit the wave-aligned shape we fall back to
:func:`block_lds_reduce` (the canonical full LDS tree).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Value
from ...helpers.distribution import (
    TileDistributionEncoding,
    block_tile_reduce_sync,
    make_reduce_tile_distribution_encoding,
    make_static_distributed_tensor,
    make_static_tile_distribution,
)
from ...helpers.io import io_ir_type, store_scalar_from_f32
from ...helpers.reduction import (
    block_lds_reduce,
    block_lds_reduce_with_wave_prologue,
    tree_reduce,
)
from ...helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ...helpers.sweep import sweep_row_chunks
from ...helpers.tensor_view import (
    make_lds_view,
    make_naive_tensor_view_packed,
    make_tile_window,
)


DType = Literal["f16", "bf16"]
ReduceOp = Literal["sum", "max", "min", "mean", "prod"]


@dataclass(frozen=True)
class Reduce2DSpec:
    """One row-reduction instance."""

    n_per_block: int
    op: ReduceOp = "sum"
    block_size: int = 256
    vec: int = 4
    dtype: DType = "f16"
    wave_size: int = 64
    name: str = "rocke_reduce2d"

    @property
    def elems_per_thread(self) -> int:
        return self.n_per_block // self.block_size

    @property
    def num_warps(self) -> int:
        """``block_size / wave_size`` -- waves per CTA for the CK Tile
        ``BlockReduce2dCrossWarpSync`` stage."""
        return self.block_size // self.wave_size

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.op,
            self.dtype,
            f"N{self.n_per_block}",
            f"b{self.block_size}",
            f"v{self.vec}",
        )


def is_valid_spec(spec: Reduce2DSpec) -> Tuple[bool, str]:
    if spec.op not in ("sum", "max", "min", "mean", "prod"):
        return False, f"unsupported op {spec.op!r}"
    return validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=spec.n_per_block,
        )
    )


_NEG_INF_F32 = -3.4028234663852886e38
_POS_INF_F32 = 3.4028234663852886e38


def _combine_scalar(
    b: IRBuilder,
    combine: Literal["sum", "max", "min", "prod"],
    a: Value,
    c: Value,
) -> Value:
    """One step of the per-element reduction combiner in f32.

    Centralised so the per-chunk tree fold, the warp XOR butterfly,
    and the cross-warp LDS tree share the exact same op selection.
    """
    if combine == "sum":
        return b.fadd(a, c)
    if combine == "max":
        return b.fmax(a, c)
    if combine == "min":
        return b.fmin(a, c)
    if combine == "prod":
        return b.fmul(a, c)
    raise ValueError(f"unsupported combine {combine!r}")


def _make_row_reduce_distribution(spec: Reduce2DSpec):
    """Reduce distribution for the per-row N collapse (CK Tile reduce2d).

    Builds the rank-4 hierarchical X encoding for the reduce axis,
    ``Hs = (Repeat, WarpPerBlock, ThreadPerWarp, Vector)``, then folds the
    reduce dim via :func:`make_reduce_tile_distribution_encoding`. The two
    kept Y dims (Repeat + Vector) are consumed by the reduce, so they
    disappear and the per-thread register tile is a single keep-axis slot;
    the ``WarpPerBlock`` and ``ThreadPerWarp`` levels become R replication
    levels driving the cross-warp LDS and warp-XOR butterfly respectively.

    Only ``WarpPerBlock`` (= ``num_warps``) and ``ThreadPerWarp``
    (= ``wave_size``) feed the butterfly plan; ``Repeat`` / ``Vector`` are
    set to 1 because the per-thread partial is already folded into the seed
    slot and :func:`block_tile_reduce_sync` never reconstructs the X
    coordinate.
    """
    nwarp = spec.block_size // spec.wave_size
    tpw = spec.wave_size
    encoding = TileDistributionEncoding(
        # (Repeat, WarpPerBlock, ThreadPerWarp, Vector)
        Hs=((1, nwarp, tpw, 1),),
        # P0 = warp id -> WarpPerBlock (level 1); P1 = lane id -> ThreadPerWarp
        # (level 2). idim_p_lane = NDimP - 1 = P1 (lane), idim_p_warp = 0.
        Ps2RHs_major=((1,), (1,)),
        Ps2RHs_minor=((1,), (2,)),
        # Y dims target the Repeat (level 0) and Vector (level 3) of X0 -- both
        # part of the reduced axis, so they fold away.
        Ys2RHs_major=(1, 1),
        Ys2RHs_minor=(0, 3),
    )
    reduce_enc = make_reduce_tile_distribution_encoding(encoding, [0])
    return make_static_tile_distribution(reduce_enc)


def build_reduce2d(spec: Reduce2DSpec) -> KernelDef:
    """Build the IR for one row-reduction instance.

    Kernel signature: ``(X: ptr, Y: ptr, M: i32, N: i32)``.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid reduce2d spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    BS, VEC, N = spec.block_size, spec.vec, spec.n_per_block

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)  # noqa: F841
    _ = b.param("N", I32)  # noqa: F841

    tid = b.thread_id_x()
    row = b.block_id_x()

    # CK Tile-style: make_naive_tensor_view_packed(X, (1, N)) gives us a
    # packed row-major view; make_tile_window pins the origin to ``row``
    # so the sweep below indexes within a single row.
    x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
    x_tile = make_tile_window(x_view, lengths=(1, N), origin=(row, b.const_i32(0)))

    lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_red").base

    # Pick the f32 identity element + the combiner for each op.
    if spec.op in ("sum", "mean"):
        acc = b.const_f32(0.0)
        combine: Literal["sum", "max", "min", "prod"] = "sum"
    elif spec.op == "max":
        acc = b.const_f32(_NEG_INF_F32)
        combine = "max"
    elif spec.op == "min":
        acc = b.const_f32(_POS_INF_F32)
        combine = "min"
    elif spec.op == "prod":
        acc = b.const_f32(1.0)
        combine = "prod"
    else:  # validated earlier; defensive
        raise ValueError(f"unsupported reduce op {spec.op!r}")

    # sweep_row_chunks plays the role of CK Tile's ``sweep_tile``: it
    # invokes ``body(n_off, x_scalars)`` once per per-thread chunk and
    # threads the f32-promoted lane scalars in. The per-chunk fold is
    # a balanced binary tree (depth ``log2(VEC)``) joined onto ``acc``
    # via one trailing combine -- shorter latency chain than the
    # natural left fold, same op count.
    def body(_n_off, x_scalars):
        nonlocal acc
        chunk_partial = tree_reduce(
            b, lambda a, c: _combine_scalar(b, combine, a, c), list(x_scalars)
        )
        acc = _combine_scalar(b, combine, acc, chunk_partial)

    sweep_row_chunks(
        b,
        x_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        elems_per_thread=spec.elems_per_thread,
        body=body,
    )

    # Cross-thread reduction. The wave-aligned path drives the CK Tile
    # ``BlockReduce2d{Sync,CrossWarpSync}`` distribution collapse directly:
    # we build the X distribution for the reduce axis (rank-4 hierarchical
    # ``Hs = (Repeat, WarpPerBlock, ThreadPerWarp, Vector)``), fold it into a
    # *reduce distribution* via
    # :func:`make_reduce_tile_distribution_encoding` (every H level not
    # consumed by a kept Y becomes an R replication level), and let
    # :func:`block_tile_reduce_sync` run the warp XOR butterfly over the
    # lane-owned R (``ThreadPerWarp``) followed by the cross-warp LDS over
    # the warp-owned R (``WarpPerBlock``). The per-thread ``acc`` is already
    # the fold over this thread's ``Repeat``/``Vector`` slots, so it seeds
    # the single keep-axis register slot. This emits the exact same six
    # shuffle stages + one ``sync`` the hand-rolled
    # ``block_lds_reduce_with_wave_prologue`` did, but the lane/warp split is
    # now derived from the encoding rather than spelled out by hand.
    #
    # The fallback is the canonical full LDS tree (kept for any future BS
    # that isn't a clean multiple of ``wave_size``).
    if spec.block_size % spec.wave_size == 0 and combine in ("sum", "max"):
        red_dist = _make_row_reduce_distribution(spec)
        reduced = make_static_distributed_tensor(red_dist, dtype=F32)
        reduced.storage[0] = acc
        block_tile_reduce_sync(
            b,
            reduced,
            combine=combine,
            lds_buf=lds,
            tid=tid,
            wave_size=spec.wave_size,
        )
        total = reduced.storage[0]
    elif spec.block_size % spec.wave_size == 0:
        # min / prod aren't expressible by block_tile_reduce_sync (sum / max
        # only). Keep the hand-built wave-XOR prologue for those combiners.
        total = block_lds_reduce_with_wave_prologue(
            b,
            acc,
            lds,
            tid,
            block_size=spec.block_size,
            combine=combine,
            wave_size=spec.wave_size,
        )
    else:
        total = block_lds_reduce(b, acc, lds, tid, block_size=BS, combine=combine)

    if spec.op == "mean":
        total = b.fmul(total, b.rcp(b.const_f32(float(N))))

    with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
        store_scalar_from_f32(b, Y, row, total, dtype=spec.dtype)

    return b.kernel


def reduce2d_grid(m: int, spec: Reduce2DSpec) -> Tuple[int, int, int]:
    return ceil_div_grid((m, 1))


def reduce2d_signature(spec: Reduce2DSpec):
    return (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Y", spec.dtype)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .build()
    )
