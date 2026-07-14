# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""SmoothQuant kernel instance (CK Tile ``12_smoothquant`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/12_smoothquant``. For an
``(M, N)`` activation tensor ``X`` and a per-channel smooth scale
``SmScale`` of shape ``(N,)``, the kernel produces:

* ``QY`` : ``(M, N)`` quantised tensor (i8 / fp8e4m3 / bf8e5m2)
* ``YScale`` : ``(M,)`` per-row dynamic quantisation scale (fp32)

via the canonical two-pass row recipe::

 pass 1 (per row m):
 y_n = x_n * smscale_n # f32, n in [0, N)
 amax_local = max_n(|y_n|) # per-thread partial
 amax = block_lds_reduce(amax_local, max) # one f32 per row

 yscale_m = max(amax, eps) / quant_max # row dynamic scale
 inv_yscale = 1 / yscale_m

 pass 2 (per row m):
 qy_n = quantize(x_n * smscale_n, inv_yscale) # rounded + saturated

The compute layer is f32 (matches CK Tile's ``ComputeDataType``); the
``out_dtype`` selects both the clamp range and the rounding op
(:func:`rocke.helpers.quant.quantize_scalar_f32` handles both).

What we cover today:

* Input dtype: ``f16`` / ``bf16``.
* Output dtype: ``i8`` (the SmoothQuant default), ``fp8e4m3``,
 ``bf8e5m2``.
* Block shapes any ``block_size in {64, 128, 256, 512, 1024}`` with
 ``vec in {2, 4, 8}`` and ``elems_per_thread <= 64`` (the same
 guard rmsnorm2d uses to bound the per-thread cache size).
* ``save_yscale=True`` (default) emits the ``YScale`` write at
 ``tid == 0``; set to ``False`` for the "I already have a scale"
 variant.

Implementation notes:

* Pass 1 reads the activation X tile through the canonical CK-Tile
 ``MakeXBlockTileDistribution`` + :func:`load_tile` triad
 (:func:`make_row_x_distribution`) and caches the f32-promoted ``x``
 scalars in the returned :class:`StaticDistributedTensor`, so pass 2
 does not re-read HBM. SmScale is re-loaded in pass 2 — it lives in
 L1 by the second pass and the re-load costs ~free.
* The amax reduction reuses :func:`block_lds_reduce` with the existing
 ``"max"`` combiner (no new IR primitive needed). Per-chunk we build a
 *tree* of pairwise ``fmax`` so the per-element ``|y| = fmax(y, -y)``
 chain has ``O(log VEC)`` critical-path depth (vs the previous
 ``O(VEC)`` serial chain). The AMDGPU backend pattern-matches the
 paired ``fmax(fmax(a, -a), fmax(b, -b))`` form into ``v_max3_f32``
 with abs input modifiers when fast-math permits, exactly the
 ``UseMax3`` trick CK Tile's ``SmoothquantPipelineTwoPass`` uses.
* ``eps`` (passed as an f32 kernel arg) guards the
 ``yscale = amax / quant_max`` division against pathological rows
 where ``amax == 0``. Matches the CK Tile reference's ``eps`` arg.

Pass-2 vector-store recipe:

* For ``out_dtype in {"fp8e4m3", "bf8e5m2"}`` and ``VEC`` a multiple
  of 4, the per-chunk quantise uses :func:`IRBuilder.cvt_pk_fp8_f32x4`
  / :func:`IRBuilder.cvt_pk_bf8_f32x4` — one packed ``v_cvt_pk_fp8_f32``
  per four f32 lanes, matching AITER's ``scaled_fp8_conversion_vec``
  in ``csrc/include/quant_common.cuh`` (``q8x4_t`` vec4 store path).
  The hardware saturates on conversion so the explicit
  :func:`clamp_f32` from :func:`quantize_scalar_f32` is skipped.
* For ``out_dtype == "i8"`` (and for ``VEC == 2`` on every quant
  dtype), the per-element :func:`quantize_scalar_f32` call is kept
  (the IR has no packed ``v_cvt_pk_i8`` primitive today), but the
  resulting ``q`` scalars are packed into a ``<VEC x q_ty>`` via
  :func:`IRBuilder.vec_pack` and the whole chunk is stored as one
  i16/i32/i64 dword via :func:`IRBuilder.bitcast` +
  :func:`IRBuilder.global_store`. That collapses ``VEC`` 8-bit
  scalar stores into a single VMEM store per chunk — the same
  ``buffer_store_dword{,x2}`` epilogue width AITER's
  ``q8x4_t`` reinterpret_cast achieves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Value
from ...helpers.distribution import (
    TileDistribution,
    TileDistributionEncoding,
    load_tile,
    make_static_distributed_tensor,
    make_static_tile_distribution,
    store_tile,
)
from ...helpers.io import io_ir_type
from ...helpers.quant import (
    QDType,
    quant_ir_type,
    quant_max_abs,
    quantize_scalar_f32,
)
from ...helpers.reduction import block_lds_reduce
from ...helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ...helpers.tensor_view import (
    make_global_view,
    make_lds_view,
    make_naive_tensor_view_packed,
    make_tile_window,
)


# ---------------------------------------------------------------------
# Local helpers (kept inside this instance file to honour the
# "instance files only" optimisation scope).
# ---------------------------------------------------------------------


def _tree_fmax(b: IRBuilder, values: List[Value]) -> Value:
    """Reduce a non-empty list of f32 scalars to a single f32 via a
    balanced pairwise ``fmax`` tree.

    The critical-path depth is ``ceil(log2(len(values)))`` ``fmax``
    ops rather than ``len(values) - 1`` for a left-fold. The total
    instruction count is identical but the AMDGPU scheduler can issue
    independent pairs in parallel, which matches the inline
    ``v_max3_f32 acc, |a|, |b|`` trick CK Tile's
    ``SmoothquantPipelineTwoPass`` reaches for via inline asm
    (``UseMax3 = true``).
    """
    cur = list(values)
    while len(cur) > 1:
        nxt: List[Value] = []
        for i in range(0, len(cur) - 1, 2):
            nxt.append(b.fmax(cur[i], cur[i + 1]))
        if len(cur) % 2:
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]


def make_row_x_distribution(
    *, block_size: int, vec: int, elems_per_thread: int
) -> TileDistribution:
    """Build the per-row activation X tile distribution (CK Tile
    ``MakeXBlockTileDistribution``, the canonical 2D row distribution).

    The DSL kernels launch one CTA per row, so the M tile is a single
    element (``Hs[X0] = (1,)``); the N axis is decomposed
    ``(Repeat_N, ThreadPerWarp_N=block_size, Vector_N=vec)`` with the
    lane id (P0) owning the ``block_size`` level and the Repeat / Vector
    levels living in the per-thread Y tile. The resulting
    :meth:`TileDistribution.calculate_x` reconstruction is exactly the
    legacy hand-rolled ``n_off = k*BS*VEC + tid*VEC`` plus ``+i`` chunk
    addressing, and the Y-space row-major linear order
    (``y = k*VEC + i``) matches the legacy ``cached[k*VEC + i]`` layout
    one-for-one, so the load addresses and the per-thread register
    cache are byte-for-byte the same as the previous
    :func:`sweep_row_chunks` path.
    """
    chunks = elems_per_thread // vec
    encoding = TileDistributionEncoding(
        # X0 = M (one row per CTA), X1 = N = (Repeat, block_size, vec).
        Hs=((1,), (chunks, block_size, vec)),
        # P0 = lane id -> X1 level 1 (the block_size / ThreadPerWarp dim).
        Ps2RHs_major=((2,),),
        Ps2RHs_minor=((1,),),
        # Y0 -> X0 level 0 (the length-1 M dim); Y1 -> X1 level 0
        # (Repeat); Y2 -> X1 level 2 (Vector). Row-major Y order gives
        # storage index (0*chunks + k)*vec + i = k*vec + i.
        Ys2RHs_major=(1, 2, 2),
        Ys2RHs_minor=(0, 0, 2),
    )
    return make_static_tile_distribution(encoding)


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class SmoothQuantSpec:
    """One concrete SmoothQuant kernel configuration."""

    n_per_block: int
    dtype: DType = "f16"
    out_dtype: QDType = "i8"
    block_size: int = 256
    vec: int = 4
    save_yscale: bool = True
    wave_size: int = 64
    name: str = "rocke_smoothquant"

    @property
    def elems_per_thread(self) -> int:
        return self.n_per_block // self.block_size

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            self.out_dtype,
            f"N{self.n_per_block}",
            f"b{self.block_size}",
            f"v{self.vec}",
            flags={"ys": self.save_yscale},
        )


def is_valid_spec(spec: SmoothQuantSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one SmoothQuant config on ``arch``.

    Pure elementwise + single LDS-tree (amax) reduction + dynamic-quant
    cast (no MFMA): architecture facts that matter are the per-WG LDS
    capacity and max threads/block, both sourced from
    :class:`rocke.core.arch.ArchTarget` so an unknown arch / over-budget
    ``block_size`` is rejected with a structured reason. The one f32
    reduction buffer (``block_size`` words) fits both gfx942 (64 KiB)
    and gfx950 (160 KiB), so gfx950 behavior is unchanged.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if spec.out_dtype not in ("i8", "fp8e4m3", "bf8e5m2"):
        return False, f"unsupported out_dtype {spec.out_dtype!r}"

    # fp8/bf8 output goes through ``v_cvt_pk_{fp8,bf8}_f32`` (the
    # ``llvm.amdgcn.cvt.pk.{fp8,bf8}.f32`` intrinsic), which only exists
    # on the CDNA (gfx9xx) ISA — RDNA3.5 (gfx1151) has no fp8/bf8 pack
    # conversion op, so selection would abort with an uncatchable
    # ``LLVM ERROR: Cannot select``. Reject it as a clean spec error so
    # callers get a structured reason. The ``i8`` path uses
    # ``v_cvt_f32_to_i8`` (available everywhere) and stays wave32-valid.
    if spec.out_dtype in ("fp8e4m3", "bf8e5m2") and target.family != "cdna":
        return False, (
            f"out_dtype {spec.out_dtype!r} needs the CDNA-only "
            f"v_cvt_pk_{{fp8,bf8}}_f32 conversion; {arch} (family "
            f"{target.family!r}) has no fp8/bf8 pack op -- use out_dtype='i8'"
        )
    ok, why = validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=spec.n_per_block,
            max_elems_per_thread=64,
        )
    )
    if not ok:
        return False, why

    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > max_threads_per_block "
            f"{target.max_threads_per_block} on {arch}"
        )

    # One f32 LDS reduction buffer (amax) of ``block_size`` words.
    bytes_lds = spec.block_size * 4
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )

    return True, ""


def build_smoothquant(spec: SmoothQuantSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one SmoothQuant forward instance.

    Kernel signature::

    (X: ptr<dtype, global>, # NxM input (row-major)
    SmScale: ptr<f32, global>, # (N,) per-channel smooth scale
    QY: ptr<out_dtype, global>, # NxM quantised output
    [YScale: ptr<f32, global>,] # (M,) per-row dynamic scale
    M: i32, N: i32, eps: f32)

    Grid: ``(M, 1, 1)`` — one CTA per row, same as rmsnorm2d.

    ``arch`` selects the hardware target. The kernel is pure elementwise
    plus a single :func:`block_lds_reduce` amax fold — the reduction is
    an LDS tree over all ``block_size`` lanes (no cross-lane wave
    shuffle), so it is wave-size agnostic and correct on both wave64
    (gfx950) and wave32 (gfx1151). ``arch`` is only used to validate the
    per-WG LDS / thread budget against :class:`ArchTarget`; the default
    ``"gfx950"`` keeps the wave64 build byte-identical.
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid smoothquant spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    q_ty = quant_ir_type(spec.out_dtype)
    qmax = quant_max_abs(spec.out_dtype)

    BS, VEC, N = spec.block_size, spec.vec, spec.n_per_block

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    SmScale = b.param(
        "SmScale", PtrType(F32, "global"), noalias=True, readonly=True, align=16
    )
    QY = b.param("QY", PtrType(q_ty, "global"), noalias=True, writeonly=True, align=16)
    if spec.save_yscale:
        YScale = b.param(
            "YScale", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
        )
    M = b.param("M", I32)  # noqa: F841 — ABI symmetry with CK Tile
    _ = b.param("N", I32)  # noqa: F841 — validated by caller; equals n_per_block
    eps = b.param("eps", F32)

    tid = b.thread_id_x()
    row = b.block_id_x()

    # CK Tile-style views. ``X`` and ``QY`` are 2D packed (row-major)
    # over the full activation; the per-row tile pins the origin to
    # ``row``. ``SmScale`` is a flat 1D view over N.
    x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
    qy_view = make_naive_tensor_view_packed(QY, shape=(1, N), dtype=q_ty)
    sm_view = make_global_view(SmScale, shape=(N,), dtype=F32)
    x_tile = make_tile_window(x_view, lengths=(1, N), origin=(row, b.const_i32(0)))
    qy_tile = make_tile_window(qy_view, lengths=(1, N), origin=(row, b.const_i32(0)))

    # LDS scratch for the block-wide amax reduction. The same lifetime
    # pattern layernorm/rmsnorm use: one ``block_size``-sized f32 buffer.
    lds = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_amax").base

    # Pass 1: load x through the CK-Tile X distribution; accumulate the
    # per-thread amax of ``y = x * smscale`` (in f32). The distributed
    # tensor caches the f32 x scalars so pass 2 doesn't re-read HBM.
    #
    # Per chunk we build the chunk's amax via a balanced
    # ``fmax``-tree (helper :func:`_tree_fmax`) rather than the
    # previous serial fold ``s_amax = fmax(fmax(... fmax(s_amax,
    # |y0|), |y1|), ..., |y_{VEC-1}|)``. The serial fold serialises
    # every chunk's contribution through the same ``s_amax`` register,
    # so VEC-wide chunks paid ``O(VEC)`` of latency on the critical
    # path. The pairwise tree collapses to ``O(log VEC)`` while keeping
    # the same instruction count; combined with the AMDGPU backend's
    # automatic ``v_max3_f32`` pattern-match on ``fmax(fmax(a, -a),
    # fmax(b, -b))`` this matches CK Tile's hand-rolled inline-asm
    # ``UseMax3`` path.
    #
    # CK-Tile adoption: the activation X tile is read via the canonical
    # ``MakeXBlockTileDistribution`` + ``load_tile`` triad
    # (:func:`make_row_x_distribution` / :func:`load_tile`) instead of
    # the bespoke ``sweep_row_chunks`` chunk addressing. The distribution
    # reconstructs the same per-element X coordinates and the returned
    # :class:`StaticDistributedTensor` holds the f32-promoted scalars in
    # the same row-major (``k*VEC + i``) order, so the load addresses and
    # the per-thread register cache are unchanged.
    x_dist = make_row_x_distribution(
        block_size=BS, vec=VEC, elems_per_thread=spec.elems_per_thread
    )
    x_dt = load_tile(b, x_tile, distribution=x_dist, ps=[[tid]])

    s_amax = b.const_f32(0.0)

    def pass1_body(n_off, x_scalars):
        nonlocal s_amax
        sm_scalars = sm_view.load_vec_as_f32(b, [n_off], n=VEC)
        abs_ys: List[Value] = []
        for i in range(VEC):
            y = b.fmul(x_scalars[i], sm_scalars[i])
            # ``|y| = max(y, -y)``: avoids a runtime call to fabs and
            # keeps the IR in pure ``arith.fmax`` / ``arith.fneg``.
            abs_ys.append(b.fmax(y, b.fneg(y)))
        chunk_amax = _tree_fmax(b, abs_ys)
        s_amax = b.fmax(s_amax, chunk_amax)

    # Sweep the distributed X tile chunk-by-chunk (``elems_per_thread /
    # VEC`` chunks); each chunk re-derives its ``n_off`` exactly as the
    # legacy path did so the SmScale gather addresses are identical.
    cached = list(x_dt.storage)
    chunks_p1 = spec.elems_per_thread // VEC
    c_vec_p1 = b.const_i32(VEC)
    for k in range(chunks_p1):
        n_off = b.add(b.mul(b.const_i32(k * BS), c_vec_p1), b.mul(tid, c_vec_p1))
        pass1_body(n_off, cached[k * VEC : (k + 1) * VEC])

    total_amax = block_lds_reduce(b, s_amax, lds, tid, block_size=BS, combine="max")

    # ``yscale = max(amax, eps) / quant_max``. ``fmax`` against ``eps``
    # avoids div-by-zero on all-zero rows (which CK Tile guards the
    # same way; without it the reciprocal is +inf and the cvt produces
    # the wrong saturation direction).
    safe_amax = b.fmax(total_amax, eps)
    yscale = b.fmul(safe_amax, b.const_f32(1.0 / qmax))
    inv_yscale = b.rcp(yscale)

    if spec.save_yscale:
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            b.global_store(YScale, row, yscale, align=4)

    # Pass 2: re-load SmScale, fuse the multiply with the quantise +
    # packed store.
    #
    # CK-Tile adoption (D2c): the quantised QY output is written through
    # the canonical ``store_tile`` triad now that ``store_tile`` handles
    # the i8 / fp8e4m3 / bf8e5m2 dtypes. We build a QY tile distribution
    # identical to the X distribution (so ``calculate_x`` reconstructs the
    # same ``n_off = k*BS*VEC + tid*VEC (+i)`` column addressing), populate
    # a :class:`StaticDistributedTensor` of dtype ``q_ty`` with the
    # already-scaled f32 chunk values, and let :func:`store_tile` do the
    # saturating cvt + ``vec_pack`` + one coalesced ``global_store_vN``
    # per access. This reproduces the same QY bytes as the previous
    # :func:`pack_quant_chunk_local_f32` + :func:`store_packed_chunk_local`
    # path while removing the hand-rolled ``byte_off`` arithmetic.
    #
    # ``VEC == 2`` keeps the per-element scalar fallback (no packed store):
    # the backend already coalesces adjacent lanes' byte stores into a
    # dword across the wave for this VEC.
    chunks = spec.elems_per_thread // VEC
    c_vec = b.const_i32(VEC)
    use_packed_store = VEC in (4, 8)
    if use_packed_store:
        qy_dist = make_row_x_distribution(
            block_size=BS, vec=VEC, elems_per_thread=spec.elems_per_thread
        )
        qy_dt = make_static_distributed_tensor(qy_dist, dtype=q_ty)
        for k in range(chunks):
            n_off = b.add(b.mul(b.const_i32(k * BS), c_vec), b.mul(tid, c_vec))
            sm_scalars = sm_view.load_vec_as_f32(b, [n_off], n=VEC)
            for i in range(VEC):
                x_f32 = cached[k * VEC + i]
                y_f32 = b.fmul(x_f32, sm_scalars[i])
                # Y slots hold the already-scaled f32 chunk; store_tile
                # applies the dtype-specific saturating cvt + pack.
                qy_dt.set([0, k, i], b.fmul(y_f32, inv_yscale))
        store_tile(b, qy_tile, qy_dt, ps=[[tid]])
    else:
        # VEC == 2: per-element scalar quant + store fallback.
        for k in range(chunks):
            n_off = b.add(b.mul(b.const_i32(k * BS), c_vec), b.mul(tid, c_vec))
            sm_scalars = sm_view.load_vec_as_f32(b, [n_off], n=VEC)
            for i in range(VEC):
                x_f32 = cached[k * VEC + i]
                y_f32 = b.fmul(x_f32, sm_scalars[i])
                q = quantize_scalar_f32(
                    b, y_f32, inv_scale=inv_yscale, qdtype=spec.out_dtype
                )
                col = b.add(n_off, b.const_i32(i))
                qy_tile.store_scalar(b, b.const_i32(0), col, value=q)

    return b.kernel


def smoothquant_grid(m: int, spec: SmoothQuantSpec) -> Tuple[int, int, int]:
    """Return the launch grid: one CTA per row."""
    return ceil_div_grid((m, 1))


def smoothquant_signature(spec: SmoothQuantSpec):
    sb = (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("SmScale", "f32")
        .ptr("QY", spec.out_dtype)
    )
    if spec.save_yscale:
        sb.ptr("YScale", "f32")
    return sb.scalar("M", "i32").scalar("N", "i32").scalar("eps", "f32").build()


# ``ptr_type_str`` from helpers.spec only knows the f16/bf16/f32
# canonicalisation; the SignatureBuilder above passes the quant dtype
# string straight through, which is what we want -- the runtime
# launcher reads the type string and forwards it as the manifest's
# dtype tag without further interpretation.
__all__ = [
    "SmoothQuantSpec",
    "build_smoothquant",
    "is_valid_spec",
    "smoothquant_grid",
    "smoothquant_signature",
]
