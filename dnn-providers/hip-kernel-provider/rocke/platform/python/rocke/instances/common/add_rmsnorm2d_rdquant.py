# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fused add + RMSNorm + round-to-quant kernel.

DSL counterpart of CK Tile's
``example/ck_tile/11_add_rmsnorm2d_rdquant``. For two ``(M, N)`` input
tensors ``A`` and ``B`` and an ``(N,)`` ``Gamma`` per-channel scale,
the kernel produces the residual sum, the RMSNorm output, and the
quantised RMSNorm output in one pass over global memory::

 x[m, n] = a[m, n] + b[m, n]
 sum_sq[m] = sum_n(x[m, n] ^ 2)
 inv_rms[m] = 1 / sqrt(sum_sq[m] / N + eps_rms)
 y[m, n] = x[m, n] * inv_rms[m] * gamma[n]
 yscale[m] = max(amax_y, eps_q) / quant_max
 qy[m, n] = quantise(y[m, n], 1 / yscale[m])

with ``amax_y = inv_rms * max_n(|x * gamma|)``.

Two output paths:

* ``QY`` : quantised RMSNorm output (i8 / fp8e4m3 / bf8e5m2)
* ``YScale`` : per-row dynamic quant scale (fp32; optional)
* ``X`` : optional residual write-back of ``a + b`` (saves the next
 layer from recomputing the add when it needs the residual stream).

Implementation:

* One CTA per row (same as rmsnorm2d).
* Pass 1 streams ``a`` and ``b`` once: computes ``x = a + b``, the
 per-thread ``sum_sq``, and the per-thread ``amax_g`` (the max of
 ``|x * gamma|`` before the inv_rms scale). Caches ``x`` in the
 per-thread f32 register file for pass 2; writes ``x`` back to the
 residual buffer when ``save_residual=True``.
* Two block-LDS reductions (sum + max) feed the per-row constants.
* Pass 2 re-reads gamma (in L1 by now), fuses the
 ``x * inv_rms * gamma`` multiply with the quant cast, and stores
 ``qy`` per element via :func:`rocke.helpers.quantize_scalar_f32`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

from ...core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Value
from ...helpers.io import io_ir_type, store_scalar_from_f32
from ...helpers.quant import (
    QDType,
    pack_quant_chunk_local_f32,
    quant_ir_type,
    quant_max_abs,
    quantize_scalar_f32,
    store_packed_chunk_local,
)
from ...helpers.reduction import block_lds_reduce_pair, tree_reduce
from ...helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ...helpers.sweep import sweep_row_chunks
from ...helpers.tensor_view import (
    make_global_view,
    make_lds_view,
    make_naive_tensor_view_packed,
    make_tile_window,
)


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class AddRmsnorm2DRdquantSpec:
    """One concrete fused add + RMSNorm + round-quant configuration."""

    n_per_block: int
    dtype: DType = "f16"
    out_dtype: QDType = "i8"
    block_size: int = 256
    vec: int = 4
    save_residual: bool = True  # write x = a + b to ``X``
    save_yscale: bool = True  # write per-row scale to ``YScale``
    wave_size: int = 64
    name: str = "rocke_add_rmsnorm2d_rdquant"

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
            flags={"sr": self.save_residual, "ys": self.save_yscale},
        )


def is_valid_spec(
    spec: AddRmsnorm2DRdquantSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for one fused add + RMSNorm + quant config on ``arch``.

    Pure elementwise + twin LDS-tree reductions + dynamic-quant cast (no
    MFMA): architecture facts that matter are the per-WG LDS capacity and
    max threads/block, both sourced from
    :class:`rocke.core.arch.ArchTarget` so an unknown arch / over-budget
    ``block_size`` is rejected with a structured reason. The two f32
    reduction buffers (``2 * block_size`` words) fit both gfx942 (64 KiB)
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
    # ``llvm.amdgcn.cvt.pk.{fp8,bf8}.f32`` intrinsic), which only exists on
    # the CDNA (gfx9xx) ISA — RDNA3.5 (gfx1151) has no fp8/bf8 pack
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

    # Two f32 LDS reduction buffers (sum + max), ``block_size`` words each.
    bytes_lds = 2 * spec.block_size * 4
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap on {arch}"
        )

    return True, ""


def build_add_rmsnorm2d_rdquant(
    spec: AddRmsnorm2DRdquantSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the IR for one fused add + RMSNorm + quantise instance.

    Kernel signature (with both optional outputs enabled)::

    (A: ptr<dtype, global>, # (M, N)
    B: ptr<dtype, global>, # (M, N)
    Gamma: ptr<dtype, global>, # (N,)
    X: ptr<dtype, global>, # (M, N) optional residual out (a+b)
    QY: ptr<out_dtype, global>, # (M, N) quantised output
    YScale: ptr<f32, global>, # (M,) optional per-row scale
    M: i32, N: i32,
    eps_rms: f32, # rmsnorm epsilon
    eps_q: f32) # amax clamp epsilon (avoid /0)

    Grid: ``(M, 1, 1)`` — one CTA per row.

    ``arch`` selects the validation target (default ``"gfx950"`` keeps
    the CDNA wave64 behavior byte-identical). The body is wave-size
    agnostic: both cross-thread folds go through the LDS tree
    (:func:`rocke.helpers.reduction.block_lds_reduce_pair`), which halves over
    ``block_size`` and barriers between steps with no wave64-only
    cross-lane op, so the same IR is correct on the gfx1151 wave32
    target. ``arch`` is therefore only threaded into
    :func:`is_valid_spec` (LDS / threads-per-block budgeting).
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid add_rmsnorm2d_rdquant spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    q_ty = quant_ir_type(spec.out_dtype)
    qmax = quant_max_abs(spec.out_dtype)

    BS, VEC, N = spec.block_size, spec.vec, spec.n_per_block

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    A = b.param("A", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Gamma = b.param(
        "Gamma", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16
    )
    if spec.save_residual:
        X = b.param(
            "X", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16
        )
    QY = b.param("QY", PtrType(q_ty, "global"), noalias=True, writeonly=True, align=16)
    if spec.save_yscale:
        YScale = b.param(
            "YScale", PtrType(F32, "global"), noalias=True, writeonly=True, align=4
        )
    M = b.param("M", I32)  # noqa: F841 — ABI symmetry with CK Tile
    _ = b.param("N", I32)  # noqa: F841 — validated by caller
    eps_rms = b.param("eps_rms", F32)
    eps_q = b.param("eps_q", F32)

    tid = b.thread_id_x()
    row = b.block_id_x()

    a_view = make_naive_tensor_view_packed(A, shape=(1, N), dtype=io_ty)
    b_view = make_naive_tensor_view_packed(Bp, shape=(1, N), dtype=io_ty)
    g_view = make_global_view(Gamma, shape=(N,), dtype=io_ty)
    qy_view = make_naive_tensor_view_packed(QY, shape=(1, N), dtype=q_ty)
    a_tile = make_tile_window(a_view, lengths=(1, N), origin=(row, b.const_i32(0)))
    bt_tile = make_tile_window(b_view, lengths=(1, N), origin=(row, b.const_i32(0)))
    qy_tile = make_tile_window(qy_view, lengths=(1, N), origin=(row, b.const_i32(0)))
    if spec.save_residual:
        x_view = make_naive_tensor_view_packed(X, shape=(1, N), dtype=io_ty)
        x_tile = make_tile_window(x_view, lengths=(1, N), origin=(row, b.const_i32(0)))

    # Two LDS scratch buffers (sum + max) feed a single twin-channel
    # reduction via :func:`rocke.helpers.reduction.block_lds_reduce_pair`
    # -- one
    # halving schedule instead of the previous two back-to-back
    # ``block_lds_reduce`` calls, halving the ``s_barrier`` count.
    # 1 KB per buffer for the default ``BS=256`` -- trivial vs LDS
    # budget.
    lds_sum = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_sum").base
    lds_max = make_lds_view(b, dtype=F32, shape=(BS,), name_hint="lds_max").base

    # Pass 1: stream A (which carries the f32-promoted ``a`` scalars
    # through ``sweep_row_chunks``'s ``x_scalars`` argument); per chunk
    # we manually load ``b`` from ``B`` and ``gamma`` from ``Gamma``
    # to compute ``x = a + b``, the per-thread sum-of-squares (over
    # ``x``), and the per-thread amax (over ``|x * gamma|``).
    #
    # Optimisation vs the prior body:
    #
    # * We cache ``xg = x * gamma`` (not raw ``x``) for pass 2. Pass 2
    #   no longer needs to re-load ``gamma`` from HBM nor re-multiply
    #   ``x * gamma`` -- it just runs ``y = xg * inv_rms``. Net per
    #   element: -1 ``f16`` HBM load, -1 ``fmul``. Per-thread cache
    #   width is unchanged (one f32 per element), so register pressure
    #   stays the same. When ``save_residual=True`` we still write
    #   ``x`` back to the residual buffer; we just don't cache it.
    # * The per-chunk ``s_sq`` and ``s_amax_g`` partials are folded
    #   via :func:`rocke.helpers.reduction.tree_reduce` so the critical-path depth is
    #   ``log2(VEC)`` (3 for VEC=8) instead of ``VEC`` (8). Total op
    #   count is unchanged -- the win is latency, not arithmetic.
    # * ``|xg|`` via ``fmax(xg, fneg(xg))`` is kept (no fabs IR op;
    #   matches the SmoothQuant idiom and lowers to one
    #   ``v_max_f32 xg, -xg``).
    s_sq = b.const_f32(0.0)
    s_amax_g = b.const_f32(0.0)
    cached_xg: List[Value] = []

    def pass1_body(n_off, a_scalars):
        nonlocal s_sq, s_amax_g
        b_scalars = bt_tile.load_vec_as_f32(b, b.const_i32(0), n_off, n=VEC)
        g_scalars = g_view.load_vec_as_f32(b, [n_off], n=VEC)
        chunk_x: List[Value] = []
        chunk_xg: List[Value] = []
        chunk_sq: List[Value] = []
        chunk_abs_xg: List[Value] = []
        for i in range(VEC):
            x_i = b.fadd(a_scalars[i], b_scalars[i])
            xg_i = b.fmul(x_i, g_scalars[i])
            chunk_x.append(x_i)
            chunk_xg.append(xg_i)
            chunk_sq.append(b.fmul(x_i, x_i))
            chunk_abs_xg.append(b.fmax(xg_i, b.fneg(xg_i)))
        # Balanced-tree fold: depth log2(VEC), 1 per-chunk merge into
        # the running scalar partials. AITER's
        # ``add_rmsnorm_dynquant`` device kernel uses the same
        # "vectorised compute, scalar accumulator at the bottom"
        # structure; we mirror it in the DSL.
        s_sq = b.fadd(s_sq, tree_reduce(b, b.fadd, chunk_sq))
        s_amax_g = b.fmax(s_amax_g, tree_reduce(b, b.fmax, chunk_abs_xg))
        cached_xg.extend(chunk_xg)
        if spec.save_residual:
            # Residual write-back of ``x = a + b`` happens here so
            # the next layer's residual stream can pick it up. The
            # pass-2 quantise path doesn't need ``x``; it reads
            # ``xg`` straight out of the cache.
            x_tile.store_vec_from_f32(b, b.const_i32(0), n_off, values=chunk_x)

    sweep_row_chunks(
        b,
        a_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        elems_per_thread=spec.elems_per_thread,
        body=pass1_body,
        cache=False,  # we manage the cache ourselves (``xg``, not raw ``a``).
    )

    # Twin-channel cross-thread reduction: ``sum`` for the sum-of-squares
    # and ``max`` for the abs-max of ``x * gamma``.
    total_sq, total_amax_g = block_lds_reduce_pair(
        b,
        s_sq,
        s_amax_g,
        lds_sum,
        lds_max,
        tid,
        block_size=BS,
        combine_a="sum",
        combine_c="max",
    )

    rcp_n = b.rcp(b.const_f32(float(N)))
    mean_sq = b.fmul(total_sq, rcp_n)
    inv_rms = b.rsqrt(b.fadd(mean_sq, eps_rms))

    # ``amax_y = inv_rms * max(|x * gamma|)``: same algebraic identity
    # CK Tile's ``add_rmsnorm2d_rdquant`` pipeline collapses to
    # (``inv_rms`` is non-negative so the per-row positive scale
    # factors out of the abs-max).
    amax_y = b.fmul(inv_rms, total_amax_g)
    safe_amax = b.fmax(amax_y, eps_q)
    yscale = b.fmul(safe_amax, b.const_f32(1.0 / qmax))
    inv_yscale = b.rcp(yscale)

    if spec.save_yscale:
        with b.scf_if(b.cmp_eq(tid, b.const_i32(0))):
            b.global_store(YScale, row, yscale, align=4)

    # Pass 2: ``y = xg * inv_rms`` straight out of the per-thread cache;
    # quantise and store. ``q = quantize(y, inv_yscale) =
    # cast(y * inv_yscale)``, so the per-element f32 fed to the cast is
    # ``xg * (inv_rms * inv_yscale)``. Folding the two per-row constants
    # into one ``rms_q`` scalar drops the inner multiply chain to a
    # single ``fmul`` per element.
    #
    # For ``VEC in {4, 8}`` the per-chunk quantise + store now uses the
    # packed path (mirror of SmoothQuant pass 2): the VEC scaled f32
    # scalars go through :func:`pack_quant_chunk_local_f32` (one packed
    # ``v_cvt_pk_fp8_f32`` per 4 lanes for fp8/bf8, or scalar
    # ``v_cvt_f32_to_i8_sat`` + :func:`vec_pack` for i8) and the whole
    # chunk is emitted as a single i32/i64 ``global_store`` via
    # :func:`store_packed_chunk_local` -- collapsing the previous VEC scalar
    # 8-bit stores into one VMEM transaction per chunk. ``VEC == 2``
    # keeps the per-element scalar path (no packed-store helper for n=2;
    # the backend already coalesces adjacent lanes' byte stores there).
    chunks = spec.elems_per_thread // VEC
    c_vec = b.const_i32(VEC)
    rms_q = b.fmul(inv_rms, inv_yscale)
    use_packed_store = VEC in (4, 8)
    row_base_byte_off = b.mul(row, b.const_i32(N))
    for k in range(chunks):
        n_off = b.add(b.mul(b.const_i32(k * BS), c_vec), b.mul(tid, c_vec))
        if use_packed_store:
            scaled_f32 = [b.fmul(cached_xg[k * VEC + i], rms_q) for i in range(VEC)]
            packed = pack_quant_chunk_local_f32(
                b, scaled_f32, q_ty=q_ty, out_dtype=spec.out_dtype
            )
            byte_off = b.add(row_base_byte_off, n_off)
            store_packed_chunk_local(b, QY, byte_off, packed, n=VEC)
        else:
            for i in range(VEC):
                xg_f32 = cached_xg[k * VEC + i]
                y_f32 = b.fmul(xg_f32, inv_rms)
                q = quantize_scalar_f32(
                    b, y_f32, inv_scale=inv_yscale, qdtype=spec.out_dtype
                )
                col = b.add(n_off, b.const_i32(i))
                qy_tile.store_scalar(b, b.const_i32(0), col, value=q)

    # ``store_scalar_from_f32`` is the canonical "f32 -> dtype scalar
    # store" path; we re-export it here as a no-op import so static
    # analysers don't flag the helper as unused (the kernel uses
    # ``global_store`` for the f32 ``YScale`` write but the
    # quantisation path keeps everything in the cvt op chain).
    _ = store_scalar_from_f32  # noqa: F841 -- public-API touch

    return b.kernel


def add_rmsnorm2d_rdquant_grid(
    m: int, spec: AddRmsnorm2DRdquantSpec
) -> Tuple[int, int, int]:
    """Return the launch grid: one CTA per row."""
    return ceil_div_grid((m, 1))


def add_rmsnorm2d_rdquant_signature(spec: AddRmsnorm2DRdquantSpec):
    sb = (
        SignatureBuilder()
        .ptr("A", spec.dtype)
        .ptr("B", spec.dtype)
        .ptr("Gamma", spec.dtype)
    )
    if spec.save_residual:
        sb.ptr("X", spec.dtype)
    sb.ptr("QY", spec.out_dtype)
    if spec.save_yscale:
        sb.ptr("YScale", "f32")
    return (
        sb.scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("eps_rms", "f32")
        .scalar("eps_q", "f32")
        .build()
    )


__all__ = [
    "AddRmsnorm2DRdquantSpec",
    "add_rmsnorm2d_rdquant_grid",
    "add_rmsnorm2d_rdquant_signature",
    "build_add_rmsnorm2d_rdquant",
    "is_valid_spec",
]
