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
(:func:`ck_dsl.helpers.quant.quantize_scalar_f32` handles both).

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

* Pass 1 caches the f32-promoted ``x`` scalars via
 :func:`sweep_row_chunks`, so pass 2 does not re-read HBM. SmScale
 is re-loaded in pass 2 — it lives in L1 by the second pass and the
 re-load costs ~free.
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

from ..core.ir import F32, I32, I64, IRBuilder, KernelDef, PtrType, Value
from ..helpers.io import io_ir_type
from ..helpers.quant import QDType, quant_ir_type, quant_max_abs, quantize_scalar_f32
from ..helpers.reduction import block_lds_reduce
from ..helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ..helpers.sweep import sweep_row_chunks
from ..helpers.tensor_view import (
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


def _pack_quant_chunk_f32(
    b: IRBuilder,
    scaled_f32: List[Value],
    *,
    q_ty,
    out_dtype: QDType,
) -> Value:
    """Quantise ``len(scaled_f32)`` f32 scalars and pack them into a
    ``<N x q_ty>`` vector.

    ``scaled_f32`` is the chunk of ``y * inv_yscale`` values **already
    multiplied by ``inv_yscale``** — i.e. ready for the dtype-specific
    saturating cast. The packing routes are:

    * For ``fp8e4m3`` / ``bf8e5m2`` with a 4-wide chunk we issue one
      packed ``v_cvt_pk_fp8_f32`` (resp. ``v_cvt_pk_bf8_f32``) via
      :func:`IRBuilder.cvt_pk_fp8_f32x4` / ``cvt_pk_bf8_f32x4``,
      saving 3 scalar ``v_cvt_*_f32`` instructions and the redundant
      ``clamp_f32`` (the hardware saturates on conversion).
    * For ``i8`` (no packed cvt today) and ``VEC == 2`` chunks (no
      2-wide packed cvt), we emit per-element
      :func:`quantize_scalar_f32` and pack via :func:`IRBuilder.vec_pack`.

    For ``len(scaled_f32) == 8`` we stitch two 4-wide chunks together
    with :func:`IRBuilder.vec_concat`; the resulting ``<8 x q_ty>``
    lowers to a single 8-byte VMEM store.
    """
    n = len(scaled_f32)
    if n not in (2, 4, 8):
        raise ValueError(f"_pack_quant_chunk_f32 expects n in {{2,4,8}}, got {n}")

    if out_dtype in ("fp8e4m3", "bf8e5m2") and n % 4 == 0:
        cvt = b.cvt_pk_fp8_f32x4 if out_dtype == "fp8e4m3" else b.cvt_pk_bf8_f32x4
        # Pack into <4 x f32> chunks and feed each through the packed cvt.
        packed_chunks: List[Value] = []
        for off in range(0, n, 4):
            quad = b.vec_pack(scaled_f32[off : off + 4], F32)
            packed_chunks.append(cvt(quad))
        if len(packed_chunks) == 1:
            return packed_chunks[0]
        out = packed_chunks[0]
        for chunk in packed_chunks[1:]:
            out = b.vec_concat(out, chunk)
        return out

    # i8 path (or VEC=2 fp8/bf8): per-element saturating cast + vec_pack.
    qs: List[Value] = []
    for sf in scaled_f32:
        if out_dtype == "i8":
            qs.append(
                b.cvt_f32_to_i8_sat(
                    b.clamp_f32(sf, b.const_f32(-127.0), b.const_f32(127.0))
                )
            )
        elif out_dtype == "fp8e4m3":
            qs.append(b.cvt_f32_to_fp8(sf))
        elif out_dtype == "bf8e5m2":
            qs.append(b.cvt_f32_to_bf8(sf))
        else:
            raise ValueError(f"unsupported out_dtype {out_dtype!r}")
    return b.vec_pack(qs, q_ty)


def _store_packed_chunk(
    b: IRBuilder,
    qy_ptr: Value,
    byte_off: Value,
    packed: Value,
    *,
    n: int,
) -> None:
    """Bitcast a ``<n x q_ty>`` (q_ty is an 8-bit dtype) to an
    integer and emit a single global store.

    The IR's :func:`IRBuilder.global_store_vN` only accepts f16/bf16
    today, so to coalesce the 8-bit-per-element output into a single
    VMEM transaction we bitcast the packed vector to an integer of
    matching width:

    * ``n == 4`` -> ``i32``, one ``buffer_store_dword``.
    * ``n == 8`` -> ``i64``, one ``buffer_store_dwordx2``.

    ``byte_off`` is the absolute byte offset within the QY buffer; the
    integer GEP stride for the chosen integer type is ``n`` bytes, so
    the helper divides ``byte_off`` by ``n`` via a logical right shift.
    Both ``byte_off`` and ``n`` are guaranteed multiples of ``n`` by
    spec validation (``N % (BS * VEC) == 0``).

    ``n == 2`` is not supported here: there is no ``I16`` IR type
    exposed today, and the AMDGPU backend already coalesces adjacent
    lanes' single-byte stores into a wave-wide dword in the scalar
    fall-back path, so the packed-store win is marginal for VEC=2.
    """
    if n == 4:
        as_int = b.bitcast(packed, I32)
        idx = b.lshr(byte_off, b.const_i32(2))
        b.global_store(qy_ptr, idx, as_int, align=4)
    elif n == 8:
        as_int = b.bitcast(packed, I64)
        idx = b.lshr(byte_off, b.const_i32(3))
        b.global_store(qy_ptr, idx, as_int, align=8)
    else:
        raise ValueError(f"_store_packed_chunk supports n in {{4, 8}}, got {n}")


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
    name: str = "ck_dsl_smoothquant"

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


def is_valid_spec(spec: SmoothQuantSpec) -> Tuple[bool, str]:
    if spec.out_dtype not in ("i8", "fp8e4m3", "bf8e5m2"):
        return False, f"unsupported out_dtype {spec.out_dtype!r}"
    return validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            vec=spec.vec,
            n_per_block=spec.n_per_block,
            max_elems_per_thread=64,
        )
    )


def build_smoothquant(spec: SmoothQuantSpec) -> KernelDef:
    """Build the IR for one SmoothQuant forward instance.

    Kernel signature::

    (X: ptr<dtype, global>, # NxM input (row-major)
    SmScale: ptr<f32, global>, # (N,) per-channel smooth scale
    QY: ptr<out_dtype, global>, # NxM quantised output
    [YScale: ptr<f32, global>,] # (M,) per-row dynamic scale
    M: i32, N: i32, eps: f32)

    Grid: ``(M, 1, 1)`` — one CTA per row, same as rmsnorm2d.
    """
    ok, why = is_valid_spec(spec)
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

    # Pass 1: stream x through ``sweep_row_chunks``; accumulate the
    # per-thread amax of ``y = x * smscale`` (in f32). Cache the f32
    # x scalars so pass 2 doesn't re-read HBM.
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

    sweep_res = sweep_row_chunks(
        b,
        x_tile,
        tid=tid,
        block_size=BS,
        vec=VEC,
        elems_per_thread=spec.elems_per_thread,
        body=pass1_body,
        cache=True,
    )

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
    # The previous formulation issued ``VEC`` scalar
    # ``quantize_scalar_f32`` calls and ``VEC`` scalar 8-bit
    # ``global_store``\\s per chunk. That is bandwidth-bound (every
    # lane pays the address-arithmetic overhead and the AMDGPU backend
    # only sometimes folds adjacent byte stores into a wave-wide
    # dword). The packed path packs ``y * inv_yscale`` into a
    # ``<VEC x f32>``, applies the dtype-specific saturating cast
    # (``v_cvt_pk_fp8_f32`` for fp8/bf8 via :func:`_pack_quant_chunk_f32`,
    # or scalar ``v_cvt_f32_to_i8_sat`` + :func:`IRBuilder.vec_pack`
    # for i8), and emits one i32 (or i64) ``global_store`` per chunk.
    cached = sweep_res.cached
    chunks = spec.elems_per_thread // VEC
    c_vec = b.const_i32(VEC)
    use_packed_store = VEC in (4, 8)
    row_base_byte_off = b.mul(row, b.const_i32(N))
    for k in range(chunks):
        n_off = b.add(b.mul(b.const_i32(k * BS), c_vec), b.mul(tid, c_vec))
        sm_scalars = sm_view.load_vec_as_f32(b, [n_off], n=VEC)
        if use_packed_store:
            scaled_f32: List[Value] = []
            for i in range(VEC):
                x_f32 = cached[k * VEC + i]
                y_f32 = b.fmul(x_f32, sm_scalars[i])
                scaled_f32.append(b.fmul(y_f32, inv_yscale))
            packed = _pack_quant_chunk_f32(
                b, scaled_f32, q_ty=q_ty, out_dtype=spec.out_dtype
            )
            byte_off = b.add(row_base_byte_off, n_off)
            _store_packed_chunk(b, QY, byte_off, packed, n=VEC)
        else:
            # VEC == 2: keep the per-element scalar path. The backend
            # already coalesces adjacent lanes' byte stores into a
            # dword across the wave for this VEC.
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
