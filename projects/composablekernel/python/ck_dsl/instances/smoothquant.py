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
 ``"max"`` combiner (no new IR primitive needed).
* ``eps`` (passed as an f32 kernel arg) guards the
 ``yscale = amax / quant_max`` division against pathological rows
 where ``amax == 0``. Matches the CK Tile reference's ``eps`` arg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import F32, I32, IRBuilder, KernelDef, PtrType
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
    s_amax = b.const_f32(0.0)

    def pass1_body(n_off, x_scalars):
        nonlocal s_amax
        sm_scalars = sm_view.load_vec_as_f32(b, [n_off], n=VEC)
        for i in range(VEC):
            y = b.fmul(x_scalars[i], sm_scalars[i])
            # ``|y| = max(y, -y)``: avoids a runtime call to fabs and
            # keeps the IR in pure ``arith.fmax`` / ``arith.fneg``.
            abs_y = b.fmax(y, b.fneg(y))
            s_amax = b.fmax(s_amax, abs_y)

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

    # Pass 2: re-load SmScale, fuse the multiply with the quantise.
    # ``store_vec_from_f32`` isn't wired for non-f16/bf16 dtypes, so
    # this pass uses the explicit ``quantize_scalar_f32`` + scalar
    # store per element. SmScale is re-read from HBM (it lives in L1
    # after pass 1 so the second read is ~free), and the cached x
    # scalars come straight from the pass 1 sweep result.
    cached = sweep_res.cached
    chunks = spec.elems_per_thread // VEC
    c_vec = b.const_i32(VEC)
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
