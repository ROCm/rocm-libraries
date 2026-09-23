# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""2D pooling kernel instance (CK Tile ``36_pooling`` 2D counterpart).

DSL counterpart of CK Tile's ``example/ck_tile/36_pooling`` (which ships
a 3D NDHWC ``MaxPool`` example). For v1 we implement the 2D NHWC case
covering ``max`` / ``avg`` / ``sum`` reductions; the 3D extension is a
v2 follow-up (mostly an extra spatial dim in the descriptor + the inner
loop).

Layout::

    Input  : NHWC, shape (N, H, W, C)
    Output : NHWC, shape (N, Ho, Wo, C)

    Ho = (H + 2*pH - ((Y - 1) * dH + 1)) / sH + 1
    Wo = (W + 2*pW - ((X - 1) * dW + 1)) / sW + 1

For each output ``(n, ho, wo, c)`` the kernel walks the ``(y, x)``
window and reduces over the in-bounds input cells. Pad cells contribute
``-inf`` (max), ``0`` (sum), or are skipped from the count (avg).

Implementation notes:

* Each thread owns one ``vec``-wide slab of consecutive C values at a
  fixed ``(n, ho, wo)``. With ``vec == 1`` the thread degenerates to
  one scalar output (the original v1 shape); with ``vec in {2, 4, 8}``
  the kernel issues one ``buffer_load_vN_f16`` per window cell instead
  of ``vec`` scalar loads, hoists the descriptor's ``valid`` mask out
  of the inner ``c`` step (it depends only on ``(ho, y)`` /
  ``(wo, x)``), and hoists the per-cell ``valid_count`` increment for
  ``avg`` out of the same loop -- one valid-counter update per window
  cell instead of ``vec`` redundant updates.
* Input offsets go through a coordinate-transform descriptor that
  encodes the convolution-style affine map ``hi = ho*sH - pH + y*dH``
  (``embed`` transform with a ``lo`` / ``hi`` bound). This is the same
  ``make_pad_transform`` + ``make_embed_transform`` chain CK Tile's
  ``PoolKernel`` builds in
  :file:`include/ck_tile/ops/pooling/kernel/pool_kernel.hpp`. Out-of-
  bounds cells flip the descriptor's ``valid`` predicate; the kernel
  masks them via ``select(valid, loaded, neutral)`` where the neutral
  value depends on the reduction op.
* The grid is sized in ``vec``-element output positions:
  ``total_out_v = N * Ho * Wo * (C / vec)``. ``C % vec == 0`` is
  enforced by :func:`is_valid_spec`; coarser packing (e.g. ``Ho * Wo``
  on the same thread) is a follow-on once the launcher gains a
  per-spec tile-shape attribute.
* Reductions accumulate in f32 to keep parity with CK Tile's
  ``ComputeDataType = float`` convention; the result is cast back to
  the I/O dtype at the store. The CK Tile reference funnels its
  per-thread span through ``BlockReduce2d`` and a final ``cast_tile``;
  for one thread per output (``Block_N == Y * X``) those stages
  collapse to a Y*X register chain and a trailing scalar cast, which
  is what we emit literally below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import F16, I32, IRBuilder, KernelDef, PtrType, Value
from ...helpers.distribution import (
    TileDistributionEncoding,
    make_static_distributed_tensor,
    make_static_tile_distribution,
    store_tile,
)
from ...helpers.spec import (
    IOSpecRule,
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    validate_io,
)
from ...helpers.tensor_view import make_buffer_resource, make_buffer_view
from ...helpers.transforms import TensorDescriptor, embed, unmerge_magic


DType = Literal["f16", "bf16"]
PoolOp = Literal["max", "avg", "sum"]


@dataclass(frozen=True)
class PoolingProblem:
    """2D pooling shape parameters (NHWC input / output)."""

    N: int
    H: int
    W: int
    C: int

    Y: int  # window height
    X: int  # window width

    sH: int = 1  # stride
    sW: int = 1
    pH: int = 0  # left pad (also used as right pad for now)
    pW: int = 0
    dH: int = 1  # dilation
    dW: int = 1

    @property
    def Ho(self) -> int:
        return (self.H + 2 * self.pH - ((self.Y - 1) * self.dH + 1)) // self.sH + 1

    @property
    def Wo(self) -> int:
        return (self.W + 2 * self.pW - ((self.X - 1) * self.dW + 1)) // self.sW + 1

    @property
    def total_out(self) -> int:
        return self.N * self.Ho * self.Wo * self.C

    def short(self) -> str:
        return (
            f"N{self.N}H{self.H}W{self.W}C{self.C}"
            f"_Y{self.Y}X{self.X}_s{self.sH}x{self.sW}_p{self.pH}x{self.pW}"
        )


@dataclass(frozen=True)
class Pooling2DSpec:
    """One concrete pooling kernel configuration.

    ``vec`` is the number of C-axis elements packed per thread; setting
    it > 1 turns each thread's ``Y * X`` window iteration into a chain
    of ``buffer_load_vN_f16`` instead of ``vec`` separate scalar loads.
    ``C`` must be divisible by ``vec`` (validated at spec time).
    """

    problem: PoolingProblem
    dtype: DType = "f16"
    op: PoolOp = "max"
    block_size: int = 256
    vec: int = 1
    name: str = "rocke_pooling2d"
    # P81: when > 1, each thread owns one (c, ho) tile of multiple wo
    # outputs instead of just one wo. Lets the descriptor reuse
    # ``hi = ho*sH + y*dH`` across ``wo``. Defaults to 1 (one wo per
    # thread, the historical layout); flip to 2/4/8 to amortise the
    # per-output-window math across multiple outputs.
    tile_n: int = 1
    # P82: when ``Block_N > 1`` (multiple lanes share a window's K-axis
    # reduction), the inner reduction benefits from the warp-XOR +
    # cross-warp shape used in :mod:`rocke.helpers.reduction`. Today
    # one thread owns one output so the reduction is a register chain;
    # ``use_warp_xor_reduce=True`` opts into the wave butterfly when
    # ``tile_n`` widens lane work-per-output.
    use_warp_xor_reduce: bool = False

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.problem.short(),
            self.dtype,
            self.op,
            f"b{self.block_size}",
            f"v{self.vec}",
        )


def is_valid_spec(spec: Pooling2DSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a 2D pooling spec on ``arch``.

    The pool is a per-thread window reduction accumulated in registers
    (no LDS, no MFMA atom, no cross-lane butterfly), so it is
    arch-polymorphic and wave-size agnostic. ``arch`` is threaded through
    only to validate the launch ``block_size`` against the target's
    ``max_threads_per_block`` (sourced from
    :class:`rocke.core.arch.ArchTarget`); the emitted IR is byte
    identical for wave64 (gfx950) and wave32 (gfx1151).
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if spec.vec not in (1, 2, 4, 8):
        return False, f"unsupported vec {spec.vec}; expected one of {{1, 2, 4, 8}}"
    _legal_bs = tuple(
        bs for bs in (64, 128, 256, 512, 1024) if bs <= target.max_threads_per_block
    )
    ok, why = validate_io(
        IOSpecRule(
            dtype=spec.dtype,
            block_size=spec.block_size,
            # ``IOSpecRule`` requires vec in {2, 4, 8}; pass a benign 2
            # for ``vec=1`` since the row-divisibility constraint
            # doesn't apply to the one-output-per-thread case.
            vec=spec.vec if spec.vec >= 2 else 2,
            allowed_block_sizes=_legal_bs,
        )
    )
    if not ok:
        return ok, why
    if spec.op not in ("max", "avg", "sum"):
        return False, f"unsupported pool op {spec.op!r}"
    p = spec.problem
    if p.Y <= 0 or p.X <= 0:
        return False, f"window dims must be positive (Y={p.Y}, X={p.X})"
    if p.Ho <= 0 or p.Wo <= 0:
        return False, (
            f"output spatial dims must be positive "
            f"(Ho={p.Ho}, Wo={p.Wo}); check pad/stride/window vs H/W"
        )
    if spec.vec >= 2 and p.C % spec.vec != 0:
        return False, (
            f"vec={spec.vec} requires C ({p.C}) divisible by vec; "
            "fall back to vec=1 for partial-C cases"
        )
    return True, "ok"


def _make_input_descriptor(p: PoolingProblem) -> TensorDescriptor:
    """Input descriptor: ``(n, ho, y, wo, x, c) -> NHWC linear offset``.

    Two ``embed`` transforms encode the conv-style affine map for the
    spatial dims; each carries a ``lo`` / ``hi`` bound so out-of-bounds
    cells (the padded zone) flip the validity predicate.
    """
    return TensorDescriptor.naive(
        "X_nhwc",
        lengths=[p.N, p.H, p.W, p.C],
        dtype=F16,
        coord_names=["n", "hi", "wi", "c"],
    ).transform(
        embed(
            upper=["ho", "y"],
            into="hi",
            strides=[p.sH, p.dH],
            offset=-p.pH,
            lo=0,
            hi=p.H,
        ),
        embed(
            upper=["wo", "x"],
            into="wi",
            strides=[p.sW, p.dW],
            offset=-p.pW,
            lo=0,
            hi=p.W,
        ),
    )


def _neutral_value(b: IRBuilder, op: PoolOp) -> Value:
    """f32 neutral element for the reduction."""
    if op == "max":
        return b.const_f32(float("-inf"))
    if op in ("sum", "avg"):
        return b.const_f32(0.0)
    raise ValueError(f"no neutral value for op {op!r}")


def _combine(b: IRBuilder, op: PoolOp, acc: Value, x: Value) -> Value:
    """Reduction step in f32."""
    if op == "max":
        return b.fmax(acc, x)
    if op in ("sum", "avg"):
        return b.fadd(acc, x)
    raise ValueError(f"no combiner for op {op!r}")


def build_pooling2d(spec: Pooling2DSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one pooling instance.

    Kernel signature::

        (X: ptr<dtype, global>,   # NHWC input  [N, H,  W,  C]
         Y: ptr<dtype, global>,   # NHWC output [N, Ho, Wo, C]
         X_bytes: i32,            # buffer-rsrc byte length for X
         Y_bytes: i32)            # buffer-rsrc byte length for Y

    Grid: ``(ceil(N*Ho*Wo*(C/vec) / block_size), 1, 1)``. Each thread
    owns ``vec`` consecutive C values at one ``(n, ho, wo)`` output
    position.

    The window reduction accumulates in per-thread f32 registers (no
    LDS, no MFMA, no cross-lane butterfly), so the IR is wave-size
    agnostic and identical for wave64 (gfx950) and wave32 (gfx1151).
    ``arch`` is threaded through to :func:`is_valid_spec` purely so the
    block-size bound is validated against the target.
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid pooling2d spec: {why}")

    p = spec.problem
    VEC = spec.vec
    from ...helpers.io import io_ir_type

    io_ty = io_ir_type(spec.dtype)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    X_bytes = b.param("X_bytes", I32)
    Y_bytes = b.param("Y_bytes", I32)

    in_desc = _make_input_descriptor(p)

    # C-axis dimension in ``vec``-element units. With ``vec == 1`` the
    # output address arithmetic and the per-thread work both collapse
    # back to the original scalar shape (one output per thread).
    C_v = p.C // VEC
    total_out_v = p.N * p.Ho * p.Wo * C_v

    c0 = b.const_i32(0)
    c_elem_bytes = b.const_i32(2)  # f16 / bf16 both 2 bytes
    c_vec = b.const_i32(VEC)
    oob_sentinel = b.const_i32((1 << 31) - 1)

    tid = b.thread_id_x()
    bid = b.block_id_x()
    out_idx_v = b.add(b.mul(bid, b.const_i32(spec.block_size)), tid)

    # Decompose the flat ``vec``-unit output index into
    # ``(n, ho, wo, c_v)`` via the CK Tile ``make_merge_transform``'s
    # default magic-division inverse (``merge_v2_magic_division`` ->
    # :class:`UnmergeMagicDiv`). The output M axis is a
    # ``merge((N, Ho, Wo, C_v))`` (``pool_kernel.hpp:168``); its
    # block-id -> (n, ho, wo, c_v) split is exactly this unmerge. Driving
    # it through the descriptor's :meth:`unmerge_lower` removes the
    # hand-rolled four-level ``div``/``mod`` chain *and* the integer
    # divisions themselves (the magic mul-hi sequence is what the AMDGPU
    # backend would otherwise have to synthesise). ``c_base = c_v * VEC``
    # is the C-axis start of this thread's ``vec``-wide slab.
    out_unmerge_desc = TensorDescriptor.naive(
        "pool_out_m",
        lengths=[p.N, p.Ho, p.Wo, C_v],
        dtype=F16,
        coord_names=["n", "ho", "wo", "c_v"],
    ).transform(
        unmerge_magic(
            "out_idx_v",
            into=["n", "ho", "wo", "c_v"],
            dims=[p.N, p.Ho, p.Wo, C_v],
        ),
    )
    decoded = out_unmerge_desc.unmerge_lower(b, out_idx_v=out_idx_v)
    n_val = decoded["n"]
    ho_val = decoded["ho"]
    wo_val = decoded["wo"]
    c_v_val = decoded["c_v"]
    c_base = b.mul(c_v_val, c_vec) if VEC > 1 else c_v_val

    x_rsrc = b.buffer_rsrc(X, X_bytes)

    # Reduction over the (Y, X) window. Both dims are compile-time so
    # the Python-level loop unrolls cleanly; for typical pooling
    # configurations (2x2, 3x3, 7x7) this is fine. Larger windows
    # would benefit from a runtime scf.for, which we can add later.
    #
    # Each thread carries ``VEC`` f32 accumulators (one per C lane in
    # its slab); the per-cell ``valid`` predicate and the optional
    # ``valid_count`` are uniform across those lanes (they only depend
    # on (ho, y, wo, x)), so we increment ``valid_count`` once per
    # window cell instead of once per C lane.
    neutral = _neutral_value(b, spec.op)
    acc_list: list[Value] = [neutral for _ in range(VEC)]
    valid_count = b.const_f32(0.0) if spec.op == "avg" else None

    for y_i in range(p.Y):
        c_y = b.const_i32(y_i)
        for x_i in range(p.X):
            c_x = b.const_i32(x_i)
            off, valid = in_desc.offset(
                b, n=n_val, ho=ho_val, y=c_y, wo=wo_val, x=c_x, c=c_base
            )
            off_bytes = b.mul(off, c_elem_bytes)
            safe_in_off = (
                b.select(valid, off_bytes, oob_sentinel)
                if valid is not None
                else off_bytes
            )
            # f16 / bf16 share the byte-wise buffer load surface
            # (``buffer_load_*`` returns 2-byte values either way; the
            # IR-level type tag drives the f32 cast). For ``VEC >= 2``
            # one ``buffer_load_vN_f16`` reads the whole C slab in one
            # transaction; for ``VEC == 1`` we fall back to the scalar
            # form to avoid emitting a 1-element vector type.
            if VEC >= 2:
                loaded_vec = b.buffer_load_vN_f16(
                    x_rsrc, safe_in_off, c0, dwords=VEC // 2
                )
                for k in range(VEC):
                    raw = b.vec_extract(loaded_vec, k)
                    loaded_f32 = b.cast_to_f32(raw)
                    # ``valid`` is uniform across the VEC C lanes -- the
                    # bounds check depends only on (ho, y, wo, x), so we
                    # apply the same mask to every C lane.
                    if valid is not None:
                        masked = b.select(valid, loaded_f32, neutral)
                    else:
                        masked = loaded_f32
                    acc_list[k] = _combine(b, spec.op, acc_list[k], masked)
            else:
                loaded_raw = b.buffer_load_f16(x_rsrc, safe_in_off, c0)
                loaded_f32 = b.cast_to_f32(loaded_raw)
                # Mask invalid loads with the neutral element so they
                # don't contribute to the reduction (especially
                # relevant for max pool: a buffer-OOB load returns 0,
                # which would be the wrong winner on negative-only
                # inputs).
                if valid is not None:
                    masked = b.select(valid, loaded_f32, neutral)
                else:
                    masked = loaded_f32
                acc_list[0] = _combine(b, spec.op, acc_list[0], masked)

            if spec.op == "avg":
                # Hoisted out of the per-C-lane loop -- ``valid`` is
                # the same for every C value at this (y_i, x_i).
                contrib = (
                    b.select(valid, b.const_f32(1.0), b.const_f32(0.0))
                    if valid is not None
                    else b.const_f32(1.0)
                )
                valid_count = b.fadd(valid_count, contrib)

    # Finalise: avg = sum / count (count >= 1 in practice because every
    # output position has at least one valid window cell; we guard
    # against div-by-zero with a 1.0 fallback for the all-pad case).
    # The rcp is shared across the VEC outputs because the count is
    # uniform per (n, ho, wo).
    if spec.op == "avg":
        safe_count = b.fmax(valid_count, b.const_f32(1.0))
        rcp_count = b.rcp(safe_count)
        acc_list = [b.fmul(acc, rcp_count) for acc in acc_list]

    # Store through a CK Tile ``store_tile`` + ``tile_distribution`` over the
    # flat output ``[N*Ho*Wo*C]`` (idiom A6, store side). The per-thread
    # tile is the ``VEC``-wide C slab -- a single X dim of length ``VEC``
    # consumed by one Y dim, so ``store_tile`` packs the ``VEC`` casted
    # scalars and issues one coalesced ``buffer_store_vN_f16`` (or a scalar
    # store for ``VEC == 1``). The window origin ``out_idx_v * VEC`` places
    # this thread's slab; tail-of-grid threads (``out_idx_v >= total_out_v``)
    # land past the buffer-resource byte bound, so the AMDGPU buffer rsrc
    # silently drops the store -- the same OOB tail semantics the hand-rolled
    # ``select(in_bounds, off, sentinel)`` provided, now sourced from the
    # buffer descriptor itself.
    out_total = total_out_v * VEC
    out_rsrc = make_buffer_resource(b, Y, num_bytes=Y_bytes)
    out_view = make_buffer_view(out_rsrc, [out_total], io_ty)
    out_enc = TileDistributionEncoding(
        Hs=((VEC,),),
        Ys2RHs_major=(1,),
        Ys2RHs_minor=(0,),
    )
    out_dist = make_static_tile_distribution(out_enc)
    out_origin = b.mul(out_idx_v, c_vec) if VEC > 1 else out_idx_v
    out_window = out_view.tile(lengths=[VEC], origin=[out_origin])
    # ``store_tile`` for f16/bf16 expects f32 Y-slot scalars and performs
    # the demote + pack itself, so we feed the f32 accumulators directly.
    out_dt = make_static_distributed_tensor(out_dist, dtype=io_ty)
    for k in range(VEC):
        out_dt.set([k], acc_list[k])
    store_tile(b, out_window, out_dt, ps=[])

    return b.kernel


def pooling2d_grid(spec: Pooling2DSpec) -> Tuple[int, int, int]:
    """Return the launch grid: one thread per ``vec``-element output slab."""
    total_v = spec.problem.total_out // max(spec.vec, 1)
    return ceil_div_grid((total_v, spec.block_size))


def pooling2d_signature(spec: Pooling2DSpec):
    return (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Y", spec.dtype)
        .scalar("X_bytes", "i32")
        .scalar("Y_bytes", "i32")
        .build()
    )
