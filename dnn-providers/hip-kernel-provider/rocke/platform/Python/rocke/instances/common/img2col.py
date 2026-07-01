# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Image-to-column (im2col) kernel instance.

DSL counterpart of CK Tile's ``example/ck_tile/04_img2col``. The kernel
materialises the implicit-GEMM A operand for a convolution as a real
tensor of shape ``[M, K]`` where::

    M = N * Ho * Wo
    K = R * S * C

For each output position ``(m, k)`` it computes the corresponding NHWC
input address using the same coordinate-transform DAG the implicit-GEMM
convolution uses (:func:`rocke.instances.common.conv_implicit_gemm.make_a_descriptor`),
and writes either the input value or zero (for padded / out-of-image
positions). The kernel is pure index transform + copy — no MFMA, no LDS
staging — so it makes a useful end-to-end test of the descriptor + buffer
load path on a non-GEMM, non-attention shape.

What we cover today:

* Dtype ``f16`` (matches the upstream CK Tile example, which only
  instantiates ``half_t``).
* 2D spatial conv (``NHWC + KRSC``, single group ``G == 1``). 1D / 3D
  / multi-group is a v2 extension.
* Padding ``pH`` / ``pW`` and dilation ``dH`` / ``dW`` from
  :class:`ConvProblem`.

Pipeline notes:

* One thread per output element. Block shape is ``(block_tile_m,
  block_tile_k)`` flattened; pick a shape that lands ``block_size <= 1024``.
* Input load uses an AMDGPU buffer descriptor for free OOB clamping:
  invalid offsets (padding zone) are redirected to a sentinel that the
  hardware silently returns as zero.
* Output store also uses the buffer descriptor; tail-of-grid writes
  past ``M`` / ``K`` are silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import F16, I32, IRBuilder, KernelDef, PtrType
from ...helpers.distribution import (
    TileDistributionEncoding,
    load_tile,
    make_load_store_traits,
    make_static_tile_distribution,
)
from ...helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join
from ...helpers.tensor_view import make_buffer_resource, view_from_transforms_descriptor
from ...helpers.transforms import TensorDescriptor, unmerge_magic
from .conv_implicit_gemm import ConvProblem, make_a_descriptor


DType = Literal["f16"]


@dataclass(frozen=True)
class Img2ColSpec:
    """One concrete image-to-column kernel configuration.

    ``problem`` selects the convolution shape (input dims, filter dims,
    stride / pad / dilation); the kernel emits a shape-specialised
    binary, just like the existing implicit-GEMM conv path.

    ``block_tile_m`` × ``block_tile_k`` is the per-block tile of the
    output matrix; each block dispatches ``block_tile_m * block_tile_k``
    threads (one per output element). The product must be <= 1024.
    """

    problem: ConvProblem
    dtype: DType = "f16"
    block_tile_m: int = 8
    block_tile_k: int = 128
    vec_k: int = 8
    name: str = "rocke_img2col"

    @property
    def block_size(self) -> int:
        return (self.block_tile_m * self.block_tile_k) // self.vec_k

    @property
    def can_vector_load(self) -> bool:
        """Whether a single ``buffer_load_vN_f16`` covers each chunk.

        True iff ``vec_k > 1`` and ``vec_k`` divides the input channel
        count ``C`` — then every chunk's K-slots live in one
        ``(r, s, c0..c0+vec_k-1)`` block and the leading NHWC offset is
        enough for the whole vector. Otherwise the kernel falls back to
        ``vec_k`` per-element scalar loads + a ``vec_pack``, still
        followed by one wide store.
        """
        return self.vec_k > 1 and (self.problem.C % self.vec_k) == 0

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.problem.short(),
            self.dtype,
            f"t{self.block_tile_m}x{self.block_tile_k}",
            f"v{self.vec_k}",
        )


def is_valid_spec(spec: Img2ColSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch``.

    img2col is a pure index-transform + copy kernel: it has no MFMA, no
    LDS staging, and emits the same IR on every target, so it is
    arch-neutral (builds identically on gfx942 and gfx950). The ``arch``
    parameter (``"gfx942"`` / ``"gfx950"``) is accepted so callers can
    pass it uniformly across the instance surface; it is validated
    against the :class:`rocke.core.arch.ArchTarget` table (an unknown
    gfx name is rejected) but otherwise does not constrain the kernel.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    wave_size = target.wave_size
    if spec.dtype != "f16":
        return False, f"unsupported dtype {spec.dtype!r} (only f16 in v1)"
    if spec.vec_k not in (1, 2, 4, 8):
        return False, f"vec_k must be one of {{1, 2, 4, 8}} (got {spec.vec_k})"
    if spec.block_tile_k % spec.vec_k != 0:
        return (
            False,
            f"block_tile_k {spec.block_tile_k} not divisible by vec_k {spec.vec_k}",
        )
    if spec.block_size <= 0:
        return False, "block_size must be positive"
    if spec.block_size > 1024:
        return (
            False,
            f"block_size {spec.block_size} > 1024 hardware cap "
            f"(block_tile_m {spec.block_tile_m} * block_tile_k "
            f"{spec.block_tile_k} / vec_k {spec.vec_k})",
        )
    if spec.block_size % wave_size != 0:
        return (
            False,
            f"block_size {spec.block_size} not a multiple of "
            f"wave_size ({wave_size}) for {arch}",
        )
    return True, "ok"


def build_img2col(spec: Img2ColSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one image-to-column instance.

    Kernel signature::

        (X: ptr<f16, global>,   # NHWC input image  [N, Hi, Wi, C]
         Y: ptr<f16, global>,   # output unfold     [M, K] = [N*Ho*Wo, R*S*C]
         X_bytes: i32,          # buffer-resource byte length for X
         Y_bytes: i32)          # buffer-resource byte length for Y

    Grid: ``(ceil(K/block_tile_k), ceil(M/block_tile_m), 1)``.

    Per-thread work: one ``vec_k``-wide K chunk at ``(m, k_base)``.
    The output store is always a single ``buffer_store_vN_f16`` per
    thread (one wide-store transaction). The input load is a single
    ``buffer_load_vN_f16`` when ``vec_k`` divides ``C`` (so all
    ``vec_k`` K-slots share one ``(r, s, c0..c0+vec_k-1)`` NHWC block),
    or a ``vec_k``-element scalar gather + ``vec_pack`` otherwise.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid img2col spec for {arch}: {why}")

    p = spec.problem
    V = spec.vec_k

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    X = b.param("X", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    X_bytes = b.param("X_bytes", I32)
    Y_bytes = b.param("Y_bytes", I32)

    A_desc = make_a_descriptor(p)

    c0 = b.const_i32(0)
    c_half_bytes = b.const_i32(2)
    c_M = b.const_i32(p.M)
    c_K = b.const_i32(p.K_gemm)
    c_block_m = b.const_i32(spec.block_tile_m)
    c_block_k = b.const_i32(spec.block_tile_k)
    # Far-OOB sentinel: buffer rsrc OOB clamping silently zero-fills
    # loads / drops stores whose byte offset is past the resource bound
    # (the same lever §6.1 of the runbook uses for tail-safe loads).
    oob_sentinel = b.const_i32((1 << 31) - 1)

    # Per-thread decomposition: tid -> (m_local, k_chunk_local). Each
    # thread owns one (m, k_base..k_base+V-1) chunk along K. With
    # cols_per_row = block_tile_k // V, the chunks tile the block in
    # row-major (m, k_chunk) order. This is the block tile's
    # ``tile_distribution`` decode -- CK Tile's img2col
    # ``MakeBlockTileDistribution`` (``image_to_column_kernel.hpp:158``)
    # splits the flat thread id into (M, K) sub-coords. We express it as
    # a ``merge((block_tile_m, cols_per_row))`` whose inverse is the
    # default magic-division unmerge (``merge_v2_magic_division`` ->
    # :class:`UnmergeMagicDiv`), so the per-thread div/mod becomes the
    # mul-hi sequence the backend would otherwise have to synthesise.
    cols_per_row = spec.block_tile_k // V
    c_V = b.const_i32(V)

    tid = b.thread_id_x()
    tid_unmerge_desc = TensorDescriptor.naive(
        "img2col_tid",
        lengths=[spec.block_tile_m, cols_per_row],
        dtype=F16,
        coord_names=["m_local", "k_chunk_local"],
    ).transform(
        unmerge_magic(
            "tid",
            into=["m_local", "k_chunk_local"],
            dims=[spec.block_tile_m, cols_per_row],
        ),
    )
    decoded = tid_unmerge_desc.unmerge_lower(b, tid=tid)
    m_local = decoded["m_local"]
    k_chunk_local = decoded["k_chunk_local"]
    k_local_base = b.mul(k_chunk_local, c_V) if V > 1 else k_chunk_local
    m_val = b.add(b.mul(b.block_id_y(), c_block_m), m_local)
    k_val_base = b.add(b.mul(b.block_id_x(), c_block_k), k_local_base)

    y_rsrc = b.buffer_rsrc(Y, Y_bytes)

    # Input load via CK Tile ``load_tile`` + a 2D tile_distribution over the
    # ``(block_tile_m, block_tile_k)`` block (A6 idiom, load side). This is
    # the DSL counterpart of ``image_to_column_kernel.hpp::
    # MakeBlockTileDistribution`` (`:158`) feeding the body's bare
    # ``load_tile(image_tile)`` (`:203`): the thread's ``(m_local,
    # k_chunk_local)`` partition (computed above by the unmerge_magic decode,
    # i.e. the distribution's lane->partition index) drives ``P``, and the
    # ``V``-wide K vector is the single ``Y`` dim.
    #
    # The input window is the conv address-transform descriptor wrapped as a
    # buffer ``TensorView`` (``view_from_transforms_descriptor`` over
    # ``A_desc``), so ``window.view.desc.offset((m, k))`` reproduces the same
    # implicit-GEMM NHWC offset the hand-rolled ``A_desc.offset`` produced.
    # The padded / out-of-image K-tail zero-fill is driven by ``mask_fn``:
    # it re-derives the descriptor's ``valid`` predicate and routes masked
    # lanes through the buffer-OOB-zero path (INT32_MAX byte offset) -- the
    # ``pad_tensor_view(..., sequence<false, true>)`` tail idiom, with
    # identical buffer-OOB zero semantics to the previous explicit sentinel.
    #
    # Load shape (a) -- ``can_vector_load`` (C % V == 0) or ``V == 1``: the V
    # K-slots share one NHWC ``(r, s, c0..c0+V-1)`` block and one uniform
    # ``valid``, so the distribution keeps a single ``V``-wide masked vector
    # load (one ``buffer_load_vN_f16``). The K-tail / padding zero-fill comes
    # from ``mask_fn`` (descriptor ``valid`` -> buffer OOB-zero).
    #
    # Load shape (b) -- ``V > 1`` but ``C % V != 0``: the V K-slots are *not*
    # contiguous in NHWC (they straddle an ``(r, s)`` boundary), so they are a
    # genuine non-contiguous gather rather than a vector load. ``load_tile``
    # models contiguous vector accesses, so this fallback keeps the
    # per-element ``A_desc.offset(m, k_base+i)`` gather + ``vec_pack`` (still
    # followed by one wide store below), with the same buffer-OOB sentinel
    # zero-fill semantics.
    x_rsrc_view = make_buffer_resource(b, X, num_bytes=X_bytes)
    if V == 1 or spec.can_vector_load:
        in_view = view_from_transforms_descriptor(
            x_rsrc_view, A_desc, addr_space="buffer", coord_order=["m", "k"]
        )
        in_enc = TileDistributionEncoding(
            Hs=((spec.block_tile_m,), (cols_per_row, V)),
            Ps2RHs_major=((1, 2),),
            Ps2RHs_minor=((0, 0),),
            Ys2RHs_major=(2,),
            Ys2RHs_minor=(1,),
        )
        in_dist = make_static_tile_distribution(in_enc)
        m_base = b.mul(b.block_id_y(), c_block_m)
        k_base = b.mul(b.block_id_x(), c_block_k)
        in_window = in_view.tile(
            lengths=[spec.block_tile_m, spec.block_tile_k], origin=[m_base, k_base]
        )

        def _conv_valid(bb, glob):
            # ``glob`` is (m, k) in the GEMM index space; re-derive the conv
            # descriptor's in-bounds predicate so the OOB / padding zone
            # routes to the buffer zero-fill. ``valid is None`` (no boundary
            # check) means always-in-bounds.
            _off, valid = A_desc.offset(bb, m=glob[0], k=glob[1])
            return valid if valid is not None else bb.cmp_eq(c0, c0)

        in_traits = make_load_store_traits(in_dist, max_vec=V)
        in_dt = load_tile(
            b,
            in_window,
            distribution=in_dist,
            ps=[[m_local, k_chunk_local]],
            traits=in_traits,
            mask_fn=_conv_valid,
        )
        # Reassemble the per-thread V-wide chunk (f32 -> f16) for the store.
        halves = [b.cast_f32_to(in_dt.get([i]), F16) for i in range(V)]
        loaded = halves[0] if V == 1 else b.vec_pack(halves, F16)
    else:
        x_rsrc = x_rsrc_view.rsrc
        gathered: list = []
        for i in range(V):
            k_i = k_val_base if i == 0 else b.add(k_val_base, b.const_i32(i))
            off_i, valid_i = A_desc.offset(b, m=m_val, k=k_i)
            off_i_bytes = b.mul(off_i, c_half_bytes)
            safe_i = (
                b.select(valid_i, off_i_bytes, oob_sentinel)
                if valid_i is not None
                else off_i_bytes
            )
            gathered.append(b.buffer_load_f16(x_rsrc, safe_i, c0))
        loaded = b.vec_pack(gathered, F16)

    # Output store: row-major ``[M, K_gemm]`` so consecutive K halves
    # are consecutive in memory and one ``buffer_store_vN_f16`` writes
    # the whole chunk. ``in_bounds`` only needs to be True for *some*
    # element of the chunk because the rsrc silently drops the trailing
    # OOB halves at the K tail; we keep the leading-element bounds
    # check (``m_val < M`` AND ``k_val_base < K_gemm``) so threads with
    # ``m_val >= M`` get the whole chunk dropped via the OOB sentinel.
    out_off_elems = b.add(b.mul(m_val, c_K), k_val_base)
    out_off_bytes = b.mul(out_off_elems, c_half_bytes)
    m_ok = b.cmp_lt(m_val, c_M)
    k_ok = b.cmp_lt(k_val_base, c_K)
    in_bounds = b.land(m_ok, k_ok)
    safe_out_off = b.select(in_bounds, out_off_bytes, oob_sentinel)
    if V == 1:
        b.buffer_store_f16(y_rsrc, safe_out_off, c0, loaded)
    else:
        b.buffer_store_vN_f16(y_rsrc, safe_out_off, c0, loaded, dwords=V // 2)

    return b.kernel


def img2col_grid(spec: Img2ColSpec) -> Tuple[int, int, int]:
    """Return the launch grid for ``spec``.

    ``grid_x`` covers the K (filter * channel) tiling, ``grid_y`` covers
    the M (batch * output-spatial) tiling. The launch is 2D; the kernel
    re-derives ``(m_local, k_local)`` from ``thread_id_x``.
    """
    return ceil_div_grid(
        (spec.problem.K_gemm, spec.block_tile_k),
        (spec.problem.M, spec.block_tile_m),
    )


_HIP_GRID_AXIS_CAP = 65535


def img2col_block_tile_m_for_M(M: int, *, default: int = 8) -> int:
    """Pick a ``block_tile_m`` that keeps ``grid_y`` under the HIP cap (P85).

    HIP's ``dim3`` grid axes cap at 65535. For ``M > 65535 *
    block_tile_m`` the launch fails with ``hipError(1) invalid
    argument``. This helper walks the standard ``block_tile_m`` ladder
    (8, 16, 32, 64, 128, 256) and returns the smallest value that
    yields ``grid_y = ceil(M / block_tile_m) <= 65535``.

    For shapes within the cap, returns ``default`` (the historical
    8). Mirrors CK Tile's
    ``TransformConvFwdToGemm::GetSplitImageInfo`` autotune table.
    """
    for cand in (default, 16, 32, 64, 128, 256):
        if (M + cand - 1) // cand <= _HIP_GRID_AXIS_CAP:
            return cand
    return 256


def img2col_signature(spec: Img2ColSpec):
    """Manifest-style signature for use with
    :class:`rocke.runtime.launcher.KernelLauncher`.
    """
    return (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Y", spec.dtype)
        .scalar("X_bytes", "i32")
        .scalar("Y_bytes", "i32")
        .build()
    )
