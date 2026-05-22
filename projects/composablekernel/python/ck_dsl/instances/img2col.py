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
convolution uses (:func:`ck_dsl.instances.conv_implicit_gemm.make_a_descriptor`),
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

from ..core.ir import F16, I32, IRBuilder, KernelDef, PtrType
from ..helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join
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
    name: str = "ck_dsl_img2col"

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


def is_valid_spec(spec: Img2ColSpec) -> Tuple[bool, str]:
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
    if spec.block_size % 64 != 0:
        return False, f"block_size {spec.block_size} not a multiple of wave_size (64)"
    return True, "ok"


def build_img2col(spec: Img2ColSpec) -> KernelDef:
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
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid img2col spec: {why}")

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
    # row-major (m, k_chunk) order.
    cols_per_row = spec.block_tile_k // V
    c_cols_per_row = b.const_i32(cols_per_row)
    c_V = b.const_i32(V)

    tid = b.thread_id_x()
    m_local = b.div(tid, c_cols_per_row)
    k_chunk_local = b.mod(tid, c_cols_per_row)
    k_local_base = b.mul(k_chunk_local, c_V) if V > 1 else k_chunk_local
    m_val = b.add(b.mul(b.block_id_y(), c_block_m), m_local)
    k_val_base = b.add(b.mul(b.block_id_x(), c_block_k), k_local_base)

    x_rsrc = b.buffer_rsrc(X, X_bytes)
    y_rsrc = b.buffer_rsrc(Y, Y_bytes)

    # Input load. Two paths:
    #
    # (a) ``can_vector_load`` (C % V == 0, V > 1): the V K-slots of this
    #     chunk live in one NHWC ``(r, s, c0..c0+V-1)`` block, so the
    #     leading offset from ``A_desc.offset(m, k_base)`` is the start
    #     of V contiguous halves and one ``buffer_load_vN_f16`` covers
    #     the chunk. Validity is uniform across the chunk (same h, w
    #     pad mask + same r/s indices), so the single ``valid`` from
    #     the descriptor is the right OOB-routing predicate.
    #
    # (b) Scalar fallback: V independent ``A_desc.offset(m, k_base+i)``
    #     queries, V scalar buffer loads (each with its own OOB
    #     routing), packed into a ``<V x f16>`` vector for the wide
    #     store. This handles ``C`` not divisible by V and the
    #     ``V == 1`` recovery case.
    if V == 1:
        offset, valid = A_desc.offset(b, m=m_val, k=k_val_base)
        off_bytes = b.mul(offset, c_half_bytes)
        safe_in_off = (
            b.select(valid, off_bytes, oob_sentinel) if valid is not None else off_bytes
        )
        loaded = b.buffer_load_f16(x_rsrc, safe_in_off, c0)
    elif spec.can_vector_load:
        offset, valid = A_desc.offset(b, m=m_val, k=k_val_base)
        off_bytes = b.mul(offset, c_half_bytes)
        safe_in_off = (
            b.select(valid, off_bytes, oob_sentinel) if valid is not None else off_bytes
        )
        loaded = b.buffer_load_vN_f16(x_rsrc, safe_in_off, c0, dwords=V // 2)
    else:
        # V > 1 but C % V != 0: fall back to per-element scalar gather.
        # We still get a wide store, just not a wide load.
        halves: list = []
        for i in range(V):
            k_i = k_val_base if i == 0 else b.add(k_val_base, b.const_i32(i))
            off_i, valid_i = A_desc.offset(b, m=m_val, k=k_i)
            off_i_bytes = b.mul(off_i, c_half_bytes)
            safe_i = (
                b.select(valid_i, off_i_bytes, oob_sentinel)
                if valid_i is not None
                else off_i_bytes
            )
            halves.append(b.buffer_load_f16(x_rsrc, safe_i, c0))
        loaded = b.vec_pack(halves, F16)

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
    :class:`ck_dsl.runtime.launcher.KernelLauncher`.
    """
    return (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Y", spec.dtype)
        .scalar("X_bytes", "i32")
        .scalar("Y_bytes", "i32")
        .build()
    )
