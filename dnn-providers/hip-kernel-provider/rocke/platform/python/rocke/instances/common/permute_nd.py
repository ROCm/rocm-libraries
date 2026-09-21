# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""N-D tensor permutation kernel instance.

DSL counterpart of CK Tile's ``example/ck_tile/06_permute`` /
``include/ck_tile/ops/permute/kernel/generic_permute_kernel.hpp``.
Computes ``Y = np.transpose(X, perm)`` (equivalently
``y = x.permute(*perm).contiguous()``): output axis ``d`` indexes input
axis ``perm[d]``, so::

    Y[o_0, o_1, ..., o_{n-1}] = X[a_0, a_1, ..., a_{n-1}]
        where a_{perm[d]} == o_d   (i.e. a == o ∘ perm^{-1})

For example ``perm=(2,1,0)`` on a rank-3 tensor is
``y = x.permute(2, 1, 0).contiguous()``. ``Y`` is row-major contiguous
with shape ``(x_shape[perm[0]], x_shape[perm[1]], ..., x_shape[perm[n-1]])``.

Scalar -> tile/vector replacements
----------------------------------

The v1 of this kernel (history: pre-2026-05) was literally the CK Tile
generic permute: one thread per output element, scalar
``global_load_f16`` / ``global_store``, hand-written ``div`` / ``mod``
chain to decompose ``out_idx`` into n-D coords plus a hand-written
``mul`` / ``add`` chain to recompose the source index. The C++
reference even has the comment ``// TODO: hard code to vector 1``,
acknowledging the scalar load was a placeholder.

This v2 keeps the same semantics but routes everything through the
DSL's tile / transform-DAG abstractions:

* The "out_idx -> (o_0, ..., o_{n-1}) -> input flat offset" arithmetic
  is expressed as a :class:`rocke.helpers.transforms.TensorDescriptor` chain
  (``naive`` + ``unmerge`` + ``pass_through``). This is the CK Tile
  ``transform_tensor_descriptor`` algebra: the kernel author writes
  *which* coords feed *which* coords, the descriptor emits the SSA.
  See :doc:`/architecture/TRANSFORM_DAG` for the formalism.

* Global loads / stores go through
  :class:`rocke.helpers.tensor_view.TensorView`'s ``load_vec_at`` /
  ``store_vec_at`` (a precomputed flat offset feeds a vector op of the
  chosen width). This collapses the ``b.global_load_f16(X, idx)`` /
  ``b.global_store(Y, idx, v)`` dispatch boilerplate to one call.

* When the *innermost* output axis equals the *innermost* input axis
  (``perm[-1] == rank - 1``), consecutive output elements come from
  consecutive input elements, so one thread fans out to ``vec`` output
  halves with a single vector load + a single vector store. This is
  the same vectorisation lever the C++ permute kernel has on its TODO
  list ("hard code to vector 1") and we now do it whenever the
  permutation allows.

For permutations that don't share the innermost axis (e.g.
``perm=(2,0,1)`` on a 3-D tensor, where the output's inner axis maps
to the input's middle axis), the per-thread inner load is still scalar
-- there is no contiguous chunk of input to vectorise without
gathering ``vec`` non-adjacent halves. The kernel falls back to the
``vec=1`` path with the same transform-DAG-derived flat offset.

What we cover today
-------------------

* Dtypes ``f16`` / ``bf16``. The CK Tile reference also handles
  ``fp8`` and ``fp32``; f8/f32 just need their own ``load_scalar`` /
  ``load_vec`` dispatch in :mod:`rocke.helpers.tensor_view`.
* Rank up to 8 (the same cap as ``GenericPermuteHostArgs::kMaxRanks``).
* Arbitrary permutations; the kernel is shape-specialised at build
  time (compile-time shape + permutation -> compile-time strides + the
  index-decomposition arithmetic folds at IR construction).
* Vec widths ``{2, 4, 8}`` automatically when ``perm[-1] == rank - 1``
  and ``x_shape[-1] % vec == 0``; falls back to scalar otherwise.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import BF16, F16, IRBuilder, KernelDef, PtrType
from ...helpers.io import io_ir_type
from ...helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join
from ...helpers.tensor_view import make_global_view
from ...helpers.transforms import TensorDescriptor as RichTensorDescriptor
from ...helpers.transforms import pass_through, unmerge


DType = Literal["f16", "bf16"]
MAX_RANK = 8


@dataclass(frozen=True)
class PermuteSpec:
    """One concrete permutation kernel configuration.

    ``x_shape`` is the input tensor shape (row-major contiguous);
    ``perm`` is a tuple of length ``len(x_shape)`` containing a
    permutation of ``range(rank)``. The output shape is then
    ``y_shape[d] = x_shape[perm[d]]``.
    """

    x_shape: Tuple[int, ...]
    perm: Tuple[int, ...]
    dtype: DType = "f16"
    block_size: int = 256
    name: str = "rocke_permute"

    @property
    def rank(self) -> int:
        return len(self.x_shape)

    @property
    def y_shape(self) -> Tuple[int, ...]:
        return tuple(self.x_shape[self.perm[i]] for i in range(self.rank))

    @property
    def total_elements(self) -> int:
        return functools.reduce(operator.mul, self.x_shape, 1)

    @property
    def vec_width(self) -> int:
        """Width of the per-thread inner vector load / store.

        Vectorisation requires that consecutive output elements come
        from consecutive input elements -- otherwise the per-thread
        "vec halves" would have to be gathered from non-adjacent input
        rows, which costs ``vec`` scalar loads anyway and a
        ``vec_pack``. The clean condition for that is the innermost
        output axis being the innermost input axis (``perm[-1] ==
        rank - 1``), with the inner length divisible by the chosen
        vec width.

        We pick the largest vec in ``{8, 4, 2}`` that divides the
        inner length; this matches the dword-x8 / dword-x4 / dword-x2
        global load widths the AMDGPU backend supports for 16-bit
        elements.
        """
        if self.perm[-1] == self.rank - 1:
            inner = self.x_shape[-1]
            for v in (8, 4, 2):
                if inner % v == 0 and self.total_elements % v == 0:
                    return v
            return 1
        # ``perm[-1] != rank - 1``: the innermost output axis maps to a
        # non-innermost input axis, so ``vec`` consecutive output
        # elements come from input elements that are ``stride[perm[-1]]``
        # apart -- not a contiguous chunk. A vec-wide load would gather
        # the wrong halves, so we fall back to the scalar (vec=1) path
        # and let the transform-DAG drive each element's source offset.
        return 1

    def kernel_name(self) -> str:
        shape_str = "x".join(str(d) for d in self.x_shape)
        perm_str = "".join(str(d) for d in self.perm)
        return kernel_name_join(
            self.name,
            f"s{shape_str}",
            f"p{perm_str}",
            self.dtype,
            f"b{self.block_size}",
            f"v{self.vec_width}" if self.vec_width > 1 else "",
        )


def is_valid_spec(spec: PermuteSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for an N-D permute spec on ``arch``.

    The permute is a pure gather/scatter (transform-DAG-driven flat
    offsets, no LDS, no MFMA atom, no gfx950-only ISA feature), so it is
    arch-polymorphic. The block-size upper bound is sourced from
    :class:`rocke.core.arch.ArchTarget` (the supported launch sizes
    ``{64,128,256,512,1024}`` are clamped to the target's
    ``max_threads_per_block``).
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if spec.dtype not in ("f16", "bf16"):
        return False, f"unsupported dtype {spec.dtype!r}"
    _legal_bs = tuple(
        bs for bs in (64, 128, 256, 512, 1024) if bs <= target.max_threads_per_block
    )
    if spec.block_size not in _legal_bs:
        return False, (
            f"block_size {spec.block_size} not in {set(_legal_bs)} on {arch}"
        )
    if spec.rank == 0:
        return False, "rank must be >= 1"
    if spec.rank > MAX_RANK:
        return False, f"rank {spec.rank} > {MAX_RANK} (CK Tile cap)"
    if len(spec.perm) != spec.rank:
        return (
            False,
            f"perm length {len(spec.perm)} != x_shape rank {spec.rank}",
        )
    if sorted(spec.perm) != list(range(spec.rank)):
        return (
            False,
            f"perm {list(spec.perm)} is not a permutation of range({spec.rank})",
        )
    if any(d <= 0 for d in spec.x_shape):
        return False, f"x_shape must be all positive, got {list(spec.x_shape)}"
    if spec.total_elements <= 0:
        return False, f"total_elements must be positive, got {spec.total_elements}"
    return True, "ok"


def _build_offset_descriptor(spec: PermuteSpec) -> RichTensorDescriptor:
    """Build the rich descriptor that maps ``out_idx`` -> input flat offset.

    The chain is the same algebra the CK Tile implicit-GEMM kernel
    uses for ``m -> (n, ho, wo) -> NHWC offset`` -- just specialised
    to a pure permutation:

    .. code-block:: text

        naive("X", lengths=x_shape, coord_names=[i_0, i_1, ..., i_{n-1}])
          + unmerge(upper="out_idx",
                    into=[o_0, o_1, ..., o_{n-1}],
                    dims=y_shape)
          + pass_through(o_{perm[d]} -> i_d)   for d in 0..n-1

    The naive descriptor's base strides are the row-major strides of
    the *input* tensor, so the final offset is the linear input
    offset. Unmerge decomposes ``out_idx`` into the n output coords
    using ``y_shape`` as dims (which is just ``[x_shape[perm[d]] for
    d in 0..n-1]``). The ``pass_through`` chain then renames each
    output coord ``o_{perm[d]}`` to the input coord ``i_d``, so the
    base offset becomes ``sum_d x_strides[d] * o_{perm[d]}`` -- the
    permuted indexing we want.

    Crucially this is *the* descriptor the kernel uses both for the
    scalar path and the vectorised path: in the vec path we just hand
    it the per-thread *base* output index (``thread_out_idx * vec``)
    and the resulting source offset is contiguous along the inner
    axis because we built the descriptor knowing that
    ``perm[-1] == rank - 1`` in the vectorisable case.
    """
    n = spec.rank
    in_coord_names = [f"i_{d}" for d in range(n)]
    out_coord_names = [f"o_{d}" for d in range(n)]

    desc = RichTensorDescriptor.naive(
        "X",
        lengths=list(spec.x_shape),
        coord_names=in_coord_names,
    )
    transforms = [
        unmerge(
            "out_idx",
            into=out_coord_names,
            dims=list(spec.y_shape),
        )
    ]
    for d in range(n):
        # ``np.transpose(X, perm)`` semantics: output axis ``d`` indexes
        # input axis ``perm[d]``, i.e. ``i_{perm[d]} == o_d``. So the
        # rename binds output coord ``o_d`` to input coord
        # ``i_{perm[d]}`` (``pass_through(upper=o_d, lower=i_{perm[d]})``
        # sets ``i_{perm[d]} := o_d``). The naive base then evaluates
        # ``sum_a x_strides[a] * i_a == sum_a x_strides[a] *
        # o_{perm^{-1}(a)}`` -- the row-major-input offset of the
        # element that lands at output position ``out_idx``.
        transforms.append(
            pass_through(out_coord_names[d], in_coord_names[spec.perm[d]])
        )
    return desc.transform(*transforms)


def build_permute(spec: PermuteSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one permutation instance.

    Kernel signature::

        (X: ptr<dtype, global>,   # input  (row-major over x_shape)
         Y: ptr<dtype, global>)   # output (row-major over y_shape)

    The permute is a pure gather/scatter (transform-DAG-driven flat
    offsets, no LDS, no MFMA atom, no cross-lane reduction), so the IR is
    wave-size agnostic and identical for wave64 (gfx950) and wave32
    (gfx1151) targets. ``arch`` is threaded through to
    :func:`is_valid_spec` purely so the block-size bound is validated
    against the target's ``max_threads_per_block``.
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid permute spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    total = spec.total_elements
    vec = spec.vec_width

    in_desc = _build_offset_descriptor(spec)

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)

    # Two CK Tile-style views over the global buffers. The
    # descriptor's offset arithmetic lives in the rich descriptor
    # above; here we just need the address-space-aware load / store
    # dispatch. The view shape is the *output* shape for ``Y_view``
    # and the *input* shape for ``X_view`` (informational; the actual
    # offsets we pass via ``..._at`` come from the rich descriptor).
    X_view = make_global_view(X, shape=spec.x_shape, dtype=io_ty)
    Y_view = make_global_view(Y, shape=spec.y_shape, dtype=io_ty)

    tid = b.thread_id_x()
    bid = b.block_id_x()
    # Each thread is responsible for ``vec`` consecutive output
    # halves starting at output offset ``thread_out_base * vec``.
    thread_out_base = b.add(b.mul(bid, b.const_i32(spec.block_size)), tid)

    if vec > 1:
        # Vectorised path: ``perm[-1] == rank - 1`` guarantees the
        # inner axis is shared between X and Y, so a vec-wide load
        # from the rich descriptor's offset gives us vec contiguous
        # input halves -- which we drop straight into a vec-wide
        # output store. One bounds check (per vec block) covers the
        # whole vec, since ``total_elements % vec == 0`` is part of
        # the spec.vec_width predicate.
        out_idx_base = b.mul(thread_out_base, b.const_i32(vec))
        c_total = b.const_i32(total)
        in_bounds = b.cmp_lt(out_idx_base, c_total)
        with b.scf_if(in_bounds):
            src_offset, _valid = in_desc.offset(b, out_idx=out_idx_base)
            x_vec = X_view.load_vec_at(b, src_offset, n=vec)
            Y_view.store_vec_at(b, out_idx_base, x_vec, n=vec)
    else:
        # Scalar fallback: ``perm[-1] != rank - 1`` (or the inner
        # length doesn't divide our supported vec widths). One thread
        # per output element, same as the C++ reference's
        # ``// TODO: hard code to vector 1`` body. The transform-DAG
        # still drives the source index -- the only thing that
        # differs from the vec path is the access width.
        out_idx = thread_out_base
        c_total = b.const_i32(total)
        in_bounds = b.cmp_lt(out_idx, c_total)
        with b.scf_if(in_bounds):
            src_offset, _valid = in_desc.offset(b, out_idx=out_idx)
            val = X_view.load_scalar_at(b, src_offset)
            Y_view.store_scalar_at(b, out_idx, val)

    return b.kernel


def permute_grid(spec: PermuteSpec) -> Tuple[int, int, int]:
    """Return the launch grid: one thread per ``vec_width`` output halves.

    When ``vec_width > 1`` the grid is ``ceil(total / (vec * block_size))``
    blocks. When the kernel falls back to the scalar path (``vec ==
    1``) it's just one thread per output element, matching the v1
    behaviour and CK Tile's :class:`GenericPermute`.
    """
    threads = (spec.total_elements + spec.vec_width - 1) // spec.vec_width
    return ceil_div_grid((threads, spec.block_size))


def permute_signature(spec: PermuteSpec):
    return SignatureBuilder().ptr("X", spec.dtype).ptr("Y", spec.dtype).build()


# ``BF16`` is intentionally re-exported so callers (e.g. tests) don't
# need to import it via the deeper ``rocke.core.ir`` path just to
# build a PermuteSpec.
__all__ = [
    "BF16",
    "F16",
    "MAX_RANK",
    "PermuteSpec",
    "build_permute",
    "is_valid_spec",
    "permute_grid",
    "permute_signature",
]
