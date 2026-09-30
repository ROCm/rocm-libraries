# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Batched contraction kernel instance (CK Tile ``41_batched_contraction`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/41_batched_contraction``.
Generalises :mod:`rocke.instances.common.batched_gemm` to arbitrary
*leading-batch* ranks: instead of one batch axis, the caller can pass
N batch dims that fold into the single ``batch`` axis the kernel
launches over.

In symbols, a CK Tile batched contraction looks like::

 A: [B_0, B_1, ..., B_{r-1}, M, K]
 B: [B_0, B_1, ..., B_{r-1}, K, N]
 C: [B_0, B_1, ..., B_{r-1}, M, N]

 for (b_0, ..., b_{r-1}) in batches:
 C[b_0, ..., b_{r-1}, :, :] = A[b_0, ..., b_{r-1}, :, :] @
 B[b_0, ..., b_{r-1}, :, :]

For rocke v1 we flatten the leading batches into a single ``batch =
B_0 * B_1 * ... * B_{r-1}`` axis and delegate to
:func:`rocke.instances.build_batched_gemm`. The caller computes the
per-batch element strides from the flattened layout (typically all
contiguous: ``stride_a = M * K``, ``stride_b = K * N``,
``stride_c = M * N``) and passes them through the standard
``(stride_a, stride_b, stride_c)`` kernel args.

Mapping to CK Tile primitives (``include/ck_tile/ops/batched_contraction``):

* Host-side flatten of ``(G_0, ..., G_{r-1}) -> G_total`` is the same
  product reduction CK Tile's
  ``BatchedContractionKernel::MakeKernelArgs`` runs on ``G_dims``;
  algebraically it is a :class:`rocke.helpers.transforms.Merge` over the
  batch axes (``make_merge_transform(dims_G)`` in
  ``tensor_descriptor_utils.hpp``).
* Per-batch device-time pointer offset (``i_batch_flat *
  batch_stride_X``) is emitted inside the universal-GEMM body
  (``gemm_universal.py``) as a single uniform i32 multiply per CTA,
  matching the C++ ``blockIdx.y * kargs.batch_stride_X``.
* The multi-dim M / N / K flatten (``make_merge_transform(dims_M)``
  etc. in ``tensor_descriptor_utils.hpp``) is expressed here as
  optional descriptor helpers (:func:`make_contraction_a_descriptor`
  and friends) that callers can compose with their own kernel
  bodies when they want non-flat M / N / K.

What this v1 covers:

* Arbitrary number of leading batch dims (rank up to 8 -- the CK Tile
 cap; we don't enforce it, since the flatten is purely host-side).
* Standard ``[..., M, K] x [..., K, N] -> [..., M, N]`` contraction.

Future v2:

* Permuted contraction (``ck_tile::Contraction``'s
 ``in_layout`` / ``out_layout`` template params).
* Multi-A / multi-B contractions that fuse a pre-tensor mul into the
 load distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from ...core.ir import F16, KernelDef, Type
from ...helpers.spec import WarpTileBlockSizeMixin
from ...helpers.transforms import (
    Merge,
    TensorDescriptor,
    Unmerge,
    merge,
    pass_through,
    unmerge,
)
from .batched_gemm import (
    BatchedGemmSpec,
    batched_gemm_grid,
    batched_gemm_signature,
    build_batched_gemm,
)
from .gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
)


def _flatten_batch(shape: Sequence[int]) -> int:
    """Total batch count = product of all leading batch dims.

    Algebraically this is the scalar evaluation of a
    :class:`rocke.helpers.transforms.Merge` over the batch axes: given
    ``dims = batch_shape``, the flattened batch index is the linear
    combination ``b_0 * prod(dims[1:]) + b_1 * prod(dims[2:]) + ...``
    and the total count is ``prod(dims)``. We keep this as a pure
    host-time helper (no SSA emission) so callers can size launches
    and torch buffers; the matching device-side reconstruction lives
    in :func:`batch_unmerge_transform` for kernels that need
    per-dim batch coordinates inside the body.

    Returns 1 for an empty ``shape`` (the natural identity of an
    empty product, matching ``math.prod([])``).
    """
    return math.prod(shape)


@dataclass(frozen=True)
class BatchedContractionSpec(WarpTileBlockSizeMixin):
    """One concrete batched-contraction kernel configuration.

    Mirrors :class:`BatchedGemmSpec` (which is itself a thin wrapper
    over :class:`UniversalGemmSpec`); the only extra field is the
    ``batch_shape`` tuple, which the launcher uses to compute the
    total ``batch`` axis at runtime.
    """

    tile: TileSpec
    batch_shape: Tuple[int, ...] = field(default_factory=tuple)
    trait: TraitSpec = field(default_factory=TraitSpec)
    wave_size: int = 64
    block_size: int = 0
    name: str = "rocke_batched_contraction"

    def __post_init__(self) -> None:
        self._init_block_size()

    @property
    def batch_count(self) -> int:
        return _flatten_batch(self.batch_shape)

    def to_batched_spec(self) -> BatchedGemmSpec:
        # See ``flatmm.py`` for the same naming convention: a custom
        # prefix tag scopes the resulting kernel symbol to the
        # batched-contraction family while delegating to
        # ``BatchedGemmSpec`` for the per-config suffix.
        shape_tag = "x".join(str(d) for d in self.batch_shape) or "scalar"
        prefix = f"{self.name}_b{shape_tag}"
        return BatchedGemmSpec(
            name=prefix,
            tile=self.tile,
            trait=self.trait,
            wave_size=self.wave_size,
            block_size=self.block_size,
            batch_size=self.batch_count,
        )

    def kernel_name(self) -> str:
        return self.to_batched_spec().kernel_name()


def is_valid_spec(
    spec: BatchedContractionSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch``.

    ``arch`` (``"gfx942"`` / ``"gfx950"``) is forwarded to the
    underlying :func:`rocke.instances.common.batched_gemm.is_valid_spec`, which
    sources the legal MFMA warp-tile atoms + LDS cap from
    :class:`rocke.core.arch.ArchTarget`. A warp-tile atom that exists on
    gfx950 but not gfx942 (e.g. the wide ``32x32x16`` f16 atom) is
    rejected for gfx942 with a structured reason rather than crashing
    comgr at lower time.
    """
    if any(d <= 0 for d in spec.batch_shape):
        return False, (
            f"batch_shape must be all positive, got {list(spec.batch_shape)}"
        )
    from .batched_gemm import is_valid_spec as _bgemm_valid

    ok, why = _bgemm_valid(spec.to_batched_spec(), arch=arch)
    if not ok:
        return False, f"base batched_gemm spec invalid: {why}"
    return True, "ok"


def build_batched_contraction(
    spec: BatchedContractionSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the IR for one batched contraction instance.

    v1 wraps :func:`build_batched_gemm` with the flattened batch
    count baked into the kernel name; the launcher passes the
    flattened ``batch`` to the standard ``(stride_a, stride_b,
    stride_c)`` arg trio.

    ``arch`` (``"gfx942"`` / ``"gfx950"``) selects the target GPU and is
    forwarded to :func:`build_batched_gemm`, which validates the MFMA
    warp-tile atom against the :class:`rocke.core.arch.ArchTarget`
    catalog before lowering. Requesting an atom that only exists on
    gfx950 (e.g. the default wide ``32x32x16`` f16 atom) for ``gfx942``
    fails with a clean structured error instead of an ``LLVM ERROR``.
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid batched_contraction spec for {arch}: {why}")
    return build_batched_gemm(spec.to_batched_spec(), arch=arch)


def batched_contraction_grid(
    spec: BatchedContractionSpec, m: int, n: int
) -> Tuple[int, int, int]:
    """Launch grid: ``(N_tiles, M_tiles, batch_count)``."""
    return batched_gemm_grid(spec.batch_count, m, n, spec.to_batched_spec())


def batched_contraction_signature(spec: BatchedContractionSpec):
    """Same signature as :func:`batched_gemm_signature`."""
    return batched_gemm_signature(spec.to_batched_spec())


def flatten_batch_strides(batch_shape: Sequence[int], inner_size: int) -> int:
    """Per-batch element stride along the flat batch axis for a
    row-major ``[*batch_shape, *inner_shape]`` tensor.

    For a contiguous row-major tensor whose batch axes pack into the
    leading rank, the increment of the flat batch index by 1 advances
    the element offset by exactly ``inner_size = prod(inner_shape)``
    -- regardless of how many leading batch dims there are. This
    mirrors what CK Tile's ``BatchedContractionKernel::MakeKernelArgs``
    extracts as ``A_strides[NumDimG - 1]`` from the user-supplied
    stride vector: the stride of the *last* batch dim in a row-major
    layout is the inner-tile element count.

    Useful for the caller pre-computing ``(stride_a, stride_b,
    stride_c)`` from a torch tensor's natural strides.

    Edge cases:

    * Empty ``batch_shape`` -> single batch; the kernel never
      multiplies by the stride (``block_id_z == 0``), so we still
      return ``inner_size`` as a defensible default.
    * Any ``batch_shape`` entry equal to 0 -> the tensor is empty;
      we return ``0`` so a launcher that still calls into the
      kernel with ``batch_count == 0`` produces a no-op pointer
      advance. This preserves the prior behaviour (the old
      implementation returned ``0`` here via ``and`` short-circuit).
    """
    if not batch_shape:
        return int(inner_size)
    if any(int(d) <= 0 for d in batch_shape):
        return 0
    return int(inner_size)


# ---------------------------------------------------------------------
# Transform-DAG helpers: opt-in tile-level descriptors for callers
# that want richer multi-dim batch / M / N / K addressing.
# ---------------------------------------------------------------------
#
# These mirror CK Tile's ``TensorDescriptorUtils`` in
# ``include/ck_tile/ops/batched_contraction/utils/
# tensor_descriptor_utils.hpp``:
#
#   make_merge_transform(dims_G) -> batch_merge_transform(batch_shape)
#   make_merge_transform(dims_M) -> M_merge inside
#                                   make_contraction_*_descriptor
#   make_merge_transform(dims_N) -> N_merge inside
#                                   make_contraction_*_descriptor
#   make_merge_transform(dims_K) -> K_merge inside
#                                   make_contraction_a_descriptor
#                                   / make_contraction_b_descriptor
#
# The v1 ``build_batched_contraction`` does NOT consume these helpers
# -- the kernel body assumes a flat 2D ``[M, K]`` / ``[N, K]`` / ``[M,
# N]`` tile layout. Callers that want to wire the richer descriptors
# through their own kernel body can use
# :func:`rocke.helpers.tensor_view.view_from_transforms_descriptor`
# to plug a returned :class:`TensorDescriptor` into
# ``make_tile_window``.


def batch_merge_transform(
    batch_shape: Sequence[int],
    *,
    into: str = "batch",
    names: Optional[Sequence[str]] = None,
) -> Merge:
    """Build a :class:`rocke.helpers.transforms.Merge` for the batch flatten.

    Algebraically the same as CK Tile's
    ``make_merge_transform(dims_G)`` -- given user-level coords
    ``(g_0, g_1, ..., g_{r-1})`` it produces the flat ``batch``
    coord via ``g_0 * prod(dims[1:]) + g_1 * prod(dims[2:]) + ... +
    g_{r-1}``.

    Use this when a fused-epilogue kernel needs to consume the
    multi-dim batch coords at the kernel body level (e.g. a residual
    tensor laid out as ``[B_0, B_1, M, N]`` where the user-facing
    epilogue wants ``(b_0, b_1)`` rather than the host-flattened
    ``b_flat``). Pair with :func:`batch_unmerge_transform` to recover
    ``(g_0, ..., g_{r-1})`` from a flat ``block_id_z`` SSA value.

    For a degenerate empty ``batch_shape`` we still return a single-
    coord ``Merge`` with one trivial axis so the transform chain is
    consistent across rank-0 / rank-N contractions.
    """
    shape = tuple(int(d) for d in batch_shape) if batch_shape else (1,)
    if names is None:
        names = tuple(f"g{i}" for i in range(len(shape)))
    else:
        names = tuple(names)
    if len(names) != len(shape):
        raise ValueError(
            f"batch_merge_transform: len(names)={len(names)} "
            f"!= len(batch_shape)={len(shape)}"
        )
    return merge(names, into=into, dims=shape)


def batch_unmerge_transform(
    batch_shape: Sequence[int],
    *,
    from_name: str = "batch",
    into: Optional[Sequence[str]] = None,
) -> Unmerge:
    """Build a :class:`rocke.helpers.transforms.Unmerge` for the inverse split.

    The dual of :func:`batch_merge_transform`: consumes the flat
    ``batch`` upper coord and produces ``(g_0, ..., g_{r-1})`` via
    the standard row-major division-and-modulo. CK Tile's
    ``ck_tile::Unmerge`` (the inverse of ``ck_tile::Merge``) is the
    direct counterpart.

    The intended use is inside a kernel that picks up ``batch =
    block_id_z`` and wants to feed per-dim batch coords into a
    descriptor chain (typically for a fused-epilogue tensor whose
    layout is ``[B_0, B_1, ..., M, N]`` with non-contiguous batch
    strides).
    """
    shape = tuple(int(d) for d in batch_shape) if batch_shape else (1,)
    if into is None:
        into = tuple(f"g{i}" for i in range(len(shape)))
    else:
        into = tuple(into)
    if len(into) != len(shape):
        raise ValueError(
            f"batch_unmerge_transform: len(into)={len(into)} "
            f"!= len(batch_shape)={len(shape)}"
        )
    return unmerge(from_name, into, dims=shape)


def _row_major_strides(lengths: Sequence[int]) -> Tuple[int, ...]:
    """Row-major (C-order) packed strides for ``lengths``.

    ``strides[i] = prod(lengths[i+1:])``. Equivalent to the
    ``make_naive_tensor_descriptor`` default in CK Tile when the
    user does not pass an explicit stride vector.
    """
    out = []
    running = 1
    for d in reversed(lengths):
        out.append(running)
        running *= int(d)
    return tuple(reversed(out))


def _validate_axes_positive(name: str, shape: Sequence[int]) -> None:
    if any(int(d) <= 0 for d in shape):
        raise ValueError(f"{name}: every axis must be positive, got {list(shape)}")


def _build_contraction_descriptor(
    *,
    batch_shape: Sequence[int],
    leading_shape: Sequence[int],
    trailing_shape: Sequence[int],
    leading_name: str,
    trailing_name: str,
    strides: Optional[Sequence[int]],
    name: str,
    dtype: Type,
    flatten_batch_axis: bool,
    batch_name: str,
) -> TensorDescriptor:
    """Internal: build a contraction operand descriptor.

    Shared by :func:`make_contraction_a_descriptor`,
    :func:`make_contraction_b_descriptor`, and
    :func:`make_contraction_e_descriptor`. The operand layout is
    ``[*batch_shape, *leading_shape, *trailing_shape]`` row-major
    (or with caller-supplied ``strides``).

    Naive coord space: ``(g_0, g_1, ..., l_0, l_1, ..., t_0, t_1, ...)``
    (matches the actual memory layout). User-facing upper coord space:
    ``(batch, leading_name, trailing_name)`` -- one flat coord per dim
    group -- when ``flatten_batch_axis=True``.

    To present a flat user-facing coord and split it back into the
    naive multi-dim coords we use :class:`Unmerge` (split, the
    inverse of :class:`Merge` in ``rocke.helpers.transforms``). For
    single-axis groups we use :func:`pass_through` for an identity
    rename so the user-facing surface is shape-invariant.

    Algebraically this matches CK Tile's
    ``transform_tensor_descriptor(naive, make_merge_transform(dims))``
    pattern from ``tensor_descriptor_utils.hpp``. CK Tile's
    ``MergeTransform`` consumes the flat new-view coord on the
    *upper* side and provides the multi-dim naive coords on the
    *lower* side; rocke's :class:`Unmerge` matches that
    upper/lower split (the multi-to-flat direction is what
    rocke names :class:`Merge`).
    """
    _validate_axes_positive(f"{leading_name}_shape", leading_shape)
    _validate_axes_positive(f"{trailing_name}_shape", trailing_shape)
    if batch_shape:
        _validate_axes_positive("batch_shape", batch_shape)

    g_names = tuple(f"g{i}" for i in range(len(batch_shape)))
    l_names = tuple(f"{leading_name}{i}" for i in range(len(leading_shape)))
    t_names = tuple(f"{trailing_name}{i}" for i in range(len(trailing_shape)))
    base_names = g_names + l_names + t_names
    base_lengths = tuple(
        int(d) for d in (*batch_shape, *leading_shape, *trailing_shape)
    )
    if strides is None:
        base_strides = _row_major_strides(base_lengths)
    else:
        strides = tuple(int(s) for s in strides)
        if len(strides) != len(base_lengths):
            raise ValueError(
                f"strides length must match len(batch_shape)+"
                f"len({leading_name}_shape)+len({trailing_name}_shape) = "
                f"{len(base_lengths)}, got {len(strides)}"
            )
        base_strides = strides

    desc = TensorDescriptor.naive(
        name,
        lengths=base_lengths,
        strides=base_strides,
        dtype=dtype,
        coord_names=base_names,
    )

    transforms = []
    if flatten_batch_axis and batch_shape:
        if len(g_names) > 1:
            transforms.append(
                unmerge(
                    batch_name,
                    g_names,
                    dims=tuple(int(d) for d in batch_shape),
                )
            )
        elif g_names[0] != batch_name:
            transforms.append(pass_through(batch_name, into=g_names[0]))
    if len(leading_shape) > 1:
        transforms.append(
            unmerge(
                leading_name,
                l_names,
                dims=tuple(int(d) for d in leading_shape),
            )
        )
    elif l_names and l_names[0] != leading_name:
        transforms.append(pass_through(leading_name, into=l_names[0]))
    if len(trailing_shape) > 1:
        transforms.append(
            unmerge(
                trailing_name,
                t_names,
                dims=tuple(int(d) for d in trailing_shape),
            )
        )
    elif t_names and t_names[0] != trailing_name:
        transforms.append(pass_through(trailing_name, into=t_names[0]))

    if not transforms:
        return desc
    return desc.transform(*transforms)


def make_contraction_a_descriptor(
    *,
    batch_shape: Sequence[int],
    m_shape: Sequence[int],
    k_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    name: str = "A_contraction",
    dtype: Type = F16,
    flatten_batch_axis: bool = True,
    batch_name: str = "batch",
    m_name: str = "m",
    k_name: str = "k",
) -> TensorDescriptor:
    """Build a CK-Tile-style descriptor for the A operand of an N-D
    batched contraction.

    Layout assumption: ``A[g_0, g_1, ..., m_0, m_1, ..., k_0, k_1,
    ...]`` row-major (or with caller-supplied ``strides``). The
    returned :class:`TensorDescriptor`:

    * exposes user-facing coords ``(batch_name, m_name, k_name)``
      (or, if ``flatten_batch_axis=False``, exposes the individual
      ``g_i`` along with ``m_name`` and ``k_name``);
    * routes those upper coords through a :class:`Merge` chain so
      ``.offset(b, batch=..., m=..., k=...)`` emits the same SSA
      arithmetic CK Tile's ``Make_A_GridDescriptor_M_K`` would emit
      after flattening the M / K dim groups.

    This mirrors CK Tile's
    ``TensorDescriptorUtils<NumDimG, NumDimM, NumDimN,
    NumDimK>::Make_A_GridDescriptor_M_K``. The shipped v1
    ``build_batched_contraction`` does NOT plug this descriptor in
    (the kernel body is still a flat 2D ``[M, K]`` walk via
    :func:`build_batched_gemm`); callers writing their own kernel
    body can route this descriptor through
    :func:`rocke.helpers.tensor_view.view_from_transforms_descriptor`
    when they need real multi-dim addressing.

    Parameters
    ----------
    batch_shape, m_shape, k_shape:
        Per-axis lengths. ``batch_shape`` may be empty (no batch
        rank); ``m_shape`` and ``k_shape`` must be at least 1D.
    strides:
        Optional explicit per-axis stride vector for the full
        ``[*batch_shape, *m_shape, *k_shape]`` shape. Default is
        row-major packed.
    flatten_batch_axis:
        When ``True`` (default) the descriptor exposes a single
        flat ``batch`` upper coord (matching the v1 kernel ABI's
        ``stride_a * block_id_z`` per-batch pointer). When
        ``False`` the per-dim ``g_i`` coords are exposed
        directly (useful for callers that compute them outside the
        descriptor, e.g. via :func:`batch_unmerge_transform` on
        ``block_id_z``).
    """
    return _build_contraction_descriptor(
        batch_shape=batch_shape,
        leading_shape=m_shape,
        trailing_shape=k_shape,
        leading_name=m_name,
        trailing_name=k_name,
        strides=strides,
        name=name,
        dtype=dtype,
        flatten_batch_axis=flatten_batch_axis,
        batch_name=batch_name,
    )


def make_contraction_b_descriptor(
    *,
    batch_shape: Sequence[int],
    n_shape: Sequence[int],
    k_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    name: str = "B_contraction",
    dtype: Type = F16,
    flatten_batch_axis: bool = True,
    batch_name: str = "batch",
    n_name: str = "n",
    k_name: str = "k",
) -> TensorDescriptor:
    """B-operand counterpart of :func:`make_contraction_a_descriptor`.

    Layout assumption: ``B[g_0, ..., n_0, n_1, ..., k_0, k_1, ...]``.
    Exposes upper coords ``(batch_name, n_name, k_name)`` and emits
    the same dim-flatten SSA chain CK Tile's
    ``Make_B_GridDescriptor_N_K`` does.
    """
    return _build_contraction_descriptor(
        batch_shape=batch_shape,
        leading_shape=n_shape,
        trailing_shape=k_shape,
        leading_name=n_name,
        trailing_name=k_name,
        strides=strides,
        name=name,
        dtype=dtype,
        flatten_batch_axis=flatten_batch_axis,
        batch_name=batch_name,
    )


def make_contraction_e_descriptor(
    *,
    batch_shape: Sequence[int],
    m_shape: Sequence[int],
    n_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    name: str = "E_contraction",
    dtype: Type = F16,
    flatten_batch_axis: bool = True,
    batch_name: str = "batch",
    m_name: str = "m",
    n_name: str = "n",
) -> TensorDescriptor:
    """Output-operand counterpart of :func:`make_contraction_a_descriptor`.

    Layout assumption: ``E[g_0, ..., m_0, m_1, ..., n_0, n_1, ...]``.
    Exposes upper coords ``(batch_name, m_name, n_name)`` and emits
    the same dim-flatten SSA chain CK Tile's
    ``Make_E_GridDescriptor_M_N`` does.
    """
    return _build_contraction_descriptor(
        batch_shape=batch_shape,
        leading_shape=m_shape,
        trailing_shape=n_shape,
        leading_name=m_name,
        trailing_name=n_name,
        strides=strides,
        name=name,
        dtype=dtype,
        flatten_batch_axis=flatten_batch_axis,
        batch_name=batch_name,
    )


__all__ = [
    "BatchedContractionSpec",
    "batched_contraction_grid",
    "batched_contraction_signature",
    "build_batched_contraction",
    "flatten_batch_strides",
    "is_valid_spec",
    # Transform-DAG helpers (opt-in tile-level extensions).
    "batch_merge_transform",
    "batch_unmerge_transform",
    "make_contraction_a_descriptor",
    "make_contraction_b_descriptor",
    "make_contraction_e_descriptor",
    # Re-exports for caller convenience.
    "DataSpec",
    "TileSpec",
    "TraitSpec",
    "UniversalGemmSpec",
]
