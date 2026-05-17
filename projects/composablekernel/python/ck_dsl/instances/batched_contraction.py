# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Batched contraction kernel instance (CK Tile ``41_batched_contraction`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/41_batched_contraction``.
Generalises :mod:`ck_dsl.instances.batched_gemm` to arbitrary
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

For ck_dsl v1 we flatten the leading batches into a single ``batch =
B_0 * B_1 * ... * B_{r-1}`` axis and delegate to
:func:`ck_dsl.instances.build_batched_gemm`. The caller computes the
per-batch element strides from the flattened layout (typically all
contiguous: ``stride_a = M * K``, ``stride_b = K * N``,
``stride_c = M * N``) and passes them through the standard
``(stride_a, stride_b, stride_c)`` kernel args.

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

import functools
import operator
from dataclasses import dataclass, field
from typing import Sequence, Tuple

from ..core.ir import KernelDef
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
    """Total batch count = product of all leading batch dims."""
    return functools.reduce(operator.mul, shape, 1) if shape else 1


@dataclass(frozen=True)
class BatchedContractionSpec:
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
    name: str = "ck_dsl_batched_contraction"

    def __post_init__(self) -> None:
        if self.block_size == 0:
            t = self.tile
            object.__setattr__(
                self,
                "block_size",
                t.warp_m * t.warp_n * t.warp_k * self.wave_size,
            )

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


def is_valid_spec(spec: BatchedContractionSpec) -> Tuple[bool, str]:
    if any(d <= 0 for d in spec.batch_shape):
        return False, (
            f"batch_shape must be all positive, got {list(spec.batch_shape)}"
        )
    from .batched_gemm import is_valid_spec as _bgemm_valid

    ok, why = _bgemm_valid(spec.to_batched_spec())
    if not ok:
        return False, f"base batched_gemm spec invalid: {why}"
    return True, "ok"


def build_batched_contraction(spec: BatchedContractionSpec) -> KernelDef:
    """Build the IR for one batched contraction instance.

    v1 wraps :func:`build_batched_gemm` with the flattened batch
    count baked into the kernel name; the launcher passes the
    flattened ``batch`` to the standard ``(stride_a, stride_b,
    stride_c)`` arg trio.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid batched_contraction spec: {why}")
    return build_batched_gemm(spec.to_batched_spec())


def batched_contraction_grid(
    spec: BatchedContractionSpec, m: int, n: int
) -> Tuple[int, int, int]:
    """Launch grid: ``(N_tiles, M_tiles, batch_count)``."""
    return batched_gemm_grid(spec.batch_count, m, n, spec.to_batched_spec())


def batched_contraction_signature(spec: BatchedContractionSpec):
    """Same signature as :func:`batched_gemm_signature`."""
    return batched_gemm_signature(spec.to_batched_spec())


def flatten_batch_strides(batch_shape: Sequence[int], inner_size: int) -> int:
    """Helper: contiguous per-batch element stride for an inner
    ``M * K`` (or ``K * N``, etc.) tensor laid out as
    ``[*batch_shape, M, K]`` row-major.

    Useful for the caller pre-computing ``(stride_a, stride_b,
    stride_c)`` from a torch tensor's natural strides.
    """
    return _flatten_batch(batch_shape) and inner_size


__all__ = [
    "BatchedContractionSpec",
    "batched_contraction_grid",
    "batched_contraction_signature",
    "build_batched_contraction",
    "flatten_batch_strides",
    "is_valid_spec",
    # Re-exports for caller convenience.
    "DataSpec",
    "TileSpec",
    "TraitSpec",
    "UniversalGemmSpec",
]
