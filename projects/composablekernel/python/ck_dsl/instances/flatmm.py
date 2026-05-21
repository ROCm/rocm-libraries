# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FlatMM kernel instance (CK Tile ``18_flatmm`` parity).

DSL counterpart of CK Tile's ``example/ck_tile/18_flatmm``. FlatMM is
CK Tile's name for a batched matmul that takes preshuffled B (and the
same A and C shapes as standard batched GEMM); it ships in the
upstream example library as an *alternative* to the preshuffled-GEMM
configuration in ``03_gemm``. Algorithmically it is one batched MFMA
GEMM per ``(batch, m_tile, n_tile)`` triple, the only difference being
the per-tile load distribution for B (which expects a preshuffled
layout that fuses the strided ``ds_write`` of the standard pipeline
into the global load).

For ck_dsl v1 we ship the **same body as batched_gemm**: the spec /
kernel name carry the ``flatmm`` tag so callers and benchmarks can
distinguish the two configurations, but the kernel body is shared.
The preshuffled-B load distribution is a v2 feature (it lands with
's ``preshuffle-B`` helper, alongside the FP8 block-scaled GEMM
that also wants the same primitive).

If you need flat-non-batched matmul today, the right move is one of:

* ``build_universal_gemm`` (single-batch, the canonical hero path) -- the
 same kernel CK Tile's ``03_gemm`` ``Preshuffle`` config produces.
* ``build_batched_gemm`` (multi-batch) -- if FlatMM's API surface is
 what you want.
* ``build_flatmm`` (this file) -- if you want a kernel name + spec
 that documents the FlatMM intent so a sweep / dispatcher can route
 it appropriately.

Once lands ``helpers/preshuffle.py`` and the FP8 block-scaled
GEMM, FlatMM gains its own kernel body and stops aliasing
``batched_gemm``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

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


@dataclass(frozen=True)
class FlatMMSpec:
    """One concrete FlatMM kernel configuration.

    Mirrors :class:`BatchedGemmSpec` (since the v1 kernel body is
    shared) with two FlatMM-specific extras:

    * ``name`` defaults to ``ck_dsl_flatmm`` so the kernel symbol
    carries the FlatMM tag.
    * ``preshuffle_b`` (default False) is on the spec surface so
    callers can write the FlatMM-with-preshuffle intent today; v1
    rejects ``True`` at build time, v2 wires the
    preshuffled-B load distribution.
    """

    tile: TileSpec
    trait: TraitSpec = field(default_factory=TraitSpec)
    wave_size: int = 64
    block_size: int = 0
    batch_size: int = 0
    preshuffle_b: bool = False
    name: str = "ck_dsl_flatmm"

    def __post_init__(self) -> None:
        if self.block_size == 0:
            t = self.tile
            object.__setattr__(
                self,
                "block_size",
                t.warp_m * t.warp_n * t.warp_k * self.wave_size,
            )

    def to_batched_spec(self) -> BatchedGemmSpec:
        # ``BatchedGemmSpec.kernel_name()`` will use this name as the
        # *prefix* and append the per-config string + flags; passing
        # our own prefix here keeps the resulting kernel symbol scoped
        # to the FlatMM family (and tags the preshuffle variant).
        prefix = self.name + ("_psb" if self.preshuffle_b else "")
        return BatchedGemmSpec(
            name=prefix,
            tile=self.tile,
            trait=self.trait,
            wave_size=self.wave_size,
            block_size=self.block_size,
            batch_size=self.batch_size,
        )

    def kernel_name(self) -> str:
        return self.to_batched_spec().kernel_name()


def is_valid_spec(spec: FlatMMSpec) -> Tuple[bool, str]:
    if spec.preshuffle_b:
        return False, (
            "preshuffle_b=True is a feature (waiting on "
            "helpers/preshuffle.py); shipping as v2"
        )
    from .batched_gemm import is_valid_spec as _bgemm_valid

    ok, why = _bgemm_valid(spec.to_batched_spec())
    if not ok:
        return False, f"base batched_gemm spec invalid: {why}"
    return True, "ok"


def build_flatmm(spec: FlatMMSpec) -> KernelDef:
    """Build the IR for one FlatMM instance.

    In v1 this is a re-export of :func:`build_batched_gemm` with the
    FlatMM kernel name. The runtime ABI is identical, so any launcher
    that already understands the batched_gemm signature can drive a
    FlatMM kernel unchanged.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid flatmm spec: {why}")
    return build_batched_gemm(spec.to_batched_spec())


def flatmm_grid(spec: FlatMMSpec, batch: int, m: int, n: int) -> Tuple[int, int, int]:
    """Same launch grid as :func:`build_batched_gemm`."""
    return batched_gemm_grid(batch, m, n, spec.to_batched_spec())


def flatmm_signature(spec: FlatMMSpec):
    """Manifest-style signature mirroring :func:`batched_gemm_signature`."""
    return batched_gemm_signature(spec.to_batched_spec())


__all__ = [
    "FlatMMSpec",
    "build_flatmm",
    "flatmm_grid",
    "flatmm_signature",
    "is_valid_spec",
    # Re-exports for caller convenience.
    "DataSpec",
    "TileSpec",
    "TraitSpec",
    "UniversalGemmSpec",
]
