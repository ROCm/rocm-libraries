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

Where the tile-level ops live
=============================

There is **no scalar inner loop in this file**. The compute body is
:func:`rocke.instances.build_batched_gemm` (which delegates to
:func:`rocke.instances.build_universal_gemm`), which:

* drives the K-loop with ``scf_for_iter`` and per-K-tile MFMA atom
  invocations (no scalar FMUL/FADD per output cell);
* loads A/B tiles via :class:`~rocke.helpers.loads.CoalescedTileLoader`
  / :class:`~rocke.helpers.loads.AsyncTileLoader` (vector
  ``buffer_load_vN_f16`` / ``raw_ptr_buffer_load_lds`` rather than
  per-element scalar loads);
* picks the **K-packed** MFMA atom (``f16_32x32x16`` on f16 by
  default; ``f16_16x16x32`` for the 16x16 hero), halving the K-loop
  trip count vs. the legacy 32x32x8 / 16x16x16 atoms -- same lever CK
  Tile's ``FlatmmConfig32`` selects via ``K_Warp_Tile = sizeof(f16) ==
  2 ? 16 : 32`` (see
  ``example/ck_tile/18_flatmm/flatmm_basic.hpp::FlatmmConfig32``);
* writes the f32 accumulator to LDS via the ``cshuffle`` epilogue
  (vector ``buffer_store_vN_f16`` per thread instead of scalar
  ``buffer_store_short`` per output cell).

So the v1 spec / wrapper carries the **dispatch knobs** (tile shape,
warp grid, pipeline, epilogue, preshuffle intent); the tile-level
ops live one layer down in ``gemm_universal``.

Preshuffled-B layout
====================

The preshuffled-B *layout descriptor* lives in
:mod:`rocke.helpers.preshuffle` today; :func:`flatmm_preshuffle_b_layout`
exposes it through this module so a host-side launcher can permute B
into ``[k_tiles, n_tiles, block_n, block_k]`` layout *before* the
kernel launch -- that is the host side of the FlatMM contract. The
matching kernel-side per-lane preshuffled-B load path lives in
``helpers/preshuffle.py::emit_preshuffleb_offset`` and is the v2
hook ``gemm_universal`` consumes once the FlatMM pipeline body lands;
``build_flatmm`` continues to reject ``preshuffle_b=True`` at IR-build
time until then.

If you need flat-non-batched matmul today, the right move is one of:

* :func:`rocke.instances.build_universal_gemm` (single-batch, the
  canonical hero path) -- the same kernel CK Tile's ``03_gemm``
  ``Preshuffle`` config produces.
* :func:`rocke.instances.build_batched_gemm` (multi-batch) -- if
  FlatMM's API surface is what you want.
* :func:`build_flatmm` (this file) -- if you want a kernel name + spec
  that documents the FlatMM intent so a sweep / dispatcher can route
  it appropriately.

Once the preshuffled-B kernel body lands in ``gemm_universal``,
FlatMM gains its own kernel body and stops aliasing ``batched_gemm``;
the spec surface (``preshuffle_b``, ``flatmm_preshuffle_b_layout``)
stays unchanged so callers don't need to follow the cutover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ...core.ir import KernelDef
from ...helpers.atoms import MfmaAtom, mfma_atom
from ...helpers.spec import WarpTileBlockSizeMixin
from ...helpers.preshuffle import PreshuffleBSpec, host_preshuffle_layout
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
class FlatMMSpec(WarpTileBlockSizeMixin):
    """One concrete FlatMM kernel configuration.

    Mirrors :class:`BatchedGemmSpec` (since the v1 kernel body is
    shared) with two FlatMM-specific extras:

    * ``name`` defaults to ``rocke_flatmm`` so the kernel symbol
      carries the FlatMM tag.
    * ``preshuffle_b`` (default False) is on the spec surface so
      callers can write the FlatMM-with-preshuffle intent today. v1
      rejects ``True`` at IR build time; the host-side preshuffle
      layout descriptor is reachable via
      :func:`flatmm_preshuffle_b_layout` so the permuted-B tensor
      can be built ahead of the v2 kernel landing.
    """

    tile: TileSpec
    trait: TraitSpec = field(default_factory=TraitSpec)
    wave_size: int = 64
    block_size: int = 0
    batch_size: int = 0
    preshuffle_b: bool = False
    name: str = "rocke_flatmm"

    def __post_init__(self) -> None:
        self._init_block_size()

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


# ---------------------------------------------------------------------
# Spec convenience constructors mirroring CK Tile's flatmm_basic.hpp
# ---------------------------------------------------------------------


def flatmm_config32(dtype: str = "f16") -> TileSpec:
    """``FlatmmConfig32`` mirror: 32x32 hero atom, K-packed when on f16.

    Drop-in match for the upstream
    ``example/ck_tile/18_flatmm/flatmm_basic.hpp::FlatmmConfig32``
    defaults: ``M_Tile=N_Tile=128``, ``M_Warp=1, N_Warp=4``, MFMA
    ``32x32x16`` for f16 / bf16 (the K-packed atom; ``K_Warp_Tile =
    sizeof(DataType) == 2 ? 16 : 32`` in the CK Tile config).

    Use as::

        spec = FlatMMSpec(tile=flatmm_config32("f16"),
                          trait=TraitSpec(epilogue="cshuffle"))
    """
    if dtype not in ("f16", "fp16", "bf16"):
        raise ValueError(
            f"flatmm_config32 ships f16 / bf16 today (got {dtype!r}); "
            "fp8 / bf8 wire through helpers.atoms.mfma_atom('fp8', ...)"
        )
    return TileSpec(
        tile_m=128,
        tile_n=128,
        tile_k=128 // 2,  # 128 bytes / sizeof(f16)
        warp_m=1,
        warp_n=4,
        warp_k=1,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
    )


def flatmm_config16(dtype: str = "f16") -> TileSpec:
    """``FlatmmConfig16`` mirror: 16x16 atom, K=32 K-packed for f16/bf16.

    Mirror of ``FlatmmConfig16`` in CK Tile's ``flatmm_basic.hpp`` for
    the 16x16 hero shape (``K_Warp_Tile = sizeof(DataType) == 2 ? 32 :
    64``). Useful when the 32x32 atom's 16-float accumulator pushes
    occupancy below break-even -- the 16x16 atom keeps the per-lane
    accumulator at 4 floats.
    """
    if dtype not in ("f16", "fp16", "bf16"):
        raise ValueError(f"flatmm_config16 ships f16 / bf16 today (got {dtype!r})")
    return TileSpec(
        tile_m=128,
        tile_n=128,
        tile_k=128 // 2,
        warp_m=1,
        warp_n=4,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )


def is_valid_spec(spec: FlatMMSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    if spec.preshuffle_b:
        return False, (
            "preshuffle_b=True is gated by the v2 preshuffled-B kernel "
            "body in gemm_universal; the host-side layout descriptor is "
            "available today via flatmm_preshuffle_b_layout(spec, N, K)"
        )
    from .batched_gemm import is_valid_spec as _bgemm_valid

    ok, why = _bgemm_valid(spec.to_batched_spec(), arch=arch)
    if not ok:
        return False, f"base batched_gemm spec invalid: {why}"
    return True, "ok"


def build_flatmm(spec: FlatMMSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one FlatMM instance.

    In v1 this is a re-export of :func:`build_batched_gemm` with the
    FlatMM kernel name. The runtime ABI is identical, so any launcher
    that already understands the batched_gemm signature can drive a
    FlatMM kernel unchanged.

    The underlying kernel body is tile-level: vector tile loads
    (``CoalescedTileLoader`` / ``AsyncTileLoader``), K-packed MFMA atom
    in the K-loop (``f16_32x32x16`` by default; see
    :func:`flatmm_atom_shape`), and the cshuffle epilogue's wide
    ``buffer_store_vN_f16``. There is no scalar per-element code path
    in this flow.

    ``arch`` selects the target GPU. The canonical FlatMM configs
    (:func:`flatmm_config32` / :func:`flatmm_config16`) use the wide
    K-packed atoms (``32x32x16`` / ``16x16x32``) that exist only on
    gfx950 (CDNA4); requesting ``arch="gfx942"`` for such a config is
    rejected with a structured Python error before comgr. For gfx942,
    use a narrow-atom :class:`TileSpec` (e.g. ``warp_tile_k=16`` for the
    32x32 atom / ``warp_tile_k=16`` for 16x16).
    """
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid flatmm spec for {arch}: {why}")
    return build_batched_gemm(spec.to_batched_spec(), arch=arch)


def flatmm_grid(spec: FlatMMSpec, batch: int, m: int, n: int) -> Tuple[int, int, int]:
    """Same launch grid as :func:`build_batched_gemm`."""
    return batched_gemm_grid(batch, m, n, spec.to_batched_spec())


def flatmm_signature(spec: FlatMMSpec):
    """Manifest-style signature mirroring :func:`batched_gemm_signature`."""
    return batched_gemm_signature(spec.to_batched_spec())


# ---------------------------------------------------------------------
# Tile-level introspection helpers
# ---------------------------------------------------------------------


def flatmm_atom_shape(spec: FlatMMSpec) -> Tuple[int, int, int]:
    """Return ``(m, n, k)`` of the per-warp MFMA atom this spec resolves to.

    Useful when sanity-checking a FlatMM config against the K-packed
    atom catalog: ``(32, 32, 16)`` for the canonical hero (CK Tile
    ``FlatmmConfig32`` on f16) and ``(16, 16, 32)`` for the 16x16
    K-packed atom (``FlatmmConfig16``).
    """
    t = spec.tile
    return (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)


def flatmm_atom(spec: FlatMMSpec) -> MfmaAtom:
    """Return the :class:`~rocke.helpers.atoms.MfmaAtom` this spec resolves to.

    Hands back the same :class:`MfmaAtom` ``build_universal_gemm`` will
    invoke for the K-loop, so a caller can introspect per-lane operand
    widths / accumulator footprint before launching.
    """
    return mfma_atom("f16", *flatmm_atom_shape(spec))


def flatmm_preshuffle_b_spec(spec: FlatMMSpec) -> PreshuffleBSpec:
    """Build the :class:`~rocke.helpers.preshuffle.PreshuffleBSpec` for
    this FlatMM config.

    Uses the per-block tile dims from ``spec.tile`` and ``elem_bytes=2``
    for the f16 / bf16 family. The descriptor feeds both the host-side
    permutation pass (via :func:`flatmm_preshuffle_b_layout`) and the
    kernel-side per-lane address calculator
    (:func:`rocke.helpers.emit_preshuffleb_offset`) once the v2
    preshuffled-B kernel body lands.
    """
    t = spec.tile
    return PreshuffleBSpec(block_n=t.tile_n, block_k=t.tile_k, elem_bytes=2)


def flatmm_preshuffle_b_layout(
    spec: FlatMMSpec, *, n: int, k: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Host-side preshuffled-B layout for the launcher to permute B with.

    Returns ``(shape, strides)`` describing the
    ``[k_tiles, n_tiles, block_n, block_k]`` packed layout the v2
    FlatMM kernel body expects. Available **today** so a host-side
    launcher can build the permuted B tensor (e.g. ``b_pre =
    b_natural.view(k_tiles, block_k, n_tiles, block_n).permute(0, 2, 3,
    1).contiguous()``) before the v2 body lands -- at which point the
    only kernel-side change is wiring ``preshuffle_b=True``.

    Wraps :func:`rocke.helpers.host_preshuffle_layout` with the
    PreshuffleBSpec derived from this FlatMM config.
    """
    return host_preshuffle_layout(flatmm_preshuffle_b_spec(spec), n=n, k=k)


__all__ = [
    "FlatMMSpec",
    "build_flatmm",
    "flatmm_atom",
    "flatmm_atom_shape",
    "flatmm_config16",
    "flatmm_config32",
    "flatmm_grid",
    "flatmm_preshuffle_b_layout",
    "flatmm_preshuffle_b_spec",
    "flatmm_signature",
    "is_valid_spec",
    # Re-exports for caller convenience.
    "DataSpec",
    "TileSpec",
    "TraitSpec",
    "UniversalGemmSpec",
]
