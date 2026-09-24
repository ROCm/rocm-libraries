# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``LayoutStyle`` -- the operand-layout STRATEGY seam (rev-7 Principle 8).

A layout style is a *profile* (canonical, interleaved, ...) that produces, PER OPERAND, the MMA-ready
register layout (and, for a style that stages through LDS, the memory-bridge descriptors). Canonical and
interleaved are two label-flow choices over ONE machine (``docs/mma_is_machinery.md``), not two machines.

The seam owns the correctness envelope; the style owns only the recipe:

- A style **composes** the style-agnostic toolbox primitives (``cooperative_load_desc``,
  ``TileDesc.swap_dims``/``.reorder_registers``) -- it never owns private copies.
- The accumulator (C) is **always derived** from the atom; a style MUST NOT supply a C descriptor. Its
  only influence on C is the K-distribution its operand layout presents to the atom. ``accumulator_desc``
  is the derived native C for the style's operands (checked against the ``derive_c_distribution`` oracle,
  by an independent path, in :class:`~rocke.helpers.tiling.mma.plan.TileMmaPlan`).
- The MMA-ready descriptor a style produces is validated identically to a derived one -- ``operand_desc``
  feeds ``TileMmaPlan.a_operand_desc``/``b_operand_desc``, which the driver polices with
  ``operand_soundness`` + the SOA slice guard. The atom-canonical ``a_layout``/``b_layout`` remain the
  immutable soundness reference; a style never overrides them.

Adding a style: subclass :class:`LayoutStyle`, implement ``operand_desc`` + ``accumulator_desc`` from the
public primitives, and (if it stages through LDS) expose bridge descriptors the kernel composes with
``cooperative_load_desc`` + the ``memory`` LDS verbs. See ``docs/tiling_api_contract.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...fragments import TileDesc
from ...traits import MmaTraits


@dataclass(frozen=True)
class AtomNumbers:
    """The handful of atom quantities a layout style needs, DERIVED from the traits SSOT (never
    hand-typed), so a style re-derives for a different atom instead of being rewritten for it."""

    m: int
    n: int
    k: int
    k_per_lane: int   # ABK -- K a lane holds per atom
    k_lanes: int      # lanes spanning K within one atom == k / k_per_lane
    c_patches: int    # CMN -- disjoint M sub-tiles a lane owns in the accumulator
    c_lane_rows: int  # M / CM -- lanes the accumulator spends on M
    c_inner: int      # CM / CMN -- contiguous accumulator M rows per patch per atom
    wave_size: int

    @classmethod
    def from_traits(cls, traits: MmaTraits) -> "AtomNumbers":
        return cls(
            m=traits.m,
            n=traits.n,
            k=traits.k,
            k_per_lane=traits.k_ab_per_lane,
            k_lanes=traits.k // traits.k_ab_per_lane,
            c_patches=traits.c_m_num_access,
            c_lane_rows=traits.m // traits.c_m_per_lane,
            c_inner=traits.c_m_per_lane // traits.c_m_num_access,
            wave_size=traits.wave_size,
        )


class LayoutStyle:
    """Base layout style. Produces, per operand, the MMA-ready operand descriptor + the derived C
    accumulator descriptor. Subclasses: :class:`CanonicalStyle`, :class:`InterleavedStyle`. A style is
    a single *profile* (one ``style=`` kwarg on ``TileMma``, shared A/B); it is resolved per operand
    from that operand's free extent (``free_lanes``), so A and B get asymmetric layouts from one profile.
    """

    name: str = "layout-style"

    def operand_desc(
        self, traits: MmaTraits, *, role: str, free_sub: int, k_sub: int
    ) -> TileDesc:
        """The MMA-ready (SOA) operand descriptor the driver consumes, for ``role`` in {"A", "B"} over
        a wave sub-grid of ``free_sub`` free-atoms x ``k_sub`` K-atoms."""
        raise NotImplementedError

    def accumulator_desc(self, traits: MmaTraits, *, m_sub: int, n_sub: int) -> TileDesc:
        """The DERIVED native C accumulator descriptor for a ``m_sub`` x ``n_sub`` wave grid. Always
        atom-derived (never style-supplied data)."""
        raise NotImplementedError

    def lds_bridge(
        self, traits: MmaTraits, *, role: str, free_sub: int, k_sub: int
    ) -> tuple[TileDesc, TileDesc] | None:
        """OPTIONAL declared extension point for a style that STAGES the operand THROUGH LDS. Returns the
        memory-bridge pair ``(lds_read_landing_desc, mma_ready_desc)`` for ``role`` in {"A", "B"}: the
        kernel loads the wide LDS read with the first descriptor, then applies the in-register reorder to
        reach the second (which equals :meth:`operand_desc`). Returns ``None`` for a NON-staging style
        (e.g. canonical loads MMA-ready directly, no LDS landing). A new LDS-staging style overrides this
        so the bridge is on the protocol, not exposed ad-hoc off a concrete subclass."""
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
