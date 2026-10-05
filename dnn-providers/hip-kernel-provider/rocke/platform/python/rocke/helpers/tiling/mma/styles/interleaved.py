# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``InterleavedStyle`` -- the free-contiguous layout style: wide coalesced global loads land free-dim
fastest, then a single in-register reorder makes them MMA-ready.

Chosen when both operands are FREE-DIM contiguous in global memory (A's M, B's N are the stride-1 axis):
a K-contiguous/canonical load would be VW=1, while this style loads the whole free run wide and the
global->LDS store is an IDENTITY reposition (the free dim is already innermost). It PRODUCES an
N-lane-major accumulator against an N-stride-1 C (the coalesced C store is produced, never recovered).

Composes the style-agnostic primitives -- ``TileDesc.reorder_registers`` for the register significance,
``cooperative_load_desc`` (in ``rocke.helpers.tiling.memory``) for the global load, ``TileDesc.swap_dims``
for the (free,K)<->(K,free) reposition -- so the interleaved recipe is a composition, not a private copy.
The accumulator is DERIVED (:meth:`accumulator_desc`): interleaved's MMA-ready operand is a register
REORDER presenting the SAME K-distribution to the atom, so the atom produces the same native C.
"""

from __future__ import annotations

from ...encoding import WarpDistributionEncoding
from ...fragments import TileDesc
from ...layouts import make_tile_desc
from ...traits import MmaTraits
from .base import AtomNumbers, LayoutStyle


class InterleavedStyle(LayoutStyle):
    name = "interleaved"

    def operand_descs(
        self, traits: MmaTraits, *, free_sub: int, k_sub: int, free_lanes: int
    ) -> tuple[TileDesc, TileDesc]:
        """``(lds_read_desc_in_(free,K), mma_ready_desc)`` for one operand. The generic ``free_lanes``-keyed
        form (public: introspected by the interleavability proof sweep); :meth:`lds_bridge` is the
        role-keyed wrapper the protocol/kernel use.

        ``free_lanes`` is PER OPERAND (``traits.m`` for A -- free axis M; ``traits.n`` for B -- free axis
        N); making it a caller argument is what lets A and B differ from one profile without B reusing
        A's span. ``k_lanes``/``k_per_lane`` are shared (square contract only).

        Base labels (identical in both; only register significance differs):
            free = free_sub*u + d        u = lane % free_lanes,  d = free-atom index
            K    = atom.k*ka + kpl*g + j g = lane // free_lanes, ka = 0..k_sub-1, j = 0..kpl-1
        ``make_tile_desc``'s canonical register order is [ka, d, j]. The two consumers want
        (ka, j, d) -- free dim FASTEST, so the LDS read vectorizes along the free dim -- and
        (d, ka, j) -- free-atom major, K-atom, K-within -- the MMA-ready (SOA) order the driver slices.
        The delta is the mandatory in-register reorder, the price of the wide LDS read.
        """
        atom = AtomNumbers.from_traits(traits)
        lanes_per_axis = (free_lanes, atom.k_lanes)  # axis 0 = free, axis 1 = K
        base = make_tile_desc(
            shape=[free_sub * free_lanes, k_sub * atom.k],
            thread_tile=[free_sub, atom.k_per_lane],
            thread_dist=[free_lanes, atom.k_lanes],
            # free dim fastest -> lane = g*free_lanes + u (K-group major). Derived, not a literal, so a
            # collapsed (thread_dist == 1) lane bucket drops out instead of naming a missing axis.
            thread_order=[axis for axis in (1, 0) if lanes_per_axis[axis] > 1],
            block_repeat=[1, k_sub],
            # REPLICATION: when free_lanes*k_lanes < wave (gfx11 WMMA duplicates the operand across the
            # two lane halves) the leftover lanes broadcast identical data. == 1 (no-op) on CDNA/gfx12.
            thread_broadcast=atom.wave_size // (free_lanes * atom.k_lanes),
            wave_size=atom.wave_size,
        )
        # make_tile_desc DROPS any register bucket whose extent is 1, so name the buckets and derive each
        # consumer's permutation by name: a collapsed level drops out of both lists instead of shifting
        # indices under a hardcoded tuple. Bookkeeping, not a change of meaning.
        keep = lambda name, extent: [name] if extent > 1 else []
        names = keep("ka", k_sub) + keep("d", free_sub) + keep("j", atom.k_per_lane)
        perm = lambda want: tuple(names.index(n) for n in want if n in names)
        return (
            base.reorder_registers(perm(("ka", "j", "d"))),
            base.reorder_registers(perm(("d", "ka", "j"))),
        )

    def lds_bridge(
        self, traits: MmaTraits, *, role: str, free_sub: int, k_sub: int
    ) -> tuple[TileDesc, TileDesc]:
        """The ``(lds_read_landing_desc, mma_ready_desc)`` bridge pair -- interleaved STAGES the operand
        through LDS. ``role`` selects the operand's free axis (A -> M, B -> N)."""
        if role == "A":
            free_lanes = traits.m
        elif role == "B":
            free_lanes = traits.n
        else:
            raise ValueError(f"operand role must be 'A' or 'B' -- got {role!r}")
        return self.operand_descs(traits, free_sub=free_sub, k_sub=k_sub, free_lanes=free_lanes)

    def operand_desc(
        self, traits: MmaTraits, *, role: str, free_sub: int, k_sub: int
    ) -> TileDesc:
        """The MMA-ready descriptor -- the second half of the LDS bridge (:meth:`lds_bridge`)."""
        return self.lds_bridge(traits, role=role, free_sub=free_sub, k_sub=k_sub)[1]

    def accumulator_desc(self, traits: MmaTraits, *, m_sub: int, n_sub: int) -> TileDesc:
        """The DERIVED accumulator -- what the interleaved A/B produce through the fixed MFMA coupling.
        Hand-built: ``make_tile_desc`` cannot express the M axis's lane-below-register level. With
        ``R=c_lane_rows, V=c_inner, P=c_patches, W=n``:

            M = mna*(R*V*m_sub) + mo*(V*m_sub) + m_in*m_sub + mi   mo = lane // W ; mna/m_in/mi regs
            N = n_in*n_sub + nj                                    n_in = lane % W ; nj register
            reg significance (major -> minor) = mi, nj, mna, m_in

        The lane owns ``P`` disjoint (V*m_sub) x n_sub rectangles (P==1 at 16x16 degenerates to one
        rectangle; P==4 at 32x32 is a legal multi-patch distribution). N is lane-major (W consecutive
        lanes span a contiguous n run), which against an N-stride-1 C is 'with the grain'. The P level is
        OMITTED when 1 so the 16x16 encoding is byte-identical to the single-patch case.
        """
        atom = AtomNumbers.from_traits(traits)
        wave_m, wave_n = m_sub * atom.m, n_sub * atom.n
        patches, rows, inner = atom.c_patches, atom.c_lane_rows, atom.c_inner
        m_levels = ([patches] if patches > 1 else []) + [rows, inner, m_sub]
        first = 1 if patches > 1 else 0
        l_patch, l_rows, l_inner, l_mi = first - 1, first, first + 1, first + 2
        patch_reg = ((1,), (l_patch,)) if patches > 1 else ((), ())
        return TileDesc(
            shape=(wave_m, wave_n),
            layout=WarpDistributionEncoding(
                replication_lengths=(),
                hierarchical_lengths=(tuple(m_levels), (atom.n, n_sub)),
                lane_to_rh_major=((1, 2),),   # lane = mo (an M level) * atom.n + n_in (N level 0)
                lane_to_rh_minor=((l_rows, 0),),
                # significance: mi (M atom index), nj (N atom index), mna (M patch), m_in (M within)
                register_to_rh_major=(1, 2) + patch_reg[0] + (1,),
                register_to_rh_minor=(l_mi, 1) + patch_reg[1] + (l_inner,),
            ),
        )
