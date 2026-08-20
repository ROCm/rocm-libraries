# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MMA warp-encoding calculators -- port of CK TileDistrEncCalc (dense no-block branch, M1).

INTERNAL machinery -- NOT part of the public author surface. Produces a
:class:`~rocke.helpers.tiling.encoding.WarpDistributionEncoding` (the coordinate-transform graph) from
typed :class:`~rocke.helpers.tiling.traits.MmaTraits`. The public layer hides this behind
:class:`~rocke.helpers.tiling.mma.TileMma`; only that class / reflection consume it directly. The
encoding TYPE itself lives at the package root (:mod:`rocke.helpers.tiling.encoding`) -- this module only
*produces* MMA-flavoured instances of it.

**rocke-native adaptation vs CK:** CK's no-block C branch emits a trivial replication
``Rs = sequence<1>`` that no P/Y references. rocke's bijection validator *requires* every
R bucket to be referenced, so we drop the trivial R (``replication_lengths = ()``), exactly
as rocke's ``make_c_warp_dstr_encoding`` does. The A/B StridedK encodings keep their R
because the lane references it.
"""

from __future__ import annotations

from ..encoding import WarpDistributionEncoding
from ..traits import MmaTraits

__all__ = ["a_warp_encoding", "b_warp_encoding", "c_warp_encoding"]

def _require_divisible(numerator: int, denominator: int, what: str, op_id: str) -> int:
    if denominator == 0 or numerator % denominator != 0:
        raise ValueError(
            f"{what} not divisible -- op_id={op_id!r}, {numerator} % {denominator} != 0"
        )
    return numerator // denominator

def c_warp_encoding(
    traits: MmaTraits, m_iter: int = 1, n_iter: int = 1
) -> WarpDistributionEncoding:
    """C/D accumulator encoding (dense, no-block, SFactor=1, no transpose).

    ``m_iter`` / ``n_iter`` = number of C atoms stacked along M / N (the wave tile's M/N /
    atom M/N). Each is prepended as an outer atom-index level (stride = one atom's M/N) and
    made the most-significant register, so the fragment is SUBTILE-CONTIGUOUS: registers for
    C subtile ``(mi, nj)`` are ``[(mi*n_iter + nj)*atom_len : +atom_len]``, with the
    within-atom layout identical to the single-atom encoding.

    For ``m_iter == n_iter == 1`` this is byte-identical to rocke ``make_c_warp_dstr_encoding``:
    gfx90a ``mfma_f32_16x16x16f16`` -> Hs=((1,4,4),(16,)), lane=((1,2),)/((1,0),),
    register=(1,1)/(0,2), no replication.
    """
    if m_iter <= 0 or n_iter <= 0:
        raise ValueError(
            f"m_iter/n_iter must be positive -- op_id={traits.op_id!r}, "
            f"m_iter={m_iter}, n_iter={n_iter}"
        )
    m_per_lane = traits.c_m_per_lane
    m_num_access = traits.c_m_num_access
    m_outer = _require_divisible(traits.m, m_per_lane, "M / c_m_per_lane", traits.op_id)
    m_inner = _require_divisible(
        m_per_lane, m_num_access, "c_m_per_lane / c_m_num_access", traits.op_id
    )

    m_levels: list[int] = []
    reg_major: list[int] = []
    reg_minor: list[int] = []
    if m_iter > 1:
        reg_major.append(1)
        reg_minor.append(len(m_levels))
        m_levels.append(m_iter)
    mna_level = len(m_levels)
    m_levels.append(m_num_access)
    mouter_level = len(m_levels)
    m_levels.append(m_outer)
    minner_level = len(m_levels)
    m_levels.append(m_inner)

    n_levels: list[int] = []
    n_reg_major: list[int] = []
    n_reg_minor: list[int] = []
    if n_iter > 1:
        n_reg_major.append(2)
        n_reg_minor.append(len(n_levels))
        n_levels.append(n_iter)
    n_level = len(n_levels)
    n_levels.append(traits.n)

    # registers (most significant first): m_iter, n_iter, then within-atom C (m_num_access, m_inner).
    reg_major += n_reg_major + [1, 1]
    reg_minor += n_reg_minor + [mna_level, minner_level]
    return WarpDistributionEncoding(
        replication_lengths=(),  # rocke-native: drop the trivial unreferenced R
        hierarchical_lengths=(tuple(m_levels), tuple(n_levels)),
        lane_to_rh_major=((1, 2),),  # lane: X-dim M (1) then N (2)
        lane_to_rh_minor=((mouter_level, n_level),),
        register_to_rh_major=tuple(reg_major),
        register_to_rh_minor=tuple(reg_minor),
    )

def _ab_strided_k_encoding(
    *,
    major_dim_size: int,
    repeat: int,
    num_access: int,
    k_per_lane: int,
    k_dim: int,
    op_id: str,
    major_iter: int = 1,
    k_iter: int = 1,
    interleaved: bool = False,
) -> WarpDistributionEncoding:
    """A/B operand StridedK encoding (CK ABWarpDstrEncStridedK), dense.

    ``major_iter`` = number of atoms stacked along the major dim (M for A, N for B = the wave
    tile's M/N / atom M/N); ``k_iter`` = number of atoms stacked along K. Each iteration count
    is prepended as an outer atom-index level (stride = one atom's extent on that axis) and
    made a most-significant register, so the fragment is SUBTILE-CONTIGUOUS: registers for
    atom ``(mi, ki)`` are ``[(mi*k_iter + ki)*atom_len : +atom_len]`` with the within-atom
    layout identical to the single-atom encoding -- the MMA driver slices per atom.

    For ``major_iter == k_iter == 1`` this is byte-identical to the single-atom encoding:
      Hs = ( (major_dim_size,), (num_access, K/k_per_lane, k_per_lane/num_access) );
      lane merges {K1, R, major_dim}; registers = {num_access, k_per_lane/num_access}.
    """
    k_outer = _require_divisible(k_dim, k_per_lane, "K / k_ab_per_lane", op_id)
    k_inner = _require_divisible(k_per_lane, num_access, "k_ab_per_lane / num_access", op_id)
    if major_iter <= 0 or k_iter <= 0:
        raise ValueError(
            f"major_iter/k_iter must be positive -- op_id={op_id!r}, "
            f"major_iter={major_iter}, k_iter={k_iter}"
        )

    # Major axis (X-dim 1): optional atom-index level, then the within-atom major dim (lane).
    major_levels: list[int] = []
    major_reg_major: list[int] = []
    major_reg_minor: list[int] = []
    if major_iter > 1:
        major_reg_major.append(1)
        major_reg_minor.append(len(major_levels))
        major_levels.append(major_iter)
    atom_major_level = len(major_levels)
    major_levels.append(major_dim_size)

    # K axis (X-dim 2): optional atom-index level, then num_access / k_outer (lane) / k_inner.
    k_levels: list[int] = []
    k_reg_major: list[int] = []
    k_reg_minor: list[int] = []
    if k_iter > 1:
        k_reg_major.append(2)
        k_reg_minor.append(len(k_levels))
        k_levels.append(k_iter)
    na_level = len(k_levels)
    k_levels.append(num_access)
    kouter_level = len(k_levels)
    k_levels.append(k_outer)
    kinner_level = len(k_levels)
    k_levels.append(k_inner)

    # Register order. Default (SOA, subtile-contiguous): atom-iteration axes are the MAJOR
    # (outer/slowest) registers, so each atom's registers are contiguous. interleaved (AOS):
    # atom-iteration is the MINOR (inner/fastest) register -- an element's atom copies are adjacent,
    # which matches a coalesced global load. The two differ only by a register reorder (same lanes,
    # same elements), so `transform_fragment` moves between them with a compile-time shuffle.
    iter_major = major_reg_major + k_reg_major
    iter_minor = major_reg_minor + k_reg_minor
    within_major = [2, 2]
    within_minor = [na_level, kinner_level]
    if interleaved:
        reg_major = within_major + iter_major
        reg_minor = within_minor + iter_minor
    else:
        reg_major = iter_major + within_major
        reg_minor = iter_minor + within_minor
    return WarpDistributionEncoding(
        replication_lengths=(repeat,),
        hierarchical_lengths=(tuple(major_levels), tuple(k_levels)),
        lane_to_rh_major=((2, 0, 1),),  # lane merges K1 (K axis), R, within-atom major
        lane_to_rh_minor=((kouter_level, 0, atom_major_level),),
        register_to_rh_major=tuple(reg_major),
        register_to_rh_minor=tuple(reg_minor),
    )

def _raise_interleaved_broken(where: str) -> None:
    raise RuntimeError(
        f"{where}(interleaved=True) is BROKEN and must not be used: it only reorders registers while "
        "keeping CANONICAL lane ownership, so it is NOT a real interleaved layout (no rectangular "
        "per-lane patch, no coalescing gain). Build interleaved layouts as custom static tile "
        "distributions with make_tile_desc (see kernels/tiling_gemm_interleaved_demo.py::"
        "_wave_descs_interleaved)."
    )


def a_warp_encoding(
    traits: MmaTraits, m_iter: int = 1, k_iter: int = 1, interleaved: bool = False
) -> WarpDistributionEncoding:
    """A operand encoding (dense StridedK). Major dim = M; repeat = a_repeat.

    ``interleaved=True`` is BROKEN and raises -- real interleaved layouts are custom static tile
    distributions (``make_tile_desc``), not this register reorder."""
    if interleaved:
        _raise_interleaved_broken("a_warp_encoding")
    return _ab_strided_k_encoding(
        major_dim_size=traits.m,
        repeat=traits.a_repeat,
        num_access=traits.a_k_num_access,
        k_per_lane=traits.k_ab_per_lane,
        k_dim=traits.k,
        op_id=traits.op_id,
        major_iter=m_iter,
        k_iter=k_iter,
        interleaved=interleaved,
    )

def b_warp_encoding(
    traits: MmaTraits, n_iter: int = 1, k_iter: int = 1, interleaved: bool = False
) -> WarpDistributionEncoding:
    """B operand encoding (dense StridedK). Major dim = N; repeat = b_repeat.

    ``interleaved=True`` is BROKEN and raises -- real interleaved layouts are custom static tile
    distributions (``make_tile_desc``), not this register reorder."""
    if interleaved:
        _raise_interleaved_broken("b_warp_encoding")
    return _ab_strided_k_encoding(
        major_dim_size=traits.n,
        repeat=traits.b_repeat,
        num_access=traits.b_k_num_access,
        k_per_lane=traits.k_ab_per_lane,
        k_dim=traits.k,
        op_id=traits.op_id,
        major_iter=n_iter,
        k_iter=k_iter,
        interleaved=interleaved,
    )
