# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Small-tile warp grids and the occupancy bound of the attention backward.

The main kernel's G0 / G2 (``S``, ``dP``: ``kM0 x kN0``) and G1 / G3 (``dV``,
``dK``: ``kN0 x D``) run on warp grids (rows, cols) of 16x16 atom tiles. The
relabel route fixes them to ``(1, W)`` and ``(W, 1)``, so every wave owns at
least 16 keys and ``kN0`` is a multiple of ``16 * W``. With P^T and dS^T read
back from LDS (``pt_route = "lds"``) the grids are free, which is what the
small tiles need (``kN0`` or ``kM0`` of 16 or 32 on two or four waves):

* G0 / G2: ``(1, k)``, ``(k, 1)`` or ``(2, k / 2)`` for a wave count ``k``
  dividing ``W``. With ``k < W`` the grid is *replicated*: wave ``w`` computes
  the sub-block of wave ``w mod k`` (every wave holds S / dP of its rows and
  keys; the softmax gradient needs no cross-wave exchange), and only the first
  ``k`` waves store P and dS to LDS. Replication is the only way to give a
  16 x 16 S tile to more than one wave without a cross-wave reduction. The
  copies are discarded: a wave-uniform branch that lets the waves ``w >= k``
  skip G0 / G2 and the softmax gradient is not built (the step keeps every
  matrix op outside branches).
* G1 / G3: ``(W, 1)``, ``(1, W)`` or ``(2, W / 2)``: the dK / dV accumulators
  are never replicated (the epilogue would count them twice).

Every grid must split its tile into whole 16x16 atoms (the block GEMM rule).
The checks below keep the messages of the shipped ``(1, W)`` / ``(2, W / 2)``
rules, so a spec the old rules refused is refused with the same text.

:func:`attn_bwd_occupancy` gives the waves per SIMD a resolved spec can reach
by its static register estimate and by its LDS plan, counted in whole
workgroups like the compiled-object gauge of the resource tests. A
``waves_per_eu`` hint above the LDS bound cannot be met (the backend drops an
unsatisfiable hint), so a sweep skips it; the validator does not refuse it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AttnBwdOccupancy",
    "attn_bwd_occupancy",
    "check_lds_route_grid",
    "g02_grids",
    "g13_grids",
    "grid_waves",
    "kv_operand_split",
    "natural_grids",
    "replicated_g02",
    "small_tile_grids",
]

# CDNA: four SIMDs per CU, a 512-entry unified (arch VGPR + AGPR) file per lane
# with an 8-register allocation granule, at most 8 waves per SIMD.
_SIMDS_PER_CU = 4
_UNIFIED_REGS = 512
_REG_GRANULE = 8
_MAX_WAVES_PER_SIMD = 8


def grid_waves(grid) -> int:
    """Waves of a warp grid ``(rows, cols)``."""
    return int(grid[0]) * int(grid[1])


def _shapes(k: int) -> tuple[tuple[int, int], ...]:
    """The block-GEMM grid shapes of ``k`` waves: ``(1, k)``, ``(k, 1)``,
    ``(2, k / 2)`` (deduplicated, in that order)."""
    out = [(1, k), (k, 1)]
    if k >= 2:
        out.append((2, k // 2))
    return tuple(dict.fromkeys(out))


def natural_grids(waves: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """The relabel route's ``(G0 / G2, G1 / G3)`` grids: ``(1, W)``, ``(W, 1)``."""
    return (1, waves), (waves, 1)


def g02_grids(waves: int) -> tuple[tuple[int, int], ...]:
    """Every G0 / G2 grid of a ``W``-wave CTA, the natural ``(1, W)`` first;
    the others need ``pt_route = "lds"``. Wave counts ``k < W`` (dividing
    ``W``) are replicated grids."""
    ks = [k for k in (1, 2, 4, 8) if k <= waves and waves % k == 0]
    out = [(1, waves)]
    for k in sorted(ks, reverse=True):
        out += _shapes(k)
    return tuple(dict.fromkeys(out))


def g13_grids(waves: int) -> tuple[tuple[int, int], ...]:
    """Every G1 / G3 grid of a ``W``-wave CTA, the natural ``(W, 1)`` first;
    the others need ``pt_route = "lds"``. Never replicated."""
    return tuple(dict.fromkeys(((waves, 1),) + _shapes(waves)))


def replicated_g02(grid, waves: int) -> bool:
    """Whether a G0 / G2 grid holds fewer waves than the CTA."""
    return grid_waves(grid) < waves


def check_lds_route_grid(
    gname: str, grid, natural, waves: int, rows: int, cols: int, pt_route: str
) -> None:
    """Legality of a non-natural G0 / G2 (``gname = "warp_grid_g02"``) or G1 /
    G3 (``"warp_grid_g13"``) grid over its ``rows x cols`` tile.

    Raises ``ValueError``: the grid is not one of the CTA's grids (message
    "must be ..."), it needs the LDS route ("needs pt_route='lds'"), or it does
    not split the tile into whole 16x16 atoms ("does not divide its ...").
    """
    g = tuple(grid)
    alt = (2, waves // 2)
    legal = g02_grids(waves) if gname == "warp_grid_g02" else g13_grids(waves)
    if waves < 2 or g not in legal:
        others = tuple(x for x in legal if x not in (natural, alt))
        more = f"; also {others}" if waves >= 2 and others else ""
        raise ValueError(
            f"{gname}={g} must be {natural} (or {alt} with pt_route='lds'{more})"
        )
    if pt_route != "lds":
        raise ValueError(
            f"{gname}={g} needs pt_route='lds' (the relabel route fixes it)"
        )
    if rows % (16 * g[0]) != 0 or cols % (16 * g[1]) != 0:
        raise ValueError(f"{gname}={g} does not divide its {rows} x {cols} tile")


def kv_operand_split(grid_g02, waves: int) -> int:
    """Waves the per-wave K / V operand term of the register estimate divides
    ``kN0`` by: the grid's columns (each wave holds the keys of its column).

    The natural ``(1, W)`` grid gives ``W`` (exact). For the shipped ``(2, W /
    2)`` grid the estimate has always counted ``W`` (each wave in fact holds
    ``kN0 / (W / 2)`` keys); that count is kept so the policy resolution and
    the legality of the existing instances do not move. Every other grid
    counts its columns.
    """
    g = tuple(grid_g02)
    if g in ((1, waves), (2, waves // 2)):
        return waves
    return g[1]


def small_tile_grids(
    head_size: int, block_m: int, block_n: int, waves: int
) -> tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], ...]:
    """``(G0 / G2, G1 / G3, G4)`` grid triples that split a ``kM0 x kN0``
    tile of head size ``D`` over ``W`` waves into whole 16x16 atoms (natural
    triple first when it splits). A triple other than the natural one needs
    ``pt_route = "lds"``. Empty when no grid splits the tile.
    """
    km, kn, d, w = block_m, block_n, head_size, waves
    out = []
    for g02 in g02_grids(w):
        if km % (16 * g02[0]) or kn % (16 * g02[1]):
            continue
        for g13 in g13_grids(w):
            if kn % (16 * g13[0]) or d % (16 * g13[1]):
                continue
            for g4 in ((1, w), (2, w // 2)) if w >= 2 else ((1, 1),):
                if grid_waves(g4) != w or km % (16 * g4[0]) or d % (16 * g4[1]):
                    continue
                out.append((g02, g13, g4))
    return tuple(dict.fromkeys(out))


@dataclass(frozen=True)
class AttnBwdOccupancy:
    """Waves per SIMD of a resolved spec, counted in whole workgroups.

    ``by_registers``: the static arch-VGPR + AGPR estimate (rounded up to the
    allocation granule) against the unified file; ``by_lds``: the LDS plan
    against the CU's capacity; ``waves``: the smaller. ``requested`` is the
    spec's ``waves_per_eu`` (None: no hint). ``hint_reachable`` is False when
    the LDS plan alone keeps the CTA below the hint.
    """

    by_registers: int
    by_lds: int
    requested: int | None

    @property
    def waves(self) -> int:
        return min(self.by_registers, self.by_lds)

    @property
    def hint_reachable(self) -> bool:
        return self.requested is None or self.by_lds >= self.requested


def _waves_per_simd(wgs: int, waves: int) -> int:
    """Waves per SIMD of ``wgs`` resident workgroups of ``waves`` waves."""
    if wgs <= 0:
        return 0
    return max(1, (wgs * waves) // _SIMDS_PER_CU)


def _wgs_by_slots(per_simd: int, waves: int) -> int:
    """Workgroups that fit ``per_simd`` wave slots on every SIMD."""
    if waves > _SIMDS_PER_CU:
        return per_simd // -(-waves // _SIMDS_PER_CU)
    return (per_simd * _SIMDS_PER_CU) // waves


def attn_bwd_occupancy(spec, arch: str) -> AttnBwdOccupancy:
    """Occupancy bound of a resolved spec on an MFMA target (raises
    ``ValueError`` on a WMMA target, whose register file and SIMD count
    differ). The register bound uses the static estimate
    (:func:`kernels.common.attention_bwd.register_estimates`); the compiled
    resource notes decide the real count."""
    from kernels.common.attention_bwd import (
        attn_bwd_arch_facts,
        lds_footprint_bytes,
        register_estimates,
    )

    facts = attn_bwd_arch_facts(arch)
    if facts.matrix_path != "mfma":
        raise ValueError(f"attn_bwd_occupancy models MFMA targets only, not {arch}")
    w = spec.waves
    est = register_estimates(spec, arch)
    used = max(est["arch_vgpr"] + est["agpr"], 1)
    used = -(-used // _REG_GRANULE) * _REG_GRANULE
    per_simd = min(_MAX_WAVES_PER_SIMD, _UNIFIED_REGS // used)
    by_regs = _waves_per_simd(_wgs_by_slots(per_simd, w), w)
    lds = lds_footprint_bytes(spec, arch)
    wgs_lds = facts.lds_capacity_bytes // lds if lds > 0 else _MAX_WAVES_PER_SIMD * 4
    wgs_lds = min(wgs_lds, _wgs_by_slots(_MAX_WAVES_PER_SIMD, w))
    by_lds = _waves_per_simd(wgs_lds, w)
    return AttnBwdOccupancy(by_regs, by_lds, spec.waves_per_eu)
