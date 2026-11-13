# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 post-processors.

gfx1250 needs none of the gfx950 post-processing steps:
  * MIArchVgpr is auto-forced True under WMMA — no augmentation needed.
  * CMS (UseCustomMainLoopSchedule) is gfx950-only — no CMS group merge.
The base class's optional MT_DU filtering still runs via ``apply``.
"""

from typing import Dict, List, Sequence, Tuple

from geko.config_generator.fork_params.hw_profiles.gfx1250.cluster_dim import (
    select_cluster_dims,
    tile_grid,
)
from geko.config_generator.fork_params.hw_profiles.gfx1250.optimization_param import (
    GFX1250GAOrigamiPolicy,
)
from geko.config_generator.fork_params.post_processor import (
    BasePostProcessor,
    mark_post_process,
    mi_macro_tile,
)
from geko.config_generator.shared_utils import ForkParameter, GroupDimension, SizeContext

MIN_MI_WAVE_TILE = 4

_MI_WAVE_TILE_IDX = (5, 6)


def _wave_tile_ok(entry: Dict[str, ForkParameter]) -> bool:
    """True when both MIWaveTile components are >= :data:`MIN_MI_WAVE_TILE`.

    Entries that are too short to carry a wave tile are kept: this filter should
    narrow a known geometry, never silently discard something it cannot parse.
    """
    mi = entry["MatrixInstruction"].values
    if len(mi) <= max(_MI_WAVE_TILE_IDX):
        return True
    return all(mi[i] >= MIN_MI_WAVE_TILE for i in _MI_WAVE_TILE_IDX)


class _GFX1250WaveTileFilter:
    """MIWaveTile filter for the gfx1250 generic post-processor."""

    @mark_post_process
    def filter_small_wave_tiles(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Drop MI groups whose MIWaveTile has a component < MIN_MI_WAVE_TILE.

        Never empties the group list. A shape whose every candidate is filtered out
        would otherwise produce a config that generates no kernels at all, which is
        strictly worse than tuning a narrow geometry -- so in that case the original
        list is kept and the shape is left as it was.
        """
        kept = [e for e in mi_groups if _wave_tile_ok(e)]
        if not kept or len(kept) == len(mi_groups):
            return fork_params, mi_groups
        return fork_params, kept


def _mi_wave_tile(mi: Sequence[int]) -> Tuple[int, int]:
    """``MIWaveTile`` as Tensile derives it: ``[mi[5], mi[6]]``.

    See ``matrixInstructionToMIParameters`` in
    ``tensilelite/SolutionStructs/Validators/MatrixInstruction.py``.
    """
    return mi[5], mi[6]


def _mi_wave_group(mi: Sequence[int]) -> Tuple[int, int]:
    """``MIWaveGroup`` as Tensile derives it -- NOT simply ``[mi[7], mi[8]]``.

    Mirrors ``matrixInstructionToMIParameters``::

        wg0   = mi[4] * mi[0] * mi[7]
        waves = mi[7] * mi[8]
        miwg0 = min((wg0 // mi[0]) // MIBlockBM, waves)
        MIWaveGroup = [miwg0, waves // miwg0]

    with ``MIBlockBM = mi[4]`` for the geometries this profile emits.
    """
    waves = mi[7] * mi[8]
    wg0 = mi[4] * mi[0] * mi[7]
    block_bm = mi[4] if mi[4] else 1
    miwg0 = min((wg0 // mi[0]) // block_bm, waves)
    miwg0 = max(miwg0, 1)
    return miwg0, waves // miwg0


def _mis(mi_groups: GroupDimension) -> List[Sequence[int]]:
    """The 9-element MatrixInstruction of every group that carries one."""
    out = []
    for entry in mi_groups:
        mi = entry["MatrixInstruction"].values
        if len(mi) == 9:
            out.append(mi)
    return out


class _GFX1250GeometryNarrowing:
    """Drop fork values that the emitted MI geometry makes structurally invalid.

    Each rule below cites the rejection it prevents (Tensile
    ``SolutionStructs/Solution.py``).
    """

    @mark_post_process
    def narrow_to_mi_geometry(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        mis = _mis(mi_groups)
        if not mis:
            return fork_params, mi_groups

        def restrict(name, usable, fallback):
            fp = fork_params.get(name)
            if fp is None or not isinstance(fp.values, list):
                return
            kept = [v for v in fp.values if usable(v)] or list(fallback)
            if kept != fp.values:
                fp.values = kept

        # Rule B -- "HalfPLR does not support odd WaveTile" (Solution.py):
        # HalfPLR bit0 gates side A, bit1 side B, and a bit may only be set when
        # that side's MIWaveTile is even.
        #
        # The A/B orientation is FIXED, not swept: Solution.py picks it from
        # ``ProblemType["Tensor0"]``, and MIWaveTile itself is never swapped by
        # SourceSwap. Verified on the generated configs -- BBS_TN and F4BS_TN both
        # report Tensor0=0 -- so MIWaveTileA = MIWaveTile[0] = mi[5] and
        # MIWaveTileB = mi[6], which holds for standard GEMM index assignments.
        # An earlier revision admitted both orientations "to be safe"; that kept
        # HalfPLR 1 for M=16 shapes where mi[5] is always 1, and Tensile rejected
        # it every time -- acceptance fell from ~80% to ~30%.
        def half_plr_usable(h):
            if h == 0:
                return True
            return any(
                (not (h & 0x01) or _mi_wave_tile(mi)[0] % 2 == 0)
                and (not (h & 0x02) or _mi_wave_tile(mi)[1] % 2 == 0)
                for mi in mis
            )

        restrict("HalfPLR", half_plr_usable, [0])

        # Rule C -- "LDSSegmentInterleave=1 requested but not applicable:
        # MIWaveGroup unsupported" (segment_interleave.py): only [2,2] and the
        # asymmetric groups, with exactly one dimension equal to 1, can
        # interleave. Same "any group" policy.
        def interleavable(wg):
            return wg == (2, 2) or (wg[0] == 1) != (wg[1] == 1)

        can_interleave = any(interleavable(_mi_wave_group(mi)) for mi in mis)
        restrict("LDSSegmentInterleave", lambda v: v != 1 or can_interleave, [0])

        return fork_params, mi_groups


class GFX1250PostProcessor(BasePostProcessor):
    """gfx1250 heuristic post-processor (no MI augmentation, no CMS)."""


class _GFX1250DropMIGroupGSU:
    """Strip the per-MI GlobalSplitU that MIDesign attaches to each MI group.

    The generic profile emits GlobalSplitU as its own flat ForkParameter
    (``[-1]``, resolved at runtime by calculateAutoGSU). Leaving MIDesign's
    per-MI value in place as well would give the parameter two sources, and the
    per-MI one wins for those MIs. Doing it here rather than in MIDesign keeps
    the change inside the generic gfx1250 search space: MIDesign is shared by
    every architecture, and the heuristic profile still wants its per-MI value.
    ``GSU_PER_MI: true`` drops the flat parameter instead, so this step finds
    nothing to resolve and the per-MI values stay.
    """

    @mark_post_process
    def drop_mi_group_gsu(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        if "GlobalSplitU" not in fork_params:
            return fork_params, mi_groups
        stripped = [{k: v for k, v in e.items() if k != "GlobalSplitU"} for e in mi_groups]
        # Keyed on values alone: MIDesign gives each candidate's MatrixInstruction
        # ForkParameter its own comment/metadata (GSU, LSU, totalGranularity, ...),
        # so two groups with identical values -- the only thing left once
        # GlobalSplitU is stripped -- would still compare unequal and defeat the
        # dedupe if it used ForkParameter/dict equality directly.
        seen = set()
        deduped = []
        for e in stripped:
            key = tuple(sorted((name, tuple(fp.values)) for name, fp in e.items()))
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        return fork_params, deduped


def _mi_num_waves(entry: Dict[str, ForkParameter]) -> int:
    """Waves per workgroup of an MI group entry, or 0 when it cannot be read.

    ``MIWaveGroup`` holds ``mi[7] * mi[8]`` waves, and an entry that carries a
    ``WorkGroup`` multiplies that by its LocalSplitU, ``WorkGroup[2]`` (Tensile:
    ``NumWaves = NumThreads // WavefrontSize``).
    """
    mi = getattr(entry.get("MatrixInstruction"), "values", None)
    if not mi or len(mi) != 9:
        return 0
    waves = mi[7] * mi[8]
    wg = getattr(entry.get("WorkGroup"), "values", entry.get("WorkGroup"))
    if isinstance(wg, list) and len(wg) == 3:
        waves *= wg[2]
    return waves


class _GFX1250TDMFuseOneWave:
    """Keep one-wave MI groups at TDMFuse=0.

    Tensile rejects TDMFuse 1-3 on one wave ("TDMFuse=%d needs more than one
    wave ... Use TDMFuse=0 on one wave"): those values split a descriptor set's
    loads across waves. A flat TDMFuse that the profile pins to 1 would leave
    every one-wave MI without a valid kernel, so when any MI group has one wave
    the flat parameter moves into the MI groups: one-wave groups get the values
    that build there (0), every other group keeps the profile's values.
    """

    @mark_post_process
    def keep_one_wave_tdm_fuse_zero(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        fp = fork_params.get("TDMFuse")
        if fp is None or not isinstance(fp.values, list):
            return fork_params, mi_groups
        if not any(_mi_num_waves(e) == 1 for e in mi_groups):
            return fork_params, mi_groups
        one_wave = [v for v in fp.values if v == 0] or [0]
        coupled = []
        for entry in mi_groups:
            values = one_wave if _mi_num_waves(entry) == 1 else list(fp.values)
            coupled.append({**entry, "TDMFuse": self._make_param("TDMFuse", values)})
        del fork_params["TDMFuse"]
        return fork_params, coupled


def _mi_is_square(entry: Dict[str, ForkParameter]) -> bool:
    """True when an MI group entry's MatrixInstruction has M == N.

    Entries whose MatrixInstruction cannot be read count as square, so the
    SourceSwap coupling never narrows something it cannot parse.
    """
    mi = getattr(entry.get("MatrixInstruction"), "values", None)
    if not isinstance(mi, list) or len(mi) < 2:
        return True
    return mi[0] == mi[1]


class _GFX1250SourceSwapCoupling:
    """Pin SourceSwap to False on non-square MI groups (32x16 on gfx1250).

    SourceSwap=True on a non-square MI swaps the effective M and N, so the
    kernel builds a different macro tile: [32, 16, 128, 1, 1, 8, 4, 2, 2] is
    512x128 with SourceSwap=False but 256x256 with SourceSwap=True.
    _GFX1250ClusterDimCoupling ranks ClusterDim shapes on the tile of the
    MatrixInstruction as written, so SourceSwap=True kernels on these MIs would
    get shapes ranked for the wrong tile (on 4096x3072 the [4, 4] that 256x256
    needs is dropped for [2, 8]). Until that coupling ranks each SourceSwap
    tile, SourceSwap moves into the MI groups: non-square groups keep only
    False, square groups keep every profile value, one entry per value so each
    group entry stays single-valued. A profile without SourceSwap=True, or MI
    groups that are all square, leaves the flat parameter untouched.
    """

    @mark_post_process
    def pin_non_square_source_swap(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        fp = fork_params.get("SourceSwap")
        if fp is None or not fp.active or not isinstance(fp.values, list) or True not in fp.values:
            return fork_params, mi_groups
        if all(_mi_is_square(e) for e in mi_groups):
            return fork_params, mi_groups
        non_square = [v for v in fp.values if v is False] or [False]
        coupled = []
        for entry in mi_groups:
            values = list(fp.values) if _mi_is_square(entry) else non_square
            for v in values:
                coupled.append({**entry, "SourceSwap": self._make_param("SourceSwap", [v])})
        del fork_params["SourceSwap"]
        return fork_params, coupled


def _source_swap_values(
    fork_params: Dict[str, ForkParameter],
    entry: Dict[str, ForkParameter],
) -> List[bool]:
    """The SourceSwap values an MI group entry can be built with.

    A SourceSwap carried by the entry itself wins over the flat parameter. An
    absent, inactive (commented-out) or empty parameter leaves Tensile's
    default, False; a scalar value counts as a one-element list.
    """
    fp = entry.get("SourceSwap") or fork_params.get("SourceSwap")
    if fp is None or not fp.active:
        return [False]
    values = fp.values if isinstance(fp.values, list) else [fp.values]
    return [bool(v) for v in values] or [False]


def _source_swapped_macro_tile(mi: Sequence[int], macro_tile: Tuple[int, int]) -> Tuple[int, int]:
    """``(MT0, MT1)`` of the same MI built with ``SourceSwap=True``.

    SourceSwap on a non-square MI swaps the instruction's effective M and N
    (TensileLite ``effectiveMatrixInstMN``), while MIBlock, MIWaveTile and
    MIWaveGroup keep their physical values. ``Solution.py`` forms
    ``MacroTile0 = MIWaveGroup[0] * MatrixInstM * MatrixInstBM * MIWaveTile[0]``
    (and the N-side analogue) from the effective extents, so MT0 scales by
    ``N / M`` and MT1 by ``M / N``: ``[32,16,128,1,1,8,4,2,2]`` is 512x128
    unswapped and 256x256 swapped. A square MI is unchanged.
    """
    m, n = mi[0], mi[1]
    if m == n:
        return macro_tile
    mt0, mt1 = macro_tile
    return mt0 * n // m, mt1 * m // n


def _buildable_macro_tiles(
    fork_params: Dict[str, ForkParameter],
    entry: Dict[str, ForkParameter],
) -> List[Tuple[int, int]]:
    """Every macro tile an MI group entry builds across its SourceSwap values."""
    mi = entry["MatrixInstruction"].values
    unswapped = mi_macro_tile(entry)
    tiles: List[Tuple[int, int]] = []
    for swap in _source_swap_values(fork_params, entry):
        tile = _source_swapped_macro_tile(mi, unswapped) if swap else unswapped
        if tile not in tiles:
            tiles.append(tile)
    return tiles


class _GFX1250ClusterDimCoupling:
    """Move ClusterDim into the MI groups, with the shapes each MI's grid admits.

    Whether a cluster shape tiles the grid, and which shape of a size fetches
    least, depend on the macro tile, so a flat ClusterDim axis would pair every
    MI with shapes chosen for none of them. Each MI group entry is instead
    repeated once per shape that :func:`select_cluster_dims` keeps for its tile
    grid at this size, and the flat parameter is dropped.

    The macro tile is the one the kernel actually builds. ``SourceSwap=True``
    on a non-square MI transposes it (see :func:`_source_swapped_macro_tile`),
    and the best shape of a size differs with it: at 4096x3072,
    ``[32,16,128,1,1,8,4,2,2]`` prefers ``[2, 8]`` as 512x128 but
    ``[4, 4]`` as 256x256. An entry that carries its own single SourceSwap
    (as _GFX1250SourceSwapCoupling leaves them) is ranked on its own tile.
    When SourceSwap is still a flat parameter that can be True, the shapes
    selected for each buildable tile are merged, so the swapped kernel can
    still draw its own best shape.

    The execution policy stays a separate group, so some pairs remain that
    Tensile rejects: persistent StreamK takes only ``[Cs, 1]`` shapes and
    DataParallel no ``[1, Ck]`` shape. Rejected solutions fail the GA's
    validity check and are never benchmarked.
    """

    @mark_post_process
    def couple_cluster_dim(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], list]:
        fp = fork_params.get("ClusterDim")
        if fp is None or not fp.active or len(fp.values) < 2:
            return fork_params, mi_groups
        coupled = []
        for entry in mi_groups:
            shapes = set()
            for macro_tile in _buildable_macro_tiles(fork_params, entry):
                grid = tile_grid(ctx.M, ctx.N, macro_tile)
                shapes.update(select_cluster_dims(fp.values, grid, macro_tile))
            # Same order select_cluster_dims returns: by size, then by Cs.
            for shape in sorted(shapes, key=lambda s: (s[0] * s[1], s[0])):
                coupled.append({**entry, "ClusterDim": self._make_param("ClusterDim", [list(shape)])})
        del fork_params["ClusterDim"]
        return fork_params, coupled


# Base order is load-bearing. ``BasePostProcessor.__init__`` collects steps by
# walking ``reversed(type(self).__mro__)``, so the LAST base listed contributes
# its steps FIRST. _GFX1250GeometryNarrowing must run after
# _GFX1250WaveTileFilter -- it derives HalfPLR / LDSSegmentInterleave from the
# surviving MI groups, and narrowing against groups the wave-tile filter is
# about to drop would be needlessly conservative. Listing the narrowing first
# here therefore runs it second. _GFX1250ClusterDimCoupling is listed first so
# it runs last, on the final, de-duplicated MI groups; _GFX1250SourceSwapCoupling
# runs just before it and _GFX1250TDMFuseOneWave before that, so each per-shape
# copy inherits its group's SourceSwap and TDMFuse.
class GFX1250GAPostProcessor(
    GFX1250GAOrigamiPolicy,
    _GFX1250ClusterDimCoupling,
    _GFX1250SourceSwapCoupling,
    _GFX1250TDMFuseOneWave,
    _GFX1250DropMIGroupGSU,
    _GFX1250GeometryNarrowing,
    _GFX1250WaveTileFilter,
    BasePostProcessor,
):
    """gfx1250 generic (GA) post-processor (no MI augmentation, no CMS)."""
