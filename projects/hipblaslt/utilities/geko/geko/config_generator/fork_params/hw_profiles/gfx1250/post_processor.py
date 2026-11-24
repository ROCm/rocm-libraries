# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 post-processors.

gfx1250 needs none of the gfx950 post-processing steps:
  * MIArchVgpr is auto-forced True under WMMA — no augmentation needed.
  * CMS (UseCustomMainLoopSchedule) is gfx950-only — no CMS group merge.
The base class's optional MT_DU filtering still runs via ``apply``.
"""

from typing import Dict, List, Sequence, Tuple

from geko.config_generator.constants import mx_format
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


def _interleavable(wave_group: Tuple[int, int]) -> bool:
    """Whether ``LDSSegmentInterleave=1`` supports this MIWaveGroup.

    "LDSSegmentInterleave=1 requested but not applicable: MIWaveGroup
    unsupported" (segment_interleave.py): only [2,2] and the asymmetric groups,
    with exactly one dimension equal to 1, can interleave.
    """
    return wave_group == (2, 2) or (wave_group[0] == 1) != (wave_group[1] == 1)


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
        # MIWaveGroup unsupported" (see _interleavable). Same "any group"
        # policy: an MI that cannot interleave is rejected by TensileLite and
        # redrawn by the GA.
        can_interleave = any(_interleavable(_mi_wave_group(mi)) for mi in mis)
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


def _tdm_fuse_builds(value: int, waves: int, both_scales: bool) -> bool:
    """Whether TensileLite builds ``TDMFuse=value`` on a workgroup of ``waves`` waves.

    Solution.py rejects TDMFuse 1-3 on one wave ("TDMFuse=%d needs more than one
    wave"), and with MX scales on both tensors it rejects TDMFuse 2 and 3 on any
    wave count but 2 and 4 ("TDMFuse=%d with MX scales on both tensors supports
    2 or 4 waves"). ``waves`` 0 means the MI cannot be read; it keeps every value.
    """
    if value == 0 or waves == 0:
        return True
    if waves == 1:
        return False
    if value in (2, 3) and both_scales:
        return waves in (2, 4)
    return True


def _flat_values(fp):
    """The values of an active flat parameter as a list, or None."""
    if fp is None or not fp.active:
        return None
    return fp.values if isinstance(fp.values, list) else [fp.values]


class _GFX1250TDMFuseNarrowing:
    """Narrow the flat TDMFuse to the values some MI group can use.

    TDMFuse, LDSSegmentInterleave and TemporalHintA/B stay flat fork
    parameters, so the GA can change any of them without redrawing the MI.
    The combinations that cannot build are rejected by TensileLite, those
    that would build another kernel's code under a new name by the Ductile
    backend's validity check, and Ductile redraws both:

      * TDMFuse 1-3 on one wave, and 2 or 3 with MX scales on both tensors on
        wave counts other than 2 and 4 (see :func:`_tdm_fuse_builds`);
      * a non-persistent ``LDSSegmentInterleave=1`` without the paired LDS
        layout ("requires paired LDS placement"), which ``TDMFuse > 0`` rules
        out;
      * ``TemporalHintA != TemporalHintB`` while A and B share one TDM load
        (TDMFuse 0 on more than one wave), whose instruction-wide hint bits
        would build the default-hint code under another name.

    Like Rule B and C of _GFX1250GeometryNarrowing, this step only drops a
    value no MI group can use: a TDMFuse above 0 that no MI builds, and
    TDMFuse 0 when nothing needs it. 0 serves the paired layout of
    ``LDSSegmentInterleave=1`` and the one-wave MIs, which build nothing
    else; with neither, the MIs keep the values above 0 the profile offers.
    geko's MX problems put one block size on A and B, so both scales are live
    whenever the problem is MX. The flat list keeps the profile's order and
    never ends up empty.
    """

    def _mx_scales_on_both(self) -> bool:
        gemm = getattr(self, "config", {}).get("GemmProblem")
        if gemm is None:
            return False
        try:
            return mx_format(gemm, self.config.get("ARCH")) is not None
        except ValueError:
            return False

    @mark_post_process
    def narrow_tdm_fuse(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        fp = fork_params.get("TDMFuse")
        if fp is None or not fp.active or not isinstance(fp.values, list) or not mi_groups:
            return fork_params, mi_groups
        both_scales = self._mx_scales_on_both()
        waves = [_mi_num_waves(e) for e in mi_groups]
        lsi = _flat_values(fork_params.get("LDSSegmentInterleave")) or []
        zero_needed = 1 in lsi or 1 in waves

        def usable(v):
            if v == 0:
                return zero_needed
            return any(_tdm_fuse_builds(v, w, both_scales) for w in waves)

        kept = [v for v in fp.values if usable(v)] or ([0] if 0 in fp.values else list(fp.values))
        if kept != fp.values:
            fp.values = kept
        return fork_params, mi_groups


def _mi_is_square(entry: Dict[str, ForkParameter]) -> bool:
    """True when an MI group entry's MatrixInstruction has M == N.

    Entries whose MatrixInstruction cannot be read count as square, so the
    SourceSwap split never acts on something it cannot parse.
    """
    mi = getattr(entry.get("MatrixInstruction"), "values", None)
    if not isinstance(mi, list) or len(mi) < 2:
        return True
    return mi[0] == mi[1]


def _cluster_dim_is_coupled(fork_params: Dict[str, ForkParameter]) -> bool:
    """True when _GFX1250ClusterDimCoupling will move ClusterDim into the MI groups."""
    fp = fork_params.get("ClusterDim")
    return fp is not None and fp.active and len(fp.values) >= 2


class _GFX1250SourceSwapSplit:
    """Split SourceSwap into the MI groups so ClusterDim is ranked per tile.

    SourceSwap=True on a non-square MI swaps the effective M and N, so the
    kernel builds a different macro tile: [32, 16, 128, 1, 1, 8, 4, 2, 2] is
    512x128 with SourceSwap=False but 256x256 with SourceSwap=True, and the
    best ClusterDim shape differs (on 4096x3072, [2, 8] versus [4, 4]).
    _GFX1250ClusterDimCoupling ranks an entry on the tile of the entry's own
    SourceSwap, so every MI group entry is repeated once per SourceSwap value,
    each copy single-valued, and the flat parameter is dropped. Each
    (MI, SourceSwap) entry then gets only the shapes of the tile it builds, and
    the GA draws SourceSwap together with a shape chosen for it.

    The split runs only where it changes the shapes: the profile offers
    SourceSwap=True, some MI is non-square, and ClusterDim is coupled. With
    all-square MIs both values build the same tile, and without a ClusterDim
    axis nothing is ranked, so the flat parameter stays.
    """

    @mark_post_process
    def split_source_swap(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        fp = fork_params.get("SourceSwap")
        if fp is None or not fp.active or not isinstance(fp.values, list) or True not in fp.values:
            return fork_params, mi_groups
        if all(_mi_is_square(e) for e in mi_groups) or not _cluster_dim_is_coupled(fork_params):
            return fork_params, mi_groups
        split = []
        for entry in mi_groups:
            for v in fp.values:
                split.append({**entry, "SourceSwap": self._make_param("SourceSwap", [v])})
        del fork_params["SourceSwap"]
        return fork_params, split


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
    (as _GFX1250SourceSwapSplit leaves them) is ranked on its own tile. When
    SourceSwap is still a flat parameter that can be True, the shapes
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
        if not _cluster_dim_is_coupled(fork_params):
            return fork_params, mi_groups
        fp = fork_params["ClusterDim"]
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
# it runs last, on the final, de-duplicated MI groups. Before it, in run order:
# _GFX1250TDMFuseNarrowing, which runs after the geometry narrowing so Rule C
# has already dropped an LDSSegmentInterleave=1 that no MI can build, and
# _GFX1250SourceSwapSplit, which repeats every MI entry once per SourceSwap
# value right before the coupling, so the coupling ranks each entry on the tile
# it builds. TDMFuse, LDSSegmentInterleave and TemporalHintA/B stay flat. The
# steps therefore run as: _GFX1250WaveTileFilter, _GFX1250GeometryNarrowing,
# _GFX1250DropMIGroupGSU, _GFX1250TDMFuseNarrowing, _GFX1250SourceSwapSplit,
# _GFX1250ClusterDimCoupling.
class GFX1250GAPostProcessor(
    GFX1250GAOrigamiPolicy,
    _GFX1250ClusterDimCoupling,
    _GFX1250SourceSwapSplit,
    _GFX1250TDMFuseNarrowing,
    _GFX1250DropMIGroupGSU,
    _GFX1250GeometryNarrowing,
    _GFX1250WaveTileFilter,
    BasePostProcessor,
):
    """gfx1250 generic (GA) post-processor (no MI augmentation, no CMS)."""
