# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 post-processors.

gfx1250 needs none of the gfx950 post-processing steps:
  * MIArchVgpr is auto-forced True under WMMA — no augmentation needed.
  * CMS (UseCustomMainLoopSchedule) is gfx950-only — no CMS group merge.
The base class's optional MT_DU filtering still runs via ``apply``.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from geko.config_generator.constants import mx_format
from geko.config_generator.fork_params.hw_profiles.gfx1250.cluster_dim import (
    select_cluster_dims,
    tile_grid,
)
from geko.config_generator.fork_params.hw_profiles.gfx1250.optimization_param import (
    SEGMENT_INTERLEAVE_AUTO_VWA,
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

# VectorWidthB values a B-rotate variant may set, widest first: the powers of
# two in Tensile's VectorWidthB range (Common/ValidParameters.py).
_B_ROTATE_VECTOR_WIDTHS = (8, 4, 2, 1)

# MX B-rotate needs MatrixInstN == WavefrontSize / 2, and the gfx1250 profiles
# pin Wave32.
_MX_B_ROTATE_MATRIX_INST_N = 16


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
    ``Tensile/SolutionStructs/Validators/MatrixInstruction.py``.
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


def _auto_vector_width(wave_tile: int, widest: int) -> int:
    """Tensile's auto VectorWidth: the widest power of two <= *widest* dividing *wave_tile*."""
    width = widest
    while wave_tile % width:
        width //= 2
    return width


def _b_rotate_vector_width(
    entry: Dict[str, ForkParameter], data_type: str, mx: bool
) -> Optional[int]:
    """VectorWidthB for a B-rotate variant of this MI group entry; None if it cannot rotate.

    Covers the part of Tensile's B-rotate gate that an MI group entry decides:
    ``b_rotate_reason`` and the [2,2] layout of ``evaluate`` it builds on, both
    in SolutionStructs/segment_interleave.py.

    * "needs MIWaveGroup [2,2], 4 waves, wave32" -- Wave32 is pinned by the profile.
    * "LocalSplitU>1 not supported" -- MIDesign puts LSU in ``WorkGroup[2]``.
    * "A: VWA must be WaveTileA, or WaveTileA/2 with TDMSplit" -- the profile pins
      TDMSplit off, so A has to be read whole: min(MatrixInstM, MatrixInstN) *
      VectorWidthA >= MacroTile0 / 2. That takes an MI without MIBlockM and an
      MIWaveTile[0] that Tensile's auto VectorWidthA reaches (a power of two of at
      most 4 VGPRs of elements). Non-square MIs are left out: SourceSwap swaps
      MatrixInstM and MatrixInstN, so at most one of the two SourceSwap values
      the profile searches reads A whole, and for MX that one fails the
      MatrixInstN check below. A pinned VectorWidthA would reach more wave tiles,
      but codegen asserts on some of them (VectorWidthA 6), and an assert aborts
      the whole GA run instead of rejecting one kernel.
    * "MIWaveTile[1]/VectorWidthB must be even". Tensile's auto VectorWidthB, up
      to 4 VGPRs of elements, makes the ratio 1 for almost every wave tile here,
      so the variant sets VectorWidthB itself.
    * MX: "needs MatrixInstN == WavefrontSize/2" and
      "needs MIWaveTile[1]/VectorWidthB == 2".

    Returns the widest width with an even ratio: two B groups where the wave tile
    allows, which is the only split MX accepts. The LDS layout checks (component
    stride, B reach, buffer alignment) depend on DepthU and padding, so they are
    left to Tensile.
    """
    mi = entry["MatrixInstruction"].values
    if len(mi) != 9 or mi[7] * mi[8] != 4 or _mi_wave_group(mi) != (2, 2):
        return None
    work_group = entry.get("WorkGroup")
    if work_group is not None and len(work_group.values) > 2 and work_group.values[2] > 1:
        return None
    wave_tile_a, wave_tile_b = _mi_wave_tile(mi)
    widest_a = SEGMENT_INTERLEAVE_AUTO_VWA.get(data_type)
    if widest_a is None or mi[0] != mi[1] or mi[4] != 1:
        return None
    if _auto_vector_width(wave_tile_a, widest_a) != wave_tile_a:
        return None
    if mx:
        if mi[1] != _MX_B_ROTATE_MATRIX_INST_N or wave_tile_b % 2:
            return None
        width = wave_tile_b // 2
        return width if width in _B_ROTATE_VECTOR_WIDTHS else None
    for width in _B_ROTATE_VECTOR_WIDTHS:
        if wave_tile_b % width == 0 and (wave_tile_b // width) % 2 == 0:
            return width
    return None


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


class _GFX1250BRotateCoupling:
    """Give each MI that can rotate B a B-rotate variant, with the VectorWidthB it needs.

    Whether B-rotate can build depends on the MI geometry, and so does the
    VectorWidthB it needs (see :func:`_b_rotate_vector_width`). A flat
    ``LDSSegmentBRotate: [0, 1]`` axis would leave VectorWidthB on auto, which
    Tensile rejects for B-rotate on almost every MI. So, for every MI that can
    rotate, the base entry is kept with ``LDSSegmentBRotate: 0`` and followed by
    a variant with ``LDSSegmentBRotate: 1``, ``LDSSegmentInterleave: 1`` (the
    layout it builds on) and that VectorWidthB. A group entry overrides a flat
    parameter, so the variant needs no other change. The flat parameter is then
    removed; MIs without a variant keep Tensile's default, 0.

    If no MI can rotate, or the flat LDSSegmentInterleave has been narrowed to
    [0], the flat parameter is pinned to [0].

    The cost offsets give the variant half its MI's group_0 draws, but the
    execution-policy group is drawn independently, and Tensile rejects B-rotate
    on every persistent kernel. With StreamK on, only draws that pair the variant
    with the non-persistent policy entry can build: 1 in 9 on gfx1250_96cu. The
    LDS layout checks also reject some DepthU values. Rejected solutions fail the
    GA's validity check and are never benchmarked.
    """

    @mark_post_process
    def couple_b_rotate(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        fp = fork_params.get("LDSSegmentBRotate")
        if fp is None or not fp.active or 1 not in fp.values:
            return fork_params, mi_groups
        interleave = fork_params.get("LDSSegmentInterleave")
        mx = bool(mx_format(self.config["GemmProblem"], self.config["ARCH"]))
        coupled = []
        n_variants = 0
        for entry in mi_groups:
            width = None
            if interleave is None or 1 in interleave.values:
                width = _b_rotate_vector_width(entry, self._gt.data_type, mx)
            if width is None:
                coupled.append(entry)
                continue
            coupled.append(
                {**entry, "LDSSegmentBRotate": self._make_param("LDSSegmentBRotate", [0])}
            )
            coupled.append(
                {
                    **entry,
                    "LDSSegmentBRotate": self._make_param("LDSSegmentBRotate", [1]),
                    "LDSSegmentInterleave": self._make_param("LDSSegmentInterleave", [1]),
                    "VectorWidthB": self._make_param("VectorWidthB", [width]),
                }
            )
            n_variants += 1
        if not n_variants:
            fp.values = [0]
            return fork_params, mi_groups
        del fork_params["LDSSegmentBRotate"]
        return fork_params, coupled


class _GFX1250ClusterDimCoupling:
    """Move ClusterDim into the MI groups, with the shapes each MI's grid admits.

    Whether a cluster shape tiles the grid, and which shape of a size fetches
    least, depend on the macro tile, so a flat ClusterDim axis would pair every
    MI with shapes chosen for none of them. Each MI group entry is instead
    repeated once per shape that :func:`select_cluster_dims` keeps for its tile
    grid at this size, and the flat parameter is dropped.

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
            macro_tile = mi_macro_tile(entry)
            grid = tile_grid(ctx.M, ctx.N, macro_tile)
            for shape in select_cluster_dims(fp.values, grid, macro_tile):
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
# it runs last, on the final, de-duplicated MI groups. _GFX1250BRotateCoupling
# could run on either side of it -- a variant's cluster shapes depend only on its
# macro tile -- and runs before it so that each MI's B-rotate-off shapes are
# listed ahead of its B-rotate-on ones.
class GFX1250GAPostProcessor(
    GFX1250GAOrigamiPolicy,
    _GFX1250ClusterDimCoupling,
    _GFX1250BRotateCoupling,
    _GFX1250DropMIGroupGSU,
    _GFX1250GeometryNarrowing,
    _GFX1250WaveTileFilter,
    BasePostProcessor,
):
    """gfx1250 generic (GA) post-processor (no MI augmentation, no CMS)."""

