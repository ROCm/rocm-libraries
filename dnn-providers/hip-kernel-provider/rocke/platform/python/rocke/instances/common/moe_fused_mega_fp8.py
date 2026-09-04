# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single-launch fused-MoE MEGA-kernel (FP8 e4m3 block-scale).

FP8 sibling of ``moe_fused_mega.py`` (the bit-exact f16 mega-kernel). A SINGLE
fused kernel computes, per (inter-slice, sorted-m-block) threadgroup, the full
MoE per-expert path with fp8 e4m3 operands and per-128-block f32 scales:

    GEMM0 (gate) + GEMM1 (up)  -- fp8 X . fp8 W -> f32 acc, per-128-group
        dequant fold (group-accumulator pattern, scale applied POST-MFMA)
      -> SiLU(gate_dq) * up_dq in f32
      -> DYNAMIC-QUANTIZE the f32 Hidden to fp8 (per-128-block-along-inter,
         per-row amax/448 scale), stage into a PERSISTENT LDS Hidden_smem,
         and stash the per-block dynamic scales in an LDS scratch.
      -> GEMM2 (down) -- fp8 Hidden . fp8 W_down -> f32 acc, dequant by
         (hidden_dyn_scale * down_scale) -> weighted atomic reduce into Y.

See ``examples/gfx950/fused_mega_moe/docs/BUILD_SPEC_FP8.md`` for the authoritative
build spec. The dequant ordering follows BUILD_SPEC_FP8 Section 1.2 (the
``block_scale_gemm.py`` group-accumulator pattern): within a 128-wide
contraction block the scales are constant, so
``sum_k (a.sa)(b.sb) = sa.sb . sum_k (a.b)`` -- the scale is applied per
K-group, post-MFMA, NOT in-instruction (which would mean the E8M0 trap of
``cvt_scalef32_pk_f32_fp8x4``).

STAGING STATUS (incremental implementation per BUILD_SPEC_FP8 Phase plan):

* STAGE 1 (THIS FILE, current state):
    - ``FusedMegaKernelSpecFp8`` (fp8 spec / signature / grid).
    - the gate+up fp8 GEMM via ``_emit_fp8_gateup_group_gemm`` (fp8 atom,
      fp8 operand loads, per-128-block scale dequant of BOTH accumulators).
    - SiLU(gate)*up in f32.
    - ``_emit_hidden_dyn_quant_stage``: per-(row, 128-inter-block) amax ->
      dynamic scale -> quantize the f32 Hidden to fp8 and stage into the
      persistent LDS ``Hidden_smem`` + stash the per-block scales in
      ``HiddenScale_smem``.
* STAGE 2 (THIS FILE, current state): the fp8 down GEMM reading ``Hidden_smem``
  as the LDS-resident A operand (``make_transposed``-equivalent implicit reshape
  -- the dynamic-quant write addr == the down MFMA A-read addr, same logical
  ``(m, inter)`` cell, BUILD_SPEC_FP8 Section 3.5), with the per-128-group
  group-accumulator dequant by (``HiddenScale_smem[row, blk]`` * ``down_scale``),
  multiply by the sorted per-token weight, and atomic-add the f32 partial into Y
  (padded rows token id -1 / >= tokens are skipped). Tiled over the H_out output
  in ``tile_n_down`` chunks; grid.x split the inter contraction so each TG
  atomic-adds a PARTIAL Y over the whole H_out.

The fp8 MFMA atom + cvt primitives are 100% reused from ``helpers/atoms.py``
and ``core/ir.py`` (never modified). This file builds the gate+up fp8 GEMM and
the dynamic-quant staging from the fp8-aware ``helpers/mfma_gemm_inner.py``
toolkit (which dispatches the fp8 atom directly) rather than the
``UniversalGemmSpec``-bound f16/bf16 helpers (whose ``io_ir_type`` rejects fp8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...core.ir import (
    CACHE_ALL,
    F32,
    FP8E4M3,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
)
from ...helpers.atoms import MfmaAtom
from ...helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    validate_arch_and_block_size,
    validate_mfma_atom_in_catalog,
)
from ...helpers.quant import quant_max_abs
from ...helpers.tensor_view import TensorDescriptor, TensorView


__all__ = [
    "FusedMegaKernelSpecFp8",
    "build_moe_fused_mega_gemm_fp8",
    "moe_fused_mega_fp8_grid",
    "moe_fused_mega_fp8_signature",
    "build_moe_split_down_fp8",
    "moe_split_down_fp8_grid",
    "moe_split_down_fp8_signature",
]


#: The cadences ``sched_cadence`` accepts. ``none`` is the explicit request for
#: no scheduler hint, which is why the field rejects Python ``None``.
_SCHED_CADENCES = frozenset({"iglp1", "none", "sgb"})


def _emit_loop_cadence_hint(b: IRBuilder, cadence: str) -> None:
    """Emit the per-loop scheduler hint at the TOP of a K-loop body.

    Only ``iglp1`` emits here (it owns the whole-loop schedule and must precede
    the loop body). ``sgb`` emits its cadence inline next to each MFMA instead.
    """
    if cadence == "iglp1":
        b.iglp_opt(1)


# sched_group_barrier mask bits.
_SGB_MFMA = 0x008
_SGB_VMEM_READ = 0x020
_SGB_VMEM_WRITE = 0x040
_SGB_DS_READ = 0x100
_SGB_DS_WRITE = 0x200


def _emit_sgb_gateup_dtla(b: IRBuilder, n_mfma: int, cadence: str) -> None:
    """compv4-style cadence for the DTLA gate/up loop body (per ni).

    The DTLA path stages B via ``global_load...lds`` (a VMEM read whose dest is
    LDS) then ``ds_read`` then ``n_mfma`` MFMAs. Impose: 1 VMEM_READ (the staged
    DMA), DS_READ feeding the MFMA, then the MFMAs -- so the in-flight DMA + LDS
    read overlap the MFMA shadow. No-op unless cadence == 'sgb'.
    """
    if cadence != "sgb":
        return
    b.sched_group_barrier(_SGB_VMEM_READ, 1, 0)
    b.sched_group_barrier(_SGB_DS_READ, n_mfma, 0)
    b.sched_group_barrier(_SGB_MFMA, int(n_mfma), 0)


def _emit_sgb_down_group(b: IRBuilder, n_mfma: int, cadence: str) -> None:
    """compv4-style VMEM<->MFMA cadence for the down loop per-group body.

    The down loop issues a global VMEM W_down load then ``n_mfma`` MFMAs per
    128-group. Impose: 1 VMEM_READ (next group's W_down) under the MFMA(s).
    No-op unless cadence == 'sgb'.
    """
    if cadence != "sgb":
        return
    b.sched_group_barrier(_SGB_VMEM_READ, 1, 0)
    b.sched_group_barrier(_SGB_MFMA, int(n_mfma), 0)


# Group block along the contraction axis (= 4 fp8_16x16x32 atoms).
GROUP_K = 128

# fp8e4m3 saturating clamp magnitude.
FP8_MAX = quant_max_abs("fp8e4m3")  # 448.0
# pyisa dynamic-quant amax floor.
AMAX_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusedMegaKernelSpecFp8:
    """Single-launch fused-MoE mega-kernel spec (FP8 e4m3 block-scale).

    Tile geometry is IDENTICAL to the f16 :class:`FusedMegaKernelSpec`
    (BUILD_SPEC_FP8 Section 2.0): the fp8 atom is also K=32 and the per-lane
    fragment width is the same, so only the operand element dtype (2B->1B) and
    the MFMA intrinsic change.

    * ``tile_m`` = sorted tokens per m-block (pyisa ``sub_x``), default 32.
    * ``tile_n_inter`` = inter columns this TG owns (pyisa ``sub_gu``); the
      GEMM0/1 N extent AND the GEMM2 contraction extent, default 256.
    * ``tile_k_gu`` = K-loop tile along the hidden contraction H for gate/up.
    * MFMA atom = fp8 ``16x16x32`` (e4m3); the 128-wide scale group = 4 atoms.

    OPTIMIZATION-LEVER FLAGS (additive; ALL defaults = the final best config, so
    the default-built kernel is golden-digest byte-identical to the pre-flag
    on-disk kernel -- see ``examples/.../fused_mega/REPRO_PLAN.md``). Each flag
    selects an ALREADY-EXISTING code path so toggling it OFF reproduces an
    earlier optimization level:

    * ``gate_up_k`` / ``down_k`` (level 7, "K=128 hero atom"): MFMA contraction
      width per atom. ``128`` (default, best) -> the ``fp8_16x16x128`` hero atom
      (1 atom / 128-group); ``32`` -> the legacy ``fp8_16x16x32`` (4 atoms /
      group). The K=32 path is the pre-L7 baseline.
    * ``use_dtla`` (level 8, "direct-to-LDS gate+up"): ``True`` (default, best)
      stages the gate/up weight operands global->LDS->MFMA; ``False`` uses the
      legacy global->VGPR->MFMA path. (DTLA requires ``atoms_per_group==1``, i.e.
      ``gate_up_k==128``; with K=32 the DTLA path is auto-bypassed.)
    * ``sched_cadence`` (level 9, "iglp_opt(1) cadence"): per-loop scheduler
      hint, one of ``"iglp1"`` (default, best) / ``"none"`` / ``"sgb"``.
      ``iglp1`` emits ``b.iglp_opt(1)`` once at the top of the gate/up + down
      K-loop bodies, so the post-RA scheduler imposes its canned MFMA/DS
      interleave over each loop region; it is mutually exclusive with the
      ``sched_group_barrier`` cadence, so ``sgb`` emits nothing here and places
      its barriers inline next to each MFMA instead. ``none`` is the pre-level-9
      baseline (no hint at all).
    """

    name: str
    tile_m: int = 16
    tile_n_inter: int = 256
    tile_k_gu: int = 32
    warp_m: int = 1
    warp_n: int = 4
    warp_tile_m: int = 16
    warp_tile_n: int = 16
    warp_tile_k: int = 32
    tile_n_down: int = 256
    tile_k_down: int = 64
    wave_size: int = 64
    block_size: int = 0
    dtype: str = "fp8e4m3"
    # -- optimization-lever flags (defaults = final best; see class docstring) --
    gate_up_k: int = 128  # level 7 (K=128 hero atom); 32 = legacy baseline
    down_k: int = 128  # level 7 (down hero atom); 32 = legacy baseline
    use_dtla: bool = True  # level 8 (direct-to-LDS gate+up); False = global->VGPR
    sched_cadence: str = "iglp1"  # level 9; "iglp1" | "none" | "sgb"
    # Gate/up emitter selection. True (default, byte-identical) = the fused
    # across-ni K-loop with DTLA staging. False = the legacy per-(mi, ni)
    # K-loop (``_emit_fp8_gateup_group_gemm``), which loads weights
    # global->VGPR exactly like the down GEMM does. The fused emitter carries
    # a_next/gb_next[ni]/ub_next[ni] live across EVERY ni at once, which is why
    # its non-DTLA form is intractable for the backend; the per-cell emitter
    # keeps one cell's operands live and compiles fine on the hero atom (the
    # down GEMM has always used that shape).
    use_fused_kloop: bool = True
    # DTLA prefetch depth: how many gate/up N-cells are staged ahead of the one
    # being consumed. 2 (default, byte-identical) = the original ping-pong,
    # which issues cell ni+1 and then immediately blocks on cell ni -- only
    # ~one iteration of MFMA work covers an HBM round trip, so waves sit in
    # s_waitcnt. Deeper costs 2*chunks*wave_size*16B of LDS per extra buffer.
    dtla_depth: int = 2
    # LDS row padding in BYTES for the Hidden staging buffers (0 = off,
    # byte-identical). The fp8 Hidden row stride is ``tile_n`` bytes, and at
    # tile_n=128 that is exactly the 32-bank x 4B span, so every row starts on
    # bank 0 and the down GEMM's A-read (16 lanes, 16 distinct rows) serializes
    # 16 ways. Padding shifts each row by ``lds_pad/4`` banks. Must stay a
    # multiple of 16 to keep ds_read_b128 / 16B stores aligned, which caps the
    # distinct-bank count at 8 (so 16 rows land 2-way rather than 16-way).
    lds_pad: int = 0
    # Windowed (no-LDS) gate/up path: how many ni cells share one sched_barrier.
    # 1 = a barrier after every cell (tightest register bound, least freedom to
    # interleave); larger groups give the scheduler a wider region to overlap
    # loads with MFMAs at the cost of more live fragments.
    window_group: int = 1
    # Down GEMM: fuse all n cells of an mi row into one K-loop so the LDS A
    # fragment is read once instead of per cell, and the W_down loads are
    # prefetched in a rolling window across cells. False (default) keeps the
    # byte-identical per-cell emitter.
    down_fused_cells: bool = False
    down_depth: int = 2
    down_group: int = 1
    # How the windowed paths fence their scheduling regions. "barrier" =
    # sched_barrier(0), a hard fence: bounds registers tightly but also caps how
    # many loads can be in flight to one region's worth. "sgb" = express the
    # VMEM/MFMA interleave with sched_group_barrier instead, which constrains
    # placement without fencing, so loads from later regions can still hoist.
    window_sched: str = "barrier"
    # Width (along inter) of ONE dynamic fp8 scale block for the INTERMEDIATE
    # hidden tensor. This is entirely internal -- gate/up produces it and the
    # down GEMM consumes it -- unlike the weight scale groups, which are fixed
    # at GROUP_K=128 by the checkpoint. The fuse-quant invariant needs a whole
    # hidden scale block to live inside one CTA's inter tile, so pinning this to
    # 128 is what forces tile_n_inter >= 128 and caps the grid at 6 inter
    # slices. Lowering it to 64 legalises tile_n_inter=64 and doubles the CTA
    # count. Requires down_k <= hidden_group_k so a down atom never straddles
    # two hidden scale blocks, and down_fused_cells (the per-cell down emitter
    # still assumes 128).
    hidden_group_k: int = GROUP_K
    # Consume the weights from the coalesced tiled layout produced by
    # ``swizzle_b_fp8_weights`` instead of natural (out_rows, K) row-major.
    # Independently settable for the gate/up streams and the down stream so the
    # two can be attributed separately. False (default) = byte-identical
    # row-major addressing. The CALLER must upload weights already permuted --
    # these flags only change how the kernel addresses them, so a mismatch is
    # silently wrong rather than an error. Each swizzled stream needs the hero
    # atom (see ``b_swizzle_supported``); the builder enforces that.
    swizzle_gu: bool = False
    swizzle_down: bool = False
    #: Build-time upper bound on the intermediate dimension, used ONLY by
    #: :func:`build_moe_split_down_fp8` to size its static LDS staging buffer.
    #: The contraction extent itself stays runtime, so one binary serves any
    #: I <= this. Ignored by the fused kernel.
    split_inter_max: int = 768
    #: Number of ``tile_n_down``-wide output tiles ONE split-down CTA walks,
    #: trading grid width for in-CTA reuse of the staged intermediate. At 1
    #: (default) H_out is purely a grid axis, so the h_out/tile_n_down CTAs of
    #: a token block each re-stage the SAME [tile_m, I] intermediate into their
    #: own LDS -- 16x redundant staging at qwen3 prefill. Raising it stages
    #: once and loops, which also gives each warp that many times more MFMAs to
    #: amortise the staging prologue over. Ignored by the fused kernel.
    down_h_loop: int = 1
    #: Software-pipeline the split-down K loop: issue group ``kg+1``'s W_down
    #: loads before consuming ``kg``'s. The existing ``down_depth`` window runs
    #: along N, which in this kernel is only 2 cells wide, so it cannot put
    #: more than 2 loads in flight no matter what it is set to. Ignored by the
    #: fused kernel, whose down half has a different N width and shares its
    #: K loop with the gate/up epilogue.
    down_pipeline_k: bool = False
    #: Take the intermediate's fp8 scale as an INPUT (in ``InterScale``, which
    #: stage 1 otherwise writes) instead of deriving it from the tile's own
    #: amax. The dynamic form is a three-pass, barrier-separated epilogue --
    #: SiLU to an f32 LDS scratch, a cross-lane and cross-warp amax, then a
    #: re-read and convert -- because a scale that depends on the whole tile
    #: cannot be applied until the whole tile exists. A known scale collapses
    #: all of that into the SiLU pass: convert and store fp8 straight from the
    #: accumulators. Costs the f32 scratch, the amax buffer, two of three
    #: barriers and the entire third pass. Split gate/up only; the fused
    #: kernel's stage 2 consumes the scratch in place, so it is untouched.
    static_inter_scale: bool = False
    #: Have the MFMAs accumulate in arch VGPRs rather than AGPRs
    #: (``-amdgpu-mfma-vgpr-form``). The block-scale fold has to bring every
    #: accumulator into the VALU each K group -- with GROUP_K == atom.k there
    #: is exactly one MFMA per scale group, so the fold runs at MFMA frequency
    #: -- and out of AGPRs that costs a ``v_accvgpr_read`` per value: 64 per
    #: K-loop iteration in gate/up, 19% of the loop body. Accumulating in VGPRs
    #: deletes them and, because the two files share one 512-register budget,
    #: also drops gate/up from 280 registers to 202 (1 -> 2 waves/SIMD).
    mfma_vgpr_form: bool = False
    #: Stage the gate/up weight tile ONCE per CTA into LDS, shared by every
    #: wave, instead of each wave streaming its own private copy global->VGPR.
    #:
    #: This is what makes a wide ``tile_m`` actually pay. ``warp_m`` splits M
    #: across waves so the accumulator count per wave stays fixed, but with
    #: private B every wave still fetches the whole weight tile, so the weight
    #: traffic is per-WAVE and ``tile_m`` buys nothing -- measured: tile_m 16 ->
    #: 64 left stage 1 unchanged. Sharing makes the traffic per-CTA, so
    #: tile_m=64 cuts the gate/up weight stream ~4x (6.63 GB -> 1.66 GB at
    #: T=4096). It is also how Triton reaches 8 waves/CU on this shape: its
    #: 24 KB of LDS backs 4 waves (6 KB/wave) where ours backs 1.
    #:
    #: Uses ``global_load_lds`` for the staging DMA, so unlike Triton there is
    #: no VGPR round trip and no ``ds_write`` at all -- only the read-back.
    #: Requires ``atoms_per_group == 1`` (gate_up_k=128) so one staged tile
    #: serves exactly one scale group, and ``n_warps`` to divide the tile's
    #: n-cell count so the staging splits evenly.
    coop_b_lds: bool = False
    #: Overlay the epilogue's fp8 Hidden staging on the cooperative B tile,
    #: which would buy 1 -> 2 workgroups/CU. OFF because it does not hold: the
    #: two live ranges look disjoint (the B tile dies at the last K group, the
    #: staging is written after it, with the K loop's trailing barrier as the
    #: handoff) but overlaying them fails parity identically at every geometry
    #: -- including tile_m=16, a single wave with no barriers at all -- so the
    #: hazard is not cross-wave. The smem pool's own liveness packer also
    #: declines to overlap them, which is consistent with a real interference
    #: this builder's local view does not show. Kept as an opt-in probe rather
    #: than a silent 2x-occupancy trap.
    coop_alias: bool = False

    def __post_init__(self) -> None:
        if self.block_size == 0:
            object.__setattr__(
                self,
                "block_size",
                self.warp_m * self.warp_n * self.wave_size,
            )
        # ``sched_cadence`` used to be ``str | None``, where None meant "defer
        # to the environment", whose default was iglp1. Now that the knob lives
        # here, None would compare unequal to every cadence name and so emit no
        # hint at all -- the level-9 speedup dropped on the floor, with a kernel
        # that still builds and still computes the right answer. A caller
        # asking for no hint spells it "none".
        if self.sched_cadence not in _SCHED_CADENCES:
            raise ValueError(
                f"sched_cadence={self.sched_cadence!r} is not one of "
                f"{sorted(_SCHED_CADENCES)}"
                + (
                    "; None used to mean the iglp1 default and now means no "
                    "scheduler hint, so it must be spelled explicitly"
                    if self.sched_cadence is None
                    else ""
                )
            )
        # The four ``mfmas_*`` properties below are floor divisions, so a warp
        # grid that does not tile its block exactly does not fail -- it silently
        # drops the remainder. tile_n_down=128 with warp_n=3 gives 2 atoms per
        # warp, i.e. 96 of 128 output columns written and the rest left at
        # whatever was there, which reads as a plausible-looking wrong answer
        # rather than a crash. Reject it at construction instead.
        for field, extent, per_warp, axis, warps in (
            ("tile_m", self.tile_m, self.warp_tile_m, "warp_m", self.warp_m),
            ("tile_n_inter", self.tile_n_inter, self.warp_tile_n, "warp_n",
             self.warp_n),
            ("tile_n_down", self.tile_n_down, self.warp_tile_n, "warp_n",
             self.warp_n),
        ):
            if extent % (per_warp * warps):
                covered = (extent // warps // per_warp) * per_warp * warps
                raise ValueError(
                    f"{field}={extent} is not a multiple of {axis}={warps} x "
                    f"{per_warp}, so the warp grid would silently cover only "
                    f"{covered} of it"
                )

    # -- derived geometry helpers ----------------------------------------

    def gate_up_atom(self) -> MfmaAtom:
        # L7 (flag ``gate_up_k``): the unscaled fp8 16x16x128 hero atom (K=128
        # per MFMA, 4x fewer K-trips than the legacy 16x16x32). The 128-wide
        # scale group then maps to EXACTLY ONE atom (atoms_per_group =
        # GROUP_K // atom.k = 1), so the per-128-block dequant fold aligns with a
        # single MFMA. Toggling ``gate_up_k=32`` selects the legacy 4-atom path
        # (the pre-L7 baseline); default 128 = best.
        if self.gate_up_k == 32:
            return MfmaAtom.fp8_16x16x32()
        return MfmaAtom.fp8_16x16x128()

    def down_atom(self) -> MfmaAtom:
        if self.down_k == 32:
            return MfmaAtom.fp8_16x16x32()
        return MfmaAtom.fp8_16x16x128()

    @property
    def mfmas_m(self) -> int:
        """Atoms along M per warp for the gate/up tile."""
        return (self.tile_m // self.warp_m) // self.warp_tile_m

    @property
    def mfmas_n(self) -> int:
        """Atoms along N per warp for the gate/up tile."""
        return (self.tile_n_inter // self.warp_n) // self.warp_tile_n

    @property
    def mfmas_m_down(self) -> int:
        """Atoms along M per warp for the down tile (M x H_out_slice)."""
        return (self.tile_m // self.warp_m) // self.warp_tile_m

    @property
    def mfmas_n_down(self) -> int:
        """Atoms along N (= H_out output) per warp for the down tile."""
        return (self.tile_n_down // self.warp_n) // self.warp_tile_n

    def kernel_name(self) -> str:
        return (
            f"{self.name}_moe_fused_mega_fp8_"
            f"m{self.tile_m}n{self.tile_n_inter}k{self.tile_k_gu}"
        )


# ---------------------------------------------------------------------------
# Grid + signature (BUILD_SPEC_FP8 Section 2.7 / 2.8)
# ---------------------------------------------------------------------------


def moe_fused_mega_fp8_grid(
    num_m_blocks: int, inter: int, spec: FusedMegaKernelSpecFp8
) -> Tuple[int, int, int]:
    """Mega-kernel launch grid (unchanged from the f16 kernel).

    ``grid = (ceil(inter / tile_n_inter), num_m_blocks, 1)``. Canonical decode
    (I=7168, tile_n_inter=256): grid.x = 28; grid.y = num_m_blocks = 8.
    """
    sub_gu = spec.tile_n_inter
    gx = (inter + sub_gu - 1) // sub_gu
    return (gx, num_m_blocks, 1)


# Default persistent-grid cap: MI355X 256 CU x ~2 blocks/CU at block_size 256.
PERSISTENT_P_CAP = 512


def moe_fused_mega_fp8_persistent_grid(
    num_m_blocks: int,
    inter: int,
    spec: FusedMegaKernelSpecFp8,
    p_cap: int = PERSISTENT_P_CAP,
) -> Tuple[Tuple[int, int, int], int, int, int]:
    """Persistent 1-D launch grid + the (grid_x, total_work, P) ABI scalars.

    Relinearizes the 2-D ``(grid_x, num_active_m_blocks)`` work space into a 1-D
    grid-stride loop. ``grid_x = ceil(inter / tile_n_inter)`` is the inter-tile
    modulus; ``total_work = grid_x * num_m_blocks`` is the number of (bx, by)
    work-items; ``P = min(total_work, p_cap)`` is the launched persistent block
    count (= the loop stride). For THIS decode shape ``total_work`` (56 at T1,
    224 at T8) <= p_cap so ``P == total_work`` and the trip count is 1 -- a pure
    1-D relinearization that removes the 2-D grid y-dim + per-block grid-tail
    (the T1 lever). The SAME compiled kernel is correct for ``total_work > P``
    via the strided ``w < total_work`` bound.

    Returns ``((P, 1, 1), grid_x, total_work, P)``.
    """
    sub_gu = spec.tile_n_inter
    grid_x = (inter + sub_gu - 1) // sub_gu
    total_work = grid_x * num_m_blocks
    P = min(total_work, p_cap) if total_work > 0 else 1
    return ((P, 1, 1), grid_x, total_work, P)


def moe_fused_mega_fp8_signature(
    spec: FusedMegaKernelSpecFp8,
    *,
    persistent: bool = False,
    split_gateup: bool = False,
):
    from ...helpers.spec import SignatureBuilder

    sb = (
        SignatureBuilder()
        .ptr("A", "fp8e4m3")  # quantized activation X (pyisa: fp8 + input_scale)
        .ptr("WGate", "fp8e4m3")
        .ptr("WUp", "fp8e4m3")
        .ptr("WDown", "fp8e4m3")
        .ptr("AScale", "f32")  # input_scale, per (token-block, H-block-of-128)
        .ptr("WGateScale", "f32")  # fc1_scale gate half, per (E, I-block, H-block)
        .ptr("WUpScale", "f32")  # fc1_scale up half
        .ptr("WDownScale", "f32")  # fc2_scale, per (E, H_out-block, I-block)
        .ptr("SortedTokenIds", "i32")
        .ptr("SortedWeights", "f32")
        .ptr("BlockExpertIds", "i32")
        .ptr("Y", "f32")  # f32 to reuse the f16 atomic epilogue unchanged
        .scalar("M", "i32")
        .scalar("N", "i32")  # = I (inter dim)
        .scalar("K", "i32")  # = H (hidden contraction)
        .scalar("H_out", "i32")  # = H (down output)
        .scalar("stride_a", "i32")
        .scalar("stride_b_gate", "i32")
        .scalar("stride_b_up", "i32")
        .scalar("stride_b_down", "i32")
        .scalar("stride_a_scale", "i32")
        .scalar("stride_gate_scale", "i32")
        .scalar("stride_up_scale", "i32")
        .scalar("stride_down_scale", "i32")
        .scalar("stride_gate_scale_e", "i32")
        .scalar("stride_up_scale_e", "i32")
        .scalar("stride_down_scale_e", "i32")
        .scalar("slot_size", "i32")
        .scalar("tokens", "i32")
    )
    if persistent:
        # Persistent ABI variant: appended AFTER ``tokens`` to match the
        # builder's b.param() append order (grid_x, total_work, P).
        sb = sb.scalar("grid_x", "i32").scalar("total_work", "i32").scalar("P", "i32")
    if split_gateup:
        # PARTIAL-FUSION stage 1: the requantized intermediate leaves in HBM
        # instead of staying in LDS for a fused stage 2. Appended after
        # ``tokens`` to match the builder's b.param() order. WDown/WDownScale/
        # SortedWeights/Y stay in the ABI but go unread, which costs nothing.
        sb = sb.ptr("Inter", "fp8e4m3").ptr("InterScale", "f32")
    return sb.build()


# ---------------------------------------------------------------------------
# STAGE 1a: gate+up fp8 GEMM with per-128-block dequant
# ---------------------------------------------------------------------------


def _emit_fp8_gateup_group_gemm(
    b: IRBuilder,
    *,
    A: Value,
    WGate: Value,
    WUp: Value,
    AScale: Value,
    WGateScale: Value,
    WUpScale: Value,
    atom: MfmaAtom,
    lane_decode,
    m_tile_base: Value,
    n_tile_base: Value,
    K: Value,
    stride_a_scale: Value,
    stride_gate_scale: Value,
    stride_up_scale: Value,
    tag: str,
    swizzle_b: bool,
    wave_size: int,
) -> Tuple[Value, Value]:
    """Gate + up fp8 GEMM, returning ``(gate_dq, up_dq)`` per-lane f32 vectors.

    Group-accumulator pattern (BUILD_SPEC_FP8 Section 1.2): the outer loop walks
    128-wide groups along the hidden contraction K (= H). Per group, 4
    fp8_16x16x32 atoms accumulate into a FRESH ``group_acc`` (one for gate, one
    for up), then the group is folded into the outer accumulator scaled by
    ``a_scale * b_scale`` -- a single ``v_pk_fma_f32`` per accumulator. The
    A read (the quantized activation) is shared across gate and up; only the
    B-side weight scale differs.

    Scale index math (BUILD_SPEC_FP8 Section 1.3), per-128 along the
    contraction and per-128 along the output:

        a_scale_off = (m_row // GROUP_K_M) * k_scale_count + kg
        b_scale_off = kg * n_scale_count + (n_col // GROUP_K)

    Here the activation amax is per-(token-block, H-block); we use the per-row
    granularity available to the lane (``m_row``) with the H-block index ``kg``,
    and the weight scale is per (output-128, contraction-128).
    """
    c_group_k = b.const_i32(GROUP_K)
    c_atom_k = b.const_i32(atom.k)
    atoms_per_group = GROUP_K // atom.k  # 4

    m_row = b.add(m_tile_base, lane_decode.m_in_atom)
    n_col = b.add(n_tile_base, lane_decode.n_in_atom)

    # Per-lane scale offsets vary along the K-group index ``kg`` only.
    a_row_scale_base = b.mul(m_row, stride_a_scale)
    # Derive the scale block from the WAVE-UNIFORM tile base, not from n_col.
    # atom.n divides GROUP_K, so [n_tile_base, n_tile_base + atom.n) never
    # straddles a scale block and the two agree exactly -- but only this form
    # is visibly uniform to the backend, which is what lets the block scale be
    # fetched with s_load into an SGPR instead of burning a vector memory slot
    # on a 4-byte broadcast per MFMA.
    n_blk = b.div(n_tile_base, c_group_k)

    zero = atom.zero_acc(b)
    gate_zero = atom.zero_acc(b)
    up_zero = atom.zero_acc(b)

    # Outer loop over 128-wide contraction groups: num_groups = K // GROUP_K.
    num_groups = b.div(K, c_group_k)
    outer = b.scf_for_iter(
        b.const_i32(0),
        num_groups,
        b.const_i32(1),
        [(f"gate_outer_{tag}", gate_zero), (f"up_outer_{tag}", up_zero)],
        iv_name=f"kg_{tag}",
    )
    with outer as (kg, (gate_outer, up_outer)):
        # Per-group scales (one f32 each).
        a_scale_off = b.add(a_row_scale_base, kg)
        a_scale_v = b.global_load_f32(AScale, a_scale_off)
        gate_scale_off = b.add(b.mul(kg, stride_gate_scale), n_blk)
        up_scale_off = b.add(b.mul(kg, stride_up_scale), n_blk)
        gate_scale_v = b.global_load_f32(WGateScale, gate_scale_off)
        up_scale_v = b.global_load_f32(WUpScale, up_scale_off)
        gate_ab = b.fmul(a_scale_v, gate_scale_v)
        up_ab = b.fmul(a_scale_v, up_scale_v)

        k_group_base = b.mul(kg, c_group_k)

        # Fresh per-group accumulators; 4 fp8 atoms cover the 128-wide group.
        ginner = b.scf_for_iter(
            b.const_i32(0),
            b.const_i32(atoms_per_group),
            b.const_i32(1),
            [(f"g_acc_{tag}", zero), (f"u_acc_{tag}", zero)],
            iv_name=f"kk_{tag}",
        )
        with ginner as (kk, (g_acc, u_acc)):
            k_tile_base = b.add(k_group_base, b.mul(kk, c_atom_k))
            a_frag = _load_a_fp8(
                b,
                A=A,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                k_tile_base=k_tile_base,
                K=K,
            )
            gb_frag = _load_b_fp8(
                b,
                B=WGate,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=k_tile_base,
                N=K,
                swizzle=swizzle_b,
                wave_size=wave_size,
            )
            ub_frag = _load_b_fp8(
                b,
                B=WUp,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=k_tile_base,
                N=K,
                swizzle=swizzle_b,
                wave_size=wave_size,
            )
            g_new = atom.emit(b, a_frag, gb_frag, g_acc)
            u_new = atom.emit(b, a_frag, ub_frag, u_acc)
            b.scf_yield(g_new, u_new)
        group_gate = ginner.results[0]
        group_up = ginner.results[1]

        # Fold (post-MFMA, per-group): outer += group * (a_scale * b_scale).
        gate_scale_vec = b.vector_splat(gate_ab, atom.c_per_lane)
        up_scale_vec = b.vector_splat(up_ab, atom.c_per_lane)
        gate_outer_new = b.vector_fma(group_gate, gate_scale_vec, gate_outer)
        up_outer_new = b.vector_fma(group_up, up_scale_vec, up_outer)
        b.scf_yield(gate_outer_new, up_outer_new)

    return outer.results[0], outer.results[1]


def _emit_fp8_gateup_fused_kloop(
    b: IRBuilder,
    *,
    A: Value,
    WGate: Value,
    WUp: Value,
    AScale: Value,
    WGateScale: Value,
    WUpScale: Value,
    atom: MfmaAtom,
    lane_decode,
    m_tile_base: Value,
    n_tile_bases,
    K: Value,
    stride_a_scale: Value,
    stride_gate_scale: Value,
    stride_up_scale: Value,
    tag: str,
    cadence: str,
    prefetch_depth: int,
    sched_group: int,
    sched_mode: str,
    swizzle_b: bool,
    wave_size: int,
    dtla=None,
    coop_b=None,
):
    """Gate + up fp8 GEMM fused across ALL ni cells of one mi row.

    COMBINATION lever (gate+up SW-pipeline + wave-pair odd/even MFMA interleave).
    The legacy ``_emit_fp8_gateup_group_gemm`` ran an INDEPENDENT K-loop per
    (mi, ni) cell, reloading the shared A operand ``mfmas_n`` times and emitting
    the two MFMAs (gate, up) of each cell back-to-back -- a bursty pattern that
    starves the compiler of any load/MFMA overlap window. This fused emitter:

    * Runs ONE outer 128-group K-loop carrying ``2 * len(n_tile_bases)`` outer
      f32 accumulators (gate[ni], up[ni]).
    * Loads the shared A fragment ONCE per atom (m_tile_base, k) and reuses it
      across every ni -- killing the (mfmas_n - 1) redundant A streams.
    * Unrolls the 4-atom inner group in Python (the trip count is the
      compile-time constant ``atoms_per_group``; an scf.for iter-arg cannot carry
      an fp8e4m3 fragment) and REGISTER-DOUBLE-BUFFERS the next atom's A + every
      WGate/WUp fragment (``a_next`` / ``gb_next[ni]`` / ``ub_next[ni]``): the
      next loads are ISSUED before the current MFMAs consume their operands,
      giving an in-flight load under the MFMA.
    * INTERLEAVES the MFMAs across ni in pyisa wave-pair odd/even order
      (gate[0], up[0], gate[1], up[1], ...) with the next-atom loads spliced in
      the middle of the burst, so the longest run of consecutive MFMAs that
      share no intervening load drops from the legacy 5 toward ~2.

    Returns ``(gate_list, up_list)`` -- per-ni f32 outer accumulators, same
    order as ``n_tile_bases``.
    """
    c_group_k = b.const_i32(GROUP_K)
    c_atom_k = b.const_i32(atom.k)
    atoms_per_group = GROUP_K // atom.k  # 4
    nni = len(n_tile_bases)

    m_row = b.add(m_tile_base, lane_decode.m_in_atom)
    a_row_scale_base = b.mul(m_row, stride_a_scale)

    # Per-ni n_col / n_blk for the weight-scale index math. The scale block
    # comes off the WAVE-UNIFORM tile base (atom.n divides GROUP_K, so an atom
    # never straddles a block); n_cols stays lane-varying for the weight data
    # addresses. Only the uniform form lets these 4-byte block scales be
    # fetched with s_load instead of a vector memory slot per MFMA.
    n_cols = [b.add(nb, lane_decode.n_in_atom) for nb in n_tile_bases]
    # One scale block per WARP, not per cell. The builder already requires a
    # warp's whole N-extent to sit inside one GROUP_K block (the fuse-quant
    # invariant), so every cell's n_tile_base floor-divides to the same block.
    # Deriving it per cell instead made the backend emit nni separate s_load
    # chains -- 16 s_load_dword plus ~80 SALU of 64-bit address math per K
    # group in the shipped gate/up loop, ~36% of the body -- to fetch two
    # distinct f32 values eight times each. It cannot CSE them itself because
    # proving (base + 16) // 128 == base // 128 needs the alignment fact.
    # Divisibility, not just "fits": it is what makes the warp's base a
    # multiple of its own extent and therefore unable to straddle a block.
    shared_blk = GROUP_K % (nni * atom.n) == 0
    n_blks = (
        [b.div(n_tile_bases[0], c_group_k)] * nni
        if shared_blk
        else [b.div(nb, c_group_k) for nb in n_tile_bases]
    )

    # Outer iter-args: gate[0..nni), up[0..nni).
    zero = atom.zero_acc(b)
    iter_args = []
    for ni in range(nni):
        iter_args.append((f"g_out_{tag}_{ni}", atom.zero_acc(b)))
    for ni in range(nni):
        iter_args.append((f"u_out_{tag}_{ni}", atom.zero_acc(b)))

    num_groups = b.div(K, c_group_k)
    outer = b.scf_for_iter(
        b.const_i32(0), num_groups, b.const_i32(1), iter_args, iv_name=f"kg_{tag}"
    )
    # The windowed global->VGPR path drives its own schedule with explicit
    # sched_barriers, and iglp_opt "owns the loop schedule" -- letting it run
    # here would undo the window and hoist every cell's loads back to the top.
    # Scoped to THIS loop so the down GEMM keeps its iglp cadence.
    _use_window = dtla is None and atoms_per_group == 1
    with outer as (kg, outs):
        _emit_loop_cadence_hint(b, "none" if _use_window else cadence)
        gate_outer = list(outs[:nni])
        up_outer = list(outs[nni:])

        # Per-group scales (hoisted alongside the operand prefetch).
        gate_ab = []
        up_ab = []
        a_scale_off = b.add(a_row_scale_base, kg)
        a_scale_v = b.global_load_f32(AScale, a_scale_off)
        kg_gate = b.mul(kg, stride_gate_scale)
        kg_up = b.mul(kg, stride_up_scale)
        for ni in range(nni):
            gsc = b.global_load_f32(WGateScale, b.add(kg_gate, n_blks[ni]))
            usc = b.global_load_f32(WUpScale, b.add(kg_up, n_blks[ni]))
            gate_ab.append(b.fmul(a_scale_v, gsc))
            up_ab.append(b.fmul(a_scale_v, usc))

        k_group_base = b.mul(kg, c_group_k)

        def _a_at(kk):
            kbase = b.add(k_group_base, b.mul(b.const_i32(kk), c_atom_k))
            return _load_a_fp8(
                b,
                A=A,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                k_tile_base=kbase,
                K=K,
            )

        def _gb_at(ni, kk):
            kbase = b.add(k_group_base, b.mul(b.const_i32(kk), c_atom_k))
            return _load_b_fp8(
                b,
                B=WGate,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_bases[ni],
                k_tile_base=kbase,
                N=K,
                swizzle=swizzle_b,
                wave_size=wave_size,
            )

        def _ub_at(ni, kk):
            kbase = b.add(k_group_base, b.mul(b.const_i32(kk), c_atom_k))
            return _load_b_fp8(
                b,
                B=WUp,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_bases[ni],
                k_tile_base=kbase,
                N=K,
                swizzle=swizzle_b,
                wave_size=wave_size,
            )

        # Fresh per-group accumulators, one gate + one up per ni.
        g_acc = [zero] * nni
        u_acc = [zero] * nni

        if coop_b is not None:
            # ---- cooperative CTA-shared B tile ------------------------------
            # Every wave in the CTA needs the SAME gate/up weight tile for this
            # K group (B does not depend on M), so the CTA stages it once and
            # all waves read it back. The A activation stays private per wave:
            # each wave owns its own rows, A is ~1/4 of the CTA's bytes, and
            # keeping it out of LDS saves both budget and a barrier.
            a_cur = _load_a_fp8(
                b,
                A=A,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                k_tile_base=k_group_base,
                K=K,
            )
            for half, Bptr in ((0, WGate), (1, WUp)):
                for j, n_base in enumerate(coop_b["stage_n_bases"]):
                    _dtla_stage_b_fp8(
                        b,
                        B=Bptr,
                        atom=atom,
                        lane_decode=lane_decode,
                        n_tile_base=n_base,
                        k_tile_base=k_group_base,
                        N=K,
                        stage_view=coop_b["view"],
                        slot=j,
                        wave_lds_base=coop_b["stage_bases"][half],
                        lane=coop_b["lane"],
                        wave_size=coop_b["wave_size"],
                        swizzle=swizzle_b,
                    )
            # b.sync() drains vmcnt+lgkmcnt BEFORE the barrier, which is what
            # makes another wave's DMA visible: s_barrier alone orders execution,
            # not memory.
            b.sync()
            for ni in range(nni):
                gb = _dtla_read_b_fp8(
                    b,
                    atom=atom,
                    stage_view=coop_b["view"],
                    slot=ni,
                    lane=coop_b["lane"],
                    warp_row_base=coop_b["read_row_base"],
                    wave_size=coop_b["wave_size"],
                )
                ub = _dtla_read_b_fp8(
                    b,
                    atom=atom,
                    stage_view=coop_b["view"],
                    slot=coop_b["n_cells_all"] + ni,
                    lane=coop_b["lane"],
                    warp_row_base=coop_b["read_row_base"],
                    wave_size=coop_b["wave_size"],
                )
                g_acc[ni] = atom.emit(b, a_cur, gb, g_acc[ni])
                u_acc[ni] = atom.emit(b, a_cur, ub, u_acc[ni])
            # Second barrier: the tile is single-buffered (a double buffer would
            # need 64 KiB and halve occupancy), so no wave may start the next
            # group's DMA until every wave has finished reading this one.
            b.sync()
        elif dtla is not None and atoms_per_group == 1:
            # ---- DTLA path (GOAL 1): direct-to-LDS gate+up B operands -------
            # The A activation stays a cheap global->VGPR load (tiny, reused).
            # The dominant WGate/WUp weight streams go global->LDS->MFMA via
            # ``b.global_load_lds``, PING-PONG double-buffered over ni so the DMA
            # for ni+1 is issued (and in flight) while ni's MFMAs run, then the
            # vmcnt drain + ds_read feed the MFMA. Per-wave LDS base + CACHE_ALL
            # are threaded via the ``dtla`` bundle.
            kbase0 = k_group_base
            a_cur = _load_a_fp8(
                b,
                A=A,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                k_tile_base=kbase0,
                K=K,
            )

            def _stage(ni, slot_pair):
                # slot_pair 0/1 -> gate slot 2*pair, up slot 2*pair+1.
                _dtla_stage_b_fp8(
                    b,
                    B=WGate,
                    atom=atom,
                    lane_decode=lane_decode,
                    n_tile_base=n_tile_bases[ni],
                    k_tile_base=kbase0,
                    N=K,
                    stage_view=dtla["view"],
                    slot=2 * slot_pair,
                    wave_lds_base=dtla["base"],
                    lane=dtla["lane"],
                    wave_size=dtla["wave_size"],
                    swizzle=swizzle_b,
                )
                _dtla_stage_b_fp8(
                    b,
                    B=WUp,
                    atom=atom,
                    lane_decode=lane_decode,
                    n_tile_base=n_tile_bases[ni],
                    k_tile_base=kbase0,
                    N=K,
                    stage_view=dtla["view"],
                    slot=2 * slot_pair + 1,
                    wave_lds_base=dtla["base"],
                    lane=dtla["lane"],
                    wave_size=dtla["wave_size"],
                    swizzle=swizzle_b,
                )

            def _read(slot_pair):
                g = _dtla_read_b_fp8(
                    b,
                    atom=atom,
                    stage_view=dtla["view"],
                    slot=2 * slot_pair,
                    lane=dtla["lane"],
                    warp_row_base=dtla["warp_row_base"],
                    wave_size=dtla["wave_size"],
                )
                u = _dtla_read_b_fp8(
                    b,
                    atom=atom,
                    stage_view=dtla["view"],
                    slot=2 * slot_pair + 1,
                    lane=dtla["lane"],
                    warp_row_base=dtla["warp_row_base"],
                    wave_size=dtla["wave_size"],
                )
                return g, u

            # Prime the pipeline: stage the first ``depth-1`` cells so that when
            # the loop blocks on cell ni there are already ``depth-1`` further
            # cells in flight behind it. depth=2 reproduces the original
            # single-cell prime exactly.
            depth = max(2, int(dtla.get("depth", 2)))
            for _j in range(min(depth - 1, nni)):
                _stage(_j, _j % depth)
            # DMAs per _stage call: gate + up, each ceil(b_per_lane/16) chunks.
            chunks_per_frag = (atom.b_per_lane + DTLA_CHUNK - 1) // DTLA_CHUNK
            dmas_per_stage = 2 * chunks_per_frag
            for ni in range(nni):
                pair = ni % depth
                # Issue cell ni+depth-1's DMA BEFORE consuming this cell, so
                # ``depth-1`` cells stay in flight across the wait. Drain only
                # DOWN TO those outstanding DMAs -- vmcnt(0) here would
                # serialize and kill the overlap (the DTLA-alone regression
                # trap). VMEM completes ~FIFO, so cell ni has landed once
                # only the cells behind it remain.
                nxt = ni + depth - 1
                if nxt < nni:
                    _stage(nxt, nxt % depth)
                ahead = min(depth - 1, nni - 1 - ni)
                b.s_waitcnt(vmcnt=ahead * dmas_per_stage)
                gb, ub = _read(pair)
                g_acc[ni] = atom.emit(b, a_cur, gb, g_acc[ni])
                u_acc[ni] = atom.emit(b, a_cur, ub, u_acc[ni])
                _emit_sgb_gateup_dtla(b, 2, cadence)
        elif atoms_per_group == 1:
            # ---- windowed global->VGPR->MFMA path (no LDS) ------------------
            # The hero atom leaves atoms_per_group == 1, so the loop below
            # degenerates to a single trip and the legacy form ends up issuing
            # ALL 2*nni weight fragments before the first MFMA -- 128 VGPRs of
            # weights at nni=8, which is what makes the backend's allocator
            # blow up. Instead roll a window of ``prefetch_depth`` cells: cell
            # ni+depth-1's loads are issued before cell ni's MFMAs consume
            # theirs, so at most ``depth`` gate/up pairs are live at once and
            # the loads stay in flight under the MFMAs.
            #
            # Unlike DTLA there is no manual vmcnt bookkeeping: the fragments
            # land in VGPRs, so the backend derives an exact per-register wait
            # and only blocks on the fragment actually being consumed.
            depth = max(2, int(prefetch_depth))
            window_group = max(1, int(sched_group))
            sched_mode = str(sched_mode)
            a_cur = _a_at(0)
            win: dict[int, tuple] = {}
            for _j in range(min(depth, nni)):
                win[_j] = (_gb_at(_j, 0), _ub_at(_j, 0))
            for ni in range(nni):
                nxt = ni + depth
                if nxt < nni:
                    win[nxt] = (_gb_at(nxt, 0), _ub_at(nxt, 0))
                gb, ub = win.pop(ni)
                g_acc[ni] = atom.emit(b, a_cur, gb, g_acc[ni])
                u_acc[ni] = atom.emit(b, a_cur, ub, u_acc[ni])
                # Pin the [issue ni+depth-1][consume ni] grouping. Without this
                # the pre-RA scheduler hoists every cell's loads to the top of
                # the block to maximise ILP, which recreates the all-cells-live
                # register pressure the window exists to bound. The barrier only
                # constrains instruction motion, so the issued loads still stay
                # in flight across it.
                if (ni + 1) % window_group == 0 or ni + 1 == nni:
                    if sched_mode == "sgb":
                        # 2 chunks x (gate, up) VMEM reads and 2 MFMAs per cell.
                        b.sched_group_barrier(_SGB_VMEM_READ, 4 * window_group, 0)
                        b.sched_group_barrier(_SGB_MFMA, 2 * window_group, 0)
                    else:
                        b.sched_barrier(0)
        else:
            # ---- legacy global->VGPR->MFMA path ----------------------------
            # Prefetch atom 0 operands (A shared + all ni B fragments).
            a_cur = _a_at(0)
            gb_cur = [_gb_at(ni, 0) for ni in range(nni)]
            ub_cur = [_ub_at(ni, 0) for ni in range(nni)]

            for kk in range(atoms_per_group):
                last = kk + 1 >= atoms_per_group
                # Issue the NEXT atom's shared A load up front (in flight while
                # the current atom's MFMAs run).
                if not last:
                    a_next = _a_at(kk + 1)
                # Wave-pair interleave: for each ni, emit gate then up, and
                # splice the next-atom B prefetch for ni before ni's MFMAs.
                for ni in range(nni):
                    if not last:
                        gb_next_ni = _gb_at(ni, kk + 1)
                        ub_next_ni = _ub_at(ni, kk + 1)
                    g_acc[ni] = atom.emit(b, a_cur, gb_cur[ni], g_acc[ni])
                    u_acc[ni] = atom.emit(b, a_cur, ub_cur[ni], u_acc[ni])
                    if not last:
                        gb_cur[ni] = gb_next_ni
                        ub_cur[ni] = ub_next_ni
                if not last:
                    a_cur = a_next

        # Fold each group accumulator (post-MFMA) by a_scale * b_scale.
        new_gate = []
        new_up = []
        for ni in range(nni):
            gvec = b.vector_splat(gate_ab[ni], atom.c_per_lane)
            uvec = b.vector_splat(up_ab[ni], atom.c_per_lane)
            new_gate.append(b.vector_fma(g_acc[ni], gvec, gate_outer[ni]))
            new_up.append(b.vector_fma(u_acc[ni], uvec, up_outer[ni]))
        b.scf_yield(*(new_gate + new_up))

    res = outer.results
    return list(res[:nni]), list(res[nni:])


def _global_load_fp8_vec(b: IRBuilder, ptr: Value, addr: Value, n: int) -> Value:
    """Coalesced fp8 vector load of ``n`` contiguous bytes at ``addr``.

    ``global_load_vN`` for fp8 caps at n=16 (ds/global payload width); the
    K=128 hero atom needs ``a_per_lane`` == ``b_per_lane`` == 32 fp8 per lane.
    Split into ceil(n/16) consecutive 16-wide loads and concat -- the bytes are
    contiguous so this is two ``global_load_dwordx4`` over the same 32-byte run,
    bit-identical to a single 32-wide load.
    """
    if n <= 16:
        return b.global_load_vN(ptr, addr, FP8E4M3, n)
    acc = None
    off = 0
    while off < n:
        chunk = min(16, n - off)
        a = addr if off == 0 else b.add(addr, b.const_i32(off))
        v = b.global_load_vN(ptr, a, FP8E4M3, chunk)
        acc = v if acc is None else b.vec_concat(acc, v)
        off += chunk
    return acc


def _load_a_fp8(
    b: IRBuilder, *, A, atom, lane_decode, m_tile_base, k_tile_base, K
) -> Value:
    """Per-lane fp8 A load for row-major (M, K) -- K contiguous; scalar loads."""
    m_row = b.add(m_tile_base, lane_decode.m_in_atom)
    k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.a_per_lane))
    k_base = b.add(k_tile_base, k_lane_start)
    a_addr = b.add(b.mul(m_row, K), k_base)
    # The a_per_lane fp8 bytes are CONTIGUOUS along K (addr + j) -> one (or, for
    # the K=128 atom's 32-wide fragment, two concatenated) coalesced vector
    # load(s) instead of a_per_lane byte-granular global_load_ubyte.
    # Bit-identical values.
    return _global_load_fp8_vec(b, A, a_addr, atom.a_per_lane)


# 16-byte VMEM payload cap (== global_load_dwordx4 / global_load_lds_dwordx4
# on gfx950). Both the plain B load and the direct-to-LDS DMA split their
# per-lane fragment into chunks of this width.
VMEM_CHUNK_BYTES = 16


# ---------------------------------------------------------------------------
# B (weight) FRAGMENT SWIZZLE
# ---------------------------------------------------------------------------
#
# In the natural (out_rows, K) row-major layout the 64 lanes of one B fragment
# read ``atom.n`` (=16) DISJOINT runs of ``atom.k`` (=128) bytes, one per output
# row, separated by the full row stride K (2048B for gate/up). Each 16-byte
# chunk instruction therefore touches 16 distinct cache lines instead of the 8
# a fully coalesced 64x16B access would, and both chunks of the fragment touch
# the same 16 lines -- 32 line-touches per fragment against a possible 16. The
# kernel is memory bound on the L1 (TCP) tag pipe rather than on HBM bytes, so
# that 2x in tag-pipe work is paid in full.
#
# The fix is to store the weights in the order the lanes consume them. A
# swizzle tile is exactly one wave's fragment -- ``atom.n`` rows x ``atom.k``
# bytes == ``wave_size * atom.b_per_lane`` bytes -- laid out so that chunk ``c``
# of lane ``L`` lives at tile offset ``(c*wave_size + L) * 16``. Then a chunk
# instruction reads ``wave_size * 16`` == 1024 CONTIGUOUS bytes (8 cache lines)
# and the whole fragment is one contiguous 2048-byte run.
#
# Tiles are ordered row-block-major, ``tile_idx = (n/atom.n) * (K/atom.k) +
# (k/atom.k)``, which keeps the K-loop's consecutive iterations adjacent in
# memory. The resulting base address is WAVE-UNIFORM::
#
#     tile_base = n_tile_base * K + k_tile_base * atom.n
#
# so the only lane-varying term left is ``lane * 16``.
#
# Nothing outside this kernel consumes the weight buffers, so the layout is
# free to change; :func:`swizzle_b_fp8_weights` is the host-side permutation
# that produces it and is the single source of truth paired with
# :func:`_load_b_fp8_swizzled` below.


def b_swizzle_tile_bytes(atom: MfmaAtom) -> int:
    """Bytes in one swizzle tile (== one wave's B fragment)."""
    return atom.n * atom.k


def b_swizzle_supported(atom: MfmaAtom, wave_size: int = 64) -> bool:
    """Whether ``atom``'s B fragment maps onto the swizzle tiling.

    Requires the fragment to be an exact wave-sized cover of the tile and the
    per-lane fragment to be a whole number of 16-byte chunks. True for the
    fp8 16x16x128 hero atom (16*128 == 64*32, 32 == 2*16); false for the
    legacy 16x16x32 atom, whose 8-byte fragment is narrower than a chunk.
    """
    return (
        atom.b_per_lane % VMEM_CHUNK_BYTES == 0
        and atom.n * atom.k == wave_size * atom.b_per_lane
    )


def swizzle_b_fp8_weights(w, *, atom_n: int = 16, atom_k: int = 128, wave_size: int = 64):
    """Permute a stack of fp8 weights into the layout described above.

    ``w`` is ``(E, out_rows, K)`` uint8/fp8 with ``out_rows % atom_n == 0`` and
    ``K % atom_k == 0``. Returns a C-contiguous array of the SAME shape and
    size -- only the byte order within each expert changes, so every stride the
    kernel signature carries stays valid.

    The inverse of the kernel's address math: byte ``j`` of chunk ``c`` of lane
    ``L`` in the tile at ``(n_tile, k_tile)`` must hold
    ``w[e, n_tile*atom_n + L % atom_n, k_tile*atom_k + (L // atom_n)*per_lane
    + c*16 + j]``.
    """
    import numpy as np

    e, rows, k = w.shape
    if rows % atom_n or k % atom_k:
        raise ValueError(
            f"weight {w.shape} not tileable by ({atom_n}, {atom_k}) for the B swizzle"
        )
    per_lane = atom_n * atom_k // wave_size
    k_blks = wave_size // atom_n  # distinct k_blk values across the wave
    chunks = per_lane // VMEM_CHUNK_BYTES
    if per_lane % VMEM_CHUNK_BYTES or k_blks * per_lane != atom_k:
        raise ValueError(f"atom ({atom_n}, {atom_k}) does not map onto a {wave_size}-lane swizzle tile")

    t = w.reshape(e, rows // atom_n, atom_n, k // atom_k, k_blks, chunks, VMEM_CHUNK_BYTES)
    #        axes: 0=E  1=n_tile          2=row  3=k_tile      4=k_blk  5=chunk  6=byte
    # Tile-internal order must be (chunk, k_blk, row, byte) so that position
    # ``c*wave_size + (k_blk*atom_n + row)`` is at offset ``pos*16``.
    t = t.transpose(0, 1, 3, 5, 4, 2, 6)
    return np.ascontiguousarray(t).reshape(e, rows, k)


def _load_b_fp8_swizzled(
    b: IRBuilder, *, B, atom, lane_decode, n_tile_base, k_tile_base, N, wave_size: int
) -> Value:
    """Fully coalesced per-lane fp8 B load from the swizzled weight layout.

    Emits the same ``b_per_lane // 16`` ``global_load_dwordx4`` as the row-major
    form and returns a bit-identical fragment, but each one reads
    ``wave_size * 16`` contiguous bytes instead of ``atom.n`` scattered runs.
    """
    tile_base = b.add(b.mul(n_tile_base, N), b.mul(k_tile_base, b.const_i32(atom.n)))
    base = b.add(tile_base, b.mul(lane_decode.lane, b.const_i32(VMEM_CHUNK_BYTES)))
    block = wave_size * VMEM_CHUNK_BYTES
    acc = None
    for c in range(atom.b_per_lane // VMEM_CHUNK_BYTES):
        addr = base if c == 0 else b.add(base, b.const_i32(c * block))
        v = b.global_load_vN(B, addr, FP8E4M3, VMEM_CHUNK_BYTES)
        acc = v if acc is None else b.vec_concat(acc, v)
    return acc


def _load_b_fp8(
    b: IRBuilder,
    *,
    B,
    atom,
    lane_decode,
    n_tile_base,
    k_tile_base,
    N,
    swizzle: bool,
    wave_size: int,
) -> Value:
    """Per-lane fp8 B load for row-major (K, N) -- col-strided scalar loads.

    ``N`` is the weight's K extent (the matrix is stored (out, K) row-major;
    B[k, n] address = (n_tile_base + n_in_atom) * K + (k_base + j)).

    With ``swizzle`` the weights are instead in the tiled layout produced by
    :func:`swizzle_b_fp8_weights` and the load becomes fully coalesced.
    """
    if swizzle:
        return _load_b_fp8_swizzled(
            b,
            B=B,
            atom=atom,
            lane_decode=lane_decode,
            n_tile_base=n_tile_base,
            k_tile_base=k_tile_base,
            N=N,
            wave_size=wave_size,
        )
    n_col = b.add(n_tile_base, lane_decode.n_in_atom)
    k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.b_per_lane))
    k_base = b.add(k_tile_base, k_lane_start)
    # W is stored (out_rows, K) row-major -> address of W[n_col, k] = n_col*K + k.
    row_base = b.mul(n_col, N)
    # The b_per_lane fp8 weight bytes are CONTIGUOUS along K (k_base + j) ->
    # one (or, for the K=128 atom's 32-wide fragment, two concatenated)
    # coalesced vector load(s) instead of b_per_lane byte-granular
    # global_load_ubyte. This is the dominant HBM weight traffic; bit-identical
    # values.
    b_addr = b.add(row_base, k_base)
    return _global_load_fp8_vec(b, B, b_addr, atom.b_per_lane)


# ---------------------------------------------------------------------------
# DIRECT-TO-LDS (DTLA) B-operand staging for the gate+up GEMM
# ---------------------------------------------------------------------------
#
# GOAL 1: replace the global->VGPR->MFMA weight load of the dominant gate/up
# weight stream with a global->LDS->MFMA path via the additive
# ``b.global_load_lds`` op (== ``global_load_lds_dwordx4`` ISA, the flat sibling
# of pyisa's ``buffer_load_dwordx4 ... offen lds``). The DMA bypasses the VGPR
# round-trip.
#
# LANE DISTRIBUTION (the correctness model): ``llvm.amdgcn.global.load.lds``
# takes a WAVE-UNIFORM LDS destination; the HARDWARE spreads the 64 lanes'
# payloads lane-contiguously -- lane L writes its ``size_bytes`` to
# ``lds_dst + L*size_bytes``. So the destination passed to the intrinsic must NOT
# carry a per-lane term (the lane term lives in the per-lane SOURCE pointer). A
# 32-byte fp8 fragment exceeds the 16-byte payload cap, so it is two 16-byte DMA
# chunks landing in two SEPARATE 64*16-byte lane-blocks: chunk c at
# ``slot_base + c*(wave_size*16)``, each lane's source at ``src + c*16``. The
# read-back gathers both half-blocks (rows (slot*chunks + c)*wave_size + L).
#
# CRITICAL coupling (DTLA regresses ALONE): the DMA for the NEXT ni cell is
# issued BEFORE the current cell's MFMAs, so it is in flight under the MFMA. The
# LDS slot is PER-WAVE (base biased by wave_id*wave_bytes) and PING-PONG
# double-buffered over ni so the next DMA does not stomp the fragment the current
# MFMA is still reading. CACHE_ALL keeps the reused weights resident in L2.

# 16-byte direct-to-LDS payload cap (== global_load_lds_dwordx4 on gfx950).
DTLA_CHUNK = VMEM_CHUNK_BYTES


def _dtla_stage_b_fp8(
    b: IRBuilder,
    *,
    B: Value,
    atom: MfmaAtom,
    lane_decode,
    n_tile_base: Value,
    k_tile_base: Value,
    N: Value,
    stage_view,
    slot: int,
    wave_lds_base: Value,
    lane: Value,
    wave_size: int,
    swizzle: bool = False,
) -> None:
    """Issue the direct-to-LDS DMA of one lane's ``b_per_lane`` fp8 weight bytes.

    Lane ``L`` streams the SAME ``b_per_lane`` contiguous-along-K fp8 weight bytes
    that :func:`_load_b_fp8` would have loaded into a VGPR, but the HARDWARE
    distributes them: chunk ``c`` lands lane-contiguously at
    ``slot_base + c*wave_size*16 + L*16`` (the destination is wave-uniform; the
    per-lane spread is implicit). The DMA completes on the VMEM counter; the
    caller drains ``s_waitcnt(vmcnt=0)`` before the read-back.

    Under ``swizzle`` the source is the tiled layout of
    :func:`swizzle_b_fp8_weights`, whose tile-internal order was chosen to be
    exactly this lane-block spread -- so chunk ``c``'s 64 source addresses are
    the ``wave_size*16`` contiguous bytes at ``tile_base + c*wave_size*16``, and
    the bytes landing in LDS (hence :func:`_dtla_read_b_fp8`) are unchanged.
    """
    frag_bytes = atom.b_per_lane  # fp8 == 1 byte/elem
    chunks = (frag_bytes + DTLA_CHUNK - 1) // DTLA_CHUNK
    block_bytes = wave_size * DTLA_CHUNK  # one DMA's lane-block footprint
    slot_base_off = slot * chunks * block_bytes

    if swizzle:
        tile_base = b.add(
            b.mul(n_tile_base, N), b.mul(k_tile_base, b.const_i32(atom.n))
        )
        src_elem = b.add(tile_base, b.mul(lane_decode.lane, b.const_i32(DTLA_CHUNK)))
        src_stride = block_bytes
    else:
        n_col = b.add(n_tile_base, lane_decode.n_in_atom)
        k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.b_per_lane))
        k_base = b.add(k_tile_base, k_lane_start)
        row_base = b.mul(n_col, N)
        src_elem = b.add(row_base, k_base)  # per-lane element (== byte) source offset
        src_stride = DTLA_CHUNK

    for c in range(chunks):
        chunk = min(DTLA_CHUNK, frag_bytes - c * DTLA_CHUNK)
        src = src_elem if c == 0 else b.add(src_elem, b.const_i32(c * src_stride))
        # WAVE-UNIFORM destination for chunk c (no per-lane term -- HW spreads).
        dst = b.smem_ptr_add(
            wave_lds_base, b.const_i64(slot_base_off + c * block_bytes)
        )
        b.global_load_lds(B, src, dst, chunk, CACHE_ALL)


def _dtla_read_b_fp8(
    b: IRBuilder,
    *,
    atom: MfmaAtom,
    stage_view,
    slot: int,
    lane: Value,
    warp_row_base: Value,
    wave_size: int,
) -> Value:
    """Read back a lane's fp8 B fragment staged in LDS by :func:`_dtla_stage_b_fp8`.

    Mirrors the staging lane-block layout: chunk ``c`` of lane ``L`` lives at row
    ``warp_row_base + (slot*chunks + c)*wave_size + L`` (the 2-D view has a
    16-byte column). Concatenate the per-chunk 16-wide ``ds_read_b128`` reads to
    re-form the full ``b_per_lane`` fragment -- bit-identical to the VGPR load.
    """
    frag = atom.b_per_lane
    chunks = (frag + DTLA_CHUNK - 1) // DTLA_CHUNK
    acc = None
    for c in range(chunks):
        chunk = min(DTLA_CHUNK, frag - c * DTLA_CHUNK)
        row = b.add(
            warp_row_base,
            b.add(b.const_i32((slot * chunks + c) * wave_size), lane),
        )
        v = b.smem_load_vN(stage_view.base, row, b.const_i32(0), dtype=FP8E4M3, n=chunk)
        acc = v if acc is None else b.vec_concat(acc, v)
    return acc


# ---------------------------------------------------------------------------
# STAGE 2: down fp8 GEMM (LDS-A) + per-128-group dequant + weighted atomic Y
# ---------------------------------------------------------------------------


def _load_a_fp8_lds(
    b: IRBuilder, *, a_view, atom, lane_decode, m_tile_base, k_tile_base
) -> Value:
    """Per-lane fp8 A load for the down GEMM, reading the LDS-resident Hidden.

    The down-GEMM A operand is ``Hidden_smem`` (``[tile_m, tile_n_inter]``,
    logical ``(row=m, col=inter)``). The contraction axis is the inter slice, so
    for output row ``m_row`` (= ``m_tile_base + m_in_atom``) the lane reads its
    ``a_per_lane`` contiguous fp8 elements starting at inter column
    ``k_tile_base + k_blk * a_per_lane``. This is the SAME logical ``(m, inter)``
    cell the dynamic-quant Pass C wrote (BUILD_SPEC_FP8 Section 3.5: the implicit
    reshape -- cshuffle/quant write addr == down MFMA A-read addr).
    """
    m_row = b.add(m_tile_base, lane_decode.m_in_atom)
    k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.a_per_lane))
    k_col = b.add(k_tile_base, k_lane_start)
    # smem_load_vN caps fp8 at n=16; the K=128 hero atom needs a 32-wide
    # fragment. Split into ceil(n/16) contiguous 16-wide ds_read_b128 loads and
    # concat -- the inter columns are contiguous in LDS so this is bit-identical
    # to a single 32-wide read.
    n = atom.a_per_lane
    if n <= 16:
        return b.smem_load_vN(a_view.base, m_row, k_col, dtype=FP8E4M3, n=n)
    acc = None
    off = 0
    while off < n:
        chunk = min(16, n - off)
        c = k_col if off == 0 else b.add(k_col, b.const_i32(off))
        v = b.smem_load_vN(a_view.base, m_row, c, dtype=FP8E4M3, n=chunk)
        acc = v if acc is None else b.vec_concat(acc, v)
        off += chunk
    return acc


def _emit_fp8_down_group_gemm(
    b: IRBuilder,
    *,
    a_view,
    WDown: Value,
    WDownScale: Value,
    atom: MfmaAtom,
    lane_decode,
    n_tile_base: Value,
    scale_view,
    inter_slice: int,
    inter_full: Value,
    inter_blk_base: Value,
    stride_down_scale: Value,
    m_row_base: Value,
    tag: str,
    cadence: str,
    swizzle_b: bool,
    wave_size: int,
) -> Value:
    """Down fp8 GEMM for one warp-atom output cell -> per-lane f32 vector.

    Group-accumulator pattern (BUILD_SPEC_FP8 Section 1.2 / 2.2): the outer loop
    walks 128-wide groups along the inter CONTRACTION (= this TG's inter slice,
    ``tile_n_inter`` = ``inter_slice``). Per group, 4 fp8_16x16x32 atoms
    accumulate into a FRESH ``group_acc``; the group is folded into the outer
    accumulator scaled by ``hidden_dyn_scale * down_b_scale`` -- a single
    ``v_pk_fma_f32``.

    Index conventions (the two-space split):
    * **A (Hidden, LDS):** read at LOCAL inter columns ``[0, inter_slice)``; its
      per-128-block dynamic scale is ``HiddenScale_smem[m_row, local_blk]`` (the
      dynamic scale is only defined over THIS TG's slice).
    * **B (W_down, global):** stored ``(H_out, I_full)`` row-major; the
      contraction column is the GLOBAL inter position ``inter_blk_base*128 + local``
      and the row stride is the FULL inter dim ``inter_full``. The B-side scale
      ``WDownScale`` is indexed per (GLOBAL inter-block, H_out-block):
      ``off = (inter_blk_base + kg) * stride_down_scale + h_out_blk``.

    The W_down per-expert pointer base is folded by the caller; the inter slice
    base is folded here (column ``inter_blk_base*128 + local``), NOT into the
    pointer, so the row stride math stays the full-inter stride.
    """
    c_group_k = b.const_i32(GROUP_K)
    c_atom_k = b.const_i32(atom.k)
    atoms_per_group = GROUP_K // atom.k  # 4

    n_col = b.add(n_tile_base, lane_decode.n_in_atom)
    # Wave-uniform scale block (see the gate/up emitter): atom.n divides
    # GROUP_K, so this equals n_col // GROUP_K but stays scalarisable.
    h_out_blk = b.div(n_tile_base, c_group_k)
    # CORRECTNESS FIX: the down-GEMM A operand row must follow the per-mi atom
    # m-base (m_row_base = down_warp_m_off + mi*atom.m), not a hardcoded 0.
    # HiddenScale is row-uniform-within-block so this term is harmless for the
    # scale read, but threading it keeps the A-read (below) and scale-read on
    # the same row basis. mfmas_m_down=1 (tile_m=16) leaves it == 0; tile_m=32
    # makes mi=1 read Hidden LDS rows 16-31 instead of 0-15.
    m_row = b.add(m_row_base, lane_decode.m_in_atom)

    # Global inter column base for this TG's slice (W_down contraction origin).
    inter_col_base = b.mul(inter_blk_base, c_group_k)

    zero = atom.zero_acc(b)
    outer_zero = atom.zero_acc(b)

    # num_groups = inter_slice // GROUP_K (local inter slice / 128).
    num_groups = b.const_i32(inter_slice // GROUP_K)
    outer = b.scf_for_iter(
        b.const_i32(0),
        num_groups,
        b.const_i32(1),
        [(f"down_outer_{tag}", outer_zero)],
        iv_name=f"dg_{tag}",
    )
    with outer as (kg, (down_outer,)):
        _emit_loop_cadence_hint(b, cadence)
        # A-side dynamic scale: HiddenScale_smem[m_row, local-inter-block kg].
        a_scale_v = b.vec_extract(
            b.smem_load_vN(scale_view.base, m_row, kg, dtype=F32, n=1), 0
        )
        # B-side W_down scale: per (GLOBAL inter-block, H_out-block).
        global_blk = b.add(inter_blk_base, kg)
        down_scale_off = b.add(b.mul(global_blk, stride_down_scale), h_out_blk)
        down_scale_v = b.global_load_f32(WDownScale, down_scale_off)
        ab_scale = b.fmul(a_scale_v, down_scale_v)

        local_k_group = b.mul(kg, c_group_k)
        global_k_group = b.add(inter_col_base, local_k_group)

        # L5: software-pipeline the W_down (B) tile -- the dominant HBM stream of
        # this stage. The inner k-group loop has a COMPILE-TIME-CONSTANT trip
        # count (``atoms_per_group`` == 4 fp8_16x16x32 atoms), so it is unrolled
        # in Python and the B=W_down fragments are register double-buffered:
        # the next atom's global B load is ISSUED before the current b_frag is
        # consumed in ``atom.emit``, keeping a load in flight under the MFMA.
        # (An scf.for iter-arg cannot carry the fp8e4m3 fragment -- the LLVM
        # lowering has no loop-carried fp8 vector type -- so the prefetch is
        # expressed by Python unroll instead, same dataflow as the f16
        # ``_emit_moe_down_kloop_lds_a`` b_next pattern. A stays a direct LDS
        # read: cheap, already resident.)
        def _load_b_at(kk_idx):
            return _load_b_fp8(
                b,
                B=WDown,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=b.add(global_k_group, b.mul(b.const_i32(kk_idx), c_atom_k)),
                N=inter_full,
                swizzle=swizzle_b,
                wave_size=wave_size,
            )

        def _load_a_at(kk_idx):
            return _load_a_fp8_lds(
                b,
                a_view=a_view,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_row_base,
                k_tile_base=b.add(local_k_group, b.mul(b.const_i32(kk_idx), c_atom_k)),
            )

        # Prefetch the first atom's W_down fragment, then pipeline.
        b_cur = _load_b_at(0)
        group_acc = zero
        for kk in range(atoms_per_group):
            a_frag = _load_a_at(kk)
            # Issue the NEXT atom's W_down load (in flight during the MFMA).
            if kk + 1 < atoms_per_group:
                b_next = _load_b_at(kk + 1)
            d_new = atom.emit(b, a_frag, b_cur, group_acc)
            group_acc = d_new
            if kk + 1 < atoms_per_group:
                b_cur = b_next

        # sgb cadence: place the next-group W_down VMEM under this group's MFMA(s).
        _emit_sgb_down_group(b, atoms_per_group, cadence)

        scale_vec = b.vector_splat(ab_scale, atom.c_per_lane)
        down_outer_new = b.vector_fma(group_acc, scale_vec, down_outer)
        b.scf_yield(down_outer_new)

    return outer.results[0]


def _emit_fp8_down_fused_cells(
    b: IRBuilder,
    *,
    a_view,
    WDown: Value,
    WDownScale: Value,
    atom: MfmaAtom,
    lane_decode,
    n_tile_bases,
    scale_view,
    inter_slice: int,
    inter_full: Value,
    inter_col_base: Value,
    stride_down_scale: Value,
    m_row_base: Value,
    tag: str,
    cadence: str,
    prefetch_depth: int,
    sched_group: int,
    sched_mode: str,
    hidden_group_k: int,
    swizzle_b: bool,
    wave_size: int,
    pipeline_k: bool = False,
    scale_row: Value | None = None,
):
    """Down fp8 GEMM fused across ALL n cells of one mi row.

    Same dataflow and index conventions as ``_emit_fp8_down_group_gemm``, but
    every output cell of the mi row shares ONE K-loop. Two wins over the
    per-cell form:

    * The A (Hidden, LDS) fragment depends only on ``m_row_base`` and k, NOT on
      the cell, so it is read from LDS ONCE and reused across cells instead of
      being re-read ``nni`` times -- and that read is the 16-way bank-conflict
      one.
    * The W_down fragments are prefetched in a rolling ``prefetch_depth`` window
      across cells. At ``down_k=128`` the per-cell form has
      ``atoms_per_group == 1``, so its register double-buffer never engages and
      each cell loads then immediately consumes -- no load is ever in flight
      under an MFMA. This is the same degeneracy the gate/up path had.

    ``pipeline_k`` adds a second, outer software pipeline. The window above is
    over the N axis and so can never be deeper than ``nni``; in the split-down
    kernel ``nni`` is 2, which is why ``prefetch_depth`` 2/4/8 all compile to
    identical ISA there and why that kernel shows only 4 loads in flight, half
    its waits full ``vmcnt(0)`` drains, and 71 wait-cycles per memory
    instruction against the gate/up kernel's 18. With the flag on, iteration
    ``kg`` issues the loads for ``kg+1`` before consuming the fragments it
    carried in, so a K group's loads are in flight underneath the previous
    group's MFMAs. The fragments ride the loop as ``iter_args``, which is what
    pushes the ``s_waitcnt`` out of the load's own iteration.

    Returns one f32 outer accumulator per entry of ``n_tile_bases``.
    """
    c_group_k = b.const_i32(GROUP_K)
    c_atom_k = b.const_i32(atom.k)
    # The contraction walks HIDDEN scale blocks (the A-side granularity). The
    # W_down scale stays on the checkpoint's 128-wide grid and is re-derived
    # from the absolute inter column below, so a 64-wide hidden block simply
    # reads the same b-scale for its two halves.
    hgk = int(hidden_group_k)
    c_hgk = b.const_i32(hgk)
    atoms_per_group = hgk // atom.k
    nni = len(n_tile_bases)
    depth = max(2, int(prefetch_depth))
    group = max(1, int(sched_group))

    n_cols = [b.add(nb, lane_decode.n_in_atom) for nb in n_tile_bases]
    # Wave-uniform scale block (see the gate/up emitter). n_cols stays
    # lane-varying -- it addresses the actual W_down data, not the scale.
    # Same per-cell-to-per-warp dedup as gate/up: when the warp's output extent
    # divides GROUP_K its base cannot straddle a block, so all nni cells read
    # one scale and the backend needs one s_load chain instead of nni.
    h_out_blks = (
        [b.div(n_tile_bases[0], c_group_k)] * nni
        if GROUP_K % (nni * atom.n) == 0
        else [b.div(nb, c_group_k) for nb in n_tile_bases]
    )
    m_row = b.add(m_row_base, lane_decode.m_in_atom)

    iter_args = [(f"down_outer_{tag}_{ni}", atom.zero_acc(b)) for ni in range(nni)]
    # ``inter_slice`` is a compile-time int for the fused kernel (it is a tile
    # width) but a runtime Value for the partial-fusion down kernel, whose
    # contraction extent is the whole intermediate dimension.
    num_groups = (
        b.const_i32(inter_slice // hgk)
        if isinstance(inter_slice, int)
        else b.div(inter_slice, c_hgk)
    )

    # Only the atoms_per_group == 1 shape is pipelined: with more than one atom
    # per group the inner kk loop already has somewhere to hide the latency,
    # and carrying a fragment per (ni, kk) would cost more registers than the
    # stall it removes.
    pipe = bool(pipeline_k) and atoms_per_group == 1

    def _b_frags_for_group(k_group):
        return [
            _load_b_fp8(
                b,
                B=WDown,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_bases[ni],
                k_tile_base=k_group,
                N=inter_full,
                swizzle=swizzle_b,
                wave_size=wave_size,
            )
            for ni in range(nni)
        ]

    if pipe:
        # Prologue: the kg=0 fragments enter the loop already in flight.
        for ni, frag in enumerate(_b_frags_for_group(inter_col_base)):
            iter_args.append((f"down_bpf_{tag}_{ni}", frag))
        c_last_group = b.sub(num_groups, b.const_i32(1))

    outer = b.scf_for_iter(
        b.const_i32(0), num_groups, b.const_i32(1), iter_args, iv_name=f"dg_{tag}"
    )
    with outer as (kg, outs):
        # As in the gate/up window: iglp_opt owns the loop schedule and would
        # hoist every cell's loads to the top, undoing the window.
        _emit_loop_cadence_hint(b, "none")
        down_outer = list(outs[:nni])
        carried_b = list(outs[nni:]) if pipe else []

        # ``scale_row`` pins the A-scale read to a single LDS row. The scale is
        # row-uniform by construction, so a [1, n_blocks] scale_view carries the
        # same information as a [tile_m, n_blocks] one -- and saves the writer
        # from broadcasting each value down tile_m rows, which is where 56 of
        # the split-down kernel's 60 DS instructions were going.
        local_k_group = b.mul(kg, c_hgk)
        global_k_group = b.add(inter_col_base, local_k_group)
        a_scale_v = b.vec_extract(
            b.smem_load_vN(
                scale_view.base,
                m_row if scale_row is None else scale_row,
                kg,
                dtype=F32,
                n=1,
            ),
            0,
        )
        global_blk = b.div(global_k_group, c_group_k)
        base_scale_off = b.mul(global_blk, stride_down_scale)
        ab_scales = [
            b.fmul(
                a_scale_v,
                b.global_load_f32(WDownScale, b.add(base_scale_off, h_out_blks[ni])),
            )
            for ni in range(nni)
        ]

        def _load_b_at(ni, kk):
            return _load_b_fp8(
                b,
                B=WDown,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_bases[ni],
                k_tile_base=b.add(global_k_group, b.mul(b.const_i32(kk), c_atom_k)),
                N=inter_full,
                swizzle=swizzle_b,
                wave_size=wave_size,
            )

        def _load_a_at(kk):
            return _load_a_fp8_lds(
                b,
                a_view=a_view,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_row_base,
                k_tile_base=b.add(local_k_group, b.mul(b.const_i32(kk), c_atom_k)),
            )

        g_acc = [atom.zero_acc(b)] * nni
        next_b: list[Value] = []
        if pipe:
            # Next group's loads go out BEFORE this group's MFMAs, and nothing
            # in this iteration reads them, so the compiler has no reason to
            # emit a wait for them until the next iteration consumes them.
            # Clamping instead of predicating keeps the last iteration's
            # (discarded) prefetch inside the tensor.
            kg_next = b.smin(b.add(kg, b.const_i32(1)), c_last_group)
            next_b = _b_frags_for_group(
                b.add(inter_col_base, b.mul(kg_next, c_hgk))
            )

        for kk in range(atoms_per_group):
            a_frag = _load_a_at(kk)
            win: dict[int, Value] = {}
            if pipe:
                win = {ni: carried_b[ni] for ni in range(nni)}
            else:
                for _j in range(min(depth, nni)):
                    win[_j] = _load_b_at(_j, kk)
            for ni in range(nni):
                nxt = ni + depth
                if not pipe and nxt < nni:
                    win[nxt] = _load_b_at(nxt, kk)
                g_acc[ni] = atom.emit(b, a_frag, win.pop(ni), g_acc[ni])
                if (ni + 1) % group == 0 or ni + 1 == nni:
                    if sched_mode == "sgb":
                        # 2 chunks of W_down VMEM and 1 MFMA per cell.
                        b.sched_group_barrier(_SGB_VMEM_READ, 2 * group, 0)
                        b.sched_group_barrier(_SGB_MFMA, group, 0)
                    else:
                        b.sched_barrier(0)

        new_outer = []
        for ni in range(nni):
            sv = b.vector_splat(ab_scales[ni], atom.c_per_lane)
            new_outer.append(b.vector_fma(g_acc[ni], sv, down_outer[ni]))
        b.scf_yield(*new_outer, *next_b)

    return list(outer.results)


def _emit_down_atomic_reduce(
    b: IRBuilder,
    *,
    atom: MfmaAtom,
    down_list,
    warp_m_off: Value,
    warp_n_off: Value,
    lane: Value,
    mfmas_m: int,
    mfmas_n: int,
    block_m_off: Value,
    ho_off: Value,
    H_out: Value,
    SortedTokenIds: Value,
    SortedWeights: Value,
    Y: Value,
    tokens: Value,
) -> None:
    """Weighted, token-validity-masked atomic reduce of the down result into Y.

    ``Y[token, h_out] += weight * down_dq`` (f32 atomic add). Padded rows
    (sorted token id < 0 or >= tokens) are skipped. Mirrors the f16
    ``_emit_down_reduce_epilogue_atomic`` but on the raw-atom lane layout used by
    the rest of this fp8 file (``atom.lane_to_output``). The sorted-token bucket
    index is the GLOBAL output row ``block_m_off + warp/atom row``; the output
    column is ``ho_off + warp/atom col`` along H_out.
    """
    c0 = b.const_i32(0)
    # L1: the sorted-token id and routing weight are per-ROW (bucket == row); they
    # do NOT depend on the output column (ni / col_in). The pre-L1 nest reloaded
    # SortedTokenIds + SortedWeights and re-checked validity for EVERY (mi, ni, i)
    # output slot, each `global_load_f32(SortedWeights) -> s_waitcnt vmcnt(0) ->
    # v_mul -> global_atomic_add` forcing one full drain per slot (~12-16 of 25
    # epilogue drains). Hoist the token/weight load + validity check to ONE per
    # (mi, i) row, then batch the per-ni atomics inside a single scf_if. This
    # collapses mfmas_n redundant drains per row into one.
    for mi in range(mfmas_m):
        # Lever G drain reduction: the per-row SortedTokenIds / SortedWeights
        # loads do NOT depend on each other. The pre-lever code loaded the weight
        # INSIDE each row's `scf_if(valid)` block, forcing a separate
        # `global_load_f32 -> vmcnt(0) -> v_mul -> atomic` serialization per row
        # (c_per_lane full drains in the epilogue). Here we ISSUE every row's
        # token + weight load up front into DISTINCT registers, then drain ONCE
        # for the whole batch, then run the per-row validity-masked atomics with
        # the operands already resident. The weight bucket index is always a
        # valid slot (padded-row slots have weights too), so the unconditional
        # hoisted load is safe; the validity check still gates the atomic store.
        rows = []
        for i in range(atom.c_per_lane):
            row_in, col_in = atom.lane_to_output(b, lane, i)
            row = b.add(
                block_m_off,
                b.add(warp_m_off, b.add(b.const_i32(mi * atom.m), row_in)),
            )
            bucket = row
            token = b.global_load_i32(SortedTokenIds, bucket)
            w = b.global_load_f32(SortedWeights, bucket)
            rows.append((i, col_in, token, w))
        # One rolling drain covers all c_per_lane (token,weight) loads instead of
        # one vmcnt(0) per row.
        b.s_waitcnt(vmcnt=0)
        for i, col_in, token, w in rows:
            valid = b.land(b.cmp_ge(token, c0), b.cmp_lt(token, tokens))
            with b.scf_if(valid):
                token_h = b.mul(token, H_out)
                for ni in range(mfmas_n):
                    flat = mi * mfmas_n + ni
                    acc = down_list[flat]
                    col = b.add(
                        ho_off,
                        b.add(warp_n_off, b.add(b.const_i32(ni * atom.n), col_in)),
                    )
                    v = b.vec_extract(acc, i)
                    contrib = b.fmul(w, v)
                    y_off = b.add(token_h, col)
                    b.global_atomic_add(Y, y_off, contrib)


# ---------------------------------------------------------------------------
# STAGE 1b: SiLU(gate)*up + dynamic-quantize Hidden to fp8 in LDS
# ---------------------------------------------------------------------------


def _silu_mul_f32(
    b: IRBuilder, g: Value, u: Value, *, one_f32: Value, c_neg_log2e: Value
) -> Value:
    """f32 SwiGLU chain ``silu(g) * u`` (sigmoid via exp2), op order matched.

    The sigmoid reciprocal uses ``rcp_fast`` (single ``v_rcp_f32``, ~1 ulp) to
    match aiter's activation epilogue; the IEEE-correct ``rcp`` here expanded to
    a 68-instr ``v_div_scale``/``v_div_fmas``/``v_div_fixup`` sequence per TG.
    """
    sig = b.rcp_fast(b.fadd(one_f32, b.exp2(b.fmul(c_neg_log2e, g))))
    silu = b.fmul(g, sig)
    return b.fmul(silu, u)


def _store_hidden_f32_pass(
    b: IRBuilder,
    *,
    atom: MfmaAtom,
    gate_list,
    up_list,
    f32_view,
    warp_m_off: Value,
    warp_n_off: Value,
    lane: Value,
    mfmas_m: int,
    mfmas_n: int,
    one_f32: Value,
    c_neg_log2e: Value,
    c_floor: Value,
) -> Value:
    """Fused Pass A: silu(gate)*up -> f32 LDS scratch AND in-register amax.

    Writes each MFMA output cell to the f32 LDS scratch (still consumed by the
    requantize pass) while accumulating, per lane, the absolute max over every
    cell this lane owns. Because ``tile_n_inter`` is partitioned so that an
    entire warp's N-extent (warp_n_off .. +mfmas_n*atom.n) lands inside ONE
    128-inter scale block, the per-lane partial here belongs to exactly one
    block; the caller reduces the 64 lane partials of a warp (and the 2 warps
    that share a block) to the final per-block amax. Returns the per-lane
    partial amax (f32), already floored at ``AMAX_FLOOR``.
    """
    amax_partial = c_floor
    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            flat = mi * mfmas_n + ni
            g_vec = gate_list[flat]
            u_vec = up_list[flat]
            for i in range(atom.c_per_lane):
                row_in, col_in = atom.lane_to_output(b, lane, i)
                row = b.add(warp_m_off, b.add(b.const_i32(mi * atom.m), row_in))
                col = b.add(warp_n_off, b.add(b.const_i32(ni * atom.n), col_in))
                g = b.vec_extract(g_vec, i)
                u = b.vec_extract(u_vec, i)
                h = _silu_mul_f32(b, g, u, one_f32=one_f32, c_neg_log2e=c_neg_log2e)
                f32_view_store(b, f32_view, row, col, h)
                amax_partial = b.fmax(amax_partial, b.fabs(h))
    return amax_partial


def _store_hidden_fp8_static_pass(
    b: IRBuilder,
    *,
    atom,
    gate_list,
    up_list,
    fp8_view,
    warp_m_off: Value,
    warp_n_off: Value,
    lane: Value,
    mfmas_m: int,
    mfmas_n: int,
    one_f32: Value,
    c_neg_log2e: Value,
    inv_scale: Value,
) -> None:
    """Single-pass epilogue for a scale that is known before the tile is.

    The dynamic path has to spill every cell to an f32 scratch and come back
    for it, because the divisor is the tile's own amax. Given the scale up
    front, each cell can go silu -> scale -> fp8 in registers and land in its
    final LDS slot immediately: no scratch, no amax reduction, no second sweep,
    and two fewer barriers. Cell order and addressing are otherwise identical
    to :func:`_store_hidden_f32_pass`, so the fp8 tile it leaves behind is the
    same one the three-pass form leaves for the same scale.
    """
    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            flat = mi * mfmas_n + ni
            g_vec = gate_list[flat]
            u_vec = up_list[flat]
            for i in range(atom.c_per_lane):
                row_in, col_in = atom.lane_to_output(b, lane, i)
                row = b.add(warp_m_off, b.add(b.const_i32(mi * atom.m), row_in))
                col = b.add(warp_n_off, b.add(b.const_i32(ni * atom.n), col_in))
                g = b.vec_extract(g_vec, i)
                u = b.vec_extract(u_vec, i)
                h = _silu_mul_f32(b, g, u, one_f32=one_f32, c_neg_log2e=c_neg_log2e)
                q = b.cvt_f32_to_fp8(b.fmul(h, inv_scale))
                b.smem_store_vN(fp8_view.base, [row, col], q, 1)


def f32_view_store(b: IRBuilder, view, row: Value, col: Value, val: Value) -> None:
    b.smem_store_vN(view.base, [row, col], val, 1)


def f32_view_load(b: IRBuilder, view, row: Value, col: Value) -> Value:
    v = b.smem_load_vN(view.base, row, col, dtype=F32, n=1)
    return b.vec_extract(v, 0)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def moe_split_down_fp8_grid(
    num_m_blocks: int, h_out: int, spec: FusedMegaKernelSpecFp8
) -> Tuple[int, int, int]:
    """Launch grid for the partial-fusion down kernel.

    ``grid = (ceil(H_out / (tile_n_down * down_h_loop)), num_m_blocks, 1)``.
    The parallel axis is the OUTPUT hidden dimension, not the intermediate,
    which is the whole point: no CTA holds a partial sum, so there is no
    cross-CTA reduction. ``down_h_loop`` folds that many output tiles into one
    CTA so the staged intermediate is read from HBM once instead of once per
    tile.
    """
    per_cta = spec.tile_n_down * spec.down_h_loop
    gx = (h_out + per_cta - 1) // per_cta
    return (gx, num_m_blocks, 1)


def moe_split_down_fp8_signature(spec: FusedMegaKernelSpecFp8):
    from ...helpers.spec import SignatureBuilder

    return (
        SignatureBuilder()
        .ptr("Inter", "fp8e4m3")
        .ptr("InterScale", "f32")
        .ptr("WDown", "fp8e4m3")
        .ptr("WDownScale", "f32")
        .ptr("SortedTokenIds", "i32")
        .ptr("SortedWeights", "f32")
        .ptr("BlockExpertIds", "i32")
        .ptr("Y", "f32")
        .scalar("N", "i32")  # = I (intermediate, the full contraction extent)
        .scalar("H_out", "i32")
        .scalar("stride_b_down", "i32")
        .scalar("stride_down_scale", "i32")
        .scalar("stride_down_scale_e", "i32")
        .scalar("tokens", "i32")
        .build()
    )


def build_moe_split_down_fp8(
    spec: FusedMegaKernelSpecFp8, arch: str = "gfx950"
) -> KernelDef:
    """Down GEMM as its own launch, tiled over H_out (PARTIAL FUSION stage 2).

    Measurement, not theory, motivates this kernel. Per-launch profiling of the
    fused mega-kernel against vLLM's Triton path showed the fused gate/up half
    is ~4 us FASTER than Triton's equivalent, but the fused down half is
    ~16.7 us SLOWER -- and 7.2 us of that is atomic traffic the fused shape
    forces. Fusing pins a token block's intermediate to one CTA's LDS, so the
    only axis left to parallelize stage 2 over is the intermediate, which is
    the CONTRACTION axis; six CTAs then each own a partial and recombine
    through 12.6 MB of fp32 atomics.

    Splitting stage 2 into its own launch frees it to tile over H_out instead.
    Each CTA then reduces the ENTIRE intermediate in-block and owns its outputs
    outright, so the only atomics left are the topk (not slice) accumulation:
    2.1 MB. The price is re-reading the intermediate from HBM (1.4 MB, plus L2
    re-reads across h-tiles) and one extra launch.

    Grid ``(H_out/tile_n_down, num_m_blocks)``; each CTA reads
    ``[tile_m, I]`` of the intermediate into LDS, then runs the same
    :func:`_emit_fp8_down_fused_cells` the fused kernel uses, with the
    contraction extent widened from one inter slice to all of ``I``.
    """
    ok, why, _ = validate_arch_and_block_size(arch, spec.block_size)
    if not ok:
        raise ValueError(f"invalid fp8 split-down spec for {arch}: {why}")
    down_atom = spec.down_atom()
    if spec.swizzle_down and not b_swizzle_supported(down_atom, spec.wave_size):
        raise ValueError("swizzle_down needs a B fragment that tiles onto lanes x 16B")

    atom = down_atom
    tile_m = spec.tile_m
    HGK = int(spec.hidden_group_k)
    cadence = spec.sched_cadence

    b = IRBuilder(spec.kernel_name() + "_splitdown")
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.mfma_vgpr_form:
        b.kernel.attrs["mfma_vgpr_form"] = True

    Inter = b.param(
        "Inter", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16
    )
    InterScale = b.param(
        "InterScale", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    WDown = b.param(
        "WDown", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16
    )
    WDownScale = b.param(
        "WDownScale", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    SortedTokenIds = b.param(
        "SortedTokenIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    SortedWeights = b.param(
        "SortedWeights", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    BlockExpertIds = b.param(
        "BlockExpertIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    Y = b.param("Y", PtrType(F32, "global"), noalias=True, align=16)
    N = b.param("N", I32)
    H_out = b.param("H_out", I32)
    stride_b_down = b.param("stride_b_down", I32)
    stride_down_scale = b.param("stride_down_scale", I32)
    stride_down_scale_e = b.param("stride_down_scale_e", I32)
    tokens = b.param("tokens", I32)

    # The down tile's output width is split across warps exactly as in the
    # fused kernel, but here a CTA owns ONE h-tile rather than looping them.
    tile_n_down = spec.tile_n_down
    warp_n = spec.warp_n
    if tile_n_down % (warp_n * atom.n):
        raise ValueError(
            f"tile_n_down ({tile_n_down}) must be a multiple of "
            f"warp_n*atom.n ({warp_n * atom.n})"
        )
    mfmas_n_down = tile_n_down // (warp_n * atom.n)
    # Must divide by warp_m: this is the count of M atoms ONE warp owns, and it
    # scales down_warp_m_off. Deriving it from tile_m alone made every warp claim
    # the whole tile, so with warp_m>1 the row offsets ran off the end of the
    # tile and the output was NaN. spec.mfmas_m_down is the shared definition the
    # fused builder already uses.
    mfmas_m_down = spec.mfmas_m_down
    if mfmas_m_down * spec.warp_m * atom.m != tile_m:
        raise ValueError(
            f"warp_m ({spec.warp_m}) must split tile_m ({tile_m}) into whole "
            f"{atom.m}-row atoms"
        )

    c_wave = b.const_i32(spec.wave_size)
    c_warps_n = b.const_i32(warp_n)
    c_block_m = b.const_i32(tile_m)
    c_threads = b.const_i32(spec.block_size)
    c_group_k = b.const_i32(GROUP_K)
    c0 = b.const_i32(0)

    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    warp_m_idx = b.div(warp_id, c_warps_n)
    warp_n_idx = b.mod(warp_id, c_warps_n)
    lane = b.mod(tid, c_wave)

    m_block_idx = b.block_id_y()
    block_m_off = b.mul(m_block_idx, c_block_m)
    h_loop = int(spec.down_h_loop)
    ho_base = b.mul(b.block_id_x(), b.const_i32(tile_n_down * h_loop))

    expert_idx = b.global_load_i32(BlockExpertIds, m_block_idx)
    WDown = b.global_ptr_add(
        WDown,
        b.mul(b.sext(expert_idx, I64), b.sext(stride_b_down, I64)),
    )
    WDownScale = b.global_ptr_add(
        WDownScale,
        b.mul(
            b.mul(b.sext(expert_idx, I64), b.sext(stride_down_scale_e, I64)),
            b.const_i64(4),
        ),
    )

    # ---- LDS: the whole intermediate tile, staged from HBM ----------------
    # inter_max is a build-time bound on I so the LDS allocation is static;
    # the CONTRACTION extent stays the runtime N, so one binary serves any
    # I <= inter_max.
    inter_max = int(spec.split_inter_max)
    if inter_max % HGK:
        raise ValueError(f"split_inter_max ({inter_max}) must be a multiple of {HGK}")
    pad_fp8 = spec.lds_pad
    hid_w = inter_max + pad_fp8
    n_blocks_full = inter_max // HGK
    Hidden_smem = b.smem_alloc(FP8E4M3, [tile_m, hid_w], name_hint="Hidden_smem")
    # ONE row, not tile_m: stage 1 reduces the amax over every row of a block,
    # so all tile_m rows of a column block hold the same f32. Storing it once
    # and reading row 0 turns 96 LDS stores per CTA into 6.
    HiddenScale_smem = b.smem_alloc(
        F32, [1, n_blocks_full], name_hint="HiddenScale_smem"
    )
    fp8_view = TensorView(
        base=Hidden_smem,
        desc=TensorDescriptor.packed((tile_m, hid_w), FP8E4M3),
        addr_space="lds",
    )
    scale_view = TensorView(
        base=HiddenScale_smem,
        desc=TensorDescriptor.packed((1, n_blocks_full), F32),
        addr_space="lds",
    )

    lane_decode = decode_mfma_lanes(b, atom, lane)

    # Stage [tile_m, N] fp8 into LDS, 16 bytes per lane per step.
    n16 = b.div(N, b.const_i32(16))
    load_sweep = b.scf_for_iter(
        tid, b.mul(b.const_i32(tile_m), n16), c_threads, [], iv_name="lcell"
    )
    with load_sweep as lcell:
        row = b.div(lcell, n16)
        col16 = b.mul(b.mod(lcell, n16), b.const_i32(16))
        v16 = b.global_load_vN(
            Inter,
            b.add(b.mul(b.add(block_m_off, row), N), col16),
            FP8E4M3,
            16,
        )
        b.smem_store_vN(fp8_view.base, [row, col16], v16, 16)
        b.scf_yield()

    # Scales are row-uniform (stage 1 reduces the amax over all rows of a
    # block), so one f32 per inter block is stored once and every lane reads
    # row 0. Broadcasting it down all tile_m rows instead -- which is what this
    # did -- cost 96 LDS stores per CTA to publish 6 distinct values, and the
    # compiler emitted them as 56 ds_write2_b32, 93% of the kernel's DS traffic.
    nb = b.div(N, c_group_k)
    scale_sweep = b.scf_for_iter(tid, nb, c_threads, [], iv_name="scell")
    with scale_sweep as blk:
        s = b.global_load_f32(InterScale, b.add(b.mul(m_block_idx, nb), blk))
        b.smem_store_vN(scale_view.base, [c0, blk], s, 1)
        b.scf_yield()
    b.sync()

    # ---- down GEMM: the FULL intermediate reduced inside this CTA ---------
    down_warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m_down * atom.m))
    down_warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n_down * atom.n))

    def emit_h_tile(ho_off):
        """One ``tile_n_down``-wide output tile against the staged LDS tile.

        Reads LDS only, so repeating it needs no extra barrier: the single
        ``b.sync()`` after staging still dominates every iteration.
        """
        down_list = []
        for mi in range(mfmas_m_down):
            m_row_base = b.add(down_warp_m_off, b.const_i32(mi * atom.m))
            n_tile_bases_down = [
                b.add(ho_off, b.add(down_warp_n_off, b.const_i32(ni * atom.n)))
                for ni in range(mfmas_n_down)
            ]
            down_list.extend(
                _emit_fp8_down_fused_cells(
                    b,
                    a_view=fp8_view,
                    WDown=WDown,
                    WDownScale=WDownScale,
                    atom=atom,
                    lane_decode=lane_decode,
                    n_tile_bases=n_tile_bases_down,
                    scale_view=scale_view,
                    inter_slice=N,  # runtime: all of I, no cross-CTA partials
                    inter_full=N,
                    inter_col_base=c0,
                    stride_down_scale=stride_down_scale,
                    m_row_base=m_row_base,
                    hidden_group_k=HGK,
                    tag=f"sd{mi}",
                    cadence=cadence,
                    prefetch_depth=spec.down_depth,
                    sched_group=spec.down_group,
                    sched_mode=spec.window_sched,
                    swizzle_b=spec.swizzle_down,
                    wave_size=spec.wave_size,
                    pipeline_k=spec.down_pipeline_k,
                    scale_row=c0,
                )
            )

        # Only the topk accumulation is atomic now -- 2.1 MB, not 12.6 MB.
        _emit_down_atomic_reduce(
            b,
            atom=atom,
            down_list=down_list,
            warp_m_off=down_warp_m_off,
            warp_n_off=down_warp_n_off,
            lane=lane,
            mfmas_m=mfmas_m_down,
            mfmas_n=mfmas_n_down,
            block_m_off=block_m_off,
            ho_off=ho_off,
            H_out=H_out,
            SortedTokenIds=SortedTokenIds,
            SortedWeights=SortedWeights,
            Y=Y,
            tokens=tokens,
        )

    if h_loop == 1:
        emit_h_tile(ho_base)
    else:
        # Runtime loop, not an unrolled one: at h_loop=16 unrolling would
        # multiply an already ~700-instruction body by 16 for no scheduling
        # gain, since consecutive tiles share no operands.
        h_sweep = b.scf_for_iter(
            c0, b.const_i32(h_loop), b.const_i32(1), [], iv_name="htile"
        )
        with h_sweep as it:
            emit_h_tile(b.add(ho_base, b.mul(it, b.const_i32(tile_n_down))))
            b.scf_yield()
    b.ret()
    return b.kernel


def build_moe_fused_mega_gemm_fp8(
    spec: FusedMegaKernelSpecFp8,
    arch: str = "gfx950",
    *,
    persistent: bool = False,
    split_gateup: bool = False,
) -> KernelDef:
    """Build the STAGE 1 fp8 fused-MoE mega-kernel.

    Current implementation covers STAGE 1 (gate+up fp8 GEMM with per-128-block
    dequant -> SiLU*up f32 -> dynamic-quant to fp8 staged in the persistent LDS
    ``Hidden_smem`` + per-block scales in ``HiddenScale_smem``). The down GEMM +
    weighted atomic reduce (STAGE 2) is a documented stub at the end of the body.

    PERSISTENT TRANSFORM (additive, default OFF -> byte-identical):
    When ``persistent=True`` the kernel is launched on a fixed 1-D grid
    ``(P, 1, 1)`` (P = host-chosen persistent block count) and TWO extra i32
    kernel params are appended -- ``grid_x`` (= ``ceil(I/tile_n_inter)``, the
    inter-tile modulus) and ``total_work`` (= ``grid_x * num_active_m_blocks``).
    Each persistent block ``p`` grid-strides the linear work-id space::

        for w = p; w < total_work; w += P:
            bx = w % grid_x   # inter-tile  (fast index, weight-tile reuse)
            by = w // grid_x  # active m-block
            <run the ENTIRE per-TG mega body for (by, bx)>

    The per-item state is re-derived each iteration: GEMM accumulators are
    ``scf.for`` iter-args (re-zeroed by re-entering the body), the dyn-quant LDS
    scratch is write-before-read within an item, the Y epilogue is a
    commutative atomic-add (host-zeroed once), and ONE extra inter-item
    ``b.sync()`` guards the LDS hand-off between the previous item's down-GEMM
    reads and the current item's Pass-A writes. The static-partition strided
    walk covers ``[0, total_work)`` exactly once (the hardened skew parity guards
    against dropped/duplicated items). No XCD remap yet (next phase).

    The ``persistent=False`` path emits the IDENTICAL op stream (no extra
    params, no loop, no extra sync) so the default build is byte-identical.
    """
    ok, why, _ = validate_arch_and_block_size(arch, spec.block_size)
    if not ok:
        raise ValueError(f"invalid fp8 fused-mega spec for {arch}: {why}")
    atom = spec.gate_up_atom()
    # The down GEMM contracts the INTERMEDIATE, whose scale-block width is
    # hidden_group_k, so its atom must be no wider than that. It shares the
    # 16x16 C layout (and therefore lane_decode) with the gate/up atom, only
    # the K extent differs. At the default down_k=128 this IS gate_up_atom, so
    # existing configs are unchanged.
    down_atom = spec.down_atom()
    # The swizzled weight layout is defined by the consuming atom's fragment
    # shape, so a stream may only be swizzled if its atom tiles cleanly.
    for _flag, _name, _at in (
        (spec.swizzle_gu, "swizzle_gu", atom),
        (spec.swizzle_down, "swizzle_down", down_atom),
    ):
        if _flag and not b_swizzle_supported(_at, spec.wave_size):
            raise ValueError(
                f"{_name} needs a B fragment that tiles onto "
                f"{spec.wave_size} lanes x 16B; atom {_at.m}x{_at.n}x{_at.k} "
                f"has b_per_lane={_at.b_per_lane}"
            )
    # L6: the unscaled fp8 16x16x128 hero atom reuses the (catalog-registered)
    # ``mfma.scale.f32.16x16x128.f8f6f4`` intrinsic with the in-instruction E8M0
    # scales pinned to the neutral value (verified numerically standalone), so it
    # is gfx950-valid even though the plain unscaled 16x16x128 fp8 SHAPE is not a
    # separate JSON catalog row. Only run the per-arch catalog guard for atoms
    # that ARE catalog shapes (the legacy 16x16x32 path); skip it for the hero
    # atom to avoid a spurious NotImplementedError (this is a guard against
    # gfx950-only intrinsics reaching comgr -- which this atom is not).
    if atom.k != 128:
        validate_mfma_atom_in_catalog(atom, arch, where="moe_fused_mega_fp8")

    cadence = spec.sched_cadence

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.mfma_vgpr_form:
        b.kernel.attrs["mfma_vgpr_form"] = True

    # ---- params (BUILD_SPEC_FP8 Section 2.7) ---------------------------
    A = b.param("A", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16)
    WGate = b.param(
        "WGate", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16
    )
    WUp = b.param(
        "WUp", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16
    )
    WDown = b.param(
        "WDown", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16
    )
    AScale = b.param(
        "AScale", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    WGateScale = b.param(
        "WGateScale", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    WUpScale = b.param(
        "WUpScale", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    WDownScale = b.param(
        "WDownScale", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    SortedTokenIds = b.param(
        "SortedTokenIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    SortedWeights = b.param(
        "SortedWeights", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    BlockExpertIds = b.param(
        "BlockExpertIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    Y = b.param("Y", PtrType(F32, "global"), noalias=True, align=16)
    M = b.param("M", I32)
    N = b.param("N", I32)  # = I (inter dim)
    K = b.param("K", I32)  # = H (hidden contraction)
    H_out = b.param("H_out", I32)  # = H (down output)
    stride_a = b.param(
        "stride_a", I32
    )  # noqa: F841 -- ABI (A is dense, gather elsewhere)
    stride_b_gate = b.param("stride_b_gate", I32)
    stride_b_up = b.param("stride_b_up", I32)
    stride_b_down = b.param("stride_b_down", I32)  # noqa: F841 -- used in STAGE 2
    stride_a_scale = b.param("stride_a_scale", I32)
    stride_gate_scale = b.param("stride_gate_scale", I32)
    stride_up_scale = b.param("stride_up_scale", I32)
    stride_down_scale = b.param("stride_down_scale", I32)  # noqa: F841 -- STAGE 2
    # Per-expert ELEMENT stride for the weight scale tensors (scales are
    # per-expert, but the scale pointer is NOT folded by _b_base; fold here).
    stride_gate_scale_e = b.param("stride_gate_scale_e", I32)
    stride_up_scale_e = b.param("stride_up_scale_e", I32)
    stride_down_scale_e = b.param("stride_down_scale_e", I32)
    slot_size = b.param("slot_size", I32)  # noqa: F841 -- ABI
    tokens = b.param("tokens", I32)  # noqa: F841 -- used in STAGE 2 epilogue

    # Persistent transform params (ONLY on the persistent ABI variant so the
    # default kernel stays byte-identical). grid_x = inter-tile modulus,
    # total_work = grid_x * num_active_m_blocks, P = launched persistent grid
    # size (the grid-stride). P is passed explicitly (no gpu.grid_dim intrinsic
    # in the IR surface) so no core lowering edit is needed.
    if persistent:
        p_grid_x = b.param("grid_x", I32)
        p_total_work = b.param("total_work", I32)
        p_P = b.param("P", I32)
    if split_gateup:
        Inter = b.param(
            "Inter", PtrType(FP8E4M3, "global"), noalias=True, align=16
        )
        InterScale = b.param(
            "InterScale", PtrType(F32, "global"), noalias=True, align=4
        )

    tile_m = spec.tile_m
    tile_n = spec.tile_n_inter
    HGK = int(spec.hidden_group_k)
    if HGK != GROUP_K:
        if not spec.down_fused_cells:
            raise ValueError(
                "hidden_group_k != 128 requires down_fused_cells=True "
                "(the per-cell down emitter assumes a 128-wide hidden block)"
            )
        # Check the ATOM's k, not spec.down_k: down_k only selects between the
        # 32- and 128-wide atoms, so an unrepresentable value like 64 would
        # otherwise silently land on the 128 atom and make atoms_per_group 0.
        _dk = spec.down_atom().k
        if _dk > HGK or HGK % _dk:
            raise ValueError(
                f"down atom k ({_dk}, from down_k={spec.down_k}) must divide "
                f"hidden_group_k ({HGK}) so a down atom stays inside one hidden "
                "scale block"
            )
    n_blocks = tile_n // HGK

    static_scale = bool(spec.static_inter_scale)
    if static_scale:
        if not split_gateup:
            raise ValueError(
                "static_inter_scale is split gate/up only: the fused kernel's "
                "stage 2 reads the f32 scratch this mode does not produce"
            )
        if n_blocks != 1:
            # With more than one scale block per tile the divisor stops being
            # CTA-uniform, so the single wave-uniform load below would apply
            # one block's scale to another block's columns.
            raise ValueError(
                f"static_inter_scale needs tile_n_inter ({tile_n}) == "
                f"hidden_group_k ({HGK}), got {n_blocks} scale blocks per tile"
            )

    # LEVER (fuse-quant) invariant: each warp's N-extent must lie inside exactly
    # one 128-inter scale block, AND each block must be covered by an integer
    # number of warps, so the register-amax combine is exact.
    #
    # The scale is row-uniform over the whole tile (see Pass A below), so a
    # block's amax must be reduced over EVERY warp that holds part of it. With
    # ``warp_m > 1`` that set is 2-D: warp ``mi*warp_n + ni`` for all mi and for
    # the ni covering the block. The combine below walks exactly that set, which
    # collapses to the old consecutive-warp run when warp_m == 1.
    #
    # warp_m is what makes wide tile_m affordable: it splits M across warps, so
    # tile_m can grow with the per-wave accumulator count held fixed. Without it
    # tile_m=64 needs 416 VGPR + 160 AGPR and drops to one wave per SIMD.
    warp_n_cols = tile_n // spec.warp_n  # mfmas_n * atom.n
    warps_per_block = HGK // warp_n_cols
    if (
        (tile_m // spec.warp_m) % spec.warp_tile_m
        or warp_n_cols * spec.warp_n != tile_n
        or HGK % warp_n_cols != 0
        or warps_per_block * n_blocks != spec.warp_n
    ):
        raise ValueError(
            "fuse-quant lever requires warps to tile the 128-inter blocks "
            f"evenly and warp_m to divide tile_m into whole atoms (got "
            f"tile_m={tile_m}, warp_m={spec.warp_m}, tile_n={tile_n}, "
            f"warp_n={spec.warp_n}, warp_n_cols={warp_n_cols})"
        )

    c_wave = b.const_i32(spec.wave_size)
    c_warps_n = b.const_i32(spec.warp_n)
    c_block_m = b.const_i32(tile_m)
    c_block_n = b.const_i32(tile_n)
    c0 = b.const_i32(0)

    # ---- block/thread prelude -----------------------------------------
    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    warp_m_idx = b.div(warp_id, c_warps_n)
    warp_n_idx = b.mod(warp_id, c_warps_n)
    lane = b.mod(tid, c_wave)

    # ---- BYTE-BASE FIX (BUILD_SPEC_FP8 Section 2.4): fp8 weights = 1 byte.
    # GOLDEN-GATE NOTE: the byte-multiplier const is created LAZILY (inside
    # ``_b_base``, i.e. right after ``expert_idx`` is loaded) instead of here, so
    # the DEFAULT (non-persistent) op-counter order is byte-identical to the
    # pre-persistent baseline (block_id_y -> expert_idx -> const_i64(1) -> ...).
    # Creating it at prelude scope bumped every following value's SSA number by 1.
    _elem_bytes_b_holder = []

    def _elem_bytes_b() -> Value:
        if not _elem_bytes_b_holder:
            _elem_bytes_b_holder.append(b.const_i64(1))
        return _elem_bytes_b_holder[0]

    # Immutable per-tensor BASE pointers (the un-rebased kernel params). The
    # per-expert rebasing is a PURE function of the work-item's expert_idx, so it
    # is re-derived per work-item inside ``_select_item`` (== once, at the
    # original prelude position, for the default non-persistent path) rather than
    # mutating the param Values. This keeps the persistent loop able to re-select
    # a different expert each iteration.
    WGate0, WUp0, WDown0 = WGate, WUp, WDown
    WGateScale0, WUpScale0, WDownScale0 = WGateScale, WUpScale, WDownScale

    def _b_base(ptr: Value, stride_b: Value, expert_idx: Value) -> Value:
        bytes_off = b.mul(
            b.mul(b.sext(expert_idx, I64), b.sext(stride_b, I64)), _elem_bytes_b()
        )
        return b.global_ptr_add(ptr, bytes_off)

    # Per-expert base for the f32 weight scale tensors (4-byte elements). The
    # scale index math inside the k-loops carries no expert term, so the
    # per-expert slice is selected here on the pointer.
    def _scale_base(ptr: Value, stride_e: Value, expert_idx: Value) -> Value:
        bytes_off = b.mul(
            b.mul(b.sext(expert_idx, I64), b.sext(stride_e, I64)),
            b.const_i64(4),
        )
        return b.global_ptr_add(ptr, bytes_off)

    # Per-work-item selected state, populated by ``_select_item`` and read by
    # ``_emit_body`` (nonlocal). For the default path these are computed ONCE in
    # the original prelude op-order; for the persistent path they are re-derived
    # each loop iteration.
    expert_idx = None
    WGate = WUp = WDown = None
    WGateScale = WUpScale = WDownScale = None
    block_m_off = gu_n_off = None

    def _select_item(m_block_idx: Value, bx_block) -> None:
        """Derive the per-(by, bx) work-item state (expert + rebased weight/scale
        pointers + m/n tile offsets). Pure function of (m_block_idx, bx_block).

        ``bx_block=None`` emits ``block_id_x()`` HERE (the default-path order,
        where block_id_x is the LAST prelude op -> byte-identical). The
        persistent path passes an explicit decoded ``bx`` Value.
        """
        nonlocal expert_idx, WGate, WUp, WDown
        nonlocal WGateScale, WUpScale, WDownScale, block_m_off, gu_n_off
        expert_idx = b.global_load_i32(BlockExpertIds, m_block_idx)
        # Create the byte-multiplier const HERE (right after expert_idx) so the
        # default-path op-counter order matches the pre-persistent baseline.
        _elem_bytes_b()
        WGate = _b_base(WGate0, stride_b_gate, expert_idx)
        WUp = _b_base(WUp0, stride_b_up, expert_idx)
        WDown = _b_base(WDown0, stride_b_down, expert_idx)
        WGateScale = _scale_base(WGateScale0, stride_gate_scale_e, expert_idx)
        WUpScale = _scale_base(WUpScale0, stride_up_scale_e, expert_idx)
        WDownScale = _scale_base(WDownScale0, stride_down_scale_e, expert_idx)
        block_m_off = b.mul(m_block_idx, c_block_m)
        if bx_block is None:
            bx_block = b.block_id_x()
        gu_n_off = b.mul(bx_block, c_block_n)

    # DEFAULT (non-persistent) path: select the single work-item HERE, in the
    # ORIGINAL prelude op-order (block_id_y -> expert_idx -> bases -> block_m_off
    # -> block_id_x -> gu_n_off), so the op stream is byte-identical to pre-edit.
    # The persistent path defers selection into the work-item loop instead.
    if not persistent:
        _select_item(b.block_id_y(), None)

    # ---- LDS allocations ----------------------------------------------
    # Persistent fp8 Hidden buffer (half the f16 bytes): silu(gate)*up quantized
    # here, reused as the down-GEMM LDS-resident A operand in STAGE 2.
    if spec.lds_pad % 16:
        raise ValueError(f"lds_pad must be a multiple of 16B (got {spec.lds_pad})")
    pad_fp8 = spec.lds_pad  # 1 byte/elem
    pad_f32 = spec.lds_pad // 4  # 4 bytes/elem
    hid_w = tile_n + pad_fp8
    f32_w = tile_n + pad_f32
    Hidden_smem = b.smem_alloc(FP8E4M3, [tile_m, hid_w], name_hint="Hidden_smem")
    # Per-(row, 128-inter-block) dynamic scales for the down dequant (STAGE 2).
    HiddenScale_smem = b.smem_alloc(
        F32, [tile_m, n_blocks], name_hint="HiddenScale_smem"
    )
    # f32 scratch for the exact per-block amax reduction (STAGE 1 only). Under
    # a supplied scale there is no amax to reduce and nothing to re-read, so
    # the scratch shrinks to a placeholder rather than costing LDS for a buffer
    # the epilogue never touches.
    HiddenF32_smem = b.smem_alloc(
        F32, [1, 1] if static_scale else [tile_m, f32_w], name_hint="HiddenF32_smem"
    )
    # Tiny per-warp amax partials (one f32 per warp). Each warp's N-extent lands
    # inside ONE 128-inter block, so warps {2*blk, 2*blk+1} cover block ``blk``.
    n_warps = spec.warp_m * spec.warp_n
    WarpAmax_smem = b.smem_alloc(
        F32, [1] if static_scale else [n_warps], name_hint="WarpAmax_smem"
    )

    # ---- DTLA gate+up B staging (GOAL 1) -----------------------------------
    # Per-wave, ping-pong (4 logical slots: gate/up x 2 ni-buffers) direct-to-LDS
    # landing zone. Each slot holds DTLA_CHUNKS lane-blocks of (wave_size x 16B)
    # -- the lane-contiguous spread the global.load.lds HW imposes. Shape
    # [n_warps*DTLA_SLOTS*DTLA_CHUNKS*wave_size, 16] fp8: 4*4*2*64*16 = 32 KiB
    # at the canonical geometry.
    # 2 B slots (gate/up) per prefetch buffer.
    _dtla_depth = max(2, int(spec.dtla_depth))
    DTLA_SLOTS = 2 * _dtla_depth
    DTLA_CHUNKS = (atom.b_per_lane + DTLA_CHUNK - 1) // DTLA_CHUNK
    bstage_rows = n_warps * DTLA_SLOTS * DTLA_CHUNKS * spec.wave_size
    BStage_smem = b.smem_alloc(
        FP8E4M3, [bstage_rows, DTLA_CHUNK], name_hint="BStage_smem"
    )
    bstage_view = TensorView(
        base=BStage_smem,
        desc=TensorDescriptor.packed((bstage_rows, DTLA_CHUNK), FP8E4M3),
        addr_space="lds",
    )

    # ---- cooperative CTA-shared gate/up B tile -----------------------------
    # One slot per (half, n-cell) for the WHOLE tile rather than per wave: gate
    # cells [0, mfmas_n_all) then up cells [mfmas_n_all, 2*mfmas_n_all). At the
    # canonical geometry that is 2*8 slots * 2 chunks * 64 lanes * 16 B = 32 KiB,
    # which with the epilogue buffer packed on top of it (their live ranges are
    # disjoint, so the smem pool's liveness packer overlaps them) leaves 2
    # workgroups per CU -- 8 waves/CU at 4 waves per workgroup.
    mfmas_n_all = tile_n // atom.n
    BCoop_smem = None
    coop_view = None
    coop_bytes = 0
    if spec.coop_b_lds:
        if GROUP_K // atom.k != 1:
            raise ValueError(
                "coop_b_lds needs one MFMA atom per scale group (gate_up_k=128); "
                f"got {GROUP_K // atom.k} atoms per group"
            )
        if mfmas_n_all % n_warps:
            raise ValueError(
                f"coop_b_lds needs n_warps ({n_warps}) to divide the tile's "
                f"n-cell count ({mfmas_n_all}) so staging splits evenly"
            )
        coop_rows = 2 * mfmas_n_all * DTLA_CHUNKS * spec.wave_size
        coop_bytes = coop_rows * DTLA_CHUNK
        BCoop_smem = b.smem_alloc(
            FP8E4M3, [coop_rows, DTLA_CHUNK], name_hint="BCoop_smem"
        )
        coop_view = TensorView(
            base=BCoop_smem,
            desc=TensorDescriptor.packed((coop_rows, DTLA_CHUNK), FP8E4M3),
            addr_space="lds",
        )

    f32_view = TensorView(
        base=HiddenF32_smem,
        desc=TensorDescriptor.packed((tile_m, f32_w), F32),
        addr_space="lds",
    )
    # The overlay only saves anything if the B tile is at least as large as the
    # staging it would absorb (see ``coop_alias`` for why it is off by default).
    hidden_base = Hidden_smem
    if spec.coop_alias and BCoop_smem is not None and coop_bytes >= tile_m * hid_w:
        hidden_base = BCoop_smem
    fp8_view = TensorView(
        base=hidden_base,
        desc=TensorDescriptor.packed((tile_m, hid_w), FP8E4M3),
        addr_space="lds",
    )
    scale_view = TensorView(
        base=HiddenScale_smem,
        desc=TensorDescriptor.packed((tile_m, n_blocks), F32),
        addr_space="lds",
    )

    lane_decode = decode_mfma_lanes(b, atom, lane)
    mfmas_m = spec.mfmas_m
    mfmas_n = spec.mfmas_n
    mfmas_m_down = spec.mfmas_m_down
    mfmas_n_down = spec.mfmas_n_down
    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * atom.m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * atom.n))

    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one_f32 = b.const_f32(1.0)
    c_fp8_max = b.const_f32(FP8_MAX)
    c_floor = b.const_f32(AMAX_FLOOR)

    c_group_k = b.const_i32(GROUP_K)
    c_hgk = b.const_i32(HGK)
    c_threads = b.const_i32(spec.block_size)
    _c_n_blocks = b.const_i32(n_blocks)
    _c_tile_n = b.const_i32(tile_n)

    def _emit_body() -> None:
        # ---- STAGE 1a: gate + up fp8 GEMM -> f32 (per-128-block dequant) ----
        # Per warp atom (mi, ni) -> one gate_dq / up_dq vector. The contraction
        # group GEMM is run per warp-atom output position; the lane decode and
        # tile bases select the rows/cols this lane owns.
        # COMBINATION lever: one fused K-loop per mi row covering ALL ni cells
        # (shared A read + register-double-buffered B prefetch + wave-pair
        # odd/even MFMA interleave). gate_list/up_list keep the row-major
        # (mi, ni) ordering the downstream Pass A / down stage expects.
        gate_list = []
        up_list = []
        # DTLA bundle (GOAL 1): per-wave LDS base + read-row base for the
        # direct-to-LDS gate+up B staging. The DMA dst i64 = smem_addr_of +
        # warp_id * (DTLA_SLOTS*wave_size*b_per_lane); the read row base =
        # warp_id * DTLA_SLOTS*wave_size -- same byte (packed, b_per_lane/row).
        # L8 flag (use_dtla): construct the DTLA bundle ONLY when direct-to-LDS
        # gate/up staging is enabled (default True = best). When False, the
        # bundle (and its ops) are skipped entirely and the legacy
        # global->VGPR->MFMA path runs. Default-True construction is byte-
        # identical to the pre-flag kernel.
        if spec.use_dtla:
            bstage_base_i64 = b.smem_addr_of(BStage_smem)
            warp_rows = DTLA_SLOTS * DTLA_CHUNKS * spec.wave_size
            warp_wave_bytes = warp_rows * DTLA_CHUNK
            wave_lds_base = b.smem_ptr_add(
                bstage_base_i64,
                b.sext(b.mul(warp_id, b.const_i32(warp_wave_bytes)), I64),
            )
            warp_row_base = b.mul(warp_id, b.const_i32(warp_rows))
            dtla_bundle = {
                "view": bstage_view,
                "base": wave_lds_base,
                "warp_row_base": warp_row_base,
                "lane": lane,
                "wave_size": spec.wave_size,
                "depth": _dtla_depth,
            }
        else:
            dtla_bundle = None

        # Cooperative-shared B bundle. Two runtime pieces, and the split between
        # them is deliberate: the STAGING half keeps ``half`` (gate vs up)
        # compile-time so the weight pointer can be selected at build time, and
        # pushes the only runtime term (this wave's n-cell range) into the LDS
        # base. The READ half likewise keeps the slot compile-time and puts
        # warp_n's contribution in the row base. Both sit inside _dtla_*'s
        # existing addressing, so no new LDS layout is introduced.
        if spec.coop_b_lds:
            coop_base_i64 = b.smem_addr_of(BCoop_smem)
            slot_bytes = DTLA_CHUNKS * spec.wave_size * DTLA_CHUNK
            ni_per_wave = mfmas_n_all // n_warps
            warp_stage_off = b.mul(warp_id, b.const_i32(ni_per_wave * slot_bytes))
            coop_bundle = {
                "view": coop_view,
                # gate slots start at 0, up slots at mfmas_n_all
                "stage_bases": [
                    b.smem_ptr_add(
                        coop_base_i64,
                        b.sext(
                            b.add(
                                warp_stage_off,
                                b.const_i32(half * mfmas_n_all * slot_bytes),
                            ),
                            I64,
                        ),
                    )
                    for half in range(2)
                ],
                # the n-cells THIS wave stages (both halves)
                "stage_n_bases": [
                    b.add(
                        gu_n_off,
                        b.mul(
                            b.add(
                                b.mul(warp_id, b.const_i32(ni_per_wave)),
                                b.const_i32(j),
                            ),
                            b.const_i32(atom.n),
                        ),
                    )
                    for j in range(ni_per_wave)
                ],
                "read_row_base": b.mul(
                    warp_n_idx, b.const_i32(mfmas_n * DTLA_CHUNKS * spec.wave_size)
                ),
                "n_cells_all": mfmas_n_all,
                "lane": lane,
                "wave_size": spec.wave_size,
            }
        else:
            coop_bundle = None
        for mi in range(mfmas_m):
            m_tile_base = b.add(
                block_m_off, b.add(warp_m_off, b.const_i32(mi * atom.m))
            )
            if not spec.use_fused_kloop:
                # Legacy per-(mi, ni) K-loop: one cell's operands live at a
                # time, weights global->VGPR (no DTLA), same shape as the down
                # GEMM's emitter.
                for ni in range(mfmas_n):
                    g_dq, u_dq = _emit_fp8_gateup_group_gemm(
                        b,
                        A=A,
                        WGate=WGate,
                        WUp=WUp,
                        AScale=AScale,
                        WGateScale=WGateScale,
                        WUpScale=WUpScale,
                        atom=atom,
                        lane_decode=lane_decode,
                        m_tile_base=m_tile_base,
                        n_tile_base=b.add(
                            gu_n_off, b.add(warp_n_off, b.const_i32(ni * atom.n))
                        ),
                        K=K,
                        stride_a_scale=stride_a_scale,
                        stride_gate_scale=stride_gate_scale,
                        stride_up_scale=stride_up_scale,
                        tag=f"{mi}_{ni}",
                        swizzle_b=spec.swizzle_gu,
                        wave_size=spec.wave_size,
                    )
                    gate_list.append(g_dq)
                    up_list.append(u_dq)
                continue
            n_tile_bases = [
                b.add(gu_n_off, b.add(warp_n_off, b.const_i32(ni * atom.n)))
                for ni in range(mfmas_n)
            ]
            g_dqs, u_dqs = _emit_fp8_gateup_fused_kloop(
                b,
                A=A,
                WGate=WGate,
                WUp=WUp,
                AScale=AScale,
                WGateScale=WGateScale,
                WUpScale=WUpScale,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                n_tile_bases=n_tile_bases,
                K=K,
                stride_a_scale=stride_a_scale,
                stride_gate_scale=stride_gate_scale,
                stride_up_scale=stride_up_scale,
                tag=f"{mi}",
                dtla=dtla_bundle,
                cadence=cadence,
                prefetch_depth=_dtla_depth,
                sched_group=spec.window_group,
                sched_mode=spec.window_sched,
                swizzle_b=spec.swizzle_gu,
                wave_size=spec.wave_size,
                coop_b=coop_bundle,
            )
            gate_list.extend(g_dqs)
            up_list.extend(u_dqs)

        def _emit_split_gateup_store(write_scale: bool = True) -> None:
            """Publish the fp8 tile (and, when it owns it, its scale) to HBM.

            Layout: Inter is [num_m_blocks*tile_m, N] fp8 row-major, so this
            CTA owns rows [block_m_off, +tile_m) x cols [gu_n_off, +tile_n).
            Scales are row-uniform by construction, so only
            [num_m_blocks, N/GROUP_K] are stored, not one per row. Under
            ``static_inter_scale`` the host supplied those scales, so writing
            them back would just copy the input over itself.
            """
            c_tile_n16 = b.const_i32(tile_n // 16)
            store_sweep = b.scf_for_iter(
                tid,
                b.const_i32((tile_m * tile_n) // 16),
                c_threads,
                [],
                iv_name="wcell",
            )
            with store_sweep as wcell:
                row = b.div(wcell, c_tile_n16)
                col16 = b.mul(b.mod(wcell, c_tile_n16), b.const_i32(16))
                v16 = b.smem_load_vN(
                    fp8_view.base, row, col16, dtype=FP8E4M3, n=16
                )
                g_off = b.add(
                    b.mul(b.add(block_m_off, row), N), b.add(gu_n_off, col16)
                )
                b.global_store_vN(Inter, g_off, v16, 16)
                b.scf_yield()
            if not write_scale:
                return
            # One f32 per (m_block, inter block); n_blocks is 1 at the shipped
            # tile_n_inter=128, so this is a single lane doing a single store.
            with b.scf_if(b.cmp_lt(tid, _c_n_blocks)):
                sc = b.smem_load_vN(scale_view.base, c0, tid, dtype=F32, n=1)
                s_off = b.add(
                    b.mul(
                        b.div(block_m_off, c_block_m), b.div(N, c_group_k)
                    ),
                    b.add(b.div(gu_n_off, c_group_k), tid),
                )
                b.global_store_vN(InterScale, s_off, sc, 1)

        # ---- STAGE 1b Pass A (FUSED): SiLU(gate)*up -> f32 LDS + amax -----
        # pyisa G_dyn_quant granularity = per-token-block (sub_x=32), NOT
        # per-row: ONE dynamic scale per 128-inter-block, reduced over ALL
        # tile_m rows of the block. This row-uniform-within-block scheme is
        # MANDATORY because the down-GEMM fold applies a single per-lane scalar
        # (indexed by the A-input Hidden row ``m_in_atom``) to output slots that
        # span 4 DIFFERENT output rows; only a row-uniform block scale stays
        # correct under that fold (the same constraint as the activation scale,
        # BUILD_SPEC_FP8 OPEN RISK #1). The scale is broadcast to every row of
        # ``scale_view`` so any ``m_in_atom`` read returns the block scale.
        #
        # LEVER (fuse-quant): the per-block amax is reduced from registers as
        # part of Pass A instead of a separate full-block LDS re-read sweep.
        # Each lane returns the abs-max over the cells it owns; a 64-lane
        # butterfly max collapses that to the warp amax, and since an entire
        # warp's N-extent lives in ONE 128-inter block, the two warps that
        # share a block (2*blk, 2*blk+1) are combined below. This removes the
        # 4096-iter per-thread re-read scan and one whole barrier.
        if static_scale:
            # One pass, one barrier. InterScale is an input here, so the
            # divisor is available before the first cell is written and the
            # scratch/amax/re-read machinery below is all dead.
            inv_scale = b.rcp_fast(
                b.global_load_f32(
                    InterScale,
                    b.add(
                        b.mul(b.div(block_m_off, c_block_m), b.div(N, c_group_k)),
                        b.div(gu_n_off, c_group_k),
                    ),
                )
            )
            _store_hidden_fp8_static_pass(
                b,
                atom=atom,
                gate_list=gate_list,
                up_list=up_list,
                fp8_view=fp8_view,
                warp_m_off=warp_m_off,
                warp_n_off=warp_n_off,
                lane=lane,
                mfmas_m=mfmas_m,
                mfmas_n=mfmas_n,
                one_f32=one_f32,
                c_neg_log2e=c_neg_log2e,
                inv_scale=inv_scale,
            )
            b.sync()
            _emit_split_gateup_store(write_scale=False)
            return

        amax_lane = _store_hidden_f32_pass(
            b,
            atom=atom,
            gate_list=gate_list,
            up_list=up_list,
            f32_view=f32_view,
            warp_m_off=warp_m_off,
            warp_n_off=warp_n_off,
            lane=lane,
            mfmas_m=mfmas_m,
            mfmas_n=mfmas_n,
            one_f32=one_f32,
            c_neg_log2e=c_neg_log2e,
            c_floor=c_floor,
        )
        # Butterfly max over the whole wave: the block amax has to be exact, so
        # every lane must see every other lane's partial. Halving strides cover
        # a power-of-two wave in log2(wave_size) steps (1,2,4,... at wave 64).
        amax_warp = amax_lane
        xm = 1
        while xm < spec.wave_size:
            amax_warp = b.fmax(amax_warp, b.warp_shuffle_xor(amax_warp, xm))
            xm *= 2
        # Lane 0 of each warp publishes its partial.
        with b.scf_if(b.cmp_eq(lane, c0)):
            b.smem_store_vN(WarpAmax_smem, [warp_id], amax_warp, 1)
        b.sync()

        # ---- STAGE 1b combine: per-block amax from its warps' partials -----
        # Block ``blk`` is held by warps ``mi*warp_n + (warps_per_block*blk+wo)``
        # for every mi in [0, warp_m) -- the whole M column of the warp grid,
        # because the scale is row-uniform. Combine + scale + broadcast to every
        # row so any ``m_in_atom`` read returns the block scale. At warp_m=1 this
        # is the original consecutive-warp run, op for op.
        sweep = b.scf_for_iter(
            tid, b.const_i32(n_blocks), c_threads, [], iv_name="cell"
        )
        with sweep as blk:
            w0 = b.mul(blk, b.const_i32(warps_per_block))
            amax = None
            for mi_w in range(spec.warp_m):
                for wo in range(warps_per_block):
                    off = mi_w * spec.warp_n + wo
                    wid = w0 if off == 0 else b.add(w0, b.const_i32(off))
                    pw = b.vec_extract(
                        b.smem_load_vN(WarpAmax_smem, wid, dtype=F32, n=1), 0
                    )
                    amax = pw if amax is None else b.fmax(amax, pw)
            scale = b.fmul(amax, b.rcp(c_fp8_max))  # amax / 448
            row_bc = b.scf_for_iter(c0, c_block_m, b.const_i32(1), [], iv_name="rb")
            with row_bc as rr:
                b.smem_store_vN(scale_view.base, [rr, blk], scale, 1)
                b.scf_yield()
            b.scf_yield()
        b.sync()

        # ---- STAGE 1b Pass C: quantize f32 Hidden -> fp8 LDS (PACKED) -----
        # PACKED-CVT lever (pyisa G_dyn_quant uses v_cvt_pk_fp8_f32, 4 f32 ->
        # 4 fp8 per pair of cvt ops, NOT the scalar one-at-a-time cvt). Each
        # thread now owns a CONTIGUOUS run of 4 columns (same row): one
        # ds_read_b128 of <4 x f32>, one rcp of the block scale, a packed
        # multiply, ONE v_cvt_pk_fp8_f32x4, and one packed fp8 store. This
        # quarters the Pass C op count (4096 scalar -> 1024 packed iters) and
        # matches pyisa's cvt cadence WITHOUT raising register pressure (only
        # 4 f32 live per iter, vs the in-register-fold variant that held the
        # whole lane tile across the amax combine and crushed occupancy).
        # GROUP_K (128) is divisible by 4, so all 4 cols share one block scale.
        total_q4 = (tile_m * tile_n) // 4
        c_tile_n4 = b.const_i32(tile_n // 4)
        qsweep = b.scf_for_iter(
            tid, b.const_i32(total_q4), c_threads, [], iv_name="qcell"
        )
        with qsweep as qcell:
            row = b.div(qcell, c_tile_n4)
            col4 = b.mul(b.mod(qcell, c_tile_n4), b.const_i32(4))
            blk = b.div(col4, c_hgk)
            hv4 = b.smem_load_vN(f32_view.base, row, col4, dtype=F32, n=4)
            sc = b.smem_load_vN(scale_view.base, row, blk, dtype=F32, n=1)
            # Fast reciprocal (~1 ulp): the divide-by-block-scale quant step
            # feeds an fp8 e4m3 cvt (~2 mantissa bits), so 1-ulp f32 error is
            # far below the fp8 rounding floor; matches aiter's quant cadence.
            inv = b.rcp_fast(b.vec_extract(sc, 0))
            scaled = b.vec_pack(
                [b.fmul(b.vec_extract(hv4, j), inv) for j in range(4)], F32
            )
            q4 = b.cvt_pk_fp8_f32x4(scaled)
            b.smem_store_vN(fp8_view.base, [row, col4], q4, 4)
            b.scf_yield()
        b.sync()

        # ---- STAGE 2: down fp8 GEMM (LDS-A) -> dequant -> weighted atomic Y --
        # grid.x split the inter contraction; this TG owns the inter slice at
        # gu_n_off and produces a PARTIAL Y over the WHOLE H_out, tiled in
        # tile_n_down chunks. Per output tile, run the LDS-A down group GEMM
        # (contracting this TG's inter slice with the group-accumulator dequant
        # fold by hidden_dyn_scale * down_b_scale) then atomic-add the weighted,
        # token-validity-masked partial into Y.
        #
        # The Hidden A operand (fp8 in Hidden_smem) and its per-128-block dynamic
        # scales (HiddenScale_smem) are already resident from STAGE 1. The
        # contraction extent is the LOCAL inter slice tile_n_inter; A reads local
        # inter columns, W_down reads GLOBAL inter columns (full inter row stride
        # N) at this slice's base. inter_blk_base = gu_n_off // GROUP_K.
        if split_gateup:
            # ---- PARTIAL FUSION: publish the intermediate, skip stage 2 ----
            # The fused kernel's stage 2 can only parallelize over the inter
            # (reduction) axis, because a token block's intermediate is pinned
            # to this CTA's LDS. That costs 12.6 MB of cross-slice fp32
            # atomics. Writing the intermediate out lets the down GEMM run as
            # its own launch tiled over H_out, which reduces all of I in-block
            # and writes 2.1 MB instead. See build_moe_split_down_fp8.
            #
            _emit_split_gateup_store()
            return

        inter_blk_base = b.div(gu_n_off, c_group_k)
        down_for = b.scf_for_iter(
            c0, H_out, b.const_i32(spec.tile_n_down), [], iv_name="ho"
        )
        with down_for as ho:
            down_warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m_down * atom.m))
            down_warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n_down * atom.n))
            down_list = []
            for mi in range(mfmas_m_down) if spec.down_fused_cells else ():
                m_row_base = b.add(down_warp_m_off, b.const_i32(mi * atom.m))
                n_tile_bases_down = [
                    b.add(ho, b.add(down_warp_n_off, b.const_i32(ni * atom.n)))
                    for ni in range(mfmas_n_down)
                ]
                down_list.extend(
                    _emit_fp8_down_fused_cells(
                        b,
                        a_view=fp8_view,
                        WDown=WDown,
                        WDownScale=WDownScale,
                        atom=down_atom,
                        lane_decode=lane_decode,
                        n_tile_bases=n_tile_bases_down,
                        scale_view=scale_view,
                        inter_slice=tile_n,
                        inter_full=N,
                        inter_col_base=gu_n_off,
                        stride_down_scale=stride_down_scale,
                        m_row_base=m_row_base,
                        hidden_group_k=HGK,
                        tag=f"dm{mi}",
                        cadence=cadence,
                        prefetch_depth=spec.down_depth,
                        sched_group=spec.down_group,
                        sched_mode=spec.window_sched,
                        swizzle_b=spec.swizzle_down,
                        wave_size=spec.wave_size,
                    )
                )
            for mi in range(mfmas_m_down) if not spec.down_fused_cells else ():
                for ni in range(mfmas_n_down):
                    # Down output column base (along H_out) for this warp-atom.
                    n_tile_base = b.add(
                        ho,
                        b.add(down_warp_n_off, b.const_i32(ni * atom.n)),
                    )
                    # A-read m-base for this atom: warp m-offset + mi*atom.m so
                    # atom mi reads its own Hidden LDS rows (the correctness fix).
                    m_row_base = b.add(down_warp_m_off, b.const_i32(mi * atom.m))
                    d_dq = _emit_fp8_down_group_gemm(
                        b,
                        a_view=fp8_view,
                        WDown=WDown,
                        WDownScale=WDownScale,
                        atom=atom,
                        lane_decode=lane_decode,
                        n_tile_base=n_tile_base,
                        scale_view=scale_view,
                        inter_slice=tile_n,
                        inter_full=N,
                        inter_blk_base=inter_blk_base,
                        stride_down_scale=stride_down_scale,
                        m_row_base=m_row_base,
                        tag=f"d{mi}_{ni}",
                        cadence=cadence,
                        swizzle_b=spec.swizzle_down,
                        wave_size=spec.wave_size,
                    )
                    down_list.append(d_dq)
            # Barrier before the next H_out tile reuses Hidden_smem reads
            # (read-only here, but keep the scf.for body well-formed).
            _emit_down_atomic_reduce(
                b,
                atom=atom,
                down_list=down_list,
                warp_m_off=down_warp_m_off,
                warp_n_off=down_warp_n_off,
                lane=lane,
                mfmas_m=mfmas_m_down,
                mfmas_n=mfmas_n_down,
                block_m_off=block_m_off,
                ho_off=ho,
                H_out=H_out,
                SortedTokenIds=SortedTokenIds,
                SortedWeights=SortedWeights,
                Y=Y,
                tokens=tokens,
            )
            b.scf_yield()
        _ = (M, stride_a, slot_size)

    if not persistent:
        # ---- DEFAULT one-shot path (byte-identical) --------------------------
        # Empty tail block (BlockExpertIds == -1) skips all work.
        with b.scf_if(b.cmp_ge(expert_idx, c0)):
            _emit_body()
    else:
        # ---- PERSISTENT path: grid-stride over linear work-ids ---------------
        # p = this persistent block's id; walk w = p, p+P, ... < total_work.
        # bx = w % grid_x (inter-tile, fast index -> same-by weight reuse);
        # by = w // grid_x (active m-block). Each item re-selects its expert +
        # rebased pointers + offsets (_select_item) and runs the FULL mega body
        # under the empty-block guard. ONE inter-item barrier at the TOP of the
        # body guards the LDS hand-off: the previous item's down-GEMM READS
        # Hidden_smem/HiddenScale_smem while the current item's Pass A WRITES
        # them -- without it a fast warp could clobber LDS a slow prior-item warp
        # still reads. (The 3 intra-item syncs are sufficient WITHIN an item.)
        # Accumulators are scf.for iter-args re-zeroed by re-entering the body;
        # the dyn-quant LDS scratch is write-before-read per item; the Y
        # atomic-add is commutative + host-zeroed once, so the strided
        # exactly-once walk is iteration-safe.
        p = b.block_id_x()
        wloop = b.scf_for_iter(p, p_total_work, p_P, [], iv_name="witem")
        with wloop as w:
            bx = b.mod(w, p_grid_x)
            by = b.div(w, p_grid_x)
            _select_item(by, bx)
            # Inter-item LDS hand-off barrier (the ONE new barrier). Block-uniform
            # (every thread runs the same trip count -> same barrier count).
            b.sync()
            with b.scf_if(b.cmp_ge(expert_idx, c0)):
                _emit_body()
            b.scf_yield()

    b.ret()
    return b.kernel
