# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 MXFP6 GEMM tile-config library: configs -> kernel families -> default ``Mxfp6GemmSpec``.

Conventions (hipBLASLt, column-major): ``D (m x n) = op(A) (m x k) op(B) (k x n)`` with hipBLASLt A = the weights
and B = the activations; the column-major D is our row-major ``C[M, N] = A[M, K] B[N, K]^T`` with our A = the
activations (M = tokens = hipBLASLt n) and our B = the weights (N = hipBLASLt m). A config ``MT0 x MT1 x DepthU``
therefore tiles the weights by MT0 (``tile_n``) and the tokens by MT1 (``tile_m``); DepthU is the K per main-loop
iteration.

Transposes (first letter opA = weights, second opB = activations): the weights are pre-shuffled offline (opA only
changes the pre-shuffle), so two kernel families per tile serve the four columns:

* ``ak`` -- activation K-contiguous (opB = N: NN, TN): canonical dense codes, stage ring (``a_layout="canonical"``);
* ``am`` -- activation M-contiguous (opB = T: NT, TT): ``[K][M]`` dense codes read with ``ds_read_b96_tr_b6``
  (``a_layout="mcontig"``).

Both families read the weights as pre-tiled c0c1 images and both operands' E8M0 scales as per-wave images
(``scale_layout="wtile"``; the activation scales are repacked per call from the raw ``[M][K/32]`` bytes, which an
M-contiguous producer writes the same way: scales stay per 32 along K).

:func:`family_spec` is the first-build default per tile and family (``tile_family`` names the structural family);
per-family searches tune every ``Mxfp6GemmSpec`` knob from there.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from .gemm import (
    Mxfp6GemmSpec,
    acc_regs,
    library_k_ref,
    lds_bytes,
    scale_img_bytes,
    validate_spec,
)

__all__ = ["FAMILIES", "LIBRARY", "TRANSPOSES", "configs", "family_spec", "tile_family", "wave_layout"]

_DU256 = (
    "32x128 32x256 32x384 32x512 64x64 64x128 64x192 64x256 64x320 64x384 64x448 64x512 96x128 96x256 96x384 "
    "128x32 128x64 128x96 128x128 128x192 128x224 128x256 128x320 128x384 128x448 160x256 160x384 192x64 192x128 "
    "192x192 192x256 192x320 192x384 224x256 256x32 256x64 256x96 256x128 256x160 256x192 256x256 256x288 256x320 "
    "288x256 320x64 320x128 320x192 320x256 384x32 384x64 384x96 384x128 384x192 448x64 448x128 512x32 512x64"
)
_DU512 = "32x32 32x128 64x64 64x128 64x192 96x128 128x32 128x64 128x96 128x128 128x160 160x128 192x64"
#: (MT0, MT1, DepthU) of every library config (user spec 2026-11-11; "x2" entries appear once)
LIBRARY: Tuple[Tuple[int, int, int], ...] = tuple(
    [tuple(map(int, t.split("x"))) + (256,) for t in _DU256.split()]
    + [tuple(map(int, t.split("x"))) + (512,) for t in _DU512.split()])
#: kernel family -> activation contiguous dimension
FAMILIES: Dict[str, str] = {"ak": "K", "am": "M"}
#: hipBLASLt transposes (opA opB) -> family
TRANSPOSES: Dict[str, str] = {"NN": "ak", "TN": "ak", "NT": "am", "TT": "am"}

#: knobs shared by every family default: prefetch fragments with the acquire at slot 0 and reads every 2nd MFMA
#: slot, the first refill one slot after the acquire, an lgkmcnt(0) per tile end, A as MFMA src0 (DPP-paired
#: dword stores), the canonical stage ring's interleaved DTLs, per-wave scale images in the LDS stage, the grouped
#: XCD-chunked CTA order, KiB-aligned stage allocations (no cross-stage DMA waits), runtime shapes, the direct
#: compiler
_COMMON = dict(operand_layout="c0c1", canon_dtl="interleave", frag_mode="prefetch", read_every=2, read_delay=0,
               dtl_delay=1, tile_lgkm0=True, swap_ab=False, scale_layout="wtile", scale_src="lds",
               cta_order="grouped", xcd_block="none", runtime_shape=True, stage_kib=True, required_compiler="rust")
#: direct-compiler levers of the family builds (best_mixed's + dsb96 / trtag for the transposing reads; trtag
#: needs a compiler >= out-68365943; bankocc -- keep the AGPR accumulators when the LDS would allow a second work
#: group (direct-operand kernels) -- needs >= out-68372344; config_spec's dlevers need >= out-68371726)
LEVERS = "scalesep,dsrebase,dshoist,m0fill,m0cse,dmamanualringw,m0share,ldsgroup,litshare,soffshare,dsb96,trtag,bankocc"
#: am lane: the ``am`` family's M-contiguous A DMA. "x3": 12-byte b96 LDS DMAs (no 4-byte overlap per 16-byte slot;
#: faster on every am config measured, notes/tile-library.md "am lane"); "x4": the 16-byte overlapping DMAs. "x3" needs
#: a direct compiler with the b96 LDS DMA (notes/requests-rust.md R11; out-68373227 has it). x3 / x4 on the shipped
#: dispatch kernel of all 70 am configs (jobs 68371919 / 68372066): min 1.002, median 1.029, max 1.204.
AM_MC_DMA = "x3"
LDS = 160 * 1024


def self_levered(spec: Mxfp6GemmSpec, levers: str = LEVERS) -> Mxfp6GemmSpec:
    """``spec`` with its direct-compiler levers in the IR (``dlevers``, default off): ``self_levered(family_spec(...))``
    builds the object ``family_spec(...)`` gives under ``ROCKE_DIRECT_LEVERS=LEVERS``, without the environment.
    Needs a direct compiler that reads kernel ``dlevers`` (rustgemm crate-up, out-<job> in notes/requests-rust.md)."""
    from .gemm import with_dlevers

    return with_dlevers(spec, levers)


#: the record kernel's A path (best_mixed): canonical A in the line2m ring (2 line slots + the head line's remnant),
#: B c0c1 in a 3-stage ring, the loop body 24 tiles; ``family_spec(..., **LINE2M)`` for the K-contiguous family
LINE2M = dict(canon_ring="line2m", stages=3, unroll=8, line_reads_first=True, line_release=34, dtl_spread=4,
              mini_copy_gap=4, canon_read="b96x2", swizzle="none", scale_slot=32, store_nt=True, peel_first=True,
              m_outer=False, dtl_imm=True, rsrc_once=True)


def configs() -> List[Tuple[int, int, int]]:
    return list(LIBRARY)


def tile_family(mt0: int, mt1: int, du: int) -> str:
    """Structural family of a config (MT0 = weight rows = tile_n, MT1 = tokens = tile_m)."""
    tm, tn = mt1, mt0
    if du == 512:
        return "depthu512"
    if tm * tn > 4 * 64 * 256:  # 4 waves cannot hold the accumulators in their AGPRs
        return "large"
    if tm * tn <= 128 * 64:
        return "tiny"
    if tn >= 3 * tm:
        return "tall"  # many weight rows, few tokens (decode-like)
    if tm >= 3 * tn:
        return "wide"  # many tokens, few weight rows
    return "square"


def wave_layout(tm: int, tn: int, acc_cap: int = 256) -> Tuple[int, int]:
    """(waves_m, waves_n): 4 waves when they hold the accumulators (<= 256 per wave), else 8 (<= 192 each), then
    the layouts with >= 4 MFMA tiles per wave, then the least LDS read traffic (each wave reads its A rows and B
    rows: waves_n tm + waves_m tn), then the squarer wave tile (2 / 1 waves only if nothing else fits)."""
    best = None
    for nw in (4, 8, 2, 1):
        for wm in (1, 2, 4, 8):
            if nw % wm:
                continue
            wn = nw // wm
            if tm % (16 * wm) or tn % (16 * wn):
                continue
            fm, fn = tm // (16 * wm), tn // (16 * wn)
            if fm > 16 or fn > 16 or tm * tn // (64 * nw) > acc_cap:
                continue
            # 8 waves = 2 per SIMD, 256 registers each: jit fragments (a few in flight) + addresses / scales +
            # accumulators (compile check: 256x288 2x4 jit = 200 registers, no spill)
            if nw == 8 and tm * tn // (64 * nw) > 192:
                continue
            # measured MFMA-only ceilings (notes/tile-library.md "Ceilings"): 4-wave work-groups are never worse than
            # 1- / 2-wave ones; a wave with < 4 MFMA tiles needs >= 2 waves / SIMD (occupancy), not fewer waves
            key = ({4: 0, 8: 1, 2: 2, 1: 3}[nw], 0 if fm * fn >= 4 else 1, wn * tm + wm * tn, abs(fm - fn))
            if best is None or key < best[0]:
                best = (key, (wm, wn))
    if best is None:
        raise ValueError(f"no wave layout for tile_m {tm} x tile_n {tn}")
    return best[1]


#: (MT0, MT1, DepthU, family) whose default loads the wtile scales straight to VGPRs (``scale_src="vgpr"``): the
#: configs where that won by >= 3 % over the LDS scale image on the folded 1 x ~16384 x ~8192 x 8192 problem (jobs
#: 68368234 / 68368235, n01-09 / n14-05, every config's two arms on one GPU; DU256 / DU512 twins agree within 1 %).
#: Everywhere else the LDS image wins (by up to 40 %), so it stays the default there.
VGPR_SCALES = frozenset({
    (32, 128, 256, "am"), (32, 128, 512, "am"), (64, 128, 256, "am"), (64, 128, 512, "am"), (32, 256, 256, "am"),
    (128, 64, 256, "am"), (128, 64, 512, "am"), (64, 256, 256, "am"), (128, 32, 256, "am"), (128, 32, 512, "am"),
    (64, 64, 256, "am"), (64, 64, 512, "am"),
    (64, 128, 256, "ak"), (64, 128, 512, "ak"), (160, 384, 256, "ak"),
})


def family_spec(mt0: int, mt1: int, du: int, family: str = "ak", fmt: str = "e2m3", *,
                k_target: Optional[int] = 8192, ring: str = "auto", waves: Optional[Tuple[int, int]] = None,
                **over) -> Mxfp6GemmSpec:
    """The first-build default ``Mxfp6GemmSpec`` of config ``MT0 x MT1 x DepthU`` for ``family`` (ak / am).

    * waves: ``wave_layout``; occupancy: the most work-groups per CU (<= 4 waves per SIMD) the register estimate
      allows whose LDS share still holds a 3- or 4-tile ring (else 1 work-group, 2 tiles at least);
    * LDS ring: 4 128-K tiles when they fit the per-work-group LDS share (160 KiB / occupancy), else 3, else 2;
      ``depth_u`` 128 (one acquire per 128-K tile; ``depth_u`` = DepthU is a search knob);
    * the loop body holds whole DepthU periods (unroll), so K = k_ref + 128 U j (rt_tail fixed) with k_ref in
      ``k_target``'s residue class;
    * runtime shapes, batch on grid z (``batch`` = the reference count), the reference shape m / n = 16 tiles;
    * ``ring="line2m"`` (``ak``, tile_m a multiple of 128, <= 4 waves): the record kernel's A path -- canonical A in
      the line2m ring, B in a stage ring -- with the slot knobs scaled to the tile's MFMAs per 128-K tile
      (``line2m_knobs``); ``ring="stage"``: the canonical stage ring; ``ring="auto"`` (default): line2m wherever it
      applies and a ring fits (it won every eligible ak config, folded A/B 2026-11-12), else stage.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {tuple(FAMILIES)}")
    tm, tn = mt1, mt0
    wm, wn = waves if waves else wave_layout(tm, tn)  # waves: an explicit (waves_m, waves_n)
    nw = wm * wn
    base = Mxfp6GemmSpec(**{**_COMMON, "tile_m": tm, "tile_n": tn, "waves_m": wm, "waves_n": wn, "elem_fmt": fmt,
                            "a_layout": "mcontig" if family == "am" else "canonical",
                            "mc_dma": AM_MC_DMA if family == "am" else "x4",
                            "m": 16 * tm, "n": 16 * tn})
    # occupancy: as many work-groups per CU as the registers allow (estimate: accumulators + two prefetch fragment
    # sets + 48), up to 4 waves per SIMD; then the deepest ring (4 / 3 tiles) the per-work-group LDS share holds
    fm, fn = tm // (16 * wm), tn // (16 * wn)
    regs = -(-(acc_regs(base) + 12 * (fm + fn) + 48) // 8) * 8
    occ_cap = 6 if fm * fn < 4 else 4  # a 1-3 MFMA-tile wave fills the pipe at 5 waves / SIMD (t1x1 probe)
    occ_max = max(1, min(occ_cap, (512 // regs) * 4 // nw)) if nw <= 4 else 1
    chosen = None
    for occ in range(occ_max, 0, -1):
        for S in ((4, 3) if occ > 1 else (4, 3, 2)):
            s = replace(base, stages=S, occupancy=occ)
            if lds_bytes(s) * occ <= LDS:
                chosen = s
                break
        if chosen is not None:
            break
    if chosen is None:
        raise ValueError(f"{mt0}x{mt1}: no LDS ring fits")
    ks = du // 128
    unroll = ks // _gcd(chosen.stages, ks)
    n_mfma = (tm // wm // 16) * (tn // wn // 16)
    chosen = replace(chosen, unroll=unroll, acq_slot=0 if n_mfma >= 2 else -1)
    # one buffer descriptor per operand built once, the K tile's offset in soffset (pre-tiled B; the canonical A of
    # the stage ring too -- the M-contiguous A keeps a per-tile exact-range descriptor)
    chosen = replace(chosen, rsrc_once=True, canon_rsrc_once=family == "ak")
    if nw <= 4 and acc_regs(chosen) > 256:  # more than the 256 AGPRs: the rest of the accumulators in VGPRs
        chosen = replace(chosen, acc_vgpr_spill=True)
    if nw == 8:  # 2 waves / SIMD: one fragment set (jit reads, 4 MFMAs ahead), accumulators in VGPRs
        chosen = replace(chosen, frag_mode="jit", acq_slot=-1, acc_agpr=False)
    if ring not in ("auto", "stage", "line2m"):
        raise ValueError("ring must be auto, stage or line2m")

    def finish(s: Mxfp6GemmSpec) -> Mxfp6GemmSpec:
        if (mt0, mt1, du, family) in VGPR_SCALES and "scale_src" not in over:
            s = replace(s, scale_src="vgpr")
        s = replace(s, **over)
        per = 4 if s.scale_raw else s.wtile_period  # K tiles per VGPR scale load
        if per > 1 and "unroll" not in over:  # the body: lcm(stages, period) tiles x whole DepthU periods
            lp = s.stages * per // _gcd(s.stages, per)
            s = replace(s, unroll=ks // _gcd(lp, ks))
        s = replace(s, k=library_k_ref(s, k_target=k_target) if s.runtime_shape else s.k)
        ok, why = validate_spec(s)
        if not ok:
            raise ValueError(f"{mt0}x{mt1}x{du} {family}: {why}")
        return s

    if ring == "stage" or (ring == "auto" and not line2m_ok(mt0, mt1, family, waves)):
        return finish(chosen)
    if ring == "line2m":
        return finish(_line2m(chosen, occ_max, ks))
    try:  # auto: line2m unless no line ring fits the LDS (64x512) or ``over`` excludes line rings (fill probes, ..)
        return finish(_line2m(chosen, occ_max, ks))
    except ValueError:
        return finish(chosen)


#: (MT0, MT1, DepthU, family) -> (g0, g1): the library config runs as a (g0 MT0) x (g1 MT1) work-group tile (g0 g1
#: output tiles per work group sharing their panel loads), with ``edge_tiles`` (any M >= its tile, N >= its tile).
#: Chosen where it beat the native tile by >= 3 % warm on the config's own S problem (jobs 68368621-23, 68368738-40;
#: every arm of a config on one GPU, verified before timing; ``tilek/mk_wgmult.py``). The comment: best / native.
WG_MULT: Dict[Tuple[int, int, int, str], Tuple[int, int]] = {
    (32, 32, 512, 'ak'): (2, 2),  # 1.75x
    (32, 32, 512, 'am'): (2, 2),  # 2.97x
    (32, 128, 256, 'ak'): (2, 2),  # 1.56x
    (32, 128, 256, 'am'): (2, 1),  # 1.80x
    (32, 128, 512, 'ak'): (2, 2),  # 1.58x
    (32, 128, 512, 'am'): (2, 2),  # 2.17x
    (32, 256, 256, 'ak'): (2, 1),  # 1.83x
    (32, 256, 256, 'am'): (2, 2),  # 2.64x
    (32, 384, 256, 'ak'): (2, 1),  # 1.79x
    (32, 384, 256, 'am'): (2, 1),  # 1.84x
    (32, 512, 256, 'ak'): (2, 1),  # 1.17x
    (32, 512, 256, 'am'): (2, 1),  # 1.82x
    (64, 64, 256, 'ak'): (2, 2),  # 2.07x
    (64, 64, 256, 'am'): (2, 2),  # 2.79x
    (64, 64, 512, 'ak'): (2, 2),  # 2.05x
    (64, 64, 512, 'am'): (2, 2),  # 2.99x
    (64, 128, 256, 'ak'): (2, 2),  # 1.49x
    (64, 128, 256, 'am'): (2, 2),  # 1.78x
    (64, 128, 512, 'ak'): (2, 2),  # 1.42x
    (64, 128, 512, 'am'): (2, 2),  # 1.96x
    (64, 192, 256, 'ak'): (2, 2),  # 1.91x
    (64, 192, 256, 'am'): (2, 2),  # 1.55x
    (64, 192, 512, 'ak'): (2, 2),  # 1.94x
    (64, 192, 512, 'am'): (2, 2),  # 1.55x
    (64, 256, 256, 'ak'): (2, 1),  # 1.58x
    (64, 256, 256, 'am'): (2, 2),  # 2.49x
    (64, 320, 256, 'ak'): (2, 1),  # 1.71x
    (64, 320, 256, 'am'): (2, 1),  # 1.31x
    (64, 384, 256, 'ak'): (2, 1),  # 1.62x
    (64, 384, 256, 'am'): (2, 1),  # 1.77x
    (64, 448, 256, 'ak'): (2, 1),  # 1.48x
    (64, 448, 256, 'am'): (2, 1),  # 1.78x
    (64, 512, 256, 'ak'): (2, 1),  # 1.78x
    (64, 512, 256, 'am'): (2, 1),  # 1.76x
    (96, 128, 256, 'ak'): (2, 2),  # 1.34x
    (96, 128, 256, 'am'): (2, 2),  # 1.82x
    (96, 128, 512, 'ak'): (2, 2),  # 1.34x
    (96, 128, 512, 'am'): (2, 2),  # 1.81x
    (96, 256, 256, 'ak'): (2, 1),  # 1.36x
    (96, 256, 256, 'am'): (2, 1),  # 1.32x
    (96, 384, 256, 'am'): (2, 1),  # 1.56x
    (128, 32, 256, 'ak'): (2, 2),  # 1.69x
    (128, 32, 256, 'am'): (2, 2),  # 2.35x
    (128, 32, 512, 'ak'): (2, 2),  # 1.69x
    (128, 32, 512, 'am'): (2, 2),  # 2.35x
    (128, 64, 256, 'ak'): (2, 2),  # 1.63x
    (128, 64, 256, 'am'): (2, 2),  # 2.08x
    (128, 64, 512, 'ak'): (2, 2),  # 1.63x
    (128, 64, 512, 'am'): (2, 2),  # 2.09x
    (128, 96, 256, 'ak'): (2, 2),  # 1.44x
    (128, 96, 256, 'am'): (2, 2),  # 1.50x
    (128, 96, 512, 'ak'): (2, 2),  # 1.44x
    (128, 96, 512, 'am'): (2, 2),  # 1.50x
    (128, 128, 256, 'ak'): (2, 2),  # 1.32x
    (128, 128, 256, 'am'): (2, 2),  # 1.57x
    (128, 128, 512, 'ak'): (2, 2),  # 1.32x
    (128, 128, 512, 'am'): (2, 2),  # 1.57x
    (128, 160, 512, 'ak'): (2, 1),  # 1.44x
    (128, 160, 512, 'am'): (2, 1),  # 1.47x
    (128, 192, 256, 'ak'): (2, 1),  # 1.42x
    (128, 192, 256, 'am'): (2, 1),  # 1.30x
    (128, 224, 256, 'ak'): (2, 1),  # 1.44x
    (128, 224, 256, 'am'): (2, 1),  # 1.18x
    (128, 256, 256, 'ak'): (2, 1),  # 1.30x
    (128, 256, 256, 'am'): (2, 1),  # 1.22x
    (128, 320, 256, 'ak'): (2, 1),  # 1.24x
    (128, 320, 256, 'am'): (2, 1),  # 1.53x
    (160, 128, 512, 'ak'): (2, 1),  # 1.17x
    (160, 128, 512, 'am'): (2, 1),  # 1.37x
    (160, 256, 256, 'am'): (2, 1),  # 1.18x
    (192, 64, 256, 'ak'): (2, 2),  # 1.75x
    (192, 64, 256, 'am'): (2, 2),  # 1.64x
    (192, 64, 512, 'ak'): (2, 2),  # 1.74x
    (192, 64, 512, 'am'): (2, 2),  # 1.63x
    (192, 128, 256, 'ak'): (2, 1),  # 1.25x
    (192, 128, 256, 'am'): (1, 2),  # 1.10x
    (192, 192, 256, 'ak'): (2, 1),  # 1.13x
    (192, 192, 256, 'am'): (2, 1),  # 1.20x
    (256, 32, 256, 'ak'): (1, 2),  # 1.84x
    (256, 32, 256, 'am'): (2, 2),  # 1.95x
    (256, 64, 256, 'ak'): (1, 2),  # 1.62x
    (256, 64, 256, 'am'): (2, 2),  # 1.59x
    (256, 96, 256, 'ak'): (1, 2),  # 1.26x
    (256, 96, 256, 'am'): (1, 2),  # 1.19x
    (256, 128, 256, 'ak'): (1, 2),  # 1.17x
    (256, 128, 256, 'am'): (2, 1),  # 1.07x
    (256, 160, 256, 'ak'): (2, 1),  # 1.07x
    (256, 160, 256, 'am'): (1, 2),  # 1.05x
    (320, 64, 256, 'ak'): (1, 2),  # 1.67x
    (320, 64, 256, 'am'): (1, 2),  # 1.49x
    (320, 128, 256, 'am'): (1, 2),  # 1.08x
    (384, 32, 256, 'ak'): (1, 2),  # 1.65x
    (384, 32, 256, 'am'): (1, 2),  # 1.78x
    (384, 64, 256, 'ak'): (1, 2),  # 1.64x
    (384, 64, 256, 'am'): (1, 2),  # 1.27x
    (384, 96, 256, 'ak'): (1, 2),  # 1.08x
    (384, 96, 256, 'am'): (1, 2),  # 1.12x
    (448, 64, 256, 'ak'): (1, 2),  # 1.32x
    (448, 64, 256, 'am'): (1, 2),  # 1.30x
    (512, 32, 256, 'ak'): (1, 2),  # 1.79x
    (512, 32, 256, 'am'): (1, 2),  # 1.85x
    (512, 64, 256, 'ak'): (1, 2),  # 1.75x
    (512, 64, 256, 'am'): (1, 2),  # 1.73x
}


_JIT = {"frag_mode": "jit", "acq_slot": -1, "ring": "stage"}
_BD14 = {**_JIT, "waves": (1, 4), "b_direct": True, "stages": 4}  # waves own their B rows: B straight to VGPRs, A ring 4
_W22 = {**_JIT, "waves": (2, 2)}  # 4 waves, accumulators past 256 in VGPRs (acc_vgpr_spill)

#: (MT0, MT1, DepthU, family) -> family_spec structure of its work-group tile, where it beat the 8-wave default by
#: >= 3 % warm on the config's S problem (jobs 68371247 / 68371250, every arm verified + mutants, one GPU per config):
#: (1,4) + direct B + 4-stage A ring: 384x192 ak 1.217 / am 1.102, 192x192 (WG 384x192) ak 1.216 / am 1.103,
#: 320x256 ak 1.160 / am 1.047 (mJ / GEMM 0.82-0.95); 4 waves (2,2) jit: 288x256 ak 1.063, 256x288 ak 1.046,
#: 192x384 ak 1.036 (mJ 0.93-0.96).
STRUCT: Dict[Tuple[int, int, int, str], Dict[str, object]] = {
    (384, 192, 256, "ak"): _BD14, (384, 192, 256, "am"): _BD14,
    (192, 192, 256, "ak"): _BD14, (192, 192, 256, "am"): _BD14,
    (320, 256, 256, "ak"): _BD14, (320, 256, 256, "am"): _BD14,
    (288, 256, 256, "ak"): _W22, (256, 288, 256, "ak"): _W22, (192, 384, 256, "ak"): _W22,
    # w1x4 work groups (each wave owns its B rows) + direct B, persistent + bankocc (job 68373683, n14-05, verified):
    # 512x32 ak (WG 512x64) 1.241, 448x64 1.044, 448x128 1.038, 384x32 (WG 384x64) 1.035; mJ 0.88-0.96
    (512, 32, 256, "ak"): {**_JIT, "b_direct": True, "stages": 4},
    (384, 32, 256, "ak"): {**_JIT, "b_direct": True, "stages": 4},
    (448, 64, 256, "ak"): {"ring": "stage", "b_direct": True, "stages": 4},
    (448, 128, 256, "ak"): {"ring": "stage", "b_direct": True, "stages": 4},
}


#: round 5 (bubble loop items b31-b46, every arm verified + mutants on the b27 tree), re-measured head to head
#: against the shipped kernel and the c5 promotion on the current tree (job 68374995, S problem; TF/s / mJ per GEMM
#: vs ship): knobs merged over the config's STRUCT entry
_R5_STRUCT: Dict[Tuple[int, int, int, str], Dict[str, object]] = {
    (32, 384, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'unroll': 2, 'read_lead': 1},  # x1.159 / mJ x0.979
    (32, 512, 256, 'ak'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 4), 'b_direct': True, 'stages': 3, 'dtl_spread': 4, 'read_lead': 1},  # x1.135 / mJ x1.027
    (64, 192, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3},  # x1.074 / mJ x0.941
    (64, 192, 512, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 4), 'b_direct': True, 'stages': 3, 'dtl_spread': 4},  # x1.051 / mJ x0.961
    (64, 320, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'read_lead': 2, 'dtl_spread': 4, 'unroll': 2},  # x1.134 / mJ x0.907
    (64, 384, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'read_lead': 3},  # x1.099 / mJ x0.935
    (64, 448, 256, 'ak'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'scale_src': 'vgpr'},  # x1.118 / mJ x0.944
    (64, 512, 256, 'ak'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'scale_src': 'vgpr'},  # x1.137 / mJ x0.955
    (96, 384, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'scale_src': 'vgpr', 'cta_order': 'linear', 'setprio': 1, 'read_lead': 2, 'dtl_delay': 2, 'unroll': 2},  # x1.140 / mJ x0.877
    (128, 128, 512, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3},  # x1.046 / mJ x0.954
    (128, 224, 256, 'am'): {'ring': 'stage', 'stages': 3, 'unroll': 2, 'scale_src': 'vgpr'},  # x1.034 / mJ x0.973
    (128, 256, 256, 'am'): {'dtl_spread': 2},  # x1.061 / mJ x0.942
    (128, 320, 256, 'ak'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'cta_order': 'linear'},  # x1.112 / mJ x0.931
    (128, 320, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'group_m': 4, 'scale_src': 'vgpr'},  # x1.127 / mJ x0.885
    (128, 384, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 4), 'b_direct': True, 'stages': 3, 'read_lead': 3},  # x1.068 / mJ x0.951
    (128, 448, 256, 'ak'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'scale_src': 'vgpr'},  # x1.107 / mJ x0.942
    (160, 256, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (1, 4), 'b_direct': True, 'stages': 4, 'scale_src': 'vgpr', 'read_lead': 3},  # x1.013 / mJ x0.983
    (160, 384, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'scale_src': 'vgpr', 'unroll': 2},  # x1.086 / mJ x0.932
    (192, 320, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'unroll': 2},  # x1.112 / mJ x0.902
    (192, 384, 256, 'ak'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'scale_src': 'vgpr'},  # x1.125 / mJ x0.891
    (192, 384, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'read_lead': 2, 'unroll': 2, 'cta_order': 'linear'},  # x1.098 / mJ x0.906
    (256, 160, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'group_m': 4, 'scale_src': 'vgpr'},  # x1.127 / mJ x0.886
    (256, 256, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (1, 4), 'b_direct': True, 'stages': 4, 'read_lead': 2},  # x1.092 / mJ x0.916
    (256, 288, 256, 'ak'): {'ring': 'stage', 'stages': 3, 'unroll': 2, 'scale_src': 'vgpr'},  # x1.166 / mJ x0.883
    (256, 288, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'read_lead': 2},  # x1.090 / mJ x0.919
    (256, 320, 256, 'ak'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'scale_src': 'vgpr'},  # x1.095 / mJ x0.929
    (256, 320, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'group_m': 4, 'scale_src': 'vgpr'},  # x1.129 / mJ x0.884
    (288, 256, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (2, 2), 'b_direct': True, 'stages': 3, 'cta_order': 'linear'},  # x1.056 / mJ x0.945
    (320, 128, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (1, 4), 'b_direct': True, 'stages': 4, 'scale_src': 'vgpr', 'read_lead': 3},  # x1.014 / mJ x0.981
    (320, 192, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (1, 4), 'b_direct': True, 'stages': 4, 'scale_src': 'vgpr'},  # x1.068 / mJ x0.940
    (384, 96, 256, 'ak'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (1, 4), 'b_direct': True, 'stages': 4, 'dtl_delay': 2},  # x1.223 / mJ x0.819
    (384, 96, 256, 'am'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (1, 4), 'b_direct': True, 'stages': 4, 'scale_src': 'vgpr'},  # x1.115 / mJ x0.894
    (512, 32, 256, 'ak'): {'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (1, 4), 'b_direct': True, 'stages': 4, 'scale_src': 'vgpr', 'unroll': 2, 'dtl_spread': 4, 'read_lead': 2},  # x1.022 / mJ x0.974
}
for _k, _v in _R5_STRUCT.items():
    STRUCT[_k] = {**STRUCT.get(_k, {}), **_v}


#: (MT0, MT1, DepthU, family) -> the complete spec of a search promotion (fields != the dataclass default, levers as a
#: dlevers delta over LEVERS), shipped where it beat the current config_spec on the config's S problem (A/B against
#: the then-shipped dispatch kernel; see the notes per entry). It replaces WG_MULT / STRUCT / the schedule choice for
#: that config on multi-entry problems (the S problems it was measured on are batched) when nothing but ``k_target``
#: is overridden; folded batch-1 problems, K tails and explicit knobs keep the ``family_spec`` path.
#: search c5 (large / square lane), A/B vs the shipped kernel on one GPU (job 68374995, n01-09, S problem, verified;
#: ratios TF/s / mJ per GEMM vs ship); "mc_dma": "x3" entries beat the c5 spec with x4 there.
PROMOTED: Dict[Tuple[int, int, int, str], Dict[str, object]] = {
    (128, 128, 256, 'am'):  # x1.035 TF/s, x0.965 mJ
        {'required_compiler': 'rust', 'stages': 2, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'scale_layout': 'wtile', 'swap_ab': False, 'peel_first': True, 'frag_mode': 'prefetch', 'acq_slot': 1, 'read_every': 3, 'read_delay': 1, 'dtl_delay': 1, 'tile_lgkm0': True, 'unroll': 2, 'rsrc_once': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': '', 'mc_dma': 'x3'},
    (128, 256, 256, 'ak'):  # x1.024 TF/s, x0.977 mJ
        {'required_compiler': 'rust', 'dlevers': 'litreuse', 'swizzle': 'none', 'canon_ring': 'line2m', 'mini_copy_gap': 4, 'mini_copy_at': 6, 'line_reads_first': True, 'line_release': 39, 'canon_read': 'b96x2', 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'line_dtl_order': 'wave', 'peel_first': True, 'm_outer': False, 'lds_alias_scopes': False, 'frag_mode': 'prefetch', 'acq_slot': 0, 'read_every': 3, 'dtl_delay': 1, 'dtl_spread': 5, 'scale_slot': 32, 'tile_lgkm0': True, 'unroll': 8, 'rsrc_once': True, 'dtl_imm': True, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 4, 'stage_kib': True, 'schedule': 'persistent', 'magic_div': True, 'edge_tiles': True},
    (128, 384, 256, 'ak'):  # x1.045 TF/s, x0.960 mJ
        {'required_compiler': 'rust', 'tile_m': 384, 'tile_n': 128, 'waves_m': 4, 'waves_n': 1, 'swizzle': 'none', 'canon_ring': 'line2m', 'mini_copy_gap': 30, 'mini_copy_at': 6, 'mini_layout': 'v4', 'line_spread': 3, 'line_reads_first': True, 'line_release': 17, 'canon_rsrc_once': True, 'canon_read': 'b96x2', 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'peel_first': True, 'm_outer': False, 'frag_mode': 'prefetch', 'acq_slot': 0, 'dtl_delay': 1, 'scale_slot': 24, 'tile_lgkm0': True, 'unroll': 8, 'rsrc_once': True, 'dtl_imm': True, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': ''},
    (160, 256, 256, 'ak'):  # x1.017 TF/s, x0.985 mJ
        {'required_compiler': 'rust', 'tile_n': 160, 'canon_ring': 'line3', 'ring_ops': 'A', 'mini_copy_gap': 0, 'mini_layout': 'v4', 'line_reads_first': True, 'line_early': True, 'line_release': 27, 'canon_read': 'b128b64', 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'peel_first': True, 'm_outer': False, 'lds_alias_scopes': False, 'frag_mode': 'prefetch', 'acq_slot': 0, 'dtl_delay': 1, 'scale_slot': 18, 'unroll': 8, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': ''},
    (160, 384, 256, 'ak'):  # x1.024 TF/s, x0.977 mJ
        {'required_compiler': 'rust', 'tile_m': 384, 'tile_n': 160, 'waves_m': 4, 'waves_n': 1, 'stages': 2, 'swizzle': 'none', 'canon_ring': 'line2m', 'mini_copy_gap': 4, 'line_reads_first': True, 'line_release': 27, 'canon_read': 'b96x2', 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'm_outer': False, 'frag_mode': 'prefetch', 'acq_slot': 0, 'dtl_delay': 1, 'dtl_spread': 4, 'scale_slot': 30, 'tile_lgkm0': True, 'unroll': 4, 'rsrc_once': True, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'edge_tiles': True, 'dlevers': '-m0share,-soffshare'},
    (192, 192, 256, 'am'):  # x1.039 TF/s, x0.967 mJ
        {'required_compiler': 'rust', 'tile_m': 192, 'tile_n': 384, 'waves_m': 1, 'waves_n': 4, 'stages': 4, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'scale_layout': 'wtile', 'scale_src': 'vgpr', 'swap_ab': False, 'acc_vgpr_spill': True, 'peel_first': True, 'dtl_delay': 1, 'tile_lgkm0': True, 'unroll': 3, 'rsrc_once': True, 'b_direct': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 4, 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': '', 'mc_dma': 'x3'},
    (192, 256, 256, 'ak'):  # x1.021 TF/s, x0.979 mJ
        {'required_compiler': 'rust', 'tile_n': 192, 'canon_ring': 'line3', 'ring_ops': 'A', 'mini_copy_gap': 0, 'line_reads_first': True, 'line_early': True, 'line_release': 31, 'canon_read': 'b96x2', 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'peel_first': True, 'm_outer': False, 'lds_alias_scopes': False, 'frag_mode': 'prefetch', 'acq_slot': 0, 'read_delay': 1, 'dtl_delay': 1, 'scale_slot': 18, 'tile_lgkm0': True, 'unroll': 8, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': ''},
    (224, 256, 256, 'am'):  # x1.146 TF/s, x0.870 mJ
        {'required_compiler': 'rust', 'tile_n': 224, 'waves_m': 4, 'waves_n': 1, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'scale_layout': 'wtile', 'scale_src': 'vgpr', 'swap_ab': False, 'peel_first': True, 'pin_mfma': False, 'read_split': False, 'read_every': 3, 'read_lead': 3, 'dtl_delay': 1, 'scale_slot': 10, 'setprio': 1, 'tile_lgkm0': True, 'unroll': 2, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 2, 'stage_kib': True, 'magic_div': True, 'edge_tiles': True, 'dlevers': 'bufoffany,loadfirst,nofence,pinvmem,sreassoc,wpecap,-dmamanualringw,-ldsgroup,-litshare,-scalesep', 'mc_dma': 'x3'},
    (256, 128, 256, 'am'):  # x1.028 TF/s, x0.987 mJ
        {'required_compiler': 'rust', 'tile_m': 128, 'tile_n': 512, 'waves_m': 1, 'waves_n': 4, 'stages': 2, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'scale_layout': 'wtile', 'swap_ab': False, 'peel_first': True, 'frag_mode': 'prefetch', 'acq_slot': 8, 'read_every': 4, 'read_delay': 4, 'tile_lgkm0': True, 'unroll': 2, 'rsrc_once': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': '', 'mc_dma': 'x3'},
    (256, 192, 256, 'ak'):  # x1.016 TF/s, x0.991 mJ
        {'required_compiler': 'rust', 'tile_m': 192, 'swizzle': 'none', 'line_spread': 1, 'canon_rsrc_once': True, 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'frag_mode': 'prefetch', 'read_delay': 2, 'dtl_delay': 1, 'tile_lgkm0': True, 'unroll': 2, 'rsrc_once': True, 'dtl_aux': 1, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': ''},
    (288, 256, 256, 'ak'):  # x1.189 TF/s, x0.861 mJ
        {'required_compiler': 'rust', 'tile_n': 288, 'line_release': 25, 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'scale_src': 'vgpr', 'swap_ab': False, 'acc_vgpr_spill': True, 'peel_first': True, 'dtl_delay': 1, 'dtl_spread': 6, 'scale_slot': 27, 'unroll': 2, 'rsrc_once': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 2, 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': 'sreassoc,-litshare'},
    (320, 256, 256, 'ak'):  # x1.037 TF/s, x0.964 mJ
        {'required_compiler': 'rust', 'tile_n': 320, 'waves_m': 1, 'waves_n': 4, 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'acc_vgpr_spill': True, 'peel_first': True, 'tile_lgkm0': True, 'rsrc_once': True, 'b_direct': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': ''},
    (320, 256, 256, 'am'):  # x1.117 TF/s, x0.895 mJ
        {'required_compiler': 'rust', 'tile_n': 320, 'waves_m': 1, 'waves_n': 4, 'stages': 4, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'scale_layout': 'wtile', 'swap_ab': False, 'acc_vgpr_spill': True, 'peel_first': True, 'read_lead': 6, 'dtl_delay': 8, 'scale_slot': 35, 'tile_lgkm0': True, 'rsrc_once': True, 'b_direct': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 4, 'stage_kib': True, 'edge_tiles': True, 'dlevers': 'copysink,nofence,vmemfirst,-soffshare', 'mc_dma': 'x3'},
    (384, 128, 256, 'am'):  # x1.208 TF/s, x0.827 mJ
        {'required_compiler': 'rust', 'tile_m': 128, 'tile_n': 384, 'waves_m': 1, 'waves_n': 4, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'scale_layout': 'wtile', 'scale_src': 'vgpr', 'swap_ab': False, 'peel_first': True, 'pin_mfma': False, 'lds_alias_scopes': False, 'read_every': 4, 'dtl_spread': 5, 'scale_slot': 15, 'setprio': 3, 'tile_lgkm0': True, 'unroll': 2, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'linear', 'group_m': 16, 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': 'dmafree,ldsquads,licm,litreuse,nofence,sreassoc,wpecap,-dshoist,-ldsgroup,-litshare,-m0cse,-m0fill,-m0share', 'mc_dma': 'x3'},
    (384, 192, 256, 'am'):  # x1.041 TF/s, x0.964 mJ
        {'required_compiler': 'rust', 'tile_m': 192, 'tile_n': 384, 'waves_m': 1, 'waves_n': 4, 'stages': 4, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'scale_layout': 'wtile', 'swap_ab': False, 'acc_vgpr_spill': True, 'read_split': False, 'read_lead': 3, 'dtl_delay': 4, 'rsrc_once': True, 'b_direct': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': 'ldsspillhi', 'mc_dma': 'x3'},
    (448, 128, 256, 'ak'):  # x1.049 TF/s, x0.953 mJ
        {'required_compiler': 'rust', 'dlevers': '-litshare', 'tile_m': 128, 'tile_n': 448, 'waves_m': 1, 'waves_n': 4, 'stages': 2, 'swizzle': 'none', 'canon_ring': 'line2m', 'mini_copy_at': 3, 'line_reads_first': True, 'line_release': 34, 'canon_read': 'b128b64', 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'peel_first': True, 'm_outer': False, 'frag_mode': 'prefetch', 'acq_slot': 0, 'read_delay': 1, 'read_lead': 3, 'dtl_spread': 1, 'scale_slot': 28, 'tile_lgkm0': True, 'unroll': 4, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True},
    (448, 128, 256, 'am'):  # x1.030 TF/s, x0.971 mJ
        {'required_compiler': 'rust', 'tile_m': 128, 'tile_n': 448, 'waves_m': 1, 'waves_n': 4, 'stages': 2, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'scale_layout': 'wtile', 'swap_ab': False, 'peel_first': True, 'pin_mfma': False, 'frag_mode': 'prefetch', 'read_delay': 4, 'read_lead': 3, 'tile_lgkm0': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': 'ldsquads', 'mc_dma': 'x3'},
}


#: FINAL search promotions c6 / c7 (large / square lane, tilesearch/results/final), head to head against the
#: tree with the entries above (job 68376478, n01-09, S problem, verified; fin / finx3 = with mc_dma x3): they
#: replace those entries where they won
PROMOTED.update({
    (128, 128, 256, 'ak'):  # fin x1.034 TF/s, x0.969 mJ
        {'required_compiler': 'rust', 'dlevers': 'm0sharebase,litreuse', 'swizzle': 'none', 'canon_ring': 'line2m', 'mini_copy_gap': 4, 'mini_copy_at': 6, 'line_reads_first': True, 'line_release': 39, 'canon_read': 'b96x2', 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'line_dtl_order': 'wave', 'peel_first': True, 'm_outer': False, 'lds_alias_scopes': False, 'frag_mode': 'prefetch', 'acq_slot': 0, 'read_every': 3, 'dtl_delay': 1, 'dtl_spread': 6, 'scale_slot': 32, 'tile_lgkm0': True, 'unroll': 8, 'rsrc_once': True, 'dtl_imm': True, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 4, 'stage_kib': True, 'schedule': 'persistent', 'magic_div': True, 'edge_tiles': True},
    (128, 128, 256, 'am'):  # finx3 x1.039 TF/s, x0.971 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink', 'stages': 4, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'scale_layout': 'wtile', 'swap_ab': False, 'read_lead': 6, 'dtl_delay': 1, 'tile_lgkm0': True, 'rsrc_once': True, 'b_direct': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 4, 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'mc_dma': 'x3'},
    (128, 256, 256, 'am'):  # fin x1.018 TF/s, x0.985 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink,wpecap', 'waves_m': 1, 'waves_n': 4, 'stages': 4, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'mc_dma': 'x3', 'scale_layout': 'wtile', 'swap_ab': False, 'read_lead': 3, 'dtl_delay': 1, 'tile_lgkm0': True, 'rsrc_once': True, 'b_direct': True, 'b_every': 4, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 4, 'stage_kib': True, 'magic_div': True, 'edge_tiles': True},
    (128, 384, 256, 'am'):  # fin x1.017 TF/s, x0.983 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink', 'tile_m': 384, 'tile_n': 128, 'waves_n': 4, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'mc_dma': 'x3', 'scale_layout': 'wtile', 'swap_ab': False, 'acc_agpr': False, 'read_lead': 3, 'dtl_delay': 4, 'tile_lgkm0': True, 'unroll': 2, 'rsrc_once': True, 'b_direct': True, 'b_every': 4, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True},
    (128, 448, 256, 'am'):  # fin x1.035 TF/s, x0.964 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink,sreassoc,vmemfirst', 'tile_m': 448, 'tile_n': 128, 'waves_m': 4, 'waves_n': 1, 'stages': 2, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'mc_dma': 'x3', 'scale_layout': 'wtile', 'swap_ab': False, 'peel_first': True, 'frag_mode': 'prefetch', 'acq_slot': 0, 'read_delay': 8, 'dtl_spread': 1, 'tile_lgkm0': True, 'rsrc_once': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'magic_div': True, 'edge_tiles': True},
    (160, 256, 256, 'am'):  # fin x1.022 TF/s, x0.981 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink,-soffshare', 'tile_n': 320, 'waves_n': 4, 'stages': 2, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'mc_dma': 'x3', 'scale_layout': 'wtile', 'swap_ab': False, 'acc_agpr': False, 'peel_first': True, 'read_lead': 2, 'dtl_spread': 2, 'tile_lgkm0': True, 'rsrc_once': True, 'b_every': 1, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True},
    (160, 384, 256, 'ak'):  # fin x1.060 TF/s, x0.944 mJ
        {'required_compiler': 'rust', 'dlevers': 'm0sharebase', 'tile_m': 384, 'tile_n': 160, 'waves_m': 4, 'waves_n': 1, 'stages': 2, 'swizzle': 'none', 'canon_ring': 'line2m', 'mini_copy_at': 3, 'line_reads_first': True, 'line_release': 32, 'canon_read': 'b96x2', 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'scale_src': 'vgpr', 'swap_ab': False, 'line_dtl_order': 'wave', 'peel_first': True, 'm_outer': False, 'frag_mode': 'prefetch', 'dtl_delay': 1, 'dtl_spread': 4, 'scale_slot': 30, 'tile_lgkm0': True, 'unroll': 4, 'rsrc_once': True, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True},
    (192, 192, 256, 'ak'):  # fin x1.013 TF/s, x0.988 mJ
        {'required_compiler': 'rust', 'tile_m': 192, 'tile_n': 384, 'waves_m': 1, 'waves_n': 4, 'stages': 4, 'canon_rsrc_once': True, 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'scale_src': 'vgpr', 'swap_ab': False, 'acc_vgpr_spill': True, 'dtl_delay': 1, 'scale_slot': 27, 'tile_lgkm0': True, 'rsrc_once': True, 'b_direct': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': 'sremat,-soffshare'},
    (192, 256, 256, 'am'):  # fin x1.035 TF/s, x0.968 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink,-dmamanualringw', 'tile_n': 192, 'waves_m': 4, 'waves_n': 1, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'mc_dma': 'x3', 'scale_layout': 'wtile', 'swap_ab': False, 'peel_first': True, 'frag_mode': 'prefetch', 'acq_slot': 0, 'dtl_delay': 2, 'tile_lgkm0': True, 'unroll': 2, 'rsrc_once': True, 'b_every': 4, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 4, 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True},
    (192, 320, 256, 'am'):  # fin x1.026 TF/s, x0.977 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink', 'tile_m': 320, 'tile_n': 192, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'mc_dma': 'x3', 'scale_layout': 'wtile', 'scale_src': 'vgpr', 'swap_ab': False, 'read_every': 3, 'read_delay': 1, 'tile_lgkm0': True, 'unroll': 2, 'rsrc_once': True, 'dtl_aux': 1, 'b_direct': True, 'b_every': 4, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True},
    (224, 256, 256, 'ak'):  # fin x1.017 TF/s, x0.983 mJ
        {'required_compiler': 'rust', 'dlevers': 'm0sharebase', 'tile_n': 224, 'swizzle': 'none', 'canon_ring': 'line2m', 'mini_copy_gap': 4, 'line_reads_first': True, 'line_release': 35, 'canon_read': 'b96x2', 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'line_dtl_order': 'wave', 'peel_first': True, 'm_outer': False, 'frag_mode': 'prefetch', 'acq_slot': 0, 'dtl_delay': 2, 'scale_slot': 35, 'tile_lgkm0': True, 'unroll': 8, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'linear', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True},
    (256, 128, 256, 'ak'):  # fin x1.025 TF/s, x0.976 mJ
        {'required_compiler': 'rust', 'dlevers': 'm0sharebase', 'canon_ring': 'line2m', 'mini_copy_gap': 4, 'line_reads_first': True, 'line_release': 39, 'canon_read': 'b96x2', 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'line_dtl_order': 'wave', 'peel_first': True, 'm_outer': False, 'frag_mode': 'prefetch', 'acq_slot': 0, 'read_every': 3, 'read_delay': 8, 'dtl_delay': 1, 'dtl_spread': 5, 'scale_slot': 32, 'tile_lgkm0': True, 'unroll': 8, 'rsrc_once': True, 'dtl_imm': True, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 2, 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True},
    (256, 160, 256, 'ak'):  # fin x1.129 TF/s, x0.888 mJ
        {'required_compiler': 'rust', 'tile_m': 160, 'tile_n': 512, 'waves_m': 1, 'waves_n': 8, 'stages': 5, 'canon_rsrc_once': True, 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'acc_agpr': False, 'read_lead': 3, 'dtl_delay': 4, 'tile_lgkm0': True, 'rsrc_once': True, 'b_direct': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': ''},
    (256, 256, 256, 'ak'):  # fin x1.026 TF/s, x0.977 mJ
        {'required_compiler': 'rust', 'dlevers': 'm0sharebase', 'canon_ring': 'line2m', 'mini_copy_gap': 4, 'line_reads_first': True, 'line_release': 38, 'canon_read': 'b96x2', 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'swap_ab': False, 'line_dtl_order': 'wave', 'peel_first': True, 'm_outer': False, 'frag_mode': 'prefetch', 'acq_slot': 0, 'read_every': 3, 'read_delay': 8, 'dtl_delay': 1, 'dtl_spread': 5, 'scale_slot': 32, 'tile_lgkm0': True, 'unroll': 8, 'rsrc_once': True, 'dtl_imm': True, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 2, 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True},
    (256, 288, 256, 'ak'):  # fin x1.045 TF/s, x0.957 mJ
        {'required_compiler': 'rust', 'tile_m': 288, 'mini_layout': 'v4', 'ring_kib': True, 'line_reads_first': True, 'line_early': True, 'canon_dtl': 'interleave', 'a_layout': 'canonical', 'scale_layout': 'wtile', 'scale_src': 'vgpr', 'swap_ab': False, 'acc_vgpr_spill': True, 'read_every': 1, 'dtl_delay': 1, 'scale_slot': 27, 'tile_lgkm0': True, 'rsrc_once': True, 'dtl_aux': 1, 'b_every': 1, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'dlevers': '-ldsgroup,-litshare'},
    (256, 320, 256, 'am'):  # fin x1.024 TF/s, x0.977 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink,dmafree,vmemfirst,-ldsgroup,-soffshare', 'tile_m': 320, 'waves_n': 4, 'stages': 2, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'mc_dma': 'x3', 'scale_layout': 'wtile', 'scale_src': 'vgpr', 'swap_ab': False, 'acc_agpr': False, 'peel_first': True, 'read_split': False, 'read_lead': 2, 'dtl_spread': 2, 'unroll': 2, 'rsrc_once': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 2, 'stage_kib': True, 'schedule': 'persistent'},
    (320, 128, 256, 'am'):  # fin x1.023 TF/s, x0.982 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink,-soffshare', 'tile_n': 320, 'waves_n': 4, 'stages': 2, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'mc_dma': 'x3', 'scale_layout': 'wtile', 'swap_ab': False, 'acc_agpr': False, 'read_lead': 2, 'dtl_spread': 2, 'tile_lgkm0': True, 'unroll': 2, 'rsrc_once': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True},
    (320, 192, 256, 'am'):  # finx3 x1.031 TF/s, x0.969 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink', 'tile_m': 192, 'tile_n': 320, 'waves_m': 1, 'waves_n': 4, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'scale_layout': 'wtile', 'swap_ab': False, 'dtl_delay': 1, 'scale_slot': 38, 'tile_lgkm0': True, 'unroll': 2, 'rsrc_once': True, 'b_direct': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True, 'mc_dma': 'x3'},
    (384, 128, 256, 'am'):  # fin x1.067 TF/s, x0.938 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink,sreassoc,licm,dmafree,litreuse,wpecap,nofence,ldsquads,-dshoist,-m0fill,-m0cse,-m0share,-ldsgroup,-litshare', 'tile_m': 128, 'tile_n': 384, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'mc_dma': 'x3', 'scale_layout': 'wtile', 'scale_src': 'vgpr', 'swap_ab': False, 'peel_first': True, 'pin_mfma': False, 'lds_alias_scopes': False, 'frag_mode': 'prefetch', 'read_delay': 4, 'dtl_spread': 5, 'scale_slot': 15, 'setprio': 3, 'tile_lgkm0': True, 'unroll': 2, 'xcd_block': 'none', 'store_nt': True, 'runtime_shape': True, 'cta_order': 'linear', 'stage_kib': True, 'schedule': 'persistent'},
    (384, 192, 256, 'am'):  # fin x1.051 TF/s, x0.953 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink', 'tile_m': 192, 'tile_n': 384, 'waves_m': 1, 'waves_n': 4, 'stages': 5, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'mc_dma': 'x3', 'scale_layout': 'wtile', 'scale_src': 'vgpr', 'swap_ab': False, 'acc_vgpr_spill': True, 'peel_first': True, 'pin_mfma': False, 'dtl_delay': 8, 'scale_slot': 32, 'tile_lgkm0': True, 'rsrc_once': True, 'b_direct': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True},
    (448, 128, 256, 'am'):  # fin x1.091 TF/s, x0.917 mJ
        {'required_compiler': 'rust', 'dlevers': 'copysink', 'tile_m': 128, 'tile_n': 448, 'waves_m': 1, 'waves_n': 4, 'stages': 4, 'canon_dtl': 'interleave', 'a_layout': 'mcontig', 'mc_dma': 'x3', 'scale_layout': 'wtile', 'swap_ab': False, 'frag_mode': 'prefetch', 'acq_slot': 0, 'dtl_delay': 2, 'tile_lgkm0': True, 'rsrc_once': True, 'b_direct': True, 'xcd_block': 'none', 'runtime_shape': True, 'cta_order': 'grouped', 'group_m': 4, 'stage_kib': True, 'schedule': 'persistent', 'edge_tiles': True},
})

#: am lane round 2 (job 68375138; head to head in job 68376478): direct B (1,4) 4-stage A rings
STRUCT.update({
    (256, 128, 256, 'am'): {**_JIT, "waves": (1, 4), "b_direct": True, "stages": 4, "scale_src": "vgpr"},  # x1.027 / mJ x0.974
    (384, 64, 256, 'am'): {**_JIT, "waves": (1, 4), "b_direct": True, "stages": 4, "dtl_delay": 2},  # x1.111 / mJ x0.901
    (448, 64, 256, 'am'): {**_JIT, "waves": (1, 4), "b_direct": True, "stages": 4, "dtl_delay": 2},  # x1.085 / mJ x0.924
})
PROMOTED.pop((256, 128, 256, 'am'), None)  # its STRUCT structure beat the c5 + x3 promotion
#: 512x32 am: direct B (1,4) 4-stage A ring with the compiler's bankpref lever (accumulators kept in AGPRs, no VGPR
#: spill; needs compiler >= out-68377488, R13): S 1.190, mJ 0.900, G 0.993 (job 68378376)
STRUCT[(512, 32, 256, "am")] = {**_JIT, "waves": (1, 4), "b_direct": True, "stages": 4, "dlevers": "bankpref"}
for _k in PROMOTED:  # the library serves any M / N: edge_tiles on (only the last tile row / column moves)
    PROMOTED[_k] = {**PROMOTED[_k], "edge_tiles": True}


def _promoted_spec(key, fmt: str, over: Dict[str, object]) -> Mxfp6GemmSpec:
    k_target = over.pop("k_target", 8192)
    spec = replace(Mxfp6GemmSpec(), **{**PROMOTED[key], "elem_fmt": fmt})
    spec = replace(spec, m=16 * spec.tile_m, n=16 * spec.tile_n, **over)
    spec = replace(spec, k=library_k_ref(spec, k_target=k_target) if spec.runtime_shape else spec.k)
    ok, why = validate_spec(spec)
    if not ok:
        raise ValueError(f"{key} promoted: {why}")
    return spec


#: am lane, mid-tile core loop: configs whose shipped am kernel takes ``mc_copy`` (the second-half tr_b6 copies as
#: explicit v_movs at their MFMA, gemm.py), measured head to head on the S problem against the same kernel with the
#: compiler's alignment copy (h/cells/mcc*.json, job 68376885 n01-09, compiler out-68373490; twins within 0.2 %, every
#: mutant incl. mc_copyswap fails): 128x224 1.056, 128x256 1.042, 128x384 1.006, 192x128 1.021, 192x256 1.009,
#: 256x128 1.013, 256x160 1.017, 256x192 1.016, 320x192 1.074 (energy per GEMM -0.6 .. -6.8 %). Not taken: 160x256 /
#: 320x128 (their direct-B jit kernel: 0.997).
AM_MC_COPY: Dict[Tuple[int, int, int], str] = {
    (128, 224, 256): "late", (128, 256, 256): "late", (128, 384, 256): "late", (192, 128, 256): "late",
    (192, 256, 256): "late", (256, 128, 256): "late", (256, 160, 256): "late", (256, 192, 256): "late",
    (320, 192, 256): "late",
}


def _am_mc_copy(key, spec: Mxfp6GemmSpec, over) -> Mxfp6GemmSpec:
    """``spec`` with the config's ``AM_MC_COPY`` (am, M-contiguous A, rust compiler; not when the caller sets it)."""
    mc = AM_MC_COPY.get(key[:3]) if key[3] == "am" else None
    if mc and "mc_copy" not in over and spec.a_layout == "mcontig" and spec.required_compiler == "rust":
        return replace(spec, mc_copy=mc)
    return spec


def config_spec(mt0: int, mt1: int, du: int, family: str = "ak", fmt: str = "e2m3", promoted: bool = True,
                **over) -> Mxfp6GemmSpec:
    """The kernel the library ships for config ``MT0 x MT1 x DepthU`` / ``family``: its ``PROMOTED`` search spec
    (``promoted`` and no override but ``k_target``), else the family default of its work-group tile (``WG_MULT``,
    native when absent) with ``edge_tiles`` on. ``family_spec`` builds native tiles."""
    if promoted and (mt0, mt1, du, family) in PROMOTED and set(over) <= {"k_target"}:
        return _am_mc_copy((mt0, mt1, du, family), _promoted_spec((mt0, mt1, du, family), fmt, dict(over)), over)
    g0, g1 = WG_MULT.get((mt0, mt1, du, family), (1, 1))
    st = STRUCT.get((mt0, mt1, du, family), {})
    spec = family_spec(mt0 * g0, mt1 * g1, du, family, fmt, **{"edge_tiles": True, **st, **over})
    if family == "am" and "dlevers" not in over:  # copysink: the tr_b6 second-half copies sink to their MFMA (am)
        spec = replace(spec, dlevers=",".join(x for x in (spec.dlevers, "copysink") if x))
    if spec.canon_ring == "line2m" and "line_dtl_order" not in over and "dlevers" not in over:
        # one wave's line DTLs on consecutive row classes + the m0sharebase lever: one M0 per DTL group (M0
        # hazard nops 54 -> 0 on 32x128 ak; +0.25-0.5 %, compiler agent, out-68371726)
        spec = replace(spec, line_dtl_order="wave",
                       dlevers=",".join(x for x in (spec.dlevers, "m0sharebase") if x))
    return _am_mc_copy((mt0, mt1, du, family), spec, over)


#: (MT0, MT1, DepthU, family) whose shipped kernel keeps the data-parallel grid on multi-entry problems too: the
#: pairs where persistent lost > 1 % in the S-problem A/B of every config_spec (jobs 68369075-77, both schedules of
#: a pair on one GPU; persistent median +6.0 %, 115 of 140 win).
PERSIST_EXCLUDE: frozenset = frozenset({
    (32, 32, 512, 'am'), (32, 128, 256, 'ak'), (32, 128, 256, 'am'), (32, 128, 512, 'ak'),
    (32, 256, 256, 'ak'), (64, 64, 256, 'ak'), (64, 64, 256, 'am'), (64, 64, 512, 'am'), (128, 32, 256, 'am'),
    (128, 32, 512, 'am'), (128, 64, 512, 'ak'), (256, 32, 256, 'ak'),
})


#: configs whose kernels stage C through LDS and store it as dwordx4 (gemm ``epi_lds``): job 68378376 (n01-09, the
#: shipped kernel of every config on its S and G problems, with and without; verified), taken where one problem gains
#: >= 1 % and neither loses > 0.5 % nor costs mJ (G 256x128 ak 1.092, 128x256 ak 1.059, 256x256 ak 1.051; S 32x256 ak
#: 1.045, 64x192 ak 1.041, 256x64 ak 1.032, ...). dispatch adds it to whatever kernel it picks when that kernel takes it.
EPI_LDS: frozenset = frozenset({
    (32, 128, 256, 'ak'), (32, 128, 512, 'ak'), (32, 256, 256, 'ak'), (32, 256, 256, 'am'), (32, 384, 256, 'ak'),
    (64, 64, 256, 'ak'), (64, 64, 512, 'ak'), (64, 128, 256, 'ak'), (64, 128, 256, 'am'), (64, 128, 512, 'ak'),
    (64, 192, 256, 'ak'), (64, 256, 256, 'ak'), (64, 320, 256, 'ak'), (64, 384, 256, 'ak'), (128, 64, 256, 'ak'),
    (128, 64, 512, 'ak'), (128, 128, 256, 'ak'), (128, 128, 512, 'ak'), (128, 256, 256, 'ak'), (128, 256, 256, 'am'),
    (128, 320, 256, 'ak'), (128, 384, 256, 'ak'), (128, 384, 256, 'am'), (256, 64, 256, 'ak'), (256, 96, 256, 'ak'),
    (256, 128, 256, 'ak'), (256, 256, 256, 'ak'), (512, 64, 256, 'ak'),
})
#: promotions measured worse than the family path on the G problem (job 68378376: 160x256 am 0.977)
PROMOTED_SKIP_G: frozenset = frozenset({(160, 256, 256, "am")})


def dispatch(mt0: int, mt1: int, du: int, family: str, fmt: str, m: int, n: int, batch: int = 1,
             **over) -> Mxfp6GemmSpec:
    """``_dispatch`` plus the per-config C staging (``EPI_LDS``) on the kernel it picks."""
    spec = _dispatch(mt0, mt1, du, family, fmt, m, n, batch, **over)
    if ((mt0, mt1, du, family) in EPI_LDS and "epi_lds" not in over and "_sk_base" not in over
            and spec.splitk == "none" and not spec.epi_lds):
        e = replace(spec, epi_lds=True)
        if validate_spec(e)[0]:
            return e
    return spec


def _dispatch(mt0: int, mt1: int, du: int, family: str, fmt: str, m: int, n: int, batch: int = 1,
              **over) -> Mxfp6GemmSpec:
    """The kernel for config ``MT0 x MT1 x DepthU`` / ``family`` on a problem (m, n, batch): ``config_spec`` with
    the persistent schedule on multi-entry problems (batched S: +7 % median, the large / square search round 1 and
    the all-config A/B) and the data-parallel grid on one entry (folded batch-1 problems: persistent 0.95-0.98)."""
    key = (mt0, mt1, du, family)
    kt = over.get("k_target")
    if kt and kt % 128 and "k_tail" not in over:
        # SB-5: K a multiple of 32, not of 128 -- ceil(K / 128) tiles, B / scales padded on the host (layout.pad_k);
        # canonical A needs the stage ring (its exact-range descriptor bounds the tail; GPU job 68374455)
        if kt % 32:
            raise ValueError(f"K = {kt}: MX needs a multiple of 32")
        over = {**over, "k_tail": True, "k_target": -(-kt // 128) * 128, **({"ring": "stage"} if family == "ak" else {})}
    sched = "persistent" if batch >= 2 and key not in PERSIST_EXCLUDE and key not in PERSIST_SPILL else "dp"
    g0, g1 = WG_MULT.get(key, (1, 1))
    sk_base = over.pop("_sk_base", False)  # (splitk.select_config asks for the plain kernel)
    if over.get("k_target") and not sk_base and (m < mt1 or batch * -(-m // (mt1 * g1)) * -(-n // (mt0 * g0)) < 512
                                                  or (family == "am" and m % 16)):
        # stream-K / split-K lane: fewer tokens than the config's tile (decode; rows past M masked, m_mask) or a grid
        # short of the GPU (< 2 x 256 CU tiles): split-K / data-parallel units chosen per problem by
        # splitk.select_config's cost model (the plain kernel below when that wins); needs the problem's K. am with
        # M % 16 != 0 (the M-padded producer layout, gemm.mcontig_rows): masked rows on the fused path at any grid
        from .splitk import select_config

        return select_config(mt0, mt1, du, family, fmt, m, n, over["k_target"], batch,
                             **{k_: v for k_, v in over.items() if k_ != "k_target"})
    gw = g_wg(mt0, mt1, du, family, m, n)
    while gw is not None and gw[0] >= 1:  # G / MoE regime: one token tile row per entry (MoE lane, G_WG)
        try:
            return family_spec(mt0 * gw[0], mt1, du, family, fmt, **{"edge_tiles": True, "schedule": "dp", **gw[1], **over})
        except ValueError:  # that WG does not build (LDS / registers): the next smaller one
            gw = (gw[0] // 2, gw[1]) if gw[0] > 1 else None
    if m < mt1 * g1 or n < mt0 * g0:
        # SB-1 (MoE lane): the shipped work group is larger than the problem (e.g. 64x64 ships a 128x128 WG; the G /
        # MoE problems have M = one token tile): the largest WG multiple that fits, native at worst (edge_tiles needs
        # M >= tile_m, N >= tile_n). Only problems that config_spec cannot run take this path.
        while g1 > 1 and m < mt1 * g1:
            g1 //= 2
        while g0 > 1 and n < mt0 * g0:
            g0 //= 2
        return family_spec(mt0 * g0, mt1 * g1, du, family, fmt, **{"edge_tiles": True, "schedule": sched, **over})
    if batch >= 2 and key in PROMOTED and set(over) <= {"k_target"} and not (
            key in PROMOTED_SKIP_G and mt1 <= m < 2 * mt1):
        spec = config_spec(mt0, mt1, du, family, fmt, **over)  # the promoted search spec, its own schedule
        if m >= spec.tile_m and n >= spec.tile_n:
            return spec
    extra = dict(PERSIST_KNOBS.get(key, {})) if sched == "persistent" else {}
    return config_spec(mt0, mt1, du, family, fmt, promoted=False, **{"schedule": sched, **extra, **over})


#: G / MoE regime (MoE lane + bubble loop; tile-library.md "MoE / grouped / HBM lane" G5 / G7 and "Round 5"): problems
#: with ONE token tile row per entry (MT1 <= M < 2 MT1: the G set, grouped MoE decode) re-read their activation tile
#: once per CTA along N, so a work group of g0 tiles along the weights (g1 = 1) reads it once for g0 N tiles (64x64 G:
#: 0.46x the L2 requests and 0.73x the energy per launch at the power cap) -- and it also fixes the shipped WG_MULT
#: tiles (g1 = 2) that are taller than the problem (SB-1 / (K)). ``G_WG``: measured (g0, knobs) per config, every arm
#: verified + every mutant failing (MoE lane jobs 68373053 h1 / h2 / h3; bubble loop items b39_0-4); configs at
#: MT1 <= G_WG_MT1 not listed take g0 = the largest power of two with MT0 g0 <= G_WG_ROWS, data-parallel.
G_WG_ROWS = 512
G_WG_MT1 = 64
_BD2 = {"b_direct": True, "b_depth": 2, "stages": 3}
_PERS = {"schedule": "persistent"}
_NTB = {"b_aux": 2}  # the weights' loads non-temporal (h4: +2-19 % on DTL-path G cells; HBM probe: nt reads +8-13 %)
G_WG: Dict[Tuple[int, int, int, str], Tuple[int, Dict[str, object]]] = {
    # MT1 <= 64 (MoE lane h1 / h3; x = vs the native tile)
    (64, 64, 256, "ak"): (8, _BD2),  # 1.40x the native tile (notes G5 / G7: % of the HBM read roof)
    (64, 64, 256, "am"): (8, _BD2),  # 1.24x
    (128, 32, 256, "ak"): (4, _NTB),  # 1.06x; + nt weights (b_aux 2) 1.108x (notes G10)
    (128, 32, 256, "am"): (4, _NTB),  # 1.05x; b_aux 2 1.113x
    (128, 64, 256, "ak"): (4, _BD2),  # 1.011x vs the 512x64 stage ring; b39: 1.02x native
    (128, 64, 256, "am"): (4, {}),  # b39: 1.005x
    (192, 64, 256, "ak"): (2, _BD2),  # 1.007x vs the stage ring
    (256, 64, 256, "ak"): (2, _BD2),  # 1.007x
    (256, 64, 256, "am"): (2, _NTB),  # b_aux 2 1.019x (h4)
    (256, 32, 256, "ak"): (2, {}),  # = the 512x32 rule (h3)
    (320, 64, 256, "ak"): (2, {}),  # b39: 640x64 1.069x
    (320, 64, 256, "am"): (2, {}),  # b39: 1.091x
    (384, 64, 256, "ak"): (1, _BD2),  # native + direct B: 1.033x
    (384, 64, 256, "am"): (1, _BD2),  # 1.018x
    (384, 32, 256, "ak"): (1, _BD2),  # 1.011x
    (512, 32, 256, "ak"): (1, _BD2),  # 1.015x
    # MT1 > 64 (bubble loop b39, persistent unless noted)
    (32, 128, 256, "ak"): (8, _PERS),  # 1.889x
    (32, 128, 256, "am"): (8, _PERS),  # 2.334x
    (32, 256, 256, "am"): (8, _PERS),  # 3.886x
    (64, 128, 256, "ak"): (8, {**_PERS, "read_every": 3}),  # 1.386x (read_every 3: no spill, as PERSIST_KNOBS)
    (64, 128, 256, "am"): (8, _PERS),  # 1.530x
    (64, 192, 256, "ak"): (4, {**_PERS, **_NTB}),  # 1.679x; b_aux 2 1.191x (h4)
    (64, 192, 256, "am"): (4, {**_PERS, **_NTB}),  # 1.646x; b_aux 2 1.125x
    (64, 192, 512, "ak"): (4, _PERS),  # 1.725x
    (64, 192, 512, "am"): (4, _PERS),  # 1.593x
    (64, 256, 256, "am"): (4, _PERS),  # 2.149x
    (96, 128, 256, "ak"): (4, _PERS),  # 1.148x
    (96, 128, 256, "am"): (4, {}),  # 1.157x (dp)
    (128, 96, 256, "ak"): (4, _PERS),  # 1.164x
    (128, 128, 256, "ak"): (4, {**_PERS, "read_every": 3, "dtl_aux": 2}),  # 1.084x; nt both operands 1.020x
    (128, 128, 256, "am"): (4, {**_PERS, "dtl_aux": 2}),  # 1.097x; 1.027x
}
#: kernel agent (job 68374996, n01-09, G problem, verified): the bubble loop's best confirmed G arm (b30 / b39-b41)
#: vs the G dispatch above, head to head (TF/s / mJ per GEMM vs it); kept where job 68375670 (same protocol, the whole
#: changed dispatch on the G problem) agreed: 32x128 ak dtl_spread 2, 160x128x512 ak and 256x128 ak did not (<= 1 %)
G_WG.update({
    (32, 128, 256, 'am'): (8, {'schedule': 'persistent', 'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (1, 2), 'b_direct': True, 'stages': 4}),  # x1.028 / mJ x0.971
    (32, 128, 512, 'ak'): (8, {}),  # x1.311 / mJ x0.765
    (32, 128, 512, 'am'): (8, {'schedule': 'persistent'}),  # x1.434 / mJ x0.699
    (32, 256, 256, 'am'): (8, {'schedule': 'persistent', 'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (1, 4), 'b_direct': True, 'stages': 4, 'read_lead': 2}),  # x1.073 / mJ x0.930
    (64, 128, 512, 'am'): (8, {}),  # x1.084 / mJ x0.921
    (64, 256, 256, 'am'): (4, {'schedule': 'persistent', 'frag_mode': 'jit', 'acq_slot': -1, 'ring': 'stage', 'waves': (1, 4), 'b_direct': True, 'stages': 4, 'read_lead': 2}),  # x1.076 / mJ x0.932
    (128, 96, 512, 'ak'): (4, {'schedule': 'persistent'}),  # x1.044 / mJ x0.954
    (128, 96, 512, 'am'): (4, {'schedule': 'persistent'}),  # x1.040 / mJ x0.959
    (128, 128, 512, 'am'): (4, {'schedule': 'persistent'}),  # x1.034 / mJ x0.967
    # the 512x128 persistent WG spills without read_every 3 (as PERSIST_KNOBS); job 68375670:
    (128, 128, 512, 'ak'): (4, {'schedule': 'persistent', 'read_every': 3}),  # x1.024 / mJ x0.975
    (64, 128, 512, 'ak'): (8, {'schedule': 'persistent', 'read_every': 3}),  # x1.072 / mJ x0.929
    (256, 96, 256, 'ak'): (2, {}),  # x1.028 / mJ x0.973
    (256, 96, 256, 'am'): (2, {}),  # x1.026 / mJ x0.974
    (384, 32, 256, 'ak'): (2, {}),  # x1.042 / mJ x0.975
})
#: MoE lane h5 (job 68376071, compiler out-68375756; every arm verified + every mutant failing): the weights' loads
#: non-temporal (b_aux 2; DTL refills and direct-B VGPR loads) on the G-table kernel, x vs that kernel on one GPU
#: (tile-library.md G13). Kept where >= 1.02x; 128x128 / 128x96 / 256x96 / 64x256 am stay plain (0.92-1.01x).
_G_NT = {
    (32, 32, 512, "ak"): 1.038, (32, 32, 512, "am"): 1.078, (32, 128, 256, "ak"): 1.131, (32, 128, 256, "am"): 1.162,
    (32, 128, 512, "ak"): 1.109, (32, 128, 512, "am"): 1.126, (32, 256, 256, "am"): 1.065,
    (64, 64, 256, "ak"): 1.102, (64, 64, 512, "ak"): 1.117, (64, 64, 512, "am"): 1.112,
    (64, 128, 256, "ak"): 1.159, (64, 128, 256, "am"): 1.133, (64, 128, 512, "ak"): 1.112, (64, 128, 512, "am"): 1.091,
    (64, 192, 512, "ak"): 1.187, (64, 192, 512, "am"): 1.126, (96, 128, 256, "am"): 1.022,
    (128, 32, 512, "ak"): 1.105, (128, 32, 512, "am"): 1.103, (128, 64, 256, "ak"): 1.096, (128, 64, 256, "am"): 1.125,
    (128, 64, 512, "ak"): 1.118, (128, 64, 512, "am"): 1.114, (192, 64, 256, "ak"): 1.042, (192, 64, 256, "am"): 1.038,
    (192, 64, 512, "ak"): 1.042, (192, 64, 512, "am"): 1.043, (256, 32, 256, "ak"): 1.048, (256, 32, 256, "am"): 1.048,
    (256, 64, 256, "ak"): 1.040, (320, 64, 256, "ak"): 1.042, (320, 64, 256, "am"): 1.046, (384, 32, 256, "ak"): 1.073,
    (384, 64, 256, "ak"): 1.036, (384, 64, 256, "am"): 1.050, (512, 32, 256, "ak"): 1.059,
}
for _k in _G_NT:
    if _k not in G_WG:  # a rule-based (G_WG_ROWS) entry: pin its g0 with the knob
        _g0 = 1
        while _k[0] * _g0 * 2 <= G_WG_ROWS:
            _g0 *= 2
        G_WG[_k] = (_g0, {})
    G_WG[_k] = (G_WG[_k][0], {**G_WG[_k][1], **_NTB})
# 64x64 am: the stage-ring (DTL) path + nt weights beats direct B + nt weights (1.114 vs 1.073x the direct-B kernel)
G_WG[(64, 64, 256, "am")] = (8, dict(_NTB))
#: measured: the native tile is best (no G work group)
G_WG_NONE = frozenset({(128, 96, 256, "am")})


def g_wg(mt0: int, mt1: int, du: int, family: str, m: int, n: int) -> Optional[Tuple[int, Dict[str, object]]]:
    """(g0, knobs) of the G-regime work group (``G_WG`` / the ``G_WG_ROWS`` rule), or None outside the regime."""
    key = (mt0, mt1, du, family)
    if not mt1 <= m < 2 * mt1 or key in G_WG_NONE:
        return None
    if key in G_WG:
        g0, kn = G_WG[key]
    elif mt1 <= G_WG_MT1:
        g0, kn = 1, {}
        while mt0 * g0 * 2 <= G_WG_ROWS:
            g0 *= 2
        if g0 == 1:
            return None
    else:
        return None
    while g0 > 1 and mt0 * g0 > n:
        g0 //= 2
    return g0, dict(kn)


def grouped_spec(mt0: int, mt1: int, du: int, family: str, fmt: str, n: int, max_tokens: int,
                 k_target: Optional[int] = 8192, **over) -> Mxfp6GemmSpec:
    """The grouped (MoE) kernel of config ``MT0 x MT1 x DepthU`` (``group_table``: one packed A / C, the group of each
    tile row from a device table): the G-regime work group (``G_WG``) when every group fits one token tile row
    (``max_tokens`` <= MT1: decode-like routing), else the shipped ``config_spec`` WG with g1 = 1 (a group's rows
    are padded to the tile's token rows). CTA order (MoE lane m2 / m3): ``group_m`` 1 when every group is one tile
    row (its N tiles back to back: +2-15 %), the default ``group_m`` 8 when groups span several tile rows (their
    rows share each weight panel: g1 0.95-0.96x there)."""
    single = max_tokens <= mt1
    base = {"edge_tiles": True, "group_table": True, "schedule": "dp", "k_target": k_target,
            **({"group_m": 1} if single else {})}
    gw = g_wg(mt0, mt1, du, family, mt1, n) if single else None
    while gw is not None and gw[0] >= 1:
        try:
            return family_spec(mt0 * gw[0], mt1, du, family, fmt, **{**base, **gw[1], **over})
        except ValueError:
            gw = (gw[0] // 2, gw[1]) if gw[0] > 1 else None
    g0 = WG_MULT.get((mt0, mt1, du, family), (1, 1))[0]
    while g0 > 1 and mt0 * g0 > n:
        g0 //= 2
    return family_spec(mt0 * g0, mt1, du, family, fmt, **{**base, **over})


#: persistent builds that spill at the family defaults and compile clean with these knobs (compile gate; GPU job
#: 68372085: persistent + read_every 3 = persistent, both ~1.07x dp): the line2m 256x256-class work groups
PERSIST_KNOBS: Dict[Tuple[int, int, int, str], Dict[str, object]] = {
    k: {"read_every": 3} for k in ((128, 128, 256, "ak"), (128, 128, 512, "ak"), (128, 256, 256, "ak"),
                                   (256, 128, 256, "ak"), (256, 256, 256, "ak"), (512, 64, 256, "ak"))}

#: persistent builds that still spill (8-wave am work groups at the 256-VGPR budget): data-parallel until fixed
#: (persistent would be ~1.07x, job 68372085 -- the am-family agent's item)
PERSIST_SPILL: frozenset = frozenset({(128, 320, 256, "am"), (256, 160, 256, "am"), (256, 320, 256, "am")})


def line2m_ok(mt0: int, mt1: int, family: str = "ak", waves: Optional[Tuple[int, int]] = None) -> bool:
    """Whether ``ring="line2m"`` applies: the K-contiguous family, A rows (MT1) a multiple of 128 (16 row classes x 8
    row blocks per line DMA), 4 waves or fewer (the line rings need prefetch fragments)."""
    if family != "ak" or mt1 % 128:
        return False
    wm, wn = waves if waves else wave_layout(mt1, mt0)
    return wm * wn <= 4


def line2m_knobs(spec: Mxfp6GemmSpec) -> Dict[str, object]:
    """The record kernel's line2m knobs (best_mixed: 64 MFMAs per 128-K tile) scaled to ``spec``'s MFMAs per tile:
    the release barrier halfway between the last ring read and the tile's last MFMA, the next period's scale load in
    the middle of the tile, the stage DMAs spread evenly below 64 MFMAs."""
    from .gemm import last_ring_read, mfmas_per_tile

    n = mfmas_per_tile(spec)
    last = last_ring_read(spec)
    rel = min(n - 2, last + max(1, (n - 1 - last) // 2))
    return {"line_release": rel, "scale_slot": min(32, max(1, n // 2)), "dtl_spread": 4 if n >= 64 else 0,
            "mini_copy_gap": min(4, max(0, rel - 1))}


def _line2m(base: Mxfp6GemmSpec, occ_max: int, ks: int) -> Mxfp6GemmSpec:
    """``ring="line2m"`` of a stage-ring family default (see ``family_spec``)."""
    if base.a_layout != "canonical" or base.tile_m % 128 or base.waves_m * base.waves_n > 4:
        raise ValueError("ring line2m: the ak family, tile_m a multiple of 128, <= 4 waves")
    l2 = replace(base, canon_ring="line2m", line_reads_first=True, canon_read="b96x2", swizzle="none",
                 store_nt=True, peel_first=True, m_outer=False, canon_rsrc_once=False, frag_mode="prefetch")
    nw = base.waves_m * base.waves_n
    if (base.tile_n * 96 // 1024) % nw == 0 and (base.tile_n * 96) % 1024 == 0:
        l2 = replace(l2, dtl_imm=True)
    per = 8 * ks // _gcd(8, ks)  # line period 8 and whole DepthU periods
    why = ""
    for occ in range(occ_max, 0, -1):
        for S in (3, 4, 2):
            # few MFMAs per tile: the ring reads must end early enough for the release before the last MFMA
            for acq, every in ((0, 2), (0, 1), (-1, 1)):
                unroll = per // _gcd(S, per)
                s = replace(l2, stages=S, occupancy=occ, unroll=unroll, acq_slot=acq, read_every=every)
                s = replace(s, **line2m_knobs(s))
                if lds_bytes(s) * occ > LDS:
                    why = "LDS"
                    continue
                try:
                    ok, why = validate_spec(replace(s, k=library_k_ref(s)))
                except ValueError as e:
                    ok, why = False, str(e)
                if ok:
                    return s
    raise ValueError(f"ring line2m: no ring fits {base.tile_n}x{base.tile_m} ({why})")


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def describe(spec: Mxfp6GemmSpec) -> Dict[str, object]:
    """Static resources of a family spec (no compile): waves, LDS, occupancy target, accumulators per wave."""
    return {"tile": f"{spec.tile_n}x{spec.tile_m}", "waves": f"{spec.waves_m}x{spec.waves_n}",
            "stages": spec.stages, "depth_u": spec.depth_u, "unroll": spec.unroll, "occupancy": spec.occupancy,
            "lds": lds_bytes(spec), "acc_per_wave": acc_regs(spec), "k_ref": spec.k,
            "scale_img": (scale_img_bytes(spec, "A"), scale_img_bytes(spec, "B"))}


# ------------------------------------------------------------------ am via "transpose + ak" (am lane; default off)
#: The ``am`` family (M-contiguous activations, hipBLASLt NT / TT) has two implementations per config:
#: * ``native``: the am kernel (``dispatch(..., "am", ...)``: transposing ``ds_read_b96_tr_b6`` reads of the [K][M] A);
#: * ``transpose``: the bandwidth-bound transpose kernel (``transpose.py``: [K][M] -> the canonical [M][K] codes in a
#:   workspace), then the K-contiguous kernel of the same config (``dispatch(..., "ak", ...)``). The transpose moves
#:   ~1.5 M K bytes per entry against ~2 M N K FLOP, so it pays off from some N on, per config.
#: ``am_plan`` picks one per problem from the measured model ``AM_MODEL``; configs without an entry stay native, so
#: an empty model is the native-only library.

#: the transpose kernel of the transpose path: ``transpose.pick(M, K, batch)`` (shape and K chunk by problem size),
#: with these ``transpose.TrSpec`` overrides on top (empty: the pick as is)
AM_TRANSPOSE: Dict[str, object] = {}
#: (MT0, MT1, DepthU) -> (rho_x3, rho_x4, kappa, f0), measured (notes/tile-library.md "am lane"; shipped dispatch
#: kernels, warm (steady-state), same GPU, every arm verified with its mutants): rho = native am time / ak time (median
#: over the config's measured problems; x3 / x4 = the native kernel with ``mc_dma`` "x3" / "x4", ``AM_MC_DMA`` picks);
#: the transpose costs t_ak (kappa / N + f0 / (b M N K)) (kappa ~ 0.75 R / BW and f0 ~ c0 R / 2 for the config's ak rate
#: R, the transpose's streaming bandwidth BW and its fixed launch / fill cost c0, all fitted). Predicted speed of the
#: transpose path over native am: rho / (1 + kappa / N + f0 / (b M N K)).
#: fit of jobs 68371919 / 68372066 (n01-09; 121 cells: S of all 70 configs + N 8192-28672 / MoE on 10; 119 / 121
#: decisions right or within 2 %, no transpose pick that loses); rho_x3 of 32 configs refitted on the current tree
#: (shipped x3 + STRUCT / round-5 direct-B kernels vs the shipped ak, job 68375138: 32 / 32 decisions right) with the
#: transpose terms of the first fit -- amlane/h/fit_model.py
AM_MODEL: Dict[Tuple[int, int, int], Tuple[float, float, float, float]] = {  # (rho_x3, rho_x4, kappa, f0)
    (32, 32, 512): (1.391, 1.434, 175, 1.67e+09),
    (32, 128, 256): (1.391, 1.426, 276, 2.64e+09),
    (32, 128, 512): (1.035, 1.241, 271, 2.59e+09),
    (32, 256, 256): (0.916, 0.996, 275, 2.63e+09),
    (32, 384, 256): (1.148, 1.413, 336, 3.21e+09),
    (32, 512, 256): (0.875, 0.847, 264, 2.52e+09),
    (64, 64, 256): (1.444, 1.485, 408, 3.9e+09),
    (64, 64, 512): (1.470, 1.505, 413, 3.95e+09),
    (64, 128, 256): (1.088, 1.126, 457, 4.37e+09),
    (64, 128, 512): (1.086, 1.124, 457, 4.37e+09),
    (64, 192, 256): (1.129, 1.350, 491, 4.69e+09),
    (64, 192, 512): (1.242, 1.347, 488, 4.66e+09),
    (64, 256, 256): (1.030, 1.105, 463, 4.43e+09),
    (64, 320, 256): (0.956, 1.096, 410, 3.92e+09),
    (64, 384, 256): (1.156, 1.333, 497, 4.75e+09),
    (64, 448, 256): (0.886, 0.893, 363, 3.47e+09),
    (64, 512, 256): (0.798, 0.879, 369, 3.53e+09),
    (96, 128, 256): (1.094, 1.153, 543, 5.2e+09),
    (96, 128, 512): (1.085, 1.155, 554, 5.3e+09),
    (96, 256, 256): (1.098, 1.164, 558, 5.33e+09),
    (96, 384, 256): (0.791, 0.987, 415, 3.96e+09),
    (128, 32, 256): (1.143, 1.151, 314, 3.01e+09),
    (128, 32, 512): (1.142, 1.144, 312, 2.98e+09),
    (128, 64, 256): (1.153, 1.181, 497, 4.75e+09),
    (128, 64, 512): (1.122, 1.151, 469, 4.49e+09),
    (128, 96, 256): (0.980, 1.083, 513, 4.9e+09),
    (128, 96, 512): (0.991, 1.094, 503, 4.81e+09),
    (128, 128, 256): (1.261, 1.300, 608, 5.82e+09),
    (128, 128, 512): (1.232, 1.292, 606, 5.79e+09),
    (128, 160, 512): (1.013, 1.019, 483, 4.62e+09),
    (128, 192, 256): (0.982, 1.092, 508, 4.86e+09),
    (128, 224, 256): (1.148, 1.203, 534, 5.1e+09),
    (128, 256, 256): (1.231, 1.250, 623, 5.95e+09),
    (128, 320, 256): (1.016, 1.019, 484, 4.63e+09),
    (128, 384, 256): (1.237, 1.429, 520, 4.97e+09),
    (128, 448, 256): (0.895, 0.901, 370, 3.54e+09),
    (160, 128, 512): (1.140, 1.163, 510, 4.88e+09),
    (160, 256, 256): (1.000, 1.010, 539, 5.15e+09),
    (160, 384, 256): (1.062, 1.230, 517, 4.94e+09),
    (192, 64, 256): (1.372, 1.411, 552, 5.28e+09),
    (192, 64, 512): (1.387, 1.430, 564, 5.4e+09),
    (192, 128, 256): (1.109, 1.175, 563, 5.38e+09),
    (192, 256, 256): (1.131, 1.167, 573, 5.48e+09),
    (192, 320, 256): (1.023, 1.134, 493, 4.71e+09),
    (192, 384, 256): (0.915, 1.011, 468, 4.47e+09),
    (224, 256, 256): (1.176, 1.207, 592, 5.65e+09),
    (256, 32, 256): (1.022, 1.081, 287, 2.74e+09),
    (256, 64, 256): (1.068, 1.089, 489, 4.68e+09),
    (256, 96, 256): (0.976, 1.101, 513, 4.91e+09),
    (256, 128, 256): (1.283, 1.314, 608, 5.81e+09),
    (256, 160, 256): (0.960, 1.093, 511, 4.88e+09),
    (256, 192, 256): (0.977, 1.191, 509, 4.86e+09),
    (256, 256, 256): (1.176, 1.254, 606, 5.79e+09),
    (256, 288, 256): (1.067, 1.021, 545, 5.21e+09),
    (256, 320, 256): (0.967, 1.012, 517, 4.94e+09),
    (288, 256, 256): (1.129, 1.021, 582, 5.56e+09),
    (320, 64, 256): (1.137, 1.157, 508, 4.86e+09),
    (320, 128, 256): (0.960, 0.991, 515, 4.92e+09),
    (320, 192, 256): (1.118, 1.283, 548, 5.24e+09),
    (384, 32, 256): (1.099, 1.200, 359, 3.44e+09),
    (384, 64, 256): (1.372, 1.425, 560, 5.35e+09),
    (384, 96, 256): (1.072, 0.974, 552, 5.28e+09),
    (384, 128, 256): (1.135, 1.393, 563, 5.38e+09),
    (384, 192, 256): (1.096, 1.100, 547, 5.23e+09),
    (448, 64, 256): (1.109, 1.082, 483, 4.62e+09),
    (448, 128, 256): (1.080, 1.095, 476, 4.55e+09),
    (512, 32, 256): (1.348, 1.169, 383, 3.66e+09),
    (512, 64, 256): (1.084, 1.098, 501, 4.79e+09),
}
#: the transpose path is taken when its predicted speed over native am is at least this
AM_MARGIN = 1.03


class AmPlan:
    """What the library runs for an M-contiguous activation (``am``) call: ``path`` "native" (``gemm`` = the am
    kernel) or "transpose" (``transpose`` = the transpose kernel spec, then ``gemm`` = the ak kernel on the
    workspace); ``chunk`` > 0: the batch goes through (transpose, GEMM) launch pairs of ``chunk`` entries sharing one
    ``chunk``-entry workspace (it stays cache-resident between the two kernels)."""

    __slots__ = ("path", "gemm", "transpose", "chunk", "predicted")

    def __init__(self, path: str, gemm: Mxfp6GemmSpec, transpose=None, chunk: int = 0, predicted: float = 1.0):
        self.path, self.gemm, self.transpose, self.chunk, self.predicted = path, gemm, transpose, chunk, predicted

    def workspace_bytes(self, m: int, k: int, batch: int = 1) -> int:
        if self.path != "transpose":
            return 0
        return (self.chunk or batch) * m * k * 6 // 8

    def __repr__(self) -> str:
        return f"AmPlan({self.path}, chunk={self.chunk}, predicted={self.predicted:.3f})"


def am_predicted(mt0: int, mt1: int, du: int, m: int, n: int, k: int, batch: int = 1) -> float:
    """Model speed of the transpose path over native am for config ``MT0 x MT1 x DepthU`` on (M, N, K, batch)
    (1.0: no model for the config)."""
    mdl = AM_MODEL.get((mt0, mt1, du))
    if mdl is None:
        return 1.0
    rho_x3, rho_x4, kappa, f0 = mdl
    rho = rho_x3 if AM_MC_DMA == "x3" else rho_x4
    return rho / (1.0 + kappa / max(1, n) + f0 / max(1, batch * m * n * k))


def am_plan(mt0: int, mt1: int, du: int, fmt: str, m: int, n: int, k: int, batch: int = 1, *, path: str = "auto",
            chunk: int = -1, **over) -> AmPlan:
    """The ``am`` implementation of config ``MT0 x MT1 x DepthU`` on problem (m, n, k, batch): ``path`` "auto" (the
    model ``AM_MODEL``), "native" or "transpose"; ``chunk`` -1: the default (``am_chunk``), 0: one launch pair."""
    from .transpose import pick as tr_pick, validate as tr_validate

    if path not in ("auto", "native", "transpose"):
        raise ValueError("path must be auto, native or transpose")
    ch0 = am_chunk(m, k, batch) if chunk < 0 else chunk
    tr = replace(tr_pick(m, k, ch0 or batch), **AM_TRANSPOSE)
    usable = m >= 1 and k % tr.tk == 0 and tr_validate(tr)[0]  # any M (the M-padded input, transpose.padded_m)
    auto, pred = _am_auto(mt0, mt1, du, fmt, m, n, k, batch, over, usable)
    pred = pred or 1.0
    if path == "auto":
        path = auto
    if path == "native":
        return AmPlan("native", dispatch(mt0, mt1, du, "am", fmt, m, n, batch, k_target=k, **over), predicted=pred)
    if not usable:
        raise ValueError(f"transpose path: K a multiple of {tr.tk}")
    ch = ch0
    gb = ch if ch else batch
    gemm = dispatch(mt0, mt1, du, "ak", fmt, m, n, gb, k_target=k, **over)
    return AmPlan("transpose", gemm, tr, ch, pred)


#: The masked native regime (the problem's native am kernel masks rows past M, ``m_mask``: M % 16 != 0 or M < the
#: tile, sk lane): AM_MODEL (fit on the plain kernels) does not apply there. Measured transpose-path speed over native
#: am, same GPU, both paths verified + mutants (h/lb_anym.json: the suite's 24 am gap problems on their select_plan
#: config; job 68376885 n01-09, lbanyT2): key (MT0, MT1, DepthU, batch, M, N, K).
AM_PLAN_MEASURED: Dict[Tuple[int, int, int, int, int, int, int], float] = {
    (32, 32, 512, 128, 30, 3072, 4096): 0.758, (32, 32, 512, 256, 4, 7168, 2048): 0.080,
    (32, 128, 512, 1, 25, 5120, 2880): 0.735, (64, 64, 256, 1, 8, 6336, 3328): 0.590,
    (64, 64, 256, 1, 645, 7168, 16384): 1.588, (64, 64, 256, 1, 889, 2321, 3584): 0.785,
    (64, 64, 256, 1, 2811, 645, 4608): 0.742, (64, 64, 512, 256, 51, 4096, 7168): 1.147,
    (64, 128, 512, 1, 667, 1391, 6144): 0.788, (64, 128, 512, 1, 2109, 2997, 2048): 1.274,
    (64, 128, 512, 1, 3749, 1797, 3840): 0.810, (64, 192, 512, 128, 174, 5760, 2880): 0.813,
    (64, 256, 256, 1, 219, 4096, 8192): 1.044, (96, 128, 256, 1, 75, 36864, 7168): 1.152,
    (96, 128, 512, 1, 1185, 835, 2816): 0.601, (96, 128, 512, 1, 1581, 1077, 2816): 0.599,
    (128, 32, 256, 1, 2, 9728, 10752): 1.047, (128, 32, 256, 1, 3, 8192, 8192): 0.678,
    (128, 32, 256, 1, 4, 14976, 4608): 0.792, (128, 32, 256, 1, 9, 24576, 1536): 0.520,
    (128, 256, 256, 1, 1896, 57344, 8192): 1.257, (256, 32, 256, 128, 1, 2880, 2880): 0.955,
    (256, 32, 256, 128, 1, 4096, 1536): 0.911, (256, 32, 256, 128, 1, 5760, 2880): 0.975,
}
#: masked regime, problems not in AM_PLAN_MEASURED: the transpose path when M >= 16 and K >= this (every measured cell
#: with both won, 1.04-1.59x: 645 / 1896 / 75 / 219 x 7168-16384, 256 x 51 x 7168; every loss has K <= 6144 or M < 16)
AM_MASKED_TR_K = 7168


def _am_auto(mt0: int, mt1: int, du: int, fmt: str, m: int, n: int, k: int, batch: int, over,
             usable: bool = True) -> Tuple[str, Optional[float]]:
    """am_plan's "auto" path and its predicted transpose-path speed over native (None: unknown, e.g. no native kernel):
    no native kernel -> transpose; the masked native regime -> AM_PLAN_MEASURED, else the AM_MASKED_TR_K rule; plain
    native kernels -> AM_MODEL (am_predicted) against AM_MARGIN."""
    from .gemm import runtime_problem_ok

    if not usable:
        return "native", None
    try:
        nat = dispatch(mt0, mt1, du, "am", fmt, m, n, batch, k_target=k, **over)
        ok = runtime_problem_ok(nat, m, n, k, batch)[0]
    except ValueError:
        nat, ok = None, False
    if not ok:
        return "transpose", None
    if nat.m_mask:  # (rows past M masked: M % 16 != 0 or M < the tile; every AM_PLAN_MEASURED cell)
        r = AM_PLAN_MEASURED.get((mt0, mt1, du, batch, m, n, k))
        if r is not None:
            return ("transpose" if r >= AM_MARGIN else "native"), r
        return ("transpose" if (m >= 16 and k >= AM_MASKED_TR_K) else "native"), None
    pred = am_predicted(mt0, mt1, du, m, n, k, batch)
    return ("transpose" if pred >= AM_MARGIN else "native"), pred


def _am_native_ok(mt0: int, mt1: int, du: int, fmt: str, m: int, n: int, k: int, batch: int, over) -> bool:
    """Whether the native am kernel of the config runs the problem (else am_plan takes the transpose + ak path)."""
    from .gemm import runtime_problem_ok

    try:
        return runtime_problem_ok(dispatch(mt0, mt1, du, "am", fmt, m, n, batch, k_target=k, **over), m, n, k, batch)[0]
    except ValueError:
        return False


#: SB-3 transpose model: the transpose kernel streams its bytes (``transpose.traffic_bytes``) at this fraction of
#: ``SB3_HBM_BPS`` (am lane fit of the shipped picks, tile-library.md "am lane") plus one launch
AM_TR_HBM_FRAC = 0.71


def am_transpose_seconds(m: int, k: int, batch: int = 1) -> float:
    """SB-3 time of the transpose kernel of an am problem."""
    from .transpose import traffic_bytes

    return SB3_LAUNCH_S + traffic_bytes(m, k, batch) / (AM_TR_HBM_FRAC * SB3_HBM_BPS)


#: the transpose path's workspace target: a chunk of entries whose transposed A fits this many bytes (the 256 MiB
#: MALL holds it between the transpose and the GEMM); 0 disables chunking
AM_CHUNK_BYTES = 0


def am_chunk(m: int, k: int, batch: int) -> int:
    """Entries per (transpose, GEMM) launch pair: 0 (one pair) unless ``AM_CHUNK_BYTES`` asks for chunks."""
    if not AM_CHUNK_BYTES or batch < 2:
        return 0
    per = max(1, m * k * 6 // 8)
    c = max(1, AM_CHUNK_BYTES // per)
    return 0 if c >= batch else c


#: SB-3 model inputs: each config's shipped kernel on its own S problem, warm, as a fraction of the nominal MFMA peak
#: (MFMA_PEAK_TFS: 16x16x128 f8f6f4 at 2.4 GHz on 256 CUs; the 1400 W cap keeps every kernel well below it). Jobs
#: 68369075-77 (every config's arms on one GPU, GPUs mixed across configs: +-5 %), the 51 configs changed by the c5 /
#: round-5 integration from job 68374995. Refreshed from the scoreboard as the shipped kernels change.
MFMA_PEAK_TFS = 10066.3
EFF_S: Dict[Tuple[int, int, int, str], float] = {
    (32, 32, 512, 'ak'): 0.13, (32, 32, 512, 'am'): 0.079, (32, 128, 256, 'ak'): 0.21, (32, 128, 256, 'am'): 0.141,
    (32, 128, 512, 'ak'): 0.201, (32, 128, 512, 'am'): 0.189, (32, 256, 256, 'ak'): 0.196,
    (32, 256, 256, 'am'): 0.18, (32, 384, 256, 'ak'): 0.197, (32, 384, 256, 'am'): 0.17, (32, 512, 256, 'ak'): 0.151,
    (32, 512, 256, 'am'): 0.156, (64, 64, 256, 'ak'): 0.26, (64, 64, 256, 'am'): 0.202, (64, 64, 512, 'ak'): 0.273,
    (64, 64, 512, 'am'): 0.191, (64, 128, 256, 'ak'): 0.291, (64, 128, 256, 'am'): 0.245,
    (64, 128, 512, 'ak'): 0.289, (64, 128, 512, 'am'): 0.259, (64, 192, 256, 'ak'): 0.29, (64, 192, 256, 'am'): 0.25,
    (64, 192, 512, 'ak'): 0.282, (64, 192, 512, 'am'): 0.245, (64, 256, 256, 'ak'): 0.263,
    (64, 256, 256, 'am'): 0.234, (64, 320, 256, 'ak'): 0.234, (64, 320, 256, 'am'): 0.244,
    (64, 384, 256, 'ak'): 0.286, (64, 384, 256, 'am'): 0.243, (64, 448, 256, 'ak'): 0.23, (64, 448, 256, 'am'): 0.23,
    (64, 512, 256, 'ak'): 0.235, (64, 512, 256, 'am'): 0.227, (96, 128, 256, 'ak'): 0.314,
    (96, 128, 256, 'am'): 0.274, (96, 128, 512, 'ak'): 0.314, (96, 128, 512, 'am'): 0.271,
    (96, 256, 256, 'ak'): 0.311, (96, 256, 256, 'am'): 0.274, (96, 384, 256, 'ak'): 0.24,
    (96, 384, 256, 'am'): 0.299, (128, 32, 256, 'ak'): 0.232, (128, 32, 256, 'am'): 0.166,
    (128, 32, 512, 'ak'): 0.232, (128, 32, 512, 'am'): 0.166, (128, 64, 256, 'ak'): 0.294,
    (128, 64, 256, 'am'): 0.258, (128, 64, 512, 'ak'): 0.3, (128, 64, 512, 'am'): 0.258, (128, 96, 256, 'ak'): 0.297,
    (128, 96, 256, 'am'): 0.255, (128, 96, 512, 'ak'): 0.297, (128, 96, 512, 'am'): 0.264,
    (128, 128, 256, 'ak'): 0.339, (128, 128, 256, 'am'): 0.278, (128, 128, 512, 'ak'): 0.339,
    (128, 128, 512, 'am'): 0.282, (128, 160, 512, 'ak'): 0.268, (128, 160, 512, 'am'): 0.27,
    (128, 192, 256, 'ak'): 0.281, (128, 192, 256, 'am'): 0.265, (128, 224, 256, 'ak'): 0.304,
    (128, 224, 256, 'am'): 0.262, (128, 256, 256, 'ak'): 0.354, (128, 256, 256, 'am'): 0.286,
    (128, 320, 256, 'ak'): 0.295, (128, 320, 256, 'am'): 0.303, (128, 384, 256, 'ak'): 0.294,
    (128, 384, 256, 'am'): 0.24, (128, 448, 256, 'ak'): 0.229, (128, 448, 256, 'am'): 0.231,
    (160, 128, 512, 'ak'): 0.292, (160, 128, 512, 'am'): 0.251, (160, 256, 256, 'ak'): 0.306,
    (160, 256, 256, 'am'): 0.306, (160, 384, 256, 'ak'): 0.292, (160, 384, 256, 'am'): 0.276,
    (192, 64, 256, 'ak'): 0.326, (192, 64, 256, 'am'): 0.223, (192, 64, 512, 'ak'): 0.325,
    (192, 64, 512, 'am'): 0.223, (192, 128, 256, 'ak'): 0.315, (192, 128, 256, 'am'): 0.274,
    (192, 192, 256, 'ak'): 0.256, (192, 192, 256, 'am'): 0.3, (192, 256, 256, 'ak'): 0.325,
    (192, 256, 256, 'am'): 0.253, (192, 320, 256, 'ak'): 0.278, (192, 320, 256, 'am'): 0.274,
    (192, 384, 256, 'ak'): 0.271, (192, 384, 256, 'am'): 0.291, (224, 256, 256, 'ak'): 0.329,
    (224, 256, 256, 'am'): 0.293, (256, 32, 256, 'ak'): 0.225, (256, 32, 256, 'am'): 0.179,
    (256, 64, 256, 'ak'): 0.279, (256, 64, 256, 'am'): 0.253, (256, 96, 256, 'ak'): 0.29, (256, 96, 256, 'am'): 0.26,
    (256, 128, 256, 'ak'): 0.327, (256, 128, 256, 'am'): 0.27, (256, 160, 256, 'ak'): 0.289,
    (256, 160, 256, 'am'): 0.303, (256, 192, 256, 'ak'): 0.292, (256, 192, 256, 'am'): 0.246,
    (256, 256, 256, 'ak'): 0.322, (256, 256, 256, 'am'): 0.292, (256, 288, 256, 'ak'): 0.311,
    (256, 288, 256, 'am'): 0.289, (256, 320, 256, 'ak'): 0.293, (256, 320, 256, 'am'): 0.303,
    (288, 256, 256, 'ak'): 0.331, (288, 256, 256, 'am'): 0.294, (320, 64, 256, 'ak'): 0.292,
    (320, 64, 256, 'am'): 0.251, (320, 128, 256, 'ak'): 0.294, (320, 128, 256, 'am'): 0.307,
    (320, 192, 256, 'ak'): 0.304, (320, 192, 256, 'am'): 0.278, (320, 256, 256, 'ak'): 0.342,
    (320, 256, 256, 'am'): 0.341, (384, 32, 256, 'ak'): 0.211, (384, 32, 256, 'am'): 0.172,
    (384, 64, 256, 'ak'): 0.314, (384, 64, 256, 'am'): 0.224, (384, 96, 256, 'ak'): 0.313,
    (384, 96, 256, 'am'): 0.293, (384, 128, 256, 'ak'): 0.322, (384, 128, 256, 'am'): 0.281,
    (384, 192, 256, 'ak'): 0.253, (384, 192, 256, 'am'): 0.303, (448, 64, 256, 'ak'): 0.261,
    (448, 64, 256, 'am'): 0.237, (448, 128, 256, 'ak'): 0.29, (448, 128, 256, 'am'): 0.255,
    (512, 32, 256, 'ak'): 0.222, (512, 32, 256, 'am'): 0.151, (512, 64, 256, 'ak'): 0.281,
    (512, 64, 256, 'am'): 0.255,
}
#: CUs of the target (MI355X)
N_CU = 256
#: SB-3 model constants: the K tiles of the S problems EFF_S was measured on (K = 8192); the per-wave fill / drain of
#: a work group in K tiles (prologue ring fill + epilogue, ~ stages + 2); the effective HBM read rate (93-96 % of the
#: 6.33 TB/s roof on the G set, tile-library.md G5); the fixed cost of a launch
SB3_KT_REF = 64
SB3_FILL_TILES = 4
SB3_HBM_BPS = 6.0e12
SB3_LAUNCH_S = 4e-6


def _sb3_kernel_seconds(spec: Mxfp6GemmSpec, rate_tfs: float, m: int, n: int, k: int, batch: int) -> float:
    """SB-3 time of one dispatch kernel: max(compute, compulsory bytes / HBM) + launch, where compute = FLOPs over
    the config's S rate, scaled by the machine fill (work groups over N_CU x occupancy, whole waves), the padded-tile
    area and the K-loop fill (``SB3_FILL_TILES`` per work group, already inside the S rate at ``SB3_KT_REF``)."""
    tm, tn = spec.tile_m, spec.tile_n
    gm, gn = -(-m // tm), -(-n // tn)
    split = max(1, spec.sk_split) if spec.splitk != "none" else 1  # split-K: each tile runs as `split` K units
    wgs = gm * gn * batch * split
    slots = N_CU * max(1, spec.occupancy)
    fill = wgs / (-(-wgs // slots) * slots)
    area = (m * n) / (gm * tm * gn * tn)
    kt = -(-k // (128 * split))
    kfill = (SB3_KT_REF + SB3_FILL_TILES) / SB3_KT_REF * kt / (kt + SB3_FILL_TILES)
    t_comp = 2.0 * batch * m * n * k / (rate_tfs * 1e12 * fill * max(area, 1e-9) * kfill)
    t_mem = batch * ((m + n) * k * (0.75 + 1 / 32) + 2 * m * n) / SB3_HBM_BPS
    # HBM-bound problems tie on t_mem: the kernel with less compute time (less energy per GEMM) ranks first
    return max(t_comp, t_mem) + SB3_LAUNCH_S + 1e-3 * t_comp


def predict_seconds(mt0: int, mt1: int, du: int, family: str, m: int, n: int, k: int, batch: int = 1) -> float:
    """SB-3 cost model of config ``MT0 x MT1 x DepthU`` on a problem (``_sb3_kernel_seconds`` of its ``dispatch``
    kernel at the config's S rate). ``am``: the faster of the native am kernel and, where ``am_plan`` takes it, the
    transpose + ak path (``am_predicted``)."""
    path, pred = _am_auto(mt0, mt1, du, "e2m3", m, n, k, batch, {}) if family == "am" else ("native", None)
    if path == "transpose" and pred is None:
        # the transpose + ak launch pair without a measured / modeled ratio (no native kernel, masked-regime rule)
        return predict_seconds(mt0, mt1, du, "ak", m, n, k, batch) + am_transpose_seconds(m, k, batch)
    spec = dispatch(mt0, mt1, du, family, "e2m3", m, n, batch, k_target=k)
    t = _sb3_kernel_seconds(spec, EFF_S.get((mt0, mt1, du, family), 0.1) * MFMA_PEAK_TFS, m, n, k, batch)
    if path == "transpose":
        t /= pred
    return t


def select_candidates(m: int, n: int, k: int, batch: int = 1, family: str = "ak", fmt: str = "e2m3", top: int = 3,
                      configs=None) -> List[Tuple[float, Tuple[int, int, int], Mxfp6GemmSpec]]:
    """SB-3: the ``top`` configs with the lowest ``predict_seconds`` for a problem, (seconds, config, its
    ``dispatch`` kernel) fastest first, among the library configs whose kernel accepts the problem (cached per
    problem: the tables are fixed at import)."""
    return list(_select_ranked(m, n, k, batch, family, fmt, tuple(configs or LIBRARY))[:top])


@functools.lru_cache(maxsize=4096)
def _select_ranked(m: int, n: int, k: int, batch: int, family: str, fmt: str, configs: tuple):
    from .gemm import runtime_problem_ok

    out = []
    for cfg in configs:
        try:
            spec = dispatch(*cfg, family, fmt, m, n, batch, k_target=k)
            ok = runtime_problem_ok(spec, m, n, k, batch)[0]
        except (ValueError, ImportError):
            spec, ok = None, False
        try:
            if not ok and family == "am":
                # am without a native kernel for the problem: the transpose + ak path (``am_plan``); the ak kernel
                # stands in as the candidate's spec
                spec = am_plan(*cfg, fmt, m, n, k, batch, path="transpose").gemm
                ok = runtime_problem_ok(spec, m, n, k, batch)[0]
            if ok:
                out.append((predict_seconds(*cfg, family, m, n, k, batch), tuple(cfg), spec))
        except (ValueError, ImportError):
            continue
    out.sort(key=lambda x: x[0])
    return tuple(out)


def select(m: int, n: int, k: int, batch: int = 1, family: str = "ak", fmt: str = "e2m3",
           configs=None) -> Tuple[Tuple[int, int, int], Mxfp6GemmSpec]:
    """SB-3: the config (and its ``dispatch`` kernel) with the lowest ``predict_seconds`` for a problem, among the
    library configs whose kernel accepts it (``runtime_problem_ok``). ``am``: run it through ``select_plan`` (the
    transpose + ak path is a launch pair, not one kernel)."""
    c = select_candidates(m, n, k, batch, family, fmt, 1, configs)
    if not c:
        raise ValueError(f"no library kernel runs {batch} x {m} x {n} x {k} ({family})")
    return c[0][1], c[0][2]


def select_plan(m: int, n: int, k: int, batch: int = 1, family: str = "ak", fmt: str = "e2m3", configs=None):
    """SB-3 out of the box: (config, plan) for a problem -- ``ak``: the selected ``dispatch`` kernel (an ``AmPlan``
    with path "native"); ``am``: ``am_plan`` of the selected config (native am or transpose + ak)."""
    cfg, spec = select(m, n, k, batch, family, fmt, configs)
    if family != "am":
        return cfg, AmPlan("native", spec)
    try:
        return cfg, am_plan(*cfg, fmt, m, n, k, batch)
    except ValueError:  # the transpose path's ak kernel does not take the problem: native am
        return cfg, AmPlan("native", spec, predicted=am_predicted(*cfg, m, n, k, batch))
