# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Plan choice for the gfx942 register-staged (``vgpr_prefetch``) GEMM body.

A *plan* is one launch configuration of the register-staged body for a
request: a geometry of the measured menu (tile M / N / K, waves,
interleave mode, LDS swizzle row shift) with a split-K degree and the split-K
reduction (``split_k_serial`` 0 = f32 atomics into a zeroed workspace + a cast
pass, 2 = the in-launch write-through reduction); or, for the arbitration
against the gemm_skinny candidate, that candidate's own plan.

* **menu** (``selector_models/g5_plan_<arch>.menu.json``): every geometry that
  compiled without spills and was timed on a full wave of tiles (one
  workgroup per CU) at two K extents: per-tile time ``a + b * K`` and the
  resident workgroups per CU (registers, LDS).
* **analytic model** (:func:`analytic_us`): waves of workgroups x (a + b * K
  slice), an operand-traffic floor and the split-K reduction cost. It ranks
  every plan of the menu; the ``G5P_PRESELECT`` cheapest go to
* **the learned model** (``g5_plan_<arch>.model.txt`` + ``.meta.json``): a
  LightGBM regression of ln(time) over :func:`plan_features` (the analytic
  estimate is one of them), fitted on interleaved gfx942 timings with every
  arm checked against an f64 reference. The plan of least predicted time
  wins; ``G5P_MODEL = False`` keeps the analytic order.

Every knob is a module attribute; ``G5P_ON = False`` restores the fixed-geometry
plan choice (``common.g5_vgpr_plan`` over ``G5_VGPR_GEOS``) exactly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

G5P_ON = True
G5P_ARCHES = ("gfx942",)
# split-K degrees and reductions the plans may use
G5P_SPLITS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
G5P_SERIAL_MODES = (0, 2)
# at most this many workgroups per CU from split-K (more only adds reduction)
G5P_SPLIT_MAX_WAVES = 4
# the analytic model's cheapest plans that the learned model ranks
G5P_PRESELECT = 24
G5P_MODEL = True
# gemm_skinny arbitration: the register-staged candidate steps aside when the
# learned model predicts the gemm_skinny plan faster by this factor (log units)
G5P_SKINNY_ARB = True
G5P_SKINNY_MARGIN = 0.0
# analytic split-K reduction cost: (fixed us, us per byte of f32 partials)
G5P_SPLIT_COST = {0: (4.0, 2.5e-6), 2: (1.0, 1.0e-6)}
# operand streaming ceiling of the analytic floor (bytes per us)
G5P_TRAFFIC_BW = 9.0e6
G5P_CUS = {"gfx942": 304}
# ``vp_unroll`` (D + 1 K tiles per loop iteration, no loop-carried prefetch
# register copies): on for plans with at most this many accumulators per lane
# (0 = never). It helps plans with few accumulators per lane and hurts above
# that bound (AGPR accumulation, register pressure)
G5P_VP_UNROLL_MAX_ACC = 128
# predicted-time factor of an unrolled plan (the learned model is fitted on
# one-tile-per-iteration bodies)
G5P_VP_UNROLL_SCALE = 0.97
# the dispatcher takes the best-ranked of the first
# ``G5P_LAUNCHABLE_TOP`` plans that the launch contract accepts (1 = the top
# plan only: a refused top plan refuses the g5 candidate, e.g. a
# padded tile with operand bytes >= 2^31 at 128 x 32768 x 32768)
G5P_LAUNCHABLE_TOP = 8
# main-loop traits applied to the chosen single-LDS-buffer
# (mode 4) RCR plan (``common._g5p_mainloop``; only when the result is valid
# and launchable), within the scope of :func:`mainloop_applies`. Empty = off.
# The hipBLASLt-style loop (read every fragment of a K tile during its first
# atom, release the LDS buffer, stream load / write pairs through the other
# atoms, pack the next tile's reads behind the second barrier, keep the tile's
# first MFMA ahead of its slot): measured faster on gfx942 for the 4-wave AGPR
# bodies (256x256 / 256x224 / 224x288 w2x2), neutral on the 8-wave
# VGPR-accumulator picks.
G5P_MAINLOOP: Dict[str, int] = {
    "vp_pair_reads": 1,
    "vp_release_atoms": 3,
    "vp_tail_pack": 3,
    "vp_mfma_first": 1,
}
# scope of G5P_MAINLOOP: minimum K tiles of the plan's K loop (per split-K
# slice), accumulators per lane in [MIN_ACC, MAX_ACC] (0 = no bound), waves
# per workgroup (empty = any) and minimum K atoms per K tile
G5P_MAINLOOP_MIN_KTILES = 0
G5P_MAINLOOP_MIN_ACC = 129
G5P_MAINLOOP_MAX_ACC = 0
G5P_MAINLOOP_WAVES: Tuple[int, ...] = (4,)
G5P_MAINLOOP_MIN_KATOMS = 4
# predicted-time factor of a plan the main-loop traits apply to (the learned
# model is fitted on bodies without these traits); 1.0 = rank as before
G5P_MAINLOOP_SCALE = 0.95
# feature value for "not applicable" (the tree walker has no NaN branch)
G5P_NA = -1.0

MODEL_DIR = Path(__file__).with_name("selector_models")

FEATURES: Tuple[str, ...] = (
    "lnM", "lnN", "lnK", "bf16", "lnMN", "lnMNK", "minMN", "k_a8", "k_a64",
    "kind", "tm", "tn", "tk", "waves", "wm", "wn", "mode", "swr", "acc", "split",
    "serial", "vgpr", "lds", "occ", "lntiles", "lnctas", "W", "fill", "fill_occ",
    "ks", "kit", "padm", "padn", "padk", "effm", "effn", "ai", "agpr", "ln_pred",
    "ln_a", "ln_b",
)  # fmt: skip


@dataclass(frozen=True)
class G5Plan:
    """One launch plan: a menu geometry + split-K, or the gemm_skinny plan."""

    geo: str  # menu name "{tm}x{tn}x{tk}_w{wm}x{wn}_m{mode}_r{swr}", or "skinny"
    split: int = 1
    serial: int = 0
    skinny: Any = None  # the SkinnyGemmSpec when geo == "skinny"


def parse_geo(name: str) -> Tuple[int, int, int, int, int, int, int]:
    a, w, m, r = name.split("_")
    tm, tn, tk = (int(x) for x in a.split("x"))
    wm, wn = (int(x) for x in w[1:].split("x"))
    return tm, tn, tk, wm, wn, int(m[1:]), int(r[1:])


def occupancy(vgpr: int, lds: int, waves: int) -> int:
    """Resident workgroups per CU (CDNA3: 512 registers per lane per SIMD,
    granule 8; 4 SIMDs; 64 KiB LDS)."""
    v = max(8, -(-int(vgpr) // 8) * 8)
    by_regs = (max(1, 512 // v) * 4) // max(1, waves)
    by_lds = 65536 // max(1, int(lds)) if lds else 8
    return max(1, min(by_regs, by_lds, 8))


@lru_cache(maxsize=8)
def load_menu(arch: str) -> Dict[str, Dict[str, float]]:
    p = MODEL_DIR / f"g5_plan_{arch}.menu.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


@lru_cache(maxsize=8)
def load_model(arch: str):
    p = MODEL_DIR / f"g5_plan_{arch}.model.txt"
    m = MODEL_DIR / f"g5_plan_{arch}.meta.json"
    if not p.exists() or not m.exists():
        return None
    from .selector import TreeEnsemble

    meta = json.loads(m.read_text())
    if tuple(meta.get("features", ())) != FEATURES:
        raise ValueError(f"{p.name}: feature list does not match g5_plan.FEATURES")
    return TreeEnsemble.from_text(p.read_text())


def analytic_us(
    M: int, N: int, K: int, geo: str, d: Dict[str, float], split: int, serial: int, cus: int
) -> Optional[float]:
    """Waves x (a + b * K slice) with the traffic floor and the split cost
    (None: the split leaves an empty K slice)."""
    tm, tn, tk = parse_geo(geo)[:3]
    tiles = -(-M // tm) * -(-N // tn)
    ctas = tiles * split
    W = -(-ctas // cus)
    ks = -(-K // (split * tk)) * tk
    if split > 1 and ks * (split - 1) >= K:
        return None
    t = W * (d["a"] + d["b"] * ks)
    t = max(t, ctas * (tm + tn) * ks * 2 / G5P_TRAFFIC_BW)
    if split > 1:
        fx, pb = G5P_SPLIT_COST.get(serial, G5P_SPLIT_COST[0])
        t += fx + pb * M * N * 4 * split
    return t


def mainloop_applies(tile, trait, M: int, N: int, K: int) -> bool:
    """Whether ``G5P_MAINLOOP`` applies to a plan (tile, trait) at M x N x K."""
    if int(trait.vgpr_prefetch) != 4:
        return False
    split = max(1, int(trait.split_k))
    ktiles = -(-K // (split * tile.tile_k))
    if ktiles < G5P_MAINLOOP_MIN_KTILES:
        return False
    if tile.tile_k // max(1, tile.warp_tile_k) < G5P_MAINLOOP_MIN_KATOMS:
        return False
    waves = tile.warp_m * tile.warp_n
    if G5P_MAINLOOP_WAVES and waves not in tuple(G5P_MAINLOOP_WAVES):
        return False
    acc = tile.tile_m * tile.tile_n // (64 * waves)
    if G5P_MAINLOOP_MIN_ACC and acc < G5P_MAINLOOP_MIN_ACC:
        return False
    return not G5P_MAINLOOP_MAX_ACC or acc <= G5P_MAINLOOP_MAX_ACC


def plan_mainloop(plan: "G5Plan", M: int, N: int, K: int) -> bool:
    """Whether the dispatcher would run ``plan`` with ``G5P_MAINLOOP`` (the
    ranking scale ``G5P_MAINLOOP_SCALE`` applies to it)."""
    if plan.geo == "skinny" or not G5P_MAINLOOP:
        return False
    tm, tn, tk, wm, wn, mode, _swr = parse_geo(plan.geo)
    if mode != 4:
        return False
    from types import SimpleNamespace as _NS

    tile = _NS(tile_m=tm, tile_n=tn, tile_k=tk, warp_m=wm, warp_n=wn, warp_tile_k=16)
    return mainloop_applies(tile, _NS(vgpr_prefetch=4, split_k=plan.split), M, N, K)


def plan_unroll(plan: G5Plan) -> bool:
    """Whether ``plan`` runs with ``vp_unroll = 1`` (``G5P_VP_UNROLL_MAX_ACC``)."""
    if plan.geo == "skinny" or not G5P_VP_UNROLL_MAX_ACC:
        return False
    tm, tn, _tk, wm, wn, mode, _swr = parse_geo(plan.geo)
    return mode in (3, 4) and tm * tn // (64 * wm * wn) <= G5P_VP_UNROLL_MAX_ACC


def plan_features(
    M: int, N: int, K: int, bf16: bool, plan: G5Plan, menu: Dict[str, Dict[str, float]], cus: int
) -> List[float]:
    """The learned model's feature row of ``plan`` (order: FEATURES)."""
    NA = G5P_NA
    f: Dict[str, float] = {
        "lnM": math.log(M),
        "lnN": math.log(N),
        "lnK": math.log(K),
        "bf16": float(bool(bf16)),
        "lnMN": math.log(M * N),
        "lnMNK": math.log(M * N * K),
        "minMN": math.log(min(M, N)),
        "k_a8": float(K % 8 == 0),
        "k_a64": float(K % 64 == 0),
    }
    if plan.geo == "skinny":
        s = plan.skinny
        f.update(kind=1.0, split=float(getattr(s, "split", 1)))
        if s is not None:
            f.update(
                tm=float(s.tile_m),
                tn=float(s.tile_n),
                tk=float(s.kc),
                waves=float(s.warp_m * s.warp_n * s.warp_k),
                wm=float(s.warp_m),
                wn=float(s.warp_n),
            )
    else:
        tm, tn, tk, wm, wn, mode, swr = parse_geo(plan.geo)
        d = menu.get(plan.geo, {})
        tiles = -(-M // tm) * -(-N // tn)
        ctas = tiles * plan.split
        W = -(-ctas // cus)
        ks = -(-K // (plan.split * tk)) * tk
        vg = float(d.get("vgpr") or 0)
        lds = float(d.get("lds") or 0)
        occ = occupancy(int(vg or 512), int(lds or 65536), wm * wn)
        acc = tm * tn / (64 * wm * wn)
        f.update(
            kind=0.0, tm=tm, tn=tn, tk=tk, waves=wm * wn, wm=wm, wn=wn, mode=mode,
            swr=swr, acc=acc, split=plan.split, serial=plan.serial, vgpr=vg, lds=lds,
            occ=occ, lntiles=math.log(tiles), lnctas=math.log(ctas), W=W,
            fill=ctas / (W * cus), fill_occ=ctas / (cus * occ), ks=ks, kit=ks / tk,
            padm=float(M % tm != 0), padn=float(N % tn != 0), padk=float(K % tk != 0),
            effm=M / (-(-M // tm) * tm), effn=N / (-(-N // tn) * tn),
            ai=tm * tn / (tm + tn), agpr=float(acc > 128),
        )  # fmt: skip
        if "a" in d:
            t = analytic_us(M, N, K, plan.geo, d, plan.split, plan.serial, cus)
            if t:
                f["ln_pred"] = math.log(t)
                f["ln_a"] = math.log(max(0.1, d["a"]))
                f["ln_b"] = math.log(max(1e-6, d["b"]))
    return [float(f.get(k, NA)) for k in FEATURES]


def candidate_plans(M: int, N: int, K: int, arch: str) -> List[Tuple[float, G5Plan]]:
    """Every applicable menu plan with its analytic time, cheapest first."""
    menu = load_menu(arch)
    cus = G5P_CUS.get(arch, 304)
    out: List[Tuple[float, G5Plan]] = []
    if K % 8:
        return out
    for geo, d in menu.items():
        tm, tn, tk = parse_geo(geo)[:3]
        tiles = -(-M // tm) * -(-N // tn)
        for s in G5P_SPLITS:
            if s > 1 and (tiles * s > G5P_SPLIT_MAX_WAVES * cus or K < 2 * s * tk):
                continue
            for ser in G5P_SERIAL_MODES if s > 1 else (0,):
                t = analytic_us(M, N, K, geo, d, s, ser, cus)
                if t is not None:
                    out.append((t, G5Plan(geo, s, ser)))
    out.sort(key=lambda x: (x[0], x[1].geo, x[1].split, x[1].serial))
    return out


def rank_plans(
    M: int,
    N: int,
    K: int,
    dtype: str,
    arch: str,
    extra: Sequence[G5Plan] = (),
    mainloop: bool = True,
) -> List[Tuple[float, G5Plan]]:
    """The preselected plans (+ ``extra``) by predicted time (microseconds),
    cheapest first: the learned model when it is on and shipped, else the
    analytic model (``extra`` plans need the learned model)."""
    pre = candidate_plans(M, N, K, arch)[:G5P_PRESELECT]
    model = load_model(arch) if G5P_MODEL else None
    if model is None:
        return pre
    plans = [p for _t, p in pre] + list(extra)
    if not plans:
        return []
    menu = load_menu(arch)
    cus = G5P_CUS.get(arch, 304)
    bf16 = dtype in ("bf16",)
    xs = [plan_features(M, N, K, bf16, p, menu, cus) for p in plans]
    ys = model.predict(xs)
    out = sorted(
        (
            (
                math.exp(y)
                * (G5P_VP_UNROLL_SCALE if plan_unroll(p) else 1.0)
                * (
                    G5P_MAINLOOP_SCALE
                    if mainloop and plan_mainloop(p, M, N, K)
                    else 1.0
                ),
                p,
            )
            for y, p in zip(ys, plans)
        ),
        key=lambda x: (x[0], x[1].geo, x[1].split, x[1].serial),
    )
    return out


def plan_tile_trait(plan: G5Plan, M: int, N: int, K: int):
    """(TileSpec, TraitSpec) of a menu plan for ``M x N x K`` (pads, K tail,
    split-K slices and reduction, ``vp_unroll``)."""
    from dataclasses import replace

    from ...instances.common.gemm_universal import TileSpec, TraitSpec

    tm, tn, tk, wm, wn, mode, swr = parse_geo(plan.geo)
    tile = TileSpec(
        tile_m=tm, tile_n=tn, tile_k=tk, warp_m=wm, warp_n=wn, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
    )  # fmt: skip
    trait = TraitSpec(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        ab_double_buffer=mode != 4,
        chiplet_swizzle=True,
        chiplet_wgm=8,
        chiplet_chunk_size=64,
        lds_swizzle=True,
        lds_swizzle_r=swr,
        acc_agpr=tm * tn // (64 * wm * wn) > 128,
        vgpr_prefetch=mode,
        pad_m=bool(M % tm),
        pad_n=bool(N % tn),
    )
    if K % tk:
        trait = replace(trait, pad_k=True)
    if plan_unroll(plan):
        trait = replace(trait, vp_unroll=1)
    if plan.split > 1:
        trait = replace(trait, split_k=int(plan.split))
        if K % (plan.split * tk):
            trait = replace(trait, pad_k=True, split_k_aligned=True)
        if plan.serial:
            trait = replace(trait, split_k_serial=int(plan.serial))
    return tile, trait
