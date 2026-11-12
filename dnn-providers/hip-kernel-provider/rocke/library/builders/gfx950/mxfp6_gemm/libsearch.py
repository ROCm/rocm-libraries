#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Library-mode search for the gfx950 MXFP6 tile-config library: one tile x kernel family x format per run.

``python -m builders.gfx950.mxfp6_gemm.libsearch <cmd>``:

* ``space``   -- a config's knob space (tile-scaled slot levels, wave-layout bundles, lever axes) and its legal
  fraction (no GPU);
* ``worker``  -- the GPU side for library problems: strided-batched runtime-shape launches, ``n_src`` distinct data
  sources cycled over the batch entries for verification, ``rotate`` replicated buffer sets for timing;
* ``run``     -- tune.py's supervisor and strategy portfolio (signed lookback, hill climb with lookback, GP + EI,
  SHADE DE, Hyperband) over the config's space, warm-started from ``library.family_spec`` (+ caller warm starts);
* ``confirm`` -- reference (the family default), its twin, the winner(s), the winner's twin and the winner's
  ``mfma_only`` probe on one GPU: every GEMM arm verified with its must-fail mutants, then cold-start (post-idle)
  rounds and warm (steady-state) windows, arms shuffled; paired bootstrap CIs.

Every candidate of a run is verified before it is timed: the integer set bit-exact, the Gaussian and uniform-code
sets within the scaled-MFMA tolerance, entries ``0 .. n_src - 1`` against their source's float64 reference and every
later entry bitwise equal to the entry with the same source (else checked against the reference too), launches
bitwise identical, a second module load bitwise identical; then every must-fail mutant that validates on the
candidate (``gemm.MUTANTS`` + ``gemm.LIB_MUTANTS``) is compiled from the candidate's own spec and must fail.

Timing in the search: candidates, the anchor (family default) and the incumbent + its recompiled copy (A-vs-A) run
in shuffled rounds of back-to-back launches (``--gap-ms 0``: the chip stays in its power-capped steady state);
value = median over rounds of ``t / t_ref``.

Absolute figures are for the operator's internal record only; repository documents carry ratios.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import tune as TU
from .space import Mxfp6Space, kernel_fingerprint, kernel_module, label, legal, overrides_of

MODULE = "builders.gfx950.mxfp6_gemm.libsearch"
WARM_PER_SUMMARY = 5  # warm starts taken from one summary.json (its finalists / top)
LAY = "lay"  # pseudo knob: wave layout bundle "<wm>x<wn><p|j>"
FRACS_SLOT = (1 / 32, 1 / 8, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 7 / 8)
#: direct-compiler levers searched as on / off axes on top of ``library.LEVERS`` (trtag stays on for ``am``)
LEVER_AXES = ("sreassoc", "licm", "bufoffany", "dmafree", "litreuse", "ldsalias", "dmamanualring", "wpecap",
              "vmemfirst", "loadfirst", "nofence", "ldsquads", "ldspairs", "m0cse", "m0share", "ldsgroup", "litshare",
              "soffshare", "dmamanualringw", "dsrebase", "dshoist", "m0fill", "scalesep", "ldsexact",
              # crate-up ports (out-68369242): searched when the compiler lists them
              "accinitkeep", "costagpr", "deadlanes", "dmaexplicit", "dmaoffwide", "fp6sched", "kpreload", "ldswaw",
              "m0csevn", "m0early", "peepshifts", "peepwaits", "pinvmem", "copysink", "m0sharebase", "ldsspillhi", "bankocc", "sremat",
              "m0remat",
              # R13 / branch relaxation (out-68377488)
              "bankpref", "nopfill", "wpebudget")


def say(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def lib_module():
    from kernels.gfx950.mxfp6_gemm import library

    return library


# ===================================================================== problem / base


def parse_tile(s: str) -> Tuple[int, int, int]:
    p = [int(x) for x in s.lower().split("x")]
    return (p[0], p[1], p[2] if len(p) > 2 else 256)


def lib_problem(tile: Tuple[int, int, int], which: str) -> Tuple[int, int, int, int, int]:
    """(batch, m, n, k, rotate) of ``which`` = "S" / "G" / "S-k<K>" (``libtiles``) or "b,m,n,k[,rotate]"."""
    from . import libtiles as T

    if re.fullmatch(r"\d+(,\d+){3,4}", which):
        v = [int(x) for x in which.split(",")]
        rot = v[4] if len(v) > 4 else 1
        return v[0], v[1], v[2], v[3], rot
    t = T.Tile.parse("x".join(map(str, tile)))
    probs = list(T.problems(t)) + list(T.k_sweep(t))
    for p in probs:
        if p.set == which:
            return p.batch, p.m, p.n, p.k, max(1, int(p.rotate))
    raise ValueError(f"no problem {which!r} for tile {tile}")


def wave_layouts(tm: int, tn: int, spill: bool = False) -> List[Tuple[int, int]]:
    """(waves_m, waves_n) with 4 or 8 waves (1- / 2-wave work-groups are never better: notes "Ceilings"), whole
    16-row MFMA tiles, <= 16 MFMA tiles per wave side, <= 256 accumulators per lane (``spill``: the 4-wave layouts
    with 257..384, held in AGPRs + VGPRs by ``acc_vgpr_spill``)."""
    out = []
    for nw in (4, 8):
        for wm in (1, 2, 4, 8):
            if nw % wm:
                continue
            wn = nw // wm
            if tm % (16 * wm) or tn % (16 * wn):
                continue
            fm, fn = tm // (16 * wm), tn // (16 * wn)
            acc = fm * fn * 4
            if fm > 16 or fn > 16:
                continue
            if (not spill and acc > 256) or (spill and not (nw == 4 and 256 < acc <= 384)):
                continue
            out.append((wm, wn))
    return out


def spec_over(over: Mapping[str, object]) -> Dict[str, object]:
    """The spec-field part of a config dict: tune's pseudo knobs and the layout bundle (already applied) dropped."""
    return {k: v for k, v in TU.spec_over(over).items() if k != LAY}


def full_spec_of(mod, sp: "LibSpace", over: Mapping[str, object]) -> dict:
    """Every spec field of a config dict (overrides on the spec defaults + pseudo knobs): the layout bundle applied
    (idempotent on a consistent dict; a hand-made dict such as the no-op prepass's ``{**base, "lay": v}`` takes the
    bundle's waves / fragment mode), the reference K fixed for the problem."""
    spec = replace(mod.Mxfp6GemmSpec(), **spec_over(over))
    if over.get(LAY):
        spec = apply_lay(spec, str(over[LAY]))
    return asdict(sp.fix(spec))


def lay_of(spec) -> str:
    v = "v" if getattr(spec, "acc_vgpr_spill", False) else ""
    return f"{spec.waves_m}x{spec.waves_n}{'j' if spec.frag_mode == 'jit' else 'p'}{v}"


def apply_lay(spec, lay: str):
    """A wave-layout bundle: waves, fragment mode (jit: acquire at the tile top), accumulator file (8 waves = 2 per
    SIMD: VGPR accumulators, as ``library.family_spec``; suffix ``v``: 4 waves with AGPR + VGPR accumulators,
    ``acc_vgpr_spill``)."""
    m = re.fullmatch(r"(\d+)x(\d+)([pj])(v?)", lay)
    if not m:
        raise ValueError(f"bad layout bundle {lay!r}")
    wm, wn, fr, v = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
    out = replace(spec, waves_m=wm, waves_n=wn, frag_mode="jit" if fr == "j" else "prefetch", acc_agpr=wm * wn <= 4)
    if hasattr(out, "acc_vgpr_spill"):
        out = replace(out, acc_vgpr_spill=bool(v))
        if v:
            out = replace(out, acc_pin=False, persist_overlap=False)
    if fr == "j":
        out = replace(out, acq_slot=-1)
    elif out.acq_slot < 0 and spec.frag_mode == "jit":
        out = replace(out, acq_slot=0)
    return out


def lib_legal(mod, spec, problem: Tuple[int, int, int, int]) -> Tuple[bool, str]:
    """``space.legal`` (validate_spec, LDS, register estimate) + 4 / 8 waves + the runtime problem predicates."""
    nw = spec.waves_m * spec.waves_n
    if nw not in (4, 8):
        return False, "library search: 4 or 8 waves"
    ok, why = legal(mod, spec)
    if not ok:
        return ok, why
    if spec.runtime_shape:
        b, m, n, k = problem
        ok, why = mod.runtime_problem_ok(spec, m, n, k, b)
        if not ok:
            return False, f"problem: {why}"
    return True, ""


class LibSpace(Mxfp6Space):
    """``Mxfp6Space`` for one library config: the reference K follows the loop body (``library_k_ref`` in the
    problem's K residue class), the pseudo knob ``lay`` applies a wave-layout bundle, legality adds the problem."""

    def __init__(self, mod, *, problem: Tuple[int, int, int, int], **kw):
        self.problem = tuple(problem)
        self.k_target = problem[3]
        self._mod_early = mod
        super().__init__(mod, **kw)

    def fix(self, spec):
        if getattr(spec, "runtime_shape", False):
            try:
                spec = replace(spec, k=self._mod_early.library_k_ref(spec, k_target=self.k_target))
            except ValueError:
                pass  # legal() refuses it (no reference K in the residue class)
        return spec

    def spec_of(self, genome_or_cfg):
        cfg = self._cfg(genome_or_cfg)
        spec = replace(self.base_spec, **{k: v for k, v in cfg.items() if k not in self.pseudo})
        lay = cfg.get(LAY, self.base_pseudo.get(LAY))
        if lay:
            spec = apply_lay(spec, lay)
        return self.fix(spec)

    def legal(self, spec) -> Tuple[bool, str]:
        return lib_legal(self.mod, self.fix(spec), self.problem)

    def genome_of(self, overrides: Mapping[str, object]):
        """The genome of a config given as overrides on the SPEC DEFAULTS: a knob the dict leaves out takes its spec
        default (not the base's value -- a winner that set a knob back to the spec default keeps it); a pseudo knob
        left out takes the base's level."""
        b = {**asdict(self.base_spec), **self.base_pseudo}
        full = {**self.defaults, **self.base_pseudo, **dict(overrides)}
        cfg = {k: v for k, v in full.items() if k in b and k in self.knobs and v != b[k]}
        return self.space.encode(cfg)

    def _is_valid(self, config: dict) -> bool:
        k = json.dumps(sorted(config.items()), default=str)
        hit = self._legal_cache.get(k)
        if hit is None:
            try:
                spec = self.spec_of(config)
            except (TypeError, ValueError) as e:
                hit = (False, str(e))
            else:
                hit = lib_legal(self.mod, spec, self.problem)
            self._legal_cache[k] = hit
            if not hit[0]:
                reason = hit[1].split(":")[0][:60]
                self.rejects[reason] = self.rejects.get(reason, 0) + 1
        return hit[0]


def base_spec_of(tile: Tuple[int, int, int], family: str, fmt: str, k: int, over: Optional[Mapping] = None,
                 ring: str = "auto"):
    lib = lib_module()
    kw = {"ring": ring} if ring != "auto" else {}
    return lib.family_spec(tile[0], tile[1], tile[2], family, fmt, k_target=k, **kw, **dict(over or {}))


def _slots_one(n: int, fracs=FRACS_SLOT, extra=()) -> Tuple[int, ...]:
    vals = []
    for v in list(extra) + [int(round(f * n)) for f in fracs]:
        v = max(0, min(n - 2, v))
        if v not in vals:
            vals.append(v)
    return tuple(vals)


def lib_table(mod, base, family: str, levers: str, lever_axes: Sequence[str] = LEVER_AXES,
              structure_only: bool = False) -> Tuple[Dict[str, tuple], Dict[str, tuple], List[str]]:
    """``(knobs, pseudo, exclude)`` of one config's space: every ``KNOBS`` entry the library structures can use, with
    slot levels scaled to the MFMAs per 128-K tile of every wave layout (the base level first), the wave-layout
    bundles and the lever on / off axes."""
    lay_n = sorted({mod.mfmas_per_tile(replace(base, waves_m=wm, waves_n=wn))
                    for wm, wn in wave_layouts(base.tile_m, base.tile_n)
                    + wave_layouts(base.tile_m, base.tile_n, spill=True)} | {mod.mfmas_per_tile(base)})
    n = max(lay_n)

    def _slots(nn, fracs=FRACS_SLOT, extra=()):  # the union over the layouts' MFMA counts (nn: unused)
        vals = []
        for ni in lay_n:
            for v in _slots_one(ni, fracs, extra):
                if v not in vals:
                    vals.append(v)
        return tuple(sorted(vals))

    def first(name, levels):
        b = getattr(base, name)
        lv = [b] + [x for x in levels if x != b]
        return tuple(dict.fromkeys(lv))

    knobs: Dict[str, tuple] = {
        "stages": first("stages", (2, 3, 4, 5, 6, 7, 8)),
        "unroll": first("unroll", (1, 2, 3, 4, 8)),
        "occupancy": first("occupancy", (1, 2)),
        "depth_u": first("depth_u", (128, 256, 512)),
        "cta_order": first("cta_order", ("grouped", "linear")),
        "group_m": first("group_m", (1, 2, 4, 8, 16)),
        "acq_slot": first("acq_slot", tuple(x for x in (0, -1, 1, 2, 4, 8) if x < n - 1)),
        "read_every": first("read_every", (1, 2, 3, 4)),
        "read_delay": first("read_delay", (0, 1, 2, 4, 8)),
        "read_lead": first("read_lead", (2, 3, 4, 6, 8, 12, 16)),
        "dtl_delay": first("dtl_delay", (0, 1, 2, 4, 8)),
        "dtl_spread": first("dtl_spread", (0, 1, 2, 3, 4, 5, 6, 8)),
        "scale_slot": first("scale_slot", _slots(n, extra=(2,))),
        "setprio": first("setprio", (0, 1, 2, 3)),
        "tile_lgkm0": first("tile_lgkm0", (True, False)),
        "peel_first": first("peel_first", (False, True)),
        "serpentine": first("serpentine", (True, False)),
        "m_outer": first("m_outer", (True, False)),
        "pin_mfma": first("pin_mfma", (True, False)),
        "read_split": first("read_split", (True, False)),
        "lds_alias_scopes": first("lds_alias_scopes", (True, False)),
        "acc_pin": first("acc_pin", (False, True)),
        "mma_order": first("mma_order", ("rows", "pairs", "blk2")),
        "rsrc_once": first("rsrc_once", (True, False)),
        "dtl_imm": first("dtl_imm", (False, True)),
        "dtl_aux": first("dtl_aux", (0, 1, 2, 3)),
        "b_aux": first("b_aux", (-1, 0, 1, 2, 3)),
        "store_nt": first("store_nt", (False, True)),
        "swap_ab": first("swap_ab", (False, True)),
        "scale_src": first("scale_src", ("lds", "vgpr")),
        "schedule": first("schedule", ("dp", "persistent")),
        "persist_overlap": first("persist_overlap", (False, True)),
        "magic_div": first("magic_div", (False, True)),
        "kernarg_preload": first("kernarg_preload", (False, True)),
        "wtile_period": first("wtile_period", (1, 2, 4)),
        "scale_raw": first("scale_raw", (False, True)),
        "stage_kib": first("stage_kib", (True, False)),
        "scale_dtl": first("scale_dtl", ("dword", "split")),
        "edge_tiles": first("edge_tiles", (False, True)),
        "a_direct": first("a_direct", (False, True)),
        "b_direct": first("b_direct", (False, True)),
        "b_depth": first("b_depth", (1, 2, 3)),
        "epi_lds": first("epi_lds", (False, True)),
        "b_slot": first("b_slot", (0, 8, 16, 32)),
        "b_every": first("b_every", (2, 1, 4)),
    }
    if family == "ak":
        knobs.update({
            "canon_rsrc_once": first("canon_rsrc_once", (True, False)),
            "canon_read": first("canon_read", ("b64x3", "b96x2", "b128b64")),
            "swizzle": first("swizzle", ("rot48", "none")),
            "canon_ring": first("canon_ring", ("stage", "line2", "line2m", "line3")),
            "ring_ops": first("ring_ops", ("A", "all")),
            "line_release": first("line_release", _slots(n, (0.3, 0.4, 0.5, 34 / 64, 0.6, 0.7, 0.8, 0.9))),
            "line_spread": first("line_spread", (0, 1, 2, 3)),
            "line_reads_first": first("line_reads_first", (False, True)),
            "line_early": first("line_early", (False, True)),
            "mini_copy_gap": first("mini_copy_gap", tuple(x for x in (0, 4, 8, 12, 16, 20, 30) if x < n)),
            "mini_copy_at": first("mini_copy_at", (-1,) + _slots(n, (0.1, 0.25, 0.4, 0.5))),
            "ring_kib": first("ring_kib", (False, True)),
            "mini_layout": first("mini_layout", ("v2", "v4")),
            "mini_src": first("mini_src", ("copy", "dtl")),
            "line_dtl_order": first("line_dtl_order", ("strided", "wave")),
            "line_merge": first("line_merge", (False, True)),
        })
    if family == "am":
        knobs["mc_dma"] = first("mc_dma", ("x4", "x3"))
        knobs["mc_copy"] = first("mc_copy", ("auto", "late", "fence"))
    knobs = {k: v for k, v in knobs.items() if hasattr(base, k)}  # fields this tree has
    if structure_only:
        keep = ("stages", "unroll", "occupancy", "depth_u", "cta_order", "group_m", "canon_ring")
        knobs = {k: v for k, v in knobs.items() if k in keep}
    pseudo: Dict[str, tuple] = {"backend": ("rust",)}
    lays = [f"{wm}x{wn}{fr}" for wm, wn in wave_layouts(base.tile_m, base.tile_n) for fr in ("p", "j")]
    if hasattr(base, "acc_vgpr_spill"):  # 4 waves, AGPR + VGPR accumulators (the large tiles; jit: prefetch spills)
        lays += [f"{wm}x{wn}{fr}v" for wm, wn in wave_layouts(base.tile_m, base.tile_n, spill=True) for fr in ("j", "p")]
    bl = lay_of(base)
    pseudo[LAY] = tuple(dict.fromkeys([bl] + lays))
    base_lv = set(x for x in levers.replace("+", ",").split(",") if x)
    if not structure_only:
        for lv in lever_axes:
            if family == "am" and lv == "trtag":
                continue
            pseudo[TU.LEVER_PREFIX + lv] = (True, False) if lv in base_lv else (False, True)
    table = getattr(mod, "KNOBS", {}) or {}
    exclude = [k for k in table if k not in knobs]
    return knobs, pseudo, exclude


def structural_seeds(sp: LibSpace, family: str, max_seeds: int = 24) -> List[Tuple[dict, str]]:
    """Warm starts for the structures one knob cannot reach from the base: every wave-layout bundle, the record's
    line2m ring (``library.LINE2M``) with its slots scaled to the tile's MFMAs per 128-K tile (ak), DepthU 256 and two
    work-groups per CU; each completed with the first legal stages / unroll / slot values. ``(config dict on the
    base, source)``; the climbers then start from whichever wins."""
    lib = lib_module()
    mod = sp.mod
    out: List[Tuple[dict, str]] = []
    seen = set()

    def add(cfg: dict, src: str) -> bool:
        key = json.dumps(sorted(cfg.items()), default=str)
        if key in seen:
            return True
        if sp._is_valid(cfg):
            seen.add(key)
            out.append((cfg, src))
            return True
        return False

    def complete(target: dict, src: str, extra_tries: Sequence[dict] = ({},)) -> bool:
        for ex in extra_tries:
            for st in (None, 3, 4, 2, 5, 6):
                for un in (None, 1, 2, 4, 3, 8):
                    cfg = {**target, **ex}
                    if st is not None:
                        cfg["stages"] = st
                    if un is not None:
                        cfg["unroll"] = un
                    if add(cfg, src):
                        return True
        return False

    base_lay = sp.base_pseudo.get(LAY)
    for lay in sp.pseudo.get(LAY, ()):
        if lay != base_lay:
            complete({LAY: lay}, f"struct:{lay}")
    if family == "ak" and "canon_ring" in sp.knobs:
        l2m = {k: v for k, v in getattr(lib, "LINE2M", {}).items()}
        for lay in sp.pseudo.get(LAY, ()):
            if not lay.endswith("p"):
                continue
            n = mod.mfmas_per_tile(apply_lay(sp.base_spec, lay))
            lr_pref = sorted(set(sp.knobs.get("line_release", (26,))), key=lambda v: abs(v - 34 * n / 64))
            tries = []
            for lr in lr_pref[:4]:
                for dimm in (True, False):
                    tries.append({"line_release": lr, "dtl_imm": dimm,
                                  "scale_slot": min(sp.knobs.get("scale_slot", (40,)), key=lambda v: abs(v - n / 2)),
                                  "mini_copy_gap": min(4, max(0, n - 4))})
            tgt = {**{k: v for k, v in l2m.items() if k in sp.knobs}, LAY: lay}
            tgt.pop("stages", None)
            tgt.pop("unroll", None)
            complete(tgt, f"struct:{lay}/line2m", tries)
    if family == "ak" and getattr(sp.base_spec, "canon_ring", "stage") != "stage":
        # the stage ring (the previous family default, ``family_spec(ring="stage")``) and its layouts
        try:
            st = lib.family_spec(sp.base_spec.tile_n, sp.base_spec.tile_m, getattr(sp, "du", 256), "ak",
                                 sp.base_spec.elem_fmt, k_target=sp.k_target, ring="stage")
            cfg = {k: getattr(st, k) for k in sp.knobs if k in sp.defaults and getattr(st, k) != getattr(sp.base_spec, k)}
            complete(cfg, "struct:stage-ring")
            for lay in sp.pseudo.get(LAY, ()):
                if lay != base_lay:
                    complete({**cfg, LAY: lay}, f"struct:stage-ring/{lay}")
        except (ValueError, TypeError):
            pass
    for du in (256, 512):
        complete({"depth_u": du}, f"struct:du{du}")
    complete({"schedule": "persistent"}, "struct:persistent")
    complete({"schedule": "persistent", "persist_overlap": True}, "struct:persistent+overlap")
    if "b_direct" in sp.knobs and getattr(sp.base_spec, "canon_ring", "stage") == "line2m":
        for bd in (2, 1, 3):  # direct B beside the line2m A ring (kernel agent, 2026-11-12)
            if add({"b_direct": True, "b_depth": bd}, f"struct:line2m+b_direct/d{bd}"):
                break
        for bd in (2, 3):  # persistent + direct B b_depth >= 2 compiles since out-68377488 (branch relaxation)
            if add({"b_direct": True, "b_depth": bd, "schedule": "persistent"}, f"struct:line2m+b_direct/d{bd}+persistent"):
                break
    if "line_merge" in sp.knobs and getattr(sp.base_spec, "canon_ring", "stage") == "line2m":
        complete({"line_merge": True}, "struct:line_merge")
    if "mc_copy" in sp.knobs:
        for mc in ("late", "fence"):
            complete({"mc_copy": mc}, f"struct:mc_copy/{mc}")
    if "epi_lds" in sp.knobs:
        complete({"epi_lds": True}, "struct:epi_lds")
    # direct-to-VGPR operands at deep rings (kernel agent STRUCT: (1,4) + direct B + 4-stage A ring on the large tiles)
    for d in ("b_direct", "a_direct"):
        if d in sp.knobs:
            for lay in [sp.base_pseudo.get(LAY)] + [x for x in sp.pseudo.get(LAY, ()) if x.endswith("j")]:
                done = False
                for ring in ({}, {"canon_ring": "stage"}) if "canon_ring" in sp.knobs else ({},):
                    for st in (4, 6, 8, 5, 7, 3):
                        if add({d: True, LAY: lay, "stages": st, **ring}, f"struct:{d}/{lay}/st{st}"):
                            done = True
                            break
                    if done:
                        break
    lx = TU.LEVER_PREFIX + "ldsexact"
    if lx in sp.pseudo:
        for st in (None, 4, 5, 3):
            cfg = {"stage_kib": False, lx: True}
            if st:
                cfg["stages"] = st
            if add(cfg, f"struct:ldsexact/st{st or 'base'}"):
                break
    complete({"occupancy": 2}, "struct:occ2")
    return out[:max_seeds]


# ===================================================================== GPU worker


class LibWorker(TU.GpuWorker):
    """tune.py's worker for strided-batched runtime-shape library problems."""

    # ------------------------------------------------------------ setup
    def init(self, cache: str, batch: int, m: int, n: int, k: int, seed: int, dists: Sequence[str], n_src: int = 3,
             rotate: int = 1, fmt: str = "e2m3", workers: int = 16, time_dist: str = "gauss") -> dict:
        from . import libbench as LB

        self.batch, self.m, self.n, self.k = int(batch), int(m), int(n), int(k)
        self.flop = 2.0 * self.batch * self.m * self.n * self.k
        self.n_src, self.rotate, self.fmt = max(1, int(n_src)), max(1, int(rotate)), fmt
        self.dists = list(dists)
        self.iters_of: Dict[str, int] = {}
        if self.fake:
            return {"bdf": "fake", "fake": True}
        t0 = time.time()
        self.data = LB.Data(Path(cache), int(seed), int(workers))
        for d in sorted(set(dists) | {time_dist}):
            for s in range(self.n_src if d in dists else 1):
                self.data.operands(d, fmt, self.m, self.n, self.k, s)
                if d in dists:
                    self.data.reference(d, fmt, self.m, self.n, self.k, s)
        from rocke.runtime.hip_module import Runtime, _resolve_hip

        self.rt = Runtime()
        hip = _resolve_hip()
        buf = ctypes.create_string_buffer(64)
        hip.hipDeviceGetPCIBusId(buf, 64, 0)
        self.bdf = buf.value.decode()
        self.clk = TU.Clock(self.bdf)
        self.c_dev = self.rt.alloc(self.batch * self.m * self.n * 2)
        self.c_host = np.empty((self.batch, self.m, self.n), dtype=np.uint16)
        self.code_dev: Dict[tuple, Tuple[int, int, int, int]] = {}
        self.scale_dev: Dict[tuple, Tuple[int, int, int, int]] = {}
        return {"bdf": self.bdf, "sysfs_sclk": self.clk.sclk.exists(), "sysfs_power": bool(self.clk.power),
                "data_s": round(time.time() - t0, 1), "problem": [self.batch, self.m, self.n, self.k],
                "n_src": self.n_src, "rotate": self.rotate}

    def _cat(self, imgs: Sequence[np.ndarray], src_of) -> Tuple[int, int]:
        """Concatenate the per-entry images (entry e = image src_of(e)) into one device buffer; (ptr, stride)."""
        flat = [np.ascontiguousarray(x).view(np.uint8).reshape(-1) for x in imgs]
        cat = np.concatenate([flat[src_of(e)] for e in range(self.batch)])
        return self._put(cat), flat[0].nbytes

    def _inputs_lib(self, spec, d: str, nsrc: int, set_idx: int) -> Tuple[dict, dict]:
        L, g = self.L, self.mod
        la, lb = g.operand_layout_of(spec, "A"), g.operand_layout_of(spec, "B")
        sla, slb = g.scale_layout_of(spec, "A"), g.scale_layout_of(spec, "B")
        kc = (d, nsrc, set_idx, la, lb, spec.tile_m, spec.tile_n)
        ks = (d, nsrc, set_idx, spec.atom, sla, slb)
        ops = [self.data.operands(d, self.fmt, self.m, self.n, self.k, s) for s in range(nsrc)]
        if kc not in self.code_dev:
            pa, sa_ = self._cat([L.pack_operand(a, la, spec.tile_m) for a, _ in ops], lambda e: e % nsrc)
            pb, sb_ = self._cat([L.pack_operand(b, lb, spec.tile_n) for _, b in ops], lambda e: e % nsrc)
            self.code_dev[kc] = (pa, pb, sa_, sb_)
        if ks not in self.scale_dev:
            pa, sa_ = self._cat([L.pack_scales_for(a.scales, spec.atom, sla) for a, _ in ops], lambda e: e % nsrc)
            pb, sb_ = self._cat([L.pack_scales_for(b.scales, spec.atom, slb) for _, b in ops], lambda e: e % nsrc)
            self.scale_dev[ks] = (pa, pb, sa_, sb_)
        a, b, st_a, st_b = self.code_dev[kc]
        sa, sb, st_sa, st_sb = self.scale_dev[ks]
        ptrs = {"a": a, "b": b, "sa": sa, "sb": sb, "c": self.c_dev}
        strides = {"stride_a": st_a, "stride_b": st_b, "stride_sa": st_sa, "stride_sb": st_sb,
                   "stride_c": self.m * self.n * 2}
        return ptrs, strides

    def _ka(self, spec, d: str, nsrc: int, set_idx: int) -> bytes:
        ptrs, strides = self._inputs_lib(spec, d, nsrc, set_idx)
        return self.mod.kernargs(spec, **ptrs, m=self.m, n=self.n, k=self.k, batch=self.batch, **strides)

    def _kargs(self, name: str, d: str) -> bytes:
        """Verification arguments: ``n_src`` distinct sources cycled over the entries."""
        x = self.loaded[name]
        key = ("kv", d)
        if key not in x:
            x[key] = self._ka(x["spec"], d, self.n_src, 0)
        return x[key]

    def _tkargs(self, name: str, d: str) -> List[bytes]:
        """Timing arguments: ``rotate`` replicated buffer sets (one source), cycled launch by launch."""
        x = self.loaded[name]
        key = ("kt", d)
        if key not in x:
            x[key] = [self._ka(x["spec"], d, 1, s) for s in range(self.rotate)]
        return x[key]

    # ------------------------------------------------------------ modules
    def load(self, name: str, hsaco: str, spec: Mapping, kernel: str) -> dict:
        sp = replace(self.mod.Mxfp6GemmSpec(), **dict(spec))
        over = overrides_of(self.mod, sp)
        if self.fake:
            self.loaded[name] = {"spec": sp, "over": over, "hsaco": hsaco, "kernel": kernel}
            return {"ok": True}
        ok, why = self.mod.runtime_problem_ok(sp, self.m, self.n, self.k, self.batch)
        if not ok:
            raise RuntimeError(f"{name}: the build does not serve the problem: {why}")
        m = self.rt.load_module(Path(hsaco).read_bytes())
        grid, block = self.mod.grid_for(sp, self.m, self.n, self.batch)
        self.loaded[name] = {"mod": m, "fn": m.get_function(kernel), "grid": grid, "block": block, "spec": sp,
                             "over": over, "hsaco": hsaco, "kernel": kernel}
        return {"ok": True}

    # ------------------------------------------------------------ verify
    def run_once(self, name: str, d: str) -> np.ndarray:
        x = self.loaded[name]
        kargs = self._kargs(name, d)
        self.rt.memset(self.c_dev, 0xFF, self.c_host.nbytes)  # NaN fill
        self.rt.launch(x["fn"], x["grid"], x["block"], kargs)
        self.rt.wait_stream(0)
        self.rt.memcpy_d2h((ctypes.c_ubyte * int(self.c_host.nbytes)).from_buffer(self.c_host), self.c_dev,
                           self.c_host.nbytes)
        return self.c_host.copy()

    def _check(self, out: np.ndarray, d: str, full: bool = True) -> dict:
        """Entries 0 .. n_src - 1 against their source's reference; every later entry bitwise equal to the entry of
        the same source (else checked against the reference as well)."""
        L = self.L
        bad, n_tot, finite, worst, bad_entries, nonident = 0, 0, True, 0.0, [], 0

        def one(e: int, src: int) -> None:
            nonlocal bad, n_tot, finite, worst
            rc, rs = self.data.reference(d, self.fmt, self.m, self.n, self.k, src)
            r = L.check_against_reference(out[e], rc, rs, exact=(d == "int"))
            bad += r["bad"]
            n_tot += r["n"]
            finite &= r["finite"]
            worst = max(worst, r.get("worst_err_over_tol", 0.0))
            if r["bad"] or not r["finite"]:
                bad_entries.append(e)

        for e in range(min(self.batch, self.n_src)):
            one(e, e)
        if full:
            for e in range(self.n_src, self.batch):
                if not np.array_equal(out[e], out[e % self.n_src]):
                    nonident += 1
                    one(e, e % self.n_src)
                else:
                    n_tot += out[e].size
        ok = bad == 0 and finite
        return {"pass": ok, "bad": bad, "n": n_tot, "finite": finite, "worst_err_over_tol": worst,
                "bad_entries": bad_entries[:16], "nonidentical_entries": nonident, "exact": d == "int"}

    def verify(self, name: str, dists: Sequence[str], launches: int, keep: bool = False, compare: str = "",
               ulp: bool = False, keep_out: bool = False, mutants: Optional[Sequence[Mapping]] = None,
               twin: bool = True) -> dict:
        res = {}
        x = self.loaded[name]
        for d in dists:
            if self.fake:
                self._hb(name, f"verify:{d}")
                bad = 0 if not TU._fake_match("MXFP6_TUNE_FAKE_BAD", x["over"]) else 17
                if x["spec"].mutant != "none":
                    bad = 1000
                res[d] = {"pass": bad == 0, "bad": bad, "deterministic": True, "all_pass": bad == 0}
                continue
            outs, chk = [], []
            for i in range(max(1, launches)):
                self._hb(name, f"verify:{d}:{i}")
                o = self.run_once(name, d)
                outs.append(o)
                chk.append(self._check(o, d, full=(i == 0)))
            same = all(np.array_equal(outs[0], o) for o in outs[1:])
            r = {**chk[0], "all_pass": all(c["pass"] for c in chk), "deterministic": same}
            if twin:  # a second module load of the same code object: bitwise identical
                self._hb(name, f"verify:{d}:twin")
                tw = name + "~tw"
                self.load(tw, x["hsaco"], asdict(x["spec"]), x["kernel"])
                ot = self.run_once(tw, d)
                self.unload([tw])
                r["twin_bitwise"] = bool(np.array_equal(outs[0], ot))
                r["deterministic"] = r["deterministic"] and r["twin_bitwise"]
            if keep:
                self.ref_out[d] = outs[0][: self.n_src].copy()
            if compare and d in self.ref_out:
                r["bitwise_vs_ref"] = bool(np.array_equal(outs[0][: self.n_src], self.ref_out[d]))
            if ulp:
                rc, _ = self.data.reference(d, self.fmt, self.m, self.n, self.k, 0)
                r["ulp"] = TU.ulp_stats(outs[0][0], rc)
            if keep_out:
                self.outs[(name, d)] = outs[0][: self.n_src].copy()
            res[d] = r
        ok = all(r["all_pass"] and r["deterministic"] for r in res.values())
        if mutants is not None:
            mres = {}
            for mu in mutants:
                mres[mu["mu"]] = self._mutant(name, mu, dists)
            caught = all(v == "caught" for v in mres.values())
            res["mutants"] = {"all_pass": caught, "deterministic": True, "results": mres,
                              "bad": sum(1 for v in mres.values() if v != "caught"), "n": len(mres)}
            ok = ok and caught
        return {"ok": ok, "dists": res}

    def _mutant(self, name: str, mu: Mapping, dists: Sequence[str]) -> str:
        """A must-fail mutant of candidate ``name``: 'caught' when some dist fails (heartbeats carry the candidate's
        name, so a hang or fault is blamed on the candidate)."""
        nm = f"{name}~{mu['mu']}"
        self.load(nm, mu["hsaco"], mu["spec"], mu["kernel"])
        try:
            order = [d for d in ("gauss", "uniform", "int") if d in dists] + [d for d in dists if d not in
                                                                             ("gauss", "uniform", "int")]
            for d in order:
                if d == "int" and mu["mu"] == "scale_sel":
                    continue  # all scales 2^0 on the integer set: op_sel cannot show
                self._hb(name, f"mutant:{mu['mu']}:{d}")
                if self.fake:
                    return "caught"
                o = self.run_once(nm, d)
                if not self._check(o, d, full=False)["pass"] or not self._check(o, d, full=True)["pass"]:
                    return "caught"
            return "PASSED"
        finally:
            self.unload([nm])

    # ------------------------------------------------------------ timing
    def estimate(self, name: str, dist: str = "gauss", launches: int = 10) -> dict:
        """ms per launch of ``name`` back to back (after 5 warmup launches)."""
        if self.fake:
            return {"ms": 1.0}
        x = self.loaded[name]
        ka = self._tkargs(name, dist)
        for i in range(5):
            self.rt.launch(x["fn"], x["grid"], x["block"], ka[i % len(ka)])
        e0, e1 = self.rt.event(), self.rt.event()
        e0.record()
        for i in range(launches):
            self.rt.launch(x["fn"], x["grid"], x["block"], ka[i % len(ka)])
        e1.record()
        e1.synchronize()
        ms = e0.elapsed_to(e1) / launches
        e0.destroy()
        e1.destroy()
        return {"ms": ms}

    def _sample(self, name: str, d: str, gap_s: float, warmup: int, iters: int) -> float:
        """One sample: idle gap, warmup launches, ``iters`` timed back to back over the rotated buffer sets."""
        self._last_window = None
        if self.fake:
            time.sleep(min(gap_s, 0.001))
            return TU._fake_time(self.loaded[name]["over"], self.rng)
        x = self.loaded[name]
        ka = self._tkargs(name, d)
        self.rt.wait_stream(0)
        if gap_s > 0:
            time.sleep(gap_s)
        for i in range(warmup):
            self.rt.launch(x["fn"], x["grid"], x["block"], ka[i % len(ka)])
        e0, e1 = self.rt.event(), self.rt.event()
        t0 = time.monotonic()
        e0.record()
        for i in range(iters):
            self.rt.launch(x["fn"], x["grid"], x["block"], ka[(warmup + i) % len(ka)])
        e1.record()
        e1.synchronize()
        self._last_window = (t0, time.monotonic())
        ms = e0.elapsed_to(e1) / iters
        e0.destroy()
        e1.destroy()
        return ms

    def sustained(self, names: Sequence[str], dist: str, seconds: float, rounds: int, seed: int,
                  idle_s: float = 0.0) -> dict:
        """Warm (steady-state) windows: arms shuffled per window, back to back (no idle gap: the chip stays in its
        power-capped state), 20 warmup launches, ``seconds`` of launches over the rotated sets; SCLK / power."""
        rng = random.Random(seed)
        out: Dict[str, list] = {a: [] for a in names}
        for _ in range(rounds):
            order = list(names)
            rng.shuffle(order)
            for a in order:
                self._hb(a, "sustained")
                if self.fake:
                    ms = TU._fake_time(self.loaded[a]["over"], self.rng) * 1.05
                    out[a].append({"ms": ms, "tflops": self.flop / (ms * 1e9)})
                    continue
                est = self._sample(a, dist, 0.0, 3, 5)
                iters = max(50, int(seconds * 1000.0 / max(0.05, est)))
                x = self.loaded[a]
                ka = self._tkargs(a, dist)
                self.rt.wait_stream(0)
                if idle_s > 0:
                    time.sleep(idle_s)
                for i in range(20):
                    self.rt.launch(x["fn"], x["grid"], x["block"], ka[i % len(ka)])
                self.clk.start()
                e0, e1 = self.rt.event(), self.rt.event()
                e0.record()
                for i in range(iters):
                    if i % 200 == 0:
                        self._hb(a, "sustained")
                    self.rt.launch(x["fn"], x["grid"], x["block"], ka[i % len(ka)])
                e1.record()
                e1.synchronize()
                ms = e0.elapsed_to(e1) / iters
                e0.destroy()
                e1.destroy()
                cs = self.clk.stop()
                mj = cs["watts_med"] * ms if cs.get("watts_med") else None
                out[a].append({"ms": ms, "tflops": self.flop / (ms * 1e9), "iters": iters, "mj": mj, **cs})
        return {"sustained": out}


def cmd_worker(args) -> int:
    proto = TU._Proto()
    w = LibWorker(proto)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cid, cmd = req.get("id"), req.get("cmd")
        try:
            if cmd == "quit":
                proto.send(ev="done", id=cid, result={"ok": True})
                break
            if w.dead:
                raise RuntimeError("HipError: hipModuleUnload: hipError(700) an illegal memory access was encountered")
            fn = {"init": w.init, "load": w.load, "unload": w.unload, "compare": w.compare, "verify": w.verify,
                  "time": w.time_rounds, "sustained": w.sustained, "estimate": w.estimate}[cmd]
            res = fn(**req.get("args", {}))
            proto.send(ev="done", id=cid, result=res)
        except Exception as e:  # noqa: BLE001 - reported to the supervisor, which decides
            import traceback

            proto.send(ev="error", id=cid, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-3000:])
    return 0


class LibWorkerClient(TU.WorkerClient):
    """The library worker; ``verify`` of a candidate arm carries that candidate's must-fail mutants."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.mutants: Dict[str, List[dict]] = {}

    def start(self) -> None:
        import queue
        import threading

        kw = {"start_new_session": True} if os.name == "posix" else {}
        self.p = subprocess.Popen([sys.executable, "-m", MODULE, "worker"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=open(self.log, "ab"), text=True, bufsize=1,
                                  env=self.env, **kw)
        self.q = queue.Queue()
        p, q = self.p, self.q

        def rd():
            for line in p.stdout:
                q.put(line)
            q.put(None)

        threading.Thread(target=rd, daemon=True).start()

    def call(self, cmd: str, hang_s: Optional[float] = None, **kw):
        if cmd == "verify" and kw.get("name") in self.mutants and "mutants" not in kw:
            kw["mutants"] = self.mutants[kw["name"]]
        return super().call(cmd, hang_s=hang_s, **kw)


# ===================================================================== compile server


_SERVE = {"rc": None, "tried": False}
TAB, NL = chr(9), chr(10)


def _serve_worker():
    """This process's long-lived ``target_compile --serve`` (compilers >= out-68371726), else None."""
    if not _SERVE["tried"]:
        _SERVE["tried"] = True
        exe = os.environ.get("MXFP6_RUST_COMPILER", "")
        if exe and os.path.exists(exe) and os.environ.get("MXFP6_RC_SERVE", "1") == "1":
            try:
                p = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=30)
                if "--serve" in (p.stdout + p.stderr):
                    _SERVE["rc"] = subprocess.Popen([exe, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                    text=True, bufsize=1)
            except (OSError, subprocess.SubprocessError):
                _SERVE["rc"] = None
    return _SERVE["rc"]


def _build_object_serve(mod, spec, backend: str, out: Path, levers: str = TU.RUST_LEVERS_DEFAULT):
    """tune._build_object through this process's ``target_compile --serve`` (byte-identical to the batch tool for
    the same IR + levers: rustgemm out-68371726 identity check; the run's identity_check re-checks the anchor)."""
    rc = _serve_worker() if backend == "rust" else None
    if rc is None:
        return _ORIG_BUILD(mod, spec, backend, out, levers)
    if hasattr(mod, "check_compiler"):
        mod.check_compiler(spec, backend)
    exe = os.environ.get("MXFP6_RUST_COMPILER", "")
    unknown = TU.unknown_levers(exe, levers)
    if unknown:
        raise RuntimeError(f"levers {unknown} are unknown to {exe}")
    import hashlib

    from rocke.core.ir_serialize import serialize

    with _no_gc():
        text = serialize(mod.build_mxfp6_gemm(spec, arch="gfx950"))
    ir = out.with_name(out.stem + ".ir")
    ir.write_text(text)
    odir = out.with_name(out.stem + ".rust")
    odir.mkdir(parents=True, exist_ok=True)
    rc.stdin.write(TAB.join(["gfx950", str(odir), str(ir), levers or "-", "-"]) + NL)
    rc.stdin.flush()
    line = rc.stdout.readline()
    if not line:
        _SERVE["rc"] = None
        raise RuntimeError("target_compile --serve exited")
    f = line.rstrip(NL).split(TAB)
    if len(f) < 2 or f[1] != "ok":
        raise RuntimeError(f"target_compile: {' '.join(f)[:500]}")
    stats = dict(zip(("instructions", "code_bytes", "vgprs", "sgprs", "lds_bytes"), (int(x) for x in f[2:7])))
    return (odir / f"{out.stem}.hsaco").read_bytes(), {"rust": {"levers": levers, "exe": exe, "serve": True,
                                                                "ir_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                                                **stats}}


_ORIG_BUILD = TU._build_object


def _server_compile(full: Mapping, out: str) -> dict:
    """One build in a long-lived compile-server process (the direct compiler's path: rocKE IR built in Python,
    serialized, compiled by this process's ``target_compile --serve`` -- no LLVM state carried between builds)."""
    TU._build_object = _build_object_serve
    try:
        meta = TU.compile_to(full, Path(out))
    except Exception as e:  # noqa: BLE001 - a build failure is the config's compile error
        meta = {"ok": False, "error": f"{type(e).__name__}: {e}"[-3000:]}
        Path(out + ".json").write_text(json.dumps(meta, default=TU._jd))
    meta["hsaco"] = out
    return meta


class LibCompilePool(TU.CompilePool):
    """tune.CompilePool with long-lived worker processes (``ProcessPoolExecutor``, spawn) instead of one fresh
    Python process per build: the Rust path has no in-process compiler state, so a build is identical either way
    (checked once per run by ``identity_check``); saves the interpreter + import start of every build."""

    def __init__(self, work: Path, jobs: int, timeout_s: float = 300.0, env: Optional[dict] = None):
        super().__init__(work, jobs, timeout_s, env)
        self._ex = None

    def _executor(self):
        if self._ex is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            self._ex = ProcessPoolExecutor(max_workers=self.jobs, mp_context=mp.get_context("spawn"))
        return self._ex

    def _reset(self) -> None:
        if self._ex is not None:
            for p in list(getattr(self._ex, "_processes", {}).values()):
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._ex.shutdown(wait=False, cancel_futures=True)
            self._ex = None

    def compile_many(self, items: Mapping[str, Mapping]) -> Dict[str, dict]:
        from concurrent.futures import TimeoutError as FTimeout
        from concurrent.futures.process import BrokenProcessPool

        res: Dict[str, dict] = {}
        todo = {}
        for tag, full in items.items():
            out = self.path_of(tag)
            cached = Path(str(out) + ".json")
            if cached.exists() and out.exists():
                meta = json.loads(cached.read_text())
                meta["hsaco"] = str(out)
                res[tag] = meta
            else:
                todo[tag] = full
        if not todo:
            return res
        if TU._fake():
            for tag, full in todo.items():
                res[tag] = _server_compile(full, str(self.path_of(tag)))
            return res
        for attempt in range(2):
            try:
                ex = self._executor()
                futs = {tag: ex.submit(_server_compile, dict(full), str(self.path_of(tag))) for tag, full in todo.items()
                        if tag not in res}
                t0 = time.time()
                for tag, f in futs.items():
                    try:
                        res[tag] = f.result(timeout=max(1.0, self.timeout_s - (time.time() - t0)))
                    except FTimeout:
                        res[tag] = {"ok": False, "error": f"compile timeout {self.timeout_s:.0f} s",
                                    "hsaco": str(self.path_of(tag))}
                        self._reset()
                        break
                break
            except BrokenProcessPool as e:
                say(f"[compile] server pool broke ({e}); restarting it")
                self._reset()
        for tag in todo:
            res.setdefault(tag, {"ok": False, "error": "compile server failed", "hsaco": str(self.path_of(tag))})
        return res


# ===================================================================== supervisor


class LibRun(TU.Run):
    def __init__(self, args, sp: LibSpace):
        super().__init__(args, sp)
        b, m, n, k = sp.problem
        self.problem = (b, m, n, k)
        self.flop = 2.0 * b * m * n * k
        self.mut_stats: Dict[str, int] = {"candidates": 0, "mutants_built": 0, "mutant_passed": 0,
                                          "mutant_build_failed": 0}
        if getattr(args, "compile_server", 0):
            self.pool = LibCompilePool(self.out / "work", args.jobs)

    def identity_check(self) -> None:
        """A build from the compile server must equal the fresh-process build byte for byte, else the run falls back
        to fresh processes."""
        if not isinstance(self.pool, LibCompilePool) or TU._fake():
            return
        item = self.compile_item(self.ref_over)
        fresh = TU.CompilePool(self.out / "work_fresh", min(2, self.args.jobs)).compile_many({"idref": item})["idref"]
        srv = self.pool.compile_many({"idsrv": item})["idsrv"]
        same = (fresh.get("ok") and srv.get("ok")
                and Path(fresh["hsaco"]).read_bytes() == Path(srv["hsaco"]).read_bytes())
        self.rec(type="compile_server_identity", identical=bool(same))
        say(f"[compile] server vs fresh-process build of the anchor: {'IDENTICAL' if same else 'DIFFERENT -> fresh processes'}")
        if not same:
            self.pool = TU.CompilePool(self.out / "work", self.args.jobs)

    def full_spec(self, key_or_over) -> dict:
        over = self.E[key_or_over]["over"] if isinstance(key_or_over, str) else key_or_over
        return full_spec_of(self.mod, self.sp, over)

    def worker_start(self) -> None:
        a = self.args
        b, m, n, k = self.sp.problem
        for attempt in range(3):
            self.worker = LibWorkerClient(self.out / "worker.log", hang_s=self.args.hang_s)
            self.worker.start()
            try:
                info = self.worker.call("init", hang_s=1800, cache=str(a.cache), batch=b, m=m, n=n, k=k,
                                        seed=a.seed_data, dists=sorted(set(self.dists()) | {a.time_dist}),
                                        n_src=a.n_src, rotate=a.rotate, fmt=a.fmt, workers=a.gen_workers,
                                        time_dist=a.time_dist)
                TU.gpu_guard(info)
                say(f"[worker] up: {info}")
                self.gpu_info = info
                self.loaded_ctl = {}
                return
            except (TU.WorkerFault, RuntimeError) as e:
                say(f"[worker] init failed ({e}); retry {attempt + 1}")
                self.worker.kill()
        raise SystemExit("worker does not start: giving up")

    def worker_restart(self, why: str) -> None:
        mut = dict(self.worker.mutants) if self.worker else {}
        super().worker_restart(why)
        self.worker.mutants = mut

    def mutants_of(self, over: Mapping) -> List[str]:
        """Every must-fail mutant that validates on this config."""
        out = []
        for mu in [m for m in getattr(self.mod, "MUTANTS", ()) if m != "none"] + list(getattr(self.mod, "LIB_MUTANTS", ())):
            spec = replace(self.mod.Mxfp6GemmSpec(), **{**self.full_spec(over), "mutant": mu})
            if self.mod.validate_spec(spec)[0]:
                out.append(mu)
        return out

    def _compile_mutants(self, keys: Sequence[str]) -> Dict[str, object]:
        """``{key: [mutant dicts] | error string}``."""
        items, mus = {}, {}
        for k in keys:
            over = self.E[k]["over"]
            mus[k] = self.mutants_of(over)
            for mu in mus[k]:
                items[self.tag_of(k, "#mut" + mu)] = self.compile_item({**over, "mutant": mu})
        metas = self.pool.compile_many(items) if items else {}
        out: Dict[str, object] = {}
        for k in keys:
            bad = [mu for mu in mus[k] if not metas[self.tag_of(k, "#mut" + mu)].get("ok")]
            self.mut_stats["candidates"] += 1
            if bad:
                self.mut_stats["mutant_build_failed"] += 1
                out[k] = f"mutant build failed: {bad}: " + str(metas[self.tag_of(k, '#mut' + bad[0])].get("error"))[:200]
                continue
            if len(mus[k]) < 4:
                out[k] = f"only {len(mus[k])} mutants validate ({mus[k]})"
                continue
            self.mut_stats["mutants_built"] += len(mus[k])
            out[k] = [{"mu": mu, "hsaco": metas[self.tag_of(k, "#mut" + mu)]["hsaco"],
                       "spec": self.full_spec({**self.E[k]["over"], "mutant": mu}),
                       "kernel": metas[self.tag_of(k, "#mut" + mu)]["name"]} for mu in mus[k]]
        return out

    def session_gate(self) -> None:
        """Compile + verify the warm start with its must-fail mutants (all must fail), pick the gate mode."""
        base_over = self.ref_over
        items = {self.tag_of(self.ref_key): self.compile_item(base_over)}
        res = self.pool.compile_many(items)
        meta = res[self.tag_of(self.ref_key)]
        spec = self.sp.spec_of(self.sp.base_genome)
        st, why = TU.compile_gate(meta, spec, self.mod, self.gate_mode)
        if st != TU.STATUS_OK and self.gate_mode == "strict" and st == TU.STATUS_SPILL:
            say(f"[gate] the warm start fails the strict gate ({why}): gate mode -> 'loop' for this run")
            self.gate_mode = "loop"
            st, why = TU.compile_gate(meta, spec, self.mod, self.gate_mode)
        if st == TU.STATUS_SPILL and self.gate_mode == "loop":
            # the shipped kernel itself has extra vmcnt(0) waits (or loop scratch): the search must still be able
            # to improve on it -- compile success only (candidates are judged by verification and time)
            say(f"[gate] the warm start fails the loop gate ({why}): gate mode -> 'off' for this run")
            self.gate_mode = "off"
            st, why = TU.compile_gate(meta, spec, self.mod, self.gate_mode)
        if st != TU.STATUS_OK:
            raise SystemExit(f"warm start fails the compile gate: {st} {why}")
        self.E[self.ref_key] = {"over": base_over, "status": TU.STATUS_OK, "meta": meta, "ratios": [], "tf": [],
                                "batches": [], "first_batch": -1, "value": 1.0, "isa": meta.get("isa"),
                                "full": True, "value_full": True}
        self.isa_index[meta.get("isa")] = self.ref_key
        muts = self._compile_mutants([self.ref_key])[self.ref_key]
        if isinstance(muts, str):
            raise SystemExit(f"warm start: {muts}")
        self.worker.call("load", name="ref", hsaco=meta["hsaco"], spec=self.full_spec(base_over), kernel=meta["name"])
        self.loaded_ctl["ref"] = self.ref_key
        v = self.worker.call("verify", name="ref", dists=self.dists(), launches=self.args.verify_launches, keep=True,
                             mutants=muts, hang_s=max(self.args.hang_s, 300))
        say(f"[gate] warm start verify + mutants: {'PASS' if v['ok'] else 'FAIL'} "
            f"{json.dumps(v['dists'], default=TU._jd)[:800]}")
        self.rec(type="session", gate_mode=self.gate_mode, ref_over=base_over, ref_meta=TU._meta_summary(meta),
                 verify=v, mutants=(v["dists"].get("mutants") or {}).get("results"), gpu=getattr(self, "gpu_info", None),
                 space=self.sp.describe()["knobs"], problem=list(self.sp.problem))
        if not v["ok"]:
            raise SystemExit("the warm start fails verification (or a must-fail mutant passed): stop")
        self.E[self.ref_key]["verify"] = v
        est = self.worker.call("estimate", name="ref", dist=self.args.time_dist)["ms"]
        if self.args.search_ms > 0:
            self.args.search_iters = max(4, int(round(self.args.search_ms / max(est, 1e-3))))
        say(f"[gate] ref {est:.3f} ms per launch ({self.flop / (est * 1e9):.0f} TF/s back to back): "
            f"{self.args.search_iters} launches per search sample")

    def _gpu_batch(self, measure: List[str], pending_keys, todo, obs, bno: int, compile_s: float) -> None:
        if getattr(self.args, "mutants", "all") == "final":  # mutants at the finalists (and every confirmation)
            self.worker.mutants = {}
            return super()._gpu_batch(measure, pending_keys, todo, obs, bno, compile_s)
        muts = self._compile_mutants(measure)
        keep = []
        for k in measure:
            if isinstance(muts[k], str):
                e = self.E[k]
                e.update(status=TU.STATUS_COMPILE, why=muts[k])
                self._finish_key(k, pending_keys[k], todo, obs, TU.STATUS_COMPILE, muts[k], bno)
                continue
            keep.append(k)
        names = {k: f"c{i}" for i, k in enumerate(keep)}  # tune.Run._gpu_batch names arms the same way
        self.worker.mutants = {names[k]: muts[k] for k in keep}
        if keep:
            super()._gpu_batch(keep, pending_keys, todo, obs, bno, compile_s)
        for k in keep:
            v = self.E[k].get("verify") or {}
            mr = ((v.get("dists") or {}).get("mutants") or {}).get("results") or {}
            if any(r != "caught" for r in mr.values()):
                self.mut_stats["mutant_passed"] += 1
                say(f"[mutant] a must-fail mutant PASSED on {label(self.E[k]['over'])[:200]}: {mr}")
        self.worker.mutants = {}


    def finalists(self, k_top: int) -> dict:
        """--mutants final: every finalist (the top configs by pooled ratio) must fail all its must-fail mutants
        before it is timed as a finalist; a config whose mutant passes (or does not build) leaves the ranking."""
        if getattr(self.args, "mutants", "all") == "final":
            checked = set()
            for _ in range(4 * k_top):
                ranked = [k for v, k in sorted((self.pooled(k), k) for k, e in self.E.items()
                                               if e.get("status") == TU.STATUS_OK and not e.get("isa_dup"))
                          if math.isfinite(v)][:k_top]
                todo = [k for k in ranked if k not in checked and k != self.ref_key]
                if not todo:
                    break
                muts = self._compile_mutants(todo)
                for k in todo:
                    checked.add(k)
                    e = self.E[k]
                    if isinstance(muts[k], str):
                        e.update(status=TU.STATUS_COMPILE, why=muts[k])
                        continue
                    self.worker.call("load", name="fin_mut", hsaco=e["meta"]["hsaco"], spec=self.full_spec(k),
                                     kernel=e["meta"]["name"])
                    v = self.worker.call("verify", name="fin_mut", dists=self.dists(), launches=1, mutants=muts[k],
                                         twin=False, hang_s=max(self.args.hang_s, 300))
                    self.worker.call("unload", names=["fin_mut"])
                    mr = (v["dists"].get("mutants") or {}).get("results")
                    self.rec(type="finalist_mutants", key=k, ok=v["ok"], mutants=mr)
                    if not v["ok"]:
                        self.mut_stats["mutant_passed"] += 1
                        e.update(status=TU.STATUS_NUMERIC, why=f"finalist mutants / verify: {mr}")
                        say(f"[final] dropped (mutant / verify): {label(e['over'])[:200]} {mr}")
        return super().finalists(k_top)


# ===================================================================== run


def warm_list(path_or_json: str) -> List[Tuple[dict, str]]:
    """Warm starts: a JSON list of override dicts (or {"label": over} / a summary.json's top / finalists)."""
    if not path_or_json:
        return []
    out = []
    for part in [x for x in path_or_json.split(";") if x] if not path_or_json.lstrip().startswith(("[", "{")) else []:
        if part.startswith("/") or ":" in part[:3] or part.endswith(".json"):
            if not os.path.exists(part):
                say(f"[warm] {part}: missing (skipped)")
                continue
            out += warm_list_one(part)
    if out or not path_or_json.lstrip().startswith(("[", "{")):
        return out
    return warm_list_one(path_or_json)


def warm_list_one(path_or_json: str) -> List[Tuple[dict, str]]:
    d = json.loads(Path(path_or_json).read_text()) if os.path.exists(path_or_json) else json.loads(path_or_json)
    out = []
    if isinstance(d, dict) and ("top" in d or "finalists" in d):
        rows = ((d.get("finalists") or {}).get("rows") or []) + (d.get("top") or [])
        seen = set()
        for r in rows:
            s = json.dumps(r.get("over"), sort_keys=True, default=str)
            if r.get("over") is not None and s not in seen:
                seen.add(s)
                out.append((dict(r["over"]), f"prev{len(out)}:{Path(path_or_json).parent.name}"))
        return out[:WARM_PER_SUMMARY]
    if isinstance(d, dict):
        return [(dict(v), f"warm:{k}") for k, v in d.items()]
    return [(dict(v), f"warm:{i}") for i, v in enumerate(d)]


def make_space(args, structure_only: bool = False) -> Tuple[LibSpace, dict]:
    mod = kernel_module()
    lib = lib_module()
    tile = parse_tile(args.tile)
    b, m, n, k, rot = lib_problem(tile, args.problem)
    if args.rotate <= 0:
        args.rotate = rot
    base_over_in = json.loads(Path(args.base).read_text()) if args.base and os.path.exists(args.base) else (
        json.loads(args.base) if args.base else {})
    base_lay = base_over_in.pop(LAY, None)
    bover = {kk: v for kk, v in base_over_in.items() if not TU.is_pseudo(kk)}
    if args.anchor == "dispatch" and hasattr(lib, "dispatch"):
        # the shipped kernel for this problem: config_spec (WG_MULT work-group tile + edge_tiles) + the schedule
        kw = {"ring": args.ring} if args.ring != "auto" else {}
        base = lib.dispatch(tile[0], tile[1], tile[2], args.family, args.fmt, m, n, b, k_target=k, **kw, **bover)
    else:
        base = base_spec_of(tile, args.family, args.fmt, k, bover, ring=args.ring)
    if base_lay:
        base = apply_lay(base, base_lay)
    levers = args.rust_levers or getattr(lib, "LEVERS", TU.RUST_LEVERS_DEFAULT)
    args.rust_levers = levers.replace(",", "+")
    axes = [x for x in re.split(r"[,:+]", args.lever_axes) if x] if args.lever_axes else list(LEVER_AXES)
    exe = os.environ.get("MXFP6_RUST_COMPILER", "")
    known = TU.compiler_levers(exe) if exe and os.path.exists(exe) else None
    if known is not None:
        axes = [x for x in axes if x in known]
    knobs, pseudo, exclude = lib_table(mod, base, args.family, levers, axes, structure_only=structure_only)
    extra = dict(knobs)
    if args.extra_knobs:
        ek = json.loads(Path(args.extra_knobs).read_text()) if os.path.exists(args.extra_knobs) else json.loads(
            args.extra_knobs)
        extra.update(ek)
    excl = exclude + [x for x in re.split(r"[,:+]", args.exclude) if x]
    base_over = overrides_of(mod, base)
    base_over.update({kk: v for kk, v in base_over_in.items() if TU.is_pseudo(kk)})
    base_over.setdefault(LAY, lay_of(base))
    sp = LibSpace(mod, problem=(b, m, n, k), base=base_over, exclude=excl, extra=extra, pseudo=pseudo)
    sp.du = tile[2]
    return sp, {"tile": tile, "problem": (b, m, n, k, args.rotate), "levers": levers}


def cmd_space(args) -> int:
    sp, info = make_space(args)
    d = sp.describe()
    print(json.dumps({"problem": info["problem"], "base": d["base"], "pseudo": d["pseudo"], "knobs": d["knobs"],
                      "size_bound": d["size_bound"]}, indent=1, default=TU._jd))
    rng = random.Random(0)
    ok = 0
    for _ in range(args.samples):
        g = sp.space.random_valid(rng, tries=8)
        ok += g is not None
    print(f"structural sampler legal draws: {ok}/{args.samples}; rejects {json.dumps(sp.rejects, indent=1)}")
    # single-knob legality around the base
    single = {}
    for name, levels in sp.knobs.items():
        single[name] = [str(v) for v in levels if sp._is_valid({name: v}) or v == sp.base_value(name)]
    print("legal single-knob levels: " + json.dumps(single, default=str))
    seeds = structural_seeds(sp, args.family)
    print(f"structural seeds ({len(seeds)}):")
    for cfg, src in seeds:
        print(f"  {src}: {label(cfg)}")
    return 0


def seed_gated(run: "LibRun", noop: Sequence[str], seeds: Sequence[Tuple[dict, str]]) -> List[str]:
    """No-op candidates (every level of the knob compiles to the base ISA on the base and under every single change)
    that change the ISA on a structural seed (e.g. the line knobs under the line2m seed, which no single legal move
    reaches): those stay searchable. Fresh compile processes, parallel."""
    if not noop or not seeds:
        return []
    sp = run.sp
    items, plan = {}, []
    for si, (over, _src) in enumerate(seeds):
        st = TU._sha(json.dumps(over, sort_keys=True, default=TU._jd) + "#seed")
        items[st] = run.compile_item(over)
        for knob in noop:
            for v in sp.knobs.get(knob, ()):
                if over.get(knob, sp.base_value(knob)) == v:
                    continue
                o = {**over, knob: v}
                try:
                    if not sp.legal(replace(sp.base_spec, **{kk: vv for kk, vv in full_spec_of(run.mod, sp, o).items()
                                                             if kk in sp.defaults}))[0]:
                        continue
                except (TypeError, ValueError):
                    continue
                t = TU._sha(json.dumps(o, sort_keys=True, default=TU._jd) + "#seedk")
                items[t] = run.compile_item(o)
                plan.append((st, t, knob))
    res = run.pool.compile_many(items) if items else {}
    live = set()
    for st, t, knob in plan:
        a, b = res.get(st) or {}, res.get(t) or {}
        if a.get("ok") and b.get("ok") and a.get("isa") != b.get("isa"):
            live.add(knob)
    run.rec(type="seed_gated", live=sorted(live), n=len(items))
    return sorted(live)


def cmd_run(args) -> int:
    t_start = time.time()
    if args.rust_compiler:
        os.environ["MXFP6_RUST_COMPILER"] = args.rust_compiler
    if not (os.environ.get("MXFP6_RUST_COMPILER") and os.path.exists(os.environ["MXFP6_RUST_COMPILER"])) \
            and not TU._fake():
        raise SystemExit("--rust-compiler / MXFP6_RUST_COMPILER is not an executable")
    sp, info = make_space(args, structure_only=args.structure_only)
    os.environ["MXFP6_RUST_LEVERS"] = args.rust_levers.replace("+", ",")
    args.m, args.n, args.k = info["problem"][1], info["problem"][2], info["problem"][3]
    run = LibRun(args, sp)
    desc = sp.describe()
    (run.out / "space.json").write_text(json.dumps({**desc, "problem": info["problem"], "tile": args.tile,
                                                    "family": args.family, "fmt": args.fmt}, indent=1,
                                                   default=TU._jd))
    say(f"[space] {args.tile} {args.family} {args.fmt} problem {info['problem']}: {len(sp.knobs)} knobs, size bound "
        f"{desc['size_bound']:.3g}, kernel {desc['kernel_fingerprint']}, base {label(desc['base'])}")
    run.identity_check()
    run.worker_start()
    run.session_gate()
    seeds = structural_seeds(sp, args.family, args.struct_seeds) if args.struct_seeds else []
    seed_over = [(sp.overrides(cfg), src) for cfg, src in seeds]
    if args.noop_prepass:
        noop = TU.noop_prepass(run)
        live = seed_gated(run, noop, seed_over)
        if live:
            say(f"[noop] live under a structural seed (kept): {sorted(live)}")
        noop = [k for k in noop if k not in live]
        if noop:
            say(f"[noop] held at the base (every level compiles to the base ISA): {sorted(noop)}")
            args.exclude = ",".join([x for x in re.split(r"[,:+]", args.exclude) if x] + sorted(noop))
            sp2, _ = make_space(args, structure_only=args.structure_only)
            run.rebind_space(sp2)
            sp = sp2
            (run.out / "space.json").write_text(json.dumps({**sp.describe(), "problem": info["problem"]}, indent=1,
                                                           default=TU._jd))
    warm = [(run.ref_over, "family-default")]
    warm += warm_list(args.warm_from)
    say(f"[warm] {len(seed_over)} structural seeds: {[s for _, s in seed_over]}")
    warm += seed_over
    good = []
    for over, src in warm:
        try:
            g = sp.genome_of({kk: v for kk, v in over.items() if kk in sp.defaults or kk in sp.pseudo
                                   or TU.is_pseudo(kk)})
        except Exception as e:  # noqa: BLE001
            say(f"[warm] {src}: not encodable ({e})")
            continue
        if sp.space.valid(g):
            good.append((over, src))
        else:
            say(f"[warm] {src}: not legal in this space: dropped")
    strat = TU.build_strategy(sp, args, good)
    if args.resume and os.path.exists(args.resume):
        if not run.restore(Path(args.resume), strat):
            strat = TU.build_strategy(sp, args, good + warm_list(str(Path(args.resume).with_name("summary.json"))))
    deadline = t_start + args.minutes * 60.0 - args.reserve_s
    asks_cap = args.budget * 6
    stop = "?"
    while True:
        if time.time() > deadline:
            stop = "clock"
            break
        if run.n_gpu >= args.budget:
            stop = "budget"
            break
        if run.stall >= args.stall_batches and run.n_gpu >= args.min_evals:
            stop = f"stall ({run.stall} batches without a gain beyond {run.min_gain():.4f})"
            break
        if run.n_told >= asks_cap:
            stop = "ask cap"
            break
        if run.idle_batches >= args.idle_batches:
            stop = f"converged ({run.idle_batches} batches proposed only measured or ISA-identical configs)"
            break
        try:
            more = run.run_batch(strat)
        except TU.GiveUp as e:
            stop = f"gave up: {e}"
            break
        if not more:
            stop = "strategy exhausted" if strat.done else "no proposals"
            break
    say(f"[stop] {stop}: {run.n_gpu} GPU evaluations, {run.batch_no} batches, {time.time() - t_start:.0f} s; "
        f"mutants {run.mut_stats}")
    fin = None
    try:
        if stop.startswith("gave up"):
            raise RuntimeError("no finalists after giving up")
        fin = run.finalists(args.top)
        for r in fin["rows"]:
            say(f"[final] warm {r['sustained_tf'] or float('nan'):7.1f} TF/s @ {r['sustained_sclk']} MHz "
                f"{r['sustained_w']} W {r.get('sustained_mj')} mJ  back-to-back {r['peak_tf']:7.1f} "
                f"({r['ratio_ref']:.4f} of ref)  {r['label'][:300]}")
    except (TU.WorkerFault, RuntimeError) as e:
        say(f"[final] finalists failed: {e}")
    summ = TU.summarize(run, strat, stop, fin)
    summ.update({"tile": args.tile, "family": args.family, "fmt": args.fmt, "problem": info["problem"],
                 "mutant_stats": run.mut_stats})
    (run.out / "summary.json").write_text(json.dumps(summ, indent=1, default=TU._jd))
    run.worker.close()
    run.sync()
    return 0


# ===================================================================== confirm


def cmd_confirm(args) -> int:
    """Family default (ref) vs the winner(s) on one GPU: verified (+ mutants), cold-start rounds and warm windows,
    twins of ref and best, the best's mfma_only probe; paired bootstrap CIs."""
    if args.rust_compiler:
        os.environ["MXFP6_RUST_COMPILER"] = args.rust_compiler
    args.warm_from = ""
    args.lever_axes = args.lever_axes or ""
    sp, info = make_space(args)
    os.environ["MXFP6_RUST_LEVERS"] = args.rust_levers.replace("+", ",")
    mod = sp.mod
    b, m, n, k, rot = info["problem"]
    arms_in = json.loads(Path(args.arms).read_text()) if os.path.exists(args.arms) else json.loads(args.arms)
    cfgs: Dict[str, dict] = {"ref": dict(sp.overrides(sp.base_genome))}
    for nm, over in arms_in.items():
        cfgs[nm] = dict(over)
    if "best" not in cfgs:
        raise SystemExit("--arms needs a 'best' arm")
    lev = args.rust_levers

    def full(over):
        return full_spec_of(mod, sp, over)

    def item(over):
        return {**full(over), "__backend__": "rust", "__levers__": TU.levers_of(over, lev)}

    probe = {**cfgs["best"], "probe": "mfma_only", "scale_src": "vgpr"}
    probe_ok, why = mod.validate_spec(replace(mod.Mxfp6GemmSpec(), **full(probe)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = TU.CompilePool(out / "work", args.jobs)
    items = {nm: item(ov) for nm, ov in cfgs.items()}
    mus: Dict[str, List[str]] = {}
    for nm, ov in cfgs.items():
        mus[nm] = []
        for mu in [x for x in getattr(mod, "MUTANTS", ()) if x != "none"] + list(getattr(mod, "LIB_MUTANTS", ())):
            if mod.validate_spec(replace(mod.Mxfp6GemmSpec(), **{**full(ov), "mutant": mu}))[0]:
                mus[nm].append(mu)
                items[f"{nm}#mut{mu}"] = item({**ov, "mutant": mu})
    if probe_ok:
        items["probe_mfma_only"] = item(probe)
    else:
        say(f"[confirm] mfma_only probe not legal on best: {why}")
    metas = pool.compile_many(items)
    w = LibWorkerClient(out / "worker.log", hang_s=args.hang_s)
    w.start()
    info_g = w.call("init", hang_s=1800, cache=args.cache, batch=b, m=m, n=n, k=k, seed=args.seed_data,
                    dists=[d for d in args.verify_dists.split(",") if d], n_src=args.n_src, rotate=rot, fmt=args.fmt,
                    workers=args.gen_workers, time_dist=args.time_dist)
    TU.gpu_guard(info_g)
    say(f"[confirm] gpu {info_g}")
    report = {"gpu": info_g, "host": os.environ.get("SLURMD_NODENAME", ""), "tile": args.tile, "family": args.family,
              "fmt": args.fmt, "problem": info["problem"], "arms": {}, "kernel_fingerprint": kernel_fingerprint(mod),
              "protocol": {"cold_rounds": args.cold_rounds, "gap_ms": 25.0, "warm_windows": args.warm_windows,
                           "warm_s": args.warm_s, "time_dist": args.time_dist, "rotate": rot},
              "env": {kk: os.environ.get(kk) for kk in ("SLURM_JOB_ID", "SLURMD_NODENAME", "ROCR_VISIBLE_DEVICES",
                                                         "MXFP6_RUST_COMPILER")}}
    timed: List[str] = []
    for nm, ov in cfgs.items():
        meta = metas[nm]
        report["arms"][nm] = {"over": ov, "label": label(ov), "compile": TU._meta_summary(meta)}
        if not meta.get("ok"):
            say(f"[confirm] {nm}: compile failed: {str(meta.get('error'))[:300]}")
            continue
        mlist = []
        bad = False
        for mu in mus[nm]:
            mm = metas[f"{nm}#mut{mu}"]
            if not mm.get("ok"):
                bad = True
                say(f"[confirm] {nm}: mutant {mu} does not build")
                continue
            mlist.append({"mu": mu, "hsaco": mm["hsaco"], "spec": full({**ov, "mutant": mu}), "kernel": mm["name"]})
        w.call("load", name=nm, hsaco=meta["hsaco"], spec=full(ov), kernel=meta["name"])
        v = w.call("verify", name=nm, dists=[d for d in args.verify_dists.split(",") if d], launches=2,
                   keep=(nm == "ref"), compare="" if nm == "ref" else "ref", ulp=True, mutants=mlist,
                   hang_s=max(args.hang_s, 300))
        report["arms"][nm]["verify"] = v
        say(f"[confirm] verify {nm}: {'PASS' if v['ok'] and not bad else 'FAIL'} "
            f"mutants {(v['dists'].get('mutants') or {}).get('results')}")
        if v["ok"] and not bad:
            timed.append(nm)
            if nm in ("ref", "best"):
                w.call("load", name=nm + "_tw", hsaco=meta["hsaco"], spec=full(ov), kernel=meta["name"])
                report["arms"][nm + "_tw"] = {"over": ov, "label": f"twin of {nm} (second module load)"}
                timed.append(nm + "_tw")
    if "probe_mfma_only" in metas and metas["probe_mfma_only"].get("ok") and "best" in timed:
        w.call("load", name="probe_mfma_only", hsaco=metas["probe_mfma_only"]["hsaco"], spec=full(probe),
               kernel=metas["probe_mfma_only"]["name"])
        report["arms"]["probe_mfma_only"] = {"over": probe, "label": "best's mfma_only probe (scale_src vgpr)"}
        timed.append("probe_mfma_only")
    flop = 2.0 * b * m * n * k
    cold = w.call("time", arms={nm: args.cold_rounds for nm in timed}, dist=args.time_dist, gap_ms=25.0, warmup=1,
                  iters=3, seed=1000 * (args.seed + 1), hang_s=max(args.hang_s, 300), discard=1)
    warm = w.call("sustained", names=timed, dist=args.time_dist, seconds=args.warm_s, rounds=args.warm_windows,
                  seed=7 + args.seed, hang_s=max(args.hang_s, 300))["sustained"]
    w.close()
    cold_ms = cold["ms"]
    warm_ms = {nm: [s["ms"] for s in warm.get(nm, [])] for nm in timed}
    pairs = {"best_vs_ref": ("ref", "best"), "noise_ref_twin": ("ref_tw", "ref"), "noise_best_twin": ("best_tw", "best"),
             "best_of_mfma_probe": ("probe_mfma_only", "best")}
    for nm in timed:
        if nm not in ("ref", "best", "ref_tw", "best_tw", "probe_mfma_only"):
            pairs[f"{nm}_vs_ref"] = ("ref", nm)
            pairs[f"{nm}_vs_best"] = ("best", nm)
    report["cold_pairs"] = {p: TU.bootstrap_ci(cold_ms[a], cold_ms[c], seed=args.seed) for p, (a, c) in pairs.items()
                            if a in cold_ms and c in cold_ms}
    report["warm_pairs"] = {p: TU.bootstrap_ci(warm_ms[a], warm_ms[c], seed=args.seed + 1)
                            for p, (a, c) in pairs.items() if a in warm_ms and c in warm_ms}
    rows = []
    for nm in timed:
        sv = warm.get(nm, [])
        rows.append({"arm": nm, "label": report["arms"][nm]["label"],
                     "cold_tf": TU.median([flop / (x * 1e9) for x in cold_ms[nm]]),
                     "warm_tf": TU.median([s["tflops"] for s in sv]) if sv else None,
                     "warm_sclk": TU.median([s["sclk_med"] for s in sv if s.get("sclk_med")]) if any(
                         s.get("sclk_med") for s in sv) else None,
                     "warm_w": TU.median([s["watts_med"] for s in sv if s.get("watts_med")]) if any(
                         s.get("watts_med") for s in sv) else None,
                     "warm_mj": TU.median([s["mj"] for s in sv if s.get("mj")]) if any(s.get("mj") for s in sv) else None})
    report["rows"] = rows
    report["cold"], report["warm"] = cold, warm
    (out / "confirm.json").write_text(json.dumps(report, indent=1, default=TU._jd))
    print(confirm_text([report]))
    return 0


def confirm_text(reports: Sequence[Mapping]) -> str:
    out = []
    for rep in reports:
        g = rep.get("gpu") or {}
        out.append(f"== {rep.get('tile')} {rep.get('family')} {rep.get('fmt')} {rep.get('problem')} GPU {g.get('bdf')} "
                   f"on {rep.get('host')}")
        for r in sorted(rep["rows"], key=lambda r: -(r["warm_tf"] or 0)):
            out.append(f"  {r['arm']:<16} warm {r['warm_tf'] or float('nan'):7.1f} @ {r['warm_sclk']} MHz {r['warm_w']} W "
                       f"{None if r['warm_mj'] is None else round(r['warm_mj'], 1)} mJ  cold-start {r['cold_tf']:7.1f}")
        for kind, nm in (("warm_pairs", "warm"), ("cold_pairs", "cold-start")):
            for name, c in (rep.get(kind) or {}).items():
                out.append(f"  {nm:<10} {name:<22} {TU._ci(c)}")
    return "\n".join(out)


def cmd_report(args) -> int:
    print(confirm_text([json.loads(Path(p).read_text()) for p in args.confirm]))
    return 0


# ===================================================================== CLI


def _common(r) -> None:
    r.add_argument("--tile", required=True, help="MT0xMT1xDepthU (weights x tokens x DepthU)")
    r.add_argument("--family", choices=("ak", "am"), default="ak")
    r.add_argument("--fmt", choices=("e2m3", "e3m2"), default="e2m3")
    r.add_argument("--problem", default="S", help="S | G | S-k<K> (libtiles) | b,m,n,k[,rotate]")
    r.add_argument("--anchor", choices=("dispatch", "family"), default="dispatch",
                   help="base: library.dispatch (shipped kernel: WG_MULT tile, edge_tiles, schedule) or family_spec")
    r.add_argument("--ring", default="auto", help="family_spec ring of the base: auto | stage | line2m")
    r.add_argument("--rotate", type=int, default=0, help="replicated timing buffer sets (0: the problem's own)")
    r.add_argument("--n-src", type=int, default=3, help="distinct data sources cycled over the entries (verify)")
    r.add_argument("--base", default="", help="JSON overrides on family_spec for the base / anchor (or a file)")
    r.add_argument("--rust-levers", default="", help="base lever set (default: library.LEVERS)")
    r.add_argument("--lever-axes", default="", help="'+'-separated levers searched on / off (default: LEVER_AXES)")
    r.add_argument("--exclude", default="")
    r.add_argument("--extra-knobs", default="")
    r.add_argument("--out", default="libsearch-out")
    r.add_argument("--cache", default="mxfp6_cache")
    r.add_argument("--seed-data", type=int, default=42, help="libbench data seed (sources: seed + 1000 s)")
    r.add_argument("--verify-dists", default="int,gauss,uniform")
    r.add_argument("--verify-launches", type=int, default=2)
    r.add_argument("--time-dist", default="gauss")
    r.add_argument("--jobs", type=int, default=16)
    r.add_argument("--gen-workers", type=int, default=16)
    r.add_argument("--hang-s", type=float, default=120.0)
    r.add_argument("--rust-compiler", default="")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--warm-from", default="", help="warm starts: JSON list / {label: over} / a summary.json")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("space")
    _common(s)
    s.add_argument("--samples", type=int, default=200)
    sub.add_parser("worker")
    r = sub.add_parser("run")
    _common(r)
    r.add_argument("--mirror", default="")
    r.add_argument("--resume", default="")
    r.add_argument("--structure-only", type=int, default=0, help="1: only the structural knobs + layout bundles")
    r.add_argument("--struct-seeds", type=int, default=32, help="max structural warm starts (0: none)")
    r.add_argument("--compile-server", type=int, default=1, help="1: long-lived compile processes (identity-checked)")
    r.add_argument("--mutants", choices=("all", "final"), default="all",
                   help="final: candidates are verified but their must-fail mutants run only for the finalists "
                        "(confirmations always run them)")
    r.add_argument("--members", default="lookback,hill,gp,de,hyperband")
    r.add_argument("--lookback-module", default="")
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--budget", type=int, default=300)
    r.add_argument("--minutes", type=float, default=110.0)
    r.add_argument("--reserve-s", type=float, default=300.0)
    r.add_argument("--stall-batches", type=int, default=15)
    r.add_argument("--min-evals", type=int, default=120)
    r.add_argument("--race", type=int, default=3)
    r.add_argument("--objective", choices=("time", "energy"), default="time")
    r.add_argument("--idle-batches", type=int, default=25)
    r.add_argument("--noop-prepass", type=int, default=1)
    r.add_argument("--gap-ms", type=float, default=0.0)
    r.add_argument("--rounds", type=int, default=10)
    r.add_argument("--screen-rounds", type=int, default=5)
    r.add_argument("--discard-rounds", type=int, default=1)
    r.add_argument("--search-warmup", type=int, default=3)
    r.add_argument("--search-iters", type=int, default=40)
    r.add_argument("--search-ms", type=float, default=120.0, help="ms of launches per search sample (0: --search-iters)")
    r.add_argument("--gain-sigmas", type=float, default=1.0)
    r.add_argument("--min-gain", type=float, default=0.003)
    r.add_argument("--final-rounds", type=int, default=20)
    r.add_argument("--sustain-s", type=float, default=2.0)
    r.add_argument("--sustained-rounds", type=int, default=3)
    r.add_argument("--top", type=int, default=5)
    r.add_argument("--gate", choices=("strict", "loop", "off"), default="strict")
    r.add_argument("--max-restarts", type=int, default=12)
    r.add_argument("--aiter-co", default="")
    r.add_argument("--aiter-symbol", default=TU.AITER_DEFAULT_SYMBOL)
    q = sub.add_parser("confirm")
    _common(q)
    q.add_argument("--arms", required=True, help="JSON {name: overrides} (needs 'best'; 'ref' = the family default)")
    q.add_argument("--cold-rounds", type=int, default=15)
    q.add_argument("--warm-windows", type=int, default=10)
    q.add_argument("--warm-s", type=float, default=2.0)
    rp = sub.add_parser("report")
    rp.add_argument("confirm", nargs="+")
    args = p.parse_args(argv)
    return {"space": cmd_space, "worker": cmd_worker, "run": cmd_run, "confirm": cmd_confirm,
            "report": cmd_report}[args.cmd](args)


def _no_gc():
    """The cyclic GC paused while an IR is built / serialized (2x faster; kernels.gfx950.mxfp6_gemm.gemm.no_gc)."""
    from kernels.gfx950.mxfp6_gemm.gemm import no_gc

    return no_gc()


if __name__ == "__main__":  # (last: `python -m` runs main() here, so every helper above must already be defined)
    raise SystemExit(main())
