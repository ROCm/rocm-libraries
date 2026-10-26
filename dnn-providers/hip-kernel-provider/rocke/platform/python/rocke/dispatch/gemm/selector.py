# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Learned kernel selection for the RCR UniversalGemm families.

The rule-based registry (:class:`CandidateRegistry` priorities plus each
candidate's admission rules) picks ONE body per request. This module ranks a
wider, still fully gated, candidate set with a gradient-boosted tree model of
``log(time / roofline)`` and returns the predicted-fastest spec:

* **candidates** (:func:`selector_candidates`): every registered candidate that
  admits the request with its ``sweep_space`` variants (``registered``), one-step
  trait mutations of those specs over the levers that tuning found effective
  (``neighbour``), and a small library of generic tile configurations that tile
  the request exactly (``library``). Generated specs pass the dispatcher's own
  gates (``universal_spec_supported``, ``request_shape_supported``,
  ``launch_shape_error``) and the stricter :func:`generated_spec_error` rules.
* **features** (:func:`feature_vector`): problem extents and alignment, the
  tile / warp / atom geometry, every ``TraitSpec`` field, and analytic
  quantities (grid waves, last-wave fill, tile fill, K iterations, a
  wave-quantised compute / memory roofline). Pure Python; the C++ scorer
  (hipTensor) mirrors the same arithmetic.
* **model**: a LightGBM text model walked by :class:`TreeEnsemble` (no
  LightGBM dependency at run time), one per arch, with a JSON sidecar that
  fixes the feature order, the library, the neighbour levers and the
  per-class decision margins.
* **decision** (:func:`select`): the predicted-fastest candidate replaces the
  rule-based pick only when its predicted gain exceeds the margin of the rule
  pick's class (``margins`` in the sidecar; classes where the evaluation data
  showed the model to be weaker than the rules keep the rule pick).

``ROCKE_GEMM_SELECTOR`` = ``0`` turns the selector off (the rule-based
registry decides alone, byte-identical to the pre-selector dispatcher);
``1`` (default) turns it on for arches that ship a model. Requests pinned to an
``algorithm`` / ``spec_id`` never go through the selector.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

FEATURE_VERSION = "r2c-v2"
# Feature versions this code computes. "g4-v1" models (UniversalGemm specs only)
# load unchanged: every v1 feature keeps its definition, and a v1 model never
# ranks gemm_skinny configurations (it has no family features).
SUPPORTED_FEATURE_VERSIONS: Tuple[str, ...] = ("g4-v1", "r2c-v2")
# Feature versions whose models know the gemm_skinny family (``s_*`` features).
SKINNY_FEATURE_VERSIONS: Tuple[str, ...] = ("r2c-v2",)
SELECTOR_ENV = "ROCKE_GEMM_SELECTOR"
MODEL_DIR_ENV = "ROCKE_GEMM_SELECTOR_MODEL_DIR"
MODEL_DIR = Path(__file__).with_name("selector_models")

# native NN / TN / TT selection (``gemm_native_<arch>``
# models). The RCR model above is unchanged; a non-RCR request is ranked by the
# native model over the native candidate set (:func:`native_candidates`).
NATIVE_FEATURE_VERSION = "r2n-v1"
# ``0`` = native requests keep the rule order (the dispatcher's priority pick
# and its preferred native form); ``1`` = on wherever a native model ships
# (default: the native sidecar's ``default_on``). ``ROCKE_GEMM_SELECTOR=0``
# turns both selectors off.
NATIVE_SELECTOR_ENV = "ROCKE_GEMM_SELECTOR_NATIVE"
SELECTOR_NATIVE_DEFAULT_ON = True
# Candidate sources of a native decision: the registered candidates' native
# specs (and sweep variants), every native form of every registered RCR spec
# (``native_layout_alternatives``), and one-step mutations of both.
SELECTOR_NATIVE_SOURCES: Tuple[str, ...] = ("registered", "form", "neighbour")
# Native-form levers of the "neighbour" source (on top of
# SELECTOR_NEIGHBOUR_LEVERS): the MN chunk width and LDS pad of the transpose-read
# block layout, the ring depth / swizzle, and the persistent grid (the native
# variants covered by the training data).
SELECTOR_NATIVE_NEIGHBOUR_LEVERS: Tuple[Tuple[str, Tuple[Any, ...]], ...] = (
    ("mn_chunk", (64, 128, 256)),
    ("mn_pad", (64,)),
    ("persistent_ctas", (0,)),
    ("dtl_stages", (2, 3)),
    ("lds_swizzle", (True, False)),
    ("kloop_unroll", (1,)),
)
# Two-stage native ranking: only the rule pick and the best
# SELECTOR_NATIVE_NEIGHBOUR_BASES registered / form specs (by the model) are
# mutated (0 = every base; the full set costs much more dispatch time on large
# shapes for no gain on the training cells).
SELECTOR_NATIVE_NEIGHBOUR_BASES = 8

# ---------------------------------------------------------------------------
# Knobs (each lever is a switch; the sidecar can only narrow, never widen, the
# generated set).
# ---------------------------------------------------------------------------
SELECTOR_DEFAULT_ON = True
# Candidate sources the default decision ranks.
SELECTOR_SOURCES: Tuple[str, ...] = ("registered", "neighbour", "library", "skinny")
# Selector-local launch-safety rules for generated (non-registered) specs, on
# top of the dispatcher's gates (see generated_spec_error).
SELECTOR_SAFE_GATE = True
# Portfolio guard: when the model's sidecar ships bootstrap bags
# (``guard.bags``) and enables them (``guard.enabled``, default true), a switch
# away from the rule pick also needs every bag to predict a gain above
# ``guard.min_gain``; otherwise the rule pick is kept (the fallback of the
# portfolio). False = the margin rule alone. (The shipped gfx950 sidecar
# leaves it off: in same-process GPU evaluations it did not improve on the
# margin rule alone.)
SELECTOR_PORTFOLIO_GUARD = True
# Margin used when a model sidecar names none (log-time units).
SELECTOR_DEFAULT_MARGIN = 0.03
# Floor of the per-class margin (log-time units). The sidecar's margins are
# fitted on single-batch timings, whose node / thermal noise sets this floor;
# below it a predicted gain is not evidence. The beat bodies were verified with
# sustained-window anchor timing, so a generated variant must be predicted
# clearly faster to replace one.
SELECTOR_MIN_MARGIN: Dict[str, float] = {
    "default": 0.02,
    "dtl": 0.03,
    "dtl:persist": 0.03,
    "dtl:beat": 0.05,
}
# One-step trait mutations ("neighbour" source). Levers that change the launch
# contract beyond split-K (persistent / stream-K pools, tile M / N, pads, DTL
# K handling) are deliberately absent: those are owned by registered rules.
SELECTOR_NEIGHBOUR_LEVERS: Tuple[Tuple[str, Tuple[Any, ...]], ...] = (
    ("waves_per_eu", (1, 2)),
    ("chiplet_chunk_size", (4, 8, 16, 32)),
    ("chiplet_wgm", (4, 8, 16)),
    ("split_k", (2, 4, 8)),
    ("tile_k", (32, 64, 128)),
    ("kloop_unroll", (2, 4)),
    ("pad_fast_tile", (True,)),
    # the DTL ring depth (prefetch depth); direct_to_lds parents only.
    # A knob: the shipped sidecar leaves it out of its neighbour_levers (on
    # the 2-stage small ring, changing only this lever to 3 stages is slower;
    # the co-tuned 3-stage ring variants are registered sweep variants already).
    ("dtl_stages", (2, 3)),
)
# Levers that only exist on direct_to_lds bodies (never offered elsewhere)
_DTL_ONLY_LEVERS = ("dtl_stages",)
# gemm_skinny (register-direct split-K body) in the ranked set: the registered
# skinny candidates' configurations (region table / fallback plan) and their
# generated variants. Only models whose feature version knows the family
# (SKINNY_FEATURE_VERSIONS) rank them.
SELECTOR_SKINNY = True
# split-K degrees offered for every registered skinny configuration (each must
# leave every K slice non-empty; see skinny_variant_error)
SELECTOR_SKINNY_SPLITS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
# one-step mutations of a registered skinny configuration (at its own split)
SELECTOR_SKINNY_LEVERS: Tuple[Tuple[str, Tuple[Any, ...]], ...] = (
    ("kc", (8, 16, 32)),
    ("prefetch", (1, 2, 4)),
    ("warp_k", (1, 2, 4, 8)),
)

_TILE_FIELDS = (
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_k",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
)

# Nominal per-arch constants of the analytic roofline features. They only set
# the compute : memory balance of a FEATURE (and the shape-only normaliser of
# the regression target); the model learns the rest. Units: per clock.
ARCH_CONSTANTS: Dict[str, Dict[str, float]] = {
    "gfx950": dict(cus=256.0, lds=160.0 * 1024, flops_clk_cu=4096.0, bytes_clk=3300.0),
    "gfx942": dict(cus=304.0, lds=64.0 * 1024, flops_clk_cu=2048.0, bytes_clk=2600.0),
}

_PIPELINE_CODE = {"mem": 0.0, "compv3": 1.0, "compv4": 2.0}
_SCHEDULER_CODE = {"intrawave": 0.0, "interwave": 1.0}
_EPILOGUE_CODE = {"default": 0.0, "cshuffle": 1.0}


# ---------------------------------------------------------------------------
# Config strings: ``t{tm}x{tn}x{tk}_w{wm}x{wn}x{wk}_a{am}x{an}x{ak}|k=v,...``
# (non-default TraitSpec fields only), the key the sweeps record.
# ---------------------------------------------------------------------------
_CFG_RE = re.compile(
    r"^t(\d+)x(\d+)x(\d+)_w(\d+)x(\d+)x(\d+)_a(\d+)x(\d+)x(\d+)\|?(.*)$"
)


def _parse_value(v: str) -> Any:
    if v == "True":
        return True
    if v == "False":
        return False
    if v == "None":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def parse_config(cfg: str) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """``(tile fields, trait overrides)`` of a UniversalGemm config string."""
    m = _CFG_RE.match(cfg.strip())
    if not m:
        raise ValueError(f"not a gemm config string: {cfg!r}")
    tile = dict(zip(_TILE_FIELDS, (int(x) for x in m.groups()[:9])))
    trait: Dict[str, Any] = {}
    rest = m.group(10)
    if rest:
        for kv in rest.split(","):
            if not kv:
                continue
            k, _, v = kv.partition("=")
            trait[k] = _parse_value(v)
    return tile, trait


SKINNY_CFG_PREFIX = "sk|"
# SkinnyGemmSpec fields that are not configuration (the dtype is a problem
# feature; the name is cosmetic)
_SKINNY_NONCONFIG = ("dtype", "name")


def is_skinny_config(cfg: str) -> bool:
    return cfg.strip().startswith(SKINNY_CFG_PREFIX)


def _skinny_defaults() -> Dict[str, Any]:
    from ...instances.common.gemm_skinny import SkinnyGemmSpec

    d = dataclasses.asdict(SkinnyGemmSpec())
    for k in _SKINNY_NONCONFIG:
        d.pop(k, None)
    return d


def skinny_config_string(spec: Any) -> str:
    """``sk|k=v,...``: the non-default SkinnyGemmSpec fields (sorted) and the
    split-K degree (always present)."""
    d = dataclasses.asdict(spec)
    dflt = _skinny_defaults()
    kv = {k: v for k, v in d.items() if k in dflt and k != "split" and dflt[k] != v}
    kv["split"] = int(spec.split)
    return SKINNY_CFG_PREFIX + ",".join(f"{k}={v}" for k, v in sorted(kv.items()))


def parse_skinny_config(cfg: str) -> Dict[str, Any]:
    """The full SkinnyGemmSpec field dict (without dtype / name) of an
    ``sk|`` config string."""
    s = cfg.strip()
    if not s.startswith(SKINNY_CFG_PREFIX):
        raise ValueError(f"not a gemm_skinny config string: {cfg!r}")
    out = _skinny_defaults()
    for kv in s[len(SKINNY_CFG_PREFIX) :].split(","):
        if not kv:
            continue
        k, _, v = kv.partition("=")
        if k not in out:
            raise ValueError(f"unknown gemm_skinny field {k!r} in {cfg!r}")
        out[k] = _parse_value(v)
    return out


def config_string(spec: Any) -> str:
    """The config string of a UniversalGemm spec (inverse of parse_config), or
    of a gemm_skinny spec (``sk|...``, inverse of parse_skinny_config)."""
    if _skinny(spec):
        return skinny_config_string(spec)
    t, tr = spec.tile, spec.trait
    d = dataclasses.asdict(tr)
    dflt = dataclasses.asdict(type(tr)())
    nd = ",".join(f"{k}={v}" for k, v in sorted(d.items()) if dflt.get(k) != v)
    return (
        f"t{t.tile_m}x{t.tile_n}x{t.tile_k}_w{t.warp_m}x{t.warp_n}x{t.warp_k}"
        f"_a{t.warp_tile_m}x{t.warp_tile_n}x{t.warp_tile_k}|{nd}"
    )


def flatten_spec(spec: Any) -> Dict[str, Any]:
    """Tile + trait fields of a spec as one flat dict (the feature input),
    plus its data ``layout`` (read by the layout features only); a gemm_skinny
    spec: ``family="skinny"`` + its configuration fields."""
    if _skinny(spec):
        d = dataclasses.asdict(spec)
        flat_s: Dict[str, Any] = {"family": "skinny"}
        flat_s.update({k: v for k, v in d.items() if k not in _SKINNY_NONCONFIG})
        return flat_s
    flat: Dict[str, Any] = {f: int(getattr(spec.tile, f)) for f in _TILE_FIELDS}
    flat.update(dataclasses.asdict(spec.trait))
    data = getattr(spec, "data", None)
    flat["layout"] = str(getattr(data, "layout", "RCR") or "RCR").upper()
    return flat


def flat_from_config(cfg: str) -> Dict[str, Any]:
    if is_skinny_config(cfg):
        flat_s: Dict[str, Any] = {"family": "skinny"}
        flat_s.update(parse_skinny_config(cfg))
        return flat_s
    tile, trait = parse_config(cfg)
    flat: Dict[str, Any] = dict(tile)
    flat.update(trait)
    return flat


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _enc(v: Any) -> float:
    """Numeric encoding of one trait value (None -> -1, str -> nonempty)."""
    if v is None:
        return -1.0
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        if v in _PIPELINE_CODE:
            return _PIPELINE_CODE[v]
        if v in _SCHEDULER_CODE:
            return _SCHEDULER_CODE[v]
        if v in _EPILOGUE_CODE:
            return _EPILOGUE_CODE[v]
        return 1.0 if v else 0.0
    return 0.0


def _ln(x: float) -> float:
    return math.log(max(float(x), 1e-30))


def roofline_cycles(arch: str, M: int, N: int, K: int) -> float:
    """Shape-only ideal roofline (cycles): the regression target normaliser."""
    a = ARCH_CONSTANTS[arch]
    tc = 2.0 * M * N * K / (a["cus"] * a["flops_clk_cu"])
    tm = 2.0 * (M * K + N * K + M * N) / a["bytes_clk"]
    return max(tc, tm)


PROBLEM_FEATURES: Tuple[str, ...] = (
    "p_ln_m",
    "p_ln_n",
    "p_ln_k",
    "p_ln_mn",
    "p_ln_min_mn",
    "p_ln_max_mn",
    "p_ln_aspect",
    "p_bf16",
    "p_m_a8",
    "p_m_a64",
    "p_m_a256",
    "p_n_a8",
    "p_n_a64",
    "p_n_a256",
    "p_k_a8",
    "p_k_a64",
    "p_k_a256",
    "p_ln_roof",
    "p_comp_share",
    "p_cold",
)

DERIVED_FEATURES: Tuple[str, ...] = (
    "d_tm",
    "d_tn",
    "d_tk",
    "d_wm",
    "d_wn",
    "d_wk",
    "d_atm",
    "d_atn",
    "d_atk",
    "d_nwave",
    "d_tiles_m",
    "d_tiles_n",
    "d_ln_tiles",
    "d_split",
    "d_ln_nwg",
    "d_occ",
    "d_waves",
    "d_last_fill",
    "d_wave_eff",
    "d_fill_m",
    "d_fill_n",
    "d_fill",
    "d_ln_kiters",
    "d_ktail",
    "d_kslice_tail",
    "d_ai",
    "d_lds_kb",
    "d_acc_regs",
    "d_ln_tcomp",
    "d_ln_tmem",
    "d_ln_tq",
    "d_ln_tq_over_roof",
    "d_tiles_per_cu",
    "d_persist_ratio",
    "d_pad_mn",
)

# Data-layout features (native models only; an RCR model's sidecar never names
# them, so its feature vector is unchanged). ``layout`` is read from the flat
# spec (:func:`flatten_spec` adds the spec's data layout; config strings carry
# none, i.e. RCR).
LAYOUT_FEATURES: Tuple[str, ...] = (
    "x_a_mn",
    "x_b_mn",
    "x_n_mn",
    "x_mn_dtl",
    "x_mn_vgpr",
)


def _layout_features(flat: Mapping[str, Any]) -> Dict[str, float]:
    """``x_a_mn`` / ``x_b_mn``: A M-contiguous / B N-contiguous; ``x_n_mn`` how
    many; ``x_mn_dtl``: an MN operand read through the DTL ring's transpose-read
    block layout; ``x_mn_vgpr``: through the VGPR-staged transposing loader."""
    lay = str(flat.get("layout", "RCR") or "RCR").upper()
    a_mn = lay[:1] == "C"
    b_mn = lay[1:2] == "R"
    mn = a_mn or b_mn
    ring = (
        bool(flat.get("direct_to_lds", False))
        and bool(flat.get("dtl_prefetch", False))
        and int(_enc(flat.get("dtl_stages", 0))) >= 2
    )
    return {
        "x_a_mn": float(a_mn),
        "x_b_mn": float(b_mn),
        "x_n_mn": float(a_mn) + float(b_mn),
        "x_mn_dtl": float(mn and ring),
        "x_mn_vgpr": float(mn and not ring),
    }


def _derived(
    arch: str, M: int, N: int, K: int, flat: Mapping[str, Any]
) -> Dict[str, float]:
    a = ARCH_CONSTANTS[arch]
    cus = a["cus"]
    tm = float(flat.get("tile_m", 128))
    tn = float(flat.get("tile_n", 128))
    tk = float(flat.get("tile_k", 32))
    wm = float(flat.get("warp_m", 1))
    wn = float(flat.get("warp_n", 1))
    wk = float(flat.get("warp_k", 1))
    nwave = max(1.0, wm * wn * wk)
    tiles_m = math.ceil(M / tm)
    tiles_n = math.ceil(N / tn)
    tiles = float(tiles_m * tiles_n)
    split = float(max(1, int(_enc(flat.get("split_k", 1)))))
    persist = float(max(0, int(_enc(flat.get("persistent_ctas", 0)))))
    stages = int(_enc(flat.get("dtl_stages", 0)))
    pipe = flat.get("pipeline", "compv4")
    nbuf = float(stages) if stages >= 2 else (1.0 if pipe == "mem" else 2.0)
    lds_bytes = (tm + tn) * tk * 2.0 * nbuf
    acc = tm * tn / (nwave * 64.0)
    vgpr = acc + (tm + tn) * tk * 2.0 / (nwave * 64.0 * 4.0) + 32.0
    wps = min(8.0, max(0.0, math.floor(512.0 / max(vgpr, 1.0))))
    occ = max(
        1.0,
        min(math.floor(wps * 4.0 / nwave), math.floor(a["lds"] / max(lds_bytes, 1.0))),
    )
    nwg = tiles * split
    slots = cus * occ
    if persist > 0:
        pool = max(1.0, min(persist, slots))
        waves = math.ceil(nwg / pool)
        last = (nwg - (waves - 1) * pool) / pool
        wave_eff = nwg / (waves * pool) * min(1.0, pool / slots)
    else:
        waves = math.ceil(nwg / slots)
        last = (nwg - (waves - 1) * slots) / slots
        wave_eff = nwg / (waves * slots)
    fill_m = M / (tiles_m * tm)
    fill_n = N / (tiles_n * tn)
    kslice = math.ceil(K / split)
    kiters = math.ceil(kslice / tk)
    t_comp = waves * occ * 2.0 * tm * tn * kiters * tk / a["flops_clk_cu"]
    bytes_ = 2.0 * (M * K + N * K + M * N) + (4.0 * M * N * split if split > 1 else 0.0)
    t_mem = bytes_ / a["bytes_clk"]
    t_q = max(t_comp, t_mem)
    roof = roofline_cycles(arch, M, N, K)
    pad_mn = float(bool(flat.get("pad_m", False))) + float(
        bool(flat.get("pad_n", False))
    )
    return {
        "d_tm": tm,
        "d_tn": tn,
        "d_tk": tk,
        "d_wm": wm,
        "d_wn": wn,
        "d_wk": wk,
        "d_atm": float(flat.get("warp_tile_m", 32)),
        "d_atn": float(flat.get("warp_tile_n", 32)),
        "d_atk": float(flat.get("warp_tile_k", 8)),
        "d_nwave": nwave,
        "d_tiles_m": float(tiles_m),
        "d_tiles_n": float(tiles_n),
        "d_ln_tiles": _ln(tiles),
        "d_split": split,
        "d_ln_nwg": _ln(nwg),
        "d_occ": occ,
        "d_waves": float(waves),
        "d_last_fill": last,
        "d_wave_eff": wave_eff,
        "d_fill_m": fill_m,
        "d_fill_n": fill_n,
        "d_fill": fill_m * fill_n,
        "d_ln_kiters": _ln(kiters),
        "d_ktail": 1.0 if K % int(tk) else 0.0,
        "d_kslice_tail": 1.0 if (K % int(split * tk)) else 0.0,
        "d_ai": tm * tn / (tm + tn),
        "d_lds_kb": lds_bytes / 1024.0,
        "d_acc_regs": acc,
        "d_ln_tcomp": _ln(t_comp),
        "d_ln_tmem": _ln(t_mem),
        "d_ln_tq": _ln(t_q),
        "d_ln_tq_over_roof": _ln(t_q) - _ln(roof),
        "d_tiles_per_cu": nwg / cus,
        "d_persist_ratio": (persist / tiles) if persist > 0 else 0.0,
        "d_pad_mn": pad_mn,
    }


# gemm_skinny family features (r2c-v2). UniversalGemm rows: s_fam = 0 and every
# other s_* = -1; gemm_skinny rows: the configuration fields and its launch
# quantities. Their derived (d_*) features use the equivalent tile geometry
# (_skinny_equiv) and their tr_* features the trait defaults.
SKINNY_FEATURES: Tuple[str, ...] = (
    "s_fam",
    "s_ma",
    "s_na",
    "s_wm",
    "s_wn",
    "s_wk",
    "s_kc",
    "s_pf",
    "s_wpe",
    "s_ntws",
    "s_bload",
    "s_kml",
    "s_kalign",
    "s_rsc1",
    "s_ru",
    "s_split",
    "s_ln_kps",
    "s_regs",
    "s_ln_nwg",
    "s_wave_fill",
)
_SKINNY_BLOAD_CODE = {"buffer": 0.0, "global": 1.0, "global_nt": 2.0}


def _skinny_equiv(M: int, N: int, K: int, flat: Mapping[str, Any]) -> Dict[str, Any]:
    """The UniversalGemm-like geometry of a gemm_skinny configuration (input
    of the shared derived features)."""
    ma, na = int(flat.get("wave_m_atoms", 1)), int(flat.get("wave_n_atoms", 4))
    wm, wn, wk = int(flat.get("warp_m", 1)), int(flat.get("warp_n", 1)), int(flat.get("warp_k", 1))
    tm, tn = 16 * ma * wm, 16 * na * wn
    return {
        "tile_m": tm,
        "tile_n": tn,
        "tile_k": 4 * int(flat.get("kc", 16)) * wk,
        "warp_m": wm,
        "warp_n": wn,
        "warp_k": wk,
        "warp_tile_m": 16,
        "warp_tile_n": 16,
        "warp_tile_k": 32,
        "split_k": max(1, int(flat.get("split", 1))),
        "pad_m": bool(M % tm),
        "pad_n": bool(N % tn),
    }


def _skinny_features(
    arch: str, M: int, N: int, K: int, flat: Optional[Mapping[str, Any]]
) -> Dict[str, float]:
    if flat is None:
        f = {n: -1.0 for n in SKINNY_FEATURES}
        f["s_fam"] = 0.0
        return f
    a = ARCH_CONSTANTS[arch]
    ma, na = int(flat.get("wave_m_atoms", 1)), int(flat.get("wave_n_atoms", 4))
    wm, wn, wk = int(flat.get("warp_m", 1)), int(flat.get("warp_n", 1)), int(flat.get("warp_k", 1))
    kc, pf = int(flat.get("kc", 16)), int(flat.get("prefetch", 2))
    split = max(1, int(flat.get("split", 1)))
    tm, tn = 16 * ma * wm, 16 * na * wn
    steps = -(-int(K) // (4 * kc))
    kps = -(-steps // split)
    nwg = -(-int(M) // tm) * -(-int(N) // tn) * split
    waves = nwg * wm * wn * wk
    acc = ma * na * 4
    frag = (ma + na) * (kc // 2)
    # waves resident per CU at most 4 SIMDs x 8 (no LDS for operands)
    slots = a["cus"] * 32.0
    return {
        "s_fam": 1.0,
        "s_ma": float(ma),
        "s_na": float(na),
        "s_wm": float(wm),
        "s_wn": float(wn),
        "s_wk": float(wk),
        "s_kc": float(kc),
        "s_pf": float(pf),
        "s_wpe": float(int(flat.get("waves_per_eu", 0))),
        "s_ntws": 1.0 if flat.get("nt_ws") else 0.0,
        "s_bload": _SKINNY_BLOAD_CODE.get(str(flat.get("b_load", "buffer")), 0.0),
        "s_kml": 1.0 if flat.get("kmap") == "lane" else 0.0,
        "s_kalign": float(int(flat.get("kalign", 8))),
        "s_rsc1": 1.0 if flat.get("red_sync") == "sc1" else 0.0,
        "s_ru": float(int(flat.get("red_unroll", 4))),
        "s_split": float(split),
        "s_ln_kps": _ln(kps),
        "s_regs": float(acc + (pf + 1) * frag),
        "s_ln_nwg": _ln(nwg),
        "s_wave_fill": min(1.0, waves / slots),
    }


def _problem(
    arch: str, dtype: str, M: int, N: int, K: int, cold: float
) -> Dict[str, float]:
    a = ARCH_CONSTANTS[arch]
    tc = 2.0 * M * N * K / (a["cus"] * a["flops_clk_cu"])
    tmm = 2.0 * (M * K + N * K + M * N) / a["bytes_clk"]
    return {
        "p_ln_m": _ln(M),
        "p_ln_n": _ln(N),
        "p_ln_k": _ln(K),
        "p_ln_mn": _ln(float(M) * N),
        "p_ln_min_mn": _ln(min(M, N)),
        "p_ln_max_mn": _ln(max(M, N)),
        "p_ln_aspect": _ln(M) - _ln(N),
        "p_bf16": 1.0 if dtype in ("bf16", "bfloat16") else 0.0,
        "p_m_a8": float(M % 8 == 0),
        "p_m_a64": float(M % 64 == 0),
        "p_m_a256": float(M % 256 == 0),
        "p_n_a8": float(N % 8 == 0),
        "p_n_a64": float(N % 64 == 0),
        "p_n_a256": float(N % 256 == 0),
        "p_k_a8": float(K % 8 == 0),
        "p_k_a64": float(K % 64 == 0),
        "p_k_a256": float(K % 256 == 0),
        "p_ln_roof": _ln(max(tc, tmm)),
        "p_comp_share": tc / (tc + tmm),
        "p_cold": float(cold),
    }


def feature_dict(
    arch: str,
    dtype: str,
    M: int,
    N: int,
    K: int,
    flat: Mapping[str, Any],
    *,
    trait_fields: Sequence[str],
    trait_defaults: Mapping[str, Any],
    cold: float = 0.0,
) -> Dict[str, float]:
    """Every feature of one (problem, config) pair by name. Fields absent from
    ``flat`` take ``trait_defaults`` (then 0). A gemm_skinny ``flat``
    (``family="skinny"``) gets its equivalent tile geometry for the derived
    features, the trait defaults for ``tr_*`` and its own ``s_*`` features."""
    full: Dict[str, Any] = dict(trait_defaults)
    skinny = flat.get("family") == "skinny"
    if skinny:
        full.update(_skinny_equiv(int(M), int(N), int(K), flat))
    else:
        full.update(flat)
    f = _problem(arch, dtype, int(M), int(N), int(K), cold)
    f.update(_derived(arch, int(M), int(N), int(K), full))
    f.update(_layout_features(full))
    f.update(_skinny_features(arch, int(M), int(N), int(K), flat if skinny else None))
    for name in trait_fields:
        f["tr_" + name] = _enc(full.get(name, 0))
    return f


def feature_vector(
    names: Sequence[str],
    arch: str,
    dtype: str,
    M: int,
    N: int,
    K: int,
    flat: Mapping[str, Any],
    *,
    trait_defaults: Mapping[str, Any],
    cold: float = 0.0,
) -> List[float]:
    """Features in ``names`` order (``tr_*`` names select the trait fields)."""
    trait_fields = [n[3:] for n in names if n.startswith("tr_")]
    d = feature_dict(
        arch,
        dtype,
        M,
        N,
        K,
        flat,
        trait_fields=trait_fields,
        trait_defaults=trait_defaults,
        cold=cold,
    )
    return [d[n] for n in names]


# ---------------------------------------------------------------------------
# LightGBM text model walker
# ---------------------------------------------------------------------------
@dataclass
class _Tree:
    split_feature: List[int]
    threshold: List[float]
    decision_type: List[int]
    left: List[int]
    right: List[int]
    leaf_value: List[float]
    num_leaves: int


class TreeEnsemble:
    """Numerical-split LightGBM text model (the subset rocKE models use)."""

    def __init__(self, trees: List[_Tree], feature_names: List[str]):
        self.trees = trees
        self.feature_names = feature_names

    @classmethod
    def from_text(cls, text: str) -> "TreeEnsemble":
        trees: List[_Tree] = []
        names: List[str] = []
        cur: Dict[str, str] = {}

        def flush():
            if not cur:
                return
            nl = int(cur.get("num_leaves", "1"))

            def ints(k):
                return [int(x) for x in cur.get(k, "").split()]

            def flts(k):
                return [float(x) for x in cur.get(k, "").split()]

            if nl > 1 and any(d & 1 for d in ints("decision_type")):
                raise ValueError("categorical splits are not supported")
            trees.append(
                _Tree(
                    ints("split_feature"),
                    flts("threshold"),
                    ints("decision_type"),
                    ints("left_child"),
                    ints("right_child"),
                    flts("leaf_value"),
                    nl,
                )
            )
            cur.clear()

        in_tree = False
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("feature_names=") and not names:
                names = line.split("=", 1)[1].split()
            elif line.startswith("Tree="):
                flush()
                in_tree = True
                cur["_"] = "1"
            elif line.startswith("end of trees"):
                flush()
                break
            elif in_tree and "=" in line:
                k, _, v = line.partition("=")
                cur[k] = v
        flush()
        if not trees:
            raise ValueError("model has no trees")
        return cls(trees, names)

    def predict_one(self, x: Sequence[float]) -> float:
        s = 0.0
        for t in self.trees:
            if t.num_leaves <= 1:
                s += t.leaf_value[0] if t.leaf_value else 0.0
                continue
            n = 0
            sf, th, dt, lc, rc = (
                t.split_feature,
                t.threshold,
                t.decision_type,
                t.left,
                t.right,
            )
            while n >= 0:
                v = x[sf[n]]
                d = dt[n]
                missing = (d >> 2) & 3
                if missing == 1 and abs(v) <= 1e-35:
                    n = lc[n] if (d & 2) else rc[n]
                elif v <= th[n]:
                    n = lc[n]
                else:
                    n = rc[n]
            s += t.leaf_value[~n]
        return s

    def predict(self, xs: Sequence[Sequence[float]]) -> List[float]:
        """Scores of every row; vectorised over rows with NumPy when it is
        importable (same arithmetic as :meth:`predict_one`, which the tests
        compare it with), else row by row."""
        if len(xs) < 4:
            return [self.predict_one(x) for x in xs]
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy is optional
            return [self.predict_one(x) for x in xs]
        X = np.asarray(xs, dtype=np.float64)
        rows = np.arange(X.shape[0])
        total = np.zeros(X.shape[0], dtype=np.float64)
        for t in self._np_trees(np):
            if t is None:
                continue
            if isinstance(t, float):
                total += t
                continue
            sf, th, dl, mz, lc, rc, lv = t
            node = np.zeros(X.shape[0], dtype=np.int64)
            active = node >= 0
            while active.any():
                idx = np.nonzero(active)[0]
                nd = node[idx]
                v = X[rows[idx], sf[nd]]
                zero = mz[nd] & (np.abs(v) <= 1e-35)
                go_left = np.where(zero, dl[nd], v <= th[nd])
                node[idx] = np.where(go_left, lc[nd], rc[nd])
                active = node >= 0
            total += lv[~node]
        return total.tolist()

    def _np_trees(self, np):
        cache = getattr(self, "_np_cache", None)
        if cache is not None:
            return cache
        out = []
        for t in self.trees:
            if t.num_leaves <= 1:
                out.append(float(t.leaf_value[0]) if t.leaf_value else None)
                continue
            dt = np.asarray(t.decision_type, dtype=np.int64)
            out.append(
                (
                    np.asarray(t.split_feature, dtype=np.int64),
                    np.asarray(t.threshold, dtype=np.float64),
                    (dt & 2) != 0,
                    ((dt >> 2) & 3) == 1,
                    np.asarray(t.left, dtype=np.int64),
                    np.asarray(t.right, dtype=np.int64),
                    np.asarray(t.leaf_value, dtype=np.float64),
                )
            )
        self._np_cache = out
        return out


@dataclass
class SelectorModel:
    arch: str
    ensemble: TreeEnsemble
    meta: Dict[str, Any]
    # the portfolio guard (sidecar ``guard.bags``): models fitted on bootstrap
    # resamples of the training cells; a switch away from the rule pick needs
    # every one of them to predict a gain (see select)
    bags: Tuple[TreeEnsemble, ...] = ()

    @property
    def features(self) -> List[str]:
        return list(self.meta["features"])

    @property
    def knows_skinny(self) -> bool:
        """Whether this model ranks gemm_skinny configurations (its feature
        version has the family features and its training data had the family)."""
        return self.meta.get("feature_version") in SKINNY_FEATURE_VERSIONS and bool(
            self.meta.get("families", {}).get("skinny", False)
        )

    def margin_for(self, cls: str) -> float:
        m = self.meta.get("margins", {})
        fitted = float(m.get(cls, m.get("default", SELECTOR_DEFAULT_MARGIN)))
        floor = SELECTOR_MIN_MARGIN.get(cls, SELECTOR_MIN_MARGIN.get("default", 0.0))
        return max(fitted, floor)

    def origin_margin(self, origin: str) -> float:
        """Extra margin for replacing the rule pick with a candidate of this
        origin class (``registered`` / ``neighbour`` / ``library``)."""
        m = self.meta.get("margins", {})
        return float(m.get("origin:" + origin_class(origin), 0.0))

    def _rows(
        self, dtype: str, M: int, N: int, K: int, flats: Sequence[Mapping[str, Any]]
    ) -> List[List[float]]:
        defaults = trait_defaults(self.meta)
        names = self.features
        return [
            feature_vector(
                names, self.arch, dtype, M, N, K, fl, trait_defaults=defaults
            )
            for fl in flats
        ]

    def score(
        self, dtype: str, M: int, N: int, K: int, flats: Sequence[Mapping[str, Any]]
    ) -> List[float]:
        return self.ensemble.predict(self._rows(dtype, M, N, K, flats))

    def bag_scores(
        self, dtype: str, M: int, N: int, K: int, flats: Sequence[Mapping[str, Any]]
    ) -> List[List[float]]:
        """Per guard bag, the scores of ``flats`` (empty: no guard shipped)."""
        if not self.bags:
            return []
        xs = self._rows(dtype, M, N, K, flats)
        return [b.predict(xs) for b in self.bags]

    @property
    def guard_min_gain(self) -> float:
        return float(self.meta.get("guard", {}).get("min_gain", 0.0))

    @property
    def guard_enabled(self) -> bool:
        """The sidecar ships the guard bags but can leave the guard off
        (``guard.enabled`` false: the bags stay loadable for the knob)."""
        return bool(self.bags) and bool(self.meta.get("guard", {}).get("enabled", True))


def origin_class(origin: str) -> str:
    if origin.startswith("library"):
        return "library"
    if origin.startswith("neighbour"):
        return "neighbour"
    if origin.startswith("form"):
        return "form"
    if origin.startswith("skinny"):
        return "skinny"
    return "registered"


def trait_defaults(meta: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Defaults of every trait field: this tree's TraitSpec, then the sidecar's
    values for fields this tree does not have (levers of other trees)."""
    from ...instances.common.gemm_universal import TraitSpec

    d: Dict[str, Any] = dict((meta or {}).get("trait_defaults", {}))
    d.update(dataclasses.asdict(TraitSpec()))
    return d


def _model_dir() -> Path:
    env = os.environ.get(MODEL_DIR_ENV, "").strip()
    return Path(env) if env else MODEL_DIR


# model kind -> (file stem, feature version)
MODEL_KINDS: Dict[str, Tuple[str, str]] = {
    "rcr": ("gemm_rcr", FEATURE_VERSION),
    "native": ("gemm_native", NATIVE_FEATURE_VERSION),
}


@lru_cache(maxsize=None)
def _load_model_cached(
    arch: str, model_dir: str, kind: str = "rcr"
) -> Optional[SelectorModel]:
    stem, version = MODEL_KINDS[kind]
    base = Path(model_dir) / f"{stem}_{arch}"
    txt, meta = base.with_suffix(".model.txt"), base.with_suffix(".meta.json")
    if not (txt.is_file() and meta.is_file()):
        return None
    m = json.loads(meta.read_text())
    ok_version = (
        m.get("feature_version") in SUPPORTED_FEATURE_VERSIONS
        if kind == "rcr"
        else m.get("feature_version") == version
    )
    if not ok_version or m.get("arch") != arch:
        return None
    ens = TreeEnsemble.from_text(txt.read_text())
    if ens.feature_names and ens.feature_names != list(m["features"]):
        raise ValueError(f"{txt}: feature names disagree with the sidecar")
    bags = []
    for nm in m.get("guard", {}).get("bags", ()):
        bp = Path(model_dir) / nm
        if not bp.is_file():
            raise ValueError(f"{bp}: guard bag named by the sidecar is missing")
        b = TreeEnsemble.from_text(bp.read_text())
        if b.feature_names and b.feature_names != list(m["features"]):
            raise ValueError(f"{bp}: feature names disagree with the sidecar")
        bags.append(b)
    return SelectorModel(arch=arch, ensemble=ens, meta=m, bags=tuple(bags))


def load_model(arch: str, kind: str = "rcr") -> Optional[SelectorModel]:
    """The arch's selector model of ``kind`` (``rcr`` / ``native``), or None
    (no such model shipped for it)."""
    return _load_model_cached(arch, str(_model_dir()), kind)


def _env_switch(name: str) -> Optional[bool]:
    v = os.environ.get(name, "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return False
    if v in ("1", "on", "true", "yes"):
        return True
    return None


def native_selector_enabled(arch: str = "") -> bool:
    """Native (NN / TN / TT) selection: off when ``ROCKE_GEMM_SELECTOR=0``;
    else ``ROCKE_GEMM_SELECTOR_NATIVE`` (0 / 1) if set; else the arch's native
    sidecar ``default_on``, else :data:`SELECTOR_NATIVE_DEFAULT_ON`."""
    if _env_switch(SELECTOR_ENV) is False:
        return False
    v = _env_switch(NATIVE_SELECTOR_ENV)
    if v is not None:
        return v
    if arch:
        m = load_model(arch, "native")
        if m is not None and not bool(m.meta.get("default_on", True)):
            return False
    return SELECTOR_NATIVE_DEFAULT_ON


def selector_enabled(arch: str = "") -> bool:
    """``ROCKE_GEMM_SELECTOR`` (0 / 1) if set; else the arch model's sidecar
    ``default_on`` (an arch whose model has not been validated on its GPU
    ships off), else :data:`SELECTOR_DEFAULT_ON`."""
    v = os.environ.get(SELECTOR_ENV, "").strip().lower()
    if v in ("0", "off", "false", "no"):
        return False
    if v in ("1", "on", "true", "yes"):
        return True
    if arch:
        m = load_model(arch)
        if m is not None and not bool(m.meta.get("default_on", True)):
            return False
    return SELECTOR_DEFAULT_ON


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SelectorCandidate:
    origin: str  # "registered" | "neighbour:<lever>=<v>" | "library"
    candidate_name: str  # registered candidate the spec came from ("" = library)
    spec: Any
    is_rule_pick: bool = False


def _fix_pads(spec: Any, M: int, N: int, K: int) -> Any:
    t, tr = spec.tile, spec.trait
    return replace(
        spec,
        trait=replace(
            tr,
            pad_m=bool(tr.pad_m or M % t.tile_m),
            pad_n=bool(tr.pad_n or N % t.tile_n),
            pad_k=bool(tr.pad_k or K % t.tile_k),
        ),
    )


def generated_spec_error(req: Any, spec: Any, parent: Any | None) -> Optional[str]:
    """Why a GENERATED spec (neighbour / library) is not offered, or None.

    The dispatcher's gates run first (:func:`_gate`). These rules keep the
    generated set inside launch contracts that the measured sweeps exercised
    with full-output checks, independent of what ``is_valid_spec`` admits:

    * M, N, K multiples of 8 (16-byte rows of the vector loads);
    * no ragged M / N except on a DTL parent with exact buffer bounds
      (``dtl_tight_rsrc``) whose tiling and pads the mutation keeps: a
      global-load body's ``pad_m`` / ``pad_n`` guards only the C stores, so
      rows past the operand would be read; a library spec must tile M and N
      exactly;
    * ``direct_to_lds`` needs whole K tiles (its DMA path ignores ``pad_k``);
    * ``split_k > 1`` only on global-load, non-persistent bodies with whole-tile
      K slices (``K % (split_k * tile_k) == 0``) and without ``pad_k``;
    * no stream-K and no persistent pool other than the parent's.
    """
    if not SELECTOR_SAFE_GATE:
        return None
    M, N, K = int(req.M), int(req.N), int(req.K)
    t, tr = spec.tile, spec.trait
    if M % 8 or N % 8 or K % 8:
        return "generated specs need M, N, K multiples of 8"
    ptile = parent.tile if parent is not None else None
    ptr = parent.trait if parent is not None else None
    for dim, tile, pad, pdim in (
        (M, t.tile_m, tr.pad_m, "m"),
        (N, t.tile_n, tr.pad_n, "n"),
    ):
        if dim % tile or pad:
            # only a DTL parent with exact buffer bounds reads no row past the
            # operand; keep its tiling and pads unchanged
            bounded = (
                (
                    bool(ptr.direct_to_lds)
                    and bool(ptr.dtl_tight_rsrc)
                    and bool(tr.direct_to_lds)
                    and bool(tr.dtl_tight_rsrc)
                )
                if ptr is not None
                else False
            )
            # trees with a row-clamp lever (``pad_row_clamp``: global-load rows
            # read min(row, extent - 1)) bound the global-load path too
            bounded = bounded or (
                ptr is not None
                and bool(getattr(ptr, "pad_row_clamp", False))
                and bool(getattr(tr, "pad_row_clamp", False))
                and not tr.direct_to_lds
            )
            same = (
                ptile is not None
                and getattr(ptile, f"tile_{pdim}") == tile
                and bool(getattr(ptr, f"pad_{pdim}")) == bool(pad)
                and bounded
            )
            if not same:
                return f"ragged {pdim.upper()} on a generated tiling"
    if tr.direct_to_lds and (K % t.tile_k or tr.pad_k):
        if not (
            ptile is not None and ptile.tile_k == t.tile_k and ptr.pad_k == tr.pad_k
        ):
            return "direct_to_lds with a partial K tile"
    if tr.split_k > 1:
        if tr.direct_to_lds or tr.persistent_ctas or tr.streamk:
            return "split_k only on plain global-load bodies"
        # whole-tile K slices, or a tree's aligned-slice lever
        # (``split_k_aligned``: slices of ceil(K / (split_k * tile_k)) tiles,
        # the last ending at K) inherited from the parent
        aligned = bool(getattr(tr, "split_k_aligned", False)) and bool(
            ptr is not None and getattr(ptr, "split_k_aligned", False)
        )
        if not aligned and (tr.pad_k or K % (tr.split_k * t.tile_k)):
            return "split_k slices must be whole K tiles"
    if tr.streamk and not (ptr is not None and ptr.streamk):
        return "stream-K is owned by the registered rules"
    if tr.persistent_ctas and not (
        ptr is not None and ptr.persistent_ctas == tr.persistent_ctas
    ):
        return "persistent pools are owned by the registered rules"
    return None


def _gate(req: Any, spec: Any) -> Optional[str]:
    from ...instances.common.gemm_universal import launch_shape_error
    from .support import request_shape_supported, universal_spec_supported

    try:
        ok, why = universal_spec_supported(spec, arch=req.arch)
        if not ok:
            return why
        ok, why = request_shape_supported(req, spec)
        if not ok:
            return why
        err = launch_shape_error(spec, int(req.M), int(req.N), int(req.K))
        if err is not None:
            return err
    except (
        Exception
    ) as exc:  # noqa: BLE001 - a spec the gates cannot evaluate is refused
        return f"{type(exc).__name__}: {exc}"
    return None


def _spec_key(spec: Any) -> str:
    if _skinny(spec):
        return "skinny:" + json.dumps(
            dataclasses.asdict(spec), sort_keys=True, default=str
        )
    return json.dumps(dataclasses.asdict(spec.tile), sort_keys=True) + json.dumps(
        dataclasses.asdict(spec.trait), sort_keys=True, default=str
    )


def skinny_variant_error(req: Any, spec: Any) -> Optional[str]:
    """Why a gemm_skinny configuration is not offered for ``req``, or None:
    the body's own static validity and launch contract
    (``is_valid_spec`` / ``skinny_launch_error``, the gates of the registered
    skinny candidates), and every split-K slice non-empty (the last slice
    starts inside K)."""
    from ...instances.common.gemm_skinny import (
        is_valid_spec as skinny_valid,
        skinny_kps,
        skinny_launch_error,
        skinny_steps,
    )

    try:
        ok, why = skinny_valid(spec, req.arch)
        if not ok:
            return why
        M, N, K = int(req.M), int(req.N), int(req.K)
        err = skinny_launch_error(spec, M, N, K, spec.split)
        if err is not None:
            return err
        steps = skinny_steps(spec, K)
        kps = skinny_kps(spec, K, spec.split)
        if (int(spec.split) - 1) * kps >= steps:
            return f"split={spec.split} leaves an empty K slice ({steps} steps)"
    except Exception as exc:  # noqa: BLE001 - a spec the gates cannot evaluate is refused
        return f"{type(exc).__name__}: {exc}"
    return None


def _skinny_variants(spec: Any) -> Iterable[Tuple[str, Any]]:
    """Generated gemm_skinny configurations of one registered configuration:
    every split of SELECTOR_SKINNY_SPLITS, and one-step lever mutations at
    the configuration's own split."""
    for s in SELECTOR_SKINNY_SPLITS:
        if s != spec.split:
            yield f"split={s}", replace(spec, split=s, red_unroll=8 if s >= 16 else 4)
    for name, values in SELECTOR_SKINNY_LEVERS:
        for v in values:
            if getattr(spec, name) != v:
                yield f"{name}={v}", replace(spec, **{name: v})


def _mutations(spec: Any, levers) -> Iterable[Tuple[str, Any]]:
    for name, values in levers:
        for v in values:
            if name in _TILE_FIELDS:
                if getattr(spec.tile, name) == v:
                    continue
                yield f"{name}={v}", replace(spec, tile=replace(spec.tile, **{name: v}))
                continue
            if not hasattr(spec.trait, name) or getattr(spec.trait, name) == v:
                continue
            if name in _DTL_ONLY_LEVERS and not spec.trait.direct_to_lds:
                continue
            tr = replace(spec.trait, **{name: v})
            if name in ("chiplet_wgm", "chiplet_chunk_size"):
                tr = replace(tr, chiplet_swizzle=True)
            yield f"{name}={v}", replace(spec, trait=tr)


def library_spec(
    arch: str, dtype: str, cfg: str, M: int, N: int, K: int
) -> Optional[Any]:
    """A library config as a spec for (M, N, K), pads derived; None if this
    tree lacks one of its trait fields."""
    from ...core.arch import ArchTarget
    from ...instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    tile, trait = parse_config(cfg)
    known = {f.name for f in dataclasses.fields(TraitSpec)}
    if any(k not in known for k in trait):
        return None
    dt = "fp16" if dtype in ("fp16", "f16") else "bf16"
    spec = UniversalGemmSpec(
        name="selector_library",
        tile=TileSpec(**tile),
        trait=TraitSpec(**trait),
        data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout="RCR"),
        wave_size=ArchTarget.from_gfx(arch).wave_size,
    )
    return _fix_pads(spec, M, N, K)


def selector_candidates(
    req: Any,
    registry: Any,
    *,
    model: Optional[SelectorModel] = None,
    sources: Optional[Sequence[str]] = None,
    rule_pick: Optional[Tuple[Any, Any]] = None,
) -> List[SelectorCandidate]:
    """The gated candidate set of ``req`` (see the module docstring); a
    non-RCR request gets :func:`native_candidates`."""
    if not _rcr(req):
        return native_candidates(
            req,
            registry,
            model=model,
            sources=sources if sources is not None else SELECTOR_NATIVE_SOURCES,
            rule_pick=rule_pick,
        )
    if sources is None:
        sources = SELECTOR_SOURCES
    model = model if model is not None else load_model(req.arch)
    meta = model.meta if model is not None else {}
    out: List[SelectorCandidate] = []
    seen: set = set()
    rule_key = _spec_key(rule_pick[1]) if rule_pick is not None else None

    def add(origin: str, cname: str, spec: Any) -> None:
        k = _spec_key(spec)
        if k in seen:
            return
        seen.add(k)
        out.append(SelectorCandidate(origin, cname, spec, k == rule_key))

    registered: List[Tuple[Any, Any]] = []
    if rule_pick is not None:
        registered.append(rule_pick)
    try:
        registered.extend(registry.combos(req, include_opt_in=False))
    except Exception:  # noqa: BLE001
        pass
    skinny_on = (
        SELECTOR_SKINNY
        and "skinny" in sources
        and model is not None
        and model.knows_skinny
    )
    for cand, spec in registered:
        if _universal(spec):
            if _gate(req, spec) is None:
                add("registered", cand.name, spec)
        elif skinny_on and _skinny(spec) and skinny_variant_error(req, spec) is None:
            add("registered", cand.name, spec)
    if skinny_on:
        for base in [c for c in out if _skinny(c.spec)]:
            for lab, spec in _skinny_variants(base.spec):
                if skinny_variant_error(req, spec) is None:
                    add(f"skinny:{lab}", base.candidate_name, spec)
    bases = [c for c in out if _universal(c.spec)]
    if "neighbour" in sources:
        allowed = meta.get("neighbour_levers")
        levers = [
            (n, tuple(v))
            for n, v in SELECTOR_NEIGHBOUR_LEVERS
            if allowed is None or n in allowed
        ]
        for base in bases:
            for lab, spec in _mutations(base.spec, levers):
                spec = _fix_pads(spec, int(req.M), int(req.N), int(req.K))
                if generated_spec_error(req, spec, base.spec) or _gate(req, spec):
                    continue
                add(f"neighbour:{lab}", base.candidate_name, spec)
    if "library" in sources:
        for cfg in meta.get("library", ()):
            try:
                spec = library_spec(
                    req.arch, req.dtype, cfg, int(req.M), int(req.N), int(req.K)
                )
            except Exception:  # noqa: BLE001
                continue
            if (
                spec is None
                or generated_spec_error(req, spec, None)
                or _gate(req, spec)
            ):
                continue
            add("library", "", spec)
    return out


def _rcr_view(req: Any) -> Any:
    """The RCR (NT) request of the same problem (``auto``)."""
    try:
        return replace(
            req,
            layout="RCR",
            trans_a=False,
            trans_b=True,
            algorithm="auto",
            spec_id="auto",
        )
    except TypeError:
        return replace(req, layout="RCR")


def _mn_ring(spec: Any) -> bool:
    try:
        from ...instances.common.gemm_mn_operand import (
            mn_layout_letters,
            uses_mn_dtl,
        )

        return any(mn_layout_letters(spec.data.layout)) and uses_mn_dtl(spec)
    except Exception:  # noqa: BLE001
        return False


_MN_RING_LEVERS = ("mn_chunk", "mn_pad")


def native_candidates(
    req: Any,
    registry: Any,
    *,
    model: Optional[SelectorModel] = None,
    sources: Sequence[str] = SELECTOR_NATIVE_SOURCES,
    rule_pick: Optional[Tuple[Any, Any]] = None,
) -> List[SelectorCandidate]:
    """The gated native candidate set of a non-RCR ``req``:

    * ``registered``: the rule pick and every registered candidate's native
      spec for ``req`` (with its sweep variants);
    * ``form:<name>``: every native form (``native_layout_alternatives``) of
      every registered candidate's RCR spec for the same problem: the
      pre-selector dispatcher only ever served the FIRST launchable form of
      the priority pick;
    * ``neighbour:<lever>=<v>``: one-step mutations of both over
      :data:`SELECTOR_NEIGHBOUR_LEVERS` + :data:`SELECTOR_NATIVE_NEIGHBOUR_LEVERS`
      (the MN block-layout levers only on specs whose MN operand uses the
      DTL ring), through :func:`generated_spec_error`.

    Every spec passes the dispatcher's gates (:func:`_gate`: validity, the
    request-shape and launch rules, incl. the MN leading-dimension rules)."""
    from .common import native_layout_alternatives

    model = model if model is not None else load_model(req.arch, "native")
    meta = model.meta if model is not None else {}
    lay = str(req.layout).upper()
    out: List[SelectorCandidate] = []
    seen: set = set()
    rule_key = _spec_key(rule_pick[1]) if rule_pick is not None else None

    def add(origin: str, cname: str, spec: Any) -> None:
        if str(getattr(spec.data, "layout", "")).upper() != lay:
            return
        k = _spec_key(spec)
        if k in seen:
            return
        if _gate(req, spec) is not None:
            return
        seen.add(k)
        out.append(SelectorCandidate(origin, cname, spec, k == rule_key))

    registered: List[Tuple[Any, Any]] = []
    if rule_pick is not None:
        registered.append(rule_pick)
    try:
        registered.extend(registry.combos(req, include_opt_in=False))
    except Exception:  # noqa: BLE001
        pass
    for cand, spec in registered:
        if _universal(spec):
            add("registered", cand.name, spec)
    if "form" in sources:
        try:
            rcombos = registry.combos(_rcr_view(req), include_opt_in=False)
        except Exception:  # noqa: BLE001
            rcombos = ()
        for cand, spec in rcombos:
            if not _universal(spec):
                continue
            try:
                alts, _why = native_layout_alternatives(req, spec)
            except Exception:  # noqa: BLE001
                continue
            for nm, s in alts:
                add(f"form:{nm}", cand.name, s)
    bases = list(out)
    nb = int(SELECTOR_NATIVE_NEIGHBOUR_BASES)
    if "neighbour" in sources and model is not None and nb > 0 and len(bases) > nb:
        dtype = "bf16" if req.dtype in ("bf16", "bfloat16") else "fp16"
        sc = model.score(
            dtype,
            int(req.M),
            int(req.N),
            int(req.K),
            [flatten_spec(c.spec) for c in bases],
        )
        order = sorted(range(len(bases)), key=lambda i: (sc[i], i))
        keep = set(order[:nb])
        bases = [c for i, c in enumerate(bases) if i in keep or c.is_rule_pick]
    if "neighbour" in sources:
        allowed = meta.get("neighbour_levers")
        levers = [
            (n, tuple(v))
            for n, v in tuple(SELECTOR_NEIGHBOUR_LEVERS)
            + tuple(SELECTOR_NATIVE_NEIGHBOUR_LEVERS)
            if allowed is None or n in allowed
        ]
        M, N, K = int(req.M), int(req.N), int(req.K)
        for base in bases:
            ring = _mn_ring(base.spec)
            lv = [(n, v) for n, v in levers if ring or n not in _MN_RING_LEVERS]
            for lab, spec in _mutations(base.spec, lv):
                spec = _fix_pads(spec, M, N, K)
                if generated_spec_error(req, spec, base.spec):
                    continue
                add(f"neighbour:{lab}", base.candidate_name, spec)
    return out


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SelectorDecision:
    pick: SelectorCandidate
    predicted: float  # model output of the pick (log time / roofline)
    rule_pick: Optional[SelectorCandidate]
    rule_predicted: Optional[float]
    ranked: Tuple[Tuple[float, SelectorCandidate], ...] = field(default=())
    reason: str = ""


def spec_class(flat: Mapping[str, Any]) -> str:
    """Decision class of a rule pick (the key of the sidecar's ``margins``):
    the body family from its trait flags. Training uses the same function."""
    if flat.get("family") == "skinny":
        return "skinny"
    if flat.get("direct_to_lds"):
        if flat.get("hb_sched") or flat.get("sia_sched"):
            return "dtl:beat"
        if flat.get("persistent_ctas"):
            return "dtl:persist"
        return "dtl"
    if int(_enc(flat.get("split_k", 1))) > 1:
        return "global:splitk"
    if flat.get("pipeline") == "mem":
        return "global:mem"
    return "global"


def model_kind(req: Any) -> str:
    """``rcr`` or ``native``: the model kind that ranks ``req``."""
    return "rcr" if _rcr(req) else "native"


def rank(
    req: Any,
    registry: Any,
    *,
    model: Optional[SelectorModel] = None,
    sources: Optional[Sequence[str]] = None,
    rule_pick: Optional[Tuple[Any, Any]] = None,
) -> List[Tuple[float, SelectorCandidate]]:
    """Every gated candidate with its predicted score, best (lowest) first."""
    model = model if model is not None else load_model(req.arch, model_kind(req))
    if model is None:
        return []
    cands = selector_candidates(
        req, registry, model=model, sources=sources, rule_pick=rule_pick
    )
    if not cands:
        return []
    dtype = "bf16" if req.dtype in ("bf16", "bfloat16") else "fp16"
    scores = model.score(
        dtype, int(req.M), int(req.N), int(req.K), [flatten_spec(c.spec) for c in cands]
    )
    order = sorted(range(len(cands)), key=lambda i: (scores[i], i))
    return [(scores[i], cands[i]) for i in order]


def select(
    req: Any,
    registry: Any,
    *,
    rule_pick: Optional[Tuple[Any, Any]] = None,
    model: Optional[SelectorModel] = None,
    sources: Optional[Sequence[str]] = None,
) -> Optional[SelectorDecision]:
    """The selector's decision for ``req``, or None (no model / no candidate)."""
    model = model if model is not None else load_model(req.arch, model_kind(req))
    if model is None:
        return None
    ranked = rank(req, registry, model=model, sources=sources, rule_pick=rule_pick)
    if not ranked:
        return None
    best_s, best = ranked[0]
    rp = next(((s, c) for s, c in ranked if c.is_rule_pick), None)
    if rp is None:
        return SelectorDecision(best, best_s, None, None, tuple(ranked), "no rule pick")
    rule_s, rule_c = rp
    cls = spec_class(flatten_spec(rule_c.spec))
    margin = max(model.margin_for(cls), model.origin_margin(best.origin))
    if best is rule_c or rule_s - best_s <= margin:
        return SelectorDecision(
            rule_c,
            rule_s,
            rule_c,
            rule_s,
            tuple(ranked),
            f"keep rule pick (class {cls}, margin {margin})",
        )
    if SELECTOR_PORTFOLIO_GUARD and model.guard_enabled:
        dtype = "bf16" if req.dtype in ("bf16", "bfloat16") else "fp16"
        bs = model.bag_scores(
            dtype,
            int(req.M),
            int(req.N),
            int(req.K),
            [flatten_spec(rule_c.spec), flatten_spec(best.spec)],
        )
        gains = [b[0] - b[1] for b in bs]
        if min(gains) <= model.guard_min_gain:
            return SelectorDecision(
                rule_c,
                rule_s,
                rule_c,
                rule_s,
                tuple(ranked),
                f"keep rule pick (portfolio guard: bag gains "
                f"{', '.join(f'{g:.4f}' for g in gains)}; predicted gain "
                f"{rule_s - best_s:.4f} > margin {margin} ({cls}))",
            )
    return SelectorDecision(
        best,
        best_s,
        rule_c,
        rule_s,
        tuple(ranked),
        f"predicted gain {rule_s - best_s:.4f} > margin {margin} ({cls})",
    )


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------
# Specs the opt-in "selected" candidate exposes through sweep_space / dispatch_*_all
# (the selector's top-k: the autotune-lite shortlist).
SELECTOR_SWEEP_TOPK = 6
_DECISION_CACHE: Dict[Tuple[Any, ...], Optional[SelectorDecision]] = {}
_DECISION_CACHE_MAX = 4096


def _universal(spec: Any) -> bool:
    from ...instances.common.gemm_universal import UniversalGemmSpec

    return isinstance(spec, UniversalGemmSpec)


def _skinny(spec: Any) -> bool:
    from ...instances.common.gemm_skinny import SkinnyGemmSpec

    return isinstance(spec, SkinnyGemmSpec)


def _rcr(req: Any) -> bool:
    return str(getattr(req, "layout", "RCR")).upper() == "RCR"


def _pinned(req: Any) -> bool:
    for f in ("algorithm", "spec_id"):
        v = getattr(req, f, "auto")
        if isinstance(v, str) and v.strip().lower() not in ("", "auto"):
            return True
    return False


def _auto(req: Any) -> Any:
    try:
        return replace(req, algorithm="auto", spec_id="auto")
    except TypeError:
        return req


def _cache_key(req: Any, registry: Any) -> Tuple[Any, ...]:
    return (
        getattr(registry, "family", id(registry)),
        json.dumps(_auto(req).normalized(), sort_keys=True),
        str(_model_dir()),
        SELECTOR_SAFE_GATE,
        tuple(SELECTOR_SOURCES),
        tuple(SELECTOR_NATIVE_SOURCES),
        tuple(SELECTOR_NATIVE_NEIGHBOUR_LEVERS),
        SELECTOR_NATIVE_NEIGHBOUR_BASES,
        native_selector_enabled(str(getattr(req, "arch", ""))),
        SELECTOR_SKINNY,
        SELECTOR_PORTFOLIO_GUARD,
    )


def selector_on_for(req: Any) -> bool:
    """Whether the learned selection applies to ``req``'s layout: the RCR
    switch (:func:`selector_enabled`) for RCR, :func:`native_selector_enabled`
    otherwise."""
    arch = str(getattr(req, "arch", ""))
    if _rcr(req):
        return selector_enabled(arch)
    return native_selector_enabled(arch)


def decide(req: Any, registry: Any) -> Optional[SelectorDecision]:
    """Cached :func:`select` for an ``auto`` view of ``req`` with the
    registry's priority pick as the rule pick (None: no model / no candidate /
    the registry refuses the request)."""
    key = _cache_key(req, registry)
    if key in _DECISION_CACHE:
        return _DECISION_CACHE[key]
    areq = _auto(req)
    kind = model_kind(areq)
    model = load_model(getattr(areq, "arch", ""), kind)
    if kind == "native" and not native_selector_enabled(getattr(areq, "arch", "")):
        model = None  # native requests keep the rule order
    dec: Optional[SelectorDecision] = None
    if model is not None:
        try:
            cand = registry.select(areq)
            rule = (cand, cand.select_spec(areq))
        except (
            Exception
        ):  # noqa: BLE001 - refused by the rules: the selector does not widen coverage
            rule = None
        # the RCR model knows UniversalGemm specs (and, from r2c-v2 on,
        # gemm_skinny configurations); the native model (NN / TN / TT)
        # UniversalGemm specs only: a rule pick of a family the model does not
        # know keeps the rule pick
        if rule is None:
            known = False
        elif kind == "native":
            known = _universal(rule[1])
        else:
            known = _universal(rule[1]) or (
                _skinny(rule[1]) and SELECTOR_SKINNY and model.knows_skinny
            )
        if known:
            dec = select(areq, registry, rule_pick=rule, model=model)
    if len(_DECISION_CACHE) >= _DECISION_CACHE_MAX:
        _DECISION_CACHE.clear()
    _DECISION_CACHE[key] = dec
    return dec


def clear_cache() -> None:
    _DECISION_CACHE.clear()
    _load_model_cached.cache_clear()


def dispatch_override(
    req: Any,
    registry: Any,
    candidate: Any,
    spec: Any,
    *,
    selected_name: str,
    selected_skinny_name: str = "",
) -> Tuple[Any, Any, Tuple[str, ...]]:
    """``(candidate, spec, explanation)`` after the selector: the rule pick
    unchanged unless the selector is on, the request is ``auto``, the arch has
    a model, and the predicted gain clears the class margin. A registered
    candidate's own spec keeps that candidate; any other spec is served by the
    family's opt-in ``selected`` candidate (a gemm_skinny spec by the
    ``selected_skinny_name`` one), so pinning its ``spec_id`` reselects it."""
    if not selector_on_for(req) or _pinned(req):
        return candidate, spec, ()
    dec = decide(req, registry)
    if dec is None:
        return candidate, spec, ("selector: no model or no candidate",)
    if dec.pick.is_rule_pick:
        return candidate, spec, (f"selector: {dec.reason}",)
    why = (
        f"selector: {dec.reason}; origin {dec.pick.origin} "
        f"(from {dec.pick.candidate_name or 'library'})",
    )
    pk = _spec_key(dec.pick.spec)
    for c in registry.candidates():
        if c.name == dec.pick.candidate_name and dec.pick.origin == "registered":
            try:
                own = c.select_spec(_auto(req))
            except Exception:  # noqa: BLE001
                break
            if _spec_key(own) == pk:
                return c, own, why
            break
    target = selected_skinny_name if _skinny(dec.pick.spec) else selected_name
    for c in registry.candidates():
        if target and c.name == target:
            return c, dec.pick.spec, why
    return candidate, spec, ("selector: no selected candidate registered",)


def _family_ok(spec: Any, family: str) -> bool:
    return _skinny(spec) if family == "skinny" else _universal(spec)


def selected_supports(
    req: Any, registry: Any, family: str = "universal"
) -> Tuple[bool, str]:
    """Whether the opt-in selected candidate of ``family`` (``universal`` /
    ``skinny``) serves ``req``: the selector is on and its pick is a non-rule
    spec of that family."""
    # off = the rule layer alone: the selected candidate serves nothing, so
    # dispatch_*_all / sweep_space are exactly the pre-selector sets
    if not selector_on_for(req):
        return (
            False,
            f"selector off ({SELECTOR_ENV} / {NATIVE_SELECTOR_ENV} / the model's "
            "default_on)",
        )
    dec = decide(req, registry)
    if dec is None:
        return (
            False,
            "selector: no model, no candidate, or the rules refuse the request",
        )
    if dec.pick.is_rule_pick:
        return False, f"selector keeps the rule pick ({dec.reason})"
    if not _family_ok(dec.pick.spec, family):
        return False, f"selector pick is not a {family} spec"
    return True, "ok"


def selected_spec(req: Any, registry: Any, family: str = "universal") -> Any:
    dec = decide(req, registry)
    if dec is None or dec.pick.is_rule_pick or not _family_ok(dec.pick.spec, family):
        raise ValueError(f"selector has no non-rule {family} pick for this request")
    return dec.pick.spec


def selected_sweep(
    req: Any, registry: Any, k: int = SELECTOR_SWEEP_TOPK, family: str = "universal"
) -> Tuple[Any, ...]:
    """The selector's top-k specs of ``family`` (autotune-lite shortlist; rule
    pick first if it is in the top-k)."""
    dec = decide(req, registry)
    if dec is None:
        return ()
    return tuple(
        c.spec for _, c in diverse_top(dec.ranked, k) if _family_ok(c.spec, family)
    )


def diverse_top(
    ranked: Sequence[Tuple[float, SelectorCandidate]], k: int
) -> List[Tuple[float, SelectorCandidate]]:
    """The best ``k`` of ``ranked`` with distinct predicted scores: specs the
    model cannot tell apart (a lever it never splits on) count once, so a
    re-timed shortlist spends its slots on different predictions."""
    out: List[Tuple[float, SelectorCandidate]] = []
    for sc, c in ranked:
        if any(abs(sc - o) < 1e-9 for o, _ in out):
            continue
        out.append((sc, c))
        if len(out) >= max(1, int(k)):
            break
    return out
