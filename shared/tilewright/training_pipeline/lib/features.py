# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generic feature catalog of the per-cell recommender.

"Generic" because every input is backend-agnostic GEMM/kernel metadata:

  problem   M, N, K, batch, transA, transB, dtypes (a/b/c/d/mi)
  config    MT (m/n/k), MI (m/n/k), occupancy, cache_hints_a/b,
            grvw_a/b, gwvw_d
  hardware  `DeviceHardware` (n_cu, lds_bytes, l2_bytes) and the model's
            `ArchConstants` (MI cycle table, parallel_mi_cu, bandwidth
            coefficients); see lib/hardware.py

The engine computes the same features (`tilewright.compute_features`); the
formulas, the order (`feature_names()`) and the names are part of every model
file through `feature_names_hash()`, so none of them may change without
retraining. Arithmetic is float64; consumers cast the vectors to float32.

Catalog design:
  1. No feature that is an exact linear combination of other features.
  2. No hand-tuned `_proxy` approximations of codegen physics.
  3. No feature that is constant over the workload.
  4. No duplicate scalings of the same input.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .hardware import ArchConstants, DeviceHardware

FEATURES_VERSION = "pipeline-v3-2026-06-drop-config-ml"


# ── dtype tables ─────────────────────────────────────────────────────────────

# Element sizes and ids as the engine computes them (unlisted dtypes: 2 bytes,
# id 0); training features must equal the runtime's.
_BPE: Dict[str, float] = {
    "float": 4.0,
    "f32": 4.0,
    "xfloat32": 4.0,
    "xf32": 4.0,
    "double": 8.0,
    "f64": 8.0,
    "half": 2.0,
    "f16": 2.0,
    "bfloat16": 2.0,
    "bf16": 2.0,
    "float8": 1.0,
    "f8": 1.0,
    "float8_fnuz": 1.0,
    "bfloat8": 1.0,
    "bf8": 1.0,
    "bfloat8_fnuz": 1.0,
    "float8bfloat8": 1.0,
    "bfloat8float8": 1.0,
    "float8bfloat8_fnuz": 1.0,
    "bfloat8float8_fnuz": 1.0,
    "float6": 0.75,
    "bfloat6": 0.75,
    "f6": 0.75,
    "bf6": 0.75,
    "float4": 0.5,
    "f4": 0.5,
    "int8": 1.0,
    "int32": 4.0,
    "int4": 0.5,
}

_DTYPE_ALIAS_EXPAND = {
    "bf16": "bfloat16",
    "f16": "half",
    "fp16": "half",
    "bf8": "bfloat8",
    "f8": "float8",
    "fp8": "float8",
    "bf6": "bfloat6",
    "f6": "float6",
    "fp6": "float6",
    "f4": "float4",
    "fp4": "float4",
    "f32": "float",
    "fp32": "float",
    "xf32": "xfloat32",
    "f64": "double",
    "fp64": "double",
}

_DTYPE_ID: Dict[str, int] = {
    "float": 0,
    "f32": 0,
    "xfloat32": 1,
    "xf32": 1,
    "half": 2,
    "f16": 2,
    "bfloat16": 3,
    "bf16": 3,
    "float8": 4,
    "f8": 4,
    "bfloat8": 5,
    "bf8": 5,
    "float6": 6,
    "bfloat6": 7,
    "float4": 8,
    "int8": 9,
    "int32": 10,
    "float8_fnuz": 4,
    "bfloat8_fnuz": 5,
    "float8bfloat8": 11,
    "bfloat8float8": 12,
    "float8bfloat8_fnuz": 11,
    "bfloat8float8_fnuz": 12,
    "double": 13,
    "int4": 14,
}

# The MI cycle tables are keyed by the OCP fp8 names; fnuz problems use the
# same instructions.
_FNUZ_PLAIN = {
    "float8_fnuz": "float8",
    "bfloat8_fnuz": "bfloat8",
    "float8bfloat8_fnuz": "float8bfloat8",
    "bfloat8float8_fnuz": "bfloat8float8",
}

# hipBLASLt type strings (CSV `a_type`, ...) -> catalog dtype names.
HIPBLASLT_DTYPE: Dict[str, str] = {
    "f32_r": "float",
    "c_f32_r": "float",
    "xf32_r": "xfloat32",
    "c_xf32_r": "xfloat32",
    "f64_r": "double",
    "f16_r": "half",
    "bf16_r": "bfloat16",
    "f8_r": "float8",
    "bf8_r": "bfloat8",
    "f8_fnuz_r": "float8_fnuz",
    "bf8_fnuz_r": "bfloat8_fnuz",
    "f6_r": "float6",
    "bf6_r": "bfloat6",
    "f4_r": "float4",
    "i8_r": "int8",
    "i32_r": "int32",
}


def normalize_dtype_name(dtype: Any) -> str:
    if dtype is None:
        return "bfloat16"
    s = str(dtype)
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    s = s.lower()
    return _DTYPE_ALIAS_EXPAND.get(s, s)


def bpe_for_dtype(dtype: Any) -> float:
    return _BPE.get(normalize_dtype_name(dtype), 2.0)


def _dtype_id(dtype: Any) -> int:
    return _DTYPE_ID.get(normalize_dtype_name(dtype), 0)


def _transpose_bit(x: Any) -> int:
    s = str(x).upper()
    return 1 if s.endswith("T") or s == "T" else 0


def dtype_from_hipblaslt(t: Any) -> str:
    """Catalog dtype name of a hipBLASLt type string (`bf16_r`, ...)."""
    key = str(t or "").strip().lower()
    if key not in HIPBLASLT_DTYPE:
        raise ValueError(f"unknown hipBLASLt data type {t!r}")
    return HIPBLASLT_DTYPE[key]


def problem_kwargs_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """`compute_generic_features` problem kwargs of an enriched-CSV GEMM.

    `mi_dtype` is the matrix-instruction input dtype: the A dtype, except for
    xf32 compute (emulated tf32), the same rule TensileLite applies when it
    builds the runtime problem."""
    compute_type = str(row.get("compute_type") or "").strip().lower()
    a_dtype = dtype_from_hipblaslt(row["a_type"])
    is_xf32 = "xf32" in compute_type or "xfloat32" in compute_type
    return {
        "m": int(row["m"]),
        "n": int(row["n"]),
        "k": int(row["k"]),
        "batch": int(row["batch_count"]),
        "a_transpose": row["transA"],
        "b_transpose": row["transB"],
        "a_dtype": a_dtype,
        "b_dtype": dtype_from_hipblaslt(row["b_type"]),
        "c_dtype": dtype_from_hipblaslt(row["c_type"]),
        "d_dtype": dtype_from_hipblaslt(row["d_type"]),
        "mi_dtype": "xfloat32" if is_xf32 else a_dtype,
    }


def config_kwargs_from_row(row: Mapping[str, Any]) -> Dict[str, int]:
    """`compute_generic_features` config kwargs of one candidate row."""

    def _i(k: str, d: int) -> int:
        v = row.get(k, "")
        if v == "" or v is None:
            return d
        try:
            return int(v)
        except (TypeError, ValueError):
            return d

    return {
        "mt_m": _i("mt_m", 1),
        "mt_n": _i("mt_n", 1),
        "mt_k": _i("mt_k", 1),
        "mi_m": _i("mi_m", 1),
        "mi_n": _i("mi_n", 1),
        "mi_k": _i("mi_k", 1),
        "occupancy": max(_i("occupancy", 1), 1),
        "cache_hints_a": _i("cache_hints_a", 0),
        "cache_hints_b": _i("cache_hints_b", 0),
        "grvw_a": max(_i("grvw_a", 1), 1),
        "grvw_b": max(_i("grvw_b", 1), 1),
        "gwvw_d": max(_i("gwvw_d", 1), 1),
    }


# ── MI latency ───────────────────────────────────────────────────────────────


def get_mi_latency(
    mi_m: int, mi_n: int, mi_k: int, mi_dtype: Any, constants: ArchConstants
) -> float:
    """Instruction cycles from the model's table divided by parallel_mi_cu."""
    dt = normalize_dtype_name(mi_dtype)
    dt = _FNUZ_PLAIN.get(dt, dt)
    raw = constants.mi_table.get((int(mi_m), int(mi_n), int(mi_k), dt))
    if raw is None:
        raw = constants.mi_default
    return raw / max(constants.parallel_mi_cu, 1)


# ── feature computation ──────────────────────────────────────────────────────


def compute_generic_features(
    m: int,
    n: int,
    k: int,
    batch: int = 1,
    a_transpose: Any = "N",
    b_transpose: Any = "N",
    a_dtype: Any = "bf16",
    b_dtype: Any = "bf16",
    c_dtype: Any = "bf16",
    d_dtype: Any = "bf16",
    mi_dtype: Any = "bf16",
    mt_m: int = 1,
    mt_n: int = 1,
    mt_k: int = 1,
    mi_m: int = 1,
    mi_n: int = 1,
    mi_k: int = 1,
    occupancy: int = 1,
    cache_hints_a: int = 0,
    cache_hints_b: int = 0,
    grvw_a: int = 1,
    grvw_b: int = 1,
    gwvw_d: int = 1,
    *,
    hardware: DeviceHardware,
    constants: ArchConstants,
    as_dict: bool = False,
) -> Any:
    """Compute the feature vector of one (problem, config) pair.

    Returns an ordered {name: value} dict when `as_dict`, else the values in
    `feature_names()` order. The `max(x, 1)` floors on tile / MI / vector
    widths only guard divisions and log2() against zero inputs."""
    N_CU: int = hardware.n_cu
    LDS: int = hardware.lds_bytes
    L2: int = hardware.l2_bytes
    mem_bw_coef = constants.bw

    m_f, n_f, k_f, b_f = float(m), float(n), float(k), float(batch)
    mt_mf = float(max(mt_m, 1))
    mt_nf = float(max(mt_n, 1))
    mt_kf = float(max(mt_k, 1))
    mi_mf = float(max(mi_m, 1))
    mi_nf = float(max(mi_n, 1))
    mi_kf = float(max(mi_k, 1))
    occ_f = float(max(occupancy, 1))
    grvw_af = float(max(grvw_a, 1))
    grvw_bf = float(max(grvw_b, 1))
    gwvw_df = float(max(gwvw_d, 1))

    bpe_a = bpe_for_dtype(a_dtype)
    bpe_b = bpe_for_dtype(b_dtype)
    bpe_c = bpe_for_dtype(c_dtype)
    bpe_d = bpe_for_dtype(d_dtype)
    nta = int(cache_hints_a)
    ntb = int(cache_hints_b)

    # xf32 runs as three bf16 instruction passes, so every FLOP-derived
    # intensity counts 3x the multiply-adds.
    flop_mult = 3.0 if normalize_dtype_name(mi_dtype) == "xfloat32" else 1.0

    # ── problem-side derived ────────────────────────────────────────────────
    mn = m_f * n_f
    mk = m_f * k_f
    nk = n_f * k_f
    total_flops = flop_mult * 2.0 * m_f * n_f * k_f * b_f
    a_bytes = mk * bpe_a * b_f
    b_bytes = nk * bpe_b * b_f
    c_bytes = mn * bpe_c * b_f
    d_bytes = mn * bpe_d * b_f
    total_bytes = a_bytes + b_bytes + c_bytes + d_bytes
    ai_prob = total_flops / max(total_bytes, 1.0)

    # ── tile / grid / utilization ──────────────────────────────────────────
    nt_m = math.ceil(m_f / mt_mf)
    nt_n = math.ceil(n_f / mt_nf)
    num_tiles = nt_m * nt_n
    num_tiles_total = num_tiles * b_f
    k_iters = math.ceil(k_f / mt_kf)
    tile_coverage = (mt_mf * mt_nf) / max(mn, 1.0)

    waves = math.ceil(num_tiles / N_CU)
    wave_eff = num_tiles / (waves * N_CU) if waves > 0 else 1.0
    rho = num_tiles_total / N_CU
    batch_tiles_ratio = b_f * num_tiles / N_CU

    launched_m = nt_m * mt_mf
    launched_n = nt_n * mt_nf
    launched_k = k_iters * mt_kf
    util_out = mn / max(launched_m * launched_n, 1.0)
    util_3d = (m_f * n_f * k_f) / max(launched_m * launched_n * launched_k, 1.0)

    # ── memory / capacity ──────────────────────────────────────────────────
    lds_bytes = mt_mf * mt_kf * bpe_a + mt_nf * mt_kf * bpe_b
    lds_ratio = lds_bytes / LDS
    l2_fit_ratio = total_bytes / L2
    bw_per_cu = total_bytes / N_CU
    l2_working_set = (nt_m * mt_mf * mt_kf * bpe_a) + (nt_n * mt_nf * mt_kf * bpe_b)
    l2_fit_ws = min(l2_working_set / L2, 2.0) / 2.0

    # ── compute latency / tile arithmetic intensity ────────────────────────
    L_MI = get_mi_latency(int(mi_mf), int(mi_nf), int(mi_kf), mi_dtype, constants)
    n_mi = (
        math.ceil(mt_mf / mi_mf) * math.ceil(mt_nf / mi_nf) * math.ceil(mt_kf / mi_kf)
    )
    L_MT = n_mi * L_MI
    ai_tile = (flop_mult * 2.0 * mt_mf * mt_nf * mt_kf) / (
        mt_mf * mt_kf + mt_nf * mt_kf + mt_mf * mt_nf
    )
    active_cus = min(num_tiles_total, float(N_CU))
    bw_occ = min(
        1.0,
        mem_bw_coef[0] * active_cus * active_cus
        + mem_bw_coef[1] * active_cus
        + mem_bw_coef[2],
    )

    # ── assemble features ──────────────────────────────────────────────────
    feats: "OrderedDict[str, float]" = OrderedDict()

    feats["log2_m"] = math.log2(max(m_f, 1.0))
    feats["log2_n"] = math.log2(max(n_f, 1.0))
    feats["log2_k"] = math.log2(max(k_f, 1.0))
    feats["log2_b"] = math.log2(max(b_f, 1.0))
    feats["ai_prob"] = min(ai_prob, 1e6)
    feats["log2_ai_prob"] = math.log2(max(ai_prob, 0.001))
    feats["aspect_mn"] = min(max(m_f / max(n_f, 1.0), 0.001), 10000.0)
    feats["aspect_mk"] = min(max(m_f / max(k_f, 1.0), 0.001), 10000.0)
    feats["aspect_nk"] = min(max(n_f / max(k_f, 1.0), 0.001), 10000.0)

    def _is_pow2(x: float) -> float:
        xi = int(x)
        return 1.0 if xi > 0 and (xi & (xi - 1)) == 0 else 0.0

    feats["is_pow2_m"] = _is_pow2(m_f)
    feats["is_pow2_n"] = _is_pow2(n_f)
    feats["is_pow2_k"] = _is_pow2(k_f)

    feats["trans_a"] = float(_transpose_bit(a_transpose))
    feats["trans_b"] = float(_transpose_bit(b_transpose))
    feats["dtype_a_id"] = float(_dtype_id(a_dtype))
    feats["dtype_b_id"] = float(_dtype_id(b_dtype))
    feats["dtype_c_id"] = float(_dtype_id(c_dtype))
    feats["dtype_d_id"] = float(_dtype_id(d_dtype))
    feats["dtype_mi_id"] = float(_dtype_id(mi_dtype))
    feats["bpe_a"] = bpe_a
    feats["bpe_b"] = bpe_b

    for base in (256, 128, 64, 32, 16, 8):
        feats[f"m_mod_{base}"] = float(int(m_f) % base)
        feats[f"n_mod_{base}"] = float(int(n_f) % base)
    for base in (64, 128, 256):
        feats[f"m_align_{base}"] = min(int(m_f) % base, base - (int(m_f) % base)) / base
        feats[f"n_align_{base}"] = min(int(n_f) % base, base - (int(n_f) % base)) / base

    # Problem-only proxies of the tile geometry at fixed candidate tile sizes,
    # usable by the query tower without knowing the config.
    for st in (32, 64, 128, 256):
        s_nt_m = math.ceil(m_f / st)
        s_nt_n = math.ceil(n_f / st)
        feats[f"log2_enum_tiles_{st}"] = math.log2(max(s_nt_m * s_nt_n, 1))
        feats[f"enum_coverage_{st}"] = mn / max(s_nt_m * st * s_nt_n * st, 1.0)
    for kd in (32, 64, 128, 256):
        feats[f"log2_enum_k_iters_{kd}"] = math.log2(max(math.ceil(k_f / kd), 1))
    for st in (128, 256):
        s_nt = math.ceil(m_f / st) * math.ceil(n_f / st)
        s_w = math.ceil(s_nt / N_CU)
        feats[f"log2_enum_waves_{st}"] = math.log2(max(s_w, 1))
        feats[f"enum_wave_eff_{st}"] = s_nt / max(s_w * N_CU, 1.0)

    feats["log2_mt_m"] = math.log2(mt_mf)
    feats["log2_mt_n"] = math.log2(mt_nf)
    feats["log2_mt_k"] = math.log2(mt_kf)
    feats["log2_mi_m"] = math.log2(mi_mf)
    feats["log2_mi_n"] = math.log2(mi_nf)
    feats["log2_mi_k"] = math.log2(mi_kf)
    feats["nta_norm"] = nta / 7.0
    feats["ntb_norm"] = ntb / 7.0
    feats["occupancy_norm"] = occ_f / 9.0
    feats["grvw_a_norm"] = grvw_af / 8.0
    feats["grvw_b_norm"] = grvw_bf / 8.0
    feats["gwvw_d_norm"] = gwvw_df / 8.0

    feats["log2_num_tiles"] = math.log2(max(num_tiles, 1))
    feats["log2_num_tiles_total"] = math.log2(max(num_tiles_total, 1))
    feats["log2_k_iters"] = math.log2(max(k_iters, 1))
    feats["log2_waves"] = math.log2(max(waves, 1))
    feats["wave_eff"] = wave_eff
    feats["log2_rho"] = math.log2(max(rho, 0.001))
    feats["log2_batch_tiles_ratio"] = math.log2(max(batch_tiles_ratio, 0.001))
    feats["util_out"] = util_out
    feats["util_3d"] = util_3d
    feats["tile_coverage"] = min(tile_coverage, 1.0)
    feats["log2_lds_bytes"] = math.log2(max(lds_bytes, 1.0))
    feats["lds_ratio"] = lds_ratio
    feats["l2_fit_ratio"] = min(l2_fit_ratio, 4.0) / 4.0
    feats["l2_fit_ws"] = l2_fit_ws
    feats["log2_bw_per_cu"] = math.log2(max(bw_per_cu, 1.0))
    feats["log2_total_bytes"] = math.log2(max(total_bytes, 1.0))
    feats["log2_total_flops"] = math.log2(max(total_flops, 1.0))
    feats["m_coverage"] = min(1.0, mt_mf / max(m_f, 1.0))
    feats["n_coverage"] = min(1.0, mt_nf / max(n_f, 1.0))
    feats["k_tail"] = (k_f - (k_iters - 1) * mt_kf) / max(k_f, 1.0)
    feats["k_aligned_128B"] = (
        1.0 if ((k_f * bpe_a) % 128.0 == 0 and (mt_kf * bpe_a) % 128.0 == 0) else 0.0
    )
    feats["small_m_tile"] = 1.0 if m_f <= 2 * mt_mf else 0.0
    feats["small_n_tile"] = 1.0 if n_f <= 2 * mt_nf else 0.0
    feats["is_batched"] = 1.0 if b_f > 1 else 0.0
    feats["log2_a_read_bytes"] = math.log2(max(mt_mf * mt_kf * bpe_a, 1.0))
    feats["log2_b_read_bytes"] = math.log2(max(mt_nf * mt_kf * bpe_b, 1.0))
    feats["log2_compute_density"] = math.log2(
        max(
            (mt_mf * mt_nf * mt_kf * 2.0 * flop_mult)
            / max(mt_mf * mt_kf * bpe_a + mt_nf * mt_kf * bpe_b, 1.0),
            0.001,
        )
    )
    feats["log2_total_work"] = math.log2(max(k_iters * num_tiles * b_f, 1))
    feats["batch_k_iters_per_tile"] = b_f * k_iters / max(num_tiles, 1)
    feats["batch_wave_eff"] = (
        num_tiles_total / (math.ceil(num_tiles_total / N_CU) * N_CU)
        if num_tiles_total > 0
        else 1.0
    )
    feats["log2_grvw_a_bytes"] = math.log2(max(grvw_af * bpe_a, 1.0))
    feats["log2_grvw_b_bytes"] = math.log2(max(grvw_bf * bpe_b, 1.0))

    # Hardware proxies: inputs derived from the device values and the model's
    # constants; the MLP does the ranking.
    feats["ai_tile"] = ai_tile
    feats["l_mi_cycles"] = L_MI
    feats["log2_l_mt_cycles"] = math.log2(max(L_MT, 1.0))
    feats["bw_occ"] = bw_occ
    feats["active_cus_norm"] = active_cus / N_CU

    if as_dict:
        return feats
    return [feats[name] for name in feature_names()]


def feature_vectors(
    problem_kwargs: Mapping[str, Any],
    config_kwargs: Mapping[str, Any],
    *,
    hardware: DeviceHardware,
    constants: ArchConstants,
) -> Tuple[List[float], List[float], List[float]]:
    """(query, item, interaction) vectors of one pair, in catalog order."""
    full = compute_generic_features(
        **problem_kwargs,
        **config_kwargs,
        hardware=hardware,
        constants=constants,
        as_dict=True,
    )
    return split_features(full)


# ── schema / two-tower split ─────────────────────────────────────────────────

FEATURE_GROUPS = {
    "problem": [
        "log2_m",
        "log2_n",
        "log2_k",
        "log2_b",
        "ai_prob",
        "log2_ai_prob",
        "aspect_mn",
        "aspect_mk",
        "aspect_nk",
        "is_pow2_m",
        "is_pow2_n",
        "is_pow2_k",
        "trans_a",
        "trans_b",
        "dtype_a_id",
        "dtype_b_id",
        "dtype_c_id",
        "dtype_d_id",
        "dtype_mi_id",
        "bpe_a",
        "bpe_b",
    ],
    "modular": [
        "m_mod_256",
        "m_mod_128",
        "m_mod_64",
        "m_mod_32",
        "m_mod_16",
        "m_mod_8",
        "n_mod_256",
        "n_mod_128",
        "n_mod_64",
        "n_mod_32",
        "n_mod_16",
        "n_mod_8",
        "m_align_64",
        "m_align_128",
        "m_align_256",
        "n_align_64",
        "n_align_128",
        "n_align_256",
    ],
    "enum_tiles": [
        "log2_enum_tiles_32",
        "log2_enum_tiles_64",
        "log2_enum_tiles_128",
        "log2_enum_tiles_256",
        "enum_coverage_32",
        "enum_coverage_64",
        "enum_coverage_128",
        "enum_coverage_256",
        "log2_enum_k_iters_32",
        "log2_enum_k_iters_64",
        "log2_enum_k_iters_128",
        "log2_enum_k_iters_256",
        "log2_enum_waves_128",
        "log2_enum_waves_256",
        "enum_wave_eff_128",
        "enum_wave_eff_256",
    ],
    "tile": [
        "log2_mt_m",
        "log2_mt_n",
        "log2_mt_k",
        "log2_mi_m",
        "log2_mi_n",
        "log2_mi_k",
        "nta_norm",
        "ntb_norm",
        "occupancy_norm",
        "grvw_a_norm",
        "grvw_b_norm",
        "gwvw_d_norm",
    ],
    "interaction": [
        "log2_num_tiles",
        "log2_num_tiles_total",
        "log2_k_iters",
        "log2_waves",
        "wave_eff",
        "log2_rho",
        "log2_batch_tiles_ratio",
        "util_out",
        "util_3d",
        "tile_coverage",
        "log2_lds_bytes",
        "lds_ratio",
        "l2_fit_ratio",
        "l2_fit_ws",
        "log2_bw_per_cu",
        "log2_total_bytes",
        "log2_total_flops",
        "m_coverage",
        "n_coverage",
        "k_tail",
        "k_aligned_128B",
        "small_m_tile",
        "small_n_tile",
        "is_batched",
        "log2_a_read_bytes",
        "log2_b_read_bytes",
        "log2_compute_density",
        "log2_total_work",
        "batch_k_iters_per_tile",
        "batch_wave_eff",
        "log2_grvw_a_bytes",
        "log2_grvw_b_bytes",
    ],
    "hw_proxies": [
        "ai_tile",
        "l_mi_cycles",
        "log2_l_mt_cycles",
        "bw_occ",
        "active_cus_norm",
    ],
}

QUERY_GROUPS = ["problem", "modular", "enum_tiles"]
ITEM_GROUPS = ["tile"]
INTERACTION_GROUPS = ["interaction", "hw_proxies"]


def query_feature_names() -> List[str]:
    return [n for g in QUERY_GROUPS for n in FEATURE_GROUPS[g]]


def item_feature_names() -> List[str]:
    return [n for g in ITEM_GROUPS for n in FEATURE_GROUPS[g]]


def interaction_feature_names() -> List[str]:
    return [n for g in INTERACTION_GROUPS for n in FEATURE_GROUPS[g]]


def feature_names() -> List[str]:
    return query_feature_names() + item_feature_names() + interaction_feature_names()


def feature_names_hash() -> str:
    """SHA-256 prefix of the ordered feature names; every model file carries
    it and the engine rejects a mismatch."""
    return hashlib.sha256("\n".join(feature_names()).encode("utf-8")).hexdigest()[:16]


def split_features(feats: Any) -> Tuple[List[float], List[float], List[float]]:
    """Split a feature dict (or a `compute_generic_features` list) into
    (query, item, interaction) vectors."""
    if isinstance(feats, list):
        feats = dict(zip(feature_names(), feats))
    q = [feats[n] for n in query_feature_names()]
    it = [feats[n] for n in item_feature_names()]
    ix = [feats[n] for n in interaction_feature_names()]
    return q, it, ix


# ── catalog v3 (MLREC_v3) ────────────────────────────────────────────────────
#
# Framework-neutral: a kernel is its signature (MT m/n/k, MI m/n/k,
# cache_hints_a/b), the parameters every GEMM framework has, and its launch
# for a problem (`LaunchPlan`), which the integration supplies. v2's
# features of the signature, the problem and the execution context, with the
# usable CUs (N_eff) wherever v2 uses the device's CU count; no occupancy,
# vector-width or `bw_occ` feature. A model's attribute spec may add item
# features of named kernel attributes (Config.attributes); only parameters
# every targeted framework defines belong there, and each one declared joins
# the view key.

SCHEDULES: Tuple[str, ...] = ("default", "dynamic", "auto")


@dataclass(frozen=True)
class ExecutionContext:
    """Where the ranked kernels run (tilewright.ExecutionContext): `cu_budget`
    CUs (0, or at least the device's, means every CU) and the tile
    scheduling (`SCHEDULES`)."""

    cu_budget: int = 0
    schedule: str = "default"

    def __post_init__(self) -> None:
        if isinstance(self.cu_budget, bool) or not isinstance(self.cu_budget, int):
            raise ValueError(f"cu_budget must be an int: {self.cu_budget!r}")
        if self.cu_budget < 0:
            raise ValueError(f"cu_budget must be >= 0: {self.cu_budget}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule {self.schedule!r} is not one of {SCHEDULES}")

    def effective_cus(self, n_cu: int) -> int:
        return self.cu_budget if 0 < self.cu_budget < n_cu else n_cu

    def is_exclusive(self, n_cu: Optional[int] = None) -> bool:
        """Every CU with the default schedule. Without `n_cu`, only a zero
        budget counts as every CU."""
        every = self.cu_budget == 0 or (n_cu is not None and self.cu_budget >= n_cu)
        return every and self.schedule == "default"


EXCLUSIVE = ExecutionContext()

ATTRIBUTE_TRANSFORMS: Tuple[str, ...] = ("identity", "log2p1", "nonzero", "onehot")
MAX_ATTRIBUTES = 64
MAX_ONEHOT_VALUES = 64
_INT64_MIN, _INT64_MAX = -(1 << 63), (1 << 63) - 1


@dataclass(frozen=True)
class AttributeFeature:
    """One entry of an item-attribute spec: the kernel attribute `name`
    (`default` when a kernel lacks it) as one or more item features:

      identity  x
      log2p1    log2(1 + max(x, 0))
      nonzero   1 if x != 0 else 0
      onehot    1 if x == v else 0, for each v of `values`"""

    name: str
    default: int = 0
    transform: str = "identity"
    values: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[\x21-\x7e]{1,65535}", str(self.name)):
            raise ValueError(
                f"attribute name {self.name!r} must be printable ASCII without spaces"
            )
        if self.transform not in ATTRIBUTE_TRANSFORMS:
            raise ValueError(
                f"attribute {self.name}: transform {self.transform!r} is not one of "
                f"{ATTRIBUTE_TRANSFORMS}"
            )
        for v in (self.default, *self.values):
            if (
                isinstance(v, bool)
                or not isinstance(v, int)
                or not _INT64_MIN <= v <= _INT64_MAX
            ):
                raise ValueError(f"attribute {self.name}: {v!r} is not an int64")
        if self.transform == "onehot":
            if not 1 <= len(self.values) <= MAX_ONEHOT_VALUES:
                raise ValueError(
                    f"attribute {self.name}: onehot needs 1..{MAX_ONEHOT_VALUES} values"
                )
            if len(set(self.values)) != len(self.values):
                raise ValueError(f"attribute {self.name}: onehot values repeat")
        elif self.values:
            raise ValueError(f"attribute {self.name}: only onehot takes values")

    def feature_names(self) -> List[str]:
        if self.transform == "identity":
            return [f"attr_{self.name}"]
        if self.transform == "log2p1":
            return [f"attr_log2p1_{self.name}"]
        if self.transform == "nonzero":
            return [f"attr_nz_{self.name}"]
        return [f"attr_{self.name}_eq_{v}" for v in self.values]

    def features(self, x: int) -> List[float]:
        if self.transform == "identity":
            return [float(x)]
        if self.transform == "log2p1":
            return [math.log2(1.0 + max(float(x), 0.0))]
        if self.transform == "nonzero":
            return [1.0 if x != 0 else 0.0]
        return [1.0 if x == v else 0.0 for v in self.values]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "default": self.default,
            "transform": self.transform,
        }
        if self.values:
            out["values"] = list(self.values)
        return out


AttributeSpec = Tuple[AttributeFeature, ...]

# Empty: a v3 kernel is its signature. Framework-specific kernel knobs
# (workgroup mapping, stagger, prefetch, vector widths, ...) do not belong in
# a spec.
DEFAULT_ATTRIBUTE_SPEC: AttributeSpec = ()


def parse_attribute_spec(raw: Any) -> AttributeSpec:
    """An attribute spec from a list of {name, default, transform, values}
    mappings (a config's `train.attribute_spec`, or a JSON/YAML file's
    content). ValueError for an invalid or duplicated entry."""
    if isinstance(raw, Mapping) and "attributes" in raw:
        raw = raw["attributes"]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "an attribute spec is a list of {name, default, transform, values}"
        )
    out: List[AttributeFeature] = []
    for entry in raw:
        if isinstance(entry, AttributeFeature):
            out.append(entry)
            continue
        if not isinstance(entry, Mapping):
            raise ValueError(f"attribute spec entry {entry!r} is not a mapping")
        unknown = set(entry) - {"name", "default", "transform", "values"}
        if unknown:
            raise ValueError(f"attribute spec entry has unknown keys {sorted(unknown)}")
        out.append(
            AttributeFeature(
                name=str(entry["name"]),
                default=int(entry.get("default", 0)),
                transform=str(entry.get("transform", "identity")),
                values=tuple(int(v) for v in entry.get("values", ()) or ()),
            )
        )
    names = [a.name for a in out]
    if len(set(names)) != len(names):
        raise ValueError("attribute spec names repeat")
    if len(out) > MAX_ATTRIBUTES:
        raise ValueError(f"attribute spec has more than {MAX_ATTRIBUTES} entries")
    return tuple(out)


V3_QUERY_EXTRA: Tuple[str, ...] = (
    "cu_fraction",
    "schedule_default",
    "schedule_dynamic",
    "schedule_auto",
)
V3_ITEM: Tuple[str, ...] = (
    "log2_mt_m",
    "log2_mt_n",
    "log2_mt_k",
    "log2_mi_m",
    "log2_mi_n",
    "log2_mi_k",
    "nta_norm",
    "ntb_norm",
)
V3_INTERACTION: Tuple[str, ...] = tuple(
    n
    for n in FEATURE_GROUPS["interaction"]
    if n not in ("log2_grvw_a_bytes", "log2_grvw_b_bytes")
)
V3_HW_PROXIES: Tuple[str, ...] = (
    "ai_tile",
    "l_mi_cycles",
    "log2_l_mt_cycles",
    "active_cus_norm",
)
V3_LAUNCH: Tuple[str, ...] = (
    "is_dynamic",
    "log2_grid",
    "log2_grid_per_cu",
    "log2_wg_waves",
    "wg_wave_eff",
    "log2_tiles_per_wg",
    "uneven_work",
    "reduction_tree",
    "reduction_parallel",
    "reduction_atomic",
)


def v3_query_feature_names() -> List[str]:
    return query_feature_names() + list(V3_QUERY_EXTRA)


def v3_item_feature_names(spec: AttributeSpec = ()) -> List[str]:
    return list(V3_ITEM) + [n for a in spec for n in a.feature_names()]


def v3_interaction_feature_names() -> List[str]:
    return list(V3_INTERACTION) + list(V3_HW_PROXIES) + list(V3_LAUNCH)


def v3_catalog_hash() -> str:
    """Hash of the v3 base names (no attribute features); MLREC_v3 files
    carry it and the attribute spec's own hash."""
    names = (
        v3_query_feature_names()
        + v3_item_feature_names()
        + v3_interaction_feature_names()
    )
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]


_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_U64 = (1 << 64) - 1


def _fnv1a(data: bytes, h: int = _FNV_OFFSET) -> int:
    for b in data:
        h = ((h ^ b) * _FNV_PRIME) & _U64
    return h


def attribute_spec_hash(spec: AttributeSpec, catalog_hash: Optional[str] = None) -> int:
    """FNV-1a 64 of "<catalog hash>\\n" and one line per entry,
    "<name>\\t<default>\\t<transform>" with "\\t<v1>,<v2>,..." for onehot."""
    text = (catalog_hash or v3_catalog_hash()) + "\n"
    for a in spec:
        text += f"{a.name}\t{a.default}\t{a.transform}"
        if a.transform == "onehot":
            text += "\t" + ",".join(str(v) for v in a.values)
        text += "\n"
    return _fnv1a(text.encode("ascii"))


def attribute_reads(spec: AttributeSpec) -> List[Tuple[str, int]]:
    """(name, default) of every attribute a v3 model reads, in view-key
    order."""
    return [(a.name, a.default) for a in spec]


SIGNATURE_FIELDS: Tuple[str, ...] = (
    "mt_m",
    "mt_n",
    "mt_k",
    "mi_m",
    "mi_n",
    "mi_k",
    "cache_hints_a",
    "cache_hints_b",
)


def v3_view_key(
    config_kwargs: Mapping[str, Any],
    attributes: Optional[Mapping[str, int]],
    spec: AttributeSpec,
) -> Tuple[int, ...]:
    """The engine's view key of a config for a v3 model: the signature
    (`SIGNATURE_FIELDS` as given, no clamping), then `attribute_reads`.
    Kernels with equal keys are one candidate to a v3 model."""
    attrs = attributes or {}
    head = tuple(int(config_kwargs[k]) for k in SIGNATURE_FIELDS)
    return head + tuple(int(attrs.get(n, d)) for n, d in attribute_reads(spec))


def v3_candidate_key(
    config_kwargs: Mapping[str, Any],
    attributes: Optional[Mapping[str, int]],
    spec: AttributeSpec,
    launch: Optional[Any] = None,
) -> Tuple[int, ...]:
    """The engine's grouping of a config for a v3 model ranked with launch
    model `launch`: the view key, then the launch key (`launch.launch_key(
    config_kwargs, attributes)`, 0 without one). Configs with equal keys are
    one candidate and share a score; whitelists still match the view key."""
    key_of = getattr(launch, "launch_key", None)
    lk = int(key_of(config_kwargs, attributes)) & _U64 if key_of is not None else 0
    return v3_view_key(config_kwargs, attributes, spec) + (lk,)


def view_key_hash(key: Sequence[int]) -> int:
    """FNV-1a 64 of the key's values as 8-byte little-endian two's
    complement: what MLREC_v3 keyed whitelists list."""
    return _fnv1a(b"".join(struct.pack("<Q", int(v) & _U64) for v in key))


PARTIAL_REDUCTIONS: Tuple[str, ...] = ("none", "tree", "parallel", "atomic")


@dataclass(frozen=True)
class LaunchPlan:
    """How a kernel is launched for one problem (tilewright.LaunchPlan):
    whether workgroups take output tiles from a shared queue, the workgroups
    launched (0: one per output tile) and how the workgroups that split an
    output tile combine their partial results (`PARTIAL_REDUCTIONS`; "none"
    when no tile is split). The default is the engine's plan without a
    LaunchModel."""

    dynamic_queue: bool = False
    grid_wgs: int = 0
    partial_reduction: str = "none"

    def __post_init__(self) -> None:
        if isinstance(self.grid_wgs, bool) or not isinstance(self.grid_wgs, int):
            raise ValueError(f"grid_wgs must be an int: {self.grid_wgs!r}")
        if not 0 <= self.grid_wgs < (1 << 64):
            raise ValueError(f"grid_wgs {self.grid_wgs} is out of range")
        if self.partial_reduction not in PARTIAL_REDUCTIONS:
            raise ValueError(
                f"partial_reduction {self.partial_reduction!r} is not one of "
                f"{PARTIAL_REDUCTIONS}"
            )


DEFAULT_LAUNCH = LaunchPlan()

# A catalog's source of launch plans: (problem kwargs, config kwargs, kernel
# attributes, execution context, device CUs) -> LaunchPlan.
LaunchSource = Callable[
    [
        Mapping[str, Any],
        Mapping[str, Any],
        Optional[Mapping[str, int]],
        ExecutionContext,
        int,
    ],
    LaunchPlan,
]


def _ceil_div(n: int, d: int) -> int:
    return 0 if d == 0 else -(-n // d)


def output_tiles(
    problem_kwargs: Mapping[str, Any], config_kwargs: Mapping[str, Any]
) -> int:
    """Output macro tiles of a pair, batch included."""
    mt_m = max(int(config_kwargs.get("mt_m", 1)), 1)
    mt_n = max(int(config_kwargs.get("mt_n", 1)), 1)
    return (
        _ceil_div(int(problem_kwargs["m"]), mt_m)
        * _ceil_div(int(problem_kwargs["n"]), mt_n)
        * int(problem_kwargs.get("batch", 1))
    )


def launch_features(
    problem_kwargs: Mapping[str, Any],
    config_kwargs: Mapping[str, Any],
    plan: LaunchPlan,
    n_eff: int,
) -> List[float]:
    """`V3_LAUNCH` of a pair launched as `plan` on `n_eff` usable CUs."""
    tiles = output_tiles(problem_kwargs, config_kwargs)
    wgs = plan.grid_wgs if plan.grid_wgs > 0 else tiles
    grid = float(wgs)
    n_eff_f = float(n_eff)
    waves = math.ceil(grid / n_eff_f)
    return [
        1.0 if plan.dynamic_queue else 0.0,
        math.log2(max(grid, 1.0)),
        math.log2(max(grid / n_eff_f, 0.001)),
        math.log2(max(float(waves), 1.0)),
        grid / (waves * n_eff_f) if waves > 0 else 1.0,
        math.log2(max(float(tiles) / max(grid, 1.0), 0.001)),
        1.0 if wgs > 0 and tiles % wgs != 0 else 0.0,
        1.0 if plan.partial_reduction == "tree" else 0.0,
        1.0 if plan.partial_reduction == "parallel" else 0.0,
        1.0 if plan.partial_reduction == "atomic" else 0.0,
    ]


def v3_feature_vectors(
    problem_kwargs: Mapping[str, Any],
    config_kwargs: Mapping[str, Any],
    attributes: Optional[Mapping[str, int]] = None,
    *,
    hardware: DeviceHardware,
    constants: ArchConstants,
    context: ExecutionContext = EXCLUSIVE,
    spec: AttributeSpec = (),
    plan: LaunchPlan = DEFAULT_LAUNCH,
) -> Tuple[List[float], List[float], List[float]]:
    """(query, item, interaction) vectors of one pair under the v3 catalog,
    in `v3_*_feature_names` order, for a kernel launched as `plan`.
    `config_kwargs` as `config_kwargs_from_row` gives them; only the
    signature fields are read."""
    n_cu = int(hardware.n_cu)
    n_eff = context.effective_cus(n_cu)
    usable = DeviceHardware(
        n_cu=n_eff, lds_bytes=hardware.lds_bytes, l2_bytes=hardware.l2_bytes
    )
    full = compute_generic_features(
        **problem_kwargs,
        **config_kwargs,
        hardware=usable,
        constants=constants,
        as_dict=True,
    )
    q = [full[n] for n in query_feature_names()]
    q += [
        float(n_eff) / float(n_cu),
        1.0 if context.schedule == "default" else 0.0,
        1.0 if context.schedule == "dynamic" else 0.0,
        1.0 if context.schedule == "auto" else 0.0,
    ]
    attrs = attributes or {}
    it = [full[n] for n in V3_ITEM]
    for a in spec:
        it += a.features(int(attrs.get(a.name, a.default)))
    x = [full[n] for n in V3_INTERACTION]
    x += [full[n] for n in V3_HW_PROXIES]
    x += launch_features(problem_kwargs, config_kwargs, plan, n_eff)
    return q, it, x


@dataclass(frozen=True)
class Catalog:
    """A feature catalog the engine computes: "v2" (MLREC_v2), or "v3"
    (MLREC_v3) with its item-attribute spec and the source of its kernels'
    launch plans (None: `DEFAULT_LAUNCH`, the engine's plan without a
    LaunchModel)."""

    name: str = "v2"
    spec: AttributeSpec = ()
    launch: Optional[LaunchSource] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.name not in ("v2", "v3"):
            raise ValueError(f"unknown feature catalog {self.name!r} (v2 or v3)")
        if self.name == "v2" and self.spec:
            raise ValueError("the v2 catalog has no attribute spec")

    @property
    def version(self) -> int:
        return 3 if self.name == "v3" else 2

    def names(self) -> Tuple[List[str], List[str], List[str]]:
        if self.name == "v2":
            return (
                query_feature_names(),
                item_feature_names(),
                interaction_feature_names(),
            )
        return (
            v3_query_feature_names(),
            v3_item_feature_names(self.spec),
            v3_interaction_feature_names(),
        )

    def hash(self) -> str:
        return feature_names_hash() if self.name == "v2" else v3_catalog_hash()

    def vectors(
        self,
        problem_kwargs: Mapping[str, Any],
        config_kwargs: Mapping[str, Any],
        attributes: Optional[Mapping[str, int]] = None,
        *,
        hardware: DeviceHardware,
        constants: ArchConstants,
        context: ExecutionContext = EXCLUSIVE,
    ) -> Tuple[List[float], List[float], List[float]]:
        """(query, item, interaction) vectors of one pair; v2 reads neither
        attributes, the context nor a launch."""
        if self.name == "v2":
            return feature_vectors(
                problem_kwargs, config_kwargs, hardware=hardware, constants=constants
            )
        plan = DEFAULT_LAUNCH
        if self.launch is not None:
            plan = self.launch(
                problem_kwargs, config_kwargs, attributes, context, int(hardware.n_cu)
            )
        return v3_feature_vectors(
            problem_kwargs,
            config_kwargs,
            attributes,
            hardware=hardware,
            constants=constants,
            context=context,
            spec=self.spec,
            plan=plan,
        )


V2_CATALOG = Catalog()
