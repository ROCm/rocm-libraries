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
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Tuple

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
