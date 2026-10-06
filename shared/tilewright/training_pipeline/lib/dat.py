# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tensile library `.dat` logic files: filename decoding and per-kernel
parameters.

A logic file is msgpack, deflated with zlib when it is named `.dat.zlib`. Each
entry of its `solutions` list carries:
    sol["index"]              runtime solution index: hipblaslt-bench prints it
                              as `--Solution index` on tested rows and as
                              `solution index = N` on `Skip solution` lines
    sol["libraryLogicIndex"]  position of the kernel in its source logic YAML
    sol["name"]               kernel name
    sol["sizeMapping"]        macroTile, depthU, matrixInstruction,
                              nonTemporalA/B, grvwA/B, gwvwD, CUOccupancy, ...
Its `library` tree routes problems to rows; the `table` of the Prediction
library lists the solution indices hipBLASLt ranks with tilewright, in the
order of its kernel pool. The file can hold solutions of other rows too.

`kernel_dat_info` turns one solution into the kernel parameters the runtime
hands tilewright (`tilewright::Config`, including its named attributes) plus
its identifiers.
"""
from __future__ import annotations

import json
import os
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import msgpack

from .fs import atomic_write_text

LOGIC_SUFFIXES = (".dat.zlib", ".dat")


def is_tensile_contraction_logic(fname: str) -> bool:
    """True for a contraction logic file, `.dat` or `.dat.zlib`."""
    return "Contraction" in fname and fname.endswith(LOGIC_SUFFIXES)


def logic_stem(fname: str) -> str:
    """`<stem>.dat` / `<stem>.dat.zlib` -> `<stem>` (the library stem the
    runtime and `tilewright_index` use)."""
    for suffix in LOGIC_SUFFIXES:
        if fname.endswith(suffix):
            return fname[: -len(suffix)]
    return fname


def read_tensile_logic(path) -> Optional[Dict[str, Any]]:
    """Unpacked msgpack dict of one logic file, or None if it cannot be read."""
    p = Path(path)
    try:
        raw = p.read_bytes()
        if p.name.endswith(".zlib"):
            raw = zlib.decompress(raw)
        return msgpack.unpackb(raw, raw=False)
    except Exception:
        return None


# ── filename -> scale mode ───────────────────────────────────────────────────

# Each row is (filename_pattern, scale_mode).
#
# `scale_mode` is the hipblaslt-bench `scaleA`/`scaleB` int the library expects
# (clients/common/include/hipblaslt_scaling_format.hpp): 0 none, 1 Scalar,
# 2 Vector, 3 Block_32_UE8M0 (MX), 4 Block_16_UE8M0, 5 Block_32_UE4M3, ...
# Two gfx1250 F8 TN libraries share data types and layout and differ only in
# scaling (`_F8F8_BF8_HA_Bias_SAB_SAV_UA_` scalar vs
# `_F8F8_BF8_HA_MXAE8B32_MXBE8B32_Bias_SAV_UA_` MX block-32). Their kernels are
# not interchangeable, so the pools must not be merged.
#
# The first pattern that is a substring of the filename wins, so an MX row
# must precede any shorter pattern that could also match its filename.
SCALE_MODE_PATTERNS: Tuple[Tuple[str, int], ...] = (
    ("_BB_BB_", 0),
    ("_HH_HH_", 0),
    ("_SS_SS_HA_Bias_SAV_MX_", 0),
    ("_SS_SS_HA_Bias_SAV_UA_", 0),
    ("_SS_SB_", 0),
    ("_F8F8_BF8_HA_MXAE8B32_MXBE8B32_Bias_SAV_UA_", 3),
    ("_F8F8_BF8_HA_Bias_SAB_SAV_UA_", 1),
)

# hipblaslt-bench scale mode -> MX block size (elements per scale); 0 means
# the mode is not a block format.
SCALE_MODE_MX_BLOCK: Dict[int, int] = {
    0: 0,  # none
    1: 0,  # Scalar
    2: 0,  # Vector
    3: 32,  # Block_32_UE8M0
    4: 16,  # Block_16_UE8M0
    5: 32,  # Block_32_UE4M3
    1001: 32,  # Block_32_UE8M0_32_8_EXT
}


def mx_block_size_for_scale_mode(scale_mode: Optional[int]) -> int:
    """MX block size for a scale mode; None and unknown modes are non-MX (0)."""
    if scale_mode is None:
        return 0
    return SCALE_MODE_MX_BLOCK.get(int(scale_mode), 0)


def parse_scale_mode(fname: str) -> Optional[int]:
    """Scale mode the library in `fname` expects, or None when the filename
    matches no known pattern."""
    for pattern, scale_mode in SCALE_MODE_PATTERNS:
        if pattern in fname:
            return int(scale_mode)
    return None


# ── kernel parameters ────────────────────────────────────────────────────────

# Kernels without a matrix instruction (Dot2) carry an all-zero
# `matrixInstruction`; the runtime hands tilewright this MI for them.
DOT2_MI = (1, 1, 64)


def _matrix_instruction(sm: Dict[str, Any]) -> Tuple[int, int, int]:
    mi = list(sm.get("matrixInstruction") or [])
    mi += [0] * (3 - len(mi))
    m, n, k = (int(v or 0) for v in mi[:3])
    if m == 0 and n == 0 and k == 0:
        return DOT2_MI
    return m, n, k


def _cache_hint(sm: Dict[str, Any], operand: str) -> int:
    # Same value TensileLite passes to tilewright (ContractionSolution::cacheHintA/B).
    if sm.get("hasTemporalHint"):
        return 4 if int(sm.get(f"temporalHint{operand}", 0) or 0) in (1, 3) else 0
    return int(sm.get(f"nonTemporal{operand}", 0) or 0)


# ── kernel attributes ────────────────────────────────────────────────────────

_EXECUTION_POLICY = "<execution policy>"

# (attribute names, sizeMapping key, C++ SizeMapping defaults, key required by
# TensileLite). A vector key (dim3, std::array) fills its names in element
# order. Members without a C++ initializer (waveNum, threadTile, WaveGroup)
# default to 0. TensileLite hands the engine the same names and values from
# its SizeMapping.
_ATTRIBUTE_KEYS: Tuple[Tuple[Tuple[str, ...], str, Tuple[int, ...], bool], ...] = (
    (("wave_num",), "waveNum", (0,), True),
    (("work_group_x", "work_group_y", "work_group_z"), "workGroup", (0, 0, 0), True),
    (
        ("thread_tile_x", "thread_tile_y", "thread_tile_z"),
        "threadTile",
        (0, 0, 0),
        True,
    ),
    (("gwvw_c",), "gwvwC", (1,), True),
    (("stagger_u",), "staggerU", (0,), True),
    (("stagger_u_mapping",), "staggerUMapping", (0,), True),
    (("global_split_u_pgr",), "globalSplitUPGR", (0,), True),
    (("global_split_u",), "globalSplitU", (0,), True),
    (("stagger_stride_shift",), "staggerStrideShift", (0,), True),
    (("workgroup_mapping",), "workGroupMapping", (0,), True),
    (("workgroup_mapping_xcc",), "workGroupMappingXCC", (0,), True),
    (("workgroup_mapping_xcc_group",), "workGroupMappingXCCGroup", (0,), True),
    (("global_split_u_coalesced",), "globalSplitUCoalesced", (0,), True),
    (
        ("global_split_u_wgm_round_robin",),
        "globalSplitUWorkGroupMappingRoundRobin",
        (0,),
        True,
    ),
    (("pack_batch_dims",), "packBatchDims", (0,), False),
    (("pack_summation_dims",), "packSummationDims", (0,), False),
    (("magic_div_alg",), "magicDivAlg", (1,), False),
    (
        ("stream_k_atomic", "tile_processing_strategy", "work_assignment"),
        _EXECUTION_POLICY,
        (0, 0, 0),
        False,
    ),
    (("prefetch_across_persistent",), "prefetchAcrossPersistent", (0,), False),
    (("persistent_kernel",), "persistentKernel", (0,), False),
    (("persistent_kernel_along_batch",), "persistentKernelAlongBatch", (0,), False),
    (("source_kernel",), "sourceKernel", (0,), True),
    (("global_accumulation",), "globalAccumulation", (0,), True),
    (("adaptive_gemm_gsua",), "adaptiveGemmGSUA", (0,), False),
    (("activation_fused",), "activationFused", (1,), False),
    (("prefetch_global_read",), "PrefetchGlobalRead", (2,), True),
    (("math_clocks_unrolled_loop",), "MathClocksUnrolledLoop", (0,), True),
    (("non_temporal_a",), "nonTemporalA", (0,), True),
    (("non_temporal_b",), "nonTemporalB", (0,), True),
    (("temporal_hint_a",), "temporalHintA", (0,), False),
    (("temporal_hint_b",), "temporalHintB", (0,), False),
    (("has_temporal_hint",), "hasTemporalHint", (0,), False),
    (("adaptive_gemm_ntab",), "adaptiveGemmNTAB", (0,), False),
    (("custom_main_loop_scheduling",), "customMainLoopScheduling", (0,), True),
    (("use_subtile_impl",), "useSubtileImpl", (0,), False),
    (("source_swap",), "SourceSwap", (0,), False),
    (("non_temporal_d",), "NonTemporalD", (0,), True),
    (("wave_separate_global_read_a",), "WaveSeparateGlobalReadA", (0,), True),
    (("wave_separate_global_read_b",), "WaveSeparateGlobalReadB", (0,), True),
    (
        ("unroll_loop_swap_global_read_order",),
        "UnrollLoopSwapGlobalReadOrder",
        (0,),
        True,
    ),
    (("direct_to_vgpr_a",), "DirectToVgprA", (0,), True),
    (("direct_to_vgpr_b",), "DirectToVgprB", (0,), True),
    (("num_loads_coalesced_a",), "NumLoadsCoalescedA", (0,), True),
    (("num_loads_coalesced_b",), "NumLoadsCoalescedB", (0,), True),
    (("wave_group_0", "wave_group_1"), "WaveGroup", (0, 0), True),
    (("vector_width_a",), "VectorWidthA", (1,), True),
    (("vector_width_b",), "VectorWidthB", (1,), True),
    (("local_split_u",), "LocalSplitU", (1,), True),
    (("direct_to_lds_a",), "DirectToLdsA", (0,), True),
    (("direct_to_lds_b",), "DirectToLdsB", (0,), True),
    (("expert_scheduling_mode",), "ExpertSchedulingMode", (0,), False),
    (
        ("cluster_dim_x", "cluster_dim_y", "cluster_dim_z"),
        "clusterDim",
        (1, 1, 1),
        False,
    ),
)

ATTRIBUTE_NAMES: Tuple[str, ...] = tuple(
    name for names, _key, _defaults, _required in _ATTRIBUTE_KEYS for name in names
)

# SizeMapping members of type bool.
_BOOL_KEYS = frozenset(
    {
        "persistentKernelAlongBatch",
        "sourceKernel",
        "activationFused",
        "globalSplitUCoalesced",
        "globalSplitUWorkGroupMappingRoundRobin",
        "hasTemporalHint",
        "useSubtileImpl",
        "SourceSwap",
        "DirectToVgprA",
        "DirectToVgprB",
        "DirectToLdsA",
        "DirectToLdsB",
    }
)

# Enumerator names in C++ declaration order: the attribute value is the
# position (ContractionSolution.hpp TileProcessingStrategy / WorkAssignment).
TILE_PROCESSING_STRATEGIES: Tuple[str, ...] = ("None", "DataParallel", "StreamK")
WORK_ASSIGNMENTS: Tuple[str, ...] = ("StaticGrid", "DynamicWorkQueue", "Hybrid")

_INT64_MIN, _INT64_MAX = -(1 << 63), (1 << 63) - 1


def _int64(value: Any, key: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(f"sizeMapping {key} {value!r} is not an integer")
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise ValueError(f"sizeMapping {key} {value} does not fit in int64")
    return value


def _scalar(sm: Mapping[str, Any], key: str, default: int) -> int:
    value = sm.get(key)
    if value is None:
        return default
    value = _int64(value, key)
    if key in _BOOL_KEYS and value not in (0, 1):
        raise ValueError(f"sizeMapping {key} {value} is not a bool")
    return value


def _vector(sm: Mapping[str, Any], key: str, defaults: Tuple[int, ...]) -> List[int]:
    value = sm.get(key)
    if value is None:
        return list(defaults)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"sizeMapping {key} {value!r} is not a list")
    given = [_int64(v, key) for v in value[: len(defaults)]]
    return given + list(defaults[len(given) :])


def _enumerator(sm: Mapping[str, Any], key: str, names: Tuple[str, ...]) -> int:
    value = sm.get(key, names[0])
    if not isinstance(value, str) or value not in names:
        raise ValueError(f"sizeMapping {key} {value!r} is not one of {names}")
    return names.index(value)


def _execution_policy(sm: Mapping[str, Any]) -> Tuple[int, int, int]:
    """(stream_k_atomic, tile_processing_strategy, work_assignment) as
    TensileLite reads them (Serialization/ContractionSolution.hpp): the legacy
    streamK / streamKForceDPOnly keys replace the canonical ones, and
    ValueError for the combinations it rejects."""
    strategy = _enumerator(sm, "tileProcessingStrategy", TILE_PROCESSING_STRATEGIES)
    assignment = _enumerator(sm, "workAssignment", WORK_ASSIGNMENTS)
    atomic = _scalar(sm, "streamKAtomic", 0)
    legacy_mode = _scalar(sm, "streamK", 0)
    legacy_dp = _scalar(sm, "streamKForceDPOnly", 0)
    if legacy_dp not in (0, 1):
        raise ValueError("streamKForceDPOnly must be 0 or 1")
    if legacy_dp and (legacy_mode != 3 or atomic != 0):
        raise ValueError("streamKForceDPOnly requires non-atomic streamK 3")
    if "streamK" in sm or "streamKForceDPOnly" in sm:
        if legacy_mode not in (0, 3, 4, 5):
            raise ValueError(f"unsupported legacy streamK mode {legacy_mode}")
        if legacy_dp:
            want_strategy = TILE_PROCESSING_STRATEGIES.index("DataParallel")
        elif legacy_mode:
            want_strategy = TILE_PROCESSING_STRATEGIES.index("StreamK")
        else:
            want_strategy = TILE_PROCESSING_STRATEGIES.index("None")
        want_assignment = WORK_ASSIGNMENTS.index(
            {4: "DynamicWorkQueue", 5: "Hybrid"}.get(legacy_mode, "StaticGrid")
        )
        if ("tileProcessingStrategy" in sm and strategy != want_strategy) or (
            "workAssignment" in sm
            and want_strategy != 0
            and assignment != want_assignment
        ):
            raise ValueError("conflicting legacy and canonical execution policy")
        strategy, assignment = want_strategy, want_assignment
        if legacy_mode == 0:
            atomic = 0
    if strategy == 0:
        assignment = 0
    if atomic not in (0, 1):
        raise ValueError("streamKAtomic must be 0 or 1")
    if atomic and strategy != TILE_PROCESSING_STRATEGIES.index("StreamK"):
        raise ValueError("streamKAtomic requires tileProcessingStrategy StreamK")
    if strategy == TILE_PROCESSING_STRATEGIES.index("DataParallel") and assignment:
        raise ValueError("DataParallel supports workAssignment StaticGrid only")
    return atomic, strategy, assignment


def kernel_attributes(sm: Mapping[str, Any]) -> Dict[str, int]:
    """Named attributes of one solution's sizeMapping, every name of
    ATTRIBUTE_NAMES as an int64: bools are 0/1, an absent key takes the C++
    SizeMapping default. ValueError for a value TensileLite cannot read."""
    out: Dict[str, int] = {}
    for names, key, defaults, _required in _ATTRIBUTE_KEYS:
        if key == _EXECUTION_POLICY:
            values: Iterable[int] = _execution_policy(sm)
        elif len(names) == 1:
            values = (_scalar(sm, key, defaults[0]),)
        else:
            values = _vector(sm, key, defaults)
        out.update(zip(names, values))
    return out


def missing_size_mapping_keys(sm: Mapping[str, Any]) -> List[str]:
    """Attribute keys TensileLite requires that `sm` lacks; their attributes
    take the C++ defaults."""
    return [
        key
        for _names, key, _defaults, required in _ATTRIBUTE_KEYS
        if required and sm.get(key) is None
    ]


def kernel_dat_info(sol: Dict[str, Any]) -> Dict[str, Any]:
    """Identifiers and `tilewright::Config` parameters of one solution:

    sol_idx_global, sol_idx_local, kernel_name,
    mt_m, mt_n, mt_k (depthU), mi_m, mi_n, mi_k,
    occupancy (CUOccupancy clamped to >= 1),
    cache_hints_a, cache_hints_b,
    grvw_a, grvw_b, gwvw_d,
    attributes (`kernel_attributes`)

    ValueError, naming the solution, when its sizeMapping holds a value
    TensileLite cannot read."""
    sm = sol.get("sizeMapping", {}) or {}
    mt = sm.get("macroTile") or [0, 0, 0]
    mi_m, mi_n, mi_k = _matrix_instruction(sm)
    try:
        attributes = kernel_attributes(sm)
    except ValueError as e:
        raise ValueError(f"solution {sol.get('index')}: {e}") from None
    return {
        "sol_idx_global": sol.get("index"),
        "sol_idx_local": sol.get("libraryLogicIndex"),
        "kernel_name": sol.get("name") or sol.get("kernelName") or "",
        "mt_m": int(mt[0]),
        "mt_n": int(mt[1]),
        "mt_k": int(sm.get("depthU", 0) or 0),
        "mi_m": mi_m,
        "mi_n": mi_n,
        "mi_k": mi_k,
        "occupancy": max(int(sm.get("CUOccupancy", 1) or 1), 1),
        "cache_hints_a": _cache_hint(sm, "A"),
        "cache_hints_b": _cache_hint(sm, "B"),
        "grvw_a": int(sm.get("grvwA", 1) or 1),
        "grvw_b": int(sm.get("grvwB", 1) or 1),
        "gwvw_d": int(sm.get("gwvwD", 1) or 1),
        "attributes": attributes,
    }


def sig_from_row(row: Dict[str, Any]) -> Tuple[int, ...]:
    """Kernel signature (mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, cache_hints_a,
    cache_hints_b) of an enriched chunk_*.csv row: the fields the engine's
    per-cell whitelist matches on."""

    def _i(k, d=0):
        v = row.get(k, d)
        try:
            return int(v) if v != "" else d
        except (TypeError, ValueError):
            return d

    return (
        _i("mt_m"),
        _i("mt_n"),
        _i("mt_k"),
        _i("mi_m"),
        _i("mi_n"),
        _i("mi_k"),
        _i("cache_hints_a"),
        _i("cache_hints_b"),
    )


# ── loading libraries ────────────────────────────────────────────────────────


def library_logic_path(dat_dir, library_stem: str) -> Path:
    """`<dat_dir>/<library_stem>.dat.zlib` or `.dat`, whichever exists.
    Raises FileNotFoundError when neither does."""
    for suffix in LOGIC_SUFFIXES:
        p = Path(dat_dir) / f"{library_stem}{suffix}"
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"library {library_stem!r}: neither {library_stem}.dat nor "
        f"{library_stem}.dat.zlib exists in {dat_dir}"
    )


def prediction_table(data: Mapping[str, Any]) -> Optional[List[int]]:
    """Solution indices of the Prediction library of a logic file (the
    `table` of TensileLite's ProblemPredictionLibrary), in table order, or
    None when the file has no Prediction library. ValueError when it has
    several that differ, or a table that is not a list of distinct solution
    indices."""
    tables: Set[Tuple[int, ...]] = set()
    stack: List[Any] = [data.get("library")]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("type") == "Prediction" and "table" in node:
                table = node["table"]
                if not isinstance(table, list) or not all(
                    isinstance(i, int) and not isinstance(i, bool) for i in table
                ):
                    raise ValueError("a Prediction table is not a list of indices")
                if len(set(table)) != len(table):
                    raise ValueError("a Prediction table repeats a solution index")
                tables.add(tuple(table))
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    if len(tables) > 1:
        raise ValueError(f"{len(tables)} different Prediction tables")
    return list(tables.pop()) if tables else None


def load_library_kernels(dat_dir, library_stem: str) -> List[Dict[str, Any]]:
    """`kernel_dat_info` of the kernel pool hipBLASLt ranks with tilewright
    for one library, in pool order: the solutions of its Prediction table in
    table order, or every solution in file order when the file has no
    Prediction library. Raises FileNotFoundError / ValueError when the
    library is missing or unreadable, its Prediction table cannot be
    followed, or it holds a solution TensileLite cannot read."""
    path = library_logic_path(dat_dir, library_stem)
    data = read_tensile_logic(path)
    if data is None:
        raise ValueError(f"unreadable Tensile logic file: {path}")
    solutions = data.get("solutions", []) or []
    try:
        table = prediction_table(data)
        if table is not None:
            by_index: Dict[Any, Dict[str, Any]] = {}
            for sol in solutions:
                by_index.setdefault(sol.get("index"), sol)
            absent = [i for i in table if i not in by_index]
            if absent:
                raise ValueError(
                    f"the Prediction table names {len(absent)} solution(s) the "
                    f"file does not hold, e.g. {absent[0]}"
                )
            solutions = [by_index[i] for i in table]
        return [kernel_dat_info(sol) for sol in solutions]
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None


@dataclass
class KernelIndex:
    """Runtime solution index -> `kernel_dat_info`, over the loaded files.
    `missing_keys` counts, per sizeMapping key TensileLite requires, the
    indexed solutions without it."""

    by_index: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)
    invalid: List[str] = field(default_factory=list)
    skipped_scale_mode: int = 0
    duplicate_indices: int = 0
    missing_keys: Dict[str, int] = field(default_factory=dict)


def load_kernel_index(
    dat_dir,
    *,
    library_stem: Optional[str] = None,
    scale_mode: Optional[int] = None,
) -> KernelIndex:
    """Index the solutions of one library (`library_stem`) or, without a stem,
    of every contraction library in `dat_dir` whose scale mode matches
    `scale_mode` (all of them when `scale_mode` is None).

    An explicit stem that is missing, unreadable or holds a solution
    TensileLite cannot read raises; in directory mode such files are skipped
    and listed in `unreadable` / `invalid` (with the reason). A solution index
    seen twice keeps its first entry and counts in `duplicate_indices`."""
    out = KernelIndex()
    missing: Counter = Counter()
    if library_stem:
        paths = [library_logic_path(dat_dir, library_stem)]
    else:
        paths = []
        for fname in sorted(os.listdir(dat_dir)):
            if not is_tensile_contraction_logic(fname):
                continue
            if scale_mode is not None and parse_scale_mode(fname) != int(scale_mode):
                out.skipped_scale_mode += 1
                continue
            paths.append(Path(dat_dir) / fname)
    for path in paths:
        data = read_tensile_logic(path)
        if data is None:
            if library_stem:
                raise ValueError(f"unreadable Tensile logic file: {path}")
            out.unreadable.append(path.name)
            continue
        solutions = [
            s for s in data.get("solutions", []) or [] if s.get("index") is not None
        ]
        try:
            infos = [kernel_dat_info(sol) for sol in solutions]
        except ValueError as e:
            if library_stem:
                raise ValueError(f"{path}: {e}") from None
            out.invalid.append(f"{path.name}: {e}")
            continue
        out.files.append(path.name)
        for sol, info in zip(solutions, infos):
            gi = sol["index"]
            if int(gi) in out.by_index:
                out.duplicate_indices += 1
                continue
            out.by_index[int(gi)] = info
            missing.update(missing_size_mapping_keys(sol.get("sizeMapping") or {}))
    out.missing_keys = dict(sorted(missing.items()))
    return out


# ── kernel attribute file ────────────────────────────────────────────────────

KERNELS_FILE = "kernels.json"


def write_kernel_attributes(
    path, kernels: Mapping[int, Mapping[str, Any]], library_stem: Optional[str]
) -> None:
    """`kernels.json`: the attributes of each solution index of `kernels`
    (`kernel_dat_info` entries), as lists in ATTRIBUTE_NAMES order."""
    doc = {
        "library_stem": library_stem,
        "attribute_names": list(ATTRIBUTE_NAMES),
        "kernels": {
            str(sid): [int(info["attributes"][n]) for n in ATTRIBUTE_NAMES]
            for sid, info in sorted(kernels.items())
        },
    }
    atomic_write_text(Path(path), json.dumps(doc, separators=(",", ":")) + "\n")


def read_kernel_attributes(path) -> Dict[int, Dict[str, int]]:
    """Solution index -> attributes of a `kernels.json` file. ValueError when
    the file is malformed."""
    try:
        doc = json.loads(Path(path).read_text())
        names = [str(n) for n in doc["attribute_names"]]
        if not all(names) or len(set(names)) != len(names):
            raise ValueError("attribute names must be non-empty and unique")
        out: Dict[int, Dict[str, int]] = {}
        for sid, values in doc["kernels"].items():
            if len(values) != len(names):
                raise ValueError(f"solution {sid} has {len(values)} values")
            out[int(sid)] = {n: _int64(v, n) for n, v in zip(names, values)}
    except (OSError, KeyError, TypeError, AttributeError, ValueError) as e:
        raise ValueError(f"{path}: not a kernel attribute file ({e})") from None
    return out
