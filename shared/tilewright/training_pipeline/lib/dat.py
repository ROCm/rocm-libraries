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

`kernel_dat_info` turns one solution into the kernel parameters the runtime
hands tilewright (`tilewright::Config`) plus its identifiers.
"""
from __future__ import annotations

import os
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgpack

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


def kernel_dat_info(sol: Dict[str, Any]) -> Dict[str, Any]:
    """Identifiers and `tilewright::Config` parameters of one solution:

    sol_idx_global, sol_idx_local, kernel_name,
    mt_m, mt_n, mt_k (depthU), mi_m, mi_n, mi_k,
    occupancy (CUOccupancy clamped to >= 1),
    cache_hints_a, cache_hints_b,
    grvw_a, grvw_b, gwvw_d"""
    sm = sol.get("sizeMapping", {}) or {}
    mt = sm.get("macroTile") or [0, 0, 0]
    mi_m, mi_n, mi_k = _matrix_instruction(sm)
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


def load_library_kernels(dat_dir, library_stem: str) -> List[Dict[str, Any]]:
    """`kernel_dat_info` of every solution of one library, in file order (the
    order the runtime builds that library's candidate pool in). Raises
    FileNotFoundError / ValueError when the library is missing or unreadable."""
    path = library_logic_path(dat_dir, library_stem)
    data = read_tensile_logic(path)
    if data is None:
        raise ValueError(f"unreadable Tensile logic file: {path}")
    return [kernel_dat_info(sol) for sol in data.get("solutions", []) or []]


@dataclass
class KernelIndex:
    """Runtime solution index -> `kernel_dat_info`, over the loaded files."""

    by_index: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)
    skipped_scale_mode: int = 0
    duplicate_indices: int = 0


def load_kernel_index(
    dat_dir,
    *,
    library_stem: Optional[str] = None,
    scale_mode: Optional[int] = None,
) -> KernelIndex:
    """Index the solutions of one library (`library_stem`) or, without a stem,
    of every contraction library in `dat_dir` whose scale mode matches
    `scale_mode` (all of them when `scale_mode` is None).

    An explicit stem that is missing or unreadable raises; in directory mode
    unreadable files are skipped and listed in `unreadable`. A solution index
    seen twice keeps its first entry and counts in `duplicate_indices`."""
    out = KernelIndex()
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
        out.files.append(path.name)
        for sol in data.get("solutions", []) or []:
            gi = sol.get("index")
            if gi is None:
                continue
            if int(gi) in out.by_index:
                out.duplicate_indices += 1
                continue
            out.by_index[int(gi)] = kernel_dat_info(sol)
    return out
