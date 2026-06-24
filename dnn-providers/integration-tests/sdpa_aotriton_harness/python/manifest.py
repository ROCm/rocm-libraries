"""Manifest schema, file-path conventions, and shared numpy helpers.

A *run directory* contains one subdirectory per case plus an ``index.json``
listing every case. Each case subdirectory holds the input ``.npy`` files, the
output ``.npy`` files produced by the driver/reference paths, and a ``manifest.json``
describing the case (the fields of :class:`sdpa_cases.Case` plus a ``files``
block and a mutable ``status`` block).

Only numpy is required here; this module never imports torch, so ``compare.py``
and ``harness.py`` can use it without a torch install.

=== Integration contract (the C++ driver obeys the same; keep in sync) ===
.npy files are standard NumPy v1.0, little-endian, C-contiguous.
  fp32 -> '<f4', fp16 -> '<f2', bf16 -> '<u2' (raw 16-bit bf16 bit patterns),
  fp8 (e4m3/e5m2 and their fnuz variants) -> '|u1' (raw 8-bit fp8 bit patterns).
  Q: [B, Hq, Sq, D]; K, V: [B, Hkv, Skv, D];
  mask (optional): fp32 '<f4' full rank-4 [B, Hq, Sq, Skv] (additive bias,
  no broadcasting); O (output): fp32 '<f4' [B, Hq, Sq, D].
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

# numpy dtype string for each on-disk element type.
NPY_DTYPE = {
    "fp32": "<f4",
    "fp16": "<f2",
    "bf16": "<u2",  # raw bf16 bit patterns stored as uint16
    # fp8 formats: raw 8-bit bit patterns stored as uint8 (numpy emits '|u1').
    "fp8_e4m3": "|u1",
    "fp8_e5m2": "|u1",
    "fp8_e4m3_fnuz": "|u1",
    "fp8_e5m2_fnuz": "|u1",
}

# Filenames within a case directory.
FILE_NAMES = {
    "q": "q.npy",
    "k": "k.npy",
    "v": "v.npy",
    "mask": "mask.npy",
    "gpuref_o": "gpuref_o.npy",
    "gpuref_lse": "gpuref_lse.npy",
    "reference_o": "reference_o.npy",
    "math_hp_o": "math_hp_o.npy",
    "math_lp_o": "math_lp_o.npy",
}


def case_dir(run_dir: str, name: str) -> str:
    """Absolute directory holding the files for a single case."""
    return os.path.join(run_dir, name)


def case_files(run_dir: str, name: str, *, has_mask: bool) -> Dict[str, Optional[str]]:
    """Build the ``files`` block (absolute paths) for a case manifest.

    ``mask`` and ``gpuref_lse`` are optional: ``mask`` is ``None`` unless the
    case provides one; ``gpuref_lse`` is always populated as a path but the
    harness may choose not to request it from the driver.
    """
    cdir = case_dir(run_dir, name)
    files: Dict[str, Optional[str]] = {
        key: os.path.join(cdir, FILE_NAMES[key])
        for key in (
            "q",
            "k",
            "v",
            "gpuref_o",
            "gpuref_lse",
            "reference_o",
            "math_hp_o",
            "math_lp_o",
        )
    }
    files["mask"] = os.path.join(cdir, FILE_NAMES["mask"]) if has_mask else None
    return files


def write_manifest(run_dir: str, manifest: Dict[str, Any]) -> str:
    """Write a single case manifest to its case directory; return the path."""
    cdir = case_dir(run_dir, manifest["name"])
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return path


def read_manifest(run_dir: str, name: str) -> Dict[str, Any]:
    """Read a single case manifest by case name."""
    path = os.path.join(case_dir(run_dir, name), "manifest.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_index(run_dir: str, names: List[str], tier: str) -> str:
    """Write ``index.json`` listing all case names for a run."""
    path = os.path.join(run_dir, "index.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"tier": tier, "cases": names}, fh, indent=2)
    return path


def read_index(run_dir: str) -> Dict[str, Any]:
    """Read ``index.json`` from a run directory."""
    path = os.path.join(run_dir, "index.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def manifest_from_case(case: Any, run_dir: str) -> Dict[str, Any]:
    """Build a full manifest dict from a :class:`sdpa_cases.Case`.

    ``case`` is accepted structurally (anything with the Case fields) to avoid a
    hard import cycle. ``asdict`` works on dataclass instances.
    """
    base = asdict(case)
    base["files"] = case_files(run_dir, case.name, has_mask=case.has_mask)
    base["status"] = {"state": "pending"}  # updated by run_torch / harness
    return base


def synthesize_window_mask(
    Sq: int,
    Skv: int,
    left: int,
    right: int,
    top_left: bool,
) -> np.ndarray:
    """Build the [Sq, Skv] additive float32 mask equivalent to gpu_ref window bounds.

    Replicates GpuRefSdpaFwd exactly:

        windowOffset = top_left ? 0 : (Skv - Sq)
        position (sq, skv) is MASKED (-inf) iff
            (right >= 0 AND skv >= max(sq + 1 + windowOffset + right, 0))
            OR (left  >= 0 AND skv <  sq + windowOffset - left)
        KEPT (0.0) otherwise.

    Returns a [Sq, Skv] float32 array of 0.0 / -inf, ready to broadcast to
    [B, Hq, Sq, Skv].
    """
    window_offset = 0 if top_left else (Skv - Sq)
    mask = np.zeros((Sq, Skv), dtype=np.float32)
    neg_inf = np.float32(-math.inf)

    sq = np.arange(Sq).reshape(Sq, 1)
    skv = np.arange(Skv).reshape(1, Skv)

    masked = np.zeros((Sq, Skv), dtype=bool)
    if right >= 0:
        start_kv = np.maximum(sq + 1 + window_offset + right, 0)
        masked |= skv >= start_kv
    if left >= 0:
        masked |= skv < (sq + window_offset - left)

    mask[masked] = neg_inf
    return mask
