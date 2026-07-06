# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Static resource / occupancy primitive - reads a compiled HSACO's ELF notes.

Kernel-agnostic and **needs no GPU**: every kernel compiles to an HSACO whose
AMDGPU metadata notes record VGPR/AGPR/SGPR/LDS + spills. We read them with
`llvm-readelf --notes` and compute a coarse VGPR-limited occupancy estimate.

Reuses the note-field regexes proven in rocke's `probe_occupancy`. The occupancy
number is conservative (relative ranking, not an absolute prediction) - for exact
occupancy use rocprofv3 at runtime.

Returns a dict for the record's `resources` section. Stdlib only.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

# AMDGPU code-object metadata note fields (same patterns as probe_occupancy).
_NOTE_FIELDS = {
    "vgpr": r"\.vgpr_count:\s+(\d+)",
    "sgpr": r"\.sgpr_count:\s+(\d+)",
    "agpr": r"\.agpr_count:\s+(\d+)",
    "vgpr_spill": r"\.vgpr_spill_count:\s+(\d+)",
    "sgpr_spill": r"\.sgpr_spill_count:\s+(\d+)",
    "lds_bytes": r"\.group_segment_fixed_size:\s+(\d+)",
}

# Minimal per-arch caps for the coarse VGPR-limited occupancy estimate.
# vgpr_per_simd = physical VGPRs per SIMD; granularity = allocation quantum.
_CAPS = {
    "rdna": {"vgpr_per_simd": 1536, "granularity": 24, "max_waves_per_simd": 16},
    "cdna": {"vgpr_per_simd": 512, "granularity": 16, "max_waves_per_simd": 8},
}


def _family(arch: str) -> str:
    m = re.match(r"gfx(\d+)", arch or "")
    return "cdna" if (m and m.group(1).startswith("9")) else "rdna"


def _readelf() -> Optional[str]:
    for c in ("/opt/rocm/llvm/bin/llvm-readelf", "llvm-readelf", "readelf"):
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    return None


def parse_notes(hsaco_bytes: bytes) -> dict:
    """Extract raw resource fields from an HSACO's ELF notes ({} on failure)."""
    readelf = _readelf()
    if not readelf:
        return {}
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as f:
        f.write(hsaco_bytes)
        f.flush()
        try:
            out = subprocess.run(
                [readelf, "--notes", f.name],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
    if out.returncode != 0:
        return {}
    notes = out.stdout or ""
    fields: dict = {}
    for name, pat in _NOTE_FIELDS.items():
        m = re.search(pat, notes)
        if m:
            fields[name] = int(m.group(1))
    return fields


def _occupancy_estimate(vgpr: int, arch: str) -> Optional[int]:
    """Coarse VGPR-limited waves/SIMD (relative ranking, not absolute)."""
    caps = _CAPS.get(_family(arch))
    if not caps or vgpr <= 0:
        return None
    alloc = -(-vgpr // caps["granularity"]) * caps["granularity"]  # round up
    if alloc <= 0:
        return caps["max_waves_per_simd"]
    return min(caps["vgpr_per_simd"] // alloc, caps["max_waves_per_simd"])


def resources(hsaco_bytes: bytes, arch: str) -> dict:
    """The record's `resources` section for a compiled HSACO (no GPU).

    Keys: vgpr, agpr, sgpr, lds_bytes, vgpr_spill, sgpr_spill, occupancy.
    Empty dict if the notes couldn't be read.
    """
    f = parse_notes(hsaco_bytes)
    if not f:
        return {}
    res = {
        "vgpr": f.get("vgpr", 0),
        "agpr": f.get("agpr", 0),
        "sgpr": f.get("sgpr", 0),
        "lds_bytes": f.get("lds_bytes", 0),
        "vgpr_spill": f.get("vgpr_spill", 0),
        "sgpr_spill": f.get("sgpr_spill", 0),
        # Tag the origin: these are the compiler's raw counts from the ELF notes.
        # They can differ from the harness's rocprofv3 `resources` (source
        # "rocprofv3"), which reports the runtime's allocated/rounded figures
        # (e.g. SGPRs incl. reserved) - so a consumer never conflates the two.
        "source": "elf_notes",
    }
    res["occupancy"] = _occupancy_estimate(res["vgpr"], arch)
    return res
