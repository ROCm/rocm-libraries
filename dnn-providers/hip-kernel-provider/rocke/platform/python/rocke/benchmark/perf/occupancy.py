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
    "max_flat_workgroup_size": r"\.max_flat_workgroup_size:\s+(\d+)",
}

# The code object records the ISA it was built for
# (`amdhsa.target: amdgcn-amd-amdhsa--gfx1201`). That beats any caller-supplied
# arch: the occupancy model must follow the binary, not a stale CLI flag.
_TARGET_FIELD = r"amdhsa\.target:\s*\S*?--(gfx[0-9a-z]+)"

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
                capture_output=True,
                text=True,
                timeout=30,
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
    m = re.search(_TARGET_FIELD, notes)
    if m:
        fields["target"] = m.group(1)
    return fields


def _occupancy_estimate(vgpr: int, arch: str) -> Optional[int]:
    """Coarse VGPR-limited waves/SIMD (relative ranking, not absolute)."""
    caps = dict(_CAPS.get(_family(arch), {}))
    if (arch or "").split(":", 1)[0] == "gfx90a":
        caps["granularity"] = 8
    if not caps or vgpr <= 0:
        return None
    alloc = -(-vgpr // caps["granularity"]) * caps["granularity"]  # round up
    if alloc <= 0:
        return caps["max_waves_per_simd"]
    return min(caps["vgpr_per_simd"] // alloc, caps["max_waves_per_simd"])


def _align_up(n: int, gran: int) -> int:
    return ((n + gran - 1) // gran) * gran


# Full per-arch caps for the multi-limiter model below (a superset of the VGPR-only
# ``_CAPS`` above). Values lifted from the ``probe_occupancy`` tool and the arch
# docs. ``combined_reg_pool`` = arch VGPRs and accumulator VGPRs share one register
# file, so occupancy is set by their SUM (CDNA4/gfx950), not the tighter of two
# separate pools (CDNA3/gfx942, CDNA2/gfx90a).
#
# gfx950 combined-pool is CALIBRATED against gfx950 hardware ``MeanOccupancyPerCU``: an
# AGPR-heavy kernel (44 VGPR + 132 AGPR) measured ~7 waves/CU, which the separate
# 256-AGPR pool mispredicted as 4 and the combined model gets as 8 (measured is a
# time-average; peak ~7.2). gfx942/gfx90a remain PROVISIONAL: their waves_per_simd
# (8) and granularity are conservative CDNA2-era figures (CDNA3 may be 10 / 8) and
# await their own rocprofv3 calibration. Once validated these belong in
# ``core/arch/data/arch_specs.json`` (the hardware SSOT), read via ArchTarget.
_ARCH_CAPS = {
    "gfx950": dict(
        lds_per_cu=163840,
        max_waves_per_cu=32,
        vgpr_per_simd=512,
        agpr_per_simd=256,
        granularity=16,
        waves_per_simd=8,
        simds_per_cu=4,
        combined_reg_pool=True,
    ),
    "gfx942": dict(
        lds_per_cu=65536,
        max_waves_per_cu=32,
        vgpr_per_simd=512,
        agpr_per_simd=256,
        granularity=16,
        waves_per_simd=8,
        simds_per_cu=4,
        combined_reg_pool=False,
    ),
    "gfx90a": dict(
        lds_per_cu=65536,
        max_waves_per_cu=32,
        vgpr_per_simd=512,
        agpr_per_simd=256,
        granularity=8,
        waves_per_simd=8,
        simds_per_cu=4,
        combined_reg_pool=False,
    ),
}


def estimate_occupancy_detail(
    hsaco_bytes: bytes, arch: str, *, waves_per_wg: Optional[int] = None
) -> dict:
    """Multi-limiter static occupancy from a compiled HSACO (no GPU).

    Unlike ``_occupancy_estimate`` (VGPR only), this takes the minimum across the
    VGPR, AGPR, LDS, workgroup and hardware-wave limiters -- the same model the
    ``probe_occupancy`` tool uses -- and reports the binding one. Reads the ELF
    notes (incl. ``max_flat_workgroup_size`` for the per-workgroup wave count when
    ``waves_per_wg`` is not supplied). Returns ``{}`` if the notes cannot be read or
    ``arch`` has no caps entry.

    Keys: ``waves_per_simd`` (achieved), ``waves_per_cu``, ``wgs_per_cu``,
    ``limited_by`` in {VGPR, AGPR, "VGPR+AGPR", LDS, WAVES_PER_CU} ("VGPR+AGPR"
    only on ``combined_reg_pool`` arches).
    """
    caps = _ARCH_CAPS.get((arch or "").split(":", 1)[0])
    if caps is None:
        return {}
    f = parse_notes(hsaco_bytes)
    if not f:
        return {}
    vgpr = max(f.get("vgpr", 0), 1)
    agpr = f.get("agpr", 0)
    lds = max(f.get("lds_bytes", 0), 1)
    if waves_per_wg is None:
        wg = f.get("max_flat_workgroup_size", 0)
        waves_per_wg = max(wg // 64, 1) if wg else 1

    if caps.get("combined_reg_pool"):
        # CDNA4: VGPRs + AGPRs come from one file, so their SUM sets the wave count.
        waves_reg = min(
            caps["vgpr_per_simd"] // _align_up(vgpr + agpr, caps["granularity"]),
            caps["waves_per_simd"],
        )
        reg_label = "VGPR+AGPR" if agpr else "VGPR"
    else:
        # CDNA2/3: separate VGPR and AGPR pools; the tighter one binds.
        waves_vgpr = min(
            caps["vgpr_per_simd"] // _align_up(vgpr, caps["granularity"]),
            caps["waves_per_simd"],
        )
        if agpr:
            waves_agpr = min(
                caps["agpr_per_simd"] // _align_up(agpr, caps["granularity"]),
                caps["waves_per_simd"],
            )
        else:
            waves_agpr = caps["waves_per_simd"]
        waves_reg = min(waves_vgpr, waves_agpr)
        reg_label = "AGPR" if waves_agpr < waves_vgpr else "VGPR"
    wgs_from_lds = caps["lds_per_cu"] // lds

    per_cu_from_regs = waves_reg * caps["simds_per_cu"]
    per_cu_from_lds = wgs_from_lds * waves_per_wg
    waves_per_cu = min(per_cu_from_regs, per_cu_from_lds, caps["max_waves_per_cu"])

    if (
        per_cu_from_regs <= per_cu_from_lds
        and per_cu_from_regs <= caps["max_waves_per_cu"]
    ):
        limited_by = reg_label
    elif per_cu_from_lds <= caps["max_waves_per_cu"]:
        limited_by = "LDS"
    else:
        limited_by = "WAVES_PER_CU"

    return {
        "waves_per_simd": waves_per_cu // caps["simds_per_cu"],
        "waves_per_cu": waves_per_cu,
        "wgs_per_cu": waves_per_cu // max(waves_per_wg, 1),
        "limited_by": limited_by,
    }


def resources(hsaco_bytes: bytes, arch: str) -> dict:
    """The record's `resources` section for a compiled HSACO (no GPU).

    Keys: vgpr, agpr, sgpr, lds_bytes, vgpr_spill, sgpr_spill, target_arch,
    occupancy. Empty dict if the notes couldn't be read.

    `arch` is only a fallback: when the code object names its own target, that wins,
    so passing a stale `--arch` can no longer produce a wave count for the wrong
    family. `target_arch` reports what was actually used, so a caller can flag the
    disagreement.
    """
    f = parse_notes(hsaco_bytes)
    if not f:
        return {}
    target = f.get("target", "")
    effective_arch = target or (arch or "").split(":", 1)[0]
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
        "target_arch": effective_arch,
    }
    res["occupancy"] = _occupancy_estimate(res["vgpr"], effective_arch)
    return res
