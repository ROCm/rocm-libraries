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
from dataclasses import dataclass
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
    "wave_size": r"\.wavefront_size:\s+(\d+)",
}

# The code object records the ISA it was built for
# (`amdhsa.target: amdgcn-amd-amdhsa--gfx1201`). That beats any caller-supplied
# arch: the occupancy model must follow the binary, not a stale CLI flag. The
# processor name may be a generic target (`gfx11-generic`); feature suffixes
# (`:xnack-`) are not part of it.
_TARGET_FIELD = r"amdhsa\.target:\s*\S*?--(gfx[0-9a-z-]+)"


@dataclass(frozen=True)
class _VgprFile:
    """One SIMD's VGPR file, as the compiler's occupancy calculation sees it.

    The waves resident on a SIMD share ``vgprs_per_simd`` registers. Each wave's
    VGPR count is rounded up to a multiple of ``granule``, so a kernel using ``v``
    VGPRs fits ``vgprs_per_simd // round_up(v, granule)`` waves, capped at
    ``max_waves_per_simd``. The numbers hold for code built at ``wave_size``.
    """

    wave_size: int
    vgprs_per_simd: int
    granule: int
    max_waves_per_simd: int


# The register file differs between processors of one family (gfx1100 has 1.5x the
# VGPRs of gfx1102), so the model is keyed by processor, never by family. A generic
# target takes the values of its smallest member: its code runs on every member,
# and the compiler budgets occupancy for that member. These RDNA values match the
# compiler's own occupancy report (llc's "; Occupancy:" comment) for every VGPR
# count from 1 to 256 at wave32; test_occupancy.py pins them and re-checks them
# against the local llc. Wave64 code on these processors is not modelled. The CDNA
# register files come from ``_ARCH_CAPS`` below (see ``_VGPR_FILES``).
_LARGE_RDNA_VGPR_FILE = _VgprFile(
    wave_size=32, vgprs_per_simd=1536, granule=24, max_waves_per_simd=16
)
_RDNA_VGPR_FILE = _VgprFile(
    wave_size=32, vgprs_per_simd=1024, granule=16, max_waves_per_simd=16
)

_RDNA_VGPR_FILES = {
    "gfx1100": _LARGE_RDNA_VGPR_FILE,
    "gfx1101": _LARGE_RDNA_VGPR_FILE,
    "gfx1102": _RDNA_VGPR_FILE,
    "gfx1103": _RDNA_VGPR_FILE,
    "gfx1150": _RDNA_VGPR_FILE,
    "gfx1151": _LARGE_RDNA_VGPR_FILE,
    "gfx1152": _RDNA_VGPR_FILE,
    "gfx1153": _RDNA_VGPR_FILE,
    "gfx1154": _RDNA_VGPR_FILE,
    "gfx11-generic": _RDNA_VGPR_FILE,
    "gfx1200": _LARGE_RDNA_VGPR_FILE,
    "gfx1201": _LARGE_RDNA_VGPR_FILE,
    "gfx12-generic": _LARGE_RDNA_VGPR_FILE,
    "gfx1250": _RDNA_VGPR_FILE,
}


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


def _occupancy_estimate(vgpr: int, arch: str, wave_size: int) -> Optional[int]:
    """Coarse VGPR-limited waves/SIMD (relative ranking, not absolute).

    ``arch`` is a processor name (feature suffixes such as ``:xnack-`` are
    ignored) and ``wave_size`` the wave size the code was built for. Raises
    ``ValueError`` for a processor or wave size the model has no data for, rather
    than guessing from a neighbouring part. Returns ``None`` when ``vgpr`` is 0.
    """
    processor = (arch or "").split(":", 1)[0]
    vgpr_file = _VGPR_FILES.get(processor)
    if vgpr_file is None:
        raise ValueError(
            f"no occupancy model for processor {processor!r}; "
            f"the VGPR file is recorded for: {', '.join(_VGPR_FILES)}"
        )
    if wave_size != vgpr_file.wave_size:
        raise ValueError(
            f"no occupancy model for wave{wave_size} code on {processor}; "
            f"its VGPR file is recorded for wave{vgpr_file.wave_size} code"
        )
    if vgpr <= 0:
        return None
    alloc = _align_up(vgpr, vgpr_file.granule)
    return min(vgpr_file.vgprs_per_simd // alloc, vgpr_file.max_waves_per_simd)


def _align_up(n: int, gran: int) -> int:
    return ((n + gran - 1) // gran) * gran


# Full per-arch caps for the CDNA multi-limiter model below (adds the AGPR, LDS and
# workgroup limiters to the VGPR file). Values lifted from the ``probe_occupancy`` tool and the arch
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


# Every processor the VGPR-limited estimate covers: the RDNA files above, and each
# CDNA processor's VGPR file as recorded in ``_ARCH_CAPS``.
_VGPR_FILES = {
    **{
        processor: _VgprFile(
            wave_size=64,
            vgprs_per_simd=caps["vgpr_per_simd"],
            granule=caps["granularity"],
            max_waves_per_simd=caps["waves_per_simd"],
        )
        for processor, caps in _ARCH_CAPS.items()
    },
    **_RDNA_VGPR_FILES,
}


def estimate_occupancy_detail(
    hsaco_bytes: bytes, arch: str, *, waves_per_wg: Optional[int] = None
) -> dict:
    """Multi-limiter static occupancy from a compiled HSACO (no GPU).

    Unlike ``_occupancy_estimate`` (VGPR only), this takes the minimum across the
    VGPR, AGPR, LDS, workgroup and hardware-wave limiters and reports the binding
    one. It is the same *shape* of model as the ``probe_occupancy`` tool, but with
    the calibrated CDNA4 combined register pool -- ``probe_occupancy`` still models
    gfx950 as separate VGPR/AGPR pools and folds in a ``waves_per_eu`` hint, so the
    two disagree on AGPR-heavy gfx950 kernels until the caps move to
    ``arch_specs.json`` (see the caps note above; TODO: converge there).

    ``waves_per_wg``, when not supplied, is derived from ``max_flat_workgroup_size``
    -- the *permitted maximum* (256 by default when a kernel does not pin
    ``max_workgroup_size``), NOT the launch geometry. That biases the LDS limiter
    optimistic for kernels that never pinned it, so a caller that knows the block
    size should pass ``waves_per_wg`` explicitly.

    Returns ``{}`` if the required notes (``vgpr``, ``lds_bytes``) are missing or
    ``arch`` has no caps entry.

    Keys: ``waves_per_simd`` (achieved, ``= waves_per_cu // simds_per_cu``; floors
    to 0 for an LDS-bound kernel below 4 waves/CU -- gate on ``waves_per_cu`` if
    that distinction matters), ``waves_per_cu``, ``wgs_per_cu``, ``limited_by`` in
    {VGPR, AGPR, "VGPR+AGPR", LDS, WAVES_PER_CU} ("VGPR+AGPR" only on
    ``combined_reg_pool`` arches).
    """
    caps = _ARCH_CAPS.get((arch or "").split(":", 1)[0])
    if caps is None:
        return {}
    f = parse_notes(hsaco_bytes)
    # Require the fields the model consumes: a partial parse (metadata-shape drift,
    # a different readelf) would otherwise clamp a missing vgpr to 1 via max(..., 1)
    # and report the arch maximum, passing the floor vacuously.
    if not f or "vgpr" not in f or "lds_bytes" not in f:
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
        # Unreachable while waves_per_simd * simds_per_cu == max_waves_per_cu for
        # every arch (32 == 32): per_cu_from_regs can never exceed the cap, so the
        # regs branch always claims it first. Kept for a future CDNA3 recalibration
        # to 10 waves/SIMD (see the caps note above), which would break that tie.
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
    processor. `target_arch` reports what was actually used, so a caller can flag
    the disagreement.

    Raises ``ValueError`` when the occupancy model has no data for that processor
    or for the code object's wave size, or when the notes do not record the wave
    size.
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
    if "wave_size" not in f:
        raise ValueError(
            "the code object's notes do not record .wavefront_size, "
            "which the occupancy estimate needs"
        )
    res["occupancy"] = _occupancy_estimate(res["vgpr"], effective_arch, f["wave_size"])
    return res
