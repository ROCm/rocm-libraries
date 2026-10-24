# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Static resource / occupancy primitive - reads a compiled HSACO's ELF notes.

Kernel-agnostic and **needs no GPU**: every kernel compiles to an HSACO whose
AMDGPU metadata notes record VGPR/AGPR/SGPR/LDS + spills. We read them with
`llvm-readelf --notes` and compute a coarse VGPR-limited occupancy estimate.

Reuses the note-field regexes proven in rocke's `probe_occupancy`. The occupancy
number is conservative (relative ranking, not an absolute prediction) - for exact
occupancy use rocprofv3 at runtime.

Returns a dict for the record's `resources` section. Uses the standard library
plus rocke's own target-name parsing.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rocke.core.arch import target_id_from_isa

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
# (`amdhsa.target: amdgcn-amd-amdhsa--gfx1201`; newer compilers write the empty
# environment field as `unknown`, `amdgcn-amd-amdhsa-unknown-gfx1201`). That beats
# any caller-supplied arch: the occupancy model must follow the binary, not a stale
# CLI flag. ``target_id_from_isa`` reads the target ID out of either form. The
# processor name may be a generic target (`gfx11-generic`); feature suffixes
# (`:xnack-`) are not part of it. An empty value is captured too, so it is refused
# like any other target without a processor.
_TARGET_FIELD = r"amdhsa\.target:[ \t]*['\"]?([^\s'\"]*)"
_PROCESSOR = r"gfx[0-9a-z]+(?:-[0-9a-z]+)*"


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


def _processor_from_target(isa: str) -> str:
    """The processor named by an ``amdhsa.target`` value.

    Raises ``ValueError`` when no processor can be read from it, so a target
    the parser does not understand is reported instead of being replaced by
    the caller's arch.
    """
    processor = target_id_from_isa(isa).split(":", 1)[0]
    if not re.fullmatch(_PROCESSOR, processor):
        raise ValueError(f"cannot read a processor from amdhsa.target {isa!r}")
    return processor


def parse_notes(hsaco_bytes: bytes) -> dict:
    """Extract raw resource fields from an HSACO's ELF notes ({} on failure).

    Raises ``ValueError`` when the notes record an ``amdhsa.target`` that names
    no processor.
    """
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
        fields["target"] = _processor_from_target(m.group(1))
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
# docs. ``combined_reg_pool`` selects how the multi-limiter model counts AGPRs: True
# adds them to the VGPRs against one file (gfx950); False gives them a separate pool
# of ``agpr_per_simd`` and the tighter pool binds (gfx942, gfx90a). The compiler
# treats all three as one shared file (see ``_VGPR_FILES``); the multi-limiter AGPR
# counting has not been reconciled with that.
#
# gfx950 combined-pool was calibrated against gfx950 hardware ``MeanOccupancyPerCU``:
# an AGPR-heavy kernel (44 VGPR + 132 AGPR) measured ~7 waves/CU, which the separate
# 256-AGPR pool mispredicted as 4 and the combined model gets as 8 (measured is a
# time-average; peak ~7.2). That calibration fed the model the architectural VGPR
# count (44). A real code object records ``.vgpr_count`` 176, which already includes
# the AGPRs, so the combined model adds the AGPRs twice and gives 4 waves/CU where
# the compiler reports 8; this is not fixed yet.
#
# The VGPR file fields (``vgpr_per_simd``, ``granularity``, ``waves_per_simd``) are
# the compiler's: VGPRs are allocated in steps of 8 from 512 per SIMD, at most 8
# waves, on all three processors, identically at clang 20, 22, 23 and 24.
# test_occupancy.py pins them and re-checks them against the local llc. The AGPR,
# combined-pool, LDS and workgroup fields are not checked against the compiler.
# Once validated these belong in ``core/arch/data/arch_specs.json`` (the hardware
# SSOT), read via ArchTarget.
_ARCH_CAPS = {
    "gfx950": dict(
        lds_per_cu=163840,
        max_waves_per_cu=32,
        vgpr_per_simd=512,
        agpr_per_simd=256,
        granularity=8,
        waves_per_simd=8,
        simds_per_cu=4,
        combined_reg_pool=True,
    ),
    "gfx942": dict(
        lds_per_cu=65536,
        max_waves_per_cu=32,
        vgpr_per_simd=512,
        agpr_per_simd=256,
        granularity=8,
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
# CDNA processor's VGPR file as recorded in ``_ARCH_CAPS``. On these CDNA processors
# the AGPRs live in the same file as the VGPRs, and the code object's
# ``.vgpr_count`` already counts both (the VGPRs rounded up to 4, plus the AGPRs),
# so the VGPR file alone gives the compiler's wave count with AGPRs in use too.
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
        # Separate VGPR and AGPR pools (see ``combined_reg_pool``); the tighter binds.
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
        # regs branch always claims it first. Kept so that caps which ever break
        # that equality still report the hardware wave cap as the limiter.
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
    or for the code object's wave size, when the notes do not record the wave
    size, or when they record a target that names no processor.
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
