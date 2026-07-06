# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""PMU counter probing + per-arch normalization (a primitive - pure-ish).

Different AMD architectures expose different hardware counters with different raw
names. Never hardcode a counter list: **probe** what the current GPU supports
(`rocprofv3 --list-avail`), intersect with what we want, and map raw arch-specific
names to stable *normalized* names so downstream code (and any consumer) is
arch-independent. A kernel is only ever compared against its own baseline on the
same arch, so missing counters on one arch (null) are fine.

`parse_list_avail` / `select` / `wanted_map` are pure (testable with a saved dump);
`discover` runs the profiler. Stdlib only.
"""
from __future__ import annotations

import re
import subprocess

# Clock/wave counters that populate on both families (verified nonzero on gfx1201).
_COMMON: dict[str, str] = {
    "total_clocks": "GRBM_COUNT",
    "busy_cycles": "GRBM_GUI_ACTIVE",   # primary regression metric
    "sq_busy_cycles": "SQ_BUSY_CYCLES",
    "waves": "SQ_WAVES",
    "wait_cycles": "SQ_WAIT_ANY",
}

# Per-family names. RDNA is wave32 (SQ_INSTS_WAVE32_*) and L2 = GL2C_*; CDNA is
# wave64 (SQ_INSTS_*) and L2 = TCC_*. select() intersects with the probed set, so
# an unavailable name drops out.
#
# ARCH COVERAGE (both verified on-box 2026-07):
#  - gfx1201/RDNA4: the instruction (VALU/LDS) and L2 counters below return 0 for
#    real kernels even though the names exist in --list-avail - a rocprofv3/RDNA4
#    support gap, not a parse bug. Only clock/wave counters (above) populate.
#  - gfx950/CDNA (the CI benchmark target, MI355X): the wave64 / TCC counters DO
#    populate. Confirmed with a memory+VALU probe kernel: SQ_INSTS_VALU, TCC_HIT,
#    TCC_MISS all nonzero; SQ_INSTS_LDS was 0 only because that kernel used no LDS.
# So the diagnostic panel is expected to fill on CDNA and be clock/wave-only on
# RDNA4. `captured_counters` in the record surfaces exactly which populated, so a
# record never overstates coverage.
_BY_FAMILY: dict[str, dict[str, str]] = {
    "rdna": {  # gfx10/11/12 - wave32
        "valu_insts": "SQ_INSTS_WAVE32_VALU",
        "lds_insts": "SQ_INSTS_WAVE32_LDS",
        "l2_hit": "GL2C_HIT",
        "l2_miss": "GL2C_MISS",
    },
    "cdna": {  # gfx9x - wave64 (verified on gfx950/MI355X)
        "valu_insts": "SQ_INSTS_VALU",
        "lds_insts": "SQ_INSTS_LDS",
        "l2_hit": "TCC_HIT",
        "l2_miss": "TCC_MISS",
    },
}


def _family(arch: str) -> str:
    """'cdna' for gfx9xx, else 'rdna' (gfx10/11/12). Default 'rdna'."""
    m = re.match(r"gfx(\d+)", arch or "")
    if m and m.group(1).startswith("9"):
        return "cdna"
    return "rdna"


def wanted_map(arch: str) -> dict[str, str]:
    """normalized -> raw counter names we *want* on this arch (pre-probe)."""
    m = dict(_COMMON)
    m.update(_BY_FAMILY[_family(arch)])
    return m


def parse_list_avail(text: str) -> set[str]:
    """Extract available raw counter names from `rocprofv3 --list-avail` output.

    rocprofv3's format varies by ROCm version, so match on the field *name* rather
    than a fixed prefix:
      - older builds print ``Name:\\t<COUNTER>`` per counter (gfx1201 dev box);
      - ROCm 7.2 prints ``Counter_Name        :\\t<COUNTER>`` (padded), and uses
        ``Name`` only for the GPU/arch line.
    We accept both ``Name`` and ``Counter_Name`` keys; a stray arch value (e.g.
    ``gfx950``) is harmless because `select` intersects with the wanted set.
    """
    names: set[str] = set()
    for line in text.splitlines():
        head, sep, val = line.partition(":")
        if not sep:
            continue
        if head.strip() in ("Name", "Counter_Name"):
            name = val.strip()
            if name:
                names.add(name)
    return names


def select(arch: str, available: set[str]) -> dict[str, str]:
    """normalized -> raw for counters we want AND the GPU actually supports."""
    return {norm: raw for norm, raw in wanted_map(arch).items() if raw in available}


def discover(arch: str, *, timeout: int = 60) -> dict[str, str]:
    """Probe the live GPU (`rocprofv3 --list-avail`) -> normalized->raw selection.

    Returns {} if rocprofv3 is missing or errors, so a caller can degrade to
    wall-time-only.
    """
    try:
        proc = subprocess.run(
            ["rocprofv3", "--list-avail"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}
    return select(arch, parse_list_avail(proc.stdout or ""))
