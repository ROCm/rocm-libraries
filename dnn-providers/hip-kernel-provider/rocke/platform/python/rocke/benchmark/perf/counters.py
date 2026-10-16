# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""PMU counter probing + per-arch normalization (a primitive - pure-ish).

Different AMD architectures expose different hardware counters with different raw
names. Never hardcode a counter list: **probe** what the current GPU supports
(`rocprofv3 --list-avail`), intersect with what we want, and map raw arch-specific
names to stable *normalized* names so downstream code (and any consumer) is
arch-independent. A kernel is only ever compared against its own baseline on the
same arch, so missing counters on one arch (null) are fine.

`parse_list_avail` / `select` / `wanted_map` / `group_counters` are pure (testable
without a GPU); `discover` runs the profiler. Stdlib only.
"""
from __future__ import annotations

import re
import subprocess

# Hardware counter-slot budget per block: how many counters that block can drive in
# ONE rocprofv3 pass. Counters from DIFFERENT blocks share a pass freely; only
# same-block counters compete for that block's slots. Collecting counters in one
# pass = one kernel execution, so they are mutually consistent (e.g. busy_cycles and
# wait_cycles from the same run). The original 9-counter set fits one pass;
# the current 14-counter CDNA selection needs two, and the 11-counter RDNA
# selection fits one. Budgets are conservative (see `group_counters`).
_BLOCK_SLOTS: dict[str, int] = {
    "GRBM": 2,
    "SQ": 8,
    "TCC": 4,
    "GL2C": 4,  # L2 is TCC on CDNA, GL2C on RDNA
    "TCP": 4,
    "TA": 4,
    "TD": 4,
    "CPC": 2,
    "CPF": 2,
}
_DEFAULT_SLOTS = 2  # unknown block: assume a small budget

# Counters requested on every arch. GRBM (total_clocks, busy_cycles), SQ_BUSY_CYCLES,
# and SQ_WAVES populate on both families; SQ_WAIT_ANY populates on CDNA but reads 0
# on gfx1201/RDNA4 (the same SQ gap as the instruction counters below).
_COMMON: dict[str, str] = {
    "total_clocks": "GRBM_COUNT",
    "busy_cycles": "GRBM_GUI_ACTIVE",  # primary regression metric
    "sq_busy_cycles": "SQ_BUSY_CYCLES",
    "waves": "SQ_WAVES",
    "wait_cycles": "SQ_WAIT_ANY",
}

# Per-family names. RDNA is wave32 (SQ_INSTS_WAVE32_*) and L2 = GL2C_*; CDNA is
# wave64 (SQ_INSTS_*) and L2 = TCC_*. select() intersects with the probed set, so
# an unavailable name drops out.
#
# ARCH COVERAGE (verified on-box):
#  - gfx1201/RDNA4 (2026-07): the instruction (VALU/LDS) and L2 counters below
#    return 0 for real kernels even though the names exist in --list-avail - a
#    rocprofv3/RDNA4 support gap, not a parse bug. Only clock/wave counters (above)
#    populate. --list-avail offers 58 counters total.
#  - gfx950/CDNA (2026-07, MI355X): the wave64 / TCC counters DO populate.
#  - gfx90a/CDNA2 (2026-09): all of the below populate, including the LDS-conflict
#    and VALU-activity counters. --list-avail offers 533 counters.
# So the diagnostic panel is expected to fill on CDNA and be clock/wave-only on
# RDNA4. `captured_counters` lists counter names present in the record, including
# counters that read zero; it does not promise meaningful hardware coverage.
#
# The lds_bank_conflict / lds_idx_active / valu_active_cycles / cu_busy_cycles
# counters are ratio INPUTS, not metrics: a consumer divides them pairwise (see
# `derived` in the record). On CDNA they are SQ-block counters, pushing SQ past
# its 8-slot budget, so `group_counters` uses two passes while keeping ratio
# inputs together. RDNA's LDS pair is in SQC and its selection fits one pass.
_BY_FAMILY: dict[str, dict[str, str]] = {
    "rdna": {  # gfx10/11/12 - wave32
        "valu_insts": "SQ_INSTS_WAVE32_VALU",
        "lds_insts": "SQ_INSTS_WAVE32_LDS",
        "l2_hit": "GL2C_HIT",
        "l2_miss": "GL2C_MISS",
        # RDNA exposes the LDS pair under the SQC (cache) block, not SQ.
        "lds_bank_conflict": "SQC_LDS_BANK_CONFLICT",
        "lds_idx_active": "SQC_LDS_IDX_ACTIVE",
    },
    "cdna": {  # gfx9x - wave64 (verified on gfx950/MI355X and gfx90a)
        "valu_insts": "SQ_INSTS_VALU",
        "lds_insts": "SQ_INSTS_LDS",
        "l2_hit": "TCC_HIT",
        "l2_miss": "TCC_MISS",
        "lds_bank_conflict": "SQ_LDS_BANK_CONFLICT",
        "lds_idx_active": "SQ_LDS_IDX_ACTIVE",
        "valu_active_cycles": "SQ_ACTIVE_INST_VALU",
        "cu_busy_cycles": "SQ_BUSY_CU_CYCLES",
        # Matrix-instruction count. Distinguishes "VALU is idle" from "this is a
        # matrix kernel whose VALU is idle by design" - the denominator a consumer
        # needs before calling low VALU utilization a problem. CDNA-only: RDNA
        # exposes no MFMA/WMMA counter at all (gfx1201 --list-avail: zero matches).
        "mfma_insts": "SQ_INSTS_MFMA",
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


def _block_of(raw: str) -> str:
    """Hardware block a raw counter belongs to (e.g. SQ_INSTS_VALU -> 'SQ')."""
    for b in _BLOCK_SLOTS:
        if raw == b or raw.startswith(b + "_"):
            return b
    return raw.split("_", 1)[0]  # fallback: leading token


def group_counters(
    raws: "list[str]", keep_together: "list[list[str]] | None" = None
) -> "list[list[str]]":
    """Pack raw counters into minimal single-pass groups honoring per-block slots.

    Counters from different blocks share a pass; same-block counters are chunked by
    that block's slot budget (`_BLOCK_SLOTS`). Returns a list of groups, each a list
    of raw names = one rocprofv3 pass = one kernel replay. Counters within a group
    are collected in a single execution, so they are mutually consistent (e.g.
    `busy_cycles`/`wait_cycles`, or `l2_hit`/`l2_miss`, from the same run). Fewer
    groups = fewer replays.

    `keep_together` names sets that MUST land in one group - the counters of a
    ratio, which is meaningless across executions (dividing one run's numerator by
    another run's denominator). Overlapping constraints form one indivisible
    unit, so an overflowing block splits *between* units, never through one.
    A connected unit larger than its block budget cannot be satisfied and raises
    rather than silently emitting an incoherent ratio.
    """
    pinned: list[dict[str, None]] = []
    for unit in keep_together or []:
        members = dict.fromkeys(r for r in unit if r in raws)
        if len(members) < 2:
            continue
        merged = None
        for prior in pinned:
            if not any(r in members for r in prior):
                continue
            if merged is None:
                merged = prior
            else:
                merged.update(prior)
                prior.clear()
        if merged is None:
            pinned.append(members)
        else:
            merged.update(members)
    seen = {r for unit in pinned for r in unit}
    # Every remaining counter is its own unit; a unit is what packing cannot split.
    units = [list(unit) for unit in pinned if unit]
    units.extend([r] for r in raws if r not in seen)

    by_block: dict[str, list[list[str]]] = {}
    for u in units:
        blocks = {_block_of(r) for r in u}
        if len(blocks) > 1:
            raise ValueError(
                f"counters {u} must share a pass but span blocks {sorted(blocks)}; "
                "cross-block ratio inputs are not supported"
            )
        by_block.setdefault(blocks.pop(), []).append(u)

    block_chunks: list[list[list[str]]] = []
    for block, block_units in by_block.items():
        lim = _BLOCK_SLOTS.get(block, _DEFAULT_SLOTS)
        chunks: list[list[str]] = []
        cur: list[str] = []
        for u in block_units:
            if len(u) > lim:
                raise ValueError(
                    f"counters {u} must share a pass but block {block} has only "
                    f"{lim} slot(s)"
                )
            if len(cur) + len(u) > lim:
                chunks.append(cur)
                cur = []
            cur.extend(u)
        if cur:
            chunks.append(cur)
        block_chunks.append(chunks)

    n_passes = max((len(c) for c in block_chunks), default=0)
    groups: list[list[str]] = []
    for i in range(n_passes):
        g: list[str] = []
        for chunks in block_chunks:
            if i < len(chunks):
                g.extend(chunks[i])
        if g:
            groups.append(g)
    return groups


# Ratio metrics: normalized name -> (numerator, denominator). Each is a pure
# function of counters this module already selects, so a consumer never has to
# know arch-specific raw names or which hardware block they came from. Defined
# here (not in the harness) because a single run and an aggregate of runs must
# derive them identically - the aggregate recomputes from median counters.
_RATIOS: dict[str, tuple[str, str]] = {
    "busy_fraction": ("busy_cycles", "total_clocks"),
    "valu_utilization": ("valu_active_cycles", "cu_busy_cycles"),
    # Fraction of vector instructions that are matrix ops. Low VALU utilization is
    # expected on a matrix kernel, so this is the guard a consumer checks before
    # calling idle VALU a problem. Declaring it here also pins its two inputs into
    # one pass, which a cross-pass quotient would otherwise make meaningless.
    "matrix_share": ("mfma_insts", "valu_insts"),
    "lds_bank_conflict_rate": ("lds_bank_conflict", "lds_idx_active"),
}

# Ratios whose denominator is the SUM of both counters (hit/(hit+miss) style).
_SHARE_RATIOS: dict[str, tuple[str, str]] = {
    "l2_hit_rate": ("l2_hit", "l2_miss"),
}


def derive(counters: "dict[str, float]") -> "dict[str, float]":
    """Ratio metrics computable from the counters actually captured.

    A ratio is emitted only when both inputs are present and the denominator is
    nonzero, so a partial capture yields fewer metrics rather than a wrong or
    zero-division one. Values are ratios, not percentages, and may exceed one.
    """
    out: dict[str, float] = {}
    for name, (num, den) in _RATIOS.items():
        n, d = counters.get(num), counters.get(den)
        if n is not None and d:
            out[name] = n / d
    for name, (num, other) in _SHARE_RATIOS.items():
        n, o = counters.get(num), counters.get(other)
        if n is not None and o is not None and (n + o) > 0:
            out[name] = n / (n + o)
    return out


def ratio_units(sel: "dict[str, str]") -> "list[list[str]]":
    """Raw-name sets that must share one pass, given a normalized->raw selection.

    Pass to `group_counters(raws, keep_together=ratio_units(sel))` so a ratio's
    inputs are never split across executions. Pairs whose counters this arch did
    not capture are omitted.
    """
    units: list[list[str]] = []
    for num, den in list(_RATIOS.values()) + list(_SHARE_RATIOS.values()):
        pair = [sel[k] for k in (num, den) if k in sel]
        if len(pair) == 2:
            units.append(pair)
    return units


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
