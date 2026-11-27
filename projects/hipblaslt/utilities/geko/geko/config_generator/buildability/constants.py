# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Settings of the buildability search: the probe, the presets and base values every kernel starts from, and the
search limits."""

from geko.config_generator.fork_params.hw_profiles.gfx942.optimization_param import GFX942GAParams
from geko.config_generator.fork_params.hw_profiles.gfx950.optimization_param import GFX950GAParams
from geko.config_generator.fork_params.hw_profiles.gfx1250.optimization_param import GFX1250GAParams

# Any ordinary size: the generic ranges do not depend on it (M or N = 1 drops DepthU on gfx942).
PROBE_SIZE = [4096, 4096, 1, 4096]

# Settings tried first, per generic (GA) profile: DepthU "K" = the smallest legal DepthU, positive read widths in
# 2-byte elements. They only order the search; a profile without an entry starts from the factorial.
_GFX9_PRESETS = [
    {
        "DepthU": "K",
        "PrefetchGlobalRead": 2,
        "1LDSBuffer": 0,
        "ClusterLocalRead": 1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "DirectToLds": 1,
        "UseSgprForGRO": 0,
    },
    {
        "DepthU": "K",
        "PrefetchGlobalRead": 1,
        "1LDSBuffer": 1,
        "ClusterLocalRead": 1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "DirectToLds": 0,
        "UseSgprForGRO": 0,
    },
    {
        "DepthU": "K",
        "PrefetchGlobalRead": 1,
        "1LDSBuffer": 0,
        "ClusterLocalRead": 1,
        "GlobalReadVectorWidthA": 2,
        "GlobalReadVectorWidthB": 2,
        "DirectToLds": 1,
        "UseSgprForGRO": 0,
    },
]
LIST_PRESETS = {
    GFX950GAParams: _GFX9_PRESETS,
    GFX942GAParams: _GFX9_PRESETS,
    # HalfPLR x ClusterLocalRead: the only register levers gfx1250's space tunes (sweeps over 220 MIs, 6 non-FP4 types).
    GFX1250GAParams: [{"DepthU": "K", "HalfPLR": h, "ClusterLocalRead": c} for c in (0, 1) for h in (0, 2, 1, 3)],
}

# Base value of each parameter outside the factorial: the cheapest the space allows (parameters a space does not tune
# are ignored). Every kernel the search tries starts from these values, so they shape what it reaches, not just order.
LOW_RESOURCE = {
    "MIArchVgpr": False,
    "TransposeLDS": -1,
    "StoreVectorWidth": -1,
    "SourceSwap": False,
    "TailloopInNll": False,
    "UnrollLoopSwapGlobalReadOrder": 0,
    "WaveSeparateGlobalReadA": 0,
    "WaveSeparateGlobalReadB": 0,
    "NumElementsPerBatchStore": 0,
    "ScheduleGROverBarrier": 0,
    "ExtraMiLatencyLeft": -1,
    "StorePriorityOpt": False,
    "StoreSyncOpt": 0,
    "WorkGroupMapping": 0,
    "PrefetchGL2": 0,
    "TDMIterateMode": 0,
    "LDSSegmentInterleave": 0,
    "AdaptiveGemm": 0,
}


OFAT_PASSES = 2
MULTI_STARTS = 4

# Repair limits: refusals repaired per MI, accepted settings kept per repair, validations for pairs of changes.
REPAIRS = 8
REPAIR_KEEP = 4
REPAIR_PAIRS = 2000

# Stop after _initKernel when a register pool is this far over its cap: pools only grow, so the
# kernel cannot build. Near misses are generated in full so stage 3 can rank them by overflow.
EARLY_EXIT_MARGIN = 1.2


# The search space whose ranges define "buildable": the GA's.
LIST_SEARCH_SPACE = "generic"

# StreamK kernels need slightly more registers, so a StreamK-off list also covers StreamK-on runs.
LIST_STREAMK = False
