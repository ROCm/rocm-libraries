# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Which candidate MatrixInstructions can TensileLite build? Code generation only (CPU, no GPU).

An MI is *buildable* for a GEMM type when TensileLite code generation builds at least one kernel
for it with every other parameter inside geko's generic (GA) tuning ranges, i.e. a kernel a geko
GA run can produce. Kernels fail over a resource limit (registers, LDS, occupancy) or on a validity rule.
Lists are generated with StreamK off, which also covers StreamK-on runs; heuristic runs read the
same list. Subtile kernels are not covered.

Search per MI, stopping at the first kernel that builds:
  1. Presets: settings that build most tiles.
  2. Factorial: every combination of the parameters that decide register and LDS use, over the
     values the architecture's space tunes.
  R. Repair: the nearest settings TensileLite accepts to each refused one (validation only), code-generated.
  3. One parameter at a time, starting from the kernels closest to fitting.
Each buildable MI is then searched at every DepthU of the space (buildable_depthus).

Modules:
  constants    probe size, presets, base values, search limits
  measurement  Measurement: one kernel's resource use, or why it was refused
  codegen      write_probe_config, TuningSpace, CodegenChecker (TensileLite set up as ``--cpu-only``)
  search       SettingSearch (presets, factorial, single changes) and repair
  verdict      Verdict, classify (the search for one MI), buildable_depthus
  driver       classify_candidates: every candidate MI of a GEMM type, in parallel
  output       write_buildable_list, write_report
"""

from geko.config_generator.buildability.codegen import CodegenChecker, TuningSpace, write_probe_config
from geko.config_generator.buildability.constants import (
    EARLY_EXIT_MARGIN,
    LIST_PRESETS,
    LIST_SEARCH_SPACE,
    LIST_STREAMK,
    LOW_RESOURCE,
    MULTI_STARTS,
    OFAT_PASSES,
    PROBE_SIZE,
    REPAIR_KEEP,
    REPAIR_PAIRS,
    REPAIRS,
)
from geko.config_generator.buildability.driver import SubtileOnly, classify_candidates
from geko.config_generator.buildability.measurement import Measurement
from geko.config_generator.buildability.output import write_buildable_list, write_report
from geko.config_generator.buildability.search import SettingSearch, repair
from geko.config_generator.buildability.verdict import Verdict, buildable_depthus, classify, searched_depthu

__all__ = [
    "EARLY_EXIT_MARGIN",
    "LIST_PRESETS",
    "LIST_SEARCH_SPACE",
    "LIST_STREAMK",
    "LOW_RESOURCE",
    "MULTI_STARTS",
    "OFAT_PASSES",
    "PROBE_SIZE",
    "REPAIR_KEEP",
    "REPAIR_PAIRS",
    "REPAIRS",
    "CodegenChecker",
    "Measurement",
    "SettingSearch",
    "SubtileOnly",
    "TuningSpace",
    "Verdict",
    "buildable_depthus",
    "classify",
    "classify_candidates",
    "repair",
    "searched_depthu",
    "write_buildable_list",
    "write_probe_config",
    "write_report",
]
