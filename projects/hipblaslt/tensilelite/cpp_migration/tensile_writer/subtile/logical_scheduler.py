# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tensile_writer.subtile.logical_scheduler — C++-backed LogicalScheduler primitives.

Thin re-export of the ``_tensile_writer.subtile.logical_scheduler`` nanobind
submodule. This slice ports only the *data / config* primitives of
``Tensile.Components.Subtile.LogicalScheduler``:

* ``Pass`` enum
* ``fmt_mt`` helper
* ``MFMATileRange`` / ``ReadGranularity``
* ``SchedulerConfig`` (incl. partition normalization + candidate generation)
* placement value types (``MFMAPlacement`` / ``LRPlacement`` / ``GRPlacement``)
  including their pass-populated fields (``deps`` / ``preOps`` / ``postOps`` /
  ``vgpr_tile_map`` [s]), which default to empty
* before-chain op value types (``WaitGRCounts`` / ``WaitGROp`` / ``WaitLROp`` /
  ``SyncOp`` / ``MaskKOp`` / ``LRIncOp`` / ``GRIncOp`` / ``SkipOp`` /
  ``InlineModuleOp``)
* dependency / slot / emitted-module value types (``Dep`` / ``SubIterKSlot`` /
  ``EmittedModule``)

The writer-free scheduling passes (place_LRs through
remove_unnecessary_wait_lr_sync, plus assign_vgpr_tiles) are ported in the
``LogicalScheduler`` class below and are the live implementation: the Python
``Tensile.Components.Subtile.LogicalScheduler`` delegates those passes to it and
rebuilds its dataclass partitions from ``LogicalScheduler.value_partitions()``.
The rocisa writer integration (build / populate_instructions, InstructionEmitter
dispatch, rocisa Module / Kernel.mainLoop emission), the ``InlineModuleOp.build``
Callable, and ``EmittedModule.instructions`` (rocisa objects) remain Python-only.
The value-type pass-populated fields above are filled by the converter from the
exported schedule.
"""

from tensile_writer import _tensile_writer as _ext

_ls = _ext.subtile.logical_scheduler

# Enum + free helpers
Pass = _ls.Pass
fmt_mt = _ls.fmt_mt

# Core primitives / config
MFMATileRange = _ls.MFMATileRange
ReadGranularity = _ls.ReadGranularity
SchedulerConfig = _ls.SchedulerConfig

# Placement value types
MFMAPlacement = _ls.MFMAPlacement
LRPlacement = _ls.LRPlacement
GRPlacement = _ls.GRPlacement

# Dependency / before-chain op value types
WaitGRCounts = _ls.WaitGRCounts
WaitGROp = _ls.WaitGROp
WaitLROp = _ls.WaitLROp
SyncOp = _ls.SyncOp
MaskKOp = _ls.MaskKOp
LRIncOp = _ls.LRIncOp
GRIncOp = _ls.GRIncOp
SkipOp = _ls.SkipOp
InlineModuleOp = _ls.InlineModuleOp

# Dependency / slot / emitted-module value types
Dep = _ls.Dep
SubIterKSlot = _ls.SubIterKSlot
EmittedModule = _ls.EmittedModule

# Writer-free pass pipeline (place_LRs through emit/build). Operates purely on
# the data-only logical schedule and exposes byte-identical print_* helpers for
# pass-by-pass parity with the Python LogicalScheduler. It does NOT populate
# rocisa instructions, allocate writer VGPR pools, or emit Kernel.mainLoop
# control flow — those remain Python-only.
LogicalScheduler = _ls.LogicalScheduler


def get_partition_candidates(tileInfoA, tileInfoB):
    """C++-backed ``SchedulerConfig.get_partition_candidates``.

    Accepts the same tileInfo objects as the Python static method and extracts
    the two ``localMMATileGrid[0]`` values before handing off to C++, returning
    ``[(partitionSizeM, partitionSizeN), ...]`` as a list of tuples.
    """
    M = tileInfoA.localMMATileGrid[0]
    N = tileInfoB.localMMATileGrid[0]
    return [tuple(c) for c in SchedulerConfig.get_partition_candidates(M, N)]


__all__ = [
    "Pass",
    "fmt_mt",
    "MFMATileRange",
    "ReadGranularity",
    "SchedulerConfig",
    "MFMAPlacement",
    "LRPlacement",
    "GRPlacement",
    "WaitGRCounts",
    "WaitGROp",
    "WaitLROp",
    "SyncOp",
    "MaskKOp",
    "LRIncOp",
    "GRIncOp",
    "SkipOp",
    "InlineModuleOp",
    "Dep",
    "SubIterKSlot",
    "EmittedModule",
    "LogicalScheduler",
    "get_partition_candidates",
]
