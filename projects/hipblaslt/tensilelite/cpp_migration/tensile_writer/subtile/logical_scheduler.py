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
* before-chain op value types (``WaitGRCounts`` / ``WaitGROp`` / ``WaitLROp`` /
  ``SyncOp`` / ``MaskKOp`` / ``LRIncOp`` / ``GRIncOp`` / ``SkipOp``)

The scheduling passes (place_LRs / place_GRs / annotate_deps / build /
populate_instructions), InstructionEmitter dispatch, and rocisa Module emission
are **not** ported here and remain pure Python. The names mirror the Python
module so it can optionally delegate its pure helpers here.
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
    "get_partition_candidates",
]
