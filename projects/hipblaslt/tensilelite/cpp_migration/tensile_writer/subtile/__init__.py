# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Subtile C++-backed nanobind re-export namespace.

Production callers (boundary contract per docs/subtile_cpp_boundary.md):
  emit               — Kernel.py (_CPP_EMIT)
  geometry           — SubtileGeometry.py (_cppgeo)
  instruction_scheduler — LogicalScheduler.py (_cppsched)
  logical_scheduler  — LogicalScheduler.py (_cppls)
  loop_orchestrator  — LogicalScheduler.py (_emit_loop / _emit_main_and_exit_loops / _emit_tail_loop)
  module_builder     — Kernel.py, SubtileGREmit.py, SubtileLREmit.py
  tile_info          — Kernel.py (_CPP_TI)
"""

from . import emit  # noqa: F401
from . import geometry  # noqa: F401
from . import instruction_scheduler  # noqa: F401
from . import logical_scheduler  # noqa: F401
from . import loop_orchestrator  # noqa: F401
from . import module_builder  # noqa: F401
from . import tile_info  # noqa: F401

__all__ = [
    "emit",
    "geometry",
    "instruction_scheduler",
    "logical_scheduler",
    "loop_orchestrator",
    "module_builder",
    "tile_info",
]
