# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.benchmark.perf - rocKE kernel performance measurement primitives.

Lives at `rocke/platform/python/rocke/benchmark/perf/`, alongside the other
`benchmark/` harnesses; invoked by rocKE kernel-launch commands.

LAYER 1 - primitives (this package). Pure: they RETURN records / values and write
nothing, so any consumer (the user tool OR an external perf framework) can use them
without inheriting file-writing behavior.

  schema.py     - measurement-record schema + validate (the seam)
  counters.py   - probe + normalized counter map per arch (rocprofv3)
  harness.py    - profile a kernel -> RETURN a record (composes below)
  occupancy.py  - VGPR/AGPR/SGPR/LDS + occupancy from ELF notes (no GPU)
  aggregate.py  - K records -> median/spread/derived
  report.py     - record -> JSON string/dict (serialize; no writes)

LAYER 2 - the user tool lives in a SEPARATE package `rocke.benchmark.perf.tool` (store, self-
check, CLI). It imports these primitives; primitives never import it.
"""

from . import schema  # noqa: F401

__all__ = ["schema"]
