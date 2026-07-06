# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.benchmark.perf - rocKE kernel performance PRIMITIVES (pure library, produce side).

Prototype (Option A1): self-contained, invokes rocke kernels but lives outside the
rocke tree; structured to later move to `rocke/platform/Python/rocke/benchmark/perf/`.

LAYER 1 - primitives (this package). Pure: they RETURN records / values and write
nothing, so any consumer (the user tool OR an external perf framework) can use them
without inheriting file-writing behavior.

  schema.py     - measurement-record schema + validate (the seam)      [done]
  counters.py   - probe + normalized counter map per arch (rocprofv3)  [PR-2]
  harness.py    - profile a kernel -> RETURN a record (composes below) [PR-2]
  occupancy.py  - VGPR/AGPR/SGPR/LDS + occupancy from ELF notes (no GPU)[PR-3]
  aggregate.py  - K records -> median/spread/derived                   [PR-4]
  report.py     - record -> JSON string/dict (serialize; no writes)    [PR-1]

LAYER 2 - the user tool lives in a SEPARATE package `rocke.benchmark.perf.tool` (store, self-
check, CLI; dev OR agent). It imports these primitives; primitives never import it.
"""

from . import schema  # noqa: F401

__all__ = ["schema"]
