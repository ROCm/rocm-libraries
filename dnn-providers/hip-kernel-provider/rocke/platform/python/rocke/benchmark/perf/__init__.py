# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.benchmark.perf - rocKE kernel performance measurement primitives.

Lives at `rocke/platform/python/rocke/benchmark/perf/`, alongside the other
`benchmark/` harnesses; invoked by rocKE kernel-launch commands.

LAYER 1 - primitives (this package). They return records / values without persisting
measurement records. The harness normally uses temporary profiler files; callers
may explicitly retain that workspace with artifacts_dir. `perfjson.emit` writes
only its launcher protocol line to a caller-selected stream.

  schema.py     - measurement-record schema + validate (the seam)
  perfjson.py   - emit/parse the `PerfJSON:` launcher line (optional wall timing)
  counters.py   - probe + normalized counter map per arch (rocprofv3)
  harness.py    - profile a kernel -> RETURN a record (composes below)
  occupancy.py  - VGPR/AGPR/SGPR/LDS + occupancy from ELF notes (no GPU)
  aggregate.py  - K records -> median/spread/derived
  report.py     - record -> JSON string/dict (serialize; no writes)

LAYER 2 - the user tool lives in a SEPARATE package `rocke.benchmark.perf.tool`
(history, artifact bundles, self-check, CLI). Primitives never import it.
"""

from . import perfjson, schema

__all__ = ["perfjson", "schema"]
