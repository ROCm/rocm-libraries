# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.benchmark.perf.tool - LAYER 2: the user tool (dev OR agent), the ONLY writer.

The primitives in `rocke.benchmark.perf` are pure and write nothing. This package is the thin
convenience layer a developer or an agent drives to actually *keep* results and see
improve/regress:

  store.py      persist/read records in a USER CACHE dir (~/.cache/rocke.benchmark.perf),
                never in the repo - only this code is committed, never its data.
  selfcheck.py  compare a current run against a previous one (advisory).
  cli.py        `python -m rocke.benchmark.perf.tool ...` entrypoint.

An external perf framework is a *different* consumer of the same primitives; it
does not import this package. Stdlib only.
"""
