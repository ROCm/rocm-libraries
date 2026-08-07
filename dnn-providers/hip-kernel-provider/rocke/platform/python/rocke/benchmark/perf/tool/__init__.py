# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.benchmark.perf.tool - local history and self-check CLI.

The primitives in `rocke.benchmark.perf` do not persist records. This package is the
thin convenience layer a developer drives to *keep* results and see
improve/regress:

  store.py      persist/read records in a USER CACHE dir (~/.cache/rocke-perf),
                never in the repo - only this code is committed, never its data.
  selfcheck.py  compare a current run against a previous one (advisory).
  cli.py        `python -m rocke.benchmark.perf.tool ...` entrypoint.

An external perf framework is a *different* consumer of the same primitives; it
does not import this package. Stdlib only.
"""
