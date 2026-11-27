# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The test directories inside the ``rocke`` package that
``platform/tests/in_tree/test_in_tree_suites.py`` runs, one child pytest process
per file.

The ``rocke_pytest_*`` entries collect ``platform/tests`` only; these
directories sit next to the code they test, so the in-tree runner is their CI
lane. ``portable_ir/tests`` is not listed: ``portable_ir/test_portable_ir.py::test_unit``
runs it.
"""

from __future__ import annotations

__all__ = ["EXCLUDED_FILES", "SUITES"]

# Importable package name -> optional modules that the directory needs.
SUITES: dict[str, tuple[str, ...]] = {
    "rocke.benchmark.gemm.tests": (),
    "rocke.benchmark.perf.tests": (),
}
# ``rocke.heuristics.tests`` runs in rocke_heuristics_pytest, a comprehensive-tier
# entry, with packages rocKE does not depend on. ``test_wavescope_pmc.py`` runs in
# no lane. ``rocke.tests.pending_lanes`` records what they wait for.
# ``test_fp16_rcr_multigpu.py`` runs from its installed copy under ``tests/benchmark``,
# which the installed ``rocke_pytest_*`` entries collect, so it is not run a second time.
EXCLUDED_FILES: tuple[str, ...] = ("test_wavescope_pmc.py", "test_fp16_rcr_multigpu.py")
