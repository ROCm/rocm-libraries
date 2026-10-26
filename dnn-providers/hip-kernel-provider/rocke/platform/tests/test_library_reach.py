# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Freeze platform's imports of the library packages, so they only shrink.

The dependency rule is ``library -> platform``. A few platform files still
import the library's top-level packages (``kernels``, ``builders``,
``dispatch``, ``benchmarks``): parity harness cases, examples and tools that
predate the library's own homes for them. They resolve only because
``tests/conftest.py`` puts the sibling library tree on ``sys.path``.

This test walks every platform Python file by AST (function-level imports
included) and holds each file to the count recorded in ``ALLOWED``:

* a file not listed may not import a library package;
* a listed file may not import more than its count;
* a listed file that imports fewer must have its count lowered here, so the
  list never hides room to grow back.

Move the code that needs the library into ``library/`` instead of raising a
count. Host-only; no GPU, no engine build.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
ROOTS = ("python", "tests", "tools", "dsl_docs", "cpp")
LIBRARY_PACKAGES = frozenset({"kernels", "builders", "dispatch", "benchmarks"})

# platform-relative path -> library imports it may still make. Burn down only.
ALLOWED = {
    "cpp/bindings/prove_parity_binding.py": 2,
    "dsl_docs/development/verify_dsl_docs.py": 2,
    "dsl_docs/optimization/utilities/tools/dsl_probes/probe_occupancy.py": 1,
    "dsl_docs/optimization/utilities/tools/utils/extract_rocke_isa.py": 1,
    "dsl_docs/optimization/utilities/tools/utils/profile_register_usage.py": 2,
    "python/rocke/core/lower_cktile.py": 2,
    "python/rocke/examples/common/hip_lowering_parity.py": 4,
    "python/rocke/examples/gfx950/conv_dgrad/run_one_dgrad.py": 5,
    "python/rocke/heuristics/gen_sweep_data.py": 2,
    "python/rocke/portable_ir/drivers/bench_jit_validation.py": 2,
    "python/rocke/portable_ir/drivers/roll_coverage.py": 1,
    "python/rocke/portable_ir/drivers/roll_gfx950_sweep.py": 4,
    "python/rocke/portable_ir/drivers/roll_hsaco_parity.py": 2,
    "python/rocke/portable_ir/drivers/roll_nd_coverage.py": 2,
    "python/rocke/portable_ir/drivers/verify_recording_production.py": 1,
    "python/rocke/portable_ir/examples/export_mha.py": 1,
    "python/rocke/portable_ir/tests/test_recording_builder.py": 1,
    "tests/core/test_cshuffle_reuse_barrier.py": 3,
    "tests/core/test_ir_serialize.py": 1,
    "tests/core/test_pipeline_basic.py": 4,
    "tests/instances/rocke_ir_parity_harness.py": 51,
    "tests/portable_ir/test_recipe_roller.py": 1,
    "tests/test_rocke.py": 12,
}


def _library_imports(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        count += sum(name.split(".")[0] in LIBRARY_PACKAGES for name in names)
    return count


def _current() -> Counter:
    found: Counter = Counter()
    for root in ROOTS:
        base = PLATFORM / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            n = _library_imports(path)
            if n:
                found[path.relative_to(PLATFORM).as_posix()] = n
    return found


def test_platform_library_imports_only_shrink():
    found = _current()
    problems = []
    for rel, n in sorted(found.items()):
        allowed = ALLOWED.get(rel)
        if allowed is None:
            problems.append(
                f"{rel}: {n} new library import(s); move the code to library/"
            )
        elif n > allowed:
            problems.append(f"{rel}: {n} library imports, allowed {allowed}")
    for rel, allowed in sorted(ALLOWED.items()):
        n = found.get(rel, 0)
        if n < allowed:
            problems.append(
                f"{rel}: now {n} library imports; lower ALLOWED to {n} "
                "(drop the entry at 0)"
            )
    assert not problems, "\n".join(problems)
