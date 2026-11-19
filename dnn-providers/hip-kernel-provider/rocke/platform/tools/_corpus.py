# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The representative-IR case corpus the emitted-IR gates walk.

By default that is the platform harness (tests/instances/
rocke_ir_parity_harness.py). A library family keeps its representative cases
beside its own golden (e.g. library/tests/test_conv_ir_golden.py) and runs the
same gates over them by passing that file with --cases, so platform never has
to import library code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_corpus(path: str = ""):
    """Return the corpus module: ``path`` if given, else the platform harness.

    A corpus module exposes ``cases()`` (dicts with case_id, family, arch and a
    zero-arg ``build``) and ``lower_case(case, flavor)``. The harness must
    already be importable (the gates' sys.path bootstrap puts it there).
    """
    if not path:
        import rocke_ir_parity_harness

        return rocke_ir_parity_harness
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"FATAL: --cases {path} is not an importable module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
