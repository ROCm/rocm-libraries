# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Enforce the one-way layering rule: platform/ must NEVER reference the library.

The rocKE platform SDK (this ``rocke`` package), its tests, and its ``dsl_docs``
tooling MUST NOT depend on the moved SDPA/MHA library packages
(``kernels`` / ``builders`` / ``dispatch``). The dependency is one-way:
``library -> platform`` only. This guard fails on ANY platform->library
reference, in every form a plain ``grep`` for ``import`` would miss:

* ``import kernels`` / ``from builders import ...`` / ``from dispatch.x import``
* ``importlib.util.find_spec("builders.gfx950...")``
* ``importlib.import_module("kernels.common...")``
* a bare dotted-module string literal (e.g. a ``python -m builders...`` arg)

``rocke.dispatch`` is a platform SDK subpackage and is explicitly NOT the library
``dispatch`` package, so it is never flagged.

Pure file/AST scanning -- no imports, no GPU, no torch -- so it runs in every
environment. This guard is what turns "we audited the separation" into
"the separation is enforced".
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

# test file: rocke/platform/tests/test_layering_isolation.py
_PLATFORM = Path(__file__).resolve().parents[1]  # tests -> platform
# Scan the ENTIRE platform tree (Python SDK, tests, dsl_docs tooling, Cpp/bindings
# helpers, tools/) so no directory can silently become a blind spot.
_SCAN_ROOTS = [_PLATFORM]
_SELF = Path(__file__).resolve()

_LIB_TOP = ("kernels", "builders", "dispatch")
# A string that is *exactly* a dotted module path rooted at a library package,
# e.g. "builders.gfx950.attention.parity_unified_attention". Catches find_spec /
# import_module / `-m` subprocess args without matching prose or comments.
_MODULE_STR = re.compile(
    r"^(?:kernels|builders|dispatch)(?:\.[A-Za-z_][A-Za-z0-9_]*)+$"
)


def _iter_py(root: Path):
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts or p.resolve() == _SELF:
            continue
        yield p


def _module_is_library(mod: str | None) -> bool:
    if not mod:
        return False
    return mod in _LIB_TOP or mod.startswith(tuple(f"{t}." for t in _LIB_TOP))


def _scan(path: Path) -> list[str]:
    """Return human-readable violation strings for one file."""
    out: list[str] = []
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_is_library(alias.name):
                    out.append(f"{path}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            # level>0 is a relative import within the current package (never library)
            if node.level == 0 and _module_is_library(node.module):
                out.append(f"{path}:{node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # dotted module path ("builders.gfx950...") OR bare top-level name
            # ("builders") -- both catch find_spec/import_module/`-m` string args.
            if _MODULE_STR.match(node.value) or node.value in _LIB_TOP:
                out.append(f"{path}:{node.lineno}: module string {node.value!r}")
    return out


class TestPlatformDoesNotReferenceLibrary(unittest.TestCase):
    def test_no_platform_to_library_references(self):
        violations: list[str] = []
        scanned = 0
        for root in _SCAN_ROOTS:
            if not root.exists():
                continue
            for py in _iter_py(root):
                scanned += 1
                violations.extend(_scan(py))
        self.assertGreater(scanned, 0, "no platform python files scanned")
        self.assertEqual(
            violations,
            [],
            "platform/ must not reference the library (kernels/builders/dispatch); "
            f"found {len(violations)} reference(s):\n  " + "\n  ".join(violations),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
