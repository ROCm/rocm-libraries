# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Enforce that the library reaches platform ONLY properly (never via raw paths).

The dependency direction ``library -> platform`` is allowed, but ONLY through
proper package imports (``from rocke.*``) and the sanctioned ``rocke.assets``
accessors (``platform_root`` / ``dsl_docs_dir`` / ``shape_utils_dir``, which are
env-overridable and encapsulate every path literal). The shipped library
packages (``kernels`` / ``builders`` / ``dispatch``) MUST NOT reach into platform
with raw path math or a ``rocke``-root ``sys.path`` bootstrap:

* no bare path-segment string literal ``"platform"`` / ``"dsl_docs"`` (use the
  rocke.assets accessor instead),
* no ``rocke/__init__.py`` discovery walk or ``ROCKE_ROOT`` bootstrap (``rocke``
  is editable-installed; see rocke/BUILDING.md).

Scope: the *shipped* packages only. Test-tree conftest bootstraps (the
chicken-and-egg ``platform/Python`` insert needed before ``rocke.assets`` is
importable) are sanctioned and intentionally excluded.

Pure file/AST scanning -- no imports, no GPU, no torch.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

# test file: rocke/library/tests/test_layering_isolation.py
_LIBRARY = Path(__file__).resolve().parents[1]  # tests -> library
_SHIPPED = [_LIBRARY / "kernels", _LIBRARY / "builders", _LIBRARY / "dispatch"]

# Path-segment literals that only appear when a file does raw path math into the
# platform tree; the rocke.assets accessors exist precisely so library code never
# writes these.
_FORBIDDEN_STR = {"platform", "dsl_docs", "__init__.py", "ROCKE_ROOT"}


def _iter_py(root: Path):
    for p in root.rglob("*.py"):
        if "__pycache__" not in p.parts:
            yield p


def _scan(path: Path) -> list[str]:
    out: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in _FORBIDDEN_STR:
                out.append(
                    f"{path}:{node.lineno}: raw platform path/bootstrap literal "
                    f"{node.value!r} (use rocke.assets / from rocke.* instead)"
                )
    return out


class TestLibraryReachesPlatformProperly(unittest.TestCase):
    def test_no_raw_platform_path_math_in_shipped_packages(self):
        violations: list[str] = []
        scanned = 0
        for root in _SHIPPED:
            if not root.exists():
                continue
            for py in _iter_py(root):
                scanned += 1
                violations.extend(_scan(py))
        self.assertGreater(scanned, 0, "no library package files scanned")
        self.assertEqual(
            violations,
            [],
            "shipped library packages must reach platform via from rocke.* and "
            f"rocke.assets accessors, not raw paths; found {len(violations)}:\n  "
            + "\n  ".join(violations),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
