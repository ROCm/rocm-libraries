# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests gate on ``gpu("gfxNNN")``, not an arch/hip ``skipif`` that drifts from it."""

import ast
from pathlib import Path

import pytest

from tensilelite.Tests.gpu_detection import arch_matches

pytestmark = pytest.mark.unit

_UNIT_DIR = Path(__file__).resolve().parent
_FORBIDDEN = ("GFX_TARGET", "HAS_GFX", "hip is None", "HIP_AVAILABLE")


def _arch_skipifs(path):
    src = path.read_text()
    for node in ast.walk(ast.parse(src, str(path))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "skipif"):
            text = ast.get_source_segment(src, node) or ""
            hits = [w for w in _FORBIDDEN if w in text]
            if hits:
                yield f"{path.relative_to(_UNIT_DIR)}:{node.lineno}: skipif on {', '.join(hits)}"


def test_no_arch_skipif_in_unit_tests():
    found = [hit for path in sorted(_UNIT_DIR.rglob("*.py")) for hit in _arch_skipifs(path)]
    assert not found, (
        "Use @pytest.mark.gpu(\"gfxNNN\"), not skipif:\n  " + "\n  ".join(found))


@pytest.mark.parametrize("found,wanted,expected", [
    ("gfx950", "gfx950", True),
    ("gfx942", "gfx950", False),
    ("gfx1250", "gfx1250", True),
    ("gfx1250-strict", "gfx1250", True),
    ("gfx1250", "gfx1250-strict", False),
    ("gfx1250-strict", "gfx1250-strict", True),
    ("gfx12500", "gfx1250", False),
])
def test_arch_matches(found, wanted, expected):
    assert arch_matches(found, wanted) is expected
