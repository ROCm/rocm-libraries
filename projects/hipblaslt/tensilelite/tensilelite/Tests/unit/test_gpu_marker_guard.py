# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests gate on ``gpu("gfxNNN")``, not an arch/hip skip that drifts from it.

Flags ``skipif(...)`` and in-body ``if ...: pytest.skip(...)`` on the GPU.
"""

import ast
from pathlib import Path

import pytest

from gpu_detection import arch_matches, check_gpu_marks

pytestmark = pytest.mark.unit

_UNIT_DIR = Path(__file__).resolve().parent
_FORBIDDEN = ("GFX_TARGET", "HAS_GFX", "hip is None", "HIP_AVAILABLE",
              "gpu_is", "detect_gpu_target")


def _is_call(node, attr):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr)


def _arch_skips(path):
    src = path.read_text()
    for node in ast.walk(ast.parse(src, str(path))):
        if _is_call(node, "skipif"):
            cond, kind = node, "skipif"
        elif isinstance(node, ast.If) and any(
                _is_call(n, "skip") for stmt in node.body for n in ast.walk(stmt)):
            cond, kind = node.test, "if ...: pytest.skip"
        else:
            continue
        text = ast.get_source_segment(src, cond) or ""
        hits = [w for w in _FORBIDDEN if w in text]
        if hits:
            yield f"{path.relative_to(_UNIT_DIR)}:{node.lineno}: {kind} on {', '.join(hits)}"


def test_no_arch_skip_in_unit_tests():
    found = [hit for path in sorted(_UNIT_DIR.rglob("*.py")) for hit in _arch_skips(path)]
    assert not found, (
        "Use @pytest.mark.gpu(\"gfxNNN\"), not a skip:\n  " + "\n  ".join(found))


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


class _Item:
    """The part of a pytest item check_gpu_marks reads: its gpu marks, nearest first."""

    def __init__(self, *marks):
        self.nodeid = "t"
        self._marks = [m.mark for m in marks]

    def iter_markers(self, name):
        return (m for m in self._marks if m.name == name)


@pytest.mark.parametrize("item", [
    _Item(pytest.mark.gpu),
    _Item(pytest.mark.gpu("gfx950", "gfx942")),
    _Item(pytest.mark.unit, pytest.mark.gpu("gfx1250")),
], ids=["bare", "archs", "other-marks"])
def test_check_gpu_marks_accepts(item):
    check_gpu_marks(item)


@pytest.mark.parametrize("item, message", [
    (_Item(pytest.mark.gpu("gfx950"), pytest.mark.gpu("gfx942")), "2 gpu marks"),
    (_Item(pytest.mark.gpu("gfx942"), pytest.mark.gpu), "2 gpu marks"),
    (_Item(pytest.mark.gpu(arch="gfx950")), "arch strings only"),
    (_Item(pytest.mark.gpu(950)), "arch strings only"),
], ids=["stacked-or-param", "function-and-class", "keyword", "non-string"])
def test_check_gpu_marks_rejects(item, message):
    with pytest.raises(pytest.UsageError, match=message):
        check_gpu_marks(item)
