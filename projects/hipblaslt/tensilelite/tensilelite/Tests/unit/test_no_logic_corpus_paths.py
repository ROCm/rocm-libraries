# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""No TensileLite test may build its own path into the production logic corpus.

A test that walks up from its own ``__file__`` into
``library/src/amd_detail/rocblaslt/.../asm_full`` reads files outside the
tensilelite tree, which a tuning PR can change and an installed-artifact lane
does not have. Tests that genuinely need the corpus go through
``logic_corpus.py``, the one allowlisted resolver, which skips visibly and
counts them. Everything else uses a fixture.

The check folds string expressions statically (literals, ``+`` and f-string
concatenation) and flags any ``os.path.join`` / ``pathlib`` expression that both
references ``__file__`` and contains one of the product-tree path components.
A synthetic corpus built under ``tmp_path`` is not anchored at ``__file__`` and
is not flagged. Building the path in two steps through an intermediate variable
evades it; review still has to catch that.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_SEGMENTS = {"amd_detail", "rocblaslt", "asm_full"}
_ALLOWLIST = {Path("unit") / "logic_corpus.py"}


def _fold(node):
    """The string a constant-only expression evaluates to, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold(node.left), _fold(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts = [_fold(value) for value in node.values]
        if all(part is not None for part in parts):
            return "".join(parts)
    return None


def _segments(node):
    found = set()
    for sub in ast.walk(node):
        folded = _fold(sub)
        if folded is not None:
            found |= _FORBIDDEN_SEGMENTS & set(folded.replace("\\", "/").split("/"))
    return found


def _references_file(node):
    return any(isinstance(sub, ast.Name) and sub.id == "__file__" for sub in ast.walk(node))


def _anchored_corpus_paths(source, filename="<probe>"):
    """Line numbers of outermost path expressions anchored at ``__file__`` that
    reach through a product-tree component."""
    tree = ast.parse(source, filename=filename)
    flagged = []
    covered = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Call, ast.BinOp)) or id(node) in covered:
            continue
        if _references_file(node) and _segments(node):
            flagged.append(node.lineno)
            covered.update(id(sub) for sub in ast.walk(node))
    return sorted(set(flagged))


def _offenders():
    offenders = {}
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        rel = path.relative_to(_TESTS_ROOT)
        lines = _anchored_corpus_paths(path.read_text(encoding="utf-8"), str(path))
        if lines:
            offenders[rel] = lines
    return offenders


def test_no_test_builds_a_path_into_the_logic_corpus():
    offenders = {rel: lines for rel, lines in _offenders().items() if rel not in _ALLOWLIST}
    assert not offenders, (
        "These tests build their own path into the production logic corpus. Use a "
        "fixture, a valid_corpus_consistency check, or logic_corpus.require_logic_file:\n"
        + "\n".join(f"  {rel}: lines {lines}" for rel, lines in sorted(offenders.items()))
    )


def test_allowlist_has_no_stale_entries():
    assert _ALLOWLIST <= set(_offenders())


@pytest.mark.parametrize(
    "source",
    [
        'import os\nP = os.path.join(os.path.dirname(__file__), "..", "library", "src", "amd_detail")\n',
        'from pathlib import Path\nP = Path(__file__).parents[4] / "library" / "Logic" / "asm_full"\n',
        'import os\nP = os.path.join(os.path.dirname(__file__), *([".."] * 4), "roc" + "blaslt")\n',
        'from pathlib import Path\nP = Path(__file__).parent / f"library/src/amd_detail/{1}"\n',
    ],
)
def test_guard_flags_a_path_anchored_at_the_test_file(source):
    assert _anchored_corpus_paths(source) == [2]


@pytest.mark.parametrize(
    "source",
    [
        'def f(tmp_path):\n    return tmp_path / "library" / "amd_detail" / "asm_full"\n',
        'import os\nP = os.path.join(os.path.dirname(__file__), "logic_yaml", "staggeru")\n',
        '"""Prose naming amd_detail and asm_full, next to __file__."""\nP = __file__\n',
        'TOOL = "rocblaslt-bench"\nP = __file__ + TOOL\n',
    ],
)
def test_guard_ignores_synthetic_corpora_fixtures_and_prose(source):
    assert _anchored_corpus_paths(source) == []
