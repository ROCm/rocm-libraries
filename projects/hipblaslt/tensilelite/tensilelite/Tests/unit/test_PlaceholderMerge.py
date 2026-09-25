################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""Regression tests for the gfx942 placeholder/predicate merge collision.

Two invariants must hold together; either alone is insufficient:

1. Sibling logic YAMLs (same arch, same basename) declare identical DeviceNames.
2. The ``_ID<chipid>`` placeholder-filename suffix is gated on
   ``supportsChipIdPredicate``, mirroring ``HardwarePredicate.FromHardware``.

Sibling-DeviceNames consistency is a fact about the shipped corpus, so it is
enforced by ``TensileLogic --check-all`` via
``tensilelite.logic.valid_corpus_consistency.check_corpus_invariants`` in
every kernel-generating build, and pinned hermetically in
``test_valid_corpus_consistency.py``. Everything in this file checks code, and
none of it reads the corpus.
"""
import ast
from pathlib import Path

import pytest

from tensilelite import SolutionLibrary
from tensilelite.Common.Architectures import SUPPORTED_GFX, supportsChipIdPredicate


_ID_SUFFIX_LITERAL = "_ID"
_GATE_FUNC_NAME = "supportsChipIdPredicate"


def _annotate_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node


def _node_contains_id_suffix_literal(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Constant)
        and isinstance(sub.value, str)
        and sub.value == _ID_SUFFIX_LITERAL
        for sub in ast.walk(node)
    )


def _if_test_calls_gate(if_node: ast.If) -> bool:
    return any(
        isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Name)
        and sub.func.id == _GATE_FUNC_NAME
        for sub in ast.walk(if_node.test)
    )


def _is_placeholder_target(node: ast.AST) -> bool:
    if isinstance(node, ast.AugAssign):
        return isinstance(node.target, ast.Name) and node.target.id == "placeholderName"
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == "placeholderName"
                   for t in node.targets)
    return False


def test_id_suffix_appends_are_gated_on_supports_chip_id_predicate():
    """Every ``placeholderName`` site embedding ``"_ID"`` must sit inside an
    ``if`` whose test calls ``supportsChipIdPredicate``."""
    src_path = Path(SolutionLibrary.__file__)
    tree = ast.parse(src_path.read_text(), filename=str(src_path))
    _annotate_parents(tree)

    sites = [
        node for node in ast.walk(tree)
        if _is_placeholder_target(node) and _node_contains_id_suffix_literal(node)
    ]
    assert sites, f"No '_ID' suffix construction found in {src_path.name}; update test."

    ungated = []
    for site in sites:
        ancestor = getattr(site, "parent", None)
        while ancestor is not None:
            if isinstance(ancestor, ast.If) and _if_test_calls_gate(ancestor):
                break
            ancestor = getattr(ancestor, "parent", None)
        else:
            ungated.append(site.lineno)

    assert not ungated, (
        f"{src_path.name}: '_ID' suffix at line(s) {ungated} not gated on "
        f"{_GATE_FUNC_NAME}(...)."
    )


_HARDWARE_CASES = [
    ("gfx942", ["Device 74a1"], False),
    ("gfx950", ["Device 75a0"], True),
]


@pytest.mark.parametrize("devicePart,deviceNames,expect_id_suffix", _HARDWARE_CASES)
def test_hardware_gates_placeholder_chip_id_suffix(
    devicePart, deviceNames, expect_id_suffix
):
    """``MasterSolutionLibrary.hardware`` appends ``_ID<chipid>`` iff
    ``supportsChipIdPredicate(devicePart)``."""
    d = {"ArchitectureName": devicePart, "CUCount": None, "DeviceNames": deviceNames}
    _, placeholderName = SolutionLibrary.MasterSolutionLibrary.hardware(
        d, library=None, placeholderName="TensileLibrary", lazyLibrary=True
    )

    has_id = "_ID" in placeholderName
    assert has_id == expect_id_suffix, (
        f"{devicePart}: _ID suffix presence={has_id}, expected={expect_id_suffix} "
        f"(name={placeholderName!r})"
    )
    assert placeholderName.endswith("_" + devicePart), placeholderName


def test_supports_chip_id_predicate_only_gfx950():
    """Lock chip-id-aware archs to gfx950 across every architecture TensileLite
    supports, whether or not logic ships for it yet; new entries require
    re-audit of YAMLs and the SolutionLibrary suffix gate."""
    assert [gfx for gfx in SUPPORTED_GFX if supportsChipIdPredicate(gfx)] == ["gfx950"]


def test_supports_chip_id_predicate_includes_gfx950():
    assert supportsChipIdPredicate("gfx950") is True
