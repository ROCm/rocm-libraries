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

Sibling-DeviceNames consistency is the enforcement point in CI: it runs
unconditionally inside ``TensileLogic --check-all`` via
``Tensile.TensileLogic.ValidCorpusConsistency.check_corpus_invariants``. The
chip-ID-arch-lock check is *not* wired into ``--check-all`` (it guards a
future source-policy change, not the artifact any one build selects -- see
``check_corpus_invariants``'s docstring); its enforcement point is this
file's own pytest assertion against the real corpus, run in CI's unit-test
job. Either way, these pytest copies exercise the real ``Logic/asm_full``
directory when it's present in *this* test environment, and are a
convenience/local-dev signal otherwise -- hence ``skipif`` (an unmet
precondition), not ``xfail`` (an expected failure).
"""
import ast
from pathlib import Path

import pytest

from Tensile import SolutionLibrary
from Tensile.Common.Architectures import supportsChipIdPredicate
from Tensile.TensileLogic.ValidCorpusConsistency import (
    find_chip_id_arch_lock_violations,
    find_sibling_device_names_violations,
)


_LOGIC_ROOT = (
    Path(__file__).resolve().parents[4]
    / "library" / "src" / "amd_detail" / "rocblaslt" / "src"
    / "Tensile" / "Logic" / "asm_full"
)

_needs_logic_dir = pytest.mark.skipif(
    not _LOGIC_ROOT.is_dir(),
    reason="Logic files not found: https://github.com/ROCm/rocm-libraries/issues/7481",
)

_ID_SUFFIX_LITERAL = "_ID"
_GATE_FUNC_NAME = "supportsChipIdPredicate"


@_needs_logic_dir
def test_logic_yaml_sibling_device_names_consistent():
    """Same-basename YAMLs in one logic tree must declare identical DeviceNames."""
    violations = find_sibling_device_names_violations(sorted(_LOGIC_ROOT.rglob("*.yaml")), _LOGIC_ROOT)
    assert not violations, "\n".join(violations)


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


@_needs_logic_dir
def test_supports_chip_id_predicate_only_gfx950():
    """Lock chip-id-aware archs (as seen in the corpus) to gfx950; new
    entries require re-audit of YAMLs and the SolutionLibrary suffix gate."""
    violations = find_chip_id_arch_lock_violations(sorted(_LOGIC_ROOT.rglob("*.yaml")))
    assert not violations, "\n".join(violations)


def test_supports_chip_id_predicate_includes_gfx950():
    assert supportsChipIdPredicate("gfx950") is True
