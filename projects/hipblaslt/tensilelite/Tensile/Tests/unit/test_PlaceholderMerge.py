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
"""
import ast
import re
from collections import defaultdict
from pathlib import Path

import pytest

from Tensile import SolutionLibrary
from Tensile.Common.Architectures import supportsChipIdPredicate


_LOGIC_ROOT = (
    Path(__file__).resolve().parents[4]
    / "library" / "src" / "amd_detail" / "rocblaslt" / "src"
    / "Tensile" / "Logic" / "asm_full"
)

_needs_logic_dir = pytest.mark.xfail(
    not _LOGIC_ROOT.is_dir(),
    reason="Logic files not found: https://github.com/ROCm/rocm-libraries/issues/7481",
)

_DEVICE_NAMES_RE = re.compile(r"^\s*-\s*\[\s*Device\s+([^\]]+)\]\s*$")
_ID_SUFFIX_LITERAL = "_ID"
_GATE_FUNC_NAME = "supportsChipIdPredicate"


def _iter_arch_dirs():
    for codename_dir in _LOGIC_ROOT.iterdir():
        if not codename_dir.is_dir():
            continue
        for arch_dir in codename_dir.iterdir():
            if arch_dir.is_dir() and arch_dir.name.startswith("gfx"):
                yield codename_dir.name, arch_dir


def _all_arch_names():
    if not _LOGIC_ROOT.is_dir():
        return []
    return sorted({arch_dir.name for _, arch_dir in _iter_arch_dirs()})


def _read_device_names(yaml_path: Path):
    """Return sorted DeviceNames tuple from a logic-YAML header, or None."""
    try:
        with yaml_path.open("r") as f:
            for _ in range(8):
                line = f.readline()
                if not line:
                    return None
                m = _DEVICE_NAMES_RE.match(line)
                if m:
                    parts = [p.strip() for p in m.group(1).split(",")]
                    parts = [p[len("Device "):].strip() if p.startswith("Device ") else p
                             for p in parts]
                    return tuple(sorted(parts))
    except OSError:
        return None
    return None


@_needs_logic_dir
def test_logic_yaml_sibling_device_names_consistent():
    """Same-basename YAMLs in one arch dir must declare identical DeviceNames."""
    assert _LOGIC_ROOT.is_dir(), f"Logic root not found: {_LOGIC_ROOT}"
    violations = []
    for codename, arch_dir in _iter_arch_dirs():
        by_basename = defaultdict(lambda: defaultdict(list))
        for yaml_path in arch_dir.rglob("*.yaml"):
            names = _read_device_names(yaml_path)
            if names is None:
                continue
            by_basename[yaml_path.name][names].append(yaml_path)
        for basename, dn_map in by_basename.items():
            if len(dn_map) > 1:
                detail = {
                    str(names): [str(p.relative_to(_LOGIC_ROOT)) for p in paths]
                    for names, paths in dn_map.items()
                }
                violations.append(f"  {codename}/{arch_dir.name}/{basename}: {detail}")

    assert not violations, "Divergent sibling DeviceNames:\n" + "\n".join(violations)


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
@pytest.mark.parametrize("arch", _all_arch_names())
def test_supports_chip_id_predicate_only_gfx950(arch):
    """Lock chip-id-aware archs to gfx950; new entries require re-audit of
    YAMLs and the SolutionLibrary suffix gate."""
    base_arch = arch.split("_", 1)[0]
    expected = base_arch == "gfx950"
    assert supportsChipIdPredicate(base_arch) is expected, (
        f"{arch} (base {base_arch}): supportsChipIdPredicate={not expected}, "
        f"expected={expected}"
    )


def test_supports_chip_id_predicate_includes_gfx950():
    assert supportsChipIdPredicate("gfx950") is True
