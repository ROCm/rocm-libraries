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
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for the defaults+overrides Logic-file compaction.

Covers both the offline conversion tool (``utilities/apply_defaults_overrides.py``)
and the runtime loaders (``_expandSolutionDefaults``) that reconstruct full
solution dicts at parse time. The runtime loaders live in modules that import the
compiled ``rocisa`` extension; those are imported defensively so the suite still
runs (with the loader parametrizations skipped) in environments without a built
extension, while exercising the identical logic via ``Tensile.Utilities.merge``.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import os

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Module loading helpers
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(__file__)
_TENSILELITE = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_ADO_PATH = os.path.join(_TENSILELITE, "utilities", "apply_defaults_overrides.py")


def _load_ado():
    """Import the standalone ``apply_defaults_overrides`` tool by file path."""
    spec = importlib.util.spec_from_file_location("apply_defaults_overrides", _ADO_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ado = _load_ado()


def _available_loaders():
    """Return {label: _expandSolutionDefaults} for every importable loader copy.

    ``Tensile.Utilities.merge`` imports without the ``rocisa`` extension, so it is
    always available and validates the shared expansion logic. ``Tensile.LibraryIO``
    and ``Tensile.TensileMergeLibrary`` require ``rocisa``; they are included when
    importable (e.g. in CI) and skipped otherwise.
    """
    loaders = {}
    for modname in (
        "Tensile.Utilities.merge",
        "Tensile.LibraryIO",
        "Tensile.TensileMergeLibrary",
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception:  # noqa: BLE001 - missing compiled ext / heavy deps
            continue
        fn = getattr(mod, "_expandSolutionDefaults", None)
        if fn is not None:
            loaders[modname] = fn
    return loaders


_LOADERS = _available_loaders()
_LOADER_ITEMS = list(_LOADERS.items())
_LOADER_IDS = [name.split(".")[-1] for name, _ in _LOADER_ITEMS]
_LOADER_FNS = [fn for _, fn in _LOADER_ITEMS]


# --------------------------------------------------------------------------- #
# Offline tool: round-trip + default computation
# --------------------------------------------------------------------------- #
def _sample_solutions():
    """Homogeneous-key solutions with scalar, list, and dict values."""
    return [
        {"ISA": [9, 4, 2], "TT0": 64, "PGR": 1, "VW": 4, "ISP": {"a": 1}, "uid": 100},
        {"ISA": [9, 4, 2], "TT0": 64, "PGR": 1, "VW": 8, "ISP": {"a": 1}, "uid": 200},
        {"ISA": [9, 4, 2], "TT0": 64, "PGR": 0, "VW": 4, "ISP": {"a": 1}, "uid": 300},
    ]


def test_expand_collapse_roundtrip_dict_equality():
    """collapse -> expand reproduces every original solution by value (not size)."""
    sols = _sample_solutions()
    defaults = ado.compute_defaults(sols)
    overrides = [ado.compute_overrides(s, defaults) for s in sols]
    expanded = [ado.expand_solution(o, defaults) for o in overrides]
    assert expanded == sols
    # Keys with a >50% majority value (present in all solutions) are hoisted;
    # an all-distinct key like 'uid' is not.
    assert "ISA" in defaults and "TT0" in defaults and "ISP" in defaults
    assert "uid" not in defaults


def test_heterogeneous_missing_key_roundtrip():
    """A key absent from some solutions must not be resurrected on expansion."""
    sols = [
        {"ISA": [9, 4, 2], "TT0": 64, "opt": 7},
        {"ISA": [9, 4, 2], "TT0": 64},            # 'opt' legitimately absent
        {"ISA": [9, 4, 2], "TT0": 64, "opt": 7},
    ]
    defaults = ado.compute_defaults(sols)
    # 'opt' must NOT be defaulted (present in only 2/3 solutions).
    assert "opt" not in defaults
    overrides = [ado.compute_overrides(s, defaults) for s in sols]
    expanded = [ado.expand_solution(o, defaults) for o in overrides]
    assert expanded == sols
    assert "opt" not in expanded[1]


def test_override_equals_default_idempotence():
    """Keys equal to the default drop out of overrides and reappear on expansion."""
    sols = [{"k": 1, "v": [1, 2], "z": "abc"} for _ in range(3)]
    defaults = ado.compute_defaults(sols)
    assert set(defaults) == {"k", "v", "z"}
    for s in sols:
        ovr = ado.compute_overrides(s, defaults)
        assert ovr == {}
        assert ado.expand_solution(ovr, defaults) == s


def test_compute_defaults_boundary_conditions():
    """Strict >50% majority; single-solution defaults all keys."""
    # Exactly 50% -> not defaulted.
    half = [{"k": 1}, {"k": 1}, {"k": 2}, {"k": 2}]
    assert "k" not in ado.compute_defaults(half)
    # Just over 50% (and present in all) -> defaulted to the majority value.
    maj = [{"k": 1}, {"k": 1}, {"k": 1}, {"k": 2}, {"k": 2}]
    d = ado.compute_defaults(maj)
    assert d.get("k") == 1
    # Single solution -> every key defaulted, empty override.
    one = {"a": 1, "b": [1, 2]}
    d1 = ado.compute_defaults([one])
    assert d1 == one
    assert ado.compute_overrides(one, d1) == {}


def test_apply_revert_roundtrip(tmp_path):
    """apply_file then revert_file reproduces the original parsed data by value."""
    sols = _sample_solutions()
    logic = [
        {"MinimumRequiredVersion": "5.0.0"},
        "sched", "gfx942", ["Device 7300"],
        {"OperationType": "GEMM"},
        sols,
        [0, 1], [], None, None, None, "Equality",
    ]
    path = tmp_path / "logic.yaml"
    ado.dump_yaml(logic, str(path))
    original = ado.load_yaml(str(path))

    ado.apply_file(str(path))
    converted = ado.load_yaml(str(path))
    assert ado._is_converted(converted[5])           # now defaults+overrides form
    assert ado.expand_solution.__module__            # sanity: tool is loaded

    ado.revert_file(str(path))
    reverted = ado.load_yaml(str(path))
    assert reverted == original


# --------------------------------------------------------------------------- #
# Runtime loaders: aliasing, passthrough, malformed input
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _LOADER_FNS, reason="no _expandSolutionDefaults loader importable")
@pytest.mark.parametrize("expand", _LOADER_FNS, ids=_LOADER_IDS)
def test_loader_expands_and_does_not_alias(expand):
    """Expanded solutions must own independent copies of collection-valued defaults."""
    element5 = {
        "SolutionDefaults": {"ISA": [9, 4, 2], "ISP": {"a": 1}, "scalar": 42},
        "Solutions": [{"SolutionIndex": 0}, {"SolutionIndex": 1}],
    }
    expanded = expand(copy.deepcopy(element5))
    assert len(expanded) == 2
    assert expanded[0]["ISA"] == [9, 4, 2] and expanded[0]["scalar"] == 42
    # Mutating one solution's nested containers must not affect its sibling.
    expanded[0]["ISA"].append(99)
    expanded[0]["ISP"]["b"] = 2
    assert expanded[1]["ISA"] == [9, 4, 2]
    assert expanded[1]["ISP"] == {"a": 1}
    assert expanded[0]["ISA"] is not expanded[1]["ISA"]


@pytest.mark.skipif(not _LOADER_FNS, reason="no _expandSolutionDefaults loader importable")
@pytest.mark.parametrize("expand", _LOADER_FNS, ids=_LOADER_IDS)
def test_loader_legacy_flat_passthrough(expand):
    """A plain list (legacy flat format) is returned unchanged."""
    flat = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    result = expand(flat)
    assert result == flat
    assert result is flat


@pytest.mark.skipif(not _LOADER_FNS, reason="no _expandSolutionDefaults loader importable")
@pytest.mark.parametrize("expand", _LOADER_FNS, ids=_LOADER_IDS)
def test_loader_malformed_missing_solutions_key(expand):
    """A defaults dict without the 'Solutions' key fails loudly, not with KeyError."""
    malformed = {"SolutionDefaults": {"k": 1}}
    with pytest.raises(SystemExit):
        expand(malformed)
