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
"""Unit tests for defaults+overrides compaction and the YAML->JSON conversion.

Covers the offline tools (``utilities/apply_defaults_overrides.py`` and
``utilities/convert_yaml_to_json.py``) and the runtime loaders
(``_expandSolutionDefaults``). Runtime loaders live in modules that import the
compiled ``rocisa`` extension; those are imported defensively so the suite still
runs (with the loader parametrizations skipped) where the extension is not built,
while the identical logic is exercised via ``Tensile.Utilities.merge``.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import math
import os

import pytest
import yaml

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Module loading helpers
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(__file__)
_TENSILELITE = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_UTIL_DIR = os.path.join(_TENSILELITE, "utilities")


def _load_by_path(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_UTIL_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ado = _load_by_path("apply_defaults_overrides", "apply_defaults_overrides.py")
cvt = _load_by_path("convert_yaml_to_json", "convert_yaml_to_json.py")


def _available_loaders():
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


def _sample_solutions():
    return [
        {"ISA": [9, 4, 2], "TT0": 64, "PGR": 1, "VW": 4, "ISP": {"a": 1}, "uid": 100},
        {"ISA": [9, 4, 2], "TT0": 64, "PGR": 1, "VW": 8, "ISP": {"a": 1}, "uid": 200},
        {"ISA": [9, 4, 2], "TT0": 64, "PGR": 0, "VW": 4, "ISP": {"a": 1}, "uid": 300},
    ]


# --------------------------------------------------------------------------- #
# Offline tool: round-trip + default computation
# --------------------------------------------------------------------------- #
def test_expand_collapse_roundtrip_dict_equality():
    sols = _sample_solutions()
    defaults = ado.compute_defaults(sols)
    overrides = [ado.compute_overrides(s, defaults) for s in sols]
    expanded = [ado.expand_solution(o, defaults) for o in overrides]
    assert expanded == sols
    assert "ISA" in defaults and "TT0" in defaults and "ISP" in defaults
    assert "uid" not in defaults


def test_heterogeneous_missing_key_roundtrip():
    sols = [
        {"ISA": [9, 4, 2], "TT0": 64, "opt": 7},
        {"ISA": [9, 4, 2], "TT0": 64},            # 'opt' legitimately absent
        {"ISA": [9, 4, 2], "TT0": 64, "opt": 7},
    ]
    defaults = ado.compute_defaults(sols)
    assert "opt" not in defaults
    overrides = [ado.compute_overrides(s, defaults) for s in sols]
    expanded = [ado.expand_solution(o, defaults) for o in overrides]
    assert expanded == sols
    assert "opt" not in expanded[1]


def test_override_equals_default_idempotence():
    sols = [{"k": 1, "v": [1, 2], "z": "abc"} for _ in range(3)]
    defaults = ado.compute_defaults(sols)
    assert set(defaults) == {"k", "v", "z"}
    for s in sols:
        ovr = ado.compute_overrides(s, defaults)
        assert ovr == {}
        assert ado.expand_solution(ovr, defaults) == s


def test_compute_defaults_boundary_conditions():
    half = [{"k": 1}, {"k": 1}, {"k": 2}, {"k": 2}]
    assert "k" not in ado.compute_defaults(half)
    maj = [{"k": 1}, {"k": 1}, {"k": 1}, {"k": 2}, {"k": 2}]
    assert ado.compute_defaults(maj).get("k") == 1
    one = {"a": 1, "b": [1, 2]}
    d1 = ado.compute_defaults([one])
    assert d1 == one
    assert ado.compute_overrides(one, d1) == {}


def test_apply_revert_roundtrip(tmp_path):
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
    assert ado._is_converted(converted[5])
    ado.revert_file(str(path))
    assert ado.load_yaml(str(path)) == original


# --------------------------------------------------------------------------- #
# YAML -> JSON conversion (8725-specific)
# --------------------------------------------------------------------------- #
def test_converter_strict_type_loader_matches_runtime():
    """The converter loads with StrictTypeLoader, not safe_load, so bool-like
    tokens and 0/1 keep the type the runtime reader would see."""
    text = "a: true\nb: false\nc: 0\nd: 1\ne: on\nf: yes\n"
    strict = yaml.load(text, cvt.StrictTypeLoader)
    assert strict["a"] is True and strict["b"] is False
    assert strict["c"] == 0 and isinstance(strict["c"], int) and not isinstance(strict["c"], bool)
    assert strict["d"] == 1 and isinstance(strict["d"], int) and not isinstance(strict["d"], bool)
    # 'on'/'yes' are preserved as strings (safe_load would coerce them to bool).
    assert strict["e"] == "on" and strict["f"] == "yes"
    assert yaml.safe_load(text)["e"] is True  # documents the divergence the fix avoids


def test_convert_yaml_json_roundtrip_type_fidelity(tmp_path):
    """load_yaml -> write_json -> json.load reproduces values and types."""
    src = tmp_path / "logic.yaml"
    src.write_text("- {MinimumRequiredVersion: 5.0.0}\n- sched\n- gfx942\n"
                   "- [Device 7300]\n- {UseBeta: true, Batched: false}\n"
                   "- [{SolutionIndex: 0, GlobalSplitU: 1, ok: on}]\n")
    data = cvt.load_yaml(str(src))
    out = tmp_path / "logic.json"
    cvt.write_json(data, str(out))
    with open(out) as f:
        back = json.load(f)
    assert back == data
    # 'on' survives as a string through the strict loader + JSON round trip.
    assert back[5][0]["ok"] == "on"
    assert back[4]["UseBeta"] is True and back[4]["Batched"] is False


def test_write_json_rejects_non_finite(tmp_path):
    """allow_nan=False makes the converter fail loudly instead of emitting the
    non-standard NaN/Infinity tokens that strict JSON readers reject."""
    out = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        cvt.write_json([{"eff": float("nan")}], str(out))
    with pytest.raises(ValueError):
        cvt.write_json([{"eff": float("inf")}], str(out))


# --------------------------------------------------------------------------- #
# Runtime loaders: aliasing, passthrough, malformed input
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _LOADER_FNS, reason="no _expandSolutionDefaults loader importable")
@pytest.mark.parametrize("expand", _LOADER_FNS, ids=_LOADER_IDS)
def test_loader_expands_and_does_not_alias(expand):
    element5 = {
        "SolutionDefaults": {"ISA": [9, 4, 2], "ISP": {"a": 1}, "scalar": 42},
        "Solutions": [{"SolutionIndex": 0}, {"SolutionIndex": 1}],
    }
    expanded = expand(copy.deepcopy(element5))
    assert len(expanded) == 2
    expanded[0]["ISA"].append(99)
    expanded[0]["ISP"]["b"] = 2
    assert expanded[1]["ISA"] == [9, 4, 2]
    assert expanded[1]["ISP"] == {"a": 1}
    assert expanded[0]["ISA"] is not expanded[1]["ISA"]


@pytest.mark.skipif(not _LOADER_FNS, reason="no _expandSolutionDefaults loader importable")
@pytest.mark.parametrize("expand", _LOADER_FNS, ids=_LOADER_IDS)
def test_loader_legacy_flat_passthrough(expand):
    flat = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    result = expand(flat)
    assert result == flat
    assert result is flat


@pytest.mark.skipif(not _LOADER_FNS, reason="no _expandSolutionDefaults loader importable")
@pytest.mark.parametrize("expand", _LOADER_FNS, ids=_LOADER_IDS)
def test_loader_malformed_missing_solutions_key(expand):
    with pytest.raises(SystemExit):
        expand({"SolutionDefaults": {"k": 1}})
