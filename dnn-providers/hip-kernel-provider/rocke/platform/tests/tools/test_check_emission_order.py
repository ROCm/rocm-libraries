# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tools/check_emission_order.py: the scanner, and the C++ tree held to it.

Source-tree only (it reads cpp/ and tools/); excluded from the installed tree.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

PLATFORM = Path(__file__).resolve().parents[2]


def _load():
    path = PLATFORM / "tools" / "check_emission_order.py"
    spec = importlib.util.spec_from_file_location("check_emission_order", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves its module here
    spec.loader.exec_module(module)
    return module


ceo = _load()


def _callees(src: str) -> list[str]:
    return [site.callee for site in ceo.find_sites(src)]


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "x = rocke_b_add(b, rocke_b_mul(b, p, q), rocke_b_sub(b, r, s));",
            ["rocke_b_add"],
        ),
        # A constant takes an SSA number like any other op.
        (
            "x = rocke_b_add(b, rocke_b_mul(b, p, q), rocke_b_const_i32(b, 1));",
            ["rocke_b_add"],
        ),
        (
            "x = rocke_b_mul(b, rocke_b_const_i32(b, 2), rocke_b_const_i32(b, 3));",
            ["rocke_b_mul"],
        ),
        # One emitting argument has nothing to reorder against.
        ("x = rocke_b_add(b, p, rocke_b_mul(b, q, rocke_b_const_i32(b, 4)));", []),
        ("x = rocke_b_add(b, p, q);", []),
        # Type and info helpers emit nothing.
        ("x = rocke_b_zext(b, rocke_b_io_ir_type(b, d), rocke_b_get_param(b, n));", []),
        # Any callee counts, not just builder ops.
        ("helper(ctx, rocke_b_mul(b, p, q), rocke_b_const_i32(b, 1));", ["helper"]),
        # A listed wrapper emits; an unlisted helper that returns a type does not.
        (
            "x = rocke_b_mul(b, fh_lane_rg(ctx), rocke_b_const_i32(b, 4));",
            ["rocke_b_mul"],
        ),
        (
            "x = rocke_b_bitcast(b, rocke_b_mul(b, p, q), rocke_vector_type(b, t, 4));",
            [],
        ),
        # Nested sites are each reported.
        (
            (
                "x = rocke_b_add(b, rocke_b_mul(b, rocke_b_const_i32(b, 1), "
                "rocke_b_const_i32(b, 2)), rocke_b_const_i32(b, 3));"
            ),
            ["rocke_b_add", "rocke_b_mul"],
        ),
    ],
)
def test_find_sites(src, expected):
    assert _callees(src) == expected


def test_comments_strings_and_preprocessor_lines_are_ignored():
    src = (
        "/* rocke_b_add(b, rocke_b_mul(b, p, q), rocke_b_const_i32(b, 1)) */\n"
        "// rocke_b_add(b, rocke_b_mul(b, p, q), rocke_b_const_i32(b, 1))\n"
        "#define X rocke_b_add(b, rocke_b_mul(b, p, q), rocke_b_const_i32(b, 1))\n"
        'log("rocke_b_add(b, rocke_b_mul(b, p, q), rocke_b_const_i32(b, 1))");\n'
        "x = rocke_b_select(b, c, rocke_b_mul(b, p, q), y);\n"
    )
    assert _callees(src) == []


def test_reports_the_line_of_the_call():
    src = "int a;\n\nx = rocke_b_add(b,\n    rocke_b_mul(b, p, q),\n    rocke_b_const_i32(b, 1));\n"
    assert ceo.find_sites(src) == [ceo.Site(3, "rocke_b_add")]


def test_cpp_tree_has_no_sites_outside_known_sites():
    found = ceo.scan(p for p in ceo.CPP_ROOT.rglob("*") if p.is_file())
    assert ceo.problems(found, check_stale=True) == []


def test_a_new_site_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(ceo, "CPP_ROOT", tmp_path)
    bad = tmp_path / "instances" / "new.cpp"
    bad.parent.mkdir()
    bad.write_text(
        "v = rocke_b_add(b, rocke_b_mul(b, p, q), rocke_b_const_i32(b, 1));\n"
    )
    assert ceo.main([str(bad)]) == 1
    errors = ceo.problems(ceo.scan([bad]), check_stale=False)
    assert errors and errors[0].startswith("cpp/instances/new.cpp:1: rocke_b_add")


def test_a_fixed_known_site_must_leave_the_allowlist(monkeypatch):
    monkeypatch.setattr(ceo, "KNOWN_SITES", {("instances/gone.cpp", "rocke_b_add"): 1})
    assert ceo.problems({}, check_stale=True) == [
        (
            "cpp/instances/gone.cpp: KNOWN_SITES allows 1 rocke_b_add site(s) but 0 "
            "remain; lower or delete the entry"
        )
    ]


def test_file_mode_ignores_paths_outside_cpp(tmp_path):
    outside = tmp_path / "x.cpp"
    outside.write_text(
        "v = rocke_b_add(b, rocke_b_mul(b, p, q), rocke_b_const_i32(b, 1));\n"
    )
    assert ceo.main([str(outside)]) == 0


@pytest.mark.parametrize("name", sorted(ceo.EMITTING_WRAPPERS))
def test_every_listed_wrapper_is_defined(name):
    pattern = re.compile(rf"rocke_value_t\*\s*{re.escape(name)}\(")
    assert any(
        pattern.search(path.read_text(encoding="utf-8", errors="replace"))
        for path in ceo.CPP_ROOT.rglob("*.cpp")
    ), f"{name} is listed in EMITTING_WRAPPERS but no longer defined under cpp/"
