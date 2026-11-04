# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tools/check_emission_order.py: the scanner, and the C++ tree held to it.

Source-tree only (it reads cpp/ and tools/); excluded from the installed tree.
The pre-commit hooks rocke-emission-order and rocke-emission-order-tests run
the tree check and this file.
"""

from __future__ import annotations

import importlib.util
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


def _callees(src: str, emitters=frozenset()) -> list[str]:
    return [site.callee for site in ceo.find_sites(src, frozenset(emitters))]


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
        # An undefined helper (e.g. one returning a type) does not emit.
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
        # A lambda body runs when the callee calls it, not with the arguments.
        ("e.if_(rocke_b_cmp_lt(b, x, y), [&]() { rocke_b_store(b, p, v); });", []),
        ("e.if_(rocke_b_cmp_lt(b, x, y), [&] { rocke_b_store(b, p, v); });", []),
        # A braced list is evaluated with its argument, so it counts.
        (
            "f(rocke_b_mul(b, p, q), Pair{rocke_b_const_i32(b, 1), y});",
            ["f"],
        ),
    ],
)
def test_find_sites(src, expected):
    assert _callees(src) == expected


def test_a_wrapper_passed_in_emits():
    src = "x = rocke_b_mul(b, fh_lane_rg(ctx), rocke_b_const_i32(b, 4));"
    assert _callees(src) == []
    assert _callees(src, {"fh_lane_rg"}) == ["rocke_b_mul"]


def test_emitting_functions_are_derived_transitively():
    src = (
        "static V lane(Ctx* c) { if(!c->v) { c->v = rocke_b_div(c->b, c->t, k); } "
        "return c->v; }\n"
        "static V lane2(Ctx* c) { return lane(c); }\n"
        "static const T* vec_t(B* b, int n) { return rocke_vector(rocke_i32(), n); }\n"
        "static V info(B* b) { return rocke_b_get_param(b, 0); }\n"
    )
    text = ceo.strip_source(src)
    assert ceo.emitting_functions([text]) == {"lane", "lane2"}


def test_member_functions_are_derived():
    src = (
        "struct Em {\n"
        "    rocke_ir_builder_t* b;\n"
        "    V ci(int64_t v) const { return rocke_b_const_i32(b, v); }\n"
        "    V mul(V x, V y) const\n    {\n        return rocke_b_mul(b, x, y);\n    }\n"
        "};\n"
        "V f(Em e, V x, V y) { return e.add(e.mul(x, y), e.ci(1)); }\n"
    )
    assert _callees(src) == ["add"]


def test_control_flow_and_lambdas_are_not_definitions():
    src = (
        "void f(B* b) {\n"
        "    if(x) { rocke_b_store(b, p, v); }\n"
        "    for(int i = 0; i < 4; ++i) { rocke_b_store(b, p, v); }\n"
        "    auto g = [&](V a) { return rocke_b_neg(b, a); };\n"
        "}\n"
    )
    assert ceo.emitting_functions([ceo.strip_source(src)]) == {"f"}


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


@pytest.fixture(scope="module")
def tree_emitters():
    texts = [
        ceo.strip_source(p.read_text(encoding="utf-8", errors="replace"))
        for root in ceo.DEFAULT_ROOTS
        for p in root.rglob("*")
        if p.is_file() and p.suffix in ceo.SUFFIXES
    ]
    return ceo.emitting_functions(texts)


def test_tree_derivation_finds_the_lane_and_member_wrappers(tree_emitters):
    for name in (
        "fh_lane_rg",
        "lane_rg_of",
        "rocke__lane_half_base",
        "rocke_g950a2d_lane_rg",
        "rocke_gfx1151_dfcp_splat_i32",
        "ci",
        "mul",
    ):
        assert name in tree_emitters, name
    assert "rocke_vector_type" not in tree_emitters


@pytest.mark.parametrize(
    "mutant",
    [
        "V m = e.add(e.mul(x, y), e.ci(1));",
        "V m = rocke_b_mul(b, fh_lane_rg(ctx), rocke_b_const_i32(b, 4));",
        "V m = rocke_b_add(b, rocke_gfx1151_dfcp_splat_i32(b, 1, n), lane_rg_of(ctx));",
    ],
)
def test_a_new_site_through_a_tree_function_is_reported(mutant, tree_emitters):
    assert len(ceo.find_sites(mutant, tree_emitters)) == 1


def test_cpp_tree_has_no_sites_outside_known_sites():
    assert ceo.problems(ceo.scan(ceo.DEFAULT_ROOTS)) == []


def test_a_new_site_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(ceo, "KNOWN_SITES", {})
    bad = tmp_path / "new.cpp"
    bad.write_text(
        "v = rocke_b_add(b, rocke_b_mul(b, p, q), rocke_b_const_i32(b, 1));\n"
    )
    found = ceo.scan([tmp_path])
    errors = ceo.problems(found)
    assert len(errors) == 1 and ": rocke_b_add(...) has two or more" in errors[0]
    monkeypatch.setattr(ceo, "DEFAULT_ROOTS", ())
    assert ceo.main(["--also", str(tmp_path)]) == 1


def test_a_fixed_known_site_must_leave_the_allowlist(monkeypatch):
    monkeypatch.setattr(
        ceo, "KNOWN_SITES", {("cpp/instances/gone.cpp", "rocke_b_add"): 1}
    )
    assert ceo.problems({}) == [
        (
            "cpp/instances/gone.cpp: KNOWN_SITES allows 1 rocke_b_add site(s) but 0 "
            "remain; lower or delete the entry"
        )
    ]


def test_known_sites_are_keyed_by_path_under_platform():
    found = ceo.scan(ceo.DEFAULT_ROOTS)
    for rel, _ in ceo.KNOWN_SITES:
        assert rel in found, rel
        assert (ceo.PLATFORM / rel).is_file(), rel
