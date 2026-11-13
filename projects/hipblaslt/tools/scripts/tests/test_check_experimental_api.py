#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for check_experimental_api.py: does a change need a feature revision bump?

Each test copies the real experimental headers and contract tests into a "base" and a
"head" tree, edits the head the way a PR would, and checks the verdict.

Pure standard library, so it runs in any Python env:
    python3 -m pytest tools/scripts/tests/test_check_experimental_api.py
"""

import re
import shutil
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS))

import check_experimental_api as cea

_REAL_ROOT = cea.PROJECT_ROOT
_DEMO_HEADER = f"{cea.HEADER_DIR}/hipblaslt-experimental-demo.h"
_DEMO_CONTRACT = f"{cea.CONTRACT_DIR}/contract_demo_gtest.cpp"
_DEMO_FUNC = "HIPBLASLT_EXPORT\nhipblasStatus_t hipblasLtExperimentalDemoAlignSize"
_DEMO_ENUM_END = "} hipblasLtExperimentalDemoRounding_t;"


def _copy_tree(dst: Path) -> Path:
    real = cea.DirTree(_REAL_ROOT)
    files = [cea.MAIN_HEADER]
    files += real.glob(cea.HEADER_DIR, cea.FEATURE_HEADER_GLOB)
    files += real.glob(cea.CONTRACT_DIR, cea.CONTRACT_GLOB)
    for rel in files:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REAL_ROOT / rel, dst / rel)
    return dst


@pytest.fixture
def trees(tmp_path):
    return _copy_tree(tmp_path / "base"), _copy_tree(tmp_path / "head")


def _edit(root: Path, rel: str, old: str, new: str) -> None:
    p = root / rel
    text = p.read_text(encoding="utf-8")
    assert old in text, f"{old!r} not in {rel}"
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def _bump_macro(root: Path, rel: str, macro: str) -> None:
    p = root / rel
    text = p.read_text(encoding="utf-8")
    p.write_text(
        re.sub(rf"(#define {macro} )(\d+)", lambda m: m.group(1) + str(int(m.group(2)) + 1), text),
        encoding="utf-8",
    )


def _bump(root: Path) -> None:
    _bump_macro(root, _DEMO_HEADER, "HIPBLASLT_EXPERIMENTAL_DEMO")


def _raise_high_water(root: Path) -> None:
    _bump_macro(root, cea.MAIN_HEADER, cea.HIGH_WATER_MACRO)


def _add_feature(root: Path, name: str = "other") -> str:
    rel = f"{cea.HEADER_DIR}/hipblaslt-experimental-{name}.h"
    (root / rel).write_text(
        f"#pragma once\n#define HIPBLASLT_EXPERIMENTAL_{name.upper()} 1\n"
        f"HIPBLASLT_EXPORT int hipblasLtExperimental{name.title()}(void);\n",
        encoding="utf-8",
    )
    return rel


def _check(base: Path, head: Path) -> cea.Report:
    return cea.compare(cea.load_surface(cea.DirTree(base)), cea.load_surface(cea.DirTree(head)))


# --- Not breaking --------------------------------------------------------------------------


def test_identical_trees_pass(trees):
    report = _check(*trees)
    assert report.ok and not report.breaking and not report.notes


def test_real_headers_parse_cleanly():
    surface = cea.load_surface(cea.DirTree(_REAL_ROOT))
    assert surface.errors == []
    assert surface.revisions == {"DEMO": 1}
    assert surface.high_water is not None
    assert surface.decls["function hipblasLtExperimentalGetFeatureRevision"][0] == cea.CORE
    assert surface.decls["function hipblasLtExperimentalDemoAlignSize"][0] == "DEMO"
    assert "macro HIPBLASLT_EXPERIMENTAL_ENUM_BIT" in surface.decls


def test_feature_names_from_paths():
    assert cea.feature_of_header("x/hipblaslt-experimental-gemm-a2a.h") == "GEMM_A2A"
    features = {"GEMM", "GEMM_A2A"}
    assert cea.feature_of_contract("x/contract_gemm_a2a_gtest.cpp", features) == "GEMM_A2A"
    assert cea.feature_of_contract("x/contract_gemm_a2a_errors_gtest.cpp", features) == "GEMM_A2A"
    assert cea.feature_of_contract("x/contract_gemm_gtest.cpp", features) == "GEMM"
    assert cea.feature_of_contract("x/contract_nope_gtest.cpp", features) == "NOPE"


def test_comment_and_whitespace_changes_pass(trees):
    base, head = trees
    _edit(head, _DEMO_HEADER, "size_t*                             result);",
          "size_t *result /* output */);")
    _edit(head, _DEMO_CONTRACT, "namespace\n", "// a new comment\nnamespace\n")
    report = _check(base, head)
    assert report.ok and not report.breaking


def test_crlf_checkout_passes(trees):
    base, head = trees
    for p in [head / _DEMO_HEADER, head / _DEMO_CONTRACT, head / cea.MAIN_HEADER]:
        p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    report = _check(base, head)
    assert report.ok and not report.breaking


def test_adding_a_function_passes_without_bump(trees):
    base, head = trees
    _edit(head, _DEMO_HEADER, "#ifdef __cplusplus\n}",
          "HIPBLASLT_EXPORT int hipblasLtExperimentalDemoNew(void);\n\n#ifdef __cplusplus\n}")
    report = _check(base, head)
    assert report.ok and not report.breaking
    assert "added    function hipblasLtExperimentalDemoNew (DEMO)" in report.notes


def test_adding_an_enumerator_above_high_water_passes_without_bump(trees):
    base, head = trees
    _edit(head, _DEMO_HEADER, _DEMO_ENUM_END,
          f"    HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_NEAREST = 2,\n{_DEMO_ENUM_END}")
    _raise_high_water(head)
    report = _check(base, head)
    assert report.ok and not report.breaking, report.errors


def test_adding_a_contract_test_passes_without_bump(trees):
    base, head = trees
    (head / cea.CONTRACT_DIR / "contract_demo_more_gtest.cpp").write_text("TEST(A, B) {}\n")
    report = _check(base, head)
    assert report.ok and not report.breaking, report.errors
    assert f"added contract test {cea.CONTRACT_DIR}/contract_demo_more_gtest.cpp" in report.notes


def test_adding_a_feature_passes(trees):
    base, head = trees
    _add_feature(head)
    report = _check(base, head)
    assert report.ok and not report.breaking
    assert "new feature OTHER at revision 1" in report.notes


# --- Breaking without a bump fails, with a bump passes ------------------------------------

BREAKING_EDITS = {
    "remove_function": (_DEMO_HEADER, _DEMO_FUNC, _DEMO_FUNC + "_v2"),
    "retire_enumerator": (_DEMO_HEADER, "    HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_UP   = 1, /**< Round up to a multiple of the alignment. */\n", ""),
    "change_contract_test": (_DEMO_CONTRACT, "EXPECT_EQ(r, 128u);\n    }\n\n    TEST(ExperimentalContractDemo, AlignedValueIsUnchanged)",
                             "EXPECT_EQ(r, 192u);\n    }\n\n    TEST(ExperimentalContractDemo, AlignedValueIsUnchanged)"),
}


@pytest.mark.parametrize("edit", sorted(BREAKING_EDITS))
def test_breaking_change_without_bump_fails(trees, edit):
    base, head = trees
    _edit(head, *BREAKING_EDITS[edit])
    report = _check(base, head)
    assert "DEMO" in report.breaking
    assert not report.ok
    assert any("without bumping HIPBLASLT_EXPERIMENTAL_DEMO" in e for e in report.errors)


@pytest.mark.parametrize("edit", sorted(BREAKING_EDITS))
def test_breaking_change_with_bump_passes(trees, edit):
    base, head = trees
    _edit(head, *BREAKING_EDITS[edit])
    _bump(head)
    report = _check(base, head)
    assert report.breaking and report.ok, report.errors
    assert "feature DEMO: revision 1 -> 2" in report.notes


def test_bump_is_per_feature(trees):
    base, head = trees
    other = _add_feature(base)
    shutil.copyfile(base / other, head / other)
    _edit(head, *BREAKING_EDITS["remove_function"])
    _bump_macro(head, other, "HIPBLASLT_EXPERIMENTAL_OTHER")  # bumps the wrong feature
    report = _check(base, head)
    assert any("without bumping HIPBLASLT_EXPERIMENTAL_DEMO" in e for e in report.errors)
    assert any("HIPBLASLT_EXPERIMENTAL_OTHER bumped" in w for w in report.warnings)


def test_deleting_a_contract_test_fails(trees):
    base, head = trees
    (head / _DEMO_CONTRACT).unlink()
    report = _check(base, head)
    assert f"deleted contract test {_DEMO_CONTRACT}" in report.breaking["DEMO"]
    assert not report.ok


def test_removing_a_feature_passes(trees):
    # Graduation or removal: the header, its revision macro and its contract tests all go.
    base, head = trees
    (head / _DEMO_HEADER).unlink()
    (head / _DEMO_CONTRACT).unlink()
    report = _check(base, head)
    assert report.ok, report.errors
    assert "feature DEMO removed or graduated (revision macro gone)" in report.notes


# --- Never allowed, even with a bump ------------------------------------------------------

NEVER_EDITS = {
    "change_signature_in_place": (
        (_DEMO_HEADER, "size_t*                             result);", "uint64_t* result);"),
        "in place",
    ),
    "renumber_enumerator": ((_DEMO_HEADER, "ROUND_UP   = 1", "ROUND_UP   = 3"), "never renumbered"),
    "change_core_macro": ((cea.MAIN_HEADER, "0x40000000", "0x20000000"), "never changes"),
    "change_runtime_query": (
        (cea.MAIN_HEADER, "int hipblasLtExperimentalGetFeatureRevision(const char* feature);",
         "int hipblasLtExperimentalGetFeatureRevision(const char* feature, int flags);"),
        "never changes",
    ),
}


@pytest.mark.parametrize("edit", sorted(NEVER_EDITS))
def test_never_allowed_even_with_bump(trees, edit):
    base, head = trees
    change, message = NEVER_EDITS[edit]
    _edit(head, *change)
    _bump(head)
    _raise_high_water(head)
    report = _check(base, head)
    assert not report.ok
    assert any(message in e for e in report.errors), report.errors


_DEMO_STRUCT = (
    "typedef struct hipblasLtExperimentalDemoParams_t\n{\n    size_t alignment;\n"
    "} hipblasLtExperimentalDemoParams_t;\n"
    "typedef int (*hipblasLtExperimentalDemoCallback_t)(size_t);\n"
)


def _add_demo_types(*roots: Path) -> None:
    for root in roots:
        _edit(root, _DEMO_HEADER, "#ifdef __cplusplus\n}", f"{_DEMO_STRUCT}\n#ifdef __cplusplus\n}}")


TYPE_EDITS = {
    "add_struct_member": ("    size_t alignment;\n", "    size_t alignment;\n    int flags;\n"),
    "change_struct_member": ("    size_t alignment;\n", "    uint32_t alignment;\n"),
    "change_typedef": ("int (*hipblasLtExperimentalDemoCallback_t)(size_t)",
                       "int (*hipblasLtExperimentalDemoCallback_t)(size_t, void*)"),
}


@pytest.mark.parametrize("edit", sorted(TYPE_EDITS))
def test_changing_a_type_in_place_fails_even_with_bump(trees, edit):
    base, head = trees
    _add_demo_types(base, head)
    _edit(head, _DEMO_HEADER, *TYPE_EDITS[edit])
    _bump(head)
    report = _check(base, head)
    assert not report.ok
    assert any("in place" in e and e.startswith("changed type ") for e in report.errors), report.errors


def test_replacing_a_type_with_a_renamed_one_passes_with_bump(trees):
    base, head = trees
    _add_demo_types(base, head)
    _edit(head, _DEMO_HEADER, "    size_t alignment;\n} hipblasLtExperimentalDemoParams_t;",
          "    size_t alignment;\n    int flags;\n} hipblasLtExperimentalDemoParams_v2_t;")
    _edit(head, _DEMO_HEADER, "typedef struct hipblasLtExperimentalDemoParams_t",
          "typedef struct hipblasLtExperimentalDemoParams_v2_t")
    report = _check(base, head)
    assert any("without bumping HIPBLASLT_EXPERIMENTAL_DEMO" in e for e in report.errors)
    _bump(head)
    report = _check(base, head)
    assert report.ok, report.errors
    assert "removed  type hipblasLtExperimentalDemoParams_t" in report.breaking["DEMO"]


def test_reusing_a_retired_enum_value_fails(trees):
    base, head = trees
    _edit(head, *BREAKING_EDITS["retire_enumerator"])
    _edit(head, _DEMO_HEADER, _DEMO_ENUM_END,
          f"    HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_NEAREST = 1,\n{_DEMO_ENUM_END}")
    _bump(head)
    report = _check(base, head)
    assert any("never reused" in e for e in report.errors)


def test_new_enumerator_must_raise_high_water(trees):
    base, head = trees
    _edit(head, _DEMO_HEADER, _DEMO_ENUM_END,
          f"    HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_NEAREST = 2,\n{_DEMO_ENUM_END}")
    report = _check(base, head)
    assert any("raise it" in e for e in report.errors)


def test_high_water_never_goes_down(trees):
    base, head = trees
    _raise_high_water(base)
    report = _check(base, head)
    assert any("went down" in e for e in report.errors)


def test_stable_enum_extension_macro_is_an_enumerator(trees):
    base, head = trees
    for root in trees:
        _edit(root, _DEMO_HEADER, "#ifdef __cplusplus\nextern",
              "#define HIPBLASLT_EPILOGUE_EXPERIMENTAL_DEMO"
              " ((hipblasLtEpilogue_t)(HIPBLASLT_EXPERIMENTAL_ENUM_BIT | 1))\n\n#ifdef __cplusplus\nextern")
    surface = cea.load_surface(cea.DirTree(head))
    assert surface.enums["enumerator hipblasLtEpilogue_t::HIPBLASLT_EPILOGUE_EXPERIMENTAL_DEMO"] == 1
    _edit(head, _DEMO_HEADER, "ENUM_BIT | 1))", "ENUM_BIT | 0))")
    report = _check(base, head)
    assert any("never renumbered" in e for e in report.errors)


# --- Other rules --------------------------------------------------------------------------


def test_revision_decrease_fails(trees):
    base, head = trees
    _bump(base)
    report = _check(base, head)
    assert any("went backwards" in e for e in report.errors)


def test_bump_without_breaking_change_warns(trees):
    base, head = trees
    _bump(head)
    report = _check(base, head)
    assert report.ok
    assert any("no breaking change" in w for w in report.warnings)


def test_missing_revision_macro_fails(trees):
    base, head = trees
    _edit(head, _DEMO_HEADER, "#define HIPBLASLT_EXPERIMENTAL_DEMO 1\n", "")
    report = _check(base, head)
    assert any("no #define HIPBLASLT_EXPERIMENTAL_DEMO" in e for e in report.errors)


def test_contract_test_for_unknown_feature_fails(trees):
    base, head = trees
    (head / cea.CONTRACT_DIR / "contract_nope_gtest.cpp").write_text("TEST(A, B) {}\n")
    report = _check(base, head)
    assert any("no experimental feature NOPE" in e for e in report.errors)


def test_enumerator_without_explicit_value_fails(trees):
    base, head = trees
    _edit(head, _DEMO_HEADER, _DEMO_ENUM_END,
          f"    HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_NEAREST,\n{_DEMO_ENUM_END}")
    report = _check(base, head)
    assert any("needs an explicit value" in e for e in report.errors)


def test_base_without_experimental_api_passes(tmp_path):
    head = _copy_tree(tmp_path / "head")
    report = cea.compare(cea.load_surface(cea.DirTree(tmp_path / "empty")), cea.load_surface(cea.DirTree(head)))
    assert report.ok


def test_main_exit_codes(trees, capsys):
    base, head = trees
    assert cea.main(["--base-dir", str(base), "--head-dir", str(head)]) == 0
    _edit(head, *BREAKING_EDITS["remove_function"])
    assert cea.main(["--base-dir", str(base), "--head-dir", str(head)]) == 1
    out = capsys.readouterr().out
    assert "[DEMO] removed  function hipblasLtExperimentalDemoAlignSize" in out
    assert "FAILED" in out
