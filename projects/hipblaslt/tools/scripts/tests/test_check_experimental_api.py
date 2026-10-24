#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for check_experimental_api.py: does a change need an experimental API bump?

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


def _bump(root: Path) -> None:
    p = root / cea.MAIN_HEADER
    text = p.read_text(encoding="utf-8")
    p.write_text(
        re.sub(
            rf"(#define {cea.VERSION_MACRO} )(\d+)",
            lambda m: m.group(1) + str(int(m.group(2)) + 1),
            text,
        ),
        encoding="utf-8",
    )


def _check(base: Path, head: Path) -> cea.Report:
    return cea.compare(cea.load_surface(cea.DirTree(base)), cea.load_surface(cea.DirTree(head)))


# --- Not breaking --------------------------------------------------------------------------


def test_identical_trees_pass(trees):
    report = _check(*trees)
    assert report.ok and not report.breaking and not report.added


def test_real_headers_parse_cleanly():
    surface = cea.load_surface(cea.DirTree(_REAL_ROOT))
    assert surface.errors == []
    assert surface.version is not None
    assert "function hipblasLtExperimentalGetApiVersion" in surface.decls
    assert "macro HIPBLASLT_EXPERIMENTAL_ENUM_BIT" in surface.decls


def test_comment_and_whitespace_changes_pass(trees):
    base, head = trees
    _edit(head, _DEMO_HEADER, "size_t*                             result);",
          "size_t *result /* output */);")
    _edit(head, _DEMO_CONTRACT, "namespace\n", "// a new comment\nnamespace\n")
    report = _check(base, head)
    assert report.ok and not report.breaking


def test_crlf_checkout_passes(trees):
    base, head = trees
    for p in [head / _DEMO_HEADER, head / _DEMO_CONTRACT]:
        p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert not _check(base, head).breaking


def test_adding_a_function_passes_without_bump(trees):
    base, head = trees
    _edit(head, _DEMO_HEADER, "#ifdef __cplusplus\n}",
          "HIPBLASLT_EXPORT int hipblasLtExperimentalDemoNew(void);\n\n#ifdef __cplusplus\n}")
    report = _check(base, head)
    assert report.ok and not report.breaking
    assert "added    function hipblasLtExperimentalDemoNew" in report.added


def test_adding_an_enumerator_passes_without_bump(trees):
    base, head = trees
    _edit(head, _DEMO_HEADER, "} hipblasLtExperimentalDemoRounding_t;",
          "    HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_NEAREST = 2,\n} hipblasLtExperimentalDemoRounding_t;")
    report = _check(base, head)
    assert report.ok and not report.breaking


def test_adding_a_contract_test_passes_without_bump(trees):
    base, head = trees
    (head / cea.CONTRACT_DIR / "contract_more_gtest.cpp").write_text("TEST(A, B) {}\n")
    report = _check(base, head)
    assert report.ok and not report.breaking


# --- Breaking without a bump fails, with a bump passes ------------------------------------

BREAKING_EDITS = {
    "remove_function": (
        _DEMO_HEADER,
        "HIPBLASLT_EXPORT\nhipblasStatus_t hipblasLtExperimentalDemoAlignSize",
        "HIPBLASLT_EXPORT\nhipblasStatus_t hipblasLtExperimentalDemoAlignSizeRenamed",
    ),
    "change_signature": (_DEMO_HEADER, "size_t*                             result);",
                         "uint64_t* result);"),
    "renumber_enumerator": (_DEMO_HEADER, "ROUND_UP   = 1", "ROUND_UP   = 2"),
    "change_macro": (cea.MAIN_HEADER, "0x40000000", "0x20000000"),
    "change_contract_test": (_DEMO_CONTRACT, "EXPECT_EQ(r, 128u);\n    }\n\n    TEST(ExperimentalContractDemo, AlignedValueIsUnchanged)",
                             "EXPECT_EQ(r, 192u);\n    }\n\n    TEST(ExperimentalContractDemo, AlignedValueIsUnchanged)"),
}


@pytest.mark.parametrize("edit", sorted(BREAKING_EDITS))
def test_breaking_change_without_bump_fails(trees, edit):
    base, head = trees
    _edit(head, *BREAKING_EDITS[edit])
    report = _check(base, head)
    assert report.breaking
    assert not report.ok
    assert any("without bumping" in e for e in report.errors)


@pytest.mark.parametrize("edit", sorted(BREAKING_EDITS))
def test_breaking_change_with_bump_passes(trees, edit):
    base, head = trees
    _edit(head, *BREAKING_EDITS[edit])
    _bump(head)
    report = _check(base, head)
    assert report.breaking and report.ok


def test_deleting_a_contract_test_fails(trees):
    base, head = trees
    (head / _DEMO_CONTRACT).unlink()
    report = _check(base, head)
    assert f"deleted contract test {_DEMO_CONTRACT}" in report.breaking
    assert not report.ok


def test_removing_a_feature_header_fails(trees):
    base, head = trees
    (head / _DEMO_HEADER).unlink()
    report = _check(base, head)
    assert "removed  function hipblasLtExperimentalDemoAlignSize" in report.breaking
    assert not report.ok


# --- Other rules --------------------------------------------------------------------------


def test_version_decrease_fails(trees):
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


def test_enumerator_without_explicit_value_fails(trees):
    base, head = trees
    _edit(head, _DEMO_HEADER, "} hipblasLtExperimentalDemoRounding_t;",
          "    HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_NEAREST,\n} hipblasLtExperimentalDemoRounding_t;")
    report = _check(base, head)
    assert any("needs an explicit value" in e for e in report.errors)


def test_base_without_experimental_api_passes(tmp_path):
    head = _copy_tree(tmp_path / "head")
    report = cea.compare(cea.load_surface(cea.DirTree(tmp_path / "empty")), cea.load_surface(cea.DirTree(head)))
    assert report.ok


def test_main_exit_codes(trees, capsys):
    base, head = trees
    assert cea.main(["--base-dir", str(base), "--head-dir", str(head)]) == 0
    _edit(head, *BREAKING_EDITS["change_signature"])
    assert cea.main(["--base-dir", str(base), "--head-dir", str(head)]) == 1
    assert "FAILED" in capsys.readouterr().out
