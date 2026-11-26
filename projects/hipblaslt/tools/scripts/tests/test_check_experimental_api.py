#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for check_experimental_api.py: does a change need a feature revision bump?

Each test copies the real experimental headers and contract tests into a "base" and a
"head" tree, edits the head the way a PR would, and checks the verdict. The trees also get the
fixture feature DEMO from data/, a small C feature that exercises the C rules.

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
_FIXTURES = Path(__file__).resolve().parent / "data"
_DEMO_HEADER = f"{cea.HEADER_DIR}/hipblaslt-experimental-demo.h"
_DEMO_CONTRACT = f"{cea.CONTRACT_DIR}/contract_demo_gtest.cpp"
_DEMO_FUNC = "HIPBLASLT_EXPORT\nhipblasStatus_t hipblasLtExperimentalDemoAlignSize"
_DEMO_ENUM_END = "} hipblasLtExperimentalDemoRounding_t;"
# The first value a new experimental enumerator may take in the real headers.
_NEXT_ENUM = cea.load_surface(cea.DirTree(_REAL_ROOT)).high_water + 1


def _copy_tree(dst: Path) -> Path:
    real = cea.DirTree(_REAL_ROOT)
    files = [cea.MAIN_HEADER]
    files += real.glob(cea.HEADER_DIR, cea.FEATURE_HEADER_GLOB)
    files += cea.contract_files(real)
    for rel in files:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REAL_ROOT / rel, dst / rel)
    for rel in (_DEMO_HEADER, _DEMO_CONTRACT):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_FIXTURES / Path(rel).name, dst / rel)
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
    assert surface.revisions == {"PARTITIONED_GEMM": 1}
    assert surface.high_water is not None
    assert surface.decls["function hipblasLtExperimentalGetFeatureRevision"][0] == cea.CORE
    assert "macro HIPBLASLT_EXPERIMENTAL_ENUM_BIT" in surface.decls


def test_fixture_feature_parses_cleanly(trees):
    surface = cea.load_surface(cea.DirTree(trees[0]))
    assert surface.errors == []
    assert surface.revisions == {"DEMO": 1, "PARTITIONED_GEMM": 1}
    assert surface.decls["function hipblasLtExperimentalDemoAlignSize"][0] == "DEMO"
    assert surface.contracts[_DEMO_CONTRACT][0] == "DEMO"


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
          f"    HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_NEAREST = {_NEXT_ENUM},\n{_DEMO_ENUM_END}")
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
          f"    HIPBLASLT_EXPERIMENTAL_DEMO_ROUND_NEAREST = {_NEXT_ENUM},\n{_DEMO_ENUM_END}")
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


# --- C++ features ----------------------------------------------------------------------------

_PG_HEADER = f"{cea.HEADER_DIR}/hipblaslt-experimental-partitioned-gemm.h"
_PG = "hipblaslt_experimental::PartitionedGemm"
_PG_COUNT = "        HIPBLASLT_EXPORT size_t getPartitionCount() const;\n"
_PG_WORKSPACE = "HIPBLASLT_EXPORT void setMaxWorkspaceBytes(size_t workspaceBytes);"
_PG_PIMPL = "        std::unique_ptr<PartitionedGemmImpl> pimpl;\n"
_PG_BATCH = "        Batch = 2,"


def _bump_pg(root: Path) -> None:
    _bump_macro(root, _PG_HEADER, "HIPBLASLT_EXPERIMENTAL_PARTITIONED_GEMM")


def _add_cpp_feature(root: Path, body: str, name: str = "cpp") -> None:
    (root / cea.HEADER_DIR / f"hipblaslt-experimental-{name}.h").write_text(
        f"#pragma once\n#define HIPBLASLT_EXPERIMENTAL_{name.upper()} 1\n{body}\n", encoding="utf-8"
    )


def test_cpp_header_declarations():
    surface = cea.load_surface(cea.DirTree(_REAL_ROOT))
    decls = {k for k, (f, _) in surface.decls.items() if f == "PARTITIONED_GEMM"}
    assert f"class {_PG}" in decls
    assert f"method {_PG}::getPartitionCount()const" in decls
    assert f"method {_PG}::setMaxWorkspaceBytes(size_t)" in decls
    # Overloads are separate declarations; parameter names are not part of the key.
    runs = sorted(k for k in decls if k.startswith(f"method {_PG}::run("))
    assert len(runs) == 2
    assert f"method {_PG}::initialize(const hipblasLtMatmulAlgo_t&,const std::vector<void*>&,bool,"\
           "const std::vector<hipStream_t>&)" in decls
    assert surface.enums["enumerator hipblaslt_experimental::SplitDim::Batch"] == 2
    # Private members are layout, not API.
    assert not any("PartitionedGemmImpl" in k for k in decls if not k.startswith("class "))
    assert "pimpl" in surface.decls[f"class {_PG}"][1]


def test_cpp_adding_a_method_passes_without_bump(trees):
    base, head = trees
    _edit(head, _PG_HEADER, _PG_COUNT, _PG_COUNT + "        HIPBLASLT_EXPORT int getNewThing() const;\n")
    report = _check(base, head)
    assert report.ok and not report.breaking, report.errors
    assert f"added    method {_PG}::getNewThing()const (PARTITIONED_GEMM)" in report.notes


def test_cpp_adding_an_overload_passes_without_bump(trees):
    base, head = trees
    _edit(head, _PG_HEADER, _PG_WORKSPACE, _PG_WORKSPACE + "\n        HIPBLASLT_EXPORT void setMaxWorkspaceBytes(const std::vector<size_t>& perPartition);")
    report = _check(base, head)
    assert report.ok and not report.breaking, report.errors


def test_cpp_renaming_a_parameter_is_no_change(trees):
    base, head = trees
    _edit(head, _PG_HEADER, _PG_WORKSPACE, "HIPBLASLT_EXPORT void setMaxWorkspaceBytes(size_t bytes);")
    report = _check(base, head)
    assert report.ok and not report.breaking and not report.notes, report.notes


def test_cpp_removing_a_method_needs_bump(trees):
    base, head = trees
    _edit(head, _PG_HEADER, _PG_COUNT, "")
    report = _check(base, head)
    assert not report.ok
    assert f"removed  method {_PG}::getPartitionCount()const" in report.breaking["PARTITIONED_GEMM"]
    _bump_pg(head)
    assert _check(base, head).ok


def test_cpp_new_parameter_list_is_a_removal_and_an_addition(trees):
    base, head = trees
    _edit(head, _PG_HEADER, _PG_WORKSPACE, "HIPBLASLT_EXPORT void setMaxWorkspaceBytes(uint64_t workspaceBytes);")
    assert not _check(base, head).ok
    _bump_pg(head)
    report = _check(base, head)
    assert report.ok, report.errors


@pytest.mark.parametrize("old,new", [
    (_PG_COUNT, "        HIPBLASLT_EXPORT int64_t getPartitionCount() const;\n"),  # return type
    ("storePeerD = false);", "storePeerD = true);"),  # default argument
    (_PG_COUNT, "        HIPBLASLT_EXPORT size_t getPartitionCount() const noexcept;\n"),  # qualifier
])
def test_cpp_changing_a_method_in_place_is_never_allowed(trees, old, new):
    base, head = trees
    _edit(head, _PG_HEADER, old, new)
    _bump_pg(head)
    report = _check(base, head)
    assert any(e.startswith(f"changed method {_PG}::") and "in place" in e for e in report.errors), report.errors


@pytest.mark.parametrize("member", [
    "        int extra = 0;\n",  # a data member
    "        HIPBLASLT_EXPORT virtual void hook();\n",  # a virtual function (vtable)
])
def test_cpp_changing_a_class_layout_is_never_allowed(trees, member):
    base, head = trees
    _edit(head, _PG_HEADER, _PG_PIMPL, _PG_PIMPL + member)
    _bump_pg(head)
    report = _check(base, head)
    assert any(e.startswith(f"changed class {_PG} in place") for e in report.errors), report.errors


def test_cpp_private_method_is_not_api(trees):
    base, head = trees
    _edit(head, _PG_HEADER, _PG_PIMPL, _PG_PIMPL + "        void helper();\n")
    report = _check(base, head)
    assert report.ok and not report.notes, report.notes


def test_cpp_scoped_enum_value_is_never_changed(trees):
    base, head = trees
    _edit(head, _PG_HEADER, _PG_BATCH, "        Batch = 3,")
    _raise_high_water(head)
    _bump_pg(head)
    report = _check(base, head)
    assert any("changed value of enumerator hipblaslt_experimental::SplitDim::Batch" in e for e in report.errors)


def test_cpp_scoped_enum_value_is_never_reused(trees):
    base, head = trees
    _edit(head, _PG_HEADER, _PG_BATCH, f"{_PG_BATCH}\n        K = 1,")
    report = _check(base, head)
    assert any("SplitDim::K has value 1, at or below" in e for e in report.errors), report.errors


def test_cpp_new_scoped_enumerator_above_high_water_passes(trees):
    base, head = trees
    _edit(head, _PG_HEADER, _PG_BATCH, f"{_PG_BATCH}\n        K = {_NEXT_ENUM},")
    _raise_high_water(head)
    report = _check(base, head)
    assert report.ok and not report.breaking, report.errors


def test_cpp_scoped_enumerator_needs_explicit_value(trees):
    base, head = trees
    _edit(head, _PG_HEADER, _PG_BATCH, f"{_PG_BATCH}\n        K,")
    report = _check(base, head)
    assert any("SplitDim::K needs an explicit value" in e for e in report.errors), report.errors


def test_cpp_template_is_rejected(trees):
    base, head = trees
    _add_cpp_feature(head, "namespace hipblaslt_experimental { template <typename T> T twice(T x); }")
    report = _check(base, head)
    assert any("templates are not supported" in e for e in report.errors), report.errors


def test_cpp_inline_bodies_and_braced_initializers():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = _copy_tree(Path(tmp))
        _add_cpp_feature(root, """
namespace hipblaslt_experimental::nested
{
    struct Options
    {
        int  count{2};
        bool on = true;
        int  twice() const { return count * 2; }
        HIPBLASLT_EXPORT void set(const std::vector<int>& v = {}, int n = 1);
    };
    inline int helper(int x) { if(x) { return 1; } return 0; }
    enum class Mode : int { A = 7, B = 8 };
}
""")
        surface = cea.load_surface(cea.DirTree(root))
    decls = {k for k, (f, _) in surface.decls.items() if f == "CPP"}
    scope = "hipblaslt_experimental::nested"
    assert f"class {scope}::Options" in decls
    assert f"method {scope}::Options::twice()const" in decls
    assert f"method {scope}::Options::set(const std::vector<int>&,int)" in decls
    assert f"function {scope}::helper(int)" in decls
    assert surface.decls[f"enum {scope}::Mode"][1] == "enum class:int"
    assert surface.enums[f"enumerator {scope}::Mode::B"] == 8
    layout = surface.decls[f"class {scope}::Options"][1]
    assert "count{2}" in layout and "on=true" in layout and "twice" not in layout


def test_contract_header_counts_as_contract(trees):
    base, head = trees
    for root in trees:
        (root / cea.CONTRACT_DIR / "contract_partitioned_gemm_more_helpers.hpp").write_text("inline int f() { return 1; }\n")
    _edit(head, f"{cea.CONTRACT_DIR}/contract_partitioned_gemm_more_helpers.hpp", "return 1", "return 2")
    report = _check(base, head)
    assert not report.ok
    assert any("changed contract test" in c for c in report.breaking["PARTITIONED_GEMM"])
    assert cea.feature_of_contract("x/contract_gemm_a2a_helpers.hpp", {"GEMM_A2A"}) == "GEMM_A2A"
