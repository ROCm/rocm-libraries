# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for check_public_abi.py.

The declaration parser is the only thing standing between the range entry
points and a divergence that links cleanly and corrupts arguments at run time,
so it is tested against inputs it must reject as well as ones it must accept.
Running the check against a tree that is already correct only exercises the
passing path.

The binary checks are covered the same way, against a stand-in that answers the
three questions they ask of a shared object. It deliberately skips the ELF
parser: that reads a real libMIOpen.so on every packaging build, so it is
covered by construction, whereas the conditions the checks exist to catch (a
stale half of a build, an entry point renamed with no stub) appear in no working
tree. The PE parser is tested directly, on images built here: only Windows runs
read a real DLL, and few of those select the ABI check.

Everything works on string fixtures and stand-ins, so no build, toolchain or GPU
is needed and the module runs in a lint lane:

    python -m pytest projects/miopen/script/test_check_public_abi.py
"""

import hashlib
import struct
import subprocess
import sys
from pathlib import Path

import pytest

# pytest's default import mode already puts this directory on sys.path; doing it
# explicitly keeps the module runnable under a bare `python -m pytest` from any
# directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_public_abi as abi  # noqa: E402

FWD = "miopenConvolutionForwardGetWorkSpaceSizeRange"
BWD = "miopenConvolutionBackwardDataGetWorkSpaceSizeRange"
WRW = "miopenConvolutionBackwardWeightsGetWorkSpaceSizeRange"

# The parameter list every range entry point carries, in the two spellings the
# tree actually contains: named in the definitions, unnamed in the consumers.
NAMED_PARAMS = """miopenHandle_t handle,
                  const miopenTensorDescriptor_t aDesc,
                  const miopenTensorDescriptor_t bDesc,
                  const miopenConvolutionDescriptor_t convDesc,
                  const miopenTensorDescriptor_t cDesc,
                  size_t* minWorkspaceSize,
                  size_t* maxWorkspaceSize"""
UNNAMED_PARAMS = """miopenHandle_t,
                    const miopenTensorDescriptor_t,
                    const miopenTensorDescriptor_t,
                    const miopenConvolutionDescriptor_t,
                    const miopenTensorDescriptor_t,
                    size_t*,
                    size_t*"""

EXPECTED_SIG = (
    "miopenStatus_t",
    "miopenHandle_t",
    "miopenTensorDescriptor_t",
    "miopenTensorDescriptor_t",
    "miopenConvolutionDescriptor_t",
    "miopenTensorDescriptor_t",
    "size_t*",
    "size_t*",
)


def definition(name: str, params: str = NAMED_PARAMS) -> str:
    """One entry point as src/convolution_api.cpp writes it."""
    return f"""
extern "C" MIOPEN_EXPORT miopenStatus_t {name}({params})
{{
    return miopen::try_([&] {{ /* body */ }});
}}
"""


def declaration(name: str, params: str = UNNAMED_PARAMS) -> str:
    """One entry point as a consumer declares it, inside extern "C"."""
    return f"miopenStatus_t {name}({params});"


def consumer(*decls: str) -> str:
    return 'extern "C" {\n' + "\n\n".join(decls) + "\n}\n"


# A declarator is recognised by what precedes it, so the fixture carries the
# kind of preamble the real file has rather than opening on a definition.
PREAMBLE = """
#include <miopen/miopen.h>
#include <miopen/convolution.hpp>

namespace {
constexpr std::size_t kUnused = 0;
}  // namespace
"""

DEFINITIONS_SOURCE = PREAMBLE + definition(FWD) + definition(BWD) + definition(WRW)
CONSUMER_SOURCE = consumer(declaration(FWD), declaration(BWD), declaration(WRW))


# --------------------------------------------------------------------------
# Parsing: the two forms a range entry point is written in
# --------------------------------------------------------------------------


def test_definitions_parse_through_export_and_extern_c():
    """`MIOPEN_EXPORT extern "C"` before the return type must not hide it."""
    found = abi.parse_range_prototypes(DEFINITIONS_SOURCE, "convolution_api.cpp")
    assert set(found) == {FWD, BWD, WRW}
    assert found[FWD] == EXPECTED_SIG


def test_declarations_parse_and_drop_parameter_names():
    """Named and unnamed spellings of the same prototype must agree."""
    found = abi.parse_range_prototypes(CONSUMER_SOURCE, "consumer.cpp")
    assert set(found) == {FWD, BWD, WRW}
    assert found[FWD] == EXPECTED_SIG


def test_matching_pair_passes(capsys):
    definitions = abi.parse_range_prototypes(DEFINITIONS_SOURCE, "convolution_api.cpp")
    declared = abi.parse_range_prototypes(CONSUMER_SOURCE, "consumer.cpp")
    assert abi.check_range_entry_point_set(declared, "consumer.cpp")
    assert abi.check_prototypes(
        definitions, declared, "consumer.cpp", "convolution_api.cpp"
    )
    assert "FAIL" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# The drift this gate exists to catch
# --------------------------------------------------------------------------


def test_drifted_parameter_type_is_reported(capsys):
    """A consumer whose parameter type differs must fail, and say which one."""
    drifted = consumer(
        declaration(FWD, UNNAMED_PARAMS.replace("size_t*,\n", "int*,\n", 1)),
        declaration(BWD),
        declaration(WRW),
    )
    definitions = abi.parse_range_prototypes(DEFINITIONS_SOURCE, "convolution_api.cpp")
    declared = abi.parse_range_prototypes(drifted, "consumer.cpp")

    # The name set is unchanged, so only the prototype comparison can see this.
    assert abi.check_range_entry_point_set(declared, "consumer.cpp")
    assert not abi.check_prototypes(
        definitions, declared, "consumer.cpp", "convolution_api.cpp"
    )
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert FWD in out
    assert "int*" in out


def test_drifted_return_type_is_reported(capsys):
    drifted = consumer(
        declaration(FWD).replace("miopenStatus_t", "int", 1),
        declaration(BWD),
        declaration(WRW),
    )
    definitions = abi.parse_range_prototypes(DEFINITIONS_SOURCE, "convolution_api.cpp")
    declared = abi.parse_range_prototypes(drifted, "consumer.cpp")
    assert not abi.check_prototypes(
        definitions, declared, "consumer.cpp", "convolution_api.cpp"
    )
    assert FWD in capsys.readouterr().out


def test_dropped_declaration_is_reported(capsys):
    """A consumer that stops declaring one entry point must fail."""
    declared = abi.parse_range_prototypes(
        consumer(declaration(FWD), declaration(BWD)), "consumer.cpp"
    )
    assert not abi.check_range_entry_point_set(declared, "consumer.cpp")
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert f"- missing from consumer.cpp: {WRW}" in out


def test_unknown_range_symbol_is_reported(capsys):
    """A name that looks like a range entry point but is not one must fail."""
    stray = "miopenConvolutionForwardBogusGetWorkSpaceSizeRange"
    declared = abi.parse_range_prototypes(
        consumer(
            declaration(FWD), declaration(BWD), declaration(WRW), declaration(stray)
        ),
        "consumer.cpp",
    )
    assert not abi.check_range_entry_point_set(declared, "consumer.cpp")
    assert f"+ present in consumer.cpp, not a known range entry point: {stray}" in (
        capsys.readouterr().out
    )


def test_declarator_at_the_very_start_of_a_file_is_not_silently_dropped(capsys):
    """The parser needs a boundary before a declarator; the miss must be loud.

    A definition with nothing before it has no preceding ';', '}' or '{' to mark
    where its return type begins, so the parser does not recognise it. No real
    source is written that way (both consumers open with `extern "C" {`, the
    library file with its includes), but nothing the gate fails to read may pass
    quietly. The entry point set check guarantees that, so pin it rather than
    the parser result: an unrecognised definition leaves the set short and fails.
    """
    found = abi.parse_range_prototypes(definition(FWD), "convolution_api.cpp")
    assert not abi.check_range_entry_point_set(found, "convolution_api.cpp")
    assert "FAIL" in capsys.readouterr().out


def test_conflicting_declarations_in_one_file_raise():
    source = consumer(
        declaration(FWD),
        declaration(FWD, UNNAMED_PARAMS.replace("size_t*,\n", "int*,\n", 1)),
    )
    with pytest.raises(abi.AbiError, match=FWD):
        abi.parse_range_prototypes(source, "consumer.cpp")


# --------------------------------------------------------------------------
# Call sites are not declarations
# --------------------------------------------------------------------------


CALLS_ONLY = f"""
static miopenStatus_t use(miopenHandle_t h)
{{
    size_t lo = 0;
    size_t hi = 0;
    {FWD}(h, a, b, c, d, &lo, &hi);
    auto rc = {BWD}(h, a, b, c, d, &lo, &hi);
    if(rc != miopenStatusSuccess)
        return {WRW}(h, a, b, c, d, &lo, &hi);
    return rc;
}}
"""


def test_call_sites_are_not_mistaken_for_declarations():
    """Statement, assignment and return positions must all read as calls."""
    assert abi.parse_range_prototypes(CALLS_ONLY, "consumer.cpp") == {}


def test_declarations_survive_alongside_call_sites():
    """The consumers both declare and call these; only the declarations count."""
    found = abi.parse_range_prototypes(CONSUMER_SOURCE + CALLS_ONLY, "consumer.cpp")
    assert set(found) == {FWD, BWD, WRW}
    assert found[FWD] == EXPECTED_SIG


def test_commented_out_declaration_is_ignored():
    source = consumer(declaration(FWD), declaration(BWD), declaration(WRW))
    source += f"\n// {declaration('miopenConvolutionStaleGetWorkSpaceSizeRange')}\n"
    source += f"/* {declaration('miopenConvolutionOlderGetWorkSpaceSizeRange')} */\n"
    assert set(abi.parse_range_prototypes(source, "consumer.cpp")) == {FWD, BWD, WRW}


# --------------------------------------------------------------------------
# The dispatch seam
#
# A stub that never reaches a dispatch macro silently loses the ability to route
# to hipDNN, and no build catches it: the macro's assert needs the macro to be
# present, and is compiled out under NDEBUG regardless.
#
# MIOPEN_WRAPPER_DISPATCH and MIOPEN_WRAPPER_FORWARD open the same seam, so both
# satisfy the check and a stub is expected to carry exactly one of them.
# --------------------------------------------------------------------------

# miopenGetErrorString is the one exempt stub, so it belongs in every fixture:
# an exemption for a stub the wrapper does not define is itself a finding.
EXEMPT_STUB = """
extern "C" const char* miopenGetErrorString(miopenStatus_t error)
{
    return miopenGetErrorString_impl(error);
}
"""


def wrapper_source(*stubs: str) -> str:
    return EXEMPT_STUB + "".join(stubs)


def stub(name: str, body: str) -> str:
    return f"""
extern "C" miopenStatus_t {name}(miopenHandle_t* handle)
{{
{body}
    return {name}_impl(handle);
}}
"""


def dispatches_of(source: str) -> dict:
    return abi.parse_wrapper(source)[2]


def test_stubs_dispatching_under_their_own_name_pass(capsys):
    source = wrapper_source(
        stub("miopenCreate", "    MIOPEN_WRAPPER_DISPATCH(miopenCreate);"),
        stub("miopenDestroy", "    MIOPEN_WRAPPER_DISPATCH(miopenDestroy);"),
    )
    assert abi.check_wrapper_dispatch(dispatches_of(source))
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "2 routable wrapper stubs" in out
    assert "1 exempt" in out


def test_stub_without_the_macro_is_reported(capsys):
    source = wrapper_source(
        stub("miopenCreate", "    MIOPEN_WRAPPER_DISPATCH(miopenCreate);"),
        stub("miopenDestroy", ""),
    )
    assert not abi.check_wrapper_dispatch(dispatches_of(source))
    out = capsys.readouterr().out
    assert (
        "miopenDestroy has no MIOPEN_WRAPPER_DISPATCH or MIOPEN_WRAPPER_FORWARD" in out
    )


def test_stub_dispatching_under_a_neighbours_name_is_reported(capsys):
    source = wrapper_source(
        stub("miopenCreate", "    MIOPEN_WRAPPER_DISPATCH(miopenCreate);"),
        stub("miopenDestroy", "    MIOPEN_WRAPPER_DISPATCH(miopenCreate);"),
    )
    assert not abi.check_wrapper_dispatch(dispatches_of(source))
    out = capsys.readouterr().out
    assert "miopenDestroy dispatches as miopenCreate" in out


def test_repeated_macro_is_reported(capsys):
    source = wrapper_source(
        stub(
            "miopenCreate",
            "    MIOPEN_WRAPPER_DISPATCH(miopenCreate);\n"
            "    MIOPEN_WRAPPER_DISPATCH(miopenCreate);",
        )
    )
    assert not abi.check_wrapper_dispatch(dispatches_of(source))
    assert "miopenCreate has 2 dispatch macros" in capsys.readouterr().out


def test_a_commented_out_macro_does_not_count_as_present(capsys):
    source = wrapper_source(
        stub("miopenCreate", "    // MIOPEN_WRAPPER_DISPATCH(miopenCreate);")
    )
    assert not abi.check_wrapper_dispatch(dispatches_of(source))
    out = capsys.readouterr().out
    assert (
        "miopenCreate has no MIOPEN_WRAPPER_DISPATCH or MIOPEN_WRAPPER_FORWARD" in out
    )


def test_forward_macro_satisfies_the_seam(capsys):
    source = wrapper_source(
        stub(
            "miopenCreate",
            "    MIOPEN_WRAPPER_FORWARD(miopenCreate,\n"
            "                           ::miopen::wrapper::hipdnn::Create(handle));",
        )
    )
    assert abi.check_wrapper_dispatch(dispatches_of(source))
    assert "1 routable wrapper stubs" in capsys.readouterr().out


def test_forward_macro_under_a_neighbours_name_is_reported(capsys):
    source = wrapper_source(
        stub("miopenCreate", "    MIOPEN_WRAPPER_DISPATCH(miopenCreate);"),
        stub(
            "miopenDestroy",
            "    MIOPEN_WRAPPER_FORWARD(miopenCreate,\n"
            "                           ::miopen::wrapper::hipdnn::Destroy(handle));",
        ),
    )
    assert not abi.check_wrapper_dispatch(dispatches_of(source))
    assert "miopenDestroy dispatches as miopenCreate" in capsys.readouterr().out


def test_stub_carrying_both_macros_is_reported(capsys):
    source = wrapper_source(
        stub(
            "miopenCreate",
            "    MIOPEN_WRAPPER_DISPATCH(miopenCreate);\n"
            "    MIOPEN_WRAPPER_FORWARD(miopenCreate,\n"
            "                           ::miopen::wrapper::hipdnn::Create(handle));",
        )
    )
    assert not abi.check_wrapper_dispatch(dispatches_of(source))
    assert "miopenCreate has 2 dispatch macros" in capsys.readouterr().out


def test_dispatch_macro_given_a_forward_call_is_reported(capsys):
    source = wrapper_source(
        stub(
            "miopenCreate",
            "    MIOPEN_WRAPPER_DISPATCH(miopenCreate,\n"
            "                            ::miopen::wrapper::hipdnn::Create(handle));",
        )
    )
    assert not abi.check_wrapper_dispatch(dispatches_of(source))
    out = capsys.readouterr().out
    assert (
        "miopenCreate has no MIOPEN_WRAPPER_DISPATCH or MIOPEN_WRAPPER_FORWARD" in out
    )


def test_forward_macro_without_a_call_is_reported(capsys):
    source = wrapper_source(
        stub("miopenCreate", "    MIOPEN_WRAPPER_FORWARD(miopenCreate);")
    )
    assert not abi.check_wrapper_dispatch(dispatches_of(source))
    out = capsys.readouterr().out
    assert (
        "miopenCreate has no MIOPEN_WRAPPER_DISPATCH or MIOPEN_WRAPPER_FORWARD" in out
    )


def test_exempt_stub_growing_the_macro_is_reported(capsys):
    source = """
extern "C" const char* miopenGetErrorString(miopenStatus_t error)
{
    MIOPEN_WRAPPER_DISPATCH(miopenGetErrorString);
    return miopenGetErrorString_impl(error);
}
"""
    assert not abi.check_wrapper_dispatch(dispatches_of(source))
    out = capsys.readouterr().out
    assert "miopenGetErrorString carries a dispatch macro but is exempt" in out
    assert "returns const char*" in out


def test_exemption_for_a_stub_that_no_longer_exists_is_reported(capsys):
    source = stub("miopenCreate", "    MIOPEN_WRAPPER_DISPATCH(miopenCreate);")
    assert not abi.check_wrapper_dispatch(dispatches_of(source))
    assert "miopenGetErrorString is exempt" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Reading a file a sparse checkout may not have materialized
# --------------------------------------------------------------------------


def test_read_tracked_source_prefers_the_working_tree(tmp_path):
    path = tmp_path / "present.hpp"
    path.write_text("#pragma once\n", encoding="utf-8")
    assert abi.read_tracked_source(path, tmp_path) == "#pragma once\n"


def test_read_tracked_source_records_why_it_skipped(tmp_path):
    """A skip must name its cause: a silent one is indistinguishable from a pass."""
    reasons: dict[str, str] = {}
    missing = tmp_path / "absent.hpp"
    assert abi.read_tracked_source(missing, tmp_path, reasons) is None
    assert reasons, "a skipped file must record a reason"
    assert any(text for text in reasons.values())


def test_read_tracked_source_rejects_paths_outside_the_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "elsewhere.hpp"
    reasons: dict[str, str] = {}
    assert abi.read_tracked_source(outside, root, reasons) is None
    assert "outside the repository root" in "".join(reasons.values())


# run_git() is the script's single shell-out, so these cases replace it instead of
# standing up a real repository. `git init` in a temporary directory is unsafe
# here: git exports GIT_DIR into the hooks it runs and GIT_DIR outranks -C, so
# under the pre-commit hook these tests run from, the init reached the
# surrounding repository and left it marked bare, which refuses every later
# commit.


def _completed(returncode=0, stdout=b"", stderr=b""):
    return subprocess.CompletedProcess(["git"], returncode, stdout, stderr)


def _unreadable_blob_git(tracked: bool, cat_file_stderr: bytes = b"fatal: bad object"):
    """Stand in for the shape a failed promisor fetch leaves behind.

    The commit's tree still names the file, so ls-tree answers, while the blob it
    points at is gone, so cat-file cannot produce the content.
    """

    def run_git(repo_root, *args):
        if args[0] == "cat-file":
            return _completed(1, stderr=cat_file_stderr)
        if args[0] == "ls-tree":
            return _completed(0, stdout=b"tracked.hpp\n" if tracked else b"")
        raise AssertionError(f"unexpected git call: {args}")

    return run_git


def test_tracked_but_unreadable_file_is_a_hard_failure(tmp_path, monkeypatch):
    """A checkout that carries the file but cannot produce it must not go green.

    Skipping here would drop a file that is under test out of the comparison
    while still reporting success, which is the one outcome this gate cannot
    afford: the drift it catches links cleanly and only misbehaves at run time.
    """
    monkeypatch.setattr(abi, "run_git", _unreadable_blob_git(tracked=True))
    with pytest.raises(abi.AbiError) as caught:
        abi.read_tracked_source(tmp_path / "tracked.hpp", tmp_path, {})
    assert "tracked.hpp" in str(caught.value)


def test_the_hard_failure_carries_git_stderr(tmp_path, monkeypatch):
    """Whatever git said is the only evidence of why the checkout is broken."""
    monkeypatch.setattr(
        abi,
        "run_git",
        _unreadable_blob_git(tracked=True, cat_file_stderr=b"fatal: missing blob abc"),
    )
    with pytest.raises(abi.AbiError) as caught:
        abi.read_tracked_source(tmp_path / "tracked.hpp", tmp_path, {})
    message = str(caught.value)
    assert "git cat-file said:" in message
    assert "fatal: missing blob abc" in message
    assert "(git printed nothing)" not in message


def test_a_file_absent_from_the_commit_is_still_only_a_skip(tmp_path, monkeypatch):
    """The escalation must stay narrow.

    A sparse checkout legitimately lacks whole subtrees the commit never
    carried, and failing on those would break every partial-checkout lane
    instead of catching drift.
    """
    monkeypatch.setattr(abi, "run_git", _unreadable_blob_git(tracked=False))
    reasons: dict[str, str] = {}
    assert (
        abi.read_tracked_source(tmp_path / "never_added.hpp", tmp_path, reasons) is None
    )
    assert "not tracked at HEAD" in "".join(reasons.values())


def test_a_non_checkout_is_a_skip_not_a_failure(tmp_path, monkeypatch):
    """Source tarballs have no git at all; that is not a broken checkout."""
    monkeypatch.setattr(
        abi,
        "run_git",
        lambda root, *args: _completed(
            128,
            stderr=b"fatal: not a git repository (or any of the parent directories)",
        ),
    )
    reasons: dict[str, str] = {}
    assert abi.read_tracked_source(tmp_path / "absent.hpp", tmp_path, reasons) is None
    assert "not a git checkout" in "".join(reasons.values())


def test_run_git_strips_an_ambient_git_environment(tmp_path, monkeypatch):
    """Stands in for the pre-commit hook this runs as, which exports GIT_DIR.

    Left in place, GIT_DIR outranks -C and every read answers from the repository
    that invoked the hook, so a file absent from the tree under test is reported on
    some other tree's contents -- and the answer looks ordinary either way.
    """
    monkeypatch.setenv("GIT_DIR", "/nowhere/.git")
    monkeypatch.setenv("GIT_WORK_TREE", "/nowhere")
    monkeypatch.setenv("GIT_INDEX_FILE", "/nowhere/index")

    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs["env"]
        return _completed()

    monkeypatch.setattr(abi.subprocess, "run", fake_run)
    abi.run_git(tmp_path, "cat-file", "-p", "HEAD:x")

    assert [key for key in seen["env"] if key.startswith("GIT_")] == []
    assert seen["cmd"][:3] == ["git", "-C", str(tmp_path)]


# --------------------------------------------------------------------------
# Installed headers
#
# The staged include tree is what a consumer compiles against. A private
# spelling that reaches it names a symbol only libMIOpen_private.so carries.
# --------------------------------------------------------------------------


def staged(tmp_path, files: dict[str, str]) -> Path:
    """Build an include tree from {relative path: text} and return its root."""
    root = tmp_path / "include"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_a_clean_include_tree_passes(tmp_path):
    root = staged(tmp_path, {"miopen/miopen.h": "miopenStatus_t miopenFoo(int);\n"})
    assert abi.check_installed_headers(str(root), []) is True


def test_an_include_tree_with_no_headers_is_an_error(tmp_path):
    """A tree that stages nothing this recognises never got checked, so passing it
    would report a clean include tree having read no files at all."""
    root = staged(tmp_path, {"miopen/miopen-config.cmake": "# miopenFoo_impl\n"})
    with pytest.raises(abi.AbiError, match="nothing was checked"):
        abi.check_installed_headers(str(root), [])


def test_an_include_tree_with_no_miopen_headers_is_an_error(tmp_path):
    """Headers from other packages sharing the prefix do not count: passing on them
    would report MIOpen's headers clean having read none of them."""
    root = staged(tmp_path, {"hip/hip_runtime.h": "void hipFoo(void);\n"})
    with pytest.raises(abi.AbiError, match="nothing was checked"):
        abi.check_installed_headers(str(root), [])


def test_other_packages_headers_are_still_scanned_for_leaks(tmp_path, capsys):
    root = staged(
        tmp_path,
        {
            "miopen/miopen.h": "miopenStatus_t miopenFoo(int);\n",
            "other/leak.h": "miopenStatus_t miopenFoo_impl(int);\n",
        },
    )
    assert abi.check_installed_headers(str(root), []) is False
    assert "other/leak.h:1: miopenFoo_impl" in capsys.readouterr().out.replace(
        "\\", "/"
    )


def test_an_include_tree_exempt_all_the_way_down_is_an_error(tmp_path):
    """Exempting every staged header leaves the same nothing-was-read result."""
    root = staged(tmp_path, {"miopen/private/rename.hpp": "void miopenFoo_impl();\n"})
    with pytest.raises(abi.AbiError, match="nothing was checked"):
        abi.check_installed_headers(str(root), ["miopen/private"])


def test_a_leaked_rename_is_reported_with_file_and_line(tmp_path, capsys):
    root = staged(
        tmp_path,
        {"miopen/miopen.h": "// header\nmiopenStatus_t miopenFoo_impl(int);\n"},
    )
    assert abi.check_installed_headers(str(root), []) is False
    out = capsys.readouterr().out.replace("\\", "/")
    assert "miopen/miopen.h:2: miopenFoo_impl" in out


def test_the_private_directory_is_exempt(tmp_path):
    """The private declarations spell the private names by definition.

    A public header is staged alongside so the exemption is what carries the pass;
    without one the tree scans nothing, which is its own failure.
    """
    root = staged(
        tmp_path,
        {
            "miopen/miopen.h": "miopenStatus_t miopenFoo(int);\n",
            "miopen/private/miopen_impl.h": "miopenStatus_t miopenFoo_impl(int);\n",
        },
    )
    assert abi.check_installed_headers(str(root), ["miopen/private"]) is True


def test_the_exemption_is_by_path_not_by_file_name(tmp_path, capsys):
    """A file named like the private header, staged elsewhere, still trips.

    Matching on the name alone would let any header called miopen_impl.h
    smuggle the rename into the public tree.
    """
    root = staged(
        tmp_path,
        {"miopen/miopen_impl.h": "miopenStatus_t miopenFoo_impl(int);\n"},
    )
    assert abi.check_installed_headers(str(root), ["miopen/private"]) is False
    assert "miopenFoo_impl" in capsys.readouterr().out


def test_a_name_merely_containing_impl_is_not_a_leak(tmp_path):
    """ck_impl_interface and friends are ordinary identifiers.

    The check matches whole identifiers against the same pattern the symbol
    checks use, so a substring search's false positives do not appear here.
    """
    root = staged(
        tmp_path,
        {
            "miopen/miopen.h": (
                "void ck_impl_interface(void);\n"
                "int miopen_impl_helper;\n"
                "void miopenFoo_implementation(void);\n"
            )
        },
    )
    assert abi.check_installed_headers(str(root), []) is True


def test_only_headers_are_scanned(tmp_path, capsys):
    """A CMake package file staged alongside the headers is not a header.

    Nothing compiles against it, and counting it would inflate the number the
    check reports as scanned.
    """
    root = staged(
        tmp_path,
        {
            "miopen/miopen.h": "miopenStatus_t miopenFoo(int);\n",
            "miopen/miopen-config.cmake": "# miopenFoo_impl\n",
        },
    )
    assert abi.check_installed_headers(str(root), []) is True
    assert "1 installed headers" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The binary stand-in
#
# The co-version and superset checks ask a shared object for three things:
# what it exports, what SONAME it declares, and what it is linked against. A
# Windows DLL is also asked for its file version, since its name carries none.
# --------------------------------------------------------------------------


class FakeLibrary:
    """A shared object as the binary checks see it."""

    def __init__(
        self,
        path,
        symbols=(),
        soname="",
        needed=(),
        naming=abi.ELF_NAMING,
        file_version="",
    ):
        self.path = Path(path)
        self._symbols = set(symbols)
        self._soname = soname
        self._needed = tuple(needed)
        self.naming = naming
        self._file_version = file_version

    def defined_dynamic_functions(self):
        return set(self._symbols)

    def soname(self):
        return self._soname

    def needed(self):
        return sorted(self._needed)

    def file_version(self):
        return self._file_version


def wrapper_elf(
    symbols=(), soname="libMIOpen.so.1", needed=("libMIOpen_private.so.1",)
):
    return FakeLibrary("libMIOpen.so", symbols, soname, needed)


def private_elf(symbols=(), soname="libMIOpen_private.so.1"):
    return FakeLibrary("libMIOpen_private.so", symbols, soname, ())


# --------------------------------------------------------------------------
# Co-versioning
#
# The two libraries are halves of one build. When they are not, the usual
# cause is a stale build directory holding one half of an earlier one.
# --------------------------------------------------------------------------


def test_matched_versions_pass():
    assert abi.check_coversioned(wrapper_elf(), private_elf()) is True


def test_a_soname_major_mismatch_fails():
    private = private_elf(soname="libMIOpen_private.so.2")
    assert abi.check_coversioned(wrapper_elf(), private) is False


def test_a_needed_version_mismatch_fails():
    """The declared SONAMEs can agree while the link edge points elsewhere.

    This is the case a SONAME-only comparison misses, and the one that
    actually decides what the loader binds.
    """
    wrapper = wrapper_elf(needed=("libMIOpen_private.so.0",))
    assert abi.check_coversioned(wrapper, private_elf()) is False


def test_the_failure_names_all_three_observed_strings(capsys):
    """Which of the three is the odd one out is the whole diagnosis."""
    wrapper = wrapper_elf(soname="libMIOpen.so.1", needed=("libMIOpen_private.so.0",))
    abi.check_coversioned(wrapper, private_elf(soname="libMIOpen_private.so.2"))
    out = capsys.readouterr().out
    assert "libMIOpen.so.1" in out
    assert "libMIOpen_private.so.2" in out
    assert "libMIOpen_private.so.0" in out


def test_an_unversioned_wrapper_soname_fails():
    """A wrapper with no soversion at all cannot be shown to match anything."""
    wrapper = wrapper_elf(soname="libMIOpen.so", needed=("libMIOpen_private.so",))
    assert (
        abi.check_coversioned(wrapper, private_elf(soname="libMIOpen_private.so"))
        is False
    )


# --------------------------------------------------------------------------
# Private -> wrapper superset
#
# The baseline check compares the wrapper against a recorded list, so an entry
# point the wrapper never re-exported is absent from both sides and passes.
# Only the private library knows the full set of renamed entry points.
# --------------------------------------------------------------------------


def test_every_renamed_entry_point_having_a_stub_passes():
    wrapper = wrapper_elf(symbols=("miopenFoo", "miopenBar"))
    private = private_elf(symbols=("miopenFoo_impl", "miopenBar_impl"))
    assert abi.check_impl_superset(wrapper, private) is True


def test_a_renamed_entry_point_with_no_stub_fails():
    wrapper = wrapper_elf(symbols=("miopenFoo",))
    private = private_elf(symbols=("miopenFoo_impl", "miopenBar_impl"))
    assert abi.check_impl_superset(wrapper, private) is False


def test_the_missing_stub_is_reported_under_its_public_name(capsys):
    """That is the name the stub would carry, and the one to grep for."""
    wrapper = wrapper_elf(symbols=("miopenFoo",))
    private = private_elf(symbols=("miopenFoo_impl", "miopenBar_impl"))
    abi.check_impl_superset(wrapper, private)
    out = capsys.readouterr().out
    assert "  miopenBar\n" in out
    assert "miopenBar_impl" not in out


def test_private_symbols_that_were_never_renamed_are_not_required():
    """Only the renamed set is forwarded; the rest stay private by design."""
    wrapper = wrapper_elf(symbols=("miopenFoo",))
    private = private_elf(symbols=("miopenFoo_impl", "miopenInternalThing"))
    assert abi.check_impl_superset(wrapper, private) is True


def test_a_private_library_with_no_renamed_symbols_is_an_error():
    """Zero renamed entry points means the wrong file -- --private-lib aimed at the
    wrapper, or a build where the rename was never applied -- not a satisfied
    superset."""
    wrapper = wrapper_elf(symbols=("miopenFoo",))
    private = private_elf(symbols=("miopenInternalThing",))
    with pytest.raises(abi.AbiError, match="not a flag-on private library"):
        abi.check_impl_superset(wrapper, private)


# --------------------------------------------------------------------------
# Exit codes
#
# Each check above returning False is only half the contract; the other half is
# that the subcommand folds it into its exit status. Dropping one `ok &=` would
# leave every test above passing while the gate ignored that check entirely.
# 0 is pass, 1 is a check that failed, 2 is a check that could not run.
# --------------------------------------------------------------------------


def run_check_wrapper(
    tmp_path, monkeypatch, wrapper, private, extra=(), wrapper_only=()
):
    """Run `check-wrapper` through main() against stand-ins for the two libraries."""
    baseline = tmp_path / "public_symbols.baseline"
    baseline.write_text("miopenFoo\nmiopenBar\nmiopenExcluded\n")
    excluded = tmp_path / "excluded_symbols.txt"
    excluded.write_text("miopenExcluded\n")
    listed = tmp_path / "wrapper_only_symbols.txt"
    listed.write_text("".join(sym + "\n" for sym in wrapper_only))
    libs = {"libMIOpen.so": wrapper, "libMIOpen_private.so": private}
    monkeypatch.setattr(abi, "open_library", lambda path, what: libs[path])
    return abi.main(
        [
            "check-wrapper",
            "libMIOpen.so",
            "--baseline",
            str(baseline),
            "--excluded",
            str(excluded),
            "--wrapper-only",
            str(listed),
            "--private-lib",
            "libMIOpen_private.so",
            *extra,
        ]
    )


def good_wrapper(**overrides):
    return wrapper_elf(**{"symbols": ("miopenFoo", "miopenBar"), **overrides})


def good_private(**overrides):
    symbols = ("miopenFoo_impl", "miopenBar_impl", "miopenExcluded")
    return private_elf(**{"symbols": symbols, **overrides})


def test_check_wrapper_passes_a_consistent_pair(tmp_path, monkeypatch, capsys):
    assert run_check_wrapper(tmp_path, monkeypatch, good_wrapper(), good_private()) == 0
    assert "wrapper public-abi symbol check: PASS" in capsys.readouterr().out


def test_check_wrapper_fails_on_a_coversion_mismatch(tmp_path, monkeypatch, capsys):
    private = good_private(soname="libMIOpen_private.so.2")
    assert run_check_wrapper(tmp_path, monkeypatch, good_wrapper(), private) == 1
    assert "wrapper public-abi symbol check: FAIL" in capsys.readouterr().out


def test_check_wrapper_fails_on_a_missing_stub(tmp_path, monkeypatch, capsys):
    private = good_private(
        symbols=("miopenFoo_impl", "miopenBar_impl", "miopenBaz_impl", "miopenExcluded")
    )
    assert run_check_wrapper(tmp_path, monkeypatch, good_wrapper(), private) == 1
    assert "wrapper public-abi symbol check: FAIL" in capsys.readouterr().out


def run_with_needed_baseline(tmp_path, monkeypatch, listed):
    needed = tmp_path / "wrapper_needed.baseline"
    needed.write_text("".join(entry + "\n" for entry in listed))
    return run_check_wrapper(
        tmp_path,
        monkeypatch,
        good_wrapper(needed=("libMIOpen_private.so.1", "libc.so.6")),
        good_private(),
        extra=("--needed-baseline", str(needed)),
    )


def test_check_wrapper_passes_a_matching_needed_baseline(tmp_path, monkeypatch, capsys):
    listed = ("libMIOpen_private.so.1", "libc.so.6")
    assert run_with_needed_baseline(tmp_path, monkeypatch, listed) == 0
    assert "DT_NEEDED list matches baseline (2 entries)" in capsys.readouterr().out


def test_check_wrapper_fails_a_differing_needed_baseline(tmp_path, monkeypatch, capsys):
    listed = ("libMIOpen_private.so.1", "libhipdnn_backend.so")
    assert run_with_needed_baseline(tmp_path, monkeypatch, listed) == 1
    out = capsys.readouterr().out
    assert "DT_NEEDED list differs from baseline" in out
    assert "missing from build: libhipdnn_backend.so" in out
    assert "unexpected in build: libc.so.6" in out
    assert "wrapper public-abi symbol check: FAIL" in out


def test_check_wrapper_fails_without_the_private_lib_dependency(
    tmp_path, monkeypatch, capsys
):
    wrapper = good_wrapper(needed=("libc.so.6",))
    assert run_check_wrapper(tmp_path, monkeypatch, wrapper, good_private()) == 1
    out = capsys.readouterr().out
    assert "flag-on wrapper has no DT_NEEDED on libMIOpen_private" in out
    assert "wrapper public-abi symbol check: FAIL" in out


def test_check_wrapper_fails_on_a_hipdnn_backend_dependency(
    tmp_path, monkeypatch, capsys
):
    wrapper = good_wrapper(needed=("libMIOpen_private.so.1", "libhipdnn_backend.so"))
    assert run_check_wrapper(tmp_path, monkeypatch, wrapper, good_private()) == 1
    out = capsys.readouterr().out
    assert "wrapper has DT_NEEDED on libhipdnn_backend.so" in out
    assert "wrapper public-abi symbol check: FAIL" in out


def test_check_wrapper_that_cannot_run_exits_2(tmp_path, monkeypatch, capsys):
    # A private library with no renamed symbols makes the superset check raise.
    private = good_private(symbols=("miopenExcluded",))
    assert run_check_wrapper(tmp_path, monkeypatch, good_wrapper(), private) == 2
    assert "not a flag-on private library" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Windows DLLs
#
# The same gates run on a Windows build's DLLs, with Windows names, case-blind
# name matching, and the version resource in place of the soversion.
# --------------------------------------------------------------------------


def wrapper_dll(
    symbols=("miopenFoo", "miopenBar"),
    soname="MIOpen.dll",
    needed=("MIOpen_private.dll", "KERNEL32.dll"),
    file_version="3.5.1.0",
):
    return FakeLibrary(
        "MIOpen.dll", symbols, soname, needed, abi.PE_NAMING, file_version
    )


def private_dll(
    symbols=("miopenFoo_impl", "miopenBar_impl", "miopenExcluded"),
    soname="MIOpen_private.dll",
    file_version="3.5.1.0",
):
    return FakeLibrary(
        "MIOpen_private.dll", symbols, soname, (), abi.PE_NAMING, file_version
    )


def test_check_wrapper_passes_a_consistent_dll_pair(tmp_path, monkeypatch, capsys):
    result = run_check_wrapper(tmp_path, monkeypatch, wrapper_dll(), private_dll())
    out = capsys.readouterr().out
    assert result == 0, out
    assert "PASS: DLL name is MIOpen.dll" in out
    assert "co-versioned at file version 3.5.1.0" in out


def test_dll_names_are_matched_without_case(capsys):
    """Windows file names, and so the loader's view of DLL names, ignore case."""
    wrapper = wrapper_dll(soname="miopen.DLL", needed=("MIOPEN_PRIVATE.dll",))
    assert abi.check_soname(wrapper) is True
    assert abi.check_private_dep(wrapper, expect_present=True) is True
    assert abi.check_coversioned(wrapper, private_dll()) is True


def test_a_linux_soname_does_not_pass_for_a_dll(capsys):
    assert abi.check_soname(wrapper_dll(soname="libMIOpen.so.1")) is False
    assert "expected 'MIOpen.dll'" in capsys.readouterr().out


def test_a_linux_soname_is_matched_with_case(capsys):
    assert abi.check_soname(wrapper_elf(soname="libmiopen.so.1")) is False


def test_a_dll_without_the_private_import_fails(capsys):
    wrapper = wrapper_dll(needed=("KERNEL32.dll",))
    assert abi.check_private_dep(wrapper, expect_present=True) is False
    assert "flag-on wrapper has no import of MIOpen_private" in capsys.readouterr().out


def test_a_dll_importing_the_hipdnn_backend_fails(capsys):
    wrapper = wrapper_dll(needed=("MIOpen_private.dll", "HIPDNN_BACKEND.dll"))
    assert abi.check_no_backend_dep(wrapper) is False
    assert "wrapper has import of HIPDNN_BACKEND.dll" in capsys.readouterr().out


def test_dlls_from_two_releases_are_not_coversioned(capsys):
    """A stale half from another release still loads on Windows: the import
    names no version, so only the version resource tells them apart."""
    private = private_dll(file_version="3.4.0.0")
    assert abi.check_coversioned(wrapper_dll(), private) is False
    out = capsys.readouterr().out
    assert "3.5.1.0" in out
    assert "3.4.0.0" in out


def test_a_dll_without_a_version_resource_is_not_coversioned():
    """Two missing versions agree, and prove nothing."""
    wrapper = wrapper_dll(file_version="")
    private = private_dll(file_version="")
    assert abi.check_coversioned(wrapper, private) is False


def test_a_wrapper_importing_another_private_dll_is_not_coversioned(capsys):
    wrapper = wrapper_dll(needed=("MIOpen_private_old.dll",))
    assert abi.check_coversioned(wrapper, private_dll()) is False
    assert "MIOpen_private_old.dll" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The PE reader
#
# Images are built here rather than linked: the module runs in a lint lane with
# no Windows toolchain. Each holds an executable .text section, which code
# exports point into, and a read-only .rdata section with everything else.
# --------------------------------------------------------------------------

TEXT_RVA, RDATA_RVA = 0x1000, 0x2000
TEXT_RAW, RDATA_RAW = 0x400, 0x600


def pe_image(
    dll_name,
    exports=(),
    imports=(),
    delay_imports=(),
    version=None,
    pe32_plus=True,
    ordinal_base=1,
    reverse_functions=False,
    old_delay_layout=False,
    decoy_resources=(),
    named_resource_entry=False,
    version_signature=0xFEEF04BD,
    text_vsize=None,
):
    """Return a DLL image.

    `exports` maps a name to "code", "data" or "forward", or to "padding_code" or
    "bss_code" for code past `text_vsize` or past the raw data. `reverse_functions`
    lays the function table out in .def order, as a .def-file link does, with
    `ordinal_base` as the first ordinal. `old_delay_layout` (PE32 only) writes
    delay-import names as addresses. `decoy_resources` adds resource types around
    RT_VERSION.
    """
    rdata = bytearray()
    image_base = 0x180000000 if pe32_plus else 0x10000000

    def put(data):
        rva = RDATA_RVA + len(rdata)
        rdata.extend(data)
        rdata.extend(b"\0" * (-len(rdata) % 4))
        return rva

    def cstr(text):
        return put(text.encode() + b"\0")

    directories = {}
    data_target = put(b"\0" * 4)

    if exports:
        names = sorted(exports)
        start = put(b"\0" * 40)
        functions = put(b"\0" * 4 * len(names))
        name_table = put(b"\0" * 4 * len(names))
        ordinals = put(b"\0" * 2 * len(names))
        own_name = cstr(dll_name)
        code_targets = {
            "code": lambda i: TEXT_RVA + 16 * i,
            "padding_code": lambda i: TEXT_RVA + 0x180,
            "bss_code": lambda i: TEXT_RVA + 0x300,
        }
        for i, name in enumerate(names):
            if exports[name] in code_targets:
                target = code_targets[exports[name]](i)
            elif exports[name] == "data":
                target = data_target
            else:
                target = cstr("KERNEL32.GetTickCount")
            slot = len(names) - 1 - i if reverse_functions else i
            struct.pack_into("<I", rdata, functions - RDATA_RVA + 4 * slot, target)
            struct.pack_into("<I", rdata, name_table - RDATA_RVA + 4 * i, cstr(name))
            struct.pack_into("<H", rdata, ordinals - RDATA_RVA + 2 * i, slot)
        end = RDATA_RVA + len(rdata)
        struct.pack_into(
            "<IIHHIIIIIII",
            rdata,
            start - RDATA_RVA,
            0,
            0,
            0,
            0,
            own_name,
            ordinal_base,
            len(names),
            len(names),
            functions,
            name_table,
            ordinals,
        )
        directories[0] = (start, end - start)

    # Each imported DLL gets an empty, terminated table of imported functions.
    if imports:
        descriptors = []
        for name in imports:
            thunks = put(b"\0" * 8)
            descriptors.append(struct.pack("<IIIII", thunks, 0, 0, cstr(name), thunks))
        table = put(b"".join(descriptors) + b"\0" * 20)
        directories[1] = (table, 20 * (len(imports) + 1))

    if delay_imports:
        assert not (old_delay_layout and pe32_plus)
        descriptors = []
        for name in delay_imports:
            thunks = put(b"\0" * 8)
            if old_delay_layout:
                attributes, name_field = 0, cstr(name) + image_base
            else:
                attributes, name_field = 1, cstr(name)
            descriptors.append(
                struct.pack(
                    "<IIIIIIII", attributes, name_field, 0, thunks, thunks, 0, 0, 0
                )
            )
        table = put(b"".join(descriptors) + b"\0" * 32)
        directories[13] = (table, 32 * (len(delay_imports) + 1))

    if version:
        high = version[0] << 16 | version[1]
        low = version[2] << 16 | version[3]
        fixed = struct.pack(
            "<13I", version_signature, 0x10000, high, low, high, low, *[0] * 7
        )
        key = "VS_VERSION_INFO\0".encode("utf-16-le")
        info = struct.pack("<HHH", 40 + len(fixed), len(fixed), 0) + key + b"\0\0"
        # Decoy data is all zeros, so reading it as a version finds none.
        subtrees = sorted(
            [(16, put(info + fixed), len(info + fixed))]
            + [(kind, put(b"\0" * 64), 64) for kind in decoy_resources]
        )

        def directory(entries, named=0):
            header = struct.pack("<IIHHHH", 0, 0, 0, 0, named, len(entries) - named)
            return header + b"".join(struct.pack("<II", *entry) for entry in entries)

        # Each type gets its own name directory, language directory and data entry
        # (64 bytes) after the root.
        root_size = 16 + 8 * (len(subtrees) + named_resource_entry)

        def subtree_at(k):
            return root_size + 64 * k

        root_entries = [
            (kind, 0x80000000 | subtree_at(k))
            for k, (kind, _, _) in enumerate(subtrees)
        ]
        if named_resource_entry:
            # Named entries come before the ID entries.
            root_entries.insert(0, (0x80000000, 0x80000000 | subtree_at(0)))
        tree = directory(root_entries, named=int(named_resource_entry))
        for k, (_, data_rva, data_size) in enumerate(subtrees):
            tree += directory([(1, 0x80000000 | (subtree_at(k) + 24))])
            tree += directory([(0x409, subtree_at(k) + 48)])
            tree += struct.pack("<IIII", data_rva, data_size, 0, 0)
        directories[2] = (put(tree), len(tree))

    rdata.extend(b"\0" * (-len(rdata) % 0x200))
    text = b"\xc3" * 0x200

    dir_offset = 112 if pe32_plus else 96
    optional = bytearray(dir_offset + 16 * 8)
    struct.pack_into("<H", optional, 0, 0x20B if pe32_plus else 0x10B)
    if pe32_plus:
        struct.pack_into("<Q", optional, 24, image_base)
    else:
        struct.pack_into("<I", optional, 28, image_base)
    struct.pack_into("<I", optional, dir_offset - 4, 16)
    for index, (rva, size) in directories.items():
        struct.pack_into("<II", optional, dir_offset + 8 * index, rva, size)

    def section(name, rva, raw_offset, raw_size, flags, virtual_size=None):
        return struct.pack(
            "<8sIIIIIIHHI",
            name,
            virtual_size or raw_size,
            rva,
            raw_size,
            raw_offset,
            0,
            0,
            0,
            0,
            flags,
        )

    machine = 0x8664 if pe32_plus else 0x14C
    headers = (
        b"MZ"
        + b"\0" * 0x3A
        + struct.pack("<I", 0x40)
        + b"PE\0\0"
        + struct.pack("<HHIIIHH", machine, 2, 0, 0, 0, len(optional), 0x2022)
        + bytes(optional)
        + section(b".text", TEXT_RVA, TEXT_RAW, len(text), 0x60000020, text_vsize)
        + section(b".rdata", RDATA_RVA, RDATA_RAW, len(rdata), 0x40000040)
    )
    return headers.ljust(TEXT_RAW, b"\0") + text + bytes(rdata)


def read_dll(tmp_path, image, name="MIOpen.dll"):
    path = tmp_path / name
    path.write_bytes(image)
    return abi.open_library(str(path), "library")


@pytest.mark.parametrize("pe32_plus", [True, False], ids=["PE32+", "PE32"])
def test_pe_reader_reads_what_the_checks_ask(tmp_path, pe32_plus):
    image = pe_image(
        "MIOpen.dll",
        exports={"miopenFoo": "code", "miopenBar": "code"},
        imports=["MIOpen_private.dll", "KERNEL32.dll"],
        version=(3, 5, 1, 0),
        pe32_plus=pe32_plus,
    )
    lib = read_dll(tmp_path, image)
    assert isinstance(lib, abi.Pe)
    assert lib.naming is abi.PE_NAMING
    assert lib.defined_dynamic_functions() == {"miopenFoo", "miopenBar"}
    assert lib.soname() == "MIOpen.dll"
    assert lib.needed() == ["KERNEL32.dll", "MIOpen_private.dll"]
    assert lib.file_version() == "3.5.1.0"


def test_pe_data_exports_are_not_functions(tmp_path):
    """As the ELF reader keeps only function symbols."""
    image = pe_image("MIOpen.dll", exports={"miopenFoo": "code", "miopenData": "data"})
    assert read_dll(tmp_path, image).defined_dynamic_functions() == {"miopenFoo"}


def test_pe_forwarded_exports_are_functions(tmp_path):
    """A forwarder points into the export table, not into code, and is still a
    function a caller can bind to."""
    image = pe_image("MIOpen.dll", exports={"miopenFoo": "forward"})
    assert read_dll(tmp_path, image).defined_dynamic_functions() == {"miopenFoo"}


def test_pe_delay_loaded_dlls_are_needed(tmp_path):
    """The link binds to a delay-loaded DLL by name, which the backend check
    forbids, even though nothing opens it until first use."""
    image = pe_image(
        "MIOpen.dll",
        exports={"miopenFoo": "code"},
        imports=["MIOpen_private.dll"],
        delay_imports=["hipdnn_backend.dll"],
    )
    assert read_dll(tmp_path, image).needed() == [
        "MIOpen_private.dll",
        "hipdnn_backend.dll",
    ]


def test_pe_without_tables_has_no_name_imports_or_version(tmp_path):
    lib = read_dll(tmp_path, pe_image("MIOpen.dll"))
    assert lib.defined_dynamic_functions() == set()
    assert lib.soname() == ""
    assert lib.needed() == []
    assert lib.file_version() == ""


def test_a_truncated_pe_is_an_error(tmp_path):
    image = pe_image("MIOpen.dll", exports={"miopenFoo": "code"})
    with pytest.raises(abi.AbiError, match="MIOpen.dll"):
        read_dll(tmp_path, image[:0x200]).defined_dynamic_functions()


@pytest.mark.parametrize("pe32_plus", [True, False], ids=["PE32+", "PE32"])
def test_pe_reader_reads_the_image_base(tmp_path, pe32_plus):
    lib = read_dll(tmp_path, pe_image("MIOpen.dll", pe32_plus=pe32_plus))
    assert lib.image_base == (0x180000000 if pe32_plus else 0x10000000)


def test_pe_exports_are_matched_through_the_ordinal_table(tmp_path):
    """A .def-file link lays the function table out in .def order, so a name's
    function is the one its ordinal picks, not the one at its own position."""
    image = pe_image(
        "MIOpen.dll",
        exports={"miopenA": "code", "miopenB": "data"},
        reverse_functions=True,
        ordinal_base=5,
    )
    assert read_dll(tmp_path, image).defined_dynamic_functions() == {"miopenA"}


@pytest.mark.parametrize("pe32_plus", [True, False], ids=["PE32+", "PE32"])
def test_pe_reader_walks_every_delay_import(tmp_path, pe32_plus):
    """The backend is the second descriptor here: reading past the first one needs
    the descriptor's full size."""
    image = pe_image(
        "MIOpen.dll",
        delay_imports=["SHLWAPI.dll", "hipdnn_backend.dll"],
        pe32_plus=pe32_plus,
    )
    assert read_dll(tmp_path, image).needed() == ["SHLWAPI.dll", "hipdnn_backend.dll"]


def test_pe_delay_import_names_in_the_old_layout_are_addresses(tmp_path):
    """Only a PE32 image can hold the old layout, whose names are image-base
    relative addresses rather than RVAs."""
    image = pe_image(
        "MIOpen.dll",
        delay_imports=["hipdnn_backend.dll"],
        pe32_plus=False,
        old_delay_layout=True,
    )
    assert read_dll(tmp_path, image).needed() == ["hipdnn_backend.dll"]


def test_pe_version_is_found_among_other_resources(tmp_path):
    """RT_VERSION sits between types that sort before and after it, behind a named
    entry."""
    image = pe_image(
        "MIOpen.dll",
        version=(3, 5, 1, 0),
        decoy_resources=(10, 24),
        named_resource_entry=True,
    )
    assert read_dll(tmp_path, image).file_version() == "3.5.1.0"


def test_pe_version_without_the_fixed_file_info_signature_is_ignored(tmp_path):
    image = pe_image("MIOpen.dll", version=(3, 5, 1, 0), version_signature=0)
    assert read_dll(tmp_path, image).file_version() == ""


@pytest.mark.parametrize(
    "kind, text_vsize",
    [("padding_code", 0x100), ("bss_code", 0x400)],
    ids=["past the virtual size", "past the raw data"],
)
def test_pe_code_counts_when_only_one_size_of_its_section_reaches_it(
    tmp_path, kind, text_vsize
):
    """A section is as large as the bigger of its virtual size and raw data."""
    image = pe_image("MIOpen.dll", exports={"miopenFoo": kind}, text_vsize=text_vsize)
    assert read_dll(tmp_path, image).defined_dynamic_functions() == {"miopenFoo"}


@pytest.mark.parametrize(
    "needle, read",
    [
        (b"MIOpen.dll\0", lambda lib: lib.soname()),
        (b"miopenFoo\0", lambda lib: lib.defined_dynamic_functions()),
        (b"MIOpen_private.dll\0", lambda lib: lib.needed()),
    ],
    ids=["DLL name", "export name", "import name"],
)
def test_a_pe_cut_inside_a_name_is_an_error(tmp_path, needle, read):
    image = pe_image(
        "MIOpen.dll",
        exports={"miopenFoo": "code"},
        imports=["MIOpen_private.dll"],
    )
    cut = image.index(needle) + 3
    with pytest.raises(abi.AbiError, match="truncated PE file"):
        read(read_dll(tmp_path, image[:cut]))


def test_an_elf_file_still_gets_the_elf_reader(tmp_path):
    path = tmp_path / "libMIOpen.so.1.0"
    path.write_bytes(b"\x7fELF" + b"\0" * 60)
    with pytest.raises(abi.AbiError, match="ELF has no section headers"):
        abi.open_library(str(path), "library")


# --------------------------------------------------------------------------
# The commands on real DLL images
#
# open_library is not stubbed here, so the commands run the format detection and
# the readers as well as the checks.
# --------------------------------------------------------------------------


def write_flag_off_dll(tmp_path, imports=("KERNEL32.dll",), extra_exports=()):
    path = tmp_path / "MIOpen.dll"
    path.write_bytes(
        pe_image(
            "MIOpen.dll",
            exports={"miopenFoo": "code", "miopenBar": "code"}
            | {name: "code" for name in extra_exports},
            imports=imports,
            version=(3, 5, 1, 0),
        )
    )
    baseline = tmp_path / "public_symbols.baseline"
    baseline.write_text("miopenFoo\nmiopenBar\n")
    return path, baseline


def test_check_passes_a_flag_off_dll(tmp_path, capsys):
    dll, baseline = write_flag_off_dll(tmp_path)
    assert abi.main(["check", str(dll), "--baseline", str(baseline)]) == 0
    assert "public-abi symbol check: PASS" in capsys.readouterr().out


def test_check_fails_a_flag_off_dll_that_imports_the_private_dll(tmp_path, capsys):
    dll, baseline = write_flag_off_dll(
        tmp_path, imports=("KERNEL32.dll", "MIOpen_private.dll")
    )
    assert abi.main(["check", str(dll), "--baseline", str(baseline)]) == 1
    out = capsys.readouterr().out
    assert "flag-off MIOpen.dll has import of MIOpen_private" in out
    assert "public-abi symbol check: FAIL" in out


def test_check_fails_a_flag_off_dll_with_a_different_export_set(tmp_path, capsys):
    dll, baseline = write_flag_off_dll(tmp_path)
    baseline.write_text("miopenFoo\nmiopenBar\nmiopenBaz\n")
    assert abi.main(["check", str(dll), "--baseline", str(baseline)]) == 1
    assert "public-abi symbol check: FAIL" in capsys.readouterr().out


def test_check_fails_a_flag_off_dll_that_exports_an_impl_symbol(tmp_path, capsys):
    dll, baseline = write_flag_off_dll(tmp_path, extra_exports=("miopenFoo_impl",))
    assert abi.main(["check", str(dll), "--baseline", str(baseline)]) == 1
    out = capsys.readouterr().out
    assert "FAIL: flag-off MIOpen.dll exported *_impl symbols" in out


def test_dump_symbols_prints_the_public_exports_of_a_dll(tmp_path, capsys):
    dll, _ = write_flag_off_dll(tmp_path)
    assert abi.main(["dump-symbols", str(dll)]) == 0
    assert capsys.readouterr().out == "miopenBar\nmiopenFoo\n"


def test_dump_symbols_writes_the_file_it_is_given(tmp_path, capsys):
    dll, _ = write_flag_off_dll(tmp_path)
    out = tmp_path / "symbols.txt"
    assert abi.main(["dump-symbols", str(dll), "-o", str(out)]) == 0
    assert out.read_text() == "miopenBar\nmiopenFoo\n"


def test_dump_writes_the_symbols_name_imports_and_hash_of_a_dll(tmp_path):
    dll, _ = write_flag_off_dll(tmp_path)
    prefix = tmp_path / "dumped"
    assert abi.main(["dump", str(dll), str(prefix)]) == 0
    assert (tmp_path / "dumped.symbols").read_text() == "miopenBar\nmiopenFoo\n"
    assert (tmp_path / "dumped.soname").read_text() == "MIOpen.dll\n"
    assert (tmp_path / "dumped.needed").read_text() == "KERNEL32.dll\n"
    assert (tmp_path / "dumped.sha256").read_text() == (
        hashlib.sha256(dll.read_bytes()).hexdigest() + "\n"
    )


def test_check_wrapper_rejects_a_dll_wrapper_with_an_elf_private_library(
    tmp_path, monkeypatch, capsys
):
    assert run_check_wrapper(tmp_path, monkeypatch, wrapper_dll(), good_private()) == 2
    assert "different formats" in capsys.readouterr().err


def test_check_wrapper_rejects_an_elf_wrapper_with_a_dll_private_library(
    tmp_path, monkeypatch, capsys
):
    assert run_check_wrapper(tmp_path, monkeypatch, good_wrapper(), private_dll()) == 2
    assert "different formats" in capsys.readouterr().err


def test_a_needed_baseline_cannot_be_used_with_a_dll(tmp_path, monkeypatch, capsys):
    needed = tmp_path / "wrapper_needed.baseline"
    needed.write_text("MIOpen_private.dll\n")
    result = run_check_wrapper(
        tmp_path,
        monkeypatch,
        wrapper_dll(),
        private_dll(),
        extra=("--needed-baseline", str(needed)),
    )
    assert result == 2
    assert "cannot be used with a DLL" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Wrapper-only functions
#
# The flag-on wrapper exports a few functions of its own, declared in a header
# installed only with the flag on and absent from the flag-off baseline. They
# have no _impl, so none of the files that mirror miopen.h lists them.
# --------------------------------------------------------------------------

ACCESSOR = "miopenGetLastForwardedError"


def test_check_wrapper_passes_an_exported_wrapper_only_symbol(
    tmp_path, monkeypatch, capsys
):
    wrapper = good_wrapper(symbols=("miopenFoo", "miopenBar", ACCESSOR))
    result = run_check_wrapper(
        tmp_path, monkeypatch, wrapper, good_private(), wrapper_only=(ACCESSOR,)
    )
    assert result == 0
    assert "wrapper public-abi symbol check: PASS" in capsys.readouterr().out


def test_check_wrapper_fails_a_missing_wrapper_only_symbol(
    tmp_path, monkeypatch, capsys
):
    result = run_check_wrapper(
        tmp_path, monkeypatch, good_wrapper(), good_private(), wrapper_only=(ACCESSOR,)
    )
    assert result == 1
    assert f"missing from build: {ACCESSOR}" in capsys.readouterr().out


def test_check_wrapper_fails_a_wrapper_only_symbol_in_the_baseline(
    tmp_path, monkeypatch, capsys
):
    """A symbol the flag-off build exports too is not wrapper-only.

    Listing it leaves the expected set unchanged, so only this finding fails.
    """
    result = run_check_wrapper(
        tmp_path,
        monkeypatch,
        good_wrapper(),
        good_private(),
        wrapper_only=("miopenFoo",),
    )
    assert result == 1
    out = capsys.readouterr().out
    assert "wrapper-only symbols file lists symbols that are in the baseline" in out
    assert "  miopenFoo\n" in out


def run_with_public_header(tmp_path, monkeypatch, header_text):
    header = tmp_path / "miopen.h"
    header.write_text(header_text)
    wrapper = good_wrapper(symbols=("miopenFoo", "miopenBar", ACCESSOR))
    return run_check_wrapper(
        tmp_path,
        monkeypatch,
        wrapper,
        good_private(),
        extra=("--public-header", str(header)),
        wrapper_only=(ACCESSOR,),
    )


def test_check_wrapper_fails_a_wrapper_only_symbol_declared_in_miopen_h(
    tmp_path, monkeypatch, capsys
):
    header = f"MIOPEN_EXPORT miopenStatus_t {ACCESSOR}(miopenStatus_t* status);\n"
    assert run_with_public_header(tmp_path, monkeypatch, header) == 1
    out = capsys.readouterr().out
    assert "wrapper-only symbols file lists entry points that ARE declared" in out
    assert f"  {ACCESSOR}\n" in out


def test_check_wrapper_ignores_a_wrapper_only_symbol_named_in_a_comment(
    tmp_path, monkeypatch, capsys
):
    """miopen.h may point readers at the function without declaring it."""
    header = f"/* Flag-on builds also offer {ACCESSOR}. */\n// See {ACCESSOR}.\n"
    assert run_with_public_header(tmp_path, monkeypatch, header) == 0
    assert "no wrapper-only symbol appears in miopen.h" in capsys.readouterr().out


def test_check_installed_headers_exempts_the_private_directory_by_default(tmp_path):
    root = staged(
        tmp_path,
        {
            "miopen/miopen.h": "miopenStatus_t miopenFoo(void);\n",
            "miopen/private/miopen_impl.h": "miopenStatus_t miopenFoo_impl(void);\n",
        },
    )
    assert abi.main(["check-installed-headers", str(root)]) == 0


def test_check_installed_headers_fails_on_a_leak(tmp_path):
    root = staged(
        tmp_path, {"miopen/miopen.h": "miopenStatus_t miopenFoo_impl(void);\n"}
    )
    assert abi.main(["check-installed-headers", str(root)]) == 1


def test_check_installed_headers_that_cannot_run_exits_2(tmp_path):
    assert abi.main(["check-installed-headers", str(tmp_path / "missing")]) == 2


# --------------------------------------------------------------------------
# The check-headers command
#
# It checks three things: the hand-written stubs in wrapper.cpp, the wrapper's
# own functions (the hipdnn_forwarding.h header, the list of them, and their
# stubs), and the provider's declarations of the range entry points against the
# library's definitions.
# --------------------------------------------------------------------------

EMPTY_WRAPPER_HEADER = "#pragma once\n#include <miopen/miopen.h>\n"
ACCESSOR_HEADER = f"""#pragma once
#include <miopen/miopen.h>
#ifdef MIOPEN_BETA_API
MIOPEN_EXPORT miopenStatus_t {ACCESSOR}(miopenStatus_t* forwardedStatus);
#endif
"""
NO_WRAPPER_ONLY_FUNCTIONS = "# nothing yet\n"


def create_wrapper(*stubs: str) -> str:
    return wrapper_source(
        stub("miopenCreate", "    MIOPEN_WRAPPER_DISPATCH(miopenCreate);"), *stubs
    )


def accessor_stub(
    body: str = "    *forwardedStatus = miopenStatusSuccess;",
    param: str = "miopenStatus_t* forwardedStatus",
) -> str:
    return f"""
extern "C" miopenStatus_t {ACCESSOR}({param})
{{
{body}
    return miopenStatusSuccess;
}}
"""


def headers_inputs(
    tmp_path,
    wrapper=None,
    provider_decls=None,
    wrapper_header=EMPTY_WRAPPER_HEADER,
    wrapper_only=NO_WRAPPER_ONLY_FUNCTIONS,
):
    """Write the inputs and return the flags that point `check-headers` at them.

    ``False`` for `wrapper`, `provider_decls`, `wrapper_header` or `wrapper_only`
    leaves that flag off, so the command has to find the file itself.
    """
    flags = []
    if wrapper is not False:
        wrapper_path = tmp_path / "wrapper.cpp"
        wrapper_path.write_text(wrapper or create_wrapper())
        flags += ["--wrapper", str(wrapper_path)]
    definitions = tmp_path / "convolution_api.cpp"
    definitions.write_text(DEFINITIONS_SOURCE)
    flags += ["--range-definitions", str(definitions)]
    if provider_decls is not False:
        provider = tmp_path / "MiopenApi.hpp"
        provider.write_text(provider_decls or CONSUMER_SOURCE)
        flags += ["--provider-api", str(provider)]
    if wrapper_header is not False:
        header = tmp_path / "hipdnn_forwarding.h"
        header.write_text(wrapper_header)
        flags += ["--wrapper-header", str(header)]
    if wrapper_only is not False:
        listed = tmp_path / "wrapper_only_symbols.txt"
        listed.write_text(wrapper_only)
        flags += ["--wrapper-only", str(listed)]
    return flags


def with_accessor(**overrides):
    """The inputs for one consistent wrapper-only function, `overrides` on top."""
    return {
        "wrapper_header": ACCESSOR_HEADER,
        "wrapper_only": f"{ACCESSOR}\n",
        "wrapper": create_wrapper(accessor_stub()),
        **overrides,
    }


def run_check_headers(tmp_path, flags, *extra):
    # Not inside a repository, so a file that is not on disk reads as absent.
    root = tmp_path / "projects" / "miopen"
    root.mkdir(parents=True, exist_ok=True)
    return abi.main(["check-headers", "--source-root", str(root), *flags, *extra])


def test_check_headers_passes_consistent_inputs(tmp_path, capsys):
    flags = headers_inputs(tmp_path)
    assert run_check_headers(tmp_path, flags, "--require-provider") == 0
    assert "header consistency check: PASS" in capsys.readouterr().out


def test_check_headers_fails_on_a_stub_calling_the_wrong_impl(tmp_path, capsys):
    wrapper = wrapper_source(
        stub("miopenCreate", "    MIOPEN_WRAPPER_DISPATCH(miopenCreate);").replace(
            "miopenCreate_impl", "miopenDestroy_impl"
        )
    )
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, wrapper=wrapper)) == 1
    assert "miopenCreate calls miopenDestroy_impl" in capsys.readouterr().out


def test_check_headers_fails_on_a_stub_without_a_dispatch_macro(tmp_path, capsys):
    wrapper = wrapper_source(stub("miopenCreate", ""))
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, wrapper=wrapper)) == 1
    out = capsys.readouterr().out
    assert "miopenCreate has no MIOPEN_WRAPPER_DISPATCH" in out
    assert "Every stub in src/private/wrapper.cpp must open" in out


def test_check_headers_fails_on_provider_range_drift(tmp_path, capsys):
    drifted = consumer(
        declaration(FWD, UNNAMED_PARAMS.replace("size_t*,\n", "int*,\n", 1)),
        declaration(BWD),
        declaration(WRW),
    )
    flags = headers_inputs(tmp_path, provider_decls=drifted)
    assert run_check_headers(tmp_path, flags) == 1
    out = capsys.readouterr().out
    assert "MiopenApi.hpp prototypes have drifted" in out
    assert "src/include/miopen/miopen_internal.h" in out


def test_check_headers_no_longer_reads_the_generated_files(tmp_path):
    """The generator covers the rename headers, miopen_impl.h and miopen.h."""
    for removed in (
        "--public-header",
        "--impl-header",
        "--rename-header",
        "--range-test",
        "--provider-rename",
    ):
        with pytest.raises(SystemExit):
            abi.main(["check-headers", removed, "x"])


def test_check_headers_skips_an_absent_provider_unless_it_is_required(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        abi,
        "run_git",
        lambda repo_root, *args: _completed(128, stderr=b"fatal: not a git repository"),
    )
    flags = headers_inputs(tmp_path, provider_decls=False)
    assert run_check_headers(tmp_path, flags) == 0
    assert "SKIP:" in capsys.readouterr().out

    assert run_check_headers(tmp_path, flags, "--require-provider") == 1
    out = capsys.readouterr().out
    assert "FAIL:" in out
    assert "--require-provider was passed" in out


def test_check_headers_skips_a_checkout_without_wrapper_cpp(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(abi, "run_git", _unreadable_blob_git(tracked=False))
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, wrapper=False)) == 0
    out = capsys.readouterr().out
    assert "SKIP" in out
    assert "src/private/wrapper.cpp: not tracked at HEAD" in out


# --------------------------------------------------------------------------
# check-headers on the wrapper's own functions
# --------------------------------------------------------------------------


def test_check_headers_passes_a_consistent_wrapper_only_function(tmp_path, capsys):
    """Its stub must be held to its own rules, not to the forwarding stubs'."""
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, **with_accessor())) == 0
    out = capsys.readouterr().out
    assert "all 1 wrapper-only functions have a stub in wrapper.cpp" in out
    assert "all 2 wrapper stubs forward to their own _impl symbol" in out
    assert "header consistency check: PASS" in out


def test_check_headers_fails_when_the_header_and_list_disagree(tmp_path, capsys):
    inputs = with_accessor(wrapper_only="")
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, **inputs)) == 1
    out = capsys.readouterr().out
    assert f"- declared in hipdnn_forwarding.h, not listed: {ACCESSOR}" in out
    assert "A function only the flag-on wrapper exports is declared in" in out


def test_check_headers_fails_a_missing_wrapper_only_stub(tmp_path, capsys):
    inputs = with_accessor(wrapper=create_wrapper())
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, **inputs)) == 1
    assert f"{ACCESSOR} has no stub in wrapper.cpp" in capsys.readouterr().out


def test_check_headers_fails_a_drifted_wrapper_only_prototype(tmp_path, capsys):
    drifted = create_wrapper(accessor_stub(param="int* forwardedStatus"))
    inputs = with_accessor(wrapper=drifted)
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, **inputs)) == 1
    out = capsys.readouterr().out
    assert "wrapper.cpp prototypes have drifted from hipdnn_forwarding.h" in out
    assert "int*" in out


def test_check_headers_fails_a_wrapper_only_stub_with_a_dispatch_macro(
    tmp_path, capsys
):
    body = (
        f"    MIOPEN_WRAPPER_DISPATCH({ACCESSOR});\n"
        "    *forwardedStatus = miopenStatusSuccess;"
    )
    inputs = with_accessor(wrapper=create_wrapper(accessor_stub(body=body)))
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, **inputs)) == 1
    assert f"{ACCESSOR} carries a dispatch macro" in capsys.readouterr().out


def test_check_headers_fails_a_wrapper_only_stub_calling_an_impl(tmp_path, capsys):
    body = f"    return {ACCESSOR}_impl(forwardedStatus);"
    inputs = with_accessor(wrapper=create_wrapper(accessor_stub(body=body)))
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, **inputs)) == 1
    assert f"{ACCESSOR} calls {ACCESSOR}_impl" in capsys.readouterr().out


def test_check_headers_treats_an_untracked_wrapper_header_and_list_as_empty(
    tmp_path, monkeypatch, capsys
):
    """Removing the optional files must not turn the gate off."""
    monkeypatch.setattr(abi, "run_git", _unreadable_blob_git(tracked=False))
    flags = headers_inputs(tmp_path, wrapper_header=False, wrapper_only=False)
    assert run_check_headers(tmp_path, flags) == 0
    out = capsys.readouterr().out
    assert "SKIP" not in out
    assert "include_wrapper/miopen/hipdnn_forwarding.h is not part of this tree" in out
    assert "test/public_abi/wrapper_only_symbols.txt is not part of this tree" in out
    assert "wrapper_only_symbols.txt lists exactly the 0 hipdnn_forwarding.h" in out
    assert "header consistency check: PASS" in out


def test_check_headers_keeps_checking_without_the_wrapper_header_and_list(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(abi, "run_git", _unreadable_blob_git(tracked=False))
    flags = headers_inputs(
        tmp_path,
        wrapper=wrapper_source(stub("miopenCreate", "")),
        wrapper_header=False,
        wrapper_only=False,
    )
    assert run_check_headers(tmp_path, flags) == 1
    out = capsys.readouterr().out
    assert "SKIP" not in out
    assert "miopenCreate has no MIOPEN_WRAPPER_DISPATCH" in out


def test_check_headers_fails_a_wrapper_only_stub_when_the_header_is_untracked(
    tmp_path, monkeypatch, capsys
):
    """With no header nothing declares the stub, so it is judged as a forwarding
    stub, and fails for having no dispatch macro."""
    monkeypatch.setattr(abi, "run_git", _unreadable_blob_git(tracked=False))
    inputs = with_accessor(wrapper_header=False, wrapper_only=False)
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, **inputs)) == 1
    out = capsys.readouterr().out
    assert "SKIP" not in out
    assert f"{ACCESSOR} has no MIOPEN_WRAPPER_DISPATCH" in out


def test_check_headers_fails_a_listed_symbol_when_the_header_is_untracked(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(abi, "run_git", _unreadable_blob_git(tracked=False))
    inputs = with_accessor(wrapper_header=False)
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, **inputs)) == 1
    out = capsys.readouterr().out
    assert f"+ listed, not declared in hipdnn_forwarding.h: {ACCESSOR}" in out


def test_check_headers_fails_a_declared_symbol_when_the_list_is_untracked(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(abi, "run_git", _unreadable_blob_git(tracked=False))
    inputs = with_accessor(wrapper_only=False)
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, **inputs)) == 1
    out = capsys.readouterr().out
    assert f"- declared in hipdnn_forwarding.h, not listed: {ACCESSOR}" in out


def test_check_headers_still_fails_a_tracked_but_unreadable_wrapper_header(
    tmp_path, monkeypatch, capsys
):
    """Only an untracked file counts as empty; a broken checkout stays an error."""
    monkeypatch.setattr(abi, "run_git", _unreadable_blob_git(tracked=True))
    flags = headers_inputs(tmp_path, wrapper_header=False)
    assert run_check_headers(tmp_path, flags) == 2
    assert "hipdnn_forwarding.h" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The reader the checks share with the generator
# --------------------------------------------------------------------------

INACTIVE_ACCESSOR_HEADER = f"""#pragma once
#include <miopen/miopen.h>
#ifndef MIOPEN_BETA_API
MIOPEN_EXPORT miopenStatus_t {ACCESSOR}(miopenStatus_t* forwardedStatus);
#endif
"""


def test_check_headers_reads_a_conditional_header_the_way_the_generator_does(
    tmp_path, capsys
):
    """MIOPEN_BETA_API is defined in every library build, so this branch is dead."""
    inputs = with_accessor(wrapper_header=INACTIVE_ACCESSOR_HEADER)
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, **inputs)) == 1
    out = capsys.readouterr().out
    assert f"+ listed, not declared in hipdnn_forwarding.h: {ACCESSOR}" in out


def test_a_comment_opener_inside_a_string_does_not_hide_a_stub(tmp_path, capsys):
    literal = 'static const char* const kGlob = "hipdnn_plugins/*.so";\n'
    wrapper = literal + create_wrapper() + "\n/* a later comment */\n"
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, wrapper=wrapper)) == 0
    out = capsys.readouterr().out
    assert "all 2 wrapper stubs forward to their own _impl symbol" in out
    assert "header consistency check: PASS" in out


def test_an_unterminated_comment_in_a_source_is_an_error(tmp_path, capsys):
    wrapper = create_wrapper() + "\n/* never closed\n"
    assert run_check_headers(tmp_path, headers_inputs(tmp_path, wrapper=wrapper)) == 2
    assert "wrapper.cpp:" in capsys.readouterr().err


def test_an_array_parameter_is_rejected_rather_than_compared_by_name():
    source = consumer(declaration(FWD, "miopenHandle_t, size_t lens[2]"))
    with pytest.raises(abi.AbiError, match="array parameter"):
        abi.parse_range_prototypes(source, "consumer.cpp")
