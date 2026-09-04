# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for check_public_abi.py.

The declaration parser is the only thing standing between the range entry
points and a divergence that links cleanly and corrupts arguments at run time,
so it is worth testing against inputs it is expected to reject as well as ones
it is expected to accept. Running the check against a tree that is already
correct only ever exercises the passing path.

The binary checks are covered the same way, against a stand-in that answers the
three questions those checks ask of a shared object. That stand-in deliberately
does not exercise the ELF parser itself: the parser reads a real libMIOpen.so on
every packaging build, so it is continuously covered by construction, whereas
the conditions the checks are meant to catch -- a stale half of a build, an
entry point renamed with no stub -- appear in no tree that is working.

Everything here works on string fixtures and stand-ins, so no build, toolchain
or GPU is needed and the module runs in a lint lane:

    python -m pytest projects/miopen/script/test_check_public_abi.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Colocated with the script under test. pytest's default import mode already
# puts this directory on sys.path; do it explicitly so the module also runs
# under a bare `python -m pytest` from any working directory.
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

    A definition with nothing at all before it has no preceding ';', '}' or
    '{' to mark where its return type begins, so the parser does not recognise
    it. No real source is written that way -- both consumers open with
    `extern "C" {` and the library file with its includes -- but the point of
    this gate is that nothing it fails to read passes quietly. The entry point
    set check is what makes that true, so pin it here rather than the parser
    result: an unrecognised definition leaves the set short and fails.
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
# The rename headers and the provider's mirror of them
# --------------------------------------------------------------------------


RENAME_HEADER = """
#pragma once
#define miopenCreate miopenCreate_impl
#define miopenDestroy miopenDestroy_impl
"""


def test_parse_renames_reads_defines():
    assert abi.parse_renames(RENAME_HEADER) == {
        "miopenCreate": "miopenCreate_impl",
        "miopenDestroy": "miopenDestroy_impl",
    }


def test_duplicate_rename_raises():
    with pytest.raises(abi.AbiError, match="miopenCreate"):
        abi.parse_renames(RENAME_HEADER + "#define miopenCreate miopenCreate_impl\n")


def test_rename_to_wrong_target_is_reported(capsys):
    renames = abi.parse_renames(
        RENAME_HEADER + "#define miopenGetVersion miopenCreate_impl\n"
    )
    assert not abi.check_rename_targets(renames)
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "miopenGetVersion -> miopenCreate_impl" in out


def test_provider_mirror_matches(capsys):
    renames = abi.parse_renames(RENAME_HEADER)
    assert abi.check_provider_rename_mirror(renames, dict(renames))
    assert "PASS" in capsys.readouterr().out


def test_provider_mirror_missing_entry_is_reported(capsys):
    lib = abi.parse_renames(RENAME_HEADER)
    provider = {k: v for k, v in lib.items() if k != "miopenDestroy"}
    assert not abi.check_provider_rename_mirror(lib, provider)
    out = capsys.readouterr().out
    assert "- in the library, missing from the provider: miopenDestroy" in out


def test_provider_mirror_divergent_target_is_reported(capsys):
    lib = abi.parse_renames(RENAME_HEADER)
    provider = dict(lib, miopenCreate="miopenCreate_private")
    assert not abi.check_provider_rename_mirror(lib, provider)
    out = capsys.readouterr().out
    assert "~ miopenCreate" in out
    assert "miopenCreate_private" in out


# --------------------------------------------------------------------------
# The dispatch seam
#
# A stub that never reaches MIOPEN_WRAPPER_DISPATCH silently loses the ability
# to route to hipDNN, and no build catches it: the macro's assert needs the
# macro to be present, and is compiled out under NDEBUG regardless.
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
    assert "miopenDestroy has no MIOPEN_WRAPPER_DISPATCH" in capsys.readouterr().out


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
    assert "miopenCreate has 2 MIOPEN_WRAPPER_DISPATCH calls" in capsys.readouterr().out


def test_a_commented_out_macro_does_not_count_as_present(capsys):
    source = wrapper_source(
        stub("miopenCreate", "    // MIOPEN_WRAPPER_DISPATCH(miopenCreate);")
    )
    assert not abi.check_wrapper_dispatch(dispatches_of(source))
    assert "miopenCreate has no MIOPEN_WRAPPER_DISPATCH" in capsys.readouterr().out


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
    assert "miopenGetErrorString carries MIOPEN_WRAPPER_DISPATCH but is exempt" in out
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


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def _repo_with_unreadable_blob(tmp_path: Path) -> tuple[Path, Path]:
    """Build the exact shape a failed promisor fetch leaves behind.

    The commit's tree still names the file -- so ls-tree answers -- while the
    blob it points at is gone, so cat-file cannot produce the content. Deleting
    the loose object reproduces that locally without needing a partial clone or
    a network.
    """
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "tracked.hpp"
    target.write_text("#pragma once\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "nobody@example.invalid")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", "tracked.hpp")
    _git(root, "commit", "-qm", "add tracked.hpp")
    blob = _git(root, "rev-parse", "HEAD:tracked.hpp")
    target.unlink()
    (root / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    return root, target


@needs_git
def test_tracked_but_unreadable_file_is_a_hard_failure(tmp_path):
    """A checkout that carries the file but cannot produce it must not go green.

    Skipping here would drop a file that is under test out of the comparison
    while still reporting success, which is the one outcome this gate cannot
    afford: the drift it catches links cleanly and only misbehaves at run time.
    """
    root, target = _repo_with_unreadable_blob(tmp_path)
    with pytest.raises(abi.AbiError) as caught:
        abi.read_tracked_source(target, root, {})
    assert "tracked.hpp" in str(caught.value)


@needs_git
def test_the_hard_failure_carries_git_stderr(tmp_path):
    """Whatever git said is the only evidence of why the checkout is broken."""
    root, target = _repo_with_unreadable_blob(tmp_path)
    with pytest.raises(abi.AbiError) as caught:
        abi.read_tracked_source(target, root, {})
    message = str(caught.value)
    assert "git cat-file said:" in message
    assert "(git printed nothing)" not in message


@needs_git
def test_a_file_absent_from_the_commit_is_still_only_a_skip(tmp_path):
    """The escalation must stay narrow.

    A sparse checkout legitimately lacks whole subtrees the commit never
    carried, and failing on those would break every partial-checkout lane
    instead of catching drift.
    """
    root, _ = _repo_with_unreadable_blob(tmp_path)
    reasons: dict[str, str] = {}
    assert abi.read_tracked_source(root / "never_added.hpp", root, reasons) is None
    assert "not tracked at HEAD" in "".join(reasons.values())


def test_a_non_checkout_is_a_skip_not_a_failure(tmp_path):
    """Source tarballs have no git at all; that is not a broken checkout."""
    reasons: dict[str, str] = {}
    assert abi.read_tracked_source(tmp_path / "absent.hpp", tmp_path, reasons) is None
    assert "not a git checkout" in "".join(reasons.values())


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


def test_a_leaked_rename_is_reported_with_file_and_line(tmp_path, capsys):
    root = staged(
        tmp_path,
        {"miopen/miopen.h": "// header\nmiopenStatus_t miopenFoo_impl(int);\n"},
    )
    assert abi.check_installed_headers(str(root), []) is False
    out = capsys.readouterr().out
    assert "miopen/miopen.h:2: miopenFoo_impl" in out


def test_the_private_directory_is_exempt(tmp_path):
    """The private declarations spell the private names by definition."""
    root = staged(
        tmp_path,
        {"miopen/private/miopen_impl.h": "miopenStatus_t miopenFoo_impl(int);\n"},
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


# --------------------------------------------------------------------------
# The binary stand-in
#
# The co-version and superset checks ask a shared object for three things:
# what it exports, what SONAME it declares, and what it is linked against.
# --------------------------------------------------------------------------


class FakeElf:
    """A shared object as the binary checks see it."""

    def __init__(self, path, symbols=(), soname="", needed=()):
        self.path = Path(path)
        self._symbols = set(symbols)
        self._soname = soname
        self._needed = tuple(needed)

    def defined_dynamic_functions(self):
        return set(self._symbols)

    def soname(self):
        return self._soname

    def needed(self):
        return sorted(self._needed)


def wrapper_elf(
    symbols=(), soname="libMIOpen.so.1", needed=("libMIOpen_private.so.1",)
):
    return FakeElf("libMIOpen.so", symbols, soname, needed)


def private_elf(symbols=(), soname="libMIOpen_private.so.1"):
    return FakeElf("libMIOpen_private.so", symbols, soname, ())


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
