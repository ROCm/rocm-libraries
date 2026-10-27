# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for generate_wrapper_sources.py.

The generator replaces four hand-copied files, so what matters is that it refuses
input it cannot read faithfully: a declaration it skipped is an entry point with
no stub, and nothing downstream would say so. Most of these tests therefore feed
it input it must reject, and check that the message names the line or function.

Text building is separate from formatting, so the parsing and rendering tests
need no clang-format. The command-line tests stand in a fake clang-format that
reports a version and echoes its input, which exercises the flow (write, check,
drift, sparse checkouts, version pin) without the real tool. That the real
clang-format reproduces the committed files is checked by running the generator
itself, as CI does.

    python -m pytest projects/miopen/script/test_generate_wrapper_sources.py
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_public_abi as abi  # noqa: E402
import generate_wrapper_sources as gen  # noqa: E402

PIN = "18.1.4"


def header(*body: str) -> str:
    """A miopen.h-shaped file: include guard, extern "C", then ``body``."""
    return (
        "#ifndef MIOPEN_GUARD_MIOPEN_H_\n"
        "#define MIOPEN_GUARD_MIOPEN_H_\n"
        "#include <miopen/export.h>\n"
        "#ifdef __cplusplus\n"
        'extern "C" {\n'
        "#endif\n"
        + "\n".join(body)
        + "\n#ifdef __cplusplus\n}\n#endif\n#endif // MIOPEN_GUARD_MIOPEN_H_\n"
    )


def entry(
    name: str, params: str = "miopenHandle_t handle", ret: str = "miopenStatus_t"
):
    return f"MIOPEN_EXPORT {ret} {name}({params});"


def parse(*body: str):
    return gen.parse_public_header(header(*body))


def by_name(entries):
    return {e.name: e for e in entries}


# --------------------------------------------------------------------------
# Reading declarations
# --------------------------------------------------------------------------


def test_entries_come_back_in_declaration_order_with_their_return_types():
    entries = parse(
        entry("miopenGetErrorString", "miopenStatus_t error", ret="const char*"),
        entry("miopenCreate", "miopenHandle_t* handle"),
        entry("miopenDestroy"),
    )
    assert [e.name for e in entries] == [
        "miopenGetErrorString",
        "miopenCreate",
        "miopenDestroy",
    ]
    assert [e.ret for e in entries] == [
        "const char*",
        "miopenStatus_t",
        "miopenStatus_t",
    ]


def test_a_declaration_may_span_lines_and_carry_comments():
    entries = parse(
        "/*! @brief doc, with MIOPEN_EXPORT in it; and a { too */",
        "MIOPEN_EXPORT miopenStatus_t miopenFoo(miopenHandle_t handle, // the handle",
        "                                       int n /* count */);",
    )
    assert [p.name for p in entries[0].params] == ["handle", "n"]


def test_a_mention_in_a_comment_is_not_a_declaration():
    entries = parse(entry("miopenCreate"), "// " + entry("miopenGhost"))
    assert [e.name for e in entries] == ["miopenCreate"]


def test_void_and_empty_parameter_lists_mean_no_parameters():
    entries = by_name(parse(entry("miopenA", "void"), entry("miopenB", "")))
    assert entries["miopenA"].params == ()
    assert entries["miopenB"].params == ()
    assert gen.render_impl_declaration(entries["miopenA"]) == (
        'extern "C" miopenStatus_t miopenA_impl(void);\n'
    )
    assert "return miopenB_impl();" in gen.render_stub(entries["miopenB"])


def test_error_lines_survive_block_comments():
    source = header(
        "/* one", "two", "three */", "MIOPEN_EXPORT miopenStatus_t miopenFoo(int);"
    )
    # The declaration is on line 11 of the fixture: 6 lines of preamble, the
    # three-line comment, then the declaration.
    with pytest.raises(abi.AbiError, match=r"miopen\.h:10:.*miopenFoo"):
        gen.parse_public_header(source)


def test_an_unterminated_comment_is_rejected():
    with pytest.raises(abi.AbiError, match="unterminated /\\* comment"):
        gen.parse_public_header(header(entry("miopenCreate"), "/* never closed"))


# --------------------------------------------------------------------------
# Default arguments and top-level const
# --------------------------------------------------------------------------

KTH = (
    "miopenHandle_t handle, size_t k, int32_t dim = -1, bool keepDim = false,"
    " void* workspace = nullptr"
)


def test_default_arguments_stay_in_the_declaration_and_leave_the_stub():
    kth = by_name(parse(entry("miopenKthvalueForward", KTH)))["miopenKthvalueForward"]
    declaration = gen.render_impl_declaration(kth)
    assert "int32_t dim = -1" in declaration
    assert "bool keepDim = false" in declaration
    assert "void* workspace = nullptr" in declaration

    stub = gen.render_stub(kth)
    assert "=" not in stub
    assert (
        "miopenKthvalueForward(miopenHandle_t handle, size_t k, int32_t dim,"
        " bool keepDim, void* workspace)"
    ) in stub
    assert "miopenKthvalueForward_impl(handle, k, dim, keepDim, workspace);" in stub


@pytest.mark.parametrize(
    "written, in_declaration",
    [
        ("const float alpha", "float alpha"),
        ("float* const out", "float* out"),
        ("const miopenTensorDescriptor_t desc", "miopenTensorDescriptor_t desc"),
        ("const float* const p", "const float* p"),
        # Not top level: the const belongs to what the pointer points at.
        ("const void* x", "const void* x"),
        ("const int* lens", "const int* lens"),
        # The inner const is below the top level, so nothing is dropped.
        ("const void* const* xs", "const void* const* xs"),
    ],
)
def test_only_a_top_level_const_is_dropped_from_the_declaration(
    written, in_declaration
):
    function = by_name(parse(entry("miopenFoo", written)))["miopenFoo"]
    assert f"miopenFoo_impl({in_declaration});" in gen.render_impl_declaration(function)
    # The stub is a definition, where a top-level const is legal: copy it as written.
    assert f"miopenFoo({written})" in gen.render_stub(function)


def test_pointer_spacing_in_the_source_does_not_change_the_output():
    spaced = by_name(parse(entry("miopenFoo", "size_t *major, const int *lens")))
    plain = by_name(parse(entry("miopenFoo", "size_t* major, const int* lens")))
    assert gen.render_stub(spaced["miopenFoo"]) == gen.render_stub(plain["miopenFoo"])


# --------------------------------------------------------------------------
# Conditionals
# --------------------------------------------------------------------------


def test_the_else_branch_of_a_known_macro_contributes_nothing():
    names = [
        e.name
        for e in parse(
            "#ifdef MIOPEN_BETA_API",
            entry("miopenBeta"),
            "#else",
            entry("miopenNotBeta"),
            "#endif",
            "#ifndef MIOPEN_BETA_API",
            entry("miopenAlsoNotBeta"),
            "#endif",
            entry("miopenAlways"),
        )
    ]
    assert names == ["miopenBeta", "miopenAlways"]


def test_directives_that_are_not_conditionals_are_ignored():
    entries = parse(
        "#define MIOPEN_DECLARE_OBJECT(name) \\",
        "    struct name {              \\",
        "    };",
        "#pragma clang diagnostic push",
        entry("miopenCreate"),
    )
    assert [e.name for e in entries] == ["miopenCreate"]


@pytest.mark.parametrize("keyword", ["if", "elif"])
def test_a_directive_needing_a_real_preprocessor_is_rejected_with_its_line(keyword):
    source = header(entry("miopenA"), f"#{keyword} 1", entry("miopenB"), "#endif")
    with pytest.raises(
        abi.AbiError, match=rf"miopen\.h:8: #{keyword} is not supported"
    ):
        gen.parse_public_header(source)


def test_an_unknown_ifdef_is_rejected_naming_the_macro_and_line():
    source = header(
        entry("miopenA"), "#ifdef SOMETHING_NEW", entry("miopenB"), "#endif"
    )
    with pytest.raises(abi.AbiError, match=r"miopen\.h:8: #ifdef SOMETHING_NEW"):
        gen.parse_public_header(source)


def test_an_unknown_ifndef_is_rejected_too():
    source = header("#ifndef SOMETHING_NEW", entry("miopenB"), "#endif")
    with pytest.raises(abi.AbiError, match="SOMETHING_NEW"):
        gen.parse_public_header(source)


def test_unbalanced_conditionals_are_rejected():
    with pytest.raises(abi.AbiError, match="#endif without"):
        gen.parse_public_header(header(entry("miopenA"), "#endif"))
    with pytest.raises(abi.AbiError, match="never closed"):
        gen.parse_public_header(header("#ifdef MIOPEN_BETA_API", entry("miopenA")))
    with pytest.raises(abi.AbiError, match="#else without"):
        gen.parse_public_header(
            header("#ifdef MIOPEN_BETA_API", "#else", "#else", "#endif")
        )


# --------------------------------------------------------------------------
# Declarations it must refuse rather than skip
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "declaration",
    [
        "MIOPEN_EXPORT miopenStatus_t;",
        "MIOPEN_EXPORT miopenFoo(int n);",
        "MIOPEN_EXPORT miopenStatus_t miopenFoo;",
    ],
)
def test_an_unparseable_declaration_is_rejected_with_its_line(declaration):
    with pytest.raises(abi.AbiError, match=r"miopen\.h:7: cannot parse"):
        gen.parse_public_header(header(declaration))


def test_a_declaration_with_a_body_is_rejected():
    with pytest.raises(abi.AbiError, match=r"miopen\.h:7: a '\{' comes before"):
        gen.parse_public_header(
            header("MIOPEN_EXPORT miopenStatus_t miopenFoo(int n) { return 0; }")
        )


def test_an_unterminated_declaration_is_rejected():
    with pytest.raises(abi.AbiError, match="never terminated"):
        gen.parse_public_header("MIOPEN_EXPORT miopenStatus_t miopenFoo(int n)\n")


@pytest.mark.parametrize(
    "unnamed",
    ["miopenHandle_t", "int*", "unsigned int", "const miopenHandle_t", "const void*"],
)
def test_an_unnamed_parameter_is_rejected_naming_the_function(unnamed):
    with pytest.raises(abi.AbiError, match=r"miopenFoo: parameter .* has no name"):
        parse(entry("miopenFoo", f"miopenHandle_t handle, {unnamed}"))


def test_a_function_pointer_parameter_is_rejected_with_advice():
    with pytest.raises(abi.AbiError, match=r"miopenFoo.*typedef"):
        parse(entry("miopenFoo", "miopenHandle_t handle, void (*callback)(int)"))


def test_an_array_parameter_is_rejected_with_advice():
    with pytest.raises(abi.AbiError, match=r"miopenFoo.*typedef"):
        parse(entry("miopenFoo", "int lens[4]"))


def test_a_variadic_parameter_is_rejected():
    with pytest.raises(abi.AbiError, match=r"miopenFoo: variadic"):
        parse(entry("miopenFoo", "miopenHandle_t handle, ..."))


def test_an_empty_parameter_is_rejected():
    with pytest.raises(abi.AbiError, match="empty parameter"):
        parse(entry("miopenFoo", "miopenHandle_t handle,"))


def test_a_duplicate_declaration_is_rejected_naming_both_lines():
    with pytest.raises(
        abi.AbiError, match=r"miopen\.h:8: miopenA is declared again.*line 7"
    ):
        parse(entry("miopenA"), entry("miopenA"))


def test_a_header_with_no_declarations_is_rejected():
    with pytest.raises(abi.AbiError, match="no MIOPEN_EXPORT declarations"):
        gen.parse_public_header(header("typedef int miopenThing;"))


def test_a_declaration_hidden_by_an_inactive_branch_cannot_empty_the_header_quietly():
    source = header("#ifndef MIOPEN_BETA_API", entry("miopenA"), "#endif")
    with pytest.raises(abi.AbiError, match="no MIOPEN_EXPORT declarations"):
        gen.parse_public_header(source)


# --------------------------------------------------------------------------
# Hand-written stubs
# --------------------------------------------------------------------------

WRAPPER_WITH_ONE_STUB = """
extern "C" miopenStatus_t miopenDestroy(miopenHandle_t handle)
{
    ::miopen::wrapper::hipdnn::ReleaseHandle(handle);
    MIOPEN_WRAPPER_DISPATCH(miopenDestroy);
    return miopenDestroy_impl(handle);
}
"""


def test_a_hand_written_stub_is_left_out_of_the_generated_ones():
    entries = parse(entry("miopenCreate"), entry("miopenDestroy"))
    hand_written = gen.hand_written_stubs(WRAPPER_WITH_ONE_STUB, entries)
    assert hand_written == {"miopenDestroy"}

    text = gen.render_passthrough(entries, hand_written)
    assert "miopenCreate(" in text
    assert 'extern "C" miopenStatus_t miopenDestroy(' not in text
    # It is still renamed and declared: the hand-written stub calls the _impl.
    assert "#define miopenDestroy miopenDestroy_impl" in gen.render_rename_header(
        entries
    )
    assert "miopenDestroy_impl(" in gen.render_impl_header(entries)


def test_a_wrapper_with_no_stubs_hand_writes_nothing():
    entries = parse(entry("miopenCreate"))
    assert gen.hand_written_stubs("// nothing\n#include <x>\n", entries) == set()


def test_a_hand_written_stub_for_an_unknown_name_is_rejected():
    entries = parse(entry("miopenCreate"))
    with pytest.raises(abi.AbiError, match="miopenDestroy"):
        gen.hand_written_stubs(WRAPPER_WITH_ONE_STUB, entries)


@pytest.mark.parametrize(
    "declaration",
    [
        entry("miopenDestroy", "int handle"),
        entry("miopenDestroy", ret="int"),
        entry("miopenDestroy", "miopenHandle_t handle, int extra"),
        entry("miopenDestroy", "void"),
    ],
)
def test_a_hand_written_stub_that_disagrees_with_miopen_h_is_rejected(declaration):
    with pytest.raises(abi.AbiError, match="miopenDestroy"):
        gen.hand_written_stubs(WRAPPER_WITH_ONE_STUB, parse(declaration))


def test_a_signature_mismatch_names_the_line_and_both_signatures():
    entries = parse(entry("miopenCreate"), entry("miopenDestroy", "int handle"))
    with pytest.raises(abi.AbiError) as error:
        gen.hand_written_stubs(WRAPPER_WITH_ONE_STUB, entries)
    message = str(error.value)
    assert "miopen.h:8: miopenStatus_t(int)" in message
    assert "wrapper.cpp: miopenStatus_t(miopenHandle_t)" in message
    assert "miopenCreate" not in message


@pytest.mark.parametrize(
    "params",
    [
        "miopenHandle_t h",
        "const miopenHandle_t handle",
        "miopenHandle_t handle = nullptr",
    ],
)
def test_a_stub_may_differ_from_miopen_h_where_the_type_does_not(params):
    entries = parse(entry("miopenDestroy", params))
    assert gen.hand_written_stubs(WRAPPER_WITH_ONE_STUB, entries) == {"miopenDestroy"}


def test_a_const_below_the_top_level_is_part_of_the_type():
    stub = WRAPPER_WITH_ONE_STUB.replace("miopenHandle_t handle", "void* handle")
    entries = parse(entry("miopenDestroy", "const void* handle"))
    with pytest.raises(abi.AbiError, match="miopenDestroy"):
        gen.hand_written_stubs(stub, entries)


def test_a_stub_body_with_a_nested_block_is_rejected_rather_than_missed():
    """parse_wrapper cannot see such a stub; leaving it out would duplicate it."""
    wrapper = """
extern "C" miopenStatus_t miopenDestroy(miopenHandle_t handle)
{
    if(handle != nullptr)
    {
        MIOPEN_WRAPPER_DISPATCH(miopenDestroy);
    }
    return miopenDestroy_impl(handle);
}
"""
    entries = parse(entry("miopenDestroy"))
    with pytest.raises(abi.AbiError, match="nested"):
        gen.hand_written_stubs(wrapper, entries)


ACCESSOR = "miopenGetLastForwardedError"
ACCESSOR_STUB = f"""
extern "C" miopenStatus_t {ACCESSOR}(miopenStatus_t* forwardedStatus)
{{
    *forwardedStatus = miopenStatusSuccess;
    return miopenStatusSuccess;
}}
"""


def test_a_wrapper_only_function_is_read_from_the_header():
    entries = parse(entry("miopenCreate"))
    source = header(
        "#ifdef MIOPEN_BETA_API",
        entry(ACCESSOR, "miopenStatus_t* forwardedStatus"),
        "#endif",
    )
    assert gen.wrapper_only_names(source, entries) == {ACCESSOR}


def test_no_forwarding_header_means_no_wrapper_only_functions():
    assert gen.wrapper_only_names(None, parse(entry("miopenCreate"))) == set()


def test_a_forwarding_header_may_declare_nothing():
    source = header("typedef int miopenThing;")
    assert gen.wrapper_only_names(source, parse(entry("miopenCreate"))) == set()


def test_a_name_declared_in_both_headers_is_rejected():
    entries = parse(entry("miopenCreate"))
    source = header(entry("miopenCreate"))
    with pytest.raises(abi.AbiError, match="both miopen.h and hipdnn_forwarding.h"):
        gen.wrapper_only_names(source, entries)


def test_a_wrapper_only_stub_is_accepted_and_never_generated():
    entries = parse(entry("miopenCreate"))
    wrapper = WRAPPER_WITH_ONE_STUB.replace("miopenDestroy", "miopenCreate") + (
        ACCESSOR_STUB
    )
    hand_written = gen.hand_written_stubs(wrapper, entries, {ACCESSOR})
    assert ACCESSOR in hand_written
    # It is not in miopen.h, so none of the generated files can mention it.
    for text in (
        gen.render_passthrough(entries, hand_written),
        gen.render_rename_header(entries),
        gen.render_impl_header(entries),
    ):
        assert ACCESSOR not in text


def test_a_stub_that_neither_header_declares_is_rejected_naming_both():
    entries = parse(entry("miopenCreate"))
    with pytest.raises(abi.AbiError, match="neither miopen.h nor hipdnn_forwarding.h"):
        gen.hand_written_stubs(ACCESSOR_STUB, entries)


def test_an_entry_point_that_is_not_status_returning_needs_a_hand_written_stub():
    entries = parse(
        entry("miopenGetErrorString", "miopenStatus_t error", ret="const char*")
    )
    with pytest.raises(
        abi.AbiError, match=r"miopen\.h:7: miopenGetErrorString returns"
    ):
        gen.check_dispatchable(entries, set())
    gen.check_dispatchable(entries, {"miopenGetErrorString"})


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

ENTRIES = parse(
    entry("miopenCreate", "miopenHandle_t* handle"),
    entry(
        "miopenSetStream", "miopenHandle_t handle, miopenAcceleratorQueue_t streamID"
    ),
    entry("miopenKthvalueForward", KTH),
)


def texts(hand_written=frozenset()):
    return gen.build_texts(ENTRIES, set(hand_written))


def test_every_file_names_its_source_and_the_command_that_regenerates_it():
    for key, text in texts().items():
        assert "GENERATED FILE. Do not edit." in text, key
        assert "Source: projects/miopen/include/miopen/miopen.h" in text, key
        assert f"Regenerate: {gen.GENERATOR_COMMAND}" in text, key


def test_no_generated_file_still_calls_itself_hand_maintained():
    for key, text in texts().items():
        assert "HAND-MAINTAINED" not in text, key
        assert "kept in sync" not in text, key
        assert "in lockstep" not in text, key


def test_each_file_keeps_its_own_copyright_line():
    out = texts()
    assert out["provider"].startswith("// Copyright © Advanced Micro Devices")
    for key in ("rename", "impl", "passthrough"):
        assert out[key].startswith("// Copyright (c) Advanced Micro Devices"), key


def test_the_rename_header_is_guarded_and_renames_every_entry_point():
    text = texts()["rename"]
    assert "#ifndef MIOPEN_BUILDING_PRIVATE\n#error" in text
    for e in ENTRIES:
        assert f"#define {e.name} {e.name}_impl\n" in text


def test_the_provider_mirror_has_the_same_defines_and_no_guard():
    out = texts()
    assert "MIOPEN_BUILDING_PRIVATE" not in out["provider"]

    def defines(text):
        return [line for line in text.splitlines() if line.startswith("#define")]

    assert defines(out["provider"]) == defines(out["rename"])


def test_the_impl_header_declares_every_entry_point_and_refuses_the_private_library():
    text = texts()["impl"]
    assert "#ifdef MIOPEN_BUILDING_PRIVATE\n#error" in text
    assert "#include <miopen/miopen.h>" in text
    assert (
        'extern "C" miopenStatus_t miopenCreate_impl(miopenHandle_t* handle);' in text
    )


def test_the_passthrough_file_guards_against_the_rename_leaking_in():
    text = texts()["passthrough"]
    assert "#ifdef MIOPEN_BUILDING_PRIVATE\n#error" in text
    assert "#ifdef miopenCreate\n#error" in text
    assert '#include "miopen_impl.h"' in text


def test_a_stub_is_the_dispatch_then_a_call_to_its_own_impl():
    stub = gen.render_stub(ENTRIES[1])
    assert stub == (
        'extern "C" miopenStatus_t miopenSetStream(miopenHandle_t handle,'
        " miopenAcceleratorQueue_t streamID)\n"
        "{\n"
        "    MIOPEN_WRAPPER_DISPATCH(miopenSetStream);\n"
        "    return miopenSetStream_impl(handle, streamID);\n"
        "}\n"
    )


def test_stubs_follow_declaration_order():
    text = texts()["passthrough"]
    positions = [text.index(f"MIOPEN_WRAPPER_DISPATCH({e.name})") for e in ENTRIES]
    assert positions == sorted(positions)


def test_the_text_is_identical_on_every_run():
    assert texts() == texts()
    assert texts({"miopenCreate"}) == texts({"miopenCreate"})


# --------------------------------------------------------------------------
# clang-format discovery and the version pin
# --------------------------------------------------------------------------

PRE_COMMIT = textwrap.dedent(
    f"""\
    repos:
      - repo: https://github.com/pre-commit/mirrors-clang-format
        rev: v{PIN}
        hooks:
          - id: clang-format
    """
)


def fake_clang_format(
    tmp_path: Path, version: str = PIN, name: str = "clang-format"
) -> Path:
    """A stand-in that reports ``version`` and echoes stdin; logs how it was called."""
    log = tmp_path / f"{name}.log"
    script = tmp_path / name
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import sys
            if "--version" in sys.argv:
                print("clang-format version {version}")
                raise SystemExit(0)
            style = next(a for a in sys.argv if a.startswith("--style=file:"))
            assumed = next(a for a in sys.argv if a.startswith("--assume-filename="))
            with open({str(log)!r}, "a") as out:
                out.write(assumed.split("=", 1)[1] + " | ")
                out.write(open(style.split(":", 1)[1]).read().strip() + "\\n")
            sys.stdout.write(sys.stdin.read())
            """
        )
    )
    script.chmod(0o755)
    return script


def test_the_pin_is_read_from_the_pre_commit_config():
    assert gen.pinned_clang_format_version(PRE_COMMIT) == PIN


def test_a_config_without_the_pin_is_an_error():
    with pytest.raises(abi.AbiError, match="mirrors-clang-format"):
        gen.pinned_clang_format_version("repos: []\n")


def test_the_real_pre_commit_config_carries_a_readable_pin():
    config = Path(__file__).resolve().parents[3] / ".pre-commit-config.yaml"
    if not config.is_file():
        pytest.skip("not run inside the monorepo")
    assert gen.pinned_clang_format_version(config.read_text(encoding="utf-8"))


def test_a_clang_format_of_the_wrong_version_is_refused_with_the_fix(tmp_path):
    wrong = fake_clang_format(tmp_path, "18.1.3")
    with pytest.raises(abi.AbiError) as caught:
        gen.find_clang_format(str(wrong), PIN)
    message = str(caught.value)
    assert PIN in message
    assert "18.1.3" in message
    assert f"pip install clang-format=={PIN}" in message
    assert "miopen-wrapper-sources-regenerate" in message


def test_the_environment_variable_is_held_to_the_pin_too(tmp_path, monkeypatch):
    monkeypatch.setenv("CLANG_FORMAT", str(fake_clang_format(tmp_path, "18.1.3")))
    with pytest.raises(abi.AbiError, match="18.1.3"):
        gen.find_clang_format(None, PIN)


def test_discovery_takes_the_first_candidate_with_the_pinned_version(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CLANG_FORMAT", raising=False)
    wrong = fake_clang_format(tmp_path, "18.1.3", name="on_path")
    right = fake_clang_format(tmp_path, PIN, name="fallback")
    monkeypatch.setattr(gen.shutil, "which", lambda _name: str(wrong))
    monkeypatch.setattr(gen, "FALLBACK_CLANG_FORMAT", str(right))
    assert gen.find_clang_format(None, PIN) == str(right)


def test_discovery_with_no_matching_candidate_lists_what_it_found(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CLANG_FORMAT", raising=False)
    wrong = fake_clang_format(tmp_path, "18.1.3")
    monkeypatch.setattr(gen.shutil, "which", lambda _name: str(wrong))
    monkeypatch.setattr(gen, "FALLBACK_CLANG_FORMAT", str(tmp_path / "absent"))
    with pytest.raises(abi.AbiError, match="18.1.3"):
        gen.find_clang_format(None, PIN)


# --------------------------------------------------------------------------
# The command line, against a small tree and the fake clang-format
# --------------------------------------------------------------------------

PUBLIC = header(
    entry("miopenCreate", "miopenHandle_t* handle"),
    entry("miopenDestroy"),
    entry(
        "miopenSetStream", "miopenHandle_t handle, miopenAcceleratorQueue_t streamID"
    ),
)


class Tree:
    """projects/miopen plus the provider and the repo-root config, in a tmp dir."""

    def __init__(self, root: Path, with_provider: bool = True):
        self.repo = root / "repo"
        self.miopen = self.repo / "projects" / "miopen"
        self.provider = self.repo / "dnn-providers" / "miopen-provider"
        self.write(self.miopen / "include/miopen/miopen.h", PUBLIC)
        self.write(self.miopen / "src/private/wrapper.cpp", WRAPPER_WITH_ONE_STUB)
        self.write(self.miopen / ".clang-format", "# miopen style\n")
        self.write(self.repo / ".pre-commit-config.yaml", PRE_COMMIT)
        if with_provider:
            self.write(self.provider / ".clang-format", "# provider style\n")
            self.write(self.provider / "MiopenApiPrivateRename.hpp", "// stale\n")
        self.cf = fake_clang_format(root)

    @staticmethod
    def write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def path(self, rel: str) -> Path:
        return self.repo / rel

    def run(self, *flags: str) -> int:
        return gen.main(
            ["--source-root", str(self.miopen), "--clang-format", str(self.cf), *flags]
        )


GENERATED = [
    "projects/miopen/src/private/miopen_private_rename.h",
    "projects/miopen/src/private/miopen_impl.h",
    "projects/miopen/src/private/wrapper_passthrough.cpp",
    "dnn-providers/miopen-provider/MiopenApiPrivateRename.hpp",
]


@pytest.fixture
def tree(tmp_path):
    return Tree(tmp_path)


def test_writing_creates_all_four_files_and_the_second_run_changes_nothing(
    tree, capsys
):
    assert tree.run() == 0
    for rel in GENERATED:
        assert tree.path(rel).is_file(), rel
    first = {rel: tree.path(rel).read_bytes() for rel in GENERATED}
    capsys.readouterr()

    assert tree.run() == 0
    assert "wrote:" not in capsys.readouterr().out
    assert {rel: tree.path(rel).read_bytes() for rel in GENERATED} == first


def test_the_passthrough_file_leaves_out_what_wrapper_cpp_defines(tree):
    assert tree.run() == 0
    text = tree.path(GENERATED[2]).read_text(encoding="utf-8")
    assert "miopenCreate(" in text
    assert "miopenSetStream(" in text
    assert 'extern "C" miopenStatus_t miopenDestroy(' not in text


def test_check_passes_on_freshly_written_files(tree, capsys):
    assert tree.run() == 0
    capsys.readouterr()
    assert tree.run("--check", "--require-provider") == 0
    assert "wrapper source generation check: PASS" in capsys.readouterr().out


def test_check_writes_nothing(tree):
    assert tree.run("--check") == 1
    assert not tree.path(GENERATED[2]).exists()
    assert tree.path(GENERATED[3]).read_text(encoding="utf-8") == "// stale\n"


def test_a_hand_edit_fails_the_check_with_a_diff_and_the_fix(tree, capsys):
    assert tree.run() == 0
    target = tree.path(GENERATED[1])
    target.write_text(
        target.read_text(encoding="utf-8").replace(
            "miopenCreate_impl", "miopenCreate_imp"
        ),
        encoding="utf-8",
    )
    capsys.readouterr()

    assert tree.run("--check") == 1
    out = capsys.readouterr().out
    assert f"FAIL: {GENERATED[1]} differs" in out
    assert '-extern "C" miopenStatus_t miopenCreate_imp(' in out
    assert '+extern "C" miopenStatus_t miopenCreate_impl(' in out
    assert gen.GENERATOR_COMMAND in out
    assert "PASS: " + GENERATED[0] in out


def test_a_changed_signature_in_miopen_h_fails_the_check_in_the_declaration_and_stub(
    tree, capsys
):
    assert tree.run() == 0
    public = tree.miopen / "include/miopen/miopen.h"
    public.write_text(
        public.read_text(encoding="utf-8").replace(
            "miopenHandle_t* handle", "int* handle"
        ),
        encoding="utf-8",
    )
    capsys.readouterr()

    assert tree.run("--check") == 1
    out = capsys.readouterr().out
    assert '+extern "C" miopenStatus_t miopenCreate_impl(int* handle);' in out
    assert "miopenCreate(int* handle)" in out


def test_a_changed_signature_under_a_hand_written_stub_exits_2_and_writes_nothing(
    tree, capsys
):
    assert tree.run() == 0
    written = {rel: tree.path(rel).read_bytes() for rel in GENERATED}
    public = tree.miopen / "include/miopen/miopen.h"
    public.write_text(
        public.read_text(encoding="utf-8").replace(
            entry("miopenDestroy"), entry("miopenDestroy", "int handle")
        ),
        encoding="utf-8",
    )
    capsys.readouterr()

    assert tree.run("--check") == 2
    assert "miopenDestroy" in capsys.readouterr().err
    assert tree.run() == 2
    assert {rel: tree.path(rel).read_bytes() for rel in GENERATED} == written


def test_a_missing_generated_file_fails_the_check(tree, capsys):
    assert tree.run() == 0
    tree.path(GENERATED[2]).unlink()
    capsys.readouterr()

    class NotInGit:
        def __call__(self, repo_root, *args):
            return subprocess.CompletedProcess(
                ["git"], 128, b"", b"fatal: not a git repo"
            )

    abi_run_git = abi.run_git
    abi.run_git = NotInGit()
    try:
        assert tree.run("--check") == 1
    finally:
        abi.run_git = abi_run_git
    assert f"FAIL: {GENERATED[2]} does not exist" in capsys.readouterr().out


def test_a_long_diff_is_capped(tree, capsys, monkeypatch):
    monkeypatch.setattr(gen, "MAX_DIFF_LINES", 5)
    assert tree.run() == 0
    tree.path(GENERATED[1]).write_text("// emptied\n", encoding="utf-8")
    capsys.readouterr()

    assert tree.run("--check") == 1
    out = capsys.readouterr().out
    lines = out.splitlines()
    marker = next(
        i for i, line in enumerate(lines) if "more diff lines not shown" in line
    )
    first = next(i for i, line in enumerate(lines) if line.startswith("--- "))
    assert marker - first == 5


def test_the_formatter_is_given_each_files_own_style_and_path(tmp_path):
    tree = Tree(tmp_path)
    assert tree.run() == 0
    log = (tmp_path / "clang-format.log").read_text(encoding="utf-8").splitlines()
    assert sorted(log) == sorted(
        [
            f"{GENERATED[0]} | # miopen style",
            f"{GENERATED[1]} | # miopen style",
            f"{GENERATED[2]} | # miopen style",
            f"{GENERATED[3]} | # provider style",
        ]
    )


def test_the_wrong_clang_format_exits_2_without_touching_a_file(tmp_path, capsys):
    tree = Tree(tmp_path)
    tree.cf = fake_clang_format(tmp_path, "18.1.3", name="old")
    assert tree.run() == 2
    assert "18.1.3" in capsys.readouterr().err
    assert tree.path(GENERATED[3]).read_text(encoding="utf-8") == "// stale\n"


def test_a_parse_error_exits_2_and_names_the_line(tree, capsys):
    public = tree.miopen / "include/miopen/miopen.h"
    public.write_text(
        header(entry("miopenCreate", "miopenHandle_t* handle"), "#if 1", "#endif"),
        encoding="utf-8",
    )
    assert tree.run() == 2
    assert "miopen.h:8: #if is not supported" in capsys.readouterr().err
    assert not tree.path(GENERATED[2]).exists()


def test_writing_without_the_provider_on_disk_exits_2(tmp_path, capsys):
    tree = Tree(tmp_path, with_provider=False)
    assert tree.run() == 2
    assert "MiopenApiPrivateRename.hpp is not on disk" in capsys.readouterr().err
    assert not tree.path(GENERATED[2]).exists()


def sparse(monkeypatch, tracked: dict[str, bytes]):
    """Answer git as a commit that tracks ``tracked`` would, whatever is on disk."""

    def run_git(repo_root, *args):
        if args[0] == "cat-file":
            rel = args[2].removeprefix("HEAD:")
            if rel in tracked:
                return subprocess.CompletedProcess(["git"], 0, tracked[rel], b"")
            return subprocess.CompletedProcess(
                ["git"], 128, b"", b"fatal: path not in HEAD"
            )
        if args[0] == "ls-tree":
            return subprocess.CompletedProcess(["git"], 0, b"", b"")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(abi, "run_git", run_git)


def test_a_tree_without_the_provider_skips_it_unless_it_is_required(
    tmp_path, monkeypatch, capsys
):
    tree = Tree(tmp_path)
    assert tree.run() == 0
    for path in (
        tree.provider / "MiopenApiPrivateRename.hpp",
        tree.provider / ".clang-format",
    ):
        path.unlink()
    sparse(monkeypatch, {})
    capsys.readouterr()

    assert tree.run("--check") == 0
    out = capsys.readouterr().out
    assert "SKIP: dnn-providers/miopen-provider/MiopenApiPrivateRename.hpp" in out

    assert tree.run("--check", "--require-provider") == 2
    assert "--require-provider" in capsys.readouterr().err


def test_a_provider_only_change_is_compared_against_miopen_as_committed(
    tmp_path, monkeypatch, capsys
):
    """MIOpen inputs missing from disk but tracked at HEAD are read from git."""
    tree = Tree(tmp_path)
    assert tree.run() == 0
    tracked = {
        "projects/miopen/include/miopen/miopen.h": (
            tree.miopen / "include/miopen/miopen.h"
        ),
        "projects/miopen/src/private/wrapper.cpp": (
            tree.miopen / "src/private/wrapper.cpp"
        ),
        "projects/miopen/.clang-format": tree.miopen / ".clang-format",
        **{rel: tree.path(rel) for rel in GENERATED[:3]},
    }
    committed = {rel: path.read_bytes() for rel, path in tracked.items()}
    for path in tracked.values():
        path.unlink()
    sparse(monkeypatch, committed)
    capsys.readouterr()

    assert tree.run("--check", "--require-provider") == 0
    out = capsys.readouterr().out
    assert "reading it from HEAD" in out
    assert "wrapper source generation check: PASS" in out

    # The provider copy is on disk; breaking it is caught against MIOpen's HEAD.
    mirror = tree.provider / "MiopenApiPrivateRename.hpp"
    mirror.write_text(mirror.read_text(encoding="utf-8") + "#define miopenGhost x\n")
    assert tree.run("--check", "--require-provider") == 1


def test_a_tree_without_miopen_has_nothing_to_generate_from(
    tmp_path, monkeypatch, capsys
):
    tree = Tree(tmp_path)
    for rel in ("include/miopen/miopen.h", "src/private/wrapper.cpp"):
        (tree.miopen / rel).unlink()
    sparse(monkeypatch, {})
    assert tree.run("--check") == 0
    assert "SKIP: MIOpen's own sources could not be read" in capsys.readouterr().out


def test_the_provider_is_formatted_with_its_own_style_even_when_only_git_has_it(
    tmp_path, monkeypatch
):
    tree = Tree(tmp_path)
    assert tree.run() == 0
    style = (tree.provider / ".clang-format").read_bytes()
    (tree.provider / ".clang-format").unlink()
    sparse(monkeypatch, {"dnn-providers/miopen-provider/.clang-format": style})
    (tmp_path / "clang-format.log").unlink()

    assert tree.run("--check") == 0
    log = (tmp_path / "clang-format.log").read_text(encoding="utf-8")
    assert f"{GENERATED[3]} | # provider style" in log


FORWARDING_HEADER_PATH = "include_wrapper/miopen/hipdnn_forwarding.h"


def add_wrapper_only_function(tree: Tree) -> None:
    tree.write(
        tree.miopen / FORWARDING_HEADER_PATH,
        header(entry(ACCESSOR, "miopenStatus_t* forwardedStatus")),
    )
    wrapper = tree.miopen / "src/private/wrapper.cpp"
    wrapper.write_text(
        wrapper.read_text(encoding="utf-8") + ACCESSOR_STUB, encoding="utf-8"
    )


def test_a_wrapper_only_function_changes_none_of_the_generated_files(tmp_path):
    plain = Tree(tmp_path / "plain")
    assert plain.run() == 0
    with_function = Tree(tmp_path / "with")
    add_wrapper_only_function(with_function)
    assert with_function.run() == 0
    for rel in GENERATED:
        assert with_function.path(rel).read_bytes() == plain.path(rel).read_bytes(), rel


def test_a_wrapper_only_stub_without_its_header_is_rejected(tmp_path, capsys):
    tree = Tree(tmp_path)
    add_wrapper_only_function(tree)
    tree.path(f"projects/miopen/{FORWARDING_HEADER_PATH}").unlink()
    assert tree.run() == 2
    assert f"{ACCESSOR}" in capsys.readouterr().err


def test_a_name_in_both_headers_exits_2(tmp_path, capsys):
    tree = Tree(tmp_path)
    tree.write(tree.miopen / FORWARDING_HEADER_PATH, header(entry("miopenCreate")))
    assert tree.run() == 2
    assert "both miopen.h and hipdnn_forwarding.h" in capsys.readouterr().err


def test_check_passes_when_the_forwarding_header_is_not_tracked(
    tmp_path, monkeypatch, capsys
):
    tree = Tree(tmp_path)
    assert tree.run() == 0
    sparse(monkeypatch, {})
    capsys.readouterr()
    assert tree.run("--check", "--require-provider") == 0
    assert "wrapper source generation check: PASS" in capsys.readouterr().out
