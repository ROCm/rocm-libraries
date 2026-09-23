# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A real compiler's verdict on the emitted C++. ``packs/<Name>Native.cpp`` is what an
agent or a human fills in to make an engine serve real graphs (see RUNBOOK.md)."""

import subprocess
import shutil
import tempfile
from pathlib import Path

import pytest

from codegen.generator import cpp_escape, mint_ids


def _find_include_dir(name: str) -> Path | None:
    # tests -> IngestorGenerator -> tools -> hipdnn
    hipdnn_root = Path(__file__).resolve().parents[3]
    candidate = hipdnn_root / name / "include"
    return candidate if candidate.is_dir() else None


@pytest.fixture(scope="module")
def compile_env():
    """Best-effort host-compile environment for the emitted native stub. The SDKs'
    ``version.h``/``CacheRootDefaults.h`` are CMake-configured, so a from-scratch
    compile needs stand-ins. Skips rather than fails when a prerequisite is absent."""
    gxx = shutil.which("g++") or shutil.which("clang++")
    if gxx is None:
        pytest.skip("no host C++ compiler (g++/clang++) found on PATH")

    plugin_sdk = _find_include_dir("plugin_sdk")
    data_sdk = _find_include_dir("data_sdk")
    flatbuffers_sdk = _find_include_dir("flatbuffers_sdk")
    provider_src = (
        Path(__file__).resolve().parents[5]
        / "dnn-providers"
        / "hip-kernel-provider"
        / "src"
    )
    if not (plugin_sdk and data_sdk and flatbuffers_sdk and provider_src.is_dir()):
        pytest.skip(
            "plugin_sdk/data_sdk/flatbuffers_sdk/provider src not found beside "
            "this checkout -- cannot attempt a real compile"
        )

    # Vendored flatbuffers headers: prefer an installed ROCm's, since this
    # repo does not vendor flatbuffers itself.
    fb_vendor = None
    for candidate in (Path("/opt/rocm/include"),):
        if (candidate / "flatbuffers" / "array.h").is_file():
            fb_vendor = candidate
            break
    if fb_vendor is None:
        pytest.skip("no flatbuffers/array.h found (checked /opt/rocm/include)")

    gen_dir = Path(tempfile.mkdtemp(prefix="ingestor_gen_include_"))
    configure_targets = {
        "hipdnn_data_sdk/utilities/CacheRootDefaults.h": (
            data_sdk
            / ".."
            / "include"
            / "hipdnn_data_sdk"
            / "utilities"
            / "CacheRootDefaults.h.in",
            {"HIPDNN_CACHE_ROOT_DEFAULT": "~/.cache/hipdnn/"},
        ),
    }
    for rel, (in_path, subs) in configure_targets.items():
        in_path = in_path.resolve()
        if not in_path.is_file():
            pytest.skip(f"missing CMake template {in_path}")
        text = in_path.read_text()
        for key, value in subs.items():
            text = text.replace(f"@{key}@", value)
        out_path = gen_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)

    for name, root, version_vals in (
        (
            "hipdnn_data_sdk",
            data_sdk,
            dict(MAJOR=0, MINOR=1, PATCH=0, TWEAK="test", STRING="0.1.0.test"),
        ),
        (
            "hipdnn_flatbuffers_sdk",
            flatbuffers_sdk,
            dict(MAJOR=0, MINOR=1, PATCH=0, TWEAK="test", STRING="0.1.0.test"),
        ),
        (
            "hipdnn_plugin_sdk",
            plugin_sdk,
            dict(MAJOR=1, MINOR=0, PATCH=0, TWEAK="test", STRING="1.0.0.test"),
        ),
    ):
        in_path = (root / ".." / "version.h.in").resolve()
        if not in_path.is_file():
            pytest.skip(f"missing version template {in_path}")
        text = in_path.read_text()
        prefix = name.upper()
        for key, value in version_vals.items():
            text = text.replace(f"@{prefix}_VERSION_{key}@", str(value))
        out = gen_dir / name / "version.h"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)

    return {
        "gxx": gxx,
        "includes": [
            gen_dir,
            plugin_sdk,
            data_sdk,
            flatbuffers_sdk,
            provider_src,
            fb_vendor,
        ],
    }


def _compile(compile_env, source: str, tmp_path: Path) -> subprocess.CompletedProcess:
    src_path = tmp_path / "Native.cpp"
    src_path.write_text(source)
    cmd = [
        compile_env["gxx"],
        "-fsyntax-only",
        "-std=c++20",
        "-D__HIP_PLATFORM_AMD__",
        "-DHIPDNN_ENABLE_KERNEL_INGESTOR",
    ]
    for inc in compile_env["includes"]:
        cmd += ["-I", str(inc)]
    cmd.append(str(src_path))
    return subprocess.run(cmd, capture_output=True, text=True)


#: A minimal stand-in for the gtest surface the emitted test files use, so parsing them
#: does not pull googletest into this tool's test environment. Each expectation macro
#: must instantiate the operation real gtest performs, not merely name its operands;
#: ``TestTheGtestStandIn`` holds those properties directly.
_GTEST_STUB_HEADER = """#pragma once

struct GTestMsg
{
    template <typename T>
    GTestMsg& operator<<(const T&)
    {
        return *this;
    }
};

template <typename A, typename B>
GTestMsg gtestEqual(const A& lhs, const B& rhs)
{
    (void)static_cast<bool>(lhs == rhs);
    return GTestMsg();
}

template <typename A, typename B>
GTestMsg gtestUnequal(const A& lhs, const B& rhs)
{
    (void)static_cast<bool>(lhs != rhs);
    return GTestMsg();
}

template <typename T>
GTestMsg gtestBoolean(const T& value)
{
    (void)static_cast<bool>(value);
    return GTestMsg();
}

namespace testing
{
class Test
{
public:
    virtual ~Test() = default;

protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};
} // namespace testing

#define TEST(a, b) void a##_##b##_generated_test()
#define TEST_F(fixture, name)                \\
    struct fixture##_##name##_case : fixture \\
    {                                        \\
        void testBody();                     \\
    };                                       \\
    void fixture##_##name##_case::testBody()
#define GTEST_SKIP() GTestMsg()
#define EXPECT_TRUE(x) gtestBoolean((x))
#define EXPECT_FALSE(x) gtestBoolean((x))
#define ASSERT_TRUE(x) gtestBoolean((x))
#define ASSERT_FALSE(x) gtestBoolean((x))
#define EXPECT_EQ(a, b) gtestEqual((a), (b))
#define EXPECT_NE(a, b) gtestUnequal((a), (b))
#define ASSERT_EQ(a, b) gtestEqual((a), (b))
#define ASSERT_NE(a, b) gtestUnequal((a), (b))
"""


@pytest.fixture(scope="module")
def host_cxx():
    """A host C++ compiler and nothing else: the stand-in cases include no SDK header,
    flatbuffers or provider source, so they must not inherit ``compile_env``'s skips."""
    gxx = shutil.which("g++") or shutil.which("clang++")
    if gxx is None:
        pytest.skip("no host C++ compiler (g++/clang++) found on PATH")
    return gxx


def _parse_against_the_stub(
    gxx: str, body: str, tmp_path: Path, stem: str
) -> subprocess.CompletedProcess:
    """``-fsyntax-only`` a fragment written against the stand-in, nothing else."""
    gtest_dir = tmp_path / "stub" / "gtest"
    gtest_dir.mkdir(parents=True, exist_ok=True)
    (gtest_dir / "gtest.h").write_text(_GTEST_STUB_HEADER)
    src_path = tmp_path / f"{stem}.cpp"
    src_path.write_text("#include <string>\n#include <gtest/gtest.h>\n\n" + body)
    cmd = [
        gxx,
        "-fsyntax-only",
        "-std=c++20",
        "-I",
        str(tmp_path / "stub"),
        str(src_path),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _parse_test_stub(
    compile_env, source: str, tmp_path: Path, stem: str
) -> subprocess.CompletedProcess:
    """``-fsyntax-only`` one emitted gtest file against the stand-in above."""
    gtest_dir = tmp_path / "stub" / "gtest"
    gtest_dir.mkdir(parents=True, exist_ok=True)
    (gtest_dir / "gtest.h").write_text(_GTEST_STUB_HEADER)
    src_path = tmp_path / f"{stem}.cpp"
    src_path.write_text(source)
    cmd = [
        compile_env["gxx"],
        "-fsyntax-only",
        "-std=c++20",
        "-D__HIP_PLATFORM_AMD__",
        "-DHIPDNN_ENABLE_KERNEL_INGESTOR",
        "-I",
        str(tmp_path / "stub"),
    ]
    for inc in compile_env["includes"]:
        cmd += ["-I", str(inc)]
    cmd.append(str(src_path))
    return subprocess.run(cmd, capture_output=True, text=True)


class TestRealCompile:
    """Host-compile the emitted stub with g++, best-effort. Every case runs
    ``-fsyntax-only``, so linking, symbol registration, loading and dispatch belong to
    the provider's own tests."""

    def test_single_pack_stub_compiles(
        self, compile_env, generator, scale_add_config, tmp_path
    ):
        rendered = generator._render_template(
            "native.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        result = _compile(compile_env, rendered, tmp_path)
        assert (
            result.returncode == 0
        ), f"emitted single-pack native stub failed to compile:\n{result.stderr}"

    def test_packaged_dialect_stub_compiles(
        self, compile_env, generator, gfx950_attention_dense_config, tmp_path
    ):
        """The only config reaching the `{% if config.is_packaged %}` block: an extra
        function definition outside the pack's anonymous namespace so IngestorPacks.cpp
        can reference it."""
        config = gfx950_attention_dense_config
        assert config.is_packaged, "fixture is no longer the packaged-dialect one"
        rendered = generator._render_template(
            "native.cpp.j2", config, ids=mint_ids(config)
        )
        result = _compile(compile_env, rendered, tmp_path)
        assert (
            result.returncode == 0
        ), f"emitted packaged native stub failed to compile:\n{result.stderr}"

    def test_matcher_test_stub_parses(
        self, compile_env, generator, scale_add_config, tmp_path
    ):
        """`test_matchers.cpp.j2` ships pre-wired `GTEST_SKIP()` stubs."""
        rendered = generator._render_template(
            "test_matchers.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        result = _parse_test_stub(compile_env, rendered, tmp_path, "TestMatchers")
        assert (
            result.returncode == 0
        ), f"emitted matcher-test stub does not parse:\n{result.stderr}"

    def test_pack_census_stub_parses(
        self, compile_env, generator, scale_add_config, tmp_path
    ):
        """Unlike the two stubs an author finishes, the census is emitted complete and
        meant to run as written, so a defect in it ships inside something that looks
        finished."""
        rendered = generator._render_template(
            "test_packs.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        result = _parse_test_stub(compile_env, rendered, tmp_path, "TestPacks")
        assert (
            result.returncode == 0
        ), f"emitted pack-census test does not parse:\n{result.stderr}"

    def test_multi_pack_census_stub_parses(
        self, compile_env, generator, binary_ops_config, tmp_path
    ):
        """The census's multi-pack arm counts one graph-scoped matcher per pack, against
        the single-pack arm's zero: different code, not a constant."""
        config = binary_ops_config
        assert config.is_multi_pack, "fixture is no longer the multi-pack one"
        rendered = generator._render_template(
            "test_packs.cpp.j2", config, ids=mint_ids(config)
        )
        result = _parse_test_stub(compile_env, rendered, tmp_path, "TestPacksMulti")
        assert (
            result.returncode == 0
        ), f"emitted multi-pack census does not parse:\n{result.stderr}"

    def test_heuristic_free_stubs_parse(
        self, compile_env, generator, heuristic_free_config, tmp_path
    ):
        """Every shipped config declares `heuristic: native`, so this is the only case
        parsing the `{% else %}` arms of all three templates."""
        config = heuristic_free_config
        assert not config.engine.has_heuristic
        ids = mint_ids(config)
        native = generator._render_template("native.cpp.j2", config, ids=ids)
        result = _compile(compile_env, native, tmp_path)
        assert (
            result.returncode == 0
        ), f"heuristic-free native stub failed to compile:\n{result.stderr}"

        for template, stem in (
            ("test_matchers.cpp.j2", "TestMatchersNoHeuristic"),
            ("test_packs.cpp.j2", "TestPacksNoHeuristic"),
        ):
            rendered = generator._render_template(template, config, ids=ids)
            result = _parse_test_stub(compile_env, rendered, tmp_path, stem)
            assert (
                result.returncode == 0
            ), f"heuristic-free {template} does not parse:\n{result.stderr}"

    def test_the_census_parse_catches_a_typo_inside_an_expectation(
        self, compile_env, generator, scale_add_config, tmp_path
    ):
        """An expectation macro that discarded its operands would let this class report
        success on a file real gtest rejects."""
        rendered = generator._render_template(
            "test_packs.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        broken = rendered.replace(
            "EXPECT_EQ(loaded, _expected->packNames)",
            "EXPECT_EQ(loaded, _expected->thisMemberDoesNotExist)",
            1,
        )
        assert broken != rendered, "the census no longer carries the mutated statement"
        result = _parse_test_stub(compile_env, broken, tmp_path, "TestPacksBroken")
        assert result.returncode != 0, (
            "an undeclared member inside EXPECT_EQ parsed cleanly -- the gtest "
            "stand-in is discarding the operands it is handed"
        )

    def test_multi_pack_stub_compiles(
        self, compile_env, generator, binary_ops_config, tmp_path
    ):
        rendered = generator._render_template(
            "native.cpp.j2", binary_ops_config, ids=mint_ids(binary_ops_config)
        )
        result = _compile(compile_env, rendered, tmp_path)
        assert (
            result.returncode == 0
        ), f"emitted multi-pack native stub failed to compile:\n{result.stderr}"

    def test_compile_catches_a_real_break(
        self, compile_env, generator, scale_add_config, tmp_path
    ):
        """An actually-broken stub must fail to compile, or this class is vacuous."""
        rendered = generator._render_template(
            "native.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        broken = rendered.replace(
            "return std::nullopt;", "return this_identifier_does_not_exist;", 1
        )
        result = _compile(compile_env, broken, tmp_path)
        assert result.returncode != 0, (
            "a deliberately broken stub compiled cleanly -- the compile check "
            "is not exercising real errors"
        )


#: Declarations the stand-in cases below are written against. ``ExplicitlyBool``'s
#: conversion is explicit because that is the form gtest's own contextual conversion
#: accepts.
_STAND_IN_PREAMBLE = """
struct NoBoolConversion
{
};

struct ExplicitlyBool
{
    explicit operator bool() const { return true; }
};
"""

#: Fragments real gtest rejects. Each names the operation the stand-in has to
#: instantiate for the rejection to happen.
_REJECTED_BY_REAL_GTEST = {
    "equality between incomparable types": "EXPECT_EQ(std::string{}, 5);",
    "asserted equality between incomparable types": "ASSERT_EQ(std::string{}, 5);",
    "inequality between incomparable types": "EXPECT_NE(std::string{}, 5);",
    "asserted inequality between incomparable types": "ASSERT_NE(std::string{}, 5);",
    "truth of a value with no bool conversion": "EXPECT_TRUE(NoBoolConversion{});",
    "falsity of a value with no bool conversion": "EXPECT_FALSE(NoBoolConversion{});",
    "an undeclared identifier as an operand": "EXPECT_EQ(noSuchIdentifier, 1);",
}

#: The other side: everything the emitted files actually do must still parse, or the
#: stand-in would reject real code instead of bad code.
_ACCEPTED_BY_REAL_GTEST = """
EXPECT_EQ(std::string("a"), std::string("a"));
EXPECT_EQ(std::string("a"), "a");
EXPECT_NE(std::string("a"), std::string("b"));
ASSERT_EQ(std::string("a").size(), 1U);
ASSERT_NE(std::string("a"), std::string("b"));
EXPECT_TRUE(ExplicitlyBool{});
EXPECT_FALSE(std::string("a").empty());
ASSERT_TRUE(ExplicitlyBool{}) << "a chained message still compiles";
ASSERT_FALSE(false) << "and so does one on an ASSERT_";
GTEST_SKIP() << "as does the skip form the matcher stubs ship";
"""


class TestTheGtestStandIn:
    """``TestRealCompile``'s verdict is only as strong as the operations this header
    instantiates."""

    @pytest.mark.parametrize(
        "statement",
        list(_REJECTED_BY_REAL_GTEST.values()),
        ids=list(_REJECTED_BY_REAL_GTEST),
    )
    def test_the_stand_in_rejects_what_real_gtest_rejects(
        self, host_cxx, statement, tmp_path
    ):
        body = f"{_STAND_IN_PREAMBLE}\nvoid subject()\n{{\n    {statement}\n}}\n"
        result = _parse_against_the_stub(host_cxx, body, tmp_path, "StandInRejects")
        assert result.returncode != 0, (
            f"{statement} parsed cleanly against the gtest stand-in, so an emitted "
            "file carrying it would pass this suite and fail the provider's build"
        )

    def test_the_stand_in_accepts_what_the_emitted_files_write(
        self, host_cxx, tmp_path
    ):
        """The positive control: the rejections above must be discriminating."""
        statements = "\n    ".join(_ACCEPTED_BY_REAL_GTEST.strip().splitlines())
        body = f"{_STAND_IN_PREAMBLE}\nvoid subject()\n{{\n    {statements}\n}}\n"
        result = _parse_against_the_stub(host_cxx, body, tmp_path, "StandInAccepts")
        assert result.returncode == 0, (
            "well-typed expectations were rejected by the gtest stand-in:\n"
            f"{result.stderr}"
        )


#: Values a generated C++ string literal must survive, keyed by what each attacks.
#: ``cpp_escape`` is the only thing between these and the emitted census: no pattern
#: charset-validates kernel names and ``engine.sdk_version`` is free-form. ASCII only,
#: since each case is checked byte-for-byte against its own code points below.
_HOSTILE_LITERAL_VALUES = {
    "quote-ends-the-literal": 'scale_add."f32"',
    "backslash-changes-the-string": "scale_add" + chr(92) + "path",
    "trailing-backslash-continues-the-line": "ends_with" + chr(92),
    "newline-runs-past-the-line": "line\nnext",
    "carriage-return-runs-past-the-line": "line\rnext",
    "maximal-munch-hex": "\x1f32",
    "tab-and-named-escapes": "tab\there",
    "delete-and-low-controls": "del\x7fsoh\x01end",
}

#: The subset whose RAW form a C++ front end refuses, each key naming the failure its
#: escaping prevents. Tab, DEL and the low control characters are LEGAL raw and denote
#: the same bytes, as ``test_a_raw_control_character_is_accepted`` pins; escaping them
#: keeps the census readable rather than making it compile.
_MUST_BE_ESCAPED_TO_COMPILE = (
    "quote-ends-the-literal",
    "backslash-changes-the-string",
    "trailing-backslash-continues-the-line",
    "newline-runs-past-the-line",
    "carriage-return-runs-past-the-line",
)


def _byte_assertions(escaped: str, value: str) -> str:
    """C++ that fails to compile unless ``escaped`` denotes exactly ``value``. Compares
    against integer code points, since a string literal would need escaping too and
    would check ``cpp_escape`` against itself."""
    lines = [
        f'constexpr std::string_view kSubject = "{escaped}";',
        f"static_assert(kSubject.size() == {len(value)}, "
        '"the escaped literal denotes a different number of characters");',
    ]
    lines += [
        f"static_assert(kSubject[{index}] == {ord(character)}, "
        f'"character {index} of the escaped literal is not the authored one");'
        for index, character in enumerate(value)
    ]
    return "\n".join(lines)


class TestEscapedLiteralsDenoteTheAuthoredBytes:
    """A real compiler's verdict on ``cpp_escape``, not a string comparison.

    The value cases in ``test_generator.py`` compare a render against an escaping
    written by the same hand. These hand the escaped text to a C++ front end instead:
    the backslash and control-character cases parse either way and differ only in what
    the literal MEANS, which ``static_assert`` settles at compile time.
    """

    @pytest.mark.parametrize(
        "value", _HOSTILE_LITERAL_VALUES.values(), ids=_HOSTILE_LITERAL_VALUES.keys()
    )
    def test_the_escaped_value_parses_and_means_what_was_authored(
        self, host_cxx, value, tmp_path
    ):
        assert value.isascii(), "byte-for-byte comparison below assumes ASCII"
        body = "#include <string_view>\n\n" + _byte_assertions(cpp_escape(value), value)
        result = _parse_against_the_stub(host_cxx, body, tmp_path, "EscapedLiteral")
        assert result.returncode == 0, (
            f"the escaping of {value!r} does not denote the authored string:\n"
            f"{result.stderr}"
        )

    @pytest.mark.parametrize("case", _MUST_BE_ESCAPED_TO_COMPILE)
    def test_the_same_value_unescaped_is_rejected(self, host_cxx, case, tmp_path):
        """Without it, a ``cpp_escape`` returning its input unchanged would pass
        whichever cases need no escaping; the lone backslash parses and is caught only
        by the byte assertions."""
        value = _HOSTILE_LITERAL_VALUES[case]
        body = "#include <string_view>\n\n" + _byte_assertions(value, value)
        result = _parse_against_the_stub(host_cxx, body, tmp_path, "RawLiteral")
        assert result.returncode != 0, (
            f"the raw form of {value!r} compiled and denoted the authored bytes, so "
            "the escaped case proves nothing about the escaping"
        )

    def test_a_raw_control_character_is_accepted(self, host_cxx, tmp_path):
        """Why the control above covers only part of the table: an assumed rejection
        would make the escape-or-fail set look stronger than it is."""
        value = _HOSTILE_LITERAL_VALUES["delete-and-low-controls"]
        body = "#include <string_view>\n\n" + _byte_assertions(value, value)
        result = _parse_against_the_stub(host_cxx, body, tmp_path, "RawControl")
        assert result.returncode == 0, (
            "a raw control character was refused inside a string literal, so it "
            f"belongs in the escape-or-fail set:\n{result.stderr}"
        )
