# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The generated native stub -- the single most important artifact this tool
produces.

``packs/<Name>Native.cpp`` is what an agent (or a human) fills in to make an
engine serve real graphs (see RUNBOOK.md). These tests observe a real compiler's
verdict on the emitted C++.

``TestRealCompile`` -- best-effort host compile of the emitted native stub and
of the emitted matcher-test stub, guarded by a module fixture that skips (not
fails) when the plugin SDK's CMake-baked version/config headers are
unavailable, so the run reports honestly whether it happened. The compile is
``-fsyntax-only``: it proves the emitted translation unit parses and
type-checks against the real SDK headers, and nothing more. Linking, symbol
registration, loader pre-flight, inventory and runtime behaviour are owned by
the provider's compiled registration/load/inventory tests and by the native
execution matrix, not by anything in this file.
"""

import subprocess
import shutil
import tempfile
from pathlib import Path

import pytest

from codegen.generator import mint_ids


def _find_include_dir(name: str) -> Path | None:
    # tests -> IngestorGenerator -> tools -> hipdnn
    hipdnn_root = Path(__file__).resolve().parents[3]
    candidate = hipdnn_root / name / "include"
    return candidate if candidate.is_dir() else None


@pytest.fixture(scope="module")
def compile_env():
    """Best-effort host-compile environment for the emitted native stub.

    The plugin SDK's ``version.h``/``CacheRootDefaults.h`` headers are
    CMake-configured (``.h.in`` templates), so a from-scratch compile needs
    stand-ins for them. Generates minimal ones from the real ``.in``
    templates (substituting placeholder values -- the macros' actual values
    are irrelevant to whether the emitted stub parses) rather than skipping
    outright, so the check actually runs rather than silently reporting
    nothing. Skips (never fails) when a prerequisite -- compiler, SDK
    sources beside this checkout, or a vendored flatbuffers -- is absent.
    """
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
    # Stand-in CMake-configured headers, generated from the real .in
    # templates with placeholder substitutions -- their content is
    # irrelevant to whether the emitted native stub parses; only their
    # presence (and the macros they define) matters.
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


class TestRealCompile:
    """Host-compile the emitted stub with g++, best-effort.

    The plugin SDK's ``version.h``/``CacheRootDefaults.h`` headers are
    CMake-configured (``.h.in`` templates), so a from-scratch compile needs
    stand-ins for them. The fixture generates minimal ones from the real
    ``.in`` templates (substituting placeholder values -- the macros' actual
    values are irrelevant to whether the emitted stub parses) rather than
    skipping outright, so the check actually runs rather than silently
    reporting nothing.

    Proof boundary: every case here runs ``-fsyntax-only``. A pass means the
    emitted translation unit parses and type-checks against the real headers;
    it is not evidence that the engine links, registers its symbols, loads, or
    dispatches.
    """

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
        """The packaged branch emits code the other two configs never exercise.

        Both compile tests above use direct_load configs, so the
        `{% if config.is_packaged %}` block -- an entire extra function
        definition, placed outside the pack's anonymous namespace so the
        IngestorPacks.cpp row can reference it -- was emitted by nothing that
        compiles. It was added to fix a link error; emitting a *syntax* error in
        its place would have passed every test.
        """
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
        """The OTHER emitted C++ file. Nothing compiled it.

        `test_matchers.cpp.j2` gained three pre-wired `GTEST_SKIP()` stubs this
        session and no test has ever fed it to a compiler -- the same gap the
        packaged native stub had, one template over. A malformed stub would ship
        and first fail inside the provider's build, days later.

        gtest is not a dependency of this tool, so this parses against a minimal
        stand-in for the handful of macros the emitted file uses. That is enough
        to catch the realistic defect (an unbalanced stub, a bad string
        concatenation) without pulling googletest into the generator's test env.
        """
        gtest_dir = tmp_path / "stub/gtest"
        gtest_dir.mkdir(parents=True)
        (gtest_dir / "gtest.h").write_text(
            "#pragma once\n"
            "struct GTestMsg { template <typename T>\n"
            "    GTestMsg& operator<<(const T&) { return *this; } };\n"
            "#define TEST(a, b) void a##_##b##_generated_test()\n"
            "#define GTEST_SKIP() GTestMsg()\n"
            "#define EXPECT_TRUE(x) (void)(x)\n"
            "#define EXPECT_FALSE(x) (void)(x)\n"
            "#define EXPECT_NE(a, b) (void)0\n"
            "#define EXPECT_EQ(a, b) (void)0\n"
        )
        rendered = generator._render_template(
            "test_matchers.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        src = tmp_path / "TestMatchers.cpp"
        src.write_text(rendered)
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
        cmd.append(str(src))
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert (
            result.returncode == 0
        ), f"emitted matcher-test stub does not parse:\n{result.stderr}"

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
        """Sanity check on the check itself: an actually-broken stub must fail
        to compile, or this whole class is silently vacuous."""
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
