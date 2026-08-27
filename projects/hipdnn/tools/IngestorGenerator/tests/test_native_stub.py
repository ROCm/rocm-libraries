# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The generated native stub -- the single most important artifact this tool
produces, and until now the least tested.

``packs/<Name>Native.cpp`` is what an agent (or a human) fills in to make an
engine serve real graphs (RUNBOOK.md Step 6, "THIS IS THE WORK"). Existing
tests cover two isolated substring traps
(``test_generator.py::TestRequiredTrapAssertions``) but nothing asserts the
file's overall shape: that every hook is genuinely a placeholder (none
silently emitted as a working body that looks finished but isn't), that the
symbol constants match what the same run's descriptors name, that the
registration block wires the right functions to the right symbols, and that
the C++ is structurally sound enough to compile.

Four classes, one per contract:

- ``TestEveryHookIsAPlaceholder`` -- a hook whose body silently does real
  work (returns something other than the documented placeholder value) would
  look finished and never get filled in; RUNBOOK.md Step 6's own gate
  (``grep -c "FILL THIS OUT"`` == 0) only fires once a human/agent starts
  editing, so nothing currently catches a *template* regression that stops
  emitting a TODO for a given hook.
- ``TestSymbolsMatchDescriptors`` -- the generator emits both the native
  symbol constants and the descriptor JSON from the same ``config``/``ids``,
  but nothing asserts they actually agree; a change to one without the other
  would silently break the loader's symbol pre-flight (native-pack.md:261
  "Nothing in the type system ties the JSON strings to the C++ constants").
- ``TestStructuralSoundness`` -- balanced braces, every declared hook
  present, and the registration block references the config's own
  register-function name and dispatch handler.
- ``TestRealCompile`` -- best-effort host compile with g++, guarded by a
  session fixture that skips (not fails) when the plugin SDK's CMake-baked
  version/config headers are unavailable, and reports honestly whether it ran.
"""

import re
import subprocess
import shutil
import tempfile
from pathlib import Path

import pytest

from codegen.generator import mint_ids

# ---------------------------------------------------------------------------
# Placeholder-value contract per hook. Keyed on a regex that isolates the
# hook's own body (never spilling into a sibling function), and the value the
# template is documented to emit while unfilled. A hook silently returning
# anything else has stopped being a stub.
# ---------------------------------------------------------------------------
_HOOK_BODY_PATTERNS = {
    "graph_match": (
        r"GraphMatches\(const MatchContext& /\*context\*/\)\n\{\n(.*?)\n\}",
        "return std::nullopt;",
    ),
    "kernel_match": (
        r"bool kernelMatches\(.*?\)\n\{\n(.*?)\n\}",
        "return false;",
    ),
    "score": (
        r"double scoreKernel\(.*?\)\n\{\n(.*?)\n\}",
        "return 0.0;",
    ),
    "dispatch_prepare": (
        r"std::unique_ptr<PreparedDispatch> prepare\(.*?\n    \{\n(.*?)\n    \}",
        "return std::make_unique<Prepared",
    ),
    "dispatch_launch": (
        r"void launch\(.*?\n    \{\n(.*?)\n    \}",
        "",  # launch's stub body is TODO-only, no return value
    ),
}


def _hook_body(rendered: str, key: str) -> str:
    pattern, _ = _HOOK_BODY_PATTERNS[key]
    match = re.search(pattern, rendered, re.S)
    assert match, f"could not isolate the '{key}' hook body in rendered output"
    return match.group(1)


def _hook_statement(body: str) -> str:
    """The hook body with '//'-only comment lines dropped, so a check can
    require the REMAINING code be EXACTLY the placeholder statement -- not
    merely end with it. `body.strip().endswith(...)` alone would still pass
    if a template regression inserted real logic (a branch, a helper call)
    ahead of the placeholder `return`; this is what actually distinguishes a
    stub from a hook that quietly started doing work."""
    kept = [
        line
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("//")
    ]
    return "\n".join(kept).strip()


class TestEveryHookIsAPlaceholder:
    """Every hook body is a TODO placeholder -- none silently emitted as a
    working body. A hook that renders real logic instead of a `TODO` marker
    would parse, validate, and enumerate cleanly while quietly no longer
    being the stub RUNBOOK.md Step 6 tells an agent to fill in."""

    @pytest.mark.parametrize("key", sorted(_HOOK_BODY_PATTERNS))
    def test_hook_body_carries_a_todo_marker(self, generator, scale_add_config, key):
        rendered = generator._render_template(
            "native.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        body = _hook_body(rendered, key)
        assert "TODO - FILL THIS OUT" in body or "TODO" in body, (
            f"'{key}' hook body carries no TODO marker -- it may have silently "
            f"become a working implementation:\n{body}"
        )

    def test_graph_match_returns_the_documented_nullopt(
        self, generator, scale_add_config
    ):
        rendered = generator._render_template(
            "native.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        body = _hook_body(rendered, "graph_match")
        assert _hook_statement(body) == "return std::nullopt;", (
            "graph_match's stub must be EXACTLY 'return std::nullopt;' "
            f"(declining every graph), no other code, until filled in; got:\n{body}"
        )

    def test_kernel_match_returns_the_documented_false(
        self, generator, scale_add_config
    ):
        rendered = generator._render_template(
            "native.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        body = _hook_body(rendered, "kernel_match")
        assert _hook_statement(body) == "return false;", (
            "kernel_match's stub must be EXACTLY 'return false;', no other "
            f"code, until filled in; got:\n{body}"
        )

    def test_score_returns_the_documented_zero(self, generator, scale_add_config):
        rendered = generator._render_template(
            "native.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        body = _hook_body(rendered, "score")
        assert _hook_statement(body) == "return 0.0;", (
            "score's stub must be EXACTLY 'return 0.0;', no other code, "
            f"until filled in; got:\n{body}"
        )

    def test_operation_matcher_is_a_placeholder_multi_pack(
        self, generator, binary_ops_config
    ):
        """Multi-pack engines emit one operationMatches() per pack -- also a
        stub, and not covered by the single-pack fixture above."""
        rendered = generator._render_template(
            "native.cpp.j2", binary_ops_config, ids=mint_ids(binary_ops_config)
        )
        matches = re.findall(
            r"OperationMatches\(const MatchContext& /\*context\*/, "
            r"const BoundTokens& /\*bound\*/\)\n\{\n(.*?)\n\}",
            rendered,
            re.S,
        )
        assert len(matches) == len(binary_ops_config.packs), (
            "expected one OperationMatches stub per pack, "
            f"found {len(matches)} of {len(binary_ops_config.packs)}"
        )
        for body in matches:
            assert _hook_statement(body) == "return false;", (
                "per-pack operation matcher stub must be EXACTLY "
                f"'return false;', no other code; got:\n{body}"
            )


class TestSymbolsMatchDescriptors:
    """The symbol constants the native stub declares must be exactly the
    symbols the SAME run's descriptors name -- the loader's symbol
    pre-flight is the only thing that would otherwise catch a mismatch
    (native-pack.md: 'Nothing in the type system ties the JSON strings to
    the C++ constants')."""

    def test_single_pack_symbols_match_descriptor_json(
        self, generator, scale_add_config, tmp_path
    ):
        written = generator.render(scale_add_config, tmp_path)
        native_rel = next(w for w in written if w.endswith("Native.cpp"))
        native_text = (tmp_path / native_rel).read_text()

        # Pull every constexpr std::string_view SYMBOL = "..."; constant.
        declared = dict(
            re.findall(r'constexpr std::string_view (\w+) = "([^"]+)";', native_text)
        )
        assert declared, "no symbol constants found -- the scan is broken, not the file"

        ued_rel = next(w for w in written if w.endswith(".ued.json"))
        udd_rel = next(w for w in written if w.endswith(".udd.json"))
        umd_rel = next(
            w for w in written if w.endswith("kernel_dtype_matches_graph.umd.json")
        )
        import json

        ued = json.loads((tmp_path / ued_rel).read_text())
        udd = json.loads((tmp_path / udd_rel).read_text())
        umd = json.loads((tmp_path / umd_rel).read_text())

        assert declared["GRAPH_MATCHER_SYMBOL"] == ued["graph_match"]["native"]
        assert declared["DISPATCH_SYMBOL"] == udd["dispatch_symbol"]
        assert declared["KERNEL_MATCHER_SYMBOL"] == umd["match_symbol"]
        assert declared["SCORE_SYMBOL"] == scale_add_config.score_symbol

    def test_multi_pack_operation_symbols_match_per_pack_umds(
        self, generator, binary_ops_config, tmp_path
    ):
        written = generator.render(binary_ops_config, tmp_path)
        native_rel = next(w for w in written if w.endswith("Native.cpp"))
        native_text = (tmp_path / native_rel).read_text()
        declared = dict(
            re.findall(r'constexpr std::string_view (\w+) = "([^"]+)";', native_text)
        )

        import json

        for pack in binary_ops_config.packs:
            umd_rel = next(
                w
                for w in written
                if w.endswith(f"operation_is_{pack.discriminator}.umd.json")
            )
            umd = json.loads((tmp_path / umd_rel).read_text())
            const_name = f"{pack.discriminator.upper()}_MATCHER_SYMBOL"
            assert const_name in declared, f"native stub never declares {const_name}"
            assert declared[const_name] == umd["match_symbol"]

    def test_registration_wires_every_declared_symbol(
        self, generator, scale_add_config
    ):
        """Every constant declared must appear as a scope.add(...) argument in
        register<Name>Symbols() -- a declared-but-unregistered symbol is dead
        and a registered-but-undeclared one would not compile."""
        rendered = generator._render_template(
            "native.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        declared = {
            name
            for name in re.findall(r"constexpr std::string_view (\w+) =", rendered)
            if name.endswith("SYMBOL")
        }
        reg_match = re.search(
            r"void register\w+Symbols\(SymbolScope<Handle>& scope\)\n\{\n(.*?)\n\}",
            rendered,
            re.S,
        )
        assert reg_match, "registration function not found"
        reg_body = reg_match.group(1)
        registered = set(re.findall(r"scope\.add\(std::string\((\w+)\)", reg_body))
        assert declared == registered, (
            f"declared symbols {declared} and registered symbols {registered} "
            "disagree -- a symbol constant with no scope.add() is dead, and "
            "vice versa would not compile"
        )


class TestStructuralSoundness:
    """Minimum mechanical soundness: balanced braces, every declared hook
    present in the emitted text, and the registration block references the
    config's own pack/dispatch names -- not another engine's, copy-pasted."""

    def test_braces_are_balanced(self, generator, scale_add_config):
        rendered = generator._render_template(
            "native.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        assert rendered.count("{") == rendered.count("}"), (
            f"unbalanced braces: {rendered.count('{')} opens, "
            f"{rendered.count('}')} closes"
        )

    def test_every_declared_hook_present(self, generator, scale_add_config):
        rendered = generator._render_template(
            "native.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        for needle in (
            "GraphMatches(const MatchContext&",
            "bool kernelMatches(",
            "double scoreKernel(",
            "size_t workspaceBytes(",
            "std::unique_ptr<PreparedDispatch> prepare(",
            "void launch(",
        ):
            assert needle in rendered, f"hook signature '{needle}' missing"

    def test_registration_references_this_configs_dispatch_handler(
        self, generator, scale_add_config
    ):
        rendered = generator._render_template(
            "native.cpp.j2", scale_add_config, ids=mint_ids(scale_add_config)
        )
        assert f"&{scale_add_config.engine.camel_name}DispatchHandler()" in rendered
        assert f"void register{scale_add_config.engine.pascal_name}Symbols(" in rendered

    def test_multi_pack_registration_references_every_pack_matcher(
        self, generator, binary_ops_config
    ):
        rendered = generator._render_template(
            "native.cpp.j2", binary_ops_config, ids=mint_ids(binary_ops_config)
        )
        for pack in binary_ops_config.packs:
            assert f"&{pack.discriminator}OperationMatches" in rendered


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
    stand-ins for them. This fixture generates minimal ones from the real
    ``.in`` templates (substituting placeholder values -- the macros' actual
    values are irrelevant to whether the emitted stub parses) rather than
    skipping outright, so the check actually runs rather than silently
    reporting nothing.
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
