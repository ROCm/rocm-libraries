################################################################################
# Characterization tests for Tensile.Toolchain.Component
#
# ADD-ONLY: pins current behavior of the ROCm toolchain wrapper classes
# (Component / Assembler / Compiler / Bundler / Linker) and the module helpers
# (_invoke, _getVersion, get_rocm_version).
#
# Real subprocess work is avoided by stubbing module-level `_invoke` (the single
# subprocess chokepoint for the __call__ methods) and `_getVersion` (the version
# probe run in every __init__). Argument *construction* is what we pin — the
# exact argv each wrapper hands to the toolchain.
################################################################################
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

C = importlib.import_module("Tensile.Toolchain.Component")
SemanticVersion = importlib.import_module("Tensile.Common").SemanticVersion


@pytest.fixture
def fixed_version(monkeypatch):
    """Make every Component.__init__ version probe deterministic & subprocess-free."""
    monkeypatch.setattr(C, "_getVersion", lambda *a, **k: SemanticVersion(1, 2, 3))
    return SemanticVersion(1, 2, 3)


@pytest.fixture
def captured_invoke(monkeypatch):
    """Replace _invoke with a recorder; returns the list of arg-lists captured."""
    calls = []

    def _rec(args, desc=""):
        calls.append(list(args))
        return b"stubbed"

    monkeypatch.setattr(C, "_invoke", _rec)
    return calls


# ---------------------------------------------------------------------------
# _invoke
# ---------------------------------------------------------------------------
def test_invoke_success_returns_output():
    out = C._invoke(["echo", "hello"], "echo test")
    assert out.strip() == b"hello"


def test_invoke_failure_raises_runtimeerror():
    with pytest.raises(RuntimeError, match="Error with"):
        C._invoke(["false"], "failing command")


# ---------------------------------------------------------------------------
# _getVersion
# ---------------------------------------------------------------------------
def test_get_version_match(monkeypatch):
    monkeypatch.setattr(C, "validateToolchain", lambda x: x)

    class _R:
        stdout = b"clang version 17.0.5-rc1 extra"

    monkeypatch.setattr(C, "run", lambda *a, **k: _R())
    v = C._getVersion("amdclang++", "--version", r"version\s+([\d.]+)")
    assert v == SemanticVersion(17, 0, 5)


def test_get_version_no_match_raises(monkeypatch):
    monkeypatch.setattr(C, "validateToolchain", lambda x: x)

    class _R:
        stdout = b"no version here"

    monkeypatch.setattr(C, "run", lambda *a, **k: _R())
    with pytest.raises(RuntimeError, match="Failed to get version"):
        C._getVersion("amdclang++", "--version", r"version\s+([\d.]+)")


def test_get_rocm_version_uses_hipconfig(monkeypatch):
    seen = {}

    def _fake(exe, flag, regex):
        seen["exe"], seen["flag"] = exe, flag
        return SemanticVersion(6, 4, 0)

    monkeypatch.setattr(C, "_getVersion", _fake)
    assert C.get_rocm_version() == SemanticVersion(6, 4, 0)
    assert seen["flag"] == "--version"
    assert seen["exe"] == C.ToolchainDefaults.HIP_CONFIG


# ---------------------------------------------------------------------------
# Component base
# ---------------------------------------------------------------------------
def test_component_properties_and_str(fixed_version):
    comp = C.Component(Path("/x/amdclang++"))
    assert comp.path == Path("/x/amdclang++")
    assert comp.version == SemanticVersion(1, 2, 3)
    assert comp.rocm_version == C.Component._rocm_version
    s = str(comp)
    assert "Component path: /x/amdclang++" in s
    assert "version: 1.2.3" in s
    assert "ROCm" in s


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------
def test_assembler_default_args_and_code_object_version(fixed_version):
    asm = C.Assembler(Path("/x/amdclang++"), co_version="5", debug=True)
    assert asm.code_object_version == "5"
    assert "-mcode-object-version=5" in asm._default_args
    assert "-g" in asm._default_args  # debug=True
    assert "assembler" in asm._default_args


def test_assembler_call_wavefront64_no_true16(fixed_version, captured_invoke):
    asm = C.Assembler(Path("/x/amdclang++"), co_version="5")
    asm("gfx942", 64, "src.s", "out.o")
    args = captured_invoke[0]
    assert "-mcpu=gfx942" in args
    assert "-mwavefrontsize64" in args
    assert "-Xclangas" not in args  # gfx942 not in true16 set
    assert args[-3:] == ["src.s", "-o", "out.o"]


def test_assembler_call_true16_and_no_wavefront64(fixed_version, captured_invoke):
    asm = C.Assembler(Path("/x/amdclang++"), co_version="5")
    asm("gfx1100", 32, "src.s", "out.o")
    args = captured_invoke[0]
    assert "-mcpu=gfx1100" in args
    assert "-mno-wavefrontsize64" in args
    assert "+real-true16" in args


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------
def test_compiler_default_args_plain(fixed_version):
    comp = C.Compiler(Path("/x/amdclang++"), build_id_kind="sha1")
    assert "--build-id=sha1" in comp.default_args
    assert "-std=c++17" in comp.default_args
    assert "-fsanitize=address" not in comp.default_args
    assert "--save-temps" not in comp.default_args


def test_compiler_asan_and_save_temps(fixed_version):
    comp = C.Compiler(
        Path("/x/amdclang++"), build_id_kind="sha1", asan_build=True, save_temps=True
    )
    assert "-fsanitize=address" in comp.default_args
    assert "-shared-libasan" in comp.default_args
    assert "--save-temps" in comp.default_args


def test_compiler_windows_branch(fixed_version, monkeypatch):
    monkeypatch.setattr(C, "os_name", "nt")
    monkeypatch.setenv("ROCM_PATH", "C:/rocm")
    comp = C.Compiler(Path("/x/clang++.exe"), build_id_kind="sha1")
    assert "-fms-extensions" in comp.default_args
    assert "-fPIC" in comp.default_args
    assert "--rocm-path=C:/rocm" in comp.default_args


def test_compiler_call_builds_arch_flags(fixed_version, captured_invoke):
    comp = C.Compiler(Path("/x/amdclang++"), build_id_kind="sha1")
    comp("/inc", ["gfx942", "gfx90a"], "k.cpp", "k.o")
    args = captured_invoke[0]
    assert "--offload-arch=gfx942" in args
    assert "--offload-arch=gfx90a" in args
    assert args[-4:] == ["k.cpp", "-c", "-o", "k.o"]
    assert "/inc" in args


# ---------------------------------------------------------------------------
# Bundler
# ---------------------------------------------------------------------------
def test_bundler_targets(fixed_version, monkeypatch):
    monkeypatch.setattr(C, "_invoke", lambda args, desc="": b"line1\nline2\n")
    b = C.Bundler(Path("/x/clang-offload-bundler"))
    assert b.targets("obj.o") == ["line1", "line2", ""]


def test_bundler_compress(fixed_version, captured_invoke):
    b = C.Bundler(Path("/x/clang-offload-bundler"))
    b.compress("src.co", "dst.co", "gfx942")
    args = captured_invoke[0]
    assert "--compress" in args
    assert any("gfx942" in str(a) for a in args)
    assert f"--output=dst.co" in args


def test_bundler_unbundle_call(fixed_version, captured_invoke):
    b = C.Bundler(Path("/x/clang-offload-bundler"))
    b("the-target", "src.o", "dst.co")
    args = captured_invoke[0]
    assert "--unbundle" in args
    assert "--targets=the-target" in args
    assert "--input=src.o" in args


# ---------------------------------------------------------------------------
# Linker
# ---------------------------------------------------------------------------
def test_linker_default_args(fixed_version):
    lk = C.Linker("/x/amdclang++", build_id_kind="sha1")
    assert "--target=amdgcn-amd-amdhsa" in lk.default_args
    assert "--build-id=sha1" in lk.default_args


def test_linker_use_response_file_posix_short_false(fixed_version):
    lk = C.Linker("/x/amdclang++", build_id_kind="sha1")
    assert lk._use_response_file(["a", "b"]) is False


def test_linker_call_short_no_response_file(fixed_version, captured_invoke):
    lk = C.Linker("/x/amdclang++", build_id_kind="sha1")
    lk(["a.o", "b.o"], "out.co")
    args = captured_invoke[0]
    assert "a.o" in args and "b.o" in args
    assert args[-2:] == ["-o", "out.co"]
    assert "@clang_args.txt" not in args


def test_linker_call_long_uses_response_file(fixed_version, captured_invoke, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(C.Linker, "_use_response_file", lambda self, args: True)
    lk = C.Linker("/x/amdclang++", build_id_kind="sha1")
    lk(["a.o", "b.o"], "out.co")
    args = captured_invoke[0]
    assert "@clang_args.txt" in args
    assert (tmp_path / "clang_args.txt").read_text() == "a.o b.o"


def test_linker_response_file_args_windows(fixed_version, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(C, "os_name", "nt")
    lk = C.Linker("/x/amdclang++", build_id_kind="sha1")
    out = lk._response_file_args(["a\\b.o", "c.o"], "out.co")
    assert "@clang_args.txt" in out
    # backslashes are doubled on Windows
    assert (tmp_path / "clang_args.txt").read_text() == "a\\\\b.o c.o"


def test_linker_use_response_file_windows_true(fixed_version, monkeypatch):
    monkeypatch.setattr(C, "os_name", "nt")
    lk = C.Linker("/x/amdclang++", build_id_kind="sha1")
    assert lk._use_response_file(["a"]) is True
