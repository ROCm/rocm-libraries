################################################################################
# Characterization tests for Tensile.TensileCreateLibrary.ParseArguments
#
# ADD-ONLY: pins the argv -> arguments-dict mapping of parseArguments.
#
# Pinned quirk: parseArguments(input) IGNORES its `input` parameter and calls
# argparse .parse_args() with no list, i.e. it parses sys.argv. Tests set
# sys.argv accordingly.
################################################################################
import importlib
import os
import sys

import pytest

pytestmark = pytest.mark.unit

PA = importlib.import_module("Tensile.TensileCreateLibrary.ParseArguments")
coVersionMap = importlib.import_module("Tensile.Common").coVersionMap

BASE = ["prog", "/logic", "/out", "HSA"]


def _run(monkeypatch, extra=None):
    argv = list(BASE) + (extra or [])
    monkeypatch.setattr(sys, "argv", argv)
    return PA.parseArguments(["this", "is", "ignored"])


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_defaults(monkeypatch):
    a = _run(monkeypatch)
    assert a["RuntimeLanguage"] == "HSA"
    assert a["LogicPath"] == "/logic"
    assert a["OutputPath"] == "/out"
    assert a["CodeObjectVersion"] == coVersionMap["4"]
    assert a["Architecture"] == "all"
    assert a["LazyLibraryLoading"] is True
    assert a["EnableMarker"] is False
    assert a["DisableAsmComments"] is False
    assert a["LogicFormat"] == "yaml"
    assert a["LibraryFormat"] == "msgpack"
    assert a["CpuThreads"] == -1
    assert a["PrintLevel"] == 1
    assert a["AsmDebug"] is False
    assert a["BuildIdKind"] == "sha1"
    assert a["KeepBuildTmp"] is False
    assert a["AsanBuild"] is False
    assert a["UseCompression"] is True  # NoCompress default False
    assert a["LogicFilter"] == "*"
    assert a["Experimental"] is False
    assert a["GenSolTable"] is True


def test_quirk_input_param_ignored(monkeypatch):
    # Even with a fully-formed `input` list, sys.argv is what gets parsed.
    monkeypatch.setattr(sys, "argv", ["prog", "/real", "/realout", "HIP"])
    a = PA.parseArguments(["prog", "/fake", "/fakeout", "OCL"])
    assert a["RuntimeLanguage"] == "HIP"
    assert a["LogicPath"] == "/real"


# ---------------------------------------------------------------------------
# store_true flags
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "flag,key",
    [
        ("--experimental", "Experimental"),
        ("--enable-marker", "EnableMarker"),
        ("--asm-debug", "AsmDebug"),
        ("--address-sanitizer", "AsanBuild"),
        ("--keep-build-tmp", "KeepBuildTmp"),
        ("--disable-asm-comments", "DisableAsmComments"),
    ],
)
def test_store_true_flags(monkeypatch, flag, key):
    a = _run(monkeypatch, [flag])
    assert a[key] is True


def test_no_compress_inverts_use_compression(monkeypatch):
    a = _run(monkeypatch, ["--no-compress"])
    assert a["UseCompression"] is False


# ---------------------------------------------------------------------------
# store_false flags
# ---------------------------------------------------------------------------
def test_no_lazy_library_loading(monkeypatch):
    a = _run(monkeypatch, ["--no-lazy-library-loading"])
    assert a["LazyLibraryLoading"] is False


def test_no_generate_solution_table(monkeypatch):
    a = _run(monkeypatch, ["--no-generate-solution-table"])
    assert a["GenSolTable"] is False


# ---------------------------------------------------------------------------
# value-carrying options
# ---------------------------------------------------------------------------
def test_code_object_version_mapping(monkeypatch):
    a = _run(monkeypatch, ["--code-object-version", "V5"])
    assert a["CodeObjectVersion"] == coVersionMap["V5"]


def test_scalar_options(monkeypatch):
    a = _run(
        monkeypatch,
        [
            "--architecture", "gfx942",
            "--jobs", "8",
            "--verbose", "3",
            "--logic-format", "json",
            "--library-format", "yaml",
            "--build-id", "md5",
            "--logic-filter", "gfx942/Equality/*",
            "--cxx-compiler", "myclang++",
            "--c-compiler", "myclang",
            "--offload-bundler", "mybundler",
            "--assembler", "myasm",
        ],
    )
    assert a["Architecture"] == "gfx942"
    assert a["CpuThreads"] == 8
    assert a["PrintLevel"] == 3
    assert a["LogicFormat"] == "json"
    assert a["LibraryFormat"] == "yaml"
    assert a["BuildIdKind"] == "md5"
    assert a["LogicFilter"] == "gfx942/Equality/*"
    assert a["CxxCompiler"] == "myclang++"
    assert a["CCompiler"] == "myclang"
    assert a["OffloadBundler"] == "mybundler"
    assert a["Assembler"] == "myasm"


def test_cmake_cxx_compiler_sets_env(monkeypatch):
    # delenv first so monkeypatch restores absence on teardown (the code mutates
    # os.environ directly, which monkeypatch would not otherwise track).
    monkeypatch.delenv("CMAKE_CXX_COMPILER", raising=False)
    a = _run(monkeypatch, ["--cmake-cxx-compiler", "/usr/bin/g++"])
    assert os.environ["CMAKE_CXX_COMPILER"] == "/usr/bin/g++"


# ---------------------------------------------------------------------------
# argparse validation
# ---------------------------------------------------------------------------
def test_invalid_runtime_language_exits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "/logic", "/out", "CUDA"])
    with pytest.raises(SystemExit):
        PA.parseArguments()


def test_invalid_code_object_version_exits(monkeypatch):
    with pytest.raises(SystemExit):
        _run(monkeypatch, ["--code-object-version", "9"])
