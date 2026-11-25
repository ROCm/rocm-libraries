# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 and its steppings build code object v5; other architectures keep the requested version.

The HIP runtime may refuse code object v4 on gfx1250 and reads the version from the amdhsa.version
metadata, not the ELF header. So the assembler flag, the metadata TensileLite writes, and the extop
sources must all agree, and a v4 source must not reach a gfx1250 code object: ld.lld refuses a v4
directive next to v5 objects, and comgr drops kernels whose metadata version differs from the first
one in a code object."""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensilelite.Common import IsaVersion, SemanticVersion
from tensilelite.Common.Architectures import codeObjectVersionFor

pytestmark = pytest.mark.unit

C = importlib.import_module("tensilelite.Toolchain.Component")

TENSILE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

V5_SOURCE = """.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 2
...
.end_amdgpu_metadata
"""


@pytest.mark.parametrize("version, arch, expected", [
    ("4", "gfx1250", "5"),
    ("default", "gfx1250-strict", "5"),
    ("V4", "gfx1250:xnack-", "5"),
    ("4", IsaVersion(12, 5, 0), "5"),
    ("4", [12, 5, 0], "5"),
    ("5", "gfx1250", "5"),
    (5, "gfx1250-strict", 5),
    ("6", "gfx1250", "6"),
    ("4", "gfx942", "4"),
    ("default", "gfx950", "default"),
    ("4", IsaVersion(12, 5, 1), "4"),
])
def test_code_object_version_for(version, arch, expected):
    assert codeObjectVersionFor(version, arch) == expected


@pytest.fixture
def captured_invoke(monkeypatch):
    monkeypatch.setattr(C, "_getVersion", lambda *a, **k: SemanticVersion(1, 2, 3))
    calls = []
    monkeypatch.setattr(C, "_invoke", lambda args, desc="": calls.append(list(args)) or b"")
    return calls


def _versionFlags(args):
    return [a for a in args if a.startswith("-mcode-object-version=")]


@pytest.mark.parametrize("arch", ["gfx1250", "gfx1250-strict"])
def test_assembler_raises_gfx1250_to_v5(tmp_path, captured_invoke, arch):
    src = tmp_path / "k.s"
    src.write_text(V5_SOURCE)
    C.Assembler(Path("/x/amdclang++"), co_version="4")(arch, 32, str(src), "k.o")
    assert _versionFlags(captured_invoke[0]) == ["-mcode-object-version=4",
                                                 "-mcode-object-version=5"]


def test_assembler_keeps_the_requested_version_elsewhere(captured_invoke):
    C.Assembler(Path("/x/amdclang++"), co_version="4")("gfx942", 64, "k.s", "k.o")
    assert _versionFlags(captured_invoke[0]) == ["-mcode-object-version=4"]


@pytest.mark.parametrize("source, message", [
    (".amdhsa_code_object_version 4\n" + V5_SOURCE, ".amdhsa_code_object_version 4"),
    (V5_SOURCE.replace("  - 2", "  - 1"), r"amdhsa.version \[1, 1\]"),
    ("amdhsa.version: [ 1, 1 ]\n", r"amdhsa.version \[1, 1\]"),
], ids=["directive", "block-metadata", "flow-metadata"])
def test_assembler_refuses_a_v4_source_for_gfx1250(tmp_path, captured_invoke, source, message):
    src = tmp_path / "k.s"
    src.write_text(source)
    with pytest.raises(RuntimeError, match=message):
        C.Assembler(Path("/x/amdclang++"), co_version="4")("gfx1250", 32, str(src), "k.o")
    assert not captured_invoke


def test_assembler_leaves_v4_sources_alone_elsewhere(tmp_path, captured_invoke):
    src = tmp_path / "k.s"
    src.write_text(".amdhsa_code_object_version 4\n" + V5_SOURCE.replace("  - 2", "  - 1"))
    C.Assembler(Path("/x/amdclang++"), co_version="4")("gfx950", 64, str(src), "k.o")
    assert len(captured_invoke) == 1


@pytest.mark.parametrize("isa, assemblerVersion, logicVersion, expected", [
    ((12, 5, 0), "4", 4, "5"),
    ((12, 5, 0), "default", "4", "5"),
    ((12, 5, 0), "6", None, "6"),
    ((9, 4, 2), "4", 4, "4"),
    ((9, 5, 0), "default", None, "default"),
    ((9, 4, 2), "5", 4, "5"),
    # LibraryLogic passes the C++ compiler, which has no code object version.
    ((12, 5, 0), None, 4, "5"),
    ((9, 4, 2), None, 4, "4"),
    ((9, 4, 2), None, None, "4"),
])
def test_solution_metadata_follows_the_assembler(monkeypatch, isa, assemblerVersion, logicVersion,
                                                 expected):
    from tensilelite.SolutionStructs.Solution import Solution

    monkeypatch.setattr(Solution, "assignDerivedParameters", lambda *a, **k: None)
    monkeypatch.setattr(Solution, "_assignCustomKernelParameters", lambda *a, **k: None)
    config = {"ISA": list(isa), "ProblemType": {"DataType": "s"}}
    if logicVersion is not None:
        config["CodeObjectVersion"] = logicVersion
    assembler = SimpleNamespace(rocm_version=None)
    if assemblerVersion is not None:
        assembler.code_object_version = assemblerVersion
    isaInfo = {IsaVersion(*isa): SimpleNamespace(asmCaps={"SupportedISA": True})}
    solution = Solution(config, False, False, False, assembler, isaInfo)
    assert solution["CodeObjectVersion"] == expected


EXTOP_GENERATORS = [
    ("amax_generator", ["-t", "S", "-d", "S", "-w", "256", "-c", "4"]),
    ("layer_norm_generator", ["-w", "256", "-c", "4", "--sweep-once", "0"]),
    ("softmax_generator", ["-m", "8", "-n", "32"]),
]


def _assembleExtop(tmp_path, compiler, module, args, arch):
    """Generate and assemble an extop the way device-library/extops/CMakeLists.txt does."""
    src = tmp_path / f"{module}_{arch}.s"
    result = subprocess.run([sys.executable, "-m", f"tensilelite._extops.{module}", "-o", str(src),
                             *args, "--arch", arch, "--toolchain", str(compiler)],
                            cwd=TENSILE_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    obj = tmp_path / f"{module}_{arch}.o"
    wave = ["-mno-wavefrontsize64"] if arch.startswith("gfx12") else ["-mwavefrontsize64"]
    result = subprocess.run([str(compiler), "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
                             "-mcode-object-version=4", f"-mcpu={arch}", *wave, "-c", str(src),
                             "-o", str(obj)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    # e_ident[EI_ABIVERSION]: 2 is code object v4, 3 is v5.
    return src.read_text(), obj.read_bytes()[8]


@pytest.mark.parametrize("module, args", EXTOP_GENERATORS, ids=[g[0] for g in EXTOP_GENERATORS])
@pytest.mark.parametrize("arch, version, minor, abi", [
    ("gfx1250", 5, 2, 3),
    ("gfx942", 4, 1, 2),
])
def test_extop_code_object_version(tmp_path, cxx_compiler, module, args, arch, version, minor, abi):
    text, abiVersion = _assembleExtop(tmp_path, cxx_compiler, module, args, arch)
    assert text.startswith(f".amdhsa_code_object_version {version}\n")
    assert f"amdhsa.version:\n- 1\n- {minor}\n" in text
    assert abiVersion == abi
