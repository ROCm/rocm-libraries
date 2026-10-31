# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Field types of a generated struct, and the guard that keeps the generated
header from naming a struct it never declares: a Literal of strings is a
string, a class another module defines uses its C twin struct from the tree,
and a declaration that uses a struct defined nowhere is refused (a hole)."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest
from trtest import FIXTURES, ROCKE, TR, Tr

MODULE = "_attention_bwd_type_probe"
HEADER = f"helper_rocke.instances.common.{MODULE}.h"

SOURCE = """\
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rocke.instances.common.gemm_universal import TileSpec

DType = Literal["f16", "bf16"]

__all__ = ["ProbeSpec", "OpaqueSpec", "probe_tile_m", "probe_opaque"]


@dataclass(frozen=True)
class ProbeSpec:
    tile: TileSpec
    vec: int = 8
    dtype: DType = "f16"
    mode: Literal["row", "col"] = "row"


@dataclass(frozen=True)
class OpaqueSpec:
    other: Mystery
    vec: int = 8


def probe_tile_m(spec: ProbeSpec) -> int:
    return spec.tile.tile_m * spec.vec


def probe_opaque(spec: OpaqueSpec) -> int:
    return spec.vec
"""


@pytest.fixture(scope="module")
def probe(tmp_path_factory, clang_format):
    """The probe module's header, --analyze output and source."""
    root = tmp_path_factory.mktemp("abi_types")
    lib = root / "library"
    shutil.copytree(FIXTURES / "library", lib)
    (lib / "kernels" / "common" / f"{MODULE}.py").write_text(SOURCE)
    tr = Tr(lib)
    abi = root / "abi"
    abi.mkdir()
    tr.run("abi", MODULE, "--header", str(abi))
    out = root / f"{MODULE}.cpp"
    subprocess.run(
        [sys.executable, "-m", "tr.translate", MODULE, "--abi", str(abi)]
        + ["--out", str(out), "--rocke", str(ROCKE)],
        cwd=TR,
        env=tr.env(),
        capture_output=True,
        text=True,
    )
    run = subprocess.run(
        [sys.executable, "-m", "tr.translate", MODULE, "--abi", str(abi)]
        + ["--analyze", "--rocke", str(ROCKE)],
        cwd=TR,
        env=tr.env(),
        capture_output=True,
        text=True,
    )
    return abi / HEADER, run.stdout + run.stderr, out.read_text()


def _struct(header: str, name: str) -> str:
    m = re.search(rf"typedef struct {name}\n\{{(.*?)\}} {name}_t;", header, re.S)
    assert m, f"no struct {name} in\n{header}"
    return m.group(1)


def test_str_literal_is_a_string(probe):
    header = probe[0].read_text()
    body = _struct(header, "rocke_attn_bwd_probe_spec")
    assert "const char* dtype;" in body  # DType = Literal["f16", "bf16"]
    assert "const char* mode;" in body  # Literal["row", "col"] written out


def test_class_of_another_module_uses_its_twin(probe):
    header = probe[0].read_text()
    body = _struct(header, "rocke_attn_bwd_probe_spec")
    assert "rocke_gemm_tile_spec_t tile;" in body
    assert '#include "rocke/instance_gemm_universal.h"' in header
    _, analyze, cpp = probe
    assert "translated probe_tile_m" in analyze
    assert "spec->tile.tile_m" in cpp


def test_undeclared_struct_is_refused(probe):
    path, analyze, cpp = probe
    header = path.read_text()
    assert "rocke_attn_bwd_opaque_spec_t" not in header
    assert "rocke_attn_bwd_probe_opaque" not in header
    assert re.search(
        r"hole +probe_opaque \(not exported: .*rocke_attn_bwd_opaque_spec_t, defined nowhere",
        analyze,
    ), analyze
    assert '#error "hole: probe_opaque' in cpp.replace("\\\n    ", "")


def test_header_compiles(probe, tmp_path):
    exe = shutil.which("g++")
    if exe is None:
        pytest.skip("no g++")
    path = probe[0]
    tu = tmp_path / "hdr.cpp"
    tu.write_text(f'#include "{path}"\n')
    inc = ROCKE / "platform" / "cpp" / "include"
    cmd = [exe, "-std=c++20", "-fsyntax-only", "-Wall", "-Werror", "-I", str(inc)]
    p = subprocess.run(cmd + [str(tu)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
