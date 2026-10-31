# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A platform instance's (ok, reason) result is the platform's validity-gate
ABI, as in every instance_*.h: `bool f(..., char* reason, size_t
reason_cap)`, the reason written into the caller's buffer and a raise turned
into false plus its text. Not a tuple struct whose `const char*` has no clear
owner. Attention backward keeps its ABI (its goldens check that)."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest
from trtest import ROCKE, TR, platform_probe, strip_holes

sys.path.insert(0, str(TR))
from tr.package import PACKAGES  # noqa: E402

PROBE = """\
def probe_check(spec: BatchedTranspose2DSpec, cap: int = 64) -> tuple[bool, str]:
    \"\"\"(ok, reason) of the probe.\"\"\"
    if spec.tile_m % spec.vec:
        return False, "tile must be divisible by vec"
    if spec.tile_m > 4 * cap:
        raise ValueError("tile too big")
    if spec.tile_m > cap:
        return False, f"tile {spec.tile_m} > {cap}"
    return True, "ok"
"""

HEADER = "instance_batched_transpose.h"


@pytest.fixture(scope="module")
def probe(tmp_path_factory, clang_format):
    """(header text, source text, abi dir) of the probe."""
    root = tmp_path_factory.mktemp("ok_reason")
    env = platform_probe(root, PROBE)
    abi = root / "abi"
    abi.mkdir()
    out = root / "batched_transpose.cpp"
    for args in (["tr.abi", "--header", str(abi)], ["tr.translate", "--out", str(out)]):
        extra = ["--abi", str(abi)] if args[0] == "tr.translate" else []
        subprocess.run(
            [sys.executable, "-m", args[0], "batched_transpose", *args[1:], *extra]
            + ["--rocke", str(ROCKE)],
            cwd=TR,
            env=env,
            capture_output=True,
            text=True,
        )
    return (abi / HEADER).read_text(), out.read_text(), abi


def test_only_platform_instances_use_it():
    assert not PACKAGES["attention_bwd"].ok_reason_abi
    assert PACKAGES["batched_transpose"].ok_reason_abi


def test_header_declares_bool_and_reason_buffer(probe):
    header = probe[0]
    decl = re.search(
        r"^(\w[^;/]*\brocke_batched_transpose_probe_check\([^;]*\));", header, re.M
    )
    assert decl, header
    assert re.sub(r"\s+", " ", decl.group(1)) == (
        "bool rocke_batched_transpose_probe_check(const rocke_batched_transpose_spec_t* spec,"
        " int cap, char* reason, size_t reason_cap)"
    )
    assert "tuple_bool_str" not in header


def test_entry_writes_the_reason_and_returns_ok(probe):
    cpp = probe[1]
    m = re.search(r"\nbool rocke_batched_transpose_probe_check\(.*?\n\}\n", cpp, re.S)
    assert m, cpp
    entry = re.sub(r"\s+", " ", m.group(0))
    assert "py_adapt::with_reason<bool>(reason, reason_cap, false," in entry
    assert "const auto res_ = py_probe_check(spec, cap);" in entry
    assert "strncpy(reason, why_.c_str(), reason_cap - 1);" in entry
    assert "return std::get<0>(res_);" in entry


def test_source_compiles(probe, tmp_path):
    exe = shutil.which("g++")
    if exe is None:
        pytest.skip("no g++")
    _, cpp, abi = probe
    inc = tmp_path / "inc"
    shutil.copytree(abi, inc / "rocke")
    src = tmp_path / "src.cpp"
    src.write_text(strip_holes(cpp))
    cmd = [exe, "-std=c++20", "-fsyntax-only", "-Wall", "-Werror", "-I", str(inc)]
    cmd += ["-isystem", str(ROCKE / "platform" / "cpp" / "include")]
    cmd += ["-I", str(TR / "tr" / "prelude"), str(src)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
