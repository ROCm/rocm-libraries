# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The extern "C" entries translated against an existing hand header (the
hand ABI, tr.regen --hand-abi), here the reviewed instance_batched_transpose.h
of PR #14538 (fixtures/hand_abi/) put into a copy of the tree.

- An (ok, reason) Python result behind `bool f(..., char* reason, size_t
  reason_cap)` writes the reason into the buffer and returns ok, as the
  generated ABI does (not a tuple cast to bool, which does not compile)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

import pytest
from trtest import FIXTURES, TR, overlay_root, strip_holes

HEADER = "instance_batched_transpose.h"
INC = "platform/cpp/include/rocke"


def hand_abi_env(root, header_text: str, fills: dict | None = None) -> dict:
    """Environment of a batched_transpose translation against header_text as
    the tree's hand header (overlay root at root/rocke, maps mined from it,
    no facts, fills as hand fills)."""
    rocke = overlay_root(root / "rocke", {f"{INC}/{HEADER}": header_text})
    holes = root / "holes" / "batched_transpose"
    holes.mkdir(parents=True)
    for name, body in (fills or {}).items():
        (holes / f"{name}.cpp").write_text(body)
    (root / "data").mkdir()
    env = dict(os.environ)
    env.pop("TR_LIBRARY", None)
    env.pop("TR_PARITY", None)
    env.update(
        TR_PACKAGE="batched_transpose",
        TR_DATA=str(root / "data"),
        TR_MAPS=str(root / "maps"),
        TR_HOLES=str(root / "holes"),
        ROCKE_ROOT=str(rocke),
        PYTHONDONTWRITEBYTECODE="1",
    )
    p = subprocess.run(
        [sys.executable, "tools/mine_maps.py", "--rocke", str(rocke)]
        + ["--out", str(root / "maps"), "--only", "sigmap", "--only", "twin_map"],
        cwd=TR,
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    return env


def tr(env, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=TR,
        env=env,
        capture_output=True,
        text=True,
    )


def entry(cpp: str, cname: str) -> str:
    """The extern "C" definition of cname, whitespace collapsed."""
    m = re.search(rf"\n\w[^\n;]*\b{cname}\([^;{{]*\)\s*\{{.*?\n\}}\n", cpp, re.S)
    assert m, cpp
    return re.sub(r"\s+", " ", m.group(0))


def compiles(cpp: str, rocke, tmp_path) -> subprocess.CompletedProcess | None:
    exe = shutil.which("g++")
    if exe is None:
        return None
    src = tmp_path / "src.cpp"
    src.write_text(cpp)
    cmd = [exe, "-std=c++20", "-fsyntax-only", "-Wall", "-Werror"]
    cmd += ["-isystem", str(rocke / "platform" / "cpp" / "include")]
    cmd += ["-I", str(TR / "tr" / "prelude"), str(src)]
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory, clang_format):
    """(source, rocke root, translate run) against the reviewed header."""
    root = tmp_path_factory.mktemp("hand_abi")
    env = hand_abi_env(root, (FIXTURES / "hand_abi" / HEADER).read_text())
    out = root / "batched_transpose.cpp"
    p = tr(env, "tr.translate", "batched_transpose", "--out", str(out))
    return out.read_text(), root / "rocke", p


def test_ok_reason_writes_the_reason_and_returns_ok(reviewed):
    e = entry(reviewed[0], "rocke_batched_transpose2d_is_valid_spec")
    assert "(bool)py_is_valid_spec" not in e
    assert "py_adapt::with_reason<bool>(reason, reason_cap, false," in e
    assert 'const auto res_ = py_is_valid_spec(spec, arch != nullptr ? arch : "gfx950");' in e
    assert "strncpy(reason, why_.c_str(), reason_cap - 1);" in e
    assert "reason[reason_cap - 1] = '\\0';" in e
    assert "return std::get<0>(res_);" in e


def test_ok_reason_same_code_as_the_generated_abi():
    """Both shim paths build the gate's body from one routine."""
    sys.path.insert(0, str(TR))
    from tr import translate

    src = open(translate.__file__).read()
    assert src.count("ok_reason_body(call)") == 2
    assert src.count("strncpy(reason, why_") == 1


def test_source_compiles(reviewed, tmp_path):
    cpp, rocke, _ = reviewed
    p = compiles(strip_holes(cpp), rocke, tmp_path)
    if p is None:
        pytest.skip("no g++")
    assert p.returncode == 0, p.stderr
