# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The extern "C" entries translated against an existing hand header (the
hand ABI, tr.regen --hand-abi), here the reviewed instance_batched_transpose.h
of PR #14538 (fixtures/hand_abi/) put into a copy of the tree.

- An (ok, reason) Python result behind `bool f(..., char* reason, size_t
  reason_cap)` writes the reason into the buffer and returns ok, as the
  generated ABI does (not a tuple cast to bool, which does not compile).
- A Python int (int64_t) behind a narrower C `int` is narrowed with a range
  check, when the header documents the value the entry returns on a failure
  ("Returns 0 for a NULL spec or vec == 0"): a NULL argument it names, a
  Python raise and a result out of range all return that value.
- Otherwise the entry is an ABI hole, and an ABI hole is a hole: --analyze
  lists and counts it, tr.translate exits 1, tr.regen writes nothing; a hand
  fill of the function replaces the entry."""

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


BLOCK = "rocke_batched_transpose2d_block_size"
DOC = (
    "/* BatchedTranspose2DSpec.block_size @property: (tile_m * tile_n) // vec, with\n"
    " * Python floor division. Returns 0 for a NULL spec or vec == 0 (where the\n"
    " * Python raises ZeroDivisionError). */"
)
UNDOCUMENTED = "/* BatchedTranspose2DSpec.block_size @property: (tile_m * tile_n) // vec. */"


def test_int_narrowing_follows_the_header(reviewed):
    cpp, _, p = reviewed
    e = re.sub(r"\s+", "", entry(cpp, BLOCK))  # the formatting does not matter
    assert "abi hole" not in cpp
    for want in (
        "if(spec==nullptr)return0;",
        "constint64_tr_=py_BatchedTranspose2DSpec_block_size(spec);",
        "if((int64_t)(int)r_==r_)return(int)r_;",
        "catch(constckc::Error&)",
    ):
        assert want in e, e
    assert e.endswith("return0;}")
    assert "0 holes" in p.stdout.splitlines()[-1], p.stdout


@pytest.fixture(scope="module")
def undocumented(tmp_path_factory, clang_format):
    """A header whose int entry documents no failure value."""
    text = (FIXTURES / "hand_abi" / HEADER).read_text()
    assert DOC in text
    root = tmp_path_factory.mktemp("hand_abi_undoc")
    return root, hand_abi_env(root, text.replace(DOC, UNDOCUMENTED))


def test_abi_hole_is_a_hole_of_the_report(undocumented):
    root, env = undocumented
    p = tr(env, "tr.translate", "batched_transpose", "--analyze")
    line = next(x for x in p.stdout.splitlines() if "block_size" in x)
    assert line.startswith("hole       BatchedTranspose2DSpec.block_size (abi hole: ")
    assert "documents no return value for a Python raise" in line
    assert ", 1 holes," in p.stdout.splitlines()[-1]
    out = root / "o.cpp"
    p = tr(env, "tr.translate", "batched_transpose", "--out", str(out))
    assert p.returncode == 1
    assert f'abi hole: {BLOCK}' in out.read_text()


def test_regen_writes_nothing_with_an_abi_hole(undocumented, tmp_path):
    _, env = undocumented
    into = tmp_path / "into"
    p = tr(env, "tr.regen", "batched_transpose", "--hand-abi", "--into", str(into))
    assert p.returncode == 1, p.stdout + p.stderr
    assert "nothing written" in p.stdout
    assert "BatchedTranspose2DSpec.block_size (abi hole:" in p.stdout
    assert not into.exists() or not any(into.rglob("*.cpp"))


def test_a_fill_replaces_the_abi_hole(tmp_path, clang_format):
    text = (FIXTURES / "hand_abi" / HEADER).read_text().replace(DOC, UNDOCUMENTED)
    fill = (
        "/* BatchedTranspose2DSpec.block_size: (tile_m * tile_n) // vec */\n"
        f"int {BLOCK}(const rocke_batched_transpose2d_spec_t* spec)\n"
        "{\n    return spec->tile_m * spec->tile_n / spec->vec;\n}\n"
    )
    env = hand_abi_env(tmp_path, text, {"BatchedTranspose2DSpec_block_size": fill})
    into = tmp_path / "into"
    p = tr(env, "tr.regen", "batched_transpose", "--hand-abi", "--into", str(into))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "1 hand fills, 0 holes" in p.stdout
    cpp = next(into.rglob("batched_transpose.cpp")).read_text()
    assert "spec->tile_m * spec->tile_n / spec->vec" in cpp
    assert "abi hole" not in cpp


def test_documented_fail_reads_only_failure_values(tmp_path):
    sys.path.insert(0, str(TR))
    from tr.env import CFunc
    from tr.translate import documented_fail

    class Env:
        class headers:
            paths = {}

    def doc(comment):
        h = tmp_path / "h.h"
        h.write_text(f"{comment}\nint f(const s_t* spec, int n);\n")
        Env.headers.paths = {"h.h": str(h)}
        line = comment.count("\n") + 2
        cf = CFunc("f", "int", [("const s_t*", "spec"), ("int", "n")], "h.h", line)
        return documented_fail(Env, cf)

    assert doc(DOC)[::2] == ("0", ["spec"])
    assert doc("/* n // 2. Returns -1 on a raise. */")[::2] == ("-1", [])
    assert doc("/* Returns 0 for a NULL spec or vec == 0. */")[::2] == ("0", ["spec"])
    assert doc("/* Returns 1 if the spec is valid. */") is None
    assert doc(UNDOCUMENTED) is None
    assert doc("// Returns 0 for a NULL spec.") is None  # not a block comment above
