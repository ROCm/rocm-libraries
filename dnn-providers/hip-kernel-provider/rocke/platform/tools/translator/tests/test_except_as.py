# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""`except KeyError as e: return False, str(e)` (RT-R1 with a bound name) and
the lifetime of a str a function returns as const char* (py::keep)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap

import pytest
from trtest import FIXTURES, ROCKE, TR, Tr

MODULE = "_attention_bwd_exc_probe"

SOURCE = """\
from rocke.core.arch import ArchTarget


def arch_reason(arch: str) -> tuple[bool, str]:
    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if target.wave_size != 64:
        return False, f"wave {target.wave_size} on {arch}"
    return True, "ok"


def arch_only(arch: str) -> tuple[bool, str]:
    try:
        ArchTarget.from_gfx(arch)
    except KeyError:
        return False, "unknown"
    return True, "supported"

"""

# repr(e) has no rule: the function is a hole
REFUSED = """

def arch_repr(arch: str) -> tuple[bool, str]:
    try:
        ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, repr(e)
    return True, "supported"
"""


@pytest.fixture(scope="module")
def probe(tmp_path_factory, clang_format):
    """The probe module translated against the fixture data: (analyze, cpp)."""
    root = tmp_path_factory.mktemp("exc")
    lib = root / "library"
    shutil.copytree(FIXTURES / "library", lib)
    (lib / "kernels" / "common" / f"{MODULE}.py").write_text(SOURCE)
    tr = Tr(lib)
    abi = root / "abi"
    abi.mkdir()
    tr.run("abi", MODULE, "--header", str(abi))
    out = root / f"{MODULE}.cpp"
    run = tr.run("translate", MODULE, "--abi", str(abi), "--out", str(out))
    cpp = out.read_text()
    (lib / "kernels" / "common" / f"{MODULE}.py").write_text(SOURCE + REFUSED)
    tr.run("abi", MODULE, "--header", str(abi))
    run = subprocess.run(
        [sys.executable, "-m", "tr.translate", MODULE, "--abi", str(abi)]
        + ["--analyze", "--rocke", str(ROCKE)],
        cwd=TR,
        env=tr.env(),
        capture_output=True,
        text=True,
    )
    return run.stdout + run.stderr, cpp


def _body(cpp: str, fn: str) -> str:
    start = cpp.index(f"py_{fn}(const char* arch)\n{{")
    return cpp[start : cpp.index("\n}\n", start)]


def test_except_as_translates(probe):
    analyze, _ = probe
    assert "translated arch_reason" in analyze
    assert "translated arch_only" in analyze


def test_str_of_caught_exception_is_its_message(probe):
    _, cpp = probe
    body = _body(cpp, "arch_reason")
    assert "catch(const ckc::KeyError& e)" in body
    assert "{false, py::keep(e.message())}" in body


def test_returned_string_is_kept(probe):
    """A std::string returned as const char* is interned, not .c_str() of a
    local that dies with the function."""
    _, cpp = probe
    body = _body(cpp, "arch_reason")
    assert "py::keep(e_1)" in body
    assert ".c_str()" not in body


def test_unnamed_handler_unchanged(probe):
    _, cpp = probe
    body = _body(cpp, "arch_only")
    assert "catch(const ckc::KeyError&)" in body


def test_other_uses_of_the_name_refuse(probe):
    analyze, _ = probe
    assert "hole       arch_repr (repr() of a caught exception" in analyze


def test_keep_is_stable(tmp_path):
    """py::keep returns one pointer per distinct text, valid after the string dies."""
    exe = shutil.which("g++")
    if exe is None:
        pytest.skip("no g++")
    src = tmp_path / "keep.cpp"
    src.write_text(
        textwrap.dedent(
            """\
            #include "rocke_py.hpp"
            #include <cstdio>
            static const char* make(int i) { std::string s = "why " + std::to_string(i); return py::keep(s); }
            int main()
            {
                const char* a = make(1);
                const char* b = make(2);
                const char* c = make(1);
                return (a == c && a != b && strcmp(a, "why 1") == 0 && strcmp(b, "why 2") == 0) ? 0 : 1;
            }
            """
        )
    )
    inc = ROCKE / "platform" / "cpp" / "include"
    prelude = TR / "tr" / "prelude"
    out = tmp_path / "keep"
    cmd = [exe, "-std=c++17", "-Wall", "-Werror", "-I", str(prelude), "-I", str(inc)]
    p = subprocess.run(cmd + [str(src), "-o", str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert subprocess.run([str(out)]).returncode == 0
