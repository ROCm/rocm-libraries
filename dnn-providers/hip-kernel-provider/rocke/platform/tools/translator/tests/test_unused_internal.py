# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""An internal function nothing calls is not emitted: a static function with no
caller fails -Wunused-function under -Wall -Werror (the build's flags). It
happens when the function's exported entry is not written (a hole) and when a
private helper is called only by such a function."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest
from trtest import FIXTURES, ROCKE, TR, Tr

sys.path.insert(0, str(TR))
from tr.translate import used_internals  # noqa: E402

MODULE = "_attention_bwd_unused_probe"

SOURCE = """\
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["OpaqueSpec", "probe_opaque", "probe_ok"]


@dataclass(frozen=True)
class OpaqueSpec:
    other: Mystery
    vec: int = 8


def _twice(n: int) -> int:
    return 2 * n


def probe_opaque(spec: OpaqueSpec) -> int:
    return _twice(spec.vec)


def probe_ok(n: int) -> int:
    return n + 1
"""


def _fn(name: str, *calls: str) -> list[str]:
    body = [f"    {c}(x);" for c in calls]
    return [f"/* py: {name} */", f"int64_t {name}(int64_t x)", "{"] + body + ["}"]


def test_used_internals_drops_chains():
    a = _fn("py_a", "py_b")  # called by nothing: dropped, then b and c with it
    b = _fn("py_b", "py_c")
    c = _fn("py_c")
    d = _fn("py_d", "py_d")  # only calls itself: dropped
    e = _fn("py_e")  # called by an exported entry: kept
    kept = used_internals([a, b, c, d, e], ["int rocke_x(int x) { return py_e(x); }"])
    assert kept == [e]


@pytest.fixture(scope="module")
def probe(tmp_path_factory, clang_format):
    """The probe's generated header directory and source."""
    root = tmp_path_factory.mktemp("unused")
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
    return abi, out.read_text()


def test_unused_internal_not_emitted(probe):
    _, cpp = probe
    assert "py_probe_ok(" in cpp
    assert "py_probe_opaque(" not in cpp  # its entry is a hole
    assert "py__twice(" not in cpp  # only probe_opaque called it
    assert '#error "hole: probe_opaque' in cpp.replace("\\\n    ", "")


def test_source_builds_with_werror(probe, tmp_path):
    """The source compiles to an object with -Wall -Werror, apart from the
    #error line of each hole (-fsyntax-only does not report unused functions)."""
    exe = shutil.which("g++")
    if exe is None:
        pytest.skip("no g++")
    abi, cpp = probe
    inc = tmp_path / "inc"
    shutil.copytree(abi, inc / "rocke")
    src = tmp_path / "src.cpp"
    src.write_text(re.sub(r'#error (?:\\\n\s*)?".*"\n', "", cpp))
    cmd = [exe, "-std=c++20", "-c", "-o", str(tmp_path / "src.o"), "-Wall", "-Werror"]
    cmd += ["-I", str(inc), "-isystem", str(ROCKE / "platform" / "cpp" / "include")]
    cmd += ["-I", str(TR / "tr" / "prelude"), str(src)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
