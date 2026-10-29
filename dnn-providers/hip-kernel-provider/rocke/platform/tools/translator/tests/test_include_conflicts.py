# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A type two headers define differently (rocke_tensor_descriptor_t: one
struct in helper_rocke.helpers.tensor_view.h, another in
helper_rocke.helpers.transforms.h) comes from the header of the functions the
source calls with it, never from both: including both does not compile. When
the source calls functions of both headers, no choice works: tr.translate
lists the conflict and exits 1, and tr.regen writes nothing."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest
from trtest import FIXTURES, ROCKE, TR, Tr

MODULE = "_attention_bwd_include_probe"
TV = "helper_rocke.helpers.tensor_view.h"
XF = "helper_rocke.helpers.transforms.h"

SOURCE = """\
from __future__ import annotations

from rocke.helpers.spec import kernel_name_join

__all__ = ["probe_name"]


def probe_name(prefix: str) -> str:
    return kernel_name_join(prefix, "x")
"""

# names the shared type and calls a transforms.h function that takes it
FILL_ONE = """\
/* probe_name: kernel_name_join(prefix, "x"); a transforms.h descriptor */
rocke_status_t rocke_attn_bwd_probe_name(const char* prefix, char* out, size_t out_cap)
{
    const rocke_tensor_descriptor_t* desc = NULL;
    if(prefix == NULL)
        (void)rocke_tensor_descriptor_unmerge_lower(NULL, desc, NULL, NULL, 0, NULL, NULL, 0);
    const char* parts[] = {"x"};
    return rocke_kernel_name_join(prefix, parts, 1, NULL, NULL, 0, out, out_cap, NULL);
}
"""

# and a tensor_view.h function too: the two definitions would meet
FILL_BOTH = FILL_ONE.replace(
    "    const char* parts[]",
    "    (void)rocke_tensor_descriptor_rank(NULL);\n    const char* parts[]",
)


def run(root, fill, tool_args):
    """tr.abi then the tool (tr.translate or tr.regen args) on the probe with fill."""
    lib = root / "library"
    if not lib.exists():
        shutil.copytree(FIXTURES / "library", lib)
        (lib / "kernels" / "common" / f"{MODULE}.py").write_text(SOURCE)
    holes = root / "holes" / MODULE
    holes.mkdir(parents=True, exist_ok=True)
    (holes / "probe_name.cpp").write_text(fill)
    env = dict(Tr(lib).env(), TR_HOLES=str(root / "holes"))
    abi = root / "abi"
    abi.mkdir(exist_ok=True)
    p = subprocess.run(
        [sys.executable, "-m", "tr.abi", MODULE, "--header", str(abi)]
        + ["--rocke", str(ROCKE)],
        cwd=TR,
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    return subprocess.run(
        [sys.executable, "-m", *tool_args, "--rocke", str(ROCKE)],
        cwd=TR,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def one(tmp_path_factory, clang_format):
    root = tmp_path_factory.mktemp("incl_one")
    out = root / f"{MODULE}.cpp"
    p = run(
        root,
        FILL_ONE,
        ["tr.translate", MODULE, "--abi", str(root / "abi"), "--out", str(out)],
    )
    assert p.returncode == 0, p.stdout + p.stderr
    return out.read_text(), root / "abi"


def test_type_comes_from_the_header_of_the_called_function(one):
    cpp = one[0]
    assert f'#include "rocke/{XF}"' in cpp
    assert TV not in cpp


def test_source_compiles(one, tmp_path):
    exe = shutil.which("g++")
    if exe is None:
        pytest.skip("no g++")
    cpp, abi = one
    inc = tmp_path / "inc"
    shutil.copytree(abi, inc / "rocke")
    src = tmp_path / "src.cpp"
    src.write_text(cpp)
    cmd = [exe, "-std=c++20", "-fsyntax-only", "-I", str(inc)]
    cmd += ["-isystem", str(ROCKE / "platform" / "cpp" / "include")]
    cmd += ["-I", str(TR / "tr" / "prelude"), str(src)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert "rocke_tensor_descriptor" not in p.stderr, p.stderr


def test_calls_into_both_headers_are_a_conflict(tmp_path, clang_format):
    out = tmp_path / f"{MODULE}.cpp"
    p = run(
        tmp_path,
        FILL_BOTH,
        ["tr.translate", MODULE, "--abi", str(tmp_path / "abi"), "--out", str(out)],
    )
    assert p.returncode == 1, p.stdout + p.stderr
    line = next(
        x for x in p.stdout.splitlines() if x.startswith("conflict ")
    )
    assert "rocke_tensor_descriptor_t" in line and TV in line and XF in line
    cpp = out.read_text()
    assert re.search(r'#error (?:\\\n\s*)?"include conflict: rocke_tensor_descriptor_t', cpp), cpp


def test_regen_writes_nothing_on_a_conflict(tmp_path, clang_format):
    into = tmp_path / "into"
    p = run(
        tmp_path,
        FILL_BOTH,
        ["tr.regen", MODULE, "--generated-abi", "--into", str(into)],
    )
    assert p.returncode == 1, p.stdout + p.stderr
    assert "include conflicts" in p.stdout and "nothing written" in p.stdout
    assert not into.exists() or not any(into.rglob("*.cpp"))
