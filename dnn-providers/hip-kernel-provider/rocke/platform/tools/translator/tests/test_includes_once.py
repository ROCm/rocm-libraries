# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A generated source includes each rocKE header once, and one header for a
function several headers declare: rocke_kernel_name_join is declared in both
helper_rocke.helpers.spec.h and helper_helper_rocke.helpers.spec.h, so a
source that also calls rocke_validate_io (helper_rocke.helpers.spec.h only)
needs only that one."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest
from trtest import FIXTURES, ROCKE, TR, Tr

MODULE = "_attention_bwd_include_probe"

SOURCE = """\
from __future__ import annotations

from rocke.helpers.spec import kernel_name_join

__all__ = ["probe_name"]


def probe_name(prefix: str) -> str:
    return kernel_name_join(prefix, "x")
"""

# the hand fill of the hole: calls a function of each spec header
FILL = """\
/* probe_name: kernel_name_join(prefix, "x"), and rocke_validate_io */
rocke_status_t rocke_attn_bwd_probe_name(const char* prefix, char* out, size_t out_cap)
{
    const char* parts[] = {"x"};
    (void)rocke_validate_io(NULL, NULL); /* arguments elided: text only, not built */
    return rocke_kernel_name_join(prefix, parts, 1, NULL, NULL, 0, out, out_cap, NULL);
}
"""


@pytest.fixture(scope="module")
def cpp(tmp_path_factory, clang_format):
    root = tmp_path_factory.mktemp("includes")
    lib = root / "library"
    shutil.copytree(FIXTURES / "library", lib)
    (lib / "kernels" / "common" / f"{MODULE}.py").write_text(SOURCE)
    holes = root / "holes" / MODULE
    holes.mkdir(parents=True)
    (holes / "probe_name.cpp").write_text(FILL)
    tr = Tr(lib)
    env = dict(tr.env(), TR_HOLES=str(root / "holes"))
    abi = root / "abi"
    abi.mkdir()
    out = root / f"{MODULE}.cpp"
    for args in (
        ["tr.abi", MODULE, "--header", str(abi)],
        ["tr.translate", MODULE, "--abi", str(abi), "--out", str(out)],
    ):
        p = subprocess.run(
            [sys.executable, "-m", *args, "--rocke", str(ROCKE)],
            cwd=TR,
            env=env,
            capture_output=True,
            text=True,
        )
        assert p.returncode == 0, p.stdout + p.stderr
    return out.read_text()


def test_fill_is_used(cpp):
    assert "rocke_kernel_name_join(prefix, parts" in cpp


def test_each_include_once(cpp):
    incs = re.findall(r"^#include .*$", cpp, re.M)
    assert len(incs) == len(set(incs)), incs


def test_one_header_for_a_function_declared_twice(cpp):
    assert '#include "rocke/helper_rocke.helpers.spec.h"' in cpp
    assert "helper_helper_rocke.helpers.spec.h" not in cpp
