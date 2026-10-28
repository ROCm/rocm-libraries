# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The reports count every function once, by its status: tr.translate
--analyze lists each function as translated, derived, filled or hole, and its
summary line and tr.regen's per-module line count the same four the same way
(a hand fill is not a hole, a derived companion is mechanical)."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest
from trtest import FIXTURES, ROCKE, TR, Tr

MODULE = "_attention_bwd_report_probe"

SOURCE = """\
from __future__ import annotations

from dataclasses import dataclass

from rocke.helpers.spec import kernel_name_join

__all__ = ["ProbeSpec", "probe_name", "probe_add"]


@dataclass
class ProbeSpec:
    n: int = 1


def probe_name(prefix: str) -> str:
    return kernel_name_join(prefix, "x")


def probe_add(a: int, b: int) -> int:
    return a + b
"""

# the hand fill of probe_name (kernel_name_join has no translation rule)
FILL = """\
/* probe_name: kernel_name_join(prefix, "x") */
rocke_status_t rocke_attn_bwd_probe_name(const char* prefix, char* out, size_t out_cap)
{
    const char* parts[] = {"x"};
    return rocke_kernel_name_join(prefix, parts, 1, NULL, NULL, 0, out, out_cap, NULL);
}
"""

SUMMARY = "2/3 functions mechanical (1 translated, 1 derived), 1 hand fills, 0 holes"


@pytest.fixture(scope="module")
def probe(tmp_path_factory, clang_format):
    """(root, env): the fixture modules plus the probe and its fill."""
    root = tmp_path_factory.mktemp("reports")
    lib = root / "library"
    shutil.copytree(FIXTURES / "library", lib)
    (lib / "kernels" / "common" / f"{MODULE}.py").write_text(SOURCE)
    holes = root / "holes" / MODULE
    holes.mkdir(parents=True)
    (holes / "probe_name.cpp").write_text(FILL)
    env = dict(Tr(lib).env(), TR_HOLES=str(root / "holes"))
    abi = root / "abi"
    abi.mkdir()
    run(env, "tr.abi", MODULE, "--header", str(abi))
    return root, env


def run(env, *args: str) -> subprocess.CompletedProcess:
    p = subprocess.run(
        [sys.executable, "-m", *args, "--rocke", str(ROCKE)],
        cwd=TR,
        env=env,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    return p


def test_analyze_counts_a_fill_as_a_fill(probe):
    root, env = probe
    p = run(env, "tr.translate", MODULE, "--abi", str(root / "abi"), "--analyze")
    lines = p.stdout.strip().splitlines()
    assert re.match(r"^filled +probe_name \(hand fill for: ", lines[0]), lines
    assert re.match(r"^translated +probe_add$", lines[1]), lines
    assert re.match(r"^derived +rocke_attn_bwd_\w+_default$", lines[2]), lines
    assert lines[-1] == SUMMARY, lines


def test_regen_reports_the_same_counts(probe, tmp_path):
    root, env = probe
    p = run(
        env,
        "tr.regen",
        MODULE,
        "--generated-abi",
        "--into",
        str(tmp_path),
        "--pretend",
    )
    assert f"{MODULE}: {SUMMARY}" in p.stdout.splitlines(), p.stdout
