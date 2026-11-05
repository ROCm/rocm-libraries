# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for tools/make_gfx1250_conv_fwd_aot_test_vectors.py: the committed golden header
regenerates from the in-tree rocKE.

hipkernel:Gfx1250ConvFwdAot's C++ port of admission and launch is unit-tested against
Gfx1250ConvFwdAotTestVectors.hpp (TestGfx1250ConvFwdAotGeometry). The header is written
from the catalog's instance lists through the packaging adapter's plan, which is the
dispatcher's. A change to the lists, the dispatcher's admission or launch plan, or the
tool that is not regenerated fails here, rather than leaving the committed header to vouch
for a port of a dispatcher that no longer exists.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="the rocKE builders need numpy")

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools"
sys.path.insert(0, str(_TOOLS))

import launch_surface  # noqa: E402
import make_gfx1250_conv_fwd_aot_test_vectors as vectors  # noqa: E402

_REPO = launch_surface.find_repo_root(_TOOLS)
_ROCKE = _REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_HEADER = (
    _REPO / "dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine"
    "/packs/Gfx1250ConvFwdAotTestVectors.hpp"
)
_INSTANCES = [
    _ROOT / "configs" / f"gfx1250_conv_fwd_aot_{dtype}.instances.json"
    for dtype in ("fp16", "bf16")
]


def _run(*args: str, cwd: Path = _ROOT) -> subprocess.CompletedProcess:
    # Nothing here needs a GPU.
    env = dict(os.environ, HIP_VISIBLE_DEVICES="-1")
    tool = _TOOLS / "make_gfx1250_conv_fwd_aot_test_vectors.py"
    return subprocess.run(
        [sys.executable, str(tool), "--rocke-root", str(_ROCKE), *args],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def _text(path: Path) -> str:
    # A Windows checkout may hold the committed header with CRLF; the tool writes LF.
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def test_the_golden_header_regenerates_from_the_in_tree_dispatcher(tmp_path):
    out = tmp_path / _HEADER.name
    result = _run(
        "--instances",
        *map(str, _INSTANCES),
        "--header",
        str(out),
        "--problems-from-header",
        str(_HEADER),
    )
    assert result.returncode == 0, result.stderr
    assert _text(out) == _text(_HEADER), (
        f"{_HEADER.name} no longer matches rocKE's dispatcher: regenerate it with "
        "make_gfx1250_conv_fwd_aot_test_vectors.py --instances <lists> --header <it> "
        "--problems-from-header <it>"
    )


def test_problems_from_header_reads_back_the_recorded_set():
    problems = vectors.problems_from_header(_HEADER)
    text = _HEADER.read_text(encoding="utf-8")
    declared = int(
        text.split("std::array<std::array<int32_t, 15>, ", 1)[1].split(">", 1)[0]
    )
    assert len(problems) == declared
    assert {p["dtype"] for p in problems} == {"fp16", "bf16"}
    assert all(p["N"] >= 1 and p["G"] >= 1 for p in problems)


@pytest.mark.parametrize(
    "args",
    [
        ("--header", "h.hpp", "--out", "o.jsonl"),
        ("--out", "o.jsonl", "--synthetic", "2"),
        ("--header", "h.hpp", "--per-kernel", "3"),
        ("--out", "o.jsonl", "--problems-from-header", "h.hpp"),
        ("--header", "h.hpp", "--problems-from-header", "h.hpp", "--shapes", "s.txt"),
        (),
    ],
)
def test_the_tool_rejects_conflicting_options(tmp_path, args):
    result = _run("--instances", *map(str, _INSTANCES), *args, cwd=tmp_path)
    assert result.returncode == 2, result.stderr
    assert not any(tmp_path.iterdir()), "a rejected invocation wrote output"
