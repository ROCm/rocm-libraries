# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SLO = Path(__file__).resolve().parents[1]
ROCKE = SLO.parents[1]
sys.path.insert(0, str(SLO))


# Folders a copy of the real tree needs: the measured code, the helpers, the IR harness.
COPY = ("library", "platform/python", "platform/tests")


def copy_tree(top: Path) -> Path:
    """Copy the real rocke tree to top/rocke and commit it in a new git repo at top."""
    root = top / "rocke"
    for d in COPY:
        shutil.copytree(
            ROCKE / d, root / d, ignore=shutil.ignore_patterns("__pycache__")
        )
    for a in (
        ["init", "-q", "-b", "main"],
        ["config", "user.email", "t@t"],
        ["config", "user.name", "t"],
        ["add", "-A"],
        ["commit", "-qm", "base"],
    ):
        subprocess.run(["git", "-C", str(top), *a], check=True, capture_output=True)
    return root


def src(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def kernel_text(n_raw=3, extra_fn_lines=0):
    body = "\n".join(f"    x{i} = b.add(x, {i})" for i in range(n_raw))
    pad = "\n".join(f"    y{i} = {i}" for i in range(extra_fn_lines))
    return f"def build(b, x):\n{body}\n{pad}\n    return x\n"


@pytest.fixture
def fake_rocke(tmp_path):
    """A tiny rocke tree: one helpers module, one kernel folder, one instances folder."""
    root = tmp_path / "rocke"
    (root / "platform/python/rocke/helpers").mkdir(parents=True)
    (root / "platform/python/rocke/helpers/attention.py").write_text(
        "def mask(b, x):\n    return x\n"
    )
    (root / "platform/python/rocke/instances/common").mkdir(parents=True)
    (root / "library/kernels/common").mkdir(parents=True)
    return root
