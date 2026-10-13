# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import sys
import textwrap
from pathlib import Path

import pytest

SLO = Path(__file__).resolve().parents[1]
ROCKE = SLO.parents[1]
sys.path.insert(0, str(SLO))


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
