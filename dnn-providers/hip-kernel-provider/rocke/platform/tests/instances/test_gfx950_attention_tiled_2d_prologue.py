# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx950 tiled 2D attention C++ port has one build prologue.

The driver TU builds the context with its file-local
``rocke_g950_build_ctx_init_local``. An external
``rocke_gfx950_attn2d_build_ctx_init`` used to sit beside it, declared in the
internal header and defined in ``attention_tiled_2d_ctx_prologue.cpp``, with
no caller. A change to the prologue then had two copies to keep in step, and
the unused one was easy to edit by mistake.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CPP = Path(__file__).resolve().parents[2] / "cpp"
GLUE = CPP / "instances" / "gfx950" / "attention_tiled_2d_public_entry_glue.cpp"

pytestmark = pytest.mark.skipif(
    not GLUE.exists(),
    reason="reads the platform sources, which an installed test layout omits",
)


def test_the_unused_external_prologue_stays_removed():
    users = [
        path.relative_to(CPP).as_posix()
        for path in CPP.rglob("*")
        if path.suffix in {".c", ".cpp", ".h", ".hpp"}
        and "rocke_gfx950_attn2d_build_ctx_init"
        in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert users == []


def test_the_driver_defines_its_prologue_once_and_file_local():
    src = GLUE.read_text(encoding="utf-8")
    definitions = re.findall(
        r"^(static\s+)?bool\s+rocke_g950_build_ctx_init_local\(", src, re.MULTILINE
    )
    assert definitions == ["static "]
