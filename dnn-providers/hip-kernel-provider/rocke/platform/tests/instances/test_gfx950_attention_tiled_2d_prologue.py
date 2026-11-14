# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx950 tiled 2D attention C++ port has one build prologue.

The driver TU builds the context with its file-local
``rocke_gfx950_attn2d_build_ctx_init`` (the name the other families use for
their prologue). A second, external copy used to sit beside it, declared in the
internal header and defined in ``attention_tiled_2d_ctx_prologue.cpp``, with no
caller. A change to the prologue then had two copies to keep in step, and the
unused one was easy to edit by mistake. The check is by signature, so a copy
under another name is found too.
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

# bool NAME(rocke_gfx950_attn2d_build_ctx_t*, rocke_ir_builder_t*,
#           const rocke_attention_tiled_2d_spec_t*, const char*)
_PROLOGUE = re.compile(
    r"\bbool\s+(\w+)\s*\(\s*"
    r"rocke_gfx950_attn2d_build_ctx_t\s*\*\s*\w*\s*,\s*"
    r"rocke_ir_builder_t\s*\*\s*\w*\s*,\s*"
    r"const\s+rocke_attention_tiled_2d_spec_t\s*\*\s*\w*\s*,\s*"
    r"const\s+char\s*\*\s*\w*\s*\)\s*([{;])"
)


def _code(path: Path) -> str:
    """The file without its comments."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.DOTALL)


def _prologues() -> list[tuple[str, str, str]]:
    """(file, name, '{' for a definition or ';' for a declaration)."""
    return [
        (path.relative_to(CPP).as_posix(), m.group(1), m.group(2))
        for path in sorted(CPP.rglob("*"))
        if path.suffix in {".c", ".cpp", ".h", ".hpp"}
        for m in _PROLOGUE.finditer(_code(path))
    ]


def test_there_is_one_prologue_and_it_is_file_local():
    glue = GLUE.relative_to(CPP).as_posix()
    assert _prologues() == [(glue, "rocke_gfx950_attn2d_build_ctx_init", "{")]
    assert re.search(
        r"^static\s+bool\s+rocke_gfx950_attn2d_build_ctx_init\(",
        _code(GLUE),
        re.MULTILINE,
    )


def test_the_driver_calls_it():
    assert re.search(
        r"!\s*rocke_gfx950_attn2d_build_ctx_init\(\s*&ctx\s*,\s*b\s*,\s*spec\s*,\s*arch\s*\)",
        _code(GLUE),
    )
