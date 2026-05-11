################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Asm-side renderer stubs (rocm-libraries-3dy, 5gd.B.1 placeholder).

This module provides the asm-source counterparts to the CMS-side
`Tensile.Components.cms_to_timeline.CmsLabelRenderer` so the
`TaggedInstructionLike` Protocol (in `Tensile.Components.Timeline`) is
viable end-to-end before the asm bridge (sub-bead `rocm-libraries-x6s`,
`assembly_to_timeline`) lands.

What lives here today: a minimal `AsmLabelRenderer` that satisfies the
Protocol with asm-native rendering (mnemonic + operands for instructions;
`@ asm_line=N`-style position for source location). The renderer is built
from already-rendered strings supplied by the caller — it does NOT walk
rocisa instructions itself, since the asm bridge that owns that walk
hasn't landed.

What does NOT live here: the asm bridge function (`assembly_to_timeline`),
the rocisa-instruction walker that produces the per-event
`(mnemonic, operands, asm_line)` triples, or any rocisa-source classifier.
Those land with `rocm-libraries-x6s`. When that bead lands, the bridge
constructs `AsmLabelRenderer` instances per event with the walked-out
strings and the renderer's existing surfaces stay unchanged.

Discipline (from the parent bead):
  * No fallback paths. Required arguments are required. If an asm-side
    caller has no asm_line, that's a caller bug — pass a meaningful
    location string or fix the bridge to populate one.
  * No backwards-compat shims. There is no prior asm-side renderer this
    one replaces; this is the first.
  * No per-source branching in formatters. Formatters consume the
    `(primary, position)` strings these renderers produce; they have no
    idea CMS- vs asm-source emitted them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AsmLabelRenderer:
    """Asm-side renderer satisfying `TaggedInstructionLike`.

    Holds two pre-walked strings supplied by the asm bridge
    (`assembly_to_timeline`, sub-bead `rocm-libraries-x6s`):

      * `rendered_inst` — the full asm rendering of one instruction
        (mnemonic + operands), e.g.
        `"buffer_load_dword v[34], s[16:19], 0 offen offset:128"`.
        For a SWaitCnt this is e.g. `"s_waitcnt vmcnt(0) lgkmcnt(0)"`.
      * `asm_line` — the asm-stream line number (or instruction offset,
        whichever the bridge picks; the renderer does not interpret it).
        Rendered as `"@ asm_line={asm_line}"`. The asm bridge picks
        whatever location semantic is most useful for a developer reading
        the failure (line number for an asm dump on disk; instruction
        offset for an in-memory rocisa stream).

    Why both surfaces are pre-rendered strings and not derived from a
    rocisa instance here: this module is a placeholder — the asm bridge
    that owns the rocisa walk hasn't landed yet (`rocm-libraries-x6s`).
    Holding pre-walked strings keeps the renderer trivially testable
    today (the asm-source pinning test in
    `Tests/unit/test_asm_source_failure_rendering.py` uses literal strings)
    and lets the bridge supply whichever rendering convention it wants
    when it lands. The rendering CONVENTION (mnemonic + operands;
    `@ asm_line=N`) is fixed by this Protocol implementation; the SOURCE
    of those strings is the bridge.
    """
    rendered_inst: str
    asm_line: int

    def render(self) -> str:
        """Asm-native instruction label: `mnemonic + operands`."""
        return self.rendered_inst

    def render_position(self) -> str:
        """Asm-native position: `@ asm_line=N`.

        The leading `"@ "` matches the CMS-side `"@ idx=N"` shape so the
        formatter's prose templates ("between SWaitCnt {position} and
        consumer ...") read the same way regardless of source.
        """
        return f"@ asm_line={self.asm_line}"


__all__ = ["AsmLabelRenderer"]
