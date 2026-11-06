# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
RecordUnemittedPass -- count what the model-only blocks are carrying away, LAST.
"""

from __future__ import annotations

from ..nodes import Mark
from .base import Pass


class RecordUnemittedPass(Pass):
    """See module docstring.  Records the discarded-Mark count for model-only blocks."""

    def run(self, prog, am):
        model_only = [b for b in prog.blocks.values() if b.model_only]
        if not model_only:
            return
        dropped = sum(1 for b in model_only for n in b.body
                      if isinstance(n, Mark) and n.kind != "phase_boundary")
        rec = (prog.meta.get("short_loop") or {}).get("model_only")
        if rec is None:
            raise RuntimeError(
                f"blocks {sorted(b.label for b in model_only)} are flagged model_only but no pass "
                f"recorded why -- an unemitted block must be a stated decision, not a flag")
        rec["marks_dropped"] = dropped
