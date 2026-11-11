# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
ApplyMarksPass -- the only BODY mutator among the swap passes.

Inserts each placed PendingMark's Mark into its block body at the resolved anchor:
 - anchor (block, node) -> insert immediately before `node`
"""

from __future__ import annotations

from .base import Pass
from ..nodes import BLOCK_EXIT, BLOCK_ENTRY


class ApplyMarksPass(Pass):
    """Insert each placed Mark into its block body, back to front so anchors stay valid."""
    def run(self, prog, am):
        # group placed marks by block so we can insert into each body once, back-to-front (so
        # earlier insertions don't shift the indices of later anchors).
        by_block = {}
        for pm in prog.pending:
            if pm.anchor is None:
                raise RuntimeError("ApplyMarksPass: PendingMark not placed (run PlacementPass first)")
            block_label, anchor = pm.anchor
            by_block.setdefault(block_label, []).append((anchor, pm.mark))

        for block_label, items in by_block.items():
            body = prog.block(block_label).body
            # resolve each anchor to an insertion index, then insert descending by index
            insertions = []
            for anchor, mark in items:
                idx = self._anchor_index(body, anchor)
                insertions.append((idx, mark))
            for idx, mark in sorted(insertions, key=lambda t: t[0], reverse=True):
                body.insert(idx, mark)

        prog.pending = []
        prog.bump()

    @staticmethod
    def _anchor_index(body, anchor):
        if anchor is BLOCK_EXIT or anchor == BLOCK_EXIT:
            return len(body)
        if anchor is BLOCK_ENTRY or anchor == BLOCK_ENTRY:
            return 0
        for i, node in enumerate(body):
            if node is anchor:
                return i
        # anchor node not found (e.g. its block differs) -> append at end, conservative
        return len(body)
