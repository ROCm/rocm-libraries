################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Renderer tests for the PRE_LOOP / POST_LOOP `@-1` disambiguation
(rocm-libraries-aprv).

`_make_node` writes `node.name = f"{category}@{mfma_index}.{sequence}"`,
which is `slot_kind`-independent. Two distinct kernel-writer code paths
both stamp `mfma_index=-1`:

  - PRE_LOOP path: pre-MFMA leaves in the iter stream and the
    `BODY_LABEL_PROLOGUE` body capture (KernelWriter.py:2666-2683 +
    :5138-5144).
  - POST_LOOP path: leftover-pack capture at the end of `_loopBody`'s
    shadow-emit branch (KernelWriter.py:4624-4628).

`LoopBodyCaptureBuilder.append` keys the per-bucket sequence counter on
`(slot_kind, mfma_index)`, so PRE_LOOP@-1 and POST_LOOP@-1 share
sequence numbers (each starts at 0). Without disambiguation, two distinct
events render as the same `node.name` string in the dump tool and
matplotlib visualization.

Per the Q6 investigation memo (Approach A), `node.name` itself stays
byte-stable; the dump-tool / visualization renderers call
`render_node_label(node)` to consume `tagged_inst.slot.slot_kind` and
substitute `@PRE-1.X` / `@POST-1.X` for the colliding `@-1.X`. These
tests pin the exact rendered strings so future drift is caught.
"""

import pytest

from rocisa.instruction import SNop

from Tensile.Components.ScheduleCapture import (
    SLOT_KIND_MFMA,
    SLOT_KIND_PRE_LOOP,
    SLOT_KIND_POST_LOOP,
    SlotKey,
    TaggedInstruction,
    WrappedInstruction,
    BODY_LABEL_ML,
    BODY_LABEL_PROLOGUE,
)
from Tensile.Components.CMSValidator import (
    GraphNode,
    make_position,
    render_node_label,
)


def _make_node(*, category: str, slot_kind: str, mfma_index: int,
               sequence: int, body_label: str = BODY_LABEL_ML) -> GraphNode:
    """Construct a synthetic GraphNode with the requested slot fields.

    Mirrors the shape `_make_node` produces in `build_dataflow_graph`:
    the renderer reads `node.tagged_inst.slot.slot_kind`, so the test
    only needs a TaggedInstruction whose slot carries the desired
    `slot_kind`. Uses an SNop as a cheap wrapped-rocisa stand-in
    (no operands; same idiom as `test_ScheduleCapture._opaque_inst`).
    """
    inst = SNop(waitState=0)
    wrapped = WrappedInstruction(inst)
    slot = SlotKey(subiter=0, slot_kind=slot_kind,
                   mfma_index=mfma_index, sequence=sequence)
    ti = TaggedInstruction(wrapped=wrapped, category=category, slot=slot)
    name = f"{category}@{slot.mfma_index}.{slot.sequence}"
    return GraphNode(
        identity=ti.identity_for(body_label),
        position=make_position(body_label, stream_index=sequence),
        category=category,
        rocisa_inst=inst,
        tagged_inst=ti,
        body_label=body_label,
        name=name,
        issue_cycles=1,
    )


class TestRenderNodeLabel:
    """Pin the exact strings emitted by `render_node_label` for each
    slot_kind. Catches drift if a future renderer change accidentally
    re-introduces the PRE/POST collision or alters the encoding.
    """

    def test_mfma_slot_passes_through_unchanged(self):
        # MFMA slots are the common case and must stay byte-stable —
        # any prefix would visibly degrade `MFMA@5.0` to e.g.
        # `MFMA@MFMA5.0`.
        node = _make_node(category="MFMA", slot_kind=SLOT_KIND_MFMA,
                          mfma_index=5, sequence=0)
        assert render_node_label(node) == "MFMA@5.0"
        # Round-trip with `node.name` is unchanged for MFMA-slot nodes.
        assert render_node_label(node) == node.name

    def test_pre_loop_at_minus_one_renders_with_PRE_prefix(self):
        node = _make_node(category="PackA0", slot_kind=SLOT_KIND_PRE_LOOP,
                          mfma_index=-1, sequence=13)
        # node.name remains the slot_kind-independent base form so any
        # in-process consumer that holds a GraphNode reference is
        # unaffected — the disambiguation is renderer-only.
        assert node.name == "PackA0@-1.13"
        assert render_node_label(node) == "PackA0@PRE-1.13"

    def test_post_loop_at_minus_one_renders_with_POST_prefix(self):
        node = _make_node(category="PackA3", slot_kind=SLOT_KIND_POST_LOOP,
                          mfma_index=-1, sequence=13)
        assert node.name == "PackA3@-1.13"
        assert render_node_label(node) == "PackA3@POST-1.13"

    def test_pre_and_post_loop_with_same_category_and_sequence_are_distinguishable(self):
        """The core regression: PRE_LOOP@-1 and POST_LOOP@-1 with the
        SAME category and SAME sequence (which can occur when the
        leftover-pack shadow emit and the iter-stream pre-MFMA capture
        both produce a `PackA{u}` leaf for the same `u` — see memo §2)
        must render distinctly even though `node.name` collides.
        """
        pre = _make_node(category="PackA0", slot_kind=SLOT_KIND_PRE_LOOP,
                         mfma_index=-1, sequence=5)
        post = _make_node(category="PackA0", slot_kind=SLOT_KIND_POST_LOOP,
                          mfma_index=-1, sequence=5)
        # Field-level collision verifies the failure mode the renderer
        # is fixing.
        assert pre.name == post.name == "PackA0@-1.5"
        # Renderer disambiguates.
        pre_label = render_node_label(pre)
        post_label = render_node_label(post)
        assert pre_label != post_label
        assert pre_label == "PackA0@PRE-1.5"
        assert post_label == "PackA0@POST-1.5"

    def test_pre_loop_in_prologue_body_renders_with_PRE_prefix(self):
        """PRE_LOOP entries from the BODY_LABEL_PROLOGUE capture
        (KernelWriter.py:5138-5144 -> ScheduleCapture.build_prologue_capture)
        share the same renderer path as in-iter PRE_LOOP entries; pin the
        rendering for the prologue-body case explicitly.
        """
        node = _make_node(category="PackA0", slot_kind=SLOT_KIND_PRE_LOOP,
                          mfma_index=-1, sequence=0,
                          body_label=BODY_LABEL_PROLOGUE)
        assert render_node_label(node) == "PackA0@PRE-1.0"

    def test_pre_loop_with_nonnegative_mfma_index_still_disambiguated(self):
        """Defensive: even though production PRE_LOOP entries always
        carry `mfma_index=-1`, the renderer's behavior is uniform on
        slot_kind, not on the index value. Pin that the prefix encoding
        is stable for any mfma_index.
        """
        node = _make_node(category="LRA0", slot_kind=SLOT_KIND_PRE_LOOP,
                          mfma_index=2, sequence=4)
        assert render_node_label(node) == "LRA0@PRE2.4"
