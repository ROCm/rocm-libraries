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
"""Source-aware-rendering pinning tests for the asm bridge stub
(rocm-libraries-3dy, 5gd.B.1).

These tests verify the asm-source half of the source-aware Failure-rendering
contract: when a Failure is constructed using `AsmLabelRenderer`-derived
labels, the formatter must produce asm-native rendering (mnemonic +
operands; `@ asm_line=N` position), NEVER CMS-native rendering
(`category[N]`; `@ idx=N` position).

Why this test matters: `Failure._format_canonical` is shared across all
sources. The Protocol-based dispatch relies on the formatter being
source-agnostic — it must consume `(primary, position)` strings and not
care which source emitted them. These tests pin that contract.

The CMS-side counterpart (byte-identical preservation of the existing
CMS rendering) lives in `test_failure_formatters.py` (the CMS tests
unchanged after this bead, which is the safety net).
"""

import pytest

from Tensile.Components.asm_to_timeline_renderers import AsmLabelRenderer
from Tensile.Components.CMSValidator import (
    FailureNodeLabel,
    InvalidCounterValueFailure,
    MissingBarrierFailure,
    MissingWaitFailure,
    OrderInvertedFailure,
    OverriddenInputFailure,
    TimingTooCloseFailure,
    WaitInsufficientFailure,
    _PositionStr,
)
from Tensile.Components.ScheduleCapture import SchedulePosition


def _asm_label(rendered_inst: str, asm_line: int) -> FailureNodeLabel:
    """Build a FailureNodeLabel via the asm-side renderer Protocol surfaces.

    Mirrors what an asm-side label provider (the eventual sibling of
    `cms_node_label`, to be added when `assembly_to_timeline` lands)
    would do: construct an AsmLabelRenderer, call its `render()` and
    `render_position()` methods, and stuff the returned strings into
    `FailureNodeLabel`. The `category` / `body_label` fields are left
    unset — asm has no CMS-style category concept.

    The Protocol surfaces (and only the Protocol surfaces) are what the
    formatter sees through; constructing the label this way exercises
    the contract end-to-end.
    """
    renderer = AsmLabelRenderer(rendered_inst=rendered_inst, asm_line=asm_line)
    return FailureNodeLabel(
        primary=renderer.render(),
        position=_PositionStr(renderer.render_position()),
    )


def _asm_position(asm_line: int, rendered_inst: str = "s_waitcnt vmcnt(0)") -> str:
    """Helper: produce an asm-native position string via the renderer.

    Used for the `wait_position` / `swait_position` / `nearby_wait_positions`
    fields that hold pre-rendered position strings (see the source-aware
    audit in rocm-libraries-3dy). The `rendered_inst` is unused by
    `render_position()` but the renderer's invariant requires both fields,
    so a SWaitCnt-shaped placeholder default is supplied.
    """
    return AsmLabelRenderer(rendered_inst=rendered_inst, asm_line=asm_line).render_position()


# =============================================================================
# Asm-source rendering: each Failure type goes through formatter
# =============================================================================


def test_order_inverted_failure_renders_asm_native():
    """OrderInverted with asm-source labels: mnemonic + operands for both
    primary fields, `@ asm_line=N` for both position fields. NEVER
    `category[N]` or `@ idx=N`."""
    failure = OrderInvertedFailure(
        producer=_asm_label("buffer_load_dword v[34], s[16:19], 0 offen offset:128", 17),
        consumer=_asm_label("v_mfma_f32_16x16x16f16 a[0:3], v[4:5], v[8:9], a[0:3]", 92),
        default_producer_position=SchedulePosition(loop_index=1, stream_index=0),
        default_consumer_position=SchedulePosition(loop_index=1, stream_index=5),
    )
    msg = failure.format()
    assert "buffer_load_dword v[34], s[16:19], 0 offen offset:128 @ asm_line=17" in msg
    assert "v_mfma_f32_16x16x16f16 a[0:3], v[4:5], v[8:9], a[0:3] @ asm_line=92" in msg
    # Negative assertions: no CMS-shape leakage.
    assert "@ idx=" not in msg
    assert "LRA" not in msg
    assert "PackA" not in msg
    # Standard prose still applies.
    assert "is issued after consumer" in msg
    assert msg.startswith("Producer ")


def test_missing_wait_failure_renders_asm_native_with_nearby_position():
    """MissingWait with asm labels and an asm-native nearby-SWaitCnt
    position. The pre-rendered nearby_wait_positions tuple holds asm-side
    strings (`@ asm_line=N`), and the formatter's hint phrasing strips the
    leading `@ ` for inline reading."""
    failure = MissingWaitFailure(
        producer=_asm_label("ds_load_b128 v[0:3], v255 offset:0", 42),
        consumer=_asm_label("v_mfma_f32_16x16x16 a[0:3], v[4:5], v[8:9], a[0:3]", 58),
        counter_kind="dscnt",
        nearby_wait_positions=(_asm_position(50),),
    )
    msg = failure.format()
    expected = (
        "SWaitCnt(dscnt) missing between producer "
        "ds_load_b128 v[0:3], v255 offset:0 @ asm_line=42 and consumer "
        "v_mfma_f32_16x16x16 a[0:3], v[4:5], v[8:9], a[0:3] @ asm_line=58 "
        "(existing SWaitCnts at asm_line=50 drain other counters)."
    )
    assert msg == expected
    assert "@ idx=" not in msg


def test_wait_insufficient_failure_renders_asm_native():
    """WaitInsufficient: `wait_position` is the pre-rendered asm-native
    position string of the failing SWaitCnt — formatter splices it
    verbatim after `SWaitCnt`."""
    failure = WaitInsufficientFailure(
        producer=_asm_label("ds_load_b128 v[0:3], v255 offset:0", 42),
        consumer=_asm_label("v_mfma_f32_16x16x16 a[0:3], v[4:5], v[8:9], a[0:3]", 58),
        wait_position=_asm_position(50),
        counter_kind="dscnt", counter_value=4,
        queue_depth_at_wait=5, producer_position=2,
    )
    msg = failure.format()
    assert msg == (
        "dscnt for SWaitCnt @ asm_line=50 is too high to guarantee producer "
        "ds_load_b128 v[0:3], v255 offset:0 @ asm_line=42 for consumer "
        "v_mfma_f32_16x16x16 a[0:3], v[4:5], v[8:9], a[0:3] @ asm_line=58. "
        "Current value of 4 must be in range [0, 2]."
    )
    assert "@ idx=" not in msg


def test_missing_barrier_failure_renders_asm_native():
    """MissingBarrier: `wait_position` is pre-rendered asm-native string."""
    failure = MissingBarrierFailure(
        producer=_asm_label("ds_load_b128 v[0:3], v255 offset:0", 42),
        consumer=_asm_label("buffer_store_dword v[34], s[16:19], 0 offen", 80),
        wait_position=_asm_position(50),
    )
    msg = failure.format()
    assert msg == (
        "SBarrier missing between SWaitCnt @ asm_line=50 and consumer "
        "buffer_store_dword v[34], s[16:19], 0 offen @ asm_line=80, "
        "needed for producer ds_load_b128 v[0:3], v255 offset:0 @ asm_line=42."
    )
    assert "@ idx=" not in msg


def test_invalid_counter_value_failure_renders_asm_native():
    """InvalidCounterValue: `swait_position` is pre-rendered asm-native."""
    failure = InvalidCounterValueFailure(
        swait_position=_asm_position(99),
        dscnt=-2, vlcnt=0, vscnt=-3,
    )
    msg = failure.format()
    assert msg == (
        "SWaitCnt @ asm_line=99 is invalid: dscnt=-2, vscnt=-3. "
        "All counter fields must be >= -1."
    )
    assert "@ idx=" not in msg


def test_overridden_input_failure_renders_asm_native():
    """OverriddenInput: all three labels (producer, consumer,
    intervening_writer) routed through asm renderer."""
    failure = OverriddenInputFailure(
        producer=_asm_label("v_pack_b32_f16 v34, v36, v38", 100),
        consumer=_asm_label("v_mfma_f32_16x16x16 a[0:3], v[34:35], v[36:37], a[0:3]", 110),
        resource="vgpr",
        intervening_writer=_asm_label("v_mov_b32 v34, v40", 105),
    )
    msg = failure.format()
    assert msg == (
        "v_mov_b32 v34, v40 @ asm_line=105 is incorrectly scheduled between producer "
        "v_pack_b32_f16 v34, v36, v38 @ asm_line=100 and consumer "
        "v_mfma_f32_16x16x16 a[0:3], v[34:35], v[36:37], a[0:3] @ asm_line=110, "
        "clobbering the vgpr that the consumer needs."
    )
    assert "@ idx=" not in msg
    assert "PackA" not in msg


def test_timing_too_close_failure_renders_asm_native():
    """TimingTooClose: producer + consumer asm-native; quad-cycle counts
    are source-agnostic scalars (unchanged)."""
    failure = TimingTooCloseFailure(
        producer=_asm_label("v_pack_b32_f16 v34, v36, v38", 100),
        consumer=_asm_label("v_mfma_f32_16x16x16 a[0:3], v[34:35], v[36:37], a[0:3]", 102),
        expected_quad_cycles=2, actual_quad_cycles=1,
    )
    msg = failure.format()
    assert "v_pack_b32_f16 v34, v36, v38 @ asm_line=100" in msg
    assert "v_mfma_f32_16x16x16 a[0:3], v[34:35], v[36:37], a[0:3] @ asm_line=102" in msg
    assert "Need at least 2 quad-cycles but only 1 guaranteed." in msg
    assert "@ idx=" not in msg


# =============================================================================
# Round-trip via the AsmLabelRenderer Protocol surfaces (Task 7 spec)
# =============================================================================


def test_asm_renderer_round_trip_no_per_source_branching_in_formatter():
    """End-to-end Protocol round-trip: build a Failure with labels produced
    by AsmLabelRenderer.render() / render_position() (NOT bare strings),
    then assert the formatter produces asm-native shape.

    This is the "build a Timeline via the asm-renderer stub, force a
    Failure, assert the rendered message uses asm-native shape" spec from
    rocm-libraries-3dy Task 7. The asm bridge itself (`assembly_to_timeline`)
    hasn't landed (`rocm-libraries-x6s`), but the renderer Protocol IS
    already viable end-to-end through this path.
    """
    producer_renderer = AsmLabelRenderer(
        rendered_inst="ds_load_b128 v[0:3], v255 offset:0",
        asm_line=42,
    )
    consumer_renderer = AsmLabelRenderer(
        rendered_inst="v_mfma_f32_16x16x16f16 a[0:3], v[4:5], v[8:9], a[0:3]",
        asm_line=58,
    )

    # The label-construction step (eventual asm-side counterpart of
    # `cms_node_label`) calls the renderer's Protocol surfaces and stuffs
    # the returned strings into FailureNodeLabel. The formatter then has
    # ZERO knowledge of which source emitted the timeline.
    producer_label = FailureNodeLabel(
        primary=producer_renderer.render(),
        position=_PositionStr(producer_renderer.render_position()),
    )
    consumer_label = FailureNodeLabel(
        primary=consumer_renderer.render(),
        position=_PositionStr(consumer_renderer.render_position()),
    )

    failure = MissingWaitFailure(
        producer=producer_label,
        consumer=consumer_label,
        counter_kind="dscnt",
    )
    msg = failure.format()

    # Asm-native shape end-to-end:
    assert "ds_load_b128 v[0:3], v255 offset:0 @ asm_line=42" in msg
    assert "v_mfma_f32_16x16x16f16 a[0:3], v[4:5], v[8:9], a[0:3] @ asm_line=58" in msg
    # NOT CMS shape:
    assert "@ idx=" not in msg
    assert "LRA0[" not in msg
    assert "MFMA[" not in msg


def test_position_protocol_does_not_lie_about_mfma_slot():
    """The asm-side renderer must NOT produce a fake `@ idx=N` — asm has
    no MFMA-slot concept. Per the bead's design directive: 'Don't fake
    @ idx=N for asm — it has no MFMA-slot concept and the lie surfaces
    in failure messages.'"""
    renderer = AsmLabelRenderer(
        rendered_inst="v_mfma_f32_16x16x16f16 a[0:3], v[4:5], v[8:9], a[0:3]",
        asm_line=200,
    )
    pos = renderer.render_position()
    assert pos == "@ asm_line=200"
    assert "@ idx=" not in pos
