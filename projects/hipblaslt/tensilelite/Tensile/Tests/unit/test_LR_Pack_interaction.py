################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
"""Cross-cutting LR/Pack negative regression tests (graph-native).

Migrated from the legacy ``CMSValidationTestBase`` shape that ran
``[add_local_read_constraints, add_pack_constraints]``. The legacy LR
rule and its ``LR.needed_by`` machinery have been removed; the equivalent
coverage graph-side is the LR -> MFMA RAW edge classified by
``validate_edge_wait_coverage``, which emits ``MissingWaitFailure`` on
dscnt when no qualifying SWaitCnt sits in the producer -> consumer window.

This file covered Category 4 from the validator regression-test plan:
schedules where BOTH the LR and the Pack are positioned after the MFMA
that consumes their data, AND the LR is strictly before the Pack and
guaranteed before it. The LR check was the one that fired first
because instructions validate in linear issue order.

Graph-native equivalent: place the LR after its consumer MFMA (no SWait
in window) -> the LR -> MFMA edge has no covering wait -> the
classifier emits a dscnt failure on that edge regardless of the Pack
ordering. The pack-side coverage of the same shape lives in
``test_validate_pack_graph.py``.
"""

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML,
    MissingWaitFailure,
)
from dataflow_fixtures import (
    make_capture, make_lr, make_mfma, make_swait,
)
from graph_native_validation_base import GraphNativeValidationTest


class TestLRAndPackAfterMFMA(GraphNativeValidationTest):
    """LR positioned after the consumer MFMA in stream order.

    Mirrors legacy ``test_LR0_both_after_MFMA_LR_before_Pack``: the LR
    finishes (graph: produces) AFTER the MFMA that reads its register.
    With no SWait sitting in the LR -> MFMA window, the dscnt drain is
    missing on that edge. The Pack node is irrelevant to the LR-side
    failure: even without a Pack chained between LR and MFMA, the
    LR -> MFMA RAW edge alone surfaces ``MissingWaitFailure(dscnt)``.

    Pack-after-MFMA negative coverage is intentionally NOT exercised
    here — that is ola.4 territory (see ``test_validate_pack_graph.py``).
    """

    def test_LR0_late_lr_no_swait_in_window(self):
        """LRA0 @ slot 6 writes v8; MFMA @ slot 7 reads v8.
        No SWait sits between the LR (slot 6) and the MFMA (slot 7) ->
        the dscnt counter is undrained on the LR -> MFMA edge ->
        ``MissingWaitFailure(counter_kind='dscnt')``.

        Mirrors the legacy LR.validate firing first because the LR check
        runs in linear issue order and the LR sits BEFORE the Pack
        which would otherwise also fail.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=6, category="LRA0"),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=7, a_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        f = self.assert_failures_contain(
            failures, cls=MissingWaitFailure, counter_kind="dscnt",
        )
        # Pin the failure: producer must be the LRA0 (graph identity);
        # consumer must be the MFMA reading vgpr 8.
        assert f.producer.category == "LRA0", (
            f"expected LRA0 as failing producer, got {f.producer.category}"
        )
        assert f.consumer.category == "MFMA", (
            f"expected MFMA as failing consumer, got {f.consumer.category}"
        )

    def test_LR0_baseline_passing(self):
        """Same skeleton with the LR positioned BEFORE the consumer
        MFMA and an SWait(dscnt=0) in the LR -> MFMA window. No failure.
        """
        cap = make_capture(BODY_LABEL_ML, [
            make_lr(dst_vgpr_start=8, dst_vgpr_count=2, lds_offset=64,
                    slot=0, category="LRA0"),
            make_swait(slot=2, dscnt=0),
            make_mfma(c_dst_start=0, a_src_start=8, b_src_start=32,
                      slot=4, a_src_count=1),
        ])
        failures = self.validate_waits(self.build_graph(
            self.wrap_single_body(cap)))
        self.assert_no_failures(failures)
