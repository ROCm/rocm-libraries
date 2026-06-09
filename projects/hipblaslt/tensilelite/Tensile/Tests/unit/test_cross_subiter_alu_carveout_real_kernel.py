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
"""Real-kernel pin for cross-subiter Pack3->MFMA edge handling under the
unrolled walk (post C3 / rocm-libraries-si5f).

Companion artifact to bead `rocm-libraries-bwfr` (the cross-subiter ALU
artifact investigation umbrella).  Earlier work pinned this behavior with
synthetic 3-instruction `_FakePack`/`_FakeMFMA` fixtures
(`test_cross_subiter_pack_artifact.py`); this file re-pins it using the
SAME real production kernel build that the production tests in
`test_ScheduleCapture.py` exercise — no synthetic constructions, no
hand-rolled `TaggedInstruction` shapes, no scaffolded edge sets.

What this file now pins (single test):

  `test_real_kernel_validates_clean_with_carveout_engaged` — the TF32
  4x4 TN canonical kernel validates green under `compare_graphs`.  As of
  C3 (rocm-libraries-si5f), the reason is NOT an exemption in
  `diagnose_missing_edge` (that code was deleted in C1 / rocm-libraries-
  5tf9).  The assertion `compare_graphs(ref, subj) == []` holds because
  the unrolled walk emits byte-key-based edge keys (8-field tuple
  including `source_module_id` and `emission_ordinal` from Option E /
  rocm-libraries-56e3).  Both ref and subj resolve the same physical
  Pack3->MFMA byte-key flows from the same physical producer, so the
  keys cancel in set-diff.  The 192-edge (post-hdem-body-collapse)
  PackA3/PackB3->MFMA artifact that formerly required a cross-graph
  exemption is now handled correctly by the principled identity model.

Tombstone — neutralization test deleted in C4 (rocm-libraries-5ryl):

  `test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures`
  was deleted because it tested the behavior of the cross-graph exemption
  in `diagnose_missing_edge`, which no longer exists (deleted C1 /
  rocm-libraries-5tf9).  The monkeypatch of `GraphNode.subiter` it used
  has no effect on `compare_graphs` output post-C1 because the subiter
  check it targeted is gone from the cross-graph path.  The principled
  behavior (0 failures for principled reasons) is already pinned by the
  remaining test.

The `_alu_cross_subiter_passthrough` GapRule in the within-graph timing
path (`_classify_edge_coverage`) is a separate concern tracked by bead
`rocm-libraries-37d3`; it is not evaluated or deleted in C4.
"""

import os
import shutil
import sys

import pytest


CANONICAL_KERNEL_CONFIG = {
    'ProblemType': {
        'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
        'F32XdlMathOp': 'X', 'TransposeA': True, 'TransposeB': False,
        'UseBeta': True, 'Batched': True,
    },
    'MatrixInstruction': [16, 16, 32, 1, 1, 4, 4, 2, 2],
    'DepthU': 32, 'PrefetchGlobalRead': 2, 'PrefetchLocalRead': 1,
    'DirectToLds': 1, 'TransposeLDS': 1, 'LocalReadVectorWidth': 4,
    'GlobalReadVectorWidthA': 4, 'GlobalReadVectorWidthB': 4,
    'UseCustomMainLoopSchedule': 1, 'ExpandPointerSwap': 0,
    'SourceSwap': 1, 'StreamK': 0,
    # rocm-libraries-2bww: required_flags for _get_schedule_128x128x32_TF32
    # ('TN', False, 1) branch — uniform validation requires these to be
    # explicitly supplied in the synthetic config.
    'UseMFMAF32XEmulation': True, 'UsePLRPack': True,
}


@pytest.fixture(scope="module")
def real_kernel_graphs(isa_infrastructure):
    """Build the canonical TF32 4x4 TN kernel through the real production
    `KernelWriterAssembly._getKernelSource` pipeline and return the (ref,
    subj) `DataflowGraph` pair the validator runs against.

    Module-scoped: the kernel build is ~3-5s; we want ONE build per pytest
    module run.

    The build path is exactly the production path:
      1. `_make_solution` reads the canonical config dict (no shortcuts).
      2. `KernelWriterAssembly._getKernelSource(solution)` runs end-to-end,
         emitting the assembly text and (because
         `UseCustomMainLoopSchedule=1` triggers
         `_captureDefaultSchedule`) populating BOTH the shadow default-
         side `FourPartCapture` AND the real CMS-side `FourPartCapture`
         on the writer object (`_last_default_capture`,
         `_last_cms_capture`).
      3. `build_dataflow_graph` consumes those captures (real
         rocisa-emitted instructions, real RegSet directives, real
         schedule slot ids) and produces the two graphs `compare_graphs`
         operates on.

    No instructions are constructed by hand.  No `TaggedInstruction`
    shapes are scaffolded.  No `_FakePack`/`_FakeMFMA` fixtures appear
    anywhere in this file.
    """
    _isa, isaInfoMap, asm = isa_infrastructure

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig
    from Tensile.Components.CMSValidator import build_dataflow_graph

    config = dict(CANONICAL_KERNEL_CONFIG)
    solution = _make_solution(config, asm, isaInfoMap)
    writer = KernelWriterAssembly(asm, DebugConfig())
    writer._getKernelSource(solution)

    default_cap = writer._last_default_capture
    cms_cap = writer._last_cms_capture
    assert default_cap is not None, (
        "Real CMS kernel build did not populate _last_default_capture; "
        "auto-activation in KernelWriter.py expected for "
        "UseCustomMainLoopSchedule=1.")
    assert cms_cap is not None, (
        "Real CMS kernel build did not populate _last_cms_capture.")

    ref_graph = build_dataflow_graph(default_cap)
    subj_graph = build_dataflow_graph(cms_cap)
    return ref_graph, subj_graph


def test_real_kernel_validates_clean_with_carveout_engaged(real_kernel_graphs):
    """Production behavior pin.

    The TF32 4x4 TN canonical kernel validates green end-to-end under
    `compare_graphs`.  As of C3 (rocm-libraries-si5f), the assertion
    `compare_graphs(ref, subj) == []` holds for the principled reason that
    the unrolled walk's byte-key edge keys (8-field tuple including
    `source_module_id` and `emission_ordinal` from Option E /
    rocm-libraries-56e3) cancel in set-diff: both ref and subj resolve the
    same physical Pack3->MFMA byte-key flows from the same physical
    producer.  No cross-graph exemption is involved — the exemption in
    `diagnose_missing_edge` was deleted in C1 (rocm-libraries-5tf9).
    """
    from Tensile.Components.CMSValidator import compare_graphs

    ref_graph, subj_graph = real_kernel_graphs
    failures = compare_graphs(ref_graph, subj_graph)

    assert failures == [], (
        f"Real-kernel validation should be clean with the carve-out engaged; "
        f"got {len(failures)} failures: "
        f"{[type(f).__name__ for f in failures[:5]]}")

