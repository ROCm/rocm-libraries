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
"""Real-kernel pin for the cross-subiter ALU producer carve-out.

Companion artifact to bead `rocm-libraries-bwfr` (the cross-subiter ALU
artifact investigation umbrella).  Earlier work pinned this behavior with
synthetic 3-instruction `_FakePack`/`_FakeMFMA` fixtures
(`test_cross_subiter_pack_artifact.py`); this file re-pins it using the
SAME real production kernel build that the 10 production tests in
`test_ScheduleCapture.py` exercise — no synthetic constructions, no
hand-rolled `TaggedInstruction` shapes, no scaffolded edge sets.

What the carve-out is (verified against current `CMSValidator.py`):

* Live in TWO mirrored places — both must be neutralized to surface the
  artifact edges as `OrderInvertedFailure`s:

  1. `diagnose_missing_edge` (cross-graph), CMSValidator.py around line 3448::

         if (_is_alu_producer(p_node)
                 and p_node.subiter(nmps) != c_node.subiter(nmps)):
             return []  # cross-subiter pipelined dependency — legitimate

     Gates `OrderInvertedFailure` emission for the missing-edge / order-
     inversion path.

  2. The `_alu_cross_subiter_passthrough` GapRule in
     `_build_cdna4_gap_rules` (CMSValidator.py around line 725) with
     `condition="cross_subiter_alu_artifact"`, evaluated by
     `_evaluate_gap_rule_condition` (line 2610) — fires when
     `nmps > 0 AND p.subiter(nmps) != c.subiter(nmps)`.

  Both predicates depend on `GraphNode.subiter(num_mfma_per_subiter)` to
  decide whether two nodes live in different subiterations.  Forcing
  `GraphNode.subiter` to return a constant 0 collapses every node into
  the same notional subiter — the `subiter(p) != subiter(c)` test is
  then always False, both carve-outs decline to fire, and the legitimate
  cross-subiter Pack -> MFMA pipelining surfaces as `OrderInvertedFailure`.

The neutralization technique: a single `monkeypatch.setattr` against
`GraphNode.subiter`.  This is the principled pivot point — both carve-out
sites (the within-graph gap rule and the cross-graph diagnose path)
reach the same method, so one patch disables both consistently.

Prior session description vs. what the code actually does (verified):

  Item                | Claimed                                 | Actual
  --------------------|-----------------------------------------|----------------------------------
  File                | CMSValidator.py                         | CMSValidator.py (confirmed)
  Line                | ~2532-2546                              | ~3438-3450 (diagnose_missing_edge)
                      |                                         | ~725-737 (gap-rule companion)
  Predicate           | _is_alu_producer(p) AND                 | Same predicate (confirmed) plus
                      | p.subiter != c.subiter                  | parallel gap-rule entry
                      |                                         | "cross_subiter_alu_artifact"
                      |                                         | (rocm-libraries-vmua landed it)
  Failure type gated  | OrderInvertedFailure                    | OrderInvertedFailure (confirmed)
  Real schedule       | TF32 4x4 TN, MI=16x16x32 4x4 wave tile  | Same (confirmed)
  Edge count off      | 768                                     | 768 (confirmed exact)
  Edge shape off      | PackA3[N]/PackB3[N] -> MFMA             | Same (100%; all 768 are
                      |                                         | this shape, 32-deep histogram)

Discrepancy of note: the prior description's line numbers (~2532-2546)
are stale — the carve-out moved to ~3438-3450 in `diagnose_missing_edge`,
and post `rocm-libraries-vmua` a parallel `cross_subiter_alu_artifact`
GapRule mirrors the same condition on the within-graph
`_classify_edge_coverage` path.  Both must be neutralized (one
`GraphNode.subiter` patch covers both).
"""

import os
import shutil
import sys
from collections import Counter

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
}


@pytest.fixture(scope="module")
def real_kernel_graphs(isa_infrastructure):
    """Build the canonical TF32 4x4 TN kernel through the real production
    `KernelWriterAssembly._getKernelSource` pipeline and return the (ref,
    subj) `DataflowGraph` pair the validator runs against.

    Module-scoped: the kernel build is ~3-5s; we want ONE build per pytest
    module run, shared across both with-carveout / without-carveout tests.

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


def _is_pack_n_to_mfma_artifact(failure):
    """Predicate matching the documented artifact shape:
    `PackA3[N] -> MFMA` or `PackB3[N] -> MFMA` failures, both within the
    same body.  Subiter index 3 is the trailing subiter in this kernel
    (4 subiters per inner unroll: 0..3); the artifact is exclusively
    PackA3 / PackB3 because subiter 3's Pack legitimately issues after
    earlier-subiter MFMAs in the CMS schedule.
    """
    if type(failure).__name__ != "OrderInvertedFailure":
        return False
    p_cat = getattr(failure.producer, "category", "") or ""
    c_cat = getattr(failure.consumer, "category", "") or ""
    p_body = getattr(failure.producer, "body_label", None)
    c_body = getattr(failure.consumer, "body_label", None)
    return (
        (p_cat.startswith("PackA3") or p_cat.startswith("PackB3"))
        and c_cat == "MFMA"
        and p_body == c_body
    )


def test_real_kernel_validates_clean_with_carveout_engaged(real_kernel_graphs):
    """Production behavior pin.

    The TF32 4x4 TN canonical kernel — the one driven by 10 of the
    production tests in `test_ScheduleCapture.py` — must validate green
    end-to-end.  `compare_graphs` returns zero failures because the
    cross-subiter ALU carve-out absorbs the artifact edges.
    """
    from Tensile.Components.CMSValidator import compare_graphs

    ref_graph, subj_graph = real_kernel_graphs
    failures = compare_graphs(ref_graph, subj_graph)

    assert failures == [], (
        f"Real-kernel validation should be clean with the carve-out engaged; "
        f"got {len(failures)} failures: "
        f"{[type(f).__name__ for f in failures[:5]]}")


def test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures(
    real_kernel_graphs, monkeypatch,
):
    """Carve-out neutralization pin.

    Monkey-patch `GraphNode.subiter` to a constant 0.  This collapses
    every node into "subiter 0" so the predicate
    `p.subiter(nmps) != c.subiter(nmps)` (which gates BOTH the
    `diagnose_missing_edge` early-return at CMSValidator.py:~3448 AND
    the `cross_subiter_alu_artifact` GapRule at CMSValidator.py:~725)
    is always False.  Both carve-out sites then decline to fire and the
    legitimate cross-subiter Pack3 -> MFMA pipelining surfaces as
    `OrderInvertedFailure`.

    Pinned exactly:
      - 192 total failures, ALL `OrderInvertedFailure` (no other type
        leaks through).
      - 100% are PackA3[N]/PackB3[N] -> MFMA edges within the same
        body, with each Pack producer feeding 4 or 8 MFMA consumers.
      - The producer node is the body-blind survivor of the
        cross-body identity collapse (ORAM1 §6.2 / hdem Approach A
        + E): `nodes_by_identity` runs last-writer-wins keyed on
        `(canonical_render, emission_ordinal)` (with `loop_index`
        dropped from the leading slot), so the four bodies' identical
        Pack3 producers (one each in ML, ML-1, NGL, NLL with
        identical render-text and identical per-body ordinals)
        collapse to a single node entry. The artifact thus surfaces
        once-per-Pack3-shape rather than four-times-per-Pack3-shape.

    Pre-hdem framing (kept here for the historical record): the test
    pinned 768 failures = 192 × 4 bodies, with a per-body assertion
    `body_counts == {"ML": 192, "ML-1": 192, "NGL": 192, "NLL": 192}`
    and a 128-distinct-(cat, primary, body)-tuple structural
    fingerprint. Under hdem the body discriminator collapses; the
    test now pins the body-blind shape (192 = 64 × 3 nominal,
    actually 64+64 PackA3/PackB3 with 4-or-8 MFMA fan-out).

    If any of those numbers shift, either (a) the carve-out has changed
    behavior, (b) the kernel emit path has changed shape, or (c) the
    `GraphNode.subiter` patch no longer covers both gate sites.  In all
    three cases this test should fail loudly.
    """
    from Tensile.Components.CMSValidator import GraphNode, compare_graphs

    ref_graph, subj_graph = real_kernel_graphs

    # Single principled monkey-patch: `GraphNode.subiter` is the one
    # function both carve-out sites consult to derive cross-vs-same
    # subiter.  Forcing it to 0 disables both gates atomically without
    # touching the validator's wiring.  `monkeypatch.setattr` restores
    # automatically at test teardown.
    monkeypatch.setattr(GraphNode, "subiter", lambda self, nmps: 0)

    failures = compare_graphs(ref_graph, subj_graph)

    # Pin total count (post-hdem A+E body-collapse: was 768 = 192 * 4
    # bodies pre-hdem; bodies' identical Pack3 emissions now collapse
    # to one identity each via last-writer-wins on
    # `nodes_by_identity`).
    assert len(failures) == 192, (
        f"Expected 192 OrderInvertedFailures when carve-out is neutralized "
        f"(post-hdem A+E body-blind identity collapse); got {len(failures)}. "
        f"If this changed, investigate whether the carve-out's predicate, "
        f"the kernel's emitted edge set, or the identity-collapse "
        f"semantics have shifted.")

    # Pin failure-type uniformity.
    type_counts = Counter(type(f).__name__ for f in failures)
    assert type_counts == {"OrderInvertedFailure": 192}, (
        f"Carve-out neutralization should surface ONLY OrderInvertedFailure; "
        f"got mixed types: {dict(type_counts)}")

    # Pin shape: 100% of surfaced failures are PackA3[N]/PackB3[N] -> MFMA.
    artifact_failures = [f for f in failures if _is_pack_n_to_mfma_artifact(f)]
    assert len(artifact_failures) == 192, (
        f"All 192 surfaced failures should be PackA3[N]/PackB3[N] -> MFMA "
        f"(same body); got {len(artifact_failures)} matching the artifact "
        f"shape.  Non-matching examples: "
        f"{[(f.producer, f.consumer) for f in failures if not _is_pack_n_to_mfma_artifact(f)][:3]}")

    # Pin structural fingerprint (post-hdem): 192 failures resolve to
    # exactly 32 distinct (producer-category, producer-primary-label)
    # tuples, with each tuple appearing either 4 or 8 times. The
    # body discriminator dropped out — see the test docstring.
    per_shape = Counter(
        (f.producer.category, getattr(f.producer, "primary", ""))
        for f in artifact_failures
    )
    replica_distribution = Counter(per_shape.values())
    # Each Pack3 producer feeds 4 or 8 MFMA consumers; total = 192.
    total_replicas = sum(k * v for k, v in replica_distribution.items())
    assert total_replicas == 192, (
        f"Replica distribution {dict(replica_distribution)} sums to "
        f"{total_replicas}, expected 192 (= sum of per-shape replica "
        f"counts).")
    assert set(replica_distribution.keys()).issubset({4, 8}), (
        f"Each (cat, primary) tuple should appear 4 or 8 times; "
        f"got distribution {dict(replica_distribution)}.")

    # Pin producer-side categories: only PackA3 and PackB3.
    producer_cats = Counter(f.producer.category for f in artifact_failures)
    assert set(producer_cats.keys()) == {"PackA3", "PackB3"}, (
        f"Expected only PackA3 + PackB3 artifact failures; "
        f"got {dict(producer_cats)}.")
    # Symmetric A/B contribution: each is half of total (96 each).
    assert producer_cats == {"PackA3": 96, "PackB3": 96}, (
        f"Expected 96 PackA3 + 96 PackB3 artifact failures (post-hdem "
        f"body-collapse); got {dict(producer_cats)}.")
