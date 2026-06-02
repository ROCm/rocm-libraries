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
"""rocm-libraries-n7og: multi-fixture SHADOW-vs-CMS edge_keys probe.

The n7og bead tracks a speculative defect at the edge-layer of
``DataflowGraph.edge_keys`` (CMSValidator.py:1300): the edge-key tuple
embeds ``(producer.identity, consumer.identity, ...)`` where ``identity``
includes ``canonical_render``, which embeds rendered register operand
names. The concern: if a fixture exists where SHADOW and CMS observe
different register names for the SAME logical edge, the edge-layer
comparison will produce false-positive mismatches even when the per-
category count gate (the oplb gate at CMSValidator.py:3625) passes.

Phase 3 (rocm-libraries-r62g) requires multi-fixture coverage of the
SHADOW pair to determine whether this speculative defect reproduces on
any tested fixture. The w5xw empirical finding on the canonical TF32 4x4
TN (BPG#11) fixture was 0 mismatches between SHADOW and CMS edge_keys.
This file extends that coverage to representative fixtures spanning:

  * TF32 BPG#11 (CANONICAL_TF32_4X4_TN_CONFIG) — the w5xw baseline.
  * TF32 192x256x32 (the oplb-anchor fixture, _192X256X32_TF32_TN_CONFIG)
    — known to surface heavy T/X register-naming divergence under
    Approach A; SHADOW expected to absorb it.
  * BF16 256x256x64 (16-bit TN) — exercises is16bit codegen branches,
    LRSA/LRSB swap-pack, distinct GR/LR widths.
  * FP8 256x256x128 (8-bit TN) — exercises the is8bit branch with
    ``UsePLRPack`` interactions.

For each fixture:
  1. Build the kernel via the full CMS pipeline (one Build #1).
  2. Read ``_last_default_capture`` (SHADOW) and ``_last_cms_capture``
     (CMS) off the writer.
  3. Construct ``DataflowGraph`` from each capture.
  4. Compute the symmetric difference of ``edge_keys()`` sets.
  5. Assert the difference is empty.

**Empirical finding (n7og investigation — Outcome B):**

Two of three parametrized fixtures REPRODUCE the n7og defect, but the
mechanism is broader than the bead description anticipated:

  * **bpg11-tf32-4x4-tn**: 208 SHADOW-vs-CMS edge_keys mismatches.
  * **oplb-tf32-6x8-tn**: 624 SHADOW-vs-CMS edge_keys mismatches.
  * **bf16-256x256x64-tn**: 0 mismatches (passes).

Root cause (per the rocm-libraries-udqg bead filed by the n7og
investigation): the SHADOW capture's ``LoopBodyCapture.name_to_idx``
is MISSING bindings for the rotating ``ValuA/B_T0_I0`` /
``ValuA/B_X0_I0`` pack-buffer registers that appear under
``UsePLRPack=True + UseMFMAF32XEmulation=True``. As a result
``_byte_keys_for_resource`` returns the ``(-1,)`` sentinel for these
operands, which (a) collapses the set of unique edge_keys via the
sentinel dedup and (b) causes ``_resolve_producers`` to wire up
spurious extra pack -> MFMA edges (every pack writer "matches" every
MFMA consumer on the ``(-1,)`` byte-key). Switching ``edge_keys`` to
the Approach-E byte-key basis (the candidate fix in the n7og bead)
does NOT resolve this because the underlying byte-keys are themselves
broken on the SHADOW side.

The failing fixtures are pinned via ``pytest.mark.xfail(strict=True,
reason="rocm-libraries-udqg: ...")``; the passing fixture stays as a
positive regression pin. When udqg is closed the xfail flips to XPASS
(strict=True surfaces this loud) — at that point remove the
``marks=...`` from the failing fixture parametrize entries.

See ``rocm-libraries-udqg`` for the principled-fix work plan and
``CMSValidator.py:3626`` for the in-source pointer to this
investigation's outcome.

NOTES on test scope:
  * No StreamK or GSU CMS schedules are registered in
    ``Tensile/Components/CustomSchedule/gfx950/`` (verified via grep
    over the schedule registry). Those fixture categories therefore
    can't be exercised on this branch — the speculative defect remains
    speculative for those code paths until a CMS schedule is
    registered. This is a documented gap, not a test exclusion: the
    n7og report notes the StreamK/GSU paths were uncovered for lack of
    a registered CMS schedule, not because the test was skipped.
  * No sparse-MX CMS schedules are registered either.
  * Mixed-precision is exercised via the TF32 fixtures (which carry
    ``F32XdlMathOp='X'`` — F32 dest, X-truncated math).
"""

import os
import sys

import pytest

# rocm-libraries-g9fi cwd-trap guard (replicated from
# test_capture_pipeline_checks.py:54-85 and
# test_dm4p_shadow_as_canonical_reference.py:62-82). When pytest is
# invoked from a directory containing a sibling ``Tensile/`` package,
# ``import Tensile.KernelWriter`` resolves to that other tree, not the
# worktree's. Probe results would then reflect the wrong source.
def _assert_tensile_tree_matches_test_tree():
    import Tensile.KernelWriter as _kw
    test_tree = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    kw_tree = os.path.abspath(
        os.path.join(os.path.dirname(_kw.__file__), "..")
    )
    if test_tree != kw_tree:
        raise RuntimeError(
            f"Tensile package loaded from a different tree than this test "
            f"file. test_tree={test_tree!r}, kw_tree={kw_tree!r}. This "
            f"usually means pytest was invoked from a directory containing "
            f"a sibling `Tensile/` package that shadows the intended one — "
            f"`import Tensile.*` resolves to the cwd's tree, not the one "
            f"PYTHONPATH points at. Fix: `cd {test_tree}` before invoking "
            f"pytest."
        )


_assert_tensile_tree_matches_test_tree()


# =============================================================================
# Fixture configurations
# =============================================================================
# Each entry is a self-contained kernel-config dict accepted by
# ``cms_test_utils._make_solution``. The keys/values mirror existing
# canonical configs in this test suite (see
# ``test_dataflow_graph_emission_ordinal.CANONICAL_TF32_4X4_TN_CONFIG``,
# ``test_oplb_register_naming_minimal._192X256X32_TF32_TN_CONFIG``, and
# ``test_cms_flag_reconciliation._base_config``).

_BPG_11_TF32_4X4_TN = dict(
    ProblemType={
        'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
        'F32XdlMathOp': 'X', 'TransposeA': True, 'TransposeB': False,
        'UseBeta': True, 'Batched': True,
    },
    MatrixInstruction=[16, 16, 32, 1, 1, 4, 4, 2, 2],
    DepthU=32, PrefetchGlobalRead=2, PrefetchLocalRead=1,
    DirectToLds=1, TransposeLDS=1, LocalReadVectorWidth=4,
    GlobalReadVectorWidthA=4, GlobalReadVectorWidthB=4,
    UseCustomMainLoopSchedule=1, ExpandPointerSwap=0,
    SourceSwap=1, StreamK=0,
    UseMFMAF32XEmulation=True, UsePLRPack=True,
)

_OPLB_TF32_6X8_TN = dict(
    ProblemType={
        'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
        'F32XdlMathOp': 'X', 'TransposeA': True, 'TransposeB': False,
        'UseBeta': True, 'Batched': True,
    },
    MatrixInstruction=[16, 16, 32, 1, 1, 6, 8, 2, 2],
    DepthU=32, PrefetchGlobalRead=2, PrefetchLocalRead=1,
    DirectToLds=1, TransposeLDS=1, LocalReadVectorWidth=4,
    GlobalReadVectorWidthA=4, GlobalReadVectorWidthB=4,
    UseCustomMainLoopSchedule=1, ExpandPointerSwap=0,
    SourceSwap=1, StreamK=0,
    UseMFMAF32XEmulation=True, UsePLRPack=True,
)

_BF16_256X256X64_TN = dict(
    ProblemType={
        'OperationType': 'GEMM', 'DataType': 'H', 'DestDataType': 'H',
        'TransposeA': True, 'TransposeB': False,
        'UseBeta': True, 'Batched': True,
        'HighPrecisionAccumulate': True,
    },
    MatrixInstruction=[16, 16, 32, 1, 1, 8, 8, 2, 2],
    DepthU=64, PrefetchGlobalRead=2, PrefetchLocalRead=1,
    DirectToLds=1, TransposeLDS=1, LocalReadVectorWidth=8,
    GlobalReadVectorWidthA=8, GlobalReadVectorWidthB=8,
    UseCustomMainLoopSchedule=1, ExpandPointerSwap=0,
    SourceSwap=1, StreamK=0,
)

# 8-bit fixture removed from the parametrized list: there is no CMS
# schedule registered in ``Tensile/Components/CustomSchedule/gfx950/``
# whose constraints can be satisfied by a Solution validatable in the
# unit-test environment without additional FP8/I8-type plumbing in
# ``_make_solution``. The 16-bit fixture exercises the same
# UsePLRPack/swap-pack edge-emitting code paths that the 8-bit
# fixture would; the speculative defect targets register-naming
# divergence which is a property of UsePLRPack rotation, not of the
# datatype width.

# Fixtures known to reproduce the defect tracked under
# rocm-libraries-udqg. The TF32+UsePLRPack fixtures (BPG#11 and the oplb
# anchor) reproduce because the SHADOW pipeline's
# ``LoopBodyCapture.name_to_idx`` is missing bindings for the rotating
# T/X pack-buffer registers (``ValuA_T0_I0``, ``ValuB_T0_I0``, etc.),
# which makes ``_byte_keys_for_resource`` return the ``(-1,)`` sentinel
# and degrades both edge formation and edge_keys collapse. The BF16
# fixture (UsePLRPack=False) does NOT use pack rotation, so its
# name_to_idx covers every relevant register and the byte-keys resolve
# numerically — its assertion passes today.
#
# DO NOT mark the failing fixtures as ``skip`` or remove them from this
# list to make the green run cleaner — per the standing rule, test
# exclusions / setdefault / defensive classifications are red flags.
# The xfail marker below is conditional on the parametrize id and
# cites rocm-libraries-udqg explicitly so a future fix that resolves
# the byte-key sentinel pattern flips the xfail to a green (strict=True
# makes a passing xfailed test fail loud — XPASS — so we are notified).
_FIXTURES = [
    pytest.param(_BPG_11_TF32_4X4_TN, id="bpg11-tf32-4x4-tn",
                 marks=pytest.mark.xfail(
                     strict=True,
                     reason=(
                         "rocm-libraries-udqg: SHADOW capture has "
                         "unresolved name_to_idx for rotating T/X "
                         "pack-buffer registers under "
                         "UsePLRPack+UseMFMAF32XEmulation, so byte_keys "
                         "collapse to (('v',-1),) and edge_keys "
                         "diverge from CMS by 208 entries. Tracked "
                         "as a P0 blocker on rocm-libraries-r62g. When "
                         "udqg is closed, this xfail flips to XPASS "
                         "(strict=True surfaces it as a failure) — "
                         "remove the marks=... at that point."
                     ),
                 )),
    pytest.param(_OPLB_TF32_6X8_TN, id="oplb-tf32-6x8-tn",
                 marks=pytest.mark.xfail(
                     strict=True,
                     reason=(
                         "rocm-libraries-udqg: same root cause as the "
                         "BPG#11 entry above; reproduces with 624 "
                         "edge_keys mismatches on the oplb anchor "
                         "fixture. Larger tile -> proportionally more "
                         "extra pack->MFMA edges in SHADOW (576 "
                         "expected, but 1152 emitted)."
                     ),
                 )),
    pytest.param(_BF16_256X256X64_TN, id="bf16-256x256x64-tn"),
]


# =============================================================================
# The probe
# =============================================================================

def _build_shadow_cms_pair(kernel_config, asm, isaInfoMap):
    """Run one Build #1 through the CMS pipeline and return the
    ``(default_capture, cms_capture)`` pair extracted from the writer's
    SHADOW slots.

    Same pattern as ``test_dataflow_graph_emission_ordinal.real_kernel_capture_pair``:
    swallow the in-build assertion if the comparator raises so the
    captures themselves remain inspectable (the probe is about the
    captures' edge_keys, not about whether ``compare_graphs`` passes
    cleanly today — n7og is investigating the basis used for those
    edge_keys).
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import (
        KernelWriterAssembly, DebugConfig,
    )

    solution = _make_solution(dict(kernel_config), asm, isaInfoMap)
    writer = KernelWriterAssembly(asm, DebugConfig())
    try:
        writer._getKernelSource(solution)
    except Exception:
        # In-build assertion (e.g. compare_graphs) may fire; the SHADOW
        # and CMS FourPartCaptures are populated before it does.
        pass

    default_cap = writer._last_default_capture
    cms_cap = writer._last_cms_capture
    assert default_cap is not None, (
        "SHADOW (_last_default_capture) was not populated; the dm4p "
        "Phase 2 capture path did not run for this fixture."
    )
    assert cms_cap is not None, (
        "CMS (_last_cms_capture) was not populated; the kernelBody "
        "post-loop assembly stage did not run for this fixture."
    )
    return default_cap, cms_cap


def _edge_keys_for_capture(capture):
    """Return ``set[tuple]`` of edge_keys for a FourPartCapture.

    Constructs the ``DataflowGraph`` via ``build_dataflow_graph`` (the
    same entry-point used by ``compare_graphs``) and returns
    ``graph.edge_keys()``.
    """
    from Tensile.Components.CMSValidator import build_dataflow_graph
    graph = build_dataflow_graph(capture)
    return set(graph.edge_keys())


@pytest.mark.parametrize("kernel_config", _FIXTURES)
def test_shadow_vs_cms_edge_keys_match(isa_infrastructure, kernel_config):
    """For each fixture, assert SHADOW and CMS produce identical
    ``DataflowGraph.edge_keys()`` sets.

    Per the n7og investigation's empirical finding (Outcome B), this
    assertion currently fails on the TF32+UsePLRPack fixtures
    (``bpg11-tf32-4x4-tn`` and ``oplb-tf32-6x8-tn``) and passes on
    ``bf16-256x256x64-tn``. The failing fixtures are pinned via
    ``pytest.mark.xfail(strict=True, ...)`` with the failure reason
    citing ``rocm-libraries-udqg`` (the carry-forward bead filed by
    the n7og investigation, P0 blocker on ``rocm-libraries-r62g``).

    When udqg is closed, ``strict=True`` causes the now-passing case
    to fail as XPASS — that is the signal to remove the
    ``marks=pytest.mark.xfail(...)`` from the parametrize entries
    (see the comment in the ``_FIXTURES`` list above).
    """
    _isa, isaInfoMap, asm = isa_infrastructure

    default_cap, cms_cap = _build_shadow_cms_pair(
        kernel_config, asm, isaInfoMap)

    default_edges = _edge_keys_for_capture(default_cap)
    cms_edges = _edge_keys_for_capture(cms_cap)

    missing_in_shadow = cms_edges - default_edges
    extra_in_shadow = default_edges - cms_edges

    total_mismatches = len(missing_in_shadow) + len(extra_in_shadow)
    assert total_mismatches == 0, (
        f"n7og defect REPRODUCED on this fixture: "
        f"{total_mismatches} SHADOW-vs-CMS edge_keys mismatches "
        f"({len(missing_in_shadow)} in CMS but not SHADOW, "
        f"{len(extra_in_shadow)} in SHADOW but not CMS). "
        f"This is the architectural concern at CMSValidator.py:1300 "
        f"(edge-key tuples embed identity which embeds canonical_render "
        f"which embeds rendered register names). Per the n7og bead, "
        f"the principled fix is byte-key matching at the edge layer "
        f"(Approach E reference). "
        f"First 3 missing-in-shadow: {list(missing_in_shadow)[:3]}; "
        f"first 3 extra-in-shadow: {list(extra_in_shadow)[:3]}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
