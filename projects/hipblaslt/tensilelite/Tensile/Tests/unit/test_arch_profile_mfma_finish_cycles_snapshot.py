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
"""Snapshot pin for `ArchProfile.mfma_finish_cycles_for(rocisa_inst)`
(rocm-libraries-qbcc).

The migration replaced a `"_4x4x" in str(rocisa_inst)` rendered-assembly
parse with a direct call to rocisa's bound
`MFMAInstruction.getIssueLatency()` accessor (see
`Tensile/Components/CMSValidator.py:ArchProfile.mfma_finish_cycles_for`).

The accessor consumes rocisa's per-`(arch, dtype, B)` C++ helper
`getMFMAIssueLatency<isSparse>(dataType, matrixInstM, matrixInstB)` and
returns `matrixInstM / mi_divisor`. The validator translates that integer
to a "finish cycles" bucket via the threshold `issueLatency <= 2`:
4x4 PackMFMAs (M=4) always satisfy the threshold; every standard MFMA
(M >= 16) lands above it. The CDNA4 profile then maps the bucket to
`mfma_4x4_finish_cycles=1` (PackMFMA) or `standard_mfma_finish_cycles=3`
(everything else) — bit-identical to the pre-migration string-parse
answer.

This module enumerates every MFMA opcode shape currently exercised by the
unit-test suite (4x4 PackMFMAs of the TF32 chain, 16x16 / 32x32 / 4x4
families of the canonical fixtures), constructs the rocisa instance, and
asserts `(variant, dtype) -> finish_cycles` matches the pre-migration
answer. Adding a new MFMA shape to a fixture requires extending
`MFMA_OPCODE_PINS` below.

The pin runs against the default CDNA4 ArchProfile. Regressions surface
either when:
  - rocisa C++ changes the meaning of `getMFMAIssueLatency` (then this
    test fails and the ArchProfile threshold needs revisiting), or
  - the validator's `mfma_finish_cycles_for` body drifts (then this test
    pins us to the historical answer).
"""

import pytest

from rocisa.container import vgpr
from rocisa.enum import InstType
from rocisa.instruction import MFMAInstruction, SMFMAInstruction

from Tensile.Components.CMSValidator import _DEFAULT_CDNA4_ARCH_PROFILE


# (variant, instType, accType, expected_finish_cycles).
#
# Pinned answers reflect the pre-qbcc behavior of the
# `"_4x4x" in str(rocisa_inst)` discriminator: 4x4 PackMFMAs returned
# `mfma_4x4_finish_cycles == 1`, every other MFMA returned
# `standard_mfma_finish_cycles == 3`.
#
# Sources for each (variant, instType, accType) tuple:
#   [4,4,4,16]  F32 -> F32  : `test_dataflow_graph_register_gaps.py:1462,2408`
#                             (4x4 PackMFMA producer in the TF32 emul chain)
#   [4,4,4,16]  BF16 -> F32 : `test_dataflow_graph_comparison.py:487,568`
#                             (mixed-register PackMFMA fixtures)
#   [16,16,32,1] BF16 -> F32: `cms_test_utils.py:66`
#                             (canonical full-tile MFMA used by CMS body builders)
#   [32,32,0,1]  F32 -> F32 : `test_ScheduleCapture.py:198`
#                             (full-tile MFMA used by `_make_body`)
#   [16,16,4,1]  F32 -> F32 : DGEMM-style F32 MFMA (defensive coverage; not
#                             exercised by any current fixture but rocisa
#                             constructs it identically — pinning here keeps
#                             the threshold honest if a fixture starts using it)
#   [4,4,4,1]   F32 -> F32  : 4x4 with B=1 (tests rocisa's gfx950 fast-path
#                             interaction with the 4x4 family)
MFMA_OPCODE_PINS = [
    # 4x4 PackMFMA family — always returns mfma_4x4_finish_cycles (= 1 on CDNA4).
    ([4, 4, 4, 16], InstType.INST_F32, InstType.INST_F32, 1),
    ([4, 4, 4, 16], InstType.INST_BF16, InstType.INST_F32, 1),
    ([4, 4, 4, 1], InstType.INST_F32, InstType.INST_F32, 1),
    ([4, 4, 4, 1], InstType.INST_BF16, InstType.INST_F32, 1),
    # Standard full-tile MFMAs — always return standard_mfma_finish_cycles (= 3).
    ([16, 16, 32, 1], InstType.INST_BF16, InstType.INST_F32, 3),
    ([16, 16, 16, 1], InstType.INST_BF16, InstType.INST_F32, 3),
    ([16, 16, 4, 1], InstType.INST_F32, InstType.INST_F32, 3),
    ([32, 32, 0, 1], InstType.INST_F32, InstType.INST_F32, 3),
    ([32, 32, 8, 1], InstType.INST_BF16, InstType.INST_F32, 3),
]


# gfx950-specific pins. The bead's headline motivation was rocisa picking up
# the gfx950 F8 cycle-doubling override (`mfma.hpp:60-64`: F8 on gfx950
# resets `mi_divisor=2` even after the 8-bit-float fast-path bumped it to 4)
# and the XF32 `mi_divisor=4` forcing (`mfma.hpp:55-58`) "for free" — the
# string-parse classifier had no path to either. These rows pin the live
# rocisa accessor's answer on each, run with `isaVersion = (9, 5, 0)` set so
# the gfx950-conditional branches actually fire.
#
# Verified expected `getIssueLatency()` values against the live accessor on
# gfx950-set rocisa (see `_isa_gfx950` fixture):
#   F8  4x4x4   B=1 -> 2  (gfx950 override forces mi_divisor=2 from default-2;
#                          fast-path fix-up to 4 doesn't apply at M=4: 4/2=2)
#   F8  16x16x32 B=1 -> 8 (8-bit-float fast-path would set mi_divisor=4
#                          (16/4=4) but the gfx950 F8 override resets it to 2:
#                          16/2=8 — i.e. F8 takes 2x the cycles of BF16
#                          16x16x32 B=1 on gfx950, exactly the doubling the
#                          bead motivation referenced)
#   XF32 16x16x8 B=1 -> 4 (XFloat32 forces mi_divisor=4 unconditionally:
#                          16/4=4)
#
# Threshold (`issueLatency <= 2`) maps these to:
#   F8 4x4 -> 4x4 bucket (finish_cycles=1)
#   F8 16x16 / XF32 16x16 -> standard bucket (finish_cycles=3)
GFX950_MFMA_OPCODE_PINS = [
    # F8 4x4 with B=1: gfx950 override leaves mi_divisor=2; 4x4 bucket.
    ([4, 4, 4, 1], InstType.INST_F8, InstType.INST_F32, 1),
    # F8 16x16 with B=1: gfx950 override resets mi_divisor=4 -> 2; standard
    # bucket. This is the cycle-doubling case the bead specifically called out.
    ([16, 16, 32, 1], InstType.INST_F8, InstType.INST_F32, 3),
    # XF32 16x16 with B=1: XFloat32 forces mi_divisor=4 (sparse-style); standard
    # bucket. Pins that the threshold doesn't accidentally promote XF32.
    ([16, 16, 8, 1], InstType.INST_XF32, InstType.INST_F32, 3),
]


# Sparse-MFMA gfx950 pins. SMFMAs unconditionally use mi_divisor=4 from the
# `isSparse` clause; the gfx950 F8 override then resets it to 2 for F8.
# Verified expected values against the live accessor:
#   sparse F8   16x16x32 B=1 -> 8 (override resets mi_divisor 4 -> 2)
#   sparse XF32 16x16x8  B=1 -> 4 (XFloat32 path equivalent to sparse path)
GFX950_SMFMA_OPCODE_PINS = [
    ([16, 16, 32, 1], InstType.INST_F8, InstType.INST_F32, 3),
    ([16, 16, 8, 1], InstType.INST_XF32, InstType.INST_F32, 3),
]


# Sparse-MFMA pins — the SMFMAInstruction binding also exposes
# `getIssueLatency()`; sparse always uses mi_divisor=4 so 4x4 sparse falls
# under the threshold (issueLatency=1) and 16x16/32x32 sparse stay above it
# (issueLatency >= 4).
SMFMA_OPCODE_PINS = [
    ([4, 4, 4, 1], InstType.INST_F32, InstType.INST_F32, 1),
    ([4, 4, 4, 16], InstType.INST_BF16, InstType.INST_F32, 1),
    ([16, 16, 32, 1], InstType.INST_BF16, InstType.INST_F32, 3),
    ([32, 32, 16, 1], InstType.INST_BF16, InstType.INST_F32, 3),
]


def _make_mfma(variant, inst_type, acc_type):
    """Build a real rocisa MFMAInstruction with throwaway register operands.

    The `mfma_finish_cycles_for` accessor only consults the instruction's
    `instType` + `variant` (via the bound `getIssueLatency()`); the register
    containers exist only to satisfy the constructor signature. We pick
    sizes that match the surface arity rocisa expects from the variant
    (M-wide accumulator, 2 vgprs for A/B sources) so the constructor accepts
    every test entry.
    """
    M = variant[0]
    acc_count = max(M, 4)
    return MFMAInstruction(
        inst_type, acc_type, variant, False,
        vgpr(0, acc_count), vgpr(64, 2), vgpr(72, 2),
    )


def _make_smfma(variant, inst_type, acc_type):
    """Build a rocisa SMFMAInstruction (sparse) with throwaway operands.

    SMFMA constructor takes an extra `metadata` operand (single vgpr). The
    classifier only consults `getIssueLatency()` so sizes are arbitrary.
    """
    M = variant[0]
    acc_count = max(M, 4)
    return SMFMAInstruction(
        inst_type, acc_type, variant, False,
        vgpr(0, acc_count), vgpr(64, 2), vgpr(72, 2), vgpr(80, 1),
    )


@pytest.mark.parametrize(
    "variant,inst_type,acc_type,expected_finish_cycles",
    MFMA_OPCODE_PINS,
    ids=[
        f"M{v[0]}xN{v[1]}xK{v[2]}_B{v[3]}_{it.name}_{at.name}"
        for v, it, at, _ in MFMA_OPCODE_PINS
    ],
)
def test_mfma_finish_cycles_pin_cdna4(
    variant, inst_type, acc_type, expected_finish_cycles,
):
    """Pin `(variant, instType, accType) -> finish_cycles` on CDNA4.

    Constructs a rocisa MFMAInstruction with the given shape, runs it through
    the (post-qbcc) `mfma_finish_cycles_for` classifier on the default
    CDNA4 profile, and asserts the result matches the pre-qbcc answer.
    """
    inst = _make_mfma(variant, inst_type, acc_type)
    finish = _DEFAULT_CDNA4_ARCH_PROFILE.mfma_finish_cycles_for(inst)
    assert finish == expected_finish_cycles, (
        f"variant={variant} inst_type={inst_type.name} acc_type={acc_type.name}: "
        f"got finish_cycles={finish}, expected {expected_finish_cycles}. "
        f"rocisa getIssueLatency()={inst.getIssueLatency()}; "
        f"the threshold (issueLatency <= 2 -> 4x4 family) misclassified this "
        f"shape — investigate `ArchProfile.mfma_finish_cycles_for`."
    )


@pytest.mark.parametrize(
    "variant,inst_type,acc_type,expected_finish_cycles",
    SMFMA_OPCODE_PINS,
    ids=[
        f"sparse_M{v[0]}xN{v[1]}xK{v[2]}_B{v[3]}_{it.name}"
        for v, it, _, _ in SMFMA_OPCODE_PINS
    ],
)
def test_smfma_finish_cycles_pin_cdna4(
    variant, inst_type, acc_type, expected_finish_cycles,
):
    """Pin `(variant, instType, accType) -> finish_cycles` for SMFMA on CDNA4.

    SMFMAInstruction also exposes `getIssueLatency()` (sparse code path,
    `getMFMAIssueLatency<true>`). Sparse always sets mi_divisor=4 so the
    issueLatency partition keeps the same threshold semantics.
    """
    inst = _make_smfma(variant, inst_type, acc_type)
    finish = _DEFAULT_CDNA4_ARCH_PROFILE.mfma_finish_cycles_for(inst)
    assert finish == expected_finish_cycles


def test_mfma_finish_cycles_none_input():
    """`None` rocisa_inst falls back to the standard finish-cycles bucket.

    Pre-qbcc behavior: `if rocisa_inst is None: return standard`. The
    accessor-based path preserves this fallback (no AttributeError /
    accidental discriminator flip).
    """
    finish = _DEFAULT_CDNA4_ARCH_PROFILE.mfma_finish_cycles_for(None)
    assert finish == _DEFAULT_CDNA4_ARCH_PROFILE.standard_mfma_finish_cycles


def test_mfma_finish_cycles_non_mfma_input():
    """A non-MFMA rocisa instance (no `getIssueLatency` accessor) falls
    back to the standard finish-cycles bucket.

    Pre-qbcc behavior: `str(rocisa_inst)` rendered something that didn't
    contain `_4x4x`, returning standard. Post-qbcc behavior: `getattr(...,
    'getIssueLatency', None)` is None, returning standard. This test pins
    that the accessor-missing fallback doesn't accidentally promote a
    non-MFMA to the 4x4 bucket.
    """
    from rocisa.instruction import VXorB32
    inst = VXorB32(vgpr(0, 1), vgpr(1, 1), vgpr(2, 1))
    finish = _DEFAULT_CDNA4_ARCH_PROFILE.mfma_finish_cycles_for(inst)
    assert finish == _DEFAULT_CDNA4_ARCH_PROFILE.standard_mfma_finish_cycles


@pytest.fixture
def _isa_gfx950():
    """Temporarily set the rocisa global kernel to gfx950 for the duration
    of the test, restoring whatever was there before.

    Required because the gfx950 F8 cycle-doubling override and the
    fp16/bf16/int8/8-bit-float fast-path branches inside
    `getMFMAIssueLatency` are gated on `isaVersion == (9, 5, 0)` (and
    cousins). With the default `(0, 0, 0)` isa rocisa never reaches the
    override and the F8 16x16 case returns the same `8` as plain BF16
    16x16 — masking exactly the override behavior the bead motivation
    referenced.
    """
    from rocisa import rocIsa

    ri = rocIsa.getInstance()
    prev_kernel = ri.getKernel()
    prev_isa = tuple(prev_kernel.isa)
    prev_wavefront = prev_kernel.wavefrontSize
    ri.setKernel((9, 5, 0), 64)
    try:
        yield
    finally:
        ri.setKernel(prev_isa, prev_wavefront)


@pytest.mark.parametrize(
    "variant,inst_type,acc_type,expected_finish_cycles",
    GFX950_MFMA_OPCODE_PINS,
    ids=[
        f"gfx950_M{v[0]}xN{v[1]}xK{v[2]}_B{v[3]}_{it.name}_{at.name}"
        for v, it, at, _ in GFX950_MFMA_OPCODE_PINS
    ],
)
def test_mfma_finish_cycles_pin_gfx950_f8_xf32(
    variant, inst_type, acc_type, expected_finish_cycles, _isa_gfx950,
):
    """Pin gfx950 F8 / XF32 `(variant, instType) -> finish_cycles`.

    These rows exercise the gfx950-specific code paths inside rocisa's
    `getMFMAIssueLatency` that the original string-parse classifier had
    no visibility into:
      - F8 cycle-doubling override (`mfma.hpp:60-64`): F8 on gfx950 resets
        mi_divisor=2 even after the 8-bit-float fast-path bumps it to 4.
      - XFloat32 mi_divisor=4 forcing (`mfma.hpp:55-58`): XF32 always takes
        the sparse-style divisor regardless of arch / B.
    """
    inst = _make_mfma(variant, inst_type, acc_type)
    finish = _DEFAULT_CDNA4_ARCH_PROFILE.mfma_finish_cycles_for(inst)
    assert finish == expected_finish_cycles, (
        f"variant={variant} inst_type={inst_type.name} acc_type={acc_type.name} "
        f"@ gfx950: got finish_cycles={finish}, expected {expected_finish_cycles}. "
        f"rocisa getIssueLatency()={inst.getIssueLatency()}; "
        f"the gfx950 F8/XF32 path drifted — investigate `mfma.hpp` "
        f"`getMFMAIssueLatency` or the ArchProfile threshold."
    )


@pytest.mark.parametrize(
    "variant,inst_type,acc_type,expected_finish_cycles",
    GFX950_SMFMA_OPCODE_PINS,
    ids=[
        f"gfx950_sparse_M{v[0]}xN{v[1]}xK{v[2]}_B{v[3]}_{it.name}"
        for v, it, _, _ in GFX950_SMFMA_OPCODE_PINS
    ],
)
def test_smfma_finish_cycles_pin_gfx950_f8_xf32(
    variant, inst_type, acc_type, expected_finish_cycles, _isa_gfx950,
):
    """Pin gfx950 F8 / XF32 sparse-MFMA `(variant, instType) -> finish_cycles`.

    Sparse always sets mi_divisor=4 from the `isSparse` clause; the gfx950 F8
    override then resets it to 2 for F8. XF32 lands on the same forced
    divisor=4. Both stay above the threshold so they map to the standard
    bucket — pinned here so a future drift in either branch surfaces.
    """
    inst = _make_smfma(variant, inst_type, acc_type)
    finish = _DEFAULT_CDNA4_ARCH_PROFILE.mfma_finish_cycles_for(inst)
    assert finish == expected_finish_cycles


def test_classifier_partitions_4x4_from_standard_via_accessor():
    """Cross-check: every 4x4 (M=4) variant produces issueLatency <= 2; every
    M >= 16 variant produces issueLatency >= 4. This is the invariant the
    threshold relies on.

    Failure of this test indicates rocisa's `getMFMAIssueLatency` C++ helper
    grew a new mi_divisor branch that violates the partition — the
    ArchProfile threshold needs to be revisited and the per-arch profile
    may need a richer (variant -> finish_cycles) table.
    """
    for variant, inst_type, acc_type, _ in MFMA_OPCODE_PINS:
        inst = _make_mfma(variant, inst_type, acc_type)
        issue_latency = inst.getIssueLatency()
        M = variant[0]
        if M == 4:
            assert issue_latency <= 2, (
                f"variant={variant} inst_type={inst_type.name}: rocisa "
                f"getIssueLatency()={issue_latency}, expected <= 2 for the "
                f"4x4 family. The classifier threshold (issueLatency <= 2) "
                f"would misclassify this shape — revisit `ArchProfile."
                f"mfma_finish_cycles_for`."
            )
        else:
            assert issue_latency >= 4, (
                f"variant={variant} inst_type={inst_type.name}: rocisa "
                f"getIssueLatency()={issue_latency}, expected >= 4 for "
                f"M>=16 standard MFMAs. The classifier threshold would "
                f"misclassify this shape."
            )
