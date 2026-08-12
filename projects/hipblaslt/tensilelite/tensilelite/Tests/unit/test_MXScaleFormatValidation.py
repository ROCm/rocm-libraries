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

"""Unit tests for `Validators.MXScaleFormat.validateMXScaleFormatCombination`.

gfx1250's v_wmma_scale_f32_16x16x128_f8f6f4 only accepts a fixed set of
(A matrix class, A scale, B matrix class, B scale) tuples. The AMDGPU
assembler does not enforce these joint constraints (see
ROCm/llvm-project#2634), so Solution.py rejects illegal candidates before
codegen.

The helper under test is a pure dict-in / bool-out function: build a
minimal Solution state, call the helper, assert on the return value and
on ``state["Valid"]``. No client build, no GPU, no rocisa device code.

Mapping in tensilelite vocabulary:
  - matrix FP8        -> DataTypeEnum.Float8           (also FP8 _fnuz)
  - matrix BF8        -> DataTypeEnum.BFloat8          (also BF8 _fnuz)
  - matrix FP6 / BF6  -> DataTypeEnum.Float6 / BFloat6
  - matrix FP4        -> DataTypeEnum.Float4
  - scale  E8         -> DataTypeEnum.E8     (UE8M0)
  - scale  E5M3       -> DataTypeEnum.E5M3
  - scale  E4M3       -> DataTypeEnum.Float8 (same byte as OCP FP8)
"""

import pytest

from rocisa.enum import DataTypeEnum
from Tensile.Common.DataType import DataType
from Tensile.SolutionStructs.Validators.MXScaleFormat import validateMXScaleFormatCombination


# ---------------------------------------------------------------------------
# Tag shorthands so the parametrize tables read like the ISA spec.
# ---------------------------------------------------------------------------
FP8       = DataTypeEnum.Float8
FP8_FNUZ  = DataTypeEnum.Float8_fnuz
BF8       = DataTypeEnum.BFloat8
BF8_FNUZ  = DataTypeEnum.BFloat8_fnuz
FP6       = DataTypeEnum.Float6
BF6       = DataTypeEnum.BFloat6
FP4       = DataTypeEnum.Float4
F32       = DataTypeEnum.Float

# Scale formats (tensilelite uses Float8 for the E4M3 byte).
E8        = DataTypeEnum.E8
E5M3      = DataTypeEnum.E5M3
E4M3      = DataTypeEnum.Float8


# The validator is gated on ``asmCaps["HasWMMA_V3"]`` so the gfx1250 MX rules
# only fire for gfx1250 candidates. Tests that exercise the rule set use
# ``_GFX1250_CAPS``; the off-arch ``_NON_GFX1250_CAPS`` is used by the gating
# tests to confirm we don't over-reject on other architectures.
_GFX1250_CAPS     = {"HasWMMA_V3": True}
_NON_GFX1250_CAPS = {"HasWMMA_V3": False}


def _state(*, dtA, dtB, dtSA=E8, dtSB=E8, mxBlockA=32, mxBlockB=32):
    """Build the smallest Solution state dict the validator reads.

    The helper only inspects ``ProblemType.{DataTypeA, DataTypeB,
    DataTypeMXSA, DataTypeMXSB, MXBlockA, MXBlockB}`` and ``state["Valid"]``
    via ``reject``; nothing else needs to be populated.
    """
    return {
        "ProblemType": {
            "DataTypeA":     DataType(dtA),
            "DataTypeB":     DataType(dtB),
            "DataTypeMXSA":  DataType(dtSA),
            "DataTypeMXSB":  DataType(dtSB),
            "MXBlockA":      mxBlockA,
            "MXBlockB":      mxBlockB,
        }
    }


def _call(state, asmCaps=_GFX1250_CAPS):
    """Invoke the helper with rejection-reason printing off (test noise).

    Defaults to a gfx1250 (``HasWMMA_V3=True``) asmCaps mapping so the rule
    set fires; gating tests pass ``_NON_GFX1250_CAPS`` explicitly.
    """
    return validateMXScaleFormatCombination(
        state, asmCaps=asmCaps, printRejectionReason=False)


# ============================================================================
# 0. Architecture gate: the rule set only applies on gfx1250 (HasWMMA_V3).
# ============================================================================
class TestArchitectureGate:
    """The joint MX scale-format rules described in ``table-valid-combinations.txt``
    are specific to gfx1250's ``v_wmma_scale_f32_16x16x128_f8f6f4``. Older
    architectures use different MX instructions whose constraints differ,
    so the validator must no-op (return True, leave ``state["Valid"]``
    untouched) whenever ``asmCaps["HasWMMA_V3"]`` is false - even for tuples
    that *would* be illegal on gfx1250."""

    @pytest.mark.parametrize("asmCaps", [
        {},                      # caps mapping missing the key entirely
        {"HasWMMA_V3": False},   # cap explicitly false
    ], ids=["missing-cap", "cap-false"])
    def test_non_gfx1250_passes_through_illegal_combo(self, asmCaps):
        # FP8 x FP8 with E5M3 scale - illegal on gfx1250, but on a non-WMMA_V3
        # arch the validator must not touch it.
        st = _state(dtA=FP8, dtB=FP8, dtSA=E5M3, dtSB=E8)
        assert _call(st, asmCaps=asmCaps) is True
        assert "Valid" not in st

    def test_non_gfx1250_passes_through_fp4_mismatched_scales(self):
        # FP4 x FP4 with mismatched scales - illegal on gfx1250 by the joint
        # rule, but on a non-WMMA_V3 arch the joint rule does not exist.
        st = _state(dtA=FP4, dtB=FP4, dtSA=E5M3, dtSB=E4M3)
        assert _call(st, asmCaps=_NON_GFX1250_CAPS) is True
        assert "Valid" not in st

    def test_gfx1250_still_rejects_the_same_combo(self):
        # Sanity check: the same illegal combo IS rejected once HasWMMA_V3
        # is true, confirming the gate is the only thing suppressing the
        # rejection above.
        st = _state(dtA=FP4, dtB=FP4, dtSA=E5M3, dtSB=E4M3)
        assert _call(st, asmCaps=_GFX1250_CAPS) is False
        assert st["Valid"] is False


# ============================================================================
# 1. Short-circuit: neither side has MX scaling -> always valid
# ============================================================================
class TestNoMXScalingShortCircuits:
    """If both MXBlockA and MXBlockB are 0 the rules do not apply at all.

    The helper must return True without inspecting the matrix dtypes - in
    particular it must NOT raise even when the dtype fields would be
    illegal under the MX rules (because for a non-MX problem those fields
    are simply unused defaults).
    """

    def test_both_blocks_zero_returns_true(self):
        st = _state(dtA=F32, dtB=F32, mxBlockA=0, mxBlockB=0)
        assert _call(st) is True
        # state["Valid"] must be untouched - the helper short-circuits.
        assert "Valid" not in st

    def test_both_blocks_zero_with_mx_dtype_fields_is_still_valid(self):
        """Even an FP4xFP4 dtype pair must not trigger the joint rule when
        neither side has MX scaling (mxBlock == 0 on both sides)."""
        st = _state(dtA=FP4, dtB=FP4, dtSA=E5M3, dtSB=E4M3,
                    mxBlockA=0, mxBlockB=0)
        assert _call(st) is True
        assert "Valid" not in st


# ============================================================================
# 2. Per-side rule: FP8 / BF8 / FP6 / BF6 require E8 scale
# ============================================================================
class TestFP8FamilyRequiresE8:
    """FP8, BF8, FP6, BF6 (incl. _fnuz variants of FP8/BF8) each accept
    only E8 (UE8M0) scale on their own side."""

    @pytest.mark.parametrize("dt", [FP8, FP8_FNUZ, BF8, BF8_FNUZ, FP6, BF6],
                             ids=lambda e: e.name)
    def test_e8_scale_is_accepted(self, dt):
        st = _state(dtA=dt, dtB=dt, dtSA=E8, dtSB=E8)
        assert _call(st) is True

    @pytest.mark.parametrize("dt", [FP8, FP8_FNUZ, BF8, BF8_FNUZ, FP6, BF6],
                             ids=lambda e: e.name)
    @pytest.mark.parametrize("badScale", [E5M3, E4M3], ids=lambda e: e.name)
    def test_non_e8_scale_on_a_is_rejected(self, dt, badScale):
        st = _state(dtA=dt, dtB=dt, dtSA=badScale, dtSB=E8)
        assert _call(st) is False
        assert st["Valid"] is False

    @pytest.mark.parametrize("dt", [FP8, FP8_FNUZ, BF8, BF8_FNUZ, FP6, BF6],
                             ids=lambda e: e.name)
    @pytest.mark.parametrize("badScale", [E5M3, E4M3], ids=lambda e: e.name)
    def test_non_e8_scale_on_b_is_rejected(self, dt, badScale):
        st = _state(dtA=dt, dtB=dt, dtSA=E8, dtSB=badScale)
        assert _call(st) is False
        assert st["Valid"] is False


# ============================================================================
# 3. Per-side rule: FP4 accepts E8, E5M3, or E4M3
# ============================================================================
class TestFP4AcceptsAllThreeScales:
    """FP4 is the only class with multiple legal scales. Each individually
    must be accepted when paired with itself on the other side."""

    @pytest.mark.parametrize("scale", [E8, E5M3, E4M3], ids=lambda e: e.name)
    def test_fp4_x_fp4_matching_scale_is_accepted(self, scale):
        st = _state(dtA=FP4, dtB=FP4, dtSA=scale, dtSB=scale)
        assert _call(st) is True


# ============================================================================
# 4. FP4 x FP4 joint rule: scales must match
# ============================================================================
class TestFP4xFP4ScalesMustMatch:
    """For a pure FP4 x FP4 problem the validator additionally requires
    AScale == BScale. (For mixed-class problems the per-side rule already
    pins the non-FP4 side to E8, so the mismatch can only arise FP4 x
    FP4.)"""

    @pytest.mark.parametrize("scaleA, scaleB", [
        (E8,   E5M3),
        (E8,   E4M3),
        (E5M3, E8),
        (E5M3, E4M3),
        (E4M3, E8),
        (E4M3, E5M3),
    ])
    def test_mismatched_scales_are_rejected(self, scaleA, scaleB):
        st = _state(dtA=FP4, dtB=FP4, dtSA=scaleA, dtSB=scaleB)
        assert _call(st) is False
        assert st["Valid"] is False


# ============================================================================
# 5. Mixed-class problems: each side independent; joint rule does not fire
# ============================================================================
class TestMixedClassProblems:
    """When A and B are different MX classes the FP4 x FP4 joint rule does
    not fire. Each side just has to satisfy its own per-side rule."""

    @pytest.mark.parametrize("aSide, bSide", [
        ((FP8, E8), (FP4, E8)),
        ((FP8, E8), (FP4, E5M3)),
        ((FP8, E8), (FP4, E4M3)),
        ((BF8, E8), (FP4, E8)),
        ((FP6, E8), (FP4, E5M3)),
        ((FP4, E8),   (FP8, E8)),
        ((FP4, E5M3), (BF8, E8)),
        ((FP4, E4M3), (FP6, E8)),
    ], ids=lambda p: f"A_{p[0].name if hasattr(p, 'name') else p}")
    def test_legal_mixed_class_combos(self, aSide, bSide):
        dtA, scaleA = aSide
        dtB, scaleB = bSide
        st = _state(dtA=dtA, dtB=dtB, dtSA=scaleA, dtSB=scaleB)
        assert _call(st) is True

    @pytest.mark.parametrize("aSide, bSide", [
        # FP8 on the non-FP4 side must use E8, not E5M3 / E4M3.
        ((FP8, E5M3), (FP4, E5M3)),
        ((FP8, E4M3), (FP4, E4M3)),
        # BF8 must use E8 even when B side is happily FP4.
        ((BF8, E5M3), (FP4, E5M3)),
        # FP4 on the A side with FP8 on B - FP8 still pinned to E8.
        ((FP4, E8),   (FP8, E5M3)),
    ])
    def test_illegal_mixed_class_combos(self, aSide, bSide):
        dtA, scaleA = aSide
        dtB, scaleB = bSide
        st = _state(dtA=dtA, dtB=dtB, dtSA=scaleA, dtSB=scaleB)
        assert _call(st) is False
        assert st["Valid"] is False


# ============================================================================
# 6. Asymmetric MX (only one side has MX scaling)
# ============================================================================
class TestOneSidedMXScaling:
    """When only one side has MX scaling (mxBlock != 0 on that side and 0
    on the other) the validator must:
      * still enforce the per-side rule on the MX side,
      * skip the non-MX side entirely (no FP4 x FP4 joint trigger across
        a mixed MX / non-MX problem).
    """

    def test_a_side_only_legal(self):
        st = _state(dtA=FP8, dtB=F32, dtSA=E8, dtSB=E8,
                    mxBlockA=32, mxBlockB=0)
        assert _call(st) is True

    def test_a_side_only_illegal_scale_rejected(self):
        st = _state(dtA=FP8, dtB=F32, dtSA=E5M3, dtSB=E8,
                    mxBlockA=32, mxBlockB=0)
        assert _call(st) is False
        assert st["Valid"] is False

    def test_b_side_only_legal(self):
        st = _state(dtA=F32, dtB=BF8, dtSA=E8, dtSB=E8,
                    mxBlockA=0, mxBlockB=32)
        assert _call(st) is True

    def test_b_side_only_illegal_scale_rejected(self):
        st = _state(dtA=F32, dtB=BF8, dtSA=E8, dtSB=E4M3,
                    mxBlockA=0, mxBlockB=32)
        assert _call(st) is False
        assert st["Valid"] is False

    def test_fp4_on_a_only_does_not_trigger_joint_rule(self):
        """A=FP4 MX, B non-MX must NOT match the joint FP4 x FP4 rule even
        if B's DataType happens to be FP4 - because B has no MX scaling
        the joint constraint does not apply across that axis."""
        st = _state(dtA=FP4, dtB=FP4, dtSA=E5M3, dtSB=E8,
                    mxBlockA=32, mxBlockB=0)
        assert _call(st) is True


# ============================================================================
# 7. Error-message contract
# ============================================================================
class TestErrorMessageShape:
    """Reject diagnostics must include the offending tuple in ISA-spec
    spelling and the rule name that failed, so a developer reading the
    log can act on the message without cross-referencing the source."""

    def _captured_reject(self, st, capsys):
        # printRejectionReason=True so reject() emits the diagnostic to
        # stdout, then capsys captures it for assertion.
        assert validateMXScaleFormatCombination(
            st, asmCaps=_GFX1250_CAPS, printRejectionReason=True) is False
        return capsys.readouterr().out

    def test_per_side_message_contains_dtype_and_scale(self, capsys):
        st = _state(dtA=FP8, dtB=FP8, dtSA=E5M3, dtSB=E8)
        out = self._captured_reject(st, capsys)
        assert "Invalid MX scale-format combination" in out
        assert "A=FP8" in out and "AScale=E5M3" in out
        assert "matrix A class FP8 does not accept scale format E5M3" in out
        assert "ROCm/llvm-project#2634" in out

    def test_fp4_joint_message_names_the_mismatch(self, capsys):
        st = _state(dtA=FP4, dtB=FP4, dtSA=E5M3, dtSB=E4M3)
        out = self._captured_reject(st, capsys)
        assert "(A=FP4, AScale=E5M3, B=FP4, BScale=E4M3)" in out
        assert "FP4 x FP4 requires AScale (E5M3) == BScale (E4M3)" in out

    def test_both_sides_illegal_reports_both(self, capsys):
        st = _state(dtA=FP8, dtB=BF8, dtSA=E5M3, dtSB=E4M3)
        out = self._captured_reject(st, capsys)
        assert "matrix A class FP8 does not accept scale format E5M3" in out
        assert "matrix B class BF8 does not accept scale format E4M3" in out


# ============================================================================
# 8. Spec table: every legal combo from table-valid-combinations.txt
# ============================================================================
#
# This is the authoritative gfx1250 valid-combination matrix expressed as
# (A matrix, A scale, B matrix, B scale) tuples. Any change to this list
# is also a change to the hardware contract.
VALID_COMBOS = []
# FP8 / BF8 / FP6 / BF6 (incl. _fnuz on the FP8/BF8 sides) all pin scale to E8
# on both sides. Every same-class and mixed-class pairing is legal.
_E8_CLASSES = [FP8, FP8_FNUZ, BF8, BF8_FNUZ, FP6, BF6]
for _a in _E8_CLASSES:
    for _b in _E8_CLASSES:
        VALID_COMBOS.append((_a, E8, _b, E8))
# FP4 against itself accepts three matching-scale variants.
for _s in [E8, E5M3, E4M3]:
    VALID_COMBOS.append((FP4, _s, FP4, _s))
# FP4 paired with an E8-class on the other side: FP4 may use any of its
# 3 legal scales; the other side must use E8.
for _other in _E8_CLASSES:
    for _s in [E8, E5M3, E4M3]:
        VALID_COMBOS.append((FP4, _s, _other, E8))
        VALID_COMBOS.append((_other, E8, FP4, _s))


class TestEveryLegalCombinationFromSpec:
    """Exercises every legal (A, scaleA, B, scaleB) tuple permitted by the
    gfx1250 ISA. This is the inverse of the per-side / joint-rule tests
    above: instead of asserting what gets rejected, we assert that the
    full positive set passes."""

    @pytest.mark.parametrize(
        "dtA, scaleA, dtB, scaleB", VALID_COMBOS,
        ids=[f"{a.name}-{sa.name}_x_{b.name}-{sb.name}"
             for (a, sa, b, sb) in VALID_COMBOS])
    def test_combo_is_accepted(self, dtA, scaleA, dtB, scaleB):
        st = _state(dtA=dtA, dtB=dtB, dtSA=scaleA, dtSB=scaleB)
        assert _call(st) is True, (
            f"Spec combo rejected: A={dtA.name}, AScale={scaleA.name}, "
            f"B={dtB.name}, BScale={scaleB.name}")


# ============================================================================
# 9. Field-shape compatibility: helper accepts both DataType and raw enum
# ============================================================================
class TestFieldShapeCompatibility:
    """ProblemType fields can arrive as ``DataType`` wrappers (typical during
    ``assignDerivedParameters``) or as raw ``DataTypeEnum`` values (when
    callers have already unwrapped to the enum). The helper must handle
    both shapes."""

    def test_raw_enum_fields_are_accepted(self):
        st = {"ProblemType": {
            "DataTypeA":    FP8,     # raw enum
            "DataTypeB":    FP8,
            "DataTypeMXSA": E8,
            "DataTypeMXSB": E8,
            "MXBlockA":     32,
            "MXBlockB":     32,
        }}
        assert _call(st) is True

    def test_raw_enum_illegal_combo_is_rejected(self):
        st = {"ProblemType": {
            "DataTypeA":    FP8,
            "DataTypeB":    FP8,
            "DataTypeMXSA": E5M3,    # illegal
            "DataTypeMXSB": E8,
            "MXBlockA":     32,
            "MXBlockB":     32,
        }}
        assert _call(st) is False
        assert st["Valid"] is False
