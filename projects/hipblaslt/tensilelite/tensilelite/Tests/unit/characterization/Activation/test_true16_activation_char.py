# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""True16 half-select emit characterization for tensilelite/Activation.py.

The existing Activation characterization tests only initialize a gfx942 (legacy
SDWA) ISA, so the new true16 code paths added on `users/ericwan/true16-half-sel`
are never exercised. This test drives the same emit code under BOTH:

  * a legacy SDWA arch (gfx942, ``NoSDWA == 0``)  -> WORD_x / v_cndmask_b32, and
  * a true16 arch      (gfx1200, ``NoSDWA == 1``) -> .l/.h operand suffix,
                                                     v_cndmask_b16.

It covers the new per-half helpers (``addTransF16Half`` / ``addTransF16Halfwise``
/ ``addCmpF16Half`` / ``addValuF16Half`` / ``addCndMaskF16Half`` / ``_t16v``) both
directly and through the Half activation modules that call them
(clippedrelu / leakyrelu / sigmoid / tanh / exp / clamp / silu / swish / gelu),
plus the ActivationInline true16 constraint path.

All calls generate assembly strings without a GPU. The shared test helper probes
the compiler on first use of each ISA; collection does not initialize either ISA.
"""

import importlib

import pytest

pytestmark = pytest.mark.unit

A = importlib.import_module("tensilelite.Activation")
DataType = importlib.import_module("tensilelite.Common.DataType").DataType

from rocisa.code import Module
from rocisa.container import sgpr, vgpr
from rocisa.instruction import VCmpGTF16, VExpF16, VMinF16
from toolchain_test_helpers import use_isa

# NoSDWA (true16) archs are gfx11/gfx12; legacy SDWA archs are gfx9/gfx10.
LEGACY_ISA = (9, 4, 2)
TRUE16_ISA = (12, 0, 0)


def _render(module):
    return "\n".join(str(item) for item in module.items())


def _no_true16_suffix(text):
    # Legacy output must never tag an operand with a .l/.h half-select.
    import re

    return re.search(r"v\d+\.[lh]\b", text) is None


# ---------------------------------------------------------------------------
# Half activation modules that route through the true16 helpers.
# ---------------------------------------------------------------------------
_HALF_ACTS = [
    "abs",
    "relu",
    "clippedrelu",
    "leakyrelu",
    "sigmoid",
    "tanh",
    "exp",
    "clamp",
    "silu",
    "swish",
    "gelu",
    "geluscaling",
]


def _module_for(act, *, isa, usePK):
    use_isa(isa, 64)
    m = A.ActivationModule()
    m.setUsePK(usePK)
    return _render(m.getModule(DataType("H"), act, 0, 1))


@pytest.mark.parametrize("usePK", [True, False], ids=["pk", "scalar"])
@pytest.mark.parametrize("act", _HALF_ACTS)
def test_half_activation_true16_renders(act, usePK):
    # true16 arch: emit must succeed and never fall back to SDWA WORD selects.
    text = _module_for(act, isa=TRUE16_ISA, usePK=usePK)
    assert text
    assert "src0_sel:WORD" not in text
    assert "v_cndmask_b32" not in text


@pytest.mark.parametrize("usePK", [True, False], ids=["pk", "scalar"])
@pytest.mark.parametrize("act", _HALF_ACTS)
def test_half_activation_legacy_renders(act, usePK):
    # legacy arch: emit must succeed and never emit a true16 .l/.h suffix.
    text = _module_for(act, isa=LEGACY_ISA, usePK=usePK)
    assert text
    assert _no_true16_suffix(text)
    assert "v_cndmask_b16" not in text


def test_cndmask_op_switches_by_arch():
    # clippedrelu + leakyrelu are the activations that emit the conditional select.
    for act in ("clippedrelu", "leakyrelu"):
        assert "v_cndmask_b16" in _module_for(act, isa=TRUE16_ISA, usePK=True)
        assert "v_cndmask_b32" in _module_for(act, isa=LEGACY_ISA, usePK=True)


# ---------------------------------------------------------------------------
# Direct per-half helper coverage (both arch branches, both halves).
# ---------------------------------------------------------------------------
def test_addCmpF16Half_branches():
    am = A.ActivationModule()

    use_isa(TRUE16_ISA, 64)
    for i, suffix in ((0, "v0.l"), (1, "v0.h")):
        mod = Module("cmp")
        am.addCmpF16Half(mod, VCmpGTF16, vgpr(0), sgpr(1), i, "x > a")
        text = _render(mod)
        assert suffix in text
        assert "s1" in text  # scalar src passes through untagged

    use_isa(LEGACY_ISA, 64)
    for i, sel in ((0, "src0_sel:WORD_0"), (1, "src0_sel:WORD_1")):
        mod = Module("cmp")
        am.addCmpF16Half(mod, VCmpGTF16, vgpr(0), sgpr(1), i, "x > a")
        text = _render(mod)
        assert sel in text
        assert _no_true16_suffix(text)


def test_addValuF16Half_branches():
    am = A.ActivationModule()

    use_isa(TRUE16_ISA, 64)
    mod = Module("valu")
    am.addValuF16Half(mod, VMinF16, vgpr(0), sgpr(1), vgpr(2), 1, "min")
    text = _render(mod)
    assert "v0.h" in text and "v2.h" in text
    assert "s1" in text

    use_isa(LEGACY_ISA, 64)
    mod = Module("valu")
    am.addValuF16Half(mod, VMinF16, vgpr(0), sgpr(1), vgpr(2), 0, "min")
    text = _render(mod)
    assert "src0_sel:WORD_0" in text
    assert _no_true16_suffix(text)


def test_addCndMaskF16Half_branches():
    am = A.ActivationModule()

    use_isa(TRUE16_ISA, 64)
    mod = Module("cnd")
    am.addCndMaskF16Half(mod, vgpr(0), vgpr(1), vgpr(2), 0, "select")
    text = _render(mod)
    assert "v_cndmask_b16" in text
    assert "v0.l" in text

    use_isa(LEGACY_ISA, 64)
    mod = Module("cnd")
    am.addCndMaskF16Half(mod, vgpr(0), vgpr(1), vgpr(2), 1, "select")
    text = _render(mod)
    assert "v_cndmask_b32" in text
    assert "src0_sel:WORD_1" in text
    assert _no_true16_suffix(text)


def test_addTransF16Half_and_halfwise_branches():
    am = A.ActivationModule()

    use_isa(TRUE16_ISA, 64)
    mod = Module("trans")
    am.addTransF16Half(mod, VExpF16, 0, 1, "exp")
    assert "v0.h" in _render(mod)

    mod = Module("trans")
    am.addTransF16Halfwise(mod, VExpF16, 0, "exp")
    text = _render(mod)
    assert "v0.l" in text and "v0.h" in text  # both halves emitted

    use_isa(LEGACY_ISA, 64)
    mod = Module("trans")
    am.addTransF16Half(mod, VExpF16, 0, 1, "exp")
    text = _render(mod)
    assert "src0_sel:WORD_1" in text
    assert _no_true16_suffix(text)


def test_t16v_only_tags_vgpr():
    use_isa(TRUE16_ISA, 64)
    from rocisa.enum import HighBitSel

    tagged = A.ActivationModule._t16v(vgpr(3), HighBitSel.HIGH)
    assert str(tagged) == "v3.h"
    # Scalar operands are returned unchanged (no half-select tagging).
    passthrough = A.ActivationModule._t16v(sgpr(2), HighBitSel.HIGH)
    assert str(passthrough) == "s2"


# ---------------------------------------------------------------------------
# ActivationInline true16 constraint path.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("isa", [LEGACY_ISA, TRUE16_ISA], ids=["legacy", "true16"])
def test_activation_inline_body_renders(isa):
    use_isa(isa, 64)
    ai = A.ActivationInline(DataType("h"), False)
    body = ai.generateInlineAssemblyBody(4, "geluscaling")
    assert isinstance(body, str) and body


# ---------------------------------------------------------------------------
# PackData_F16: F32->F16 pack now goes through ECvtF32toF16(sel=LOW).
# ---------------------------------------------------------------------------
def test_packdata_f16_gwvw1_branches():
    from tensilelite.Components.PackData import PackData_F16

    packer = PackData_F16()

    use_isa(TRUE16_ISA, 64)
    text = _render(packer(gwvw=1, destIdx=0, elementSumIdx=4))
    assert "v_cvt_f16_f32" in text
    assert "v0.l" in text  # true16 writes the low half explicitly

    use_isa(LEGACY_ISA, 64)
    text = _render(packer(gwvw=1, destIdx=0, elementSumIdx=4))
    assert "v_cvt_f16_f32" in text
    assert _no_true16_suffix(text)

    # gwvw>=2 pack loop should also emit on both arches.
    for isa in (TRUE16_ISA, LEGACY_ISA):
        use_isa(isa, 64)
        assert _render(packer(gwvw=2, destIdx=0, elementSumIdx=4))
