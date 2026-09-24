# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Adaptor-side tests for true16 half-select support.

Mirrors ``rocisa/test/test_true16.py`` so the pure-Python adaptor and native
rocisa are held to the same observable contract:

  * ``RegisterContainer`` half-select rendering (.l / .h),
  * the true16 16-bit conditional select ``VCndMaskB16``,
  * the ``ECvt*`` helpers picking the true16 (NoSDWA) vs legacy (SDWA)
    encoding,
  * the NoSDWA-gated ``t16()`` helper.

Plus two things the native suite cannot cover, because they only exist on the
adaptor's logical-IR path:

  * ``_apply_true16`` re-hanging the operand half onto the instruction as
    True16Modifiers, in stinkytofu's wire values -- the
    ``t16 -> _apply_true16 -> to_stinky_logical`` derivation that
    ``LogicalToAsmPipelineTest.True16HalfSelectSurvivesLowering`` skips by
    injecting an already-derived modifier in C++.
  * the negative case: an untagged instruction must attach no modifier at
    all, so 32-bit and packed ops stay untouched.

Byte-equality of the emitted assembly across all three production paths for
t16-tagged instructions lives in ``tests/test_emission_consistency.py``.
"""

from __future__ import annotations

import copy
import os
import pickle
import shutil
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor import base as abase  # noqa: E402
from rocisa_stinkytofu_adaptor.code import Module  # noqa: E402
from rocisa_stinkytofu_adaptor.container import HolderContainer, vgpr  # noqa: E402
from rocisa_stinkytofu_adaptor.enum import HighBitSel  # noqa: E402
from rocisa_stinkytofu_adaptor.instruction import (  # noqa: E402
    CommonInstruction,
    ECvtF16toF32,
    ECvtF32toF16,
    VCndMaskB16,
    VMaxF16,
    _apply_true16,
    t16,
)

try:
    import stinkytofu as _stinky  # noqa: F401
    _STINKY_OK = True
except ImportError:
    _STINKY_OK = False


# gfx1250 is the NoSDWA (true16) target stinkytofu registers caps for; gfx942
# is a legacy SDWA target, reached through the native-rocisa cap fallback.
TRUE16_ISA = (12, 5, 0)
LEGACY_ISA = (9, 4, 2)


def _assembler_path():
    """Assembler for the legacy ISA's native cap fallback.

    ``caps.getCaps`` only knows stinkytofu-registered backends, so selecting
    a legacy ISA shells out to native rocisa -- which probes a real
    assembler. gfx1250 needs none.
    """
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    search_path = os.pathsep.join(
        [
            os.path.join(rocm_path, "bin"),
            os.path.join(rocm_path, "lib", "llvm", "bin"),
            os.environ.get("PATH", ""),
        ]
    )
    return shutil.which("amdclang++", path=search_path)


_ASM_PATH = _assembler_path()


class _IsaCase(unittest.TestCase):
    """Base restoring the process-global ISA state after each test.

    ``rocisa_stinkytofu_adaptor.base`` is a module-level singleton, so a test
    that pins an ISA would otherwise leak caps (notably gfx1250's
    ``HasVgprMSB``) into sibling test modules sharing the process.
    """

    def setUp(self):
        self._saved_data = dict(abase.getData())
        self._saved_kernel = abase.getKernel()

    def tearDown(self):
        abase.setData(self._saved_data)
        abase.setKernelInfo(self._saved_kernel)

    def use_isa(self, isa):
        """Pin @p isa, skipping when its caps need an unavailable assembler."""
        if isa != TRUE16_ISA and _ASM_PATH is None:
            self.skipTest(f"caps for {isa} need an assembler (none found)")
        abase.init(isa, _ASM_PATH or "", False)
        abase.setKernel(isa, 64)


class TestHalfSelectRender(_IsaCase):
    """``RegisterContainer`` half-select rendering (native: test_half_select_render)."""

    def test_lo_hi_render(self):
        # gfx942 keeps HasVgprMSB off, so a single VGPR renders without an
        # _hi pad -- same reason the native test picks the legacy ISA here.
        self.use_isa(LEGACY_ISA)

        self.assertEqual(str(vgpr(0)), "v0")
        self.assertEqual(str(vgpr(0).lo()), "v0.l")
        self.assertEqual(str(vgpr(0).hi()), "v0.h")

    def test_set_half_select(self):
        self.use_isa(LEGACY_ISA)

        reg = vgpr(0)
        reg.setHalfSelect(HighBitSel.HIGH)
        self.assertEqual(str(reg), "v0.h")
        reg.setHalfSelect(HighBitSel.LOW)
        self.assertEqual(str(reg), "v0.l")
        # NONE renders like an unset halfSelect, matching the native
        # ``*halfSelect != HighBitSel::NONE`` guard.
        reg.setHalfSelect(HighBitSel.NONE)
        self.assertEqual(str(reg), "v0")

    def test_lo_hi_do_not_mutate_source(self):
        # Native lo()/hi() return by value; the same VGPR is routinely reused
        # for both halves, so tagging one must not disturb the other.
        self.use_isa(LEGACY_ISA)

        reg = vgpr(0)
        self.assertEqual(str(reg.hi()), "v0.h")
        self.assertIsNone(reg.halfSelect)
        self.assertEqual(str(reg), "v0")

    def test_multi_register_has_no_half(self):
        # Native only appends halfStr on the regNum == 1 branches: a
        # half-word of a register *range* is not expressible.
        self.use_isa(LEGACY_ISA)

        self.assertEqual(str(vgpr(0, 2).hi()), "v[0:1]")

    def test_survives_deepcopy(self):
        self.use_isa(LEGACY_ISA)

        self.assertEqual(str(copy.deepcopy(vgpr(0).hi())), "v0.h")

    def test_survives_pickle(self):
        # Tensile ships RegisterContainers to ParallelMap2 workers by pickle,
        # so a tagged operand has to make the trip. The state is a plain int
        # rather than the enum member: the adaptor's enums claim
        # ``__module__ = "rocisa.enum"``, which pickle would resolve to native
        # rocisa's HighBitSel whenever both are loaded.
        self.use_isa(LEGACY_ISA)

        self.assertEqual(str(pickle.loads(pickle.dumps(vgpr(0).hi()))), "v0.h")
        self.assertEqual(str(pickle.loads(pickle.dumps(vgpr(0)))), "v0")

    def test_holder_copy_keeps_half(self):
        # ``replaceHolder`` swaps a HolderContainer for its getCopiedRC()
        # snapshot; dropping the half there would emit a suffix-less 16-bit
        # operand, which NoSDWA rejects.
        self.use_isa(LEGACY_ISA)

        holder = HolderContainer("v", 0, 1)
        holder.setHalfSelect(HighBitSel.HIGH)
        self.assertEqual(str(holder.getCopiedRC()), "v0.h")


class TestT16Gating(_IsaCase):
    """``t16`` NoSDWA gating (native: test_t16_gating)."""

    def test_tags_on_true16_target(self):
        self.use_isa(TRUE16_ISA)

        self.assertEqual(str(t16(vgpr(1), HighBitSel.HIGH)), "v1.h")
        self.assertEqual(str(t16(vgpr(1), HighBitSel.LOW)), "v1.l")

    def test_noop_on_legacy_target(self):
        self.use_isa(LEGACY_ISA)

        self.assertEqual(str(t16(vgpr(1), HighBitSel.HIGH)), "v1")
        self.assertEqual(str(t16(vgpr(1), HighBitSel.LOW)), "v1")

    def test_passes_through_non_registers(self):
        # ``inputWithHalf`` guards on the operand being a register, so an
        # immediate is returned untouched rather than rendered as "1.0.l".
        self.use_isa(TRUE16_ISA)

        self.assertEqual(t16(1.0, HighBitSel.LOW), 1.0)
        self.assertEqual(t16("vcc", HighBitSel.LOW), "vcc")
        self.assertIsNone(t16(None, HighBitSel.LOW))


class TestVCndMaskB16Construction(_IsaCase):
    """VCndMaskB16 (true16 16-bit select; native: test_vcndmask_b16_renders_halves)."""

    def _tagged(self):
        dst, src0, src1 = vgpr(0), vgpr(1), vgpr(2)
        dst.setHalfSelect(HighBitSel.LOW)
        src0.setHalfSelect(HighBitSel.LOW)
        src1.setHalfSelect(HighBitSel.HIGH)
        return VCndMaskB16(dst=dst, src0=src0, src1=src1)

    def test_construction_and_str(self):
        self.use_isa(TRUE16_ISA)

        inst = self._tagged()
        self.assertIsInstance(inst, CommonInstruction)
        self.assertEqual(inst.instStr, "v_cndmask_b16")
        text = str(inst)
        self.assertIn("v0.l", text)
        self.assertIn("v1.l", text)
        self.assertIn("v2.h", text)

    def test_deepcopy(self):
        self.use_isa(TRUE16_ISA)

        inst = self._tagged()
        c = copy.deepcopy(inst)
        self.assertIsInstance(c, VCndMaskB16)
        self.assertEqual(str(c), str(inst))

    def test_has_to_stinky_logical(self):
        self.use_isa(TRUE16_ISA)

        self.assertTrue(
            callable(getattr(self._tagged(), "to_stinky_logical", None))
        )

    @unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built")
    def test_collected_by_module(self):
        self.use_isa(TRUE16_ISA)

        m = Module()
        m.add(self._tagged())
        self.assertEqual(len(m._collect_logical_insts()), 1)


class TestECvtF16toF32(_IsaCase):
    """``ECvtF16toF32`` encoding choice (native: test_ecvt_f16_to_f32_true16_vs_legacy)."""

    def test_true16_uses_operand_suffix(self):
        self.use_isa(TRUE16_ISA)

        text = str(ECvtF16toF32(dst=vgpr(0), src=vgpr(1), sel=HighBitSel.HIGH))
        self.assertIn("v_cvt_f32_f16", text)
        self.assertIn("v1.h", text)
        self.assertNotIn("src0_sel", text)

    def test_legacy_uses_sdwa_src_sel(self):
        self.use_isa(LEGACY_ISA)

        text = str(ECvtF16toF32(dst=vgpr(0), src=vgpr(1), sel=HighBitSel.HIGH))
        self.assertIn("v_cvt_f32_f16", text)
        self.assertIn("src0_sel:WORD_1", text)
        self.assertNotIn(".h", text)


class TestECvtF32toF16(_IsaCase):
    """``ECvtF32toF16`` encoding choice (native: test_ecvt_f32_to_f16_true16_vs_legacy)."""

    def test_true16_defaults_to_low_half(self):
        self.use_isa(TRUE16_ISA)

        # A suffix-less 16-bit dst is illegal on NoSDWA, so sel=None has to
        # resolve to the low half rather than emit a bare register.
        self.assertIn("v0.l", str(ECvtF32toF16(dst=vgpr(0), src=vgpr(1))))

    def test_true16_honours_explicit_sel(self):
        self.use_isa(TRUE16_ISA)

        self.assertIn(
            "v0.h",
            str(ECvtF32toF16(dst=vgpr(0), src=vgpr(1), sel=HighBitSel.HIGH)),
        )

    def test_legacy_without_sel_is_plain_cvt(self):
        self.use_isa(LEGACY_ISA)

        text = str(ECvtF32toF16(dst=vgpr(0), src=vgpr(1)))
        self.assertIn("v_cvt_f16_f32 v0, v1", text)
        self.assertNotIn(".l", text)
        self.assertNotIn(".h", text)

    def test_legacy_with_sel_uses_sdwa_dst_sel(self):
        self.use_isa(LEGACY_ISA)

        self.assertIn(
            "dst_sel:WORD_1",
            str(ECvtF32toF16(dst=vgpr(0), src=vgpr(1), sel=HighBitSel.HIGH)),
        )


class _RecordingInst:
    """Stand-in for a stinkytofu logical instruction.

    The binding exposes ``set_true16`` but no getter, so recording the call is
    the only way to assert the derived wire values directly.
    """

    def __init__(self):
        self.calls = []

    def set_true16(self, dst0, dst1, srcs):
        self.calls.append((dst0, dst1, list(srcs)))


class TestApplyTrue16Derivation(_IsaCase):
    """``_apply_true16`` operand-half -> instruction-modifier derivation.

    This is the step ``LogicalToAsmPipelineTest.True16HalfSelectSurvivesLowering``
    deliberately skips (it injects a ready-made True16Modifiers in C++), and
    the port of ``attachTrue16ModifiersFromOperands``.
    """

    def test_derives_wire_values_from_operand_halves(self):
        self.use_isa(TRUE16_ISA)

        inst = _RecordingInst()
        _apply_true16(
            inst,
            vgpr(0).hi(),
            [vgpr(1).lo(), vgpr(2).hi()],
        )
        # stinkytofu's HighBitSel integers, not 0-based indices: the C++
        # bridge static_casts straight between the two enums.
        self.assertEqual(inst.calls, [(1, -1, [0, 1])])

    def test_untagged_operands_attach_no_modifier(self):
        self.use_isa(TRUE16_ISA)

        inst = _RecordingInst()
        _apply_true16(inst, vgpr(0), [vgpr(1), vgpr(2)])
        self.assertEqual(inst.calls, [])

    def test_non_register_operands_read_as_none(self):
        self.use_isa(TRUE16_ISA)

        inst = _RecordingInst()
        _apply_true16(inst, vgpr(0).lo(), [1.0, "vcc"])
        self.assertEqual(inst.calls, [(0, -1, [-1, -1])])

    def test_dst1_half_is_carried(self):
        self.use_isa(TRUE16_ISA)

        inst = _RecordingInst()
        _apply_true16(inst, vgpr(0).lo(), [vgpr(1).lo()], vgpr(2).hi())
        self.assertEqual(inst.calls, [(0, 1, [0])])

    def test_t16_tagged_operands_reach_the_derivation(self):
        # The full chain generators actually use: t16() tags the operand and
        # the factory's to_stinky_logical hands it to _apply_true16.
        self.use_isa(TRUE16_ISA)

        inst = _RecordingInst()
        _apply_true16(
            inst,
            t16(vgpr(0), HighBitSel.LOW),
            [t16(vgpr(1), HighBitSel.LOW), t16(vgpr(2), HighBitSel.HIGH)],
        )
        self.assertEqual(inst.calls, [(0, -1, [0, 1])])

    def test_legacy_target_derives_nothing(self):
        # t16 is a no-op off NoSDWA, so the same generator code must not grow
        # a true16 modifier on legacy targets.
        self.use_isa(LEGACY_ISA)

        inst = _RecordingInst()
        _apply_true16(
            inst,
            t16(vgpr(0), HighBitSel.LOW),
            [t16(vgpr(1), HighBitSel.LOW)],
        )
        self.assertEqual(inst.calls, [])

    def test_instruction_without_set_true16_is_ignored(self):
        self.use_isa(TRUE16_ISA)

        class _NoModifierSupport:
            pass

        # Must not raise: the collector calls _apply_true16 for every shim,
        # including ones whose stinkytofu counterpart has no true16 slot.
        _apply_true16(_NoModifierSupport(), vgpr(0).hi(), [vgpr(1).hi()])


class TestVMaxF16True16(_IsaCase):
    """A t16-tagged binary f16 ALU op, the shape AMax / Activation emit."""

    def test_renders_halves(self):
        self.use_isa(TRUE16_ISA)

        text = str(VMaxF16(
            dst=t16(vgpr(0), HighBitSel.LOW),
            src0=t16(vgpr(1), HighBitSel.LOW),
            src1=t16(vgpr(2), HighBitSel.HIGH),
        ))
        self.assertIn("v_max_f16", text)
        self.assertIn("v0.l", text)
        self.assertIn("v1.l", text)
        self.assertIn("v2.h", text)

    def test_legacy_renders_no_halves(self):
        self.use_isa(LEGACY_ISA)

        text = str(VMaxF16(
            dst=t16(vgpr(0), HighBitSel.LOW),
            src0=t16(vgpr(1), HighBitSel.LOW),
            src1=t16(vgpr(2), HighBitSel.HIGH),
        ))
        self.assertIn("v_max_f16", text)
        self.assertNotIn(".l", text)
        self.assertNotIn(".h", text)


if __name__ == "__main__":
    unittest.main()
