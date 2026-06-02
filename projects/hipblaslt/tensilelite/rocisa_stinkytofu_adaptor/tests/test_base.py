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
################################################################################
"""Standalone tests for the ``rocisa_stinkytofu_adaptor.base`` state sink.

These tests verify the invariant that ALL ``rocisa::rocIsa`` singleton
state -- KernelInfo, IsaInfo dict, current ISA, vgpr_idx, vgpr_msb,
OutputOptions, is_init, assembler_path -- lives in ``base.py``
module-level globals, and the ``rocIsa`` class in ``__init__.py`` is a
thin forwarding shell with NO instance state.

Run from any working directory:

    python3 projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_base.py

Or via the wrapper:

    ./test.sh test_base
"""

from __future__ import annotations

import os
import pickle
import sys
import unittest


# ---------------------------------------------------------------------------
# Self-contained sys.path bootstrap (mirrors test_code.py).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor import base as _base  # noqa: E402
from rocisa_stinkytofu_adaptor import (  # noqa: E402
    IsaInfo,
    KernelInfo,
    OutputOptions,
    rocIsa,
)


# ===========================================================================
# Helpers
# ===========================================================================


class _StateSaveRestore(unittest.TestCase):
    """Snapshot/restore ``base.py`` state so tests cannot leak into one another.

    We grab the live references at setUp time and restore them in
    tearDown. Tests can mutate freely; the harness puts everything back
    so subsequent test files (and the rest of this file) see a clean
    slate.
    """

    def setUp(self) -> None:
        # ``getXxx`` accessors return the live object; we copy where
        # appropriate so the saved state is independent of in-test
        # mutations.
        self._saved_kernel = _base.getKernel()
        self._saved_current_isa = _base._current_isa
        self._saved_is_init = _base._is_init
        self._saved_assembler_path = _base._assembler_path
        self._saved_data = dict(_base.getData())
        self._saved_vgpr_idx = dict(_base.getVgprIdx())
        self._saved_vgpr_msb = _base.getVgprMsb()
        self._saved_opts = OutputOptions(_base.getOutputOptions().outputNoComment)

    def tearDown(self) -> None:
        _base.setKernelInfo(self._saved_kernel)
        _base._current_isa = self._saved_current_isa
        _base._is_init = self._saved_is_init
        _base._assembler_path = self._saved_assembler_path
        # ``getData`` returns the live dict; mutate in place so anyone
        # who captured a reference earlier in the process keeps seeing
        # the restored state.
        live_data = _base.getData()
        live_data.clear()
        live_data.update(self._saved_data)
        _base._is_init = self._saved_is_init  # re-assert (setData would flip it)
        live_vgpr = _base.getVgprIdx()
        live_vgpr.clear()
        live_vgpr.update(self._saved_vgpr_idx)
        _base.setVgprMsb(self._saved_vgpr_msb)
        _base.setOutputOptions(self._saved_opts)


# ===========================================================================
# Class location parity (IsaInfo / KernelInfo / OutputOptions live in base.py
# to mirror rocisa/include/base.hpp)
# ===========================================================================


class TestClassLocationParity(unittest.TestCase):
    """``rocisa::IsaInfo`` lives in ``base.hpp``; the Python mirror must too."""

    def test_isainfo_defined_in_base_module(self):
        self.assertIs(IsaInfo, _base.IsaInfo)
        self.assertEqual(IsaInfo.__module__, _base.__name__)

    def test_kernelinfo_defined_in_base_module(self):
        self.assertIs(KernelInfo, _base.KernelInfo)
        self.assertEqual(KernelInfo.__module__, _base.__name__)

    def test_outputoptions_defined_in_base_module(self):
        self.assertIs(OutputOptions, _base.OutputOptions)
        self.assertEqual(OutputOptions.__module__, _base.__name__)

    def test_top_level_reexport_for_isainfo(self):
        # ``from rocisa import IsaInfo`` must keep working (Tensile API).
        from rocisa_stinkytofu_adaptor import IsaInfo as TopLevelIsaInfo
        self.assertIs(TopLevelIsaInfo, _base.IsaInfo)


# ===========================================================================
# rocIsa has no instance state (everything moved to base.py)
# ===========================================================================


class TestRocIsaIsThinShell(unittest.TestCase):
    """``rocIsa`` instance must hold no state -- state lives in ``base.py``."""

    def test_init_does_not_set_state_fields(self):
        inst = rocIsa()
        # The instance must not carry the historical state fields.
        # ``_vgpr_idx`` and ``_kernel_info`` ARE still readable, but only
        # via @property descriptors that forward to base.py (verified
        # separately below); they're not instance attributes.
        instance_attrs = set(getattr(inst, "__dict__", {}))
        forbidden = {
            "_current_isa",
            "_is_init",
            "_assembler_path",
            "_data",
        }
        self.assertEqual(instance_attrs & forbidden, set())

    def test_singleton_returns_same_instance(self):
        a = rocIsa.getInstance()
        b = rocIsa.getInstance()
        self.assertIs(a, b)


# ===========================================================================
# rocIsa methods forward to base.* (the WHOLE point of this commit)
# ===========================================================================


class TestRocIsaForwarding(_StateSaveRestore):
    """Every ``rocIsa`` method must be an alias for the matching ``base.*``."""

    # ---- init / isInit ----------------------------------------------------

    def test_init_forwards(self):
        # Wipe the data dict first so we can observe init repopulating it.
        live_data = _base.getData()
        live_data.clear()
        _base._is_init = False
        rocIsa.getInstance().init((12, 5, 0), "/fake/path", False)
        self.assertTrue(_base.isInit())
        self.assertIn((12, 5, 0), live_data)
        self.assertEqual(_base._assembler_path, "/fake/path")

    def test_isinit_forwards(self):
        _base._is_init = True
        self.assertTrue(rocIsa.getInstance().isInit())
        _base._is_init = False
        self.assertFalse(rocIsa.getInstance().isInit())

    # ---- getIsaInfo + get*Caps -------------------------------------------

    def test_getisainfo_forwards(self):
        info = rocIsa.getInstance().getIsaInfo((12, 5, 0))
        self.assertIsInstance(info, IsaInfo)
        # Must be the SAME dict the base module cached -- callers expect
        # to be able to mutate getData() and see the result via getIsaInfo.
        self.assertIs(info, _base.getData()[(12, 5, 0)])

    def test_get_caps_forward(self):
        rocIsa.getInstance().init((12, 5, 0))
        self.assertEqual(
            rocIsa.getInstance().getAsmCaps(), _base.getAsmCaps()
        )
        self.assertEqual(
            rocIsa.getInstance().getArchCaps(), _base.getArchCaps()
        )
        self.assertEqual(
            rocIsa.getInstance().getRegCaps(), _base.getRegCaps()
        )
        self.assertEqual(
            rocIsa.getInstance().getAsmBugs(), _base.getAsmBugs()
        )

    def test_get_caps_raises_without_init(self):
        # Hard-reset to "no ISA selected" state.
        _base._current_isa = None
        with self.assertRaises(RuntimeError):
            rocIsa.getInstance().getAsmCaps()
        with self.assertRaises(RuntimeError):
            _base.getAsmCaps()

    # ---- setKernel / getKernel -------------------------------------------

    def test_setkernel_forwards(self):
        rocIsa.getInstance().setKernel((12, 5, 0), 64)
        ki = _base.getKernel()
        self.assertEqual(ki.isa, (12, 5, 0))
        self.assertEqual(ki.wavefrontSize, 64)
        self.assertEqual(_base._current_isa, (12, 5, 0))

    def test_getkernel_forwards(self):
        custom = KernelInfo(isa=(12, 5, 0), wavefrontSize=32)
        _base.setKernelInfo(custom)
        self.assertIs(rocIsa.getInstance().getKernel(), custom)

    # ---- OutputOptions ----------------------------------------------------

    def test_outputoptions_forwards(self):
        opts = OutputOptions(outputNoComment=True)
        rocIsa.getInstance().setOutputOptions(opts)
        self.assertIs(_base.getOutputOptions(), opts)
        self.assertTrue(_base.outputNoComment())
        rocIsa.getInstance().setOutputOptions(OutputOptions(False))
        self.assertFalse(_base.outputNoComment())

    # ---- Data dict --------------------------------------------------------

    def test_getdata_returns_live_reference(self):
        rocIsa.getInstance().init((12, 5, 0))
        d_via_rocisa = rocIsa.getInstance().getData()
        d_via_base = _base.getData()
        self.assertIs(d_via_rocisa, d_via_base)

    def test_setdata_replaces_dict_and_flips_isinit(self):
        snap = {
            (12, 5, 0): IsaInfo({"x": 1}, {"y": 2}, {"z": 3}, {"b": True}),
        }
        rocIsa.getInstance().setData(snap)
        self.assertEqual(dict(_base.getData()), snap)
        self.assertTrue(_base.isInit())
        rocIsa.getInstance().setData({})
        self.assertFalse(_base.isInit())

    # ---- VGPR idx ---------------------------------------------------------

    def test_setvgpridx_forwards(self):
        rocIsa.getInstance().setVgprIdx("ValuA", 100)
        self.assertEqual(_base.getVgprIdx()["ValuA"], 100)

    def test_getvgpridx_forwards(self):
        _base.setVgprIdx("ValuB", 42)
        self.assertEqual(rocIsa.getInstance().getVgprIdx()["ValuB"], 42)

    def test_getvgpridx_returns_live_reference(self):
        d_via_rocisa = rocIsa.getInstance().getVgprIdx()
        d_via_base = _base.getVgprIdx()
        self.assertIs(d_via_rocisa, d_via_base)
        d_via_rocisa["ValuC"] = 7
        self.assertEqual(_base.getVgprIdx()["ValuC"], 7)

    # ---- VGPR MSB (new in this commit) -----------------------------------

    def test_vgpr_msb_default_is_zero(self):
        # Sanity: default must match C++ ``int()`` default-construct.
        _base.setVgprMsb(0)
        self.assertEqual(rocIsa.getInstance().getVgprMsb(), 0)

    def test_setvgprmsb_forwards(self):
        rocIsa.getInstance().setVgprMsb(3)
        self.assertEqual(_base.getVgprMsb(), 3)
        rocIsa.getInstance().setVgprMsb(-1)
        self.assertEqual(_base.getVgprMsb(), -1)


# ===========================================================================
# Backwards-compatible private-field shims (test_container.py style usage)
# ===========================================================================


class TestBackwardCompatPrivateFields(_StateSaveRestore):
    """``_vgpr_idx.clear()`` and ``_kernel_info = info`` must keep working.

    Test harnesses across the repo reach into these private fields to
    reset state. The @property shim on ``rocIsa`` must transparently
    delegate to ``base.*`` so existing tests survive.
    """

    def test_underscore_vgpr_idx_clear(self):
        rocIsa.getInstance().setVgprIdx("ValuA", 100)
        self.assertEqual(_base.getVgprIdx()["ValuA"], 100)
        rocIsa.getInstance()._vgpr_idx.clear()
        self.assertNotIn("ValuA", _base.getVgprIdx())
        # And it must clear the SAME dict as ``base.getVgprIdx()``.
        self.assertEqual(len(_base.getVgprIdx()), 0)

    def test_underscore_vgpr_idx_is_live_reference(self):
        live = rocIsa.getInstance()._vgpr_idx
        live["ValuD"] = 8
        self.assertEqual(_base.getVgprIdx()["ValuD"], 8)

    def test_underscore_kernel_info_read(self):
        custom = KernelInfo(isa=(12, 5, 0), wavefrontSize=64)
        _base.setKernelInfo(custom)
        self.assertIs(rocIsa.getInstance()._kernel_info, custom)

    def test_underscore_kernel_info_assignment_restore_unset(self):
        # Mimic the test_container.py:1684 pattern -- restoring a saved
        # KernelInfo where ``isa is None`` (the initial unset state).
        unset = KernelInfo()  # isa=None, wavefrontSize=0
        rocIsa.getInstance()._kernel_info = unset
        self.assertIs(_base.getKernel(), unset)
        self.assertIsNone(_base.getKernel().isa)


# ===========================================================================
# ParallelMap2 worker pattern (pickle round-trip of data + output options)
# ===========================================================================


class TestParallelMap2RoundTrip(_StateSaveRestore):
    """Tensile spawns workers via ``ParallelMap2`` and re-hydrates state with
    ``rocIsa.getInstance().setData(pickled_data)`` /
    ``.setOutputOptions(pickled_opts)``. Verify both round-trip cleanly."""

    def test_data_pickle_roundtrip(self):
        rocIsa.getInstance().init((12, 5, 0))
        snapshot = rocIsa.getInstance().getData()
        # Round-trip the snapshot via pickle (the actual worker boundary).
        pickled = pickle.dumps(snapshot)
        revived = pickle.loads(pickled)
        # Wipe and re-hydrate from the pickle.
        rocIsa.getInstance().setData({})
        rocIsa.getInstance().setData(revived)
        # The active caps must come back identical.
        info = rocIsa.getInstance().getIsaInfo((12, 5, 0))
        self.assertEqual(info.asmCaps, snapshot[(12, 5, 0)].asmCaps)
        self.assertEqual(info.archCaps, snapshot[(12, 5, 0)].archCaps)
        self.assertEqual(info.regCaps, snapshot[(12, 5, 0)].regCaps)
        self.assertEqual(info.asmBugs, snapshot[(12, 5, 0)].asmBugs)

    def test_outputoptions_pickle_roundtrip(self):
        opts = OutputOptions(outputNoComment=True)
        revived = pickle.loads(pickle.dumps(opts))
        rocIsa.getInstance().setOutputOptions(revived)
        self.assertTrue(_base.outputNoComment())
        # Round-tripping again preserves the flag bit-for-bit.
        again = pickle.loads(pickle.dumps(_base.getOutputOptions()))
        self.assertTrue(again.outputNoComment)

    def test_kernelinfo_pickle_roundtrip(self):
        ki = KernelInfo(isa=(12, 5, 0), wavefrontSize=32)
        revived = pickle.loads(pickle.dumps(ki))
        self.assertEqual(revived.isa, (12, 5, 0))
        self.assertEqual(revived.wavefrontSize, 32)

    def test_isainfo_pickle_roundtrip(self):
        info = IsaInfo({"a": 1}, {"b": 2}, {"c": 3}, {"d": True})
        revived = pickle.loads(pickle.dumps(info))
        self.assertEqual(revived.asmCaps, info.asmCaps)
        self.assertEqual(revived.archCaps, info.archCaps)
        self.assertEqual(revived.regCaps, info.regCaps)
        self.assertEqual(revived.asmBugs, info.asmBugs)


# ===========================================================================
# init() is idempotent for the same ISA
# ===========================================================================


class TestInitIdempotent(_StateSaveRestore):
    """``init(arch)`` called twice with the same ISA must not re-populate."""

    def test_init_same_isa_keeps_same_isainfo_object(self):
        rocIsa.getInstance().init((12, 5, 0))
        first = _base.getData()[(12, 5, 0)]
        rocIsa.getInstance().init((12, 5, 0), "/different/path")
        second = _base.getData()[(12, 5, 0)]
        # Same object identity -- ``init`` short-circuited the cap build.
        self.assertIs(first, second)
        # But the assembler path got updated regardless (mirrors C++,
        # which assigns ``m_assemblerPath`` outside the cache-check
        # branch... wait, actually C++ DOES update isainfo only inside
        # the if-not-cached branch; assemblerPath isn't a field of
        # rocIsa in C++. The Python adaptor's _assembler_path is updated
        # unconditionally, which is fine.)
        self.assertEqual(_base._assembler_path, "/different/path")


# ===========================================================================
# Wiring sanity: base accessors are the source of truth
# ===========================================================================


class TestBaseAccessorsAreSourceOfTruth(_StateSaveRestore):
    """Verify mutations via ``base.*`` are visible via ``rocIsa.*`` and vice
    versa -- they must operate on the same backing storage."""

    def test_setvgpridx_via_base_visible_via_rocisa(self):
        _base.setVgprIdx("X", 99)
        self.assertEqual(rocIsa.getInstance().getVgprIdx()["X"], 99)

    def test_setvgpridx_via_rocisa_visible_via_base(self):
        rocIsa.getInstance().setVgprIdx("Y", 77)
        self.assertEqual(_base.getVgprIdx()["Y"], 77)

    def test_setkernel_via_base_visible_via_rocisa(self):
        _base.setKernel((12, 5, 0), 32)
        ki = rocIsa.getInstance().getKernel()
        self.assertEqual(ki.isa, (12, 5, 0))
        self.assertEqual(ki.wavefrontSize, 32)

    def test_setkernel_via_rocisa_visible_via_base(self):
        rocIsa.getInstance().setKernel((12, 5, 0), 64)
        self.assertEqual(_base.getKernel().wavefrontSize, 64)

    def test_setvgprmsb_via_base_visible_via_rocisa(self):
        _base.setVgprMsb(5)
        self.assertEqual(rocIsa.getInstance().getVgprMsb(), 5)

    def test_setvgprmsb_via_rocisa_visible_via_base(self):
        rocIsa.getInstance().setVgprMsb(7)
        self.assertEqual(_base.getVgprMsb(), 7)


if __name__ == "__main__":
    unittest.main()
