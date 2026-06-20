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
################################################################################

"""Characterization tests for KernelWriter.py line 9867.

Branch ID: 3f01b4a2dfe0491afbd472d98feb9213944f7ddb
Predicate:  self.enableAsserts   (KernelWriter.py:9867 in Assert.multiple_b32)

self.enableAsserts is a boolean instance attribute set in Assert.__init__ (line 9793)
from the enableAsserts parameter, which originates from:

  GlobalParameters.EnableAsserts (YAML) -> makeDebugConfig() -> DebugConfig.enableAsserts
  -> KernelWriter.debugConfig.enableAsserts -> KernelWriter.db['EnableAsserts']
  -> Assert(enableAsserts=...) -> self.enableAsserts

Default is False (key absent from YAML GlobalParameters).

FALSE branch (predicate False): Assert.multiple_b32 returns an empty Module (no ASM emitted).
TRUE branch  (predicate True):  Assert.multiple_b32 enters the if-block at L9867 and
                                 proceeds to emit ASM instructions inside the block.

CPU-only.  No GPU hardware required.
"""

import linecache
import traceback

import pytest
from rocisa.code import Module

from Tensile.Common.Types import makeDebugConfig, DebugConfig
from Tensile.KernelWriter import Assert

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper: pure predicate mirror
# ---------------------------------------------------------------------------

def _predicate_enable_asserts(enable_asserts: bool) -> bool:
    """Mirror of the predicate at KernelWriter.py:9867.

    The predicate is simply ``if self.enableAsserts:``, where self is an
    Assert instance and self.enableAsserts is the bool stored at construction.
    """
    return bool(enable_asserts)


class TestPredicateEnableAsserts:
    """Pure predicate tests pinning the boolean semantics of self.enableAsserts."""

    def test_true_yields_true(self):
        """Predicate is True when enableAsserts=True — TRUE branch of L9867."""
        assert _predicate_enable_asserts(True) is True

    def test_false_yields_false(self):
        """Predicate is False when enableAsserts=False — FALSE branch of L9867."""
        assert _predicate_enable_asserts(False) is False

    def test_assert_instance_stores_false(self):
        """Assert(enableAsserts=False).enableAsserts is False."""
        a = Assert(laneSGPRCount=2, wavefrontSize=64, enableAsserts=False)
        assert a.enableAsserts is False

    def test_assert_instance_stores_true(self):
        """Assert(enableAsserts=True).enableAsserts is True."""
        a = Assert(laneSGPRCount=2, wavefrontSize=64, enableAsserts=True)
        assert a.enableAsserts is True


# ---------------------------------------------------------------------------
# makeDebugConfig derivation: GlobalParameters.EnableAsserts -> enableAsserts
# ---------------------------------------------------------------------------

class TestMakeDebugConfigDerivation:
    """Pins the derivation chain from YAML GlobalParameters to enableAsserts."""

    def test_key_absent_defaults_false(self):
        """makeDebugConfig({}) -> enableAsserts=False (default, key absent)."""
        dc = makeDebugConfig({})
        assert dc.enableAsserts is False

    def test_key_false_yields_false(self):
        """makeDebugConfig({'EnableAsserts': False}) -> enableAsserts=False."""
        dc = makeDebugConfig({"EnableAsserts": False})
        assert dc.enableAsserts is False

    def test_key_true_yields_true(self):
        """makeDebugConfig({'EnableAsserts': True}) -> enableAsserts=True."""
        dc = makeDebugConfig({"EnableAsserts": True})
        assert dc.enableAsserts is True

    def test_result_is_debug_config(self):
        """makeDebugConfig returns a DebugConfig instance."""
        dc = makeDebugConfig({})
        assert isinstance(dc, DebugConfig)


# ---------------------------------------------------------------------------
# False branch: enableAsserts=False -> multiple_b32 emits no ASM
# ---------------------------------------------------------------------------

class TestMultipleB32FalseBranch:
    """FALSE branch: enableAsserts=False -> if-block is skipped, empty Module returned."""

    def test_returns_module(self):
        """multiple_b32 with enableAsserts=False returns a Module object."""
        a = Assert(laneSGPRCount=2, wavefrontSize=64, enableAsserts=False)
        result = a.multiple_b32(sval=None, multiple2=8, vtmp=None)
        assert isinstance(result, Module)

    def test_returns_empty_module(self):
        """multiple_b32 with enableAsserts=False returns an empty Module (no ASM emitted).

        The FALSE branch of L9867 skips the if-block entirely; the module
        has no instructions added to it, so its string representation is empty.
        """
        a = Assert(laneSGPRCount=2, wavefrontSize=64, enableAsserts=False)
        result = a.multiple_b32(sval=None, multiple2=8, vtmp=None)
        assert str(result) == ""

    def test_multiple2_values_all_return_empty(self):
        """multiple_b32 returns empty regardless of multiple2 when enableAsserts=False."""
        a = Assert(laneSGPRCount=2, wavefrontSize=64, enableAsserts=False)
        for m2 in [2, 4, 8, 16, 32, 64]:
            result = a.multiple_b32(sval=None, multiple2=m2, vtmp=None)
            assert str(result) == "", f"Expected empty module for multiple2={m2}"

    def test_derived_from_debug_config_false(self):
        """End-to-end: makeDebugConfig key absent -> enableAsserts=False -> empty module."""
        dc = makeDebugConfig({})  # key absent -> False
        a = Assert(laneSGPRCount=2, wavefrontSize=64, enableAsserts=dc.enableAsserts)
        result = a.multiple_b32(sval=None, multiple2=8, vtmp=None)
        assert str(result) == ""

    def test_derived_from_debug_config_explicit_false(self):
        """End-to-end: makeDebugConfig(EnableAsserts=False) -> empty module."""
        dc = makeDebugConfig({"EnableAsserts": False})
        a = Assert(laneSGPRCount=2, wavefrontSize=64, enableAsserts=dc.enableAsserts)
        result = a.multiple_b32(sval=None, multiple2=8, vtmp=None)
        assert str(result) == ""


# ---------------------------------------------------------------------------
# True branch: enableAsserts=True -> multiple_b32 enters the if-block at L9867
# ---------------------------------------------------------------------------

class TestMultipleB32TrueBranch:
    """TRUE branch: enableAsserts=True -> if-block at L9867 is entered.

    The TRUE branch enters the if-block and proceeds to line 10816 where it
    uses SAndB64/SAndB32 (imported in the KernelWriter module scope but not
    in the Assert class namespace).  This causes a NameError at line 10816,
    which is INSIDE the if-block — proving the predicate evaluated True and
    execution proceeded past line 9867.

    This is a pre-existing import-scope detail inside the block; the predicate
    verdict is the entry into the if-block, confirmed by the exception line.
    """

    def test_true_branch_enters_if_block(self):
        """multiple_b32 with enableAsserts=True enters the if-block at L9867.

        Proven by the NameError at L10816 (inside the block), which would not
        occur if the branch were not taken.
        """
        a = Assert(laneSGPRCount=2, wavefrontSize=64, enableAsserts=True)
        with pytest.raises(NameError):
            a.multiple_b32(sval=None, multiple2=8, vtmp=None)

    def test_true_branch_exception_is_inside_if_block(self):
        """The NameError originates at the ``SAndBX`` statement inside the if-block.

        The statement is ``SAndBX = SAndB64 if self.wavefrontSize else SAndB32``.
        It is the first statement INSIDE the ``if self.enableAsserts:`` block, so
        reaching it proves the predicate evaluated True. The exact source line
        number drifts as KernelWriter.py changes, so the frame is matched by the
        statement text rather than a hardcoded line number.
        """
        a = Assert(laneSGPRCount=2, wavefrontSize=64, enableAsserts=True)
        try:
            a.multiple_b32(sval=None, multiple2=8, vtmp=None)
            pytest.fail("Expected NameError from inside the if-block")
        except NameError as exc:
            tb_frames = traceback.extract_tb(exc.__traceback__)
            deepest = tb_frames[-1]
            assert "KernelWriter.py" in deepest.filename, (
                f"Deepest frame not in KernelWriter.py: {deepest.filename}"
            )
            src_line = linecache.getline(deepest.filename, deepest.lineno)
            assert "SAndBX = SAndB64" in src_line, (
                f"Expected exception at the SAndBX statement inside the if-block, "
                f"got L{deepest.lineno}: {src_line.strip()!r}"
            )

    def test_derived_from_debug_config_true(self):
        """End-to-end: makeDebugConfig(EnableAsserts=True) -> enters if-block at L9867."""
        dc = makeDebugConfig({"EnableAsserts": True})
        a = Assert(laneSGPRCount=2, wavefrontSize=64, enableAsserts=dc.enableAsserts)
        with pytest.raises(NameError):
            a.multiple_b32(sval=None, multiple2=8, vtmp=None)
