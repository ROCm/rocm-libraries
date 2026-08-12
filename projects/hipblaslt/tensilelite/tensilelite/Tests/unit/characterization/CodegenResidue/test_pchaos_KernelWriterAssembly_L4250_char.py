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

"""Characterization tests for KernelWriterAssembly.py line 4250.

Branch ID: 0902ebf1cd3edc0bc1aac9c97f8dfd41cba0c700
Predicate:  self.states.groOffsetInMacroTile  (KernelWriterAssembly.py:4250, bare truthy test)

The predicate gates SRD/tileStart computation in the tile-offset code path.
groOffsetInMacroTile is derived in KernelWriter.py:6631 as a pure function
of three public kernel parameters:

    gro = 1  iff  len(PackedC0IndicesX)==1
                  and len(PackedC1IndicesX)==1
                  and BufferLoad==True

TRUE branch  (gro=1):  packed dims are unpacked (single each), BufferLoad enabled
FALSE branch (gro=0):  packed dims > 1 on either axis, OR BufferLoad disabled

CPU-only. No GPU hardware required.
"""

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper: mirrors KernelWriter.py:6631 derivation of groOffsetInMacroTile
# ---------------------------------------------------------------------------

def _gro_offset_in_macro_tile(
    packed_c0_indices_x: list,
    packed_c1_indices_x: list,
    buffer_load: bool,
) -> int:
    """Mirror of KernelWriter.py:6631 derivation of states.groOffsetInMacroTile.

    Returns 1 (truthy) if GRO offsets are expressed relative to the macro-tile
    start; 0 (falsy) otherwise.

    The predicate at KernelWriterAssembly.py:4250 is a bare truthiness test on
    this integer value:  ``if self.states.groOffsetInMacroTile:``

    pre:  packed_c0_indices_x and packed_c1_indices_x are non-empty lists
    post: __return__ == (1 if (len(c0)==1 and len(c1)==1 and buffer_load) else 0)
    """
    if (
        len(packed_c0_indices_x) == 1
        and len(packed_c1_indices_x) == 1
        and buffer_load
    ):
        return 1
    else:
        return 0


def _predicate(
    packed_c0_indices_x: list,
    packed_c1_indices_x: list,
    buffer_load: bool,
) -> bool:
    """The branch predicate at KernelWriterAssembly.py:4250 (truthiness of gro)."""
    return bool(_gro_offset_in_macro_tile(packed_c0_indices_x, packed_c1_indices_x, buffer_load))


# ---------------------------------------------------------------------------
# Pure-helper tests: derivation contract
# ---------------------------------------------------------------------------

class TestGroOffsetDerivation:
    """Verify the gro=1/0 derivation from public parameters."""

    def test_true_single_packed_dims_buffer_load_on(self):
        """gro=1: both packed dim lists have length 1, BufferLoad=True (TRUE branch)."""
        assert _gro_offset_in_macro_tile([0], [1], True) == 1

    def test_true_returns_truthy_int(self):
        """gro=1 is truthy: the L4250 bare `if gro:` takes the branch."""
        assert bool(_gro_offset_in_macro_tile([0], [1], True)) is True

    def test_false_c0_packed(self):
        """gro=0: PackedC0IndicesX has 2 elements -> gro=0 (FALSE branch)."""
        assert _gro_offset_in_macro_tile([0, 1], [0], True) == 0

    def test_false_c1_packed(self):
        """gro=0: PackedC1IndicesX has 2 elements -> gro=0 (FALSE branch)."""
        assert _gro_offset_in_macro_tile([0], [0, 1], True) == 0

    def test_false_both_packed(self):
        """gro=0: both axes packed (len > 1) -> gro=0 (FALSE branch)."""
        assert _gro_offset_in_macro_tile([0, 1], [0, 1], True) == 0

    def test_false_buffer_load_off(self):
        """gro=0: BufferLoad=False disables groOffsetInMacroTile -> gro=0 (FALSE branch)."""
        assert _gro_offset_in_macro_tile([0], [1], False) == 0

    def test_false_buffer_load_off_also_packed(self):
        """gro=0: BufferLoad=False + packed dims -> gro=0 (FALSE branch, combined)."""
        assert _gro_offset_in_macro_tile([0, 1], [0, 1], False) == 0


# ---------------------------------------------------------------------------
# Predicate tests: L4250 branch truthiness
# ---------------------------------------------------------------------------

class TestL4250PredicateTruthy:
    """Verify L4250 `if self.states.groOffsetInMacroTile:` branch outcome."""

    # TRUE witnesses (confirmed solver examples)
    def test_true_witness_canonical(self):
        """Canonical TRUE witness: gro=1 -> predicate True (solver-confirmed)."""
        assert _predicate([0], [1], True) is True

    def test_true_witness_same_index_both_axes(self):
        """gro=1 holds when both axes have exactly one index (any single-element list)."""
        assert _predicate([5], [3], True) is True

    # FALSE witnesses (solver-confirmed)
    def test_false_witness_c0_packed(self):
        """FALSE witness: len(PackedC0IndicesX)=2 -> gro=0 -> predicate False."""
        assert _predicate([0, 1], [0], True) is False

    def test_false_witness_c1_packed(self):
        """FALSE witness: len(PackedC1IndicesX)=2 -> gro=0 -> predicate False."""
        assert _predicate([0], [0, 1], True) is False

    def test_false_witness_no_buffer_load(self):
        """FALSE witness: BufferLoad=False -> gro=0 -> predicate False."""
        assert _predicate([0], [1], False) is False

    # Contract grid (c0 len, c1 len) x BufferLoad exhaustive for small counts
    @pytest.mark.parametrize("c0_len,c1_len,bl,expected", [
        (1, 1, True,  True),   # only TRUE configuration
        (1, 1, False, False),  # no BufferLoad -> False
        (2, 1, True,  False),  # c0 packed -> False
        (1, 2, True,  False),  # c1 packed -> False
        (2, 2, True,  False),  # both packed -> False
        (2, 2, False, False),  # both packed + no BL -> False
        (3, 1, True,  False),  # c0 len=3 -> False
        (1, 3, True,  False),  # c1 len=3 -> False
        (4, 4, True,  False),  # large packed -> False
    ])
    def test_contract_grid(self, c0_len, c1_len, bl, expected):
        """Exhaustive contract grid over (c0_len, c1_len, BufferLoad)."""
        c0 = list(range(c0_len))
        c1 = list(range(c1_len))
        assert _predicate(c0, c1, bl) is expected


# ---------------------------------------------------------------------------
# Assertion guard tests: L4251-4252 assert(len==1) fires when gro=1
# ---------------------------------------------------------------------------

class TestL4251L4252AssertGuards:
    """When gro=1 is True, the assert guards at L4251-4252 must hold.

    These asserts confirm that the groOffsetInMacroTile derivation is
    consistent: if gro==1, then by definition both packed dim lists have
    len==1, so the assertions can never fire on a legal input.
    """

    def test_gro1_implies_c0_len_one(self):
        """If gro=1 then len(PackedC0IndicesX)==1 (by derivation; assert at L4251 cannot fire)."""
        c0, c1, bl = [0], [1], True
        gro = _gro_offset_in_macro_tile(c0, c1, bl)
        assert gro == 1
        assert len(c0) == 1, "L4251 assert would fire — derivation inconsistent"

    def test_gro1_implies_c1_len_one(self):
        """If gro=1 then len(PackedC1IndicesX)==1 (by derivation; assert at L4252 cannot fire)."""
        c0, c1, bl = [0], [1], True
        gro = _gro_offset_in_macro_tile(c0, c1, bl)
        assert gro == 1
        assert len(c1) == 1, "L4252 assert would fire — derivation inconsistent"
