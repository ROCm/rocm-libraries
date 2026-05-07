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
"""Unit tests for ``Tensile.Components.register.Register``.

Each test pins one behaviour. The Register API exposed:

  * Form predicates: ``is_numeric()``, ``is_symbolic()``
  * Range accessors: ``lo()``, ``hi()``
  * Comparison: ``overlaps(other)``, ``contains(other)``,
    ``intersection(other)``
  * Construction: ``from_rocisa(rc)``
  * Dedup support: ``signature()``
  * Type predicate: ``is_register(x)`` (static)
"""
import pytest

from rocisa.container import vgpr, sgpr, mgpr, accvgpr

from Tensile.Components.register import Register


# =============================================================================
# Numeric construction + form predicates
# =============================================================================


class TestNumericConstruction:
    def test_from_rocisa_numeric_vgpr(self):
        r = Register.from_rocisa(vgpr(8, 4))
        assert r.reg_type == "v"
        assert r.name is None
        assert r.base == 8
        assert r.count == 4

    def test_numeric_form_predicates(self):
        r = Register.from_rocisa(vgpr(8, 4))
        assert r.is_numeric() is True
        assert r.is_symbolic() is False

    def test_lo_and_hi_for_numeric(self):
        r = Register.from_rocisa(vgpr(8, 4))
        assert r.lo() == 8
        assert r.hi() == 12  # 8 + 4

    def test_rocisa_container_preserved(self):
        rc = vgpr(8, 4)
        r = Register.from_rocisa(rc)
        assert r.rocisa_container is rc

    def test_count_default_one(self):
        # mgpr produces a single-register container
        r = Register.from_rocisa(mgpr(0))
        assert r.count == 1
        assert r.lo() == 0
        assert r.hi() == 1

    def test_post_init_rejects_zero_count(self):
        with pytest.raises(ValueError):
            Register(reg_type="v", name=None, base=0, count=0)

    def test_post_init_rejects_negative_count(self):
        with pytest.raises(ValueError):
            Register(reg_type="v", name=None, base=0, count=-1)


# =============================================================================
# Numeric overlap
# =============================================================================


class TestNumericOverlap:
    def test_self_overlaps(self):
        r = Register.from_rocisa(vgpr(8, 4))
        assert r.overlaps(r) is True

    def test_disjoint_no_overlap(self):
        a = Register.from_rocisa(vgpr(8, 4))     # v[8:12)
        b = Register.from_rocisa(vgpr(20, 4))    # v[20:24)
        assert a.overlaps(b) is False
        assert b.overlaps(a) is False

    def test_partial_overlap(self):
        a = Register.from_rocisa(vgpr(8, 4))     # v[8:12)
        b = Register.from_rocisa(vgpr(10, 4))    # v[10:14)
        assert a.overlaps(b) is True
        assert b.overlaps(a) is True

    def test_boundary_no_overlap(self):
        # v[8:12) and v[12:16) — touch at 12 but don't overlap (hi exclusive)
        a = Register.from_rocisa(vgpr(8, 4))
        b = Register.from_rocisa(vgpr(12, 4))
        assert a.overlaps(b) is False
        assert b.overlaps(a) is False

    def test_overlap_is_symmetric(self):
        a = Register.from_rocisa(vgpr(8, 8))
        b = Register.from_rocisa(vgpr(10, 2))
        assert a.overlaps(b) == b.overlaps(a)


# =============================================================================
# Numeric containment
# =============================================================================


class TestNumericContainment:
    def test_self_contains(self):
        r = Register.from_rocisa(vgpr(8, 4))
        assert r.contains(r) is True

    def test_proper_containment(self):
        outer = Register.from_rocisa(vgpr(8, 8))   # v[8:16)
        inner = Register.from_rocisa(vgpr(10, 4))  # v[10:14)
        assert outer.contains(inner) is True
        assert inner.contains(outer) is False

    def test_equal_ranges_contain_each_other(self):
        a = Register.from_rocisa(vgpr(8, 4))
        b = Register.from_rocisa(vgpr(8, 4))
        assert a.contains(b) is True
        assert b.contains(a) is True

    def test_disjoint_no_containment(self):
        a = Register.from_rocisa(vgpr(8, 4))
        b = Register.from_rocisa(vgpr(20, 4))
        assert a.contains(b) is False
        assert b.contains(a) is False

    def test_partial_overlap_no_containment(self):
        a = Register.from_rocisa(vgpr(8, 4))     # v[8:12)
        b = Register.from_rocisa(vgpr(10, 4))    # v[10:14)
        assert a.contains(b) is False
        assert b.contains(a) is False


# =============================================================================
# Cross-type rejection
# =============================================================================


class TestCrossTypeRejection:
    def test_vgpr_sgpr_no_overlap_even_at_same_index(self):
        v = Register.from_rocisa(vgpr(8, 4))
        s = Register.from_rocisa(sgpr(8, 4))
        assert v.overlaps(s) is False
        assert s.overlaps(v) is False

    def test_vgpr_sgpr_no_containment_even_at_same_index(self):
        v = Register.from_rocisa(vgpr(8, 4))
        s = Register.from_rocisa(sgpr(8, 4))
        assert v.contains(s) is False
        assert s.contains(v) is False

    def test_vgpr_accvgpr_no_overlap(self):
        v = Register.from_rocisa(vgpr(0, 4))
        a = Register.from_rocisa(accvgpr(0, 4))
        assert v.overlaps(a) is False
        assert a.overlaps(v) is False

    def test_vgpr_mgpr_no_overlap(self):
        v = Register.from_rocisa(vgpr(0, 1))
        m = Register.from_rocisa(mgpr(0))
        assert v.overlaps(m) is False


# =============================================================================
# Symbolic construction
# =============================================================================


class TestSymbolicConstruction:
    def test_from_rocisa_symbolic_vgpr(self):
        r = Register.from_rocisa(vgpr("ValuA", 4))
        assert r.reg_type == "v"
        assert r.name == "ValuA"
        assert r.base == 0  # offset is 0
        assert r.count == 4

    def test_symbolic_form_predicates(self):
        r = Register.from_rocisa(vgpr("ValuA", 4))
        assert r.is_symbolic() is True
        assert r.is_numeric() is False

    def test_symbolic_with_offset(self):
        r = Register.from_rocisa(vgpr("ValuA+2", 4))
        assert r.name == "ValuA"
        assert r.base == 2
        assert r.count == 4
        assert r.lo() == 2
        assert r.hi() == 6


# =============================================================================
# Symbolic overlap / containment
# =============================================================================


class TestSymbolicOverlap:
    def test_same_name_self_overlap(self):
        r = Register.from_rocisa(vgpr("ValuA", 4))
        assert r.overlaps(r) is True

    def test_same_name_partial_overlap(self):
        a = Register.from_rocisa(vgpr("ValuA", 4))     # offset 0..3
        b = Register.from_rocisa(vgpr("ValuA+2", 4))   # offset 2..5
        assert a.overlaps(b) is True
        assert b.overlaps(a) is True

    def test_same_name_disjoint(self):
        a = Register.from_rocisa(vgpr("ValuA", 4))     # offset 0..3
        b = Register.from_rocisa(vgpr("ValuA+8", 4))   # offset 8..11
        assert a.overlaps(b) is False

    def test_different_names_never_overlap(self):
        a = Register.from_rocisa(vgpr("ValuA", 4))
        b = Register.from_rocisa(vgpr("ValuB", 4))
        assert a.overlaps(b) is False
        assert b.overlaps(a) is False

    def test_same_name_containment(self):
        outer = Register.from_rocisa(vgpr("ValuA", 8))      # 0..7
        inner = Register.from_rocisa(vgpr("ValuA+2", 4))    # 2..5
        assert outer.contains(inner) is True
        assert inner.contains(outer) is False

    def test_different_names_no_containment(self):
        a = Register.from_rocisa(vgpr("ValuA", 4))
        b = Register.from_rocisa(vgpr("ValuB", 4))
        assert a.contains(b) is False


# =============================================================================
# Mixed numeric / symbolic — always non-overlapping
# =============================================================================


class TestMixedFormRejection:
    def test_numeric_vs_symbolic_no_overlap(self):
        n = Register.from_rocisa(vgpr(8, 4))
        s = Register.from_rocisa(vgpr("ValuA", 4))
        assert n.overlaps(s) is False
        assert s.overlaps(n) is False

    def test_numeric_vs_symbolic_no_containment(self):
        n = Register.from_rocisa(vgpr(8, 4))
        s = Register.from_rocisa(vgpr("ValuA", 4))
        assert n.contains(s) is False
        assert s.contains(n) is False

    def test_mixed_intersection_is_none(self):
        n = Register.from_rocisa(vgpr(8, 4))
        s = Register.from_rocisa(vgpr("ValuA", 4))
        assert n.intersection(s) is None
        assert s.intersection(n) is None


# =============================================================================
# Intersection
# =============================================================================


class TestIntersection:
    def test_self_intersection_returns_self_range(self):
        r = Register.from_rocisa(vgpr(8, 4))
        x = r.intersection(r)
        assert x is not None
        assert x.reg_type == "v"
        assert x.base == 8
        assert x.count == 4
        assert x.is_numeric()

    def test_partial_numeric_intersection(self):
        a = Register.from_rocisa(vgpr(8, 4))     # v[8:12)
        b = Register.from_rocisa(vgpr(10, 4))    # v[10:14)
        x = a.intersection(b)
        assert x is not None
        assert x.base == 10
        assert x.count == 2

    def test_disjoint_intersection_is_none(self):
        a = Register.from_rocisa(vgpr(8, 4))
        b = Register.from_rocisa(vgpr(20, 4))
        assert a.intersection(b) is None

    def test_cross_type_intersection_is_none(self):
        a = Register.from_rocisa(vgpr(8, 4))
        b = Register.from_rocisa(sgpr(8, 4))
        assert a.intersection(b) is None

    def test_symbolic_partial_intersection(self):
        a = Register.from_rocisa(vgpr("ValuA", 4))    # 0..3
        b = Register.from_rocisa(vgpr("ValuA+2", 4))  # 2..5
        x = a.intersection(b)
        assert x is not None
        assert x.is_symbolic()
        assert x.name == "ValuA"
        assert x.base == 2
        assert x.count == 2  # overlap is 2..3

    def test_symbolic_different_names_intersection_none(self):
        a = Register.from_rocisa(vgpr("ValuA", 4))
        b = Register.from_rocisa(vgpr("ValuB", 4))
        assert a.intersection(b) is None

    def test_intersection_is_symmetric(self):
        a = Register.from_rocisa(vgpr(8, 8))
        b = Register.from_rocisa(vgpr(10, 4))
        x = a.intersection(b)
        y = b.intersection(a)
        assert x == y

    def test_intersection_drops_rocisa_container(self):
        a = Register.from_rocisa(vgpr(8, 4))
        b = Register.from_rocisa(vgpr(10, 4))
        x = a.intersection(b)
        assert x.rocisa_container is None


# =============================================================================
# signature() — hashable, stable, dedup-friendly
# =============================================================================


class TestSignature:
    def test_signature_shape(self):
        r = Register.from_rocisa(vgpr(8, 4))
        assert r.signature() == ("v", None, 8, 4)

    def test_symbolic_signature_shape(self):
        r = Register.from_rocisa(vgpr("ValuA+2", 4))
        assert r.signature() == ("v", "ValuA", 2, 4)

    def test_signature_is_hashable(self):
        r = Register.from_rocisa(vgpr(8, 4))
        # Should not raise
        hash(r.signature())

    def test_signature_stability_across_independent_construction(self):
        r1 = Register.from_rocisa(vgpr(8, 4))
        r2 = Register.from_rocisa(vgpr(8, 4))
        assert r1.signature() == r2.signature()

    def test_signature_distinguishes_reg_type(self):
        v = Register.from_rocisa(vgpr(8, 4))
        s = Register.from_rocisa(sgpr(8, 4))
        assert v.signature() != s.signature()

    def test_signature_distinguishes_symbolic_names(self):
        a = Register.from_rocisa(vgpr("ValuA", 4))
        b = Register.from_rocisa(vgpr("ValuB", 4))
        assert a.signature() != b.signature()


# =============================================================================
# Equality / hashing semantics
# =============================================================================


class TestEqualityAndHashing:
    def test_two_registers_from_equivalent_containers_compare_equal(self):
        r1 = Register.from_rocisa(vgpr(8, 4))
        r2 = Register.from_rocisa(vgpr(8, 4))
        assert r1 == r2

    def test_two_registers_from_equivalent_containers_hash_equal(self):
        r1 = Register.from_rocisa(vgpr(8, 4))
        r2 = Register.from_rocisa(vgpr(8, 4))
        assert hash(r1) == hash(r2)

    def test_set_dedup_works(self):
        r1 = Register.from_rocisa(vgpr(8, 4))
        r2 = Register.from_rocisa(vgpr(8, 4))
        assert {r1, r2} == {r1}

    def test_rocisa_container_excluded_from_equality(self):
        # Two distinct rocisa containers with identical (regType, regIdx,
        # regNum) must yield Registers that compare equal — equality must
        # NOT depend on container identity.
        rc_a = vgpr(8, 4)
        rc_b = vgpr(8, 4)
        assert rc_a is not rc_b  # sanity: independent objects
        r_a = Register.from_rocisa(rc_a)
        r_b = Register.from_rocisa(rc_b)
        assert r_a == r_b
        assert hash(r_a) == hash(r_b)

    def test_distinct_registers_compare_unequal(self):
        a = Register.from_rocisa(vgpr(8, 4))
        b = Register.from_rocisa(vgpr(12, 4))
        assert a != b

    def test_numeric_and_symbolic_compare_unequal(self):
        n = Register.from_rocisa(vgpr(8, 4))
        s = Register.from_rocisa(vgpr("ValuA", 4))
        assert n != s


# =============================================================================
# Malformed input rejection
# =============================================================================


class TestMalformedInputRejection:
    def test_none_container_raises(self):
        with pytest.raises(ValueError):
            Register.from_rocisa(None)

    def test_object_without_regType_raises(self):
        class _NotARegister:
            regIdx = 0
        with pytest.raises(ValueError):
            Register.from_rocisa(_NotARegister())

    def test_object_without_regIdx_raises(self):
        class _NotARegister:
            regType = "v"
        with pytest.raises(ValueError):
            Register.from_rocisa(_NotARegister())

    def test_negative_idx_without_regName_raises(self):
        # Half-symbolic shape: regIdx = -1 but regName is None.
        class _BadHalfSymbolic:
            regType = "v"
            regIdx = -1
            regNum = 4
            regName = None
        with pytest.raises(ValueError):
            Register.from_rocisa(_BadHalfSymbolic())


# =============================================================================
# Register.is_register predicate
# =============================================================================


class TestIsRegisterPredicate:
    def test_real_register_passes(self):
        assert Register.is_register(vgpr(8, 4)) is True

    def test_symbolic_register_passes(self):
        assert Register.is_register(vgpr("ValuA", 4)) is True

    def test_none_rejected(self):
        assert Register.is_register(None) is False

    def test_int_rejected(self):
        assert Register.is_register(42) is False

    def test_string_rejected(self):
        assert Register.is_register("not a register") is False

    def test_arbitrary_object_rejected(self):
        class _Other:
            pass
        assert Register.is_register(_Other()) is False

    def test_duck_type_with_both_fields_accepted(self):
        # Documented surface: anything with regType + regIdx walks like a
        # register. Used by the operand-rule registry to filter getParams().
        class _DuckTypedReg:
            regType = "v"
            regIdx = 7
        assert Register.is_register(_DuckTypedReg()) is True
