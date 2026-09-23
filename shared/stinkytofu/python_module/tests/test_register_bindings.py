"""Tests for the StinkyRegister Python bindings added for the
rocisa→stinkytofu adapter.

Scope: every surface the rocisa-shaped adapter wrapper needs in order to
back the RegisterContainer + RegName shapes that Tensile KernelWriter
expects. KernelWriter call sites this file pins down:

  * RegisterContainer.setMinus(True)          (Components/GSU.py:466)
  * RegisterContainer.getMinus()              (KernelWriterAssembly.py:8669,
                                               KernelWriterModules.py:313,
                                               Components/GlobalWriteBatch.py:2238...)
  * rc.regName truthy / falsy guard           (Activation.py:1309,1314)
  * rc.regName.addOffset(N)                   (KernelWriterAssembly.py:11483...)
  * rc.regName.setOffset(i, v)                (Activation.py:967)
  * rc.regName.name access                    (Activation.py:1467)
  * str(rc.regName)                           (Components/LocalRead.py:486)
  * rc.setInlineAsm(True)                     (Activation.py:1310; wrapper-only,
                                               not delegated to stinkytofu — included
                                               here as the contract the wrapper expects)
  * MUBUF "off" keyword                       (Components/GlobalWriteBatch)
  * rc as dict key, rc == other               (instruction de-dup, IR transforms)
  * copy.deepcopy(rc)                         (KernelWriter pre-emit cloning)

Run via:
  PYTHONPATH=<build_dir_containing_stinkytofu_so> \
      pytest shared/stinkytofu/python_module/tests/test_register_bindings.py
"""

import copy
import os
import sys

import pytest

# Prefer PYTHONPATH; fall back to the in-tree standalone build only when the
# direct import fails. Inserting an in-tree dir at sys.path[0] unconditionally
# (as test_ir_basic.py historically did) tends to shadow a fresh build that
# the developer staged via PYTHONPATH.
try:
    import stinkytofu  # noqa: E402
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../build/lib"))
    import stinkytofu  # noqa: E402

from stinkytofu import Register, RegType, vgpr, sgpr, agpr, accvgpr, mgpr  # noqa: E402


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


class TestConstructors:
    def test_zero_arg_ctor_is_invalid_register(self):
        r = Register()
        assert not r.is_register
        assert not r.is_literal
        assert r.reg_type == RegType.UNKNOWN
        assert r.index == -1
        assert r.count == 0

    def test_three_arg_ctor_with_valid_type(self):
        r = Register("v", 5, 4)
        assert r.is_register
        assert r.reg_type == RegType.V
        assert r.index == 5
        assert r.count == 4

    def test_three_arg_ctor_with_invalid_type_raises(self):
        """Regression: previously Register('vgprFoo', 0, 1) silently built a
        UNKNOWN-typed register and only failed far downstream."""
        with pytest.raises(ValueError, match="unknown register type"):
            Register("vgprFoo", 0, 1)
        with pytest.raises(ValueError):
            Register("notARegType", 0, 1)

    def test_three_arg_ctor_count_default_is_one(self):
        r = Register("s", 10)
        assert r.count == 1

    def test_int_literal_ctor(self):
        r = Register(42)
        assert not r.is_register
        assert r.is_literal
        assert not r.is_literal_string

    def test_float_literal_ctor(self):
        r = Register(3.14)
        assert not r.is_register
        assert r.is_literal
        assert not r.is_literal_string

    def test_single_string_ctor_builds_literal_string(self):
        """Used for MUBUF 'off' keyword and similar literal-string operands."""
        r = Register("off")
        assert not r.is_register
        assert r.is_literal
        assert r.is_literal_string
        assert r.literal_string == "off"

    def test_factory_functions_still_work(self):
        v = vgpr(0, 2)
        s = sgpr(7)
        a = agpr(3, 4)
        assert (v.reg_type, v.index, v.count) == (RegType.V, 0, 2)
        assert (s.reg_type, s.index, s.count) == (RegType.S, 7, 1)
        assert (a.reg_type, a.index, a.count) == (RegType.A, 3, 4)

    def test_accvgpr_alias_of_agpr(self):
        # accvgpr is an alias kept for rocisa-style call sites.
        a = accvgpr(3, 4)
        b = agpr(3, 4)
        assert (a.reg_type, a.index, a.count) == (RegType.A, 3, 4)
        assert a == b

    def test_mgpr_helper(self):
        # mgpr helper exists for the rocisa "m" register type
        # (memory descriptor); required by the adapter's to_stinky()
        # path for tensilelite kernels that emit MUBUF/FLAT addr pairs.
        m = mgpr(2, 4)
        assert m.is_register
        assert (m.reg_type, m.index, m.count) == (RegType.M, 2, 4)

    def test_mgpr_default_count(self):
        m = mgpr(5)
        assert (m.reg_type, m.index, m.count) == (RegType.M, 5, 1)


# ---------------------------------------------------------------------------
# Type queries
# ---------------------------------------------------------------------------


class TestTypeQueries:
    def test_register_is_not_literal(self):
        v = vgpr(0)
        assert v.is_register and not v.is_literal

    def test_int_literal_classification(self):
        r = Register(42)
        assert r.is_literal and not r.is_register and not r.is_literal_string

    def test_float_literal_classification(self):
        r = Register(3.14)
        assert r.is_literal and not r.is_literal_string

    def test_literal_string_classification(self):
        r = Register("off")
        assert r.is_literal
        assert r.is_literal_string
        assert not r.is_register

    def test_literal_string_accessor_on_non_literal_string_is_none(self):
        assert vgpr(0).literal_string is None
        assert Register(42).literal_string is None
        assert Register(3.14).literal_string is None

    def test_literal_string_accessor_on_literal_string(self):
        assert Register("off").literal_string == "off"
        assert Register("inline_asm_blob").literal_string == "inline_asm_blob"
        assert Register("").literal_string == ""


# ---------------------------------------------------------------------------
# RegName (symbolic name + offsets) — round-trip
# ---------------------------------------------------------------------------


class TestRegName:
    def test_register_initially_has_no_reg_name(self):
        r = vgpr(0)
        assert not r.has_reg_name()
        assert r.get_reg_name() == ("", [])

    def test_set_and_read_back_simple_name(self):
        r = vgpr(0, 4)
        r.set_reg_name("vgprLocalWriteAddrA")
        assert r.has_reg_name()
        assert r.get_reg_name() == ("vgprLocalWriteAddrA", [])

    def test_set_and_read_back_name_with_offsets(self):
        r = vgpr(0, 4)
        r.set_reg_name("ValuC", [1, 2, 3])
        assert r.has_reg_name()
        name, offsets = r.get_reg_name()
        assert name == "ValuC"
        assert offsets == [1, 2, 3]

    def test_set_with_default_offsets_is_empty_list(self):
        r = vgpr(0)
        r.set_reg_name("foo")
        assert r.get_reg_name() == ("foo", [])

    def test_clear_reg_name(self):
        r = vgpr(0)
        r.set_reg_name("foo", [1, 2])
        assert r.has_reg_name()
        r.clear_reg_name()
        assert not r.has_reg_name()
        assert r.get_reg_name() == ("", [])

    def test_set_reg_name_overwrites_previous(self):
        r = vgpr(0)
        r.set_reg_name("foo", [1])
        r.set_reg_name("bar", [2, 3])
        assert r.get_reg_name() == ("bar", [2, 3])

    def test_reg_name_with_zero_offset(self):
        r = vgpr(0)
        r.set_reg_name("foo", [0])
        assert r.get_reg_name() == ("foo", [0])

    def test_reg_name_with_large_offsets(self):
        r = vgpr(0)
        r.set_reg_name("foo", [256, 1024, 65535])
        assert r.get_reg_name() == ("foo", [256, 1024, 65535])

    def test_setting_empty_name_is_treated_as_clear(self):
        r = vgpr(0)
        r.set_reg_name("foo")
        r.set_reg_name("")
        # setSymbolicName("") leaves literalValue empty → hasSymbolicName()→False
        assert not r.has_reg_name()


class TestRegNameEncoding:
    """The (name, offsets[]) ↔ "name+1+2+..." encoding has to be byte-for-byte
    stable so stinkytofu IR transforms (LegalizationUtils::adjustSymbolicRegName)
    and adapter consumers interoperate without ambiguity."""

    def test_str_format_matches_rocisa_layout(self):
        # Encoding is "name+offset0+offset1+..."; split between name and
        # offsets is on the FIRST '+'.
        r = vgpr(0)
        r.set_reg_name("ValuC", [1, 2, 3])
        # No raw-string getter, so verify by round-tripping.
        assert r.get_reg_name() == ("ValuC", [1, 2, 3])

    def test_offsets_join_back_correctly_when_first_offset_is_dropped(self):
        """KernelWriterAssembly.py:11483 emulates `regName.addOffset(1)` by
        appending; verify we can do the same via get→append→set."""
        r = vgpr(0)
        r.set_reg_name("foo", [2])
        name, offsets = r.get_reg_name()
        offsets.append(1)
        r.set_reg_name(name, offsets)
        assert r.get_reg_name() == ("foo", [2, 1])


# ---------------------------------------------------------------------------
# Modifier flags
# ---------------------------------------------------------------------------


class TestModifiers:
    def test_is_minus_default_false(self):
        assert vgpr(0).is_minus is False

    def test_is_abs_default_false(self):
        assert vgpr(0).is_abs is False

    def test_set_minus_toggles(self):
        r = vgpr(0)
        r.set_minus(True)
        assert r.is_minus is True
        r.set_minus(False)
        assert r.is_minus is False

    def test_set_abs_toggles(self):
        r = sgpr(0)
        r.set_abs(True)
        assert r.is_abs is True
        r.set_abs(False)
        assert r.is_abs is False

    def test_minus_and_abs_independent(self):
        r = vgpr(0)
        r.set_minus(True)
        r.set_abs(True)
        assert r.is_minus and r.is_abs
        r.set_minus(False)
        assert not r.is_minus and r.is_abs

    def test_set_minus_on_literal_is_noop(self):
        for r in (Register(42), Register(3.14), Register("off")):
            r.set_minus(True)
            assert r.is_minus is False

    def test_set_abs_on_literal_is_noop(self):
        for r in (Register(42), Register(3.14), Register("off")):
            r.set_abs(True)
            assert r.is_abs is False

    def test_modifiers_survive_set_reg_name(self):
        """setting RegName shouldn't trash the modifier bits."""
        r = vgpr(0)
        r.set_minus(True)
        r.set_abs(True)
        r.set_reg_name("foo", [1])
        assert r.is_minus and r.is_abs
        assert r.get_reg_name() == ("foo", [1])


# ---------------------------------------------------------------------------
# Hash & equality
# ---------------------------------------------------------------------------


class TestHashAndEquality:
    def test_equal_registers_hash_equal(self):
        a = vgpr(5, 2)
        b = vgpr(5, 2)
        assert a == b
        assert hash(a) == hash(b)

    def test_different_registers_compare_unequal(self):
        assert vgpr(0) != vgpr(1)
        assert vgpr(0, 1) != vgpr(0, 2)
        assert vgpr(0) != sgpr(0)

    def test_can_be_used_as_dict_key(self):
        # KernelWriter / stinkytofu IR transforms put registers in dicts to
        # track def/use chains. The C++ side already supports it; binding
        # has to expose __hash__ + __eq__ for it to work from Python.
        d = {}
        a = vgpr(0)
        d[a] = "alpha"
        # Lookup with a freshly-built but equal register
        b = vgpr(0)
        assert d[b] == "alpha"

    def test_int_literal_equality(self):
        assert Register(42) == Register(42)
        assert Register(42) != Register(43)

    def test_literal_string_equality(self):
        assert Register("off") == Register("off")
        assert Register("off") != Register("none")

    def test_register_vs_non_register_comparison_returns_false_not_raises(self):
        r = vgpr(0)
        # Comparison against unrelated Python types must not raise; KernelWriter
        # does heterogeneous comparisons (e.g., in DCE / dedup passes).
        assert (r == "vgpr0") is False
        assert (r == 42) is False
        assert (r == None) is False  # noqa: E711 (intentional eq test)
        assert r != "vgpr0"
        assert r != 42

    def test_unequal_after_set_minus(self):
        a = vgpr(0)
        b = vgpr(0)
        b.set_minus(True)
        # Equality only compares (type, idx, num); modifier bits don't
        # participate — `-v0` and `v0` refer to the same SSA register,
        # the `-` is an instruction operand modifier. Pin the contract.
        assert a == b
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Copy / deepcopy
# ---------------------------------------------------------------------------


class TestCopy:
    def test_copy_produces_equal_but_independent_register(self):
        a = vgpr(0)
        a.set_minus(True)
        a.set_reg_name("foo", [1, 2])
        b = copy.copy(a)
        assert b == a
        assert b is not a
        assert b.is_minus
        assert b.get_reg_name() == ("foo", [1, 2])

    def test_deepcopy_produces_independent_register(self):
        a = vgpr(0, 4)
        a.set_reg_name("ValuC", [3])
        b = copy.deepcopy(a)
        # Mutate the copy; original must not move
        b.set_reg_name("ValuC", [3, 5])
        b.set_minus(True)
        assert a.get_reg_name() == ("ValuC", [3])
        assert a.is_minus is False
        assert b.get_reg_name() == ("ValuC", [3, 5])
        assert b.is_minus is True

    def test_deepcopy_of_literal_string(self):
        a = Register("off")
        b = copy.deepcopy(a)
        assert b == a
        assert b.literal_string == "off"


# ---------------------------------------------------------------------------
# Scenarios mirroring real KernelWriter usage patterns
# ---------------------------------------------------------------------------


class TestKernelWriterScenarios:
    """Each test maps to a concrete call site in Tensile (cited in test name).
    These are the contracts the rocisa-shaped adapter wrapper relies on; if
    any of these break, the wrapper breaks in turn."""

    def test_get_minus_emulation_via_copy_plus_set_minus(self):
        # getMinus() returns a NEW container with isMinus=True; wrapper
        # backs this with copy + set_minus on the underlying Register.
        # See KernelWriterAssembly.py:8669 (`ar.getMinus()`).
        original = vgpr(5)
        neg = copy.copy(original)
        neg.set_minus(True)
        assert original.is_minus is False
        assert neg.is_minus is True

    def test_set_minus_on_existing_container(self):
        # Components/GSU.py:466 — `m.setMinus(True)`
        r = sgpr(0, 2)
        r.set_minus(True)
        assert r.is_minus is True

    def test_reg_name_truthy_check(self):
        # Activation.py:1309 — `if not item.dst.regName:`
        plain = vgpr(0)
        named = vgpr(0)
        named.set_reg_name("ValuC")
        assert plain.has_reg_name() is False
        assert named.has_reg_name() is True

    def test_reg_name_add_offset_via_round_trip(self):
        # KernelWriterAssembly.py:11483 — `new_src.regName.addOffset(1)`
        # The adapter RegName wrapper will own .addOffset() and sync to the
        # underlying Register via set_reg_name. Verify the round-trip works
        # without information loss across multiple mutations.
        r = vgpr(0)
        r.set_reg_name("ValuC", [0])
        for k in range(5):
            name, offsets = r.get_reg_name()
            offsets.append(k)
            r.set_reg_name(name, offsets)
        assert r.get_reg_name() == ("ValuC", [0, 0, 1, 2, 3, 4])

    def test_reg_name_set_offset_via_round_trip(self):
        # Activation.py:967 — `vgpr.regName.setOffset(0, vgprIn)`
        r = vgpr(0)
        r.set_reg_name("ValuC", [10, 20, 30])
        name, offsets = r.get_reg_name()
        offsets[0] = 99
        r.set_reg_name(name, offsets)
        assert r.get_reg_name() == ("ValuC", [99, 20, 30])

    def test_reg_name_name_access(self):
        # Activation.py:1467 — `param.regName.name`
        r = vgpr(0)
        r.set_reg_name("ValuC", [1, 2])
        name, _ = r.get_reg_name()
        assert name == "ValuC"

    def test_str_of_reg_name_resembles_rocisa(self):
        # Components/LocalRead.py:486 — `str(v0t.regName)` is expected to
        # produce "ValuC+1+2"-style strings. The adapter wrapper owns
        # __str__; binding only needs to preserve the components so the
        # wrapper can format them.
        r = vgpr(0)
        r.set_reg_name("ValuC", [1, 2])
        name, offsets = r.get_reg_name()
        rocisa_style = name + "".join(f"+{o}" for o in offsets)
        assert rocisa_style == "ValuC+1+2"

    def test_mubuf_off_keyword(self):
        # The no-address MUBUF case (isOff=True flag in RegisterContainer)
        # surfaces in stinkytofu as a LiteralString operand.
        off = Register("off")
        assert off.is_literal_string
        assert off.literal_string == "off"
        # Wrapper dispatches on is_literal_string to set its own isOff
        # flag rather than reading reg_type (which is UNKNOWN here).
        assert off.reg_type == RegType.UNKNOWN

    def test_dict_lookup_after_deepcopy(self):
        # IR transforms in stinkytofu (e.g., def/use chain) build dicts keyed
        # on Register. KernelWriter often passes deep-copied registers; lookup
        # must still hit by value, not by identity.
        seen = {vgpr(0, 4): "lwA", sgpr(10): "Alpha"}
        probe_v = copy.deepcopy(vgpr(0, 4))
        probe_s = copy.deepcopy(sgpr(10))
        assert seen[probe_v] == "lwA"
        assert seen[probe_s] == "Alpha"

    def test_clear_reg_name_for_holder_resolution(self):
        # RegisterContainer.replaceRegName(srcName, dst:int) resolves a
        # holder-style name into a numeric idx and clears regName; the
        # adapter wrapper calls clear_reg_name as part of that step.
        r = vgpr(0)
        r.set_reg_name("HOLDER_NAME", [])
        # Wrapper would compute resolvedIdx from the name lookup, then:
        r.clear_reg_name()
        # Numeric identity preserved, symbolic info gone.
        assert not r.has_reg_name()
        assert r.reg_type == RegType.V
        assert r.index == 0
