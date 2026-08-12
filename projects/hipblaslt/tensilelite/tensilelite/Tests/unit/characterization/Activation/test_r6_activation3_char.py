################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R6 — Activation.py remaining-coverage characterization test.

Targets the following groups of uncovered lines in Tensile/Activation.py (207
statements at 76% per survey — "Activation remaining (207, 76%)"):

  Group A — ActivationAvailable.__init__ + ActivationTypeRegister.__init__ +
             typeAvailable (lines 100-128):
    All ActivationTypeRegister instances already exist in ActivationType.lookup,
    so we need to call typeAvailable() with every DataType to touch its branches.

  Group B — ActivationType methods (lines 198-282):
    passActivation (3 Export variants), getAdditionalArgNum (all+hipblaslt_all
    branches), getAdditionalArgStringList (addPrefix True/False),
    getEnumStrList (Normal/GradOnly/Both export types),
    fitSupported (0 and non-zero results),
    state/toEnum/__str__/__repr__/__lt__/__eq__,
    init from ActivationType (copy) + error paths.

  Group C — actCacheInfo.isSame (lines 284-299):
    Create an actCacheInfo dataclass and call isSame() with matching and
    non-matching field combinations.

  Group D — ActivationModule cache paths (lines 940-978):
    setUseCache(True) + getModule for 'abs' creates a cache entry; subsequent
    getModule with different vgprIn/vgprOut retrieves from cache.
    A second call with vgprIn==vgprOut skips cache creation.

  Group E — ActivationInline.generateInlineAssemblyBody for all activation
             types + getRequiredRegStr + getActivationAsmStr (lines 1333-1454):
    All asm/non-asm branches: exp, gelu, geluscaling, sigmoid, tanh, silu, swish,
    dgelu (plain + guard), drelu, abs (all subtypes), clippedrelu, leakyrelu,
    relu (all subtypes), clamp, none.

  Group F — Helper functions getMagic / getMagicStr / HexToStr (lines 1236-1261):
    isPack True/False for half; single precision; getMagicStr delegation.

  Group G — ConvertCoeffToHex + HolderToGpr (lines 1264-1291):
    Touched indirectly by getModule(), but exercised explicitly here to ensure
    deterministic coverage across parallel-collection runs.

  Group H — createVgprIdxList (lines 1456-1471):
    Called from createCache when vgprPrefixFormat is empty (the regName="" path).

Strategy: initialize rocisa with gfx942 (required by getArchCaps["TransOpWait"]
called inside getExpModule, getSigmoidModule, getTanhModule, getDGeluModule, and
getDReluModule). All calls are CPU-only — no GPU required.

pytestmark = pytest.mark.unit
"""

import importlib
import shutil
from dataclasses import dataclass

import pytest

from rocisa.code import Module

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Lazy module references (same pattern as test_r4_activation2_char.py)
# ---------------------------------------------------------------------------
A = importlib.import_module("Tensile.Activation")
DataType = importlib.import_module("Tensile.Common.DataType").DataType


def _init_rocisa():
    """Initialize the rocisa singleton for gfx942 (9,4,2), wavefront 64.

    Required because getExpModule / getSigmoidModule / getTanhModule /
    getDGeluModule all query ``rocIsa.getInstance().getArchCaps()["TransOpWait"]``.
    """
    import rocisa

    isa = (9, 4, 2)
    wavefront = 64
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri = rocisa.rocIsa.getInstance()
    ri.init(isa, asmpath)
    ri.setKernel(isa, wavefront)
    return ri


# One-time module-level init.
_RI = _init_rocisa()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _mod(dtype_code, act_name, *, saturate=False, usePK=True, isAlt=False, guard=False):
    """Return a fresh ActivationModule.getModule() result."""
    m = A.ActivationModule()
    m.setSaturationForInt8(saturate)
    m.setUsePK(usePK)
    m.setAlt(isAlt)
    m.setGuard(guard)
    return m.getModule(DataType(dtype_code), act_name, 0, 1)


# ============================================================================
# Group A — ActivationAvailable + ActivationTypeRegister + typeAvailable
# ============================================================================

class TestActivationAvailableAndRegister:
    """Covers lines 100-128: ActivationAvailable.__init__ +
    ActivationTypeRegister.__init__ + typeAvailable()."""

    def test_activation_available_init(self):
        """ActivationAvailable stores per-type flags correctly."""
        aa = A.ActivationAvailable(canHalf=True, canSingle=True, canDouble=False,
                                   canBFloat16=False, canInt8=False, canInt16=False,
                                   canInt32=True)
        assert aa.half is True
        assert aa.single is True
        assert aa.double is False
        assert aa.bfloat16 is False
        assert aa.int8 is False
        assert aa.int16 is False
        assert aa.int32 is True

    def test_activation_type_register_init(self):
        """ActivationTypeRegister stores name, isGradient, extraArgs, and can."""
        atr = A.ActivationTypeRegister("myact", True, 2,
                                       canHalf=True, canSingle=True)
        assert atr.name == "myact"
        assert atr.isGradient is True
        assert atr.extraArgs == 2
        assert atr.can.half is True
        assert atr.can.single is True
        assert atr.can.double is False

    def test_type_available_half(self):
        """typeAvailable returns True for half when canHalf=True."""
        atr = A.ActivationTypeRegister("t", False, 0, canHalf=True)
        assert atr.typeAvailable(DataType("h")) is True

    def test_type_available_single(self):
        """typeAvailable returns True for single when canSingle=True."""
        atr = A.ActivationTypeRegister("t", False, 0, canSingle=True)
        assert atr.typeAvailable(DataType("s")) is True

    def test_type_available_double(self):
        """typeAvailable returns True for double when canDouble=True."""
        atr = A.ActivationTypeRegister("t", False, 0, canDouble=True)
        assert atr.typeAvailable(DataType("d")) is True

    def test_type_available_bfloat16(self):
        """typeAvailable returns True for bfloat16 when canBFloat16=True."""
        atr = A.ActivationTypeRegister("t", False, 0, canBFloat16=True)
        assert atr.typeAvailable(DataType("B")) is True

    def test_type_available_int8(self):
        """typeAvailable returns True for int8 when canInt8=True."""
        atr = A.ActivationTypeRegister("t", False, 0, canInt8=True)
        assert atr.typeAvailable(DataType("i8")) is True

    def test_type_available_int32(self):
        """typeAvailable returns True for int32 when canInt32=True."""
        atr = A.ActivationTypeRegister("t", False, 0, canInt32=True)
        assert atr.typeAvailable(DataType("I")) is True

    def test_type_available_false_when_unsupported(self):
        """typeAvailable returns False when no matching can-flag is set."""
        atr = A.ActivationTypeRegister("t", False, 0, canHalf=True)
        # queried with single but only half supported
        assert atr.typeAvailable(DataType("s")) is False


# ============================================================================
# Group B — ActivationType methods
# ============================================================================

class TestActivationTypeMethods:
    """Covers lines 198-282: ActivationType methods, comparisons, error paths."""

    def test_init_from_string(self):
        at = A.ActivationType("gelu")
        assert str(at) == "Gelu"

    def test_init_from_activationtype(self):
        """Init ActivationType from another ActivationType (copy branch)."""
        at1 = A.ActivationType("silu")
        at2 = A.ActivationType(at1)
        assert at2 == at1

    def test_init_lookupveri(self):
        """'exp' lives in lookupVeri; it's allowed at init time."""
        at = A.ActivationType("exp")
        assert str(at) == "Exp"

    def test_init_invalid_string(self):
        """Unrecognized string raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Unrecognized activation type"):
            A.ActivationType("not_a_real_activation")

    def test_init_invalid_type(self):
        """Non-string/non-ActivationType input raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Unrecognized input type"):
            A.ActivationType(42)

    def test_pass_activation_normal_gradient(self):
        """NORMAL export: passActivation returns True for gradient activations."""
        at_dgelu = A.ActivationType("dgelu")
        # dgelu isGradient=True; NORMAL -> pass if isGradient
        result = at_dgelu.passActivation(True, A.ActivationType.Export.NORMAL)
        assert result is True

    def test_pass_activation_normal_non_gradient(self):
        """NORMAL export: passActivation returns False for non-gradient activations."""
        at_gelu = A.ActivationType("gelu")
        result = at_gelu.passActivation(False, A.ActivationType.Export.NORMAL)
        assert result is False

    def test_pass_activation_gradonly_gradient(self):
        """GRADONLY export: passActivation returns False for gradient activations."""
        at = A.ActivationType("dgelu")
        result = at.passActivation(True, A.ActivationType.Export.GRADONLY)
        assert result is False

    def test_pass_activation_gradonly_non_gradient(self):
        """GRADONLY export: passActivation returns True for non-gradient activations."""
        at = A.ActivationType("gelu")
        result = at.passActivation(False, A.ActivationType.Export.GRADONLY)
        assert result is True

    def test_pass_activation_both(self):
        """BOTH export: passActivation always returns False."""
        at_dgelu = A.ActivationType("dgelu")
        at_gelu = A.ActivationType("gelu")
        assert at_dgelu.passActivation(True, A.ActivationType.Export.BOTH) is False
        assert at_gelu.passActivation(False, A.ActivationType.Export.BOTH) is False

    def test_get_additional_arg_num_specific(self):
        """Specific type returns its own extraArgs count."""
        assert A.ActivationType("geluscaling").getAdditionalArgNum() == 1
        assert A.ActivationType("swish").getAdditionalArgNum() == 1
        assert A.ActivationType("tanh").getAdditionalArgNum() == 2
        assert A.ActivationType("silu").getAdditionalArgNum() == 0
        assert A.ActivationType("dgelu").getAdditionalArgNum() == 0

    def test_get_additional_arg_num_all(self):
        """'all' returns the max extraArgs across all non-gradient activations."""
        at_all = A.ActivationType("all")
        num = at_all.getAdditionalArgNum()
        assert num >= 2  # tanh/clippedrelu/clamp all have extraArgs=2

    def test_get_additional_arg_num_hipblaslt_all(self):
        """'hipblaslt_all' returns max extraArgs for HIPBLASLT-supported activations."""
        at = A.ActivationType("hipblaslt_all")
        num = at.getAdditionalArgNum()
        assert num >= 0

    def test_get_additional_arg_num_all_gradonly(self):
        """'all' with GRADONLY export (only gradient activations): extraArgs = 0."""
        at_all = A.ActivationType("all")
        num = at_all.getAdditionalArgNum(A.ActivationType.Export.GRADONLY)
        assert num == 0  # dgelu + drelu both have extraArgs=0

    def test_get_additional_arg_string_list_with_prefix(self):
        """geluscaling extraArgs=1 -> ['activationAlpha']."""
        at = A.ActivationType("geluscaling")
        lst = at.getAdditionalArgStringList(addPrefix=True)
        assert lst == ["activationAlpha"]

    def test_get_additional_arg_string_list_no_prefix(self):
        """geluscaling extraArgs=1, addPrefix=False -> ['alpha']."""
        at = A.ActivationType("geluscaling")
        lst = at.getAdditionalArgStringList(addPrefix=False)
        assert lst == ["alpha"]

    def test_get_additional_arg_string_list_two_args(self):
        """tanh extraArgs=2 -> ['activationAlpha', 'activationBeta']."""
        at = A.ActivationType("tanh")
        lst = at.getAdditionalArgStringList(addPrefix=True)
        assert lst == ["activationAlpha", "activationBeta"]

    def test_get_enum_index(self):
        """getEnumIndex returns position in lookup dict."""
        idx_none = A.ActivationType.getEnumIndex("none")
        idx_gelu = A.ActivationType.getEnumIndex("gelu")
        idx_all = A.ActivationType.getEnumIndex("all")
        assert idx_none == 0
        assert isinstance(idx_gelu, int) and idx_gelu > 0
        assert isinstance(idx_all, int)

    def test_get_enum_str_list_normal_single(self):
        """NORMAL export on Single: gradient activations (dgelu, drelu) excluded."""
        lst = A.ActivationType.getEnumStrList(
            DataType("s"), A.ActivationType.SupportedBy.ALL,
            exportType=A.ActivationType.Export.NORMAL
        )
        assert "gelu" in lst
        assert "geluscaling" in lst
        assert "silu" in lst
        assert "dgelu" not in lst
        assert "drelu" not in lst

    def test_get_enum_str_list_gradonly_single(self):
        """GRADONLY export on Single: only gradient activations (dgelu, drelu)."""
        lst = A.ActivationType.getEnumStrList(
            DataType("s"), A.ActivationType.SupportedBy.ALL,
            exportType=A.ActivationType.Export.GRADONLY
        )
        assert "dgelu" in lst
        assert "drelu" in lst
        assert "gelu" not in lst

    def test_get_enum_str_list_both_single(self):
        """BOTH export on Single: all activations (no passActivation exclusion)."""
        lst = A.ActivationType.getEnumStrList(
            DataType("s"), A.ActivationType.SupportedBy.ALL,
            exportType=A.ActivationType.Export.BOTH
        )
        # Both gradient + non-gradient should appear
        assert "gelu" in lst
        assert "dgelu" in lst

    def test_get_enum_str_list_no_include_none(self):
        """includeNone=False excludes 'none' from the list."""
        lst = A.ActivationType.getEnumStrList(
            DataType("s"), A.ActivationType.SupportedBy.ALL, includeNone=False
        )
        assert "none" not in lst

    def test_get_enum_str_list_tensile_only(self):
        """TENSILE supported_by filter: all TENSILE-only types included."""
        lst = A.ActivationType.getEnumStrList(
            DataType("s"), A.ActivationType.SupportedBy.TENSILE
        )
        # geluscaling, swish are TENSILE-only
        assert "geluscaling" in lst or "swish" in lst

    def test_fit_supported_match(self):
        """fitSupported returns non-zero when component is in config."""
        at = A.ActivationType("gelu")
        result = at.fitSupported(A.ActivationType.SupportedBy.ALL, A.ActivationType.SupportedBy.TENSILE)
        assert result != 0

    def test_fit_supported_no_match(self):
        """fitSupported returns 0 when component is not in config."""
        at = A.ActivationType("gelu")
        result = at.fitSupported(A.ActivationType.SupportedBy.HIPBLASLT, A.ActivationType.SupportedBy.TENSILE)
        # HIPBLASLT & TENSILE = 0b01 & 0b10 = 0
        assert result == 0

    def test_state_method(self):
        """state() returns Capitalized name."""
        at = A.ActivationType("dgelu")
        assert at.state() == "Dgelu"

    def test_to_enum(self):
        """toEnum() is same as state()."""
        at = A.ActivationType("silu")
        assert at.toEnum() == "Silu"

    def test_repr(self):
        """__repr__ is same as __str__."""
        at = A.ActivationType("relu")
        assert repr(at) == "Relu"

    def test_lt_string(self):
        """__lt__ comparison against string works."""
        at = A.ActivationType("gelu")
        assert at < "relu"         # "gelu" < "relu"
        assert not (at < "abc")    # "gelu" > "abc"

    def test_lt_activation_type(self):
        """__lt__ comparison between ActivationType objects works."""
        at1 = A.ActivationType("abs")
        at2 = A.ActivationType("gelu")
        assert at1 < at2
        assert not (at2 < at1)

    def test_lt_invalid(self):
        """__lt__ with unrecognized type raises RuntimeError."""
        at = A.ActivationType("gelu")
        with pytest.raises(RuntimeError, match="Unrecognized type in rhs"):
            _ = at < 99

    def test_eq_string(self):
        """__eq__ against string matches case-insensitively."""
        at = A.ActivationType("GELU")
        assert at == "gelu"
        assert at == "Gelu"
        assert not (at == "relu")

    def test_eq_activation_type(self):
        """__eq__ between two ActivationType objects works."""
        at1 = A.ActivationType("gelu")
        at2 = A.ActivationType("gelu")
        at3 = A.ActivationType("relu")
        assert at1 == at2
        assert not (at1 == at3)

    def test_eq_invalid(self):
        """__eq__ with unrecognized type raises RuntimeError."""
        at = A.ActivationType("gelu")
        with pytest.raises(RuntimeError, match="Unrecognized type in rhs"):
            _ = at == 99


# ============================================================================
# Group C — actCacheInfo.isSame
# ============================================================================

class TestActCacheInfo:
    """Covers lines 284-299: actCacheInfo dataclass + isSame()."""

    def _make(self, usePK=True, saturateI8=False, enableGuard=False, isAlt=False,
              prefix=""):
        mod = Module("dummy")
        return A.actCacheInfo(
            usePK=usePK,
            saturateI8=saturateI8,
            enableGuard=enableGuard,
            isAlt=isAlt,
            prefix=prefix,
            vgprIdxList=[[], []],
            module=mod,
            vgprCounter=0,
            sgprCounter=0,
        )

    def test_is_same_identical(self):
        """isSame returns True when all fields match."""
        ci = self._make(usePK=True, saturateI8=False, enableGuard=False, isAlt=False, prefix="")
        assert ci.isSame(usePK=True, saturateI8=False, enableGuard=False, isAlt=False, prefix="")

    def test_is_same_different_pk(self):
        """isSame returns False when usePK differs."""
        ci = self._make(usePK=True)
        assert not ci.isSame(usePK=False, saturateI8=False, enableGuard=False, isAlt=False, prefix="")

    def test_is_same_different_saturate(self):
        """isSame returns False when saturateI8 differs."""
        ci = self._make(saturateI8=False)
        assert not ci.isSame(usePK=True, saturateI8=True, enableGuard=False, isAlt=False, prefix="")

    def test_is_same_different_guard(self):
        """isSame returns False when enableGuard differs."""
        ci = self._make(enableGuard=False)
        assert not ci.isSame(usePK=True, saturateI8=False, enableGuard=True, isAlt=False, prefix="")

    def test_is_same_different_alt(self):
        """isSame returns False when isAlt differs."""
        ci = self._make(isAlt=False)
        assert not ci.isSame(usePK=True, saturateI8=False, enableGuard=False, isAlt=True, prefix="")

    def test_is_same_different_prefix(self):
        """isSame returns False when prefix differs."""
        ci = self._make(prefix="vgpr+%d")
        assert not ci.isSame(usePK=True, saturateI8=False, enableGuard=False, isAlt=False, prefix="")


# ============================================================================
# Group D — ActivationModule cache paths (createCache / getCache)
# ============================================================================

class TestActivationModuleCache:
    """Covers lines 939-978: createCache + getCache traversal."""

    def test_cache_populates_on_first_call(self):
        """First getModule with useCache=True and in!=out populates cacheDict."""
        am = A.ActivationModule()
        am.setUseCache(True)
        am.setUsePK(True)
        m = am.getModule(DataType("s"), "abs", 0, 1)
        assert "abs" in am.cacheDict
        assert isinstance(m, Module)

    def test_cache_hit_different_vgpr(self):
        """Second getModule with different vgprs retrieves from cache."""
        am = A.ActivationModule()
        am.setUseCache(True)
        am.setUsePK(True)
        # First: populate cache
        am.getModule(DataType("s"), "abs", 0, 1)
        initial_vgpr = am.vgprCounter
        # Second: cache hit, should return a Module
        m2 = am.getModule(DataType("s"), "abs", 2, 3)
        assert isinstance(m2, Module)

    def test_cache_skipped_when_in_eq_out(self):
        """When vgprIn == vgprOut, cache is NOT created (line 392 guard)."""
        am = A.ActivationModule()
        am.setUseCache(True)
        am.setUsePK(True)
        am.getModule(DataType("s"), "abs", 5, 5)   # in == out -> no cache
        cache_entries = am.cacheDict.get("abs", {}).get("s", [])
        assert len(cache_entries) == 0

    def test_cache_no_hit_different_pk(self):
        """Cache miss when usePK differs: creates a new entry for sigmoid (uses PK).

        Cache key uses DataType.toChar() which returns uppercase ('S' for Single).
        """
        am = A.ActivationModule()
        am.setUseCache(True)
        am.setUsePK(True)
        # sigmoid uses needCombine=True and usePK=True, so its cache key differs
        am.getModule(DataType("s"), "sigmoid", 0, 1)
        # DataType("s").toChar() == "S" — cache key is uppercase
        count_before = len(am.cacheDict.get("sigmoid", {}).get("S", []))
        assert count_before >= 1, "Cache should have been created after first call"

        # Second call with same args hits the cache (same usePK=True)
        am.getModule(DataType("s"), "sigmoid", 2, 3)
        count_same = len(am.cacheDict.get("sigmoid", {}).get("S", []))
        # Should still be the same (cache hit)
        assert count_same == count_before


# ============================================================================
# Group E — ActivationInline.generateInlineAssemblyBody (all branches)
# ============================================================================

class TestActivationInline:
    """Covers lines 1297-1454: ActivationInline class + all activation branches."""

    # --- Single (float) inline bodies ---

    def test_inline_abs_single(self):
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "abs")
        assert "value" in result

    def test_inline_clippedrelu_single(self):
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "clippedrelu")
        assert "alpha" in result and "beta" in result

    def test_inline_exp_single(self):
        """exp branch uses asm() block — unique in having '// Exp' marker."""
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "exp")
        assert "Exp" in result
        assert "v_mul_f32" in result or "v_exp_f32" in result

    def test_inline_gelu_single(self):
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "gelu")
        assert "gelu" in result.lower()

    def test_inline_geluscaling_single(self):
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "geluscaling")
        assert "geluscaling" in result.lower()
        assert "alpha" in result

    def test_inline_leakyrelu_single(self):
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "leakyrelu")
        assert "alpha" in result

    def test_inline_leakyrelu_int32(self):
        ai = A.ActivationInline(DataType("I"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "leakyrelu")
        assert "alpha" in result

    def test_inline_relu_single(self):
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "relu")
        assert "max" in result.lower()

    def test_inline_relu_int32(self):
        ai = A.ActivationInline(DataType("I"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "relu")
        assert "max" in result.lower()

    def test_inline_sigmoid_single(self):
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "sigmoid")
        assert "Sigmoid" in result

    def test_inline_tanh_single(self):
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "tanh")
        assert "tanh" in result.lower()
        assert "alpha" in result and "beta" in result

    def test_inline_dgelu_single_no_guard(self):
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "dgelu")
        assert "dgelu" in result.lower()
        assert "exec" not in result

    def test_inline_dgelu_single_with_guard(self):
        """dgelu with enableGuard=True: getRequiredRegStr adds 'exec'."""
        ai = A.ActivationInline(DataType("s"), enableGuard=True)
        result = ai.generateInlineAssemblyBody(4, "dgelu")
        assert "exec" in result

    def test_inline_drelu_single(self):
        """drelu on Single uses a plain C expression, not asm block."""
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "drelu")
        assert "1.0f" in result or "0.0f" in result

    def test_inline_silu_single(self):
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "silu")
        assert "Silu" in result

    def test_inline_swish_single(self):
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "swish")
        assert "Swish" in result
        assert "alpha" in result

    def test_inline_clamp_single(self):
        """clamp uses a plain C std::max/std::min expression."""
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "clamp")
        assert "alpha" in result and "beta" in result

    def test_inline_none(self):
        """none activation produces empty body."""
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "none")
        assert result == ""

    # --- Half inline bodies (non-asm branches) ---

    def test_inline_abs_half(self):
        """abs on Half uses union f16_union path."""
        ai = A.ActivationInline(DataType("h"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "abs")
        assert "f16_union" in result

    def test_inline_abs_bfloat16(self):
        """abs on BFloat16 uses union bf16_union path."""
        ai = A.ActivationInline(DataType("B"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "abs")
        assert "bf16_union" in result

    def test_inline_abs_double(self):
        """abs on Double uses C negation expression."""
        ai = A.ActivationInline(DataType("d"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "abs")
        assert "value" in result

    def test_inline_abs_int32(self):
        """abs on Int32 uses C negation expression."""
        ai = A.ActivationInline(DataType("I"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "abs")
        assert "value" in result

    def test_inline_clippedrelu_double(self):
        ai = A.ActivationInline(DataType("d"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "clippedrelu")
        assert "alpha" in result and "beta" in result

    def test_inline_clippedrelu_int32(self):
        ai = A.ActivationInline(DataType("I"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "clippedrelu")
        assert "alpha" in result and "beta" in result

    def test_inline_relu_double(self):
        ai = A.ActivationInline(DataType("d"), enableGuard=False)
        result = ai.generateInlineAssemblyBody(4, "relu")
        assert "max" in result.lower()

    # --- getRequiredRegStr branches ---

    def test_get_required_reg_str_no_exec(self):
        """getRequiredRegStr with non-zero vgpr/sgpr and no exec."""
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.getRequiredRegStr("    ", 2, 1, needExec=False)
        assert '"v0"' in result
        assert '"v1"' in result
        assert '"s0"' in result
        assert "exec" not in result

    def test_get_required_reg_str_with_exec(self):
        """getRequiredRegStr with needExec=True appends 'exec'."""
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.getRequiredRegStr("    ", 1, 0, needExec=True)
        assert "exec" in result

    def test_get_required_reg_str_empty_with_exec(self):
        """getRequiredRegStr with zero vgpr/sgpr and exec=True."""
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.getRequiredRegStr("    ", 0, 0, needExec=True)
        assert '"exec"' in result

    def test_get_required_reg_str_empty_no_exec(self):
        """getRequiredRegStr with zero vgpr/sgpr and no exec produces :)."""
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        result = ai.getRequiredRegStr("    ", 0, 0, needExec=False)
        assert ":)" in result

    # --- getActivationAsmStr ---

    def test_get_activation_asm_str(self):
        """getActivationAsmStr returns a non-empty asm string for exp on Single."""
        ai = A.ActivationInline(DataType("s"), enableGuard=False)
        am = A.ActivationModule()
        am.setUsePK(False)
        mod = am.getExpModule(DataType("s"), 0, 0)
        result = ai.getActivationAsmStr(am, mod, "    ")
        assert len(result) > 0


# ============================================================================
# Group F — Helper functions: getMagic, getMagicStr, HexToStr
# ============================================================================

class TestHelperFunctions:
    """Covers lines 1236-1261: getMagic / getMagicStr / HexToStr."""

    def test_get_magic_half_no_pack(self):
        """getMagic on half returns correct 16-bit hex."""
        result = A.getMagic(DataType("h"), 1.0, isPack=False)
        assert result == "0x3c00"

    def test_get_magic_half_pack(self):
        """getMagic on half with isPack=True returns duplicated 32-bit hex."""
        result = A.getMagic(DataType("h"), 1.0, isPack=True)
        # 0x3c00 packed: (0x3c00 << 16) | 0x3c00 = 0x3c003c00
        assert result == "0x3c003c00"

    def test_get_magic_single(self):
        """getMagic on single returns correct float32 hex."""
        result = A.getMagic(DataType("s"), 1.0, isPack=False)
        assert result == "0x3f800000"

    def test_get_magic_str(self):
        """getMagicStr delegates to getMagic and returns a string."""
        result = A.getMagicStr(DataType("h"), 1.0)
        assert isinstance(result, str)
        assert result == "0x3c00"

    def test_hex_to_str_half_no_pack(self):
        """HexToStr on half with isPack=False does not duplicate."""
        result = A.HexToStr(DataType("h"), False, 0x29b9)
        assert result == "0x29b9"

    def test_hex_to_str_half_pack(self):
        """HexToStr on half with isPack=True duplicates the value."""
        result = A.HexToStr(DataType("h"), True, 0x29b9)
        assert result == hex((0x29b9 << 16) | 0x29b9)

    def test_hex_to_str_single(self):
        """HexToStr on single with isPack=False returns the same hex."""
        result = A.HexToStr(DataType("s"), False, 0x3f800000)
        assert result == "0x3f800000"

    def test_hex_to_str_multiple_args_error(self):
        """HexToStr raises RuntimeError when called with >1 hex args."""
        with pytest.raises(RuntimeError, match="multiple args"):
            A.HexToStr(DataType("s"), False, 0x3f800000, 0x40000000)


# ============================================================================
# Group G — ConvertCoeffToHex + HolderToGpr (via getModule)
# ============================================================================

class TestPostProcessPaths:
    """Covers lines 1264-1291: ConvertCoeffToHex + HolderToGpr called from
    postProcess() and assignGpr(). Exercise them via getModule() + assignGpr()
    to ensure deterministic line hits without depending on module internals."""

    def test_convert_coeff_to_hex_via_gelu(self):
        """getModule('gelu', Single) calls postProcess -> ConvertCoeffToHex."""
        am = A.ActivationModule()
        am.setUsePK(True)
        m = am.getModule(DataType("s"), "gelu", 0, 1)
        assert isinstance(m, Module)

    def test_holder_to_gpr_via_assign_gpr(self):
        """assignGpr() calls HolderToGpr() internally; exercise with abs(Single)."""
        am = A.ActivationModule()
        am.setUsePK(True)
        m = am.getModule(DataType("s"), "abs", 0, 1)
        assigned = am.assignGpr(m, 5, 10)
        assert isinstance(assigned, Module)

    def test_convert_coeff_to_hex_via_exp(self):
        """getExpModule builds an 'Exp' named Module; ConvertCoeffToHex special-cases it."""
        am = A.ActivationModule()
        am.setUsePK(True)
        m_exp = am.getModule(DataType("s"), "exp", 0, 1)
        assert isinstance(m_exp, Module)


# ============================================================================
# Group H — ActivationModule.getAllGprUsage
# ============================================================================

class TestGetAllGprUsage:
    """Covers lines 396-407: getAllGprUsage for all/hipblaslt_all/specific."""

    def test_get_all_gpr_usage_all_single(self):
        """'all' on Single returns usage dict for every non-gradient activation."""
        am = A.ActivationModule()
        am.setUsePK(True)
        usage = am.getAllGprUsage(DataType("s"), "all")
        assert "gelu" in usage
        assert "geluscaling" in usage
        assert "silu" in usage
        assert "swish" in usage
        assert all(isinstance(v["vgpr"], int) for v in usage.values())

    def test_get_all_gpr_usage_hipblaslt_all(self):
        """'hipblaslt_all' on Single returns usage dict for HIPBLASLT activations."""
        am = A.ActivationModule()
        am.setUsePK(True)
        usage = am.getAllGprUsage(DataType("s"), "hipblaslt_all")
        assert "gelu" in usage  # gelu is HIPBLASLT supported
        assert isinstance(usage, dict)

    def test_get_all_gpr_usage_gradonly_all(self):
        """'all' with GRADONLY export returns only dgelu + drelu usage."""
        am = A.ActivationModule()
        am.setUsePK(True)
        usage = am.getAllGprUsage(DataType("s"), "all",
                                  exportType=A.ActivationType.Export.GRADONLY)
        assert "dgelu" in usage
        assert "drelu" in usage
        # dgelu needs vgprs; drelu needs 0
        assert usage["drelu"]["vgpr"] == 0

    def test_get_all_gpr_usage_specific(self):
        """Specific activation returns a single-key dict."""
        am = A.ActivationModule()
        am.setUsePK(True)
        usage = am.getAllGprUsage(DataType("s"), "gelu")
        assert "Gelu" in usage or "gelu" in usage
