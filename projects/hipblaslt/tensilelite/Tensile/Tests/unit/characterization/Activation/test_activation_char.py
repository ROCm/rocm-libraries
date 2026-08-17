################################################################################
# Characterization tests for Tensile.Activation
#
# ADD-ONLY. Activation.py is ~1040 statements, the large majority of which is
# rocisa **assembly codegen** (the getXModule emitters, CombineInstructions /
# FuseInstruction, ConvertCoeffToHex / HolderToGpr / createVgprIdxList,
# ActivationInline). Per the project's codegen/asm exclusion that surface is out
# of scope, and in this environment most emitters raise immediately anyway
# (NameError 'SelectBit'/'VMaxF16', KeyError 'TransOpWait' — see DECISIONS D13).
#
# This suite pins the PURE configuration/type/numeric layer plus the asm
# entry-points that DO run cleanly with dummy vgprs (abs/relu/none/clippedrelu/
# leakyrelu/clamp/drelu).
################################################################################
import importlib

import pytest

pytestmark = pytest.mark.unit

A = importlib.import_module("Tensile.Activation")
DataType = importlib.import_module("Tensile.Common.DataType").DataType
AT = A.ActivationType


# ---------------------------------------------------------------------------
# ActivationAvailable / ActivationTypeRegister.typeAvailable
# ---------------------------------------------------------------------------
def test_activation_available_flags():
    av = A.ActivationAvailable(canHalf=True, canInt32=True)
    assert av.half is True and av.int32 is True
    assert av.single is False and av.double is False


@pytest.mark.parametrize(
    "code,kw,expected",
    [
        ("H", {"canHalf": True}, True),
        ("S", {"canSingle": True}, True),
        ("D", {"canDouble": True}, True),
        ("B", {"canBFloat16": True}, True),
        ("H", {"canSingle": True}, False),  # half but only single allowed
        ("S", {}, False),                   # nothing allowed
    ],
)
def test_type_available(code, kw, expected):
    reg = A.ActivationTypeRegister("x", False, 0, **kw)
    assert reg.typeAvailable(DataType(code)) is expected


# ---------------------------------------------------------------------------
# ActivationType construction
# ---------------------------------------------------------------------------
def test_activation_type_from_str_lookup():
    assert AT("Relu").value == "relu"


def test_activation_type_from_veri():
    assert AT("exp").value == "exp"


def test_activation_type_from_instance():
    base = AT("gelu")
    assert AT(base).value == "gelu"


def test_activation_type_invalid_str_raises():
    with pytest.raises(RuntimeError, match="Unrecognized activation type"):
        AT("not-a-real-activation")


def test_activation_type_invalid_input_raises():
    with pytest.raises(RuntimeError, match="Unrecognized input type"):
        AT(123)


# ---------------------------------------------------------------------------
# passActivation / getAdditionalArgNum / arg strings
# ---------------------------------------------------------------------------
def test_pass_activation():
    a = AT("relu")
    assert a.passActivation(True, AT.Export.NORMAL) is True
    assert a.passActivation(True, AT.Export.GRADONLY) is False
    assert a.passActivation(False, AT.Export.GRADONLY) is True
    assert a.passActivation(True, AT.Export.BOTH) is False


def test_additional_arg_num_specific():
    assert AT("clippedrelu").getAdditionalArgNum() == 2
    assert AT("leakyrelu").getAdditionalArgNum() == 1
    assert AT("relu").getAdditionalArgNum() == 0


def test_additional_arg_num_veri_is_zero():
    assert AT("exp").getAdditionalArgNum() == 0  # 'exp' not in lookup -> 0


def test_additional_arg_num_all_aggregates():
    # 'all' takes the max extraArgs over non-skipped activations -> 2 (clamp/tanh)
    assert AT("all").getAdditionalArgNum() == 2


def test_additional_arg_string_list():
    a = AT("clippedrelu")  # 2 args
    assert a.getAdditionalArgStringList() == ["activationAlpha", "activationBeta"]
    assert a.getAdditionalArgStringList(addPrefix=False) == ["alpha", "beta"]


def test_fit_supported():
    assert AT("relu").fitSupported(AT.SupportedBy.ALL, AT.SupportedBy.HIPBLASLT)
    assert not AT("relu").fitSupported(AT.SupportedBy.TENSILE, AT.SupportedBy.HIPBLASLT)


# ---------------------------------------------------------------------------
# getEnumIndex / getEnumStrList
# ---------------------------------------------------------------------------
def test_get_enum_index():
    assert AT.getEnumIndex("none") == 0
    assert AT.getEnumIndex("abs") == 1


def test_get_enum_str_list_single():
    lst = AT.getEnumStrList(DataType("S"), configSupported=AT.SupportedBy.ALL)
    assert "relu" in lst and "none" in lst
    # 'all'/'hipblaslt_all' are never listed
    assert "all" not in lst and "hipblaslt_all" not in lst


def test_get_enum_str_list_exclude_none():
    lst = AT.getEnumStrList(DataType("S"), configSupported=AT.SupportedBy.ALL, includeNone=False)
    assert "none" not in lst


def test_get_enum_str_list_gradonly_keeps_gradients():
    # GRADONLY skips non-gradient activations -> dgelu/drelu remain, relu drops
    lst = AT.getEnumStrList(
        DataType("S"), configSupported=AT.SupportedBy.ALL, exportType=AT.Export.GRADONLY
    )
    assert "dgelu" in lst or "drelu" in lst
    assert "relu" not in lst


def test_get_enum_str_list_empty_warns(capsys):
    # double type has essentially no hipblaslt activations -> warning path
    AT.getEnumStrList(
        DataType("D"), configSupported=AT.SupportedBy.HIPBLASLT, includeNone=False
    )
    # the warning is emitted via printWarning; just ensure it didn't raise
    capsys.readouterr()


# ---------------------------------------------------------------------------
# dunders: state/repr/str/eq/lt/toEnum
# ---------------------------------------------------------------------------
def test_state_repr_str_toenum():
    a = AT("relu")
    assert a.state() == "Relu"
    assert str(a) == "Relu"
    assert repr(a) == "Relu"
    assert a.toEnum() == "Relu"


def test_eq():
    assert AT("relu") == "Relu"
    assert AT("relu") == AT("relu")
    assert not (AT("relu") == AT("gelu"))
    with pytest.raises(RuntimeError):
        AT("relu") == 5


def test_lt():
    assert AT("abs") < "relu"
    assert AT("abs") < AT("relu")
    with pytest.raises(RuntimeError):
        AT("abs") < 5


# ---------------------------------------------------------------------------
# actCacheInfo.isSame
# ---------------------------------------------------------------------------
def _cache_info(**over):
    base = dict(
        usePK=True, saturateI8=False, enableGuard=False, isAlt=False, prefix="p",
        vgprIdxList=[], module=None, vgprCounter=0, sgprCounter=0,
    )
    base.update(over)
    return A.actCacheInfo(**base)


def test_cache_info_is_same():
    ci = _cache_info()
    assert ci.isSame(True, False, False, False, "p") is True
    assert ci.isSame(False, False, False, False, "p") is False  # usePK differs
    assert ci.isSame(True, False, False, False, "q") is False   # prefix differs


# ---------------------------------------------------------------------------
# numeric helpers: getMagic / getMagicStr / HexToStr / addSpace
# ---------------------------------------------------------------------------
def test_get_magic_single():
    assert A.getMagic(DataType("S"), 1.0) == hex(0x3f800000)


def test_get_magic_half_and_pack():
    plain = A.getMagic(DataType("H"), 1.0)
    packed = A.getMagic(DataType("H"), 1.0, isPack=True)
    assert int(plain, 16) == 0x3c00
    assert int(packed, 16) == (0x3c00 << 16) | 0x3c00


def test_get_magic_double_exits():
    with pytest.raises(SystemExit):
        A.getMagic(DataType("D"), 1.0)


def test_get_magic_str():
    assert A.getMagicStr(DataType("S"), 1.0) == str(hex(0x3f800000))


def test_hex_to_str_single_and_pack():
    assert A.HexToStr(DataType("S"), False, 0x1234) == str(hex(0x1234))
    packed = A.HexToStr(DataType("H"), True, 0xABCD)
    assert int(packed, 16) == (0xABCD << 16) | 0xABCD


def test_hex_to_str_multiple_args_raises():
    with pytest.raises(RuntimeError, match="multiple args"):
        A.HexToStr(DataType("S"), False, 1, 2)


def test_add_space():
    out = A.addSpace("1234", "ab")
    assert out.endswith("ab")
    assert len(out) == len("1234") + len("ab")


# ---------------------------------------------------------------------------
# ActivationModule: defaults, setters, counters, vgprPrefix, getModule (working)
# ---------------------------------------------------------------------------
def test_module_defaults_and_reduce():
    m = A.ActivationModule()
    assert m.usePK is True and m.saturateI8 is False
    assert m.useCache is False
    # __reduce__ supports pickling to a fresh instance
    cls, args = m.__reduce__()
    assert cls is A.ActivationModule and args == ()


def test_module_setters():
    m = A.ActivationModule()
    m.setUsePK(False); m.setSaturationForInt8(True); m.setVgprPrefixFormat("v%u")
    m.setUseCache(True); m.setGuard(True); m.setAlt(True)
    assert (m.usePK, m.saturateI8, m.vgprPrefixFormat) == (False, True, "v%u")
    assert (m.useCache, m.enableGuard, m.isAlt) == (True, True, True)


def test_module_gpr_counters():
    m = A.ActivationModule()
    assert m.getVgpr(2) == 0 and m.vgprCounter == 2
    assert m.getSgpr(3) == 0 and m.sgprCounter == 3
    m.resetGprCounter()
    assert m.vgprCounter == 0 and m.sgprCounter == 0


def test_module_vgpr_prefix():
    m = A.ActivationModule()
    m.setVgprPrefixFormat("v%u")
    # int + format -> formatted name
    r = m.vgprPrefix(5)
    assert r is not None
    # explicit string passthrough + index arg form
    r2 = m.vgprPrefix("myreg", 2)
    assert r2 is not None


@pytest.mark.parametrize("dtype,act", [
    ("S", "abs"), ("S", "relu"), ("S", "clippedrelu"), ("S", "leakyrelu"),
    ("S", "clamp"), ("S", "drelu"), ("H", "abs"), ("H", "relu"),
])
def test_get_module_working_paths(dtype, act):
    # these emitters run cleanly with dummy vgprs; pin that they return a Module
    m = A.ActivationModule()
    mod = m.getModule(DataType(dtype), act, 0, 1)
    assert type(mod).__name__ == "Module"


def test_get_module_none_and_unknown():
    m = A.ActivationModule()
    assert type(m.getModule(DataType("S"), "none", 0, 1)).__name__ == "Module"
    # unknown type returns a placeholder Module rather than raising
    assert type(m.getModule(DataType("S"), "bogus", 0, 1)).__name__ == "Module"


def test_get_all_gpr_usage_single_type():
    m = A.ActivationModule()
    usage = m.getAllGprUsage(DataType("S"), "relu")
    assert "relu" in usage
    assert set(usage["relu"].keys()) == {"vgpr", "sgpr"}
