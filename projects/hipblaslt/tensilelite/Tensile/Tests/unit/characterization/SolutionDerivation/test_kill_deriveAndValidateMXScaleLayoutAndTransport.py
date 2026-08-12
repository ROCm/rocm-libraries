import importlib

import pytest

S = importlib.import_module("Tensile.SolutionStructs.Solution")
pytestmark = pytest.mark.unit

F = S._deriveAndValidateMXScaleLayoutAndTransport

def _state(mxLoadInst="Auto", mxScaleFormat="Auto", tdmInst=0, isa=(9, 4, 2),
           mxBlockA=False, mxBlockB=False):
    return {
        "ProblemType": {"MXBlockA": mxBlockA, "MXBlockB": mxBlockB},
        "MXLoadInst": mxLoadInst,
        "MXScaleFormat": mxScaleFormat,
        "TDMInst": tdmInst,
        "ISA": isa,
        "StreamK": 0,
        "Valid": True,
    }

def test_auto_mxloadinst_tdm_when_tdminst_nonzero(capsys):
    st = _state(mxLoadInst="Auto", tdmInst=2)
    asm = {"HasTDM": True}
    arch = {"HasMXScaleSwizzle": True}
    ret = F(st, asm, arch, True)
    assert ret is True
    assert st["Valid"] is True
    assert st["MXLoadInst"] == "TDM"
    assert st["TDMInst"] == 2
    assert st["MXScaleFormat"] == "InMemorySwizzle"
    assert capsys.readouterr().out == ""

def test_auto_mxloadinst_bufferload_when_tdminst_zero_default_noswizzle():
    st = _state(mxLoadInst="Auto", tdmInst=0, isa=(9, 4, 2), mxBlockA=False, mxBlockB=False)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": False}, True)
    assert ret is True
    assert st["MXLoadInst"] == "BufferLoad"
    assert st["MXScaleFormat"] == "NoSwizzle"

def test_auto_mxscaleformat_hostpreswizzle_on_gfx950_bufferload():
    st = _state(mxLoadInst="Auto", tdmInst=0, isa=(9, 5, 0), mxBlockA=True, mxBlockB=False)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": True}, True)
    assert ret is True
    assert st["MXLoadInst"] == "BufferLoad"
    assert st["MXScaleFormat"] == "HostPreSwizzle"

def test_auto_mxscaleformat_noswizzle_when_not_gfx950_and_no_mxblock():
    st = _state(mxLoadInst="Auto", tdmInst=0, isa=(9, 5, 0), mxBlockA=False, mxBlockB=False)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": False}, True)
    assert ret is True
    assert st["MXScaleFormat"] == "NoSwizzle"

def test_tdminst_promoted_to_three_when_tdm_and_zero():
    st = _state(mxLoadInst="TDM", mxScaleFormat="InMemorySwizzle", tdmInst=0)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": True}, True)
    assert ret is True
    assert st["TDMInst"] == 3

def test_reject_globalload_not_implemented(capsys):
    st = _state(mxLoadInst="GlobalLoad", mxScaleFormat="NoSwizzle", tdmInst=0)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": True}, True)
    assert ret is False
    assert st["Valid"] is False
    assert "MXLoadInst=GlobalLoad not implemented yet" in capsys.readouterr().out

def test_reject_tdm_requires_hastdm(capsys):
    st = _state(mxLoadInst="TDM", mxScaleFormat="NoSwizzle", tdmInst=0)
    ret = F(st, {"HasTDM": False}, {"HasMXScaleSwizzle": True}, True)
    assert ret is False
    assert st["Valid"] is False
    assert "MXLoadInst=TDM requires asmCaps.HasTDM" in capsys.readouterr().out

def test_reject_incompatible_tdminst(capsys):
    st = _state(mxLoadInst="BufferLoad", mxScaleFormat="NoSwizzle", tdmInst=5)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": True}, True)
    assert ret is False
    assert st["Valid"] is False
    out = capsys.readouterr().out
    assert "incompatible with TDMInst=5" in out
    assert "MXLoadInst=BufferLoad is incompatible" in out

def test_reject_swizzled_format_requires_swizzle_cap(capsys):
    st = _state(mxLoadInst="BufferLoad", mxScaleFormat="HostPreSwizzle", tdmInst=0)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": False}, True)
    assert ret is False
    assert st["Valid"] is False
    assert "MXScaleFormat=HostPreSwizzle requires archCaps.HasMXScaleSwizzle" in capsys.readouterr().out

def test_reject_inmemoryswizzle_requires_tdm(capsys):
    st = _state(mxLoadInst="BufferLoad", mxScaleFormat="InMemorySwizzle", tdmInst=0,
                mxBlockA=False, mxBlockB=True)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": True}, True)
    assert ret is False
    assert st["Valid"] is False
    assert "MXScaleFormat=InMemorySwizzle requires MXLoadInst=TDM" in capsys.readouterr().out

def test_reject_hostpreswizzle_requires_bufferload(capsys):
    st = _state(mxLoadInst="TDM", mxScaleFormat="HostPreSwizzle", tdmInst=0, isa=(9, 5, 0),
                mxBlockA=True)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": True}, True)
    assert ret is False
    assert st["Valid"] is False
    assert "MXScaleFormat=HostPreSwizzle requires MXLoadInst=BufferLoad" in capsys.readouterr().out

def test_reject_hostpreswizzle_only_on_gfx950(capsys):
    st = _state(mxLoadInst="BufferLoad", mxScaleFormat="HostPreSwizzle", tdmInst=0, isa=(9, 4, 2),
                mxBlockA=True)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": True}, True)
    assert ret is False
    assert st["Valid"] is False
    assert "MXScaleFormat=HostPreSwizzle is only implemented on the gfx950 host pipeline" in capsys.readouterr().out

def test_reject_tdm_must_produce_inmemoryswizzle(capsys):
    st = _state(mxLoadInst="TDM", mxScaleFormat="NoSwizzle", tdmInst=0, mxBlockA=True)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": True}, True)
    assert ret is False
    assert st["Valid"] is False
    assert "MXLoadInst=TDM currently always produces MXScaleFormat=InMemorySwizzle" in capsys.readouterr().out

def test_reject_gfx1250_noswizzle_unsupported(capsys):
    st = _state(mxLoadInst="BufferLoad", mxScaleFormat="NoSwizzle", tdmInst=0, isa=(12, 5, 0),
                mxBlockA=True)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": True}, True)
    assert ret is False
    assert st["Valid"] is False
    assert "MXScaleFormat=NoSwizzle is not supported on gfx1250" in capsys.readouterr().out

def test_elif_is_and_not_or_for_hostpreswizzle_default():
    st = _state(mxLoadInst="Auto", mxScaleFormat="Auto", tdmInst=0, isa=(9, 5, 0),
                mxBlockA=False, mxBlockB=False)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": False}, True)
    assert ret is True
    assert st["MXScaleFormat"] == "NoSwizzle"

def test_reject_message_suppressed_when_flag_false(capsys):
    st = _state(mxLoadInst="GlobalLoad", mxScaleFormat="NoSwizzle", tdmInst=0)
    ret = F(st, {"HasTDM": True}, {"HasMXScaleSwizzle": True}, False)
    assert ret is False
    assert st["Valid"] is False
    assert capsys.readouterr().out == ""
