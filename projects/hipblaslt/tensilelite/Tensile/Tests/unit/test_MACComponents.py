# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import inspect
from types import SimpleNamespace

from Tensile import Components  # noqa: F401 - registers component implementations
from Tensile.Component import MAC
from Tensile.Components import MAC_F16, MAC_F16_HPA
from Tensile.KernelWriterAssembly import KernelWriterAssembly


def _concrete_implementations(component):
    for implementation in component.implementations.values():
        if inspect.isabstract(implementation):
            yield from _concrete_implementations(implementation)
        else:
            yield implementation


def test_mac_components_share_call_interface():
    expected = ["self", "writer", "tPA", "tPB", "m", "innerUnroll"]

    for implementation in _concrete_implementations(MAC):
        assert list(inspect.signature(implementation.__call__).parameters) == expected, \
            implementation.__name__


def test_packed_f16_mac_generates_both_accumulator_halves():
    kernel = {
        "ThreadTile0": 2,
        "ThreadTile1": 2,
    }
    writer = SimpleNamespace(states=SimpleNamespace(
        archCaps={},
        asmCaps={"v_pk_fma_f16": True},
        kernel=kernel,
    ))

    module = MAC_F16.FMA_F16_Packed()(writer, {}, {}, 0, 1)
    generated = str(module)

    assert generated.count("v_pk_fma_f16") == 2
    assert "v[vgprValuC + 0]" in generated
    assert "v[vgprValuC + 1]" in generated


def test_gfx90c_packed_f16_mac_iter_does_not_use_legacy_macro(monkeypatch):
    kernel = {
        "AssertSummationElementMultiple": 2,
        "InnerUnroll": 1,
        "NumDotElements": 1,
        "NumWaveSplitK": 1,
        "ProblemType": {
            "DataType": MAC_F16.DataType(MAC_F16.DataTypeEnum.Half),
            "MacDataTypeA": MAC_F16.DataType(MAC_F16.DataTypeEnum.Half),
            "MacDataTypeB": MAC_F16.DataType(MAC_F16.DataTypeEnum.Half),
            "HighPrecisionAccumulate": False,
        },
        "ThreadTile0": 2,
        "ThreadTile1": 2,
        "UseDotInstruction": False,
    }
    writer = object.__new__(KernelWriterAssembly)
    writer.states = SimpleNamespace(
        asmCaps={"HasVgprMSB": False, "v_pk_fma_f16": True},
        kernel=kernel,
        laneSGPRCount=2,
        unrollIdx=0,
    )
    writer.do = {"MAC": True}
    writer.vgprPool = SimpleNamespace(checkIn=lambda _register: None)
    monkeypatch.setattr(writer, "loopCounterName", lambda *_args: "LoopCounterL")
    monkeypatch.setattr(MAC, "find", lambda *_args: MAC_F16.FMA_F16_Packed())

    generated = str(writer.macIter(kernel, {}, {}, 0, 1, True))
    assert generated.count("v_pk_fma_f16") == 2
    assert "MAC_2x2_X0" not in generated


def test_gfx90c_f16_hpa_emits_typed_mixed_mad_for_each_half_pair():
    kernel = {
        "ThreadTile0": 2,
        "ThreadTile1": 2,
    }
    writer = SimpleNamespace(states=SimpleNamespace(
        archCaps={},
        asmCaps={"v_fma_mix_f32": False},
        kernel=kernel,
        version=(9, 0, 12),
    ))

    module = MAC_F16_HPA.FMA_F16_HPA_MAD_MIX()(
        writer,
        {"tileIdx": 0},
        {"tileIdx": 1},
        0,
        1,
    )
    generated = str(module)

    assert generated.count("v_mad_mix_f32") == 4
    assert "v[vgprValuA_X0_I0+0]" in generated
    for op_sel in ([0,0,0], [1,0,0], [0,1,0], [1,1,0]):
        modifiers = f"op_sel:{op_sel} op_sel_hi:[1,1,0]".replace(" ", "")
        assert modifiers in generated.replace(" ", "")
    assert "v_cvt_f32_f16" not in generated
    assert "v_lshrrev_b32" not in generated
    assert "v_fma_f32" not in generated
