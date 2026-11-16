# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ tiled-attention entries refuse what the Python builders refuse, with
the same reason, instead of crashing.

* A zero or non-dividing head pair: the 2D ``_new`` entries used to fail at
  naming before initialising the builder, and the caller then freed that
  uninitialised builder (signal 11).
* fastKV register-P on gfx942: the body is gfx950-only, so the C++ builder must
  refuse it as Python does rather than emit gfx950 MFMA for gfx942.
* The grouped-KV2 softmax and early-V schedule rules: the C++ entries that
  build gfx950 specs (tiled 2D and fastKV register-P; lower, serialize and
  verify) refuse them with the Python text rather than build the spec, for every
  rule whose fields the entry's binding passes. The gfx942 entry refuses
  grouped-KV2 and the fp8 K/V cache outright, on every arch, as the gfx942 spec
  does.
"""

from __future__ import annotations

import dataclasses
import re
from typing import NamedTuple

import pytest
from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx942Spec2D,
)
from kernels.gfx950.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx950Spec2D,
)
from kernels.gfx950.attention_tiled_2d_fastkv_regp import (
    build_unified_attention_2d_fastkv_register_p,
    make_fastkv_register_p_spec,
)

rocke_engine = pytest.importorskip("rocke_engine")

_PROBLEM = {
    "head_size": 128,
    "block_size": 16,
    "dtype": "bf16",
    "use_sinks": False,
    "sliding_window": 0,
    "has_softcap": False,
}


@pytest.mark.parametrize(
    "arch, spec_cls",
    [("gfx942", Gfx942Spec2D), ("gfx950", Gfx950Spec2D)],
)
@pytest.mark.parametrize(
    "heads, reason",
    [((32, 0), "must be > 0"), ((32, 6), "must be divisible by num_kv_heads")],
)
def test_2d_entry_refuses_bad_head_counts_like_python(arch, spec_cls, heads, reason):
    hq, hkv = heads
    spec = spec_cls(num_query_heads=hq, num_kv_heads=hkv, **_PROBLEM)
    with pytest.raises(ValueError, match=reason) as py_err:
        spec.kernel_name()
    lower = getattr(rocke_engine, f"{arch}_attention_tiled_2d_lower_llvm")
    with pytest.raises(Exception) as cpp_err:
        lower(dataclasses.asdict(spec), arch)
    assert str(py_err.value) in str(cpp_err.value)


def test_fastkv_register_p_entry_refuses_gfx942_like_python():
    spec = make_fastkv_register_p_spec(
        Gfx950Spec2D(
            num_query_heads=64,
            num_kv_heads=8,
            **{**_PROBLEM, "head_size": 64, "block_size": 32},
        )
    )
    with pytest.raises(ValueError, match="gfx950 kernel") as py_err:
        build_unified_attention_2d_fastkv_register_p(spec, arch="gfx942")
    with pytest.raises(Exception) as cpp_err:
        rocke_engine.gfx950_attention_tiled_2d_fastkv_regp_lower_llvm(
            dataclasses.asdict(spec), "gfx942"
        )
    assert str(py_err.value) in str(cpp_err.value)


class _Refusal(NamedTuple):
    rule: str
    change: dict
    py_reason: str
    # None when the C++ text is the Python one.
    cpp_reason: str | None = None
    # The fastKV binding passes only part of the spec to C++.
    fastkv_binding_carries: bool = True


_GKV2_R4 = {
    "tile_size": 32,
    "block_m_per_warp": 32,
    "use_mfma_32x32": True,
    "use_transposed_qk_32x32": True,
    "use_grouped_kv2_softmax": True,
}
# Specs each spec class accepts; every case changes one of them.
_GFX950_GKV2 = {"num_query_heads": 64, "num_kv_heads": 8, **_PROBLEM, **_GKV2_R4}
_FASTKV_GKV2 = {
    **_GFX950_GKV2,
    "head_size": 64,
    "block_size": 32,
    "tile_size": 64,
    "num_warps": 4,
    "use_fast_paged_kv_desc": True,
}
_GFX942_NARROW = {"num_query_heads": 64, "num_kv_heads": 8, **_PROBLEM}

_FP8_KV = {"kv_storage_dtype": "fp8e4m3"}

_SHARED_RULES = [
    _Refusal(
        rule="bf16 only",
        change={"dtype": "fp16"},
        py_reason=(
            "use_grouped_kv2_softmax v1 is restricted to bf16 - dtype='fp16', "
            "expected 'bf16'"
        ),
    ),
    _Refusal(
        rule="transposed 32x32",
        change={"use_transposed_qk_32x32": False},
        py_reason=(
            "use_grouped_kv2_softmax requires the transposed 32x32 R4 path - "
            "use_mfma_32x32=True, use_transposed_qk_32x32=False, expected both True"
        ),
    ),
    # Python reaches the next two through its 32x32 rules first; the C++ entries
    # do not carry those, so the grouped-KV2 rule refuses there.
    _Refusal(
        rule="32x32 atom",
        change={"use_mfma_32x32": False},
        py_reason="use_transposed_qk_32x32 requires use_mfma_32x32",
        cpp_reason=(
            "use_grouped_kv2_softmax requires the transposed 32x32 R4 path - "
            "use_mfma_32x32=False, use_transposed_qk_32x32=True, expected both True"
        ),
    ),
    _Refusal(
        rule="32-row warp",
        change={"block_m_per_warp": 16},
        py_reason=(
            "use_mfma_32x32 requires block_m_per_warp=32: one 32-row MFMA atom per warp"
        ),
        cpp_reason=(
            "use_grouped_kv2_softmax requires one 32-row MFMA atom per warp - "
            "block_m_per_warp=16, expected 32"
        ),
    ),
    _Refusal(
        rule="sliding window",
        change={"sliding_window": 64},
        py_reason=(
            "use_grouped_kv2_softmax v1 does not support a sliding window - "
            "sliding_window=64, expected 0"
        ),
    ),
    _Refusal(
        rule="softcap",
        change={"has_softcap": True},
        py_reason=(
            "use_grouped_kv2_softmax v1 does not support softcap - "
            "has_softcap=True, expected False"
        ),
    ),
    _Refusal(
        rule="ALiBi",
        change={"use_alibi": True},
        py_reason=(
            "use_grouped_kv2_softmax v1 does not support ALiBi - use_alibi=True, "
            "expected False"
        ),
        fastkv_binding_carries=False,
    ),
    _Refusal(
        rule="QQ bias",
        change={"use_qq_bias": True},
        py_reason=(
            "use_grouped_kv2_softmax v1 does not support QQ bias - "
            "use_qq_bias=True, expected False"
        ),
        fastkv_binding_carries=False,
    ),
    _Refusal(
        rule="fp8 K/V",
        change=_FP8_KV,
        py_reason=(
            "use_grouped_kv2_softmax v1 does not support FP8 KV - "
            "kv_storage_dtype='fp8e4m3', expected None"
        ),
        fastkv_binding_carries=False,
    ),
    _Refusal(
        rule="early-V with grouped-KV2",
        change={"use_early_v_schedule": True},
        py_reason=(
            "use_early_v_schedule does not support grouped-KV2 softmax - "
            "use_grouped_kv2_softmax=True, expected False"
        ),
        fastkv_binding_carries=False,
    ),
    _Refusal(
        rule="early-V with fp8 K/V",
        change={
            "use_grouped_kv2_softmax": False,
            "use_early_v_schedule": True,
            **_FP8_KV,
        },
        py_reason=(
            "use_early_v_schedule v1 does not support FP8 KV - "
            "kv_storage_dtype='fp8e4m3', expected None"
        ),
        fastkv_binding_carries=False,
    ),
]


class _Entry(NamedTuple):
    name: str
    spec_cls: type
    accepted: dict
    binding: str
    archs: tuple[str, ...]
    rules: list[_Refusal]


# The gfx950 builders admit only gfx950.
_GFX950_ENTRIES = [
    _Entry(
        "gfx950 tiled 2D",
        Gfx950Spec2D,
        _GFX950_GKV2,
        "gfx950_attention_tiled_2d",
        ("gfx950",),
        _SHARED_RULES,
    ),
    _Entry(
        "gfx950 fastKV register-P",
        Gfx950Spec2D,
        _FASTKV_GKV2,
        "gfx950_attention_tiled_2d_fastkv_regp",
        ("gfx950",),
        [refusal for refusal in _SHARED_RULES if refusal.fastkv_binding_carries],
    ),
]
# Each binding entry point builds through the same C++ entry.
_ENTRY_POINTS = ("lower_llvm", "serialize_ir", "verify")


@pytest.mark.parametrize(
    "spec_cls, accepted, binding, arch",
    [
        pytest.param(e.spec_cls, e.accepted, e.binding, arch, id=f"{e.name} on {arch}")
        for e in _GFX950_ENTRIES
        for arch in e.archs
    ],
)
def test_2d_entries_build_the_accepted_specs(spec_cls, accepted, binding, arch):
    """The C++ entries build each accepted spec, so a refusal below is the rule's."""
    fields = dataclasses.asdict(spec_cls(**accepted))
    assert getattr(rocke_engine, f"{binding}_serialize_ir")(fields, arch)


def _assert_both_engines_refuse(spec_cls, accepted, entry_point, arch, refusal):
    accepted_spec = spec_cls(**accepted)
    with pytest.raises(ValueError) as py_err:
        dataclasses.replace(accepted_spec, **refusal.change)
    assert str(py_err.value) == refusal.py_reason
    refused_fields = {**dataclasses.asdict(accepted_spec), **refusal.change}
    cpp_reason = refusal.cpp_reason or refusal.py_reason
    with pytest.raises(RuntimeError, match=re.escape(cpp_reason)):
        getattr(rocke_engine, entry_point)(refused_fields, arch)


@pytest.mark.parametrize(
    "spec_cls, accepted, entry_point, arch, refusal",
    [
        pytest.param(
            e.spec_cls,
            e.accepted,
            f"{e.binding}_{entry_point}",
            arch,
            refusal,
            id=f"{e.name} {entry_point} on {arch}: {refusal.rule}",
        )
        for e in _GFX950_ENTRIES
        for arch in e.archs
        for entry_point in _ENTRY_POINTS
        for refusal in e.rules
    ],
)
def test_gfx950_2d_entries_refuse_grouped_kv2_and_early_v_rules_like_python(
    spec_cls, accepted, entry_point, arch, refusal
):
    """Python refuses the changed spec at construction; the C++ entry refuses it too."""
    _assert_both_engines_refuse(spec_cls, accepted, entry_point, arch, refusal)


# The gfx942 spec refuses grouped-KV2 and the fp8 K/V cache in its first checks,
# whatever the target; its C++ rules word them differently.
_GFX942_GATE_PY = (
    "gfx942 tiled-2D attention supports only the narrow 16x16x16 default path and "
    "the K=8 32x32x8 wide path; these are not available on gfx942: {knobs}"
)
_GFX942_GATE_CPP = (
    "gfx942 tiled-2D attention supports only the narrow 16x16x16 default path; "
    "gfx950-only knobs are not available on gfx942"
)
_GFX942_REFUSALS = [
    _Refusal(
        rule="grouped-KV2",
        change={"use_grouped_kv2_softmax": True},
        py_reason=_GFX942_GATE_PY.format(knobs="use_grouped_kv2_softmax"),
        cpp_reason=_GFX942_GATE_CPP,
    ),
    _Refusal(
        rule="grouped-KV2 on the transposed 32x32 path",
        change=_GKV2_R4,
        py_reason=_GFX942_GATE_PY.format(
            knobs="use_mfma_32x32, use_grouped_kv2_softmax"
        ),
        cpp_reason=_GFX942_GATE_CPP,
    ),
    _Refusal(
        rule="early-V with grouped-KV2",
        change={"use_grouped_kv2_softmax": True, "use_early_v_schedule": True},
        py_reason=_GFX942_GATE_PY.format(knobs="use_grouped_kv2_softmax"),
        cpp_reason=_GFX942_GATE_CPP,
    ),
    _Refusal(
        rule="early-V with fp8 K/V",
        change={"use_early_v_schedule": True, **_FP8_KV},
        py_reason=(
            "gfx942 tiled-2D attention does not support the fp8 K/V cache "
            "(its PV path uses ds_read_tr_b8, a gfx950-only transpose read)"
        ),
        cpp_reason=(
            "gfx942 tiled-2D attention has no fp8 K/V cache path "
            "(kv_storage_dtype must be None on gfx942)"
        ),
    ),
]


@pytest.mark.parametrize(
    "entry_point, arch, refusal",
    [
        pytest.param(
            f"gfx942_attention_tiled_2d_{entry_point}",
            arch,
            refusal,
            id=f"{entry_point} on {arch}: {refusal.rule}",
        )
        for arch in ("gfx942", "gfx950")
        for entry_point in _ENTRY_POINTS
        for refusal in _GFX942_REFUSALS
    ],
)
def test_gfx942_2d_entry_refuses_grouped_kv2_and_fp8_kv_on_every_arch(
    entry_point, arch, refusal
):
    """The gfx942 entry runs the gfx942 spec checks for every target, so the
    grouped-KV2 and early-V specs the gfx942 spec refuses are refused on both
    arches its builder admits."""
    _assert_both_engines_refuse(
        Gfx942Spec2D, _GFX942_NARROW, entry_point, arch, refusal
    )


@pytest.mark.parametrize(
    "change, reason",
    [
        pytest.param(
            {"use_additive_bias": True},
            "use_grouped_kv2_softmax v1 does not support additive bias - "
            "use_additive_bias=True, expected False",
            id="additive bias",
        ),
        pytest.param(
            {"use_additive_bias": True, "has_softcap": True},
            "use_grouped_kv2_softmax v1 does not support softcap - has_softcap=True, "
            "expected False",
            id="additive bias after the shared rules",
        ),
    ],
)
def test_gfx950_2d_spec_refuses_grouped_kv2_with_additive_bias(change, reason):
    """The C++ spec has no additive-bias field, so only Python carries this rule."""
    with pytest.raises(ValueError) as py_err:
        dataclasses.replace(Gfx950Spec2D(**_GFX950_GKV2), **change)
    assert str(py_err.value) == reason
