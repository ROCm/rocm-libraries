# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ tiled-attention entries refuse what the Python builders refuse, with
the same reason, instead of crashing.

* A zero or non-dividing head pair: the 2D ``_new`` entries used to fail at
  naming before initialising the builder, and the caller then freed that
  uninitialised builder (signal 11).
* fastKV register-P on gfx942: the body is gfx950-only, so the C++ builder must
  refuse it as Python does rather than emit gfx950 MFMA for gfx942.
* The tiled-2D spec rules the C++ entries mirror -- the gfx950 32x32 atom
  geometry, the transposed-softmax VALU opts, grouped-KV2 softmax and the
  early-V schedule: every C++ tiled-2D entry (gfx942, gfx950, gfx950 fastKV
  register-P; lower, serialize and verify) on every arch its builder admits
  refuses them with the Python text rather than build the spec, for every rule
  whose fields the entry's binding passes. The gfx942 entry refuses grouped-KV2
  and the fp8 K/V cache outright, as the gfx942 spec does.
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
from rocke.tests.lanes import BUILD_VARIANT, import_or_skip_until


@pytest.fixture(scope="module")
def rocke_engine():
    """The C++ entries' binding. Only the tests that call it take it, so the
    Python-only spec checks run where the binding is not built."""
    return import_or_skip_until("rocke_engine", BUILD_VARIANT)


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
def test_2d_entry_refuses_bad_head_counts_like_python(
    rocke_engine, arch, spec_cls, heads, reason
):
    hq, hkv = heads
    spec = spec_cls(num_query_heads=hq, num_kv_heads=hkv, **_PROBLEM)
    with pytest.raises(ValueError, match=reason) as py_err:
        spec.kernel_name()
    lower = getattr(rocke_engine, f"{arch}_attention_tiled_2d_lower_llvm")
    with pytest.raises(Exception) as cpp_err:
        lower(dataclasses.asdict(spec), arch)
    assert str(py_err.value) in str(cpp_err.value)


def test_fastkv_register_p_entry_refuses_gfx942_like_python(rocke_engine):
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


_PROBLEM_64_8 = {"num_query_heads": 64, "num_kv_heads": 8, **_PROBLEM}
_GKV2_R4 = {
    "tile_size": 32,
    "block_m_per_warp": 32,
    "use_mfma_32x32": True,
    "use_transposed_qk_32x32": True,
    "use_grouped_kv2_softmax": True,
}
_S1MASK = {
    "tile_size": 32,
    "block_m_per_warp": 32,
    "use_transposed_qk_32x32": True,
    "use_transposed_scalar_state": True,
    "use_transposed_mask_once": True,
    "use_transposed_mask_limit": True,
}
# Specs each spec class accepts; every case changes one of them.
_GFX950_GKV2 = {**_PROBLEM_64_8, **_GKV2_R4}
_GFX950_VALU = {**_PROBLEM_64_8, **_S1MASK, "use_mfma_32x32": True}
_GFX942_VALU = {**_PROBLEM_64_8, **_S1MASK, "use_mfma_32x32x8": True}
_FASTKV_GKV2 = {
    **_GFX950_GKV2,
    "head_size": 64,
    "block_size": 32,
    "tile_size": 64,
    "num_warps": 4,
    "use_fast_paged_kv_desc": True,
}

_FP8_KV = {"kv_storage_dtype": "fp8e4m3"}

# The gfx950 32x32x16 atom geometry and the transposed QK it needs.
_GFX950_32X32_RULES = [
    _Refusal(
        rule="32x32 atom per warp",
        change={"block_m_per_warp": 16},
        py_reason=(
            "use_mfma_32x32 requires one 32-row MFMA atom per warp - "
            "block_m_per_warp=16, expected 32"
        ),
    ),
    _Refusal(
        rule="32x32 whole tiles",
        change={"tile_size": 48},
        py_reason=(
            "use_mfma_32x32 requires whole 32-row tiles - tile_size=48 "
            "(block_size when unset), expected a multiple of 32"
        ),
    ),
    _Refusal(
        rule="32x32 head steps",
        change={"head_size": 72},
        py_reason=(
            "use_mfma_32x32 requires whole 16-element head steps - head_size=72, "
            "expected a multiple of 16"
        ),
    ),
    # Python runs these rules before the register-P ones; so must C++.
    _Refusal(
        rule="32x32 rules before register-P",
        change={"use_register_pv": True, "block_m_per_warp": 16},
        py_reason=(
            "use_mfma_32x32 requires one 32-row MFMA atom per warp - "
            "block_m_per_warp=16, expected 32"
        ),
    ),
    _Refusal(
        rule="transposed QK needs the 32x32 atom",
        change={"use_mfma_32x32": False},
        py_reason=(
            "use_transposed_qk_32x32 requires the 32x32 MFMA atom - "
            "use_mfma_32x32=False, expected True"
        ),
    ),
]

_GKV2_RULES = [
    _Refusal(
        rule="grouped-KV2 bf16 only",
        change={"dtype": "fp16"},
        py_reason=(
            "use_grouped_kv2_softmax v1 is restricted to bf16 - dtype='fp16', "
            "expected 'bf16'"
        ),
    ),
    _Refusal(
        rule="grouped-KV2 transposed 32x32",
        change={"use_transposed_qk_32x32": False},
        py_reason=(
            "use_grouped_kv2_softmax requires the transposed 32x32 R4 path - "
            "use_mfma_32x32=True, use_transposed_qk_32x32=False, expected both True"
        ),
    ),
    _Refusal(
        rule="grouped-KV2 sliding window",
        change={"sliding_window": 64},
        py_reason=(
            "use_grouped_kv2_softmax v1 does not support a sliding window - "
            "sliding_window=64, expected 0"
        ),
    ),
    _Refusal(
        rule="grouped-KV2 softcap",
        change={"has_softcap": True},
        py_reason=(
            "use_grouped_kv2_softmax v1 does not support softcap - "
            "has_softcap=True, expected False"
        ),
    ),
    _Refusal(
        rule="grouped-KV2 ALiBi",
        change={"use_alibi": True},
        py_reason=(
            "use_grouped_kv2_softmax v1 does not support ALiBi - use_alibi=True, "
            "expected False"
        ),
        fastkv_binding_carries=False,
    ),
    _Refusal(
        rule="grouped-KV2 QQ bias",
        change={"use_qq_bias": True},
        py_reason=(
            "use_grouped_kv2_softmax v1 does not support QQ bias - "
            "use_qq_bias=True, expected False"
        ),
        fastkv_binding_carries=False,
    ),
    _Refusal(
        rule="grouped-KV2 fp8 K/V",
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


_VALU_R4_S1MASK = _Refusal(
    rule="VALU opts R4_s1mask path",
    change={"use_transposed_mask_once": False},
    py_reason=(
        "transposed softmax VALU opts require the R4_s1mask path - "
        "use_transposed_qk_32x32=True, use_transposed_scalar_state=True, "
        "use_transposed_mask_once=False, expected all True"
    ),
)
_VALU_RULES = [
    _VALU_R4_S1MASK,
    _Refusal(
        rule="VALU opts sliding window",
        change={"sliding_window": 64},
        py_reason=(
            "transposed softmax VALU opts do not support a sliding window - "
            "sliding_window=64, expected 0"
        ),
    ),
    _Refusal(
        rule="VALU opts softcap",
        change={"has_softcap": True},
        py_reason=(
            "transposed softmax VALU opts do not support softcap - "
            "has_softcap=True, expected False"
        ),
    ),
    _Refusal(
        rule="VALU opts ALiBi",
        change={"use_alibi": True},
        py_reason=(
            "transposed softmax VALU opts do not support ALiBi - "
            "use_alibi=True, expected False"
        ),
    ),
    _Refusal(
        rule="VALU opts QQ bias",
        change={"use_qq_bias": True},
        py_reason=(
            "transposed softmax VALU opts do not support QQ bias - "
            "use_qq_bias=True, expected False"
        ),
    ),
]


_VALU_GKV2 = _Refusal(
    rule="VALU opts grouped-KV2",
    change={"use_grouped_kv2_softmax": True},
    py_reason=(
        "transposed softmax VALU opts do not support grouped-KV2 softmax - "
        "use_grouped_kv2_softmax=True, expected False"
    ),
)


def _sub_knob_rule(field: str, fastkv_binding_carries: bool = True) -> _Refusal:
    """A transposed sub-knob turned on without the transposed QK."""
    return _Refusal(
        rule=f"{field} without the transposed QK",
        change={"use_transposed_qk_32x32": False, field: True},
        py_reason=(
            f"{field} requires use_transposed_qk_32x32 - "
            "use_transposed_qk_32x32=False, expected True"
        ),
        fastkv_binding_carries=fastkv_binding_carries,
    )


# gfx950 rules after the 32x32 ones, in the Python order. The fastKV binding
# does not pass use_transposed_invariant_hoist.
_GFX950_SUB_KNOB_RULES = [
    _sub_knob_rule("use_transposed_scalar_state"),
    _sub_knob_rule("use_transposed_invariant_hoist", fastkv_binding_carries=False),
    _sub_knob_rule("use_transposed_mask_once"),
    _sub_knob_rule("use_transposed_half_local_pv"),
    _Refusal(
        rule="skip-legacy-qreg without the 32x32 atom",
        change={
            "use_mfma_32x32": False,
            "use_transposed_qk_32x32": False,
            "use_mfma32_skip_legacy_qreg": True,
        },
        py_reason=(
            "use_mfma32_skip_legacy_qreg requires use_mfma_32x32 - "
            "use_mfma_32x32=False, expected True"
        ),
    ),
]
# The gfx942 spec refuses use_transposed_half_local_pv and the skip-legacy knob
# outright, so only these sub-knob rules reach the shared rule there.
_GFX942_SUB_KNOB_RULES = [
    _sub_knob_rule("use_transposed_scalar_state"),
    _sub_knob_rule("use_transposed_invariant_hoist"),
    _sub_knob_rule("use_transposed_mask_once"),
]
# Python runs the VALU opts rule before the fast descriptor's K/V read rule.
_GFX942_VALU_BEFORE_FAST_DESCRIPTOR = _Refusal(
    rule="VALU opts before the fast descriptor's K read",
    change={
        "use_fast_paged_kv_desc": True,
        "use_k_hbm_direct": True,
        "use_transposed_mask_once": False,
    },
    py_reason=_VALU_R4_S1MASK.py_reason,
)


class _Entry(NamedTuple):
    name: str
    spec_cls: type
    accepted: dict
    binding: str
    archs: tuple[str, ...]
    rules: list[_Refusal]


# The gfx942 builder admits gfx942 and gfx950; the gfx950 builders admit gfx950.
# The fastKV binding does not pass the VALU opts knob, so fastKV has no VALU rows.
_ENTRIES = [
    _Entry(
        "gfx950 tiled 2D",
        Gfx950Spec2D,
        _GFX950_GKV2,
        "gfx950_attention_tiled_2d",
        ("gfx950",),
        _GFX950_32X32_RULES + _GFX950_SUB_KNOB_RULES + _GKV2_RULES,
    ),
    _Entry(
        "gfx950 tiled 2D, VALU opts",
        Gfx950Spec2D,
        _GFX950_VALU,
        "gfx950_attention_tiled_2d",
        ("gfx950",),
        [*_VALU_RULES, _VALU_GKV2],
    ),
    _Entry(
        "gfx950 fastKV register-P",
        Gfx950Spec2D,
        _FASTKV_GKV2,
        "gfx950_attention_tiled_2d_fastkv_regp",
        ("gfx950",),
        [
            refusal
            for refusal in _GFX950_32X32_RULES + _GFX950_SUB_KNOB_RULES + _GKV2_RULES
            if refusal.fastkv_binding_carries
        ],
    ),
    # The gfx942 spec refuses grouped-KV2 before its VALU rules, so no gfx942
    # grouped-KV2 VALU row.
    _Entry(
        "gfx942 tiled 2D, VALU opts",
        Gfx942Spec2D,
        _GFX942_VALU,
        "gfx942_attention_tiled_2d",
        ("gfx942", "gfx950"),
        [*_VALU_RULES, _GFX942_VALU_BEFORE_FAST_DESCRIPTOR],
    ),
    _Entry(
        "gfx942 tiled 2D",
        Gfx942Spec2D,
        _PROBLEM_64_8,
        "gfx942_attention_tiled_2d",
        ("gfx942", "gfx950"),
        _GFX942_SUB_KNOB_RULES,
    ),
]
# Each binding entry point builds through the same C++ entry.
_ENTRY_POINTS = ("lower_llvm", "serialize_ir", "verify")


@pytest.mark.parametrize(
    "spec_cls, accepted, binding, arch",
    [
        pytest.param(e.spec_cls, e.accepted, e.binding, arch, id=f"{e.name} on {arch}")
        for e in _ENTRIES
        for arch in e.archs
    ],
)
def test_2d_entries_build_the_accepted_specs(
    rocke_engine, spec_cls, accepted, binding, arch
):
    """The C++ entries build each accepted spec, so a refusal below is the rule's."""
    fields = dataclasses.asdict(spec_cls(**accepted))
    assert getattr(rocke_engine, f"{binding}_serialize_ir")(fields, arch)


def _assert_both_engines_refuse(
    rocke_engine, spec_cls, accepted, entry_point, arch, refusal
):
    # The C++ message is matched as a substring: the binding prefixes it with
    # the entry's "build failed:" context.
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
        for e in _ENTRIES
        for arch in e.archs
        for entry_point in _ENTRY_POINTS
        for refusal in e.rules
    ],
)
def test_2d_entries_refuse_the_spec_rules_like_python(
    rocke_engine, spec_cls, accepted, entry_point, arch, refusal
):
    """Python and the C++ entry refuse the changed spec with the same text."""
    _assert_both_engines_refuse(
        rocke_engine, spec_cls, accepted, entry_point, arch, refusal
    )


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
    rocke_engine, entry_point, arch, refusal
):
    """The gfx942 entry runs the gfx942 spec checks for every target, so the
    grouped-KV2 and early-V specs the gfx942 spec refuses are refused on both
    arches its builder admits."""
    _assert_both_engines_refuse(
        rocke_engine, Gfx942Spec2D, _PROBLEM_64_8, entry_point, arch, refusal
    )


@pytest.mark.parametrize(
    "accepted, change, reason",
    [
        pytest.param(
            _GFX950_GKV2,
            {"use_additive_bias": True},
            "use_grouped_kv2_softmax v1 does not support additive bias - "
            "use_additive_bias=True, expected False",
            id="grouped-KV2 additive bias",
        ),
        pytest.param(
            _GFX950_GKV2,
            {"use_additive_bias": True, "has_softcap": True},
            "use_grouped_kv2_softmax v1 does not support softcap - has_softcap=True, "
            "expected False",
            id="grouped-KV2 additive bias after the shared rules",
        ),
        pytest.param(
            _GFX950_VALU,
            {"use_additive_bias": True},
            "transposed softmax VALU opts do not support additive bias - "
            "use_additive_bias=True, expected False",
            id="VALU opts additive bias",
        ),
        pytest.param(
            _GFX950_VALU,
            {"use_additive_bias": True, "use_alibi": True},
            "transposed softmax VALU opts do not support ALiBi - use_alibi=True, "
            "expected False",
            id="VALU opts additive bias after the shared rules",
        ),
    ],
)
def test_gfx950_2d_spec_refuses_additive_bias_with_shared_rule_knobs(
    accepted, change, reason
):
    """Python alone refuses additive bias here, after the shared rules.

    The C++ spec has no additive-bias field.
    """
    with pytest.raises(ValueError) as py_err:
        dataclasses.replace(Gfx950Spec2D(**accepted), **change)
    assert str(py_err.value) == reason
