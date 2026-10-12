# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The dense attention backward tuning registry (CPU).

Every dense spec tuning field has one axis whose default is the field's
default; not-built values are registered and refused by the kernel; every
built value of every dense axis builds a legal spec on its arch; the body
knobs are the general registry's axes; the shipped tuned table is empty.
"""

from __future__ import annotations

import dataclasses

import pytest

from dispatch.attention_bwd import (
    DENSE_BWD_KNOB_AXES,
    DENSE_BWD_TUNED,
    BWD_KNOB_AXES,
    dense_axis,
    dense_body_axes,
    dense_runtime_axes,
    dense_spec_legality,
    dense_tuning_axes,
)
from dispatch.attention_bwd.tuning_common import KIND_RUNTIME
from kernels.common.attention_dense_bwd_spec import DENSE_BODY_KNOBS
from kernels.gfx942.attention_dense_bwd import Gfx942AttentionDenseBwdSpec
from kernels.gfx950.attention_dense_bwd import Gfx950AttentionDenseBwdSpec

_SPEC = {"gfx942": Gfx942AttentionDenseBwdSpec, "gfx950": Gfx950AttentionDenseBwdSpec}
_BASE = dict(
    seqlen_q=256, seqlen_kv=256, num_query_heads=8, num_kv_heads=2, head_size=128
)


def test_defaults_match_the_spec():
    fields = {
        f.name: f.default for f in dataclasses.fields(Gfx942AttentionDenseBwdSpec)
    }
    for ax in DENSE_BWD_KNOB_AXES:
        if ax.kind == KIND_RUNTIME:
            continue
        assert ax.default == fields[ax.name], ax.name


def test_body_axes_are_the_general_axes():
    names = {ax.name for ax in dense_body_axes()}
    assert names == set(DENSE_BODY_KNOBS)
    general = {ax.name: ax for ax in BWD_KNOB_AXES}
    for ax in dense_body_axes():
        assert ax is general[ax.name]
    assert dense_axis("waves") is general["waves"]
    assert dense_axis("oob").name == "oob"
    with pytest.raises(KeyError):
        dense_axis("head_size")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_every_built_value_is_legal_and_not_built_values_refused(arch):
    cls = _SPEC[arch]
    for ax in DENSE_BWD_KNOB_AXES:
        if ax.kind == KIND_RUNTIME or ax.plan_owned:
            continue
        for v in ax.values(arch, "full"):
            kw = dict(_BASE)
            kw[ax.name] = v
            if ax.name == "persist_decode" and v != "kv_tile_major":
                kw["persistent"] = True
            try:
                spec = cls(**kw)
            except ValueError:
                assert v in ax.not_built, (ax.name, v)
                continue
            got = dense_spec_legality(spec, arch)
            want = "not_built" if v in ax.not_built else "legal"
            assert got == want, (ax.name, v, got)
    spec = cls(**_BASE, dkv_mode="atomic")
    assert dense_spec_legality(spec, arch) == "legal"


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_arch_values_of_the_body_variant(arch):
    vals = dense_axis("body_variant").values(arch)
    assert vals[0] == "general" and len(vals) == 2
    assert all(v in _SPEC[arch].BODY_VARIANTS for v in vals)


def test_tuning_axes_and_table():
    names = {ax.name for ax in dense_tuning_axes("gfx942", "full")}
    assert {"oob", "epilogue", "persistent", "waves", "block_n"} <= names
    assert {"g_split", "lb_order", "xcd_chunk", "scale_placement"} <= names
    assert dict(DENSE_BWD_TUNED) == {}


def test_runtime_axes_are_the_general_runtime_axes():
    from kernels.common.attention_bwd_plan import AUX_KNOBS, RUNTIME_KNOBS

    general = {ax.name: ax for ax in BWD_KNOB_AXES}
    got = {ax.name for ax in dense_runtime_axes()}
    assert got == (set(RUNTIME_KNOBS) - {"use_worklist"}) | set(AUX_KNOBS)
    for ax in dense_runtime_axes():
        assert ax is general[ax.name]
        assert ax.kind == KIND_RUNTIME
    assert dense_axis("lb_order") is general["lb_order"]
    with pytest.raises(KeyError):
        dense_axis("use_worklist")
    # the dense remap is derived by the plan, like the general one
    assert dense_axis("xcd_remap").plan_owned
