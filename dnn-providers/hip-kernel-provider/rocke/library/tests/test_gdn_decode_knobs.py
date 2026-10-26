# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Performance knobs of the gfx950 GDN/KDA decode emitter, without a GPU.

Covers the ``spec -> IR`` contract of the ten knobs (ALGORITHM.md §4.10):

* the validator rejects every non-bool flag, every unknown hint and every
  out-of-range ``waves_per_eu``, and each dependency rule;
* a tile knob that is inert on a tile (its ``*_applies`` predicate is False)
  emits the knob-off kernel under the knob-off name, and one that applies
  changes both, so the name stays a faithful cache key;
* names stay injective across knob combinations;
* the hints, ``waves_per_eu`` and ``dpp_reduce`` change the IR in the way
  their docs claim.

The numerics of every knob-on path are covered on a gfx950 device by
``test_gdn_decode_knobs_gfx950_numeric.py``; the knob-off IR is pinned by the
golden test. Lowering needs no device and no comgr.
"""

from __future__ import annotations

import dataclasses as dc
import itertools

import pytest

from kernels.gfx950.gdn_decode import (
    STATE_HINTS,
    GdnDecodeSpec,
    build_gdn_decode,
    dpp_reduce_applies,
    interleave_cols_applies,
    is_valid_spec,
    stream_rows_applies,
    xcd_remap_applies,
)

ARCH = "gfx950"

BOOL_KNOBS = (
    "dpp_reduce",
    "xcd_remap",
    "stream_rows",
    "interleave_cols",
    "conv_once",
    "norm_gate_once",
    "out_lds",
)
# One workgroup per head, Hk == Hv: the tile every fused mode accepts.
_FUSED = dict(
    num_k_heads=16, num_v_heads=16, num_warps=4, warp_threads_k=16, blocks_per_v_dim=1
)


def _spec(**kw) -> GdnDecodeSpec:
    return dc.replace(GdnDecodeSpec(), **kw)


def _lower(spec: GdnDecodeSpec) -> str:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(
        build_gdn_decode(spec, arch=ARCH), arch=ARCH, llvm_flavor="llvm20"
    )


def _lower_modulo_name(spec: GdnDecodeSpec) -> str:
    """The spec's IR with its kernel name replaced by ``K``."""
    return _lower(spec).replace(spec.kernel_name(), "K")


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("field", BOOL_KNOBS)
@pytest.mark.parametrize("value", ["false", "True", 1, 0, None])
def test_non_bool_flag_is_rejected(field, value):
    # A truthy string must not select the knob-on kernel, nor 0 the knob-off
    # one; every dependency is met so only the type can fail.
    knobs = {"conv_once": True, field: value}
    spec = _spec(**_FUSED, fuse_conv=True, fuse_out_norm=True, **knobs)
    ok, why = is_valid_spec(spec, arch=ARCH)
    assert not ok
    assert f"{field} must be a bool" in why


@pytest.mark.parametrize("field", BOOL_KNOBS)
@pytest.mark.parametrize("value", ["false", "True", 1, 0, None])
def test_launcher_cache_hit_does_not_bypass_flag_validation(field, value, monkeypatch):
    # A malformed flag shares its kernel name with the valid spec it imitates
    # ("false" and 1 with the knob on, 0 and None with it off). Warm the cache
    # under both names: ``launcher_for`` must still raise, never hand back the
    # cached launcher.
    pytest.importorskip("torch", reason="the builder module imports torch")
    from builders.gfx950.gdn import gdn_decode as builder

    def spec_for(flag):
        # f32 state: ``interleave_cols`` is inert (and unnamed) with a 2-byte one.
        deps = {"conv_once": True, field: flag}
        return _spec(
            **_FUSED, state_dtype="f32", fuse_conv=True, fuse_out_norm=True, **deps
        )

    cached = object()
    cache = {(spec_for(flag).kernel_name(), ARCH): cached for flag in (True, False)}
    assert len(cache) == 2  # the knob changes the name, so both keys are distinct
    monkeypatch.setattr(builder, "_LAUNCHER_CACHE", cache)

    with pytest.raises(ValueError, match=f"{field} must be a bool"):
        builder.launcher_for(spec_for(value), arch=ARCH)
    # the valid specs keep hitting the cache
    assert builder.launcher_for(spec_for(True), arch=ARCH) is cached
    assert builder.launcher_for(spec_for(False), arch=ARCH) is cached


@pytest.mark.parametrize("field", ["state_load_hint", "state_store_hint"])
@pytest.mark.parametrize("value", ["nontemporal", "STREAMING", "", None, True])
def test_unknown_state_hint_is_rejected(field, value):
    ok, why = is_valid_spec(_spec(**{field: value}), arch=ARCH)
    assert not ok
    assert field in why and "must be one of" in why


@pytest.mark.parametrize("field", ["state_load_hint", "state_store_hint"])
@pytest.mark.parametrize("value", STATE_HINTS)
def test_every_state_hint_is_accepted(field, value):
    for simple in (False, True):
        ok, why = is_valid_spec(_spec(simple=simple, **{field: value}), arch=ARCH)
        assert ok, why


@pytest.mark.parametrize("value", [-1, 9, True, False, 4.0, "4", None])
def test_invalid_waves_per_eu_is_rejected(value):
    ok, why = is_valid_spec(_spec(waves_per_eu=value), arch=ARCH)
    assert not ok
    assert "waves_per_eu" in why


@pytest.mark.parametrize("value", range(9))
def test_waves_per_eu_range_is_accepted(value):
    ok, why = is_valid_spec(_spec(waves_per_eu=value), arch=ARCH)
    assert ok, why


@pytest.mark.parametrize(
    "fused, knobs, reason",
    [
        # conv_once has no conv to compute once
        (dict(fuse_out_norm=True), dict(conv_once=True), "conv_once requires"),
        (dict(), dict(conv_once=True), "conv_once requires"),
        # norm_gate_once reads the norm's out_gate / norm_weight; its
        # products live in conv_once's LDS block (an emitter coupling)
        (
            dict(fuse_conv=True, fuse_out_norm=True),
            dict(norm_gate_once=True),
            "NOT_YET_IMPLEMENTED: norm_gate_once requires conv_once",
        ),
        (
            dict(fuse_out_norm=True),
            dict(norm_gate_once=True),
            "NOT_YET_IMPLEMENTED: norm_gate_once requires conv_once",
        ),
        (
            dict(fuse_conv=True),
            dict(conv_once=True, norm_gate_once=True),
            "norm_gate_once requires fuse_out_norm",
        ),
        # out_lds reorganizes the fused norm's epilogue
        (dict(fuse_conv=True), dict(out_lds=True), "out_lds requires"),
        (dict(), dict(out_lds=True), "out_lds requires"),
    ],
    ids=[
        "conv_once_rn_only",
        "conv_once_unfused",
        "norm_gate_once_without_conv_once",
        "norm_gate_once_rn_only",
        "norm_gate_once_without_norm",
        "out_lds_cv_only",
        "out_lds_unfused",
    ],
)
def test_knob_dependency_is_enforced(fused, knobs, reason):
    ok, why = is_valid_spec(_spec(**_FUSED, **fused, **knobs), arch=ARCH)
    assert not ok
    assert reason in why


@pytest.mark.parametrize(
    "fused, knobs",
    [
        (dict(fuse_conv=True), dict(conv_once=True)),
        (dict(fuse_conv=True, fuse_out_norm=True), dict(conv_once=True)),
        (
            dict(fuse_conv=True, fuse_out_norm=True),
            dict(conv_once=True, norm_gate_once=True),
        ),
        (dict(fuse_out_norm=True), dict(out_lds=True)),
        (
            dict(fuse_conv=True, fuse_out_norm=True),
            dict(conv_once=True, norm_gate_once=True, out_lds=True),
        ),
    ],
)
def test_fused_knob_with_its_dependencies_is_accepted(fused, knobs):
    for gate in ("gdn", "kda"):
        ok, why = is_valid_spec(
            _spec(**_FUSED, gate_kind=gate, **fused, **knobs), arch=ARCH
        )
        assert ok, why


# ------------------------------------------------- applies predicates / names

# (knob, spec with the knob off, predicate, expected predicate value). The
# tile knobs are legal everywhere; where the predicate is False they are inert.
_TILE_CASES = [
    (
        "dpp_reduce",
        _spec(warp_threads_k=4, blocks_per_v_dim=4),
        dpp_reduce_applies,
        False,
    ),
    (
        "dpp_reduce",
        _spec(warp_threads_k=8, num_warps=4, blocks_per_v_dim=4),
        dpp_reduce_applies,
        True,
    ),
    ("dpp_reduce", _spec(), dpp_reduce_applies, True),
    ("dpp_reduce", _spec(simple=True), dpp_reduce_applies, False),
    ("xcd_remap", _spec(), xcd_remap_applies, True),
    ("xcd_remap", _spec(simple=True), xcd_remap_applies, False),
    # (2, 16, 8): 8 value lanes over a 16-row tile -> 2 rows per lane
    ("stream_rows", _spec(), stream_rows_applies, True),
    # (2, 16, 16): 8 value lanes over an 8-row tile -> 1 row per lane
    ("stream_rows", _spec(blocks_per_v_dim=16), stream_rows_applies, False),
    ("stream_rows", _spec(simple=True), stream_rows_applies, False),
    ("interleave_cols", _spec(state_dtype="f32"), interleave_cols_applies, True),
    (
        "interleave_cols",
        _spec(state_dtype="f32", warp_threads_k=2, num_warps=1, blocks_per_v_dim=4),
        interleave_cols_applies,
        True,
    ),
    (
        "interleave_cols",
        _spec(state_dtype="f32", warp_threads_k=1, num_warps=1, blocks_per_v_dim=2),
        interleave_cols_applies,
        False,
    ),
    ("interleave_cols", _spec(), interleave_cols_applies, False),
    (
        "interleave_cols",
        _spec(state_dtype="f32", simple=True),
        interleave_cols_applies,
        False,
    ),
    (
        "interleave_cols",
        _spec(**_FUSED, gate_kind="kda", state_dtype="f32", fuse_conv=True),
        interleave_cols_applies,
        True,
    ),
    (
        "stream_rows",
        _spec(**_FUSED, gate_kind="kda", fuse_conv=True, fuse_out_norm=True),
        stream_rows_applies,
        True,
    ),
]


@pytest.mark.parametrize(
    "knob, off, applies, expected",
    _TILE_CASES,
    ids=[f"{c[0]}-{c[1].kernel_name()}" for c in _TILE_CASES],
)
def test_tile_knob_name_and_ir_follow_its_predicate(knob, off, applies, expected):
    on = dc.replace(off, **{knob: True})
    assert is_valid_spec(on, arch=ARCH)[0], is_valid_spec(on, arch=ARCH)[1]
    assert applies(on) is expected
    assert getattr(on, f"{knob}_on") is expected
    # The name is inside the IR, so compare the IR with it normalized out: a
    # predicate that admits the knob while the emitter adds nothing fails.
    same_ir = _lower_modulo_name(on) == _lower_modulo_name(off)
    same_name = on.kernel_name() == off.kernel_name()
    # Inert: the knob-off kernel under the knob-off name (one cache entry for
    # one kernel). Applied: a new kernel under a new name.
    assert same_ir is (not expected), (knob, on.kernel_name())
    assert same_name is (not expected), (knob, on.kernel_name())


def _knob_variants():
    """Every knob alone where it applies, and the fused combinations."""
    plain = _spec()
    f32 = _spec(state_dtype="f32")
    fused = _spec(**_FUSED, fuse_conv=True, fuse_out_norm=True)
    rn = _spec(**_FUSED, fuse_out_norm=True)
    out = {
        "plain": plain,
        "f32": f32,
        "fused": fused,
        "rn": rn,
        "lh": dc.replace(plain, state_load_hint="streaming"),
        "sh": dc.replace(plain, state_store_hint="streaming"),
        "lh_sh": dc.replace(
            plain, state_load_hint="streaming", state_store_hint="streaming"
        ),
        "wpe1": dc.replace(plain, waves_per_eu=1),
        "wpe4": dc.replace(plain, waves_per_eu=4),
        "dppr": dc.replace(plain, dpp_reduce=True),
        "xcd": dc.replace(plain, xcd_remap=True),
        "sr": dc.replace(plain, stream_rows=True),
        "ic": dc.replace(f32, interleave_cols=True),
        "co": dc.replace(fused, conv_once=True),
        "co_ngo": dc.replace(fused, conv_once=True, norm_gate_once=True),
        "olds": dc.replace(fused, out_lds=True),
        "rn_olds": dc.replace(rn, out_lds=True),
        "co_olds": dc.replace(fused, conv_once=True, out_lds=True),
        "co_ngo_olds": dc.replace(
            fused, conv_once=True, norm_gate_once=True, out_lds=True
        ),
        "simple_lh": _spec(simple=True, state_load_hint="streaming"),
        "simple_wpe": _spec(simple=True, waves_per_eu=2),
    }
    tile = ("dpp_reduce", "xcd_remap", "stream_rows")
    for r in range(2, len(tile) + 1):
        for combo in itertools.combinations(tile, r):
            out["+".join(combo)] = dc.replace(plain, **{k: True for k in combo})
    return out


def test_knob_names_are_injective():
    names = {}
    for label, spec in _knob_variants().items():
        assert is_valid_spec(spec, arch=ARCH)[0], (label, is_valid_spec(spec)[1])
        name = spec.kernel_name()
        assert name not in names, (
            f"{label!r} collides with {names.get(name)!r} on {name!r}; two "
            "different kernels would share one cache entry"
        )
        names[name] = label


# ------------------------------------------------------------ emitted effect


def _nontemporal(ir: str, op: str) -> int:
    return sum(
        1 for line in ir.splitlines() if f" {op} " in line and "!nontemporal" in line
    )


@pytest.mark.parametrize("simple", [False, True])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_state_hints_mark_only_their_own_accesses(simple, state_dtype):
    base = _spec(simple=simple, state_dtype=state_dtype)
    off = _lower(base)
    assert _nontemporal(off, "load") == 0 and _nontemporal(off, "store") == 0
    loads = _lower(dc.replace(base, state_load_hint="streaming"))
    stores = _lower(dc.replace(base, state_store_hint="streaming"))
    # A swapped or dropped hint fails here: each knob reaches only its side.
    assert _nontemporal(loads, "load") > 0 and _nontemporal(loads, "store") == 0
    assert _nontemporal(stores, "store") > 0 and _nontemporal(stores, "load") == 0


@pytest.mark.parametrize("simple", [False, True])
def test_waves_per_eu_sets_the_occupancy_floor(simple):
    assert "amdgpu-waves-per-eu" not in _lower(_spec(simple=simple))
    ir = _lower(_spec(simple=simple, waves_per_eu=4))
    assert '"amdgpu-waves-per-eu"="4,8"' in ir


@pytest.mark.parametrize(
    "wtk, dk, swizzles_left", [(8, 128, False), (16, 128, False), (32, 256, True)]
)
def test_dpp_reduce_replaces_the_xor4_and_xor8_swizzles(wtk, dk, swizzles_left):
    # WTK 8 / 16 reduce over offsets up to 4 / 8, all of which move to DPP;
    # WTK 32 (legal from head_k_dim 256) keeps ds_swizzle for its xor 16.
    base = _spec(warp_threads_k=wtk, head_k_dim=dk, num_warps=1, blocks_per_v_dim=2)
    assert is_valid_spec(base, arch=ARCH)[0]
    assert "ds.swizzle" in _lower(base)
    assert ("ds.swizzle" in _lower(dc.replace(base, dpp_reduce=True))) is swizzles_left
