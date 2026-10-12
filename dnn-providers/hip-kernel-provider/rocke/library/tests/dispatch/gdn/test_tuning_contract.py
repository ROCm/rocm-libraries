# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx950 GDN/KDA decode tuned candidates keep the shared tuning contract.

:func:`rocke.dispatch.tuning.testing.assert_tuning_contract` checks pin
resolution, id uniqueness, replay and portability for representative tiles in
every mode. The rest pins what is specific to this family: the knob space each
mode exposes, that it holds one kernel per configuration, that ``auto`` is
each tile's base configuration, and that out-of-scope knobs are dropped while
illegal ones are refused.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from dispatch.gdn import (
    GDN_REGISTRY,
    GdnDecodeRequest,
    dispatch_gdn_decode,
    dispatch_gdn_decode_all,
    request_errors,
)
from dispatch.gdn.gfx950 import ARCH
from rocke.dispatch.core import PinRefused
from rocke.dispatch.tuning import sweep_level
from rocke.dispatch.tuning.testing import assert_tuning_contract, representative

_FUSED = dict(fuse_conv=True, fuse_out_norm=True)
_REFUSED = (
    {"state_load_hint": "nontemporal"},
    {"waves_per_eu": 9},
    {"waves_per_eu": "4"},
    # Never read as 1, unset or 4: a pin is not rewritten into another kernel.
    {"waves_per_eu": True},
    {"waves_per_eu": None},
    {"waves_per_eu": 4.0},
    {"dpp_reduce": "true"},
    {"num_warps": 4},
    {"fuse_conv": True},
    {"not_a_knob": True},
)


def _req(batch: int = 64, heads: int = 32, **kw) -> GdnDecodeRequest:
    return GdnDecodeRequest(
        batch=batch, arch=ARCH, num_k_heads=heads, num_v_heads=heads, **kw
    )


# (candidate name, request, other problem of the same mode, knobs restating
# the base configuration's values)
_CASES = (
    (
        "gdn_decode_gfx950_nw2_wtk16_bpv8",
        _req(heads=16, gate_kind="gdn"),
        _req(1, heads=4, gate_kind="gdn"),
        {"state_load_hint": "default", "dpp_reduce": False, "waves_per_eu": 0},
    ),
    (
        "gdn_decode_gfx950_kda_nw1_wtk16_bpv16",
        _req(gate_kind="kda", state_dtype="f32"),
        _req(3, heads=8, gate_kind="kda", state_dtype="f32"),
        {"xcd_remap": False, "state_store_hint": "default", "interleave_cols": False},
    ),
    (
        "gdn_decode_gfx950_kda_nw2_wtk4_bpv2",
        _req(gate_kind="kda"),
        _req(7, heads=16, gate_kind="kda"),
        # No xor-4 stage on this tile and a 2-byte state: dpp_reduce and
        # interleave_cols do not change its code, so they are out of scope.
        {"dpp_reduce": True, "interleave_cols": True},
    ),
    (
        "gdn_decode_gfx950_kda_nw4_wtk8_bpv1",
        _req(gate_kind="kda", state_dtype="f32", **_FUSED),
        _req(5, heads=8, gate_kind="kda", state_dtype="f32", **_FUSED),
        {"conv_once": False, "norm_gate_once": False, "out_lds": False},
    ),
    (
        "gdn_decode_gfx950_nw4_wtk16_bpv1",
        _req(gate_kind="gdn", state_dtype="bf16", fuse_conv=True),
        _req(2, heads=16, gate_kind="gdn", state_dtype="bf16", fuse_conv=True),
        # norm_gate_once and out_lds need fuse_out_norm: out of scope, dropped.
        {"conv_once": False, "norm_gate_once": True, "out_lds": True},
    ),
)


@pytest.mark.parametrize("name,req,other,defaults", _CASES, ids=[c[0] for c in _CASES])
def test_representative_candidates_keep_the_tuning_contract(name, req, other, defaults):
    (candidate,) = representative(GDN_REGISTRY.candidates(), (name,))
    counts = assert_tuning_contract(
        candidate,
        [req],
        other_requests=[other],
        default_knobs=defaults,
        refused_knobs=_REFUSED,
    )
    assert counts["replays"] > 0
    assert counts["portable"] > 0


def _expected_configs_per_tile(req: GdnDecodeRequest, kernel) -> int:
    """The knob space size from the scope rules, stated independently of
    ``rules.py``: two values per state hint, conv_once (fuse_conv) with
    norm_gate_once legal only under it (fuse_out_norm), dpp_reduce where the
    in-group sum has an xor-4 stage, xcd_remap, stream_rows where a lane owns
    several state rows, interleave_cols with an f32 state and more than one K
    lane, out_lds with fuse_out_norm, and the two production waves_per_eu
    values (0 and 4)."""
    hints = 2 * 2
    if req.fuse_conv:
        conv = 3 if req.fuse_out_norm else 2
    else:
        conv = 1
    dpp = 2 if kernel.warp_threads_k >= 8 else 1
    lanes_v = (kernel.wave_size // kernel.warp_threads_k) * kernel.num_warps
    rows = (kernel.head_v_dim // kernel.blocks_per_v_dim) // lanes_v
    stream = 2 if rows > 1 else 1
    ic = 2 if req.state_dtype == "f32" and kernel.warp_threads_k > 1 else 1
    olds = 2 if req.fuse_out_norm else 1
    xcd = 2
    return hints * conv * dpp * xcd * stream * ic * olds * 2


_MODES = (
    _req(gate_kind="gdn"),
    _req(gate_kind="kda", state_dtype="f32"),
    _req(gate_kind="kda", state_dtype="f32", **_FUSED),
    _req(gate_kind="gdn", state_dtype="bf16", fuse_conv=True),
    _req(gate_kind="kda", state_dtype="bf16", fuse_out_norm=True),
)


@pytest.mark.parametrize(
    "req",
    _MODES,
    ids=("gdn_bf16", "kda_f32", "kda_f32_fused", "gdn_conv_only", "kda_norm_only"),
)
def test_every_canonical_configuration_is_registered_once(req):
    results = dispatch_gdn_decode_all(req)
    by_tile: dict = {}
    for result in results:
        by_tile.setdefault(result.candidate.spec_id, []).append(result)
    assert by_tile
    # Tiles come in registry priority order, each tile's rows together.
    order = [c.spec_id for c in GDN_REGISTRY.candidates() if c.spec_id in by_tile]
    assert list(by_tile) == order
    assert [r.candidate.spec_id for r in results] == [
        sid for sid in by_tile for _ in by_tile[sid]
    ]
    for spec_id, rows in by_tile.items():
        kernel = rows[0].spec.kernel_spec
        assert len(rows) == _expected_configs_per_tile(req, kernel), spec_id
        # The tile's base configuration (no knobs) leads its rows and is what
        # an unknobbed pin of the tile selects.
        assert rows[0].spec.knobs == (), spec_id
        assert sum(1 for row in rows if not row.spec.knobs) == 1, spec_id
        pinned = dispatch_gdn_decode(replace(req, spec_id=spec_id))
        assert rows[0].spec == pinned.spec, spec_id
        # Every row replays from the request it carries.
        for row in rows[:: max(1, len(rows) // 7)]:
            assert dispatch_gdn_decode(row.request).spec == row.spec
    # One kernel per configuration: no two rows compile to the same kernel.
    names = [result.spec.kernel_name() for result in results]
    assert len(names) == len(set(names))
    assert len({result.spec.tuning_id for result in results}) == len(results)
    assert len({result.kernel_id.compile_key for result in results}) == len(results)


@pytest.mark.parametrize("req", _MODES)
def test_auto_is_the_auto_tile_base_configuration(req):
    auto = dispatch_gdn_decode(req)
    assert auto.spec.knobs == ()
    assert auto.spec.tuning_id.startswith(f"{auto.candidate.spec_id}_wpe0@")
    kernel = auto.spec.kernel_spec
    for name in ("state_load_hint", "state_store_hint"):
        assert getattr(kernel, name) == "default"
    for name in (
        "dpp_reduce",
        "xcd_remap",
        "stream_rows",
        "interleave_cols",
        "conv_once",
        "norm_gate_once",
        "out_lds",
    ):
        assert getattr(kernel, name) is False, name
    assert kernel.waves_per_eu == 0
    # The result's request is pinned to what ran, so it replays it.
    assert (auto.request.spec_id, auto.request.tuning_id) == (
        auto.candidate.spec_id,
        auto.spec.tuning_id,
    )
    assert dispatch_gdn_decode(auto.request).spec == auto.spec


def test_tuning_id_and_knobs_round_trip():
    req = _req(gate_kind="kda", state_dtype="f32", **_FUSED)
    spec_id = "kda_nw4_wtk16_bpv1"
    knobs = {
        "conv_once": True,
        "norm_gate_once": True,
        "out_lds": True,
        "dpp_reduce": True,
        "interleave_cols": True,
        "state_load_hint": "streaming",
        "waves_per_eu": 4,
    }
    by_knobs = dispatch_gdn_decode(replace(req, spec_id=spec_id, tuning_knobs=knobs))
    assert dict(by_knobs.spec.knobs) == knobs
    kernel = by_knobs.spec.kernel_spec
    assert all(getattr(kernel, name) == value for name, value in knobs.items())
    assert by_knobs.spec.tuning_id.startswith(f"{spec_id}_wpe4@")
    # The stored (id, knobs) pair and the knobs as JSON gives them back (a
    # list of pairs) replay the same configuration on another problem.
    other = _req(3, heads=8, gate_kind="kda", state_dtype="f32", **_FUSED)
    for stored in (
        replace(
            other,
            spec_id=spec_id,
            tuning_id=by_knobs.spec.tuning_id,
            tuning_knobs=[list(pair) for pair in by_knobs.spec.knobs],
        ),
        replace(other, spec_id=spec_id, tuning_knobs=knobs),
    ):
        moved = dispatch_gdn_decode(stored)
        assert moved.spec.config_key == by_knobs.spec.config_key
        assert moved.spec.knobs == by_knobs.spec.knobs
    # A bare id of a production configuration resolves without its knobs.
    bare = dispatch_gdn_decode(
        replace(req, spec_id=spec_id, tuning_id=by_knobs.spec.tuning_id)
    )
    assert bare.spec == by_knobs.spec


def test_out_of_scope_knobs_are_dropped_not_refused():
    """A ``tuning_knobs`` pin that travels to a problem where a knob changes
    no code (no fused mode, a tile without the predicate's shape) keeps
    working: the knob is dropped, so the configuration is the base one."""
    plain = _req(gate_kind="gdn")
    for spec_id, knobs in (
        ("nw2_wtk16_bpv8", {"conv_once": True, "norm_gate_once": True}),
        ("nw2_wtk16_bpv8", {"out_lds": True}),
        ("nw2_wtk16_bpv8", {"interleave_cols": True}),  # 2-byte state
        ("nw1_wtk4_bpv4", {"dpp_reduce": True}),  # no xor-4 stage
        ("nw8_wtk16_bpv4", {"stream_rows": True}),  # one state row per lane
    ):
        got = dispatch_gdn_decode(replace(plain, spec_id=spec_id, tuning_knobs=knobs))
        base = dispatch_gdn_decode(replace(plain, spec_id=spec_id))
        assert got.spec == base.spec, (spec_id, knobs)
        assert got.spec.knobs == ()


@pytest.mark.parametrize(
    "spec_id,knobs,recorded_on,moved_to,dropped",
    (
        # stream_rows needs more than one state row per lane: DV 128 has 2 on
        # this tile, DV 64 has 1.
        (
            "nw4_wtk16_bpv4",
            {"stream_rows": True, "xcd_remap": True},
            _req(gate_kind="gdn"),
            _req(gate_kind="gdn", head_v_dim=64),
            "stream_rows",
        ),
        # interleave_cols needs an f32 state; the fused-only knobs need fusion.
        (
            "kda_nw4_wtk16_bpv1",
            {"interleave_cols": True, "conv_once": True, "dpp_reduce": True},
            _req(gate_kind="kda", state_dtype="f32", **_FUSED),
            _req(gate_kind="kda"),
            "conv_once",
        ),
    ),
    ids=("dv64_drops_stream_rows", "bf16_plain_drops_f32_fused_knobs"),
)
def test_a_stored_tuning_id_does_not_travel_where_scope_drops_a_knob(
    spec_id, knobs, recorded_on, moved_to, dropped
):
    """Only ``tuning_knobs`` travel across shapes. A stored (tuning_id,
    knobs) pair names the knob set canonical where it was recorded; where
    scope drops one of those knobs it names no configuration, and the
    refusal says so (not "stale, re-sweep")."""
    recorded = dispatch_gdn_decode(
        replace(recorded_on, spec_id=spec_id, tuning_knobs=knobs)
    )
    assert dict(recorded.spec.knobs) == knobs
    stored = replace(
        moved_to,
        spec_id=spec_id,
        tuning_id=recorded.request.tuning_id,
        tuning_knobs=recorded.request.tuning_knobs,
    )
    with pytest.raises(PinRefused) as raised:
        dispatch_gdn_decode(stored)
    message = str(raised.value)
    assert "out of scope" in message and dropped in message, message
    assert "stale" not in message, message
    # The knobs alone resolve there, minus the ones that problem does not read.
    moved = dispatch_gdn_decode(replace(stored, tuning_id="auto"))
    assert dropped not in dict(moved.spec.knobs)
    assert set(dict(moved.spec.knobs)) < set(knobs)


def test_waves_per_eu_listed_per_sweep_level_and_pinnable_at_any_legal_value():
    """``production`` lists waves_per_eu 0 and 4 on every knob set, ``full``
    every legal value 0-8; any legal value is pinnable at either level."""
    req = _req(gate_kind="gdn")
    spec_id = "nw2_wtk16_bpv8"

    def listed():
        by_knobs: dict = {}
        for row in dispatch_gdn_decode_all(req):
            if row.candidate.spec_id != spec_id:
                continue
            knobs = dict(row.spec.knobs)
            wpe = knobs.pop("waves_per_eu", 0)
            by_knobs.setdefault(tuple(sorted(knobs.items())), set()).add(wpe)
        return by_knobs

    production = listed()
    with sweep_level("full"):
        full = listed()
    assert set(production) == set(full)
    assert all(values == {0, 4} for values in production.values())
    assert all(values == set(range(9)) for values in full.values())
    for wpe in range(9):
        pinned = dispatch_gdn_decode(
            replace(req, spec_id=spec_id, tuning_knobs={"waves_per_eu": wpe})
        )
        assert pinned.spec.kernel_spec.waves_per_eu == wpe
        assert pinned.spec.tuning_id.startswith(f"{spec_id}_wpe{wpe}@")


def test_illegal_pins_are_refused_with_the_candidate_reason():
    req = _req(gate_kind="kda", state_dtype="f32", **_FUSED)
    for knobs, reason in (
        ({"state_store_hint": "nt"}, "state_store_hint"),
        ({"waves_per_eu": 9}, "waves_per_eu"),
        ({"waves_per_eu": True}, "waves_per_eu=True is a bool"),
        ({"waves_per_eu": None}, "waves_per_eu=None"),
        ({"waves_per_eu": 4.0}, "waves_per_eu=4.0 is a float"),
        ({"blocks_per_v_dim": 2}, "not tunable"),
        ({"not_a_knob": True}, "not tunable"),
        ({"dpp_reduce": "yes"}, "dpp_reduce"),
        # Legal only with conv_once on: the kernel validator refuses it.
        ({"norm_gate_once": True}, "norm_gate_once requires conv_once"),
    ):
        with pytest.raises(PinRefused) as raised:
            dispatch_gdn_decode(
                replace(req, spec_id="kda_nw8_wtk8_bpv1", tuning_knobs=knobs)
            )
        assert reason in str(raised.value), (knobs, str(raised.value))
        assert "kda_nw8_wtk8_bpv1" in str(raised.value)
    # A stale id (knobs that no longer reproduce it) is refused, never remapped.
    auto = dispatch_gdn_decode(req)
    stale = replace(auto.request, tuning_knobs={"xcd_remap": True})
    with pytest.raises(PinRefused, match="stale"):
        dispatch_gdn_decode(stale)
    # An id no production configuration carries is refused by name.
    with pytest.raises(PinRefused, match="unknown tuning_id"):
        dispatch_gdn_decode(
            replace(
                req,
                spec_id="kda_nw4_wtk16_bpv1",
                tuning_id="kda_nw4_wtk16_bpv1_wpe0@0000000000000000",
            )
        )
    # A tile that is illegal for the request is refused by name.
    with pytest.raises(PinRefused, match="kda_nw8_wtk16_bpv4"):
        dispatch_gdn_decode(replace(req, spec_id="kda_nw8_wtk16_bpv4"))


def test_a_tuning_pin_needs_a_spec_id():
    for changes in (
        {"tuning_id": "nw2_wtk16_bpv8_wpe0@0123456789abcdef"},
        {"tuning_knobs": {"dpp_reduce": True}},
    ):
        req = _req(gate_kind="gdn", **changes)
        errors = request_errors(req)
        assert any("pin its spec_id too" in e for e in errors), errors
        with pytest.raises(ValueError, match="pin its spec_id too"):
            dispatch_gdn_decode(req)
        assert dispatch_gdn_decode_all(req) == ()


def test_malformed_tuning_fields_are_rejected_at_the_request():
    errors = request_errors(_req(spec_id="nw2_wtk16_bpv8", tuning_id=None))
    assert any("tuning_id must be a str" in e for e in errors), errors
    with pytest.raises(TypeError):
        _req(spec_id="nw2_wtk16_bpv8", tuning_knobs={"dpp_reduce": [True]})
    with pytest.raises(ValueError, match="twice"):
        _req(
            spec_id="nw2_wtk16_bpv8",
            tuning_knobs=[("dpp_reduce", True), ("dpp_reduce", False)],
        )
