# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx950 GDN/KDA decode tuned candidates keep the shared tuning contract.

:func:`rocke.dispatch.tuning.testing.assert_tuning_contract` checks pin
resolution, id uniqueness, replay and portability for representative tiles in
every mode. The rest pins what is specific to this family: the knob space each
mode exposes, that it holds one kernel per configuration, that ``auto`` is
one configuration of that space (its case's knob stack on the case's tile),
and that out-of-scope knobs are dropped while illegal ones are refused.
"""

from __future__ import annotations

from dataclasses import replace
from unittest import mock

import pytest

from dispatch.gdn import (
    GDN_REGISTRY,
    GdnDecodeRequest,
    dispatch_gdn_decode,
    dispatch_gdn_decode_all,
    request_errors,
)
from dispatch.gdn import gfx950, rules
from dispatch.gdn.gfx950 import ARCH, auto_default, spec_id_for
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
    # Refused even where row_loads is out of scope, never read as a value.
    {"row_loads": "early"},
    {"row_loads": 1},
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
    (
        # One wave with fuse_conv: row_loads is in scope; "auto" restates the
        # base, and so does "late" without conv_once (auto's order there).
        "gdn_decode_gfx950_kda_nw1_wtk8_bpv1",
        _req(gate_kind="kda", state_dtype="f32", **_FUSED),
        _req(5, heads=8, gate_kind="kda", state_dtype="f32", **_FUSED),
        {"row_loads": "late", "conv_once": False},
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
    values (0 and 4). One wave with fuse_conv adds row_loads' forced order
    where it is not auto's: one per knob set at each waves_per_eu value
    ("first", except "late" with conv_once at DK = DV = 128 and WTK 4 / 8 at
    waves_per_eu 0, where auto loads first)."""
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
    knob_sets = hints * dpp * xcd * stream * ic * olds * conv
    orders = 2 if req.fuse_conv and kernel.num_warps == 1 else 1
    return knob_sets * 2 * orders


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
def test_auto_is_its_case_stack_on_the_case_tile(req):
    """``auto`` is a registered configuration, not a new one: the case's
    knobs pinned on the case's tile give the same spec, and the knobs it
    records are the stack less the knobs this request cannot read."""
    auto = dispatch_gdn_decode(req)
    default = auto_default(req)
    spec_id = spec_id_for(default.tile, req.gate_kind)
    assert auto.candidate.spec_id == spec_id
    wpe = default.knobs.get("waves_per_eu", 0)
    assert auto.spec.tuning_id.startswith(f"{spec_id}_wpe{wpe}@")
    assert dict(auto.spec.knobs).items() <= dict(default.knobs).items()
    kernel = auto.spec.kernel_spec
    for name, value in auto.spec.knobs:
        assert getattr(kernel, name) == value, name
    stack_pin = replace(req, spec_id=spec_id, tuning_knobs=dict(default.knobs))
    assert dispatch_gdn_decode(stack_pin).spec == auto.spec
    # One of the tile's enumerated configurations.
    assert auto.spec in {
        r.spec for r in dispatch_gdn_decode_all(req) if r.candidate.spec_id == spec_id
    }
    # The result's request is pinned to what ran, so it replays it.
    assert (auto.request.spec_id, auto.request.tuning_id) == (
        auto.candidate.spec_id,
        auto.spec.tuning_id,
    )
    assert auto.request.tuning_knobs == auto.spec.knobs
    assert dispatch_gdn_decode(auto.request).spec == auto.spec


@pytest.mark.parametrize("state_dtype", ("bf16", "f16"))
def test_norm_only_auto_drops_norm_gate_once_with_conv_once(state_dtype):
    """The GDN fused bf16 row carries conv_once and norm_gate_once. On a
    norm-only request conv_once is out of scope, and norm_gate_once without
    it is refused by the kernel, so auto drops both and runs the rest of
    the row instead of raising."""
    req = _req(gate_kind="gdn", state_dtype=state_dtype, heads=16, fuse_out_norm=True)
    default = auto_default(req)
    assert default.knobs.get("conv_once") and default.knobs.get("norm_gate_once")
    auto = dispatch_gdn_decode(req)
    assert auto.candidate.spec_id == spec_id_for(default.tile, "gdn")
    expected = {
        k: v
        for k, v in default.knobs.items()
        if k not in ("conv_once", "norm_gate_once")
    }
    assert dict(auto.spec.knobs) == expected
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
    working: the knob is dropped, so the configuration is the base one, and
    the result's explanation names every dropped knob so a caller that logs
    the pin cannot record a knob that did not run."""
    plain = _req(gate_kind="gdn")
    for spec_id, knobs in (
        ("nw2_wtk16_bpv8", {"conv_once": True, "norm_gate_once": True}),
        ("nw2_wtk16_bpv8", {"out_lds": True}),
        ("nw2_wtk16_bpv8", {"interleave_cols": True}),  # 2-byte state
        ("nw1_wtk4_bpv4", {"dpp_reduce": True}),  # no xor-4 stage
        ("nw8_wtk16_bpv4", {"stream_rows": True}),  # one state row per lane
        ("nw1_wtk4_bpv4", {"row_loads": "first"}),  # no fuse_conv
    ):
        got = dispatch_gdn_decode(replace(plain, spec_id=spec_id, tuning_knobs=knobs))
        base = dispatch_gdn_decode(replace(plain, spec_id=spec_id))
        assert got.spec == base.spec, (spec_id, knobs)
        assert got.spec.knobs == ()
        note = f"dropped tuning_knobs that change no code here {sorted(knobs)}"
        assert any(note in line for line in got.explanation), got.explanation
        assert not any("dropped" in line for line in base.explanation)


def test_in_scope_knobs_are_not_reported_as_dropped():
    """Only knobs that were dropped are named: a pin that partly applies names
    just the dropped part, and a fully applied pin names nothing."""
    plain = _req(gate_kind="gdn")
    part = dispatch_gdn_decode(
        replace(
            plain,
            spec_id="nw2_wtk16_bpv8",
            tuning_knobs={"dpp_reduce": True, "conv_once": True},
        )
    )
    assert dict(part.spec.knobs) == {"dpp_reduce": True}
    assert any(
        "dropped tuning_knobs that change no code here ['conv_once']" in line
        for line in part.explanation
    ), part.explanation
    full = dispatch_gdn_decode(
        replace(plain, spec_id="nw2_wtk16_bpv8", tuning_knobs={"dpp_reduce": True})
    )
    assert dict(full.spec.knobs) == {"dpp_reduce": True}
    assert not any("dropped" in line for line in full.explanation)


def test_a_restating_row_loads_is_reported_as_dropped_for_an_aliased_flag():
    """The restated order is judged on the knobs as the kit types them, so
    ``conv_once: 1`` reads as ``True`` and the dropped ``row_loads`` is named
    exactly as with ``conv_once: True``."""
    plain = _req(gate_kind="gdn", heads=16, **_FUSED)
    for conv_once in (True, 1):
        got = dispatch_gdn_decode(
            replace(
                plain,
                spec_id="nw1_wtk8_bpv1",
                tuning_knobs={"conv_once": conv_once, "row_loads": "first"},
            )
        )
        assert dict(got.spec.knobs) == {"conv_once": True}, conv_once
        note = "dropped tuning_knobs that change no code here ['row_loads']"
        assert any(note in line for line in got.explanation), (
            conv_once,
            got.explanation,
        )


def test_row_loads_is_one_configuration_per_order_auto_does_not_choose():
    """On a single-wave fuse_conv tile, a forced order that is auto's own (for
    its knobs and waves_per_eu) is dropped like an out-of-scope knob, and the
    other one is a configuration of its own; at every waves_per_eu value the
    walk lists the latter and never lists auto's kernel twice."""
    req = _req(gate_kind="kda", state_dtype="f32", **_FUSED)
    spec_id = "kda_nw1_wtk8_bpv1"

    def pin(**knobs):
        return dispatch_gdn_decode(replace(req, spec_id=spec_id, tuning_knobs=knobs))

    co = {"conv_once": True}
    for restated, base in (
        ({"row_loads": "late"}, {}),
        ({**co, "row_loads": "first"}, co),
        ({**co, "row_loads": "late", "waves_per_eu": 4}, {**co, "waves_per_eu": 4}),
    ):
        assert pin(**restated).spec == pin(**base).spec, restated
    for forced, base in (
        ({"row_loads": "first"}, {}),
        ({**co, "row_loads": "late"}, co),
        ({**co, "row_loads": "first", "waves_per_eu": 4}, {**co, "waves_per_eu": 4}),
    ):
        got = pin(**forced)
        assert dict(got.spec.knobs) == forced, forced
        assert (
            got.spec.kernel_spec.kernel_name()
            != pin(**base).spec.kernel_spec.kernel_name()
        )
    for level, floors in (("production", (4,)), ("full", range(1, 9))):
        listed = {}
        for row in dispatch_gdn_decode_all(req, sweep_level=level):
            if row.candidate.spec_id == spec_id:
                knobs = dict(row.spec.knobs)
                listed.setdefault(
                    (knobs.get("conv_once", False), knobs.get("waves_per_eu", 0)),
                    set(),
                ).add(knobs.get("row_loads", "auto"))
        want = {(False, 0): {"auto", "first"}, (True, 0): {"auto", "late"}}
        for wpe in floors:
            want[(False, wpe)] = want[(True, wpe)] = {"auto", "first"}
        assert listed == want, level


def test_an_out_of_range_waves_per_eu_is_refused_by_name_on_a_stored_id():
    """Whether a ``row_loads`` restates auto is judged only on a legal spec:
    a stored id replayed with ``waves_per_eu`` 9 (where auto would load late)
    is refused for the range, as the same knobs without the id are."""
    req = _req(gate_kind="kda", state_dtype="f32", **_FUSED)
    spec_id = "kda_nw1_wtk8_bpv1"
    knobs = {"conv_once": True, "row_loads": "late"}
    recorded = dispatch_gdn_decode(replace(req, spec_id=spec_id, tuning_knobs=knobs))
    for tid in (recorded.spec.tuning_id, "auto"):
        bad = replace(
            req,
            spec_id=spec_id,
            tuning_id=tid,
            tuning_knobs={**knobs, "waves_per_eu": 9},
        )
        with pytest.raises(PinRefused) as raised:
            dispatch_gdn_decode(bad)
        message = str(raised.value)
        assert "waves_per_eu must be in [0, 8], got 9" in message, (tid, message)
        assert "out of scope" not in message, (tid, message)


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
        # Recorded where auto loads first (conv_once, WTK 8, DK = DV = 128);
        # at DK = DV = 64 auto loads late, so "late" changes no code there.
        (
            "nw1_wtk8_bpv1",
            {"conv_once": True, "row_loads": "late"},
            _req(gate_kind="gdn", heads=16, **_FUSED),
            _req(gate_kind="gdn", heads=16, head_k_dim=64, head_v_dim=64, **_FUSED),
            "row_loads",
        ),
        # Recorded where auto loads late (DK = DV = 64); at DK = DV = 128 auto
        # loads first, so "first" restates it there.
        (
            "nw1_wtk8_bpv1",
            {"conv_once": True, "row_loads": "first"},
            _req(gate_kind="gdn", heads=16, head_k_dim=64, head_v_dim=64, **_FUSED),
            _req(gate_kind="gdn", heads=16, **_FUSED),
            "row_loads",
        ),
    ),
    ids=(
        "dv64_drops_stream_rows",
        "bf16_plain_drops_f32_fused_knobs",
        "d64_drops_auto_row_loads",
        "d128_drops_restating_first",
    ),
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
    # An aliased value (1 for True) names the same configuration, so it is
    # refused under the same name.
    aliased = replace(
        stored,
        tuning_knobs={k: 1 if v is True else v for k, v in knobs.items()},
    )
    with pytest.raises(PinRefused) as raised:
        dispatch_gdn_decode(aliased)
    message = str(raised.value)
    assert "out of scope" in message and dropped in message, message
    assert "stale" not in message, message
    # The knobs alone resolve there, minus the ones that problem does not read.
    moved = dispatch_gdn_decode(replace(stored, tuning_id="auto"))
    assert dropped not in dict(moved.spec.knobs)
    assert set(dict(moved.spec.knobs)) < set(knobs)


def test_a_changed_default_refuses_the_stored_pin_instead_of_drifting():
    """A pin stores only its knob delta; the defaults it is relative to are
    fingerprinted into config_key. When a default moves, the stored
    (tuning_id, knobs) pin is refused rather than rebuilt into a kernel that
    was never measured. Candidates cache resolution per request, so each side
    builds a fresh candidate, as a new release would."""

    def fresh_candidate():
        return gfx950._make_candidate(tile=(2, 16, 8), priority=1, gate_kind="gdn")

    req = _req(gate_kind="gdn", heads=16)
    stored = dispatch_gdn_decode(
        replace(req, spec_id="nw2_wtk16_bpv8", tuning_knobs={"dpp_reduce": True})
    ).spec
    pin = replace(
        req,
        spec_id="nw2_wtk16_bpv8",
        tuning_id=stored.tuning_id,
        tuning_knobs={"dpp_reduce": True},
    )
    assert fresh_candidate().select_spec(pin) == stored

    with mock.patch.dict(rules._SPEC_DEFAULTS, {"state_load_hint": "streaming"}):
        upgraded = fresh_candidate()
        ok, why = upgraded.admits(pin)
        assert not ok
        assert stored.tuning_id in why
        assert "the defaults they are relative to" in why
        assert "re-sweep/revalidate" in why
        assert "no fallback" in why
        moved = upgraded.select_spec(replace(pin, tuning_id="auto"))
        assert moved.config_key != stored.config_key


def test_every_spec_field_is_classified_exactly_once():
    """Every GdnDecodeSpec field is a problem field, a tile field, a knob, or
    a named untunable field, exactly one of them. A new spec field (a future
    knob) fails here until someone decides whether to sweep it, instead of
    falling silently into the defaults."""
    import dataclasses as dc

    from dispatch.gdn.axes import AXES
    from kernels.gfx950.gdn_decode import GdnDecodeSpec
    from rocke.dispatch.tuning import axis_knob_names

    groups = {
        "problem": rules.PROBLEM_FIELDS,
        "tile": rules.TILE_FIELDS,
        "knob": frozenset(axis_knob_names(AXES)) | {"waves_per_eu"},
        "untunable": rules.UNTUNABLE_FIELDS,
    }
    fields = {f.name for f in dc.fields(GdnDecodeSpec)}
    for name, group in groups.items():
        assert group <= fields, (name, sorted(group - fields))  # no stale member
    for field in sorted(fields):
        owners = [name for name, group in groups.items() if field in group]
        assert len(owners) == 1, (field, owners)


def test_pinned_xcd_remap_keeps_the_launch_grid():
    """xcd_remap renumbers workgroups over the grid it computes from
    batch_size, so the launch grid must stay gdn_decode_grid(batch, spec)
    when the knob is pinned: the same grid with it on and off, in every mode."""
    from kernels.gfx950.gdn_decode import gdn_decode_grid

    for req in (
        _req(gate_kind="gdn"),
        _req(gate_kind="kda", state_dtype="f32"),
        _req(gate_kind="kda", state_dtype="f32", **_FUSED),
    ):
        base = dispatch_gdn_decode(req)
        spec_id = base.candidate.spec_id
        on = dispatch_gdn_decode(
            replace(req, spec_id=spec_id, tuning_knobs={"xcd_remap": True})
        )
        assert dict(on.spec.knobs) == {"xcd_remap": True}, req
        assert on.grid == gdn_decode_grid(req.batch, on.spec.kernel_spec), req
        assert on.grid == base.grid and on.block == base.block, req


@pytest.mark.parametrize(
    "req",
    (
        _req(gate_kind="gdn"),
        _req(gate_kind="gdn", head_k_dim=64, head_v_dim=64),
        _req(gate_kind="kda", state_dtype="f32", heads=16, **_FUSED),
        _req(gate_kind="gdn", heads=16, fuse_out_norm=True),
        _req(gate_kind="gdn", spec_id="nw4_wtk16_bpv8"),
        _req(
            gate_kind="gdn", spec_id="nw2_wtk16_bpv8", tuning_knobs={"dpp_reduce": True}
        ),
        # Refused pins list nothing in either function, never a base row.
        _req(
            gate_kind="gdn",
            spec_id="nw2_wtk16_bpv8",
            tuning_id="nw2_wtk16_bpv8_wpe0@deadbeefdeadbeef",
        ),
        _req(
            gate_kind="gdn",
            spec_id="nw2_wtk16_bpv8",
            tuning_knobs={"dpp_reduce": "yes"},
        ),
    ),
    ids=(
        "gdn",
        "gdn_d64",
        "kda_f32_fused",
        "gdn_norm_only",
        "gdn_pinned",
        "gdn_knob_pin",
        "gdn_unknown_tuning_id",
        "gdn_invalid_knob",
    ),
)
def test_bases_equal_the_knob_free_rows_of_all(req):
    """dispatch_gdn_decode_bases builds only each tile's base configuration
    (an opt-in probe per candidate, as iter_combos walks), yet returns
    exactly the knob-free rows dispatch_gdn_decode_all lists: same
    candidates, order, specs, ids, grid, block and pinned request."""
    from dispatch.gdn import dispatch_gdn_decode_bases

    def key(row):
        return (
            row.candidate.name,
            row.spec,
            row.kernel_id,
            row.grid,
            row.block,
            row.request,
        )

    expected = [key(r) for r in dispatch_gdn_decode_all(req) if not r.spec.knobs]
    got = [key(r) for r in dispatch_gdn_decode_bases(req)]
    assert got == expected
    assert len(got) > 1 or req.spec_id != "auto"  # route_auto must not prune


def test_kda_admits_exactly_its_d128_tiles():
    """KDA accepts only DK = DV = 128, so 54 of its 180 registered tiles are
    admissible and the rest are unselectable until that limit is lifted."""
    from dispatch.gdn import dispatch_gdn_decode_bases

    kda = [c for c in GDN_REGISTRY.candidates() if c.spec_id.startswith("kda_")]
    assert len(kda) == 180
    assert len(dispatch_gdn_decode_bases(_req(gate_kind="kda"))) == 54


def test_empty_values_default_only_where_documented():
    """build() defaults the arch only when none is given; an empty arch is
    passed through and refused, never replaced by the spec's. An empty or
    whitespace tuning_id means auto, as documented for spec_id."""
    from dispatch.gdn import common

    req = _req(gate_kind="gdn")
    spec = dispatch_gdn_decode(req).spec
    seen = []
    module = common._decode_kernel_module(spec.arch)
    with mock.patch.object(
        module, "build_gdn_decode", side_effect=lambda s, arch: seen.append(arch)
    ):
        spec.build(None)
        spec.build("")
    assert seen == [spec.arch, ""]
    with pytest.raises(ValueError, match="unknown gfx target"):
        spec.build("")

    base = dispatch_gdn_decode(replace(req, spec_id="nw2_wtk16_bpv8"))
    for blank in ("", "   "):
        got = dispatch_gdn_decode(
            replace(req, spec_id="nw2_wtk16_bpv8", tuning_id=blank)
        )
        assert got.spec == base.spec, repr(blank)


def test_waves_per_eu_listed_per_sweep_level_and_pinnable_at_any_legal_value():
    """``production`` lists waves_per_eu 0 and 4 on every knob set, ``full``
    every legal value 0-8; any legal value is pinnable at either level."""
    req = _req(gate_kind="gdn")
    spec_id = "nw2_wtk16_bpv8"

    def listed(level):
        by_knobs: dict = {}
        for row in dispatch_gdn_decode_all(req, sweep_level=level):
            if row.candidate.spec_id != spec_id:
                continue
            knobs = dict(row.spec.knobs)
            wpe = knobs.pop("waves_per_eu", 0)
            by_knobs.setdefault(tuple(sorted(knobs.items())), set()).add(wpe)
        return by_knobs

    production = listed("production")
    full = listed("full")
    # The argument decides, not the caller's ambient level.
    with sweep_level("full"):
        assert listed("production") == production
    assert set(production) == set(full)
    assert all(values == {0, 4} for values in production.values())
    assert all(values == set(range(9)) for values in full.values())
    for wpe in range(9):
        pinned = dispatch_gdn_decode(
            replace(req, spec_id=spec_id, tuning_knobs={"waves_per_eu": wpe})
        )
        assert pinned.spec.kernel_spec.waves_per_eu == wpe
        assert pinned.spec.tuning_id.startswith(f"{spec_id}_wpe{wpe}@")


def test_full_level_samples_per_tile_reproducibly_and_production_ignores_it():
    """``tuning_sample`` draws at most that many configurations per tile at
    ``full``, the same ones for the same ``seed``; ``production`` ignores it."""
    req = _req(gate_kind="kda")
    sampled = dispatch_gdn_decode_all(req, sweep_level="full", tuning_sample=3, seed=7)
    per_tile: dict = {}
    for row in sampled:
        per_tile[row.candidate.spec_id] = per_tile.get(row.candidate.spec_id, 0) + 1
    assert per_tile and max(per_tile.values()) <= 3
    again = dispatch_gdn_decode_all(req, sweep_level="full", tuning_sample=3, seed=7)
    assert [r.spec.tuning_id for r in again] == [r.spec.tuning_id for r in sampled]
    assert dispatch_gdn_decode_all(
        req, sweep_level="production", tuning_sample=3
    ) == dispatch_gdn_decode_all(req)
    with pytest.raises(ValueError, match="sweep level"):
        dispatch_gdn_decode_all(req, sweep_level="everything")


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
        ({"row_loads": "early"}, "row_loads"),
        ({"row_loads": True}, "row_loads"),
        # Legal only with conv_once on: the kernel validator refuses it.
        ({"norm_gate_once": True}, "norm_gate_once requires conv_once"),
    ):
        with pytest.raises(PinRefused) as raised:
            dispatch_gdn_decode(
                replace(req, spec_id="kda_nw8_wtk8_bpv1", tuning_knobs=knobs)
            )
        assert reason in str(raised.value), (knobs, str(raised.value))
        assert "kda_nw8_wtk8_bpv1" in str(raised.value)
    # Scope is the kernel's own predicate (norm_gate_once applies with
    # fuse_out_norm), so on a norm-only request norm_gate_once is in scope and
    # the validator's NOT_YET_IMPLEMENTED coupling to conv_once refuses it,
    # exactly as with conv on and conv_once off; it is not silently dropped.
    norm_only = _req(gate_kind="kda", state_dtype="f32", fuse_out_norm=True)
    for mode in (norm_only, req):
        with pytest.raises(PinRefused, match="NOT_YET_IMPLEMENTED"):
            dispatch_gdn_decode(
                replace(
                    mode,
                    spec_id="kda_nw8_wtk8_bpv1",
                    tuning_knobs={"norm_gate_once": True},
                )
            )
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
