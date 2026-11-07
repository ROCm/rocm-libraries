# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Search spaces: parameters, gating, keys, distance, repair, constructors."""

from __future__ import annotations

import itertools
import random

import pytest
from rocke.dispatch.tuning import flag, gated, values
from rocke.dispatch.tuning.axes import KnobAxis, choices
from rocke.dispatch.tuning.search import BOOL, CATEGORICAL, ORDINAL, Param, SearchSpace
from rocke.dispatch.tuning.search.params import flatten_spec, params_from_axes
from rocke.dispatch.tuning.search.synthetic import brute_force_optimum, gemm_like


def _gated_space(is_valid=None):
    params = (
        Param.ordinal("tile", (32, 64, 128, 256)),
        Param.boolean("prefetch"),
        Param.ordinal("depth", (1, 2, 3), gate="prefetch"),
        Param.categorical("pipe", ("a", "b", "c")),
    )
    return SearchSpace(params, is_valid=is_valid, name="gated")


# ------------------------------------------------------------- Param guards
@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"name": "", "levels": ((),)}, "non-empty"),
        ({"name": "x", "levels": ()}, "expected at least one"),
        ({"name": "x", "levels": ((), ())}, "expected no level to repeat"),
        ({"name": "x", "levels": ((),), "kind": "weird"}, "kind="),
        (
            {"name": "x", "levels": ((), (("x", 1),), (("x", 2),)), "kind": BOOL},
            "bool levels=3, expected 2",
        ),
        ({"name": "x", "levels": ((),), "gate": "x"}, "gate itself"),
    ],
)
def test_param_rejects_malformed(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        Param(**kwargs)


def test_param_ordinal_sorts_and_rejects_non_numeric():
    p = Param.ordinal("t", (128, 32, 64, 32))
    assert [lv[0][1] for lv in p.levels] == [32, 64, 128]
    with pytest.raises(ValueError, match="numeric"):
        Param.ordinal("t", ("a", "b"))


def test_space_rejects_repeated_names_and_forward_gates():
    with pytest.raises(ValueError, match="repeat"):
        SearchSpace((Param.boolean("a"), Param.boolean("a")))
    with pytest.raises(ValueError, match="earlier param"):
        SearchSpace((Param.ordinal("d", (1, 2, 3), gate="g"), Param.boolean("g")))
    with pytest.raises(ValueError, match="at least one"):
        SearchSpace(())


def test_check_rejects_bad_genomes():
    sp = _gated_space()
    with pytest.raises(ValueError, match="length"):
        sp.check((0, 0))
    with pytest.raises(ValueError, match="level"):
        sp.check((4, 0, 0, 0))


# ------------------------------------------------------------------ gating
def test_inactive_gated_genes_do_not_change_config_or_key():
    sp = _gated_space()
    off_a, off_b = (1, 0, 0, 2), (1, 0, 2, 2)
    assert sp.active(off_a) == (True, True, False, True)
    assert (
        sp.decode(off_a)
        == sp.decode(off_b)
        == {"tile": 64, "prefetch": False, "pipe": "c"}
    )
    assert sp.key(off_a) == sp.key(off_b)
    assert sp.canonical_genome(off_b) == off_a
    on_a, on_b = (1, 1, 0, 2), (1, 1, 2, 2)
    assert sp.key(on_a) != sp.key(on_b)
    assert sp.decode(on_b)["depth"] == 3


def test_distance_kinds():
    sp = _gated_space()
    assert sp.distance((0, 0, 0, 0), (3, 0, 0, 0)) == pytest.approx(1.0)
    assert sp.distance((0, 0, 0, 0), (1, 0, 0, 0)) == pytest.approx(1 / 3)
    assert sp.distance((0, 0, 0, 0), (0, 0, 0, 1)) == 1.0
    # gate flip: the gate differs (1) and depth is active on one side only (1)
    assert sp.distance((0, 0, 0, 0), (0, 1, 0, 0)) == 2.0
    # inactive genes are ignored
    assert sp.distance((0, 0, 0, 0), (0, 0, 2, 0)) == 0.0


def test_moves_are_nearest_first_and_respect_radius():
    sp = _gated_space()
    moves = sp.moves((1, 0, 0, 0), ordinal_radius=1)
    dists = [sp.distance((1, 0, 0, 0), m) for m in moves]
    assert dists == sorted(dists)
    tiles = {m[0] for m in moves if m[0] != 1}
    assert tiles == {0, 2}
    # inactive depth is never moved
    assert all(m[2] == 0 for m in moves)
    wide = sp.moves((1, 0, 0, 0), ordinal_radius=None)
    assert {m[0] for m in wide if m[0] != 1} == {0, 2, 3}


# --------------------------------------------------------------- validity
def _rule(c):
    if c["tile"] == 256 and c.get("depth") == 3:
        return False
    return not (c["pipe"] == "c" and c["tile"] < 64)


def test_repair_returns_valid_nearest_single_move():
    sp = _gated_space(_rule)
    rng = random.Random(0)
    all_g = [g for g in itertools.product(*(range(p.size) for p in sp.params))]
    for g in all_g:
        r = sp.repair(g, rng)
        assert r is not None and sp.valid(r)
        if sp.valid(g):
            assert r == g
            continue
        singles = [m for m in sp.moves(g, ordinal_radius=None) if sp.valid(m)]
        if singles:
            best = min(sp.distance(g, m) for m in singles)
            assert sp.distance(g, r) == pytest.approx(best)


def test_repair_falls_back_to_archive_then_random():
    # Only one config is valid and it is far from everything.
    only = {"tile": 256, "prefetch": True, "depth": 2, "pipe": "a"}
    sp = _gated_space(lambda c: c == only)
    target = sp.encode(only)
    got = sp.repair((0, 0, 0, 2), random.Random(1), archive=[target], max_checks=4)
    assert got == target
    got = sp.repair((0, 0, 0, 2), random.Random(1), max_checks=4)
    assert got is None or sp.valid(got)


def test_repair_none_when_nothing_is_valid():
    sp = _gated_space(lambda c: False)
    assert sp.repair((0, 0, 0, 0), random.Random(0), max_checks=8) is None


def test_validity_is_cached():
    calls = []

    def rule(c):
        calls.append(1)
        return True

    sp = _gated_space(rule)
    for _ in range(5):
        sp.key((1, 1, 1, 1))
    sp.key((1, 1, 1, 1))
    assert len(calls) == 1 and sp.checks == 1


# ---------------------------------------------------------- enumeration
def test_enumerate_visits_each_config_once():
    sp = _gated_space()
    gs = list(sp.enumerate())
    keys = [sp.key(g) for g in gs]
    assert len(keys) == len(set(keys))
    # 4 tiles x (off + 3 depths) x 3 pipes
    assert len(gs) == 4 * 4 * 3
    assert len(list(sp.enumerate(limit=5))) == 5


def test_encode_inverts_decode_on_canonical_genomes():
    sp = _gated_space()
    for g in sp.enumerate():
        assert sp.encode(sp.decode(g)) == g


def test_stratified_covers_levels_and_has_distinct_keys():
    sp = _gated_space()
    gs = sp.stratified(random.Random(3), 12)
    keys = [sp.key(g) for g in gs]
    assert len(set(keys)) == len(keys) == 12
    assert {g[0] for g in gs} == {0, 1, 2, 3}
    assert {g[3] for g in gs} == {0, 1, 2}


def test_stratified_repair_cap_bounds_validity_checks():
    # A predicate space where few genomes are valid: an uncapped repair walks
    # singles and pairs per pool point; the cap (used for large candidate
    # pools) bounds the checks and still returns distinct valid genomes.
    params = tuple(Param.ordinal(f"x{i}", range(6)) for i in range(5))
    calls = [0]

    def ok(c):
        calls[0] += 1
        return (sum(c.values()) % 9 == 0) and c["x0"] != c["x1"]

    def run(cap):
        calls[0] = 0
        sp = SearchSpace(params, is_valid=ok)
        gs = sp.stratified(random.Random(5), 40, repair_checks=cap)
        assert all(sp.valid(g) for g in gs)
        assert len({sp.key(g) for g in gs}) == len(gs) >= 30
        return calls[0]

    full, capped = run(None), run(4)
    assert capped < full / 2, (capped, full)


def test_fingerprint_tracks_params():
    a = _gated_space().fingerprint()
    b = _gated_space().fingerprint()
    c = SearchSpace((Param.ordinal("tile", (32, 64)),)).fingerprint()
    assert a == b and a != c


# ----------------------------------------------------------- constructors
def test_from_configs_membership_and_kinds():
    rows = [
        {"tile": 64, "pipe": "a", "swz": False, "k": 1, "opt": None},
        {"tile": 128, "pipe": "b", "swz": True, "k": 1, "opt": 3},
        {"tile": 256, "pipe": "a", "swz": True, "k": 1},
    ]
    sp = SearchSpace.from_configs(rows)
    kinds = {p.name: p.kind for p in sp.params}
    assert kinds == {
        "tile": ORDINAL,
        "pipe": CATEGORICAL,
        "swz": BOOL,
        "opt": CATEGORICAL,
    }
    valid = [g for g in sp.enumerate() if sp.valid(g)]
    assert len(valid) == 3
    decoded = sorted((sp.decode(g) for g in valid), key=lambda c: c["tile"])
    assert decoded[0] == {**rows[0]}
    assert decoded[2] == {**rows[2], "opt": None}
    assert sp.members[sp.key(sp.encode(rows[1]))] == 1


def test_from_configs_single_config_and_empty():
    sp = SearchSpace.from_configs([{"a": 1}])
    (g,) = [g for g in sp.enumerate() if sp.valid(g)]
    assert sp.decode(g) == {"a": 1}
    with pytest.raises(ValueError, match="expected at least one config"):
        SearchSpace.from_configs([])


def test_params_from_axes_kinds_and_base_position():
    axes = (
        values("unroll", 2, (1, 2, 4, 8)),
        flag("swizzle"),
        gated("prefetch", {"depth": (1, 2, 3), "mode": ("x", "y")}),
        choices("sched", ("a", "b")),
        KnobAxis("combo", ((), (("p", 1), ("q", 2)), (("p", 2),))),
    )
    params = params_from_axes(axes, {"unroll": 2})
    by = {p.name: p for p in params}
    unroll = by["unroll"]
    assert unroll.kind == ORDINAL
    assert unroll.levels == ((("unroll", 1),), (), (("unroll", 4),), (("unroll", 8),))
    assert by["swizzle"].kind == BOOL
    assert by["prefetch"].kind == BOOL and by["prefetch"].levels[1] == (
        ("prefetch", True),
    )
    assert by["depth"].gate == "prefetch" and by["depth"].kind == ORDINAL
    assert by["mode"].gate == "prefetch" and by["mode"].kind == CATEGORICAL
    assert by["sched"].kind == CATEGORICAL and len(by["sched"].levels) == 3
    assert by["combo"].kind == CATEGORICAL and len(by["combo"].levels) == 3


def test_params_from_axes_keeps_non_product_gated_axis_whole():
    axis = KnobAxis(
        "g", ((), (("g", True), ("d", 1)), (("g", True), ("d", 2), ("e", 1)))
    )
    (p,) = params_from_axes((axis,), {})
    assert p.kind == CATEGORICAL and len(p.levels) == 3


def test_flatten_spec_nested():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Tile:
        m: int
        n: int

    @dataclass(frozen=True)
    class Spec:
        tile: Tile
        pipe: str
        shape: tuple

    flat = flatten_spec(Spec(Tile(1, 2), "x", (3, 4)))
    assert flat == {"tile.m": 1, "tile.n": 2, "pipe": "x", "shape": "(3, 4)"}


def test_synthetic_optimum_is_valid_and_unique_value():
    sp, f = gemm_like()
    key, best, count = brute_force_optimum(sp, f)
    assert key is not None and count > 1000
    assert best > 0


# ------------------------------------------------- structured spaces
def _chain_space(n=10, with_structure=True):
    """Flag k requires flag k-1 (constraints point backwards, like the
    dispatcher's axis order); uniform random genomes are almost never valid."""
    params = tuple(Param.boolean(f"f{k}") for k in range(n))

    def ok(c):
        return all(not c[f"f{k}"] or c[f"f{k - 1}"] for k in range(1, n))

    def prefix_ok(c):
        return ok(c)

    def sampler(rng):
        depth = rng.randrange(n + 1)
        return {f"f{k}": k < depth for k in range(n)}

    if not with_structure:
        return SearchSpace(params, is_valid=ok)
    return SearchSpace(params, is_valid=ok, prefix_valid=prefix_ok, sampler=sampler)


def test_prefix_repair_keeps_the_valid_prefix():
    sp = _chain_space()
    rng = random.Random(0)
    # f0..f3 on, f4 off, f5..f9 on: nearest valid along the order keeps f0..f3
    g = (1, 1, 1, 1, 0, 1, 1, 1, 1, 1)
    r = sp.repair(g, rng, max_checks=0)
    assert r == (1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
    for _ in range(50):
        r = sp.repair(sp.random_genome(rng), rng, max_checks=0)
        assert r is not None and sp.valid(r)


def test_structural_sampler_beats_uniform_draws():
    rng = random.Random(1)
    plain = _chain_space(with_structure=False)
    hits = sum(plain.random_valid(rng, tries=1) is not None for _ in range(200))
    sp = _chain_space()
    keys = {sp.key(sp.random_valid(rng, tries=1)) for _ in range(200)}
    assert hits < 60
    assert None not in keys and len(keys) == 11
    assert len(sp.stratified(random.Random(2), 11)) >= 9
