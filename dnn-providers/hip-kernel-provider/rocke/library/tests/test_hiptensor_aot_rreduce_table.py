# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU checks of the tuned reduction classes' route table and host rules
(``kernels.common.hiptensor_aot.rreduce_host``; table
``rocke/dispatch/families/hiptensor_rreduce.json``).

* the table is valid (``builders/hiptensor/gen_rreduce_table.py --check``);
* every pack target can run every reduction with some class (the universal knobs of each mode, data
  type, operator, with and without opA, 32- and 64-bit offsets);
* the tuned problems' routes and the prototypes' records round-trip;
* the untuned choice returns runnable plans for random problems, and the kernel arguments a plan
  sets are exactly the parameters of the class kernels it names;
* the quarter-octave scale of the untuned choice's features (integer arithmetic, mirrored in
  hipTensor's C++ planner).
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from kernels.common.hiptensor_aot import rreduce_host as H
from kernels.common.hiptensor_aot.host import canonicalize
from kernels.common.hiptensor_aot.rreduce import (
    OPS,
    build_rreduce,
    build_rreduce_final,
)

ARCHES = ("gfx90a", "gfx942", "gfx950")
DTYPES = ("f16", "bf16", "f32", "f64")


def _gen():
    from builders.hiptensor import gen_rreduce_table

    return gen_rreduce_table


def test_table_is_valid():
    errs = _gen().check(H.load_table())
    assert not errs, "\n".join(errs[:20])


@pytest.mark.parametrize("arch", ARCHES)
def test_every_problem_class_has_a_class(arch):
    """The universal knobs of each mode run every problem: the pack has them for every data
    type, operator, opA flag and offset width (8 reduced slots)."""
    classes = H.pack_classes(arch)
    have = {
        (s.mode, s.a, s.op, s.una, s.idx64)
        for s in classes
        if (s.block, s.vec, s.lanes, s.unroll, s.rows) == H.UNIVERSAL_KNOBS[s.mode]
        and s.nr == 8
    }
    for mode, a, op, una, idx64 in itertools.product(
        ("row", "col"), DTYPES, OPS, (False, True), (False, True)
    ):
        assert (mode, a, op, una, idx64) in have, (mode, a, op, una, idx64)


@pytest.mark.parametrize("arch", ARCHES)
def test_routes_round_trip(arch):
    table = H.load_table()
    routes = H.routes(arch, table)
    entries = [e for e in table["entries"] if e["arch"] == arch]
    for e in entries:
        key = H.problem_key(e["a"], e["d"], e["op"], e["kept"], e["red"])
        name = f"reduce.tuned.{H.fnv1a64(key):016x}"
        assert routes[name + ".key"] == key
        assert routes[name] == H.route_record(e)
    protos = table.get("prototypes", [])
    assert routes["reduce.proto.count"] == str(len(protos))
    parsed = H.prototypes(table)
    for i, (p, q) in enumerate(zip(protos, parsed)):
        rec = routes[f"reduce.proto.{i}"]
        assert rec == H.proto_record(p)
        assert f"waves={q.waves};" in rec and rec.startswith(
            f"v1;mode={q.mode};eb={q.eb};"
        )


def test_fnv1a64_known_values():
    # FNV-1a 64 reference values (hipTensor's planner and pack builder hash the same keys)
    assert H.fnv1a64("") == 0xCBF29CE484222325
    assert H.fnv1a64("a") == 0xAF63DC4C8601EC8C


@pytest.mark.parametrize(
    "x,want",
    [
        (0, 0),
        (1, 0),
        (2, 4),
        (3, 6),
        (4, 8),
        (5, 9),
        (6, 10),
        (7, 11),
        (8, 12),
        (1 << 20, 80),
        ((1 << 20) + (1 << 19), 82),
        ((1 << 31) - 1, 4 * 30 + 3),
    ],
)
def test_log_q(x, want):
    assert H.log_q(x) == want


def _random_problem(rng):
    rank = rng.randrange(1, 7)
    shape = [rng.choice([1, 2, 3, 5, 8, 16, 33, 64, 128, 1000]) for _ in range(rank)]
    perm = list(range(rank))
    rng.shuffle(perm)
    sa, acc = [0] * rank, 1
    for i in perm:
        sa[i] = acc
        acc *= shape[i] + (rng.choice([0, 0, 3]) if i == perm[0] else 0)
    kept = sorted(rng.sample(range(rank), rng.randrange(0, rank + 1)))
    sd, acc = {}, 1
    order = list(kept)
    rng.shuffle(order)
    for i in order:
        sd[i] = acc
        acc *= shape[i]
    kd = canonicalize([(shape[i], [sa[i], sd[i]]) for i in kept], 0)
    rd = canonicalize([(shape[i], [sa[i]]) for i in range(rank) if i not in kept], 0)
    return [[e, s[0], s[1]] for e, s in kd], [[e, s[0]] for e, s in rd]


@pytest.mark.parametrize("arch", ARCHES)
def test_untuned_choice_runs_random_problems(arch):
    """Every random problem gets a choice; its plan builds; the arguments it sets are the
    kernels' parameters (the host rules and the kernel ABI agree)."""
    rng = random.Random(7 + ARCHES.index(arch))
    classes = H.pack_classes(arch)
    protos = H.prototypes()
    built = {}
    for _ in range(60):
        kept, red = _random_problem(rng)
        a = rng.choice(DTYPES)
        op = rng.choice(tuple(OPS))
        una = rng.random() < 0.3
        ch = H.untuned_choices(
            kept,
            red,
            a,
            a,
            op,
            classes,
            protos,
            una=una,
            arch=arch,
            cus=rng.choice([104, 256, 304]),
        )
        mode = H.mode_of(kept, red)
        ub, uv, ul, uu, ur = H.UNIVERSAL_KNOBS[mode]
        universal = H.Knobs(mode, ub, uv, ul, uu, 1, ur, arch != "gfx90a")
        if (not kept and not red) or H.check_knobs(kept, red, a, universal, arch=arch):
            # a single element, or counts past the classes' 32-bit indices: no class runs it
            # (the generic kernels do)
            assert ch == []
            continue
        assert ch, (kept, red, a, op, una)
        for k, s in ch:
            assert H.check_knobs(kept, red, a, k, arch=arch) is None
            p = H.plan(
                kept, red, a, a, op, k, una=una, arch=arch, nr=s.nr, idx64=s.idx64
            )
            assert p.spec == s
            key = s.kernel_name()
            if key not in built:
                built[key] = {q.name for q in build_rreduce(s).params}
            want = set(p.args) | {"A", "C", "D", "W", "alpha", "beta", "opC"}
            if s.una:
                want.add("opA")
            assert built[key] == want, key
            if p.final is not None:
                fk = {q.name for q in build_rreduce_final(p.final).params}
                assert fk == set(p.final_args) | {"W", "C", "D", "alpha", "beta", "opC"}


def test_tuned_entries_plan():
    """Every tuned problem plans with its knobs and gets the class the table records."""
    table = H.load_table()
    for e in table["entries"]:
        k = H.knobs_of(e["knobs"])
        p = H.plan(e["kept"], e["red"], e["a"], e["d"], e["op"], k, arch=e["arch"])
        assert H.class_dict(p.spec, e["arch"]) == e["class"]
        assert p.grid[1] == k.split and p.block[0] == k.block
        if k.split > 1:
            assert p.workspace == math.prod(x[0] for x in e["kept"]) * k.split * (
                8 if e["a"] == "f64" else 4
            )


# hipTensor's planner mirrors untuned_choices: rocke_backend_test's RockeTunedPick.SameAsRocke
# plans these problems (packed column-major A and D, gfx950 with 256 compute units) and expects
# the same first candidate. When the table's palette or prototypes change, regenerate both lists
# (the assertion message prints the new pick).
PICKS_GFX950 = [
    (
        "f16",
        "add",
        "ab",
        "b",
        (3000, 70),
        False,
        "rreduce:hiptensor_aot_rreduce_row_add_f16_f16_r1_b256_v8_l256_u4_nt_i32:split2",
    ),
    (
        "f32",
        "max",
        "ab",
        "a",
        (70, 3000),
        False,
        "rreduce:hiptensor_aot_rreduce_col_max_f32_f32_r1_b512_v2_l32_u8_nt_i32:split128",
    ),
    (
        "bf16",
        "min",
        "abc",
        "b",
        (100, 50, 30),
        False,
        "rreduce:hiptensor_aot_rreduce_row_min_bf16_bf16_r2_b512_v4_l512_u4_nt_i32:split2",
    ),
    (
        "f64",
        "add",
        "abcd",
        "bd",
        (17, 33, 9, 40),
        False,
        "rreduce:hiptensor_aot_rreduce_row_add_f64_f64_r8_b256_v1_l1_u4_nt_i32:split128",
    ),
    (
        "f32",
        "add",
        "a",
        "",
        (1000003,),
        False,
        "rreduce:hiptensor_aot_rreduce_row_add_f32_f32_r8_b256_v1_l1_u4_nt_i32:split1024",
    ),
    (
        "f16",
        "max",
        "abc",
        "ac",
        (8, 2000, 12),
        False,
        "rreduce:hiptensor_aot_rreduce_col_max_f16_f16_r2_b256_v8_l1_w8_u4_nt_i32:split128",
    ),
    (
        "f32",
        "mul",
        "abc",
        "c",
        (640, 3, 50),
        True,
        "rreduce:hiptensor_aot_rreduce_row_mul_f32_f32_r2_b256_v4_l32_u8_nt_una_i32:split8",
    ),
    (
        "f64",
        "max",
        "abcde",
        "eb",
        (6, 7, 8, 9, 10),
        False,
        "rreduce:hiptensor_aot_rreduce_row_max_f64_f64_r2_b1024_v2_l16_u4_nt_i32:split8",
    ),
]


def _packed_problem(modes_a, ext, modes_d):
    """hipTensor's canonical form of a reduction of packed column-major A and D."""
    sa, acc = {}, 1
    for m, e in zip(modes_a, ext):
        sa[m] = acc
        acc *= e
    ex = dict(zip(modes_a, ext))
    sd, acc = {}, 1
    for m in modes_d:
        sd[m] = acc
        acc *= ex[m]
    kd = canonicalize([(ex[m], [sa[m], sd[m]]) for m in modes_d], 0)
    rd = canonicalize([(ex[m], [sa[m]]) for m in modes_a if m not in modes_d], 0)
    return [[e, s[0], s[1]] for e, s in kd], [[e, s[0]] for e, s in rd]


@pytest.mark.parametrize(
    "case", PICKS_GFX950, ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}-{c[3]}"
)
def test_untuned_pick_is_hiptensors(case):
    a, op, modes_a, modes_d, ext, una, want = case
    table = H.load_table()
    kept, red = _packed_problem(modes_a, ext, modes_d)
    key = H.problem_key(a, a, op, kept, red)
    assert all(
        H.problem_key(e["a"], e["d"], e["op"], e["kept"], e["red"]) != key
        for e in table["entries"]
        if e["arch"] == "gfx950"
    ), "a tuned problem: pick an untuned one"
    ch = H.untuned_choices(
        kept,
        red,
        a,
        a,
        op,
        H.pack_classes("gfx950"),
        H.prototypes(table),
        una=una,
        cus=256,
        arch="gfx950",
    )
    k, s = ch[0]
    assert f"rreduce:{s.kernel_name()}:split{k.split}" == want
