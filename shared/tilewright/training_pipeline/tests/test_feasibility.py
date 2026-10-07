# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import random

import pytest

from lib import feasibility as fz
from lib import grid, mlrec
from lib.hardware import DeviceHardware, arch_constants

from test_mlrec import random_bundle


def cfg(mt=(128, 128, 64), mi=(16, 16, 32), cha=0, chb=0, grvw=(8, 8)):
    return dict(
        mt_m=mt[0],
        mt_n=mt[1],
        mt_k=mt[2],
        mi_m=mi[0],
        mi_n=mi[1],
        mi_k=mi[2],
        occupancy=1,
        cache_hints_a=cha,
        cache_hints_b=chb,
        grvw_a=grvw[0],
        grvw_b=grvw[1],
        gwvw_d=4,
    )


def prob(m, n, k, batch=1, ta="T", tb="N", dt="bfloat16"):
    return dict(
        m=m,
        n=n,
        k=k,
        batch=batch,
        a_transpose=ta,
        b_transpose=tb,
        a_dtype=dt,
        b_dtype=dt,
        c_dtype=dt,
        d_dtype=dt,
        mi_dtype=dt,
    )


def ok(p, c, nt=(False, False), lds=1 << 20):
    return fz.passes_gates(
        p, c, lds_bytes=lds, nt_a_available=nt[0], nt_b_available=nt[1]
    )


def test_dot2_only_for_tiny_m():
    dot2 = cfg(mi=(1, 1, 64))
    assert ok(prob(2, 4096, 100), dot2)
    assert not ok(prob(3, 4096, 100), dot2)


def test_contiguous_free_dimension_holds_one_global_read_vector():
    wide_a, wide_b = cfg(grvw=(8, 1)), cfg(grvw=(1, 8))
    assert not ok(prob(7, 64, 256, ta="N", tb="N"), wide_a)
    assert ok(prob(8, 64, 256, ta="N", tb="N"), wide_a)
    assert ok(prob(1, 64, 256, ta="T", tb="N"), wide_a)
    assert not ok(prob(64, 7, 256, ta="T", tb="T"), wide_b)
    assert ok(prob(64, 8, 256, ta="T", tb="T"), wide_b)
    assert ok(prob(64, 1, 256, ta="T", tb="N"), wide_b)
    assert ok(prob(4, 4, 256, ta="N", tb="T", dt="int8x4"), cfg(grvw=(8, 8)))


def test_small_batched_problem_needs_one_tile():
    assert not ok(prob(200, 200, 512, batch=4), cfg(mt=(128, 256, 64)))
    assert ok(prob(200, 200, 512, batch=4), cfg(mt=(256, 256, 64)))
    assert ok(prob(200, 200, 512, batch=1), cfg(mt=(128, 128, 64)))
    assert ok(prob(200, 200, 1024, batch=4), cfg(mt=(128, 128, 64)))


def test_skinny_b_branch_needs_nt_b_only_when_available():
    p = prob(64, 8192, 1024, tb="N")
    assert ok(p, cfg(), nt=(False, False))
    assert not ok(p, cfg(), nt=(False, True))
    assert ok(p, cfg(chb=4), nt=(False, True))
    assert ok(p, cfg(), nt=(True, False))


def test_skinny_a_branch_needs_nt_a_only_when_available():
    p = prob(8192, 64, 1024, ta="T", tb="T")
    assert ok(p, cfg(), nt=(False, True))
    assert not ok(p, cfg(), nt=(True, False))
    assert ok(p, cfg(cha=4), nt=(True, False))


def test_single_operand_pools():
    a_only = [cfg(cha=4), cfg()]
    b_only = [cfg(chb=4), cfg()]
    assert fz.non_temporal_availability(a_only) == (True, False)
    assert fz.non_temporal_availability(b_only) == (False, True)
    skinny_b = prob(64, 8192, 1024, tb="N")
    assert ok(skinny_b, cfg(), fz.non_temporal_availability(a_only))
    assert not ok(skinny_b, cfg(), fz.non_temporal_availability(b_only))


@pytest.mark.parametrize("hint", [1, 2, 3])
def test_temporal_hints_are_not_non_temporal(hint):
    pool = [cfg(cha=hint, chb=hint), cfg()]
    assert fz.non_temporal_availability(pool) == (False, False)
    square = prob(2048, 2048, 1024)
    assert not ok(square, cfg(cha=hint))
    assert not ok(square, cfg(chb=hint))
    assert ok(square, cfg())


def test_unaligned_k_rejects_any_hint():
    p = prob(64, 8192, 1000)
    assert ok(p, cfg(), nt=(True, True))
    assert not ok(p, cfg(chb=4), nt=(True, True))
    assert not ok(prob(64, 8192, 1024), cfg(mt=(128, 128, 60), chb=4), nt=(True, True))


def test_lds_gate():
    c = cfg(mt=(128, 128, 64))
    p = prob(1024, 1024, 1024)
    assert ok(p, c, lds=2 * 128 * 64 * 2)
    assert not ok(p, c, lds=2 * 128 * 64 * 2 - 1)
    assert fz.lds_fits(128, 128, 64, "float8", "half", 128 * 64 * 3)
    assert not fz.lds_fits(16, 16, 16, "complexhalf", "half", 1 << 30)


def test_rule_bits():
    assert fz.rule_bits("f8") == 8 and fz.rule_bits("xf32") == 32
    assert fz.rule_bits("float8bfloat8") == 16 and fz.rule_bits("float4") == 16
    assert fz.lds_bits("float4") == 4 and fz.lds_bits("float8bfloat8") == 8


def test_matches_engine_on_random_pools():
    tw = pytest.importorskip("tilewright")
    from lib import evaluate as ev

    bundle = random_bundle(
        grid.all_cell_labels(),
        signatures={lbl: [] for lbl in grid.all_cell_labels()},
    )
    model = tw.load_model_from_memory(
        mlrec.write_model(bundle, None, "gfx950", arch_constants("gfx950"), "fp32")
    )
    rng = random.Random(3)
    dtypes = ["bfloat16", "half", "float", "xfloat32", "float8", "bfloat8"]
    dtypes += ["float8_fnuz", "float8bfloat8", "float4", "int8", "double", "float6"]
    dtypes += ["int8x4"]
    partial = 0
    for _ in range(600):
        mode = rng.choice(["none", "a", "b", "both", "temporal"])
        pool = []
        for _ in range(rng.randint(1, 30)):
            cha = chb = 0
            if mode in ("a", "both") and rng.random() < 0.4:
                cha = 4
            if mode in ("b", "both") and rng.random() < 0.4:
                chb = 4
            if mode == "temporal":
                cha, chb = rng.choice([0, 1, 2, 3]), rng.choice([0, 1, 2, 3])
            pool.append(
                cfg(
                    mt=(
                        rng.choice([16, 32, 64, 128, 256, 512]),
                        rng.choice([16, 32, 64, 128, 256]),
                        rng.choice([16, 32, 64, 128, 256]),
                    ),
                    mi=rng.choice(
                        [(16, 16, 32), (32, 32, 16), (1, 1, 64), (16, 16, 128)]
                    ),
                    cha=cha,
                    chb=chb,
                    grvw=(rng.choice([1, 2, 4, 8, 16]), rng.choice([1, 2, 4, 8, 16])),
                )
            )
        a = rng.choice(dtypes)
        p = dict(
            prob(
                rng.choice([1, 2, 3, 8, 64, 200, 256, 257, 1000, 5000]),
                rng.choice([1, 16, 100, 256, 257, 3000, 40000]),
                rng.choice([16, 32, 64, 96, 512, 1000, 1023, 1024, 4096]),
                batch=rng.choice([1, 1, 2, 16]),
                ta=rng.choice("TN"),
                tb=rng.choice("TN"),
                dt=a,
            ),
            b_dtype=rng.choice(dtypes),
        )
        hw = DeviceHardware(100, rng.choice([16384, 65536, 163840]), 4 << 20)
        cs = tw.CandidateSet(
            model, [ev.make_config(tw, k, i) for i, k in enumerate(pool)]
        )
        res = cs.rank(ev.make_problem(tw, p), ev.make_hardware(tw, hw), 0)
        engine = {r.config_index for r in res if r.scored}
        nt = fz.non_temporal_availability(pool)
        python = {i for i, k in enumerate(pool) if ok(p, k, nt, hw.lds_bytes)}
        assert engine == python, (p, mode)
        partial += 0 < len(python) < len(pool)
    assert partial > 50


def test_soft_rules_by_version():
    assert fz.soft_rules({}, 2) == {
        "small_batched_rule": True,
        "dot2_rule": True,
        "non_temporal_rule": True,
    }
    assert fz.soft_rules({"non_temporal_rule": False}, 2)["non_temporal_rule"] is False
    assert not any(fz.soft_rules({}, 3).values())
    assert fz.soft_rules({"dot2_rule": True}, 3)["dot2_rule"] is True
    dot2, p = cfg(mi=(1, 1, 64)), prob(3, 4096, 100)
    gates = dict(lds_bytes=1 << 20, nt_a_available=False, nt_b_available=False)
    assert fz.passes_gates(p, dot2, **gates, **fz.soft_rules({}, 3))
    small = prob(200, 200, 512, batch=4)
    assert fz.passes_gates(small, cfg(mt=(128, 256, 64)), **gates, dot2_rule=False,
                           small_batched_rule=False)  # fmt: skip
    wide_a, narrow = cfg(grvw=(8, 1)), prob(7, 64, 256, ta="N", tb="N")
    assert fz.passes_gates(narrow, wide_a, **gates, **fz.soft_rules({}, 3))
    assert not fz.passes_gates(narrow, wide_a, **gates, **fz.soft_rules({}, 2))
    assert not fz.passes_gates(narrow, wide_a, **gates)


def test_v3_soft_rules_match_engine():
    tw = pytest.importorskip("tilewright")
    from lib import evaluate as ev
    from test_v3 import v3_bundle

    rng = random.Random(5)
    dtypes = ["bfloat16", "half", "float", "float8", "int8x4"]
    for flags in range(8):
        serving = {
            "small_batched_rule": bool(flags & 1),
            "dot2_rule": bool(flags & 2),
            "non_temporal_rule": bool(flags & 4),
            "whitelist_tier": False,
        }
        bundle = v3_bundle(labels=["Large|Large|LargeK|Bnone"], serving=serving)
        model = tw.load_model_from_memory(
            mlrec.write_model(bundle, None, "gfx950", arch_constants("gfx950"), "fp32")
        )
        rules = fz.soft_rules(serving, 3)
        for _ in range(150):
            pool = [
                cfg(
                    mt=(rng.choice([16, 64, 128, 256]), rng.choice([16, 64, 256]), 64),
                    mi=rng.choice([(16, 16, 32), (1, 1, 64)]),
                    cha=rng.choice([0, 0, 4, 1]),
                    chb=rng.choice([0, 0, 4]),
                    grvw=(rng.choice([1, 8, 16]), rng.choice([1, 8])),
                )
                for _ in range(rng.randint(1, 20))
            ]
            p = prob(
                rng.choice([1, 2, 3, 8, 200, 1000]),
                rng.choice([1, 16, 200, 3000, 40000]),
                rng.choice([64, 512, 1024, 4096]),
                batch=rng.choice([1, 4]),
                ta=rng.choice("TN"),
                tb=rng.choice("TN"),
                dt=rng.choice(dtypes),
            )
            hw = DeviceHardware(100, 65536, 4 << 20)
            cs = tw.CandidateSet(
                model, [ev.make_config(tw, k, i) for i, k in enumerate(pool)]
            )
            res = cs.rank(ev.make_problem(tw, p), ev.make_hardware(tw, hw), 0)
            engine = {r.config_index for r in res if r.scored}
            nt = fz.non_temporal_availability(pool)
            python = {
                i
                for i, k in enumerate(pool)
                if fz.passes_gates(
                    p, k, lds_bytes=hw.lds_bytes, nt_a_available=nt[0],
                    nt_b_available=nt[1], **rules,
                )  # fmt: skip
            }
            assert engine == python, (p, flags)
