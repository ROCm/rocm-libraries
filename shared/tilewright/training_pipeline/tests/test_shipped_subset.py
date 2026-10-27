# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shipped-subset simulation (lib/shipped_subset.py) against the engine
and lib/evaluate, its greedy selection against brute force, and the
choose_shipped_subset.py JSON / report contract, on synthetic data."""

import json
import math
import os
from html import escape as html_escape
import random
import stat
import subprocess
import sys

import msgpack
import numpy as np
import pytest

import bench_fakes as bf
import choose_shipped_subset as css
from lib import evaluate as ev
from lib import kernel_identity as ki
from lib import feasibility
from lib import features as fs
from lib import hipblaslt_launch as hl
from lib import run_analysis as ra
from lib import shipped_subset as ss
from lib import stage05_split as s5split
from lib.hardware import DeviceHardware
from test_v3 import LABELS, random_launch_attributes, v3_bundle, write

# Made-up device values.
HW = DeviceHardware(n_cu=300, lds_bytes=163840, l2_bytes=4 << 20)
SPEC = (fs.AttributeFeature("param_a"),)
TILES = [
    (64, 64, 64),
    (128, 128, 64),
    (256, 128, 64),
    (128, 256, 32),
    (32, 32, 128),
    (16, 16, 256),
    (256, 256, 128),
    (64, 128, 64),
    (128, 64, 64),
    (64, 256, 64),
]
MIS = [(16, 16, 32), (32, 32, 16)]
SHAPES = [
    (4096, 4096, 4096, 1),
    (8192, 1024, 2048, 1),
    (16, 32, 64, 1),
    (64, 128, 4096, 1),
    (512, 512, 512, 1),
    (128, 64, 256, 8),
    (2048, 16, 8192, 1),
    (300, 700, 1000, 1),
    (1, 4096, 1024, 1),
    (1024, 1024, 128, 4),
    (100, 100, 100, 1),
    (6000, 3000, 64, 1),
]
CONTEXTS = [
    fs.EXCLUSIVE,
    fs.ExecutionContext(128),
    fs.ExecutionContext(64, "dynamic"),
    fs.ExecutionContext(32),
]


def kernels(rng, n, sid0, hints=((0, 0),)):
    out = []
    for i in range(n):
        mt = rng.choice(TILES)
        mi = rng.choice(MIS)
        cha, chb = rng.choice(hints)
        out.append(
            dict(
                mt_m=mt[0], mt_n=mt[1], mt_k=mt[2], mi_m=mi[0], mi_n=mi[1], mi_k=mi[2],
                occupancy=rng.choice([1, 2]), cache_hints_a=cha, cache_hints_b=chb,
                grvw_a=8, grvw_b=8, gwvw_d=4, sol_idx_global=sid0 + i,
                attributes=random_launch_attributes(
                    rng, {"param_a": rng.choice([0, 1])}
                ),
            )
        )  # fmt: skip
    return out


def gemms(rng, kern, n_extra_rows=0):
    """ra.Gemm rows over `kern`: random latencies, some screened rows, some
    missing kernels, planned kernel lists on some GEMMs, a fallback pick on
    most."""
    out = []
    sids = [int(k["sol_idx_global"]) for k in kern]
    for m, n, k, b in SHAPES:
        for ctx in CONTEXTS:
            measured = [s for s in sids if rng.random() < 0.7]
            us = np.array([rng.uniform(5.0, 30.0) for _ in measured])
            skip = np.array([rng.random() < 0.15 for _ in measured])
            if skip.all():
                skip[0] = False
            listed = None
            if rng.random() < 0.5:
                listed = frozenset(measured + [s for s in sids if rng.random() < 0.2])
            tested = us[~skip]
            fb = rng.choice(measured) if rng.random() < 0.8 else None
            key = (
                m,
                n,
                k,
                b,
                "T",
                "N",
                "bf16_r",
                "bf16_r",
                "bf16_r",
                "bf16_r",
                "f32_r",
            )
            if not ctx.is_exclusive():
                key += (ctx.cu_budget, ctx.schedule)
            out.append(
                ra.Gemm(
                    key=key,
                    leaf="",
                    sids=np.array(measured, dtype=np.int64),
                    us=us,
                    skip=skip,
                    winner_us=float(tested.min()),
                    listed=listed,
                    fallback=fb,
                )
            )
    return out


def model(cascade=4, whitelist=None, nt=False, seed=0, launch_keys=False):
    bundle = v3_bundle(labels=LABELS, spec=SPEC, seed=seed)
    bundle["serving"] = {
        "cascade_keys": cascade,
        "non_temporal_rule": nt,
        "launch_keys": launch_keys,
    }
    if whitelist:
        for label, hashes in whitelist.items():
            bundle["models"][label]["view_key_hashes"] = sorted(hashes)
    return write(bundle)


@pytest.fixture(scope="module")
def tw():
    return pytest.importorskip("tilewright")


def setup(tw, *, seed=0, hints=((0, 0),), bank_hints=None, pm="timed", **mk):
    rng = random.Random(seed)
    pool = kernels(rng, 24, 1000, hints)
    pool += [dict(pool[2], occupancy=3, sol_idx_global=1100)]
    bank = kernels(rng, 50, 2000, bank_hints or hints)
    bank += [dict(pool[0], sol_idx_global=1000)]
    bank += [dict(pool[3], occupancy=4, sol_idx_global=2100)]
    data = model(**mk)
    mv = ss.model_view(data, HW, tw)
    u = ss.build_universe(pool, bank, mv.spec, mv.key_launch)
    t = ss.build_table(gemms(rng, u.kernels), u, mv, pick_measurement=pm)
    return u, mv, t, data


def subsets(u, rng, n=4):
    out = [u.present(), u.present(u.bank)]
    for _ in range(n):
        size = rng.randint(1, len(u.bank) - 1)
        out.append(u.present(rng.sample(list(u.bank), size)))
    return out


# ── feasibility and the universe ─────────────────────────────────────────────


def test_feasible_keys_match_passes_gates():
    rng = random.Random(4)
    kern = kernels(rng, 60, 0, ((0, 0), (4, 0), (0, 4), (4, 4), (1, 0)))
    kern += [dict(kern[0], mi_m=1, mi_n=1, mi_k=64, sol_idx_global=999)]
    u = ss.build_universe(kern, [], ())
    probs = []
    for m, n, k, b in SHAPES + [(2, 64, 64, 4), (3, 3, 4096, 1), (4096, 8, 2048, 1)]:
        for ta, tb in (("T", "N"), ("N", "T"), ("N", "N")):
            probs.append(
                fs.problem_kwargs_from_row(
                    dict(
                        m=m, n=n, k=k, batch_count=b, transA=ta, transB=tb,
                        a_type="bf16_r", b_type="bf16_r", c_type="bf16_r",
                        d_type="bf16_r", compute_type="f32_r",
                    )
                )
            )  # fmt: skip
    first = [u.kernels[i] for i in u.members[:, 0]]
    for rules in (
        feasibility.soft_rules({}, 3),
        feasibility.soft_rules(
            {"small_batched_rule": True, "dot2_rule": True, "non_temporal_rule": True},
            3,
        ),
        feasibility.soft_rules({}, 2),
    ):
        for nt_a, nt_b in ((False, False), (True, False), (False, True), (True, True)):
            got = ss.feasible_keys(
                u, probs, lds_bytes=65536, rules=rules, nt_a=nt_a, nt_b=nt_b
            )
            want = [
                [
                    feasibility.passes_gates(
                        p, fs.config_kwargs_from_row(c), lds_bytes=65536,
                        nt_a_available=nt_a, nt_b_available=nt_b, **rules,
                    )
                    for c in first
                ]
                for p in probs
            ]  # fmt: skip
            assert got.tolist() == want


def test_universe_keys_and_members():
    rng = random.Random(1)
    pool = kernels(rng, 6, 10)
    bank = [
        dict(pool[1], occupancy=2, sol_idx_global=20),
        dict(pool[4], sol_idx_global=10),
    ]
    bank += kernels(rng, 3, 30, ((4, 0),))
    u = ss.build_universe(pool, bank, SPEC)
    assert u.n_pool == 6 and u.n == 10
    assert u.sids.tolist() == [10, 11, 12, 13, 14, 15, 20, 30, 31, 32]
    for i, k in enumerate(u.kernels):
        vk = fs.v3_view_key(fs.config_kwargs_from_row(k), k["attributes"], SPEC)
        same = [
            j
            for j, o in enumerate(u.kernels)
            if fs.v3_view_key(fs.config_kwargs_from_row(o), o["attributes"], SPEC) == vk
        ]
        row = u.members[u.key[i]]
        assert row[row >= 0].tolist() == same
    assert u.key[6] == u.key[1]
    p = u.present([7])
    assert (
        p.sum() == 7
        and u.flags(p) == (True, False)
        and u.flags(u.present())
        == (
            False,
            False,
        )
    )
    fp = u.first_present(u.present([6]))
    assert fp[u.key[6]] == 1 and fp[u.key[8]] == -1
    with pytest.raises(ValueError):
        u.present([0])
    assert u.n_views == u.n_keys and u.view.tolist() == list(range(u.n_keys))


def test_universe_groups_launch_classes_as_the_engine():
    launch = hl.HipblasltLaunch("gfx950")
    rng = random.Random(6)
    pool = kernels(rng, 30, 0)
    bank = [dict(k, sol_idx_global=100 + i) for i, k in enumerate(kernels(rng, 30, 0))]
    u = ss.build_universe(pool, bank, SPEC, launch)
    plain = ss.build_universe(pool, bank, SPEC)
    assert u.n_views == plain.n_keys < u.n_keys
    for i, k in enumerate(u.kernels):
        ck = fs.v3_candidate_key(
            fs.config_kwargs_from_row(k), k["attributes"], SPEC, launch
        )
        same = [
            j
            for j, o in enumerate(u.kernels)
            if fs.v3_candidate_key(
                fs.config_kwargs_from_row(o), o["attributes"], SPEC, launch
            )
            == ck
        ]
        row = u.members[u.key[i]]
        assert row[row >= 0].tolist() == same
        assert u.view[u.key[i]] == plain.key[i]
        assert int(u.hashes[u.key[i]]) == fs.view_key_hash(ck[:-1])
    for v in range(u.n_views):
        keys = np.flatnonzero(u.view == v)
        assert u.view_first[v] == keys.min()
    has = u.first_present(u.present(u.bank[::3])) >= 0
    want = [[has[u.view == v].any() for v in range(u.n_views)]]
    assert u.any_by_view(has[None, :]).tolist() == want


# ── against the engine ──────────────────────────────────────────────────────


@pytest.mark.parametrize("launch_keys", [False, True])
def test_full_scores_are_the_engines(tw, launch_keys):
    u, mv, t, data = setup(tw, launch_keys=launch_keys)
    assert mv.launch_keys == launch_keys
    assert (u.n_views < u.n_keys) == launch_keys
    m = ev.load_model_bytes(data, tw)
    hw = ev.make_hardware(tw, HW)
    rng = random.Random(2)
    checked = 0
    for present in subsets(u, rng):
        idx = np.flatnonzero(present)
        cs = ev.candidate_set(tw, m, [u.kernels[i] for i in idx], HW)
        fp = u.first_present(present)
        for i, g in enumerate(t.gemms):
            ranked = cs.rank(
                ev.make_problem(tw, t.problems[i]), hw, len(idx),
                ev.make_context(tw, g.context),
            )  # fmt: skip
            for r in ranked:
                if not r.scored:
                    continue
                c = idx[int(r.config_index)]
                assert t.full[i, fp[u.key[c]]] == np.float32(r.score)
                checked += 1
    assert checked > 1000
    # The engine ranks every context of a schedule the model lists, a CU
    # budget below its range at the nearest budget in it; it ranks none of
    # a schedule the model does not list.
    assert (t.cell >= 0).all()
    unlisted = [ss.ScoreJob(t.problems[0], fs.ExecutionContext(0, "auto"))]
    emb, full, cell = ss.engine_scores(mv, u, unlisted)
    assert cell.tolist() == [-1] and np.isnan(full).all() and np.isnan(emb).all()


def test_a_budget_below_the_range_is_simulated_at_its_scoring_context(tw):
    """The simulator scores a CU budget below the model's range as the
    engine does: exactly as the budget it ranks it at, its embedding term
    included."""
    u, mv, t, data = setup(tw)
    hw = ev.make_hardware(tw, HW)
    low = fs.ExecutionContext(32)
    at = tw.scoring_context(ev.load_model_bytes(data, tw), ev.make_context(tw, low), hw)
    assert at is not None and int(at.cu_budget) not in (0, 32)
    nearest = fs.ExecutionContext(int(at.cu_budget))
    for p in t.problems[:8]:
        got = ss.engine_scores(mv, u, [ss.ScoreJob(p, low)])
        want = ss.engine_scores(mv, u, [ss.ScoreJob(p, nearest)])
        assert int(got[2][0]) >= 0
        for x, y in zip(got, want):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize(
    "mk",
    [
        dict(cascade=4),
        dict(cascade=0),
        dict(cascade=3, nt=True, hints=((0, 0), (4, 0)), bank_hints=((0, 0), (0, 4))),
        dict(cascade=4, whitelist="some"),
        dict(cascade=4, launch_keys=True),
        dict(cascade=0, launch_keys=True),
        dict(
            cascade=3,
            nt=True,
            hints=((0, 0), (4, 0)),
            bank_hints=((0, 0), (0, 4)),
            launch_keys=True,
        ),
        dict(cascade=4, whitelist="some", launch_keys=True),
    ],
)
def test_cascade_and_picks_match_the_engine(tw, mk):
    mk = dict(mk)
    if mk.get("whitelist") == "some":
        rng = random.Random(0)
        pool = kernels(rng, 24, 1000)
        hashes = {
            fs.view_key_hash(
                fs.v3_view_key(fs.config_kwargs_from_row(k), k["attributes"], SPEC)
            )
            for k in pool[:4] + kernels(random.Random(9), 5, 0)
        }
        mk["whitelist"] = {LABELS[0]: hashes, LABELS[2]: hashes}
    u, mv, t, data = setup(tw, **mk)
    rows = range(t.n)
    for present in subsets(u, random.Random(5)):
        cp = ss.cascade_parity(t, present, rows, data=data, hardware=HW, tw=tw)
        assert cp["n_same"] == cp["n"], cp
        for pm in ev.PICK_MEASUREMENTS:
            tp = (
                t
                if pm == "timed"
                else ss.build_table(t.gemms, u, mv, pick_measurement=pm)
            )
            p = ss.parity(
                tp, present, rows, pick_measurement=pm, data=data, hardware=HW, tw=tw
            )
            assert p["n_same"] == p["n"] == t.n, p


# ── selection ───────────────────────────────────────────────────────────────


def brute(t, present):
    return ss.State(t, present).charged


@pytest.mark.parametrize(
    "mk",
    [
        dict(cascade=4),
        dict(cascade=2, nt=True, hints=((0, 0),), bank_hints=((0, 0), (4, 0), (0, 4))),
        dict(cascade=4, whitelist="pool"),
        dict(cascade=4, launch_keys=True),
        dict(
            cascade=2,
            nt=True,
            hints=((0, 0),),
            bank_hints=((0, 0), (4, 0), (0, 4)),
            launch_keys=True,
        ),
        dict(cascade=4, whitelist="pool", launch_keys=True),
    ],
)
def test_incremental_gains_are_exact(tw, mk):
    mk = dict(mk)
    if mk.get("whitelist") == "pool":
        rng = random.Random(0)
        pool = kernels(rng, 24, 1000)
        hashes = {
            fs.view_key_hash(
                fs.v3_view_key(fs.config_kwargs_from_row(k), k["attributes"], SPEC)
            )
            for k in pool[5:9] + kernels(random.Random(0), 50, 2000)[:6]
        }
        mk["whitelist"] = {lab: hashes for lab in LABELS[:2]}
    u, mv, t, data = setup(tw, **mk)
    rng = random.Random(7)
    for present in subsets(u, rng, 3)[:-1] + [u.present()]:
        st = ss.State(t, present)
        absent = np.array([c for c in u.bank if not present[c]])
        if not absent.size:
            continue
        if u.flags(present) != st.flags:
            continue
        rr, cc, dd = st.deltas(np.arange(t.n), absent)
        for j, c in enumerate(absent):
            nxt = present.copy()
            nxt[c] = True
            if u.flags(nxt) != u.flags(present) and mv.rules["non_temporal_rule"]:
                continue
            want = brute(t, nxt)
            ok = t.winner > 0
            d = np.zeros(t.n)
            d[rr[cc == j]] = dd[cc == j]
            got_cost = np.exp(np.log(st.charged) - d)
            assert np.allclose(got_cost[ok], want[ok], rtol=1e-9), (c, j)
        c = int(absent[0])
        st2 = ss.State(t, present)
        st2.add(c)
        nxt = present.copy()
        nxt[c] = True
        if u.flags(nxt) == u.flags(present):
            fresh = ss.State(t, nxt)
            assert np.array_equal(st2.charged, fresh.charged)
            assert np.array_equal(st2.kept, fresh.kept)


@pytest.mark.parametrize("launch_keys", [False, True])
def test_greedy_takes_the_best_step(tw, launch_keys):
    u, mv, t, data = setup(
        tw, cascade=3, nt=True, hints=((0, 0),), bank_hints=((0, 0), (4, 0)),
        launch_keys=launch_keys,
    )  # fmt: skip
    g = ss.greedy_order(t, 6)
    assert len(g.order) == 6 == len(set(g.order))
    present = u.present()
    for step, c in enumerate(g.order):
        base = ss.State(t, present).objective()
        assert math.isclose(base, g.objective[step], rel_tol=1e-9, abs_tol=1e-9)
        gains = {}
        for b in u.bank:
            if present[b]:
                continue
            nxt = present.copy()
            nxt[b] = True
            gains[int(b)] = ss.State(t, nxt).objective() - base
        assert gains[c] >= max(gains.values()) - 1e-6
        present[c] = True
    assert math.isclose(
        ss.State(t, present).objective(), g.objective[-1], rel_tol=1e-9, abs_tol=1e-9
    )


def test_identity_ceiling_is_lib_identity_ceiling_under_the_models_grouping(tw):
    from lib import identity_ceiling as ic

    got = {}
    for launch_keys in (False, True):
        u, mv, t, data = setup(tw, launch_keys=launch_keys)
        grouping = ic.engine_grouping(mv.spec, mv.key_launch)
        dicts = [ss.gemm_dict(g) for g in t.gemms]
        for present in subsets(u, random.Random(4), 2):
            shipped = u.shipped(present)
            want = ic.ceiling(
                dicts, grouping, pools={ev.pool_key(d): shipped for d in dicts}
            )
            mine = ss.identity_ceiling(t, present)
            assert mine["geomean"] == want.geomean
            assert (mine["n_groups"], mine["n_losing"]) == (
                want.n_groups,
                want.n_losing,
            )
            got.setdefault(launch_keys, []).append(mine)
            assert ss.set_metrics(t, present)["identity_ceiling"] == want.geomean
    for off, on in zip(got[False], got[True]):
        assert on["n_groups"] > off["n_groups"]
        assert on["geomean"] >= off["geomean"] - 1e-12


def test_win_order_counts_wins():
    rng = random.Random(3)
    pool = kernels(rng, 3, 0)
    bank = kernels(rng, 4, 10)
    u = ss.build_universe(pool, bank, ())
    cost = np.full((3, u.n), np.inf, dtype=np.float32)
    cost[:, 3] = [1.0, 2.0, 2.0]
    cost[:, 4] = [1.0, 1.0, 1.0]
    cost[:, 5] = [2.0, 1.0, np.inf]
    cost[:, 6] = [4.0, 4.0, 4.0]
    t = ss.Table(
        gemms=[None] * 3, contexts=["x"] * 3, problems=[{}] * 3,
        winner=np.ones(3), cost=cost, fallback_us=np.full(3, np.nan),
        charge=np.ones(3), emb=np.zeros((3, u.n_keys), np.float32),
        full=np.zeros((3, u.n), np.float32), cell=np.zeros(3, np.int32),
        lds_bytes=1, rules={}, cascade=0, has_wl=np.zeros(1, bool),
        wl=np.zeros((1, u.n_keys), bool), u=u,
    )  # fmt: skip
    assert ss.win_order(t).tolist() == [4, 3, 5, 6]


def test_recommend_takes_the_smallest_close_size():
    def row(nb, eff, p99):
        return {
            "size": str(nb),
            "n_bank": nb,
            "held_out": {"sel_eff": eff},
            "latency_us": {"p99": p99} if p99 is not None else None,
        }

    rows = [
        row(0, 0.95, 8.0),
        row(32, 0.969, 9.0),
        row(64, 0.970, 9.9),
        row(128, 0.975, 12.0),
        row(256, 0.990, 19.0),
        row(512, 0.9905, 25.0),
        row(1024, 0.999, None),
    ]
    rec = ss.recommend(rows, (10.0, 20.0, 5.0))
    assert rec["10"]["n_bank"] == 32 and rec["10"]["best_held_out_sel_eff"] == 0.970
    assert rec["20"]["n_bank"] == 256
    assert rec["5"] is None


# ── selection time ──────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("TILEWRIGHT_SHIPPED_RANK_BINARY"),
    reason="set TILEWRIGHT_SHIPPED_RANK_BINARY to a built bench_shipped_rank",
)
@pytest.mark.parametrize("arch", ["gfx950", "gfx942"])
@pytest.mark.parametrize("dtype", ["bf16_r", "f64_r"])
def test_the_bench_launch_rule_matches_the_pipeline(tmp_path, arch, dtype):
    from lib import hipblaslt_launch as hl

    rng = random.Random(5)
    kern = []
    for i, k in enumerate(kernels(rng, 300, 0)):
        attrs = dict(k["attributes"])
        attrs.update(
            tile_processing_strategy=rng.choice([0, 1, 2, 2]),
            work_assignment=rng.choice([0, 1, 2, 2]),
            stream_k_atomic=rng.choice([0, 1]),
            work_group_x=rng.choice([0, 64, 256, 1 << 20]),
            work_group_y=rng.choice([1, 4, 1 << 12]),
            work_group_z=rng.choice([1, 2]),
            hand_written=rng.choice([0, 1]),
        )
        if attrs["tile_processing_strategy"] == 1:
            attrs["work_assignment"] = 0
        kern.append(dict(k, occupancy=rng.choice([1, 2, 3, 4, 8]), attributes=attrs))
    shapes = [
        (rng.choice([1, 7, 64, 300, 4096, 20000]), rng.choice([1, 33, 512, 8192, 30000]),
         rng.choice([1, 16, 64, 1000, 8192, 70000, 1 << 22]), rng.choice([1, 1, 4, 64]))
        for _ in range(60)
    ]  # fmt: skip
    prob = fs.problem_kwargs_from_row(
        dict(
            m=1, n=1, k=1, batch_count=1, transA="T", transB="N", a_type=dtype,
            b_type=dtype, c_type=dtype, d_type=dtype, compute_type="f32_r",
        )
    )  # fmt: skip
    inputs = tmp_path / "inputs.txt"
    inputs.write_text(ss.bench_inputs(prob, HW, shapes, kern))
    red = {"none": 0, "tree": 1, "parallel": 2, "atomic": 3}
    sched = {"default": 0, "dynamic": 1, "auto": 2}
    n, seen = 0, set()
    for cu, schedule in (
        (0, "default"),
        (96, "default"),
        (200, "auto"),
        (224, "dynamic"),
    ):
        out = subprocess.run(
            [
                os.environ["TILEWRIGHT_SHIPPED_RANK_BINARY"], "--inputs", str(inputs),
                "--dump-plans", str(len(shapes)), "--dump-arch", arch,
                "--dump-context", f"{cu},{sched[schedule]}",
            ],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        ctx = fs.ExecutionContext(cu_budget=cu, schedule=schedule)
        for line in out.splitlines():
            _, si, ci, dyn, grid, r, key = line.split()
            m, nn, k, b = shapes[int(si)]
            c = kern[int(ci)]
            want = hl.launch_plan(
                dict(m=m, n=nn, k=k, batch=b, mi_dtype=prob["mi_dtype"]),
                c, c["attributes"], ctx, HW.n_cu, arch,
            )  # fmt: skip
            assert (bool(int(dyn)), int(grid), int(r)) == (
                want.dynamic_queue,
                want.grid_wgs,
                red[want.partial_reduction],
            ), (shapes[int(si)], c, cu, schedule)
            assert int(key) == hl.launch_key(c, c["attributes"]), c
            n += 1
            seen.add((int(dyn), int(r)))
    assert n == 4 * len(shapes) * len(kern)
    assert {d for d, _ in seen} == {0, 1} and {r for _, r in seen} == {0, 1, 2, 3}


def test_launch_parity_reads_the_dumped_plans(tmp_path):
    from lib import hipblaslt_launch as hl

    rng = random.Random(2)
    kern = kernels(rng, 4, 0)
    for k in kern:
        k["attributes"] = dict(k["attributes"], tile_processing_strategy=2)
    shapes = [(4096, 4096, 512, 1), (64, 64, 8192, 1)]
    prob = {"mi_dtype": "bfloat16"}
    lines = []
    for si, (m, n, k, b) in enumerate(shapes):
        for ci, c in enumerate(kern):
            p = hl.launch_plan(
                dict(m=m, n=n, k=k, batch=b, mi_dtype="bfloat16"), c, c["attributes"],
                fs.ExecutionContext(), HW.n_cu, "gfx950",
            )  # fmt: skip
            grid = p.grid_wgs + (1 if (si, ci) == (1, 2) else 0)
            red = ss._REDUCTIONS[p.partial_reduction]
            key = hl.launch_key(c, c["attributes"]) + (1 if (si, ci) == (0, 3) else 0)
            lines.append(f"PLAN {si} {ci} {int(p.dynamic_queue)} {grid} {red} {key}")
    fake = tmp_path / "bench"
    fake.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "assert sys.argv[sys.argv.index('--dump-arch') + 1] == 'gfx950'\n"
        f"print({chr(10).join(lines)!r})\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    got = ss.launch_parity(
        fake,
        tmp_path / "i",
        prob,
        shapes,
        kern,
        HW,
        f"hipblaslt:gfx950:{hl.RULE_VERSION}",
    )
    assert (got["n"], got["n_same"], got["checked"]) == (8, 6, True)
    assert got["first_mismatch"] == {
        "shape": list(shapes[0]),
        "sol_idx": int(kern[3]["sol_idx_global"]),
    }
    other = ss.launch_parity(fake, tmp_path / "i", prob, shapes, kern, HW, "tl:x:9")
    assert not other["checked"]


def test_bench_inputs_and_pinned_run(tmp_path):
    rng = random.Random(0)
    kern = kernels(rng, 3, 0)
    prob = fs.problem_kwargs_from_row(
        dict(
            m=1, n=1, k=1, batch_count=1, transA="T", transB="N", a_type="bf16_r",
            b_type="bf16_r", c_type="bf16_r", d_type="bf16_r", compute_type="f32_r",
        )
    )  # fmt: skip
    text = ss.bench_inputs(prob, HW, [(64, 64, 64, 1), (1, 2, 3, 4)], kern)
    lines = text.splitlines()
    assert lines[0].split()[0] == "PROBLEM" and lines[0].split()[-2:] == ["0", "1"]
    assert lines[1] == f"HARDWARE {HW.n_cu} {HW.lds_bytes} {HW.l2_bytes}"
    assert lines[2:5] == ["SHAPES 2", "64 64 64 1", "1 2 3 4"]
    assert lines[5] == "CONFIGS 3"
    first = lines[6].split()
    assert [int(v) for v in first[:12]] == [int(kern[0][f]) for f in ss.BENCH_FIELDS]
    assert f"param_a={kern[0]['attributes']['param_a']}" in first

    fake = tmp_path / "bench"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "cpus = sorted(os.sched_getaffinity(0))\n"
        "assert len(cpus) == 1, cpus\n"
        "assert '--min-scored' in sys.argv and sys.argv[sys.argv.index('--iters') + 1] == '7'\n"
        "print('# header')\n"
        "print('candidate_set       2       1.00       2.50       2.60       3.00       4.25       5.00')\n"
        "print('rank_configs        2       9.00       9.00       9.00       9.00       9.00       9.00')\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    core = min(os.sched_getaffinity(0))
    got = ss.run_bench(
        fake, tmp_path / "m", tmp_path / "i", core=core, iters=7, warmup=1
    )
    assert got["p50"] == 2.5 and got["p99"] == 4.25 and got["shapes"] == 2


def test_the_bench_is_built_without_the_re_rank_memo(tmp_path, monkeypatch):
    # The memo can answer every other one of the bench's repeated rankings of
    # a shape; without it every timed call ranks afresh.
    calls = []
    monkeypatch.setattr(ss.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    src, out = tmp_path / "src", tmp_path / "build"
    assert ss.build_bench(src, out, jobs=3) == out / ss.BENCH_TARGET
    configure, build = calls
    assert configure[:5] == ["cmake", "-S", str(src), "-B", str(out)]
    assert "-DTILEWRIGHT_LEVER_MEMO=OFF" in configure
    assert "-DCMAKE_BUILD_TYPE=Release" in configure
    assert not [a for a in configure if a.startswith("-DTILEWRIGHT_LEVER_PREFETCH")]
    assert build[-4:] == ["--target", ss.BENCH_TARGET, "-j", "3"]


# ── the script and the report ───────────────────────────────────────────────


def stage05_outputs(s5dir, data_dir, frac, seed, min_gemms=1):
    """cells.json / metrics.json as stage05 writes them for `data_dir`:
    its own loader and split, cells below `min_gemms` not trained. Returns
    the validation GEMM keys."""
    s5 = s5split.stage05_module()
    cells, _ = s5.load_enriched_chunks(str(data_dir), True, None)
    entries, val_keys = [], set()
    for label, gs in cells.items():
        if len(gs) < min_gemms:
            continue
        tr, va = s5.split_validation(gs, frac, seed)
        entries.append(
            {"label": label, "n_gemms": len(gs), "n_train_gemms": len(tr),
             "n_val_gemms": len(va)}
        )  # fmt: skip
        val_keys |= {ev.gemm_key(g) for g in va}
    s5dir.mkdir(parents=True, exist_ok=True)
    (s5dir / "cells.json").write_text(json.dumps({"cells": entries}))
    met = s5dir / "metrics.json"
    doc = json.loads(met.read_text()) if met.exists() else {"args": {}}
    doc["args"].update(
        validation_frac=frac, seed=seed, enriched_csv_dir=[str(data_dir)],
        prior_round_dir=None, current_round_dir=str(s5dir.parent),
    )  # fmt: skip
    met.write_text(json.dumps(doc))
    return val_keys, cells


def test_stage05_split_is_recovered_exactly(tmp_path, tw):
    run, _ = write_run(tmp_path, tw, held_round=False)
    data = run / "round_0" / "stage04"
    s5dir = run / "round_0" / "stage05"
    val_keys, cells = stage05_outputs(s5dir, data, 0.3, 42, min_gemms=4)
    gemms = ra.load_bench([data]).gemms
    assert all(ev.gemm_key(g.row) == g.key for g in gemms)

    got = s5split.recover(s5dir, gemms, data_dirs=[data])
    assert got.exact, got.reason
    assert {g.key for g in got.validation} == val_keys and val_keys
    small = {k for c, gs in cells.items() if len(gs) < 4 for k in map(ev.gemm_key, gs)}
    assert {g.key for g in got.untrained} == small and small
    assert len(got.train) + len(got.validation) + len(got.untrained) == len(gemms)
    assert not {g.key for g in got.train} & val_keys

    doc = json.loads((s5dir / "cells.json").read_text())
    doc["cells"][0]["n_val_gemms"] += 1
    (s5dir / "cells.json").write_text(json.dumps(doc))
    bad = s5split.recover(s5dir, gemms, data_dirs=[data])
    assert not bad.exact and "1 of" in bad.reason
    assert bad.mismatched_cells == [doc["cells"][0]["label"]]
    assert not s5split.recover(s5dir, gemms, data_dirs=[tmp_path]).exact
    stage05_outputs(s5dir, data, 0.0, 42)
    assert "validation_frac 0" in s5split.recover(s5dir, gemms).reason


def write_run(tmp_path, tw, *, held_round=True):
    """A run whose round_0 trained the bundle and (optionally) round_1
    benched new GEMMs, with a library and a bank library."""
    import csv

    rng = random.Random(11)
    pool = kernels(rng, 10, 0)
    bank = kernels(rng, 20, 100)
    lib = tmp_path / "lib"
    lib.mkdir()
    for stem, kern in ((bf.LIBRARY_STEM, pool), (bf.OTHER_STEM, bank)):
        path = bf.write_logic(
            lib / f"{stem}.dat",
            [
                bf.solution(
                    int(k["sol_idx_global"]), j, (k["mt_m"], k["mt_n"], k["mt_k"]),
                    (k["mi_m"], k["mi_n"], k["mi_k"], 1),
                    nta=k["cache_hints_a"], ntb=k["cache_hints_b"],
                    occupancy=k["occupancy"],
                )
                for j, k in enumerate(kern)
            ],
        )  # fmt: skip
        if stem == bf.OTHER_STEM:
            doc = msgpack.unpackb(path.read_bytes(), raw=False)
            doc["library"] = {
                "type": "Problem",
                "rows": [
                    {
                        "predicate": {"type": "EqualityMatching"},
                        "library": {
                            "type": "Matching",
                            "table": [
                                {"key": [1, 1, 1, 1], "index": int(k["sol_idx_global"])}
                                for k in kern
                            ],
                        },
                    }
                ],
            }
            path.write_bytes(msgpack.packb(doc, use_bin_type=True))
    kern = ev.library_pool(lib, bf.LIBRARY_STEM) + ev.library_pool(lib, bf.OTHER_STEM)
    by_sid = {int(k["sol_idx_global"]): k for k in kern}
    run = tmp_path / "run"
    all_g = gemms(rng, kern)
    rounds = {
        "round_0": [g for i, g in enumerate(all_g) if i % 5 != 2],
        "round_1": [g for i, g in enumerate(all_g) if i % 5 == 2],
    }
    cols = list(ra.PROBLEM_FIELDS) + ["cu_budget", "schedule", "sol_idx_global", "us"]
    cols += ["is_skip", "rank", "is_origami_pick"] + list(ev.CONFIG_FIELDS)
    for rd, gs in rounds.items():
        if rd == "round_1" and not held_round:
            continue
        d = run / rd / "stage04"
        d.mkdir(parents=True)
        with (d / "chunk_0000.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for g in gs:
                for s, us, sk in zip(g.sids, g.us, g.skip):
                    k = by_sid[int(s)]
                    w.writerow(
                        dict(
                            g.row,
                            sol_idx_global=int(s), us=float(us), is_skip=str(bool(sk)),
                            rank=0, is_origami_pick=1 if s == g.fallback else 0,
                            **{c: k[c] for c in ev.CONFIG_FIELDS},
                        )
                    )  # fmt: skip
    s5 = run / "round_0" / "stage05"
    s5.mkdir(parents=True)
    (s5 / "cells.json").write_text(json.dumps({"cells": []}))
    (s5 / "metrics.json").write_text(
        json.dumps(
            {
                "args": {
                    "library_dir": str(lib),
                    "library_stem": bf.LIBRARY_STEM,
                    "arch": "gfx950",
                    "weight_dtype": "fp32",
                    "pick_measurement": "any",
                },
                "hardware": {"n_cu": HW.n_cu, "lds_bytes": HW.lds_bytes, "l2_bytes": HW.l2_bytes},
            }
        )
    )  # fmt: skip
    (run / "configs").mkdir()
    (run / "configs" / "20000101_000000_initial.yaml").write_text(
        f"arch: gfx1250\nhardware:\n  n_cu: {HW.n_cu}\n"
        f"planner:\n  enabled: true\n  bank_library_stems: [{bf.OTHER_STEM}]\n"
        "validate:\n  pick_measurement: timed\n"
    )
    mf = tmp_path / "model.bin"
    mf.write_bytes(model(cascade=4))
    return run, mf


@pytest.mark.parametrize("held_round", [True, False, "stage05"])
def test_script_writes_the_subset_and_the_report_shows_it(tmp_path, tw, held_round):
    import make_report as mr

    run, mf = write_run(tmp_path, tw, held_round=held_round is True)
    if held_round == "stage05":
        val_keys, _ = stage05_outputs(
            run / "round_0" / "stage05", run / "round_0" / "stage04", 0.3, 42
        )
    out = tmp_path / "out"
    rc = css.main(
        [
            str(run), "--out-dir", str(out), "--model-file", str(mf),
            "--sizes", "0,4,8,all", "--jobs", "1", "--no-latency", "--parity", "20",
        ]
    )  # fmt: skip
    assert rc == 0
    doc = json.loads((out / css.OUT_NAME).read_text())
    assert doc["pick_measurement"] == "any"
    assert doc["pool"]["n"] == 10 and doc["bank"]["n"] == 20
    identity = ki.library_identity(tmp_path / "lib", [bf.LIBRARY_STEM, bf.OTHER_STEM])
    assert ki.recorded_ids(ki.read_record(doc)) == dict(identity.ids)
    assert doc["kernels"] == {
        str(sid): {"signature": ki.label_of_id(kid), "kernel_id": kid}
        for sid, kid in identity.ids.items()
    }
    assert [r["size"] for r in doc["sizes"]] == ["0", "4", "8", "all"]
    order = doc["greedy"]["order_sol_idx"]
    assert len(order) == 8 == len(set(order))
    for r in doc["sizes"]:
        if r["size"] != "all":
            assert r["bank_sol_idx"] == order[: r["n_bank"]]
        assert r["n_configs"] == 10 + r["n_bank"]
        for side in ("selection", "held_out"):
            m = r[side]
            assert m["n"] > 0 and 0 < m["sel_eff"] <= 1.0 + 1e-9
            assert m["ceiling"] >= m["sel_eff"] - 1e-12
            assert 0 < m["identity_ceiling"] <= 1.0 + 1e-9
            assert set(m["identity"]["per_context"]) <= set(m["per_context"])
            assert set(m["per_context"]) == {
                "exclusive",
                "cu128",
                "cu64_dynamic",
                "cu32",
            }
    assert (
        doc["sizes"][1]["wins_baseline"]["bank_sol_idx"]
        == doc["wins_order_sol_idx"][:4]
    )
    source = {True: "held_out", False: "selection_split"}.get(
        held_round, "stage05_validation"
    )
    assert doc["held_out"]["source"] == source
    assert doc["held_out"]["label"] == css.HELD_LABELS[source]
    assert bool(doc["notes"]) is (held_round is not True)
    if held_round == "stage05":
        assert doc["held_out"]["n_gemms"] == len(val_keys)
        assert doc["held_out"]["stage05_split"]["exact"]
        assert "early-stopping" in doc["notes"][0]
    elif held_round is False:
        assert not doc["held_out"]["stage05_split"]["exact"]
        assert "could not be recovered exactly" in doc["notes"][0]
    for k, v in doc["parity"].items():
        assert v["n_same"] == v["n"] > 0, (k, v)
    assert doc["recommendation"] == {"10": None, "20": None}
    obj = doc["greedy"]["objective_geomean"]
    assert len(obj) == 9 and all(0 < v <= 1 for v in obj)
    sel = doc["sizes"][0]["selection"]
    if sel["n"] == doc["selection"]["n_gemms"]:
        assert math.isclose(obj[0], sel["sel_eff"], rel_tol=1e-9)
    assert doc["sizes"][0]["selection"]["n_fallback"] > 0

    html = ss.report_html(doc)
    assert "<table>" in html and "Recommended" in html
    report = tmp_path / "report.html"
    mr.build(str(run), str(report), bench=False, shipped_subset=str(out / css.OUT_NAME))
    page = report.read_text()
    assert 'id="subset"' in page and "Shipped candidate set" in page
    assert html_escape(doc["held_out"]["label"]) in page
    assert f"{doc['sizes'][1]['n_configs']:,}" in page
    mr.build(str(run), str(report), bench=False)
    assert "no shipped_subset.json" in report.read_text()


def renumber(lib, stem, new_index):
    """Rewrite a library file of `write_run` with every solution index `i`
    numbered `new_index(i)`."""
    path = lib / f"{stem}.dat"
    doc = msgpack.unpackb(path.read_bytes(), raw=False)
    for sol in doc["solutions"]:
        sol["index"] = new_index(sol["index"])
    for row in (doc.get("library") or {}).get("rows") or []:
        for entry in row["library"]["table"]:
            entry["index"] = new_index(entry["index"])
    path.write_bytes(msgpack.packb(doc, use_bin_type=True))


@pytest.mark.parametrize("recorded", [False, True])
def test_script_refuses_bench_data_of_another_build(tmp_path, tw, recorded, capsys):
    """Rows of the run, with or without the kernels.json record of the
    library they were enriched with, against a rebuild that numbers every
    kernel otherwise: the subset's indices could not name them."""
    from lib.dat import KERNELS_FILE, write_kernel_attributes

    run, mf = write_run(tmp_path, tw)
    lib = tmp_path / "lib"
    if recorded:
        identity = ki.library_identity(lib, [bf.LIBRARY_STEM, bf.OTHER_STEM])
        for rd in ("round_0", "round_1"):
            write_kernel_attributes(
                run / rd / "stage04" / KERNELS_FILE,
                dict(identity.kernels),
                bf.LIBRARY_STEM,
                extra={ki.RECORD_KEY: ki.record(identity)},
            )
    renumber(lib, bf.LIBRARY_STEM, lambda i: (i + 1) % 10)
    renumber(lib, bf.OTHER_STEM, lambda i: 100 + (i - 99) % 20)
    capsys.readouterr()
    argv = [str(run), "--out-dir", str(tmp_path / "out"), "--model-file", str(mf)]
    argv += ["--sizes", "0,4", "--jobs", "1", "--no-latency", "--parity", "0"]
    assert css.main(argv) == 2
    err = capsys.readouterr().err
    assert "choose with the library the run was benched with" in err
    assert ("hipBLASLt build changed" if recorded else "differ from the current") in err
    assert not (tmp_path / "out" / css.OUT_NAME).exists()


@pytest.mark.parametrize("bank_stem", [False, True])
def test_script_leaves_blocked_kernels_out_of_the_universe(tmp_path, tw, bank_stem):
    """The config's blocklist (lib/correctness.py) applies with --bank-stem,
    which names the bank's files itself, as it does with the config's
    bank."""
    from lib.dat import load_library_kernels

    run, mf = write_run(tmp_path, tw)
    lib = tmp_path / "lib"
    pool = load_library_kernels(lib, bf.LIBRARY_STEM)
    bank = load_library_kernels(lib, bf.OTHER_STEM)
    names = {pool[3]["kernel_name"], bank[5]["kernel_name"]}
    blocked = {
        int(k["sol_idx_global"]) for k in pool + bank if k["kernel_name"] in names
    }
    lst = tmp_path / "blocked.tsv"
    lst.write_text("".join(f"C\t{n}\n" for n in sorted(names)))
    snap = run / "configs" / "20000101_000000_initial.yaml"
    snap.write_text(snap.read_text() + f"correctness:\n  lists: [{lst}]\n")
    out = tmp_path / "out"
    argv = [str(run), "--out-dir", str(out), "--model-file", str(mf)]
    argv += ["--sizes", "0,4,all", "--jobs", "1", "--no-latency", "--parity", "0"]
    if bank_stem:
        argv += ["--bank-stem", bf.OTHER_STEM]
    assert css.main(argv) == 0
    doc = json.loads((out / css.OUT_NAME).read_text())
    n_pool = sum(k["kernel_name"] not in names for k in pool)
    assert doc["pool"]["n"] == n_pool < 10
    assert doc["bank"]["n"] == sum(k["kernel_name"] not in names for k in bank) < 20
    assert not blocked & set(doc["greedy"]["order_sol_idx"])
    assert not blocked & set(doc["wins_order_sol_idx"])


def test_script_measures_latency_with_the_bench(tmp_path, tw):
    run, mf = write_run(tmp_path, tw)
    fake = tmp_path / "bench"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "n = int(open(sys.argv[sys.argv.index('--inputs') + 1]).read().split('CONFIGS ')[1].split()[0])\n"
        "print('candidate_set 3 1 %.2f 1 1 %.2f 9' % (n / 4.0, n / 2.0))\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    out = tmp_path / "out"
    assert css.main(
        [
            str(run), "--out-dir", str(out), "--model-file", str(mf),
            "--sizes", "0,4,8,16", "--jobs", "1", "--parity", "0",
            "--bench-binary", str(fake),
        ]
    ) == 0  # fmt: skip
    doc = json.loads((out / css.OUT_NAME).read_text())
    p99 = {r["n_bank"]: r["latency_us"]["p99"] for r in doc["sizes"]}
    assert p99 == {0: 5.0, 4: 7.0, 8: 9.0, 16: 13.0}
    rec = doc["recommendation"]
    fits10 = [r for r in doc["sizes"] if r["latency_us"]["p99"] <= 10]
    best = max(r["held_out"]["sel_eff"] for r in fits10)
    assert rec["10"]["held_out_sel_eff"] >= best - ss.RECOMMEND_TOLERANCE
    assert rec["10"]["n_bank"] == min(
        r["n_bank"] for r in fits10 if r["held_out"]["sel_eff"] >= best - 0.002
    )
    assert rec["20"]["n_bank"] <= 16
    assert doc["latency"]["n_shapes"] == len({(p[0], p[1], p[2], p[3]) for p in SHAPES})


def test_models_the_simulation_cannot_follow_are_refused(tw):
    for serving in (
        {"non_temporal_rule": True, "non_temporal_family_rule": True},
        {"trained_key_gate": True},
    ):
        bundle = v3_bundle(labels=LABELS, spec=SPEC, serving=serving,
                           trained_view_keys=[1])  # fmt: skip
        with pytest.raises(ValueError, match="does not model"):
            ss.model_view(write(bundle), HW, tw)
