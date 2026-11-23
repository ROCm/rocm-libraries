# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""lib/standins.py: the choice of a cell's stand-ins on its ScoreTable, the
records a model lists, and the mirror of the engine's serving rule, against
CandidateSet.standins() and the engine's [TILEWRIGHT_DIAG STANDINS] line on
synthetic and shipped models."""
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import stage05_train as s5
from lib import drift_fallback
from lib import evaluate as ev
from lib import features as fs
from lib import mlrec
from lib import standins as sl
from test_drift_fallback import NOT_FLAGGED, _Launch, _pool
from test_mlrec import shipped_models
from test_v3 import BASE_CFG, HW, LARGE, SAMPLE_SPEC, SPEC, _gemm, v3_bundle, write

INF = math.inf
U64 = (1 << 64) - 1


def _table(us, score):
    """A ScoreTable of keys (rows) x GEMMs from `us` (INF: not measured) and
    `score`, each GEMM's winner its fastest measured key."""
    us = np.asarray(us, dtype=np.float64)
    score = np.where(np.isfinite(us), np.asarray(score, dtype=np.float64), -INF)
    return s5.ScoreTable(
        [(j,) for j in range(us.shape[0])],
        list(range(us.shape[1])),
        us,
        score,
        us.min(axis=0),
    )


# Rows: 0 and 1 the tier, 2-5 candidates. Without row 0 the tier loses GEMMs
# 0 and 2, without row 1 GEMM 1; row 4 would replace row 0 exactly.
LOST = _table(
    [[10, INF, 30, 40],
     [INF, 20, INF, 41],
     [12, 22, INF, INF],
     [11, 21, 33, INF],
     [10, INF, 30, INF],
     [INF, 24, INF, INF]],
    [[5, 0, 5, 3],
     [0, 5, 0, 4],
     [1, 1, 0, 0],
     [2, 1, 1, 0],
     [9, 0, 1, 0],
     [0, 0, 0, 0]],
)  # fmt: skip


def test_keys_that_cover_the_lost_gemms_come_first_by_coverage_then_gain():
    cand = np.array([True, True, True, True, False, True])
    # Row 3 covers both GEMMs row 0 alone covers, row 2 one; for row 1 all
    # three cover its GEMM, the fastest first. Row 4 is not a candidate, and
    # the tier's rows never are.
    assert sl.choose(LOST, [0, 1], cand) == {0: [3, 2], 1: [3, 2, 5]}
    assert sl.choose(LOST, [0, 1], cand, depth=2) == {0: [3, 2], 1: [3, 2]}
    assert sl.choose(LOST, [0, 1], cand, depth=1) == {0: [3], 1: [3]}
    every = np.ones(6, dtype=bool)
    # Row 4 covers what row 0 covers at its latencies: first, then row 3.
    assert sl.choose(LOST, [0, 1], every) == {0: [4, 3, 2], 1: [3, 2, 5]}
    assert sl.choose(LOST, [0, 1], np.zeros(6, dtype=bool)) == {0: [], 1: []}


def test_a_key_whose_removal_loses_nothing_lists_only_keys_that_speed_the_picks():
    # Rows 0 and 1 the tier; row 1 is redundant (row 0 outscores it where both
    # are measured). Row 2 makes GEMM 0's pick faster, row 3 makes GEMM 1's
    # slower, row 4 makes both faster.
    t = _table(
        [[12, 25], [13, INF], [10, INF], [INF, 30], [11, 20]],
        [[5, 5], [1, 0], [6, 0], [0, 9], [7, 6]],
    )
    cand = np.array([False, False, True, True, True])
    got = sl.choose(t, [0, 1], cand)
    # Row 1: no GEMM is lost; by mean log efficiency after the key, row 4
    # (-0.048) then row 2 (-0.112); row 3 slows the pick it changes.
    assert got[1] == [4, 2]
    # Row 0: GEMM 1 is lost; rows 4 and 3 cover it (row 4 gains), then row 2
    # speeds GEMM 0's pick.
    assert got[0] == [4, 3, 2]
    # Without a helping key, no stand-in.
    assert sl.choose(t, [0, 1], np.array([False, False, False, True, False]))[1] == []


def test_equal_keys_are_taken_in_table_order():
    t = _table(
        [[10, INF], [INF, 10], [11, INF], [11, INF]],
        [[5, 0], [0, 5], [1, 0], [1, 0]],
    )
    cand = np.array([False, False, True, True])
    # Row 1's GEMM no candidate covers, and none speeds a pick: no list.
    assert sl.choose(t, [0, 1], cand) == {0: [2, 3], 1: []}
    # Equal scores: the earlier tier row keeps the GEMM, and the other takes
    # it over when it goes, so neither removal loses it; the faster key is
    # listed for both.
    tie = _table([[10], [12], [9]], [[1], [1], [2]])
    assert sl.choose(tie, [0, 1], np.array([False, False, True])) == {
        0: [2],
        1: [2],
    }


def test_records_are_the_sections_and_the_reference_text():
    big = U64
    lists = {"b": {5: [7, 8, 9], 3: [4]}, "a": {big: [1, 2]}, "c": {6: []}}
    recs = sl.records(lists, ["a", "b", "c"], 2)
    assert recs == [(0, big, [1, 2]), (1, 3, [4]), (1, 5, [7, 8])]
    assert sl.text(recs) == f"0\t{big}\t1,2\n1\t3\t4\n1\t5\t7,8\n"
    assert sl.text(reversed(recs)) == sl.text(recs)
    assert sl.records({"a": {-1: [-2]}}, ["a"], 3) == [(0, big, [big - 1])]
    assert sl.records(lists, ["a", "b", "c"], 3)[-1] == (1, 5, [7, 8, 9])
    section = mlrec.standins_bytes(recs)
    assert len(section) == 4 + 3 * 13 + 5 * 8


def test_the_serving_rule_takes_one_held_stand_in_per_missing_key():
    assert sl.missing([5, 3, 9, 3], {3, 5}, {5}) == [3]
    assert sl.missing([5, 3, 9], None, set()) == [3, 5, 9]
    assert sl.missing([5, 3], {3, 5}, {3, 5, 7}) == []
    lists = {3: [10, 11], 5: [10, 12], 9: [13]}
    # 3 takes 10, so 5 takes 12; 9's stand-in is not held.
    assert sl.active([3, 5, 9], lists, {10, 11, 12}) == [10, 12]
    assert sl.active([5], lists, {10}) == [10]
    assert sl.active([3, 5], lists, {11}) == [11]
    assert sl.active([3, 5], lists, {10}) == [10]
    assert sl.active([7], lists, {10}) == []


def _standin_model(rng, keys, labels, spec, *, drift=None):
    """A gated model over the view keys `keys` (its first two untrained)
    whose cells whitelist twelve of them and list up to three stand-ins for
    most of their trained ones, and the bundle it was written from."""
    keys = sorted(set(keys))
    trained = keys[2:]
    serving = {"trained_key_gate": True, "cascade_keys": 4}
    if drift is not None:
        serving["drift_keep_ppm"] = drift
    bundle = v3_bundle(
        labels=list(labels), spec=spec, serving=serving, trained_view_keys=trained
    )
    for entry in bundle["models"].values():
        entry["view_key_hashes"] = sorted(rng.sample(keys, min(12, len(keys))))
    records = []
    for ci, label in enumerate(sorted(bundle["models"])):
        wl = bundle["models"][label]["view_key_hashes"]
        others = [h for h in trained if h not in wl]
        for key in wl:
            if key in trained and rng.random() < 0.8:
                n = rng.randint(1, min(3, len(others)))
                records.append((ci, key, rng.sample(others, n)))
    bundle["serving"]["standins"] = records
    return write(bundle), bundle


@pytest.mark.parametrize("spec", [SPEC, SAMPLE_SPEC], ids=["default", "sample"])
def test_mirror_matches_the_engine_on_synthetic_pools(spec):
    tw = pytest.importorskip("tilewright")
    rng = random.Random(5)
    checked = joined = 0
    for round_ in range(6):
        pool = _pool(rng, spec=spec)
        per_kernel = drift_fallback.pool_hashes(pool, spec)
        hashes = sorted(set(per_kernel))
        data, _ = _standin_model(
            rng, hashes, (LARGE, "Mid|Mid|MidK|Bnone"), spec,
            drift=1000000 if round_ % 3 == 2 else None,
        )  # fmt: skip
        parsed = mlrec.read_model(data)
        model = tw.load_model_from_memory(data)
        info = tw.describe(model)
        assert info.standin_lists == len(parsed["serving"]["standins"]) > 0
        for _ in range(12):
            gone = set(rng.sample(hashes, rng.randrange(0, 12)))
            variant = [k for k, h in zip(pool, per_kernel) if h not in gone]
            if rng.random() < 0.3:
                variant = variant[::-1]
            if rng.random() < 0.3:
                variant.append(dict(BASE_CFG, mt_m=512, mt_n=32, attributes={}))
            held = sl.pool_hashes(variant, spec)
            want = sl.active_standins(parsed, held)
            cs = ev.candidate_set(tw, model, variant, HW)
            got = {int(c): [int(h) for h in cs.standins(c)] for c in cs.standin_cells()}
            assert got == want
            assert list(cs.standins(len(parsed["cells"]))) == []
            checked += 1
            joined += bool(want)
            # The stand-ins are the candidate set's, whatever its order.
            rev = ev.candidate_set(tw, model, variant[::-1], HW)
            assert [list(rev.standins(c)) for c in range(2)] == [
                list(cs.standins(c)) for c in range(2)
            ]
            summary = sl.summary(parsed, held)
            assert summary["cells"] == sorted(want)
            assert summary["standins"] == sum(len(v) for v in want.values())
        # Every listed key held: no stand-in joins, and the model ranks as it
        # does without them.
        assert sl.active_standins(parsed, hashes) == {}
        cs = ev.candidate_set(tw, model, pool, HW)
        assert list(cs.standin_cells()) == []
        plain = ev.candidate_set(
            tw, tw.load_model_from_memory(mlrec.without_standins(data)), pool, HW
        )
        h = ev.make_hardware(tw, HW)
        for shape in ((64, 64, 64), (4096, 4096, 512), (1, 8192, 8192)):
            m, n, k = shape
            p = ev.make_problem(
                tw, fs.problem_kwargs_from_row(dict(_gemm(), m=m, n=n, k=k))
            )
            assert [(r.config_index, r.score) for r in cs.rank(p, h, len(pool))] == [
                (r.config_index, r.score) for r in plain.rank(p, h, len(pool))
            ]
    assert checked == 72 and 0 < joined < checked


def test_the_mirror_counts_trained_whitelisted_keys_alone():
    parsed = {
        "serving": {
            "trained_key_gate": True,
            "standins": [[0, 3, [7, 8]], [1, 9, [3]]],
        },
        "trained_view_keys": [3, 4, 7, 8, 9],
        "cells": [{"view_key_hashes": [3, 4, 5]}, {"view_key_hashes": [9]}],
    }
    # 5 is untrained: never missing; 3 is missing and takes 7.
    assert sl.active_standins(parsed, {4, 7, 8}) == {0: [7]}
    assert sl.summary(parsed, {4, 7, 8}) == {
        "cells": [0],
        "cells_lacking_keys": 2,
        "standins": 1,
        "missing": 2,
    }
    assert sl.active_standins(parsed, {3, 4, 9}) == {}
    assert sl.active_standins(parsed, {4, 3}) == {1: [3]}


_DIAG_CHILD = """
import json, sys
sys.path.insert(0, sys.argv[1])
import tilewright as tw
from lib import evaluate as ev
from lib.hardware import DeviceHardware
hw = DeviceHardware(**json.loads(sys.argv[4]))
model = tw.load_model_from_memory(open(sys.argv[2], "rb").read())
for pool in json.load(open(sys.argv[3])):
    ev.candidate_set(tw, model, pool, hw)
"""


def test_the_engines_diagnostic_line_is_the_mirrors_summary(tmp_path):
    """With TILEWRIGHT_DIAG the engine prints one [TILEWRIGHT_DIAG STANDINS]
    line per candidate set whose pool lacks a trained whitelisted key:
    summary's counts."""
    pytest.importorskip("tilewright")
    rng = random.Random(13)
    pool = _pool(rng, n_kinds=40)
    per_kernel = drift_fallback.pool_hashes(pool, SPEC)
    hashes = sorted(set(per_kernel))
    labels = (LARGE, "Mid|Mid|MidK|Bnone", "Tiny|Tiny|TinyK|Bnone")
    data, _ = _standin_model(rng, hashes, labels, SPEC)
    parsed = mlrec.read_model(data)
    pools = [pool]
    for n in (3, 9, 20):
        gone = set(rng.sample(hashes, n))
        pools.append([k for k, h in zip(pool, per_kernel) if h not in gone])
    want = []
    for p in pools:
        s = sl.summary(parsed, sl.pool_hashes(p, SPEC))
        if s["cells_lacking_keys"]:
            want.append(
                f"[TILEWRIGHT_DIAG STANDINS] cells={len(s['cells'])}/"
                f"{s['cells_lacking_keys']} standins={s['standins']} "
                f"missing={s['missing']}"
            )
    assert want and len(want) < len(pools)
    (tmp_path / "model.bin").write_bytes(data)
    (tmp_path / "pools.json").write_text(json.dumps(pools))
    env = {k: v for k, v in os.environ.items() if not k.startswith("TILEWRIGHT_")}
    env["TILEWRIGHT_DIAG"] = "1"
    hw = {"n_cu": HW.n_cu, "lds_bytes": HW.lds_bytes, "l2_bytes": HW.l2_bytes}
    run = subprocess.run(
        [sys.executable, "-c", _DIAG_CHILD,
         str(Path(__file__).resolve().parents[1]), str(tmp_path / "model.bin"),
         str(tmp_path / "pools.json"), json.dumps(hw)],
        env=env, capture_output=True, text=True, check=True,
    )  # fmt: skip
    got = [ln for ln in run.stderr.splitlines() if "TILEWRIGHT_DIAG STANDINS" in ln]
    assert got == want
    assert all(re.fullmatch(r"\[TILEWRIGHT_DIAG STANDINS\] cells=\d+/\d+ "
                            r"standins=\d+ missing=\d+", ln) for ln in got)  # fmt: skip


def test_mirror_matches_the_engine_on_shipped_models():
    """Each shipped model that can take stand-ins (not the TF32 TT one), its
    trained keys widened by a synthetic pool's keys and stand-ins listed from
    them: the mirror's stand-ins are the engine's on samples of that pool."""
    tw = pytest.importorskip("tilewright")
    models = [
        (p, d)
        for p, d in shipped_models()
        if d[:8] == mlrec.MAGIC_V3 and Path(p).name != NOT_FLAGGED
    ]
    if not models:
        pytest.skip("no materialized MLREC_v3 shipped model")
    rng = random.Random(17)
    pool = []
    for mt_m in (16, 32, 64, 128, 256):
        for mt_n in (16, 32, 64, 128, 256):
            for mt_k in (32, 64, 128):
                for mi in ((16, 16, 32), (32, 32, 16)):
                    pool.append(dict(BASE_CFG, mt_m=mt_m, mt_n=mt_n, mt_k=mt_k,
                                     mi_m=mi[0], mi_n=mi[1], mi_k=mi[2],
                                     attributes={}))  # fmt: skip
    for i, k in enumerate(pool):
        k["sol_idx_global"] = i
    flagged = 0
    for path, data in models:
        header = mlrec.read_header(data)
        serving = header["serving"]
        if (
            not serving["whitelist_tier"]
            or serving["launch_keys"]
            or not serving["trained_key_gate"]
        ):
            continue
        spec = mlrec.read_model(data)["attribute_spec"]
        mine = sorted(sl.pool_hashes(pool, spec))
        widened = mlrec.with_trained_keys(
            data, sorted(set(header["trained_view_keys"]) | set(mine))
        )
        parsed = mlrec.read_model(widened)
        records = []
        for ci, cell in enumerate(parsed["cells"]):
            wl = {int(h) for h in cell.get("view_key_hashes") or []}
            others = [h for h in mine if h not in wl]
            for key in sorted(wl):
                if len(others) >= 3:
                    records.append((ci, key, rng.sample(others, 3)))
        flagged += 1
        listed = mlrec.with_standins(widened, records)
        parsed = mlrec.read_model(listed)
        model = tw.load_model_from_memory(listed)
        launch = _Launch(tw, tw.describe(model).provenance.launch_model)
        for _ in range(3):
            variant = rng.sample(pool, rng.randrange(len(pool) // 4, len(pool)))
            want = sl.active_standins(parsed, sl.pool_hashes(variant, spec))
            configs = [ev.make_config(tw, k, i) for i, k in enumerate(variant)]
            cs = tw.CandidateSet(model, configs, launch=launch)
            got = {int(c): [int(h) for h in cs.standins(c)] for c in cs.standin_cells()}
            assert got == want, path.name
            assert want
    if not flagged:
        pytest.skip("no shipped model takes stand-ins")


def test_deployed_depth_is_the_configs():
    assert sl.deployed_depth(None) is None
    assert sl.deployed_depth({"deploy": {}}) is None
    assert sl.deployed_depth({"deploy": {"standins": None}}) is None
    for depth in (1, 2, 3):
        assert sl.deployed_depth({"deploy": {"standins": {"depth": depth}}}) == depth
    for bad in (0, 4, -1, True, 2.0, "2", None):
        with pytest.raises(ValueError, match="whole number from 1 to 3"):
            sl.deployed_depth({"deploy": {"standins": {"depth": bad}}})
    with pytest.raises(ValueError):
        sl.deployed_depth({"deploy": {"standins": 2}})


def test_evaluate_model_refuses_a_pool_whose_cells_take_standins():
    """A model with stand-ins evaluated over a pool that lacks a trained
    whitelisted key (one built from a GEMM's measured candidates) is refused
    unless the caller allows the pool's drift, as for the drift fallback;
    allowed, the evaluation names the cells whose whitelist stand-ins
    join."""
    tw = pytest.importorskip("tilewright")
    rng = random.Random(5)
    pool = _pool(rng, n_kinds=30)
    per_kernel = drift_fallback.pool_hashes(pool, SPEC)
    keys = sorted(set(per_kernel))
    bundle = v3_bundle(
        labels=[LARGE], serving={"cascade_keys": 4, "trained_key_gate": True},
        trained_view_keys=keys,
    )  # fmt: skip
    bundle["models"][LARGE]["view_key_hashes"] = keys[:8]
    plain = write(bundle)
    flagged = mlrec.with_standins(plain, [(0, keys[0], [keys[9], keys[10]])])
    gemm = _gemm()
    gemm["candidates"] = [dict(k, us=10.0 + i, is_skip=0, rank=i, is_origami_pick=0)
                          for i, k in enumerate(pool)]  # fmt: skip
    whole = {ev.pool_key(gemm): pool}
    a = ev.evaluate_model(flagged, [gemm], hardware=HW, pools=whole, tw=tw)
    b = ev.evaluate_model(plain, [gemm], hardware=HW, pools=whole, tw=tw)
    assert [r.pick_sol_idx for r in a.results] == [r.pick_sol_idx for r in b.results]
    assert a.standin_cells == {} and b.standin_cells == {}
    lacking = {ev.pool_key(gemm): [k for k, h in zip(pool, per_kernel) if h != keys[0]]}
    with pytest.raises(ValueError, match="stand-ins.*Large\\|Large\\|LargeK\\|Bnone"):
        ev.evaluate_model(flagged, [gemm], hardware=HW, pools=lacking, tw=tw)
    out = ev.evaluate_model(flagged, [gemm], hardware=HW, pools=lacking, tw=tw,
                            allow_pool_drift=True)  # fmt: skip
    assert out.standin_cells == {"|".join(ev.pool_key(gemm)): [LARGE]}
    assert out.degraded_cells == {} and out.results[0].served_by == "model"
    assert ev.evaluate_model(plain, [gemm], hardware=HW, pools=lacking,
                             tw=tw).standin_cells == {}  # fmt: skip
