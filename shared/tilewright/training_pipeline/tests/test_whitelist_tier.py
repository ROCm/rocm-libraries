# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The whitelist tier (lib/whitelist_tier.py): how the engine serves it when
the pool changes, how it is chosen, stage05's --whitelist-tier-k, its
rebuild without training (scripts/build_whitelist_tier.py), stage06's
record, the orchestrator's option and the report's bench analysis."""

import csv
import json
import random
import subprocess
import sys

import msgpack
import numpy as np
import pytest
import yaml

import bench_fakes as bf
import run_orchestrator as ro
import stage05_train as s5
from lib import evaluate as ev
from lib import feasibility
from lib import features as fs
from lib import mlrec
from lib import run_analysis as ra
from lib import shipped_subset as ss
from lib import whitelist_tier as wt
from lib.hardware import DeviceHardware
from test_kernel_identity import pool_library, record_rows
from test_stage05_train import (
    CELLS3,
    HARDWARE,
    PIPELINE_DIR,
    kernel_pool,
    load_bundle,
    shapes_for,
    train,
    write_round,
)
from test_v3 import BASE_CFG, BF16_NT, LARGE, LABELS, v3_bundle, write

HW = DeviceHardware(n_cu=256, lds_bytes=163840, l2_bytes=4 << 20)
STEM = (
    "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_"
    "Cijk_Dijk_gfx950"
)
BANK_STEM = (
    "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_"
    "Cijk_Dijk_CU256_gfx950"
)
PROBLEM = dict(m=4096, n=4096, k=4096, batch=1, **BF16_NT)
TILES = [
    (64, 64, 64),
    (128, 128, 64),
    (256, 128, 64),
    (128, 256, 32),
    (64, 128, 64),
    (128, 64, 64),
    (64, 256, 64),
    (256, 256, 64),
]


@pytest.fixture(scope="module")
def tw():
    return pytest.importorskip("tilewright")


# ── the engine when the pool changes ─────────────────────────────────────────


def engine_pool(tiles=TILES, sid0=0):
    return [
        dict(BASE_CFG, mt_m=m, mt_n=n, mt_k=k, sol_idx_global=sid0 + i)
        for i, (m, n, k) in enumerate(tiles)
    ]


def tier_model(pool_hashes, cascade=2, tier=True):
    bundle = v3_bundle(LABELS, spec=(), seed=3)
    bundle["serving"] = {"cascade_keys": cascade, "whitelist_tier": tier}
    for e in bundle["models"].values():
        e["view_key_hashes"] = sorted(pool_hashes)
    return write(bundle)


def ranked(tw, data, pool, depth=1):
    model = ev.load_model_bytes(data, tw)
    cs = ev.candidate_set(tw, model, pool, HW)
    p = ev.make_problem(tw, PROBLEM)
    ctx = ev.make_context(tw, fs.EXCLUSIVE)
    return [
        (int(r.config_index), float(r.score) if r.scored else None)
        for r in cs.rank(p, ev.make_hardware(tw, HW), depth, ctx)
    ]


def scored(results):
    return [i for i, s in results if s is not None]


def hashes_of(kernels):
    return wt.view_key_hashes(kernels, ())


def test_the_engine_skips_tier_keys_the_pool_lacks(tw):
    pool = engine_pool()
    gone = dict(BASE_CFG, mt_m=32, mt_n=32, mt_k=32, sol_idx_global=99)
    listed = hashes_of([pool[0], pool[3], pool[5]])
    with_gone = ranked(tw, tier_model(listed + hashes_of([gone])), pool)
    assert sorted(scored(with_gone)) == [0, 3, 5]
    assert with_gone == ranked(tw, tier_model(listed), pool)


def test_a_cell_left_without_tier_keys_ranks_through_the_cascade(tw):
    pool = engine_pool()
    listed = hashes_of(pool[:3])
    rest = pool[3:]
    cascade_only = tier_model(listed, tier=False)
    for depth in (1, 4):
        got = ranked(tw, tier_model(listed), rest, depth)
        assert got == ranked(tw, cascade_only, rest, depth)
        assert len(scored(got)) == max(2, depth)
    # A tier key the pool holds but the LDS gate rejects counts as missing.
    big = dict(BASE_CFG, mt_m=512, mt_n=512, mt_k=256, sol_idx_global=50)
    assert not feasibility.passes_gates(
        PROBLEM,
        fs.config_kwargs_from_row(big),
        lds_bytes=HW.lds_bytes,
        nt_a_available=False,
        nt_b_available=False,
    )
    infeasible = tier_model(hashes_of([big]))
    assert ranked(tw, infeasible, rest + [big]) == ranked(
        tw, tier_model(hashes_of([big]), tier=False), rest + [big]
    )


def test_kernels_a_pool_adds_rank_only_behind_the_tier(tw):
    pool = engine_pool()
    listed = hashes_of(pool[:2])
    added = engine_pool([(32, 64, 64), (64, 32, 64)], sid0=200)
    grown = pool + added
    got = ranked(tw, tier_model(listed), grown)
    assert sorted(scored(got)) == [0, 1]
    deep = scored(ranked(tw, tier_model(listed), grown, depth=6))
    assert sorted(deep[:2]) == [0, 1] and len(deep) > 2


# ── choosing the tier ────────────────────────────────────────────────────────


def _row(cell, keys, costs, scores):
    return (
        cell,
        np.array(keys, dtype=np.int64),
        np.array(costs, dtype=np.float64),
        np.array(scores, dtype=np.float64),
    )


def test_choose_covers_first_then_keeps_k_keys_per_cell():
    hashes = np.array([1000 + i for i in range(5)], dtype=np.uint64)
    scored_rows = [
        _row(0, [1, 2], [10.0, 12.0], [5.0, 1.0]),
        _row(0, [2, 3], [8.0, 9.0], [1.0, 0.0]),
        _row(0, [4], [20.0], [0.0]),
        _row(1, [0], [5.0], [1.0]),
        _row(-1, [0, 1], [1.0, 1.0], [1.0, 1.0]),
        _row(2, [], [], []),
    ]
    winners = [10.0, 8.0, 20.0, 5.0, 1.0, 1.0]
    one = wt.choose(["a", "b", "c"], scored_rows, winners, 1, hashes, s5)
    assert one.keys == {"a": [1002], "b": [1000]}
    assert one.cells["c"] == {"n_gemms": 0}
    assert one.cells["a"]["n_gemms"] == 3 and one.cells["a"]["n_keys_measured"] == 4
    every = wt.choose(["a", "b", "c"], scored_rows, winners, 9, hashes, s5)
    assert every.keys["a"][0] == 1002 and sorted(every.keys["a"]) == [
        1001,
        1002,
        1003,
        1004,
    ]
    assert every.cells["a"]["covered_gemms"] == 3
    assert every.cells["a"]["in_sample_sel_eff"] == pytest.approx(
        every.cells["a"]["in_sample_sel_eff_all_keys"]
    )


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250-strict"])
def test_a_key_costs_its_first_pool_kernel_with_a_row_and_infeasible_keys_none(
    tw, arch
):
    pool = engine_pool()
    twin = dict(pool[1], occupancy=2, sol_idx_global=300)
    big = dict(BASE_CFG, mt_m=512, mt_n=512, mt_k=256, sol_idx_global=301)
    kernels = pool + [twin, big]
    u = ss.build_universe(kernels, [], ())
    assert u.key[len(pool)] == u.key[1] and u.n_keys == len(pool) + 1
    row = dict(zip(ra.PROBLEM_FIELDS, (4096, 4096, 4096, 1, "N", "T")))
    row.update(a_type="bf16_r", b_type="bf16_r", c_type="bf16_r", d_type="bf16_r")
    row.update(compute_type="f32_r")
    key = tuple(row[f] for f in ra.PROBLEM_FIELDS)
    sids = np.array([300, 1, 2, 301], dtype=np.int64)
    us = np.array([5.0, 9.0, 7.0, 1.0])
    g = ra.Gemm(key, "", sids, us, np.zeros(4, bool), float(us.min()))
    data = write(wt.scoring_bundle(v3_bundle(LABELS, spec=(), seed=1)), arch=arch)
    ((cell, keys, costs, scores),) = wt.engine_scores(data, u, HW, [g])
    assert cell >= 0
    cost = dict(zip(keys.tolist(), costs.tolist()))
    assert cost == {int(u.key[1]): 9.0, int(u.key[2]): 7.0}
    assert np.isfinite(scores).all()


def test_training_gemms_are_each_leafs_training_side():
    rng = random.Random(2)
    gemms = []
    for i in range(80):
        m, n, k = rng.choice([(4096, 4096, 4096), (64, 64, 64), (300, 700, 128)])
        key = (m + i, n, k, 1, "T", "N", "bf16_r", "bf16_r", "bf16_r", "bf16_r")
        key += ("f32_r",)
        empty = (np.zeros(0, np.int64), np.zeros(0), np.zeros(0, bool))
        gemms.append(ra.Gemm(key, "", *empty, 1.0))
    got = wt.training_gemms(gemms, {}, 0.3, 7, s5)
    by_leaf = {}
    for g in gemms:
        by_leaf.setdefault(s5._leaf_of(*(int(v) for v in g.key[:4]), {}), []).append(g)
    want = []
    for group in by_leaf.values():
        tr, _ = s5.split_validation([g.row for g in group], 0.3, 7)
        keys = {ev.gemm_key(r) for r in tr}
        want += [g for g in group if ev.gemm_key(g.row) in keys]
    assert [g.key for g in got] == [g.key for g in want]
    assert 0 < len(got) < len(gemms)
    every = wt.training_gemms(gemms, {}, 0.0, 7, s5)
    assert sorted(g.key for g in every) == sorted(g.key for g in gemms)
    counts = wt.leaf_counts(gemms, {}, 0.3, 7, s5)
    assert sum(a for a, _ in counts.values()) == len(got)


def test_the_bundle_serves_the_tier_it_is_given():
    bundle = v3_bundle(LABELS, spec=())
    bundle["serving"] = {"cascade_keys": 16, "whitelist_tier": False}
    tier = wt.Tier(k=2, keys={LARGE: [7, 3], LABELS[1]: [5]}, record={"k": 2})
    served = wt.apply(bundle, tier)
    assert served["serving"] == {"cascade_keys": 16, "whitelist_tier": True}
    assert served[wt.RECORD] == {"k": 2}
    parsed = mlrec.read_model(write(served))
    assert (
        parsed["serving"]["whitelist_tier"] and parsed["serving"]["cascade_keys"] == 16
    )
    keys = {c["label"]: c["view_key_hashes"] for c in parsed["cells"]}
    assert keys[LARGE] == [3, 7] and keys[LABELS[1]] == [5] and keys[LABELS[2]] == []
    assert not mlrec.v3_serving_flags(served["serving"]) & mlrec.V3_NO_WHITELIST_TIER
    plain = wt.strip(served)
    assert wt.RECORD not in plain and not plain["serving"]["whitelist_tier"]
    assert all("view_key_hashes" not in e for e in plain["models"].values())
    scoring = wt.scoring_bundle(served)
    assert scoring["serving"] == {"cascade_keys": 0, "whitelist_tier": False}
    assert bundle["serving"]["cascade_keys"] == 16 and not bundle.get(wt.RECORD)


def test_bundles_that_cannot_get_a_tier():
    assert "v3" in wt.refusal({"catalog": "v2", "models": {}})
    keyed = v3_bundle(LABELS, spec=())
    keyed["serving"] = {"launch_keys": True}
    assert "launch" in wt.refusal(keyed)
    smart = v3_bundle(LABELS, spec=())
    smart["models"][LARGE]["view_key_hashes"] = [1]
    assert "smart_K" in wt.refusal(smart)
    tiered = wt.apply(v3_bundle(LABELS, spec=()), wt.Tier(k=1, keys={LARGE: [1]}))
    assert wt.refusal(tiered) is None
    with pytest.raises(ValueError, match="smart_K"):
        wt.build(smart, [], k=2, u=None, hardware=HW, arch="gfx950",
                 weight_dtype="fp32")  # fmt: skip


def test_the_pool_check_counts_the_tier_keys_a_pool_lacks():
    pool = engine_pool()
    h = hashes_of(pool)
    bundle = wt.apply(
        v3_bundle(LABELS, spec=()),
        wt.Tier(k=2, keys={LARGE: h[:2], LABELS[1]: [h[0], 123]}, record={"k": 2}),
    )
    full = wt.pool_check(bundle, pool)
    assert full["tier_keys_missing"] == 1 and full["cells_without_tier_keys"] == []
    assert full["keys_digest"] == wt.keys_digest(h)
    lean = wt.pool_check(bundle, pool[1:])
    assert lean["tier_keys_missing"] == 3
    assert lean["cells_without_tier_keys"] == [LABELS[1]]
    assert wt.pool_check(v3_bundle(LABELS, spec=()), pool) is None


# ── stage05, the rebuild and stage06 ─────────────────────────────────────────


def bank_library(lib, bank):
    """A bank file of `bank` (an Equality row of every kernel) in `lib`."""
    from test_stage05_train import library_solutions

    sols = [
        dict(s, name=f"{s['name']}_K{k['sol_idx_global']}")
        for s, k in zip(library_solutions(bank), bank)
    ]
    path = bf.write_logic(lib / f"{BANK_STEM}.dat", sols)
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
                        for k in bank
                    ],
                },
            }
        ],
    }
    path.write_bytes(msgpack.packb(doc, use_bin_type=True))


def run_config(tmp, *, bank=False, blocked=()):
    doc = {
        "config_id": "unit_tier",
        "arch": "gfx950",
        "hardware": dict(HARDWARE, pci_chip_id=0x75A0),
    }
    if bank:
        doc["planner"] = {"enabled": True, "bank_library_stems": [BANK_STEM]}
    if blocked:
        lst = tmp / "blocked.tsv"
        lst.write_text("".join(f"C\t{n}\n" for n in blocked))
        doc["correctness"] = {"lists": [str(lst)]}
    path = tmp / "tier_config.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def tier_round(tmp, *, bank=False, cells=CELLS3[:2], per_cell=16):
    """A v3 round whose rows name the kernels of a library (and with `bank`
    a bank file of its last four kernels), with their identities recorded."""
    pool = kernel_pool()
    table = pool[:-4] if bank else pool
    lib = tmp / "lib"
    ident = pool_library(lib, table, stem=STEM)
    if bank:
        bank_library(lib, pool[-4:])
        from lib import kernel_identity as ki

        ident = ki.library_identity(lib, [STEM, BANK_STEM])
    data = write_round(
        tmp / "round_0" / "stage04", shapes_for(4, per_cell, cells), pool
    )
    stems = [STEM, BANK_STEM] if bank else [STEM]
    record_rows(data, ident, stems[0])
    return data, lib, ident


def tier_train(tmp, data, lib, out, *extra, cfg=None, expect_rc=0):
    """stage05 v3 on the enriched folder(s) `data` with the library `lib`."""
    args = [
        "--catalog", "v3", "--min-cell-gemms", 8, "--min-base-cell-gemms", 8,
        "--library-dir", lib, "--library-stem", STEM, *extra,
    ]  # fmt: skip
    if cfg is not None:
        args += ["--config-yaml", cfg]
    dirs = data if isinstance(data, list) else [data]
    return train(tmp, dirs, out, *args, expect_rc=expect_rc)


def measured_keys(data, ident, catalog_spec=()):
    """{hash of every view key with a row in the GEMMs of `data`}."""
    bench = ra.load_bench([data], identity=ident)
    configs = bench.configs
    return {
        fs.view_key_hash(
            fs.v3_view_key(fs.config_kwargs_from_row(configs[s]), None, catalog_spec)
        )
        for g in bench.gemms
        for s in g.sids.tolist()
        if s in configs
    }


def library_without_first_keys(bundle, dest):
    """({the first tier key of every cell of `bundle`}, the kernels of a
    library in `dest` without the kernels of those keys)."""
    first = {h for e in bundle["models"].values() for h in e["view_key_hashes"][:1]}
    pool = kernel_pool()
    hashes = dict(zip((k["sol_idx_global"] for k in pool), hashes_of(pool)))
    kept = [k for k in pool if hashes[k["sol_idx_global"]] not in first]
    assert first and len(kept) < len(pool)
    pool_library(dest, kept, stem=STEM)
    return first, kept


@pytest.fixture
def tiered(tmp_path, tw):
    data, lib, ident = tier_round(tmp_path)
    out = tmp_path / "round_0" / "stage05"
    tier_train(tmp_path, data, lib, out, "--whitelist-tier-k", 3)
    return tmp_path, data, lib, ident, out


def test_stage05_gives_every_cell_a_tier_of_measured_keys(tiered):
    tmp, data, lib, ident, out = tiered
    bundle = load_bundle(out / "models.pt")
    assert bundle["serving"]["whitelist_tier"] and bundle["serving"]["cascade_keys"]
    rec = bundle[wt.RECORD]
    assert rec["k"] == 3 and rec["n_cells"] == len(bundle["models"]) == 2
    assert rec["pool"]["n_kernels"] == len(kernel_pool())
    measured = measured_keys(data, ident)
    for entry in bundle["models"].values():
        assert 0 < len(entry["view_key_hashes"]) <= 3
        assert set(entry["view_key_hashes"]) <= measured
    doc = json.loads((out / wt.TIER_FILE).read_text())
    assert {k: c["keys"] for k, c in doc["cells"].items()} == {
        k: e["view_key_hashes"] for k, e in bundle["models"].items()
    }
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["whitelist_tier"]["k"] == 3 and metrics["serving"]["whitelist_tier"]
    options = json.loads((out / s5.MODEL_OPTIONS_FILE).read_text())
    assert options["whitelist_tier_k"] == 3
    cells = json.loads((out / "cells.json").read_text())["cells"]
    assert all(c["n_whitelist_tier_keys"] == len(
        bundle["models"][c["label"]]["view_key_hashes"]) for c in cells)  # fmt: skip
    info = __import__("tilewright").describe(
        ev.load_model_bytes(write(bundle, dtype="bf16"))
    )
    assert info.whitelist_tier and info.cascade_keys == 32


def test_the_rebuild_reproduces_stage05s_tier_and_follows_a_pool(tiered):
    tmp, data, lib, ident, out = tiered
    tool = PIPELINE_DIR / "scripts" / "build_whitelist_tier.py"

    def rebuild(dest, *extra, rc=0):
        proc = subprocess.run(
            [sys.executable, str(tool), "--train-dir", str(out), "--out-dir",
             str(dest), "--quiet", *map(str, extra)],
            capture_output=True, text=True, timeout=600,
        )  # fmt: skip
        assert proc.returncode == rc, proc.stdout[-3000:] + proc.stderr[-3000:]
        return proc

    rebuild(tmp / "same")
    a, b = load_bundle(out / "models.pt"), load_bundle(tmp / "same" / "models.pt")
    assert {k: e["view_key_hashes"] for k, e in a["models"].items()} == {
        k: e["view_key_hashes"] for k, e in b["models"].items()
    }
    assert a[wt.RECORD] == b[wt.RECORD]
    met = json.loads((tmp / "same" / "metrics.json").read_text())
    assert met["whitelist_tier"]["split"]["exact"]
    assert met["whitelist_tier"]["rebuilt_from"] == str(out.resolve())
    for name in ("cells.json", s5.MEMBER_ORDER_FILE, s5.MODEL_OPTIONS_FILE):
        assert (tmp / "same" / name).is_file()

    # A pool without the kernels the tier chose first: no tier key is theirs.
    first, kept = library_without_first_keys(a, tmp / "lib2")
    rebuild(tmp / "moved", "--library-dir", tmp / "lib2", "--library-stem", STEM)
    moved = load_bundle(tmp / "moved" / "models.pt")
    for entry in moved["models"].values():
        assert entry["view_key_hashes"] and not set(entry["view_key_hashes"]) & first
    assert moved[wt.RECORD]["pool"]["n_kernels"] == len(kept)
    assert moved["models"][CELLS3[0]]["state_dict"]["q_proj.0.weight"].equal(
        a["models"][CELLS3[0]]["state_dict"]["q_proj.0.weight"]
    )
    assert load_bundle(out / "models.pt")[wt.RECORD] == a[wt.RECORD]

    rebuild(tmp / "none", "--k", 0)
    none = load_bundle(tmp / "none" / "models.pt")
    assert wt.RECORD not in none and not none["serving"]["whitelist_tier"]
    assert not (tmp / "none" / wt.TIER_FILE).exists()
    opts = json.loads((tmp / "none" / s5.MODEL_OPTIONS_FILE).read_text())
    assert opts["whitelist_tier_k"] == 0
    proc = rebuild(tmp / "none2", "--train-dir", tmp / "none", rc=1)
    assert "has no whitelist tier" in proc.stdout + proc.stderr
    same_dir = rebuild(out, rc=1)
    assert "must not be" in same_dir.stdout + same_dir.stderr


def test_stage06_records_the_tier_and_how_it_meets_the_pool(tiered):
    tmp, data, lib, ident, out = tiered
    library_without_first_keys(load_bundle(out / "models.pt"), tmp / "lib2")
    cfg = run_config(tmp)
    s6 = PIPELINE_DIR / "stages" / "stage06_deploy_weights.py"
    for name, library in (("same", lib), ("moved", tmp / "lib2")):
        proc = subprocess.run(
            [sys.executable, str(s6), "--train-dir", str(out), "--config-yaml",
             str(cfg), "--out-dir", str(tmp / f"s6_{name}"), "--library-stem", STEM,
             "--library-dir", str(library), "--weight-dtype", "bf16"],
            capture_output=True, text=True, timeout=600,
        )  # fmt: skip
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        rec = json.loads((tmp / f"s6_{name}" / "deploy_record.json").read_text())
        tier = rec["whitelist_tier"]
        assert tier["k"] == 3 and tier["served_pool"] is not None
        served = tier["served_pool"]
        assert (served["tier_keys_missing"] == 0) == (name == "same")
        assert ("not in the pool" in proc.stdout + proc.stderr) == (name == "moved")
        manifest = json.loads((tmp / f"s6_{name}" / "manifest.json").read_text())
        assert manifest["serving"]["whitelist_tier"]
        assert manifest["whitelist_tier"]["k"] == 3
    no_lib = subprocess.run(
        [sys.executable, str(s6), "--train-dir", str(out), "--config-yaml", str(cfg),
         "--out-dir", str(tmp / "s6_bare"), "--library-stem", STEM, "--quiet"],
        capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    assert no_lib.returncode == 0, no_lib.stdout[-3000:] + no_lib.stderr[-3000:]
    rec = json.loads((tmp / "s6_bare" / "deploy_record.json").read_text())
    assert (
        rec["whitelist_tier"]["k"] == 3 and rec["whitelist_tier"]["served_pool"] is None
    )


def test_a_round_carries_cells_without_the_tier_of_the_bundle_before_it(tiered):
    tmp, data, lib, ident, out = tiered
    more = write_round(tmp / "round_1" / "stage04", shapes_for(9, 16, CELLS3[2:3]))
    record_rows(more, ident, STEM)
    nxt = tmp / "round_1" / "stage05"
    tier_train(tmp, more, lib, nxt, "--prior-round-dir", tmp / "round_0")
    bundle = load_bundle(nxt / "models.pt")
    assert sorted(bundle["models"]) == sorted(CELLS3)
    assert wt.RECORD not in bundle and not bundle["serving"]["whitelist_tier"]
    assert all(not e.get("view_key_hashes") for e in bundle["models"].values())
    # With a tier, the carried cells get one from their GEMMs the round reads.
    again = tmp / "round_1b" / "stage05"
    tier_train(tmp, [data, more], lib, again, "--prior-round-dir", tmp / "round_0",
               "--only-cells", CELLS3[2], "--whitelist-tier-k", 2)  # fmt: skip
    tiered_again = load_bundle(again / "models.pt")
    assert tiered_again[wt.RECORD]["k"] == 2
    assert tiered_again["n_models_carried_forward"] == 2
    keys = {k: e["view_key_hashes"] for k, e in tiered_again["models"].items()}
    assert sorted(keys) == sorted(CELLS3) and all(
        0 < len(v) <= 2 for v in keys.values()
    )


def test_the_tier_holds_with_the_bank_the_family_rule_the_gate_and_blocklist(
    tmp_path, tw
):
    data, lib, ident = tier_round(tmp_path, bank=True, cells=CELLS3[:1], per_cell=20)
    wins = {}
    with open(sorted(data.glob("chunk_*.csv"))[0]) as f:
        for r in csv.DictReader(f):
            if r["is_winner"] == "True":
                wins[r["sol_idx_global"]] = wins.get(r["sol_idx_global"], 0) + 1
    top = int(max(wins, key=wins.get))
    name_of = {sid: k["kernel_name"] for sid, k in ident.kernels.items()}
    cfg = run_config(tmp_path, bank=True, blocked=[name_of[top]])
    out = tmp_path / "round_0" / "stage05"
    rules = ("--non-temporal-rule", "--non-temporal-family-rule")
    tier_train(tmp_path, data, lib, out, *rules, "--recipe", "pooled_ft",
               "--pooled-epochs", 1, "--pooled-eval-points", 1,
               "--pooled-n-estimators", 8, "--fine-tune-epochs", 1,
               "--whitelist-tier-k", 4, cfg=cfg)  # fmt: skip
    bundle = load_bundle(out / "models.pt")
    serving = bundle["serving"]
    assert serving["bank"] and serving["trained_key_gate"]
    assert serving["non_temporal_family_rule"] and serving["whitelist_tier"]
    assert bundle["recipe"] == "pooled_ft"
    pool = kernel_pool()
    by_sid = {k["sol_idx_global"]: k for k in pool}
    blocked_key = hashes_of([by_sid[top]])[0]
    others = {h for k, h in zip(pool, hashes_of(pool)) if k["sol_idx_global"] != top}
    tier_keys = {h for e in bundle["models"].values() for h in e["view_key_hashes"]}
    assert tier_keys and set(bundle["trained_view_keys"]) >= tier_keys
    if blocked_key not in others:
        assert blocked_key not in tier_keys
    rec = bundle[wt.RECORD]
    assert rec["pool"]["n_kernels"] == len(pool) - 1
    bank_keys = set(hashes_of(pool[-4:])) - set(hashes_of(pool[:-4]))
    universe = wt.universe(yaml.safe_load(cfg.read_text()), lib, STEM, 0, ())
    assert universe.n_pool == len(pool) - 4 - (
        top in [k["sol_idx_global"] for k in pool[:-4]]
    )
    assert bank_keys <= set(int(h) for h in universe.hashes) | {blocked_key}
    data_v3 = write(bundle, dtype="int8")
    flags = mlrec.v3_serving_flags(mlrec.read_header(data_v3)["serving"])
    want = (mlrec.V3_NON_TEMPORAL_RULE | mlrec.V3_NON_TEMPORAL_FAMILY_RULE
            | mlrec.V3_BANK | mlrec.V3_TRAINED_KEY_GATE)  # fmt: skip
    assert flags & want == want and not flags & mlrec.V3_NO_WHITELIST_TIER


def test_stage05_refuses_a_tier_it_cannot_build(tmp_path):
    data = write_round(tmp_path / "stage04", shapes_for(4, 10, CELLS3[:1]))
    lib = tmp_path / "lib"
    pool_library(lib, kernel_pool(), stem=STEM)
    tier = ("--whitelist-tier-k", 4)
    with_lib = ("--library-dir", lib, "--library-stem", STEM)
    for extra, text in (
        ((*tier, *with_lib), "needs --catalog v3"),
        ((*tier, *with_lib, "--catalog", "v3", "--whitelist", "view-key"),
         "needs --whitelist none"),
        ((*tier, *with_lib, "--catalog", "v3", "--launch-keys"), "--launch-keys"),
        ((*tier, "--catalog", "v3"), "--library-dir and --library-stem"),
        ((*tier, *with_lib, "--catalog", "v3", "--skip-deployed-eval"),
         "--skip-deployed-eval"),
        (("--whitelist-tier-k", -1), "out of range"),
    ):  # fmt: skip
        proc = train(tmp_path, [data], tmp_path / "out", *extra, expect_rc=1)
        assert text in proc.stdout + proc.stderr, extra


# ── the orchestrator ─────────────────────────────────────────────────────────


def orchestrator_config(tmp_path, **train_keys):
    """A valid v3 config with `train_keys` set afterwards."""
    from test_run_orchestrator import make_config

    cfg, _ = ro.load_config(make_config(tmp_path, **{"train.catalog": "v3"}))
    cfg["train"].update(train_keys)
    return cfg


def test_the_orchestrator_passes_the_tier_to_stage05(tmp_path):
    from test_run_orchestrator import _pipeline, make_config

    v3 = {"train.catalog": "v3"}
    p = _pipeline(tmp_path, make_config(tmp_path, **v3))
    assert "--whitelist-tier-k" not in p._catalog_args()
    p = _pipeline(
        tmp_path, make_config(tmp_path, **v3, **{"train.whitelist_tier_k": 32})
    )
    args = p._catalog_args()
    assert args[args.index("--whitelist-tier-k") + 1] == "32"
    assert ro.cfg_get({}, "train.whitelist_tier_k") == 0
    options = s5.command_model_options(
        ["--enriched-csv-dir", "-", "--output-dir", "-", "--arch", "gfx950", *args]
    )
    assert options["whitelist_tier_k"] == 32


def test_the_tier_is_a_model_shaping_option_old_records_trained_without():
    assert "whitelist_tier_k" not in s5.NOT_MODEL_SHAPING
    assert s5.changed_model_options({}, {"whitelist_tier_k": 0}) == {}
    assert s5.changed_model_options({}, {"whitelist_tier_k": 32}) == {
        "whitelist_tier_k": (0, 32)
    }


@pytest.mark.parametrize(
    "keys, text",
    [
        ({"catalog": "v2"}, "needs train.catalog v3"),
        ({"catalog": "v3", "whitelist": "view-key"}, "replaces the smart_K"),
        ({"catalog": "v3", "launch_keys": True}, "launch class"),
    ],
)
def test_config_validation_refuses_a_tier_it_cannot_build(tmp_path, keys, text):
    cfg = orchestrator_config(tmp_path, whitelist_tier_k=32, **keys)
    errors, _ = ro.validate_config(cfg)
    assert any(text in e and e.startswith("train.whitelist_tier_k") for e in errors)


def test_config_validation_checks_the_final_retrains_tier(tmp_path):
    cfg = orchestrator_config(tmp_path)
    assert not [e for e in ro.validate_config(cfg)[0] if "whitelist_tier" in e]
    cfg["full"] = {"final_retrain": {"recipe": {"whitelist_tier_k": 32}}}
    assert not [e for e in ro.validate_config(cfg)[0] if "whitelist_tier" in e]
    cfg["full"]["final_retrain"]["recipe"]["whitelist"] = "view-key"
    errors = ro.validate_config(cfg)[0]
    assert any(
        e.startswith("full.final_retrain.recipe: train.whitelist_tier_k 32")
        and "replaces the smart_K" in e
        for e in errors
    )


# ── the report's bench analysis ──────────────────────────────────────────────


def test_the_bench_analysis_takes_the_tier_as_tier_1():
    pool = ra.Pool(engine_pool(), "library")
    h = hashes_of(pool.kernels)
    mask = pool.whitelist_mask([], [h[2], h[5]], ())
    assert mask.tolist() == [i in (2, 5) for i in range(len(pool))]
    sig = ra.signature_of(pool.kernels[0])
    assert pool.whitelist_mask([sig], [h[5]]).tolist() == [
        i in (0, 5) for i in range(len(pool))
    ]


def test_the_bench_analysis_stops_tier_1_at_the_tier():
    kernels = engine_pool()
    pool = ra.Pool(kernels, "library")
    key = (4096, 4096, 4096, 1, "N", "T", "bf16_r", "bf16_r", "bf16_r", "bf16_r")
    key += ("f32_r",)
    us = np.array([1.0 if i == 0 else 2.0 if i == 3 else 3.0 for i in range(8)])
    sids = np.array([k["sol_idx_global"] for k in kernels], dtype=np.int64)
    g = ra.Gemm(key, LARGE, sids, us, np.zeros(8, bool), 1.0)
    bench = ra.Bench(gemms=[g], configs={}, stats={})
    tier, soft = ra.LEVELS.index("tier"), ra.LEVELS.index("soft")

    def levels(**model):
        info = ra.ModelInfo(
            labels=[LARGE], whitelists={}, hardware=HW, non_temporal_rule=False, **model
        )
        return ra.evaluate(bench, pool, info).levels[0]

    assert levels()[tier] == levels()[soft] == 1.0
    on_tier = levels(view_keys={LARGE: hashes_of([kernels[3]])})
    assert on_tier[soft] == 1.0 and on_tier[tier] == 2.0
