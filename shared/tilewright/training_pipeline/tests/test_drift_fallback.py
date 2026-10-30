# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""lib/drift_fallback.py: the drift fallback's share and its mirror of the
engine's trigger, against CandidateSet.degraded_cells() and the engine's
[TILEWRIGHT_DIAG DRIFT] line on synthetic and shipped models."""
import json
import os
import random
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from lib import drift_fallback
from lib import evaluate as ev
from lib import features as fs
from lib import mlrec
from test_mlrec import shipped_models
from test_v3 import BASE_CFG, HW, LARGE, SAMPLE_SPEC, SPEC, v3_bundle, write

# The shipped model the drift-fallback tests leave without the fallback (the
# TF32 TT one).
NOT_FLAGGED = "gfx950_Cijk_Alik_Bjlk_S_MX_B_Bias_HAS_SAV_UserArgs.tilewright.bin"


@pytest.mark.parametrize(
    "keep,ppm",
    [(1, 1000000), (1.0, 1000000), (0.9, 900000), (0.95, 950000), (0.000001, 1),
     (Decimal("0.75"), 750000), (0.5, 500000)],
)  # fmt: skip
def test_keep_ppm_is_exact(keep, ppm):
    assert drift_fallback.keep_ppm(keep) == ppm


@pytest.mark.parametrize(
    "bad",
    [0, -0.5, 1.000001, 0.1234567, 2, True, None, "x", float("nan"), float("inf")],
)
def test_keep_ppm_refuses_what_the_file_cannot_hold(bad):
    with pytest.raises(ValueError, match="keep"):
        drift_fallback.keep_ppm(bad)


def _pool(rng, n_kinds=40, spec=SPEC):
    """Kernels of `n_kinds` signatures, some with a second member, shuffled."""
    kinds = []
    for mt_m in (16, 32, 64, 128, 256):
        for mt_n in (16, 64, 256):
            for mt_k in (32, 64, 128):
                kinds.append(dict(BASE_CFG, mt_m=mt_m, mt_n=mt_n, mt_k=mt_k))
    rng.shuffle(kinds)
    kinds = kinds[:n_kinds]
    pool = [dict(k, attributes={}) for k in kinds]
    for k in kinds[:6]:
        pool.append(dict(k, occupancy=2, attributes={}))
    if spec != SPEC:
        for k in pool:
            k["attributes"] = {"param_a": rng.choice([0, 16])}
    rng.shuffle(pool)
    for i, k in enumerate(pool):
        k["sol_idx_global"] = i
    return pool


def _model(
    rng, hashes, keep, gate=True, labels=(LARGE, "Mid|Mid|MidK|Bnone"), spec=SPEC
):
    hashes = sorted(set(hashes))
    trained = hashes[2:] if gate else None
    bundle = v3_bundle(
        labels=list(labels),
        spec=spec,
        serving={"trained_key_gate": gate, "cascade_keys": 4,
                 **({"drift_keep_ppm": keep} if keep is not None else {})},
        **({"trained_view_keys": trained} if gate else {}),
    )  # fmt: skip
    for entry in bundle["models"].values():
        entry["view_key_hashes"] = sorted(rng.sample(hashes, min(12, len(hashes))))
    return write(bundle)


@pytest.mark.parametrize("spec", [SPEC, SAMPLE_SPEC], ids=["default", "sample"])
def test_mirror_matches_the_engine_on_synthetic_pools(spec):
    tw = pytest.importorskip("tilewright")
    rng = random.Random(3)
    checked = degraded = 0
    for round_ in range(6):
        pool = _pool(rng, spec=spec)
        hashes = drift_fallback.pool_hashes(pool, spec)
        keep = rng.choice([1000000, 950000, 900000, 750000, 500000, 1])
        data = _model(rng, hashes, keep, gate=round_ % 2 == 0, spec=spec)
        parsed = mlrec.read_model(data)
        assert drift_fallback.model_keep_ppm(parsed) == keep
        model = tw.load_model_from_memory(data)
        for _ in range(12):
            gone = set(rng.sample(sorted(set(hashes)), rng.randrange(0, 8)))
            variant = [k for k, h in zip(pool, hashes) if h not in gone]
            if rng.random() < 0.3:
                variant = variant[::-1]
            if rng.random() < 0.3:
                variant.append(dict(BASE_CFG, mt_m=512, mt_n=32, attributes={}))
            vh = drift_fallback.pool_hashes(variant, spec)
            want = drift_fallback.degraded_cells(parsed, vh)
            got = list(ev.candidate_set(tw, model, variant, HW).degraded_cells())
            assert got == want
            checked += 1
            degraded += bool(want)
            missing = drift_fallback.missing_keys(parsed, vh)
            counts = drift_fallback.cell_counts(parsed, vh)
            for i, (present, total) in enumerate(counts):
                assert total - present == len(missing.get(i, []))
            summary = drift_fallback.drift_summary(parsed, vh)
            assert summary["degraded"] == want
            assert summary["missing"] == sum(len(missing[i]) for i in want)
        # Every listed key held: no cell is degraded, whatever the pool adds.
        assert drift_fallback.degraded_cells(parsed, hashes) == []
        assert list(ev.candidate_set(tw, model, pool[::-1], HW).degraded_cells()) == []
    assert checked == 72 and 0 < degraded < checked


def test_mirror_counts_trained_whitelisted_keys_once():
    parsed = {
        "serving": {"trained_key_gate": True, "drift_keep_ppm": 1000000},
        "trained_view_keys": [1, 2, 3],
        "cells": [
            {"view_key_hashes": [1, 1, 2, 9]},  # 9 is untrained: counts in neither
            {"view_key_hashes": [9]},
            {"view_key_hashes": []},
        ],
    }
    assert drift_fallback.cell_counts(parsed, [1, 2]) == [(2, 2), (0, 0), (0, 0)]
    assert drift_fallback.degraded_cells(parsed, [1, 2, 9]) == []
    assert drift_fallback.degraded_cells(parsed, [1, 9]) == [0]
    assert drift_fallback.missing_keys(parsed, [1, 9]) == {0: [2]}
    assert drift_fallback.degraded_cells(parsed, [1], keep=500000) == []
    assert drift_fallback.degraded_cells(parsed, [], keep=500000) == [0]
    ungated = dict(parsed, serving={"drift_keep_ppm": 1000000})
    assert drift_fallback.degraded_cells(ungated, [1, 2]) == [0, 1]
    assert drift_fallback.degraded_cells(dict(parsed, serving={}), [], keep=None) == []
    with pytest.raises(ValueError):
        drift_fallback.degraded_cells(parsed, [], keep=0)


class _Launch:
    """The launch model a shipped model names (its plans never reach the
    trigger)."""

    def __init__(self, tw, name):
        self.tw, self.id = tw, name or None

    def plan(self, config, problem, context):
        return self.tw.LaunchPlan()


def test_mirror_matches_the_engine_on_shipped_models():
    tw = pytest.importorskip("tilewright")
    models = [
        (p, d)
        for p, d in shipped_models()
        if d[:8] == mlrec.MAGIC_V3 and Path(p).name != NOT_FLAGGED
    ]
    if not models:
        pytest.skip("no materialized MLREC_v3 shipped model")
    rng = random.Random(7)
    pool = []
    for mt_m in (16, 32, 64, 128, 256):
        for mt_n in (16, 32, 64, 128, 256):
            for mt_k in (32, 64, 128):
                for mi in ((16, 16, 32), (32, 32, 16), (16, 16, 128)):
                    for hint in (0, 4):
                        pool.append(dict(BASE_CFG, mt_m=mt_m, mt_n=mt_n, mt_k=mt_k,
                                         mi_m=mi[0], mi_n=mi[1], mi_k=mi[2],
                                         cache_hints_a=hint, attributes={}))  # fmt: skip
    for i, k in enumerate(pool):
        k["sol_idx_global"] = i
    flagged = 0
    for path, data in models:
        header = mlrec.read_header(data)
        serving = header["serving"]
        if not serving["whitelist_tier"] or serving["launch_keys"]:
            continue
        try:
            data = mlrec.with_drift_fallback(data, 1000000)
        except ValueError:
            continue
        flagged += 1
        parsed = mlrec.read_model(data)
        spec = parsed["attribute_spec"]
        model = tw.load_model_from_memory(data)
        launch = _Launch(tw, tw.describe(model).provenance.launch_model)
        hashes = drift_fallback.pool_hashes(pool, spec)
        for _ in range(4):
            variant = rng.sample(pool, rng.randrange(len(pool) // 2, len(pool)))
            want = drift_fallback.degraded_cells(
                parsed, drift_fallback.pool_hashes(variant, spec)
            )
            configs = [ev.make_config(tw, k, i) for i, k in enumerate(variant)]
            cs = tw.CandidateSet(model, configs, launch=launch)
            assert list(cs.degraded_cells()) == want, path.name
        assert drift_fallback.degraded_cells(
            parsed, hashes
        ) == drift_fallback.degraded_cells(parsed, hashes[::-1])
    if not flagged:
        pytest.skip("no shipped model takes the drift fallback")


def test_evaluate_model_refuses_a_pool_that_degrades_a_cell():
    tw = pytest.importorskip("tilewright")
    from test_v3 import _gemm

    rng = random.Random(5)
    pool = _pool(rng, n_kinds=30)
    hashes = drift_fallback.pool_hashes(pool, SPEC)
    bundle = v3_bundle(labels=[LARGE], serving={"cascade_keys": 4})
    bundle["models"][LARGE]["view_key_hashes"] = sorted(set(hashes))[:8]
    plain = write(bundle)
    flagged = mlrec.with_drift_fallback(plain, 1000000)
    gemm = _gemm()
    gemm["candidates"] = [dict(k, us=10.0 + i, is_skip=0, rank=i, is_origami_pick=0)
                          for i, k in enumerate(pool)]  # fmt: skip
    gemm["winner_us"] = 10.0
    whole = {ev.pool_key(gemm): pool}
    # Over the pool it was built for, the model with the fallback picks as without it.
    a = ev.evaluate_model(flagged, [gemm], hardware=HW, pools=whole, tw=tw)
    b = ev.evaluate_model(plain, [gemm], hardware=HW, pools=whole, tw=tw)
    assert [r.pick_sol_idx for r in a.results] == [r.pick_sol_idx for r in b.results]
    # A pool without a listed key serves the cell degraded: refused unless allowed.
    gone = sorted(set(hashes))[0]
    lacking = {ev.pool_key(gemm): [k for k, h in zip(pool, hashes) if h != gone]}
    with pytest.raises(
        ValueError, match="drift fallback.*Large\\|Large\\|LargeK\\|Bnone"
    ):
        ev.evaluate_model(flagged, [gemm], hardware=HW, pools=lacking, tw=tw)
    out = ev.evaluate_model(flagged, [gemm], hardware=HW, pools=lacking, tw=tw,
                            allow_drift=True)  # fmt: skip
    assert out.results[0].served_by == "model"
    # Allowed, the evaluation names the cells it measured degraded.
    assert out.degraded_cells == {"|".join(ev.pool_key(gemm)): [LARGE]}
    assert a.degraded_cells == {} and b.degraded_cells == {}
    assert ev.evaluate_model(plain, [gemm], hardware=HW, pools=lacking,
                             tw=tw).degraded_cells == {}  # fmt: skip


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
    """With TILEWRIGHT_DIAG the engine prints one [TILEWRIGHT_DIAG DRIFT]
    line per candidate set with a degraded cell: drift_summary's counts."""
    pytest.importorskip("tilewright")
    rng = random.Random(11)
    pool = _pool(rng, n_kinds=40)
    hashes = drift_fallback.pool_hashes(pool, SPEC)
    keys = sorted(set(hashes))
    labels = (LARGE, "Mid|Mid|MidK|Bnone", "Tiny|Tiny|TinyK|Bnone")
    data = _model(rng, keys, 900000, labels=labels)
    parsed = mlrec.read_model(data)
    # The first pool holds every key; the others lack a few.
    pools = [pool]
    for n in (3, 9):
        gone = set(rng.sample(keys, n))
        pools.append([k for k, h in zip(pool, hashes) if h not in gone])
    want = []
    for p in pools:
        summary = drift_fallback.drift_summary(
            parsed, drift_fallback.pool_hashes(p, SPEC)
        )
        if summary["degraded"]:
            want.append(
                f"[TILEWRIGHT_DIAG DRIFT] cells={len(summary['degraded'])}/"
                f"{summary['cells_with_whitelist']} keep_ppm={summary['keep_ppm']} "
                f"missing={summary['missing']}"
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
    got = [ln for ln in run.stderr.splitlines() if "TILEWRIGHT_DIAG DRIFT" in ln]
    assert got == want
    assert all(re.fullmatch(r"\[TILEWRIGHT_DIAG DRIFT\] cells=\d+/\d+ "
                            r"keep_ppm=\d+ missing=\d+", ln) for ln in got)  # fmt: skip
