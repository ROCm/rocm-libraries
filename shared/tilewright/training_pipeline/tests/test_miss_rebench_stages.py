# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The deciding stages with their misses re-benched on the fake bench."""
import json

import pytest

import miss_fakes as mf
from test_stage05_train import run_stage

pytest.importorskip("tilewright")

CELL = "Large|Large|LargeK|Bnone"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A fake build, a bundle that always picks mf.PICK, and GEMMs of one
    cell on which that pick is within 10% of the winner but has only an
    inflated skip-screen row (or, planned, no row)."""
    tmp = tmp_path_factory.mktemp("world")
    root = mf.build(tmp)
    shapes = mf.near_pick_shapes(8, 0.9)
    mf.bundle(tmp / "round_0" / "stage05", shapes)
    screened, planned = shapes[:5], shapes[5:]
    rows = [r for s in screened for r in mf.rows(s)]
    rows += [r for s in planned for r in mf.rows(s, miss="not_benched")]
    mf.write_chunks(tmp / "round_1" / "stage04", rows)
    mf.write_chunks(tmp / "gate", [r for s in screened for r in mf.rows(s)])
    mf.write_plan(tmp / "plan.json", planned)
    mf.config(tmp / "config.yaml")
    return {"tmp": tmp, "root": root, "screened": screened, "planned": planned}


def rebench_flags(world, runs=2):
    return [
        "--rebench-misses",
        "--rebench-runs",
        runs,
        "--bench-binary",
        mf.bench_binary(world["root"]),
        "--devices",
        "0",
    ]


def geomean(values):
    import math

    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def warm(shapes):
    return geomean(mf.true_sel_eff(s) / mf.WARMUP_FACTOR for s in shapes)


def true(shapes):
    return geomean(mf.true_sel_eff(s) for s in shapes)


def test_the_gate_decides_on_the_rebenched_misses(world, tmp_path, monkeypatch):
    import stage06_deploy_weights as s6
    from lib import miss_rebench as mr

    for k, v in mf.env(world["root"]).items():
        monkeypatch.setenv(k, v)
    cfg_path = world["tmp"] / "config.yaml"
    screened = world["screened"]

    def gate(rebench):
        return s6.Gate(
            csv_dirs=[world["tmp"] / "gate"],
            target=0.9,
            library_dir=mf.library_dir(world["root"]),
            rebench=rebench,
        )

    def deploy(out, rebench):
        rc = s6.deploy(
            train_dir=world["tmp"] / "round_0" / "stage05",
            config_yaml=cfg_path,
            out_dir=out,
            library_stem=mf.STEM,
            gate=gate(rebench),
            quiet=True,
        )
        assert rc == 0
        return (
            json.loads((out / "deploy_record.json").read_text()),
            json.loads((out / "gate.json").read_text()),
        )

    rec, plain = deploy(tmp_path / "plain", None)
    assert rec["deployed"] is False and "below the target" in rec["not_deployed"]
    assert plain["sel_eff"] == pytest.approx(warm(screened), rel=1e-3)
    assert plain["pick_measurement"] == "warmup" and "pre_rebench" not in plain

    bench = mr.MissBench(
        bench_binary=mf.bench_binary(world["root"]),
        work_dir=tmp_path / "measured" / "rebench",
        cfg=mf.config_dict(),
        runs=2,
    )
    rec, measured = deploy(tmp_path / "measured", bench)
    assert rec["deployed"] is True and "reaches the target" in rec["gate"]["reason"]
    assert measured["sel_eff"] == pytest.approx(true(screened), rel=1e-3)
    assert measured["pick_measurement"] == "rebench" and measured["n_miss"] == 5
    assert measured["pre_rebench"]["sel_eff"] == pytest.approx(plain["sel_eff"])
    assert measured["pre_rebench"]["pick_measurement"] == "warmup"
    assert (measured["n_rebenched"], measured["n_rebench_failed"]) == (5, 0)
    assert rec["gate"]["n_rebenched"] == 5
    assert rec["gate"]["pre_rebench"]["sel_eff"] == pytest.approx(plain["sel_eff"])
    assert measured["rebench"]["library_digest"] == library_digest(world)
    assert_named(measured["rebench"]["gemms"], world)
    assert measured["per_cell"][CELL]["sel_eff_new"] == pytest.approx(
        measured["sel_eff"]
    )


def library_digest(world):
    """The digest of the library the stages' solution indices name."""
    from lib import kernel_identity as ki

    return ki.library_identity(mf.library_dir(world["root"]), [mf.STEM]).digest


def assert_named(rows, world):
    """Every miss row names its pick, and its re-benched winner, by signature
    and kernel id next to the solution index."""
    from lib import kernel_identity as ki

    ids = ki.library_identity(mf.library_dir(world["root"]), [mf.STEM]).ids
    assert rows
    for r in rows:
        keys = list(r)
        at = keys.index("pick")
        assert keys[at + 1 : at + 3] == ["pick_signature", "pick_kernel_id"]
        assert r["pick_kernel_id"] == ids[r["pick"]]
        assert r["pick_signature"] == ki.label_of_id(ids[r["pick"]])
        if r["status"] == "measured":
            assert r["winner_kernel_id"] == ids[r["winner_sol"]]
            assert r["winner_signature"] == ki.label_of_id(ids[r["winner_sol"]])


def stage08(world, out, *extra, env=None):
    run_stage(
        "stage08_optional_validate_selection_efficiency.py",
        "--train-dir",
        world["tmp"] / "round_0" / "stage05",
        "--arch",
        mf.ARCH,
        "--config-yaml",
        world["tmp"] / "config.yaml",
        "--weight-dtype",
        "bf16",
        "--library-dir",
        mf.library_dir(world["root"]),
        "--library-stem",
        mf.STEM,
        "--out-dir",
        out,
        "--dataset",
        f"gate={world['tmp'] / 'gate'}",
        "--bootstrap",
        0,
        "--quiet",
        *extra,
        env=dict(mf.env(world["root"]), **(env or {})),
    )
    return json.loads((out / "summary.json").read_text())["datasets"][0]


def test_stage08_reports_both_numbers(world, tmp_path):
    screened = world["screened"]
    plain = stage08(world, tmp_path / "plain")
    assert plain["model_sel_eff"] == pytest.approx(warm(screened), rel=1e-3)
    assert "pre_rebench" not in plain
    measured = stage08(world, tmp_path / "measured", *rebench_flags(world))
    assert measured["model_sel_eff"] == pytest.approx(true(screened), rel=1e-3)
    assert measured["pre_rebench"]["sel_eff"] == pytest.approx(plain["model_sel_eff"])
    assert (measured["n_rebenched"], measured["n_rebench_failed"]) == (5, 0)
    assert measured["summary"]["pick_measurement"] == "rebench"
    rec = json.loads(
        (tmp_path / "measured" / "gate" / "rebench_misses.json").read_text()
    )
    assert [r["status"] for r in rec["gemms"]] == ["measured"] * 5
    assert all(r["n_runs"] == 2 for r in rec["gemms"])
    assert rec["library_digest"] == library_digest(world)
    assert_named(rec["gemms"], world)


def stage04b(world, out, *extra, env=None):
    run_stage(
        "stage04b_iterate_active_learning.py",
        "--models-dir",
        world["tmp"] / "round_0" / "stage05",
        "--enriched-csv-dir",
        world["tmp"] / "round_1" / "stage04",
        "--out-dir",
        out,
        "--arch",
        mf.ARCH,
        "--config-yaml",
        world["tmp"] / "config.yaml",
        "--weight-dtype",
        "bf16",
        "--library-dir",
        mf.library_dir(world["root"]),
        "--library-stem",
        mf.STEM,
        "--plan",
        world["tmp"] / "plan.json",
        "--round-idx",
        1,
        "--min-cell-gemms",
        2,
        "--train-min-cell-gemms",
        2,
        "--sel-eff-threshold",
        0.9,
        "--quiet",
        *extra,
        env=dict(mf.env(world["root"]), **(env or {})),
    )
    return json.loads((out / "decisions.json").read_text())


def test_stage04b_decides_on_rebenched_misses(world):
    screened, planned = world["screened"], world["planned"]
    out = world["tmp"] / "round_1"
    plain = stage04b(world, out / "plain")
    row = {r["cell"]: r for r in plain["per_cell"]}[CELL]
    assert row["decision"] == "retrain" and row["n_miss"] == 3
    assert plain["aggregate"]["pick_measurement"] == "warmup"
    assert "pre_rebench" not in plain["aggregate"]

    measured = stage04b(world, out / "measured", *rebench_flags(world))
    agg = measured["aggregate"]
    row = {r["cell"]: r for r in measured["per_cell"]}[CELL]
    assert row["decision"] == "pass"
    assert row["sel_eff_new"] == pytest.approx(true(screened + planned), rel=1e-3)
    assert row["sel_eff_new_pre_rebench"] == pytest.approx(
        {r["cell"]: r for r in plain["per_cell"]}[CELL]["sel_eff_new"]
    )
    assert agg["pick_measurement"] == "rebench"
    assert agg["pre_rebench"]["n_miss_by_reason"] == {"screened": 0, "not_benched": 3}
    assert agg["rebench"]["held_out"]["n_misses_by_reason"] == {
        "screened": 5,
        "not_benched": 3,
    }
    assert (agg["n_rebenched"], agg["n_rebench_failed"]) == (8, 0)
    rec = json.loads((out / "measured" / "rebench_misses.json").read_text())
    assert rec["held_out"]["library_digest"] == library_digest(world)
    planned_rows = [r for r in rec["held_out"]["gemms"] if r["reason"] == "not_benched"]
    assert len(planned_rows) == 3
    assert all(r["status"] == "measured" for r in planned_rows)
    assert_named(rec["held_out"]["gemms"], world)

    failed = stage04b(
        world,
        out / "failed",
        *rebench_flags(world, runs=1),
        env={"FAKE_UNSUPPORTED": str(mf.PICK)},
    )
    agg = failed["aggregate"]
    assert (agg["n_rebenched"], agg["n_rebench_failed"]) == (0, 8)
    row = {r["cell"]: r for r in failed["per_cell"]}[CELL]
    assert row["n_miss"] == 8 and row["decision"] == "retrain"
    assert row["sel_eff_new"] == pytest.approx(row["sel_eff_new_pre_rebench"])
