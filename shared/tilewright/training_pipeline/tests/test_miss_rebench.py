# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse

import pytest

import miss_fakes as mf
from lib import evaluate as ev
from lib import miss_rebench as mr
from lib import rebench as rb


@pytest.fixture
def fake(tmp_path, monkeypatch):
    root = mf.build(tmp_path)
    for k, v in mf.env(root).items():
        monkeypatch.setenv(k, v)
    return root


def miss_bench(root, work, cfg=None, **kw):
    return mr.MissBench(
        bench_binary=mf.bench_binary(root),
        work_dir=work,
        cfg=cfg if cfg is not None else {"hipblaslt": _hipblaslt()},
        **kw,
    )


def _hipblaslt():
    return mf.config_dict()["hipblaslt"]


def result(index, pick, served_by="model"):
    return ev.GemmResult(
        index=index,
        m=1,
        n=1,
        k=1,
        batch=1,
        leaf="x",
        model_cell="x",
        served_by=served_by,
        winner_us=1.0,
        pick_sol_idx=pick,
    )


def test_find_misses_names_the_picks_without_a_timed_row():
    shape = (640, 1536, 1024, 1)
    gemms = [
        mf.gemm(shape),
        mf.gemm(shape, miss="not_benched"),
        mf.gemm(shape, miss="none"),
        mf.gemm(shape),
    ]
    evaluation = ev.Evaluation(
        results=[
            result(0, mf.PICK),
            result(1, mf.PICK),
            result(2, mf.PICK),
            result(3, 0, served_by="origami_fallback"),
        ],
        weight_dtype="bf16",
        min_scored=1,
    )
    assert mr.find_misses(evaluation, gemms) == [
        mr.Miss(0, mf.PICK, "screened"),
        mr.Miss(1, mf.PICK, "not_benched"),
    ]


def test_misses_are_measured_with_the_winner(fake, tmp_path):
    shapes = mf.near_pick_shapes(3, 0.9)
    gemms = [
        mf.gemm(shapes[0]),
        mf.gemm(shapes[1], miss="not_benched"),
        mf.gemm(shapes[2]),
    ]
    gemms[2]["rebench"] = {mf.PICK: 0.5}
    misses = [
        mr.Miss(0, mf.PICK, "screened"),
        mr.Miss(1, mf.PICK, "not_benched"),
        mr.Miss(2, mf.PICK, "screened"),
    ]
    rec = miss_bench(fake, tmp_path / "work", runs=2).measure(gemms, misses, "t")
    assert (rec["n_misses"], rec["n_rebenched"], rec["n_rebench_reused"]) == (3, 2, 1)
    assert rec["n_rebench_failed"] == rec["n_context_skipped"] == 0
    assert rec["n_misses_by_reason"] == {"screened": 2, "not_benched": 1}
    assert rec["n_benched_gemms"] == 2 and rec["bench_rc"] == [0]
    for g, row in zip(gemms[:2], rec["gemms"][:2]):
        want = mf.true_sel_eff((g["m"], g["n"], g["k"], 1))
        assert row["status"] == "measured" and row["n_runs"] == 2
        assert g["rebench"][mf.PICK] == pytest.approx(want, rel=1e-3)
        assert row["ratio"] == pytest.approx(want, rel=1e-3)
    assert rec["gemms"][2]["status"] == "reused" and gemms[2]["rebench"] == {
        mf.PICK: 0.5
    }
    lines = rb.read_log_lines(
        (tmp_path / "work" / "t" / "exclusive" / "run_1" / "shapes.yaml").read_text()
    )
    from lib.bench_yaml import line_solution_list

    assert [line_solution_list(ln)[0] for ln in lines] != [mf.PICK, mf.PICK]
    assert all(mf.PICK in line_solution_list(ln) for ln in lines)


def test_the_off_list_charge_is_the_geomean_the_bench_measured(fake, tmp_path):
    shapes = mf.near_pick_shapes(4, 0.8)
    gemms = [mf.gemm(s, miss="not_benched") for s in shapes[:3]] + [mf.gemm(shapes[3])]
    misses = [mr.Miss(i, mf.PICK, "not_benched") for i in range(3)]
    misses.append(mr.Miss(3, mf.PICK, "screened"))
    rec = miss_bench(fake, tmp_path / "work", runs=1).measure(gemms, misses, "t")
    want = ev.geomean(mf.true_sel_eff(s) for s in shapes[:3])
    charge, report = mr.off_list_charge({"held_out": rec}, min_misses=3)
    assert charge.sel_eff == pytest.approx(want, rel=1e-3) and charge.by_context == {}
    assert (report["source"], report["n"], report["n_by_context"]) == (
        "rebench",
        3,
        {"0/default": 3},
    )
    charge, report = mr.off_list_charge({"held_out": rec}, min_misses=4, fallback=0.9)
    assert (charge.sel_eff, report["source"], report["n"]) == (0.9, "fallback", 3)
    assert mr.off_list_charge(None, min_misses=1)[0] == ev.OffListCharge(None)


def test_the_off_list_charge_has_a_context_of_its_own_with_enough_misses():
    def row(budget, ratio, reason="not_benched", status="measured"):
        return dict(
            cu_budget=budget,
            schedule="default",
            reason=reason,
            status=status,
            ratio=ratio,
        )

    rows = [row(0, 0.5), row(0, 1.2), row(64, 0.25), row(64, 0.25), row(128, 0.8)]
    rows += [row(0, 0.1, reason="screened"), row(64, None, status="failed")]
    charge, report = mr.off_list_charge({"held_out": {"gemms": rows}}, min_misses=2)
    assert charge.sel_eff == pytest.approx((0.5 * 1.0 * 0.25 * 0.25 * 0.8) ** 0.2)
    assert charge.by_context == {
        (0, "default"): pytest.approx(0.5**0.5),
        (64, "default"): pytest.approx(0.25),
    }
    assert report["n_by_context"] == {"0/default": 2, "128/default": 1, "64/default": 2}
    assert charge.for_gemm({"cu_budget": 64}) == pytest.approx(0.25)
    assert charge.for_gemm({"cu_budget": 128}) == charge.sel_eff
    assert charge.for_gemm({}) == pytest.approx(0.5**0.5)


def test_an_off_list_pick_the_bench_cannot_run_stays_a_miss(
    fake, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_UNSUPPORTED", str(mf.PICK))
    shape = mf.near_pick_shapes(1, 0.9)[0]
    gemms = [mf.gemm(shape, miss="not_benched")]
    rec = miss_bench(fake, tmp_path / "work", runs=1).measure(
        gemms, [mr.Miss(0, mf.PICK, "not_benched")], "t"
    )
    assert (rec["n_rebenched"], rec["n_rebench_failed"]) == (0, 1)
    assert rec["gemms"][0]["status"] == "failed" and "rebench" not in gemms[0]


def test_a_context_gemm_is_rebenched_in_its_context(fake, tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "5")
    budget = 32
    shape = mf.near_pick_shapes(1, 0.85, width=budget)[0]
    cfg = {
        "hipblaslt": _hipblaslt(),
        "contexts": {"list": [{"cu_budget": budget, "cotenant": True}]},
    }
    gemms = [
        mf.gemm(shape, width=budget, cu_budget=budget),
        mf.gemm(shape, width=budget, cu_budget=16),
    ]
    misses = [mr.Miss(0, mf.PICK, "screened"), mr.Miss(1, mf.PICK, "screened")]
    bench = miss_bench(
        fake,
        tmp_path / "work",
        cfg,
        runs=1,
        n_cu=mf.N_CU,
        cotenant=fake / "clients" / "hipblaslt-cotenant",
    )
    rec = bench.measure(gemms, misses, "t")
    assert [r["status"] for r in rec["gemms"]] == ["measured", "context_not_configured"]
    assert (rec["n_rebenched"], rec["n_context_skipped"]) == (1, 1)
    want = mf.true_sel_eff(shape, width=budget)
    assert want != pytest.approx(mf.true_sel_eff(shape), rel=1e-3)
    assert gemms[0]["rebench"][mf.PICK] == pytest.approx(want, rel=1e-3)
    logs = list((tmp_path / "work" / "t" / "cu32_cot" / "cotenant_logs").iterdir())
    assert logs and logs[0].read_text() == f"cus {mf.N_CU - budget} device 5\n"
    without = miss_bench(fake, tmp_path / "w2", cfg, runs=1).measure(
        [mf.gemm(shape, width=budget, cu_budget=budget)], misses[:1], "t"
    )
    assert without["gemms"][0]["status"] == "cotenant_unavailable"


def test_measured_evaluation_reevaluates_with_the_ratios(fake, tmp_path):
    shape = mf.near_pick_shapes(1, 0.9)[0]
    gemms = [mf.gemm(shape)]
    calls = []

    def evaluate(rule):
        calls.append(rule)
        timed, screened = ev.measurements(gemms[0])
        _, cost, miss = ev.pick_cost(
            mf.PICK, timed, screened, None, rule, gemms[0].get("rebench")
        )
        res = result(0, mf.PICK)
        res.winner_us, res.pick_us, res.miss = gemms[0]["winner_us"], cost, miss
        return ev.Evaluation([res], "bf16", 1, pick_measurement=rule)

    out = mr.measured_evaluation(
        evaluate, gemms, miss_bench(fake, tmp_path / "w", runs=1), "warmup", "t"
    )
    assert calls == ["warmup", "rebench"]
    assert out.pre.results[0].sel_eff == pytest.approx(
        mf.true_sel_eff(shape) / mf.WARMUP_FACTOR, rel=1e-3
    )
    assert out.evaluation.results[0].sel_eff == pytest.approx(
        mf.true_sel_eff(shape), rel=1e-3
    )
    assert out.record["n_rebenched"] == 1
    plain = mr.measured_evaluation(evaluate, gemms, None, "warmup", "t")
    assert plain.evaluation is plain.pre and plain.record is None
    calls.clear()
    mr.measured_evaluation(evaluate, [mf.gemm(shape)], None, "any", "t")
    assert calls == ["any"]


def test_stage_options_build_the_bench(fake, tmp_path):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-yaml")
    ap.add_argument("--library-dir")
    mr.add_arguments(ap)
    lines = tmp_path / "lines.yaml"
    from bench_fakes import Gemm, bench_line

    lines.write_text(bench_line(Gemm(64, 64, 64, 1)) + "\n")
    args = ap.parse_args(
        [
            "--rebench-misses",
            "--bench-binary",
            str(mf.bench_binary(fake)),
            "--rebench-lines",
            str(lines),
            "--pool-only-library",
            mf.STEM,
            "--library-dir",
            "L",
            "--stall-timeout-s",
            "5",
        ]
    )
    bench = mr.from_args(args, {}, tmp_path / "w", 64)
    assert bench.runs == rb.DEFAULT_RUNS and set(bench.lines) == {(64, 64, 64, 1)}
    assert bench.stage03_args == [
        "--stall-timeout-s",
        "5.0",
        "--library-dir",
        "L",
        "--pool-only-library",
        mf.STEM,
    ]
    assert bench.cotenant == fake / "clients" / "hipblaslt-cotenant"
    assert bench.order_seed == 42
    listed = {"bench": {"kernel_order": "listed"}}
    assert mr.from_args(args, listed, tmp_path / "w", 64).order_seed is None
    seeded = {"bench": {"order_seed": 7}}
    assert mr.from_args(args, seeded, tmp_path / "w", 64).order_seed == 7
    assert mr.from_args(ap.parse_args([]), {}, tmp_path, 64) is None
    with pytest.raises(ValueError, match="bench-binary"):
        mr.from_args(ap.parse_args(["--rebench-misses"]), {}, tmp_path, 64)
