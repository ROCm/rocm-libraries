# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage02 on CPU: the fake hipblaslt-bench benches the seed shapes, the
planner writes the round's bench lines, and stage03 / stage04 run them."""

import csv
import json
import shutil
import sys
from collections import defaultdict

import pytest
import yaml

import bench_fakes as bf
import fake_hipblaslt as fh
import run_orchestrator as ro
import stage02_plan_benchmarks as s02
import stage03_load_balance_offline_tuning as s03
import stage04_convert_to_enriched_dataset as s04
from lib import kernel_identity as ki
from lib import planner as pl
from lib.bench_yaml import line_has_prescreen, line_solution_list, read_bench_lines
from lib.bench_yaml import shape_of_line
from lib.grid import cell_key

ARCH = "gfx950"
GEMMS = [bf.Gemm(m, 2048, 1024) for m in (1024, 1536, 2048, 3072)] + [
    bf.Gemm(m, 256, 256) for m in (160, 224, 288, 352)
]


def _kernels():
    out = fh.kernels_grid()
    idx = len(out)
    for mt in ((64, 128, 64), (128, 32, 64), (32, 128, 64)):
        out.append(bf.solution(idx, idx, mt=mt, mi=(16, 16, 128, 1)))
        idx += 1
    return out


@pytest.fixture
def round_dir(tmp_path, monkeypatch):
    bench_root = tmp_path / "bench"
    build = bench_root / "projects" / "hipblaslt" / "build" / "release"
    fh.write_fake_build(build, ARCH, bf.LIBRARY_STEM, _kernels())
    monkeypatch.setenv("FAKE_LIBRARY_DIR", str(build / "Tensile" / "library" / ARCH))
    monkeypatch.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    n_cu, lds, l2 = (int(v) for v in fh.FAKE_HW.split(","))
    cfg = {
        "config_id": "plan",
        "arch": ARCH,
        "hipblaslt": {
            "a_type": "f8_r",
            "b_type": "f8_r",
            "c_type": "bf16_r",
            "d_type": "bf16_r",
            "scale_type": "f32_r",
            "compute_type": "c_f32_r",
            "scaleA": 3,
            "scaleB": 3,
            "transA": "T",
            "transB": "N",
            "library_stem": bf.LIBRARY_STEM,
        },
        "hardware": {"n_cu": n_cu, "lds_bytes": lds, "l2_bytes": l2},
        "runtime": {"devices": [0], "rocm_libraries_root": str(bench_root)},
        "seed": {"target_per_cell": 40, "shapes_per_cell": 40},
        "bench": {
            "blocks_per_gpu": 1,
            "skip_slow_solution_ratio": 0.5,
            "adaptive": {"measure_time": 1.0},
        },
        "train": {"min_cell_gemms": 2},
        "validate": {"min_cell_gemms": 1},
        "planner": {
            "enabled": True,
            "mode": "planned",
            "seed_per_cell": 2,
            "top_k": 2,
            "families_top_f": 1,
            "explore_random_k": 1,
            "min_coverage": 1,
        },
    }
    assert ro.validate_config(cfg)[0] == []
    cfg_path = tmp_path / "plan.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    rd = tmp_path / "run" / "round_0"
    (rd / "stage01").mkdir(parents=True)
    lines = [bf.bench_line(g, skip_slow_solution_ratio=0.5) for g in GEMMS]
    (rd / "stage01" / "shapes.yaml").write_text("".join(lines))
    loaded, _ = ro.load_config(cfg_path)
    pipe = ro.Pipeline(
        cfg=loaded,
        snapshot=cfg_path,
        run_dir=tmp_path / "run",
        paths=ro.derive_paths(loaded),
        stages_dir=ro.THIS_DIR / "stages",
        timing=ro.Timing(),
        force_deploy=False,
    )
    return rd, pipe, cfg_path


def _main(monkeypatch, module, args):
    monkeypatch.setattr(sys, "argv", [module.__name__] + [str(a) for a in args])
    return module.main()


def _rows(out_dir):
    by_gemm = defaultdict(list)
    for path in sorted(out_dir.glob("chunk_*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                by_gemm[(int(r["m"]), int(r["n"]), int(r["k"]))].append(r)
    return by_gemm


def test_seed_bootstrap_plans_and_planned_rows_parse(round_dir, monkeypatch):
    rd, pipe, _ = round_dir
    step = ro.RoundStep(0, "initial", [], None, False)
    args = pipe.args_stage02(rd, step) + ["--quiet"]
    assert s02.main(args) == 0

    out = rd / "stage02"
    plan = json.loads((out / "plan.json").read_text())
    roles = defaultdict(list)
    for e in plan["shapes"]:
        roles[e["role"]].append((e["m"], e["n"], e["k"]))
    assert len(roles["seed"]) == 4 and len(roles["planned"]) == 4
    assert plan["mode"] == "planned" and "neighbors" in plan["ranker"]
    seed_cells = defaultdict(int)
    for m, n, k in roles["seed"]:
        seed_cells[cell_key(m, n, k, 1)] += 1
    assert set(seed_cells.values()) == {2}
    assert (plan["kernel_order"], plan["order_seed"]) == ("shuffled", 42)
    seed_lines = read_bench_lines(out / "seed" / "shapes.yaml")
    every = list(range(len(_kernels())))
    for ln in seed_lines:
        listed = pl.shuffled(every, 42, *shape_of_line(ln))
        assert line_solution_list(ln) == [listed[0], *listed]
    assert all(line_has_prescreen(ln) for ln in seed_lines)
    assert (out / "seed" / "stage04" / "chunk_0000.csv").is_file()
    lines = read_bench_lines(out / "shapes.yaml")
    assert sorted(shape_of_line(ln)[:3] for ln in lines) == sorted(roles["planned"])
    by_shape = {(e["m"], e["n"], e["k"]): e for e in plan["shapes"]}
    for ln in lines:
        e = by_shape[shape_of_line(ln)[:3]]
        listed = pl.shuffled(e["kernels"], 42, *shape_of_line(ln))
        assert line_solution_list(ln) == [listed[0], *listed]
        assert line_has_prescreen(ln)
        assert 2 <= len(e["kernels"]) <= e["n_feasible"]
        assert "probe" not in e
    summary = json.loads((out / "plan_summary.json").read_text())
    assert summary["roles"] == {"seed": 4, "planned": 4, "exhaustive": 0}
    # Each seed is ranked from the other seed of its cell.
    rec = summary["seed_recall"]
    assert (rec["ranker"], rec["plan_ranker"], rec["top_k"]) == (
        "neighbors",
        "neighbors",
        2,
    )
    assert (rec["n_seeds"], rec["n_cells"], rec["n_alone"], rec["n_unscored"]) == (
        4,
        2,
        0,
        0,
    )
    assert 0 < rec["ceiling"] <= 1 and rec["by_top_k"][0]["top_k"] == 2
    assert set(rec["cells"]) == set(seed_cells)
    assert summary["params"]["top_k"] == plan["params"]["top_k"] == 2

    first = (out / "shapes.yaml").read_text()
    shutil.rmtree(out)
    assert s02.main(args) == 0
    assert (out / "shapes.yaml").read_text() == first

    identity = ki.library_identity(pipe.paths.library_dir, [bf.LIBRARY_STEM])
    for doc in (plan, json.loads((out / "seed" / "plan.json").read_text())):
        rec = ki.read_record(doc)
        assert rec["digest"] == identity.digest
        assert ki.recorded_ids(rec) == dict(identity.ids)
    seed_bench = json.loads(
        (out / "seed" / "stage03" / "stage03_summary.json").read_text()
    )
    assert ki.read_record(seed_bench) == ki.record(identity, ())

    assert _main(monkeypatch, s03, pipe.args_stage03(rd) + ["--quiet"]) == 0
    bench = json.loads((rd / "stage03" / "stage03_summary.json").read_text())
    assert ki.read_record(bench) == ki.record(identity, ())
    assert _main(monkeypatch, s04, pipe.args_stage04(rd) + ["--quiet"]) == 0
    summary = json.loads((rd / "stage04" / "enrich_summary.json").read_text())
    assert summary["bench_records"] == 2
    by_gemm = _rows(rd / "stage04")
    assert len(by_gemm) == len(GEMMS)
    for key, rows in by_gemm.items():
        e = by_shape[key]
        assert {r["is_origami_pick"] for r in rows} == {"0"}
        assert sum(r["is_winner"] == "True" for r in rows) == 1
        if e["role"] == "seed":
            assert {(r["is_planned"], r["n_planned"]) for r in rows} == {("1", "")}
        else:
            assert {r["n_planned"] for r in rows} == {str(len(e["kernels"]))}
            assert {int(r["sol_idx_global"]) for r in rows} <= set(e["kernels"])
            assert all(r["is_planned"] == "1" for r in rows)
            ranks = sorted(int(r["rank"]) for r in rows)
            assert ranks == list(range(1, len(e["kernels"]) + 1))
    summary = json.loads((rd / "stage04" / "enrich_summary.json").read_text())
    assert summary["counters"]["sacrificial_rows"] == len(GEMMS)


def test_exhaustive_mode_and_all_seed_rounds(round_dir, monkeypatch):
    rd, pipe, cfg_path = round_dir
    step = ro.RoundStep(0, "initial", [], None, False)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["planner"]["mode"] = "exhaustive"
    cfg_path.write_text(yaml.safe_dump(cfg))
    args = pipe.args_stage02(rd, step) + ["--quiet"]
    assert s02.main(args) == 0
    plan = json.loads((rd / "stage02" / "plan.json").read_text())
    assert {e["role"] for e in plan["shapes"]} == {"exhaustive"}
    assert all(e["kernels"] is None and "probe" not in e for e in plan["shapes"])
    lines = read_bench_lines(rd / "stage02" / "shapes.yaml")
    assert len(lines) == len(GEMMS)
    every = list(range(len(_kernels())))
    for ln in lines:
        listed = pl.shuffled(every, 42, *shape_of_line(ln))
        assert line_solution_list(ln) == [listed[0], *listed]
        assert line_has_prescreen(ln)
    assert len({tuple(line_solution_list(ln)) for ln in lines}) > 1
    assert not (rd / "stage02" / "seed").exists()
    summary = json.loads((rd / "stage02" / "plan_summary.json").read_text())
    assert summary["seed_recall"] is None

    cfg["planner"].update(mode="planned", seed_per_cell=8)
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = rd / "all_seeds"
    assert s02.main(args[:-1] + ["--out-dir", str(out), "--quiet"]) == 0
    plan = json.loads((out / "plan.json").read_text())
    assert plan["mode"] == "exhaustive" and not (out / "seed").exists()

    cfg["planner"].update(mode="auto", exhaustive_max_kernels=len(_kernels()))
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = rd / "auto"
    assert s02.main(args[:-1] + ["--out-dir", str(out), "--quiet"]) == 0
    assert json.loads((out / "plan.json").read_text())["mode"] == "exhaustive"


def test_bad_settings_and_missing_seed_commands(round_dir):
    rd, pipe, cfg_path = round_dir
    step = ro.RoundStep(0, "initial", [], None, False)
    args = pipe.args_stage02(rd, step)
    i = args.index("--seed-commands")
    no_seed = args[:i] + args[i + 2 :] + ["--quiet"]
    assert s02.main(no_seed) == 1
    stale = json.loads(args[i + 1])
    stale["stage02"] = stale["stage03"]
    assert s02.main(args[: i + 1] + [json.dumps(stale)] + args[i + 2 :]) == 1
    cfg = yaml.safe_load(cfg_path.read_text())
    for bad in ({"top_k_round0": -1}, {"recall_min_ceiling": 1.5}):
        planner = {**cfg["planner"], **bad}
        cfg_path.write_text(yaml.safe_dump({**cfg, "planner": planner}))
        assert s02.main(args + ["--quiet"]) == 1
    assert not (rd / "stage02" / "seed" / "stage03").exists()
    cfg["planner"]["ranker"] = "origami"
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert s02.main(args + ["--quiet"]) == 1


def test_top_k_round0_lengthens_only_the_lists_the_seeds_rank(round_dir, monkeypatch):
    """planner.top_k_round0 is the top_k of the lists the seed bootstrap
    ranks, the lists the seed recall scores; a round a model ranks keeps
    planner.top_k and scores nothing."""
    rd, pipe, cfg_path = round_dir
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["planner"]["top_k_round0"] = 5
    cfg_path.write_text(yaml.safe_dump(cfg))
    step = ro.RoundStep(0, "initial", [], None, False)
    args = pipe.args_stage02(rd, step) + ["--quiet"]
    assert s02.main(args) == 0
    out = rd / "stage02"
    plan = json.loads((out / "plan.json").read_text())
    summary = json.loads((out / "plan_summary.json").read_text())
    assert plan["params"]["top_k"] == summary["seed_recall"]["top_k"] == 5
    planned = [e for e in plan["shapes"] if e["role"] == "planned"]
    assert planned and {e["sources"]["top_k"] for e in planned} == {5}

    class FixedModelRanker(s02.Ranker):
        name = "model"

        def __init__(self, data, catalog, ptype, hardware, **kw):
            self.ids = list(catalog.pool_ids)
            self.rules = dict(kw["rules"])

        def rank(self, shape):
            return list(self.ids)

    monkeypatch.setattr(s02, "ModelRanker", FixedModelRanker)
    monkeypatch.setattr(s02, "model_bytes", lambda *a, **kw: b"model")
    model = rd.parent / "m.pt"
    model.write_text("m")
    i = args.index("--seed-commands")
    assert s02.main(args[:i] + args[i + 2 :] + ["--model-file", str(model)]) == 0
    plan = json.loads((out / "plan.json").read_text())
    summary = json.loads((out / "plan_summary.json").read_text())
    assert plan["params"]["top_k"] == 2 and summary["seed_recall"] is None
    planned = [e for e in plan["shapes"] if e["role"] == "planned"]
    assert len(planned) == len(GEMMS)
    assert {e["sources"]["top_k"] for e in planned} == {2}


def test_bench_lines_list_their_kernels_in_a_seeded_shuffle(round_dir):
    rd, pipe, cfg_path = round_dir
    step = ro.RoundStep(0, "initial", [], None, False)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["planner"]["mode"] = "exhaustive"
    every = list(range(len(_kernels())))
    lists = {}
    for order, seed in (("shuffled", 7), ("listed", 7), ("shuffled", 8)):
        cfg["bench"].update(kernel_order=order, order_seed=seed)
        cfg["bench"]["sacrificial_kernel"] = False
        cfg_path.write_text(yaml.safe_dump(cfg))
        out = rd / f"{order}_{seed}"
        args = pipe.args_stage02(rd, step)
        i = args.index("--out-dir")
        assert s02.main(args[:i] + ["--out-dir", str(out)] + args[i + 2 :]) == 0
        plan = json.loads((out / "plan.json").read_text())
        summary = json.loads((out / "plan_summary.json").read_text())
        recorded = (plan["kernel_order"], plan["order_seed"])
        assert recorded == (summary["kernel_order"], summary["order_seed"])
        assert plan["sacrificial_kernel"] is summary["sacrificial_kernel"] is False
        lines = read_bench_lines(out / "shapes.yaml")
        lists[order, seed] = {shape_of_line(ln): line_solution_list(ln) for ln in lines}
        if order == "listed":
            assert recorded == ("listed", None)
            assert all(v == every for v in lists[order, seed].values())
            continue
        assert recorded == ("shuffled", seed)
        for shape, listed in lists[order, seed].items():
            assert listed == pl.shuffled(every, seed, *shape)
    assert lists["shuffled", 7] != lists["shuffled", 8]


def test_render_shuffles_explicit_lists_only():
    line = bf.bench_line(bf.Gemm(64, 64, 64), skip_slow_solution_ratio=0.5)
    plan = s02.ShapePlan((64, 64, 64, 1), "planned", [5, 1, 9, 3])
    assert line_solution_list(s02.render_planned_line(line, plan)) == [5, 1, 9, 3]
    out = s02.render_planned_line(line, plan, None, 42)
    assert line_solution_list(out) == pl.shuffled([5, 1, 9, 3], 42, 64, 64, 64, 1)
    every = s02.ShapePlan((64, 64, 64, 1), "exhaustive", None)
    assert "algo_method: 1" in s02.render_planned_line(line, every, None, 42)


def test_screened_lines_lead_with_a_sacrificial_copy(monkeypatch):
    plan = s02.ShapePlan((64, 64, 64, 1), "planned", [5, 1, 9, 3])
    screened = bf.bench_line(bf.Gemm(64, 64, 64), skip_slow_solution_ratio=0.5)
    out = s02.render_planned_line(screened, plan, None, 42, True)
    listed = pl.shuffled([5, 1, 9, 3], 42, 64, 64, 64, 1)
    assert line_solution_list(out) == [listed[0], *listed]
    out = s02.render_planned_line(screened, plan, None, None, True)
    assert line_solution_list(out) == [5, 5, 1, 9, 3]
    unscreened = bf.bench_line(bf.Gemm(64, 64, 64))
    assert "skip_slow_solution_ratio" not in unscreened
    out = s02.render_planned_line(unscreened, plan, None, None, True)
    assert line_solution_list(out) == [5, 1, 9, 3]
    zero = bf.bench_line(bf.Gemm(64, 64, 64), skip_slow_solution_ratio=0)
    out = s02.render_planned_line(zero, plan, None, None, True)
    assert line_solution_list(out) == [5, 1, 9, 3]
    monkeypatch.setattr(s02, "MAX_BENCH_LIST", 4)
    out = s02.render_planned_line(screened, plan, None, None, True)
    assert line_solution_list(out) == [5, 1, 9, 3]


@pytest.mark.parametrize("sacrificial", [True, False])
def test_the_sacrificial_copy_keeps_the_first_kernel_from_the_screen(
    round_dir, monkeypatch, sacrificial
):
    """The fake bench reads the first listed kernel's screen call twenty
    times too slow, as the real bench reads it after the data
    initialisation: the fastest kernel listed first is screened out unless a
    copy of it absorbs that call."""
    rd, pipe, _ = round_dir
    monkeypatch.setenv("FAKE_FIRST_WARMUP", "20")
    sols = {s["index"]: s for s in _kernels()}

    def us(g, sid):
        return fh.kernel_time_us(g.m, g.n, g.k, g.batch, sols[sid]["sizeMapping"])

    stage = rd / "manual"
    stage.mkdir()
    lines, fastest, plans = [], {}, []
    for g in GEMMS:
        order = sorted(sols, key=lambda sid: us(g, sid))
        fastest[(g.m, g.n, g.k)] = order[0]
        plan = s02.ShapePlan((g.m, g.n, g.k, 1), "planned", order[:6])
        plans.append(plan)
        line = bf.bench_line(g, skip_slow_solution_ratio=0.5)
        lines.append(s02.render_planned_line(line, plan, None, None, sacrificial))
    (stage / "shapes.yaml").write_text("".join(lines))
    (stage / "plan.json").write_text(json.dumps(pl.plan_document(plans, {})))
    args = pipe.args_stage03(rd, in_dir=stage, out_dir=stage / "stage03")
    assert _main(monkeypatch, s03, args + ["--quiet"]) == 0
    logs = [stage / "stage03" / "logs"]
    args = pipe.args_stage04(
        rd, log_dirs=logs, out_dir=stage / "stage04", plan=stage / "plan.json"
    )
    assert _main(monkeypatch, s04, args + ["--quiet"]) == 0
    for key, rows in _rows(stage / "stage04").items():
        best = next(r for r in rows if int(r["sol_idx_global"]) == fastest[key])
        if sacrificial:
            assert best["is_skip"] == "False" and best["is_winner"] == "True"
            assert min(int(r["rank"]) for r in rows) == 1
        else:
            assert best["is_skip"] == "True" and best["is_winner"] == "False"
            assert int(best["rank"]) == 0


def test_exhaustive_beyond_the_bench_list_limit_lists_every_kernel(monkeypatch):
    plan = s02.ShapePlan((64, 64, 64, 1), "exhaustive", None, n_feasible=3)
    line = bf.bench_line(bf.Gemm(64, 64, 64), skip_slow_solution_ratio=0.5)
    cat = s02.Catalog.build(_kernels(), [])
    listed = s02.render_planned_line(line, plan, s02.exhaustive_list(cat))
    assert line_solution_list(listed) == cat.pool_ids
    monkeypatch.setattr(s02, "MAX_BENCH_LIST", len(cat.pool_ids) - 1)
    assert s02.exhaustive_list(cat) is None
    every = s02.render_planned_line(listed, plan, None)
    assert line_solution_list(every) is None
    assert "algo_method: 1" in every and "requested_solution_num: -1" in every
    assert line_has_prescreen(every)


def test_contexts_are_replanned_with_the_model(round_dir, monkeypatch):
    rd, pipe, cfg_path = round_dir
    seen = []
    own = {"dot2_rule": False, "non_temporal_rule": False}

    class FakeModelRanker(s02.Ranker):
        name = "model"

        def __init__(self, data, catalog, ptype, hardware, context=None, **kw):
            self.ids = list(catalog.pool_ids)
            self.rules = dict(kw["rules"]) if kw.get("rules") is not None else own
            self.budget = context.cu_budget if context is not None else 0
            seen.append(
                None if context is None else (context.cu_budget, context.schedule)
            )

        def rank(self, shape):
            return self.ids[::-1] if self.budget else self.ids

    monkeypatch.setattr(s02, "ModelRanker", FakeModelRanker)
    monkeypatch.setattr(s02, "model_bytes", lambda *a, **kw: b"model")
    model = rd.parent / "m.pt"
    model.write_text("m")
    step = ro.RoundStep(0, "initial", [], None, False)
    args = pipe.args_stage02(rd, step)
    i = args.index("--seed-commands")
    args = args[:i] + args[i + 2 :] + ["--model-file", str(model), "--quiet"]
    wanted = [{"name": "cu32_cot", "cu_budget": 32, "schedule": "dynamic"}]
    assert s02.main(args + ["--contexts", json.dumps(wanted)]) == 0
    assert seen == [None, (32, "dynamic")]
    out = rd / "stage02"
    rules = json.loads((out / "plan.json").read_text())["rules"]
    assert rules == s02.config_rules(yaml.safe_load(cfg_path.read_text()))
    base = {shape_of_line(ln): ln for ln in read_bench_lines(out / "shapes.yaml")}
    ctx_dir = out / "contexts" / "cu32_cot"
    ctx = {shape_of_line(ln): ln for ln in read_bench_lines(ctx_dir / "shapes.yaml")}
    assert set(ctx) == set(base) and len(ctx) == len(GEMMS)
    assert any(line_solution_list(ctx[s]) != line_solution_list(base[s]) for s in ctx)
    plan = json.loads((ctx_dir / "plan.json").read_text())
    assert plan["context"] == {"cu_budget": 32, "schedule": "dynamic"}
    identity = ki.library_identity(pipe.paths.library_dir, [bf.LIBRARY_STEM])
    assert ki.recorded_ids(ki.read_record(plan)) == dict(identity.ids)
    assert s02.main(args + ["--contexts", "[{}]"]) == 1

    from lib.contexts import BenchContext, select_shapes

    seen.clear()
    half = [dict(wanted[0], fraction=0.5)]
    assert s02.main(args + ["--contexts", json.dumps(half)]) == 0
    ctx = {shape_of_line(ln) for ln in read_bench_lines(ctx_dir / "shapes.yaml")}
    stage03c = select_shapes(
        list(base), BenchContext.from_mapping(dict(half[0], cotenant=True)), 42
    )
    assert ctx == set(stage03c) and 0 < len(ctx) < len(base)
    plan = json.loads((ctx_dir / "plan.json").read_text())
    assert {(e["m"], e["n"], e["k"], e["batch"]) for e in plan["shapes"]} == ctx
    assert (plan["fraction"], plan["context_seed"]) == (0.5, 42)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["planner"]["rules"] = "model"
    own_cfg = cfg_path.with_name("own_rules.yaml")
    own_cfg.write_text(yaml.safe_dump(cfg))
    i = args.index("--config-yaml")
    own_args = args[: i + 1] + [str(own_cfg)] + args[i + 2 :]
    assert s02.main(own_args) == 0
    assert json.loads((out / "plan.json").read_text())["rules"] == own
    cfg["planner"]["rules"] = "serving"
    own_cfg.write_text(yaml.safe_dump(cfg))
    assert s02.main(own_args) == 1


@pytest.mark.parametrize("family_rule", [True, False])
def test_the_family_rule_lists_the_members_it_keeps_in_its_region(
    round_dir, monkeypatch, family_rule
):
    """A batched, M-thin shape with K on the DepthU alignment: with a
    non-temporal B variant in the pool, the non-temporal rule rejects a
    family that exists only with cache hints, and the family-wise rule keeps
    its lowest hints. Its plan (and each context's) lists that member
    whatever the ranker, in the shuffled, screened bench line."""
    rd, pipe, cfg_path = round_dir
    kernels = _kernels()
    base = len(kernels)
    kernels += [
        bf.solution(base, base, mt=(16, 128, 128), ntb=4),
        bf.solution(base + 1, base + 1, mt=(16, 256, 128), nta=1, ntb=1),
        bf.solution(base + 2, base + 2, mt=(16, 256, 128), nta=1, ntb=3),
    ]
    fh.write_fake_build(
        pipe.paths.library_dir.parent.parent.parent, ARCH, bf.LIBRARY_STEM, kernels
    )
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["train"].update(catalog="v3", non_temporal_rule=True)
    cfg["train"]["non_temporal_family_rule"] = family_rule
    cfg["planner"].update(top_k=1, families_top_f=0, explore_random_k=0, min_coverage=0)
    cfg_path.write_text(yaml.safe_dump(cfg))
    region, batch1 = bf.Gemm(16, 2048, 256, 4), bf.Gemm(16, 2048, 256, 1)
    (rd / "stage01" / "shapes.yaml").write_text(
        "".join(
            bf.bench_line(g, skip_slow_solution_ratio=0.5) for g in (region, batch1)
        )
    )

    class FakeModelRanker(s02.Ranker):
        def __init__(self, data, catalog, ptype, hardware, context=None, **kw):
            self.ids = list(catalog.pool_ids)
            self.rules = dict(kw["rules"])

        def rank(self, shape):
            return self.ids

    monkeypatch.setattr(s02, "ModelRanker", FakeModelRanker)
    monkeypatch.setattr(s02, "model_bytes", lambda *a, **kw: b"model")
    model = rd.parent / "m.pt"
    model.write_text("m")
    args = pipe.args_stage02(rd, ro.RoundStep(0, "initial", [], None, False))
    i = args.index("--seed-commands")
    args = args[:i] + args[i + 2 :] + ["--model-file", str(model), "--quiet"]
    wanted = [{"name": "cu32_cot", "cu_budget": 32, "schedule": "default"}]
    assert s02.main(args + ["--contexts", json.dumps(wanted)]) == 0
    out = rd / "stage02"
    plan = json.loads((out / "plan.json").read_text())
    assert plan["family_rule_coverage"] is family_rule
    summary = json.loads((out / "plan_summary.json").read_text())
    assert summary["family_rule_shapes"] == (1 if family_rule else 0)
    for doc, folder in (
        (plan, out),
        (json.loads((out / "contexts" / "cu32_cot" / "plan.json").read_text()), None),
    ):
        entries = {(e["m"], e["n"], e["k"], e["batch"]): e for e in doc["shapes"]}
        kept = entries[(16, 2048, 256, 4)]
        assert (base + 1 in kept["kernels"]) is family_rule
        assert kept["sources"].get("family_rule", 0) == (1 if family_rule else 0)
        assert base + 2 not in kept["kernels"]
        other = entries[(16, 2048, 256, 1)]
        assert (
            base + 1 not in other["kernels"] and "family_rule" not in other["sources"]
        )
        if folder is None:
            continue
        for ln in read_bench_lines(folder / "shapes.yaml"):
            listed = pl.shuffled(
                entries[shape_of_line(ln)]["kernels"], 42, *shape_of_line(ln)
            )
            assert line_solution_list(ln) == [listed[0], *listed]


EXIT_AS = """
import os, signal, sys
how = sys.argv[1]
if how.startswith("SIG"):
    os.kill(os.getpid(), getattr(signal, how))
sys.exit(int(how))
"""


@pytest.mark.parametrize(
    "stage03, stage04, rc",
    [
        ("2", "0", 2),
        ("SIGKILL", "0", 2),
        ("1", "0", 1),
        ("0", "2", 1),
        ("0", "SIGTERM", 2),
        ("0", "0", 1),
    ],
    ids=[
        "bench_gpus",
        "bench_killed",
        "bench_setup",
        "enrich_empty",
        "enrich_killed",
        "no_measurements",
    ],
)
def test_the_seed_bootstrap_exits_2_only_for_what_a_retry_may_cure(
    round_dir, tmp_path, stage03, stage04, rc
):
    rd, pipe, _ = round_dir
    step = ro.RoundStep(0, "initial", [], None, False)
    args = pipe.args_stage02(rd, step)
    script = tmp_path / "exit_as.py"
    script.write_text(EXIT_AS)
    i = args.index("--seed-commands")
    cmds = {"stage03": [str(script), stage03], "stage04": [str(script), stage04]}
    args[i + 1] = json.dumps(cmds)
    assert s02.main(args + ["--quiet"]) == rc


def test_config_rules_follow_the_catalog_serving_options():
    v2 = s02.config_rules({})
    assert v2["small_batched_rule"] and v2["dot2_rule"] and v2["non_temporal_rule"]
    assert not s02.config_rules({"train": {"non_temporal_rule": False}})[
        "non_temporal_rule"
    ]
    v3 = s02.config_rules({"train": {"catalog": "v3"}})
    assert not any(v3.values())
    v3_nt = s02.config_rules(
        {"train": {"catalog": "v3", "non_temporal_rule": True, "dot2_rule": True}}
    )
    assert v3_nt == {
        "leading_free_rule": False,
        "small_batched_rule": False,
        "dot2_rule": True,
        "non_temporal_rule": True,
    }


def _bank_file(lib, cu_count, indices):
    """A `_CU<cu_count>` file of the library's problem type whose Equality
    row holds `indices`, for devices of that many CUs."""
    import msgpack

    sols = [
        dict(
            bf.solution(i, i, mt=(64, 64, 64), mi=(16, 16, 128, 1)),
            hardwarePredicate={
                "type": "AMDGPU",
                "value": {"type": "CUCount", "value": cu_count},
            },
        )
        for i in indices
    ]
    table = [{"key": [i], "index": i} for i in indices]
    row = {
        "predicate": {"type": "EqualityMatching"},
        "library": {"type": "Matching", "table": table},
    }
    stem = bf.LIBRARY_STEM.replace("_gfx1250", f"_CU{cu_count}_gfx1250")
    doc = {"solutions": sols, "library": {"type": "Problem", "rows": [row]}}
    (lib / f"{stem}.dat").write_bytes(msgpack.packb(doc))
    return stem


def test_bank_auto_follows_the_runtime_pool(round_dir):
    """`planner.bank: auto` adds the Equality-row kernels of the problem
    type's other files that the device accepts, in ascending index."""
    rd, pipe, cfg_path = round_dir
    n_cu = int(fh.FAKE_HW.split(",")[0])
    lib = pipe.paths.library_dir
    base = len(_kernels())
    accepted = _bank_file(lib, n_cu, [base + 1, base, 2])
    _bank_file(lib, n_cu + 1, [base + 2])
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["planner"].update(mode="exhaustive", bank="auto")
    cfg["hardware"]["pci_chip_id"] = 0x75A0
    cfg_path.write_text(yaml.safe_dump(cfg))
    step = ro.RoundStep(0, "initial", [], None, False)
    assert s02.main(pipe.args_stage02(rd, step) + ["--quiet"]) == 0
    every = list(range(base)) + [base, base + 1]
    lines = read_bench_lines(rd / "stage02" / "shapes.yaml")
    assert all(sorted(set(line_solution_list(ln))) == every for ln in lines)
    plan = json.loads((rd / "stage02" / "plan.json").read_text())
    assert plan["bank"] == "auto" and plan["bank_stems_used"] == [accepted]
    assert plan["library_kind"] == ["origami", "bok"]


def test_a_tdm_kernel_and_its_th_twin_are_benched_once(round_dir):
    """A kernel whose operands the tensor data mover loads and its THA/THB
    twin are one kernel: the lines list the first of them in pool order,
    plan.json records the other as its alias; a variant that stores D with
    another hint is a kernel of its own."""
    from lib import dat
    from test_kernel_identity import tdm_solution

    rd, pipe, cfg_path = round_dir
    base = len(_kernels())
    twins = [
        tdm_solution(base, base),
        tdm_solution(base + 1, base + 1, tha=1, thb=1),
        tdm_solution(base + 2, base + 2, thd=1),
    ]
    bf.write_logic(
        pipe.paths.library_dir / f"{bf.LIBRARY_STEM}.dat", _kernels() + twins
    )
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["planner"]["mode"] = "exhaustive"
    cfg_path.write_text(yaml.safe_dump(cfg))
    step = ro.RoundStep(0, "initial", [], None, False)
    assert s02.main(pipe.args_stage02(rd, step) + ["--quiet"]) == 0
    benched = list(range(base)) + [base, base + 2]
    lines = read_bench_lines(rd / "stage02" / "shapes.yaml")
    assert lines and all(sorted(set(line_solution_list(ln))) == benched for ln in lines)
    plan = json.loads((rd / "stage02" / "plan.json").read_text())
    rec = plan[ki.RECORD_KEY]
    assert rec["rule"] == ki.IDENTITY_RULE and rec[ki.ALIASES] == {str(base + 1): base}
    assert ki.recorded_ids(rec)[base] == ki.kernel_id(dat.kernel_dat_info(twins[1]))
    summary = json.loads((rd / "stage02" / "plan_summary.json").read_text())
    assert plan["n_aliases"] == summary["n_aliases"] == 1


def test_library_kind_plans_the_same_pool(round_dir):
    """`library.kind [origami, bok]` plans over the pool of `planner.bank:
    auto` with the library's Prediction table, and stage02 sets no bank key
    of the planner beside it."""
    rd, pipe, cfg_path = round_dir
    n_cu = int(fh.FAKE_HW.split(",")[0])
    lib = pipe.paths.library_dir
    kernels = _kernels()
    base = len(kernels)
    bf.write_logic(lib / f"{bf.LIBRARY_STEM}.dat", kernels, table=list(range(base)))
    accepted = _bank_file(lib, n_cu, [base + 1, base, 2])
    _bank_file(lib, n_cu + 1, [base + 2])
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["planner"]["mode"] = "exhaustive"
    cfg["library"] = {"kind": ["origami", "bok"]}
    cfg["train"]["catalog"] = "v3"
    cfg["hardware"]["pci_chip_id"] = 0x75A0
    assert ro.validate_config(cfg)[0] == []
    cfg_path.write_text(yaml.safe_dump(cfg))
    step = ro.RoundStep(0, "initial", [], None, False)
    assert s02.main(pipe.args_stage02(rd, step) + ["--quiet"]) == 0
    every = list(range(base)) + [base, base + 1]
    lines = read_bench_lines(rd / "stage02" / "shapes.yaml")
    assert all(sorted(set(line_solution_list(ln))) == every for ln in lines)
    plan = json.loads((rd / "stage02" / "plan.json").read_text())
    assert plan["library_kind"] == ["origami", "bok"]
    assert plan["bank_stems_used"] == [accepted]
