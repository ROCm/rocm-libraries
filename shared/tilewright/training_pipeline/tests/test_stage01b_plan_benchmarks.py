# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage01b on CPU: the fake hipblaslt-bench benches the seed shapes, the
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
import stage01b_plan_benchmarks as s01b
import stage03_load_balance_offline_tuning as s03
import stage04_convert_to_enriched_dataset as s04
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
            "adaptive": {"enabled": True, "measure_time": 1.0},
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
    args = pipe.args_stage01b(rd, step) + ["--quiet"]
    assert s01b.main(args) == 0

    out = rd / "stage01b"
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
    seed_lines = read_bench_lines(out / "seed" / "shapes.yaml")
    every = list(range(len(_kernels())))
    assert all(line_solution_list(ln) == every for ln in seed_lines)
    assert all(line_has_prescreen(ln) for ln in seed_lines)
    assert (out / "seed" / "stage04" / "chunk_0000.csv").is_file()
    lines = read_bench_lines(out / "shapes.yaml")
    assert sorted(shape_of_line(ln)[:3] for ln in lines) == sorted(roles["planned"])
    by_shape = {(e["m"], e["n"], e["k"]): e for e in plan["shapes"]}
    for ln in lines:
        e = by_shape[shape_of_line(ln)[:3]]
        assert line_solution_list(ln) == e["kernels"] and line_has_prescreen(ln)
        assert 2 <= len(e["kernels"]) <= e["n_feasible"]
        assert set(e["probe"]) <= set(e["kernels"])
    summary = json.loads((out / "plan_summary.json").read_text())
    assert summary["roles"] == {"seed": 4, "planned": 4, "exhaustive": 0}

    first = (out / "shapes.yaml").read_text()
    shutil.rmtree(out)
    assert s01b.main(args) == 0
    assert (out / "shapes.yaml").read_text() == first

    assert _main(monkeypatch, s03, pipe.args_stage03(rd) + ["--quiet"]) == 0
    assert _main(monkeypatch, s04, pipe.args_stage04(rd) + ["--quiet"]) == 0
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
            assert ranks == list(range(len(e["kernels"])))


def test_exhaustive_mode_and_all_seed_rounds(round_dir, monkeypatch):
    rd, pipe, cfg_path = round_dir
    step = ro.RoundStep(0, "initial", [], None, False)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["planner"]["mode"] = "exhaustive"
    cfg_path.write_text(yaml.safe_dump(cfg))
    args = pipe.args_stage01b(rd, step) + ["--quiet"]
    assert s01b.main(args) == 0
    plan = json.loads((rd / "stage01b" / "plan.json").read_text())
    assert {e["role"] for e in plan["shapes"]} == {"exhaustive"}
    assert all(e["kernels"] is None and len(e["probe"]) == 3 for e in plan["shapes"])
    lines = read_bench_lines(rd / "stage01b" / "shapes.yaml")
    assert len(lines) == len(GEMMS)
    every = list(range(len(_kernels())))
    assert all(
        line_solution_list(ln) == every and line_has_prescreen(ln) for ln in lines
    )
    assert not (rd / "stage01b" / "seed").exists()

    cfg["planner"].update(mode="planned", seed_per_cell=8)
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = rd / "all_seeds"
    assert s01b.main(args[:-1] + ["--out-dir", str(out), "--quiet"]) == 0
    plan = json.loads((out / "plan.json").read_text())
    assert plan["mode"] == "exhaustive" and not (out / "seed").exists()

    cfg["planner"].update(mode="auto", exhaustive_max_kernels=len(_kernels()))
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = rd / "auto"
    assert s01b.main(args[:-1] + ["--out-dir", str(out), "--quiet"]) == 0
    assert json.loads((out / "plan.json").read_text())["mode"] == "exhaustive"


def test_bad_settings_and_missing_seed_commands(round_dir):
    rd, pipe, cfg_path = round_dir
    step = ro.RoundStep(0, "initial", [], None, False)
    args = pipe.args_stage01b(rd, step)
    i = args.index("--seed-commands")
    no_seed = args[:i] + args[i + 2 :] + ["--quiet"]
    assert s01b.main(no_seed) == 1
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["planner"]["ranker"] = "origami"
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert s01b.main(args + ["--quiet"]) == 1


def test_exhaustive_beyond_the_bench_list_limit_lists_every_kernel(monkeypatch):
    plan = s01b.ShapePlan((64, 64, 64, 1), "exhaustive", None, [], 3, 0, {})
    line = bf.bench_line(bf.Gemm(64, 64, 64), skip_slow_solution_ratio=0.5)
    cat = s01b.Catalog.build(_kernels(), [])
    listed = s01b.render_planned_line(line, plan, s01b.exhaustive_list(cat))
    assert line_solution_list(listed) == cat.pool_ids
    monkeypatch.setattr(s01b, "MAX_BENCH_LIST", len(cat.pool_ids) - 1)
    assert s01b.exhaustive_list(cat) is None
    every = s01b.render_planned_line(listed, plan, None)
    assert line_solution_list(every) is None
    assert "algo_method: 1" in every and "requested_solution_num: -1" in every
    assert line_has_prescreen(every)


def test_contexts_are_replanned_with_the_model(round_dir, monkeypatch):
    rd, pipe, _ = round_dir
    seen = []

    class FakeModelRanker(s01b.Ranker):
        name = "model"

        def __init__(self, data, catalog, ptype, hardware, context=None, **kw):
            self.ids = list(catalog.pool_ids)
            self.rules = {"dot2_rule": False, "non_temporal_rule": False}
            self.budget = context.cu_budget if context is not None else 0
            seen.append(
                None if context is None else (context.cu_budget, context.schedule)
            )

        def rank(self, shape):
            return self.ids[::-1] if self.budget else self.ids

    monkeypatch.setattr(s01b, "ModelRanker", FakeModelRanker)
    monkeypatch.setattr(s01b, "model_bytes", lambda *a, **kw: b"model")
    model = rd.parent / "m.pt"
    model.write_text("m")
    step = ro.RoundStep(0, "initial", [], None, False)
    args = pipe.args_stage01b(rd, step)
    i = args.index("--seed-commands")
    args = args[:i] + args[i + 2 :] + ["--model-file", str(model), "--quiet"]
    wanted = [{"name": "cu32_cot", "cu_budget": 32, "schedule": "dynamic"}]
    assert s01b.main(args + ["--contexts", json.dumps(wanted)]) == 0
    assert seen == [None, (32, "dynamic")]
    out = rd / "stage01b"
    rules = json.loads((out / "plan.json").read_text())["rules"]
    assert rules == {"dot2_rule": False, "non_temporal_rule": False}
    base = {shape_of_line(ln): ln for ln in read_bench_lines(out / "shapes.yaml")}
    ctx_dir = out / "contexts" / "cu32_cot"
    ctx = {shape_of_line(ln): ln for ln in read_bench_lines(ctx_dir / "shapes.yaml")}
    assert set(ctx) == set(base) and len(ctx) == len(GEMMS)
    assert any(line_solution_list(ctx[s]) != line_solution_list(base[s]) for s in ctx)
    plan = json.loads((ctx_dir / "plan.json").read_text())
    assert plan["context"] == {"cu_budget": 32, "schedule": "dynamic"}
    assert s01b.main(args + ["--contexts", "[{}]"]) == 1


def test_config_rules_follow_the_catalog_serving_options():
    v2 = s01b.config_rules({})
    assert v2["small_batched_rule"] and v2["dot2_rule"] and v2["non_temporal_rule"]
    assert not s01b.config_rules({"train": {"non_temporal_rule": False}})[
        "non_temporal_rule"
    ]
    v3 = s01b.config_rules({"train": {"catalog": "v3"}})
    assert not any(v3.values())
    v3_nt = s01b.config_rules(
        {"train": {"catalog": "v3", "non_temporal_rule": True, "dot2_rule": True}}
    )
    assert v3_nt == {
        "leading_free_rule": False,
        "small_batched_rule": False,
        "dot2_rule": True,
        "non_temporal_rule": True,
    }


def test_bank_auto_follows_the_runtime_pool(round_dir):
    """`planner.bank: auto` adds the Equality-row kernels of the problem
    type's other files that the device accepts, in ascending index."""
    import msgpack

    rd, pipe, cfg_path = round_dir
    n_cu = int(fh.FAKE_HW.split(",")[0])
    lib = pipe.paths.library_dir
    base = len(_kernels())

    def write(tag, cu_count, indices):
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
        stem = bf.LIBRARY_STEM.replace("_gfx1250", f"_CU{tag}_gfx1250")
        doc = {"solutions": sols, "library": {"type": "Problem", "rows": [row]}}
        (lib / f"{stem}.dat").write_bytes(msgpack.packb(doc))
        return stem

    accepted = write(n_cu, n_cu, [base + 1, base, 2])
    write(n_cu + 1, n_cu + 1, [base + 2])
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["planner"].update(mode="exhaustive", bank="auto")
    cfg["hardware"]["pci_chip_id"] = 0x75A0
    cfg_path.write_text(yaml.safe_dump(cfg))
    step = ro.RoundStep(0, "initial", [], None, False)
    assert s01b.main(pipe.args_stage01b(rd, step) + ["--quiet"]) == 0
    every = list(range(base)) + [base, base + 1]
    lines = read_bench_lines(rd / "stage01b" / "shapes.yaml")
    assert all(line_solution_list(ln) == every for ln in lines)
    plan = json.loads((rd / "stage01b" / "plan.json").read_text())
    assert plan["bank"] == "auto" and plan["bank_stems_used"] == [accepted]
