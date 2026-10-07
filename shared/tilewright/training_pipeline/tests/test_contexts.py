# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import msgpack
import pytest

import bench_fakes as bf
import fake_hipblaslt as fh
import run_orchestrator as ro
import stage03c_bench_contexts as s03c
from lib import contexts as cx
from lib import pool_libdir as pl
from lib.dat import read_tensile_logic
from test_run_orchestrator import (
    STEM,
    _pipeline,
    arg,
    calls,
    defined_options,
    env,
    make_config,
    run,
    stubs,
)

PIPELINE_DIR = Path(__file__).resolve().parent.parent

CTX_LIST = [
    {"cu_budget": 0},
    {"cu_budget": 128, "cotenant": True},
    {"cu_budget": 64, "cotenant": True, "fraction": 0.5},
]


def _cfg(**sec):
    return {"contexts": {"list": list(CTX_LIST), **sec}}


# ── config ───────────────────────────────────────────────────────────────────


def test_contexts_from_config_skips_the_exclusive_entry():
    ctxs = cx.contexts_from_config(_cfg(holdout=[{"cu_budget": 96}]))
    assert [c.label for c in ctxs] == ["cu128_cot", "cu64_cot", "cu96"]
    assert [c.holdout for c in ctxs] == [False, False, True]
    assert ctxs[1].fraction == 0.5 and ctxs[0].cotenant_cus(256) == 128
    assert ctxs[2].cotenant_cus(256) == 0
    assert cx.BenchContext(schedule="dynamic").label == "allcu_dynamic"
    assert cx.BenchContext(cu_budget=8, name="small").label == "small"
    assert cx.trained_fractions(ctxs, 256) == [0.25, 0.5, 1.0]


@pytest.mark.parametrize(
    "sec, n_cu, needle",
    [
        ({"list": [{"cu_budget": 0, "cotenant": True}]}, None, "cotenant needs"),
        ({"holdout": [{"cu_budget": 0}]}, None, "cannot be held out"),
        ({"list": [{"cu_budget": 0, "fraction": 0.5}]}, None, "fraction must be 1"),
        ({"list": [{"cu_budget": 256}]}, 256, "must be below"),
        (
            {"list": [{"cu_budget": 64}], "holdout": [{"cu_budget": 64}]},
            None,
            "also contexts.list[0]",
        ),
        (
            {"list": [{"cu_budget": 64, "name": "a"}, {"cu_budget": 32, "name": "a"}]},
            None,
            "name 'a'",
        ),
    ],
)
def test_context_errors(sec, n_cu, needle):
    errors = cx.context_errors({"contexts": sec}, n_cu)
    assert any(needle in e for e in errors), errors


def test_orchestrator_validates_contexts(tmp_path):
    base, _ = ro.load_config(make_config(tmp_path))
    good = dict(base, contexts=_cfg()["contexts"])
    good["train"] = dict(base["train"], catalog="v3")
    assert ro.validate_config(good)[0] == []

    def errors(**patch):
        cfg = json.loads(json.dumps(good))
        for dotted, value in patch.items():
            cur = cfg
            *head, last = dotted.split(".")
            for p in head:
                cur = cur.setdefault(p, {})
            cur[last] = value
        return ro.validate_config(cfg)[0]

    assert any("need train.catalog v3" in e for e in errors(**{"train.catalog": "v2"}))
    assert any("unknown key" in e for e in errors(**{"contexts.list": [{"cus": 3}]}))
    assert any(
        "fraction" in e for e in errors(**{"contexts.list": [{"fraction": 0.0}]})
    )
    sched = errors(
        **{
            "contexts.list": [{"cu_budget": 64, "schedule": "dynamic"}],
            "train.schedules": ["default"],
        }
    )
    assert any("leaves out the benched schedule(s) ['dynamic']" in e for e in sched)
    low = errors(**{"hardware.n_cu": 256, "train.min_cu_fraction": 0.5})
    assert any("does not cover the trained budget fraction 0.25" in e for e in low)
    assert errors(**{"hardware.n_cu": 256, "train.min_cu_fraction": 0.25}) == []
    assert any(
        "needs planner.enabled" in e for e in errors(**{"contexts.replan": True})
    )
    held = dict(base, contexts={"holdout": [{"cu_budget": 96}]})
    assert ro.validate_config(held)[0] == []


def test_preflight_needs_the_cotenant_kernel(tmp_path, monkeypatch):
    clients = tmp_path / "clients"
    clients.mkdir()
    launcher = clients / "hipblaslt-cotenant"
    assert any("not found" in e for e in ro._cotenant_errors(launcher))
    launcher.write_text("#!/bin/sh\n")
    launcher.chmod(0o755)
    monkeypatch.delenv("HIPCC", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    monkeypatch.setattr(ro, "rocm_bin", lambda: str(tmp_path / "nothing"))
    (err,) = ro._cotenant_errors(launcher)
    assert "hipblaslt-cotenant-kernel" in err and "no hipcc" in err
    source = clients / "hipblaslt-cotenant-kernel.hip"
    source.write_text("")
    (err,) = ro._cotenant_errors(launcher)
    assert "nor" not in err and "no hipcc" in err
    monkeypatch.setenv("HIPCC", "/opt/hipcc")
    assert ro._cotenant_errors(launcher) == []
    source.unlink()
    kernel = clients / "hipblaslt-cotenant-kernel"
    kernel.write_text("#!/bin/sh\n")
    kernel.chmod(0o755)
    assert ro._cotenant_errors(launcher) == []


# ── shape selection ──────────────────────────────────────────────────────────


def test_select_shapes_by_fraction_per_cell():
    shapes = [(m, n, 256, 1) for m in (16, 32, 4096, 8192) for n in (64, 8192)] * 2
    full = cx.select_shapes(shapes, cx.BenchContext(cu_budget=64), 1)
    assert full == list(dict.fromkeys(shapes))
    half = cx.BenchContext(cu_budget=64, fraction=0.5)
    a = cx.select_shapes(shapes, half, 1)
    assert set(a) == set(cx.select_shapes(list(reversed(shapes)), half, 1))
    assert a == cx.select_shapes(shapes, half, 1)
    assert [s for s in full if s in a] == a
    from lib.planner import shape_cell

    cells = {}
    for s in full:
        cells.setdefault(shape_cell(s), []).append(s)
    for members in cells.values():
        picked = [s for s in members if s in a]
        assert len(picked) == max(1, -(-len(members) // 2))
    other = cx.BenchContext(cu_budget=32, fraction=0.5)
    assert len(cx.select_shapes(shapes, other, 1)) == len(a)


# ── bench wrapper ────────────────────────────────────────────────────────────


def _spec(tmp_path, ctx, **kw):
    return cx.bench_spec(
        ctx, 256, "/b/hipblaslt-bench", "/b/hipblaslt-cotenant", str(tmp_path), **kw
    )


def test_bench_command_with_a_cotenant(tmp_path, monkeypatch):
    fh.unhide_devices(monkeypatch)
    monkeypatch.setenv("PATH", "/usr/bin")
    ctx = cx.BenchContext(cu_budget=64, schedule="dynamic", cotenant=True)
    spec = _spec(tmp_path, ctx, library_path="/lib", path_append=["/rocm/bin"])
    cmd, env = cx.bench_command(spec, ["--device", "3", "--yaml", "f.yaml"], "s1")
    assert cmd == [
        "/b/hipblaslt-cotenant",
        "--cus",
        "192",
        "--device",
        "3",
        "--log",
        str(tmp_path / "cotenant_192_s1.log"),
        "--",
        "/b/hipblaslt-bench",
        "--device",
        "0",
        "--yaml",
        "f.yaml",
        "--sm_count_target",
        "64",
        "--hybrid_assignment_policy",
        "DynamicWorkQueue",
    ]
    assert env == {
        "HIPBLASLT_TENSILE_LIBPATH": "/lib",
        "PATH": os.pathsep.join(["/usr/bin", "/rocm/bin"]),
    }
    assert "LD_LIBRARY_PATH" not in env


def test_bench_command_without_a_cotenant(tmp_path):
    spec = _spec(tmp_path, cx.BenchContext(cu_budget=96))
    assert spec["cotenant"] is None and spec["hybrid_assignment_policy"] is None
    cmd, env = cx.bench_command(spec, ["--device=2", "--yaml", "f.yaml"], "s")
    assert cmd == [
        "/b/hipblaslt-bench",
        "--device",
        "2",
        "--yaml",
        "f.yaml",
        "--sm_count_target",
        "96",
    ]
    assert env == {}
    auto = _spec(tmp_path, cx.BenchContext(schedule="auto"))
    cmd, _ = cx.bench_command(auto, ["--yaml", "f.yaml"], "s")
    assert cmd == ["/b/hipblaslt-bench", "--yaml", "f.yaml"] + [
        "--hybrid_assignment_policy",
        "Auto",
    ]


def test_bench_spec_rejects_impossible_contexts(tmp_path):
    with pytest.raises(ValueError, match="needs hipblaslt-cotenant"):
        cx.bench_spec(cx.BenchContext(64, cotenant=True), 256, "b", None, "l")


def test_wrapper_runs_the_bench_through_the_cotenant(tmp_path, monkeypatch):
    fh.unhide_devices(monkeypatch)
    build = fh.write_fake_build(tmp_path / "build", "gfx950", bf.LIBRARY_STEM)
    echo = tmp_path / "echo-bench"
    echo.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'lib': "
        "os.environ.get('HIPBLASLT_TENSILE_LIBPATH'), "
        "'path': os.environ['PATH']}))\n"
    )
    echo.chmod(0o755)
    ctx = cx.BenchContext(cu_budget=48, cotenant=True)
    spec = cx.bench_spec(
        ctx,
        64,
        str(echo),
        str(build / "clients" / "hipblaslt-cotenant"),
        str(tmp_path / "cot"),
        library_path="/pool",
        path_append=["/rocm/bin"],
    )
    wrapper = cx.write_wrapper(
        tmp_path / "ctx", spec, sys.executable, PIPELINE_DIR / "lib/context_bench.py"
    )
    assert json.loads((tmp_path / "ctx" / cx.SPEC_FILE).read_text()) == spec
    proc = subprocess.run(
        [str(wrapper), "--device", "1", "--yaml", "x.yaml"],
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout.splitlines()
    assert out[0] == "launching cotenant on 16 CUs"
    got = json.loads(out[-1])
    assert got["argv"] == [
        "--device",
        "0",
        "--yaml",
        "x.yaml",
        "--sm_count_target",
        "48",
    ]
    assert got["lib"] == "/pool" and got["path"].endswith(os.pathsep + "/rocm/bin")
    (log,) = (tmp_path / "cot").glob("cotenant_16_*.log")
    assert log.read_text() == "cus 16 device 1\n"


def test_cotenant_device_resolves_the_stage_visible_devices():
    assert cx.cotenant_device("1", {}) == "1"
    assert cx.cotenant_device("1", {"HIP_VISIBLE_DEVICES": "6, 3"}) == "3"
    assert cx.cotenant_device("0", {"CUDA_VISIBLE_DEVICES": "2"}) == "2"
    assert cx.cotenant_device("0", {"ROCR_VISIBLE_DEVICES": "4"}) == "0"
    for device, visible in (("1", "3"), ("0", "GPU-ab,1")):
        with pytest.raises(ValueError, match="would run elsewhere"):
            cx.cotenant_device(device, {"HIP_VISIBLE_DEVICES": visible})


def test_wrapper_keeps_the_cotenant_on_the_stage_devices(tmp_path):
    build = fh.write_fake_build(tmp_path / "build", "gfx950", bf.LIBRARY_STEM)
    echo = tmp_path / "echo-bench"
    echo.write_text(
        f"#!{sys.executable}\nimport os\n"
        "print(os.environ.get('HIP_VISIBLE_DEVICES'))\n"
    )
    echo.chmod(0o755)
    spec = cx.bench_spec(
        cx.BenchContext(cu_budget=48, cotenant=True),
        64,
        str(echo),
        str(build / "clients" / "hipblaslt-cotenant"),
        str(tmp_path / "cot"),
    )
    wrapper = cx.write_wrapper(
        tmp_path / "ctx", spec, sys.executable, PIPELINE_DIR / "lib/context_bench.py"
    )
    env = {k: v for k, v in os.environ.items() if not k.endswith("VISIBLE_DEVICES")}

    def run(device, visible):
        return subprocess.run(
            [str(wrapper), "--device", device, "--yaml", "x.yaml"],
            capture_output=True,
            text=True,
            env={**env, "HIP_VISIBLE_DEVICES": visible},
        )

    proc = run("1", "6,3")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[-1] == "3"
    (log,) = (tmp_path / "cot").glob("cotenant_16_*.log")
    assert log.read_text() == "cus 16 device 3\n"
    proc = run("1", "3")
    assert proc.returncode == 2 and "would run elsewhere" in proc.stderr


# ── pool-only library directory ──────────────────────────────────────────────


def _write_logic(path, n_rows):
    doc = {
        "solutions": [{"index": i} for i in range(n_rows)],
        "library": {"type": "Matching", "rows": [[i] for i in range(n_rows)]},
    }
    raw = msgpack.packb(doc, use_bin_type=True)
    path.write_bytes(zlib.compress(raw) if path.name.endswith(".zlib") else raw)


def test_pool_libdir_empties_the_other_libraries_of_the_type(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    pool = "TensileLibrary_BB_BB_HA_Bias_Contraction_l_Alik_Bljk_ID75a0_gfx950"
    bank = pool.replace("_ID75a0", "_CU256_ID75a0")
    other_type = "TensileLibrary_HH_HH_HA_Contraction_l_Alik_Bljk_ID75a0_gfx950"
    for name in (f"{pool}.dat", f"{bank}.dat.zlib", f"{other_type}.dat"):
        _write_logic(src / name, 3)
    (src / f"{bank}.co").write_bytes(b"code")
    assert pl.split_stem(pool) == (pool[: -len("_ID75a0_gfx950")], "_ID75a0_gfx950")
    assert pl.split_stem("TensileLibrary_X_gfx1250") == ("TensileLibrary_X", "_gfx1250")
    with pytest.raises(ValueError):
        pl.split_stem("TensileLibrary_noarch")
    dst = tmp_path / "dst"
    summary = pl.build_pool_libdir(src, [pool], dst)
    assert summary["emptied"] == [f"{bank}.dat.zlib"]
    assert (dst / f"{pool}.dat").is_symlink() and (dst / f"{bank}.co").is_symlink()
    assert (dst / f"{other_type}.dat").is_symlink()
    emptied = read_tensile_logic(dst / f"{bank}.dat.zlib")
    assert emptied["solutions"] == [] and emptied["library"]["rows"] == []
    assert len(read_tensile_logic(dst / f"{pool}.dat")["solutions"]) == 3
    assert pl.build_pool_libdir(src, [pool, bank], dst)["emptied"] == []
    assert (dst / f"{bank}.dat.zlib").is_symlink()
    with pytest.raises(FileNotFoundError):
        pl.build_pool_libdir(src, [pool.replace("BB_BB", "SS_SS")], dst)


# ── orchestrator wiring ──────────────────────────────────────────────────────


def _context_config(tmp_path, **extra):
    return make_config(
        tmp_path,
        **{
            "train.catalog": "v3",
            "contexts.list": list(CTX_LIST),
            "contexts.holdout": [{"cu_budget": 96, "cotenant": True}],
            **extra,
        },
    )


def test_context_round_wiring(tmp_path, stubs, env):
    cfg = _context_config(tmp_path)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    log = calls(env)
    rd0 = run_dir / "round_0"
    stages0 = [c["stage"] for c in log if "round_0" in (arg(c, "--out-dir") or "")]
    assert stages0[:4] == ["stage01", "stage03", "stage03c", "stage04"]
    s03c = [c for c in log if c["stage"] == "stage03c"][0]
    assert arg(s03c, "--out-dir") == str(rd0 / "stage03c")
    assert arg(s03c, "--shapes-yaml") == str(rd0 / "stage01" / "shapes.yaml")
    cmds = json.loads(arg(s03c, "--commands"))
    assert cmds["stage02"] is None
    s03 = {"argv": cmds["stage03"]}
    assert arg(s03, "--in-dir") == "{ctx_dir}"
    assert arg(s03, "--out-dir") == "{ctx_dir}/stage03"
    assert arg({"argv": cmds["stage04"]}, "--out-dir") == "{ctx_dir}/stage04"
    assert "--log-dir" not in cmds["stage04"]
    assert arg(s03c, "--path-append").endswith("bin")
    s04 = [c for c in log if c["stage"] == "stage04"][0]["argv"]
    i = s04.index("--context-log-dir")
    assert s04[i : i + 4] == [
        "--context-log-dir",
        "128",
        "default",
        str(rd0 / "stage03c" / "cu128_cot" / "stage03" / "logs"),
    ]
    assert s04.count("--context-log-dir") == 2
    s04b = [c for c in log if c["stage"] == "stage04b"][0]
    assert arg(s04b, "--context-holdout-dir") == str(
        run_dir / "round_1" / "stage03c" / "holdout" / "stage04"
    )
    single = tmp_path / "single"
    assert run(stubs, cfg, single, "--mode", "initial") == 0
    env.write_text("")
    assert run(stubs, cfg, single, "--mode", "initial", "--from-stage", "3.5") == 0
    assert [c["stage"] for c in calls(env)][:2] == ["stage03c", "stage04"]


def test_rebench_runs_before_the_context_benches(tmp_path, stubs, env):
    cfg = _context_config(tmp_path, **{"bench.rebench": {"enabled": True}})
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    rd0 = run_dir / "round_0"
    log = [c for c in calls(env) if "round_0" in (arg(c, "--out-dir") or "")]
    assert [c["stage"] for c in log][:5] == [
        "stage01",
        "stage03",
        "stage03b",
        "stage03c",
        "stage04",
    ]
    s04 = [c for c in log if c["stage"] == "stage04"][0]
    assert arg(s04, "--log-dir") == str(rd0 / "stage03b" / "logs")
    assert s04["argv"].count("--context-log-dir") == 2
    env.write_text("")
    assert run(stubs, cfg, run_dir, "--mode", "initial", "--from-stage", "3.2") == 0
    assert [c["stage"] for c in calls(env)][:3] == ["stage03b", "stage03c", "stage04"]
    env.write_text("")
    assert run(stubs, cfg, run_dir, "--mode", "initial", "--from-stage", "3.5") == 0
    assert [c["stage"] for c in calls(env)][:2] == ["stage03c", "stage04"]


def test_without_contexts_there_is_no_context_stage(tmp_path, stubs, env):
    assert run(stubs, make_config(tmp_path), tmp_path / "run", "--mode", "initial") == 0
    log = calls(env)
    assert "stage03c" not in [c["stage"] for c in log]
    s04 = [c for c in log if c["stage"] == "stage04"][0]["argv"]
    assert "--context-log-dir" not in s04


def test_pool_only_library_reaches_stage03(tmp_path):
    rd = tmp_path / "run" / "round_0"
    plain = _pipeline(tmp_path, make_config(tmp_path)).args_stage03(rd)
    assert "--pool-only-library" not in plain
    cfg = make_config(tmp_path, **{"bench.pool_only_library": True})
    args = _pipeline(tmp_path, cfg).args_stage03(rd)
    assert arg({"argv": args}, "--pool-only-library") == STEM
    assert arg({"argv": args}, "--library-dir").endswith("gfx950")


def test_replanned_contexts_reach_stage01b_and_stage03c(tmp_path):
    cfg = _context_config(
        tmp_path,
        **{
            "planner.enabled": True,
            "planner.model_file": str(tmp_path / "m.pt"),
            "contexts.replan": True,
        },
    )
    p = _pipeline(tmp_path, cfg)
    rd = tmp_path / "run" / "round_0"
    step = ro.RoundStep(0, "initial", [], None, False)
    wanted = json.loads(arg({"argv": p.args_stage01b(rd, step)}, "--contexts"))
    assert [c["name"] for c in wanted] == ["cu128_cot", "cu64_cot", "cu96_cot"]
    args = p.args_stage03c(rd)
    assert arg({"argv": args}, "--context-shapes-dir") == str(
        rd / "stage01b" / "contexts"
    )
    replan = rd / "stage01b" / "contexts" / "cu64_cot" / "plan.json"
    replan.parent.mkdir(parents=True)
    replan.write_text("{}")
    s04 = p.args_stage04(rd)
    i = s04.index("--context-plan")
    assert s04[i : i + 4] == ["--context-plan", "64", "default", str(replan)]


def test_context_stage_cli_contract(tmp_path, pipeline_dir):
    cfg = _context_config(
        tmp_path,
        **{
            "bench.pool_only_library": True,
            "planner.enabled": True,
            "planner.model_file": str(tmp_path / "m.pt"),
            "contexts.replan": True,
        },
    )
    p = _pipeline(tmp_path, cfg)
    rd = tmp_path / "run" / "round_1"
    step = ro.RoundStep(0, "initial", [], None, False)
    active = ro.RoundStep(1, "active", [rd.parent / "round_0"], None, True)
    s03c = p.args_stage03c(rd)
    cmds = json.loads(arg({"argv": s03c}, "--commands"))
    built = {
        "stage01b": p.args_stage01b(rd, step),
        "stage03": p.args_stage03(rd) + cmds["stage03"][1:],
        "stage03c": s03c,
        "stage04": p.args_stage04(rd) + cmds["stage04"][1:],
        "stage04b": p.args_stage04b(rd, active, rd.parent / "round_0" / "stage05"),
    }
    for key, args in built.items():
        script = pipeline_dir / "stages" / ro.STAGE_DEFS[key].script
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            env=dict(os.environ, COLUMNS="80"),
        )
        assert proc.returncode == 0, proc.stderr
        accepted = defined_options(proc.stdout)
        for flag in sorted({a for a in args if a.startswith("--")}):
            assert flag in accepted, f"{key} does not accept {flag}"


# ── stage03c on the fake bench ───────────────────────────────────────────────


def test_stage03c_benches_each_context(tmp_path, monkeypatch):
    fh.unhide_devices(monkeypatch)
    """stage03c with the fake bench and cotenant, then stage04 with the
    context log directories: every trained context's rows carry its
    columns, a narrower budget changes winners, the held-out context is
    enriched on its own."""
    arch = "gfx950"
    build = fh.write_fake_build(tmp_path / "build", arch, bf.LIBRARY_STEM)
    libdir = build / "Tensile" / "library" / arch
    monkeypatch.setenv("FAKE_LIBRARY_DIR", str(libdir))
    monkeypatch.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    shapes = [(4096, 4096, 1024, 1), (2048, 8192, 512, 1), (8192, 1024, 2048, 1)]
    shapes += [(64, 128, 256, 1), (1024, 1024, 4096, 1)]
    yaml_path = tmp_path / "shapes.yaml"
    yaml_path.write_text(
        "".join(
            f"- {{M: {m}, N: {n}, K: {k}, batch_count: {b}, transA: T, transB: N, "
            f"a_type: f8_r, b_type: f8_r, c_type: bf16_r, d_type: bf16_r, "
            f"compute_type: c_f32_r, requested_solution_num: -1, "
            f"skip_slow_solution_ratio: 0, iters: 1, cold_iters: 0}}\n"
            for m, n, k, b in shapes
        )
    )
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        json.dumps(
            {
                "hardware": {"n_cu": 64, "lds_bytes": 65536, "l2_bytes": 4194304},
                "contexts": {
                    "list": [
                        {"cu_budget": 0},
                        {"cu_budget": 16, "cotenant": True},
                        {"cu_budget": 32, "fraction": 0.5},
                    ],
                    "holdout": [{"cu_budget": 8, "cotenant": True}],
                    "seed": 3,
                },
            }
        )
    )
    s03 = [
        str(PIPELINE_DIR / "stages" / "stage03_load_balance_offline_tuning.py"),
        "--in-dir",
        "{ctx_dir}",
        "--out-dir",
        "{ctx_dir}/stage03",
        "--bench-binary",
        str(build / "clients" / "hipblaslt-bench"),
        "--devices",
        "0",
        "--blocks-per-gpu",
        "1",
        "--stall-timeout-s",
        "120",
        "--startup-grace-s",
        "120",
    ]
    s04 = [
        str(PIPELINE_DIR / "stages" / "stage04_convert_to_enriched_dataset.py"),
        "--out-dir",
        "{ctx_dir}/stage04",
        "--dat-dir",
        str(libdir),
        "--library-stem",
        bf.LIBRARY_STEM,
        "--quiet",
    ]
    out = tmp_path / "stage03c"
    rc = s03c.main(
        [
            "--shapes-yaml",
            str(yaml_path),
            "--out-dir",
            str(out),
            "--config-yaml",
            str(cfg),
            "--bench-binary",
            str(build / "clients" / "hipblaslt-bench"),
            "--commands",
            json.dumps({"stage02": None, "stage03": s03, "stage04": s04}),
            "--pool-only-library",
            bf.LIBRARY_STEM,
            "--library-dir",
            str(libdir),
            "--path-append",
            str(tmp_path / "rocm" / "bin"),
            "--quiet",
        ]
    )
    summary = json.loads((out / "contexts.json").read_text())
    assert rc == 0, summary
    by_label = {r["name"]: r for r in summary["contexts"]}
    assert set(by_label) == {"cu16_cot", "cu32", "cu8_cot"}
    assert by_label["cu16_cot"]["n_shapes"] == len(shapes)
    assert by_label["cu32"]["n_shapes"] < len(shapes)
    assert by_label["cu16_cot"]["cotenant_cus"] == 48
    assert list((out / "cu16_cot" / "cotenant_logs").glob("cotenant_48_*.log"))
    assert not (out / "cu32" / "cotenant_logs").exists() or not list(
        (out / "cu32" / "cotenant_logs").glob("*.log")
    )
    spec = json.loads((out / "cu16_cot" / cx.SPEC_FILE).read_text())
    assert spec["library_path"] == str(out / "libdir_pool")

    rows = _enrich(
        tmp_path / "stage04",
        libdir,
        [
            ("--log-dir", None, None, None),
            ("--context-log-dir", "16", "default", by_label["cu16_cot"]["log_dir"]),
            ("--context-log-dir", "32", "default", by_label["cu32"]["log_dir"]),
        ],
        yaml_path,
        build,
    )
    by_ctx = {}
    for r in rows:
        key = (r.get("cu_budget") or "0", r.get("schedule") or "default")
        by_ctx.setdefault(key, set()).add((r["m"], r["n"], r["k"]))
    assert set(by_ctx) == {("0", "default"), ("16", "default"), ("32", "default")}
    assert len(by_ctx[("0", "default")]) == len(by_ctx[("16", "default")]) == 5
    assert len(by_ctx[("32", "default")]) == by_label["cu32"]["n_shapes"]
    summary04 = json.loads((tmp_path / "stage04" / "enrich_summary.json").read_text())
    assert summary04["gemms_by_context"]["16/default"] == 5
    win = {}
    for r in rows:
        key = (r.get("cu_budget") or "0", r["m"], r["n"], r["k"])
        if key not in win or float(r["us"]) < win[key][0]:
            win[key] = (float(r["us"]), r["sol_idx_global"])
    assert any(
        win[("0", m, n, k)][1] != win[("16", m, n, k)][1]
        for (b, m, n, k) in win
        if b == "16"
    )
    held = _csv_rows(out / "holdout" / "stage04")
    assert {r["cu_budget"] for r in held} == {"8"}


def _csv_rows(directory):
    import csv

    rows = []
    for path in sorted(Path(directory).glob("chunk_*.csv")):
        with path.open() as f:
            rows += list(csv.DictReader(f))
    return rows


def _enrich(out, libdir, sources, yaml_path, build):
    """Run the exclusive bench of `yaml_path` through stage03 and stage04
    with the given context log directories."""
    s03_out = out.parent / "exclusive"
    s03_in = out.parent / "exclusive_in"
    s03_in.mkdir()
    (s03_in / "shapes.yaml").write_text(yaml_path.read_text())
    subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "stages" / "stage03_load_balance_offline_tuning.py"),
            "--in-dir",
            str(s03_in),
            "--out-dir",
            str(s03_out),
            "--bench-binary",
            str(build / "clients" / "hipblaslt-bench"),
            "--devices",
            "0",
            "--blocks-per-gpu",
            "1",
            "--stall-timeout-s",
            "120",
            "--startup-grace-s",
            "120",
        ],
        check=True,
        capture_output=True,
    )
    args = [
        sys.executable,
        str(PIPELINE_DIR / "stages" / "stage04_convert_to_enriched_dataset.py"),
        "--out-dir",
        str(out),
        "--dat-dir",
        str(libdir),
        "--library-stem",
        bf.LIBRARY_STEM,
        "--quiet",
    ]
    for flag, budget, sched, d in sources:
        if flag == "--log-dir":
            args += [flag, str(s03_out / "logs")]
        else:
            args += [flag, budget, sched, d]
    proc = subprocess.run(args, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return _csv_rows(out)
