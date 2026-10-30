# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The audit set (lib/audit.py) through its entry points: `run_orchestrator
--mode audit` on the fake bench (layout, exhaustive rows, contexts, resume,
kernel increments), training rounds that point at it (shape exclusion,
stage04b / stage06 / stage05 audit.json, nothing of it reaching stage05),
scripts/audit_eval.py (null until complete) and the config keys."""

import json
import math
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import bench_fakes as bf
import fake_hipblaslt as fh
import run_orchestrator as ro
from lib import audit as al
from lib import audit_v2 as v2
from lib import evaluate as ev
from lib import rebench as rb
from lib import run_layout
from lib.bench_yaml import (
    ADAPTIVE_KEYS,
    adaptive_settings,
    line_has_prescreen,
    line_is_adaptive,
    line_solution_list,
    parse_shapes_yaml,
    read_bench_lines,
    set_line_knob,
    shape_of_line,
)
from lib.grid import cell_key

PIPELINE = Path(ro.__file__).resolve().parent
ARCH = "gfx1250-strict"
AUDIT = {
    "shapes_per_cell": 4,
    "shapes_per_context": 2,
    "seed": 11,
    "bootstrap": 50,
    "part_shapes": 8,
}
N_CELLS = 6
CONTEXTS = ("cu16_cot", "cu32_cot")


def base_config(bench_root: Path, audit_dir: Path) -> dict:
    n_cu, lds, l2 = (int(v) for v in fh.FAKE_HW.split(","))
    return {
        "config_id": "audit",
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
        "seed": {
            "target_per_cell": 8,
            "shapes_per_cell": 8,
            "seed": 3,
            "n_strata": 1,
            "exclude": [{"max_m": 512}, {"max_n": 128}, {"min_batch": 2}],
        },
        "bench": {
            "blocks_per_gpu": 1,
            "startup_grace_s": 120,
            "stall_timeout_s": 120,
            "device_memory_gib": 64,
            "skip_slow_solution_ratio": 0.5,
            "pool_only_library": True,
            "adaptive": {"measure_time": 1.0},
            "canary": {"enabled": True, "backoff_s": 0},
            # The slow plans of the v2 world count bench processes; a health
            # canary's would shift them.
            "health": {"canary": {"enabled": False}},
        },
        "contexts": {
            "list": [{"cu_budget": 0}, {"cu_budget": 16, "cotenant": True}],
            "holdout": [{"cu_budget": 32, "cotenant": True, "fraction": 0.5}],
        },
        "train": {
            "epochs": 2,
            "workers": 1,
            "min_cell_gemms": 6,
            "smart_k": 4,
            "catalog": "v3",
        },
        "validate": {
            "min_cell_gemms": 3,
            "max_split_depth": 0,
            "sel_eff_threshold": 0.999,
        },
        "full": {"total_rounds": 2},
        "deploy": {"weight_dtype": "int8", "targets": [ARCH]},
        "audit": {"dir": str(audit_dir), "v2": "off", **AUDIT},
    }


def extra_kernels(n: int, start: int):
    return [
        bf.solution(start + i, start + i, mt=(64, 128, 64), mi=(16, 16, 128, 1))
        for i in range(n)
    ]


def _probe_era(part: Path) -> dict:
    """The outputs the stage02 probe left in an audit part: calibrated lines
    of its shapes with another kernel list, its marker and log. Returns
    {path: bytes} of what it wrote."""
    out = part / "stage02"
    out.mkdir(parents=True)
    lines = read_bench_lines(part / "shapes.yaml")
    (out / "shapes_probed.yaml").write_text(
        "".join(
            set_line_knob(set_line_knob(ln, "solution_index_list", "[0]"), "iters", 40)
            for ln in lines
        )
    )
    ro.mark_stage_complete(out, ["stage02_probe.py"])
    (part / "stage02.log").write_text("probe\n")
    return {
        p: p.read_bytes()
        for p in [*sorted(out.rglob("*")), part / "stage02.log"]
        if p.is_file()
    }


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """An audit set built (with one interruption and resume) beside a
    source run, then a two-round training run pointing at it."""
    pytest.importorskip("tilewright")
    root = tmp_path_factory.mktemp("audit")
    mp = pytest.MonkeyPatch()
    try:
        yield _build_world(root, mp)
    finally:
        mp.undo()


def _build_world(root: Path, mp: pytest.MonkeyPatch) -> dict:
    fh.unhide_devices(mp)
    kernels = fh.kernels_grid()
    bench_root = root / "bench"
    build = bench_root / "projects" / "hipblaslt" / "build" / "release"
    fh.write_fake_build(build, ARCH, bf.LIBRARY_STEM, kernels)
    libdir = build / "Tensile" / "library" / ARCH
    mp.setenv("FAKE_LIBRARY_DIR", str(libdir))
    mp.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    benched = root / "benched_yamls"
    benched.mkdir()
    mp.setenv("FAKE_BENCH_YAML_DIR", str(benched))
    audit_dir = root / "audit_set"
    cfg = base_config(bench_root, audit_dir)
    errors, warnings = ro.validate_config(cfg)
    assert errors == [] and warnings == []
    cfg_path = root / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # A source run whose round drew exactly what the audit seed draws first.
    source = root / "source_run"
    src_stage01 = source / "round_0" / "stage01"
    rc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE / "stages" / "stage01_generate_OOB_shapes.py"),
            "--mode",
            "initial",
            "--out-dir",
            str(src_stage01),
            "--config-yaml",
            str(cfg_path),
            "--seed",
            str(AUDIT["seed"]),
            "--round-index",
            "0",
            "--target-per-cell",
            "2",
            "--quiet",
        ]
    ).returncode
    assert rc == 0
    source_shapes = set(parse_shapes_yaml(src_stage01 / "shapes.yaml"))
    assert source_shapes

    audit_cmd = [
        "--config",
        str(cfg_path),
        "--mode",
        "audit",
        "--run-dir",
        str(audit_dir),
        "--source-run-dir",
        str(source),
    ]
    calls = []
    real = ro.Pipeline.run_stage

    def interrupting(self, owner_dir, key, args, round_idx, out_dir=None):
        calls.append((Path(owner_dir), key))
        if Path(owner_dir).name == "part_0001" and key == "stage03b":
            raise ro.Interrupted(signal.SIGTERM)
        return real(self, owner_dir, key, args, round_idx, out_dir)

    mp.setattr(ro.Pipeline, "run_stage", interrupting)
    first_rc = ro.main(audit_cmd)
    mp.setattr(ro.Pipeline, "run_stage", real)
    first_status = al.status(audit_dir)
    part0 = audit_dir / "bench" / "exclusive" / "part_0000"
    part0_marker = (part0 / ro.STAGE_MARKER).read_text()
    probed_part = audit_dir / "bench" / "exclusive" / "part_0001"
    probe_era = _probe_era(probed_part)
    resumed = []
    resumed_args = {}
    leased = {}

    def recording(self, owner_dir, key, args, round_idx, out_dir=None):
        resumed.append((Path(owner_dir).parent.name, Path(owner_dir).name, key))
        resumed_args[(Path(owner_dir), key)] = list(args)
        leased[(Path(owner_dir), key)] = self.stage_devices(key, args)
        return real(self, owner_dir, key, args, round_idx, out_dir)

    mp.setattr(ro.Pipeline, "run_stage", recording)
    second_rc = ro.main(audit_cmd)
    mp.setattr(ro.Pipeline, "run_stage", real)

    run_dir = root / "run"
    train_rc = ro.main(
        ["--config", str(cfg_path), "--mode", "full", "--run-dir", str(run_dir)]
    )
    return {
        "root": root,
        "cfg": cfg,
        "cfg_path": cfg_path,
        "audit": audit_dir,
        "source_shapes": source_shapes,
        "libdir": libdir,
        "kernels": kernels,
        "first_rc": first_rc,
        "first_status": first_status,
        "part0_marker": part0_marker,
        "second_rc": second_rc,
        "resumed": resumed,
        "resumed_args": resumed_args,
        "leased": leased,
        "probed_part": probed_part,
        "probe_era": probe_era,
        "benched": benched,
        "run": run_dir,
        "train_rc": train_rc,
        "audit_cmd": audit_cmd,
        "mp": mp,
    }


# ── config ───────────────────────────────────────────────────────────────────


def test_audit_keys_default_and_validate(tmp_path):
    cfg = base_config(tmp_path, tmp_path / "a")
    del cfg["audit"]
    assert ro.validate_config(cfg)[0] == []
    for key, default in al.DEFAULTS.items():
        assert ro.cfg_get(cfg, f"audit.{key}") == default
    cfg["audit"] = {"seed": 3}
    assert any("audit.seed equals seed.seed" in e for e in ro.validate_config(cfg)[0])
    cfg["audit"] = {"shapes_per_cell": 2, "shapes_per_context": 3}
    assert any("audit.shapes_per_context" in e for e in ro.validate_config(cfg)[0])
    cfg["audit"] = {"path": "x"}
    assert "audit.path: unknown key" in ro.validate_config(cfg)[0]


def test_audit_mode_options_are_checked(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("{}")
    for argv in (
        ["--mode", "audit"],
        ["--mode", "audit", "--run-dir", "x", "--from-stage", "3"],
        ["--mode", "full", "--source-run-dir", "x"],
        ["--mode", "full", "--devices", "0"],
    ):
        with pytest.raises(SystemExit):
            ro.parse_args(["--config", str(cfg), *argv])
    args = ro.parse_args(
        ["--config", str(cfg), "--mode", "audit", "--run-dir", "a", "--devices", "2,3"]
    )
    assert args.devices == [2, 3] and args.no_report


def test_bench_config_benches_every_context_exhaustively(tmp_path):
    cfg = base_config(tmp_path, tmp_path / "a")
    out = al.bench_config(cfg, [5])
    assert out["planner"]["enabled"] and out["planner"]["mode"] == "exhaustive"
    assert out["planner"]["model_file"] is None
    assert out["bench"]["rebench"]["enabled"] is True
    assert out["contexts"]["holdout"] == [] and out["contexts"]["replan"] is False
    assert [(c["name"], c["fraction"]) for c in out["contexts"]["list"]] == [
        ("cu16_cot", 0.5),
        ("cu32_cot", 0.5),
    ]
    assert out["runtime"]["devices"] == [5]
    assert out["deploy"]["apply_patch"] is False
    assert cfg["contexts"]["holdout"]


# ── stage01 exclusion ────────────────────────────────────────────────────────


def test_stage01_never_draws_excluded_shapes(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(base_config(tmp_path, tmp_path / "a")))
    script = PIPELINE / "stages" / "stage01_generate_OOB_shapes.py"

    def stage01(out, *extra):
        cmd = [sys.executable, str(script), "--config-yaml", str(cfg_path)]
        cmd += ["--out-dir", str(out), "--quiet", "--mode", "initial"]
        cmd += ["--target-per-cell", "3", *extra]
        assert subprocess.run(cmd).returncode == 0
        return set(parse_shapes_yaml(out / "shapes.yaml"))

    first = stage01(tmp_path / "a1")
    assert stage01(tmp_path / "a2") == first
    excluded = stage01(
        tmp_path / "a3", "--exclude-shapes-yaml", str(tmp_path / "a1" / "shapes.yaml")
    )
    assert len(excluded) == len(first) and not first & excluded
    cat = json.loads((tmp_path / "a3" / "categorization.json").read_text())
    assert cat["n_excluded_shapes"] == len(first)
    assert cat["n_excluded_dropped"] >= len(first)


def test_the_top_up_never_reuses_excluded_shapes(monkeypatch):
    import stage01_generate_OOB_shapes as s1

    pool = [(600 + i, 600, 600, 1) for i in range(6)]
    monkeypatch.setattr(s1, "synthesize_cell_shapes", lambda *a, **k: list(pool))
    history = set(pool[:4])
    reused = s1.sample_shapes({"C": 2}, 1, history, None, min_per_cell=6)
    assert set(reused.reused) == history
    out = s1.sample_shapes(
        {"C": 2}, 1, history, None, min_per_cell=6, excluded=set(pool[:3])
    )
    assert set(out.shapes) == set(pool[3:]) and set(out.reused) == {pool[3]}
    assert out.n_excluded == 3


# ── statistics ───────────────────────────────────────────────────────────────


def _rows(cell, sel, ceiling, n=8, context=al.EXCLUSIVE, served=None):
    return [
        al.Row(
            shape=(i, 1, 1, 1),
            cell=cell,
            context=context,
            sel_eff=sel * (1 + 0.001 * (i % 2)),
            ceiling_rules=ceiling,
            ceiling_table_rules=ceiling,
            ceiling_table=1.0,
            ceiling_served=served,
        )
        for i in range(n)
    ]


def test_cell_statistics_and_classes():
    rows = (
        _rows("A", 0.80, 0.85)
        + _rows("B", 0.90, 1.0)
        + _rows("C", 0.99, 1.0, n=2)
        + [al.Row((9, 9, 9, 1), "C", al.EXCLUSIVE, None, 1.0, 1.0, 1.0, miss="m")]
        + _rows("D", 0.90, 1.0, served=0.92)
        + _rows("E", 0.90, 1.0, served=0.97)
    )
    out = al.summarize_rows(rows, 0.95, bootstrap=200)
    cells = out["per_cell"]
    assert {c: v["class"] for c, v in cells.items()} == {
        "A": "pool_limited",
        "B": "model_limited",
        "C": "passing",
        "D": "identity_limited",
        "E": "model_limited",
    }
    assert all(cells[c]["confident"] for c in "ABDE")
    assert cells["D"]["ceiling_served"] == pytest.approx(0.92)
    assert cells["B"]["ceiling_served"] is None
    assert out["ceiling_served_cell_mean"] == pytest.approx(math.sqrt(0.92 * 0.97))
    out = al.summarize_rows(rows[: 8 + 8 + 3], 0.95, bootstrap=200)
    cells = out["per_cell"]
    assert cells["C"]["n"] == 2 and cells["C"]["n_not_evaluated"] == 1
    assert cells["C"]["n_miss"] == 1
    geo = {c: v["sel_eff"] for c, v in cells.items()}
    assert out["cell_mean"] == pytest.approx(
        math.exp(sum(math.log(v) for v in geo.values()) / 3)
    )
    assert out["gemm_geomean"] == pytest.approx(
        math.exp(
            (8 * math.log(geo["A"]) + 8 * math.log(geo["B"]) + 2 * math.log(geo["C"]))
            / 18
        )
    )
    lo, hi = out["cell_mean_ci"]
    assert lo <= out["cell_mean"] <= hi
    assert out["min_cell"]["cell"] == "A"
    assert out["n_cells_below_target"] == 2
    assert out["classes"] == {
        "pool_limited": 1,
        "identity_limited": 0,
        "model_limited": 1,
        "passing": 1,
    }
    assert out["ceiling_served_cell_mean"] is None
    assert al.summarize_rows([], 0.95, 10) is None
    assert al.classify(None, None, 1.0, None, 0.95) == (None, False)
    assert al.classify(0.97, None, 0.90, None, 0.95)[0] == "passing"
    assert al.classify(0.9, None, 0.97, None, 0.95, 0.93, [0.9, 0.96]) == (
        "identity_limited",
        False,
    )
    # The served ceiling's interval, not the rules', settles a model-limited cell.
    got = al.classify(0.9, [0.88, 0.92], 0.97, [0.96, 0.98], 0.95, 0.96, [0.94, 0.97])
    assert got == ("model_limited", False)


def _kernel(sid, mt_n, **kw):
    k = dict(
        sol_idx_global=sid, mt_m=64, mt_n=mt_n, mt_k=64, mi_m=16, mi_n=16, mi_k=32,
        occupancy=1, cache_hints_a=0, cache_hints_b=0, grvw_a=8, grvw_b=8, gwvw_d=4,
    )  # fmt: skip
    k.update(kw)
    return k


def test_the_served_ceiling_costs_each_key_by_the_kernel_the_runtime_runs():
    from lib import features as fs

    pool = [_kernel(1, 64), _kernel(2, 64, grvw_a=4), _kernel(3, 128), _kernel(4, 128)]
    v3 = {"version": 3, "arch": "gfx950", "attribute_spec": fs.DEFAULT_ATTRIBUTE_SPEC}
    key_of, members = al.candidate_keys(pool, dict(v3, serving={}))
    assert key_of[1] == key_of[2] != key_of[3] == key_of[4]
    assert members[key_of[1]] == [1, 2] and members[key_of[3]] == [3, 4]
    v2_keys, _ = al.candidate_keys(pool, {"version": 2})
    assert len(set(v2_keys.values())) == 3
    cands = [
        {"sol_idx_global": 2, "us": 10.0, "is_skip": 0},
        {"sol_idx_global": 3, "us": 12.0, "is_skip": 0},
        {"sol_idx_global": 4, "us": 8.0, "is_skip": 0},
    ]
    # Kernel 1 has no row (its predicates fail), so kernel 2 serves its key;
    # kernel 3 serves the other key although kernel 4 is faster.
    got = al._served_ceiling(8.0, cands, {2, 3, 4}, key_of, members)
    assert got == pytest.approx(0.8)
    assert al._served_ceiling(8.0, cands, {3, 4}, key_of, members) == pytest.approx(
        8.0 / 12.0
    )
    assert al._served_ceiling(8.0, cands, set(), key_of, members) is None
    launch = dict(v3, serving={"launch_keys": True})
    pool[1]["attributes"] = {"tile_processing_strategy": 2, "work_assignment": 1}
    lk, _ = al.candidate_keys(pool, launch)
    assert lk[1] != lk[2]


def test_a_model_with_the_alignment_veto_has_ceilings_without_the_kernels_it_rejects():
    """The audit's rule and served ceilings honour the model file's alignment
    veto, a serving rule: a kernel it rejects is no ceiling, unless it
    rejects every kernel the other rules pass."""
    from types import SimpleNamespace

    from lib import features as fs

    pool = [_kernel(1, 64), _kernel(2, 64, cache_hints_a=4), _kernel(3, 128)]
    v3 = {"version": 3, "arch": "gfx950", "attribute_spec": fs.DEFAULT_ATTRIBUTE_SPEC}

    def scorer(veto):
        sc = object.__new__(al._Scorer)
        sc.header = dict(v3, serving={"alignment_veto": veto} if veto else {})
        sc.rules = {"non_temporal_rule": False}
        sc.veto = veto
        sc.nt, sc.families = (True, False), None
        sc.table_ids = {1, 2, 3}
        sc.key_of, sc.members = al.candidate_keys(pool, sc.header)
        sc.hardware = SimpleNamespace(lds_bytes=65536)
        sc.labels = {}
        return sc

    def gemm(k, sids):
        us = {1: 12.0, 2: 10.0, 3: 15.0}
        return {
            "m": 4096, "n": 4096, "k": k, "batch_count": 1, "transA": "T",
            "transB": "T", "a_type": "bf16_r", "b_type": "bf16_r",
            "c_type": "bf16_r", "d_type": "bf16_r", "compute_type": "c_f32_r",
            "winner_us": 10.0,
            "candidates": [
                dict(next(k for k in pool if k["sol_idx_global"] == s), us=us[s],
                     is_skip=0)
                for s in sids
            ],
        }  # fmt: skip

    veto = {"k_min": 512, "align_bits": 1024}
    r = SimpleNamespace(pick_sol_idx=1, miss=None)
    # K 1000 in TT bf16: the veto rejects kernel 2, the winner
    plain = scorer(None).row(gemm(1000, [1, 2, 3]), r, 0.8)
    vetoed = scorer(veto).row(gemm(1000, [1, 2, 3]), r, 0.8)
    assert plain.ceiling_rules == 1.0 and plain.ceiling_served == 1.0
    assert vetoed.ceiling_rules == pytest.approx(10.0 / 12.0)
    assert vetoed.ceiling_served == pytest.approx(10.0 / 12.0)
    # at an aligned K, or when it rejects every kernel, the veto keeps them
    assert scorer(veto).row(gemm(1024, [1, 2, 3]), r, 0.8).ceiling_rules == 1.0
    assert scorer(veto).row(gemm(1000, [2]), r, 0.8).ceiling_rules == 1.0


def test_model_bytes_without_the_alignment_veto(tmp_path):
    """model_bytes(..., alignment_veto=False) gives the bytes a round's
    decisions score: an MLREC_v3 file without its veto, a models.pt packed
    without the config's, anything else as it is."""
    from lib import feasibility as fz

    from test_v3 import v3_bundle

    cfg = {"deploy": {"alignment_veto": {"k_min": 512, "align_bytes": 128}}}
    plain = mlrec_write(v3_bundle(), None)
    flagged = mlrec_write(v3_bundle(), fz.deployed_alignment_veto(cfg))
    f = tmp_path / "m.tilewright.bin"
    f.write_bytes(flagged)
    assert al.model_bytes(f, "gfx950", "bf16", cfg) == flagged
    assert al.model_bytes(f, "gfx950", "bf16", cfg, alignment_veto=False) == plain
    other = tmp_path / "other.bin"
    other.write_bytes(b"not a model")
    assert al.model_bytes(other, "gfx950", "bf16", alignment_veto=False) == (
        b"not a model"
    )
    assert al.without_alignment_veto(plain) == plain


def test_audit_eval_scores_without_the_veto_when_asked(tmp_path, monkeypatch):
    """scripts/audit_eval.py --without-alignment-veto (the rounds' audit
    evaluations) hands the audit evaluation a staged model file without its
    veto, and a models.pt packed without the config's; without the option
    both carry the config's veto."""
    import types

    import audit_eval
    from lib import feasibility as fz
    from lib import mlrec

    from test_v3 import v3_bundle

    cfg = {
        "arch": "gfx950",
        "hipblaslt": {"library_stem": "stem"},
        "runtime": {"devices": [0]},
        "deploy": {"alignment_veto": {"k_min": 512, "align_bytes": 128}},
    }
    staged = tmp_path / "round_1" / "stage06" / "m.tilewright.bin"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(mlrec_write(v3_bundle(), fz.deployed_alignment_veto(cfg)))
    (staged.parent / "deploy_record.json").write_text(
        json.dumps({"staged_model": str(staged)})
    )
    monkeypatch.setattr(ro, "load_config", lambda path: (cfg, ""))
    monkeypatch.setattr(ro, "resolve_contexts", lambda c, d: (c, None))
    monkeypatch.setattr(
        ro, "derive_paths", lambda c: types.SimpleNamespace(library_dir=tmp_path)
    )
    monkeypatch.setattr(audit_eval.hwlib, "device_hardware", lambda c, d: None)

    def evaluate(data, spec, cfg, **kw):
        veto = mlrec.read_header(data)["serving"].get("alignment_veto")
        return {"status": "ok", "veto": veto}

    monkeypatch.setattr(al, "evaluate", evaluate)
    argv = ["--audit-dir", str(tmp_path / "audit"), "--config", "c.yaml"]
    argv += ["--round-dir", str(tmp_path / "round_1"), "--weight-dtype", "bf16"]
    argv.append("--quiet")
    for flag, want in (
        ([], {"k_min": 512, "align_bits": 1024}),
        (["--without-alignment-veto"], None),
    ):
        out = tmp_path / f"out{len(flag)}.json"
        assert audit_eval.main([*argv, "--out", str(out), *flag]) == 0
        assert json.loads(out.read_text())["veto"] == want


def mlrec_write(bundle, veto):
    from lib import hardware as hwlib
    from lib import mlrec

    serving = dict(bundle.get("serving") or {})
    if veto:
        serving["alignment_veto"] = veto
    return mlrec.write_model(
        dict(bundle, serving=serving),
        None,
        "gfx950",
        hwlib.arch_constants("gfx950"),
        "bf16",
    )


def test_ceilings_cost_kernels_like_picks_and_cap_at_one():
    timed = {"us": 10.0, "is_skip": 0}
    warm = {"us": 8.0, "is_skip": 1}
    assert al._ceiling(9.0, [timed]) == pytest.approx(0.9)
    assert al._ceiling(9.0, [timed, warm]) == 1.0
    assert al._ceiling(9.0, [timed, warm]) >= al._ceiling(9.0, [timed])
    assert al._ceiling(9.0, []) is None
    assert al._capped(1.08) == 1.0 and al._capped(0.9) == 0.9
    assert al._capped(None) is None


def test_status_is_null_until_complete_and_covering(tmp_path):
    assert al.status(None)[0] == "missing"
    assert al.status(tmp_path)[0] == "missing"
    m = {"schema": al.SCHEMA_VERSION, "generations": [{"kernels": {"1": "a"}}]}
    (tmp_path / al.MANIFEST).write_text(json.dumps(m))
    (tmp_path / al.PROGRESS).write_text(
        json.dumps({"shapes_done": 3, "shapes_total": 9})
    )
    st, why = al.status(tmp_path)
    assert st == "incomplete" and "3/9" in why
    (tmp_path / al.COMPLETE_MARKER).write_text("{}")
    assert al.status(tmp_path, {1: "a"}) == ("ok", None)
    assert al.status(tmp_path, {1: "b"})[0] == "incomplete"
    assert al.status(tmp_path, {1: "a", 2: "c"})[0] == "incomplete"
    rec = al.null_record("incomplete", "why")
    assert rec["headline"] is None and rec["groups"] is None


# ── the audit set on the fake bench ──────────────────────────────────────────


def test_audit_mode_builds_the_set(world):
    a = world["audit"]
    assert world["first_rc"] == 128 + signal.SIGTERM
    assert world["first_status"][0] == "incomplete"
    assert world["second_rc"] == 0
    assert al.status(a) == ("ok", None)
    m = al.manifest(a)
    assert m["shapes"] == {
        "exclusive": AUDIT["shapes_per_cell"] * N_CELLS,
        **{c: AUDIT["shapes_per_context"] * N_CELLS for c in CONTEXTS},
    }
    assert len(m["generations"]) == 1 and m["status"] == "complete"
    assert set(m["generations"][0]["kernels"]) == {
        str(k["index"]) for k in world["kernels"]
    }
    progress = json.loads((a / al.PROGRESS).read_text())
    assert (
        progress["shapes_done"] == progress["shapes_total"] == sum(m["shapes"].values())
    )
    shapes = set(parse_shapes_yaml(al.shapes_yaml(a)))
    assert len(shapes) == m["shapes"]["exclusive"]
    assert not shapes & world["source_shapes"]
    cat = json.loads((a / "stage01" / "categorization.json").read_text())
    assert cat["n_history_dropped"] > 0 and cat["seed"] == AUDIT["seed"]
    assert run_layout.read(a) == run_layout.CURRENT
    for line in read_bench_lines(a / "stage02" / "shapes.yaml"):
        assert sorted(set(line_solution_list(line))) == [
            k["index"] for k in world["kernels"]
        ]
    spec = json.loads((a / "bench" / "cu16_cot" / "context.json").read_text())
    assert spec["sm_count_target"] == 16 and spec["cotenant_cus"] == 48
    assert spec["library_path"].endswith("libdir_pool")
    for label, n in m["shapes"].items():
        parts = sorted((a / "bench" / label).glob("part_*"))
        assert len(parts) == math.ceil(n / AUDIT["part_shapes"])
        for p in parts:
            assert (p / ro.STAGE_MARKER).is_file()
            assert (p / "stage03b" / "rebench.json").is_file()


def test_audit_mode_resumes_the_interrupted_part(world):
    part0 = world["audit"] / "bench" / "exclusive" / "part_0000"
    assert (part0 / ro.STAGE_MARKER).read_text() == world["part0_marker"]
    assert not (part0 / "stale").exists()
    resumed = world["resumed"]
    assert resumed[0] == ("exclusive", "part_0001", "stage03b")
    exclusive = [r for r in resumed if r[0] == "exclusive"]
    assert [r[1] for r in exclusive[1:]] == ["part_0002", "part_0002"]
    enrich = [r for r in resumed if r[2] == "stage04"]
    assert {r[0] for r in resumed if r not in enrich} == {"exclusive", *CONTEXTS}
    assert enrich == [("bench", g, "stage04") for g in (al.EXCLUSIVE, *CONTEXTS)]
    for g in (al.EXCLUSIVE, *CONTEXTS):
        i = resumed.index(("bench", g, "stage04"))
        assert all(r[0] != g for r in resumed[i + 1 :])
    assert resumed[-1] == ("bench", CONTEXTS[-1], "stage04")
    gen0 = al.manifest(world["audit"])["generations"][0]
    assert gen0["stage04"] == [f"bench/{g}/stage04" for g in (al.EXCLUSIVE, *CONTEXTS)]


def test_every_part_bench_runs_under_the_lease_of_its_gpus(world):
    devices = list(world["cfg"]["runtime"]["devices"])
    benches = {
        (owner, key): gpus
        for (owner, key), gpus in world["leased"].items()
        if owner.name.startswith("part_")
    }
    assert {key for (_, key) in benches} == {"stage03", "stage03b"}
    assert all(gpus == devices for gpus in benches.values())


def test_a_part_the_probe_calibrated_resumes_on_its_own_lines(world):
    part = world["probed_part"]
    args = world["resumed_args"][(part, "stage03b")]
    assert args[args.index("--in-dir") + 1] == str(part)
    assert not [k for (owner, k) in world["resumed_args"] if k == "stage02"]
    own = {shape_of_line(ln): ln for ln in read_bench_lines(part / "shapes.yaml")}
    assert all(line_is_adaptive(ln) for ln in own.values())
    for run in sorted((part / "stage03b" / "runs").glob("run_*")):
        for ln in read_bench_lines(run / "shapes.yaml"):
            line = own[shape_of_line(ln)]
            assert set(line_solution_list(ln)) <= set(line_solution_list(line))
            for key in ("adaptive", *ADAPTIVE_KEYS):
                if key == "min_iters":
                    assert _knob(ln, key) in _reread_min_iters(_knob(line, key))
                else:
                    assert _knob(ln, key) == _knob(line, key)
    assert {p: p.read_bytes() for p in world["probe_era"]} == world["probe_era"]


def _knob(line, key):
    import re

    m = re.search(rf"(?<![\w]){key}:\s*([^,}}]+)", line)
    return m.group(1).strip() if m else None


def _reread_min_iters(own):
    """The `min_iters` a near-winner re-read line of a GEMM line with `own`
    may carry: its own, or the slow band's when that is more."""
    slow = rb.REREAD_DEFAULTS["slow_min_iters"]
    return {own, str(max(int(own), slow))}


def test_every_bench_line_is_timed_adaptively(world):
    """What each line the bench ran carries, over the audit set and the
    training run: adaptive timing with the config's fields (on a re-read
    line of the slow band the band's `min_iters`) and no iteration counts;
    the config's skip ratio on bench lines (with the order-free screen on
    planned ones), none on re-bench lines. The only other lines are
    stage03's warm-up line, first in every yaml (the fixed problem, timed
    and screened as the stage's lines), its discarded warm-up shapes, next,
    and its canary lines, which time one kernel alone without a screen (a
    planned line's first, else, the lines being block-scaled, the first of
    the pool's library file). Every entry of a yaml has a name of its own."""
    timing = {k: str(v) for k, v in adaptive_settings(world["cfg"]).items()}
    ratio = str(world["cfg"]["bench"]["skip_slow_solution_ratio"])
    seen = {"screened": 0, "planned": 0, "rebench": 0, "warmup": 0, "canary": 0}
    seen["warmup_line"] = 0
    for path in sorted(world["benched"].iterdir()):
        lines = read_bench_lines(path)
        names = [_knob(line, "name") for line in lines]
        assert None not in names and len(set(names)) == len(names), path
        lead = names[0] == "warmup_line"
        for i, line in enumerate(lines):
            if names[i] == "warmup_line":
                assert i == 0 and shape_of_line(line) == (512, 512, 512, 1), path
                got = {k: _knob(line, k) for k in timing}
                assert got == timing and _knob(line, "iters") is None, line
                listed = line_solution_list(line)
                assert listed or _knob(line, "requested_solution_num") == "4", line
                seen["warmup_line"] += 1
                continue
            if not line_is_adaptive(line):
                assert i == int(lead) and names[i] == "warmup", path
                assert [_knob(line, k) for k in ("iters", "cold_iters")] == ["1", "1"]
                assert _knob(line, "rotating") == "0" and not line_has_prescreen(line)
                seen["warmup"] += 1
                continue
            got = {k: _knob(line, k) for k in timing}
            if _knob(line, "skip_slow_solution_ratio") == "0":
                assert got.pop("min_iters") in _reread_min_iters(timing["min_iters"])
                got["min_iters"] = timing["min_iters"]
            assert got == timing, line
            assert _knob(line, "iters") is None and _knob(line, "cold_iters") is None
            if names[i].startswith("canary_"):
                assert _knob(line, "skip_slow_solution_ratio") == "0"
                assert len(line_solution_list(line) or []) == 1
                assert not line_has_prescreen(line)
                seen["canary"] += 1
                continue
            assert _knob(line, "requested_solution_num") == "-1"
            assert int(_knob(line, "rotating")) >= 1
            if _knob(line, "skip_slow_solution_ratio") == "0":
                assert line_solution_list(line) and not line_has_prescreen(line)
                seen["rebench"] += 1
            else:
                assert _knob(line, "skip_slow_solution_ratio") == ratio, line
                seen["screened"] += 1
                seen["planned"] += line_has_prescreen(line)
    assert all(seen.values()), seen


def test_audit_rows_are_exhaustive_per_context(world):
    gemms = al.load_gemms(world["audit"])
    n_kernels = len(world["kernels"])
    by_ctx = {}
    for g in gemms:
        c = ev.context_of(g)
        by_ctx.setdefault(c.cu_budget, []).append(g)
        assert len(g["candidates"]) == n_kernels
        assert g["winner_us"] == min(
            c["us"] for c in g["candidates"] if not c["is_skip"]
        )
    assert sorted(by_ctx) == [0, 16, 32]
    for budget in (16, 32):
        per_cell = {}
        for g in by_ctx[budget]:
            key = cell_key(g["m"], g["n"], g["k"], g["batch_count"])
            per_cell[key] = per_cell.get(key, 0) + 1
        assert set(per_cell.values()) == {AUDIT["shapes_per_context"]}
    exclusive = {(g["m"], g["n"], g["k"], g["batch_count"]) for g in by_ctx[0]}
    assert exclusive == set(parse_shapes_yaml(al.shapes_yaml(world["audit"])))


def test_an_audit_dir_is_not_a_training_run(world, tmp_path):
    cfg = world["cfg_path"]
    rc = ro.main(
        ["--config", str(cfg), "--mode", "full", "--run-dir", str(world["audit"])]
    )
    assert rc == 2
    rc = ro.main(
        ["--config", str(cfg), "--mode", "audit", "--run-dir", str(world["run"])]
    )
    assert rc == 2
    other = yaml.safe_load(cfg.read_text())
    other["audit"]["shapes_per_cell"] = 5
    p = tmp_path / "other.yaml"
    p.write_text(yaml.safe_dump(other))
    rc = ro.main(
        ["--config", str(p), "--mode", "audit", "--run-dir", str(world["audit"])]
    )
    assert rc == 2


# ── training with audit.dir ──────────────────────────────────────────────────


def _rounds(world):
    return [world["run"] / "round_0", world["run"] / "round_1"]


def test_rounds_never_draw_audit_shapes(world):
    assert world["train_rc"] == 0
    audit_shapes = set(parse_shapes_yaml(al.shapes_yaml(world["audit"])))
    for rd in _rounds(world):
        drawn = set(parse_shapes_yaml(rd / "stage01" / "shapes.yaml"))
        assert drawn and not drawn & audit_shapes
        cat = json.loads((rd / "stage01" / "categorization.json").read_text())
        assert cat["n_excluded_shapes"] == len(audit_shapes)


def test_no_training_command_reaches_the_audit_set(world):
    audit = str(world["audit"])
    for rd in _rounds(world):
        for marker in rd.rglob(ro.STAGE_MARKER):
            cmd = json.loads(marker.read_text()).get("cmd") or []
            stage = marker.parent.name
            touching = [c for c in cmd if audit in str(c)]
            if stage in ("stage04b", "stage06"):
                assert touching == [audit], (stage, touching)
            elif stage == "stage01":
                assert touching == [str(al.shapes_yaml(world["audit"]))]
            else:
                assert touching == [], (marker, touching)
    snapshot = sorted((world["run"] / "configs").glob("*_full.yaml"))[-1]
    pipe = ro.Pipeline(
        cfg=yaml.safe_load(snapshot.read_text()),
        snapshot=snapshot,
        run_dir=world["run"],
        paths=ro.derive_paths(world["cfg"]),
        stages_dir=PIPELINE / "stages",
        timing=ro.Timing(),
        force_deploy=False,
    )
    step = ro.RoundStep(1, "active", [_rounds(world)[0]], None, True)
    args = pipe.args_stage05(_rounds(world)[1], step, None)
    assert not [a for a in args if audit in a]
    assert al.audit_inputs(al.existing_paths(args)) == []


def test_stage05_refuses_audit_inputs(world, tmp_path):
    cells = ",".join(f"Large|Mid|LargeK|Bnone#M<={1000 + i}" for i in range(200))
    cmd = [
        sys.executable,
        str(PIPELINE / "stages" / "stage05_train.py"),
        "--enriched-csv-dir",
        str(al.group_stage04(world["audit"], al.EXCLUSIVE)),
        "--output-dir",
        str(tmp_path / "out"),
        "--arch",
        ARCH,
        "--only-cells",
        cells,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 2, proc.stderr
    assert "never trained on" in proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    assert not (tmp_path / "out").exists()


def test_stage05_checks_every_path_argument():
    s5 = ro.stage05_module()
    dests = {a.dest: a for a in s5.arg_parser()._actions}
    assert set(s5.PATH_ARGUMENTS) <= set(dests)
    typed = {d for d, a in dests.items() if a.type is Path}
    assert typed <= set(s5.PATH_ARGUMENTS)


def _check_record(rec, n_gemms):
    assert rec["status"] == "ok", rec.get("reason")
    assert rec["n_gemms"] == n_gemms and rec["n_excluded_trained"] == 0
    assert set(rec["groups"]) == {al.EXCLUSIVE, *CONTEXTS}
    for g in rec["groups"].values():
        assert g["n_cells"] == len(g["per_cell"]) == N_CELLS
        assert sum(g["classes"].values()) == g["n_cells"]
        assert sum(c["n"] + c["n_not_evaluated"] for c in g["per_cell"].values()) == (
            g["n_gemms"] + g["n_not_evaluated"]
        )
        for c in g["per_cell"].values():
            assert 0 < c["ceiling_rules"] <= 1 and 0 < c["ceiling_table"] <= 1
            assert 0 < c["ceiling_served"] <= c["ceiling_rules"] + 1e-12
            assert c["ceiling_table_rules"] <= c["ceiling_rules"] + 1e-12
            assert c["ceiling_table_rules"] <= c["ceiling_table"] + 1e-12
            assert c["sel_eff"] <= 1
            assert (c["class"] == "passing") == (c["sel_eff"] >= rec["target"])
    head = rec["headline"]
    assert head["cell_mean"] == rec["groups"][al.EXCLUSIVE]["cell_mean"]
    assert head["min_cell"]["cell"] in rec["groups"][al.EXCLUSIVE]["per_cell"]


def test_every_round_reports_the_audit(world):
    n = sum(al.manifest(world["audit"])["shapes"].values())
    r0, r1 = _rounds(world)
    _check_record(json.loads((r0 / "stage05" / "audit.json").read_text()), n)
    entering = json.loads((r1 / "stage04b" / "audit.json").read_text())
    _check_record(entering, n)
    assert entering["model"] == str(r0 / "stage05" / "models.pt")
    candidate = json.loads((r1 / "stage06" / "audit.json").read_text())
    _check_record(candidate, n)
    record = json.loads((r1 / "stage06" / "deploy_record.json").read_text())
    assert record["audit"]["status"] == "ok"
    assert record["audit"]["headline"] == candidate["headline"]
    report = (world["run"] / "report.html").read_text()
    assert 'id="audit"' in report and "Weakest cells" in report
    assert "Candidate against the previous round" in report
    assert "round_0 &rarr; round_1" in report
    for rec in (entering, candidate):
        assert rec["status_text"] == "complete"
        per = rec["groups"][al.EXCLUSIVE]["per_gemm"]
        assert len(per) == rec["groups"][al.EXCLUSIVE]["n_gemms"]


def test_the_audit_measures_like_the_engine(world, tmp_path):
    from lib.bank_pool import config_pool
    from lib.hardware import DeviceHardware

    r1 = _rounds(world)[1]
    rec = json.loads((r1 / "stage06" / "audit.json").read_text())
    data = Path(
        json.loads((r1 / "stage06" / "deploy_record.json").read_text())["staged_model"]
    ).read_bytes()
    assert rec["model_sha256"] == __import__("hashlib").sha256(data).hexdigest()
    pool = config_pool(world["cfg"], world["libdir"], bf.LIBRARY_STEM, 0)
    gemms = [
        g for g in al.load_gemms(world["audit"]) if ev.context_of(g).cu_budget == 0
    ]
    out = ev.evaluate_model(
        data,
        gemms,
        hardware=DeviceHardware(**world["cfg"]["hardware"]),
        pools=ev.pools_for(gemms, ev.pool_from_kernels(pool.kernels)),
        pick_measurement="warmup",
    )
    by_cell = {}
    for g, r in zip(gemms, out.results):
        if r.sel_eff is not None:
            key = cell_key(g["m"], g["n"], g["k"], g["batch_count"])
            by_cell.setdefault(key, []).append(math.log(r.sel_eff))
    cells = rec["groups"][al.EXCLUSIVE]["per_cell"]
    for key, logs in by_cell.items():
        assert cells[key]["sel_eff"] == pytest.approx(math.exp(sum(logs) / len(logs)))


def test_audit_eval_scores_rounds_and_is_null_without_the_set(world, tmp_path):
    script = PIPELINE / "scripts" / "audit_eval.py"
    rounds = _rounds(world)

    def run(audit_dir, *extra):
        out = tmp_path / f"out_{len(list(tmp_path.iterdir()))}.json"
        cmd = [sys.executable, str(script), "--audit-dir", str(audit_dir)]
        cmd += ["--config", str(world["cfg_path"]), "--out", str(out), *extra]
        assert subprocess.run(cmd).returncode == 0
        return json.loads(out.read_text())

    both = run(
        world["audit"], "--round-dir", str(rounds[0]), "--round-dir", str(rounds[1])
    )
    assert [r["what"] for r in both["models"]] == ["round_0", "round_1"]
    n = sum(al.manifest(world["audit"])["shapes"].values())
    for r in both["models"]:
        _check_record(r, n)
    own = json.loads((rounds[1] / "stage06" / "audit.json").read_text())
    assert both["models"][1]["headline"] == own["headline"]

    missing = run(tmp_path / "nowhere", "--round-dir", str(rounds[0]))
    assert missing["status"] == "missing" and missing["headline"] is None


def test_audit_eval_reads_a_run_started_with_the_probe(world, tmp_path):
    """An old run's config snapshot has a probe section and its rounds have
    stage02 outputs; audit_eval scores the rounds as before."""
    old = json.loads(json.dumps(world["cfg"]))
    old["probe"] = {"duration_us": 2000, "skip_ratio_mode": "sigmoid"}
    old["bench"]["adaptive"] = {"enabled": False}
    snapshot = tmp_path / "20000101_000000_full.yaml"
    snapshot.write_text(yaml.safe_dump(old))
    rounds = _rounds(world)
    for rd in rounds:
        probed = rd / "stage02"
        if not probed.exists():
            probed.mkdir()
            (probed / "shapes_probed.yaml").write_text(
                (rd / "stage01" / "shapes.yaml")
                .read_text()
                .replace(", adaptive: true", "")
            )
            ro.mark_stage_complete(probed, ["stage02_probe.py"])
    script = PIPELINE / "scripts" / "audit_eval.py"
    out = tmp_path / "old.json"
    cmd = [sys.executable, str(script), "--audit-dir", str(world["audit"])]
    cmd += ["--config", str(snapshot), "--out", str(out)]
    cmd += ["--round-dir", str(rounds[0]), "--round-dir", str(rounds[1])]
    assert subprocess.run(cmd).returncode == 0
    got = json.loads(out.read_text())["models"]
    n = sum(al.manifest(world["audit"])["shapes"].values())
    for rec in got:
        _check_record(rec, n)
    own = json.loads((rounds[1] / "stage06" / "audit.json").read_text())
    assert got[1]["headline"] == own["headline"]


def test_a_leaked_shape_is_left_out(world, tmp_path):
    fake = tmp_path / "round_0" / "stage01"
    fake.mkdir(parents=True)
    lines = read_bench_lines(world["audit"] / "stage02" / "shapes.yaml")[:3]
    (fake / "shapes.yaml").write_text("".join(lines))
    script = PIPELINE / "scripts" / "audit_eval.py"
    out = tmp_path / "leak.json"
    cmd = [sys.executable, str(script), "--audit-dir", str(world["audit"])]
    cmd += ["--config", str(world["cfg_path"]), "--out", str(out)]
    cmd += ["--model-file", str(_rounds(world)[0] / "stage05" / "models.pt")]
    cmd += ["--trained-round-dir", str(tmp_path / "round_0")]
    assert subprocess.run(cmd).returncode == 0
    rec = json.loads(out.read_text())
    assert rec["status"] == "ok" and rec["n_excluded_trained"] == 3
    total = sum(al.manifest(world["audit"])["shapes"].values())
    assert rec["n_gemms"] < total


# ── partial audits ───────────────────────────────────────────────────────────


def audit_eval(world, audit_dir, out, *extra):
    cmd = [sys.executable, str(PIPELINE / "scripts" / "audit_eval.py")]
    cmd += ["--audit-dir", str(audit_dir), "--config", str(world["cfg_path"])]
    cmd += ["--out", str(out), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(Path(out).read_text()), proc.stdout + proc.stderr


def _files(root):
    return sorted(
        (str(p.relative_to(root)), p.stat().st_mtime_ns)
        for p in Path(root).rglob("*")
        if p.is_file()
    )


def _benching_copy(world, dst):
    """The complete audit set as it was while generation 0 was benched:
    no generation, no completion marker."""
    import shutil

    shutil.copytree(world["audit"], dst)
    (dst / al.COMPLETE_MARKER).unlink()
    m = al.manifest(dst)
    m.update(generations=[], status="benching")
    (dst / al.MANIFEST).write_text(json.dumps(m))
    return dst


def test_an_interrupted_build_evaluates_its_finished_groups(world, tmp_path):
    a = tmp_path / "audit_set"
    cmd = [*world["audit_cmd"]]
    cmd[cmd.index("--run-dir") + 1] = str(a)
    real = ro.Pipeline.run_stage

    def stop_at_contexts(self, owner_dir, key, args, round_idx, out_dir=None):
        if Path(owner_dir).parent.name == CONTEXTS[0] and key == "stage03":
            raise ro.Interrupted(signal.SIGTERM)
        return real(self, owner_dir, key, args, round_idx, out_dir)

    mp = pytest.MonkeyPatch()
    mp.setattr(ro.Pipeline, "run_stage", stop_at_contexts)
    try:
        assert ro.main(cmd) == 128 + signal.SIGTERM
    finally:
        mp.undo()
    assert (al.group_stage04(a, al.EXCLUSIVE) / ro.STAGE_MARKER).is_file()
    st = al.state(a)
    assert st.status == "partial" and st.evaluable == [al.EXCLUSIVE]
    assert {k: g.status for k, g in st.groups.items()} == {
        al.EXCLUSIVE: "ok",
        **{c: "benching" for c in CONTEXTS},
    }

    rounds = _rounds(world)
    args = ["--round-dir", str(rounds[0]), "--round-dir", str(rounds[1])]
    out, printed = audit_eval(world, a, tmp_path / "partial.json", *args)
    text = f"partial: exclusive complete, {len(CONTEXTS)} contexts benching"
    assert text in printed
    for rec in out["models"]:
        assert rec["status"] == "partial" and rec["status_text"] == text
        assert rec["groups"][al.EXCLUSIVE]["n_cells"] == N_CELLS
        assert rec["headline"]["cell_mean"] == rec["groups"][al.EXCLUSIVE]["cell_mean"]
        for c in CONTEXTS:
            assert rec["groups"][c] is None
            gs = rec["group_status"][c]
            assert gs["status"] == "benching" and gs["parts_done"] < gs["parts_total"]
        assert rec["n_gemms"] == AUDIT["shapes_per_cell"] * N_CELLS
    change = out["changes"][0]["groups"]
    assert change[al.EXCLUSIVE]["n_gemms"] == AUDIT["shapes_per_cell"] * N_CELLS
    for c in CONTEXTS:
        assert "reason" in change[c] and "partial" in change[c]["reason"]

    assert ro.main(cmd) == 0
    assert al.state(a).status == "ok"
    done, _ = audit_eval(world, a, tmp_path / "done.json", *args)
    for rec in done["models"]:
        _check_record(rec, sum(al.manifest(a)["shapes"].values()))
        assert rec["status_text"] == "complete"


def test_a_partial_record_scores_finished_groups_as_the_complete_one(world, tmp_path):
    a = _benching_copy(world, tmp_path / "audit_set")
    marker = al.part_dir(a, CONTEXTS[1], 0) / ro.STAGE_MARKER
    marker.unlink()
    import shutil

    shutil.rmtree(al.group_stage04(a, CONTEXTS[0]))
    rd = ["--round-dir", str(_rounds(world)[1])]
    full, _ = audit_eval(world, world["audit"], tmp_path / "full.json", *rd)
    before = _files(a)
    part, _ = audit_eval(world, a, tmp_path / "part.json", *rd)
    assert part["status"] == "partial"
    assert part["status_text"] == (
        "partial: exclusive complete, 1 context not enriched, 1 context benching"
    )
    assert part["groups"][al.EXCLUSIVE] == full["groups"][al.EXCLUSIVE]
    assert part["headline"] == full["headline"]
    assert part["group_status"][CONTEXTS[0]]["status"] == "not_enriched"
    assert part["group_status"][CONTEXTS[1]]["status"] == "benching"
    assert part["groups"][CONTEXTS[0]] is None and part["groups"][CONTEXTS[1]] is None

    enrich = tmp_path / "enriched"
    got, _ = audit_eval(
        world, a, tmp_path / "enriched.json", *rd, "--enrich-dir", str(enrich)
    )
    assert _files(a) == before
    src = json.loads((enrich / CONTEXTS[0] / "stage04" / al.GROUP_SOURCE).read_text())
    assert src["label"] == CONTEXTS[0] and src["logs"]
    assert got["status_text"] == (
        f"partial: {al.EXCLUSIVE}, {CONTEXTS[0]} complete, 1 context benching"
    )
    assert got["groups"][CONTEXTS[0]] == full["groups"][CONTEXTS[0]]
    assert got["groups"][CONTEXTS[1]] is None
    assert not (enrich / CONTEXTS[1]).exists()
    again, _ = audit_eval(
        world, a, tmp_path / "again.json", *rd, "--enrich-dir", str(enrich)
    )
    assert again["groups"] == got["groups"]
    log = al.group_logs(a, CONTEXTS[0], al.units(a)[CONTEXTS[0]])[0]
    log.write_text(log.read_text() + "\n")
    assert al.enriched_copy(enrich, a, CONTEXTS[0], al.units(a)[CONTEXTS[0]]) is None

    for bad in (a / "x", a / "bench"):
        cmd = [sys.executable, str(PIPELINE / "scripts" / "audit_eval.py")]
        cmd += ["--audit-dir", str(a), "--config", str(world["cfg_path"])]
        cmd += ["--out", str(tmp_path / "bad.json"), *rd, "--enrich-dir", str(bad)]
        assert subprocess.run(cmd, capture_output=True).returncode == 2


def _before_the_rename(audit_dir):
    """`audit_dir` as the pipeline wrote it before the planner became
    stage02: its plan in stage01b/, no run_layout.json."""
    for suffix in ("", ".log"):
        (audit_dir / f"stage02{suffix}").rename(audit_dir / f"stage01b{suffix}")
    (audit_dir / run_layout.LAYOUT_FILE).unlink()
    assert run_layout.of(audit_dir) == run_layout.LEGACY
    return audit_dir


def test_the_audit_tools_read_an_audit_set_written_before_the_rename(world, tmp_path):
    """audit_v2.py plans an audit set whose plan is in stage01b/ as one of
    this pipeline, and audit_eval.py enriches a finished group of it with
    that plan, writing nothing into it."""
    import shutil

    groups = []
    for name in ("now", "old"):
        a = tmp_path / name / "audit_set"
        shutil.copytree(world["audit"], a)
        if name == "old":
            _before_the_rename(a)
        cmd = [sys.executable, str(PIPELINE / "scripts" / "audit_v2.py"), "plan"]
        cmd += ["--audit-dir", str(a), "--config", str(world["cfg_path"])]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        plan = json.loads((v2.v2_dir(a) / "gen_000" / "plan.json").read_text())
        groups.append(plan["groups"])
    assert groups[0] == groups[1] and groups[0][al.EXCLUSIVE]

    a = _before_the_rename(_benching_copy(world, tmp_path / "audit_set"))
    shutil.rmtree(al.group_stage04(a, CONTEXTS[0]))
    rd = ["--round-dir", str(_rounds(world)[1])]
    full, _ = audit_eval(world, world["audit"], tmp_path / "full.json", *rd)
    before = _files(a)
    enrich = tmp_path / "enriched"
    got, _ = audit_eval(
        world, a, tmp_path / "enriched.json", *rd, "--enrich-dir", str(enrich)
    )
    assert got["groups"][CONTEXTS[0]] == full["groups"][CONTEXTS[0]]
    assert _files(a) == before


def test_a_partial_audit_without_a_finished_group_is_null(world, tmp_path):
    a = _benching_copy(world, tmp_path / "audit_set")
    for label in (al.EXCLUSIVE, *CONTEXTS):
        (al.part_dir(a, label, 0) / ro.STAGE_MARKER).unlink()
    rec, printed = audit_eval(
        world, a, tmp_path / "none.json", "--round-dir", str(_rounds(world)[1])
    )
    assert rec["status"] == "incomplete" and rec["groups"] is None
    assert rec["headline"] is None and "null" in printed
    assert {g["status"] for g in rec["group_status"].values()} == {"benching"}


def test_stage_records_are_partial_while_the_audit_is_benched(world, tmp_path):
    a = _benching_copy(world, tmp_path / "audit_set")
    (al.part_dir(a, CONTEXTS[0], 0) / ro.STAGE_MARKER).unlink()
    from lib.hardware import DeviceHardware

    r1 = _rounds(world)[1]
    data = Path(
        json.loads((r1 / "stage06" / "deploy_record.json").read_text())["staged_model"]
    ).read_bytes()
    spec = al.Spec(
        audit_dir=a,
        library_dir=world["libdir"],
        library_stem=bf.LIBRARY_STEM,
        bootstrap=AUDIT["bootstrap"],
        trained_rounds=al.rounds_until(r1, True),
    )
    rec = al.run_for_stage(
        tmp_path / "audit.json",
        data,
        spec,
        world["cfg"],
        hardware=DeviceHardware(**world["cfg"]["hardware"]),
        what="stage06: the candidate bundle",
        source="x",
        quiet=True,
    )
    on_disk = json.loads((tmp_path / "audit.json").read_text())
    assert rec["status"] == on_disk["status"] == "partial"
    assert on_disk["status_text"] == (
        f"partial: {al.EXCLUSIVE}, {CONTEXTS[1]} complete, 1 context benching"
    )
    lines = al.summary_lines(on_disk)
    assert lines[0] == f"audit {on_disk['status_text']}"
    assert any(
        line.startswith(f"audit {CONTEXTS[0]}: null (benching") for line in lines
    )


def test_compare_pairs_gemms_and_needs_both_groups():
    def rec(what, values, status="ok", ctx=None):
        per = [[100 + i, 100, 100, 1, v] for i, v in enumerate(values)]
        groups = {al.EXCLUSIVE: {"per_gemm": per}, "c1": ctx}
        gs = {"c1": {"status": "ok" if ctx else "benching"}}
        return {"what": what, "status": status, "groups": groups, "group_status": gs}

    before = rec("r0", [0.80] * 6 + [0.95] * 6)
    after = rec("r1", [0.88, 0.87, 0.89, 0.88, 0.86, 0.88] + [0.95] * 6)
    c, why = al.compare(before, after, bootstrap=200)
    assert why is None and c["n_gemms"] == 12 and c["n_cells"] == 1
    cell = cell_key(100, 100, 100, 1)
    assert c["per_cell"][cell]["moved"] == "improved" and c["improved"] == [cell]
    assert c["cell_mean_ratio"] > 1 and c["cell_mean_ratio_ci"][0] > 1
    back, _ = al.compare(after, before, bootstrap=200)
    assert back["regressed"] == [cell]
    flat, _ = al.compare(before, before, bootstrap=200)
    assert flat["cell_mean_ratio"] == pytest.approx(1.0) and not flat["improved"]
    none, why = al.compare(before, after, "c1")
    assert none is None and "c1" in why
    nulled = dict(after, status="incomplete", groups=None)
    assert al.compare(before, nulled)[0] is None
    old = {"what": "r", "status": "ok", "groups": {al.EXCLUSIVE: {"n": 1}}}
    assert "per-GEMM" in al.compare(old, after)[1]
    ch = al.changes([before, after, nulled], bootstrap=50)
    assert [list(c["groups"]) for c in ch] == [
        [al.EXCLUSIVE, "c1"],
        [al.EXCLUSIVE, "c1"],
    ]
    assert "reason" in ch[1]["groups"][al.EXCLUSIVE]
    assert al.change_lines(ch[0])[0].startswith(f"r0 -> r1 {al.EXCLUSIVE}: cell mean")


def test_not_worse_pairs_the_cell_means_under_the_shared_protocol():
    shapes = [(64, 64, 64, 1), (64, 64, 128, 1), (4096, 4096, 4096, 1)]

    def rec(values, v2=None):
        group = {"per_gemm": [[*s, v] for s, v in zip(shapes, values)]}
        out = {"status": "ok", "what": "m", "groups": {al.EXCLUSIVE: group}}
        if v2 is not None:
            g2 = {"per_gemm": [[*s, v] for s, v in zip(shapes, v2)]}
            out["v2"] = {"status": "ok", "groups": {al.EXCLUSIVE: g2}}
        return out

    base = rec([0.90, 0.90, 0.95])
    verdict, why, change, proto = al.not_worse(base, rec([0.90, 0.90, 0.95]))
    assert verdict is True and proto == "v1" and "x1.0000" in why
    verdict, _, change, _ = al.not_worse(base, rec([0.80, 0.80, 0.99]))
    assert verdict is False and change["cell_mean_ratio"] < 1
    verdict, _, _, proto = al.not_worse(
        rec([0.90] * 3, v2=[0.95] * 3), rec([0.89] * 3, v2=[0.96] * 3)
    )
    assert (verdict, proto) == (True, "v2")
    verdict, why, change, _ = al.not_worse(base, {"status": "missing"})
    assert verdict is None and change is None and "has no exclusive group" in why


# ── the per-cell gate ──────────────────────────────────────────────────────

SMALL = [(100 + i, 100, 100, 1) for i in range(8)]
LARGE_SHAPES = [(4096 + i, 4096, 4096, 1) for i in range(8)]


def _gate_rec(groups, v2=None, what="m"):
    """An evaluated record of {group: {shape: selection efficiency}}."""

    def group(per):
        return {"per_gemm": [[*sh, v] for sh, v in per.items()]}

    rec = {
        "status": "ok",
        "what": what,
        "groups": {k: group(v) for k, v in groups.items()},
    }
    if v2 is not None:
        rec["v2"] = {"status": "ok", "groups": {k: group(v) for k, v in v2.items()}}
    return rec


def _cells(values, shapes=SMALL):
    return dict(zip(shapes, values))


def _grid(n_cells, per_cell=8):
    """Shapes of `n_cells` distinct cells (M and N tiers x K tiers x batch),
    `per_cell` GEMMs each."""
    tiers = (16, 100, 300, 2000)
    out = []
    for c in range(n_cells):
        m, n = tiers[c % 4], tiers[(c // 4) % 4]
        k = (16, 200, 1000)[(c // 16) % 3]
        b = 1 if c < 48 else 4
        out.append([(m + i, n, k, b) for i in range(per_cell)])
    assert len({cell_key(*s[0]) for s in out}) == n_cells
    return out


def _noisy_pair(rng, cells, base=0.9, spread=0.04, changed=0.4, scale=0.15):
    """(before, after) {shape: efficiency}: per GEMM a base efficiency, and
    for a share `changed` of the GEMMs a change of heavy-tailed size."""
    before, after = {}, {}
    for shapes in cells:
        for s in shapes:
            a = min(1.0, base * math.exp(rng.normal(0, spread)))
            d = rng.standard_t(3) * scale if rng.random() < changed else 0.0
            before[s], after[s] = a, min(1.0, a * math.exp(d))
    return before, after


def test_the_cell_gate_finds_cells_significantly_worse_beyond_the_tolerance():
    import numpy as np

    rng = np.random.default_rng(3)
    cells = _grid(12)
    before, after = _noisy_pair(rng, cells, changed=0.3, scale=0.05)
    worse = cells[5]
    for s in worse:
        after[s] = before[s] * 0.9 * math.exp(rng.normal(0, 0.01))
    out = al.cell_gate(
        _gate_rec({al.EXCLUSIVE: before}, what="shipped"),
        _gate_rec({al.EXCLUSIVE: after}, what="new"),
        bootstrap=400,
    )
    cell = cell_key(*worse[0])
    assert out["passed"] is False and out["n_regressed"] == 1
    assert out["n_cells_tested"] == 12 and out["cell_rules"] == "on"
    (reg,) = out["groups"][al.EXCLUSIVE]["regressed"]
    assert reg["cell"] == cell and reg["ratio"] == pytest.approx(0.9, abs=0.01)
    assert reg["p"] < al.CELL_GATE_DEFAULTS["alpha"]
    assert out["groups"][al.EXCLUSIVE]["p"][cell] == reg["p"]
    assert out["reason"].startswith(
        f"1 cell(s) significantly worse beyond 1.0%: exclusive {cell} x0."
    )
    assert out["reason"].endswith("(exclusive v1)")
    assert (out["before"], out["after"]) == ("shipped", "new")
    # a decline within the tolerance, however consistent, is no finding
    slight = {**after, **{s: before[s] * 0.995 for s in worse}}
    out = al.cell_gate(
        _gate_rec({al.EXCLUSIVE: before}),
        _gate_rec({al.EXCLUSIVE: slight}),
        bootstrap=400,
    )
    assert out["passed"] is True and out["n_regressed"] == 0
    strict = al.cell_gate(
        _gate_rec({al.EXCLUSIVE: before}),
        _gate_rec({al.EXCLUSIVE: slight}),
        {"cell_tolerance": 0.001},
        bootstrap=400,
    )
    assert strict["n_regressed"] == 1
    # half the cell's GEMMs up, half down: no significant decline
    mixed = {**after, **{s: before[s] * (0.8, 1.2)[i % 2] for i, s in enumerate(worse)}}
    out = al.cell_gate(
        _gate_rec({al.EXCLUSIVE: before}),
        _gate_rec({al.EXCLUSIVE: mixed}),
        bootstrap=400,
    )
    assert out["n_regressed"] == 0
    same = al.cell_gate(
        _gate_rec({al.EXCLUSIVE: before}), _gate_rec({al.EXCLUSIVE: before})
    )
    assert same["passed"] is True
    assert same["reason"].startswith(
        "no cell significantly worse beyond 1.0% or confidently fallen below 85.0% "
        "(family-wise alpha 0.05 over 12 cells of exclusive), and no new pick "
        "below 50.0% in any group (exclusive v1)"
    )


def test_the_cell_rules_hold_their_level_across_cells():
    """When the two bundles are exchangeable (each GEMM's two efficiencies
    as likely the other way round), the cell rules keep a bundle out at most
    alpha of the time over all their cells; a 95% interval per cell would do
    so far more often."""
    import numpy as np

    rng = np.random.default_rng(11)
    cells = _grid(32)
    before, after = _noisy_pair(rng, cells, base=0.87, changed=0.5, scale=0.12)
    shapes = sorted(before)
    settings = {"tail_floor": 0}
    draws, refused, per_cell = 150, 0, 0
    for _ in range(draws):
        swap = rng.random(len(shapes)) < 0.5
        x = {s: (after[s] if w else before[s]) for s, w in zip(shapes, swap)}
        y = {s: (before[s] if w else after[s]) for s, w in zip(shapes, swap)}
        rx, ry = _gate_rec({al.EXCLUSIVE: x}), _gate_rec({al.EXCLUSIVE: y})
        out = al.cell_gate(rx, ry, settings, bootstrap=200)
        refused += int(bool(out["n_regressed"] or out["n_floor"]))
        change, _ = al.compare(rx, ry, bootstrap=200)
        per_cell += int(
            any(
                c["ratio_ci"] is not None and c["ratio_ci"][1] < 0.99
                for c in change["per_cell"].values()
            )
        )
    alpha = al.CELL_GATE_DEFAULTS["alpha"]
    assert refused / draws <= alpha + 2 * math.sqrt(alpha * (1 - alpha) / draws)
    assert per_cell / draws > 4 * alpha


def test_a_cell_falls_below_the_floor_only_when_its_own_interval_does():
    import numpy as np

    rng = np.random.default_rng(5)
    cells = _grid(8)
    before, after = _noisy_pair(rng, cells, changed=0.2, scale=0.05)
    cell = cells[2]
    for s in cell:
        before[s] = 0.88 * math.exp(rng.normal(0, 0.02))
        after[s] = 0.78 * math.exp(rng.normal(0, 0.02))
    out = al.cell_gate(
        _gate_rec({al.EXCLUSIVE: before}),
        _gate_rec({al.EXCLUSIVE: after}),
        {"cell_tolerance": 0.5},
        bootstrap=400,
    )
    (fell,) = out["groups"][al.EXCLUSIVE]["floor"]
    assert (out["passed"], out["n_floor"], out["n_regressed"]) == (False, 1, 0)
    assert fell["cell"] == cell_key(*cell[0]) and fell["after_ci"][1] < 0.85
    assert fell["before"] >= 0.85 and fell["after"] < 0.85
    assert out["reason"].startswith(
        f"1 cell(s) confidently fell below 85.0%: exclusive {fell['cell']} "
        f"{fell['after'] * 100:.2f}% (was {fell['before'] * 100:.2f}%) p "
    )
    # across the floor by a step its own interval straddles: no finding
    for s in cell:
        before[s] = 0.852 * math.exp(rng.normal(0, 0.03))
        after[s] = before[s] * 0.996
    out = al.cell_gate(
        _gate_rec({al.EXCLUSIVE: before}),
        _gate_rec({al.EXCLUSIVE: after}),
        bootstrap=400,
    )
    assert out["n_floor"] == 0 and out["passed"] is True
    off = al.cell_gate(
        _gate_rec({al.EXCLUSIVE: before}),
        _gate_rec({al.EXCLUSIVE: after}),
        {"cell_floor": 0},
        bootstrap=400,
    )
    assert off["passed"] is True and "confidently fallen" not in off["reason"]


def test_decline_p_is_family_wise_with_one_sign_per_shape():
    shapes = [(i + 1, 1, 1, 1) for i in range(8)]
    down = [-0.1] * 8
    calm = [0.0] * 8
    p = al.decline_p({"a": (shapes, down), "b": (shapes[:4], calm[:4])}, 400)
    assert p["a"] < 0.05 and p["b"] == 1.0
    one = al.decline_p({"a": (shapes, down)}, 400)
    many = al.decline_p(
        {"a": (shapes, down), **{i: ([(100 + i, 1, 1, 1)], [0.3]) for i in range(30)}},
        400,
    )
    assert many["a"] >= one["a"]
    assert al.decline_p({}, 400) == {}
    assert al.decline_p({"a": (shapes, down)}, 0) == {"a": 1.0}
    assert al.decline_p({"a": ([], [])}, 400) == {"a": 1.0}
    # the smallest p it can give is 2 / (flips + 1): min_flips reaches alpha
    assert al.min_flips(0.05) == 40
    assert al.decline_p({"a": (shapes, down)}, 40)["a"] < 0.05
    assert al.decline_p({"a": (shapes, down)}, 39)["a"] >= 0.05


def test_the_tail_rule_counts_every_new_collapse_and_reports_the_healed_ones():
    vals = [0.9, 0.6, 0.9, 0.3, 0.55, 0.4] + [0.99] * 2
    before = _gate_rec({al.EXCLUSIVE: _cells(vals)}, v2={al.EXCLUSIVE: _cells(vals)})

    def gate(after_vals, **st):
        per = {al.EXCLUSIVE: _cells(after_vals)}
        after = _gate_rec(per, v2=per)
        return al.cell_gate(before, after, {"cell_floor": 0, **st}, bootstrap=200)

    # 0.9 -> 0.3 halves, 0.6 -> 0.45 and 0.55 -> 0.4 only step across the
    # line: all three are new; 0.3 -> 0.95 is healed, 0.4 -> 0.2 persists
    out = gate([0.3, 0.45, 0.9, 0.95, 0.4, 0.2] + [0.99] * 2)
    g = out["groups"][al.EXCLUSIVE]
    assert [t["shape"] for t in g["tail"]] == [
        list(SMALL[0]),
        list(SMALL[1]),
        list(SMALL[4]),
    ]
    assert [t["shape"] for t in g["tail_lifted"]] == [list(SMALL[3])]
    assert [t["shape"] for t in g["tail_persisting"]] == [list(SMALL[5])]
    assert (g["n_tail_new"], g["n_tail_lifted"], g["tail_refuses"]) == (3, 1, True)
    assert all(t["decides"] for t in g["tail"])
    # the healed pick offsets none of them
    assert (out["passed"], out["n_tail"], out["tail_groups"]) == (
        False,
        3,
        [al.EXCLUSIVE],
    )
    assert (out["n_tail_new"], out["n_tail_healed"], out["n_tail_persisting"]) == (
        3,
        1,
        1,
    )
    assert out["reason"].startswith(
        "3 new pick(s) below 50.0% in 1 group(s): exclusive (3 new, 1 healed: "
        "100x100x100 (Small|Small|MidK|Bnone) 30.00% (was 90.00%), "
    )
    assert out["reason"].endswith(" (exclusive v2)")
    # the budget allows that many, over every group
    assert gate([0.3, 0.45, 0.9, 0.95, 0.4, 0.2] + [0.99] * 2, tail_budget=3)["passed"]
    within = gate([0.3, 0.45, 0.9, 0.95, 0.4, 0.2] + [0.99] * 2, tail_budget=3)
    assert within["n_tail"] == 0 and within["n_tail_new"] == 3
    assert "3 finding(s) that do not decide" in within["reason"]
    # healing alone passes, and says so
    healed = gate([0.9, 0.6, 0.9, 0.95, 0.55, 0.4] + [0.99] * 2)
    assert healed["passed"] is True and healed["n_tail_healed"] == 1
    assert "no new pick below 50.0% in any group (1 healed)" in healed["reason"]
    # tail_collapse below 1 counts only the drops to that fraction of before
    halving = gate([0.3, 0.45, 0.9, 0.95, 0.4, 0.2] + [0.99] * 2, tail_collapse=0.5)
    assert halving["n_tail"] == 1 and halving["passed"] is False
    assert halving["reason"].startswith(
        "1 new pick(s) below 50.0% at most 50.0% of the shipped pick's "
        "efficiency in 1 group(s)"
    )
    off = gate([0.3, 0.45, 0.9, 0.3, 0.4] + [0.99] * 3, tail_floor=0)
    assert off["passed"] is True and off["groups"][al.EXCLUSIVE]["tail"] == []
    assert "the tail rule off (gate.tail_floor 0)" in off["reason"]


def test_the_tail_rule_judges_every_group_and_the_cell_rules_their_own():
    def rec(per):
        return _gate_rec(per, v2=per)

    before = rec(
        {
            al.EXCLUSIVE: _cells([0.95] * 8),
            "cu64": _cells([0.95] * 7 + [0.9], LARGE_SHAPES),
        }
    )
    after = rec(
        {
            al.EXCLUSIVE: _cells([0.95] * 8),
            "cu64": _cells([0.6] * 7 + [0.2], LARGE_SHAPES),
        }
    )
    no_floor = {"cell_floor": 0}
    out = al.cell_gate(before, after, no_floor, bootstrap=200)
    cu64 = out["groups"]["cu64"]
    assert cu64["deciding"] is False and len(cu64["regressed"]) == 1
    assert cu64["tail_refuses"] is True and out["tail_groups"] == ["cu64"]
    assert out["passed"] is False and out["n_groups"] == 1
    assert (out["n_regressed"], out["n_tail"], out["n_reported"]) == (0, 1, 1)
    assert out["reason"].startswith("1 new pick(s) below 50.0% in 1 group(s): cu64")
    every = al.cell_gate(before, after, {**no_floor, "groups": "all"}, bootstrap=200)
    assert every["n_groups"] == 2 and every["n_regressed"] == 1
    assert every["n_reported"] == 0 and "cu64" in every["reason"]
    calm = rec(
        {
            al.EXCLUSIVE: _cells([0.95] * 8),
            "cu64": _cells([0.6] * 7 + [0.9], LARGE_SHAPES),
        }
    )
    out = al.cell_gate(before, calm, no_floor, bootstrap=200)
    assert out["passed"] is True and out["n_reported"] == 1


def test_a_tail_under_v1_decides_only_when_a_v2_measurement_confirms_it():
    vals = [0.9] + [0.99] * 7
    before = _gate_rec({al.EXCLUSIVE: _cells(vals)})
    after_v1 = _cells([0.3] + [0.99] * 7)

    def tails(v2):
        after = _gate_rec(
            {al.EXCLUSIVE: after_v1},
            v2=None if v2 is None else {al.EXCLUSIVE: _cells(v2)},
        )
        out = al.cell_gate(before, after, {"cell_floor": 0}, bootstrap=200)
        assert out["groups"][al.EXCLUSIVE]["protocol"] == "v1"
        return out

    out = tails([0.32] + [0.99] * 7)
    (t,) = out["groups"][al.EXCLUSIVE]["tail"]
    assert t["confirmed"] and t["after_v2"] == 0.32 and out["passed"] is False
    out = tails([0.7] + [0.99] * 7)
    (t,) = out["groups"][al.EXCLUSIVE]["tail"]
    assert not t["confirmed"] and out["passed"] is True and out["n_reported"] == 1
    out = tails(None)
    assert out["groups"][al.EXCLUSIVE]["tail"][0]["confirmed"] is False
    assert out["passed"] is True
    both = al.cell_gate(
        _gate_rec({al.EXCLUSIVE: _cells(vals)}, v2={al.EXCLUSIVE: _cells(vals)}),
        _gate_rec({al.EXCLUSIVE: after_v1}, v2={al.EXCLUSIVE: after_v1}),
        {"cell_floor": 0},
        bootstrap=200,
    )
    (t,) = both["groups"][al.EXCLUSIVE]["tail"]
    assert both["groups"][al.EXCLUSIVE]["protocol"] == "v2"
    assert t["confirmed"] and "after_v2" not in t and both["passed"] is False


def test_the_cell_gate_pairs_each_group_under_v2_when_both_have_it():
    before = _gate_rec(
        {al.EXCLUSIVE: _cells([0.95] * 8)}, v2={al.EXCLUSIVE: _cells([0.97] * 8)}
    )
    after = _gate_rec(
        {al.EXCLUSIVE: _cells([0.80] * 8)}, v2={al.EXCLUSIVE: _cells([0.97] * 8)}
    )
    out = al.cell_gate(before, after, bootstrap=200)
    assert out["passed"] is True and out["groups"][al.EXCLUSIVE]["protocol"] == "v2"
    v1_only = dict(after)
    v1_only.pop("v2")
    out = al.cell_gate(before, v1_only, bootstrap=200)
    assert out["passed"] is False and out["groups"][al.EXCLUSIVE]["protocol"] == "v1"


def test_the_cell_gate_cannot_decide_without_a_paired_group_or_enough_flips():
    after = _gate_rec({al.EXCLUSIVE: _cells([0.9] * 8)})
    out = al.cell_gate({"status": "missing", "what": "shipped"}, after)
    assert out["passed"] is None and out["n_groups"] == 0
    assert out["not_compared"] == [al.EXCLUSIVE]
    assert out["reason"].startswith("no group pairs the two bundles")
    before = _gate_rec({al.EXCLUSIVE: _cells([0.95] * 8)})
    for flips in (0, 39):
        out = al.cell_gate(before, after, bootstrap=flips)
        assert out["passed"] is None and out["n_regressed"] == 0
        assert out["cell_rules"].startswith(f"off: {flips} sign flips")
        assert out["reason"].startswith("the cell rules could not decide")
        assert "the cell rules were off" in out["reason"]
    collapsed = {al.EXCLUSIVE: _cells([0.3] + [0.95] * 7)}
    out = al.cell_gate(
        _gate_rec(
            {al.EXCLUSIVE: _cells([0.95] * 8)}, v2={al.EXCLUSIVE: _cells([0.95] * 8)}
        ),
        _gate_rec(collapsed, v2=collapsed),
        bootstrap=0,
    )
    assert out["passed"] is False and out["tail_groups"] == [al.EXCLUSIVE]
    assert "the cell rules were off" in out["reason"]


def test_the_gate_section_gives_the_cell_gate_settings(tmp_path):
    import test_run_orchestrator as tro

    assert al.cell_gate_settings({}) == al.CELL_GATE_DEFAULTS
    cfg = {
        "gate": {
            "cell_tolerance": 0.02,
            "tail_floor": 0,
            "groups": "all",
            "alpha": 0.1,
            "tail_budget": 2,
        }
    }
    st = al.cell_gate_settings(cfg)
    assert st == {
        **al.CELL_GATE_DEFAULTS,
        "cell_tolerance": 0.02,
        "tail_floor": 0.0,
        "groups": "all",
        "alpha": 0.1,
        "tail_budget": 2,
    }
    assert al.cell_gate_settings({"gate": {"per_cell": False}}) is None
    base = yaml.safe_load(tro.make_config(tmp_path).read_text())
    errors = ro.validate_config(
        dict(
            base,
            gate={
                "cell_floor": 1.5,
                "groups": "x",
                "alpha": 1,
                "tail_collapse": 0,
                "tail_budget": -1,
            },
        )
    )[0]
    for key in ("cell_floor", "groups", "alpha", "tail_collapse", "tail_budget"):
        assert any(f"gate.{key}" in e for e in errors), key
    assert ro.validate_config(dict(base, gate={"per_cell": False}))[0] == []
    assert ro.cfg_get(base, "gate.cell_tolerance") == 0.01
    assert ro.cfg_get(base, "gate.tail_collapse") == 1.0
    assert ro.cfg_get(base, "report.pool_gap_fraction") == 0.3
    # the cell tests take audit.bootstrap as their sign flips
    audited = dict(base, audit={"dir": str(tmp_path / "audit"), "bootstrap": 10})
    errors = ro.validate_config(audited)[0]
    assert any("audit.bootstrap 10" in e and "gate.alpha 0.05" in e for e in errors)
    assert ro.validate_config(dict(audited, gate={"per_cell": False}))[0] == []
    assert ro.validate_config(dict(audited, gate={"alpha": 0.2}))[0] == []


def test_a_bundle_names_its_model_and_the_rounds_it_trained_on(tmp_path):
    run = tmp_path / "run"
    for i in range(3):
        (run / f"round_{i}" / "stage05").mkdir(parents=True)
    staged = run / "round_1" / "stage06" / "staged" / "m.tilewright.bin"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"model")
    (run / "round_1" / "stage06" / "deploy_record.json").write_text(
        json.dumps({"staged_model": str(staged)})
    )
    assert al.bundle_model(run / "round_1") == (staged, "stage06 staged bundle")
    assert al.bundle_model(run / "round_2") == (
        run / "round_2" / "stage05" / "models.pt",
        "stage05 models.pt",
    )
    moved = tmp_path / "moved"
    (run / "round_1").rename(moved)
    assert al.bundle_model(moved)[1] == "stage06 staged bundle"
    (run / "round_1").mkdir()
    assert [p.name for p in al.bundle_trained_rounds(run / "round_1")] == [
        "round_0",
        "round_1",
    ]
    final = run / "final_retrain"
    final.mkdir()
    (final / "final_retrain.json").write_text(
        json.dumps({"rounds": {"round_0": [None, None], "round_2": [None, None]}})
    )
    assert al.bundle_trained_rounds(final) == [
        final.resolve().parent / "round_0",
        final.resolve().parent / "round_2",
    ]


def test_status_text_names_each_group_state():
    G = al.Group
    groups = {
        al.EXCLUSIVE: G(al.EXCLUSIVE, "ok", None, 2, 2),
        "a": G("a", "benching", None, 0, 2),
        "b": G("b", "benching", None, 1, 2),
    }
    assert al.status_text("partial", None, groups) == (
        "partial: exclusive complete, 2 contexts benching"
    )
    groups[al.EXCLUSIVE].status = "benching"
    groups["a"].status = "ok"
    assert al.status_text("partial", None, groups) == (
        "partial: a complete, exclusive and 1 context benching"
    )
    assert al.status_text("ok", None, groups) == "complete"
    assert al.status_text("missing", "why", {}) == "missing: why"


# ── kernel pool changes (last: they rewrite the fake library) ───────────────


def test_a_pool_change_benches_only_the_new_kernels(world, tmp_path):
    a = world["audit"]
    old = world["kernels"]
    new = extra_kernels(2, len(old))
    bf.write_logic(world["libdir"] / f"{bf.LIBRARY_STEM}.dat", old + new)
    st, why = al.status(a, al.pool_digests(_pool(world)))
    assert st == "incomplete" and "2 kernel(s)" in why
    stale = tmp_path / "stale.json"
    cmd = [sys.executable, str(PIPELINE / "scripts" / "audit_eval.py")]
    cmd += ["--audit-dir", str(a), "--config", str(world["cfg_path"])]
    cmd += ["--round-dir", str(_rounds(world)[1]), "--out", str(stale)]
    assert subprocess.run(cmd).returncode == 0
    rec = json.loads(stale.read_text())
    assert rec["status"] == "incomplete" and rec["groups"] is None

    for part in (a / "bench").glob("*/part_*"):
        if not (part / "stage02").exists():
            _probe_era(part)
    assert ro.main(world["audit_cmd"]) == 0
    m = al.manifest(a)
    assert len(m["generations"]) == 2
    assert set(m["generations"][1]["kernels"]) == {str(k["index"]) for k in new}
    inc = a / "increments" / "inc_001"
    ids = "[" + ", ".join(str(k["index"]) for k in new) + "]"
    parts = sorted((inc / "bench").glob("*/part_*"))
    assert parts
    for part in parts:
        assert not (part / "stage02").exists()
        gen0 = a / "bench" / part.parent.name / part.name / "shapes.yaml"
        own = {shape_of_line(ln): ln for ln in read_bench_lines(gen0)}
        for line in read_bench_lines(part / "shapes.yaml"):
            assert line_solution_list(line) == [k["index"] for k in new]
            assert line == set_line_knob(
                own[shape_of_line(line)], "solution_index_list", ids
            )
            assert line_is_adaptive(line) and _knob(line, "iters") is None
    gemms = al.load_gemms(a, current=al.pool_digests(_pool(world)))
    assert all(len(g["candidates"]) == len(old) + len(new) for g in gemms)
    assert al.status(a, al.pool_digests(_pool(world))) == ("ok", None)

    bf.write_logic(world["libdir"] / f"{bf.LIBRARY_STEM}.dat", old[:6])
    assert al.status(a, al.pool_digests(_pool(world))) == ("ok", None)
    assert ro.main(world["audit_cmd"]) == 0
    assert len(al.manifest(a)["generations"]) == 2
    assert not (a / "increments" / "inc_002").exists()
    bf.write_logic(world["libdir"] / f"{bf.LIBRARY_STEM}.dat", old)


def _pool(world):
    from lib.bank_pool import config_pool

    return config_pool(world["cfg"], world["libdir"], bf.LIBRARY_STEM, 0).kernels
