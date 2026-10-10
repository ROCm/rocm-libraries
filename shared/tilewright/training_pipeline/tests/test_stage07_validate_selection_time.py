# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import bench_fakes as bf
import fake_hipblaslt as fh
import gpu_fakes as gf
import stage07_optional_validate_selection_time as s7
from lib import kfd
from lib.dat import kernel_dat_info
from test_evaluate import TWINS

PIPELINE_ROOT = Path(__file__).resolve().parent.parent

ENTRY = {
    "function": "matmul",
    "transA": "T",
    "transB": "N",
    "a_type": "f8_r",
    "b_type": "f8_r",
    "c_type": "bf16_r",
    "d_type": "bf16_r",
    "scale_type": "f32_r",
    "compute_type": "c_f32_r",
    "scaleA": 3,
    "scaleB": 3,
    "batch_count": 1,
}
CFG = {
    "arch": "gfx1250-strict",
    "hipblaslt": {
        "a_type": "f8_r",
        "b_type": "f8_r",
        "c_type": "bf16_r",
        "d_type": "bf16_r",
        "compute_type": "f32_r",
        "scaleA": 3,
        "scaleB": 3,
        "transA": "T",
        "transB": "N",
        "library_stem": bf.LIBRARY_STEM,
    },
}


def entry(m, n, k, b=1, **kw):
    return dict(ENTRY, M=m, N=n, K=k, batch_count=b, **kw)


def problem_lines(g, times, sol=7, rows=1):
    out = [f"Solution selection time: {t} us" for t in times]
    out.append(f"Is supported {rows} / Total solutions: 9")
    for r in range(rows):
        out += bf.measured_lines(r, g, sol + r, 10.0 + r)
    return out


def test_parse_groups_selection_times_per_problem(tmp_path):
    g1, g2, g3 = bf.Gemm(64, 32, 16), bf.Gemm(1, 4096, 512, 8), bf.Gemm(9, 9, 9)
    lines = list(bf.BANNER)
    lines += problem_lines(g1, ["12.5"])
    lines += problem_lines(g2, ["3.0", "4.5"], rows=3)
    lines += problem_lines(g3, ["1e1"])
    lines += ["Solution selection time: 99 us"]
    log = tmp_path / "bench.log"
    log.write_text("\n".join(lines) + "\n")
    parsed = s7.parse_bench_log(log)
    assert [c.key for c in parsed.calls] == [
        (64, 32, 16, 1, "T", "N"),
        (1, 4096, 512, 8, "T", "N"),
        (9, 9, 9, 1, "T", "N"),
    ]
    assert [c.sel_us for c in parsed.calls] == [12.5, 7.5, 10.0]
    assert [c.n_lines for c in parsed.calls] == [1, 2, 1]
    assert [c.supported for c in parsed.calls] == [1, 3, 1]
    assert parsed.unattributed_sel_lines == 1


def test_parse_diag_and_pick_lines(tmp_path):
    log = tmp_path / "bench.log"
    log.write_text(
        "[TILEWRIGHT_DIAG FILE] path=/x/m.bin arch=gfx950 qhash=abc qdim=55 idim=12 "
        "xdim=37 n_cells=3 n_splits=1 weights=int8\n"
        "[TILEWRIGHT_DIAG FAIL] /x/bad.bin: CRC mismatch\n"
        "[TILEWRIGHT_PICK] m=1 n=2 k=3 b=4 tA=T tB=N leaf=Tiny|Tiny|TinyK|Bany "
        "top1_sig=(mt_m=64,mt_n=64,mt_k=64,mi_m=16,mi_n=16,mi_k=128,cha=0,chb=4) "
        "top1_score=nan top1_index=17 n_configs=243\n"
        "[TILEWRIGHT_PICK] m=1 n=2 k=3 b=4 tA=T tB=N leaf=Tiny|Tiny|TinyK|Bany "
        "top1_sig=(mt_m=64,mt_n=64,mt_k=64,mi_m=16,mi_n=16,mi_k=128,cha=0,chb=4) "
        "top1_score=1.5 n_configs=243\n"
    )
    p = s7.parse_bench_log(log)
    assert p.diag_files[0]["path"] == "/x/m.bin" and p.diag_files[0]["n_cells"] == "3"
    assert p.diag_fails == ["/x/bad.bin: CRC mismatch"]
    (pick,) = p.picks
    assert pick["key"] == (1, 2, 3, 4, "T", "N")
    assert pick["top1_sig"] == (64, 64, 64, 16, 16, 128, 0, 4)
    assert pick["top1_index"] == 17
    assert math.isnan(pick["top1_score"]) and pick["n_configs"] == 243
    assert (
        len(p.unparsed_picks) == 1 and "top1_score=1.5 n_configs" in p.unparsed_picks[0]
    )


PICK_PROBE = """
import sys
sys.path[:0] = [{root!r}, {tests!r}]
import tilewright as tw
from lib import evaluate as ev, features as fs
from test_evaluate import CELLS, HW, TWINS, gemm, model_bytes
model = tw.load_model_from_memory(model_bytes(CELLS[:1])[1])
g = gemm(2048, 2048, 2048, [dict(TWINS[0], us=1.0)])
problem = ev.make_problem(tw, fs.problem_kwargs_from_row(g))
ranked = ev.candidate_set(tw, model, TWINS).rank(problem, ev.make_hardware(tw, HW), 1)
print(ranked[0].config_index, ranked[0].score)
"""


def test_the_engine_pick_log_parses(tmp_path):
    pytest.importorskip("tilewright")
    tests = Path(__file__).resolve().parent
    code = PICK_PROBE.format(root=str(tests.parent), tests=str(tests))
    env = dict(os.environ, TILEWRIGHT_PICK_LOG="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("TILEWRIGHT_FORCE_CELL", None)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    index, score = proc.stdout.split()
    log = tmp_path / "bench.log"
    log.write_text(proc.stderr)
    parsed = s7.parse_bench_log(log)
    assert parsed.unparsed_picks == []
    (pick,) = parsed.picks
    assert pick["key"] == (2048, 2048, 2048, 1, "T", "N")
    assert pick["top1_index"] == int(index) and pick["n_configs"] == len(TWINS)
    assert pick["cell"] == "Large|Large|LargeK|Bnone"
    assert pick["top1_score"] == pytest.approx(float(score), abs=1e-5)


def test_check_diag(tmp_path):
    model = tmp_path / "lib" / "m.tilewright.bin"
    model.parent.mkdir()
    model.write_bytes(b"x")
    link = tmp_path / "alias"
    link.symlink_to(model.parent)
    rec = {
        "n_cells": 3,
        "n_splits": 1,
        "feature_catalog_hash": "abc",
        "weight_dtype": "int8",
        "arch": "gfx950",
    }
    line = {
        "path": str(link / "m.tilewright.bin"),
        "arch": "gfx950",
        "qhash": "abc",
        "n_cells": "3",
        "n_splits": "1",
        "weights": "int8",
    }
    log = s7.BenchLog(diag_files=[line])
    assert s7.check_diag(log, model, rec) == (True, [])
    ok, problems = s7.check_diag(
        s7.BenchLog(diag_files=[dict(line, n_cells="4")]), model, rec
    )
    assert not ok and "n_cells=4" in problems[0]
    ok, problems = s7.check_diag(s7.BenchLog(), model, rec)
    assert not ok and "never loaded" in problems[0]
    ok, problems = s7.check_diag(
        s7.BenchLog(diag_files=[dict(line, path="/elsewhere/m.bin")]), model, rec
    )
    assert not ok and "/elsewhere/m.bin" in problems[0]


def _cpp(m, sig, index, cell="c", n=10):
    return {
        "key": (m, 1, 1, 1, "T", "N"),
        "cell": cell,
        "top1_sig": sig,
        "top1_score": 1.0,
        "top1_index": index,
        "n_configs": n,
    }


def _py(m, sig=None, index=None, cell="c", reason="ok", n=10):
    d = {
        "m": m,
        "n": 1,
        "k": 1,
        "batch": 1,
        "transA": "T",
        "transB": "N",
        "cell": cell,
        "reason": reason,
        "n_configs": n,
    }
    if sig is not None:
        d["top1_sig"], d["top1_score"], d["top1_index"] = sig, 1.0, index
    return d


def test_compare_picks_categories():
    a, b = (1,) * 8, (2,) * 8
    problems = [entry(m, 1, 1) for m in range(1, 10)] + [entry(1, 1, 1)]
    cpp = [
        _cpp(1, a, 4),
        _cpp(2, a, 4),
        _cpp(3, a, 4, cell="other"),
        _cpp(5, a, 4),
        _cpp(6, a, 4, n=12),
        _cpp(6, b, 7),
        _cpp(8, a, 4),
        _cpp(9, a, 4),
    ]
    py = [
        _py(1, a, 4),
        _py(2, b, 7),
        _py(3, a, 4),
        _py(4, a, 4),
        _py(5, reason="no_survivors"),
        _py(6, a, 4),
        _py(7, reason="no_model", cell=None),
        _py(8, a, 5),
        _py(9, b, 4),
    ]
    r = s7.compare_picks(problems, cpp, py)
    assert [row["result"] for row in r["per_problem"]] == [
        "match",
        "pick_mismatch",
        "cell_mismatch",
        "no_cpp_pick",
        "no_py_pick",
        "match",
        "agree_unscored",
        "pick_mismatch",
        "match",
    ]
    assert r["n_problems"] == 9 and r["n_agree"] == 4
    assert r["match_rate"] == pytest.approx(4 / 9)
    assert r["counts"]["pool_size_differs"] == 1
    assert r["counts"]["pick_mismatch_same_sig"] == 1
    assert r["counts"]["same_index_other_sig"] == 1
    assert r["n_inconsistent_cpp_picks"] == 1
    assert r["library_pool_size"] == 10
    rows = {row["m"]: row for row in r["per_problem"]}
    assert (rows[8]["cpp_index"], rows[8]["py_index"]) == (4, 5)
    assert rows[8]["cpp_sig"] == rows[8]["py_sig"] == list(a)
    assert rows[9]["cpp_sig"] != rows[9]["py_sig"]


def _log(times):
    calls = [
        s7.Call(sel_us=t, n_lines=1, key=(i, 1, 1, 1, "T", "N"))
        for i, t in enumerate(times)
    ]
    return s7.BenchLog(calls=calls)


def test_paired_timing():
    off = [100.0, 10.0, 20.0, 30.0]
    reps = [
        (_log([t + 2.0 for t in off]), _log(off)),
        (_log([t + 4.0 for t in off]), _log(off)),
        (_log([t + 3.0 for t in off]), _log(off)),
    ]
    st = s7.paired_timing(reps)
    assert st["repetitions"] == 3 and st["n_problems_paired"] == 3
    assert st["delta_us"]["p50"] == pytest.approx(3.0)
    assert st["delta_us"]["n"] == 3
    assert st["ml_off_us"]["p50"] == pytest.approx(20.0)
    assert st["first_call_us"] == {
        "ml_on": [102.0, 104.0, 103.0],
        "ml_off": [100.0] * 3,
    }
    short = _log(off[:3])
    st = s7.paired_timing([(_log(off), short)])
    assert st["n_unpaired"] == 1 and st["n_problems_paired"] == 2


def test_paired_timing_without_identities_pairs_by_position():
    on = s7.BenchLog(calls=[s7.Call(sel_us=t, n_lines=1) for t in (50.0, 5.0, 6.0)])
    off = s7.BenchLog(calls=[s7.Call(sel_us=t, n_lines=1) for t in (40.0, 4.0, 4.0)])
    st = s7.paired_timing([(on, off)])
    assert st["n_problems_paired"] == 2
    assert st["delta_us"]["p50"] == pytest.approx(1.5)


def test_workload_filter_and_normalization(tmp_path):
    wf = s7.workload_filter(CFG)
    assert wf["compute_type"] == "f32_r" and wf["scaleA"] == "3"
    kept = entry(64, 64, 64, adaptive="true", measure_time=1.5, rotating=16)
    other_scale = entry(64, 64, 64, scaleA=1)
    other_layout = entry(64, 64, 64, transB="T")
    assert s7._entry_matches(kept, wf)
    assert not s7._entry_matches(other_scale, wf)
    assert not s7._entry_matches(other_layout, wf)
    norm = s7.normalized_entries([kept], 8)[0]
    assert norm["requested_solution_num"] == 8
    assert norm["iters"] == 1 and norm["cold_iters"] == 0
    for k in ("adaptive", "measure_time", "rotating"):
        assert k not in norm
    path = tmp_path / "b.yaml"
    s7.write_bench_yaml([norm, norm], path)
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and all(ln.startswith("- {") for ln in lines)
    assert yaml.safe_load(path.read_text()) == [norm, norm]


def test_describe_skips_non_finite():
    d = s7.describe([1.0, float("nan"), 3.0])
    assert d["n"] == 2 and d["p50"] == 2.0
    assert s7.describe([]) == {"n": 0}


BANK_STEM = bf.LIBRARY_STEM.replace("_gfx1250-strict", "_Bank_gfx1250-strict")
BANK = [
    bf.solution(100 + i, 100 + i, mt, mi)
    for i, (mt, mi) in enumerate(
        (mt, mi)
        for mt in ((64, 128, 64), (128, 64, 64), (192, 256, 64))
        for mi in ((16, 16, 128, 1), (32, 32, 64, 1))
    )
]


def deploy_fake_build(
    tmp_path, make_bundle, kernels, table=None, bank=None, blocked=None
):
    """A fake deploy-checkout build of `kernels` (Prediction table `table`;
    with `bank`, its solutions in a bank library the config names) with the
    model co-located and a stage06 record, as stage06 and the hipBLASLt
    model build leave them; with `blocked` (kernel names), the deploy's
    config lists them and its blocklist sidecar sits next to the model."""
    import stage06_deploy_weights as s6

    build = fh.write_fake_build(
        tmp_path / "build", "gfx1250-strict", bf.LIBRARY_STEM, kernels, table
    )
    if bank:
        lib = build / "Tensile" / "library" / "gfx1250-strict"
        fh.write_bank_library(lib, BANK_STEM, bank)
    train = tmp_path / "stage05"
    sigs = []
    for s in fh.kernels_grid():
        sm = s["sizeMapping"]
        sigs.append(
            [sm["macroTile"][0], sm["macroTile"][1], sm["depthU"]]
            + list(sm["matrixInstruction"][:3])
            + [0, 0]
        )
    labels = ["Large|Large|LargeK|Bnone", "Mid|Mid|MidK|Bnone"]
    make_bundle(train, labels, signatures={lab: sigs for lab in labels})
    cfg_path = tmp_path / "cfg.yaml"
    n_cu, lds, l2 = (int(v) for v in fh.FAKE_HW.split(","))
    cfg = dict(CFG, config_id="s7", deploy={"weight_dtype": "int8"})
    cfg["hardware"] = {"n_cu": n_cu, "lds_bytes": lds, "l2_bytes": l2}
    cfg["bench"] = {"startup_grace_s": 60, "stall_timeout_s": 60}
    cfg["stage07"] = {"timing_request_sizes": [1, 4], "timing_repetitions": 2}
    if bank:
        cfg["planner"] = {"enabled": True, "bank_library_stems": [BANK_STEM]}
    if blocked:
        lst = tmp_path / "blocked.tsv"
        lst.write_text("".join(f"C\t{n}\n" for n in blocked))
        cfg["correctness"] = {"lists": [str(lst)]}
    cfg_path.write_text(yaml.safe_dump(cfg))
    out6 = tmp_path / "stage06"
    assert (
        s6.deploy(train_dir=train, config_yaml=cfg_path, out_dir=out6, quiet=True) == 0
    )
    rec = json.loads((out6 / "deploy_record.json").read_text())
    lib = build / "Tensile" / "library" / "gfx1250-strict"
    t = rec["targets"][0]
    (lib / t["weights_file"]).write_bytes(Path(t["staged"]).read_bytes())
    (lib / "tilewright_index").write_text(f"{t['library_stem']}\t{t['weights_file']}\n")
    if t.get("blocklist"):
        side = Path(t["staged"]).parent / t["blocklist"]
        (lib / t["blocklist"]).write_bytes(side.read_bytes())
    yml = tmp_path / "work.yaml"
    yml.write_text(
        "".join(
            bf.bench_line(bf.Gemm(m, n, k), compute_type="c_f32_r")
            for m, n, k in ((1024, 2048, 4096), (256, 384, 256), (4096, 4096, 1024))
        )
        + bf.bench_line(bf.Gemm(64, 64, 64, transB="T"))
    )
    return {
        "build": build,
        "cfg": cfg_path,
        "record": out6 / "deploy_record.json",
        "yaml": yml,
    }


@pytest.fixture
def deployed(tmp_path, make_bundle):
    return deploy_fake_build(tmp_path, make_bundle, fh.kernels_grid())


@pytest.fixture
def deployed_twins(tmp_path, make_bundle):
    return deploy_fake_build(tmp_path, make_bundle, fh.kernels_grid(twins=True))


@pytest.fixture
def deployed_bank(tmp_path, make_bundle, monkeypatch):
    monkeypatch.setenv("FAKE_BANK_STEMS", BANK_STEM)
    return deploy_fake_build(tmp_path, make_bundle, fh.kernels_grid(), bank=BANK)


@pytest.fixture
def deployed_table(tmp_path, make_bundle):
    """A library whose file also holds two solutions of a grid-based row, in
    front of the Prediction ones, and whose Prediction table reverses the
    file order."""
    extra = [bf.solution(100 + i, i, (512, 512, 64)) for i in range(2)]
    grid = fh.kernels_grid()
    table = [s["index"] for s in reversed(grid)]
    return deploy_fake_build(tmp_path, make_bundle, extra + grid, table)


def run_stage07(deployed, out, monkeypatch, *extra):
    monkeypatch.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    return s7.main(
        [
            "--config-yaml",
            str(deployed["cfg"]),
            "--build-dir",
            str(deployed["build"]),
            "--deploy-record",
            str(deployed["record"]),
            "--bench-yaml",
            str(deployed["yaml"]),
            "--out-dir",
            str(out),
            "--quiet",
            *extra,
        ]
    )


def test_bench_runs_keep_the_variables_the_evaluation_drops(tmp_path, monkeypatch):
    from lib import evaluate as ev

    for name in ev.ENGINE_DEBUG_ENV:
        monkeypatch.setenv(name, "1")
    assert ev.drop_engine_debug_env() == list(ev.ENGINE_DEBUG_ENV)
    ctx = s7.Context(
        cfg={},
        record={},
        bench=tmp_path / "hipblaslt-bench",
        library_dir=tmp_path / "library",
        stem=bf.LIBRARY_STEM,
        weights_path=tmp_path / "model.bin",
        device=0,
        out_dir=tmp_path,
        startup_grace_s=1.0,
        stall_timeout_s=1.0,
        quiet=True,
    )
    on = {
        "TENSILE_USE_TILEWRIGHT": "1",
        "TILEWRIGHT_DIAG": "1",
        "TILEWRIGHT_PICK_LOG": "1",
    }
    parity = s7._env(ctx, on, timing=False)
    assert {k: parity[k] for k in on} == on
    assert "TILEWRIGHT_FORCE_CELL" not in parity
    assert parity["HIPBLASLT_TENSILE_LIBPATH"] == str(tmp_path / "library")
    off = s7._env(ctx, None, timing=True)
    assert not [k for k in off if k.startswith("TILEWRIGHT_") or k in on]


def test_ml_on_runs_serve_the_bank_of_the_config(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSILE_TILEWRIGHT_BANK", "1")
    ctx = s7.Context(
        cfg={},
        record={},
        bench=tmp_path / "hipblaslt-bench",
        library_dir=tmp_path / "library",
        stem=bf.LIBRARY_STEM,
        weights_path=tmp_path / "model.bin",
        device=0,
        out_dir=tmp_path,
        startup_grace_s=1.0,
        stall_timeout_s=1.0,
        quiet=True,
    )
    on = {"TENSILE_USE_TILEWRIGHT": "1"}
    assert s7._env(ctx, on, timing=False)["TENSILE_TILEWRIGHT_BANK"] == "0"
    ctx.serve_bank = True
    assert s7._env(ctx, on, timing=True)["TENSILE_TILEWRIGHT_BANK"] == "1"
    assert "TENSILE_TILEWRIGHT_BANK" not in s7._env(ctx, None, timing=True)


def test_the_fake_runtime_serves_the_bank_as_the_adapter_decides(tmp_path, monkeypatch):
    from test_v3 import SAMPLE_SPEC, v3_bundle, write

    reads = tmp_path / "reads.bin"
    reads.write_bytes(write(v3_bundle(spec=SAMPLE_SPEC)))
    plain = tmp_path / "plain.bin"
    plain.write_bytes(write(v3_bundle(spec=())))
    for var in ("TENSILE_USE_TILEWRIGHT", "TENSILE_TILEWRIGHT_BANK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("FAKE_BANK_UNSERVED", raising=False)
    assert fh.bank_served(None)
    monkeypatch.setenv("TENSILE_USE_TILEWRIGHT", "1")
    assert fh.bank_served(reads) and not fh.bank_served(plain)
    monkeypatch.setenv("TENSILE_TILEWRIGHT_BANK", "0")
    assert not fh.bank_served(reads)
    monkeypatch.setenv("TENSILE_TILEWRIGHT_BANK", "1")
    assert fh.bank_served(plain)
    monkeypatch.setenv("FAKE_BANK_UNSERVED", "1")
    assert not fh.bank_served(plain)


@pytest.mark.parametrize(
    "fixture, shell_bank, stated_bank",
    [("deployed", "1", "0"), ("deployed_bank", "0", "1")],
)
def test_bench_runs_ignore_the_callers_tilewright_settings(
    tmp_path, monkeypatch, request, fixture, shell_bank, stated_bank
):
    pytest.importorskip("tilewright")
    build = request.getfixturevalue(fixture)
    monkeypatch.setenv("TENSILE_TILEWRIGHT_BANK", shell_bank)
    monkeypatch.setenv("TENSILE_TILEWRIGHT_PREPARE", "0")
    out = tmp_path / "s7"
    envs = {}
    real = s7.run_bench

    def spy(bench, yaml_path, log_path, device, env, **kw):
        envs[log_path.relative_to(out).as_posix()] = env
        return real(bench, yaml_path, log_path, device, env, **kw)

    monkeypatch.setattr(s7, "run_bench", spy)
    assert run_stage07(build, out, monkeypatch) == 0
    ml_on = {p for p, e in envs.items() if e.get("TENSILE_USE_TILEWRIGHT") == "1"}
    assert {"work/parity/bench.log", "work/timing/rsn1/rep0_ml_on.log"} <= ml_on
    assert "work/timing/rsn1/rep0_ml_off.log" in envs
    for path, env in envs.items():
        assert "TENSILE_TILEWRIGHT_PREPARE" not in env, path
        want = stated_bank if path in ml_on else None
        assert env.get("TENSILE_TILEWRIGHT_BANK") == want, path


def test_timing_with_fake_bench(tmp_path, deployed, monkeypatch):
    monkeypatch.setenv("TENSILE_USE_TILEWRIGHT", "1")
    out = tmp_path / "s7"
    assert run_stage07(deployed, out, monkeypatch, "--no-parity") == 0
    summ = json.loads((out / "summary.json").read_text())
    assert summ["ok"] and summ["failures"] == []
    work = summ["per_yaml"]["work"]
    assert work["n_entries"] == 4 and work["n_kept"] == 3
    for rsn in ("rsn1", "rsn4"):
        t = work["timing"][rsn]
        assert t["repetitions"] == 2 and t["n_problems_paired"] == 2
        assert t["delta_us"]["p50"] == pytest.approx(3.0)
    logs = sorted(p.name for p in (out / "work" / "timing" / "rsn1").glob("*.log"))
    assert logs == [
        "rep0_ml_off.log",
        "rep0_ml_on.log",
        "rep1_ml_off.log",
        "rep1_ml_on.log",
    ]
    assert (
        "TILEWRIGHT_DIAG"
        not in (out / "work" / "timing" / "rsn1" / "rep0_ml_off.log").read_text()
    )
    norm = yaml.safe_load((out / "work" / "timing" / "rsn4" / "bench.yaml").read_text())
    assert {e["requested_solution_num"] for e in norm} == {4}
    assert all("print_kernel_info" not in e for e in norm)


def test_the_summary_records_the_platform_of_the_timing(
    tmp_path, deployed, monkeypatch
):
    gf.write_cpus(tmp_path / "sys", n=4)
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(tmp_path / "sys"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0})
    monkeypatch.setenv("TENSILE_FIXED_WGM", "4")
    out = tmp_path / "s7"
    assert run_stage07(deployed, out, monkeypatch, "--no-parity") == 0
    summ = json.loads((out / "summary.json").read_text())
    platform = summ["platform"]
    assert platform == json.loads((out / "platform.json").read_text())
    assert [g["device"] for g in platform["gpus"]] == [0]
    launch = platform["launch"]
    assert set(launch) == {"overrides", "stripped", "heuristic", "workspace_bytes"}
    assert launch["overrides"] == {} and launch["stripped"] == ["TENSILE_FIXED_WGM"]
    cpu = platform["cpu"]
    assert (cpu["affinity"], cpu["pinned"]) == ("0", True)
    assert cpu["smt_siblings_outside_affinity"] == "2"
    timing_cpu = summ["per_yaml"]["work"]["timing_cpu"]
    assert timing_cpu["cpus"] == "2"
    assert set(timing_cpu) == {
        "cpus",
        "busy_max",
        "busy_mean",
        "loadavg_start",
        "loadavg_end",
    }


def test_stage07_validates_a_bundle_with_its_own_blocklist(
    tmp_path, make_bundle, monkeypatch, capsys
):
    """The runtime leaves out the kernels of the bundle's sidecar, whatever
    the validating invocation's blocklist says: parity counts the pool as
    the runtime does, the difference is warned about, and a library
    directory without the bundle's sidecar fails."""
    from lib import correctness as cx

    monkeypatch.setenv("TENSILE_USE_TILEWRIGHT", "1")
    blocked = bf.kernel_name((256, 256, 64))
    d = deploy_fake_build(tmp_path, make_bundle, fh.kernels_grid(), blocked=[blocked])
    lib = d["build"] / "Tensile" / "library" / "gfx1250-strict"
    side = lib / cx.sidecar_name(bf.LIBRARY_STEM)
    assert cx.read_sidecar(side) == [blocked]
    later = yaml.safe_load(d["cfg"].read_text())
    later.pop("correctness")
    d["cfg"] = tmp_path / "later.yaml"
    d["cfg"].write_text(yaml.safe_dump(later))
    out = tmp_path / "s7"
    assert run_stage07(d, out, monkeypatch, "--no-timing") == 0
    summ = json.loads((out / "summary.json").read_text())
    assert summ["blocklist"]["bundle"]["kernels"] == 1
    assert summ["blocklist"]["served"] == summ["blocklist"]["bundle"]
    assert summ["blocklist"]["invocation"] == {"digest": None, "kernels": 0}
    parity = summ["per_yaml"]["work"]["parity"]
    assert parity["runtime_pool_sizes"] == [len(fh.kernels_grid()) - 2]
    assert "the bundle's sidecar 1" in capsys.readouterr().err
    side.unlink()
    assert run_stage07(d, tmp_path / "s7b", monkeypatch, "--no-timing") == 1
    summ = json.loads((tmp_path / "s7b" / "summary.json").read_text())
    assert any(
        "the deployed bundle's blocklist sidecar 1" in f for f in summ["failures"]
    )


def test_a_refused_lease_stops_stage07(tmp_path, deployed, monkeypatch):
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            f"sys.path.insert(0, {str(PIPELINE_ROOT)!r})\n"
            "from lib import gpu_lease\n"
            "with gpu_lease.hold_gpus([0], 'another run'):\n"
            "    print('held', flush=True)\n"
            "    time.sleep(60)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        out = tmp_path / "s7"
        assert run_stage07(deployed, out, monkeypatch, "--no-parity") == 2
    finally:
        holder.kill()
        holder.wait()
    summ = json.loads((out / "summary.json").read_text())
    assert summ["failures"][0].startswith("refused (one process tree per GPU): GPU 0")
    assert summ["per_yaml"] == {}


def test_a_record_without_the_arch_is_a_setup_error(tmp_path):
    """Not GPU trouble: the orchestrator retries a stage07 exit 2 only."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(CFG))
    record = tmp_path / "deploy_record.json"
    record.write_text(json.dumps({"targets": [{"dir": "gfx950"}]}))
    argv = ["--config-yaml", str(cfg), "--build-dir", str(tmp_path / "build")]
    argv += ["--deploy-record", str(record), "--bench-yaml", str(tmp_path / "w.yaml")]
    assert s7.main(argv + ["--out-dir", str(tmp_path / "s7"), "--quiet"]) == 1


def test_stale_colocated_model_fails(tmp_path, deployed, monkeypatch):
    rec = json.loads(deployed["record"].read_text())
    lib = deployed["build"] / "Tensile" / "library" / "gfx1250-strict"
    (lib / rec["targets"][0]["weights_file"]).write_bytes(b"older model")
    out = tmp_path / "s7"
    assert run_stage07(deployed, out, monkeypatch) == 1
    summ = json.loads((out / "summary.json").read_text())
    assert "not the deployed model" in summ["failures"][0]


def test_parity_with_fake_bench(tmp_path, deployed, monkeypatch):
    pytest.importorskip("tilewright")
    out = tmp_path / "s7"
    assert run_stage07(deployed, out, monkeypatch, "--no-timing") == 0
    par = json.loads((out / "work" / "parity" / "parity.json").read_text())
    assert par["diag_ok"] and par["match_rate"] == 1.0
    assert par["counts"] == {"match": 3}
    assert {r["cpp_cell"] for r in par["per_problem"]} == {
        "Large|Large|LargeK|Bnone",
        "Mid|Mid|MidK|Bnone",
    }


def test_parity_detects_a_broken_integration(tmp_path, deployed, monkeypatch):
    pytest.importorskip("tilewright")
    monkeypatch.setenv("FAKE_PICK_RANK", "1")
    out = tmp_path / "s7"
    assert run_stage07(deployed, out, monkeypatch, "--no-timing") == 1
    par = json.loads((out / "work" / "parity" / "parity.json").read_text())
    assert par["counts"].get("pick_mismatch", 0) >= 1 and not par["ok"]
    summ = json.loads((out / "summary.json").read_text())
    assert any("pick parity" in f for f in summ["failures"])


def test_parity_with_duplicate_kernels(tmp_path, deployed_twins, monkeypatch):
    pytest.importorskip("tilewright")
    out = tmp_path / "s7"
    assert run_stage07(deployed_twins, out, monkeypatch, "--no-timing") == 0
    par = json.loads((out / "work" / "parity" / "parity.json").read_text())
    assert par["counts"] == {"match": 3} and par["library_pool_size"] == 20
    for row in par["per_problem"]:
        assert row["cpp_index"] == row["py_index"] < 10


def test_parity_serves_the_bank_and_names_a_runtime_without_it(
    tmp_path, deployed_bank, monkeypatch
):
    pytest.importorskip("tilewright")
    out = tmp_path / "s7"
    assert run_stage07(deployed_bank, out, monkeypatch, "--no-timing") == 0
    par = json.loads((out / "work" / "parity" / "parity.json").read_text())
    pool = len(fh.kernels_grid()) + len(BANK)
    assert par["runtime_pool_sizes"] == [pool] and par["library_pool_size"] == pool
    assert par["ok"] and "pool_size_differs" not in par["counts"]
    assert json.loads((out / "summary.json").read_text())["bank_served"] is True
    monkeypatch.setenv("FAKE_BANK_UNSERVED", "1")
    out = tmp_path / "s7_unserved"
    assert run_stage07(deployed_bank, out, monkeypatch, "--no-timing") == 1
    par = json.loads((out / "work" / "parity" / "parity.json").read_text())
    assert par["runtime_pool_sizes"] == [len(fh.kernels_grid())] and not par["ok"]
    summ = json.loads((out / "summary.json").read_text())
    assert (
        f"work: the runtime ranks {len(fh.kernels_grid())} kernels with "
        f"TENSILE_TILEWRIGHT_BANK=1, the model's pool has {pool} (the library's "
        "Prediction table and the bank): the bank is not served" in summ["failures"][0]
    )


def test_cpus_pin_the_stage_before_the_platform_is_recorded(
    tmp_path, deployed, monkeypatch
):
    gf.write_cpus(tmp_path / "sys", n=4)
    monkeypatch.setenv(kfd.SYSFS_ROOT_VAR, str(tmp_path / "sys"))
    affinity = {0, 1, 2, 3}
    pinned = []

    def pin(pid, cpus):
        pinned.append(sorted(cpus))
        affinity.clear()
        affinity.update(cpus)

    monkeypatch.setattr(os, "sched_setaffinity", pin)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(affinity))
    out = tmp_path / "s7"
    assert run_stage07(deployed, out, monkeypatch, "--no-parity", "--cpus", "1-2") == 0
    assert pinned == [[1, 2]]
    cpu = json.loads((out / "summary.json").read_text())["platform"]["cpu"]
    assert (cpu["affinity"], cpu["pinned"]) == ("1-2", True)
    with pytest.raises(SystemExit):
        run_stage07(deployed, tmp_path / "s7_bad", monkeypatch, "--cpus", "two")


def test_parity_ranks_the_prediction_table(tmp_path, deployed_table, monkeypatch):
    pytest.importorskip("tilewright")
    out = tmp_path / "s7"
    assert run_stage07(deployed_table, out, monkeypatch, "--no-timing") == 0
    par = json.loads((out / "work" / "parity" / "parity.json").read_text())
    assert par["counts"] == {"match": 3} and par["library_pool_size"] == 10


def test_parity_tells_duplicate_kernels_apart(tmp_path, deployed_twins, monkeypatch):
    pytest.importorskip("tilewright")
    monkeypatch.setenv("FAKE_PICK_TIE", "last")
    out = tmp_path / "s7"
    assert run_stage07(deployed_twins, out, monkeypatch, "--no-timing") == 1
    par = json.loads((out / "work" / "parity" / "parity.json").read_text())
    assert par["counts"] == {"pick_mismatch": 3, "pick_mismatch_same_sig": 3}
    for row in par["per_problem"]:
        assert row["cpp_index"] == row["py_index"] + 10
        assert row["cpp_sig"] == row["py_sig"]
    summ = json.loads((out / "summary.json").read_text())
    assert any("pick parity" in f for f in summ["failures"])


def test_pick_lines_without_the_pool_position_fail(tmp_path, deployed, monkeypatch):
    pytest.importorskip("tilewright")
    monkeypatch.setenv("FAKE_PICK_NO_INDEX", "1")
    out = tmp_path / "s7"
    assert run_stage07(deployed, out, monkeypatch, "--no-timing") == 1
    par = json.loads((out / "work" / "parity" / "parity.json").read_text())
    assert par["n_unparsed_pick_lines"] == 3 and not par["ok"]
    assert par["n_cpp_picks"] == 0
    summ = json.loads((out / "summary.json").read_text())
    assert any("top1_index" in f for f in summ["failures"])


def test_bench_yamls_with_the_same_name_keep_separate_results():
    paths = [Path("a/w.yaml"), Path("b/w.yaml"), Path("x.yaml"), Path("c/w.yaml")]
    assert s7.yaml_tags(paths) == ["w", "w_2", "x", "w_3"]
    assert s7.yaml_tags([Path("w_2.yaml"), Path("w.yaml"), Path("b/w.yaml")]) == [
        "w_2",
        "w",
        "w_3",
    ]


def _rank_row(m, ranked, reason="ok"):
    return dict(
        _py(m, (1,) * 8, ranked[0] if ranked else None, reason=reason), ranked=ranked
    )


def test_compare_orders_categories():
    problems = [entry(m, 1, 1) for m in range(1, 9)]
    position_of = {100 + i: i for i in range(10)}
    sids = {
        1: [100, 101, 102],
        2: [100, 102, 103],
        3: [100, 102, 103],
        4: [101, 100, 102],
        5: [100, 107, 101],
        6: [109],
        7: [100, 101, 108],
    }
    runtime = {(m, 1, 1, 1, "T", "N"): v for m, v in sids.items()}
    py = [_rank_row(m, [0, 1, 2, 3]) for m in range(1, 6)]
    py += [_rank_row(6, [], reason="no_model"), _rank_row(7, [0, 1, 2, 3])]
    py += [_rank_row(8, [0, 1])]
    measured = {(3, 1, 1, 1, "T", "N"): {101}}
    r = s7.compare_orders(problems, runtime, py, position_of, measured)
    assert [row["result"] for row in r["per_problem"]] == [
        "match",
        "predicate_skips",
        "order_mismatch",
        "order_mismatch",
        "order_mismatch",
        "agree_unscored",
        "match",
        "no_runtime_list",
    ]
    rows = {row["m"]: row for row in r["per_problem"]}
    assert rows[2]["left_out"] == [1] and rows[7]["n_filled"] == 1
    assert r["n_agree"] == 4 and r["match_rate"] == pytest.approx(4 / 8)


def test_runtime_selection_joins_the_picks_to_bench_timings():
    pool = [{"sol_idx_global": 100 + i} for i in range(4)]
    problems = [entry(m, 1, 1) for m in range(1, 7)]
    cands = [
        {"sol_idx_global": 100, "us": 10.0, "is_skip": False},
        {"sol_idx_global": 101, "us": 20.0, "is_skip": False},
        {"sol_idx_global": 102, "us": 40.0, "is_skip": True},
    ]
    timings = {
        (m, 1, 1, 1, "T", "N"): {"candidates": cands, "winner_us": 10.0}
        for m in range(1, 6)
    }
    cpp = [_cpp(1, (1,) * 8, 0), _cpp(2, (1,) * 8, 1), _cpp(3, (1,) * 8, 2)]
    cpp += [_cpp(4, (1,) * 8, 3), _cpp(5, (1,) * 8, 7), _cpp(6, (1,) * 8, 0)]
    got = s7.runtime_selection(problems, cpp, pool, timings, "timed", bootstrap=50)
    assert got["n_evaluated"] == 3 and got["n_miss"] == 1
    assert got["sel_eff"] == pytest.approx((1.0 * 0.5 * 0.25) ** (1 / 3))
    assert (got["n_top1_without_row"], got["n_outside_the_pool"]) == (1, 1)
    assert got["n_without_timings"] == 1 and got["sel_eff_ci95"] is not None
    warm = s7.runtime_selection(problems, cpp, pool, timings, "any")
    assert warm["n_miss"] == 0
    assert warm["sel_eff"] == pytest.approx((1.0 * 0.5 * 0.25) ** (1 / 3))


def test_paired_selection_pairs_the_runtime_with_origami_on_the_same_problems():
    sig = dict(mt_m=128, mt_n=128, mt_k=64, mi_m=16, mi_n=16, mi_k=32)
    pool = [
        dict(sig, sol_idx_global=100 + i, cache_hints_a=0, cache_hints_b=i)
        for i in range(4)
    ]
    problems = [entry(m, 1, 1) for m in range(1, 7)]
    cands = [
        {"sol_idx_global": 100, "us": 10.0, "is_skip": False},
        {"sol_idx_global": 101, "us": 20.0, "is_skip": False},
        {"sol_idx_global": 102, "us": 40.0, "is_skip": True},
    ]
    timings = {
        (m, 1, 1, 1, "T", "N"): {"candidates": cands, "winner_us": 10.0}
        for m in (1, 2, 3, 4, 6)
    }
    cpp = [_cpp(m, (1,) * 8, i) for m, i in ((1, 0), (2, 1), (3, 0), (4, 3), (6, 0))]
    origami = {(m, 1, 1, 1, "T", "N"): s for m, s in ((1, 101), (2, 100), (3, 102))}
    origami.update({(m, 1, 1, 1, "T", "N"): 100 for m in (4, 5)})
    got = s7.paired_selection(problems, cpp, origami, pool, timings, "timed", 50)
    p = got["paired"]
    assert p["n"] == 3 and (p["wins"], p["ties"], p["losses"]) == (2, 0, 1)
    assert p["model_sel_eff"] == pytest.approx(0.5 ** (1 / 3))
    assert p["origami_sel_eff"] == pytest.approx(0.125 ** (1 / 3))
    assert "speedup_ci95" in p
    assert (got["n_problems"], got["n_model_evaluated"], got["n_origami"]) == (6, 3, 4)
    assert got["n_without_timings"] == 1 and got["n_without_origami_pick"] == 1
    assert got["n_origami_miss"] == 1 and got["n_origami_without_row"] == 0
    rows = {r["m"]: r for r in got["per_problem"]}
    assert rows[1]["model_signature"] == "MT128x128x64_MI16x16x32_NTA0_NTB0"
    assert rows[1]["origami_signature"] == "MT128x128x64_MI16x16x32_NTA0_NTB1"
    assert rows[4]["model_sel_eff"] is None and rows[4]["origami_sel_eff"] == 1.0
    assert "sol_idx_global" not in json.dumps(got)


def test_paired_timing_tells_first_calls_into_a_cell_apart():
    off = [100.0, 10.0, 20.0, 30.0, 40.0]
    on = [101.0, 50.0, 12.0, 31.0, 41.0]
    cells = {(i, 1, 1, 1, "T", "N"): c for i, c in enumerate("AABAB")}
    st = s7.paired_timing([(_log(on), _log(off))] * 3, cell_of=cells, bootstrap=20)
    assert st["first_in_cell"]["n_problems"] == 1
    assert st["first_in_cell"]["ml_on_us"]["max"] == 12.0
    assert st["warm"]["n_problems"] == 3
    assert st["warm"]["delta_us"]["p50"] == pytest.approx(1.0)
    assert st["delta_us"]["p50_ci95"] is not None
    assert "first_in_cell" not in s7.paired_timing([(_log(on), _log(off))])


def write_timings(path, deployed_problems):
    lines = ["m,n,k,batch_count,transA,transB,sol_idx_global,us,is_skip"]
    for m, n, k in deployed_problems:
        for s in fh.kernels_grid():
            us = fh.kernel_time_us(m, n, k, 1, s["sizeMapping"])
            lines.append(f"{m},{n},{k},1,T,N,{s['index']},{us},False")
    path.mkdir(parents=True, exist_ok=True)
    (path / "chunk_0000.csv").write_text("\n".join(lines) + "\n")
    return path


KEPT = ((1024, 2048, 4096), (256, 384, 256), (4096, 4096, 1024))


def test_order_parity_and_runtime_selection(tmp_path, deployed, monkeypatch):
    pytest.importorskip("tilewright")
    out = tmp_path / "s7"
    timings = write_timings(tmp_path / "timings", KEPT)
    rc = run_stage07(
        deployed, out, monkeypatch, "--no-timing", "--timings-csv-dir", str(timings)
    )
    assert rc == 0
    par = json.loads((out / "work" / "parity_rsn8" / "parity.json").read_text())
    assert par["ok"] and par["requested_solution_num"] == 8
    assert par["top1"]["counts"] == {"match": 3}
    assert par["order"]["counts"] == {"match": 3}
    bench = yaml.safe_load((out / "work" / "parity_rsn8" / "bench.yaml").read_text())
    assert {e["requested_solution_num"] for e in bench} == {8}
    assert all(e["print_kernel_info"] == 1 for e in bench)
    summ = json.loads((out / "summary.json").read_text())
    sel = summ["per_yaml"]["work"]["parity"]["runtime_selection"]
    assert sel["n_evaluated"] == 3 and sel["n_miss"] == 0
    assert 0 < sel["sel_eff"] <= 1.0
    assert summ["per_yaml"]["work"]["parity_top_k"]["ok"]


def test_origami_picks_pair_with_the_runtime_on_held_out_problems(
    tmp_path, deployed, monkeypatch, capsys
):
    pytest.importorskip("tilewright")
    timings = write_timings(tmp_path / "timings", KEPT)
    held = tmp_path / "audit_sample.yaml"
    held.write_text(deployed["yaml"].read_text())
    argv = ["--no-timing", "--timings-csv-dir", str(timings)]
    argv += ["--held-out-yaml", str(held)]
    out = tmp_path / "plain"
    assert run_stage07(deployed, out, monkeypatch, *argv) == 0
    summ = json.loads((out / "summary.json").read_text())
    assert summ["origami_reference"] is False
    assert not summ["per_yaml"]["work"]["held_out"]
    assert summ["per_yaml"]["audit_sample"]["held_out"]
    assert "origami_reference" not in summ["per_yaml"]["audit_sample"]
    assert not list(out.rglob("origami"))
    assert "Origami" not in capsys.readouterr().out

    out = tmp_path / "reference"
    assert run_stage07(deployed, out, monkeypatch, *argv, "--origami-reference") == 0
    assert "the runtime's picks with tilewright on and off" in capsys.readouterr().out
    summ = json.loads((out / "summary.json").read_text())
    ref = summ["per_yaml"]["audit_sample"]["origami_reference"]
    assert "origami_reference" not in summ["per_yaml"]["work"]
    assert ref["n_problems"] == 3 and ref["n_without_timings"] == 0
    assert ref["paired"]["n"] == 3 and ref["n_origami_without_row"] == 0
    sel = summ["per_yaml"]["audit_sample"]["parity"]["runtime_selection"]
    assert ref["paired"]["model_sel_eff"] == pytest.approx(sel["sel_eff"])
    bench = yaml.safe_load(
        (out / "audit_sample" / "origami" / "bench.yaml").read_text()
    )
    assert {(e["requested_solution_num"], e["print_kernel_info"]) for e in bench} == {
        (1, 1)
    }
    lib = deployed["build"] / "Tensile" / "library" / "gfx1250-strict"
    sols, _ = fh._library(lib, bf.LIBRARY_STEM)
    rows = json.loads((out / "audit_sample" / "origami" / "origami.json").read_text())[
        "per_problem"
    ]
    for r in rows:
        first = sols[fh.origami_order(r["m"], r["n"], r["k"], r["batch"], sols)[0]]
        sm = first["sizeMapping"]
        assert r["origami_signature"].startswith(
            f"MT{sm['macroTile'][0]}x{sm['macroTile'][1]}x{sm['depthU']}_"
        )
        assert r["origami_sel_eff"] == pytest.approx(r["winner_us"] / r["origami_us"])


def test_order_parity_catches_a_reordered_list(tmp_path, deployed, monkeypatch):
    pytest.importorskip("tilewright")
    monkeypatch.setenv("FAKE_ORDER_SWAP", "1")
    out = tmp_path / "s7"
    assert run_stage07(deployed, out, monkeypatch, "--no-timing") == 1
    par = json.loads((out / "work" / "parity_rsn8" / "parity.json").read_text())
    assert par["order"]["counts"] == {"order_mismatch": 3}
    assert par["top1"]["counts"] == {"match": 3}
    summ = json.loads((out / "summary.json").read_text())
    assert any("order parity" in f for f in summ["failures"])


def test_order_parity_and_left_out_kernels(tmp_path, deployed, monkeypatch):
    pytest.importorskip("tilewright")
    monkeypatch.setenv("FAKE_DROP_RANK", "1")
    out = tmp_path / "s7"
    assert run_stage07(deployed, out, monkeypatch, "--no-timing") == 0
    par = json.loads((out / "work" / "parity_rsn8" / "parity.json").read_text())
    assert par["order"]["counts"] == {"predicate_skips": 3}
    timings = write_timings(tmp_path / "timings", KEPT)
    out2 = tmp_path / "s7b"
    rc = run_stage07(
        deployed, out2, monkeypatch, "--no-timing", "--timings-csv-dir", str(timings)
    )
    assert rc == 1
    par = json.loads((out2 / "work" / "parity_rsn8" / "parity.json").read_text())
    assert par["order"]["counts"] == {"order_mismatch": 3}
    out3 = tmp_path / "s7c"
    assert (
        run_stage07(deployed, out3, monkeypatch, "--no-timing", "--parity-top-k", "1")
        == 0
    )
    assert not (out3 / "work" / "parity_rsn8").exists()


def test_timing_splits_first_calls_and_prints_the_reference_on_request(
    tmp_path, deployed, monkeypatch, capsys
):
    pytest.importorskip("tilewright")
    out = tmp_path / "s7"
    monkeypatch.setenv("FAKE_LIBRARY_STEM", bf.LIBRARY_STEM)
    argv = [
        "--config-yaml",
        str(deployed["cfg"]),
        "--build-dir",
        str(deployed["build"]),
        "--deploy-record",
        str(deployed["record"]),
        "--bench-yaml",
        str(deployed["yaml"]),
        "--no-parity",
        "--bootstrap",
        "20",
    ]
    capsys.readouterr()
    assert s7.main(argv + ["--out-dir", str(out / "a")]) == 0
    plain = capsys.readouterr().out
    assert (
        "ML-on selection time per problem" in plain
        and "first call into a cell" in plain
    )
    assert "ML-off" not in plain and "origami" not in plain.lower()
    t = json.loads((out / "a" / "summary.json").read_text())["per_yaml"]["work"][
        "timing"
    ]
    for rsn in ("rsn1", "rsn4"):
        assert t[rsn]["first_in_cell"]["n_problems"] == 1
        assert t[rsn]["warm"]["n_problems"] == 1
        assert t[rsn]["ml_on_us"]["p50_ci95"] is not None
    assert s7.main(argv + ["--out-dir", str(out / "b"), "--origami-reference"]) == 0
    assert "[reference] ML-on minus ML-off" in capsys.readouterr().out


def test_timings_of_another_build_are_read_at_their_kernels(tmp_path, capsys):
    """--timings-csv-dir rows recorded as enriched with build A, read for a
    library of build B (every kernel numbered otherwise), name B's indices;
    a library whose identity cannot be read leaves them as they are, with
    one warning."""
    from lib import evaluate as ev
    from test_kernel_identity import pool_library, record_rows
    from test_stage05_train import kernel_pool, shapes_for, write_round

    stem = bf.LIBRARY_STEM
    ident_a = pool_library(tmp_path / "a", kernel_pool(), stem=stem)
    ident_b = pool_library(tmp_path / "b", kernel_pool(), offset=1000, stem=stem)
    rows = record_rows(write_round(tmp_path / "rows", shapes_for(3, 2)), ident_a, stem)

    def context(lib):
        return s7.Context(
            cfg={},
            record={},
            bench=tmp_path / "hipblaslt-bench",
            library_dir=lib,
            stem=stem,
            weights_path=tmp_path / "model.bin",
            device=0,
            out_dir=tmp_path,
            startup_grace_s=1.0,
            stall_timeout_s=1.0,
            quiet=True,
            timing_dirs=[rows],
        )

    def by_kernel(timings, ids):
        return {
            key: {ids[c["sol_idx_global"]]: c["us"] for c in g["candidates"]}
            for key, g in timings.items()
        }

    raw = ev.load_timings([rows])
    assert raw and by_kernel(
        context(tmp_path / "b").timings(list(raw)), ident_b.ids
    ) == (by_kernel(raw, ident_a.ids))
    missing = context(tmp_path / "missing")
    capsys.readouterr()
    assert missing.identity() is None and missing.identity() is None
    assert by_kernel(missing.timings(list(raw)), ident_a.ids) == by_kernel(
        raw, ident_a.ids
    )
    assert capsys.readouterr().err.count("no kernel identity of") == 1


def _quarantine(deployed, tmp_path, name):
    """The deployed config with a canary store that quarantines `name`."""
    from lib import hazards as hz

    store = tmp_path / "canary_store"
    rec = {"event": "end", "token": "t", "name": name, "arch": CFG["arch"]}
    hz.CanaryStore(store).add([dict(rec, verdict="hang")])
    cfg = yaml.safe_load(deployed["cfg"].read_text())
    cfg["hazards"] = {"cache_dir": str(store)}
    deployed["cfg"].write_text(yaml.safe_dump(cfg))


def test_with_a_quarantined_kernel_the_analytical_ranking_runs_nothing(
    tmp_path, make_bundle, monkeypatch
):
    """A kernel the hazard canary keeps out never runs among others: the
    ML-off timing, the Origami reference, the order parity and the
    production route, whose kernels the analytical ranking picks, are
    skipped, and every run is when the bundle's blocklist leaves the kernel
    in."""
    pytest.importorskip("tilewright")
    monkeypatch.setenv("TENSILE_USE_TILEWRIGHT", "1")
    held = bf.kernel_name((256, 256, 64))
    d = deploy_fake_build(tmp_path, make_bundle, fh.kernels_grid(), blocked=[held])
    _quarantine(d, tmp_path, held)
    audit = tmp_path / "audit_sample.yaml"
    audit.write_text(d["yaml"].read_text())
    timings = write_timings(tmp_path / "timings", KEPT)
    argv = ["--timings-csv-dir", str(timings), "--held-out-yaml", str(audit)]
    out = tmp_path / "s7"
    assert run_stage07(d, out, monkeypatch, *argv, "--origami-reference") == 0
    summ = json.loads((out / "summary.json").read_text())
    assert summ["hazards"] == {"kernels": 1, "left_in": 0, "skipped": "analytical"}
    analytical = {"timing", "parity_top_k", "production_route"}
    for tag, extra in (("work", set()), ("audit_sample", {"origami_reference"})):
        y = summ["per_yaml"][tag]
        assert y["parity"]["ok"] and set(y["skipped"]) == analytical | extra
        assert not (analytical | {"origami_reference"}) & set(y)
    assert not list(out.rglob("*ml_off*")) and not list(out.rglob("origami"))

    plain = deploy_fake_build(tmp_path / "plain", make_bundle, fh.kernels_grid())
    _quarantine(plain, tmp_path / "plain", held)
    out = tmp_path / "s7_plain"
    assert run_stage07(plain, out, monkeypatch) == 0
    summ = json.loads((out / "summary.json").read_text())
    assert summ["hazards"]["left_in"] == 1 and summ["hazards"]["skipped"] == "every run"
    (why,) = summ["per_yaml"]["work"]["skipped"].values()
    assert "deploy the bundle again" in why
    assert not list(out.rglob("*.log"))


# ── the production route ────────────────────────────────────────────────────

ROUTE_STEM = bf.LIBRARY_STEM.replace("_Bias_SAV_UA_", "_Bias_UA_")
ANCHOR_STEM = bf.LIBRARY_STEM.replace("_HA_MXAE8B32_MXBE8B32_Bias_SAV_UA_", "_UA_")


def _served(sid, kernel):
    from lib.bench_log import TestedRow

    return [TestedRow(rank=0, fields={}, sol_index=sid, kernel_name=kernel)]


GRID_NAME = bf.kernel_name((128, 128, 256))


def test_production_route_names_what_served_each_problem():
    problems = [entry(m, 64, 64) for m in (1, 2, 3, 4, 5, 6, 7, 8)]
    problems.append(entry(1, 64, 64))
    picks = [_cpp(1, (0,) * 8, 0), _cpp(2, (0,) * 8, 0)]
    picks = [dict(p, key=(p["key"][0], 64, 64, 1, "T", "N")) for p in picks]
    served = {
        (1, 64, 64, 1, "T", "N"): _served(7, "K7"),
        (2, 64, 64, 1, "T", "N"): _served(7, "K7"),
        (3, 64, 64, 1, "T", "N"): _served(200, GRID_NAME),
        (4, 64, 64, 1, "T", "N"): _served(200, GRID_NAME),
        (5, 64, 64, 1, "T", "N"): _served(999, "LOST"),
        # The model's row without a pick: the model scored nothing.
        (7, 64, 64, 1, "T", "N"): _served(7, "K7"),
        # The Prediction row of another file answered ahead of the model's.
        (8, 64, 64, 1, "T", "N"): _served(9, "K9"),
    }
    rows = {
        7: ("host", "Prediction"),
        9: ("other", "Prediction"),
        200: ("grid", "GridBased"),
    }
    kernels = {200: kernel_dat_info(bf.solution(200, 0, (128, 128, 256)))}
    route = s7.production_route(problems, picks, served, rows, kernels, host="host")
    assert route["n_problems"] == 8 and route["n_tilewright"] == 2
    assert route["n_unscored"] == 1 and route["n_reached"] == 3
    assert route["n_ahead"] == 3 and route["warned"]
    assert route["tilewright_share"] == pytest.approx(2 / 8)
    assert route["rows"] == {"GridBased": 2, "Prediction": 2, "none": 1, "unknown": 1}
    assert route["served_by"][0] == {
        "row": "GridBased",
        "library": "grid",
        "kernel": GRID_NAME,
        "signature": "MT128x128x256 MI16x16x128 NTA0 NTB0",
        "solution_index": 200,
        "unscored": False,
        "n": 2,
    }
    by_row = {(s["row"], s["library"]): s for s in route["served_by"]}
    assert by_row[("Prediction", "host")]["unscored"]
    assert not by_row[("Prediction", "other")]["unscored"]
    # A kernel no file gives a signature of is named by what its name spells.
    assert by_row[("unknown", None)]["signature"] == s7.label_of_name("LOST")
    first = route["per_problem"][0]
    assert first["tilewright"] and (first["row"], first["kernel"]) == (
        "Prediction",
        "K7",
    )
    # The lines name kernels by signature and leave out the model's row.
    text = s7.route_text(route, most=1)
    assert text == (
        "2 by the GridBased row of grid (MT128x128x256 MI16x16x128 NTA0 NTB0); "
        "3 more kernel(s)"
    )
    assert "of host" not in s7.route_text(route) and "no solution" in (
        s7.route_text(route)
    )
    assert GRID_NAME not in s7.route_status(dict(route, model_loaded=True))
    assert s7.production_route([], [], {}, {})["tilewright_share"] is None

    # The model's row answering every problem the model left: no warning.
    reached = s7.production_route(
        problems[:2] + problems[6:7], picks, served, rows, kernels, host="host"
    )
    assert reached["n_unscored"] == 1 and reached["n_reached"] == 3
    assert not reached["warned"] and reached["n_ahead"] == 0
    assert s7.route_status(reached) == (
        "every problem reached the model: tilewright served 2 of 3 and scored "
        "nothing for the other 1, which its Prediction row served"
    )
    # A runtime that never loaded the model filled no pick from it.
    unloaded = s7.production_route(
        problems[6:7], [], served, rows, kernels, host="host", loaded=False
    )
    assert unloaded["n_unscored"] == 0 and unloaded["warned"]
    assert unloaded["n_ahead"] == 0 and unloaded["served_by"][0]["unscored"] is False
    assert s7.route_status(dict(unloaded, model_loaded=False)).endswith(
        "; the runtime never loaded the model"
    )


def test_the_runtime_loaded_the_model_when_a_diag_line_names_its_file(tmp_path):
    model = tmp_path / "m.bin"
    log = tmp_path / "bench.log"
    log.write_text(
        f"[TILEWRIGHT_DIAG FILE] path={model} arch=gfx950 qhash=x qdim=1 idim=1 "
        f"xdim=1 n_cells=3 n_splits=0 weights=int8\n"
    )
    parsed = s7.parse_bench_log(log)
    assert s7.model_loaded(parsed, model)
    assert not s7.model_loaded(parsed, tmp_path / "other.bin")
    # Loaded, though not the deployed contents: check_diag fails it, the
    # route counts it loaded.
    ok, problems = s7.check_diag(parsed, model, {"n_cells": 4})
    assert not ok and problems and "never loaded" not in problems[0]


def test_the_route_summary_adds_the_yamls_up():
    routed = {"n_problems": 6, "n_tilewright": 3, "n_unscored": 1, "n_reached": 4}
    per_yaml = {
        "a": {"production_route": {"n_problems": 4, "n_tilewright": 4}},
        "b": {"production_route": dict(routed, n_ahead=2, warned=True)},
        "c": {"production_route": {"error": "bench failed"}},
        "d": {"skipped": {"production_route": "hazards"}},
        "e": {
            "production_route": {"n_problems": 2, "n_tilewright": 1, "n_unscored": 1},
            "skipped": {"production_route_problems": [{"m": 1}, {"m": 2}]},
        },
    }
    assert s7.route_summary(per_yaml) == {
        "n_problems": 12,
        "n_tilewright": 8,
        "n_unscored": 2,
        "n_reached": 9,
        "n_ahead": 2,
        "tilewright_share": pytest.approx(8 / 12),
        "warned_yamls": ["b"],
        "n_left_out": 2,
    }
    assert s7.route_summary({"d": per_yaml["d"]}) is None


def deploy_routed_build(tmp_path, make_bundle, anchor=False):
    """A fake build whose library file holds a Prediction table of the grid,
    beside a file of the problem type with a GridBased row of one kernel that
    sorts before it; with `anchor`, also a file with an Equality row of one
    kernel tuned for 256x384x256."""
    grid = fh.kernels_grid()
    d = deploy_fake_build(tmp_path, make_bundle, grid, [s["index"] for s in grid])
    lib = d["build"] / "Tensile" / "library" / "gfx1250-strict"
    fh.write_route_library(lib, ROUTE_STEM, [bf.solution(200, 0, (128, 128, 256))])
    stems = [ROUTE_STEM]
    if anchor:
        fh.write_route_library(
            lib,
            ANCHOR_STEM,
            [bf.solution(201, 0, (64, 128, 128))],
            "EqualityMatching",
            [(256, 384, 1, 256)],
        )
        stems.insert(0, ANCHOR_STEM)
    d["route_stems"] = ",".join(stems)
    return d


def test_the_production_route_names_the_rows_ahead_of_the_model(
    tmp_path, make_bundle, monkeypatch, capsys
):
    pytest.importorskip("tilewright")
    d = deploy_routed_build(tmp_path, make_bundle, anchor=True)
    monkeypatch.setenv("FAKE_ROUTE_STEMS", d["route_stems"])
    monkeypatch.delenv("FAKE_ROUTE_MODEL_FIRST", raising=False)
    envs = {}
    real = s7.run_bench

    def spy(bench, yaml_path, log_path, device, env, **kw):
        envs[log_path.relative_to(out).as_posix()] = env
        return real(bench, yaml_path, log_path, device, env, **kw)

    monkeypatch.setattr(s7, "run_bench", spy)
    out = tmp_path / "s7"
    timings = write_timings(tmp_path / "timings", KEPT)
    held = tmp_path / "audit_sample.yaml"
    held.write_text(d["yaml"].read_text())
    argv = ["--no-timing", "--timings-csv-dir", str(timings), "--held-out-yaml"]
    argv += [str(held), "--origami-reference"]
    assert run_stage07(d, out, monkeypatch, *argv) == 0
    log = "".join(capsys.readouterr())
    assert (
        "tilewright served 0 of 3 problems (0.00%), rows ahead of the Prediction "
        "row answered 3" in log
    )
    # Kernels are named by signature, never by their solution names.
    assert (
        f"2 by the GridBased row of {ROUTE_STEM} (MT128x128x256 MI16x16x128 NTA0 "
        f"NTB0)" in log
    )
    assert GRID_NAME not in log

    route = envs["work/route/bench.log"]
    assert route["TENSILE_USE_TILEWRIGHT"] == "1"
    assert "TENSILE_PREDICTION_LIB" not in route
    others = [e for p, e in envs.items() if not p.endswith("route/bench.log")]
    assert others and all(e["TENSILE_PREDICTION_LIB"] == "1" for e in others)
    bench = yaml.safe_load((out / "work" / "route" / "bench.yaml").read_text())
    assert {(e["requested_solution_num"], e["print_kernel_info"]) for e in bench} == {
        (1, 1)
    }

    summ = json.loads((out / "summary.json").read_text())
    r = summ["per_yaml"]["work"]["production_route"]
    assert r["n_problems"] == 3 and r["n_tilewright"] == 0
    assert r["prediction_lib"] is False
    assert r["rows"] == {"Equality": 1, "GridBased": 2} and r["n_ahead"] == 3
    assert r["model_loaded"] and r["warned"] and r["n_unscored"] == 0
    assert r["served_by"][0]["library"] == ROUTE_STEM
    assert r["served_by"][0]["kernel"] == GRID_NAME
    assert r["served_by"][0]["signature"] == "MT128x128x256 MI16x16x128 NTA0 NTB0"
    assert r["served_by"][1]["library"] == ANCHOR_STEM
    assert "per_problem" not in r
    route_json = json.loads((out / "work" / "route" / "route.json").read_text())
    rows = route_json["per_problem"]
    assert {(p["m"], p["row"], p["solution_index"]) for p in rows} == {
        (1024, "GridBased", 200),
        (256, "Equality", 201),
        (4096, "GridBased", 200),
    }
    assert summ["production_route"] == {
        "n_problems": 6,
        "n_tilewright": 0,
        "n_unscored": 0,
        "n_reached": 0,
        "n_ahead": 6,
        "tilewright_share": 0.0,
        "warned_yamls": ["audit_sample", "work"],
        "n_left_out": 0,
    }
    # The canary's precaution found no kernel of a hazard class in the rows.
    assert summ["hazards"]["route"] == {"kernels": [], "skipped": None}
    # The Origami reference benches with TENSILE_PREDICTION_LIB=1: the
    # Prediction row's ranking, whatever the rows ahead of it.
    ref = summ["per_yaml"]["audit_sample"]["origami_reference"]
    assert ref["origami_rows"] == {"Prediction": 3}
    assert "(Origami, from the Prediction row with TENSILE_PREDICTION_LIB=1)" in log


def test_a_runtime_that_looks_ahead_to_a_loaded_model_serves_it(
    tmp_path, make_bundle, monkeypatch, capsys
):
    pytest.importorskip("tilewright")
    d = deploy_routed_build(tmp_path, make_bundle, anchor=True)
    monkeypatch.setenv("FAKE_ROUTE_STEMS", d["route_stems"])
    monkeypatch.setenv("FAKE_ROUTE_MODEL_FIRST", "1")
    out = tmp_path / "s7"
    assert run_stage07(d, out, monkeypatch, "--no-timing") == 0
    r = json.loads((out / "summary.json").read_text())["per_yaml"]["work"][
        "production_route"
    ]
    # The Equality row still answers the size it was tuned for.
    assert r["n_tilewright"] == 2 and r["rows"] == {"Equality": 1}
    assert "tilewright served 2 of 3 problems" in capsys.readouterr().err

    monkeypatch.setenv("FAKE_ROUTE_STEMS", ROUTE_STEM)
    out = tmp_path / "s7_grid"
    assert run_stage07(d, out, monkeypatch, "--no-timing") == 0
    r = json.loads((out / "summary.json").read_text())["per_yaml"]["work"][
        "production_route"
    ]
    assert r["n_tilewright"] == 3 and r["rows"] == {} and r["model_loaded"]
    assert "tilewright served all 3 problems" in capsys.readouterr().out

    out = tmp_path / "s7_off"
    assert run_stage07(d, out, monkeypatch, "--no-timing", "--no-route") == 0
    summ = json.loads((out / "summary.json").read_text())
    assert "production_route" not in summ["per_yaml"]["work"]
    assert summ["production_route"] is None and not (out / "work" / "route").exists()


def test_problems_the_model_scores_nothing_for_reach_it_without_a_warning(
    tmp_path, make_bundle, monkeypatch, capsys
):
    """The model's Prediction row serves a problem the model scores nothing
    for without a pick-log line: the problem reached the model, so the
    route counts it apart and does not warn."""
    pytest.importorskip("tilewright")
    d = deploy_routed_build(tmp_path, make_bundle)
    monkeypatch.delenv("FAKE_ROUTE_STEMS", raising=False)
    monkeypatch.setenv("FAKE_UNSCORED", "256")
    out = tmp_path / "s7"
    argv = ("--no-timing", "--no-parity")
    assert run_stage07(d, out, monkeypatch, *argv) == 0
    captured = capsys.readouterr()
    summ = json.loads((out / "summary.json").read_text())
    r = summ["per_yaml"]["work"]["production_route"]
    assert (r["n_tilewright"], r["n_unscored"], r["n_reached"]) == (2, 1, 3)
    assert r["n_ahead"] == 0 and not r["warned"] and r["model_loaded"]
    assert r["rows"] == {"Prediction": 1} and r["served_by"][0]["unscored"]
    assert summ["production_route"]["warned_yamls"] == []
    assert "scored nothing for the other 1, which its Prediction row" in captured.out
    assert "[route]" not in captured.err

    # Rows ahead of the model answering one problem warn, naming them alone.
    monkeypatch.setenv("FAKE_ROUTE_STEMS", ROUTE_STEM)
    monkeypatch.setenv("FAKE_UNSCORED", "999")
    out = tmp_path / "s7_ahead"
    assert run_stage07(d, out, monkeypatch, *argv) == 0
    r = json.loads((out / "summary.json").read_text())["per_yaml"]["work"][
        "production_route"
    ]
    assert r["warned"] and r["n_ahead"] == 3 and r["n_unscored"] == 0


def test_a_route_bench_that_fails_is_a_warning(tmp_path, deployed, monkeypatch, capsys):
    pytest.importorskip("tilewright")
    real = s7.run_bench

    def failing_route(bench, yaml_path, log_path, device, env, **kw):
        if log_path.parent.name == "route":
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("Segmentation fault\n")
            return -11
        return real(bench, yaml_path, log_path, device, env, **kw)

    monkeypatch.setattr(s7, "run_bench", failing_route)
    out = tmp_path / "s7"
    assert run_stage07(deployed, out, monkeypatch, "--no-timing") == 0
    summ = json.loads((out / "summary.json").read_text())
    assert summ["ok"] and not summ["failures"]
    assert summ["per_yaml"]["work"]["production_route"] == {"error": "bench failed"}
    assert summ["production_route"] is None
    assert "hipblaslt-bench failed (rc=-11)" in capsys.readouterr().err


def _hazard_kernel(index, mt, **size_mapping):
    """A solution of the hazard classes: ClusterDim [2, 4] by its logic
    parameters and PrefetchGL2 2 by its name."""
    sol = bf.solution(index, 0, mt, clusterDim=[2, 4], **size_mapping)
    sol["name"] = sol["name"].replace("_SN_", "_PGL2_SN_")
    return sol


def test_only_the_kernels_the_canary_runs_alone_are_vetted():
    """A kernel of a hazard class is vetted only as one of the pool's
    kernels the served blocklist leaves in, which the canary runs alone, by
    solution index or name; the model's Prediction row serves only those.
    A problem whose size a held Equality row answers is left out once."""
    from lib import correctness as cx
    from lib import library_pool as lp

    hazard = kernel_dat_info(_hazard_kernel(1, (64, 128, 128)))
    plain = kernel_dat_info(bf.solution(2, 0, (64, 128, 128)))
    listed = kernel_dat_info(_hazard_kernel(4, (128, 128, 128)))
    other = kernel_dat_info(_hazard_kernel(5, (256, 128, 128)))
    rows = lp.TypeRows(
        entries=[
            lp.RowEntry("host", "Prediction", 1),
            lp.RowEntry("host", "Prediction", 4),
            lp.RowEntry("anchors", "Equality", 2, (8, 8, 1, 8)),
            # The pool's kernel by name under another solution index.
            lp.RowEntry("anchors", "Equality", 3, (8, 8, 1, 16)),
            lp.RowEntry("anchors", "Equality", 4, (8, 8, 1, 32)),
            lp.RowEntry("anchors", "Equality", 5, (8, None, 1, 64)),
            lp.RowEntry("grid", "GridBased", 5),
        ],
        kernels={
            1: hazard,
            2: plain,
            3: dict(hazard, sol_idx_global=3),
            4: listed,
            5: other,
        },
    )
    blocked = cx.Blocklist(
        [cx.Entry(name=listed["kernel_name"], reason="wrong", source="lists")]
    )
    held = s7.unvetted_hazards(rows, [hazard, listed], host="host", blocked=blocked)
    assert [(e.stem, e.kind, e.index) for e, _ in held] == [
        ("anchors", "Equality", 4),
        ("anchors", "Equality", 5),
        ("grid", "GridBased", 5),
    ]
    assert [rec["size"] for _, rec in held] == [[8, 8, 1, 32], [8, None, 1, 64], None]
    assert held[0][1]["kernel"] == "MT128x128x128 MI16x16x128 NTA0 NTB0"
    assert held[0][1]["classes"] == ["cluster_dim_y4", "prefetch_gl2_2"]
    # Without the blocklist the listed kernel is vetted.
    assert len(s7.unvetted_hazards(rows, [hazard, listed], host="host")) == 2

    problems = [entry(8, 8, 32), entry(8, 9, 64), entry(8, 9, 64), entry(9, 8, 64)]
    routed, left_out = s7.route_problems(problems + [entry(8, 8, 8)], held[:2])
    assert [(e["M"], e["N"], e["K"]) for e in routed] == [(9, 8, 64), (8, 8, 8)]
    assert [(p["m"], p["n"], p["k"], p["row"]) for p in left_out] == [
        (8, 8, 32, "Equality"),
        (8, 9, 64, "Equality"),
    ]


def test_the_route_leaves_out_what_an_unvetted_hazard_kernel_answers(
    tmp_path, make_bundle, monkeypatch, capsys
):
    """An Equality row ahead of the model whose kernel is of a hazard class
    and outside the pool, which the canary never ran alone, answers its size
    in production: the route leaves the problems of that size out of its
    bench, and the yaml's `skipped` names them with the kernel's signature.
    The other runs set TENSILE_PREDICTION_LIB, which skips the row."""
    pytest.importorskip("tilewright")
    d = deploy_routed_build(tmp_path, make_bundle)
    lib = d["build"] / "Tensile" / "library" / "gfx1250-strict"
    fh.write_route_library(
        lib,
        ANCHOR_STEM,
        [_hazard_kernel(201, (64, 128, 128))],
        "EqualityMatching",
        [(256, 384, 1, 256)],
    )
    monkeypatch.setenv("FAKE_ROUTE_STEMS", f"{ANCHOR_STEM},{ROUTE_STEM}")
    held = tmp_path / "audit_sample.yaml"
    held.write_text(d["yaml"].read_text())
    out = tmp_path / "s7"
    assert run_stage07(d, out, monkeypatch, "--held-out-yaml", str(held)) == 0
    log = "".join(capsys.readouterr())
    summ = json.loads((out / "summary.json").read_text())
    kernel = "MT64x128x128 MI16x16x128 NTA0 NTB0"
    assert summ["hazards"]["route"] == {
        "kernels": [
            {
                "row": "Equality",
                "library": ANCHOR_STEM,
                "size": [256, 384, 1, 256],
                "kernel": kernel,
                "classes": ["cluster_dim_y4", "prefetch_gl2_2"],
            }
        ],
        "skipped": "sizes",
    }
    for tag in ("work", "audit_sample"):
        y = summ["per_yaml"][tag]
        assert y["skipped"]["production_route_problems"] == [
            {
                "m": 256,
                "n": 384,
                "k": 256,
                "batch": 1,
                "transA": "T",
                "transB": "N",
                "row": "Equality",
                "library": ANCHOR_STEM,
                "kernel": kernel,
                "classes": ["cluster_dim_y4", "prefetch_gl2_2"],
            }
        ]
        bench = yaml.safe_load((out / tag / "route" / "bench.yaml").read_text())
        assert sorted(e["M"] for e in bench) == [1024, 4096]
        r = y["production_route"]
        assert r["n_problems"] == 2 and r["rows"] == {"GridBased": 2}
        # Every other run still benches the problem.
        assert y["parity"]["n_problems"] == 3
    assert summ["production_route"]["n_left_out"] == 2
    assert f"{kernel} of {ANCHOR_STEM}" in log
    assert "problem(s) left out of the production route" in log
    # No bench process ran the kernel: the route leaves its size out, every
    # other run sets TENSILE_PREDICTION_LIB.
    logs = list(out.rglob("*.log"))
    assert logs and any("--Solution index: 200" in p.read_text() for p in logs)
    assert not any("--Solution index: 201" in p.read_text() for p in logs)


def test_rows_of_every_size_with_an_unvetted_hazard_kernel_skip_the_route(
    tmp_path, make_bundle, monkeypatch, capsys
):
    """A GridBased row answers every size: one that holds a kernel of a
    hazard class the canary never ran alone skips the route, and so do
    library files whose rows cannot be read. A kernel of the pool, which
    the canary runs alone, and the canary off leave the route alone."""
    pytest.importorskip("tilewright")
    d = deploy_routed_build(tmp_path, make_bundle)
    lib = d["build"] / "Tensile" / "library" / "gfx1250-strict"
    fh.write_route_library(lib, ROUTE_STEM, [_hazard_kernel(200, (128, 128, 256))])
    monkeypatch.setenv("FAKE_ROUTE_STEMS", ROUTE_STEM)
    out = tmp_path / "s7"
    assert run_stage07(d, out, monkeypatch, "--no-timing") == 0
    summ = json.loads((out / "summary.json").read_text())
    y = summ["per_yaml"]["work"]
    assert "production_route" not in y and y["parity"]["ok"]
    why = y["skipped"]["production_route"]
    assert f"the GridBased row of {ROUTE_STEM} answers problems of every size" in why
    assert "MT128x128x256 MI16x16x128 NTA0 NTB0" in why
    assert summ["hazards"]["route"]["skipped"] == "route"
    assert summ["hazards"]["route"]["kernels"][0]["size"] is None
    assert not (out / "work" / "route").exists()
    assert summ["production_route"] is None

    # The canary off: no precaution.
    cfg = yaml.safe_load(d["cfg"].read_text())
    cfg["hazards"] = {"enabled": False}
    d["cfg"].write_text(yaml.safe_dump(cfg))
    out = tmp_path / "s7_off"
    assert run_stage07(d, out, monkeypatch, "--no-timing") == 0
    summ = json.loads((out / "summary.json").read_text())
    assert summ["hazards"]["route"] is None
    assert summ["per_yaml"]["work"]["production_route"]["rows"] == {"GridBased": 3}

    # Files of the data types whose rows cannot be read: nothing can tell.
    cfg["hazards"] = {"enabled": True}
    d["cfg"].write_text(yaml.safe_dump(cfg))
    fh.write_route_library(lib, ROUTE_STEM, [bf.solution(200, 0, (128, 128, 256))])
    broken = bf.LIBRARY_STEM.replace("_Bias_SAV_UA_", "_SAV_UA_")
    (lib / f"{broken}.dat").write_bytes(b"\xc1")
    out = tmp_path / "s7_unread"
    assert run_stage07(d, out, monkeypatch, "--no-timing") == 0
    summ = json.loads((out / "summary.json").read_text())
    why = summ["per_yaml"]["work"]["skipped"]["production_route"]
    assert why.startswith("the rows of the library's files cannot be read")
    assert broken in why
