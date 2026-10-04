# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import math
from pathlib import Path

import pytest
import yaml

import bench_fakes as bf
import stage07_optional_validate_selection_time as s7

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
    "arch": "gfx1250v0",
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
        "top1_score=nan n_configs=243\n"
    )
    p = s7.parse_bench_log(log)
    assert p.diag_files[0]["path"] == "/x/m.bin" and p.diag_files[0]["n_cells"] == "3"
    assert p.diag_fails == ["/x/bad.bin: CRC mismatch"]
    pick = p.picks[0]
    assert pick["key"] == (1, 2, 3, 4, "T", "N")
    assert pick["top1_sig"] == (64, 64, 64, 16, 16, 128, 0, 4)
    assert math.isnan(pick["top1_score"]) and pick["n_configs"] == 243


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


def _cpp(m, sig, cell="c", n=10):
    return {
        "key": (m, 1, 1, 1, "T", "N"),
        "cell": cell,
        "top1_sig": sig,
        "top1_score": 1.0,
        "n_configs": n,
    }


def _py(m, sig=None, cell="c", reason="ok", n=10):
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
        d["top1_sig"], d["top1_score"] = sig, 1.0
    return d


def test_compare_picks_categories():
    a, b = (1,) * 8, (2,) * 8
    problems = [entry(m, 1, 1) for m in range(1, 8)] + [entry(1, 1, 1)]
    cpp = [
        _cpp(1, a),
        _cpp(2, a),
        _cpp(3, a, cell="other"),
        _cpp(5, a),
        _cpp(6, a, n=12),
        _cpp(6, b),
    ]
    py = [
        _py(1, a),
        _py(2, b),
        _py(3, a),
        _py(4, a),
        _py(5, reason="no_survivors"),
        _py(6, a),
        _py(7, reason="no_model", cell=None),
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
    ]
    assert r["n_problems"] == 7 and r["n_agree"] == 3
    assert r["match_rate"] == pytest.approx(3 / 7)
    assert r["counts"]["pool_size_differs"] == 1
    assert r["n_inconsistent_cpp_picks"] == 1
    assert r["library_pool_size"] == 10


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


@pytest.fixture
def deployed(tmp_path, make_bundle):
    """A fake deploy-checkout build with the model co-located and a stage06
    record, as stage06 and the hipBLASLt model build leave them."""
    import fake_hipblaslt as fh
    import stage06_deploy_weights as s6

    build = fh.write_fake_build(tmp_path / "build", "gfx1250v0", bf.LIBRARY_STEM)
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
    cfg_path.write_text(yaml.safe_dump(cfg))
    out6 = tmp_path / "stage06"
    assert (
        s6.deploy(train_dir=train, config_yaml=cfg_path, out_dir=out6, quiet=True) == 0
    )
    rec = json.loads((out6 / "deploy_record.json").read_text())
    lib = build / "Tensile" / "library" / "gfx1250v0"
    t = rec["targets"][0]
    (lib / t["weights_file"]).write_bytes(Path(t["staged"]).read_bytes())
    (lib / "tilewright_index").write_text(f"{t['library_stem']}\t{t['weights_file']}\n")
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


def test_stale_colocated_model_fails(tmp_path, deployed, monkeypatch):
    rec = json.loads(deployed["record"].read_text())
    lib = deployed["build"] / "Tensile" / "library" / "gfx1250v0"
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
