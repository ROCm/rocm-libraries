# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import stage06_deploy_weights as s6
from lib import mlrec

STEM = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_ID75a0_gfx950"
OTHER = "TensileLibrary_SS_SS_HA_Bias_SAV_MX_UA_Type_SS_Contraction_l_Alik_Bljk_Cijk_Dijk_ID75a0_gfx950"
MX_STEM = "TensileLibrary_F8F8_BF8_HA_MXAE8B32_MXBE8B32_Bias_SAV_UA_Type_F8B_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1250-strict"
LABELS = [
    "Large|Large|LargeK|Bnone#M<=4096",
    "Large|Large|LargeK|Bnone#M>4096",
    "Tiny|Tiny|TinyK|Bnone",
]


def write_cfg(path, arch="gfx950", stem=STEM, targets=None, dtype="int8"):
    cfg = {
        "config_id": "deploy_test",
        "arch": arch,
        "hipblaslt": {"transA": "T", "transB": "N", "library_stem": stem},
        "deploy": {"weight_dtype": dtype},
    }
    if targets:
        cfg["deploy"]["targets"] = targets
    path.write_text(yaml.safe_dump(cfg))
    return path


def weights_dir(repo, d):
    return repo / "shared" / "tilewright" / "weights" / "hipblaslt" / d / d


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "repo"
    wd = weights_dir(repo, "gfx950")
    wd.mkdir(parents=True)
    (wd / "tilewright_index").write_text(
        "# tilewright per-library weights index: <logic_stem>\\t<weights.bin>\n"
        "# kept comment\n"
        f"{OTHER}\tother.tilewright.bin\n"
        f"{STEM}\tlegacy.bin\n"
        f"{STEM}\tshadowed.tilewright.bin\n"
        "lonely_token\n"
    )
    (wd / "other.tilewright.bin").write_bytes(b"other model")
    (wd / "legacy.bin").write_bytes(b"old model")
    return repo


@pytest.fixture
def trained(tmp_path, make_bundle):
    d = tmp_path / "round_1" / "stage05"
    make_bundle(d, LABELS)
    return d


def deploy(trained, cfg, out, **kw):
    return s6.deploy(train_dir=trained, config_yaml=cfg, out_dir=out, quiet=True, **kw)


def leftovers(d):
    return sorted(p.name for p in d.iterdir() if p.name.startswith("."))


def test_parse_index_matches_the_engine():
    idx = s6.parse_index(
        "# header\r\n"
        "a\tone.bin # trailing comment\r\n"
        "b two.bin extra\n"
        "a\tlater.bin\n"
        "single\n"
        "\n"
        "  # indented comment\n"
    )
    assert idx.as_dict() == {"a": "one.bin", "b": "two.bin"}
    assert idx.comments == ["# header", "  # indented comment"]
    assert idx.dropped == ["a\tlater.bin", "single"]


def test_render_index_sorts_and_keeps_comments():
    text = s6.render_index(["# c"], {"z": "z.bin", "a": "a.bin"})
    assert text == "# c\na\ta.bin\nz\tz.bin\n"
    assert s6.render_index([], {}).startswith("# tilewright per-library weights index")


def test_choose_weights_name():
    assert s6.choose_weights_name({}, "S") == "S.tilewright.bin"
    assert s6.choose_weights_name({"S": "x.tilewright.bin"}, "S") == "x.tilewright.bin"
    shared = {"S": "x.tilewright.bin", "T": "x.tilewright.bin"}
    assert s6.choose_weights_name(shared, "S") == "S.tilewright.bin"
    assert (
        s6.choose_weights_name({"S": "../x.tilewright.bin"}, "S") == "S.tilewright.bin"
    )
    with pytest.raises(s6.DeployError):
        s6.choose_weights_name({"T": "S.tilewright.bin"}, "S")


def test_staging_only(tmp_path, trained):
    cfg = write_cfg(tmp_path / "cfg.yaml")
    out = tmp_path / "out"
    assert deploy(trained, cfg, out) == 0
    staged = out / "staged_for_deploy" / s6.WEIGHTS_REL / "gfx950" / "gfx950"
    data = (staged / f"{STEM}.tilewright.bin").read_bytes()
    assert (
        staged / "tilewright_index.fragment"
    ).read_text() == f"{STEM}\t{STEM}.tilewright.bin\n"
    assert not (staged / "tilewright_index").exists()
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["applied"] is False and rec["sha256"] == hashlib.sha256(data).hexdigest()
    assert rec["n_cells"] == 3 and rec["n_splits"] == 1
    assert rec["launch"] == {
        "benched": None,
        "build_heuristic": None,
        "build_matches": None,
    }
    parsed = mlrec.read_model(data)
    assert parsed["format"] == "MLREC_v2" and parsed["weight_dtype_name"] == "int8"
    assert [c["label"] for c in parsed["cells"]] == sorted(LABELS)
    manifest = (out / "manifest.json").read_text()
    assert "sel_eff" not in manifest and "best_epoch" not in manifest
    install = (out / "staged_for_deploy" / "INSTALL.md").read_text()
    for stale in ("origami", "backends/hipblaslt", "tilewright_weights.bin", "%"):
        assert stale not in install


def test_the_record_says_how_the_training_benches_launched_kernels(tmp_path, capsys):
    """The deploy record keeps stage05's `bench_launch` and, with a build
    directory, whether that build launches as the training benches did."""
    from lib import launch_settings as ls
    from test_launch_settings import write_build, write_sources

    tl, origami = write_sources(tmp_path / "src")
    build = write_build(tmp_path / "build", tl, origami)
    for p in (tmp_path / "src").rglob("*"):
        os.utime(p, (900_000_000, 900_000_000))
    digest = ls.heuristic(build)["digest"]
    benched = ls.summarize([{"overrides": {}, "heuristic": {"digest": digest}}])
    rec = s6.launch_record({"bench_launch": benched}, build, quiet=False)
    assert rec == {"benched": benched, "build_heuristic": digest, "build_matches": True}
    other = ls.summarize(
        [{"overrides": {}, "heuristic": {"digest": "sha256:" + "c" * 64}}]
    )
    rec = s6.launch_record({"bench_launch": other}, build, quiet=False)
    assert rec["build_matches"] is False
    err = capsys.readouterr().err
    assert "c" * 16 in err and "launch-rule sources" in err
    assert s6.launch_record({}, build, quiet=True)["build_matches"] is None
    assert s6.launch_record({"bench_launch": benched}, None, quiet=True) == {
        "benched": benched,
        "build_heuristic": None,
        "build_matches": None,
    }
    # Sources edited after the build compiled them name no heuristic of it.
    (origami / "src" / "origami" / "streamk.cpp").write_text("// edited\n")
    os.utime(origami / "src" / "origami" / "streamk.cpp", (2e9, 2e9))
    assert ls.heuristic(build)["newer_than_build"]
    rec = s6.launch_record({"bench_launch": other}, build, quiet=False)
    assert rec["build_heuristic"] is None and rec["build_matches"] is None
    assert "differ" not in capsys.readouterr().err


def test_apply_merges_index_and_replaces_files(tmp_path, trained, checkout):
    cfg = write_cfg(tmp_path / "cfg.yaml")
    out = tmp_path / "out"
    old_umask = os.umask(0o022)
    try:
        assert deploy(trained, cfg, out, repo=checkout, apply_patch=True) == 0
    finally:
        os.umask(old_umask)
    wd = weights_dir(checkout, "gfx950")
    new_name = f"{STEM}.tilewright.bin"
    assert (wd / "tilewright_index").read_text() == (
        "# tilewright per-library weights index: <logic_stem>\\t<weights.bin>\n"
        "# kept comment\n"
        f"{STEM}\t{new_name}\n"
        f"{OTHER}\tother.tilewright.bin\n"
    )
    assert not (wd / "legacy.bin").exists()
    assert (wd / "other.tilewright.bin").read_bytes() == b"other model"
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["applied"] and rec["targets"][0]["removed"] == "legacy.bin"
    assert hashlib.sha256((wd / new_name).read_bytes()).hexdigest() == rec["sha256"]
    assert stat.S_IMODE((wd / new_name).stat().st_mode) == 0o644
    assert leftovers(wd) == []
    staged = out / "staged_for_deploy" / s6.WEIGHTS_REL / "gfx950" / "gfx950"
    assert (staged / "tilewright_index").read_text() == (
        wd / "tilewright_index"
    ).read_text()

    assert deploy(trained, cfg, tmp_path / "out2", repo=checkout, apply_patch=True) == 0
    assert sorted(p.name for p in wd.iterdir()) == [
        new_name,
        "other.tilewright.bin",
        "tilewright_index",
    ]


def test_shared_targets_get_one_model(tmp_path, trained):
    repo = tmp_path / "repo"
    (repo / "shared" / "tilewright").mkdir(parents=True)
    cfg = write_cfg(
        tmp_path / "cfg.yaml",
        arch="gfx1250-strict",
        stem=MX_STEM,
        targets=["gfx1250-strict", "gfx1250v0", "gfx1250"],
    )
    out = tmp_path / "out"
    assert deploy(trained, cfg, out, repo=repo, apply_patch=True) == 0
    blobs = set()
    for d, stem in (
        ("gfx1250-strict", MX_STEM),
        ("gfx1250v0", MX_STEM[: -len("-strict")]),
        ("gfx1250", MX_STEM[: -len("-strict")]),
    ):
        entries = s6.parse_index(
            (weights_dir(repo, d) / "tilewright_index").read_text()
        )
        assert entries.as_dict() == {stem: f"{stem}.tilewright.bin"}
        blobs.add((weights_dir(repo, d) / f"{stem}.tilewright.bin").read_bytes())
    assert len(blobs) == 1
    assert mlrec.read_model(blobs.pop())["arch"] == "gfx1250-strict"


def test_failure_restores_previous_files(tmp_path, trained, checkout, monkeypatch):
    cfg = write_cfg(tmp_path / "cfg.yaml", targets=["gfx950", "gfx942"])
    wd = weights_dir(checkout, "gfx950")
    before = {p.name: p.read_bytes() for p in wd.iterdir()}
    real_put = s6.Transaction.put

    def put(self, dst, data):
        if dst.name == "tilewright_index" and dst.parent.name == "gfx942":
            raise s6.DeployError("simulated failure")
        return real_put(self, dst, data)

    monkeypatch.setattr(s6.Transaction, "put", put)
    assert deploy(trained, cfg, tmp_path / "out", repo=checkout, apply_patch=True) == 1
    assert {p.name: p.read_bytes() for p in wd.iterdir()} == before
    assert not weights_dir(checkout, "gfx942").exists()
    rec = json.loads((tmp_path / "out" / "deploy_record.json").read_text())
    assert rec["applied"] is False and "simulated failure" in rec["error"]


def fake_cmake(tmp_path, monkeypatch, body):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    script = bindir / "cmake"
    script.write_text(f"#!{sys.executable}\n{body}")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")


COPY_BODY = """
import shutil, sys
from pathlib import Path
build = Path(sys.argv[sys.argv.index("--build") + 1])
assert sys.argv[sys.argv.index("--target") + 1] == "hipblaslt-tilewright-models"
src = Path({src!r})
for arch in ("gfx950",):
    dst = build / "Tensile" / "library" / arch
    for f in (src / arch / arch).iterdir():
        if not f.name.startswith("."):
            shutil.copy(f, dst / f.name)
print("colocated")
"""


def test_build_after_apply_runs_colocate_and_verifies(
    tmp_path, trained, checkout, monkeypatch
):
    build = tmp_path / "build"
    (build / "Tensile" / "library" / "gfx950").mkdir(parents=True)
    src = checkout / "shared" / "tilewright" / "weights" / "hipblaslt"
    fake_cmake(tmp_path, monkeypatch, COPY_BODY.format(src=str(src)))
    cfg = write_cfg(tmp_path / "cfg.yaml")
    out = tmp_path / "out"
    rc = deploy(
        trained,
        cfg,
        out,
        repo=checkout,
        apply_patch=True,
        build_after_apply=True,
        build_dir=build,
    )
    assert rc == 0
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["built"] is True and rec["colocation"] == "verified"
    assert rec["colocated_targets"] == ["gfx950"] and rec["engine_checked"]
    log = (out / "build.log").read_text()
    assert "hipblaslt-tilewright-models" in log and "colocated" in log and "rc=0" in log


CORRUPT_BODY = COPY_BODY.replace(
    'print("colocated")',
    'p = dst / "{name}"\np.write_bytes(p.read_bytes()[:-1])\nprint("corrupted")',
)


@pytest.mark.parametrize(
    "body",
    [
        "import sys\nsys.exit(3)\n",
        "print('copied nothing')\n",
        CORRUPT_BODY,
    ],
)
def test_build_failure_rolls_back(tmp_path, trained, checkout, monkeypatch, body):
    wd = weights_dir(checkout, "gfx950")
    before = {p.name: p.read_bytes() for p in wd.iterdir()}
    build = tmp_path / "build"
    lib = build / "Tensile" / "library" / "gfx950"
    lib.mkdir(parents=True)
    for name, data in before.items():
        (lib / name).write_bytes(data)
    built_before = dict(before)
    src = checkout / "shared" / "tilewright" / "weights" / "hipblaslt"
    fake_cmake(
        tmp_path,
        monkeypatch,
        body.format(src=str(src), name=f"{STEM}.tilewright.bin"),
    )
    cfg = write_cfg(tmp_path / "cfg.yaml")
    rc = deploy(
        trained,
        cfg,
        tmp_path / "out",
        repo=checkout,
        apply_patch=True,
        build_after_apply=True,
        build_dir=build,
    )
    assert rc == 1
    assert {p.name: p.read_bytes() for p in wd.iterdir()} == before
    assert {p.name: p.read_bytes() for p in lib.iterdir()} == built_before
    rec = json.loads((tmp_path / "out" / "deploy_record.json").read_text())
    assert rec["applied"] is False and rec["built"] is False


def test_build_without_a_library_directory_is_unverified(
    tmp_path, trained, checkout, monkeypatch
):
    build = tmp_path / "build"
    build.mkdir()
    fake_cmake(tmp_path, monkeypatch, "print('nothing to copy')\n")
    out = tmp_path / "out"
    rc = deploy(
        trained,
        write_cfg(tmp_path / "cfg.yaml"),
        out,
        repo=checkout,
        apply_patch=True,
        build_after_apply=True,
        build_dir=build,
    )
    assert rc == 0
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["applied"] is True and rec["built"] is False
    assert rec["colocation"] == "unverified" and rec["colocated_targets"] == []


def test_apply_patch_requires_the_engine(tmp_path, trained, checkout, monkeypatch):
    def missing():
        raise RuntimeError("no module named tilewright")

    monkeypatch.setattr(s6.ev, "tilewright_module", missing)
    wd = weights_dir(checkout, "gfx950")
    before = {p.name: p.read_bytes() for p in wd.iterdir()}
    cfg = write_cfg(tmp_path / "cfg.yaml")
    assert deploy(trained, cfg, tmp_path / "o1", repo=checkout, apply_patch=True) == 1
    assert {p.name: p.read_bytes() for p in wd.iterdir()} == before
    assert deploy(trained, cfg, tmp_path / "o2") == 0
    rec = json.loads((tmp_path / "o2" / "deploy_record.json").read_text())
    assert rec["engine_checked"] is False and rec["applied"] is False


def test_missing_torch_is_a_deploy_error(tmp_path, trained, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(s6.DeployError, match="needs torch"):
        s6.load_bundle(trained)
    assert deploy(trained, write_cfg(tmp_path / "cfg.yaml"), tmp_path / "o") == 1


def test_catalog_mismatch_is_fatal(tmp_path, make_bundle, checkout):
    bad = tmp_path / "bad"
    from lib import features as fs

    make_bundle(bad, LABELS, names={"q_names": fs.query_feature_names()[:-1]})
    cfg = write_cfg(tmp_path / "cfg.yaml")
    wd = weights_dir(checkout, "gfx950")
    before = {p.name: p.read_bytes() for p in wd.iterdir()}
    assert deploy(bad, cfg, tmp_path / "o1", repo=checkout, apply_patch=True) == 1
    good = tmp_path / "good"
    make_bundle(good, LABELS)
    (good / "cells.json").write_text(
        json.dumps({"feature_names_hash": "0123456789abcdef"})
    )
    assert deploy(good, cfg, tmp_path / "o2", repo=checkout, apply_patch=True) == 1
    assert {p.name: p.read_bytes() for p in wd.iterdir()} == before


def test_concurrent_deploys_keep_both_index_entries(tmp_path, make_bundle, checkout):
    procs = []
    for n, stem in enumerate((STEM, OTHER)):
        train = tmp_path / f"t{n}"
        make_bundle(train, LABELS, seed=n)
        cfg = write_cfg(tmp_path / f"cfg{n}.yaml", stem=stem)
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(s6.__file__)),
                    "--train-dir",
                    str(train),
                    "--config-yaml",
                    str(cfg),
                    "--out-dir",
                    str(tmp_path / f"out{n}"),
                    "--rocm-libraries-to-deploy",
                    str(checkout),
                    "--apply-patch",
                    "--quiet",
                ]
            )
        )
    assert [p.wait() for p in procs] == [0, 0]
    entries = s6.parse_index(
        (weights_dir(checkout, "gfx950") / "tilewright_index").read_text()
    )
    assert entries.as_dict() == {
        STEM: f"{STEM}.tilewright.bin",
        OTHER: "other.tilewright.bin",
    }
    rec = json.loads((tmp_path / "out1" / "deploy_record.json").read_text())
    data = (weights_dir(checkout, "gfx950") / "other.tilewright.bin").read_bytes()
    assert hashlib.sha256(data).hexdigest() == rec["sha256"]


def test_engine_loads_the_deployed_bytes(tmp_path, trained):
    tilewright = pytest.importorskip("tilewright")
    cfg = write_cfg(tmp_path / "cfg.yaml", dtype="int4")
    out = tmp_path / "out"
    assert deploy(trained, cfg, out) == 0
    rec = json.loads((out / "deploy_record.json").read_text())
    data = Path(rec["staged_model"]).read_bytes()
    model = tilewright.load_model_from_memory(data)
    info = tilewright.describe(model)
    assert (info.n_cells, info.n_splits) == (3, 1)
    assert info.feature_catalog_hash == rec["feature_catalog_hash"]


# ── deploy gate ─────────────────────────────────────────────────────────────

GATE_HW = {"n_cu": 100, "lds_bytes": 65536, "l2_bytes": 4194304}


def gate_cfg(path, **deploy):
    cfg = yaml.safe_load(write_cfg(path).read_text())
    cfg["hardware"] = dict(GATE_HW)
    cfg["deploy"].update(deploy)
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def gate_dir(tmp_path):
    from test_stage05_train import shapes_for, write_round

    return write_round(
        tmp_path / "round_1" / "stage04b" / "gate",
        shapes_for(8, 12, ["Large|Large|LargeK|Bnone", "Mid|Large|MidK|Bnone"]),
    )


def test_the_gate_measures_the_bytes_that_ship(tmp_path, trained, gate_dir, capsys):
    pytest.importorskip("tilewright")
    import stage05_train as s5
    from lib import evaluate as ev
    from lib.hardware import DeviceHardware

    out = tmp_path / "out"
    gate = s6.Gate(csv_dirs=[gate_dir], bootstrap=0)
    assert deploy(trained, gate_cfg(tmp_path / "cfg.yaml"), out, gate=gate) == 0
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["deployed"] is True and rec["gate"]["passed"] is True
    measured = json.loads((out / "gate.json").read_text())
    cells, _ = s5.load_enriched_chunks(str(gate_dir), True)
    gemms = [g for gs in cells.values() for g in gs]
    data = Path(rec["staged_model"]).read_bytes()
    want = ev.evaluate_model(
        data, gemms, hardware=DeviceHardware(**GATE_HW), pick_measurement="warmup"
    ).summary()
    assert measured["n_gemms"] == len(gemms) == 24
    assert measured["sel_eff"] == pytest.approx(want["sel_eff"])
    assert measured["n_miss"] == want["n_miss"] == rec["gate"]["n_miss"]
    assert rec["gate"]["sel_eff"] == pytest.approx(want["sel_eff"])
    assert sum(c["n_eval"] for c in measured["per_cell"].values()) == 24
    assert set(measured["per_cell"]) <= set(LABELS) | {"Mid|Large|MidK|Bnone"}
    for c in measured["per_cell"].values():
        assert {"sel_eff_new", "origami_sel_eff_new", "paired_sel_eff_new"} <= set(c)
    assert "origami" not in capsys.readouterr().out.lower()


def test_the_gate_keeps_a_model_below_the_target_out(
    tmp_path, trained, checkout, gate_dir
):
    pytest.importorskip("tilewright")
    wd = weights_dir(checkout, "gfx950")
    before = {p.name: p.read_bytes() for p in wd.iterdir()}
    out = tmp_path / "out"
    gate = s6.Gate(csv_dirs=[gate_dir], target=1.01, bootstrap=0)
    cfg = gate_cfg(tmp_path / "cfg.yaml")
    rc = deploy(trained, cfg, out, repo=checkout, apply_patch=True, gate=gate)
    assert rc == 0
    assert {p.name: p.read_bytes() for p in wd.iterdir()} == before
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["deployed"] is False and rec["applied"] is False
    assert "below the target" in rec["not_deployed"]
    assert Path(rec["staged_model"]).is_file()
    fallback = s6.Gate(csv_dirs=[gate_dir], target=1.01, fallback="not-worse")
    out2 = tmp_path / "out2"
    assert deploy(trained, cfg, out2, gate=fallback) == 0
    rec = json.loads((out2 / "deploy_record.json").read_text())
    assert rec["deployed"] and "no bundle was deployed before" in rec["gate"]["reason"]
    same = s6.Gate(
        csv_dirs=[gate_dir],
        target=1.01,
        fallback="not-worse",
        baseline_train_dir=trained,
    )
    out3 = tmp_path / "out3"
    assert deploy(trained, cfg, out3, gate=same) == 0
    rec = json.loads((out3 / "deploy_record.json").read_text())
    base = rec["gate"]["baseline"]
    assert base["n"] == 24 and base["sel_eff"] == pytest.approx(base["bundle_sel_eff"])
    assert rec["deployed"] and "not worse than the deployed bundle" in (
        rec["gate"]["reason"]
    )


def test_a_gate_without_gemms_measures_nothing(tmp_path, trained):
    pytest.importorskip("tilewright")
    (tmp_path / "empty").mkdir()
    gate = s6.Gate(csv_dirs=[tmp_path / "empty"], target=0.5)
    out = tmp_path / "out"
    assert deploy(trained, gate_cfg(tmp_path / "cfg.yaml"), out, gate=gate) == 0
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["gate"]["n_gemms"] == 0 and rec["gate"]["sel_eff"] is None
    assert rec["deployed"] is False and rec["not_deployed"] == "no gate GEMM measured"


def test_gate_decision():
    m = {"sel_eff": 0.9, "n_evaluated": 10}
    gate = s6.Gate(csv_dirs=[], target=0.95)
    assert s6.gate_decision(m, s6.Gate(csv_dirs=[])) == (True, "no target")
    assert s6.gate_decision(dict(m, sel_eff=0.96), gate)[0]
    assert s6.gate_decision(m, gate) == (
        False,
        "90.00% on the gate GEMMs is below the target",
    )
    assert not s6.gate_decision(dict(m, n_evaluated=0), gate)[0]
    fb = s6.Gate(csv_dirs=[], target=0.95, fallback="not-worse")
    assert s6.gate_decision(m, fb)[0]
    fb.baseline_train_dir = Path("b")
    worse = dict(m, baseline={"n": 10, "sel_eff": 0.92, "bundle_sel_eff": 0.9})
    ok, why = s6.gate_decision(worse, fb)
    assert not ok and "worse than the deployed bundle" in why
    better = dict(m, baseline={"n": 10, "sel_eff": 0.88, "bundle_sel_eff": 0.9})
    assert s6.gate_decision(better, fb)[0]
    ok, why = s6.gate_decision(dict(m, baseline={"error": "x"}), fb)
    assert not ok and "not comparable" in why


def test_the_gate_needs_the_engine(tmp_path, trained, gate_dir, monkeypatch):
    def missing():
        raise RuntimeError("no module named tilewright")

    monkeypatch.setattr(s6.ev, "tilewright_module", missing)
    gate = s6.Gate(csv_dirs=[gate_dir])
    cfg = gate_cfg(tmp_path / "cfg.yaml")
    assert deploy(trained, cfg, tmp_path / "o", gate=gate) == 1
    assert not (tmp_path / "o" / "deploy_record.json").exists()


def test_the_external_reference_prints_only_on_request(
    tmp_path, trained, gate_dir, capsys
):
    pytest.importorskip("tilewright")
    from test_stage05_train import shapes_for, write_round

    write_round(gate_dir, shapes_for(9, 6, ["Large|Large|LargeK|Bnone"]))
    gate = s6.Gate(csv_dirs=[gate_dir], bootstrap=0)
    cfg = gate_cfg(tmp_path / "cfg.yaml")
    capsys.readouterr()
    assert (
        s6.deploy(train_dir=trained, config_yaml=cfg, out_dir=tmp_path / "a", gate=gate)
        == 0
    )
    plain = capsys.readouterr().out
    assert "[gate] model on" in plain and "origami" not in plain.lower()
    assert (
        s6.deploy(
            train_dir=trained,
            config_yaml=cfg,
            out_dir=tmp_path / "b",
            gate=gate,
            origami_reference=True,
        )
        == 0
    )
    assert "[reference] Origami on the same" in capsys.readouterr().out


def write_v3_bundle(d, labels, **extra):
    import torch

    from test_v3 import v3_bundle

    from lib import features as fs

    bundle = v3_bundle(labels=labels, clip=True, **extra)
    q, i, x = fs.Catalog("v3", fs.DEFAULT_ATTRIBUTE_SPEC).names()
    bundle.update(q_names=q, i_names=i, x_names=x)
    for entry in bundle["models"].values():
        entry["state_dict"] = {
            k: torch.tensor(v) for k, v in entry["state_dict"].items()
        }
        entry["view_key_hashes"] = [7, 2**63 + 5]
    d.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, d / "models.pt")
    return bundle


def test_deploys_a_v3_bundle(tmp_path):
    from lib import features as fs

    train = tmp_path / "v3" / "stage05"
    write_v3_bundle(train, LABELS)
    out = tmp_path / "out"
    assert deploy(train, write_cfg(tmp_path / "cfg.yaml", dtype="bf16"), out) == 0
    rec = json.loads((out / "deploy_record.json").read_text())
    data = Path(rec["staged_model"]).read_bytes()
    parsed = mlrec.read_model(data)
    assert parsed["format"] == "MLREC_v3" and len(parsed["cells"]) == 3
    assert all(c["view_key_hashes"] == [7, 2**63 + 5] for c in parsed["cells"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["format"] == "MLREC_v3"
    assert manifest["feature_spec_hash"] == (
        f"{mlrec.feature_spec_hash(fs.DEFAULT_ATTRIBUTE_SPEC):016x}"
    )
    assert manifest["contexts"]["schedules"] == ["default", "dynamic"]
    assert manifest["cells"][0]["n_view_keys"] == 2
    install = (out / "staged_for_deploy" / "INSTALL.md").read_text()
    assert "One MLREC_v3 model" in install
    tw = pytest.importorskip("tilewright")
    info = tw.describe(tw.load_model_from_memory(data))
    assert info.format_version == 3 and info.n_cells == 3


def test_a_v3_bundle_with_another_spec_is_refused(tmp_path):
    train = tmp_path / "v3" / "stage05"
    bundle = write_v3_bundle(train, LABELS)
    import torch

    bundle["attribute_spec"] = bundle["attribute_spec"] + [{"name": "param_a"}]
    torch.save(bundle, train / "models.pt")
    assert deploy(train, write_cfg(tmp_path / "cfg.yaml"), tmp_path / "o") == 1


def test_the_gate_reads_rows_of_another_build_at_their_kernels(
    tmp_path, trained, gate_dir
):
    """Gate rows recorded as enriched with build A, measured against build
    B, which numbers every kernel otherwise, give the numbers of build A's
    rows against build A."""
    pytest.importorskip("tilewright")
    import shutil

    from test_kernel_identity import pool_library, record_rows
    from test_stage05_train import kernel_pool

    ident_a = pool_library(tmp_path / "a", kernel_pool(), stem=STEM)
    ident_b = pool_library(tmp_path / "b", kernel_pool(), offset=1000, stem=STEM)
    moved = record_rows(shutil.copytree(gate_dir, tmp_path / "gate_a"), ident_a, STEM)
    cfg = gate_cfg(tmp_path / "cfg.yaml")
    got = {}
    for name, rows in (("a", gate_dir), ("b", moved)):
        gate = s6.Gate(csv_dirs=[rows], library_dir=tmp_path / name, bootstrap=0)
        out = tmp_path / f"out_{name}"
        assert deploy(trained, cfg, out, gate=gate, library_dir=tmp_path / name) == 0
        got[name] = json.loads((out / "gate.json").read_text())
        rec = json.loads((out / "deploy_record.json").read_text())
        ident = ident_a if name == "a" else ident_b
        assert rec["member_order"]["library_digest"] == ident.digest
    for key in ("n_gemms", "sel_eff", "n_miss"):
        assert got["b"][key] == got["a"][key]
    assert {c: v["sel_eff_new"] for c, v in got["b"]["per_cell"].items()} == {
        c: v["sel_eff_new"] for c, v in got["a"]["per_cell"].items()
    }


def test_the_deploy_record_holds_each_keys_member_order(tmp_path, trained, capsys):
    """Twin kernels (one engine config, other names) are one candidate; the
    record digests each key's members in pool order and warns when the
    library the model is served from orders them otherwise than the build
    it trained with (stage05's member_order.json, for a bundle carried
    forward without one that of the round it came from)."""
    import shutil

    import stage05_train as s5
    from lib import evaluate as ev
    from test_kernel_identity import pool_library
    from test_stage05_train import twin_pool

    pool = twin_pool()
    cfg = gate_cfg(tmp_path / "cfg.yaml")
    order = list(range(len(pool)))[::-1]
    ident_x = pool_library(tmp_path / "x", pool, stem=STEM)
    pool_library(tmp_path / "y", pool, stem=STEM, order=order)
    lib_x = ev.library_pool(tmp_path / "x", STEM, yaml.safe_load(cfg.read_text()))
    trained_with = s5.member_order(lib_x, None, False, False, "gfx950", ident_x)
    assert len(trained_with["keys"]) == 6
    (trained / s5.MEMBER_ORDER_FILE).write_text(json.dumps(trained_with))

    def deployed(lib, out):
        capsys.readouterr()
        assert deploy(trained, cfg, out, library_dir=lib) == 0
        rec = json.loads((out / "deploy_record.json").read_text())
        return rec["member_order"], capsys.readouterr().err

    same, err = deployed(tmp_path / "x", tmp_path / "out_x")
    assert same["keys"] == trained_with["keys"] and same["differs"] == []
    assert same["training"] == {"library_digest": ident_x.digest}
    assert "order them differently" not in err
    other, err = deployed(tmp_path / "y", tmp_path / "out_y")
    assert other["differs"] == sorted(trained_with["keys"])
    assert "6 of the 6 candidate keys with several members order them" in err
    assert other["differs"][0] in err
    out = tmp_path / "out_none"
    assert deploy(trained, cfg, out) == 0
    assert json.loads((out / "deploy_record.json").read_text())["member_order"] is None

    first = tmp_path / "round_2" / "stage05"
    second = tmp_path / "round_3" / "stage05"
    for src, dst in ((trained, first), (first, second)):
        dst.mkdir(parents=True)
        for name in ("models.pt", "cells.json"):
            shutil.copy2(src / name, dst / name)
        (dst / "carried_forward.json").write_text(json.dumps({"source": str(src)}))
    capsys.readouterr()
    assert deploy(second, cfg, tmp_path / "out_c", library_dir=tmp_path / "y") == 0
    carried = json.loads((tmp_path / "out_c" / "deploy_record.json").read_text())
    assert carried["member_order"]["differs"] == sorted(trained_with["keys"])
    assert "6 of the 6 candidate keys" in capsys.readouterr().err
    (trained / s5.MEMBER_ORDER_FILE).unlink()
    assert deploy(second, cfg, tmp_path / "out_n", library_dir=tmp_path / "y") == 0
    rec = json.loads((tmp_path / "out_n" / "deploy_record.json").read_text())
    assert rec["member_order"]["training"] is None


def test_deployed_serving_decides_the_bank_and_the_gate(capsys):
    v3 = {"catalog": "v3", "serving": {"cascade_keys": 8}, "trained_view_keys": [3, 5]}
    cfg = {"arch": "gfx950"}
    out = s6.deployed_serving(v3, cfg, "the bundle", quiet=False)
    assert out["serving"] == {
        "cascade_keys": 8,
        "bank": False,
        "trained_key_gate": True,
    }
    assert v3["serving"] == {"cascade_keys": 8}
    off = dict(cfg, train={"trained_key_gate": False})
    assert not s6.deployed_serving(v3, off, "b", quiet=True)["serving"][
        "trained_key_gate"
    ]
    banked = dict(cfg, planner={"enabled": True, "bank_library_stems": [STEM]})
    assert s6.deployed_serving(v3, banked, "b", quiet=True)["serving"]["bank"]
    recorded = dict(v3, serving={"bank": False})
    assert not s6.deployed_serving(recorded, banked, "b", quiet=True)["serving"]["bank"]
    old = {"catalog": "v3", "serving": {}}
    assert not s6.deployed_serving(old, cfg, "the old bundle", quiet=False)["serving"][
        "trained_key_gate"
    ]
    assert "the old bundle records no trained view keys" in capsys.readouterr().err
    v2 = {"serving": {"non_temporal_rule": True}}
    assert s6.deployed_serving(v2, cfg, "b", quiet=True) == v2


def test_a_v3_bundle_ships_with_the_trained_key_gate(tmp_path):
    train = tmp_path / "v3" / "stage05"
    write_v3_bundle(train, LABELS, trained_view_keys=[11, 7])
    out = tmp_path / "out"
    assert deploy(train, write_cfg(tmp_path / "cfg.yaml", dtype="bf16"), out) == 0
    staged = json.loads((out / "deploy_record.json").read_text())["staged_model"]
    gated = Path(staged).read_bytes()
    parsed = mlrec.read_model(gated)
    assert parsed["serving"]["trained_key_gate"] and not parsed["serving"]["bank"]
    assert parsed["trained_view_keys"] == [7, 11]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_trained_keys"] == 2 and manifest["serving"]["trained_key_gate"]
    cfg = yaml.safe_load((tmp_path / "cfg.yaml").read_text())
    cfg["train"] = {"trained_key_gate": False}
    (tmp_path / "off.yaml").write_text(yaml.safe_dump(cfg))
    assert deploy(train, tmp_path / "off.yaml", tmp_path / "off") == 0
    data = json.loads((tmp_path / "off" / "deploy_record.json").read_text())[
        "staged_model"
    ]
    assert mlrec.read_model(Path(data).read_bytes())["trained_view_keys"] is None
    tw = pytest.importorskip("tilewright")
    info = tw.describe(tw.load_model_from_memory(gated))
    assert info.trained_key_gate and info.trained_keys == 2


# ── per-cell gate ───────────────────────────────────────────────────────────

SMALL_CELL = [(100 + i, 100, 100, 1) for i in range(8)]
LARGE_CELL = [(4096 + i, 4096, 4096, 1) for i in range(8)]


def audit_record(small, large=0.9, status="ok"):
    per = [[*s, small] for s in SMALL_CELL] + [[*s, large] for s in LARGE_CELL]
    return {
        "status": status,
        "status_text": "complete" if status == "ok" else status,
        "groups": {"exclusive": {"per_gemm": per}} if status == "ok" else None,
        "model_sha256": "ab" * 32,
    }


@pytest.fixture
def shipped(tmp_path):
    """round_0, the bundle the run ships: its stage06 staged model."""
    d = tmp_path / "round_0"
    staged = d / "stage06" / "staged_for_deploy" / "model.tilewright.bin"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"the shipped bundle")
    (d / "stage06" / "deploy_record.json").write_text(
        json.dumps({"staged_model": str(staged)})
    )
    return d


def stub_audit(monkeypatch, candidate, baseline):
    """The candidate's audit record (stage06's run_for_stage) and the
    baseline's (evaluate on the shipped bundle's bytes)."""
    seen = {}

    def run_for_stage(path, data, spec, cfg, **kw):
        path.write_text(json.dumps(candidate))
        return dict(candidate)

    def evaluate(data, spec, cfg, **kw):
        seen["data"], seen["trained"] = data, [p.name for p in spec.trained_rounds]
        return dict(baseline)

    monkeypatch.setattr(s6.auditlib, "run_for_stage", run_for_stage)
    monkeypatch.setattr(s6.auditlib, "evaluate", evaluate)
    return seen


def al_defaults():
    from lib import audit as al

    return dict(al.CELL_GATE_DEFAULTS)


def cell_deploy(tmp_path, trained, out, shipped, record_only=False, quiet=True, **kw):
    from lib import audit as al

    spec = al.Spec(audit_dir=tmp_path / "audit", library_dir=None, library_stem="")
    cells = s6.CellGate(shipped, dict(al.CELL_GATE_DEFAULTS), record_only=record_only)
    cfg = gate_cfg(tmp_path / "cfg.yaml")
    return s6.deploy(
        train_dir=trained,
        config_yaml=cfg,
        out_dir=out,
        quiet=quiet,
        audit=spec,
        cell_gate=cells,
        **kw,
    )


def test_the_per_cell_gate_keeps_a_regressing_bundle_out(
    tmp_path, trained, shipped, monkeypatch, capsys
):
    seen = stub_audit(monkeypatch, audit_record(0.85), audit_record(0.95))
    out = tmp_path / "out"
    assert cell_deploy(tmp_path, trained, out, shipped, quiet=False) == 0
    assert seen == {"data": b"the shipped bundle", "trained": ["round_0"]}
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["deployed"] is False
    assert rec["not_deployed"].startswith(
        "the per-cell gate against round_0's bundle keeps it out: 1 cell(s) "
        "significantly worse beyond 1.0%: exclusive Small|Small|MidK|Bnone x0.8947"
    )
    cells = json.loads((out / s6.CELL_GATE_FILE).read_text())
    assert cells["passed"] is False and cells["n_regressed"] == 1
    assert cells["cell_rules"] == "on" and cells["n_cells_tested"] == 2
    assert cells["groups"]["exclusive"]["p"]["Small|Small|MidK|Bnone"] < 0.05
    assert rec["cell_gate"]["settings"] == al_defaults()
    assert cells["baseline"]["dir"] == str(shipped)
    assert cells["baseline"]["model_source"] == "stage06 staged bundle"
    assert cells["baseline"]["status"] == "ok"
    assert (cells["before"], cells["after"]) == ("round_0", "this bundle")
    assert rec["cell_gate"]["passed"] is False
    assert rec["cell_gate"]["file"] == s6.CELL_GATE_FILE
    assert rec["gate"] is None
    said = capsys.readouterr()
    said = said.out + said.err
    assert "per-cell gate against round_0's bundle: failed" in said
    assert "not deploying: the per-cell gate against round_0's bundle" in said

    stub_audit(monkeypatch, audit_record(0.95), audit_record(0.95))
    good = tmp_path / "good"
    assert cell_deploy(tmp_path, trained, good, shipped) == 0
    rec = json.loads((good / "deploy_record.json").read_text())
    assert rec["deployed"] is True and rec["cell_gate"]["passed"] is True


def test_the_per_cell_gate_records_only_when_asked(
    tmp_path, trained, shipped, monkeypatch
):
    stub_audit(monkeypatch, audit_record(0.85), audit_record(0.95))
    out = tmp_path / "out"
    assert cell_deploy(tmp_path, trained, out, shipped, record_only=True) == 0
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["deployed"] is True and rec["cell_gate"]["record_only"] is True
    cells = json.loads((out / s6.CELL_GATE_FILE).read_text())
    assert cells["passed"] is False and cells["record_only"] is True


def test_a_per_cell_gate_the_audit_cannot_measure_leaves_the_decision(
    tmp_path, trained, shipped, monkeypatch
):
    stub_audit(monkeypatch, audit_record(0.9, status="missing"), audit_record(0.95))
    out = tmp_path / "out"
    assert cell_deploy(tmp_path, trained, out, shipped) == 0
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["deployed"] is True and rec["cell_gate"]["passed"] is None
    assert "the audit set did not measure this bundle (missing)" in (
        rec["cell_gate"]["reason"]
    )
    (shipped / "stage06" / "deploy_record.json").unlink()
    stub_audit(monkeypatch, audit_record(0.9), audit_record(0.95))
    gone = tmp_path / "gone"
    assert cell_deploy(tmp_path, trained, gone, shipped) == 0
    rec = json.loads((gone / "deploy_record.json").read_text())
    assert rec["deployed"] is True and "does not exist" in rec["cell_gate"]["reason"]


def test_the_per_cell_gate_adds_to_the_deploy_gate(
    tmp_path, trained, shipped, gate_dir, monkeypatch
):
    pytest.importorskip("tilewright")
    stub_audit(monkeypatch, audit_record(0.85), audit_record(0.95))
    out = tmp_path / "out"
    gate = s6.Gate(csv_dirs=[gate_dir], bootstrap=0)
    assert cell_deploy(tmp_path, trained, out, shipped, gate=gate) == 0
    measured = json.loads((out / "gate.json").read_text())
    assert measured["passed"] is False
    assert measured["reason"].startswith("no target; the per-cell gate against")
    assert measured["cell_gate"]["n_regressed"] == 1
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["deployed"] is False and rec["gate"]["passed"] is False


def test_the_deploy_record_keeps_the_deploy_gates_own_verdict(
    tmp_path, trained, shipped, gate_dir, monkeypatch
):
    """What a repair of a bundle the per-cell gate keeps out reads
    (lib/cell_merge.py): the deploy gate's verdict before the per-cell
    gate's (`deploy_gate`), the sha256 of the models.pt the deploy packed
    (`models_sha256`), and what a repair changed in the bundle (`repair`,
    with the note that the audit GEMMs that flagged its cells judged it)."""
    pytest.importorskip("tilewright")
    import hashlib

    from lib import cell_merge as cm

    repair = {"schema": "collapse_repair_v1", "outcome": "1 base cell(s) carried"}
    (trained / "metrics.json").write_text(json.dumps({"repair": repair}))
    stub_audit(monkeypatch, audit_record(0.85), audit_record(0.95))
    out = tmp_path / "out"
    gate = s6.Gate(csv_dirs=[gate_dir], bootstrap=0)
    assert cell_deploy(tmp_path, trained, out, shipped, gate=gate) == 0
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["deployed"] is False and rec["cell_gate"]["passed"] is False
    assert rec["deploy_gate"] == {"passed": True, "reason": "no target"}
    want = hashlib.sha256((trained / "models.pt").read_bytes()).hexdigest()
    assert rec["models_sha256"] == want
    assert rec["repair"] == {
        **repair,
        "selected_by_audit": True,
        "selection_note": cm.SELECTION_NOTE,
    }
    # a deploy gate that keeps the bundle out says so itself
    below = tmp_path / "below"
    gate = s6.Gate(csv_dirs=[gate_dir], target=1.01, bootstrap=0)
    assert cell_deploy(tmp_path, trained, below, shipped, gate=gate) == 0
    rec = json.loads((below / "deploy_record.json").read_text())
    assert rec["deployed"] is False and rec["deploy_gate"]["passed"] is False
    assert rec["deploy_gate"]["reason"].endswith(
        "on the gate GEMMs is below the target"
    )
    # a bundle no repair changed records none
    (trained / "metrics.json").write_text(json.dumps({}))
    stub_audit(monkeypatch, audit_record(0.95), audit_record(0.95))
    good = tmp_path / "good"
    assert cell_deploy(tmp_path, trained, good, shipped) == 0
    rec = json.loads((good / "deploy_record.json").read_text())
    assert rec["deployed"] is True and rec["repair"] is None
    assert rec["deploy_gate"] == {"passed": True, "reason": "no gate"}


def test_cell_gate_decision():
    gate = s6.CellGate(Path("run/round_2"), {})
    failed = {"passed": False, "reason": "1 cell(s) worse"}
    assert s6.cell_gate_decision(True, "no gate", failed, gate) == (
        False,
        "the per-cell gate against round_2's bundle keeps it out: 1 cell(s) worse",
    )
    ok, why = s6.cell_gate_decision(False, "below the target", failed, gate)
    assert not ok and why.startswith("below the target; the per-cell gate")
    ok, why = s6.cell_gate_decision(
        True, "no gate", {"passed": True, "reason": "r"}, gate
    )
    assert ok and why == "per cell not worse than round_2's bundle: r"
    ok, why = s6.cell_gate_decision(True, "x", {"passed": None, "reason": "r"}, gate)
    assert (
        ok
        and why == "x; the per-cell gate could not compare it with round_2's bundle: r"
    )
    final = s6.CellGate(Path("run/final_retrain"), {}, record_only=True)
    ok, why = s6.cell_gate_decision(True, "no gate", failed, final)
    assert ok and why.startswith(
        "the per-cell gate against the final retrain in final_retrain would keep it out"
    )


def test_the_per_cell_gate_needs_the_audit_set(tmp_path, trained):
    cfg = gate_cfg(tmp_path / "cfg.yaml")
    argv = ["--train-dir", str(trained), "--config-yaml", str(cfg)]
    argv += ["--out-dir", str(tmp_path / "o"), "--cell-gate-baseline", str(tmp_path)]
    with pytest.raises(SystemExit) as e:
        s6.main(argv)
    assert e.value.code == 2


def test_the_per_cell_gate_options_reach_the_gate(tmp_path, trained, monkeypatch):
    cfg = gate_cfg(tmp_path / "cfg.yaml")
    argv = ["--train-dir", str(trained), "--config-yaml", str(cfg)]
    argv += ["--out-dir", str(tmp_path / "o"), "--cell-gate-baseline", str(tmp_path)]
    argv += ["--audit-dir", str(tmp_path / "audit")]
    seen = {}

    def deploy(**kw):
        seen.update(kw)
        return 0

    monkeypatch.setattr(s6, "deploy", deploy)
    more = ["--cell-gate-alpha", "0.1", "--tail-collapse", "0.25"]
    more += ["--tail-budget", "2", "--cell-gate-groups", "all"]
    assert s6.main(argv + more) == 0
    assert seen["cell_gate"].settings == {
        **al_defaults(),
        "alpha": 0.1,
        "tail_collapse": 0.25,
        "tail_budget": 2,
        "groups": "all",
    }
    for bad in (
        ["--cell-gate-alpha", "1"],
        ["--tail-collapse", "0"],
        ["--tail-budget", "-1"],
        ["--audit-bootstrap", "39"],
    ):
        with pytest.raises(SystemExit) as e:
            s6.main(argv + bad)
        assert e.value.code == 2, bad
    assert s6.main(argv + ["--audit-bootstrap", "10", "--cell-gate-alpha", "0.2"]) == 0
