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
MX_STEM = "TensileLibrary_F8F8_BF8_HA_MXAE8B32_MXBE8B32_Bias_SAV_UA_Type_F8B_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1250"
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
    parsed = mlrec.read_model(data)
    assert parsed["format"] == "MLREC_v2" and parsed["weight_dtype_name"] == "int8"
    assert [c["label"] for c in parsed["cells"]] == sorted(LABELS)
    manifest = (out / "manifest.json").read_text()
    assert "sel_eff" not in manifest and "best_epoch" not in manifest
    install = (out / "staged_for_deploy" / "INSTALL.md").read_text()
    for stale in ("origami", "backends/hipblaslt", "tilewright_weights.bin", "%"):
        assert stale not in install


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
        arch="gfx1250v0",
        stem=MX_STEM,
        targets=["gfx1250v0", "gfx1250", "gfx1250-strict"],
    )
    out = tmp_path / "out"
    assert deploy(trained, cfg, out, repo=repo, apply_patch=True) == 0
    blobs = set()
    for d, stem in (
        ("gfx1250v0", MX_STEM),
        ("gfx1250", MX_STEM),
        ("gfx1250-strict", MX_STEM[: -len("gfx1250")] + "gfx1250-strict"),
    ):
        entries = s6.parse_index(
            (weights_dir(repo, d) / "tilewright_index").read_text()
        )
        assert entries.as_dict() == {stem: f"{stem}.tilewright.bin"}
        blobs.add((weights_dir(repo, d) / f"{stem}.tilewright.bin").read_bytes())
    assert len(blobs) == 1
    assert mlrec.read_model(blobs.pop())["arch"] == "gfx1250v0"


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
