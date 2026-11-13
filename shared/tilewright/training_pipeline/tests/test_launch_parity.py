# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The launch parity gate of a deploy, with a stand-in for TensileLite's
tests that dumps the twin's own plans (or wrong ones)."""

import json
import stat
import sys
from pathlib import Path

import pytest
import yaml

import stage06_deploy_weights as s6
from bench_fakes import LIBRARY_STEM, solution, write_logic
from lib import launch_parity as lp
from test_stage06_deploy_weights import LABELS, write_v3_bundle

PIPELINE_DIR = Path(__file__).resolve().parent.parent

# DumpsPoolPlans as the twin would dump it: the launch line, each kernel's
# key and its plans of two calls under the uniform summation order without
# a workspace; FAKE_DUMP_WRONG shifts every grid by one.
FAKE_DUMP = f"""#!{sys.executable}
import os, sys
sys.path.insert(0, {str(PIPELINE_DIR)!r})
from pathlib import Path
from lib import dat, features as fs, hipblaslt_launch as hl
from lib.hardware import DATATYPE_VALUE

env = os.environ
lines = Path(env["TILEWRIGHT_LAUNCH_PLANS_POOL"]).read_text().splitlines()
library = Path([ln.split(" ", 1)[1] for ln in lines if ln.startswith("library ")][0])
sids = [int(s) for ln in lines if ln.startswith("pool ") for s in ln.split()[1:]]
kernels = {{k["sol_idx_global"]: k for k in dat.load_library_kernels(library.parent, library.name)}}
arch = env["TILEWRIGHT_LAUNCH_PLANS_PROCESSOR"]
n_cu = int(env["TILEWRIGHT_LAUNCH_PLANS_CU_COUNT"])
version = int(env["TILEWRIGHT_LAUNCH_PLANS_RULE_VERSION"])
out = [f"# launch hipblaslt:{{arch}}:{{version}} cu_count {{n_cu}}"]
ctx = fs.ExecutionContext(uniform_summation_order=True, workspace_bytes=0)
bf16 = DATATYPE_VALUE["bfloat16"]
for pos, sid in enumerate(sids):
    k = kernels[sid]
    a = k["attributes"]
    out.append(f"key {{pos}} {{sid}} {{hl.launch_key(k, a, version)}}")
    for m, n, kk in ((1024, 256, 8192), (512, 512, 64)):
        prob = dict(m=m, n=n, k=kk, batch=1, mi_dtype="bfloat16")
        p = hl.launch_plan(prob, k, a, ctx, n_cu, arch, None, version)
        grid = p.grid_wgs + (1 if env.get("FAKE_DUMP_WRONG") else 0)
        out.append(" ".join(str(f) for f in (
            pos, sid, m, n, kk, 1, bf16, 0, 0, 1, 0, int(p.dynamic_queue), grid,
            p.partial_reduction, k["mt_m"], k["mt_n"], k["mt_k"], k["occupancy"])))
Path(env["TILEWRIGHT_LAUNCH_PLANS_OUT"]).write_text("\\n".join(out) + "\\n")
"""


def write_library(d):
    sk = dict(
        tileProcessingStrategy="StreamK",
        workAssignment="StaticGrid",
        workspaceSizePerElemC=4,
    )
    sols = [solution(5, 0, (256, 256, 32), **sk), solution(7, 1, (128, 128, 64))]
    d.mkdir(parents=True)
    write_logic(d / f"{LIBRARY_STEM}.dat", sols, table=[5, 7])
    return d


@pytest.fixture
def library(tmp_path):
    return write_library(tmp_path / "lib")


@pytest.fixture
def tests_binary(tmp_path):
    path = tmp_path / "TilewrightLaunch_test"
    path.write_text(FAKE_DUMP)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_a_model_of_an_older_rule_needs_no_parity(tmp_path, library):
    rec = lp.gate(lp.DEFAULTS, library, LIBRARY_STEM, "gfx1250", 96, 2, tmp_path / "p")
    assert rec["status"] == "not needed" and not lp.refuses(rec, lp.DEFAULTS)


def test_without_a_tests_build_the_gate_does_not_run(tmp_path, library):
    rec = lp.gate(lp.DEFAULTS, library, LIBRARY_STEM, "gfx1250", 96, 3, tmp_path / "p")
    assert rec["status"] == "not run" and "tests_binary" in rec["why"]
    assert not lp.refuses(rec, lp.DEFAULTS)
    assert lp.refuses(rec, dict(lp.DEFAULTS, required=True))


def test_the_gate_names_the_input_it_lacks(
    tmp_path, library, tests_binary, monkeypatch
):
    settings = dict(lp.DEFAULTS, tests_binary=str(tests_binary), required=True)
    rec = lp.gate(settings, None, LIBRARY_STEM, "gfx1250", 40, 3, tmp_path / "p")
    assert rec["status"] == "not run" and lp.refuses(rec, settings)
    assert LIBRARY_STEM in rec["why"] and "--library-dir" in rec["why"]
    cfg = {"deploy": {"launch_parity": settings}}
    v3 = {"provenance": {"launch_model": "hipblaslt:gfx1250:3"}}
    rec = s6.parity_record(cfg, v3, library, LIBRARY_STEM, None, tmp_path / "q")
    assert rec["status"] == "not run" and lp.refuses(rec, settings)
    assert "--hardware-device" in rec["why"] and "runtime.devices" in rec["why"]

    def unreadable(cfg, device_index=0, topology=None):
        raise ValueError("the KFD node gives no CU count")

    monkeypatch.setattr(s6.hardware, "device_hardware", unreadable)
    rec = s6.parity_record(cfg, v3, library, LIBRARY_STEM, 3, tmp_path / "r")
    assert rec["status"] == "not run" and lp.refuses(rec, settings)
    assert "bench device 3" in rec["why"] and "no CU count" in rec["why"]


def test_the_bench_device_is_the_option_else_the_config_s_first_device():
    cfg = {"runtime": {"devices": [3, 1]}}
    assert s6.bench_device(cfg, None) == 3
    assert s6.bench_device(cfg, 0) == 0
    assert s6.bench_device({}, None) is None
    assert s6.bench_device({"runtime": {"devices": []}}, None) is None


def deploy_config(d, tests_binary, runtime=None):
    cfg = {
        "config_id": "parity_test",
        "arch": "gfx1250",
        "hipblaslt": {
            "transA": "T",
            "transB": "N",
            "library_stem": LIBRARY_STEM,
            "logic_name": "gfx1250_Cijk_Alik_Bljk_Fake_UserArgs",
        },
        "hardware": {"n_cu": 40, "lds_bytes": 65536, "l2_bytes": 4 << 20},
        "deploy": {
            "weight_dtype": "int8",
            "launch_parity": {"tests_binary": str(tests_binary), "required": True},
        },
    }
    if runtime is not None:
        cfg["runtime"] = runtime
    d.mkdir(parents=True, exist_ok=True)
    (d / "deploy.yaml").write_text(yaml.safe_dump(cfg))
    return d / "deploy.yaml"


def v3_round(tmp_path):
    train = tmp_path / "round_0" / "stage05"
    write_v3_bundle(train, LABELS, provenance={"launch_model": "hipblaslt:gfx1250:3"})
    return train


def test_a_deploy_without_gates_checks_the_pool_of_its_config_s_bench_build(
    tmp_path, tests_binary, monkeypatch
):
    """Without --library-dir and --hardware-device (a round without gate
    GEMMs or an audit set) the gate reads the library of the config's bench
    build, on the CUs of its first bench device."""
    build = tmp_path / "build"
    write_library(build / "Tensile" / "library" / "gfx1250")
    seen = []
    real = s6.hardware.device_hardware

    def device_hardware(cfg, device_index=0, topology=None):
        seen.append(device_index)
        return real(cfg, device_index, topology)

    monkeypatch.setattr(s6.hardware, "device_hardware", device_hardware)
    runtime = {"rocm_libraries_root": str(tmp_path), "build_dir": str(build)}
    cfg = deploy_config(tmp_path, tests_binary, dict(runtime, devices=[3, 1]))
    out = tmp_path / "stage06"
    argv = ["--train-dir", str(v3_round(tmp_path)), "--config-yaml", str(cfg)]
    assert s6.main(argv + ["--out-dir", str(out), "--quiet"]) == 0
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["launch_parity"]["status"] == "ok", rec["launch_parity"]
    assert rec["deployed"] is True and "not_deployed" not in rec
    assert (out / "launch_parity" / f"{LIBRARY_STEM}.plans").is_file()
    assert seen and set(seen) == {3}


def test_a_deploy_that_cannot_run_the_gate_is_refused_and_says_why(
    tmp_path, tests_binary
):
    build = tmp_path / "build"
    write_library(build / "Tensile" / "library" / "gfx1250")
    train = v3_round(tmp_path)
    runtime = {"rocm_libraries_root": str(tmp_path), "build_dir": str(build)}
    cases = {
        "no_library": (None, "--library-dir"),
        "no_device": (runtime, "runtime.devices"),
    }
    for name, (rt, lacks) in cases.items():
        cfg = deploy_config(tmp_path / name, tests_binary, rt)
        out = tmp_path / name / "stage06"
        assert s6.deploy(train_dir=train, config_yaml=cfg, out_dir=out, quiet=True) == 0
        rec = json.loads((out / "deploy_record.json").read_text())
        assert rec["launch_parity"]["status"] == "not run", name
        assert lacks in rec["launch_parity"]["why"], name
        assert rec["deployed"] is False and lacks in rec["not_deployed"], name


def test_the_engine_s_plans_of_the_pool_must_be_the_twin_s(
    tmp_path, library, tests_binary, monkeypatch
):
    settings = dict(lp.DEFAULTS, tests_binary=str(tests_binary), env=["HIPSHIM_CUS=96"])
    rec = lp.gate(settings, library, LIBRARY_STEM, "gfx1250", 96, 3, tmp_path / "ok")
    assert rec["status"] == "ok", rec
    assert (tmp_path / "ok" / f"{LIBRARY_STEM}.plans").is_file()
    assert not lp.refuses(rec, settings)
    monkeypatch.setenv("FAKE_DUMP_WRONG", "1")
    rec = lp.gate(settings, library, LIBRARY_STEM, "gfx1250", 96, 3, tmp_path / "bad")
    assert rec["status"] == "mismatch" and lp.refuses(rec, settings)
    assert any("differ" in line for line in rec["summary"])
    broken = dict(settings, tests_binary=str(tmp_path / "missing"))
    rec = lp.gate(broken, library, LIBRARY_STEM, "gfx1250", 96, 3, tmp_path / "err")
    assert rec["status"] == "error" and lp.refuses(rec, broken)


def test_the_library_options_name_the_chip_of_a_stem():
    args = lp.library_args(Path("/l"), "TensileLibrary_X_ID75a0_gfx950", "gfx950", 256)
    assert args[-2:] == ["--chip-id", "0x75a0"]
    assert "--chip-id" not in lp.library_args(Path("/l"), LIBRARY_STEM, "gfx1250", 96)


def test_the_deploy_gates_a_model_by_the_launch_model_it_records(tmp_path, library):
    cfg = {
        "hardware": {"n_cu": 96, "lds_bytes": 65536, "l2_bytes": 4 << 20},
        "deploy": {"launch_parity": {"required": True}},
    }
    v3 = {"provenance": {"launch_model": "hipblaslt:gfx1250:3"}}
    rec = s6.parity_record(cfg, v3, library, LIBRARY_STEM, 0, tmp_path / "p")
    assert (rec["status"], rec["rule_version"]) == ("not run", 3)
    assert lp.refuses(rec, lp.settings_of(cfg))
    v2 = {"provenance": {"launch_model": "hipblaslt:gfx1250:2"}}
    rec = s6.parity_record(cfg, v2, library, LIBRARY_STEM, 0, tmp_path / "p")
    assert rec["status"] == "not needed"
    assert s6.parity_record(cfg, {}, library, LIBRARY_STEM, 0, tmp_path / "p")[
        "status"
    ] == ("not needed")
