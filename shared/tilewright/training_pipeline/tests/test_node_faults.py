# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The GPU faults that tell against a node, their runtime records, and the
store of the nodes they hold."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from bench_fakes import LAUNCH_FAILURE_LINES, MEMORY_FAULT_LINES
from lib import node_faults as nf

PIPELINE_DIR = Path(__file__).resolve().parent.parent


def test_the_faults_a_bench_reports():
    fault = "\n".join(MEMORY_FAULT_LINES)
    assert nf.fault_of(fault) == nf.ILLEGAL_ADDRESS == 700
    assert nf.fault_of("\n".join(LAUNCH_FAILURE_LINES)) == nf.LAUNCH_FAILURE == 719
    assert nf.fault_of("hipblaslt error: hipErrorLaunchFailure") == 719
    assert nf.fault_of("hip error code: 'hipErrorIllegalAddress'") == 700
    assert nf.fault_of(fault + "\n" + LAUNCH_FAILURE_LINES[0]) == 719
    for text in ("", "error: NO solution found!", "M=700 N=719 K=64", "us: 700.5"):
        assert nf.fault_of(text) is None, text


def test_the_runtime_record_of_a_fault():
    text = "\n".join(
        [
            "--Solution name: Cijk_Ailk_Bljk_fake_MT256x256x64",
            "Dispatch packet: header=0x1402 grid=(1024,1,1) workgroup=(256,1,1)",
            *MEMORY_FAULT_LINES,
            "--Solution name: Cijk_after_the_fault",
        ]
    )
    rec = nf.runtime_record(text)
    assert rec["code"] == 700
    assert rec["kernel"] == "Cijk_Ailk_Bljk_fake_MT256x256x64"
    assert rec["address"] == "0x7f3a00001000"
    assert rec["packet"] == [
        "Dispatch packet: header=0x1402 grid=(1024,1,1) workgroup=(256,1,1)"
    ]
    assert rec["lines"][0].startswith("Memory access fault by GPU node-2")
    assert nf.runtime_record("nothing")["code"] is None


def test_a_marked_node_is_held_until_its_reset(tmp_path, monkeypatch):
    store = nf.NodeStore(tmp_path / "store")
    assert store.held("n1") == [] and store.faults("n1") == []
    monkeypatch.setattr(nf.time, "time", lambda: 1000.0)
    first = store.mark("n1", {"code": 719, "why": "error 719 (launch failure)"})
    assert (first["node"], first["epoch"]) == ("n1", 1000.0)
    assert first["at"] == "1970-01-01T00:16:40Z"
    assert [f["code"] for f in store.held("n1")] == [719]
    assert store.held("n2") == []
    text = nf.describe_hold("n1", store.held("n1"), store.root)
    assert "node n1 is held by GPU faults since its last reset (1 kept)" in text
    assert "node_faults.py reset n1" in text
    reset = store.reset("n1", "rebooted, checked by someone")
    os.utime(reset, (1500.0, 1500.0))
    assert store.held("n1") == [] and len(store.faults("n1")) == 1
    monkeypatch.setattr(nf.time, "time", lambda: 2000.0)
    store.mark("n1", {"code": 700, "why": "x"})
    assert [f["code"] for f in store.held("n1")] == [700]
    assert [f["code"] for f in store.faults("n1")] == [719, 700]
    (store.root / "n3.json").write_text("not json")
    assert store.faults("n3") == []


def test_the_store_a_run_uses(tmp_path, monkeypatch):
    monkeypatch.setenv(nf.STORE_VAR, str(tmp_path / "env"))
    assert nf.store_dir("~/x") == Path("~/x").expanduser()
    assert nf.store_dir() == tmp_path / "env"
    monkeypatch.delenv(nf.STORE_VAR)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert nf.store_dir() == tmp_path / "state" / "tilewright" / "node_faults"
    assert nf.node_name() and "." not in nf.node_name()
    assert nf.RERUN_ENV == {"HIP_LAUNCH_BLOCKING": "1", "TENSILE_DB": "0x200040"}


def test_only_a_person_resets_a_node(tmp_path):
    script = PIPELINE_DIR / "scripts" / "node_faults.py"
    store = nf.NodeStore(tmp_path)
    store.mark("n1", {"code": 719, "why": "error 719 (launch failure)"})

    def run(*argv):
        return subprocess.run(
            [sys.executable, str(script), *argv, "--store", str(tmp_path)],
            capture_output=True,
            text=True,
        )

    shown = run("show", "n1")
    assert shown.returncode == 0
    assert "node n1 is held by GPU faults since its last reset (1 kept)" in shown.stdout
    assert run("reset", "n1", "--note", " ").returncode != 0
    assert store.held("n1")
    done = run("reset", "n1", "--note", "rebooted and checked")
    assert done.returncode == 0 and "n1: reset" in done.stdout
    assert store.reset_path("n1").is_file()
    os.utime(store.reset_path("n1"), (2e9, 2e9))
    assert store.held("n1") == [] and "n1: not held" in run("show", "n1").stdout


@pytest.mark.parametrize("line", MEMORY_FAULT_LINES[:1] + LAUNCH_FAILURE_LINES)
def test_the_fault_lines_are_fatal_to_the_bench(line):
    from lib.bench_log import fatal_marker

    assert fatal_marker(line) is not None
