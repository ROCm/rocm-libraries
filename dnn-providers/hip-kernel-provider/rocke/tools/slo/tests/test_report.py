# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`slo.py report`: JSON fields, scope, exit codes, and a run on the real tree."""
import json
import time

from conftest import ROCKE, kernel_text, write

import slo

KERNEL = "library/kernels/common/k.py"


def cli(root, *args):
    return slo.main([*args[:1], "--root", str(root), *args[1:]])


def test_report_json_fields_and_scope(fake_rocke, tmp_path):
    write(fake_rocke, KERNEL, kernel_text(n_raw=3))
    write(
        fake_rocke, "library/builders/drv.py", "def run(b):\n    return b.add(1, 2)\n"
    )
    out = tmp_path / "r.json"
    assert cli(fake_rocke, "report", "--json", str(out)) == 0
    files = json.loads(out.read_text())["files"]
    assert set(files) == {KERNEL}  # builders are outside the default scope
    need = {
        "loc",
        "sloc",
        "n_funcs",
        "longest_fn",
        "longest_fn_name",
        "raw_ops",
        "raw_per_100",
        "helper_calls",
        "hcr",
        "ops",
        "helpers",
        "funcs",
        "patterns",
        "spec_bools_max",
        "overlap_max",
    }
    assert need <= set(files[KERNEL])
    assert files[KERNEL]["ops"] == {"add": 3}
    assert (
        cli(fake_rocke, "report", "--json", str(out), "--scope", "library/builders")
        == 0
    )
    assert set(json.loads(out.read_text())["files"]) == {"library/builders/drv.py"}


def test_report_exit_codes(fake_rocke, tmp_path):
    write(fake_rocke, KERNEL, kernel_text())
    assert cli(fake_rocke, "report") == 0
    assert slo.main(["report", "--root", str(tmp_path)]) == 2  # not a rocke folder
    write(fake_rocke, KERNEL, "def broken(:\n")
    assert cli(fake_rocke, "report") == 2  # a file that does not parse


def test_real_tree_measures_cleanly_and_fast(tmp_path):
    out = tmp_path / "r.json"
    t0 = time.monotonic()
    assert slo.main(["report", "--json", str(out)]) == 0
    assert time.monotonic() - t0 < 10  # speed budget for the static checks
    doc = json.loads(out.read_text())
    assert doc["errors"] == []
    assert len(doc["files"]) > 50
    assert doc["root"] == str(ROCKE)
