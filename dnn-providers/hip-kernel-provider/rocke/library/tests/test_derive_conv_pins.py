# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# benchmarks/common/derive_conv_pins.py: pins from benchmark_conv_compare run
# directories (result rows + logs). Host-only.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "common" / "derive_conv_pins.py"
CMD = "MIOpenDriver convbfp16 -n 1 -c 64 -H 8 -W 8 -k 64 -y 3 -x 3 -F 1"


def _run_dir(tmp: Path, name: str, winner: str, finalists: dict) -> Path:
    d = tmp / name
    d.mkdir()
    row = {"implicit_kernel": winner, "implicit_ms": finalists[winner], "miopen_cmd": CMD}
    (d / "fwd_s1.jsonl").write_text(json.dumps(row) + "\n")
    lines = [f"  [ {i}/9]  100.0 TFLOPS  {ms:8.3f} ms  {lab}" for i, (lab, ms) in enumerate(finalists.items(), 1)]
    lines.append("  [ 1/9] probe    0.100 ms  probe-only-label")
    (d / "fwd_s1.jsonl.log").write_text("\n".join(lines) + "\n")
    return d


def _derive(tmp: Path, *dirs, extra=()):
    out = tmp / "pins.json"
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(out), *extra, *map(str, dirs)], check=True)
    return json.loads(out.read_text())["cases"]


def test_single_run_pins_its_winner(tmp_path):
    a = _run_dir(tmp_path, "a", "k1 @split_k=4", {"k1 @split_k=4": 1.0, "k2": 1.1})
    pins = _derive(tmp_path, a)
    assert pins["fwd_s1"]["kernel"] == "k1 @split_k=4"
    assert pins["fwd_s1"]["calls"] == 2 and pins["fwd_s1"]["cmd"] == CMD


def test_several_runs_pin_the_lowest_median(tmp_path):
    # k1 won run a by luck; k2 is faster in the median over three runs.
    a = _run_dir(tmp_path, "a", "k1", {"k1": 0.90, "k2": 0.95})
    b = _run_dir(tmp_path, "b", "k2", {"k1": 1.10, "k2": 0.96})
    c = _run_dir(tmp_path, "c", "k2", {"k1": 1.05, "k2": 0.94})
    pins = _derive(tmp_path, a, b, c, extra=("--alts", "3", "--alts-window", "0.2"))
    assert pins["fwd_s1"]["kernel"] == "k2"
    assert pins["fwd_s1"]["alts"][0] == "k2" and "k1" in pins["fwd_s1"]["alts"]
    assert "probe-only-label" not in pins["fwd_s1"]["alts"]
