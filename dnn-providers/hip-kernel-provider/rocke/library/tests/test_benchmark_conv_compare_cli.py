# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# benchmarks/common/benchmark_conv_compare.py builds its command line: every
# option is declared once (a duplicate makes argparse refuse the whole CLI).
# Host-only.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

LIBRARY = Path(__file__).resolve().parents[1]


def test_help_builds_and_declares_two_stage_once():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(LIBRARY)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    r = subprocess.run(
        [sys.executable, "-m", "benchmarks.common.benchmark_conv_compare", "--help"],
        capture_output=True,
        text=True,
        cwd=LIBRARY,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    opts = [ln.split()[0] for ln in r.stdout.splitlines() if ln.startswith("  --")]
    assert opts.count("--two-stage") == 1
    assert len(opts) == len(set(opts))
