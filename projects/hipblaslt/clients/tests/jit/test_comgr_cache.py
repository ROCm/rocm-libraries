# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check that JIT processes leave comgr's on-disk cache off unless the user set it.

Each case runs the code-object test's comgr cache check in a fresh process with
private XDG_CACHE_HOME and HOME directories, because comgr fixes its cache
setting at the first cached action in a process.
"""

import argparse
import os
from pathlib import Path
import subprocess


CASES = (
    # name, environment, expected cache directory, reported setting
    ("jit-mode", {"HIPBLASLT_JIT": "1"}, "absent", "disabled at library load"),
    ("control", {"AMD_COMGR_CACHE": "1"}, "present", "comgr's default"),
    (
        "user-value",
        {"HIPBLASLT_JIT": "1", "AMD_COMGR_CACHE": "1"},
        "present",
        "the user's AMD_COMGR_CACHE",
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("fresh_output", type=Path)
    parser.add_argument("--architecture", default="gfx950")
    args = parser.parse_args()
    args.fresh_output.mkdir(parents=True, exist_ok=False)
    base = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HIPBLASLT_JIT", "AMD_COMGR_CACHE", "AMD_COMGR_CACHE_DIR")
    }
    for name, overrides, expected, setting in CASES:
        root = args.fresh_output.resolve() / name
        (root / "xdg").mkdir(parents=True)
        (root / "home").mkdir()
        env = dict(base, XDG_CACHE_HOME=str(root / "xdg"), HOME=str(root / "home"))
        env.update(overrides)
        command = [
            str(args.executable.resolve(strict=True)),
            "--target",
            args.architecture,
            "--out",
            str(root / "results"),
            "--only",
            "h_comgr_cache",
            "--expect-comgr-cache",
            expected,
        ]
        result = subprocess.run(
            command, env=env, text=True, capture_output=True, timeout=240
        )
        log = result.stdout + result.stderr
        (root / "run.log").write_text(log)
        assert result.returncode == 0, (name, result.returncode, log)
        assert "h_comgr_cache_environment" in log and setting in log, (name, log)
        print(f"PASS {name}: {setting}, cache directory {expected}", flush=True)
    print(f"PASS: {len(CASES)} comgr cache cases")


if __name__ == "__main__":
    main()
