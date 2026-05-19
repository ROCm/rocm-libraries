#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the hipDNN frontend Python wheel."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Path to projects/hipdnn/python",
    )
    parser.add_argument(
        "--ext-dir",
        type=Path,
        required=True,
        help="Directory containing the pre-built hipdnn_frontend_python extension",
    )
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        required=True,
        help="Directory where the built wheel will be written",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    ext_dir = args.ext_dir.resolve()
    wheel_dir = args.wheel_dir.resolve()

    if not (source_dir / "pyproject.toml").exists():
        raise RuntimeError(f"Missing pyproject.toml in {source_dir}")

    import glob

    if not glob.glob(str(ext_dir / "hipdnn_frontend_python*")):
        raise RuntimeError(f"No hipdnn_frontend_python extension found in {ext_dir}")

    shutil.rmtree(wheel_dir, ignore_errors=True)
    wheel_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HIPDNN_EXT_DIR"] = str(ext_dir)

    cmd = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(wheel_dir),
        str(source_dir),
    ]

    print("::: Building hipdnn-frontend wheel")
    print(f"::: HIPDNN_EXT_DIR={ext_dir}")
    print("::: " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=source_dir, env=env)

    wheels = sorted(wheel_dir.glob("hipdnn_frontend-*.whl"))
    if not wheels:
        raise RuntimeError(f"No hipdnn_frontend wheel produced in {wheel_dir}")

    print(f"::: Built wheel: {wheels[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
