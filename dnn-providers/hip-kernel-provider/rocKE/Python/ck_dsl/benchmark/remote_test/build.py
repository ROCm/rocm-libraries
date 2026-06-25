# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Local cross-arch build: run the registered example with
``--no-verify --output-dir <stage>`` so it produces HSACO + manifest.json
without trying to launch on this machine (which probably lacks the target GPU).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from . import config


def _parse_shape(args: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    keys = {"--m": "m", "--n": "n", "--k": "k"}
    it = iter(args)
    for tok in it:
        if tok in keys:
            try:
                out[keys[tok]] = int(next(it))
            except (StopIteration, ValueError):
                pass
    return out


def build_arch(
    arch: str, *, clean: bool = True, extra_args: List[str] | None = None
) -> Path:
    if arch not in config.ARCHES:
        raise KeyError(f"unknown arch {arch!r}; have {list(config.ARCHES)}")
    profile = config.ARCHES[arch]
    out_dir = config.stage_dir(arch)
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{config.CK_PY_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    ).rstrip(os.pathsep)

    cmd: List[str] = [
        sys.executable,
        "-m",
        profile.example_module,
        "--no-verify",
        "--output-dir",
        str(out_dir),
        *profile.example_args,
        *(extra_args or []),
    ]
    print(f"[build:{arch}] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env, cwd=config.CK_PY_ROOT)

    # Sanity: at least one .hsaco + manifest.json must exist.
    hsacos = list(out_dir.glob("*.hsaco"))
    manifest = out_dir / "manifest.json"
    if not hsacos or not manifest.exists():
        raise RuntimeError(
            f"build for {arch} produced no artifacts under {out_dir}: "
            f"hsacos={hsacos}, manifest_exists={manifest.exists()}"
        )
    # Sidecar with run hints so the remote runner doesn't need to re-parse args.
    shape = _parse_shape([*profile.example_args, *(extra_args or [])])
    (out_dir / "run_spec.json").write_text(
        json.dumps(
            {
                "arch": arch,
                "hsaco": hsacos[0].name,
                "manifest": "manifest.json",
                "shape": shape,
            },
            indent=2,
        )
    )
    print(f"[build:{arch}] OK -> {out_dir} ({len(hsacos)} hsaco, manifest, run_spec)")
    return out_dir
