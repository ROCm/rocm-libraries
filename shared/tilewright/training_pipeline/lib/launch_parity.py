# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The launch parity gate of a deploy (stage06): a model of launch rule
version 3 ranks with the plans the engine (TensileLite's tilewright launch
rule) makes for its pool, so it is deployed only when the engine's plans
on that pool are the twin's (lib/hipblaslt_launch.py), the plans it
trained with.

`gate` checks it host-only, with a build of TensileLite's tests
(`deploy.launch_parity.tests_binary`) on a stand-in HIP runtime
(`hip_lib`, prepended to LD_LIBRARY_PATH, and `env`, its KEY=VALUE
settings): scripts/check_launch_plans.py writes the pool's file, the
tests' TilewrightLaunchTest.DumpsPoolPlans dumps the engine's plans of
`samples` drawn calls per kernel under the rule version, and
check_launch_plans.py compares them with the twin's. Its `status`:

  ok          the plans agree
  mismatch    they do not: the deploy is refused
  error       a step failed: the deploy is refused
  not run     an input is missing, which `why` names: the tests build, a
              library holding the pool, or the bench device's values;
              refused with `required`
  not needed  the model's rule is older than version 3
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PIPELINE_DIR = Path(__file__).resolve().parent.parent
CHECK_SCRIPT = PIPELINE_DIR / "scripts" / "check_launch_plans.py"
DUMP_FILTER = "TilewrightLaunchTest.DumpsPoolPlans"
DEFAULTS: Dict[str, Any] = {
    "tests_binary": None,
    "hip_lib": None,
    "env": [],
    "samples": 8,
    "timeout_s": 3600.0,
    "required": False,
}
# The statuses that keep a model out of the deploy (`not run` only with
# `required`).
REFUSING = ("mismatch", "error")
_CHIP_RE = re.compile(r"_ID([0-9a-fA-F]+)_")


def settings_of(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """`deploy.launch_parity` over DEFAULTS."""
    given = (cfg.get("deploy") or {}).get("launch_parity") or {}
    return {**DEFAULTS, **dict(given)}


def missing_input(
    settings: Mapping[str, Any], library_dir: Optional[Path], stem: str
) -> Optional[str]:
    """Why the gate cannot run with `settings` on the pool `stem` of
    `library_dir`, naming the input it lacks; None when it can."""
    if not settings.get("tests_binary"):
        return (
            "no host-only build of TensileLite's tests "
            "(deploy.launch_parity.tests_binary)"
        )
    if library_dir is None:
        return (
            f"no Tensile library holding {stem} to read the pool from (stage06's "
            f"--library-dir, or the library of the config's bench build)"
        )
    return None


def not_run(version: int, out_dir: Path, why: str) -> Dict[str, Any]:
    """The record of a gate of rule `version` that cannot run, `why` naming
    the input it lacks."""
    return {
        "rule_version": int(version),
        "dir": str(out_dir),
        "status": "not run",
        "why": why,
    }


def refuses(record: Mapping[str, Any], settings: Mapping[str, Any]) -> bool:
    """Whether the gate's `record` keeps the model out."""
    status = record.get("status")
    return status in REFUSING or (
        status == "not run" and bool(settings.get("required"))
    )


def library_args(library_dir: Path, stem: str, arch: str, n_cu: int) -> List[str]:
    """check_launch_plans.py's options of the pool of `stem` in
    `library_dir` on `arch` with `n_cu` CUs (the chip id a stem names
    with `_ID<hex>_`)."""
    args = [
        "--library-dir",
        str(library_dir),
        "--library-stem",
        stem,
        "--processor",
        arch,
        "--cu-count",
        str(int(n_cu)),
    ]
    m = _CHIP_RE.search(stem)
    if m:
        args += ["--chip-id", f"0x{m.group(1).lower()}"]
    return args


def _run(cmd: List[str], log: Path, env: Mapping[str, str], timeout: float) -> int:
    with log.open("w") as f:
        try:
            return subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(env), timeout=timeout
            ).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n*** timed out after {timeout:g}s ***\n")
            return -1
        except OSError as e:
            f.write(f"\n*** {e} ***\n")
            return -1


def gate(
    settings: Mapping[str, Any],
    library_dir: Optional[Path],
    stem: str,
    arch: str,
    n_cu: int,
    version: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """The gate's record (see the module docstring) for a model of rule
    `version` served from the pool `stem` of `library_dir`, its files in
    `out_dir`."""
    base = {"rule_version": int(version), "dir": str(out_dir)}
    if int(version) < 3:
        return {**base, "status": "not needed", "why": "a rule older than version 3"}
    why = missing_input(settings, library_dir, stem)
    if why is not None:
        return not_run(version, out_dir, why)
    binary = settings["tests_binary"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pool, plans = out_dir / f"{stem}.pool", out_dir / f"{stem}.plans"
    lib = library_args(Path(library_dir), stem, arch, n_cu)
    timeout = float(settings.get("timeout_s") or DEFAULTS["timeout_s"])
    env = dict(os.environ)
    py = [sys.executable, "-B", str(CHECK_SCRIPT)]
    if _run([*py, "pool", str(pool), *lib], out_dir / "pool.log", env, timeout) != 0:
        return {**base, "status": "error", "why": "the pool file could not be written"}
    dump_env = dict(env)
    for item in settings.get("env") or ():
        key, _, value = str(item).partition("=")
        dump_env[key] = value
    if settings.get("hip_lib"):
        dump_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            p for p in (str(settings["hip_lib"]), env.get("LD_LIBRARY_PATH", "")) if p
        )
    dump_env.update(
        TILEWRIGHT_LAUNCH_PLANS_POOL=str(pool),
        TILEWRIGHT_LAUNCH_PLANS_OUT=str(plans),
        TILEWRIGHT_LAUNCH_PLANS_PROCESSOR=arch,
        TILEWRIGHT_LAUNCH_PLANS_CU_COUNT=str(int(n_cu)),
        TILEWRIGHT_LAUNCH_PLANS_RULE_VERSION=str(int(version)),
        TILEWRIGHT_LAUNCH_PLANS_SAMPLES=str(int(settings.get("samples") or 1)),
    )
    dumped = _run(
        [str(binary), f"--gtest_filter={DUMP_FILTER}"],
        out_dir / "dump.log",
        dump_env,
        timeout,
    )
    if dumped != 0 or not plans.is_file():
        return {
            **base,
            "status": "error",
            "why": f"the engine's dump failed (rc={dumped})",
        }
    log = out_dir / "compare.log"
    rc = _run(
        [*py, "compare", str(plans), *lib, "--rule-version", str(int(version))],
        log,
        env,
        timeout,
    )
    tail = log.read_text(errors="replace").strip().splitlines()[-5:]
    if rc == 0:
        return {**base, "status": "ok", "summary": tail}
    if rc == 1:
        return {
            **base,
            "status": "mismatch",
            "why": "the engine's plans on the pool differ from the twin's",
            "summary": tail,
        }
    return {
        **base,
        "status": "error",
        "why": f"the comparison failed (rc={rc})",
        "summary": tail,
    }
