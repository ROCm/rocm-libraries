#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure which LLVM backend options (``-mllvm``) the loaded compiler accepts.

The read side is ``rocke.runtime.compiler_options``; see it for why the answer
has to be known before COMGR is called. This tool writes its table,
``python/rocke/runtime/data/compiler_options.json``.

For every option in the table (plus any ``--option``), a child process loads
the COMGR this process resolved and compiles an empty kernel with
``-O3 -mllvm <option>``. The child goes straight to the COMGR pipeline, past
the table check, because measuring is what fills the table:

    child exits 0                              -> accepted
    child output has "Unknown command line"    -> not accepted (the parser
                                                  exited the child)
    anything else                              -> no answer; the tool stops
                                                  and records nothing

A host can only measure its own compiler, so a run writes one column, keyed by
the loaded compiler's LLVM major version, and leaves the others alone.

Usage:
  python rocke/platform/tools/gen_compiler_options.py            # update this compiler's column
  python rocke/platform/tools/gen_compiler_options.py --check    # fail if the column differs
  python rocke/platform/tools/gen_compiler_options.py --option=-amdgpu-new-option
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parent  # tools -> rocke/platform
PYTHON_ROOT = ROCKE / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from rocke.runtime import comgr
from rocke.runtime import compiler_options as table

# The limit on one probe compile, which runs in its own child process.
PROBE_TIMEOUT_S = 120

# Runs in the child. It compiles through the private pipeline on purpose: the
# public entry points refuse an option the table does not know yet.
_CHILD = """
import sys
from rocke.runtime import comgr
isa = comgr.supported_isas()[0]
ir = "define amdgpu_kernel void @rocke_option_probe() {\\n  ret void\\n}\\n"
comgr._build_hsaco((ir,), isa=isa, options=["-O3", "-mllvm", sys.argv[1]])
print("accepted")
"""

_UNKNOWN = "Unknown command line argument"


def measure(option: str, lib_path: str) -> tuple:
    """``(accepted, evidence)`` for one option on the COMGR at ``lib_path``."""
    env = dict(os.environ, ROCKE_COMGR_LIB=lib_path, AMD_COMGR_CACHE="0")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(PYTHON_ROOT), env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, option],
        capture_output=True,
        text=True,
        env=env,
        timeout=PROBE_TIMEOUT_S,
        check=False,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode == 0 and "accepted" in proc.stdout:
        return True, ""
    for line in output.splitlines():
        if _UNKNOWN in line:
            # Keep the diagnostic itself: drop the program name the parser
            # prints before it and the "Try: --help" advice after it.
            return False, line[line.index(_UNKNOWN) :].split("  Try:")[0].strip()
    raise SystemExit(
        f"no answer for {option}: the probe exited {proc.returncode} without "
        f"saying whether the option is known:\n{output[-2000:]}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="measure and fail if the committed column for this compiler differs",
    )
    ap.add_argument(
        "--option",
        action="append",
        default=[],
        help="also measure this option (name only, e.g. -amdgpu-mfma-vgpr-form)",
    )
    args = ap.parse_args()

    info = comgr.loaded_compiler_info()
    if info is None or info.llvm_version is None:
        raise SystemExit(
            f"cannot identify the loaded compiler: {info and info.describe()}"
        )
    lib_path = info.comgr_path or info.requested_comgr
    if not lib_path:
        raise SystemExit(
            f"the loaded COMGR has no path to hand the probe: {info.describe()}"
        )
    major = str(info.llvm_version[0])
    verified_on = "LLVM " + ".".join(map(str, info.llvm_version))

    doc = json.loads(table.TABLE_PATH.read_text(encoding="utf-8"))
    options = doc["options"]
    names = list(options) + [o for o in args.option if o not in options]
    drift = []
    for name in names:
        accepted, evidence = measure(name, lib_path)
        print(f"{name}: {'accepted' if accepted else 'not accepted'} on {verified_on}")
        committed = options.get(name, {}).get(major)
        if committed is None or committed["accepted"] != accepted:
            drift.append(
                f"{name} on LLVM {major}: committed {committed}, measured {accepted}"
            )
        options.setdefault(name, {})[major] = {
            "accepted": accepted,
            "evidence": evidence,
            "verified_on": verified_on,
        }

    if args.check:
        if drift:
            print(
                "compiler option table differs from this compiler:", *drift, sep="\n  "
            )
            return 1
        return 0
    for name in options:
        options[name] = dict(sorted(options[name].items(), key=lambda kv: int(kv[0])))
    table.TABLE_PATH.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {table.TABLE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
