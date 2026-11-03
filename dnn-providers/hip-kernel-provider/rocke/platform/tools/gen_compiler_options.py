#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure which LLVM backend options (``-mllvm``) the loaded compiler accepts.

The read side is ``rocke.runtime.compiler_options``; see it for why the answer
has to be known before COMGR is called. This tool writes its table,
``python/rocke/runtime/data/compiler_options.json``.

For every option in the table (plus any ``--option``), a child process loads
the COMGR this process loaded and compiles an empty kernel with
``-O3 -mllvm <option>``. The child goes straight to the COMGR pipeline, past
the table check, because measuring is what fills the table:

    child exits 0                              -> accepted
    child output has "Unknown command line"    -> not accepted (the parser
                                                  exited the child)
    anything else                              -> no answer; the tool stops
                                                  and records nothing

A host can only measure its own compiler, so a run writes one column, keyed by
the loaded compiler's LLVM major version, and leaves the others alone.
``--check`` writes nothing and fails when that column differs from the
measurement or is missing; CTest runs it as
``rocke_compiler_options_reproducibility``, so CI fails when its compiler's
committed answers drift.

Usage:
  python rocke/platform/tools/gen_compiler_options.py            # update this compiler's column
  python rocke/platform/tools/gen_compiler_options.py --check    # fail if the column differs
  python rocke/platform/tools/gen_compiler_options.py --option=-amdgpu-new-option
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# A source checkout keeps the package beside tools/. An installed test
# artifact stages this script elsewhere and supplies the package on
# PYTHONPATH instead.
PYTHON_ROOT = Path(__file__).resolve().parents[1] / "python"
if (PYTHON_ROOT / "rocke").is_dir() and str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from rocke.runtime import comgr
from rocke.runtime import compiler_options as table
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

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


def measure(option: str) -> tuple[bool, str]:
    """``(accepted, evidence)`` for one option on the COMGR this process loaded."""
    proc = subprocess.run(
        rocke_python_command("-c", _CHILD, option),
        capture_output=True,
        text=True,
        env=rocke_subprocess_env(comgr.comgr_child_env()),
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
    major = info.llvm_version[0]
    verified_on = "LLVM " + ".".join(map(str, info.llvm_version))

    committed = table.measured_answers()
    names = list(committed) + [o for o in args.option if o not in committed]
    # A separate copy takes the new column, so ``committed`` stays what the
    # file says for --check to compare against.
    answers = {option: dict(column) for option, column in committed.items()}
    measured = {}
    for name in names:
        accepted, evidence = measure(name)
        print(f"{name}: {'accepted' if accepted else 'not accepted'} on {verified_on}")
        measured[name] = accepted
        answers.setdefault(name, {})[major] = table.Answer(
            accepted=accepted, evidence=evidence, verified_on=verified_on
        )

    if args.check:
        drift = table.column_drift(committed, measured, llvm_major=major)
        if drift:
            print(
                "compiler option table differs from this compiler; run this tool "
                "without --check to update it:",
                *drift,
                sep="\n  ",
            )
            return 1
        return 0
    table.TABLE_PATH.write_text(table.format_table(answers), encoding="utf-8")
    print(f"wrote {table.TABLE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
