#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocKE SLO linter: measures kernel-file shape. No LLM, no network, no GPU.

  slo.py report [--json FILE] [--scope DIR]   every number, no pass/fail

Exit codes:
  0  ran fine
  2  the tool itself could not run (bad args, unparsable file)

Checks: SLO101 file SLOC, SLO102 longest function, SLO103 raw IR ops per 100 SLOC,
SLO104 helper-call ratio, SLO105 bool fields per frozen spec, SLO106 copy-paste
overlap, SLO107 repeated patterns. Limits and scope live in slo_config.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import slo_config as C  # noqa: E402
import slo_measure as MS  # noqa: E402
import slo_metrics as M  # noqa: E402
import slo_output as OUT  # noqa: E402

VERSION = "0.1"
DEFAULT_ROOT = HERE.parents[1]  # rocke/tools/slo -> rocke


def _die(msg: str) -> int:
    print(f"slo: error: {msg}", file=sys.stderr)
    return 2


def _write(path: str | None, text: str) -> None:
    if path == "-":
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
    elif path:
        Path(path).write_text(text if text.endswith("\n") else text + "\n")


def cmd_report(a) -> int:
    root = a.root
    files = M.scope_files(root, a.scope)
    current = MS.measure(root, files)
    summary = MS.over_target_summary(current)
    if a.json:
        doc = {
            "tool": "rocke-slo",
            "version": VERSION,
            "root": str(root),
            "targets": C.TARGETS,
            "scope": a.scope or list(C.SCOPE),
            "over_target": summary,
            **current,
        }
        _write(a.json, OUT.to_json(doc))
    if a.json != "-":
        print(OUT.report_text(current, summary))
    for e in current["errors"]:
        print(f"ERROR {e['path']}:{e['line']} {e['message']}", file=sys.stderr)
    return 2 if current["errors"] else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="slo.py", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="version", version=f"rocke-slo {VERSION}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument(
            "--root",
            type=Path,
            default=DEFAULT_ROOT,
            help="rocke folder (default: two levels above this file)",
        )

    r = sub.add_parser("report", help="print every number, no pass/fail")
    common(r)
    r.add_argument(
        "--json", metavar="FILE", help="write all numbers as JSON ('-' for stdout)"
    )
    r.add_argument(
        "--scope",
        action="append",
        metavar="DIR",
        help="measure this rocke-relative folder instead of the configured scope (repeatable)",
    )

    a = p.parse_args(argv)
    a.root = a.root.resolve()
    if not (a.root / C.SCOPE[0]).is_dir():
        return _die(
            f"{a.root} does not look like the rocke folder (no {C.SCOPE[0]}); use --root"
        )
    try:
        return {"report": cmd_report}[a.cmd](a)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    except Exception as e:  # anything unexpected is a tool failure, never a pass
        return _die(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
