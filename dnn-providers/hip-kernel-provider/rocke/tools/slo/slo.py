#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocKE SLO linter: measures kernel-file shape. No LLM, no network, no GPU.

  slo.py report   [--json FILE] [--scope DIR]   every number, no pass/fail
  slo.py check    [--base REF] [outputs]        what got worse than the baseline
  slo.py baseline --update                      save today's numbers as the baseline

Exit codes (the CI contract):
  0  ran fine (default: warnings never fail)
  1  something got worse, and --fail-on increase was given
  2  the tool itself could not run (bad args, unparsable file, git missing when asked)

Checks: SLO101 file SLOC, SLO102 longest function, SLO103 raw IR ops per 100 SLOC,
SLO104 helper-call ratio, SLO105 bool fields per frozen spec, SLO106 copy-paste
overlap, SLO107 repeated patterns. Limits and scope live in slo_config.py.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import slo_check as K  # noqa: E402
import slo_config as C  # noqa: E402
import slo_measure as MS  # noqa: E402
import slo_metrics as M  # noqa: E402
import slo_output as OUT  # noqa: E402

VERSION = "0.1"
DEFAULT_ROOT = HERE.parents[1]  # rocke/tools/slo -> rocke
DEFAULT_BASELINE = HERE / "baseline.json"


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


def cmd_check(a) -> int:
    root = a.root
    only = renamed = None
    if a.base:
        import slo_git

        try:
            ch = slo_git.changes(root, a.base)
        except slo_git.GitError as e:
            return _die(f"--base {a.base}: {e}")
        only = {p for p in ch["changed"] if M.in_scope(p) and (root / p).exists()}
        renamed = ch["renamed"]
    current = MS.measure(root)
    baseline = K.load_baseline(a.baseline)
    if not baseline["files"]:
        print(
            f"slo: note: no baseline at {a.baseline}; every file is judged as new.",
            file=sys.stderr,
        )
    findings = K.compare(current, baseline, only, renamed)
    if a.base:  # keep the summary about the files the PR touched
        current = {
            **current,
            "files": {p: r for p, r in current["files"].items() if p in only},
        }

    if a.json:
        _write(
            a.json,
            OUT.to_json(
                {"tool": "rocke-slo", "version": VERSION, "findings": findings}
            ),
        )
    if a.junit:
        _write(a.junit, OUT.junit(findings, current, a.fail_on))
    if a.markdown:
        _write(a.markdown, OUT.markdown(findings, current))
    if a.github:
        _write("-", OUT.github(findings, a.fail_on, a.path_prefix))
    if "-" not in (a.json, a.junit, a.markdown):
        print(OUT.text(findings, current, a.fail_on))
    if any(f["kind"] == "error" for f in findings):
        return 2
    return 1 if OUT.failing(findings, a.fail_on) else 0


def cmd_baseline(a) -> int:
    if not a.update:
        return _die("baseline: pass --update to rewrite the baseline")
    current = MS.measure(a.root)
    if current["errors"]:
        for e in current["errors"]:
            print(f"ERROR {e['path']}:{e['line']} {e['message']}", file=sys.stderr)
        return 2
    old = K.load_baseline(a.baseline)
    looser = (
        [f for f in K.compare(current, old) if f["kind"] == "worse"]
        if old["files"]
        else []
    )
    meta = {
        "tool_version": VERSION,
        "written": datetime.date.today().isoformat(),
        "files": len(current["files"]),
    }
    K.save_baseline(a.baseline, current, meta)
    print(f"wrote {a.baseline}: {len(current['files'])} files")
    if looser:
        print(f"WARNING: {len(looser)} numbers are now looser than the old baseline:")
        for f in looser[:20]:
            print(f"  {f['check']} {f['path']}  {f['message']}")
    return 0


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

    c = sub.add_parser("check", help="compare with the baseline")
    common(c)
    c.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    c.add_argument(
        "--base", metavar="REF", help="git ref; only check files changed since REF"
    )
    c.add_argument(
        "--fail-on",
        choices=("never", "increase"),
        default="never",
        help="never (default): exit 0 with warnings. increase: exit 1 if anything got worse",
    )
    c.add_argument("--json", metavar="FILE")
    c.add_argument("--junit", metavar="FILE")
    c.add_argument("--markdown", metavar="FILE")
    c.add_argument(
        "--github", action="store_true", help="print GitHub ::warning annotations"
    )
    c.add_argument(
        "--path-prefix",
        default="",
        help="prepended to paths in --github output, "
        "e.g. dnn-providers/hip-kernel-provider/rocke/",
    )

    b = sub.add_parser("baseline", help="save today's numbers")
    common(b)
    b.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    b.add_argument("--update", action="store_true")

    a = p.parse_args(argv)
    a.root = a.root.resolve()
    if not (a.root / C.SCOPE[0]).is_dir():
        return _die(
            f"{a.root} does not look like the rocke folder (no {C.SCOPE[0]}); use --root"
        )
    try:
        return {"report": cmd_report, "check": cmd_check, "baseline": cmd_baseline}[
            a.cmd
        ](a)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    except Exception as e:  # anything unexpected is a tool failure, never a pass
        return _die(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
