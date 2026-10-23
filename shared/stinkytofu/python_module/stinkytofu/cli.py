# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""stinkytofu passes | stinkytofu run FILE [--pass NAME[=args]]...

The command calls the analysis API. It does not run the optimization pipeline
and it does not emit assembler source. Exit status: 0 success, 2 for a missing
file, a parse error, or an unknown pass, 1 when a pass refuses the function
or reports an analysis failure.
"""

import argparse
import sys


def _module():
    try:
        import stinkytofu
    except ImportError as exc:
        raise ValueError(
            "stinkytofu is not importable (%s). From shared/stinkytofu run "
            "`python3 tasks.py build`, then run build/bin/stinkytofu" % exc
        ) from exc
    return stinkytofu


def _arch(text):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("arch must be major,minor,stepping, for example 12,5,0")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("arch must be major,minor,stepping, for example 12,5,0") from exc


def cmd_passes():
    for name in _module().pass_names():
        sys.stdout.write(name + "\n")
    return 0


def cmd_run(args):
    stinkytofu = _module()
    if args.input == "-":
        text = sys.stdin.read()
        module = stinkytofu.load_stir(text, arch=_arch(args.arch))
    else:
        module = stinkytofu.load_stir_file(args.input, arch=_arch(args.arch))
    if args.passes:
        module.run_passes(args.passes)
    out = module.serialize(ssa=args.ssa)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(out)
    else:
        sys.stdout.write(out)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stinkytofu",
        description="Load STIR, run named StinkyTofu passes, and print the result.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("passes", help="print one pass name per line")
    run = sub.add_parser("run", help="load a STIR file, run passes, and print STIR")
    run.add_argument("input", help="STIR file, or - for stdin")
    run.add_argument("--arch", default="12,5,0", help="major,minor,stepping (default 12,5,0)")
    run.add_argument(
        "--pass",
        dest="passes",
        action="append",
        default=[],
        metavar="NAME[=args]",
        help="pass to run, repeatable, in order. Text after = is the pass argument list",
    )
    run.add_argument("--ssa", action="store_true", help="print the diagnostic SSA view")
    run.add_argument("-o", "--output", help="write STIR to this file instead of stdout")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "passes":
            return cmd_passes()
        return cmd_run(args)
    except ValueError as exc:
        sys.stderr.write("stinkytofu: %s\n" % exc)
        return 2
    except RuntimeError as exc:
        sys.stderr.write("stinkytofu: %s\n" % exc)
        return 1
    except OSError as exc:
        where = ": %s" % exc.filename if exc.filename else ""
        sys.stderr.write("stinkytofu: %s%s\n" % (exc.strerror or exc, where))
        return 2
