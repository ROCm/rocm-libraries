# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""stinkytofu passes | stinkytofu run FILE [--pass NAME[=args]]...

The command calls the analysis API. It does not run the optimization pipeline.
A .s file, or stdin with --asm, loads as assembler source and prints assembler
source; anything else loads as STIR and prints STIR. Exit status: 0 success,
2 for a missing file, a load error, or an unknown pass, 1 when a pass refuses
the function or reports an analysis failure.
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
        raise ValueError(
            "arch must be major,minor,stepping, for example 12,5,0"
        ) from exc


def cmd_passes():
    for name in _module().pass_names():
        sys.stdout.write(name + "\n")
    return 0


def cmd_run(args):
    stinkytofu = _module()
    arch = _arch(args.arch)
    is_asm = args.asm or args.input.endswith(".s")
    if args.input == "-":
        text = sys.stdin.read()
        load = stinkytofu.load_asm if is_asm else stinkytofu.load_stir
        module = load(text, arch=arch)
    elif is_asm:
        module = stinkytofu.load_asm_file(args.input, arch=arch)
    else:
        module = stinkytofu.load_stir_file(args.input, arch=arch)
    if args.passes:
        module.run_passes(args.passes)
    if args.emit_asm or (is_asm and not args.ssa):
        out = module.emit_asm()
    else:
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
        description="Load STIR or assembler source, run named StinkyTofu passes, "
        "and print the result.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("passes", help="print one pass name per line")
    run = sub.add_parser("run", help="load a file, run passes, and print the result")
    run.add_argument("input", help="STIR file, .s file, or - for stdin")
    run.add_argument(
        "--arch", default="12,5,0", help="major,minor,stepping (default 12,5,0)"
    )
    run.add_argument(
        "--asm",
        action="store_true",
        help="read the input as assembler source; a .s file always is",
    )
    run.add_argument(
        "--pass",
        dest="passes",
        action="append",
        default=[],
        metavar="NAME[=args]",
        help="pass to run, repeatable, in order. Text after = is the pass argument list",
    )
    output = run.add_mutually_exclusive_group()
    output.add_argument(
        "--ssa", action="store_true", help="print the diagnostic SSA view"
    )
    output.add_argument(
        "--emit-asm",
        action="store_true",
        help="print assembler source; the default for assembler input",
    )
    run.add_argument(
        "-o", "--output", help="write the result to this file instead of stdout"
    )
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
