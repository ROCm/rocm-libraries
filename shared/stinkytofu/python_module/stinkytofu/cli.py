# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""stinkytofu passes | stinkytofu run FILE [--pass NAME[=args]]...
| stinkytofu reduce FILE --test CMD [--test-arg ARG]...

The command calls the analysis API. It does not run the optimization pipeline.
A .s file, or stdin with --asm, loads as assembler source and prints assembler
source; anything else loads as STIR and prints STIR. --print-before and
--print-after write each pass's dump to stderr; stdout stays the module.
--feature and --tile set run_passes' features and tile, so a command that
module.run_info() records runs again as it is. Exit status: 0 success, 2 for a
missing file, a load error, an unknown pass, function, feature, or tile field,
or a bad value, 1 when a pass refuses the function or reports an analysis
failure.

reduce shrinks FILE while the test command still reports it failing: each
candidate is written in FILE's form, its path goes last on the command line,
and exit status 0 means it still fails. The smallest module goes to stdout and
progress to stderr. Exit status: 0 success, 2 for a missing file, a load error,
a test that cannot start, or a test that does not report FILE as failing.
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


def _settings(items, option, convert):
    """NAME=VALUE arguments as a dict, or None when there are none."""
    settings = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise ValueError("%s takes NAME=VALUE, not %r" % (option, item))
        settings[name.strip()] = convert(option, name.strip(), value.strip())
    return settings or None


def _boolean(option, name, text):
    if text.lower() not in ("true", "false"):
        raise ValueError("%s %s must be true or false, not %r" % (option, name, text))
    return text.lower() == "true"


def _integer(option, name, text):
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(
            "%s %s must be an integer, not %r" % (option, name, text)
        ) from exc


def cmd_passes():
    for name in _module().pass_names():
        sys.stdout.write(name + "\n")
    return 0


def _load(args):
    """The module args.input names, and whether it is assembler source."""
    stinkytofu = _module()
    arch = _arch(args.arch)
    is_asm = args.asm or args.input.endswith(".s")
    if args.input == "-":
        load = stinkytofu.load_asm if is_asm else stinkytofu.load_stir
        return load(sys.stdin.read(), arch=arch), is_asm
    if is_asm:
        return stinkytofu.load_asm_file(args.input, arch=arch), is_asm
    return stinkytofu.load_stir_file(args.input, arch=arch), is_asm


def _write(args, text):
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)


def cmd_run(args):
    module, is_asm = _load(args)
    records = module.run_passes(
        args.passes,
        function=args.function,
        features=_settings(args.features, "--feature", _boolean),
        tile=_settings(args.tile, "--tile", _integer),
        print_before=args.print_before,
        print_after=args.print_after,
    )
    for record in records:
        for dump in (record.before, record.after):
            if dump is not None:
                sys.stderr.write(dump)
    if args.emit_asm or (is_asm and not args.ssa):
        _write(args, module.emit_asm())
    else:
        _write(args, module.serialize(ssa=args.ssa))
    return 0


def cmd_reduce(args):
    stinkytofu = _module()
    module, is_asm = _load(args)
    check = stinkytofu.command_check(
        [args.test] + args.test_args,
        form="asm" if is_asm else "stir",
        timeout=args.timeout,
    )
    result = stinkytofu.reduce(
        module,
        check,
        max_checks=args.max_checks,
        log=lambda line: sys.stderr.write(line + "\n"),
    )
    sys.stderr.write(str(result) + "\n")
    _write(args, result.module.emit_asm() if is_asm else result.module.serialize())
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="stinkytofu",
        description="Load STIR or assembler source, run named StinkyTofu passes, "
        "and print the result, or shrink a file a test reports failing.",
    )
    loaded = argparse.ArgumentParser(add_help=False)
    loaded.add_argument("input", help="STIR file, .s file, or - for stdin")
    loaded.add_argument(
        "--arch", default="12,5,0", help="major,minor,stepping (default 12,5,0)"
    )
    loaded.add_argument(
        "--asm",
        action="store_true",
        help="read the input as assembler source; a .s file always is",
    )
    loaded.add_argument(
        "-o", "--output", help="write the result to this file instead of stdout"
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("passes", help="print one pass name per line")
    run = sub.add_parser(
        "run", parents=[loaded], help="load a file, run passes, and print the result"
    )
    run.add_argument(
        "--pass",
        dest="passes",
        action="append",
        default=[],
        metavar="NAME[=args]",
        help="pass to run, repeatable, in order. Text after = is the pass argument list",
    )
    run.add_argument(
        "--function",
        help="run the passes on this function only; default every function",
    )
    run.add_argument(
        "--feature",
        dest="features",
        action="append",
        default=[],
        metavar="NAME=true|false",
        help="set a pass feature, repeatable: unroll_gemm or distribute_global_read",
    )
    run.add_argument(
        "--tile",
        action="append",
        default=[],
        metavar="NAME=N",
        help="set a GEMM tile field, repeatable, such as NumWaves=2",
    )
    run.add_argument(
        "--print-before",
        choices=("stir", "asm"),
        help="write the function to stderr before each pass, as STIR or assembly",
    )
    run.add_argument(
        "--print-after",
        choices=("stir", "asm"),
        help="write the function to stderr after each pass, as STIR or assembly",
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

    reduce = sub.add_parser(
        "reduce",
        parents=[loaded],
        help="shrink a file while a test command still reports it failing",
    )
    reduce.add_argument(
        "--test",
        required=True,
        metavar="CMD",
        help="command run on each candidate, with its path last; exit 0 means it "
        "still fails",
    )
    reduce.add_argument(
        "--test-arg",
        dest="test_args",
        action="append",
        default=[],
        metavar="ARG",
        help="argument for the test command, before the path, repeatable",
    )
    reduce.add_argument(
        "--timeout",
        type=float,
        metavar="SECONDS",
        help="stop a test after this long; it counts as not failing",
    )
    reduce.add_argument(
        "--max-checks",
        type=int,
        metavar="N",
        help="stop after N tests and print the smallest failing module so far",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "passes":
            return cmd_passes()
        if args.command == "reduce":
            return cmd_reduce(args)
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
