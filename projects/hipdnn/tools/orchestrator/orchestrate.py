#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Run a declarative agent flow: locate tools, resolve context, execute, record evidence.

orchestrate.py run configs/flows/rtc-kernel-review.yaml --input graph=graph.json
orchestrate.py run  <flow> --dry-run      # resolved argv + rendered prompts, no launch
orchestrate.py validate <flow>            # schema and every ${ref}, no launch
orchestrate.py inputs   <flow>            # what this flow needs to be given
orchestrate.py doctor                     # can every tool be found on this machine?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runner import Engine, Flow, ToolRegistry, bind_inputs, validate_refs
from runner.errors import ConfigError, OrchestratorError

HERE = Path(__file__).resolve().parent
DEFAULT_TOOLS = HERE / "configs" / "tools.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="orchestrate.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute a flow")
    _common(run)
    run.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="run input; KEY=@file reads the value from a file",
    )
    run.add_argument("--inputs-file", help="YAML mapping of input -> value")
    run.add_argument(
        "--run-dir",
        help="write this run's evidence here (default: ./runs/<flow>/<utc>)",
    )
    run.add_argument(
        "--max-iterations",
        type=int,
        help="override every loop's budget (1 = single pass)",
    )
    run.add_argument("--only", help="run only this top-level step or loop")
    run.add_argument("--from", dest="start_from", help="start at this top-level step")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved argv, env and rendered prompts; launch nothing",
    )
    run.add_argument(
        "--tee", action="store_true", help="mirror step output to this terminal"
    )

    validate = sub.add_parser(
        "validate", help="check schema and references without running"
    )
    _common(validate)

    inputs_cmd = sub.add_parser("inputs", help="list the inputs a flow declares")
    inputs_cmd.add_argument("flow")

    doctor = sub.add_parser(
        "doctor", help="resolve every tool in the registry on this machine"
    )
    doctor.add_argument("--tools", default=str(DEFAULT_TOOLS))
    doctor.add_argument("--profile")
    doctor.add_argument("--flow", help="check only the tools this flow uses")

    tools_cmd = sub.add_parser("tools", help="registry contents")
    tools_cmd.add_argument("action", choices=["list"])
    tools_cmd.add_argument("--tools", default=str(DEFAULT_TOOLS))
    tools_cmd.add_argument("--profile")

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2
    except OrchestratorError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("flow")
    parser.add_argument("--tools", default=str(DEFAULT_TOOLS))
    parser.add_argument("--profile")


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "inputs":
        return _inputs(args)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "tools":
        return _tools(args)

    registry = ToolRegistry.load(args.tools, args.profile)
    flow = Flow.load(args.flow)
    validate_refs(flow, registry.vars)

    if args.command == "validate":
        missing = sorted(flow.tools_used() - set(registry.tools))
        if missing:
            raise ConfigError(
                f"{flow.path}: uses tool(s) not in {registry.path.name}: {', '.join(missing)}"
            )
        print(
            f"{flow.path.name}: OK - {len(list(flow.all_steps()))} step(s), "
            f"{len(flow.inputs)} input(s), tools: {', '.join(sorted(flow.tools_used()))}"
        )
        return 0

    bound = bind_inputs(flow, args.input, args.inputs_file)
    engine = Engine(
        flow,
        registry,
        bound,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        run_root=HERE / "runs",
        max_iterations=args.max_iterations,
        tee=args.tee,
        only=args.only,
        start_from=args.start_from,
        profile=args.profile,
    )

    if args.dry_run:
        for entry in engine.plan():
            where = f"[{entry['group']}] " if entry["group"] else ""
            print(f"\n=== {where}{entry['id']} ({entry['tool']}) ===")
            print("argv: " + json.dumps(entry["argv"]))
            if entry["cwd"]:
                print(f"cwd:  {entry['cwd']}")
            if entry["env"]:
                print("env:  " + json.dumps(entry["env"]))
            if entry["timeout"]:
                print(f"timeout: {entry['timeout']}s")
            if entry["result_file"]:
                print(f"result_file: {entry['result_file']}")
            if entry["stdin"]:
                print("--- stdin ---")
                print(entry["stdin"].rstrip())
                print("--- end stdin ---")
        return 0

    report = engine.run()
    print(f"\nrun {report.run_id}: {report.status}  ({report.run_dir})")
    for loop in report.loops:
        verdict = "satisfied" if loop.satisfied else "NOT satisfied"
        print(
            f"  loop {loop.id}: {loop.iterations} iteration(s), {verdict}: {loop.until}"
        )
    if report.error:
        print(f"  {report.error}", file=sys.stderr)
    return 0 if report.status == "ok" else 1


def _inputs(args: argparse.Namespace) -> int:
    flow = Flow.load(args.flow)
    if not flow.inputs:
        print(f"{flow.name}: declares no inputs")
        return 0
    print(f"{flow.name} inputs:")
    for name, spec in flow.inputs.items():
        marker = "required" if spec.required else f"default={spec.default!r}"
        print(f"  {name} ({spec.type}, {marker})")
        if spec.description:
            print(f"      {spec.description}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    registry = ToolRegistry.load(args.tools, args.profile)
    wanted = sorted(registry.tools)
    if args.flow:
        wanted = sorted(Flow.load(args.flow).tools_used())
    failures = 0
    print(f"registry: {registry.path}")
    for name in wanted:
        try:
            tool = registry.get(name)
            print(f"  OK      {name:<20} {registry.resolve_exe(tool)}")
        except ConfigError as error:
            failures += 1
            print(f"  MISSING {name:<20} {error}")
    print(f"\n{len(wanted) - failures}/{len(wanted)} tool(s) resolved")
    return 1 if failures else 0


def _tools(args: argparse.Namespace) -> int:
    registry = ToolRegistry.load(args.tools, args.profile)
    for name, tool in sorted(registry.tools.items()):
        print(f"{name}: {tool.exe}")
        if tool.env:
            print(f"    env: {json.dumps(tool.env)}")
        if tool.path_prepend:
            print(f"    path_prepend: {list(tool.path_prepend)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
