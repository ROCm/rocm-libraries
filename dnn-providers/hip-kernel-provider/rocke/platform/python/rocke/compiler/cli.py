# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Command-line interface for installed COMGR evidence and policy."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import _support as support
from .evidence import default_cache_dir, load_or_probe, validate_report
from .policy import resolve_policy

__all__ = ["main"]


def _print_summary(
    report: dict[str, object],
    cache_hit: bool,
    output: Path,
    policy_output: Path | None,
) -> None:
    compiler = report["compiler"]
    version = compiler.get("llvm_version")
    version_text = ".".join(str(part) for part in version) if version else "unknown"
    print(f"COMGR: {compiler['library']}")
    print(f"COMGR sha256: {compiler['sha256'][:12]}…")
    print(f"embedded LLVM: {version_text} via {compiler['version_source']}")
    print(f"cache: {'hit' if cache_hit else 'miss'}")
    for name, result in report["capabilities"].items():
        observations = result["value"]["observations"]
        counts = {
            outcome: sum(entry["outcome"] == outcome for entry in observations)
            for outcome in ("accepted", "rejected", "crash", "timeout", "error")
        }
        detail = ", ".join(f"{key}={value}" for key, value in counts.items() if value)
        print(f"{name}: {detail}")
    print(f"output: {output}")
    if policy_output is not None:
        print(f"policy output: {policy_output}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="probe the COMGR contract that compiles rocKE LLVM IR"
    )
    parser.add_argument(
        "--comgr-lib",
        type=Path,
        default=None,
        help="explicit libamd_comgr path (default: rocKE runtime discovery)",
    )
    parser.add_argument("--processor", default="gfx950")
    parser.add_argument("--output", type=Path, default=Path("capabilities.json"))
    parser.add_argument(
        "--policy-output",
        type=Path,
        default=None,
        help="also write the canonical CompilerPolicy manifest",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
    )
    parser.add_argument(
        "--jobs", type=int, default=0, help="parallel workers; 0=CPU count"
    )
    parser.add_argument("--timeout", type=int, default=60, help="seconds per candidate")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    try:
        report, cache_hit = load_or_probe(
            comgr_lib=args.comgr_lib,
            processor=args.processor,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            jobs=args.jobs,
            timeout=args.timeout,
        )
        if args.validate:
            validate_report(report)
        support._write_json(args.output, report)
        if args.policy_output is not None:
            args.policy_output.parent.mkdir(parents=True, exist_ok=True)
            args.policy_output.write_text(
                resolve_policy(report).to_json(), encoding="utf-8"
            )
    except (OSError, TypeError, ValueError, support.ProbeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        _print_summary(report, cache_hit, args.output, args.policy_output)
        if args.validate:
            print("schema validation: passed")
        print(f"elapsed: {time.perf_counter() - started:.3f} s")
    outcomes = [
        observation["outcome"]
        for result in report["capabilities"].values()
        for observation in result["value"]["observations"]
    ]
    return 0 if all(outcome != "error" for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
