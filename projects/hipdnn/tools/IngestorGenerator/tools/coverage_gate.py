"""Three checks that answer three different questions. Counts answer none of them.

A descriptor-count gate once passed on an arm that served ZERO graphs. The count was
right: the descriptors were on disk, all of them, correctly named. They just never
reached a GPU, because a duplicate catalog tuple made the loader reject the whole
engine and every graph fell through to a different one -- while the phase ran to
completion and exited 0.

So "did it work" decomposes, and each rung sees a failure the others cannot:

  1. STATIC -- do the descriptors describe what they claim? (`verify_variant_sets.py`:
     binary nesting, catalog-tuple uniqueness, no sentinel, metadata matches binary,
     matcher vocabulary.) Runs on any machine, needs no build.
  2. LOADS  -- does the ENGINE survive the loader's own rules?
     (`hipdnn_validate_descriptors`, which round-trips a bundle exactly as a provider
     would at plugin-load time.) Needs a build, no GPU. This is the rung that catches
     the dropped engine, and the only cheap one that can.
  3. SERVES -- does it serve graphs ON A DEVICE, and how many? Needs a GPU. The
     preflight that caught two failures every static check passed.

This tool runs 1 and 2 and reports 3's requirement explicitly rather than pretending
the first two imply it. Rungs 1 and 2 both passing means the descriptors are
well-formed and the engine loads. It does NOT mean anything was served.

    coverage_gate.py --tree <descriptors> --profile <profile.yaml> \\
                     --validator <build>/bin/hipdnn_validate_descriptors \\
                     --expect-engine hipkernel:Gfx942AttentionDense

DIALECTS. Rung 2 wants the PACKED tree, not the authored one. A `kind: rocke`
descriptor is an authoring form that `hkp_pack` lowers to `kind: kpack` at build
time; the runtime loader has never heard of `builder` and rejects it. Pointing rung 2
at the authored tree therefore fails with a genuine-looking error about an unknown
key, which is the loader being right. The gate says so rather than leaving it to be
rediscovered.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_static(tree: Path, profile: Path | None, tool: Path) -> tuple[bool, str]:
    """Rung 1. Structural properties of the SET."""
    argv = [sys.executable, str(tool), "set", str(tree)]
    if profile:
        argv += ["--profile", str(profile)]
    result = subprocess.run(argv, capture_output=True, text=True)
    narrowed = "NOT CHECKED" in result.stdout
    ok = result.returncode == 0 and not narrowed
    detail = result.stdout.strip() or result.stderr.strip()
    if narrowed:
        detail += (
            "\n      a NOT CHECKED line is a NARROWED run, not a pass -- supply a "
            "profile so the policy and vocabulary checks actually execute"
        )
    return ok, detail


def run_loads(
    tree: Path, validator: Path, expect_engines: list[str]
) -> tuple[bool, str, list[str]]:
    """Rung 2. Does the loader accept the engine, under its own rules?

    Reports the engine LIST, not a boolean, because the historical failure is an
    engine that silently vanishes: the file count is unchanged, the exit code is 0,
    and the only observable is that a name is missing from this list.
    """
    argv = [str(validator), str(tree), "--json"]
    for name in expect_engines:
        argv += ["--expect-engine", name]
    result = subprocess.run(argv, capture_output=True, text=True)
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False, (result.stdout or result.stderr).strip()[:400], []

    engines = list(report.get("engines") or [])
    missing = list(report.get("expected_engines_missing") or [])
    errors = [
        d.get("message", "")
        for d in report.get("diagnostics") or []
        if d.get("severity") == "ERROR"
    ]
    lines = [f"engines loaded: {len(engines)}"]
    for name in engines:
        lines.append(f"        {name}")
    if missing:
        lines.append(f"      MISSING: {missing}")
    for message in errors[:3]:
        lines.append(f"      ERROR: {message[:160]}")
    ok = result.returncode == 0 and not missing and not errors
    return ok, "\n      ".join(lines), engines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the static and loader rungs, and state what rung 3 needs.",
    )
    parser.add_argument("--tree", required=True, help="Descriptor root to check.")
    parser.add_argument("--profile", help="Kernel profile for the static rung.")
    parser.add_argument(
        "--validator",
        help="Path to hipdnn_validate_descriptors. Without it rung 2 is reported as "
        "NOT RUN rather than skipped silently.",
    )
    parser.add_argument(
        "--expect-engine",
        action="append",
        default=[],
        help="An engine name that MUST be present after loading (repeatable).",
    )
    parser.add_argument(
        "--min-served",
        type=int,
        default=0,
        help="Graphs rung 3 must serve on a device. Recorded in the summary as the "
        "threshold a GPU preflight has to clear; this tool cannot check it.",
    )
    args = parser.parse_args(argv)

    tree = Path(args.tree)
    if not tree.exists():
        print(f"FAIL: {tree} does not exist", file=sys.stderr)
        return 2

    static_tool = Path(__file__).resolve().parent / "verify_variant_sets.py"
    profile = Path(args.profile) if args.profile else None

    print("coverage gate")
    failures = []

    ok, detail = run_static(tree, profile, static_tool)
    print(f"  1. STATIC   {'PASS' if ok else 'FAIL'}")
    for line in detail.splitlines():
        print(f"      {line}")
    if not ok:
        failures.append("static")

    if (
        args.validator
        and shutil.which(str(args.validator))
        or (args.validator and Path(args.validator).exists())
    ):
        ok, detail, engines = run_loads(tree, Path(args.validator), args.expect_engine)
        print(f"  2. LOADS    {'PASS' if ok else 'FAIL'}")
        print(f"      {detail}")
        if not ok:
            failures.append("loads")
    else:
        print("  2. LOADS    NOT RUN")
        print(
            "      no --validator given. Build with HIPDNN_ENABLE_KERNEL_INGESTOR=ON "
            "and pass\n      <build>/bin/hipdnn_validate_descriptors. This is the rung "
            "that catches a\n      dropped engine, and no static check substitutes for "
            "it."
        )
        failures.append("loads-not-run")

    # Rung 3 is stated, never inferred. The whole point of the three-rung split is
    # that "the descriptors are fine and the engine loads" has been true of an arm
    # that served nothing.
    print("  3. SERVES   NOT RUN (needs a GPU)")
    print(
        f"      Run the corpus on a device and require at least "
        f"{args.min_served or '<N>'} graphs served BY THIS ENGINE.\n"
        "      Filter on engine_name: a graph another engine served is not coverage,\n"
        "      and an aggregate that does not filter reports its work as yours."
    )

    print()
    if failures:
        print(f"GATE FAILED ({', '.join(failures)})")
        return 1
    print("GATE PASSED on rungs 1 and 2: descriptors are well-formed and the engine")
    print("loads. That is NOT evidence anything was served -- rung 3 is still owed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
