"""Three checks that answer three different questions; a descriptor count
answers none of them.

  1. STATIC -- do the descriptors describe what they claim?
     (`verify_variant_sets.py`.) Runs anywhere, no build and no rocKE.
     `--mode` picks which claim is made and has no default.
  2. LOADS  -- does the engine survive the loader's own rules?
     (`hipdnn_validate_descriptors`.) Needs a build, no GPU; the only cheap
     rung that catches an engine dropped for a duplicate catalog tuple.
  3. SERVES -- does it serve graphs on a device, and how many? Needs a GPU.

This tool runs 1 and 2 and reports 3's requirement explicitly.

    coverage_gate.py --tree <descriptors> --mode full \\
                     --profile <profile.yaml> \\
                     --validator <build>/bin/hipdnn_validate_descriptors \\
                     --expect-engine hipkernel:Gfx942AttentionDense

Rung 2 and `--mode full` want the packed tree: `hkp_pack` lowers a
`kind: rocke` descriptor to `kind: kpack` at build time, and the runtime
loader rejects `builder` as an unknown key.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

#: Rung 1's two claims, spelled exactly as `verify_variant_sets.py` spells them.
MODES = ("full", "structural")


def run_static(
    tree: Path,
    profile: Path | None,
    tool: Path,
    mode: str,
    arch: str | None = None,
    kpack_python_dir: str | None = None,
) -> tuple[bool, str]:
    """Rung 1: structural properties of the set, plus compiled agreement in
    full mode, read off `verify_variant_sets`' exit code.

    Under `--mode full` a check that could not run fails the tool; under
    `--mode structural` compiled specialization agreement is not checked and
    the rung reports itself as structural-only.
    """
    argv = [sys.executable, str(tool), "set", str(tree), "--mode", mode]
    if profile:
        argv += ["--profile", str(profile)]
    if arch:
        argv += ["--arch", arch]
    if kpack_python_dir:
        argv += ["--kpack-python-dir", kpack_python_dir]
    result = subprocess.run(argv, capture_output=True, text=True)
    # Both streams on failure: progress goes to stdout and refusals -- an
    # unresolvable reference, an ambiguous tree -- to stderr.
    parts = [result.stdout.strip()]
    if result.returncode != 0:
        parts.append(result.stderr.strip())
    detail = "\n".join(p for p in parts if p)
    return result.returncode == 0, detail


def run_loads(
    tree: Path, validator: Path, expect_engines: list[str]
) -> tuple[bool, str, list[str]]:
    """Rung 2: does the loader accept the engine? Reports the engine list
    rather than a boolean, since an engine dropped at load leaves the file
    count and exit code unchanged."""
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
    parser.add_argument(
        "--mode",
        required=True,
        choices=MODES,
        help="What rung 1 is allowed to claim. full: check the producing compiler's "
        "evidence against these descriptors, this schema, this architecture and the "
        "payload bytes they name; a missing or mismatched record FAILS. structural: "
        "run only the checks that need no compiled evidence and report compiled "
        "specialization agreement as NOT CHECKED. There is no default: the two make "
        "different claims and a silent choice would print one under the other's "
        "name.",
    )
    parser.add_argument(
        "--arch",
        help="The architecture whose shard rung 1 is checking, forwarded to "
        "verify_variant_sets.py. Required under --mode full when the tree does not "
        "pin exactly one.",
    )
    parser.add_argument(
        "--kpack-python-dir",
        help="Directory holding the rocm_kpack package, forwarded to "
        "verify_variant_sets.py so --mode full can read the payload bytes a packed "
        "descriptor names. Omit it to use the installed one.",
    )
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

    ok, detail = run_static(
        tree, profile, static_tool, args.mode, args.arch, args.kpack_python_dir
    )
    if not ok:
        verdict = "FAIL"
    elif args.mode == "structural":
        # Named on the rung's own line, not only in the summary: a reader
        # scanning for "1. STATIC PASS" must not find it on a run that never
        # checked a shipped binary against the metadata selecting it.
        verdict = (
            "PASS (STRUCTURAL ONLY -- compiled specialization agreement NOT checked)"
        )
    else:
        verdict = "PASS"
    print(f"  1. STATIC   {verdict}")
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

    # Rung 3 is stated, never inferred: descriptors that are fine and an engine
    # that loads can still serve nothing.
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
    if args.mode == "structural":
        print(
            "GATE PASSED on rungs 1 and 2, STRUCTURALLY: the descriptors are "
            "well-formed and the engine"
        )
        print(
            "loads. Nothing here checked that any shipped binary agrees with the "
            "metadata that"
        )
        print(
            "selects it -- only --mode full reads the producing compiler's evidence "
            "-- and nothing"
        )
        print("was served either; rung 3 is still owed.")
        return 0
    print("GATE PASSED on rungs 1 and 2: descriptors are well-formed and the engine")
    print("loads. That is NOT evidence anything was served -- rung 3 is still owed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
