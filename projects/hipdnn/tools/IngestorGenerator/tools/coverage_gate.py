"""Three checks that answer three different questions. Counts answer none of them.

A descriptor-count gate once passed on an arm that served ZERO graphs. The count was
right: the descriptors were on disk, all of them, correctly named. They just never
reached a GPU, because a duplicate catalog tuple made the loader reject the whole
engine and every graph fell through to a different one -- while the phase ran to
completion and exited 0.

So "did it work" decomposes, and each rung sees a failure the others cannot:

  1. STATIC -- do the descriptors describe what they claim? (`verify_variant_sets.py`:
     binary nesting, catalog-tuple uniqueness, no sentinel, metadata matches binary,
     matcher vocabulary.) Runs on any machine, needs no build and no rocKE.

     This rung makes ONE OF TWO CLAIMS and `--mode` says which. `--mode full` checks
     the producing compiler's own evidence -- the specialization declaration each UKD
     carries and the effective-spec record the compile wrote onto it -- against these
     descriptors, this schema, this architecture and the payload bytes they name; a
     missing or mismatched record fails. `--mode structural` runs only the checks
     that need no compiled evidence and reports compiled specialization agreement as
     NOT CHECKED. A structural pass is reported as structural throughout and can
     never be read as compiled agreement, which is why there is no default: a gate
     that silently picks the weaker claim and prints the stronger one's line is the
     defect this whole tool exists to prevent.
  2. LOADS  -- does the ENGINE survive the loader's own rules?
     (`hipdnn_validate_descriptors`, which round-trips a bundle exactly as a provider
     would at plugin-load time.) Needs a build, no GPU. This is the rung that catches
     the dropped engine, and the only cheap one that can.
  3. SERVES -- does it serve graphs ON A DEVICE, and how many? Needs a GPU. The
     preflight that caught two failures every static check passed.

This tool runs 1 and 2 and reports 3's requirement explicitly rather than pretending
the first two imply it. Rungs 1 and 2 both passing means the descriptors are
well-formed and the engine loads. It does NOT mean anything was served.

    coverage_gate.py --tree <descriptors> --mode full \\
                     --profile <profile.yaml> \\
                     --validator <build>/bin/hipdnn_validate_descriptors \\
                     --expect-engine hipkernel:Gfx942AttentionDense

DIALECTS. Rung 2 wants the PACKED tree, not the authored one. A `kind: rocke`
descriptor is an authoring form that `hkp_pack` lowers to `kind: kpack` at build
time; the runtime loader has never heard of `builder` and rejects it. Pointing rung 2
at the authored tree therefore fails with a genuine-looking error about an unknown
key, which is the loader being right. The gate says so rather than leaving it to be
rediscovered. `--mode full` wants the packed tree for the same reason from the other
side: the compiler's evidence exists only once the bytes do.
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
    """Rung 1. Structural properties of the SET, plus compiled agreement in full mode.

    Both modes are read off `verify_variant_sets`' own exit code, which encodes
    what each mode claims: under `--mode full` a check that could not run is a
    gap and the tool fails on it, while under `--mode structural` compiled
    specialization agreement is NOT CHECKED BY DEFINITION -- that is what the
    mode is -- so the line is expected, the rung reports itself as
    structural-only, and the caller is told in the same breath that the strong
    claim was never made.
    """
    argv = [sys.executable, str(tool), "set", str(tree), "--mode", mode]
    if profile:
        argv += ["--profile", str(profile)]
    if arch:
        argv += ["--arch", arch]
    if kpack_python_dir:
        argv += ["--kpack-python-dir", kpack_python_dir]
    result = subprocess.run(argv, capture_output=True, text=True)
    # Both streams when it failed: the tool prints its progress on stdout and its
    # refusals -- an unresolvable reference, an ambiguous tree -- on stderr, so
    # preferring one drops the only line that says why.
    parts = [result.stdout.strip()]
    if result.returncode != 0:
        parts.append(result.stderr.strip())
    detail = "\n".join(p for p in parts if p)
    return result.returncode == 0, detail


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
        # Named on the rung's own line, not only in the summary. A reader scanning
        # for "1. STATIC PASS" must not be able to find it on a run that never
        # checked whether a single shipped binary matches the metadata selecting it.
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
