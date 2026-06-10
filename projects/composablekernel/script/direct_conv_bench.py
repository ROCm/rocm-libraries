#!/usr/bin/env python3
"""Direct-convolution profiler benchmark CLI.

A single entry point for the direct-convolution profiler tooling.
All shared logic lives in ``direct_conv_lib.py``; this file is a thin CLI over it.

Subcommands:
  run     - smoke / correctness: run every case and print a text summary.
  regress - perf gating: compare best TFLOPS against per-arch expected values
            (10% tolerance), write a markdown report, exit nonzero on FAIL.
  compare - iGEMM-vs-direct-conv comparison: markdown table (+ optional PNG).

All subcommands share ``--bin-path``, ``--cases`` and ``--verbose``.

For running the profiler, use the Dispatcher codegen build (``CK_TILE_DISPATCHER=ON``).
To focus on direct convolution only, add ``-D DISABLE_IMPLICIT_GEMM_INSTANCES=ON``.
The ``compare`` subcommand needs an implicit-GEMM-enabled build to populate the
iGEMM rows.
"""

import argparse
import datetime
import sys
from pathlib import Path

import direct_conv_lib as lib


_DTYPE_TOKEN = {"fp16": "1", "bf16": "2"}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    bin_path = Path(args.bin_path)
    if not bin_path.is_dir():
        print(f"ERROR: --bin-path '{bin_path}' is not a directory.", file=sys.stderr)
        return 1

    cases = lib.parse_cases(args.cases, arch=None)
    if not cases:
        print(f"ERROR: no cases parsed from '{args.cases}'.", file=sys.stderr)
        return 1

    if args.category:
        flt = args.category.lower()
        cases = [c for c in cases if flt in c.section.lower()]
    if args.dtype:
        tok = _DTYPE_TOKEN[args.dtype]
        cases = [c for c in cases if c.data_type == tok]
    if not cases:
        print("ERROR: no cases matched the given filters.", file=sys.stderr)
        return 1

    print(f"Running {len(cases)} case(s) from '{args.cases}'")
    print(f"Binary path: {bin_path}\n")

    results: list[lib.Result] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {lib._SECTION_TITLE[case.section]}  args: {case.args}")
        r = lib.run_case(bin_path, case, args.verbose)
        results.append(r)
        if r.ran:
            if r.best_instance:
                print(
                    f"  PASS  {r.avg_time_ms:.3f} ms  {r.tflops:.2f} TFlops"
                    f"  {r.gb_s:.1f} GB/s"
                )
                print(f"        best: {r.best_instance}")
            else:
                print("  PASS  (no best-instance line found in output)")
        else:
            print("  FAIL")
            if r.error:
                print(f"  error: {r.error}")

    lib.print_summary(results)
    return 0 if all(r.ran for r in results) else 1


# ---------------------------------------------------------------------------
# regress
# ---------------------------------------------------------------------------

def cmd_regress(args: argparse.Namespace) -> int:
    bin_path = Path(args.bin_path)
    if not (bin_path / "ckProfiler").exists():
        print(f"ERROR: ckProfiler not found in '{bin_path}'.", file=sys.stderr)
        return 1

    report_dir = args.report_dir or bin_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which architecture's expected values to use.
    if args.arch:
        arch = args.arch.strip().lower()
        arch_source = "--arch override"
    else:
        arch, arch_source = lib.detect_arch()
        if arch is None:
            arch = lib.DEFAULT_ARCH
            arch_source = f"auto-detect failed, defaulting to '{lib.DEFAULT_ARCH}'"

    cases = lib.parse_cases(args.cases, arch)
    if not cases:
        print(f"ERROR: no cases parsed from '{args.cases}'.", file=sys.stderr)
        return 1

    if args.only_thresholded:
        cases = [c for c in cases if c.expected is not None]
        if not cases:
            print("ERROR: no thresholded cases found.", file=sys.stderr)
            return 1

    print(f"Running {len(cases)} regression case(s) from '{args.cases}'")
    print(f"Architecture: {arch}  ({arch_source})")
    print(f"Binary path: {bin_path}\n")

    results: list[lib.Result] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {lib._SECTION_TITLE[case.section]}  {case.args}")
        r = lib.run_case(bin_path, case, args.verbose)
        results.append(r)
        v = r.verdict(args.tolerance)
        if r.ran:
            dp = r.delta_pct
            delta = "" if dp is None else f"  (expected {case.expected:.0f}, {dp:+.1f}%)"
            print(f"  {v}  {r.tflops:.2f} TFLOPS{delta}")
            print(f"       best: {r.best_instance}")
        else:
            print(f"  {v}  {r.error or 'did not run'}")
            for fi in r.failed_instances:
                print(f"       [Error] {fi}")

    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "arch": f"{arch} ({arch_source})",
        "bin_path": str(bin_path),
        "cases_file": str(args.cases),
    }
    md = lib.render_markdown(results, args.tolerance, meta)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped = report_dir / f"direct_conv_regression_{stamp}.md"
    latest = report_dir / "direct_conv_regression_latest.md"
    timestamped.write_text(md)
    latest.write_text(md)
    print(f"\nReport written to:\n  {timestamped}\n  {latest}")

    if args.save_baseline:
        baseline = report_dir / "direct_conv_regression_baseline.md"
        baseline.write_text(md)
        print(f"  {baseline}  (baseline)")

    passed = sum(1 for r in results if r.verdict(args.tolerance) == "PASS")
    failed = sum(1 for r in results if r.verdict(args.tolerance) == "FAIL")
    info = sum(1 for r in results if r.verdict(args.tolerance) == "INFO")
    print(f"\nResult: {passed} passed, {failed} failed, {info} report-only")

    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> int:
    bin_path = Path(args.bin_path)
    if not (bin_path / "ckProfiler").exists():
        print(f"ERROR: ckProfiler not found in '{bin_path}'.", file=sys.stderr)
        return 1

    cases = lib.parse_cases(args.cases, arch=None)
    if not cases:
        print(f"ERROR: no cases parsed from '{args.cases}'.", file=sys.stderr)
        return 1

    print(f"Comparing {len(cases)} case(s) from '{args.cases}'\n")

    labels: list[str] = []
    igemm_best: list[float | None] = []
    igemm_names: list[str | None] = []
    direct_best: list[float | None] = []
    direct_names: list[str | None] = []
    direct_statuses: list[lib.DirectConvStatus] = []

    for i, case in enumerate(cases, 1):
        label = lib.compare_label(case)
        labels.append(label)
        print(f"[{i}/{len(cases)}] {lib._SECTION_TITLE[case.section]}  {label}")

        stdout, stderr, returncode = lib.run_profiler(
            bin_path, case.binary, case.args, timeout=300
        )
        if returncode != 0 and not stdout:
            print("  ERROR detected — case marked as failed.")
            igemm_best.append(None)
            igemm_names.append(None)
            direct_best.append(None)
            direct_names.append(None)
            direct_statuses.append(lib.DirectConvStatus.INCORRECT)
            continue

        ig, ig_name = lib.parse_valid_perf(stdout, args.igemm_prefix)
        dc, dc_name = lib.parse_valid_perf(stdout, args.direct_prefix)
        dc_status = lib.direct_conv_status(stderr, dc)

        print(f"  iGEMM best:       {ig:.4f} TFlops  ({ig_name})" if ig else "  iGEMM best:       N/A")
        if dc_status == lib.DirectConvStatus.INCORRECT:
            print(f"  Direct conv best: INCORRECT (stderr: {stderr.strip()[:120]})")
        elif dc_status == lib.DirectConvStatus.NO_INSTANCE:
            print("  Direct conv best: no applicable instance")
        else:
            print(f"  Direct conv best: {dc:.4f} TFlops  ({dc_name})")

        igemm_best.append(ig)
        igemm_names.append(ig_name)
        direct_best.append(dc)
        direct_names.append(dc_name)
        direct_statuses.append(dc_status)

    md = lib.render_compare_markdown(
        labels, igemm_best, igemm_names, direct_best, direct_names, direct_statuses
    )
    md_path = Path(args.markdown)
    md_path.write_text(md)
    print(f"\nMarkdown summary saved to {md_path}")

    if args.plot:
        lib.make_figure(labels, igemm_best, direct_best, direct_statuses, Path(args.plot))

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    default_cases = here / "direct_conv_cases.txt"

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_shared(p: argparse.ArgumentParser, *, bin_required: bool) -> None:
        p.add_argument("--bin-path", required=bin_required,
                       default=None if bin_required else here.parent / "build-gfx950" / "bin",
                       type=Path,
                       help="Directory containing the ckProfiler executable.")
        p.add_argument("--cases", default=default_cases, type=Path,
                       help="Path to the cases file "
                            "(default: direct_conv_cases.txt next to this script).")
        p.add_argument("--verbose", "-v", action="store_true",
                       help="Print commands and full stdout for each case.")

    # run -------------------------------------------------------------------
    p_run = sub.add_parser("run", help="Smoke / correctness: run cases, text summary.")
    add_shared(p_run, bin_required=True)
    p_run.add_argument("--category", "-c",
                       help="Only run cases whose section matches this substring "
                            "(e.g. 'fwd', 'bwd').")
    p_run.add_argument("--dtype", choices=sorted(_DTYPE_TOKEN),
                       help="Only run cases of this data type (fp16 or bf16).")
    p_run.set_defaults(func=cmd_run)

    # regress ---------------------------------------------------------------
    p_reg = sub.add_parser("regress", help="Perf gating against per-arch expected values.")
    add_shared(p_reg, bin_required=False)
    p_reg.add_argument("--tolerance", default=lib.DEFAULT_TOLERANCE, type=float,
                       help="Fractional tolerance below expected TFLOPS (default: 0.10).")
    p_reg.add_argument("--arch", default=None,
                       help="GPU architecture key for expected-value selection "
                            "(e.g. mi355, mi350). Default: auto-detect.")
    p_reg.add_argument("--save-baseline", action="store_true",
                       help="Also write the report as direct_conv_regression_baseline.md.")
    p_reg.add_argument("--report-dir", default=None, type=Path,
                       help="Directory for the markdown report (default: parent of --bin-path).")
    p_reg.add_argument("--only-thresholded", action="store_true",
                       help="Only run cases that carry an expected (|) value.")
    p_reg.set_defaults(func=cmd_regress)

    # compare ---------------------------------------------------------------
    p_cmp = sub.add_parser("compare", help="iGEMM vs direct conv comparison.")
    add_shared(p_cmp, bin_required=True)
    p_cmp.add_argument("--plot", default=None,
                       help="Output PNG path (lazy matplotlib). Omit to skip the figure.")
    p_cmp.add_argument("--markdown", default="ck_profiler_comparison.md",
                       help="Output markdown path (default: ck_profiler_comparison.md).")
    p_cmp.add_argument("--igemm-prefix", default="GroupedConvolutionForwardKernel",
                       help="Instance-name prefix identifying iGEMM kernels.")
    p_cmp.add_argument("--direct-prefix", default="direct_tile_conv",
                       help="Instance-name prefix identifying direct-conv kernels.")
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
