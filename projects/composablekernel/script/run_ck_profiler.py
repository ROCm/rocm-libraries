#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIRECT_TILE_PREFIX = "direct_tile_conv"
IGEMM_PREFIX = "GroupedConvolutionForwardKernel"

# Extracts TFLOPS and instance name from a [Valid] Perf line
PERF_NAME_RE = re.compile(
    r"\[Valid\]\s+Perf:\s+[\d.]+ ms,\s+([\d.]+) TFlops,\s+[\d.]+ GB/s,\s+(\S+)"
)


def parse_test_cases(test_cases_path: Path) -> list[list[str]]:
    """Return a list of argument lists, one per non-empty, non-comment line."""
    cases = []
    with open(test_cases_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Remove commas (some lines have "150, 150" style values)
            line = line.replace(",", " ")
            args = line.split()
            # Skip the header line (starts with non-numeric tokens like "data_type")
            if not args[0].lstrip("-").isdigit():
                continue
            cases.append(args)
    return cases


def make_label(args: list[str]) -> str:
    """Create a short human-readable label for a test case from its arguments.

    Column order from test-cases.txt header:
    data_type layout indexing_type verify init_type print time_kernel nDims
    G N K C Y X Hi Wi Sx Sy Dy Dx LPy Lpx RPy RPx
    """
    try:
        # nDims is args[7]; after that: G N K C Y X Hi Wi ...
        g, n, k, c, y, x, hi, wi = args[8:16]
        return f"G{g}N{n}K{k}C{c}_{y}x{x}_{hi}x{wi}"
    except (IndexError, ValueError):
        return " ".join(args)


def run_profiler(binary_dir: Path, args: list[str]) -> tuple[str, bool]:
    """Run ckProfiler with args and return (stdout+stderr, had_error)."""
    exe = binary_dir / "ckProfiler"
    cmd = [str(exe)] + ["grouped_conv_fwd_tile"] + args
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )
        output = result.stdout
        had_error = result.returncode != 0
        return output, had_error
    except subprocess.TimeoutExpired:
        return "TIMEOUT", True
    except Exception as e:
        return f"ERROR: {e}", True


def parse_best(output: str, instance_prefix: str) -> tuple[float, str] | tuple[None, None]:
    """Return (best_tflops, kernel_name) for instances matching instance_prefix, or (None, None)."""
    best_val = None
    best_name = None

    for line in output.splitlines():
        if "[Valid]" not in line or instance_prefix not in line:
            continue
        m = PERF_NAME_RE.search(line)
        if m:
            val = float(m.group(1))
            name = m.group(2)
            if best_val is None or val > best_val:
                best_val = val
                best_name = name

    return best_val, best_name


def check_output_for_errors(output: str) -> bool:
    """Return True if the output contains signs of an error beyond 'returncode != 0'."""
    lower = output.lower()
    error_keywords = ["error:", "exception", "segfault", "illegal instruction", "timeout"]
    for kw in error_keywords:
        if kw in lower:
            return True
    return False


def make_figure(
    labels: list[str],
    igemm_values: list[float | None],
    direct_values: list[float | None],
    output_path: Path,
) -> None:
    """Save a grouped bar chart comparing CK Tile iGEMM vs CK Tile Direct Conv TFLOPS."""
    n = len(labels)
    x = np.arange(n)
    width = 0.35

    igemm_vals = [v if v is not None else 0.0 for v in igemm_values]
    direct_vals = [v if v is not None else 0.0 for v in direct_values]
    igemm_failed = [v is None for v in igemm_values]
    direct_failed = [v is None for v in direct_values]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), 6))

    bars_igemm = ax.bar(x - width / 2, igemm_vals, width, label="iGEMM", color="steelblue")
    bars_direct = ax.bar(x + width / 2, direct_vals, width, label="Direct Conv", color="darkorange")

    # Mark failed bars
    for i, (bar, failed) in enumerate(zip(bars_igemm, igemm_failed)):
        if failed:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.5,
                "FAIL",
                ha="center",
                va="bottom",
                fontsize=7,
                color="red",
                rotation=90,
            )
    for i, (bar, failed) in enumerate(zip(bars_direct, direct_failed)):
        if failed:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.5,
                "FAIL",
                ha="center",
                va="bottom",
                fontsize=7,
                color="red",
                rotation=90,
            )

    ax.set_xlabel("Test case")
    ax.set_ylabel("Best TFLOPS")
    ax.set_title("iGEMM vs Direct Conv — Best TFLOPS per test case")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Figure saved to {output_path}")


def _write_markdown(
    labels: list[str],
    igemm_best: list[float | None],
    igemm_names: list[str | None],
    direct_best: list[float | None],
    direct_names: list[str | None],
    md_path: Path,
) -> None:
    """Write the summary table as a Markdown file."""
    header = (
        "| Test case | iGEMM (TFlops) | Best iGEMM kernel"
        " | Direct Conv (TFlops) | Best Direct kernel | Improvement |\n"
        "|-----------|---------------:|-------------------|"
        "---------------------:|--------------------|------------:|"
    )
    rows = [header]
    for label, ig, ig_name, dc, dc_name in zip(
        labels, igemm_best, igemm_names, direct_best, direct_names
    ):
        ig_str = f"{ig:.4f}" if ig else "FAIL"
        dc_str = f"{dc:.4f}" if dc else "FAIL"
        ig_name_str = f"`{ig_name}`" if ig_name else "—"
        dc_name_str = f"`{dc_name}`" if dc_name else "—"
        improvement_str = f"{dc/ig:.3f}x" if ig and dc else "N/A"
        rows.append(
            f"| {label} | {ig_str} | {ig_name_str} | {dc_str} | {dc_name_str} | {improvement_str} |"
        )

    with open(md_path, "w") as f:
        f.write("# CK Profiler: iGEMM vs Direct Conv\n\n")
        f.write("\n".join(rows) + "\n")

    print(f"Markdown summary saved to {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run CK Profiler test cases and compare iGEMM vs Direct Conv performance."
    )
    parser.add_argument(
        "--binary-dir",
        help="Directory containing the ckProfiler executable.",
    )
    parser.add_argument(
        "--test-cases",
        help="Path to the test cases .txt file.",
    )
    parser.add_argument(
        "--output",
        default="ck_profiler_comparison.png",
        help="Output figure path (default: ck_profiler_comparison.png).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    binary_dir = Path(args.binary_dir)
    test_cases_path = Path(args.test_cases)
    output_path = Path(args.output)

    if not (binary_dir / "ckProfiler").exists() and not args.dry_run:
        print(f"ERROR: ckProfiler not found in {binary_dir}", file=sys.stderr)
        sys.exit(1)

    cases = parse_test_cases(test_cases_path)
    if not cases:
        print("No test cases found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(cases)} test case(s).")

    labels = []
    igemm_best = []
    igemm_names = []
    direct_best = []
    direct_names = []

    for i, case_args in enumerate(cases):
        label = make_label(case_args)
        labels.append(label)
        print(f"\n[{i+1}/{len(cases)}] Running: {label}")

        if args.dry_run:
            print(f"  CMD: ckProfiler grouped_conv_fwd_tile {' '.join(case_args)}")
            igemm_best.append(None)
            igemm_names.append(None)
            direct_best.append(None)
            direct_names.append(None)
            continue

        output, had_error = run_profiler(binary_dir, case_args)

        if had_error or check_output_for_errors(output):
            print(f"  ERROR detected — case marked as failed.")
            igemm_best.append(None)
            igemm_names.append(None)
            direct_best.append(None)
            direct_names.append(None)
            continue

        ig, ig_name = parse_best(output, IGEMM_PREFIX)
        dc, dc_name = parse_best(output, DIRECT_TILE_PREFIX)

        print(f"  iGEMM best:       {ig:.4f} TFlops  ({ig_name})" if ig else "  iGEMM best:       N/A")
        print(f"  Direct conv best: {dc:.4f} TFlops  ({dc_name})" if dc else "  Direct conv best: N/A")

        igemm_best.append(ig)
        igemm_names.append(ig_name)
        direct_best.append(dc)
        direct_names.append(dc_name)

    if not args.dry_run:
        make_figure(labels, igemm_best, direct_best, output_path)

        md_path = output_path.with_suffix(".md")
        _write_markdown(labels, igemm_best, igemm_names, direct_best, direct_names, md_path)

        # Print summary table to stdout
        print("\n" + "=" * 80)
        print(f"{'Test case':<30} {'iGEMM (TF)':>12} {'Direct (TF)':>12} {'Improvement':>12}")
        print("=" * 80)
        for label, ig, dc in zip(labels, igemm_best, direct_best):
            ig_str = f"{ig:.4f}" if ig else "FAIL"
            dc_str = f"{dc:.4f}" if dc else "FAIL"
            improvement_str = f"{dc/ig:.3f}x" if ig and dc else "N/A"
            print(f"{label:<30} {ig_str:>12} {dc_str:>12} {improvement_str:>12}")
        print("=" * 80)


if __name__ == "__main__":
    main()
