#!/usr/bin/env python3
"""Direct convolution regression test.

Runs the FWD / BWD-data profiler case matrix defined in
``direct_conv_regression_cases.txt`` (mirroring the tables in
``docs/direct_convolution/kernel_configuration_refactoring_plan.md``), parses the
best-configuration performance from the profiler output, compares against the
expected TFLOPS for each case (10% tolerance, improvements always accepted), and
writes a markdown report under the build directory.

Expected TFLOPS are architecture-specific: the cases file may carry per-arch
values (e.g. ``mi355=573 mi350=488``). The GPU architecture is auto-detected
from ``rocminfo`` / ``rocm-smi`` (override with ``--arch``) and the matching
value is selected.

Use this after each refactoring stage to detect correctness or performance
regressions:

    python3 direct_conv_regression_test.py --bin-path <build>/bin

To establish a baseline:

    python3 direct_conv_regression_test.py --bin-path <build>/bin --save-baseline

Parsing of the profiler "Best configuration parameters" block is reused from
``test_direct_conv.py`` to avoid duplication.
"""

import argparse
import datetime
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the profiler-output parser from the sibling script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_direct_conv import parse_best_perf  # noqa: E402


# Default tolerance: a case fails only if it is more than this fraction *below*
# its expected value. Performance improvements above expected are accepted.
DEFAULT_TOLERANCE = 0.10

# Architecture key used when the GPU cannot be detected or no per-arch value
# matches. Falls back to the original reference values.
DEFAULT_ARCH = "mi355"

# Bare "expected=<v>" tokens are stored under this key as an
# architecture-independent fallback.
_FALLBACK_KEY = "expected"


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

# Known AMD Instinct marketing names mapped to the short keys used in the cases
# file. Matching is done case-insensitively against the detected product name.
_ARCH_PATTERNS = [
    ("mi355", re.compile(r"mi355", re.IGNORECASE)),
    ("mi350", re.compile(r"mi350", re.IGNORECASE)),
]


def detect_arch() -> tuple[str | None, str]:
    """Detect the GPU architecture key (e.g. "mi350").

    Returns a ``(arch_key, source)`` tuple. ``arch_key`` is ``None`` if no
    known architecture could be matched; ``source`` is a human-readable
    description of how the detection was made (for the report).
    """
    probes = [
        (["rocminfo"], r"Marketing Name:\s*(.+)"),
        (["rocm-smi", "--showproductname"], r"Card Series:\s*(.+)"),
    ]
    for cmd, pat in probes:
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        for m in re.finditer(pat, proc.stdout):
            name = m.group(1).strip()
            for key, rx in _ARCH_PATTERNS:
                if rx.search(name):
                    return key, f"{cmd[0]} -> '{name}'"
    return None, "auto-detect failed"


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

_SECTION_BINARY = {
    "fwd": "grouped_conv_fwd_tile",
    "bwd_data": "grouped_conv_bwd_data_tile",
}


@dataclass
class Case:
    section: str          # "fwd" or "bwd_data"
    binary: str           # profiler subcommand
    args: str             # space-separated argument string
    expected: float | None  # expected TFLOPS for the selected arch, or None
    expected_by_arch: dict[str, float] = field(default_factory=dict)


def _parse_expected_suffix(suffix: str) -> dict[str, float]:
    """Parse "mi355=573 mi350=488" or legacy "expected=573" into a dict."""
    values: dict[str, float] = {}
    for token in suffix.split():
        if "=" not in token:
            continue
        key, _, val = token.partition("=")
        try:
            values[key.strip().lower()] = float(val)
        except ValueError:
            continue
    return values


def _select_expected(values: dict[str, float], arch: str | None) -> float | None:
    """Pick the expected value for ``arch`` with sensible fallbacks."""
    if not values:
        return None
    if arch is not None and arch in values:
        return values[arch]
    if _FALLBACK_KEY in values:
        return values[_FALLBACK_KEY]
    if DEFAULT_ARCH in values:
        return values[DEFAULT_ARCH]
    return None


def parse_cases(path: Path, arch: str | None) -> list[Case]:
    cases: list[Case] = []
    section: str | None = None

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        lower = line.lower()
        if not line[0].isdigit():
            if "bwd" in lower and "data" in lower:
                section = "bwd_data"
            elif "fwd" in lower:
                section = "fwd"
            # else: a column-header line inside a section -> ignore
            continue

        if section is None:
            continue

        # Split optional "| <key>=<val> ..." suffix.
        expected_by_arch: dict[str, float] = {}
        body = line
        if "|" in line:
            body, _, suffix = line.partition("|")
            body = body.strip()
            expected_by_arch = _parse_expected_suffix(suffix.strip())

        cases.append(
            Case(
                section=section,
                binary=_SECTION_BINARY[section],
                args=" ".join(body.split()),
                expected=_select_expected(expected_by_arch, arch),
                expected_by_arch=expected_by_arch,
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Running cases
# ---------------------------------------------------------------------------

@dataclass
class Result:
    case: Case
    ran: bool
    best_instance: str = ""
    avg_time_ms: float = 0.0
    tflops: float = 0.0
    gb_s: float = 0.0
    error: str = ""

    @property
    def delta_pct(self) -> float | None:
        if self.case.expected is None or self.case.expected == 0:
            return None
        return (self.tflops - self.case.expected) / self.case.expected * 100.0

    def verdict(self, tolerance: float) -> str:
        if not self.ran:
            return "FAIL"
        if self.case.expected is None:
            return "INFO"
        # Accept if within -tolerance of expected (improvements always pass).
        if self.tflops >= self.case.expected * (1.0 - tolerance):
            return "PASS"
        return "FAIL"


def run_case(bin_path: Path, case: Case, verbose: bool) -> Result:
    exe = bin_path / "ckProfiler"
    cmd = [str(exe), case.binary] + case.args.split()

    if verbose:
        print(f"  $ {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError:
        return Result(case=case, ran=False, error=f"executable not found: {exe}")

    stderr = proc.stderr.strip()
    name, avg_time, tflops, gb_s = parse_best_perf(proc.stdout)

    # A case "ran" if we found a best configuration and there was no stderr noise.
    ran = bool(name) and len(stderr) == 0
    if verbose and proc.stdout:
        for ln in proc.stdout.splitlines():
            print(f"    {ln}")

    return Result(
        case=case,
        ran=ran,
        best_instance=name,
        avg_time_ms=avg_time,
        tflops=tflops,
        gb_s=gb_s,
        error=stderr,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_SECTION_TITLE = {"fwd": "FWD", "bwd_data": "BWD data"}


def render_markdown(results: list[Result], tolerance: float, meta: dict) -> str:
    lines: list[str] = []
    lines.append("# Direct convolution regression report")
    lines.append("")
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append(f"- **tolerance**: {tolerance * 100:.0f}% below expected")
    lines.append("")

    passed = sum(1 for r in results if r.verdict(tolerance) == "PASS")
    failed = sum(1 for r in results if r.verdict(tolerance) == "FAIL")
    info = sum(1 for r in results if r.verdict(tolerance) == "INFO")
    lines.append(
        f"**Result: {passed} passed, {failed} failed, {info} report-only "
        f"({len(results)} total)**"
    )
    lines.append("")

    for section in ("fwd", "bwd_data"):
        sec_results = [r for r in results if r.case.section == section]
        if not sec_results:
            continue
        lines.append(f"## {_SECTION_TITLE[section]} cases")
        lines.append("")
        lines.append(
            "| # | Verdict | TFLOPS | Expected | Delta% | Time(ms) | GB/s | "
            "Best instance | Args |"
        )
        lines.append(
            "|---|---------|--------|----------|--------|----------|------|"
            "---------------|------|"
        )
        for i, r in enumerate(sec_results, 1):
            exp = "-" if r.case.expected is None else f"{r.case.expected:.0f}"
            dp = r.delta_pct
            delta = "-" if dp is None else f"{dp:+.1f}"
            if r.ran:
                tflops = f"{r.tflops:.2f}"
                time = f"{r.avg_time_ms:.4f}"
                gbs = f"{r.gb_s:.1f}"
                inst = r.best_instance
            else:
                tflops = time = gbs = "N/A"
                inst = r.error or "did not run"
            lines.append(
                f"| {i} | {r.verdict(tolerance)} | {tflops} | {exp} | {delta} "
                f"| {time} | {gbs} | `{inst}` | `{r.case.args}` |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bin-path",
        default=here.parent / "build-gfx950" / "bin",
        type=Path,
        help="Directory containing the ckProfiler executable "
        "(default: ../build-gfx950/bin).",
    )
    parser.add_argument(
        "--cases",
        default=here / "direct_conv_regression_cases.txt",
        type=Path,
        help="Path to the regression cases file.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        type=Path,
        help="Directory to write the markdown report "
        "(default: parent of --bin-path).",
    )
    parser.add_argument(
        "--tolerance",
        default=DEFAULT_TOLERANCE,
        type=float,
        help="Fractional tolerance below expected TFLOPS (default: 0.10).",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Also write the report as direct_conv_regression_baseline.md.",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="GPU architecture key for expected-value selection "
        "(e.g. mi355, mi350). Default: auto-detect from rocminfo/rocm-smi.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    bin_path = args.bin_path
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
        arch, arch_source = detect_arch()
        if arch is None:
            arch = DEFAULT_ARCH
            arch_source = f"auto-detect failed, defaulting to '{DEFAULT_ARCH}'"

    cases = parse_cases(args.cases, arch)
    if not cases:
        print(f"ERROR: no cases parsed from '{args.cases}'.", file=sys.stderr)
        return 1

    print(f"Running {len(cases)} regression case(s) from '{args.cases}'")
    print(f"Architecture: {arch}  ({arch_source})")
    print(f"Binary path: {bin_path}\n")

    results: list[Result] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {_SECTION_TITLE[case.section]}  {case.args}")
        r = run_case(bin_path, case, args.verbose)
        results.append(r)
        v = r.verdict(args.tolerance)
        if r.ran:
            dp = r.delta_pct
            delta = "" if dp is None else f"  (expected {case.expected:.0f}, {dp:+.1f}%)"
            print(f"  {v}  {r.tflops:.2f} TFLOPS{delta}")
            print(f"       best: {r.best_instance}")
        else:
            print(f"  {v}  {r.error or 'did not run'}")

    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "arch": f"{arch} ({arch_source})",
        "bin_path": str(bin_path),
        "cases_file": str(args.cases),
    }
    md = render_markdown(results, args.tolerance, meta)

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


if __name__ == "__main__":
    sys.exit(main())
