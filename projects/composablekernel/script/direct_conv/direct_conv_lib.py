#!/usr/bin/env python3
"""Shared library for direct-convolution profiler tooling.

This module consolidates the case-file parsing, GPU-architecture detection,
profiler invocation, output parsing, and reporting. 
It is consumed by the ``direct_conv_bench.py`` CLI.

Sections:
  - cases    : sectioned case-file parser (FWD / BWD-data -> binary), optional
               ``| key=val …`` per-arch expected suffix.
  - arch     : ``detect_arch`` + per-arch expected-value selection.
  - profiler : single ``run_profiler`` subprocess runner.
  - parse    : ``parse_best_perf`` (best-config block), ``parse_valid_perf``
               (per-instance ``[Valid]`` lines), ``parse_failed_instances``.
  - report   : ``Result`` dataclass + verdict logic, text/markdown renderers,
               and a lazily-imported matplotlib comparison plot.
"""

import enum
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Data-type token (first profiler argument) keyed by short name.
DTYPE_TOKEN = {"fp16": "1", "bf16": "2"}


# ===========================================================================
# arch
# ===========================================================================

# Default tolerance: a case fails only if it is more than this fraction *below*
# its expected value. Performance improvements above expected are accepted.
DEFAULT_TOLERANCE = 0.075

# Bare "expected=<v>" tokens are stored under this key as an
# architecture-independent fallback.
_FALLBACK_KEY = "expected"

# Known AMD Instinct marketing names mapped to the short keys used in the cases
# file. Matching is done case-insensitively against the detected product name.
_ARCH_PATTERNS = [
    ("mi355", re.compile(r"mi355", re.IGNORECASE)),
    ("mi350", re.compile(r"mi350", re.IGNORECASE)),
    ("mi300", re.compile(r"mi300", re.IGNORECASE)),
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


# ===========================================================================
# cases
# ===========================================================================

_SECTION_BINARY = {
    "fwd": "grouped_conv_fwd_tile",
    "bwd_data": "grouped_conv_bwd_data_tile",
}

_SECTION_TITLE = {"fwd": "FWD", "bwd_data": "BWD data"}

# Reverse map: profiler subcommand token -> section. Lets a case file lead each
# line with the subcommand (e.g. "grouped_conv_fwd_tile <args...>") instead of
# grouping rows under a section header.
_BINARY_SECTION = {v: k for k, v in _SECTION_BINARY.items()}


@dataclass
class Case:
    section: str            # "fwd" or "bwd_data"
    binary: str             # profiler subcommand
    args: str               # space-separated argument string
    expected: float | None  # expected TFLOPS for the selected arch, or None
    expected_by_arch: dict[str, float] = field(default_factory=dict)

    @property
    def data_type(self) -> str:
        """First argument token: "1" (FP16) or "2" (BF16)."""
        toks = self.args.split()
        return toks[0] if toks else ""

    @property
    def group_count(self) -> int | None:
        """Group count (G) token, located by section column layout.

        FWD rows carry an extra ``indexing_type`` column, so G sits one column
        later than in BWD-data rows.
        """
        toks = self.args.split()
        idx = 8 if self.section == "fwd" else 7
        try:
            return int(toks[idx])
        except (IndexError, ValueError):
            return None

    @property
    def filter_size(self) -> str | None:
        """Convolution filter size as ``"<Y>x<X>"`` (e.g. ``"3x3"``).

        The Y/X columns follow G N K C; FWD rows carry an extra
        ``indexing_type`` column, so they sit one column later than in BWD-data
        rows.
        """
        toks = self.args.split()
        y_idx = 12 if self.section == "fwd" else 11
        try:
            return f"{int(toks[y_idx])}x{int(toks[y_idx + 1])}"
        except (IndexError, ValueError):
            return None


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
    return None


def _make_case(section: str, body: str, arch: str | None) -> Case:
    """Build a ``Case`` from an argument string with an optional ``| key=val`` suffix."""
    expected_by_arch: dict[str, float] = {}
    if "|" in body:
        body, _, suffix = body.partition("|")
        expected_by_arch = _parse_expected_suffix(suffix.strip())
    return Case(
        section=section,
        binary=_SECTION_BINARY[section],
        args=" ".join(body.split()),
        expected=_select_expected(expected_by_arch, arch),
        expected_by_arch=expected_by_arch,
    )


def parse_cases(path: Path, arch: str | None) -> list[Case]:
    """Parse a case file in either supported layout.

    Two line shapes are accepted (and may be mixed):
      - **Subcommand-prefixed**: a line leading with a known profiler
        subcommand (``grouped_conv_fwd_tile`` / ``grouped_conv_bwd_data_tile``)
        is a self-contained case; the section comes from the token and the rest
        is the verbatim argument string.
      - **Sectioned**: a section header ("FWD" / "BWD data") selects the section
        for the digit-first case rows that follow it.

    In both shapes ``#`` comments and blank lines are ignored, and a case may
    carry an optional trailing ``| key=val …`` suffix giving expected TFLOPS.
    """
    cases: list[Case] = []
    section: str | None = None

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        first_tok = line.split()[0]
        if first_tok in _BINARY_SECTION:
            cases.append(
                _make_case(_BINARY_SECTION[first_tok],
                           line[len(first_tok):].strip(), arch)
            )
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

        cases.append(_make_case(section, line, arch))

    return cases


def parse_miopen_cases(path: Path, arch: str | None = None) -> list[Case]:
    """Parse a file of MIOpenDriver commands (one per line) into ``Case`` objects.

    Each command is converted to ckProfiler argument strings via
    ``convert_miopen_driver_to_tile_profiler`` (the same converter used
    standalone), yielding a FWD and/or BWD-data case per ``-F`` direction.
    MIOpen commands carry no expected performance, so cases are unthresholded
    (``expected=None``); ``arch`` is accepted for signature parity but unused.
    """
    # The converter lives one directory up (script/), next to this package.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import convert_miopen_driver_to_tile_profiler as conv

    parser = conv.build_parser()
    cases: list[Case] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            converted = conv.convert_to_profiler_cases(line, parser)
        except SystemExit:
            # The converter calls exit() on unsupported drivers/layouts/dtypes;
            # skip such lines rather than aborting the whole run.
            print(f"WARNING: skipping unsupported MIOpen command: {line}")
            continue
        for section, args in converted:
            cases.append(
                Case(
                    section=section,
                    binary=_SECTION_BINARY[section],
                    args=args,
                    expected=None,
                )
            )
    return cases


def filter_cases(
    cases: list[Case],
    category: str | None = None,
    dtype: str | list[str] | None = None,
    group_count: int | None = None,
    filter_size: str | list[str] | None = None,
) -> list[Case]:
    """Filter cases by section substring, data type(s), group count, and/or filter size.

    ``dtype`` may be a single name (e.g. ``"fp16"``) or a list of names
    (e.g. ``["fp16", "bf16"]``); only cases of those types are kept, which is
    how unsupported types such as fp32 are excluded.

    ``filter_size`` may be a single ``"<Y>x<X>"`` token (e.g. ``"3x3"``) or a
    list of them (e.g. ``["3x3", "1x1"]``); only cases with a matching
    convolution filter size are kept.
    """
    out = cases
    if category:
        flt = category.lower()
        out = [c for c in out if flt in c.section.lower()]
    if dtype:
        names = [dtype] if isinstance(dtype, str) else dtype
        toks = {DTYPE_TOKEN[d] for d in names}
        out = [c for c in out if c.data_type in toks]
    if group_count is not None:
        out = [c for c in out if c.group_count == group_count]
    if filter_size:
        sizes = [filter_size] if isinstance(filter_size, str) else filter_size
        wanted = {s.lower() for s in sizes}
        out = [c for c in out if c.filter_size in wanted]
    return out


# ===========================================================================
# profiler
# ===========================================================================

# Some lines the profiler (or its native runtime) writes to stderr are benign
# and must not be treated as a case failure. The classic offender is the
# OpenMP / ROCr thread-affinity setup racing against thread teardown, which
# prints "pthread_setaffinity_np failed: No such process" (ESRCH) -- the kernel
# still ran and verified fine. Extend this list with other known-benign noise.
_BENIGN_STDERR_RES = [
    re.compile(r"pthread_setaffinity_np\s+failed", re.IGNORECASE),
]


def filter_stderr(stderr: str) -> str:
    """Drop known-benign runtime noise from ``stderr``.

    Returns the stderr text with lines matching any ``_BENIGN_STDERR_RES``
    pattern removed (and surrounding blank lines trimmed). A run is only
    considered failed when real diagnostics remain after this filtering.
    """
    kept = [
        ln for ln in stderr.splitlines()
        if not any(rx.search(ln) for rx in _BENIGN_STDERR_RES)
    ]
    return "\n".join(kept).strip()


def run_profiler(
    bin_path: Path, binary: str, args: str, timeout: float | None = None
) -> tuple[str, str, int]:
    """Run ``ckProfiler <binary> <args>`` and return (stdout, stderr, returncode).

    On timeout, returns ("TIMEOUT", "", 1). If the executable is missing,
    returns ("", <message>, 127).
    """
    exe = bin_path / "ckProfiler"
    cmd = [str(exe), binary] + args.split()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return proc.stdout, proc.stderr, proc.returncode
    except FileNotFoundError:
        return "", f"executable not found: {exe}", 127
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "", 1


# ===========================================================================
# parse
# ===========================================================================

# Matches the "Best configuration parameters:" block.
_BEST_NAME_RE = re.compile(r"^\s*name:\s*(.+)$")
_BEST_TIME_RE = re.compile(r"^\s*avg_time:\s*([\d.]+)ms$")
_BEST_TFLOPS_RE = re.compile(r"^\s*tflops:\s*([\d.]+)$")
_BEST_GBS_RE = re.compile(r"^\s*GB/s:\s*([\d.]+)$")


def parse_best_perf(stdout: str) -> tuple[str, float, float, float]:
    """Return (name, avg_time_ms, tflops, gb_s) from the profiler best-config block."""
    in_best = False
    name = ""
    avg_time = 0.0
    tflops = 0.0
    gb_s = 0.0

    for line in stdout.splitlines():
        if "Best configuration parameters:" in line:
            in_best = True
            continue
        if not in_best:
            continue
        m = _BEST_NAME_RE.match(line)
        if m:
            name = m.group(1).strip()
            continue
        m = _BEST_TIME_RE.match(line)
        if m:
            avg_time = float(m.group(1))
            continue
        m = _BEST_TFLOPS_RE.match(line)
        if m:
            tflops = float(m.group(1))
            continue
        m = _BEST_GBS_RE.match(line)
        if m:
            gb_s = float(m.group(1))
            continue

    return name, avg_time, tflops, gb_s


# Extracts TFLOPS and instance name from a "[Valid] Perf:" line.
_PERF_NAME_RE = re.compile(
    r"\[Valid\]\s+Perf:\s+[\d.]+ ms,\s+([\d.]+) TFlops,\s+[\d.]+ GB/s,\s+(\S+)"
)


def parse_valid_perf(
    stdout: str, prefix: str
) -> tuple[float, str] | tuple[None, None]:
    """Return (best_tflops, kernel_name) over ``[Valid]`` lines matching ``prefix``."""
    best_val: float | None = None
    best_name: str | None = None

    for line in stdout.splitlines():
        if "[Valid]" not in line or prefix not in line:
            continue
        m = _PERF_NAME_RE.search(line)
        if m:
            val = float(m.group(1))
            name = m.group(2)
            if best_val is None or val > best_val:
                best_val = val
                best_name = name

    return best_val, best_name


# The profiler tags each instance on stdout with a status marker. Instances that
# fail verification are reported as "[Error] <name>, SplitK N"; "[Invalid]" is
# used for the same purpose by some profiler ops. We extract the kernel name so
# failures name the offending instance instead of dumping the raw mismatch diff.
_FAILED_INSTANCE_RE = re.compile(r"^\[(?:Error|Invalid)\]\s+(.+)$")
_SPLITK_SUFFIX_RE = re.compile(r",\s*SplitK\s+\S+\s*$")


def parse_failed_instances(stdout: str) -> list[str]:
    """Return the names of instances the profiler flagged as Error/Invalid."""
    names: list[str] = []
    for line in stdout.splitlines():
        m = _FAILED_INSTANCE_RE.match(line.strip())
        if not m:
            continue
        name = _SPLITK_SUFFIX_RE.sub("", m.group(1).strip()).strip()
        if name:
            names.append(name)
    return names


# ===========================================================================
# report
# ===========================================================================

# The profiler reports "name:  (instance -1)" in its best-config block when no
# applicable instance exists for the problem (empty instance name, index -1).
_NO_INSTANCE_RE = re.compile(r"^\(instance\s+-1\)$")


@dataclass
class Result:
    case: Case
    ran: bool
    best_instance: str = ""
    avg_time_ms: float = 0.0
    tflops: float = 0.0
    gb_s: float = 0.0
    error: str = ""
    failed_instances: list[str] = field(default_factory=list)

    @property
    def delta_pct(self) -> float | None:
        if self.case.expected is None or self.case.expected == 0:
            return None
        return (self.tflops - self.case.expected) / self.case.expected * 100.0

    @property
    def no_instance(self) -> bool:
        """True if the profiler found no applicable instance (best is "(instance -1)")."""
        return bool(_NO_INSTANCE_RE.match(self.best_instance.strip()))

    def verdict(self, tolerance: float) -> str:
        if not self.ran:
            return "FAIL"
        # No applicable instance for this problem -> nothing was exercised.
        if self.no_instance:
            return "NOT TESTED"
        if self.case.expected is None:
            return "INFO"
        # Accept if within -tolerance of expected (improvements always pass).
        if self.tflops >= self.case.expected * (1.0 - tolerance):
            return "PASS"
        return "FAIL"


def run_case(bin_path: Path, case: Case, verbose: bool = False) -> Result:
    """Run a single case and classify the result."""
    if verbose:
        print(f"  $ ckProfiler {case.binary} {case.args}")

    stdout, stderr, returncode = run_profiler(bin_path, case.binary, case.args)
    # Drop known-benign runtime noise (e.g. the pthread_setaffinity_np ESRCH
    # warning) so it does not get mistaken for a verification failure below.
    stderr = filter_stderr(stderr)
    name, avg_time, tflops, gb_s = parse_best_perf(stdout)
    failed = parse_failed_instances(stdout)

    if verbose and stdout:
        for ln in stdout.splitlines():
            print(f"    {ln}")

    if returncode == 127:
        return Result(case=case, ran=False, error=stderr or "executable not found")

    # A case "ran" if we found a best configuration and there was no stderr noise.
    ran = bool(name) and len(stderr) == 0

    # Build a concise error string. Prefer naming the offending instance(s) over
    # dumping the raw verification diff that the profiler writes to stderr.
    error = ""
    if not ran:
        if failed:
            error = "failing instance(s): " + ", ".join(failed)
        elif stderr:
            error = stderr.splitlines()[0].strip()
        elif not name:
            error = "no best configuration found"

    return Result(
        case=case,
        ran=ran,
        best_instance=name,
        avg_time_ms=avg_time,
        tflops=tflops,
        gb_s=gb_s,
        error=error,
        failed_instances=failed,
    )


def print_summary(results: list[Result]) -> None:
    """Print a plain-text per-section summary (smoke / ``run`` output)."""
    total = len(results)
    passed = sum(1 for r in results if r.ran)
    failed = total - passed

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for section in ("fwd", "bwd_data"):
        sec = [r for r in results if r.case.section == section]
        if not sec:
            continue
        print(f"\n[{_SECTION_TITLE[section]}]")
        print(f"  {'#':<4} {'Status':<6}  {'Time(ms)':>10}  {'TFlops':>8}  {'GB/s':>8}  Args")
        print(f"  {'-'*4}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*30}")
        for i, r in enumerate(sec, 1):
            status = "PASS" if r.ran else "FAIL"
            if r.ran and r.best_instance:
                print(
                    f"  {i:<4} {status:<6}  {r.avg_time_ms:>10.3f}  {r.tflops:>8.2f}"
                    f"  {r.gb_s:>8.1f}  {r.case.args}"
                )
                print(f"       {'':6}  best: {r.best_instance}")
            else:
                print(f"  {i:<4} {status:<6}  {'N/A':>10}  {'N/A':>8}  {'N/A':>8}  {r.case.args}")
                if r.error:
                    print(f"       error: {r.error}")

    print()
    print(f"Result: {passed}/{total} ran cleanly", end="")
    if failed:
        print(f", {failed} FAILED")
    else:
        print(" (all passed)")
    print("=" * 80)


def render_markdown(results: list[Result], tolerance: float, meta: dict) -> str:
    """Render the regression report as markdown."""
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
    not_tested = sum(1 for r in results if r.verdict(tolerance) == "NOT TESTED")
    lines.append(
        f"**Result: {passed} passed, {failed} failed, {info} report-only, "
        f"{not_tested} not tested ({len(results)} total)**"
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
                if r.failed_instances:
                    inst = "; ".join(r.failed_instances)
                else:
                    inst = r.error or "did not run"
            lines.append(
                f"| {i} | {r.verdict(tolerance)} | {tflops} | {exp} | {delta} "
                f"| {time} | {gbs} | `{inst}` | `{r.case.args}` |"
            )
        lines.append("")

    # Consolidated list of every instance the profiler flagged as incorrect,
    # de-duplicated across cases, so buggy instances are easy to copy/blacklist.
    failing = sorted({fi for r in results for fi in r.failed_instances})
    if failing:
        lines.append("## Failing instances")
        lines.append("")
        lines.append(
            "Instances flagged `[Error]`/`[Invalid]` by the profiler "
            "(incorrect results):"
        )
        lines.append("")
        for fi in failing:
            lines.append(f"- `{fi}`")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# compare: iGEMM vs direct conv
# ---------------------------------------------------------------------------

class DirectConvStatus(enum.Enum):
    OK = "ok"
    INCORRECT = "incorrect"     # stderr was non-empty -> wrong results
    NO_INSTANCE = "no_instance"  # ran cleanly but no applicable direct conv kernel


def direct_conv_status(
    stderr: str, dc_tflops: float | None
) -> DirectConvStatus:
    """Classify the direct conv outcome for a single run.

    Priority:
      1. Non-empty stderr -> profiler reported incorrect results. A numerical
         verification failure routes through run_cpu_validation -> check_err,
         which writes the mismatch to std::cerr, so this branch also covers
         "applicable instance(s) present but failed verification".
      2. No [Valid] direct conv line in stdout -> no applicable instance.
      3. Otherwise -> OK.

    Known-benign runtime noise (e.g. the pthread_setaffinity_np ESRCH warning)
    is stripped first so it is not misread as a verification failure.
    """
    if filter_stderr(stderr):
        return DirectConvStatus.INCORRECT
    if dc_tflops is None:
        return DirectConvStatus.NO_INSTANCE
    return DirectConvStatus.OK


# Instance-name prefixes identifying iGEMM (implicit-GEMM) and direct-conv
# kernels in the profiler's "[Valid] Perf:" lines, per section. Under
# CK_EXPERIMENTAL_BUILDER (the build used by the profiler) the iGEMM instance
# name is reflection-derived from the kernel struct name.
IGEMM_PREFIX = {
    "fwd": "GroupedConvolutionForwardKernel",
    "bwd_data": "GroupedConvolutionBackwardDataKernel",
}
DIRECT_PREFIX = {
    "fwd": "direct_tile_conv",
    "bwd_data": "direct_tile_conv",
}


def compare_label(case: Case) -> str:
    """Short human-readable label for a comparison case from its arguments.

    FWD rows carry an extra ``indexing_type`` column, so G/N/K/C start one
    column later than in BWD-data rows. A direction tag disambiguates rows that
    would otherwise share the same shape label.
      FWD: data_type layout indexing_type verify init print time nDims G N K C Y X Hi Wi
      BWD: data_type layout verify init print time nDims G N K C Y X Hi Wi
    """
    args = case.args.split()
    start = 8 if case.section == "fwd" else 7
    tag = "FWD" if case.section == "fwd" else "BWD"
    try:
        g, n, k, c, y, x, hi, wi = args[start:start + 8]
        return f"{tag} G{g}N{n}K{k}C{c}_{y}x{x}_{hi}x{wi}"
    except (IndexError, ValueError):
        return f"{tag} {case.args}"


_COMPARE_TITLE = "# CK Profiler: iGEMM vs Direct Conv"
_COMPARE_TABLE_HEADER = (
    "| Test case | iGEMM (TFlops) | Best iGEMM kernel"
    " | Direct Conv (TFlops) | Best Direct kernel | Direct status | Improvement |\n"
    "|-----------|---------------:|-------------------|"
    "---------------------:|--------------------|---------------|------------:|"
)
_DIRECT_STATUS_STR = {
    DirectConvStatus.OK: "✓ ok",
    DirectConvStatus.INCORRECT: "✗ incorrect",
    DirectConvStatus.NO_INSTANCE: "— no instance",
}


def compare_markdown_header() -> str:
    """Markdown title + table header (the part written once, before any rows)."""
    return f"{_COMPARE_TITLE}\n\n{_COMPARE_TABLE_HEADER}\n"


def compare_markdown_row(
    label: str,
    ig: float | None,
    ig_name: str | None,
    dc: float | None,
    dc_name: str | None,
    dc_status: DirectConvStatus,
) -> str:
    """Format a single comparison row (no trailing newline)."""
    ig_str = f"{ig:.4f}" if ig else "FAIL"
    dc_str = f"{dc:.4f}" if dc else "—"
    ig_name_str = f"`{ig_name}`" if ig_name else "—"
    dc_name_str = f"`{dc_name}`" if dc_name else "—"
    status_str = _DIRECT_STATUS_STR[dc_status]
    improvement_str = f"{dc/ig:.3f}x" if ig and dc else "N/A"
    return (
        f"| {label} | {ig_str} | {ig_name_str} | {dc_str} | {dc_name_str} "
        f"| {status_str} | {improvement_str} |"
    )


def render_compare_markdown(
    labels: list[str],
    igemm_best: list[float | None],
    igemm_names: list[str | None],
    direct_best: list[float | None],
    direct_names: list[str | None],
    direct_statuses: list[DirectConvStatus],
) -> str:
    """Render the full iGEMM-vs-direct comparison table as markdown."""
    rows = [
        compare_markdown_row(label, ig, ig_name, dc, dc_name, dc_status)
        for label, ig, ig_name, dc, dc_name, dc_status in zip(
            labels, igemm_best, igemm_names, direct_best, direct_names, direct_statuses
        )
    ]
    return compare_markdown_header() + "\n".join(rows) + "\n"


def make_figure(
    labels: list[str],
    igemm_values: list[float | None],
    direct_values: list[float | None],
    direct_statuses: list[DirectConvStatus],
    output_path: Path,
) -> None:
    """Save a grouped bar chart comparing iGEMM vs Direct Conv TFLOPS.

    matplotlib / numpy are imported lazily so that ``run`` / ``regress`` (which
    never call this) do not require them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(labels)
    x = np.arange(n)
    width = 0.35

    igemm_vals = [v if v is not None else 0.0 for v in igemm_values]
    direct_vals = [v if v is not None else 0.0 for v in direct_values]
    igemm_failed = [v is None for v in igemm_values]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), 6))

    bars_igemm = ax.bar(x - width / 2, igemm_vals, width, label="iGEMM", color="steelblue")
    bars_direct = ax.bar(x + width / 2, direct_vals, width, label="Direct Conv", color="darkorange")

    # Mark failed iGEMM bars.
    for bar, failed in zip(bars_igemm, igemm_failed):
        if failed:
            ax.text(
                bar.get_x() + bar.get_width() / 2, 0.5, "FAIL",
                ha="center", va="bottom", fontsize=7, color="red", rotation=90,
            )

    # Mark direct conv bars by status, and annotate relative perf on OK bars.
    _status_label = {
        DirectConvStatus.INCORRECT: ("INCORRECT", "red"),
        DirectConvStatus.NO_INSTANCE: ("N/A", "gray"),
    }
    for bar, status, ig, dc in zip(bars_direct, direct_statuses, igemm_values, direct_values):
        if status in _status_label:
            text, color = _status_label[status]
            ax.text(
                bar.get_x() + bar.get_width() / 2, 0.5, text,
                ha="center", va="bottom", fontsize=7, color=color, rotation=90,
            )
        elif status == DirectConvStatus.OK and ig and dc:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{dc / ig:.2f}x",
                ha="center", va="bottom", fontsize=7, color="black",
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
