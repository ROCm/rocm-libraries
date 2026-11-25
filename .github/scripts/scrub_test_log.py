#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Scrub gfx1250 CI output on its way to the public job log.

Why: the job runs on unreleased gfx1250 hardware and its log and artifacts are public.
Its performance must stay out of them: the TensileLite client's problem-progress rows
(GFLOPS, kernel times, clocks, temperatures) and pytest and TensileLite timings; neither
tool can turn them off. Host names and PCI addresses identify lab machines.
Device details pass through only where TheRock's public MI455 logs already show them:
name, memory, core clock, compute capability and launch limits.

What it does not do: decide pass or fail. gfx1250_blas_tests.py reads results from the
gtest and pytest XML reports, and uploads copies of those with timings and host names
removed. The scrubbed text logs are what a person reads to debug a failure.

Per line:
  1. drops pytest and TensileLite timing (``timing="pytest"``); gtest timing is off at
     the source (GTEST_PRINT_TIME=0);
  2. drops rocminfo-style device lines TheRock's public logs don't show (serial and
     unique IDs, VBIOS, firmware, CU count, L2 size, marketing name);
  3. redacts host names, PCI addresses, the PCI chip ID and the memory clock, which
     TheRock's public logs don't show either;
  4. blanks the per-hardware columns (clocks, temperatures, GFLOPS, ...) of the
     TensileLite client's ``problem-progress`` CSV rows.

gfx1250_blas_tests.py uses Scrubber for the test steps. As a filter (stdin to stdout)
it scrubs the build log: ``install.sh ... 2>&1 | python3 scrub_test_log.py``.
"""

import argparse
import re
import sys

# Lines dropped whole: rocminfo-style device fields that TheRock's public logs don't
# show, e.g. "  Serial Number:   1234". One label per alternative; \s+ allows any
# whitespace inside it. Partial redaction could leave the value behind, so the whole
# line goes.
_DROP = re.compile(
    r"Marketing\s+Name|Product\s+Name|Serial\s+Number|Unique\s+ID|VBIOS|[Ff]irmware"
    r"|multiProcessorCount|l2CacheSize|Compute\s+Unit"
)
# Values redacted inside a line, replaced by "<redacted>":
_REDACT = re.compile(
    # 1. AMD host names (anything.amd.com). The lookbehind starts a match only where a
    #    run of host characters starts; without it a long run with no ".amd.com" is
    #    rescanned from every position (quadratic).
    r"(?<![A-Za-z0-9.-])[A-Za-z0-9.-]+[.]amd[.]com"
    # 2. PCI addresses, domain:bus:device.function (0000:c1:00.0).
    r"|[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}[.][0-7]"
    # 3. PCI chip ID: hipblaslt-test's "(PCI Chip ID: 0x1234)" and the TensileLite
    #    client's "* Chip ID: 0x1234".
    r"|(?<=Chip ID: )0x[0-9A-Fa-f]+"
    # 4. Memory clock in hipblaslt-test's header: "max. MCLK 1000 MHz".
    r"|(?<=MCLK )[0-9]+"
)
_PROGRESS_HEADER = "run,problem-progress,"
_PROGRESS_ROW = re.compile(r"^[0-9]+,[0-9]+/[0-9]+,")
# problem-progress columns that measure this machine rather than the kernel's result.
_HW_COLUMNS = frozenset(
    "num-cus tiles-per-cu total-gran cu-gran wave-gran temp-edge clock-sys clock-soc"
    " clock-mem fan-rpm hardware-samples gfx-frequency(maximum) time-us gflops"
    " gflops-per-cu fastest-gflops peak-gflops efficiency gbps-bandwidth enqueue-time"
    " rocprof-counters".split()
)
# pytest summary/progress times and TensileLite's own phase timers, removed or cut down to
# their label. In order: "== 3 passed in 12.34s ==", "(0:20:34)", "(12.3 seconds)",
# "(5.0 secs elapsed)", "End - 1.2s" / "_Final 1.2s", "done (3.4s)".
_PYTEST_TIMING = [
    (re.compile(r" in [0-9]+(\.[0-9]+)?s\b"), ""),
    (re.compile(r" \([0-9]+:[0-9]{2}:[0-9]{2}\)"), ""),
    (re.compile(r" \([0-9][^()]* seconds\)"), ""),
    (re.compile(r" \([0-9]+(\.[0-9]+)? secs elapsed\)"), ""),
    (re.compile(r"(End -|_Final) [0-9]+(\.[0-9]+)?s\b"), r"\1"),
    (re.compile(r" done \([0-9]+(\.[0-9]+)?s\)"), " done"),
]


class Scrubber:
    """Line-by-line scrubber; keeps the last problem-progress header's column map."""

    def __init__(self, timing="none"):
        self._subs = _PYTEST_TIMING if timing == "pytest" else []
        self._header_fields = 0
        self._blank_from_end = set()

    def scrub(self, line):
        """Return the scrubbed line (no newline), or None to drop it."""
        for pattern, repl in self._subs:
            line = pattern.sub(repl, line)
        if _DROP.search(line):
            return None
        line = _REDACT.sub("<redacted>", line)
        if line.startswith(_PROGRESS_HEADER):
            # Problem sizes such as "(m,n,b,k)" add commas before these columns,
            # so they are located by their distance from the end.
            fields = line.split(",")
            self._header_fields = len(fields)
            self._blank_from_end = {
                len(fields) - 1 - i for i, f in enumerate(fields) if f in _HW_COLUMNS
            }
        elif self._header_fields and _PROGRESS_ROW.match(line):
            fields = line.split(",")
            if len(fields) < self._header_fields:
                return None  # truncated: drop rather than half-scrub
            for k in self._blank_from_end:
                fields[len(fields) - 1 - k] = ""
            line = ",".join(fields)
        return line


def text_stream(fileno, mode="r"):
    """A line stream on fileno that passes bytes that aren't UTF-8 through unchanged."""
    return open(
        fileno,
        mode,
        encoding="utf-8",
        errors="surrogateescape",
        newline="",
        closefd=False,
    )


def main(argv=None):
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    scrubber = Scrubber()
    stdout = text_stream(sys.stdout.fileno(), "w")
    for raw in text_stream(sys.stdin.fileno()):
        line = scrubber.scrub(raw.rstrip("\n"))
        if line is not None:
            stdout.write(line + "\n")
            stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
