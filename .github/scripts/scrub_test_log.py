#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Scrub a gfx1250 test or build log on its way to the public job log.

Reads stdin line by line and, per line:
  1. drops wall-clock timing (``--timing gtest|pytest``) and tool versions
     (``--versions``), so logs from different runs compare;
  2. drops lines that describe the hardware, and redacts product names,
     host names and PCI addresses;
  3. blanks the per-hardware columns (clocks, temperatures, GFLOPS, ...) of the
     client's ``problem-progress`` CSV rows;
  4. writes the line to ``--log`` as is and to stdout, prefixed ``[TAG] `` with
     ``--tag``, flushing each line so the job log streams.

Nothing unscrubbed reaches the file or stdout. Exits 0 at end of input; the
caller's pipeline status (``set -o pipefail``) still reports the producer's.
"""

import argparse
import re
import sys

_WS = r"[ \t\n\r\f\v]+"  # awk's [[:space:]]: ASCII whitespace only
# Lines that only describe the device (hipblaslt-test's header, rocminfo-style dumps).
_DROP = re.compile(
    "|".join(
        [
            rf"Query{_WS}device",
            rf"Device{_WS}ID{_WS}[0-9]",
            rf"PCI{_WS}Chip{_WS}ID",
            rf"GB{_WS}memory",
            "SCLK",
            "MCLK",
            rf"compute{_WS}capability",
            "maxGridDimX",
            "sharedMemPerBlock",
            "maxThreadsPerBlock",
            "warpSize",
            rf"Marketing{_WS}Name",
            rf"Product{_WS}Name",
            rf"Serial{_WS}Number",
            rf"Unique{_WS}ID",
            "VBIOS",
            "[Ff]irmware",
            "multiProcessorCount",
            "totalGlobalMem",
            "[Cc]lockRate",
            "l2CacheSize",
            rf"Compute{_WS}Unit",
            rf"Max{_WS}Clock",
        ]
    )
)
# Host names first: "mi300x.amd.com" is one host, not a product name plus a domain.
# A host name only starts where a run of host characters does; without the lookbehind a
# long run with no ".amd.com" is rescanned from every position (quadratic).
_REDACT = re.compile(
    r"(?<![A-Za-z0-9.-])[A-Za-z0-9.-]+[.]amd[.]com"
    r"|[Mm][Ii][0-9]{3}[A-Za-z]*"
    r"|[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}[.][0-7]"
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

_TIMING = {
    "none": [],
    # gtest's per-test and per-suite times.
    "gtest": [(re.compile(r" \([0-9]+ ms( total)?\)"), "")],
    # pytest summary/progress times and TensileLite's own phase timers.
    "pytest": [
        (re.compile(r" in [0-9]+(\.[0-9]+)?s\b"), ""),
        (re.compile(r" \([0-9]+:[0-9]{2}:[0-9]{2}\)"), ""),
        (re.compile(r" \([0-9][^()]* seconds\)"), ""),
        (re.compile(r" \([0-9]+(\.[0-9]+)? secs elapsed\)"), ""),
        (re.compile(r"(End -|_Final) [0-9]+(\.[0-9]+)?s\b"), r"\1"),
        (re.compile(r" done \([0-9]+(\.[0-9]+)?s\)"), " done"),
    ],
}
_VERSIONS = [
    (re.compile(r"[Cc]lang( version)? [0-9.]+"), "clang <ver>"),
    (re.compile(r"ROCm[ -]?[0-9.]+"), "ROCm <ver>"),
]


class Scrubber:
    """Line-by-line scrubber; keeps the last problem-progress header's column map."""

    def __init__(self, timing="none", versions=False):
        self._subs = list(_TIMING[timing]) + (list(_VERSIONS) if versions else [])
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timing", choices=sorted(_TIMING), default="none")
    parser.add_argument(
        "--versions",
        action="store_true",
        help="replace clang and ROCm version numbers with <ver>",
    )
    parser.add_argument("--log", help="also write the scrubbed lines here (no tag)")
    parser.add_argument("--tag", help="prefix stdout lines with [TAG]")
    args = parser.parse_args(argv)

    # Bytes that aren't UTF-8 pass through unchanged.
    stdin = open(
        sys.stdin.fileno(),
        encoding="utf-8",
        errors="surrogateescape",
        newline="",
        closefd=False,
    )
    stdout = open(
        sys.stdout.fileno(),
        "w",
        encoding="utf-8",
        errors="surrogateescape",
        newline="",
        closefd=False,
    )
    log = (
        open(args.log, "w", encoding="utf-8", errors="surrogateescape", newline="")
        if args.log
        else None
    )
    prefix = f"[{args.tag}] " if args.tag else ""
    scrubber = Scrubber(args.timing, args.versions)
    try:
        for raw in stdin:
            body = raw.rstrip("\n")
            out = scrubber.scrub(body)
            if out is None:
                continue
            if log:
                log.write(out + "\n")
                log.flush()
            stdout.write(prefix + out + "\n")
            stdout.flush()
    finally:
        if log:
            log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
