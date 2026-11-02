#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check prebuilt C++ parity emitters against the differential golden.

The golden (`golden/llvm_gfx_all.json`) anchors the Python reference .ll of
every parity family, one sha256 per config, and rocke_differential_golden_ll
checks Python against it. This runner checks the C++ side against the same
hashes, so the two checks together hold C++ == Python on a host with no
compiler and no engine source: the emitters are built with the engine and
installed beside this file's test manifest.

Each emitter is given as FAMILY=PATH. Per family and flavor, every config
the golden blesses must emit exactly that .ll, and every config it does not
bless (the Python reference rejected it, or it is past the family's last
config) must be rejected by the C++ emitter too. A family with no golden, a
missing executable or an emitter that fails is an error, never a skip.

Usage:
  check_native_golden.py --flavor llvm20,llvm22 FAMILY=PATH [FAMILY=PATH ...]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from golden_store import LL, GoldenStore  # noqa: E402 -- this directory
from run_diff import GOLDEN_FILE, TIMEOUT, config_limit, sh  # noqa: E402


def run_config(exe: Path, idx: int) -> tuple[int, bytes, str, bool]:
    """(returncode, stdout, stderr, stdout had CR) of one emitter config.

    A C++ emitter's stdout is in text mode on Windows, which writes every "\n"
    as "\r\n"; the golden hashes the Python reference's "\n" text, so CRLF is
    folded back before hashing there.
    """
    p = subprocess.run([str(exe), str(idx)], capture_output=True, timeout=TIMEOUT)
    out = p.stdout
    had_cr = b"\r" in out
    if os.name == "nt":
        out = out.replace(b"\r\n", b"\n")
    return p.returncode, out, p.stderr.decode("utf-8", "replace"), had_cr


def check_family(family: str, exe: Path, blessed: dict[str, str]) -> list[str]:
    """Problems for one family at the flavor already exported to the emitter."""
    problems = []
    seen = set()
    for idx in range(config_limit(family)):
        try:
            rc, out, err, had_cr = run_config(exe, idx)
        except subprocess.TimeoutExpired:
            problems.append(f"config {idx}: emitter timed out after {TIMEOUT} s")
            break
        if rc != 0 and not out and "unknown config" in err.lower():
            break
        key = str(idx)
        seen.add(key)
        want = blessed.get(key)
        if rc < 0:
            problems.append(
                f"config {idx}: emitter crashed (exit {rc}): {err.strip()[:200]}"
            )
            break
        if want is None and out:
            problems.append(f"config {idx}: C++ emits, the Python reference rejects it")
        elif want is not None and not out:
            problems.append(
                f"config {idx}: C++ rejects (exit {rc}): {err.strip()[:200]}"
            )
        elif want is not None and sh(out) != want:
            problems.append(
                f"config {idx}: .ll differs from the Python reference "
                f"({len(out)} bytes, CR in output: {had_cr}, exit {rc})"
            )
    missing = sorted(set(blessed) - seen, key=int)
    if missing:
        problems.append(f"C++ ends before blessed configs {missing[:10]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flavor", required=True, help="comma-separated LLVM flavors")
    ap.add_argument("--golden-file", default=str(GOLDEN_FILE))
    ap.add_argument("emitters", nargs="+", metavar="FAMILY=PATH")
    args = ap.parse_args()

    emitters = []
    for spec in args.emitters:
        family, sep, path = spec.partition("=")
        if not sep or not family or not path:
            ap.error(f"expected FAMILY=PATH, got {spec!r}")
        emitters.append((family, Path(path)))

    store = GoldenStore.load(Path(args.golden_file))
    failed = 0
    for flavor in [f.strip() for f in args.flavor.split(",") if f.strip()]:
        os.environ["ROCKE_LLVM_FLAVOR"] = flavor
        section = store.section(LL, flavor) or {}
        for family, exe in emitters:
            if not exe.is_file():
                problems = [f"emitter not found: {exe}"]
            elif family not in section:
                problems = [f"no golden for {LL}[{flavor}]; bless it with run_diff.py"]
            else:
                problems = check_family(family, exe, section[family])
            n = len(section.get(family, {}))
            if problems:
                failed += 1
                print(f"FAIL  {flavor}  {family}  ({n} blessed configs)")
                for problem in problems[:20]:
                    print(f"        {problem}")
            else:
                print(f"OK    {flavor}  {family}  ({n} blessed configs)")
    if failed:
        print(f"\nRESULT: FAIL - {failed} family/flavor checks differ from the golden")
        return 1
    print("\nRESULT: OK - every C++ emitter matches the Python golden")
    return 0


if __name__ == "__main__":
    sys.exit(main())
