#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check prebuilt C++ parity emitters against the differential golden.

The golden anchors the Python reference .ll of every parity family, one
sha256 per config, in the family's home file (run_diff.golden_file_for: the
library's `tests/parity/golden/llvm_gfx_all.json` for a library family, else
`golden/llvm_gfx_all.json` here), and rocke_differential_golden_ll
checks Python against it. This runner checks the C++ side against the same
hashes, so the two checks together hold C++ == Python on a host with no
compiler and no engine source: the emitters are built with the engine and
installed beside this file's test manifest.

Each emitter is given as FAMILY=PATH. Per family and flavor, every config
the golden blesses must emit exactly that .ll, and every config it does not
bless (the Python reference rejected it, or it is past the family's last
config) must be rejected by the C++ emitter too. A family with no golden, a
missing executable or an emitter that fails is an error, never a skip.

Emitter exit codes (the parity emitters' contract):
  0  emitted: stdout is the .ll; the config must be blessed and match it.
  1  refused: stdout is empty, the reason is on stderr; the config must not
     be blessed.
  2  "unknown config" on stderr, no stdout: past the last config, the end.
Any other exit code is a failure, whatever stdout holds: a crash is negative
on POSIX and a large positive code on Windows (0xC0000005), so it is never
read as a refusal.

A family must end ("unknown config") by config_limit(family): when every
index up to the cap answers, the check asks for one more, and a family that
answers it fails (raise FAMILY_MAX_CFG in run_diff.py), so a config past the
cap is never silently unchecked.

A family must also end where the Python reference ends. The golden holds no
entry for a refused config, so it cannot show a C++ table that drops trailing
refusals, or adds some: where C++ ends, the Python emitter is asked about that
index and the one before it, through run_diff.py's own Python runner.

What it cannot see: a refusal's message (the golden holds no messages;
check_byte_identity.py compares them where the engine builds).

Usage:
  check_native_golden.py --flavor llvm20,llvm22 FAMILY=PATH [FAMILY=PATH ...]
"""

from __future__ import annotations

import argparse
import functools
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from golden_store import LL, GoldenStore  # noqa: E402 -- this directory
from run_diff import (  # noqa: E402
    TIMEOUT,
    _is_end_emission,
    _run_python_batch,
    config_limit,
    find_families,
    golden_file_for,
    sh,
)


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


EMITTED, REFUSED, UNKNOWN = 0, 1, 2


@functools.lru_cache(maxsize=None)
def reference_is_end(family: str, indices: tuple[int, ...]) -> tuple[bool, ...]:
    """Whether the Python reference answers "unknown config" at each index.

    The end does not depend on the LLVM flavor, so one answer per family is
    cached. Raises RuntimeError when the reference cannot be run.
    """
    src_dir = dict(find_families(python_only=True)).get(family)
    if src_dir is None:
        raise RuntimeError(f"no Python reference {family}_emit.py")
    batch = _run_python_batch(family, indices, "ll", src_dir=src_dir)
    if batch.error:
        raise RuntimeError(batch.error.strip()[:200])
    return tuple(_is_end_emission(e) for e in batch.emissions)


def same_end_as_python(family: str, end: int) -> list[str]:
    """Problems when C++ ends at `end` and the Python reference does not."""
    indices = (end - 1, end) if end else (end,)
    try:
        ends = reference_is_end(family, indices)
    except RuntimeError as e:
        return [f"cannot ask the Python reference where the family ends: {e}"]
    problems = []
    if not ends[-1]:
        problems.append(
            f"C++ ends at config {end}; the Python reference has more configs "
            "(refusals included)"
        )
    if end and ends[0]:
        problems.append(
            f"C++ answers config {end - 1}, past the Python reference's last config"
        )
    return problems


def check_family(family: str, exe: Path, blessed: dict[str, str]) -> list[str]:
    """Problems for one family at the flavor already exported to the emitter."""
    problems = []
    seen = set()
    end = None
    limit = config_limit(family)
    for idx in range(limit):
        try:
            rc, out, err, had_cr = run_config(exe, idx)
        except subprocess.TimeoutExpired:
            problems.append(f"config {idx}: emitter timed out after {TIMEOUT} s")
            break
        if rc == UNKNOWN and not out and "unknown config" in err.lower():
            end = idx
            break
        key = str(idx)
        seen.add(key)
        want = blessed.get(key)
        if rc not in (EMITTED, REFUSED):
            problems.append(
                f"config {idx}: emitter failed (exit {rc}): {err.strip()[:200]}"
            )
            if rc < 0:
                break  # a crash: the next configs would only repeat it
            continue
        if want is None:
            if rc == EMITTED:
                problems.append(
                    f"config {idx}: C++ emits, the golden blesses no .ll for it"
                    if out
                    else f"config {idx}: C++ exits 0 with no output (a refusal exits 1)"
                )
            elif out:
                problems.append(f"config {idx}: C++ refuses (exit 1) but prints output")
        elif rc == REFUSED:
            problems.append(
                f"config {idx}: C++ rejects (exit {rc}): {err.strip()[:200]}"
            )
        elif sh(out) != want:
            problems.append(
                f"config {idx}: .ll differs from the golden "
                f"({len(out)} bytes, CR in output: {had_cr}, exit {rc})"
            )
    else:
        # every index up to the cap answered: the family must end here
        try:
            rc, out, err, _cr = run_config(exe, limit)
        except subprocess.TimeoutExpired:
            rc, out, err = None, b"", "timed out"
        if rc == UNKNOWN and not out and "unknown config" in err.lower():
            end = limit
        else:
            problems.append(
                f"config {limit} exists past the cap of {limit} configs "
                f"(exit {rc}); raise FAMILY_MAX_CFG['{family}'] in run_diff.py"
            )
    if end is not None:
        problems += same_end_as_python(family, end)
    missing = sorted(set(blessed) - seen, key=int)
    if missing:
        problems.append(f"C++ ends before blessed configs {missing[:10]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flavor", required=True, help="comma-separated LLVM flavors")
    ap.add_argument(
        "--golden-file",
        help="check every family against this file (default: each family's home)",
    )
    ap.add_argument("emitters", nargs="+", metavar="FAMILY=PATH")
    args = ap.parse_args()

    emitters = []
    for spec in args.emitters:
        family, sep, path = spec.partition("=")
        if not sep or not family or not path:
            ap.error(f"expected FAMILY=PATH, got {spec!r}")
        emitters.append((family, Path(path)))

    homes = dict(find_families(python_only=True))
    stores: dict[Path, GoldenStore] = {}

    def store_for(family: str) -> GoldenStore:
        if args.golden_file:
            path = Path(args.golden_file)
        else:
            src_dir = homes.get(family)
            path = golden_file_for(src_dir) if src_dir is not None else None
        if path is None:
            raise KeyError(family)
        if path not in stores:
            stores[path] = GoldenStore.load(path)
        return stores[path]

    failed = 0
    for flavor in [f.strip() for f in args.flavor.split(",") if f.strip()]:
        os.environ["ROCKE_LLVM_FLAVOR"] = flavor
        for family, exe in emitters:
            try:
                section = store_for(family).section(LL, flavor) or {}
            except KeyError:
                section = None
            if not exe.is_file():
                problems = [f"emitter not found: {exe}"]
            elif section is None:
                section = {}
                problems = [f"no Python reference {family}_emit.py, so no home golden"]
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
    print("\nRESULT: OK - every C++ emitter matches the golden")
    return 0


if __name__ == "__main__":
    sys.exit(main())
