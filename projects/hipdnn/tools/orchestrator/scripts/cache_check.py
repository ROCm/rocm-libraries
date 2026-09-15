#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Prove the build the flow is about to test was configured with the feature on.

`HIPDNN_ENABLE_KERNEL_INGESTOR=OFF` compiles the whole descriptor-driven kernel
ingestor engine out of the binary: `discoverDescriptorSets()` is never called,
`ingestorPacks()` is compiled to an empty table, the engine simply does not
exist at runtime. A `--test-engine` run against an engine that was configured
out and a run against an engine whose matcher declined every candidate graph
are indistinguishable at the process exit code -- both report "no match" and
both exit however the harness exits on "no match". Nothing downstream of that
run can tell the difference between "the kernel is wrong" and "the kernel was
never built", so by the time a benchmark or a census gate reports a number,
that number may describe a plugin that has nothing in it.

So the flags are read directly out of the configured `CMakeCache.txt` and
asserted before an agent is allowed to launch against that build, and again
after any step that reconfigures it. A missing cache means the directory was
never configured at all; that is reported the same way a wrong value is --
`satisfied: 0` -- rather than raised as an error, because both are equally
"do not proceed", and the flow's condition step reads the same field either
way.

    cache_check.py --build-dir DIR --out report.json \\
        --require HIPDNN_ENABLE_KERNEL_INGESTOR=ON --forbid SOME_STALE_FLAG

CMake booleans are compared the way CMake compares them: `ON`/`1`/`TRUE`/`YES`
are one value, `OFF`/`0`/`FALSE`/`NO`/empty are another, case-insensitively.
Anything else -- a path, a target list -- is compared as an exact string, so a
`--require CMAKE_BUILD_TYPE=Release` behaves the way it reads.

Exit codes: 0 the report was written (an unsatisfied gate is data for the flow
to assert on, not a script failure), 1 the build directory or cache file could
not be read, 2 a `--require`/`--forbid` argument is malformed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json

#: CMake's own boolean vocabulary (`cmake --help-policy CMP0012`-adjacent):
#: these string forms are interchangeable, case-insensitively, for both a
#: cache entry's actual value and a `--require`/`--forbid` VALUE.
_TRUTHY = {"on", "1", "true", "yes"}
_FALSEY = {"off", "0", "false", "no", ""}


def _cmake_equal(expected: str, actual: str | None) -> bool:
    """Compare the way CMake's own `if()` would: booleans against booleans,
    regardless of spelling; anything else, verbatim. `actual is None` (the
    variable is simply absent from the cache) is treated as the empty/falsey
    value, which is what an unset CMake boolean behaves as."""
    e = expected.strip().lower()
    a = (actual or "").strip().lower()
    if e in _TRUTHY and a in _TRUTHY:
        return True
    if e in _FALSEY and a in _FALSEY:
        return True
    return e == a


def _is_falsey(value: str | None) -> bool:
    return (value or "").strip().lower() in _FALSEY


def _parse_cache(path: Path) -> dict[str, str]:
    """`NAME:TYPE=VALUE` -> {NAME: VALUE}, skipping `#`/`//` comments and blank
    lines. NAME may itself contain `-`/`_`, so split on the FIRST `:`; VALUE may
    itself contain `=` (a path, a semicolon list with an `=` in it), so split
    the TYPE=VALUE remainder on its FIRST `=` only."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        name, sep, rest = line.partition(":")
        if not sep or "=" not in rest:
            continue
        _type, _eq, value = rest.partition("=")
        values[name] = value
    return values


def _split_require(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"--require expects VAR=VALUE, got {item!r}")
    var, _eq, value = item.partition("=")
    if not var:
        raise ValueError(f"--require expects VAR=VALUE, got {item!r}")
    return var, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir", required=True, help="configured CMake build directory"
    )
    parser.add_argument("--out", required=True, help="where to write the JSON report")
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="VAR=VALUE",
        help="cache entry that must equal VALUE (CMake-boolean-aware); repeatable",
    )
    parser.add_argument(
        "--forbid",
        action="append",
        default=[],
        metavar="VAR",
        help="cache entry that must be absent or falsey; repeatable",
    )
    args = parser.parse_args()

    try:
        requires = [_split_require(item) for item in args.require]
    except ValueError as error:
        parser.error(str(error))
        return 2  # unreachable; parser.error() exits, but satisfies type checkers

    build_dir = Path(args.build_dir)
    out_path = Path(args.out)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"error: could not create --out directory: {error}", file=sys.stderr)
        return 1

    cache_file = build_dir / "CMakeCache.txt"
    if not cache_file.is_file():
        report = {
            "build_dir": build_dir.as_posix(),
            "cache_present": 0,
            "checked": 0,
            "satisfied": 0,
            "mismatches": 0,
            "values": {var: None for var, _ in requires}
            | {var: None for var in args.forbid},
            "install_prefix": None,
            "generator": None,
            "gpu_targets": None,
            "feedback": (
                f"{build_dir.as_posix()} has never been configured: no CMakeCache.txt "
                "there. Nothing downstream of this gate can be trusted -- there is no "
                "build to test an engine against."
            ),
        }
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"cache absent at {cache_file.as_posix()}; satisfied=0 -> {out_path}")
        return 0

    try:
        cache = _parse_cache(cache_file)
    except OSError as error:
        print(f"error: could not read {cache_file}: {error}", file=sys.stderr)
        return 1

    mismatch_lines: list[str] = []
    values: dict[str, str | None] = {}
    for var, want in requires:
        actual = cache.get(var)
        values[var] = actual
        if not _cmake_equal(want, actual):
            mismatch_lines.append(f"{var}: want {want}, cache has {actual!r}")
    for var in args.forbid:
        actual = cache.get(var)
        values[var] = actual
        if not _is_falsey(actual):
            mismatch_lines.append(f"{var}: want unset, cache has {actual!r}")

    checked = len(requires) + len(args.forbid)
    mismatches = len(mismatch_lines)

    feedback = ""
    if mismatches:
        feedback = (
            "\n".join(mismatch_lines) + "\n\n"
            "The affected targets are compiled out of this build, not failing at "
            "runtime -- a `--test-engine` run against them and a run against an "
            "engine whose matcher declined everything look identical from the exit "
            "code alone. `hipdnn_validate_descriptors` reporting nothing is the "
            "visible symptom of HIPDNN_ENABLE_KERNEL_INGESTOR=OFF; reconfigure the "
            "build with the required cache values before launching an agent against it."
        )

    report = {
        "build_dir": build_dir.as_posix(),
        "cache_present": 1,
        "checked": checked,
        "satisfied": int(mismatches == 0),
        "mismatches": mismatches,
        "values": values,
        "install_prefix": cache.get("CMAKE_INSTALL_PREFIX"),
        "generator": cache.get("CMAKE_GENERATOR"),
        "gpu_targets": cache.get("GPU_TARGETS") or cache.get("AMDGPU_TARGETS"),
        "feedback": feedback,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{cache_file.as_posix()}: checked {checked}, {mismatches} mismatch(es), "
        f"satisfied={report['satisfied']} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
