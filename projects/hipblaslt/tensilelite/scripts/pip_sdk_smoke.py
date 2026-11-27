#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only smoke check: TensileLite with TheRock's pip-installed ROCm SDK.

Run in a Python environment that has ``rocm-sdk-core`` and TensileLite from
this checkout installed (``tox -e pip-sdk-smoke``). It checks that importing
TensileLite selects the Python SDK as its ROCm, identified by
``rocm_sdk_core.__version__``, and that the compiler tools TensileLite needs
resolve through this interpreter's scripts directory. No GPU, rocisa or client
is needed: TensileLite built from a checkout is a +g<commit> development build,
which the ROCm check only warns about, and no client is requested.

The results, including the installed rocm-sdk-core version, are printed and
appended to ``$GITHUB_STEP_SUMMARY`` when it is set.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import warnings

# Tools TensileLite runs from the selected ROCm when generating code objects.
_REQUIRED_TOOLS = ("clang-offload-bundler",)


def _scripts_dirs() -> set[Path]:
    schemes = (None, sysconfig.get_preferred_scheme("user"))
    dirs = set()
    for scheme in schemes:
        path = sysconfig.get_path("scripts", scheme) if scheme else sysconfig.get_path("scripts")
        if path:
            dirs.add(Path(path).resolve())
    return dirs


def run() -> tuple[list[str], list[str]]:
    """Return (report lines, failures)."""
    report: list[str] = []
    failures: list[str] = []

    sdk_version = metadata.version("rocm-sdk-core")
    report.append(f"rocm-sdk-core: {sdk_version}")
    import rocm_sdk_core

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import tensilelite
    from tensilelite import _rocm, _runtime

    report.append(f"tensilelite: {tensilelite.__version__} ({Path(tensilelite.__file__).parent})")
    for warning in caught:
        report.append(f"warning: {str(warning.message).splitlines()[0]}")

    rocm = _runtime._installation
    report.append(
        f"selected ROCm: {type(rocm).__name__} {rocm.version} at {rocm.path} ({rocm.source})"
    )
    if not isinstance(rocm, _rocm.PythonRocm):
        failures.append(f"expected PythonRocm to be selected, got {type(rocm).__name__}")
    expected = _rocm._canonical_rocm_version(rocm_sdk_core.__version__)
    if rocm.version != expected:
        failures.append(f"expected ROCm identity {expected}, got {rocm.version}")

    search_paths = _runtime.executable_search_paths()
    scripts = _scripts_dirs()
    for tool in _REQUIRED_TOOLS:
        found = shutil.which(tool, path=os.pathsep.join(str(path) for path in search_paths))
        report.append(f"{tool}: {found or 'not found'}")
        if found is None:
            failures.append(
                f"{tool} was not found in the selected ROCm's executable paths: "
                + ", ".join(str(path) for path in search_paths)
            )
        elif Path(found).parent.resolve() not in scripts:
            failures.append(f"{tool} resolved outside this interpreter's scripts directory: {found}")
    return report, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--summary",
        type=Path,
        default=os.environ.get("GITHUB_STEP_SUMMARY") or None,
        help="Markdown file to append the results to (default: $GITHUB_STEP_SUMMARY)",
    )
    args = parser.parse_args(argv)

    try:
        report, failures = run()
    except Exception as exc:  # report import-time failures the same way
        report, failures = [], [f"{type(exc).__name__}: {exc}"]

    status = "FAILED" if failures else "passed"
    lines = [f"TensileLite pip SDK smoke check {status}", *report]
    lines += [f"FAIL: {failure}" for failure in failures]
    print("\n".join(lines))
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as stream:
            stream.write(f"### TensileLite pip SDK smoke check: {status}\n\n")
            stream.writelines(f"- {line}\n" for line in lines[1:])
            stream.write("\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
