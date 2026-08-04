#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fail when a TensileLite runtime wheel violates its explicit boundary."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import sys
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


_HEADERS = {
    "KernelHeader.h",
    "ReductionTemplate.h",
    "TensileTypes.h",
    "memory_gfx.h",
    "tensile_bfloat16.h",
    "tensile_float8_bfloat8.h",
}

_OPTIONAL_REQUIREMENTS = {
    "hip-query": "hip-python",
    "orjson": "orjson",
    "profile": "yappi",
    "simplejson": "simplejson",
    "ujson": "ujson",
}


def _wheel_names(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def _wheel_metadata(wheel: Path):
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError(f"expected one METADATA file, found {metadata_files}")
        return Parser().parsestr(archive.read(metadata_files[0]).decode("utf-8"))


def _forbidden_entries(names: set[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if name.startswith("Tensile/")
        or name.startswith("rocisa/")
        or name.startswith("tests/")
        or "/Tests/" in name
        or "/__pycache__/" in name
        or "/bin/" in name
        or "/Utilities/archive/" in name
        or name.endswith(
            (
                ".pyc",
                ".pyo",
                ".so",
                ".pyd",
                ".dll",
                ".dylib",
                "CMakeCache.txt",
                "install_manifest.txt",
            )
        )
        or "/build/" in name
    )


def errors(wheel: Path, source_root: Path) -> list[str]:
    names = _wheel_names(wheel)
    problems = []
    forbidden = _forbidden_entries(names)
    if forbidden:
        problems.append("forbidden entries:\n  " + "\n  ".join(forbidden))

    expected_headers = {f"tensilelite/Source/{name}" for name in _HEADERS}
    missing_headers = sorted(expected_headers - names)
    if missing_headers:
        problems.append("missing headers: " + ", ".join(missing_headers))

    source_kernels = {
        f"tensilelite/CustomKernels/{path.name}"
        for path in (source_root / "tensilelite" / "CustomKernels").glob("*.s")
    }
    wheel_kernels = {
        name
        for name in names
        if name.startswith("tensilelite/CustomKernels/") and name.endswith(".s")
    }
    if source_kernels != wheel_kernels:
        problems.append(
            "custom-kernel set mismatch: "
            f"missing={sorted(source_kernels - wheel_kernels)}, "
            f"unexpected={sorted(wheel_kernels - source_kernels)}"
        )

    for required in (
        "tensilelite/tensilelite_logic/known_bugs.yaml",
        "tensilelite/ductile/config/defaults.yaml",
    ):
        if required not in names:
            problems.append(f"missing resource: {required}")

    metadata = _wheel_metadata(wheel)
    provided_extras = set(metadata.get_all("Provides-Extra", []))
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    for extra, dependency in _OPTIONAL_REQUIREMENTS.items():
        if extra not in provided_extras:
            problems.append(f"missing optional extra: {extra}")
        matching = [
            requirement
            for requirement in requirements
            if canonicalize_name(requirement.name) == canonicalize_name(dependency)
        ]
        if not any(
            requirement.marker and requirement.marker.evaluate({"extra": extra})
            for requirement in matching
        ):
            problems.append(
                f"missing conditional requirement: {dependency} for extra {extra}"
            )
    return problems


def compatibility_errors(wheel: Path) -> list[str]:
    forbidden = _forbidden_entries(_wheel_names(wheel))
    return ["forbidden entries:\n  " + "\n  ".join(forbidden)] if forbidden else []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    args = parser.parse_args(argv)
    wheel_input = args.wheel
    compatibility_wheels = []
    if wheel_input.is_dir():
        matches = sorted(
            path
            for path in wheel_input.glob("tensilelite-*.whl")
            if "tensile_compat" not in path.name
        )
        if len(matches) != 1:
            print(
                f"Expected one canonical TensileLite wheel in {wheel_input}, found {matches}",
                file=sys.stderr,
            )
            return 1
        wheel = matches[0]
        compatibility_wheels = sorted(wheel_input.glob("tensilelite_tensile_compat-*.whl"))
        if len(compatibility_wheels) > 1:
            print(
                f"Expected at most one compatibility wheel in {wheel_input}, "
                f"found {compatibility_wheels}",
                file=sys.stderr,
            )
            return 1
    else:
        wheel = wheel_input
    problems = errors(wheel, args.source_root)
    for compatibility_wheel in compatibility_wheels:
        problems.extend(
            f"{compatibility_wheel}: {problem}"
            for problem in compatibility_errors(compatibility_wheel)
        )
    if problems:
        print(f"Invalid TensileLite wheels:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    checked = [wheel, *compatibility_wheels]
    print("TensileLite wheel contents are valid: " + ", ".join(map(str, checked)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
