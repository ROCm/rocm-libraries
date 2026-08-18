#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validate the exact boundary of a TensileLite release wheel."""

from __future__ import annotations

import argparse
import configparser
from email.parser import Parser
from pathlib import Path
import sys
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


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
_COMPATIBILITY_SCRIPTS = {
    "Tensile": "tensilelite_tensile_compat.commands:tensile",
    "TensileBenchmarkCluster": "tensilelite_tensile_compat.commands:benchmark_cluster",
    "TensileCreateLibrary": "tensilelite_tensile_compat.commands:create_library",
    "TensileGenerateSummations": "tensilelite_tensile_compat.commands:generate_summations",
    "TensileGetPath": "tensilelite_tensile_compat.commands:get_path",
    "TensileLibLogicToYaml": "tensilelite_tensile_compat.commands:logic_to_yaml",
    "TensileLogic": "tensilelite_tensile_compat.commands:logic",
    "TensileMergeLibrary": "tensilelite_tensile_compat.commands:merge_library",
    "TensileRetuneLibrary": "tensilelite_tensile_compat.commands:retune_library",
    "TensileUpdateLibrary": "tensilelite_tensile_compat.commands:update_library",
    "TensileVerifyStinkyElfText": "tensilelite_tensile_compat.commands:verify_stinky_elf",
}


def _archive(wheel: Path):
    archive = zipfile.ZipFile(wheel)
    names = set(archive.namelist())
    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
    entry_names = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
    if len(metadata_names) != 1 or len(wheel_names) != 1 or len(entry_names) != 1:
        archive.close()
        raise ValueError(
            "expected exactly one METADATA, WHEEL, and entry_points.txt file; "
            f"got {metadata_names}, {wheel_names}, {entry_names}"
        )
    metadata = Parser().parsestr(archive.read(metadata_names[0]).decode())
    wheel_metadata = Parser().parsestr(archive.read(wheel_names[0]).decode())
    entries = configparser.ConfigParser()
    entries.optionxform = str
    entries.read_string(archive.read(entry_names[0]).decode())
    return archive, names, metadata, wheel_metadata, dict(entries["console_scripts"])


def _forbidden(names: set[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if name.startswith(("Tensile/", "rocisa/", "tests/"))
        or "/Tests/" in name
        or "/__pycache__/" in name
        or "/bin/" in name
        or "/Utilities/archive/" in name
        or "/build/" in name
        or name.endswith(
            (".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib", "CMakeCache.txt")
        )
    )


def validate(wheel: Path, mode: str, source_root: Path, expected_version: str) -> list[str]:
    problems = []
    try:
        archive, names, metadata, wheel_metadata, scripts = _archive(wheel)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return [str(exc)]
    with archive:
        expected_name = "tensilelite" if mode == "canonical" else "tensilelite-tensile-compat"
        filename_name = "tensilelite" if mode == "canonical" else "tensilelite_tensile_compat"
        expected_filename = f"{filename_name}-{expected_version}-py3-none-any.whl"
        if wheel.name != expected_filename:
            problems.append(f"filename must be {expected_filename}, got {wheel.name}")
        if canonicalize_name(metadata["Name"]) != expected_name:
            problems.append(f"unexpected distribution name: {metadata['Name']}")
        if Version(metadata["Version"]) != Version(expected_version):
            problems.append(f"unexpected version: {metadata['Version']}")
        if metadata.get("Requires-Python") != ">=3.10":
            problems.append(f"Requires-Python must be >=3.10, got {metadata.get('Requires-Python')}")
        tags = wheel_metadata.get_all("Tag", [])
        if tags != ["py3-none-any"]:
            problems.append(f"WHEEL tags must be exactly py3-none-any, got {tags}")
        forbidden = _forbidden(names)
        if forbidden:
            problems.append("forbidden entries:\n  " + "\n  ".join(forbidden))
        bound_clients = sorted(name for name in names if name.endswith("client.json"))
        if bound_clients:
            problems.append("wheel must not contain client bindings: " + ", ".join(bound_clients))

        requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
        if mode == "canonical":
            expected_scripts = {
                "tensilelite": "tensilelite.cli:main",
                "tensilelite-configure-client": "tensilelite_configure_client:main",
            }
            if scripts != expected_scripts:
                problems.append(f"unexpected canonical console scripts: {scripts}")
            for module in ("_tensilelite_client_binding.py", "tensilelite_configure_client.py"):
                if module not in names:
                    problems.append(f"missing top-level module: {module}")
            expected_headers = {f"tensilelite/Source/{name}" for name in _HEADERS}
            if expected_headers - names:
                problems.append(f"missing headers: {sorted(expected_headers - names)}")
            for resource in (
                "tensilelite/tensilelite_logic/known_bugs.yaml",
                "tensilelite/ductile/config/defaults.yaml",
            ):
                if resource not in names:
                    problems.append(f"missing resource: {resource}")
            source_kernels = {
                f"tensilelite/CustomKernels/{path.name}"
                for path in (source_root / "tensilelite/CustomKernels").glob("*.s")
            }
            wheel_kernels = {
                name for name in names if name.startswith("tensilelite/CustomKernels/")
            }
            if source_kernels != wheel_kernels:
                problems.append("custom-kernel resource set does not match the source tree")
            rocisa = [req for req in requirements if canonicalize_name(req.name) == "rocisa"]
            if len(rocisa) != 1 or str(rocisa[0]) != "rocisa":
                problems.append(f"canonical Requires-Dist must contain exact rocisa, got {rocisa}")
            extras = set(metadata.get_all("Provides-Extra", []))
            for extra, dependency in _OPTIONAL_REQUIREMENTS.items():
                matches = [req for req in requirements if canonicalize_name(req.name) == dependency]
                if extra not in extras or not any(
                    req.marker and req.marker.evaluate({"extra": extra}) for req in matches
                ):
                    problems.append(f"missing optional dependency {dependency} for extra {extra}")
        else:
            if scripts != _COMPATIBILITY_SCRIPTS:
                problems.append(f"unexpected compatibility console scripts: {scripts}")
            canonical = [
                req for req in requirements if canonicalize_name(req.name) == "tensilelite"
            ]
            expected_pin = f"tensilelite=={expected_version}"
            if len(canonical) != 1 or str(canonical[0]) != expected_pin:
                problems.append(f"compatibility dependency must be {expected_pin}, got {canonical}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("canonical", "compatibility"))
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    problems = validate(args.wheel, args.mode, args.source_root, args.expected_version)
    if problems:
        print("Invalid TensileLite release wheel:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    print(f"Valid {args.mode} TensileLite release wheel: {args.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
