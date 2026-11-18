#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validate the exact boundary of a TensileLite release wheel."""

from __future__ import annotations

import argparse
import ast
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
_CANONICAL_REQUIREMENTS = {
    ("filelock", ""),
    ("joblib", ">=1.4.0"),
    ("msgpack", ""),
    ("numpy", ""),
    ("packaging", ""),
    ("pandas", ""),
    ("pyyaml", ""),
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


_EXPECTED_SCRIPTS = {
    "tensilelite": "tensilelite.cli:main",
    "tensilelite-configure-client": "tensilelite_configure_client:main",
}
_BUILD_INFO = "tensilelite/_build_info.py"


def _stamped_revision(archive: zipfile.ZipFile, names: set[str]):
    """Return (found, revision) for the wheel's tensilelite/_build_info.py."""
    if _BUILD_INFO not in names:
        return False, None
    tree = ast.parse(archive.read(_BUILD_INFO).decode())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SOURCE_REVISION"
        ):
            return True, ast.literal_eval(node.value)
    return True, ...


def validate(
    wheel: Path,
    source_root: Path,
    expected_version: str,
    expected_source_revision: str | None = None,
) -> list[str]:
    """Return the problems with a tensilelite release wheel; empty when valid.

    ``expected_source_revision`` is the commit the wheel must carry in its
    ``_build_info.py``; ``"none"`` requires the wheel to record no commit, and
    ``None`` skips the check.
    """
    problems = []
    try:
        archive, names, metadata, wheel_metadata, scripts = _archive(wheel)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return [str(exc)]
    with archive:
        expected_name = "tensilelite"
        expected_filename = f"tensilelite-{expected_version}-py3-none-any.whl"
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

        found, stamped = _stamped_revision(archive, names)
        if not found:
            problems.append(f"missing {_BUILD_INFO}")
        elif stamped is ...:
            problems.append(f"{_BUILD_INFO} does not define SOURCE_REVISION")
        elif expected_source_revision is not None:
            expected = None if expected_source_revision == "none" else expected_source_revision
            if stamped != expected:
                problems.append(
                    f"source revision must be {expected_source_revision}, "
                    f"got {stamped or 'none'}"
                )

        requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
        # The legacy Tensile* command names are gone; the tensilelite command
        # family is the only command surface.
        if scripts != _EXPECTED_SCRIPTS:
            problems.append(f"unexpected console scripts: {scripts}")
        for module in ("_tensilelite_client_binding.py", "tensilelite_configure_client.py"):
            if module not in names:
                problems.append(f"missing top-level module: {module}")
        expected_headers = {f"tensilelite/Source/{name}" for name in _HEADERS}
        if expected_headers - names:
            problems.append(f"missing headers: {sorted(expected_headers - names)}")
        for resource in (
            "tensilelite/logic/known_bugs.yaml",
            "tensilelite/ductile/config/defaults.yaml",
        ):
            if resource not in names:
                problems.append(f"missing resource: {resource}")
        custom_kernel_root = source_root / "tensilelite/CustomKernels"
        source_kernels = {
            "tensilelite/CustomKernels/"
            + path.relative_to(custom_kernel_root).as_posix()
            for path in custom_kernel_root.rglob("*.s")
        }
        wheel_kernels = {
            name for name in names if name.startswith("tensilelite/CustomKernels/")
        }
        if source_kernels != wheel_kernels:
            problems.append("custom-kernel resource set does not match the source tree")
        rocisa = [req for req in requirements if canonicalize_name(req.name) == "rocisa"]
        if rocisa:
            problems.append(
                f"wheel must not declare source-provisioned rocisa, got {rocisa}"
            )
        mandatory = {
            (canonicalize_name(requirement.name), str(requirement.specifier))
            for requirement in requirements
            if requirement.marker is None
        }
        if mandatory != _CANONICAL_REQUIREMENTS:
            problems.append(
                "runtime dependencies must be exactly "
                f"{sorted(_CANONICAL_REQUIREMENTS)}, got {sorted(mandatory)}"
            )
        extras = set(metadata.get_all("Provides-Extra", []))
        if extras != set(_OPTIONAL_REQUIREMENTS):
            problems.append(
                "extras must be exactly "
                f"{sorted(_OPTIONAL_REQUIREMENTS)}, got {sorted(extras)}"
            )
        marked_requirements = {
            canonicalize_name(requirement.name)
            for requirement in requirements
            if requirement.marker is not None
        }
        if marked_requirements != set(_OPTIONAL_REQUIREMENTS.values()):
            problems.append(
                "optional dependencies must be exactly "
                f"{sorted(_OPTIONAL_REQUIREMENTS.values())}, "
                f"got {sorted(marked_requirements)}"
            )
        for extra, dependency in _OPTIONAL_REQUIREMENTS.items():
            matches = [req for req in requirements if canonicalize_name(req.name) == dependency]
            if extra not in extras or not any(
                req.marker and req.marker.evaluate({"extra": extra}) for req in matches
            ):
                problems.append(f"missing optional dependency {dependency} for extra {extra}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument(
        "--expected-source-revision",
        help="commit the wheel's _build_info.py must record, or 'none'",
    )
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    problems = validate(
        args.wheel, args.source_root, args.expected_version, args.expected_source_revision
    )
    if problems:
        print("Invalid TensileLite release wheel:\n" + "\n".join(problems), file=sys.stderr)
        return 1
    print(f"Valid TensileLite release wheel: {args.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
