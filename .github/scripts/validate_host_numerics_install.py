#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validate the host-numerics install tree against its source and manifest."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


class ValidationError(RuntimeError):
    """An install invariant was not satisfied."""


def _files_below(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }


def _display_paths(paths: set[Path]) -> str:
    return "\n".join(f"  {path.as_posix()}" for path in sorted(paths))


def _manifest_entries(manifest: Path, install_prefix: Path) -> set[Path]:
    try:
        entries = [Path(line) for line in manifest.read_text().splitlines() if line]
    except OSError as error:
        raise ValidationError(
            f"could not read install manifest {manifest}: {error}"
        ) from error

    prefix = Path(os.path.abspath(install_prefix))
    relative_entries: set[Path] = set()
    outside_prefix: set[Path] = set()
    for entry in entries:
        absolute_entry = Path(os.path.abspath(entry))
        try:
            relative_entries.add(absolute_entry.relative_to(prefix))
        except ValueError:
            outside_prefix.add(absolute_entry)

    if outside_prefix:
        raise ValidationError(
            "install manifest contains paths outside the install prefix:\n"
            f"{_display_paths(outside_prefix)}"
        )
    return relative_entries


def _validate_install_tree(
    source_dir: Path, build_dir: Path, install_prefix: Path
) -> None:
    public_include_dir = source_dir / "include"
    installed_include_dir = install_prefix / "include"
    manifest = build_dir / "install_manifest.txt"

    if not public_include_dir.is_dir():
        raise ValidationError(
            f"public include directory is missing: {public_include_dir}"
        )
    if not installed_include_dir.is_dir():
        raise ValidationError(
            f"installed include directory is missing: {installed_include_dir}"
        )

    expected_public_files = _files_below(public_include_dir)
    if not expected_public_files:
        raise ValidationError(f"no public files found below {public_include_dir}")

    installed_public_files = _files_below(installed_include_dir)
    manifest_entries = _manifest_entries(manifest, install_prefix)
    manifest_public_files = {
        path.relative_to("include")
        for path in manifest_entries
        if path.is_relative_to("include")
    }

    missing_installed = expected_public_files - installed_public_files
    unexpected_installed = installed_public_files - expected_public_files
    missing_manifest = expected_public_files - manifest_public_files
    unexpected_manifest = manifest_public_files - expected_public_files

    errors: list[str] = []
    if missing_installed:
        errors.append(
            "public source files missing from the install tree:\n"
            f"{_display_paths(missing_installed)}"
        )
    if unexpected_installed:
        errors.append(
            "unexpected files installed under include/:\n"
            f"{_display_paths(unexpected_installed)}"
        )
    if missing_manifest:
        errors.append(
            "public source files missing from the install manifest:\n"
            f"{_display_paths(missing_manifest)}"
        )
    if unexpected_manifest:
        errors.append(
            "unexpected include files recorded in the install manifest:\n"
            f"{_display_paths(unexpected_manifest)}"
        )

    detail_dir = installed_include_dir / "roc" / "host_numerics" / "detail"
    if detail_dir.exists() or detail_dir.is_symlink():
        errors.append(f"private detail directory was installed: {detail_dir}")

    missing_outputs = {
        path
        for path in manifest_entries
        if not (install_prefix / path).exists()
        and not (install_prefix / path).is_symlink()
    }
    if missing_outputs:
        errors.append(
            "install manifest entries missing from the install tree:\n"
            f"{_display_paths(missing_outputs)}"
        )

    source_license = source_dir / "LICENSE.md"
    if source_license.is_file():
        installed_licenses = [
            install_prefix / path
            for path in manifest_entries
            if path.name == source_license.name
        ]
        if not any(
            path.is_file() and path.read_bytes() == source_license.read_bytes()
            for path in installed_licenses
        ):
            errors.append(
                f"{source_license.name} was not installed from the source tree"
            )

    if errors:
        raise ValidationError("\n".join(errors))

    print(
        f"Validated {len(expected_public_files)} public install files; "
        "no private detail directory is installed."
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--install-prefix", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    try:
        _validate_install_tree(
            arguments.source_dir, arguments.build_dir, arguments.install_prefix
        )
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
