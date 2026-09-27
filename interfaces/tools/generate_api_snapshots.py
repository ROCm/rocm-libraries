#!/usr/bin/env python3
"""Generate deterministic public API snapshots with rocm-api-extract."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    name: str
    header: Path
    language: str
    arguments: tuple[str, ...]


def load_profiles(path: Path, include_root: Path) -> tuple[list[Profile], list[str]]:
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: root must be an object")
    raw_profiles = document.get("profiles")
    raw_definitions = document.get("common_definitions", [])
    if not isinstance(raw_profiles, dict) or not isinstance(raw_definitions, list):
        raise ValueError(f"{path}: malformed profiles or common_definitions")
    profiles: list[Profile] = []
    for name, raw_profile in sorted(raw_profiles.items()):
        if not isinstance(name, str) or not isinstance(raw_profile, dict):
            raise ValueError(f"{path}: malformed profile")
        header = raw_profile.get("header")
        language = raw_profile.get("language")
        raw_arguments = raw_profile.get("arguments", [])
        if not isinstance(header, str) or language not in {"c", "c++"}:
            raise ValueError(f"{path}: malformed profile {name}")
        if not isinstance(raw_arguments, list) or not all(
            isinstance(item, str) for item in raw_arguments
        ):
            raise ValueError(f"{path}: malformed arguments for profile {name}")
        profiles.append(
            Profile(
                name, (include_root / header).resolve(), language, tuple(raw_arguments)
            )
        )
    if not all(isinstance(item, str) for item in raw_definitions):
        raise ValueError(f"{path}: definitions must be strings")
    return profiles, raw_definitions


def generate(
    extractor: Path,
    profile: Profile,
    include_root: Path,
    resource_dir: Path,
    definitions: list[str],
    extra_includes: list[Path],
    output: Path,
) -> None:
    if not profile.header.is_file():
        raise FileNotFoundError(
            f"{profile.name}: header does not exist: {profile.header}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(extractor),
        "--output",
        str(output),
        "--header-root",
        str(profile.header.parent),
        str(profile.header),
        "--",
        f"-x{profile.language}",
        "-std=c++17" if profile.language == "c++" else "-std=c11",
        f"-resource-dir={resource_dir}",
        f"-I{include_root}",
    ]
    command.extend(f"-I{path.resolve()}" for path in extra_includes)
    command.extend(f"-D{definition}" for definition in definitions)
    command.extend(profile.arguments)
    subprocess.run(command, check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"{profile.name}: extractor produced no output")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--include-root", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-include", type=Path, action="append", default=[])
    parser.add_argument("--profile", action="append", dest="selected")
    args = parser.parse_args()

    extractor = args.extractor.resolve()
    include_root = args.include_root.resolve()
    resource_dir = args.resource_dir.resolve()
    if not extractor.is_file():
        raise FileNotFoundError(f"extractor does not exist: {extractor}")
    if not resource_dir.is_dir():
        raise FileNotFoundError(
            f"Clang resource directory does not exist: {resource_dir}"
        )
    profiles, definitions = load_profiles(args.profiles, include_root)
    selected = set(args.selected or [])
    unknown = selected.difference(profile.name for profile in profiles)
    if unknown:
        raise ValueError(f"unknown profiles: {', '.join(sorted(unknown))}")
    for profile in profiles:
        if selected and profile.name not in selected:
            continue
        generate(
            extractor,
            profile,
            include_root,
            resource_dir,
            definitions,
            args.extra_include,
            args.output_dir / f"{profile.name}.json",
        )


if __name__ == "__main__":
    main()
