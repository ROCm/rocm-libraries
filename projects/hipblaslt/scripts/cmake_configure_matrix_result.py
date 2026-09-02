#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Capture a configured CMake build tree for the local configure matrix.

The configure-matrix shell runner deliberately owns topology setup.  This
helper only snapshots the resulting build tree and updates one JSON document,
so it is also useful when a cell fails after CMake has written a partial cache.
It uses only the Python standard library because it runs in published ROCm
container images as well as developer environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
MATRIX_KIND = "rocm-cmake-configure-matrix"
INSTALL_INCLUDE_RE = re.compile(
    r"^\s*include\s*\(\s*[\"']?([^\s\)\"']*cmake_install\.cmake)[\"']?",
    re.IGNORECASE | re.MULTILINE,
)
NINJA_RULE_RE = re.compile(
    r"^rule (?P<name>[^\n]+)\n(?P<body>(?:  [^\n]*\n)+)", re.MULTILINE
)


def utc_now() -> str:
    """Return an ISO-8601 timestamp that is stable across Python versions."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def canonical_path(path: Path) -> Path:
    """Make paths in the artifact unambiguous without requiring they exist."""

    return path.expanduser().resolve(strict=False)


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    """Describe a file without making a missing or unreadable file fatal."""

    path = canonical_path(path)
    record: dict[str, Any] = {
        "path": str(path),
        "relative_path": relative_path(path, root),
        "available": path.is_file(),
    }
    if not path.is_file():
        return record

    try:
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = file_sha256(path)
    except OSError as error:
        record["read_error"] = str(error)
    return record


def read_text(path: Path) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except OSError as error:
        return None, str(error)


def parse_cmake_cache(cache_path: Path, build_dir: Path) -> dict[str, Any]:
    record = file_record(cache_path, build_dir)
    record["entries"] = {}
    if not record["available"]:
        return record

    contents, error = read_text(cache_path)
    if error is not None or contents is None:
        record["read_error"] = error
        return record

    entries: dict[str, dict[str, str]] = {}
    for line in contents.splitlines():
        if not line or line.startswith("#") or line.startswith("//") or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        if ":" not in key_and_type:
            continue
        name, entry_type = key_and_type.rsplit(":", 1)
        if name:
            entries[name] = {"type": entry_type, "value": value}
    record["entries"] = entries
    return record


def collect_compile_commands(build_dir: Path) -> dict[str, Any]:
    compile_commands = build_dir / "compile_commands.json"
    record = file_record(compile_commands, build_dir)
    record["count"] = 0
    record["entries"] = []
    if not record["available"]:
        return record

    contents, error = read_text(compile_commands)
    if error is not None or contents is None:
        record["read_error"] = error
        return record

    try:
        entries = json.loads(contents)
    except json.JSONDecodeError as error:
        record["parse_error"] = str(error)
        return record
    if not isinstance(entries, list):
        record["parse_error"] = "compile_commands.json did not contain a JSON array"
        return record

    record["count"] = len(entries)
    record["entries"] = entries
    return record


def collect_log(log_path: Path, build_dir: Path) -> dict[str, Any]:
    """Keep a bounded, human-readable tail alongside the immutable log file."""

    record = file_record(log_path, build_dir)
    if not record["available"]:
        return record
    contents, error = read_text(log_path)
    if error is not None or contents is None:
        record["read_error"] = error
        return record
    record["tail"] = contents.splitlines()[-80:]
    return record


def sorted_files(build_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted(
        (path for path in build_dir.rglob(pattern) if path.is_file()),
        key=lambda path: str(path),
    )


def collect_link_commands(build_dir: Path) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for link_path in sorted_files(build_dir, "link.txt"):
        record = file_record(link_path, build_dir)
        contents, error = read_text(link_path)
        if error is not None:
            record["read_error"] = error
        elif contents is not None:
            record["command"] = contents.strip()
        commands.append(record)
    return commands


def collect_targets(build_dir: Path) -> list[dict[str, str]]:
    """Collect CMake's generated target-directory indexes when present."""

    targets: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index_path in sorted_files(build_dir, "TargetDirectories.txt"):
        contents, error = read_text(index_path)
        if error is not None or contents is None:
            continue
        for directory in contents.splitlines():
            directory = directory.strip()
            if not directory:
                continue
            key = (str(index_path), directory)
            if key in seen:
                continue
            seen.add(key)
            name = Path(directory).name
            targets.append(
                {
                    "index": relative_path(canonical_path(index_path), build_dir),
                    "directory": directory,
                    "name": name[:-4] if name.endswith(".dir") else name,
                }
            )
    return targets


def collect_package_configs_and_exports(build_dir: Path) -> list[dict[str, Any]]:
    """Collect generated package config and export files, excluding CPack."""

    paths: set[Path] = set()
    for pattern in ("*Config.cmake", "*-config.cmake", "*Targets.cmake", "*-targets.cmake"):
        paths.update(sorted_files(build_dir, pattern))
    paths = {
        path
        for path in paths
        if path.name not in {"CPackConfig.cmake", "CPackSourceConfig.cmake"}
    }
    return [file_record(path, build_dir) for path in sorted(paths, key=lambda item: str(item))]


def collect_cpack_files(build_dir: Path) -> list[dict[str, Any]]:
    paths = set(sorted_files(build_dir, "CPackConfig.cmake"))
    paths.update(sorted_files(build_dir, "CPackSourceConfig.cmake"))
    return [file_record(path, build_dir) for path in sorted(paths, key=lambda item: str(item))]


def load_json(path: Path) -> tuple[Any | None, str | None]:
    contents, error = read_text(path)
    if error is not None or contents is None:
        return None, error
    try:
        return json.loads(contents), None
    except json.JSONDecodeError as error:
        return None, str(error)


def compact_file_api_target(
    target_summary: dict[str, Any], target_data: dict[str, Any], reply_dir: Path, build_dir: Path
) -> dict[str, Any]:
    json_file = target_summary.get("jsonFile")
    target_file = reply_dir / json_file if isinstance(json_file, str) else None
    dependencies = target_data.get("dependencies", [])
    compile_groups = target_data.get("compileGroups", [])
    if not isinstance(dependencies, list):
        dependencies = []
    if not isinstance(compile_groups, list):
        compile_groups = []
    dependency_ids = [
        dependency["id"]
        for dependency in dependencies
        if isinstance(dependency, dict) and isinstance(dependency.get("id"), str)
    ]
    record: dict[str, Any] = {
        "id": target_summary.get("id"),
        "name": target_data.get("name", target_summary.get("name")),
        "type": target_data.get("type"),
        "artifacts": target_data.get("artifacts", []),
        "dependencies": dependency_ids,
        "compile_group_count": len(compile_groups),
        "link": target_data.get("link"),
        "archive": target_data.get("archive"),
    }
    if target_file is not None:
        record["reply_file"] = file_record(target_file, build_dir)
    return record


def compact_file_api_codemodel(
    codemodel: dict[str, Any], reply_dir: Path, build_dir: Path
) -> list[dict[str, Any]]:
    configurations: list[dict[str, Any]] = []
    for configuration in codemodel.get("configurations", []):
        if not isinstance(configuration, dict):
            continue
        targets: list[dict[str, Any]] = []
        for target_summary in configuration.get("targets", []):
            if not isinstance(target_summary, dict):
                continue
            json_file = target_summary.get("jsonFile")
            if not isinstance(json_file, str):
                targets.append(
                    {
                        "id": target_summary.get("id"),
                        "name": target_summary.get("name"),
                        "reply_error": "target summary had no jsonFile",
                    }
                )
                continue
            target_file = reply_dir / json_file
            target_data, error = load_json(target_file)
            if error is not None or not isinstance(target_data, dict):
                targets.append(
                    {
                        "id": target_summary.get("id"),
                        "name": target_summary.get("name"),
                        "reply_file": file_record(target_file, build_dir),
                        "reply_error": error or "target reply was not a JSON object",
                    }
                )
                continue
            targets.append(
                compact_file_api_target(target_summary, target_data, reply_dir, build_dir)
            )
        configurations.append({"name": configuration.get("name", ""), "targets": targets})
    return configurations


def compact_file_api_cache(cache: dict[str, Any]) -> dict[str, Any]:
    entries = cache.get("entries", [])
    compact_entries = []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            compact_entries.append(
                {
                    "name": entry.get("name"),
                    "type": entry.get("type"),
                    "value": entry.get("value"),
                }
            )
    return {"entry_count": len(compact_entries), "entries": compact_entries}


def collect_file_api(build_dir: Path) -> dict[str, Any]:
    """Read CMake File API v2 replies when the runner requested them."""

    reply_dir = build_dir / ".cmake" / "api" / "v1" / "reply"
    record: dict[str, Any] = {
        "available": False,
        "query": {
            "codemodel_v2": file_record(
                build_dir
                / ".cmake"
                / "api"
                / "v1"
                / "query"
                / "client-hipblaslt-configure-matrix"
                / "codemodel-v2",
                build_dir,
            ),
            "cache_v2": file_record(
                build_dir
                / ".cmake"
                / "api"
                / "v1"
                / "query"
                / "client-hipblaslt-configure-matrix"
                / "cache-v2",
                build_dir,
            ),
        },
        "codemodel": {"configurations": []},
        "cache_v2": {"entry_count": 0, "entries": []},
    }
    if not reply_dir.is_dir():
        return record

    index_paths = sorted(
        (path for path in reply_dir.glob("index-*.json") if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    if not index_paths:
        return record

    index_path = index_paths[-1]
    record["index"] = file_record(index_path, build_dir)
    index, error = load_json(index_path)
    if error is not None or not isinstance(index, dict):
        record["read_error"] = error or "index reply was not a JSON object"
        return record
    record["available"] = True
    record["cmake"] = index.get("cmake")

    objects = index.get("objects", [])
    if not isinstance(objects, list):
        record["read_error"] = "index reply had no objects array"
        return record
    by_kind: dict[str, dict[str, Any]] = {}
    for item in objects:
        if isinstance(item, dict) and isinstance(item.get("kind"), str):
            by_kind.setdefault(item["kind"], item)

    codemodel_object = by_kind.get("codemodel")
    if codemodel_object is not None:
        codemodel_file_name = codemodel_object.get("jsonFile")
        if isinstance(codemodel_file_name, str):
            codemodel_file = reply_dir / codemodel_file_name
            record["codemodel"]["reply_file"] = file_record(codemodel_file, build_dir)
            codemodel, codemodel_error = load_json(codemodel_file)
            if isinstance(codemodel, dict):
                record["codemodel"]["version"] = codemodel_object.get("version")
                record["codemodel"]["configurations"] = compact_file_api_codemodel(
                    codemodel, reply_dir, build_dir
                )
            else:
                record["codemodel"]["read_error"] = (
                    codemodel_error or "codemodel reply was not a JSON object"
                )

    cache_object = by_kind.get("cache")
    if cache_object is not None:
        cache_file_name = cache_object.get("jsonFile")
        if isinstance(cache_file_name, str):
            cache_file = reply_dir / cache_file_name
            record["cache_v2"]["reply_file"] = file_record(cache_file, build_dir)
            cache, cache_error = load_json(cache_file)
            if isinstance(cache, dict):
                record["cache_v2"].update(compact_file_api_cache(cache))
                record["cache_v2"]["version"] = cache_object.get("version")
            else:
                record["cache_v2"]["read_error"] = (
                    cache_error or "cache reply was not a JSON object"
                )
    return record


def collect_build_system_files(
    build_dir: Path, link_commands: list[dict[str, Any]], file_api: dict[str, Any]
) -> dict[str, Any]:
    """Keep generator files as an inspectable fallback for link.txt-free Ninja trees."""

    paths = (
        build_dir / "build.ninja",
        build_dir / "rules.ninja",
        build_dir / "CMakeFiles" / "rules.ninja",
        build_dir / "Makefile",
        build_dir / "CMakeFiles" / "Makefile2",
    )
    if link_commands:
        link_command_source = "link.txt"
    elif file_api.get("available"):
        link_command_source = "file-api-codemodel-v2"
    else:
        link_command_source = "generator-files"
    ninja_link_rules: list[dict[str, str]] = []
    for rules_path in (build_dir / "rules.ninja", build_dir / "CMakeFiles" / "rules.ninja"):
        if not rules_path.is_file():
            continue
        contents, error = read_text(rules_path)
        if error is None and contents is not None:
            for match in NINJA_RULE_RE.finditer(contents):
                name = match.group("name")
                if "LINKER" not in name and "ARCHIVER" not in name:
                    continue
                command = ""
                for line in match.group("body").splitlines():
                    if line.startswith("  command = "):
                        command = line[len("  command = ") :]
                        break
                ninja_link_rules.append(
                    {
                        "file": relative_path(rules_path, build_dir),
                        "name": name,
                        "command": command,
                    }
                )

    return {
        "link_command_source": link_command_source,
        "generator_files": [
            file_record(path, build_dir) for path in paths if path.is_file()
        ],
        "ninja_link_rules": ninja_link_rules,
    }


def resolve_install_include(script: Path, include_path: str) -> Path:
    expanded = include_path.replace("${CMAKE_CURRENT_LIST_DIR}", str(script.parent))
    path = Path(expanded)
    if not path.is_absolute():
        path = script.parent / path
    return canonical_path(path)


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def collect_install_scripts(build_dir: Path) -> dict[str, Any]:
    scripts = list(sorted_files(build_dir, "cmake_install.cmake"))
    edges: list[dict[str, Any]] = []
    adjacency: dict[Path, list[Path]] = {}

    for script in scripts:
        script = canonical_path(script)
        contents, error = read_text(script)
        if error is not None or contents is None:
            continue
        destinations: list[Path] = []
        for raw_destination in INSTALL_INCLUDE_RE.findall(contents):
            destination = resolve_install_include(script, raw_destination)
            destinations.append(destination)
            edges.append(
                {
                    "from": relative_path(script, build_dir),
                    "include": raw_destination,
                    "to": relative_path(destination, build_dir),
                    "available": destination.is_file(),
                }
            )
        adjacency[script] = destinations

    root_script = canonical_path(build_dir / "cmake_install.cmake")
    reachable: list[str] = []
    pending = [root_script] if root_script.is_file() else []
    seen: set[Path] = set()
    while pending:
        script = pending.pop()
        if script in seen or not script.is_file() or not path_is_within(script, build_dir):
            continue
        seen.add(script)
        reachable.append(relative_path(script, build_dir))
        pending.extend(adjacency.get(script, []))

    return {
        "scripts": [file_record(script, build_dir) for script in scripts],
        "include_edges": edges,
        "reachable": sorted(reachable),
    }


def parse_context(values: list[str]) -> dict[str, str]:
    context: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"context value must be KEY=VALUE: {value}")
        key, setting = value.split("=", 1)
        if not key:
            raise ValueError(f"context key must not be empty: {value}")
        context[key] = setting
    return context


def parse_assertions(values: list[str]) -> list[dict[str, str | bool]]:
    assertions: list[dict[str, str | bool]] = []
    for value in values:
        fields = value.split("|", 2)
        if len(fields) != 3:
            assertions.append({"description": value, "passed": True})
            continue
        kind, path, expected = fields
        assertions.append(
            {"kind": kind, "path": path, "expected": expected, "passed": True}
        )
    return assertions


def make_snapshot(arguments: argparse.Namespace, context: dict[str, str]) -> dict[str, Any]:
    build_dir = canonical_path(Path(arguments.build_dir))
    source_dir = canonical_path(Path(arguments.source_dir))
    stage_dir = canonical_path(Path(arguments.stage_dir))
    command = list(arguments.configure_command)

    link_commands = collect_link_commands(build_dir)
    file_api = collect_file_api(build_dir)
    snapshot: dict[str, Any] = {
        "id": arguments.cell,
        "status": arguments.status,
        "exit_code": arguments.exit_code,
        "captured_at": utc_now(),
        "source_dir": str(source_dir),
        "build_dir": str(build_dir),
        "stage_dir": str(stage_dir),
        "matrix_context": context,
        "configure": {
            "command": command,
            "program": command[0] if command else None,
            "arguments": command[1:],
        },
        "configure_log": collect_log(Path(arguments.log_file), build_dir)
        if arguments.log_file
        else None,
        "assertions": parse_assertions(arguments.assertion),
        "cache": parse_cmake_cache(build_dir / "CMakeCache.txt", build_dir),
        "build_tree": {
            "compile_commands": collect_compile_commands(build_dir),
            "link_commands": link_commands,
            "file_api": file_api,
            "build_system": collect_build_system_files(build_dir, link_commands, file_api),
            "target_directory_index": collect_targets(build_dir),
            "package_configs_and_exports": collect_package_configs_and_exports(build_dir),
            "cpack": collect_cpack_files(build_dir),
            "install_scripts": collect_install_scripts(build_dir),
        },
    }
    if arguments.error:
        snapshot["error"] = arguments.error
    return snapshot


def read_existing_result(output: Path) -> dict[str, Any]:
    if not output.exists():
        return {
            "kind": MATRIX_KIND,
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now(),
            "cells": [],
            "comparisons": [],
        }
    try:
        data = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read existing result {output}: {error}") from error
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"existing result {output} does not use schema_version {SCHEMA_VERSION}"
        )
    if not isinstance(data.get("cells"), list):
        raise RuntimeError(f"existing result {output} has no cells array")
    if data.get("kind") not in (None, MATRIX_KIND):
        raise RuntimeError(f"existing result {output} has unexpected kind {data['kind']!r}")
    if "comparisons" in data and not isinstance(data["comparisons"], list):
        raise RuntimeError(f"existing result {output} has a non-array comparisons field")
    return data


def atomic_write(output: Path, data: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            json.dump(data, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def update_result(arguments: argparse.Namespace) -> None:
    output = canonical_path(Path(arguments.output))
    result = read_existing_result(output)
    context = parse_context(arguments.context)
    snapshot = make_snapshot(arguments, context)

    result["updated_at"] = utc_now()
    result["kind"] = MATRIX_KIND
    result.setdefault("comparisons", [])
    result["matrix"] = {
        "runner": str(canonical_path(Path(arguments.runner))),
        "results_dir": str(output.parent),
    }

    cells = result["cells"]
    for index, cell in enumerate(cells):
        if isinstance(cell, dict) and cell.get("id") == arguments.cell:
            cells[index] = snapshot
            break
    else:
        cells.append(snapshot)
    atomic_write(output, result)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update a JSON snapshot for one CMake configure-matrix cell."
    )
    parser.add_argument("--output", required=True, help="Matrix JSON artifact to update.")
    parser.add_argument("--runner", required=True, help="Path to the shell matrix runner.")
    parser.add_argument("--cell", required=True, help="Stable configure-matrix cell name.")
    parser.add_argument(
        "--status",
        required=True,
        choices=("passed", "failed", "skipped", "unavailable"),
        help="Cell result status.",
    )
    parser.add_argument("--exit-code", required=True, type=int, help="Cell process exit code.")
    parser.add_argument("--source-dir", required=True, help="CMake source directory for this cell.")
    parser.add_argument("--build-dir", required=True, help="CMake binary directory for this cell.")
    parser.add_argument("--stage-dir", required=True, help="Cell install staging directory.")
    parser.add_argument(
        "--log-file", default="", help="Configure log captured by the shell runner."
    )
    parser.add_argument(
        "--error", default="", help="Failure detail captured by the shell runner."
    )
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Resolved matrix-wide setting; repeatable.",
    )
    parser.add_argument(
        "--assertion",
        action="append",
        default=[],
        metavar="KIND|PATH|EXPECTED",
        help="A passing assertion made by the shell runner; repeatable.",
    )
    parser.add_argument(
        "--configure-command",
        nargs=argparse.REMAINDER,
        default=[],
        metavar="ARG",
        help="Exact CMake configure command. This option must be last.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        update_result(arguments)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
