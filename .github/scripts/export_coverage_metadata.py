#!/usr/bin/env python3
"""Export coverage metadata from a build tree to JSON for use in the report job.

This is the build-time half of the coverage flow described in TheRock's coverage
design docs. It reads ``test_categories_coverage.yaml``, verifies the configured
coverage objects exist in the build/dist tree, and writes a ``coverage_metadata.json``
that ``coverage_runner.py`` consumes later to merge profraw files and produce a report.

Object/tool names are recorded by file name (basename). The report job locates the
actual files in whatever layout it has (build dist or staged artifact), so version
suffixes such as ``libhiprand.so.1.1`` are handled by the runner.
"""
import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_coverage_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_project_key(config: dict, project: str) -> str:
    """Return the config key for ``project`` (case-insensitive)."""
    projects = config.get("projects", {})
    if project in projects:
        return project
    lowered = project.lower()
    if lowered in projects:
        return lowered
    raise ValueError(
        f"Project '{project}' not found in coverage config "
        f"(known: {sorted(projects)})"
    )


def find_by_basename(build_dir: Path, basename: str) -> list[str]:
    """Find files in ``build_dir`` whose name starts with ``basename``.

    Matches versioned shared libraries (e.g. libhiprand.so -> libhiprand.so.1.1).
    Returns paths relative to ``build_dir`` so the metadata is location independent.
    """
    matches: list[str] = []
    for path in build_dir.rglob(f"{basename}*"):
        if path.is_file():
            matches.append(str(path.relative_to(build_dir)))
    return sorted(set(matches))


def export_metadata(
    build_dir: Path, project: str, config_path: Path, output_path: Path
):
    config = load_coverage_config(config_path)
    key = resolve_project_key(config, project)
    project_config = config["projects"][key]

    if not project_config.get("enabled", False):
        logging.info("Coverage disabled for %s; nothing to export.", key)
        return

    objects = project_config.get("coverage_objects", {})
    found = {"libraries": [], "test_binaries": []}
    for kind in ("libraries", "test_binaries"):
        for basename in objects.get(kind, []) or []:
            hits = find_by_basename(build_dir, basename)
            if hits:
                found[kind].extend(hits)
            else:
                logging.warning("Coverage object not found for %s: %s", kind, basename)

    metadata = {
        "project": key,
        "coverage_objects": found,
        # Basenames are kept so the runner can re-resolve in a different layout.
        "object_basenames": {
            "libraries": objects.get("libraries", []) or [],
            "test_binaries": objects.get("test_binaries", []) or [],
        },
        "ignore_filename_regex": project_config.get("ignore_filename_regex", ""),
        "llvm_profile_pattern": project_config.get("llvm_profile_pattern", "%m"),
        "test_category": project_config.get("test_category", ""),
        "llvm_tools": {
            "llvm_profdata": "llvm-profdata",
            "llvm_cov": "llvm-cov",
            "llvm_cxxfilt": "llvm-cxxfilt",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info("Exported coverage metadata to %s", output_path)
    logging.info("  Libraries found:     %d", len(found["libraries"]))
    logging.info("  Test binaries found: %d", len(found["test_binaries"]))


def main():
    parser = argparse.ArgumentParser(description="Export coverage metadata to JSON")
    parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="Build/dist tree to search for coverage objects",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Project key or name (e.g. hiprand / HIPRAND)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to test_categories_coverage.yaml",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output coverage_metadata.json path"
    )
    args = parser.parse_args()
    export_metadata(args.build_dir, args.project, args.config, args.output)


if __name__ == "__main__":
    main()
