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
import shutil
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


def _looks_like_executable(path: Path) -> bool:
    """Heuristic: a built test executable (ELF), not a source/object/CMake file."""
    if not path.is_file() or path.is_symlink():
        return False
    if path.suffix:  # test executables have no extension (test_foo, not test_foo.cpp)
        return False
    if "CMakeFiles" in path.parts:
        return False
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def find_test_binaries(build_dir: Path, prefix: str, scope: str | None) -> list[Path]:
    """Resolve instrumented test executables whose name starts with ``prefix``.

    Header-only libraries have no shared object to instrument; coverage instead
    comes from the test binaries. ``scope`` (e.g. the component build subdir name
    like ``rocPRIM``) restricts matches to that component so a shared group build
    (PRIM builds rocprim + hipcub + rocthrust together) does not pull in the other
    components' binaries. De-duplicates by basename, preferring the copy under a
    ``test`` directory.
    """
    by_name: dict[str, Path] = {}
    for path in build_dir.rglob(f"{prefix}*"):
        if not _looks_like_executable(path):
            continue
        if scope and scope not in path.parts:
            continue
        name = path.name
        prev = by_name.get(name)
        if prev is None or ("test" in path.parts and "test" not in prev.parts):
            by_name[name] = path
    return [by_name[k] for k in sorted(by_name)]


def export_metadata(
    build_dir: Path,
    project: str,
    config_path: Path,
    output_path: Path,
    cmake_target: str | None = None,
    stage_dir: Path | None = None,
):
    config = load_coverage_config(config_path)
    key = resolve_project_key(config, project)
    project_config = config["projects"][key]

    if not project_config.get("enabled", False):
        logging.info("Coverage disabled for %s; nothing to export.", key)
        return

    objects = project_config.get("coverage_objects", {})

    # Shared libraries are located by basename and staged separately (by the
    # workflow) from dist/lib; the runner re-resolves them by basename.
    found = {"libraries": [], "test_binaries": []}
    for basename in objects.get("libraries", []) or []:
        hits = find_by_basename(build_dir, basename)
        if hits:
            found["libraries"].extend(hits)
        else:
            logging.warning("Coverage object not found for libraries: %s", basename)

    # Header-only libraries: the configured test_binaries entries are name
    # prefixes. Resolve them to the actual instrumented executables, then copy
    # them into the staged coverage-objects dir so the report job can pass them
    # to llvm-cov as -object. We record their paths relative to the stage dir so
    # the runner finds them directly.
    staged_test_binaries: list[str] = []
    staged_basenames: list[str] = []
    test_stage = (stage_dir / "test") if stage_dir is not None else None
    total_bytes = 0
    for prefix in objects.get("test_binaries", []) or []:
        binaries = find_test_binaries(build_dir, prefix, cmake_target)
        if not binaries:
            logging.warning(
                "No instrumented test binaries found for prefix '%s' (scope=%s)",
                prefix,
                cmake_target,
            )
            continue
        logging.info(
            "Resolved %d test binary(ies) for prefix '%s'", len(binaries), prefix
        )
        for binary in binaries:
            if test_stage is not None:
                test_stage.mkdir(parents=True, exist_ok=True)
                dest = test_stage / binary.name
                if not dest.exists():
                    shutil.copy2(binary, dest)
                    total_bytes += dest.stat().st_size
                staged_test_binaries.append(f"test/{binary.name}")
            else:
                staged_test_binaries.append(str(binary.relative_to(build_dir)))
            staged_basenames.append(binary.name)
    found["test_binaries"] = sorted(set(staged_test_binaries))
    if test_stage is not None and staged_basenames:
        logging.info(
            "Staged %d test binary(ies) (%.1f MiB) into %s",
            len(set(staged_basenames)),
            total_bytes / (1024 * 1024),
            test_stage,
        )

    metadata = {
        "project": key,
        "coverage_objects": found,
        # Basenames are kept so the runner can re-resolve in a different layout.
        "object_basenames": {
            "libraries": objects.get("libraries", []) or [],
            "test_binaries": sorted(set(staged_basenames)),
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
    parser.add_argument(
        "--cmake-target",
        type=str,
        default=None,
        help="Component build subdir name used to scope header-only test binaries "
        "(e.g. rocPRIM), so a shared group build does not pull in siblings.",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Coverage-objects dir to copy resolved test binaries into "
        "(header-only libs). Defaults to the output file's directory.",
    )
    args = parser.parse_args()
    stage_dir = args.stage_dir or args.output.parent
    export_metadata(
        args.build_dir,
        args.project,
        args.config,
        args.output,
        cmake_target=args.cmake_target,
        stage_dir=stage_dir,
    )


if __name__ == "__main__":
    main()
