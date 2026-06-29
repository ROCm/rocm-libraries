#!/usr/bin/env python3
"""Run tests with coverage profiling and/or generate coverage reports.

This is the report-time half of the coverage flow described in TheRock's coverage
design docs. It is driven by the ``coverage_metadata.json`` produced by
``export_coverage_metadata.py`` and can either:

  * run tests to produce profraw files (default), or
  * skip tests and merge pre-collected profraw files (``--skip-tests``), which is
    how the GitHub Actions flow uses it (tests run in the shared package-test
    workflow and upload their profraw).

Object and llvm tool locations are resolved by name within ``--build-dir`` so the
script works whether it is pointed at a full build/dist tree or a staged artifact.
"""
import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def find_tool(build_dir: Path, name: str) -> Path | None:
    """Locate an executable ``name`` inside ``build_dir`` (prefer bin/ dirs)."""
    candidates = [p for p in build_dir.rglob(name) if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (0 if p.parent.name == "bin" else 1, len(str(p))))
    return candidates[0]


def resolve_objects(build_dir: Path, metadata: dict) -> list[Path]:
    """Resolve coverage object files (libraries + test binaries) within build_dir."""
    objects: list[Path] = []

    # Prefer the explicit relative paths recorded at export time, if present.
    rel_objects = []
    for kind in ("libraries", "test_binaries"):
        rel_objects.extend(metadata.get("coverage_objects", {}).get(kind, []) or [])
    for rel in rel_objects:
        candidate = build_dir / rel
        if candidate.is_file():
            objects.append(candidate)

    # Fall back to resolving by basename (handles a different layout than export).
    if not objects:
        basenames = []
        for kind in ("libraries", "test_binaries"):
            basenames.extend(metadata.get("object_basenames", {}).get(kind, []) or [])
        for basename in basenames:
            hits = [p for p in build_dir.rglob(f"{basename}*") if p.is_file()]
            # Prefer concrete files (e.g. libhiprand.so.1.1) over bare symlinks,
            # and the most specific (longest) name.
            hits.sort(key=lambda p: (not p.is_symlink(), len(p.name)), reverse=True)
            if hits:
                objects.append(hits[0])

    # De-duplicate while preserving order.
    seen, unique = set(), []
    for obj in objects:
        key = obj.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(obj)
    return unique


def set_coverage_environment(metadata: dict, coverage_dir: Path):
    pattern = metadata.get("llvm_profile_pattern", "%m")
    profraw_dir = coverage_dir / "profraw"
    profraw_dir.mkdir(parents=True, exist_ok=True)
    profile_file = str(profraw_dir / f"{pattern}.profraw")
    os.environ["LLVM_PROFILE_FILE"] = profile_file
    logging.info("Set LLVM_PROFILE_FILE=%s", profile_file)


def run_tests(test_dir: Path, metadata: dict) -> int:
    category = metadata.get("test_category", "")
    cmd = [
        "ctest",
        "--test-dir",
        str(test_dir),
        "--output-on-failure",
        "--parallel",
        "8",
        "--timeout",
        "7200",
    ]
    if category:
        cmd += ["-L", category]
    logging.info("Running tests: %s", " ".join(cmd))
    return subprocess.run(cmd).returncode


def merge_profraw_files(llvm_profdata: Path, profraw_dir: Path, out_path: Path) -> Path:
    profraw_files = [str(p) for p in profraw_dir.rglob("*.profraw")]
    if not profraw_files:
        raise RuntimeError(f"No .profraw files found under {profraw_dir}")
    logging.info("Merging %d profraw file(s)", len(profraw_files))
    subprocess.run(
        [str(llvm_profdata), "merge", "-sparse", "-o", str(out_path), *profraw_files],
        check=True,
    )
    logging.info("Created %s", out_path)
    return out_path


def generate_reports(
    llvm_cov: Path,
    llvm_cxxfilt: Path | None,
    objects: list[Path],
    ignore_regex: str,
    profdata: Path,
    coverage_dir: Path,
    project: str,
):
    object_args: list[str] = []
    for obj in objects:
        object_args += ["-object", str(obj)]
    ignore_args = [f"-ignore-filename-regex={ignore_regex}"] if ignore_regex else []

    text_report = coverage_dir / f"code_cov_{project}.report"
    logging.info("Generating text report -> %s", text_report)
    with open(text_report, "w") as f:
        subprocess.run(
            [
                str(llvm_cov),
                "report",
                *object_args,
                f"-instr-profile={profdata}",
                *ignore_args,
            ],
            stdout=f,
            check=True,
        )
    print(text_report.read_text())

    logging.info("Generating HTML report -> %s", coverage_dir)
    show_cmd = [
        str(llvm_cov),
        "show",
        *object_args,
        f"-instr-profile={profdata}",
        *ignore_args,
        "--format=html",
        f"--output-dir={coverage_dir}",
    ]
    if llvm_cxxfilt is not None:
        show_cmd.insert(2, f"-Xdemangler={llvm_cxxfilt}")
    subprocess.run(show_cmd, check=True)

    lcov_file = coverage_dir / "coverage.info"
    logging.info("Generating LCOV export -> %s", lcov_file)
    with open(lcov_file, "w") as f:
        subprocess.run(
            [
                str(llvm_cov),
                "export",
                *object_args,
                f"-instr-profile={profdata}",
                *ignore_args,
                "--format=lcov",
            ],
            stdout=f,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Run/aggregate coverage and report")
    parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="Tree to resolve coverage objects and llvm tools from",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="coverage_metadata.json from export_coverage_metadata.py",
    )
    parser.add_argument(
        "--coverage-dir", type=Path, required=True, help="Output directory for reports"
    )
    parser.add_argument(
        "--profraw-dir",
        type=Path,
        default=None,
        help="Directory of profraw files (default: <coverage-dir>/profraw)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="ctest directory (default: --build-dir) when running tests",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Do not run tests; merge existing profraw files",
    )
    args = parser.parse_args()

    with open(args.metadata) as f:
        metadata = json.load(f)
    project = metadata.get("project", "project")
    logging.info("Coverage for project: %s", project)

    args.coverage_dir.mkdir(parents=True, exist_ok=True)
    profraw_dir = args.profraw_dir or (args.coverage_dir / "profraw")

    # Resolve tooling. Tools link libLLVM and bundled sysdeps that sit next to
    # them, so add their directory to LD_LIBRARY_PATH.
    llvm_profdata = find_tool(args.build_dir, "llvm-profdata")
    llvm_cov = find_tool(args.build_dir, "llvm-cov")
    llvm_cxxfilt = find_tool(args.build_dir, "llvm-cxxfilt")
    if llvm_profdata is None or llvm_cov is None:
        raise FileNotFoundError(
            f"llvm-profdata/llvm-cov not found under {args.build_dir}"
        )
    tool_dirs = {str(t.parent) for t in (llvm_profdata, llvm_cov, llvm_cxxfilt) if t}
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*tool_dirs, os.environ.get("LD_LIBRARY_PATH", "")]
    )
    for t in (llvm_profdata, llvm_cov, llvm_cxxfilt):
        if t:
            t.chmod(0o755)

    objects = resolve_objects(args.build_dir, metadata)
    if not objects:
        raise RuntimeError("No coverage object files could be resolved.")
    logging.info("Coverage objects: %s", ", ".join(str(o) for o in objects))

    if not args.skip_tests:
        set_coverage_environment(metadata, args.coverage_dir)
        rc = run_tests(args.test_dir or args.build_dir, metadata)
        if rc != 0:
            logging.warning("Tests exited with %d; continuing with coverage.", rc)

    # Some test runners (e.g. GPU families whose machines have no matching GPU)
    # run only host-side quick tests that never exercise the instrumented
    # library, so they upload no profraw. With nothing to merge there is no
    # coverage to report; treat that as a no-op success rather than failing the
    # job, so the absence of data on one GPU family does not break CI.
    if not list(profraw_dir.rglob("*.profraw")):
        logging.warning(
            "No .profraw files found under %s; skipping coverage report "
            "(no instrumented test data was produced for this build).",
            profraw_dir,
        )
        return

    profdata = merge_profraw_files(
        llvm_profdata, profraw_dir, args.coverage_dir / f"{project}.profdata"
    )
    generate_reports(
        llvm_cov,
        llvm_cxxfilt,
        objects,
        metadata.get("ignore_filename_regex", ""),
        profdata,
        args.coverage_dir,
        project,
    )
    logging.info("Coverage generation complete.")


if __name__ == "__main__":
    main()
