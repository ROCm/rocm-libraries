#!/usr/bin/env python3
import os
import stat
import resource
import subprocess
import json
import sys
from pathlib import Path


def is_executable(file_path: Path) -> bool:
    """Check if a file is an executable (not a directory)."""
    return (
        file_path.is_file()
        and os.access(file_path, os.X_OK)
        and not file_path.name.startswith(".")  # skip hidden files
    )


def disable_core_dump():
    """Disable core dump generation."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def list_gtest_fixtures(executable: Path):
    """Run the executable with --gtest_list_tests and return fixture names."""
    if "miopen_gtest" in str(executable):
        print(f"Info: Skipping single-binary {executable}.", file=sys.stderr)
        return []

    try:
        # Run the command and capture output
        result = subprocess.run(
            [str(executable), "--gtest_list_tests"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,  # prevent hanging
            preexec_fn=disable_core_dump,
        )
        if result.returncode != 0:
            print(
                f"Warning: Skipping '{executable}' due to non-zero exit code. Not a gtest..?",
                file=sys.stderr,
            )
            return []

        fixtures = []
        for line in result.stdout.splitlines():
            if (
                line.strip() and not line.startswith(" ") and line.endswith(".")
            ):  # non-indented = fixture name
                fixtures.append(line.strip())
        return fixtures

    except subprocess.TimeoutExpired:
        print(f"Error: {executable} timed out", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error running {executable}: {e}", file=sys.stderr)
        return []


def main(directory: str, output_file: str):
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Error: {directory} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    results = {}
    fixture_count = 0
    for file in dir_path.iterdir():
        if is_executable(file):
            fixtures = list_gtest_fixtures(file)
            if fixtures:
                src_file = f"bin/{file.name}"
                results[src_file] = fixtures
                fixture_count += len(fixtures)

    # Write results to JSON
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(
            f"List of {fixture_count} fixtures from {len(results)} files written to {output_file}"
        )
    except Exception as e:
        print(f"Error writing JSON file: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        bin_path = Path(".") / "build" / "bin"
        if not bin_path.is_dir():
            print(f"Usage: {sys.argv[0]} <directory> <output.json>", file=sys.stderr)
            sys.exit(1)
        bin_dir = str(bin_path)
    else:
        if not os.path.isdir(sys.argv[1]):
            print(f"Error: {sys.argv[1]} is not a valid directory", file=sys.stderr)
            sys.exit(1)
        bin_dir = sys.argv[1]

    if len(sys.argv) < 3:
        output_file = "miopen_dapper_fixtures.json"
    else:
        output_file = sys.argv[2]

    main(bin_dir, output_file)
