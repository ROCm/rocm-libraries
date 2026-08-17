#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

_INCLUDED_SUFFIXES = {".cpp", ".hpp", ".c", ".h"}
_EXCLUDED_RELATIVE_PREFIXES = (
    "build/",
    "flatbuffers_sdk/include/hipdnn_flatbuffers_sdk/data_objects/",
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run clang-format over hipDNN sources."
    )
    parser.add_argument("--clang-format", required=True)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("check", "format"))
    parser.add_argument("--files-per-invocation", required=True, type=int)
    parser.add_argument("--jobs", required=True, type=int)
    return parser.parse_args()


def _relative_format_path(source_dir, path):
    relative_path = path.relative_to(source_dir).as_posix()
    return relative_path


def _is_excluded(relative_path):
    return any(
        relative_path.startswith(prefix) for prefix in _EXCLUDED_RELATIVE_PREFIXES
    )


def _find_format_files(source_dir):
    files = []
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _INCLUDED_SUFFIXES:
            continue

        relative_path = _relative_format_path(source_dir, path)
        if not _is_excluded(relative_path):
            files.append(path)

    files.sort(key=lambda path: str(path).lower())
    return files


def _chunks(items, chunk_size):
    for index in range(0, len(items), chunk_size):
        yield items[index : index + chunk_size]


def _run_clang_format(clang_format, mode_args, files, source_dir):
    command = [clang_format, *mode_args, *(str(path) for path in files)]
    completed = subprocess.run(
        command,
        cwd=source_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        check=False,
    )
    return completed.returncode, completed.stdout


def _emit_output(output):
    for line in output.splitlines():
        if line.strip():
            print(line)


def main():
    args = _parse_args()

    if args.files_per_invocation < 1:
        print("files-per-invocation must be greater than 0", file=sys.stderr)
        return 2

    if args.jobs < 0:
        print("jobs must be greater than or equal to 0", file=sys.stderr)
        return 2

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    source_dir = args.source_dir.resolve()
    clang_format = str(Path(args.clang_format).resolve())

    mode_args = (
        ["--dry-run", "--Werror"] if args.mode == "check" else ["--verbose", "-i"]
    )
    format_files = _find_format_files(source_dir)
    if not format_files:
        print("No source files found for clang-format")
        return 0

    had_failure = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                _run_clang_format, clang_format, mode_args, chunk, source_dir
            )
            for chunk in _chunks(format_files, args.files_per_invocation)
        ]

        for future in concurrent.futures.as_completed(futures):
            returncode, output = future.result()
            _emit_output(output)
            if returncode != 0:
                had_failure = True

    return 1 if had_failure else 0


if __name__ == "__main__":
    sys.exit(main())
