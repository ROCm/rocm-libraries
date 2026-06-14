#!/usr/bin/env python3
import concurrent.futures
import json
import os
import re
import resource
import shlex
import stat
import subprocess
import sys
from pathlib import Path


def preprocess_and_find_gtests(gtest):
    pp_folder = "test/gtest/pp"
    build_command = shlex.split(gtest)
    build_command.insert(1, "/P" if sys.platform == "win32" else "-E")

    preprocess_command = []
    flagged = False
    skip = -1
    for index, token in enumerate(build_command):
        if token.startswith("-I") and not flagged:
            preprocess_command.append(f"-I{pp_folder}")
            preprocess_command.append("-I../test")
            flagged = True
        elif token == "-o":
            skip = index + 1
            outfile = (
                pp_folder + "/" + Path(build_command[index + 1]).with_suffix(".i").name
            )

        if index > skip:
            preprocess_command.append(token)

    try:
        result = subprocess.run(
            preprocess_command, capture_output=True, text=True, check=True
        )

    except subprocess.CalledProcessError as e:
        # 3. Gracefully catch compilation errors if the command fails
        print("❌ Compilation failed!", file=sys.stderr)
        print(f"Exit Code: {e.returncode}", file=sys.stderr)
        print(f"Compiler Error Messages:\n{e.stderr}", file=sys.stderr)
        sys.exit(e.returncode)

    new_fixtures = re.findall(
        r"\b(?:TEST_F|TEST_P|TYPED_TEST)\s*\(\s*(\w+)", result.stdout
    )

    return new_fixtures


def extract_gtext_fixtures(json_path: str, output_file: str, pp_folder: str):
    exes = []
    gtests = []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            for entry in data:
                file_path = Path(entry.get("file", ""))
                parts = file_path.parts

                # Check if the file is located in the test/gtest folder
                if any(
                    parts[i] == "test" and parts[i + 1] == "gtest"
                    for i in range(len(parts) - 1)
                ):
                    exes.append(f"bin/test_{file_path.stem}")
                    gtests.append(entry.get("command"))

    except FileNotFoundError:
        print(f"Error: {json_path} not found.")
    except json.JSONDecodeError:
        print(f"Error: {json_path} is not a valid JSON file.")

    gtest_hpp = Path(pp_folder) / "gtest" / "gtest.h"
    gtest_hpp.parent.mkdir(parents=True, exist_ok=True)
    with open(gtest_hpp, "w") as f:
        pass

    thread_count = 384
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        all_fixtures = executor.map(preprocess_and_find_gtests, gtests)

    results = {}
    fixture_count = 0
    for exe, fixtures in zip(exes, all_fixtures):
        if fixtures:
            results[exe] = fixtures
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


def main():
    if len(sys.argv) < 2:
        compile_commands = "compile_commands.json"
    else:
        compile_commands = sys.argv[1]

    compile_commands_path = Path(compile_commands)
    if not compile_commands_path.is_file():
        print(
            f"Usage: {sys.argv[0]} [<path_to_compile_commands.json> [<path_to_miopen_dapper_fixtures.json>]]",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(sys.argv) < 3:
        output_file = "miopen_dapper_fixtures.json"
    else:
        output_file = sys.argv[2]

    pp_folder = "test/gtest/pp"
    extract_gtext_fixtures(compile_commands, output_file, pp_folder)


if __name__ == "__main__":
    main()
