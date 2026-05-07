#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

"""
This script splits a C++ test file into individual test files and a main file.

Usage:
    python split_wcnn_tests.py <input_cpp_file> <output_directory> [test_filter]

Example:
    python split_wcnn_tests.py ${WCNN_SOURCE_DIR}/${SOURCE_FILE} ${VARIANT_BUILD_DIR}
    python split_wcnn_tests.py ${WCNN_SOURCE_DIR}/${SOURCE_FILE} ${VARIANT_BUILD_DIR} ".*bf8.*"

The script will:
1. Extract each TEST() block into a separate .cpp file named:
   <test_suite>__<test_name>.cpp
2. Generate a main.cpp file named <base_name>_main.cpp containing the header
   includes and main() function from the original file.
3. If test_filter is provided (and not ".*"), only tests matching the regex
   pattern will be generated.
"""

import sys
import re
import shutil
from pathlib import Path

# Internal verbose flag - set to True to see detailed output (individual test names and generated files)
VERBOSE = False


def log(msg=""):
    """Print a message with 2-tab prefix to distinguish from CMake output."""
    print(f"\t[split_wcnn_tests.py]: {msg}")


def extract_includes(content):
    """Extract include statements from the beginning of the file."""
    lines = content.split("\n")
    includes = []
    for line in lines:
        stripped = line.strip()
        # Stop at first TEST or other code (but include comments and includes)
        if stripped.startswith("TEST(") or stripped.startswith("int main("):
            break
        if (
            stripped.startswith("#include")
            or stripped.startswith("// SPDX")
            or stripped.startswith("// Copyright")
            or stripped == ""
        ):
            includes.append(line)
        elif stripped.startswith("#if 0"):
            # Skip the #if 0 block entirely
            break

    # Clean up - keep only license and includes
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("// SPDX") or stripped.startswith("// Copyright"):
            result.append(line)
        elif stripped == "":
            if result:  # Only add empty line after we have some content
                result.append(line)
        elif stripped.startswith("#include"):
            result.append(line)
        elif (
            stripped.startswith("#if 0")
            or stripped.startswith("TEST(")
            or stripped.startswith("int main(")
        ):
            break

    return "\n".join(result)


def extract_main_function(content):
    """Extract the main function from the end of the file."""
    # Find the last main function definition
    pattern = r"int\s+main\s*\(\s*int\s+argc\s*,\s*char\s*\*\s*argv\s*\[\s*\]\s*\)"
    matches = list(re.finditer(pattern, content))

    if not matches:
        return None

    # Get the last match (the actual main function, not one in #if 0 block)
    last_match = matches[-1]
    start_pos = last_match.start()

    # Find the opening brace
    brace_pos = content.find("{", last_match.end())
    if brace_pos == -1:
        return None

    # Find the matching closing brace (handle nested braces)
    brace_count = 1
    end_pos = brace_pos + 1
    while brace_count > 0 and end_pos < len(content):
        if content[end_pos] == "{":
            brace_count += 1
        elif content[end_pos] == "}":
            brace_count -= 1
        end_pos += 1

    return content[start_pos:end_pos]


def extract_tests(content):
    """Extract all TEST() blocks from the content."""
    tests = []

    # Pattern to match TEST(suite, name) { ... }
    # We need to handle nested braces properly
    pattern = r"TEST\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)"

    pos = 0
    while True:
        match = re.search(pattern, content[pos:])
        if not match:
            break

        test_suite = match.group(1).strip()
        test_name = match.group(2).strip()

        # Find the opening brace
        start_pos = pos + match.end()
        brace_pos = content.find("{", start_pos)
        if brace_pos == -1:
            pos = start_pos
            continue

        # Find the matching closing brace
        brace_count = 1
        end_pos = brace_pos + 1
        while brace_count > 0 and end_pos < len(content):
            if content[end_pos] == "{":
                brace_count += 1
            elif content[end_pos] == "}":
                brace_count -= 1
            end_pos += 1

        # Extract the full test block
        full_test = content[pos + match.start() : end_pos]

        tests.append({"suite": test_suite, "name": test_name, "content": full_test})

        pos = end_pos

    return tests


def generate_test_file(header, test_content):
    """Generate content for an individual test file."""
    return f"{header}\n\n{test_content}\n"


def generate_main_file(header, main_func):
    """Generate content for the main file."""
    return f"{header}\n\n{main_func}\n"


def sanitize_filename(name):
    """Sanitize a string to be used as a filename."""
    # Replace any characters that might be problematic in filenames
    return re.sub(r"[^\w\-]", "_", name)


def main():
    print("\n")
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        log(f"Usage: {sys.argv[0]} <input_cpp_file> <output_directory> [test_filter]")
        log(f"Example: {sys.argv[0]} grouped_conv_fwd_wcnn.cpp ./output")
        log(f"Example: {sys.argv[0]} grouped_conv_fwd_wcnn.cpp ./output '.*bf8.*'")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    test_filter = sys.argv[3] if len(sys.argv) == 4 else ".*"

    if not input_file.exists():
        log(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    # Clean output directory before generating new files
    # This prevents duplicate files when cmake is re-run
    if output_dir.exists():
        log(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get base name from input file (e.g., 'grouped_conv_fwd_wcnn' from 'grouped_conv_fwd_wcnn.cpp')
    base_name = input_file.stem

    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract components
    header = extract_includes(content)
    main_func = extract_main_function(content)
    tests = extract_tests(content)

    if not header:
        log("Warning: Could not extract header includes.")

    if not main_func:
        log("Warning: Could not extract main function.")

    if not tests:
        log("Warning: No TEST() blocks found.")
        sys.exit(1)

    # Filter tests based on test_filter regex
    if test_filter != ".*":
        try:
            filter_pattern = re.compile(test_filter)
        except re.error as e:
            log(f"Error: Invalid regex pattern '{test_filter}': {e}")
            sys.exit(1)

        original_count = len(tests)
        filtered_tests = []
        matched_names = []
        for test in tests:
            # Match against suite__name format (same as filename)
            test_full_name = f"{test['suite']}__{test['name']}"
            if filter_pattern.search(test_full_name):
                filtered_tests.append(test)
                matched_names.append(test_full_name)

        tests = filtered_tests

        # Output filter information
        log(f"TEST_FILTER applied: '{test_filter}'")
        log(f"Matched {len(tests)} of {original_count} tests")
        if VERBOSE:
            for name in matched_names:
                log(f"   - {name}")

        if not tests:
            log(f"Warning: No tests matched filter '{test_filter}'.")
            sys.exit(1)
    else:
        log(f"Found {len(tests)} test(s)")

    # Generate individual test files
    generated_files = []
    for test in tests:
        # Create filename: suite__testname.cpp (double underscore to separate suite and name)
        # e.g., grouped_conv_fwd_wcnn__half_float_s4x2_f1x1.cpp
        test_filename = (
            f"{sanitize_filename(test['suite'])}__{sanitize_filename(test['name'])}.cpp"
        )
        test_filepath = output_dir / test_filename

        test_file_content = generate_test_file(header, test["content"])

        with open(test_filepath, "w", encoding="utf-8") as f:
            f.write(test_file_content)

        generated_files.append(test_filename)
        if VERBOSE:
            log(f"Generated: {test_filename}")

    # Generate main file
    if main_func:
        main_filename = f"{base_name}_main.cpp"
        main_filepath = output_dir / main_filename

        main_file_content = generate_main_file(header, main_func)

        with open(main_filepath, "w", encoding="utf-8") as f:
            f.write(main_file_content)

        if VERBOSE:
            log(f"Generated: {main_filename}")

    # Print summary
    log()
    log("Summary:")
    log(f"  Input file: {input_file}")
    log(f"  Output directory: {output_dir}")
    log(f"  Generated {len(generated_files)} test file(s)")
    if main_func:
        log("  Generated 1 main file")

    # Write a list of generated files for CMake integration
    manifest_file = output_dir / f"{base_name}_files.txt"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for filename in generated_files:
            f.write(f"{filename}\n")
        if main_func:
            f.write(f"{base_name}_main.cpp\n")

    log(f"  File manifest: {manifest_file}")
    print("\n")


if __name__ == "__main__":
    main()
