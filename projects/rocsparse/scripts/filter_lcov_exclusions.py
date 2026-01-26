#!/usr/bin/env python3
# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

"""
Filter LCOV coverage files to respect LCOV_EXCL_START/LCOV_EXCL_STOP markers.

This script reads an LCOV format coverage file and source files, then removes
coverage data for lines marked with exclusion comments:
  - LCOV_EXCL_START / LCOV_EXCL_STOP: Exclude a block of lines
  - LCOV_EXCL_LINE: Exclude a single line
  - GCOV_EXCL_START / GCOV_EXCL_STOP: Same as LCOV (for compatibility)
  - GCOVR_EXCL_START / GCOVR_EXCL_STOP: Same as LCOV (for compatibility)

Usage:
    python filter_lcov_exclusions.py input.info output.info
"""

import re
import sys
import os
from pathlib import Path


# Regex patterns for exclusion markers (supporting LCOV_, GCOV_, GCOVR_ prefixes)
EXCL_START_PATTERN = re.compile(r'(LCOV|GCOV|GCOVR)_EXCL_START')
EXCL_STOP_PATTERN = re.compile(r'(LCOV|GCOV|GCOVR)_EXCL_STOP')
EXCL_LINE_PATTERN = re.compile(r'(LCOV|GCOV|GCOVR)_EXCL_LINE')


def get_excluded_lines(source_file: str) -> set:
    """
    Parse a source file and return the set of line numbers that should be excluded.
    """
    excluded_lines = set()
    
    if not os.path.isfile(source_file):
        return excluded_lines
    
    try:
        with open(source_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Warning: Could not read {source_file}: {e}", file=sys.stderr)
        return excluded_lines
    
    in_exclusion_block = False
    
    for line_num, line in enumerate(lines, start=1):
        # Check for EXCL_START
        if EXCL_START_PATTERN.search(line):
            in_exclusion_block = True
            excluded_lines.add(line_num)
            continue
        
        # Check for EXCL_STOP
        if EXCL_STOP_PATTERN.search(line):
            in_exclusion_block = False
            excluded_lines.add(line_num)
            continue
        
        # If in exclusion block, add this line
        if in_exclusion_block:
            excluded_lines.add(line_num)
            continue
        
        # Check for single-line exclusion
        if EXCL_LINE_PATTERN.search(line):
            excluded_lines.add(line_num)
    
    return excluded_lines


def filter_lcov_file(input_path: str, output_path: str) -> tuple:
    """
    Filter an LCOV file to remove excluded lines.
    
    Returns a tuple of (lines_removed, files_processed).
    """
    lines_removed = 0
    files_processed = 0
    current_source_file = None
    excluded_lines_cache = {}
    
    output_lines = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            # Track current source file
            if line.startswith('SF:'):
                current_source_file = line[3:]
                if current_source_file not in excluded_lines_cache:
                    excluded_lines_cache[current_source_file] = get_excluded_lines(current_source_file)
                    if excluded_lines_cache[current_source_file]:
                        files_processed += 1
                output_lines.append(line)
                continue
            
            # Filter line data (DA:line_number,count)
            if line.startswith('DA:'):
                parts = line[3:].split(',')
                if len(parts) >= 2:
                    try:
                        line_num = int(parts[0])
                        if current_source_file and line_num in excluded_lines_cache.get(current_source_file, set()):
                            lines_removed += 1
                            continue  # Skip this line
                    except ValueError:
                        pass
            
            # Filter branch data (BRDA:line,block,branch,taken)
            if line.startswith('BRDA:'):
                parts = line[5:].split(',')
                if len(parts) >= 4:
                    try:
                        line_num = int(parts[0])
                        if current_source_file and line_num in excluded_lines_cache.get(current_source_file, set()):
                            lines_removed += 1
                            continue  # Skip this branch
                    except ValueError:
                        pass
            
            output_lines.append(line)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    return lines_removed, files_processed


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.info> <output.info>", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.isfile(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    lines_removed, files_processed = filter_lcov_file(input_path, output_path)
    
    print(f"Filtered LCOV coverage file:")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Files with exclusions: {files_processed}")
    print(f"  Lines/branches removed: {lines_removed}")


if __name__ == '__main__':
    main()
