#!/usr/bin/env python3
# ########################################################################
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

The following LCOV record types are filtered for excluded lines:
  - DA:   per-line execution counts
  - BRDA: per-branch execution counts
  - FN:   function declarations (line number + name)
  - FNDA: function execution counts (matched by name to dropped FN records)

Dropping FN/FNDA records is required so that genhtml does not fail with
"function ... found on line but no corresponding 'line' coverage data point.
Cannot derive function end line." when an excluded block surrounds a whole
function definition. FNF (functions found) and FNH (functions hit) totals
are recomputed accordingly per source-file section.

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
    # Track function names whose FN: record was dropped, so the matching
    # FNDA: record (which has no line number) can also be dropped. Reset
    # whenever a new SF: section starts.
    excluded_function_names = set()
    # Counters for FNF/FNH adjustment within the current SF section.
    fn_records_dropped = 0
    fnda_hits_dropped = 0
    
    output_lines = []
    
    def flush_section_counters():
        # Walk back through the buffered output for the current SF section
        # and adjust FNF: (functions found) and FNH: (functions hit).
        nonlocal fn_records_dropped, fnda_hits_dropped
        if fn_records_dropped == 0 and fnda_hits_dropped == 0:
            return
        # Find the start of the current SF section in output_lines.
        for idx in range(len(output_lines) - 1, -1, -1):
            entry = output_lines[idx]
            if entry.startswith('SF:'):
                break
            if entry.startswith('FNF:'):
                try:
                    val = int(entry[4:])
                    output_lines[idx] = f'FNF:{max(0, val - fn_records_dropped)}'
                except ValueError:
                    pass
            elif entry.startswith('FNH:'):
                try:
                    val = int(entry[4:])
                    output_lines[idx] = f'FNH:{max(0, val - fnda_hits_dropped)}'
                except ValueError:
                    pass
        fn_records_dropped = 0
        fnda_hits_dropped = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            # End-of-record: adjust FNF/FNH for the just-finished section.
            if line == 'end_of_record':
                flush_section_counters()
                excluded_function_names = set()
                output_lines.append(line)
                continue
            
            # Track current source file
            if line.startswith('SF:'):
                # Defensive: also flush if we somehow see SF: without a prior
                # end_of_record (shouldn't normally happen).
                flush_section_counters()
                excluded_function_names = set()
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
            
            # Filter function records (FN:line_number,function_name).
            # If the FN record points at an excluded line, drop it AND
            # remember the function name so we also drop its FNDA: record.
            # Without this, genhtml errors out with:
            #   "function ... found on line but no corresponding 'line'
            #    coverage data point. Cannot derive function end line."
            if line.startswith('FN:'):
                parts = line[3:].split(',', 1)
                if len(parts) == 2:
                    try:
                        line_num = int(parts[0])
                        if current_source_file and line_num in excluded_lines_cache.get(current_source_file, set()):
                            excluded_function_names.add(parts[1])
                            fn_records_dropped += 1
                            lines_removed += 1
                            continue
                    except ValueError:
                        pass
            
            # Filter function execution data (FNDA:count,function_name)
            # tied to a previously dropped FN: record.
            if line.startswith('FNDA:'):
                parts = line[5:].split(',', 1)
                if len(parts) == 2 and parts[1] in excluded_function_names:
                    try:
                        if int(parts[0]) > 0:
                            fnda_hits_dropped += 1
                    except ValueError:
                        pass
                    lines_removed += 1
                    continue
            
            output_lines.append(line)
    
    # Defensive flush in case the file didn't end with end_of_record.
    flush_section_counters()
    
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
