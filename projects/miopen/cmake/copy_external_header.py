#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Script to copy external header files with nested headers inlined.  Since this is
intended for run time compilation, HIP headers are skipped entirely.
"""

import sys
import os
import re
from pathlib import Path

def inline_includes(filepath, root_dir, processed=None, depth=0, max_depth=100):
    """Recursively inline #include directives."""
    if processed is None:
        processed = set()

    # Prevent infinite recursion
    if depth > max_depth:
        raise RuntimeError(f"Include depth exceeded {max_depth}")

    # Normalize path
    filepath = os.path.normpath(filepath)

    if filepath in processed:
        return []
    processed.add(filepath)

    result = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Check for #include directives
                match = re.match(r'^\s*#include\s+[<"]([^>"]+)[>"]', line)
                if match:
                    include_file = match.group(1)

                    # Skip HIP headers entirely - do not include them at all
                    if include_file.startswith('hip/'):
                        continue

                    # Try to find the included file relative to root
                    include_path = os.path.join(root_dir, include_file)

                    if os.path.exists(include_path):
                        # Add comment showing what's being inlined
                        result.append(f'// BEGIN INLINED: {include_file}')
                        # Recursively inline
                        inlined_content = inline_includes(include_path, root_dir, processed, depth + 1, max_depth)
                        result.extend(inlined_content)
                        result.append(f'// END INLINED: {include_file}')
                    else:
                        # Keep the include if file not found (might be a system header)
                        result.append(line.rstrip())
                else:
                    # Keep non-include lines as-is
                    result.append(line.rstrip())
    except Exception as e:
        print(f'Error processing {filepath}: {e}', file=sys.stderr)
        raise

    return result

def main():
    if len(sys.argv) != 4:
        print("Usage: InlineHeader.py <source_header> <root_include_dir> <output_file>", file=sys.stderr)
        sys.exit(1)

    source_file = sys.argv[1]
    root_dir = sys.argv[2]
    output_file = sys.argv[3]

    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(root_dir):
        print(f"Error: Root directory not found: {root_dir}", file=sys.stderr)
        sys.exit(1)

    # Generate guard name from output filename
    guard_name = Path(output_file).stem.upper().replace('.', '_') + '_H'

    try:
        content_lines = inline_includes(source_file, root_dir)

        # Write output with header guards
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f'#ifndef {guard_name}\n')
            f.write(f'#define {guard_name}\n\n')
            f.write('\n'.join(content_lines))
            f.write(f'\n\n#endif // {guard_name}\n')

        print(f'Successfully created {output_file}')

    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
