#!/usr/bin/env python3
################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""
Generic YAML Type Fixer for Tensile Configuration Files

This script scans YAML files and fixes type mismatches for specified parameters.
It's designed to be extensible - simply add new parameters to the PARAMETER_TYPES
dictionary to fix additional type mismatches.

Usage:
    python fix_yaml_types.py <directory> [--dry-run] [--verbose]

Examples:
    # Dry run to see what would be changed
    python fix_yaml_types.py ./Logic/asm_full --dry-run

    # Actually fix the files
    python fix_yaml_types.py ./Logic/asm_full

    # Verbose output
    python fix_yaml_types.py ./Logic/asm_full --verbose
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Set, Tuple
from collections import defaultdict


################################################################################
# CONFIGURATION SECTION - EXTEND THIS TO ADD NEW PARAMETERS
################################################################################

# Define the expected types for each parameter
# Format: "ParameterName": expected_python_type
#
# To add a new parameter:
# 1. Add it to this dictionary with its expected type
# 2. Add conversion rules in TYPE_CONVERSION_RULES if needed
PARAMETER_TYPES = {
    # Boolean parameters (should be True/False, not 0/1)
    "TransposeA": bool,
    "TransposeB": bool,
    "UseBeta": bool,
    "UseE": bool,
    "Gradient": bool,
    "UseScaleCD": bool,
    "HighPrecisionAccumulate": bool,
    "SilentHighPrecisionAccumulate": bool,
    "ComplexConjugateA": bool,
    "ComplexConjugateB": bool,
    "StochasticRounding": bool,
    "Batched": bool,
    "StridedBatched": bool,
    "GroupedGemm": bool,
    "UseInitialStridesAB": bool,
    "UseInitialStridesCD": bool,
    "AllowNoFreeDims": bool,
    "Activation": bool,
    "BetaOnlyUseBias": bool,

    # Integer parameters (should be int, not bool)
    "UseBias": int,
    "UseScaleAlphaVec": int,
    "Sparse": int,
    "DataType": int,
    "DataTypeA": int,
    "DataTypeB": int,
    "DataTypeC": int,
    "DataTypeD": int,
    "DataTypeE": int,
    "MacDataTypeA": int,
    "MacDataTypeB": int,
    "DestDataType": int,
    "ComputeDataType": int,
    "F32XdlMathOp": int,
    "ActivationComputeDataType": int,

    # Add more parameters here as needed
    # "YourParameter": bool,  # or int, or str, etc.
}

# Type conversion rules: how to convert from one type to another
# Format: (from_type, to_type): conversion_function
TYPE_CONVERSION_RULES = {
    # Convert int to bool: 0 -> False, any non-zero -> True
    (int, bool): lambda x: bool(x),

    # Convert bool to int: False -> 0, True -> 1
    (bool, int): lambda x: int(x),

    # Convert string to bool
    (str, bool): lambda x: x.lower() in ('true', '1', 'yes', 'on'),

    # Convert string to int
    (str, int): lambda x: int(x),

    # Add more conversion rules here as needed
}

# Parameters to skip (e.g., complex types that need special handling)
SKIP_PARAMETERS = {
    "ISA",  # ISA is a list/tuple, converted separately
    "ProblemType",  # Nested structure
    # Add parameters to skip here
}

################################################################################
# END OF CONFIGURATION SECTION
################################################################################


class YAMLTypeFixer:
    """Fixes type mismatches in YAML files based on parameter type definitions."""

    def __init__(self, parameter_types: Dict[str, type],
                 conversion_rules: Dict[Tuple[type, type], callable],
                 skip_parameters: Set[str],
                 verbose: bool = False):
        """
        Initialize the YAML type fixer.

        Args:
            parameter_types: Dictionary mapping parameter names to expected types
            conversion_rules: Dictionary mapping (from_type, to_type) to conversion functions
            skip_parameters: Set of parameter names to skip
            verbose: Whether to print verbose output
        """
        self.parameter_types = parameter_types
        self.conversion_rules = conversion_rules
        self.skip_parameters = skip_parameters
        self.verbose = verbose

        # Statistics
        self.files_scanned = 0
        self.files_modified = 0
        self.parameters_fixed = defaultdict(int)  # parameter_name -> count
        self.conversion_stats = defaultdict(int)  # (param, from_type, to_type) -> count

    def convert_value(self, param_name: str, value: Any, expected_type: type) -> Tuple[Any, bool]:
        """
        Convert a value to the expected type if needed.

        Args:
            param_name: Name of the parameter
            value: Current value
            expected_type: Expected type

        Returns:
            Tuple of (converted_value, was_changed)
        """
        current_type = type(value)

        # Already correct type
        if current_type == expected_type:
            return value, False

        # Look for conversion rule
        conversion_key = (current_type, expected_type)
        if conversion_key not in self.conversion_rules:
            if self.verbose:
                print(f"  WARNING: No conversion rule for {param_name}: "
                      f"{current_type.__name__} -> {expected_type.__name__}")
            return value, False

        # Apply conversion
        converter = self.conversion_rules[conversion_key]
        try:
            new_value = converter(value)
            self.conversion_stats[(param_name, current_type.__name__, expected_type.__name__)] += 1

            if self.verbose:
                print(f"  Fixed {param_name}: {value} ({current_type.__name__}) -> "
                      f"{new_value} ({expected_type.__name__})")

            return new_value, True
        except Exception as e:
            if self.verbose:
                print(f"  ERROR converting {param_name}: {e}")
            return value, False

    def fix_dict(self, data: Any) -> Tuple[Any, bool]:
        """
        Recursively fix types in a dictionary or list.

        Args:
            data: The data structure to fix (dict, list, or scalar)

        Returns:
            Tuple of (fixed_data, was_changed)
        """
        changed = False

        if isinstance(data, dict):
            fixed_dict = {}
            for key, value in data.items():
                # Check if this parameter needs type fixing
                if key in self.parameter_types and key not in self.skip_parameters:
                    expected_type = self.parameter_types[key]
                    new_value, value_changed = self.convert_value(key, value, expected_type)
                    fixed_dict[key] = new_value
                    if value_changed:
                        changed = True
                        self.parameters_fixed[key] += 1

                # Recursively fix nested structures
                elif isinstance(value, (dict, list)):
                    fixed_value, nested_changed = self.fix_dict(value)
                    fixed_dict[key] = fixed_value
                    if nested_changed:
                        changed = True
                else:
                    fixed_dict[key] = value

            return fixed_dict, changed

        elif isinstance(data, list):
            fixed_list = []
            for item in data:
                if isinstance(item, (dict, list)):
                    fixed_item, item_changed = self.fix_dict(item)
                    fixed_list.append(fixed_item)
                    if item_changed:
                        changed = True
                else:
                    fixed_list.append(item)

            return fixed_list, changed

        else:
            # Scalar value, return as-is
            return data, False

    def fix_yaml_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """
        Fix type mismatches in a single YAML file.

        Args:
            file_path: Path to the YAML file
            dry_run: If True, don't write changes

        Returns:
            True if file was modified, False otherwise
        """
        self.files_scanned += 1

        try:
            # Load YAML file
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)

            if data is None:
                if self.verbose:
                    print(f"Skipping empty file: {file_path}")
                return False

            # Fix types
            fixed_data, changed = self.fix_dict(data)

            if changed:
                self.files_modified += 1

                if not dry_run:
                    # Write back to file
                    with open(file_path, 'w') as f:
                        yaml.dump(fixed_data, f, default_flow_style=False, sort_keys=False)

                    if self.verbose:
                        print(f"✓ Fixed: {file_path}")
                else:
                    if self.verbose:
                        print(f"[DRY RUN] Would fix: {file_path}")

                return True
            else:
                if self.verbose:
                    print(f"  No changes needed: {file_path}")
                return False

        except Exception as e:
            print(f"ERROR processing {file_path}: {e}", file=sys.stderr)
            return False

    def process_directory(self, directory: Path, dry_run: bool = False):
        """
        Process all YAML files in a directory recursively.

        Args:
            directory: Directory to process
            dry_run: If True, don't write changes
        """
        yaml_files = list(directory.rglob("*.yaml")) + list(directory.rglob("*.yml"))

        if not yaml_files:
            print(f"No YAML files found in {directory}")
            return

        print(f"Found {len(yaml_files)} YAML files in {directory}")
        if dry_run:
            print("=" * 70)
            print("DRY RUN MODE - No files will be modified")
            print("=" * 70)

        for yaml_file in yaml_files:
            if self.verbose:
                print(f"\nProcessing: {yaml_file}")
            self.fix_yaml_file(yaml_file, dry_run=dry_run)

    def print_summary(self, dry_run: bool = False):
        """Print a summary of the changes made."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Files scanned: {self.files_scanned}")
        print(f"Files {'that would be ' if dry_run else ''}modified: {self.files_modified}")

        if self.parameters_fixed:
            print(f"\nParameters fixed:")
            for param, count in sorted(self.parameters_fixed.items()):
                print(f"  {param}: {count} occurrences")

        if self.conversion_stats and self.verbose:
            print(f"\nDetailed conversion statistics:")
            for (param, from_type, to_type), count in sorted(self.conversion_stats.items()):
                print(f"  {param}: {from_type} -> {to_type}: {count} times")

        if dry_run and self.files_modified > 0:
            print(f"\nTo apply these changes, run without --dry-run flag")

        print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fix type mismatches in Tensile YAML configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to preview changes
  %(prog)s ./Logic/asm_full --dry-run

  # Fix files
  %(prog)s ./Logic/asm_full

  # Verbose output
  %(prog)s ./Logic/asm_full --verbose

To extend this script:
  1. Edit PARAMETER_TYPES dictionary to add new parameters
  2. Add conversion rules to TYPE_CONVERSION_RULES if needed
  3. Add parameters to SKIP_PARAMETERS if they need special handling
        """
    )

    parser.add_argument(
        "directory",
        type=str,
        nargs='?',  # Make directory optional when using --list-parameters
        help="Directory containing YAML files to fix"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )

    parser.add_argument(
        "--list-parameters",
        action="store_true",
        help="List all configured parameters and exit"
    )

    args = parser.parse_args()

    # List parameters and exit if requested
    if args.list_parameters:
        print("Configured parameters:")
        print("\nBoolean parameters:")
        for param, typ in sorted(PARAMETER_TYPES.items()):
            if typ == bool:
                print(f"  {param}")

        print("\nInteger parameters:")
        for param, typ in sorted(PARAMETER_TYPES.items()):
            if typ == int:
                print(f"  {param}")

        print("\nOther types:")
        for param, typ in sorted(PARAMETER_TYPES.items()):
            if typ not in (bool, int):
                print(f"  {param}: {typ.__name__}")

        print(f"\nTotal: {len(PARAMETER_TYPES)} parameters configured")
        return 0

    # Validate directory (required if not listing parameters)
    if not args.directory:
        print("Error: directory argument is required", file=sys.stderr)
        parser.print_help()
        return 1

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        return 1

    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}", file=sys.stderr)
        return 1

    # Create fixer and process directory
    fixer = YAMLTypeFixer(
        parameter_types=PARAMETER_TYPES,
        conversion_rules=TYPE_CONVERSION_RULES,
        skip_parameters=SKIP_PARAMETERS,
        verbose=args.verbose
    )

    fixer.process_directory(directory, dry_run=args.dry_run)
    fixer.print_summary(dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
