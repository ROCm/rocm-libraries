#!/usr/bin/env python3
"""
Organize generated .bin files into REFERENCE_OUTPUT directory structure

This script moves .bin files from the build directory into the appropriate
subdirectories under REFERENCE_OUTPUT/, creating the directory structure as needed.

Usage:
    python3 organize_reference_outputs.py [--source <build_dir>] [--dest <ref_output_dir>] [--copy]
"""

import os
import shutil
import argparse
import re
from pathlib import Path


def extract_function_name(filename):
    """
    Extract the function name from a .bin filename.

    Examples:
        brightness_u8.bin -> brightness
        resize_f32_interpolationTypeNEAREST_NEIGHBOR.bin -> resize
        noise_u8_noiseTypeSHOT.bin -> noise
        tensor_add_tensor_f32.bin -> tensor_add_tensor
    """
    # Remove .bin extension
    base = filename.replace(".bin", "")

    # Match pattern: {function_name}_{datatype}_{optional_details}
    # Datatypes: u8, f32, f16, i8
    pattern = r"^(.+?)_(u8|f32|f16|i8)(?:_.*)?$"
    match = re.match(pattern, base)

    if match:
        return match.group(1)

    # Fallback: assume everything before first datatype match is the function name
    for datatype in ["_u8_", "_f32_", "_f16_", "_i8_"]:
        if datatype in base:
            return base.split(datatype)[0]

    # If no datatype found, return base name
    print(
        f"Warning: Could not extract function name from '{filename}', using full base name"
    )
    return base


def organize_files(source_dir, dest_dir, copy_mode=False, force_overwrite=False):
    """
    Organize .bin files from source directory into dest directory structure.

    Args:
        source_dir: Directory containing generated .bin files (usually build/)
        dest_dir: Destination directory (usually REFERENCE_OUTPUT/)
        copy_mode: If True, copy files; if False, move files
        force_overwrite: If True, overwrite existing files; if False, skip existing files
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # Find all .bin files in source directory
    bin_files = list(source_path.glob("*.bin"))

    if not bin_files:
        print(f"No .bin files found in {source_dir}")
        return

    print(f"Found {len(bin_files)} .bin files in {source_dir}")
    print(f"{'Copying' if copy_mode else 'Moving'} files to {dest_dir}\n")

    organized_count = 0
    skipped_count = 0

    for bin_file in sorted(bin_files):
        filename = bin_file.name

        # Extract function name
        func_name = extract_function_name(filename)

        # Create subdirectory for this function
        func_dir = dest_path / func_name
        func_dir.mkdir(exist_ok=True)

        # Destination file path
        dest_file = func_dir / filename

        # Check if file already exists (capture before copy/move)
        existed_before = dest_file.exists()

        if existed_before and not force_overwrite:
            print(f"  ⚠️  Skipping {filename} (already exists in {func_name}/)")
            skipped_count += 1
            continue

        # Copy or move the file
        try:
            if copy_mode:
                shutil.copy2(bin_file, dest_file)
                action = "Copied"
            else:
                shutil.move(str(bin_file), str(dest_file))
                action = "Moved"

            if existed_before:
                print(
                    f"  ✓ {action} (overwritten): {filename} → {func_name}/{filename}"
                )
            else:
                print(f"  ✓ {action}: {filename} → {func_name}/{filename}")
            organized_count += 1
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Organized: {organized_count} files")
    print(f"  Skipped:   {skipped_count} files (already exist)")
    print(f"  Total:     {len(bin_files)} files")
    print(f"{'=' * 60}")


def list_organized_structure(dest_dir):
    """Display the organized directory structure."""
    dest_path = Path(dest_dir)

    if not dest_path.exists():
        print(f"Directory {dest_dir} does not exist")
        return

    print(f"\nOrganized structure in {dest_dir}:")
    print("=" * 60)

    # Get all subdirectories
    subdirs = sorted([d for d in dest_path.iterdir() if d.is_dir()])

    for subdir in subdirs:
        bin_files = list(subdir.glob("*.bin"))
        print(f"\n{subdir.name}/ ({len(bin_files)} files)")
        for bin_file in sorted(bin_files)[:5]:  # Show first 5 files
            size_mb = bin_file.stat().st_size / (1024 * 1024)
            print(f"  - {bin_file.name} ({size_mb:.2f} MB)")
        if len(bin_files) > 5:
            print(f"  ... and {len(bin_files) - 5} more files")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Organize .bin reference output files into directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Move files from build/ to ../REFERENCE_OUTPUT/
  python3 organize_reference_outputs.py

  # Copy files (keep originals in build/)
  python3 organize_reference_outputs.py --copy

  # Overwrite existing files
  python3 organize_reference_outputs.py --force

  # Specify custom directories
  python3 organize_reference_outputs.py --source ./build --dest ../REFERENCE_OUTPUT

  # Preview the organized structure
  python3 organize_reference_outputs.py --list-only
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        default="./build",
        help="Source directory containing .bin files (default: ./build)",
    )

    parser.add_argument(
        "--dest",
        type=str,
        default="../REFERENCE_OUTPUT",
        help="Destination REFERENCE_OUTPUT directory (default: ../REFERENCE_OUTPUT)",
    )

    parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of moving them"
    )

    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files in destination"
    )

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list the organized structure, do not move files",
    )

    args = parser.parse_args()

    # Resolve paths
    source_dir = os.path.abspath(args.source)
    dest_dir = os.path.abspath(args.dest)

    if args.list_only:
        list_organized_structure(dest_dir)
    else:
        # Verify source directory exists
        if not os.path.exists(source_dir):
            print(f"Error: Source directory '{source_dir}' does not exist")
            return 1

        # Organize the files
        organize_files(
            source_dir, dest_dir, copy_mode=args.copy, force_overwrite=args.force
        )

        # Show the resulting structure
        list_organized_structure(dest_dir)

    return 0


if __name__ == "__main__":
    exit(main())
