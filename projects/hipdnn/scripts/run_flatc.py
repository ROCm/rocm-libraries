# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import sys
import subprocess
import shutil
import os
import argparse

# Supported FlatBuffers versions - must match HIPDNN_SUPPORTED_FLATBUFFERS_VERSIONS in CMakeLists.txt
SUPPORTED_VERSIONS = ["24.12.23", "25.9.23"]
DEFAULT_VERSION = "25.9.23"


def get_version_dir(version):
    """Convert version string to directory name (e.g., '25.9.23' -> 'v25_9_23')."""
    return "v" + version.replace(".", "_")


def find_and_validate_flatc(required_version):
    """Find flatc in PATH and validate its version."""
    flatc_path = shutil.which("flatc")
    current_ver = ""

    if flatc_path:
        try:
            current_ver = subprocess.check_output(
                [flatc_path, "--version"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            pass

    if required_version not in current_ver:
        print(
            f"ERROR: flatc version {required_version} required. Found: {current_ver or 'None'}",
            file=sys.stderr,
        )
        print(
            "Download the following and include the executable in PATH:",
            file=sys.stderr,
        )
        print(
            f"  Windows: Download https://github.com/google/flatbuffers/releases/download/v{required_version}/Windows.flatc.binary.zip",
            file=sys.stderr,
        )
        print(
            f"  Linux:   wget https://github.com/google/flatbuffers/releases/download/v{required_version}/Linux.flatc.binary.g++-13.zip",
            file=sys.stderr,
        )
        sys.exit(1)

    return flatc_path


def compile_schemas(flatc_path, schemas_dir, output_dir, schema_files):
    """Compile schema files using flatc."""
    os.makedirs(output_dir, exist_ok=True)

    for f in schema_files:
        try:
            subprocess.run(
                [
                    flatc_path,
                    "-I",
                    schemas_dir,
                    "--cpp",
                    "--gen-object-api",
                    "--gen-mutable",
                    "--gen-compare",
                    "--defaults-json",
                    "--scoped-enums",
                    "-o",
                    output_dir,
                    f,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to compile {f}", file=sys.stderr)
            print("STDOUT:", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
            print("STDERR:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            sys.exit(1)


def main():
    """Run flatc compiler on FlatBuffers schema files on Linux or Windows."""
    parser = argparse.ArgumentParser(
        description="Run flatc on FlatBuffers schema files to generate C++ headers."
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        choices=SUPPORTED_VERSIONS,
        help=f"FlatBuffers version to generate for (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "schemas",
        nargs="+",
        help="Schema files (.fbs) to compile",
    )
    args = parser.parse_args()

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    schemas_dir = os.path.join(script_dir, "..", "data_sdk", "schemas")

    # Output to versioned directory
    version_dir = get_version_dir(args.version)
    output_dir = os.path.join(
        script_dir,
        "..",
        "data_sdk",
        "include",
        "generated",
        version_dir,
        "hipdnn_data_sdk",
        "data_objects",
    )

    flatc_path = find_and_validate_flatc(args.version)

    print(f"Generating FlatBuffer headers for version {args.version}")
    print(f"Output directory: {output_dir}")

    compile_schemas(flatc_path, schemas_dir, output_dir, args.schemas)

    print(f"Successfully generated {len(args.schemas)} header(s)")


if __name__ == "__main__":
    main()
