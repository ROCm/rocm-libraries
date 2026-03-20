# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import sys
import subprocess
import shutil
import os

# Needs to subscribe to HIPDNN_FLATBUFFERS_VERSION CMake variable
REQUIRED_VER = "25.9.23"


def main():
    """Run flatc compiler on FlatBuffers schema files on Linux or Windows."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    schemas_dir = os.path.join(script_dir, "..", "data_sdk", "schemas")

    # Convert version to directory format (e.g., "25.9.23" -> "v25_9_23")
    # Output path matches multi-version structure: include/.../data_objects/v{ver}/hipdnn_data_sdk/data_objects/
    ver_tag = REQUIRED_VER.replace(".", "_")
    output_dir = os.path.join(
        script_dir,
        "..",
        "data_sdk",
        "include",
        "hipdnn_data_sdk",
        "data_objects",
        f"v{ver_tag}",
        "hipdnn_data_sdk",
        "data_objects",
    )

    # Find flatc in PATH and prepare to validate its version
    flatc_path = shutil.which("flatc")
    current_ver = ""

    if flatc_path:
        try:
            current_ver = subprocess.check_output(
                [flatc_path, "--version"], text=True
            ).strip()
        except subprocess.CalledProcessError:
            pass

    if REQUIRED_VER not in current_ver:
        print(
            f'ERROR: flatc version {REQUIRED_VER} required. Found: {current_ver or "None"}',
            file=sys.stderr,
        )
        print(
            "Download the following and include the executable in PATH:",
            file=sys.stderr,
        )
        print(
            f"  Windows: Download https://github.com/google/flatbuffers/releases/download/v{REQUIRED_VER}/Windows.flatc.binary.zip",
            file=sys.stderr,
        )
        print(
            f"  Linux:   wget https://github.com/google/flatbuffers/releases/download/v{REQUIRED_VER}/Linux.flatc.binary.g++-13.zip",
            file=sys.stderr,
        )
        sys.exit(1)

    for f in sys.argv[1:]:
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


if __name__ == "__main__":
    main()
