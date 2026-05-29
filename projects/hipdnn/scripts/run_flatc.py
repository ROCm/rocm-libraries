# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import sys
import subprocess
import shutil
import os
import re


_HIPDNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_required_version(cmake_file=None):
    """Read HIPDNN_FLATBUFFERS_VERSION from projects/hipdnn/CMakeLists.txt
    so the script stays in sync with the single source of truth.
    """
    if cmake_file is None:
        cmake_file = os.path.join(_HIPDNN_DIR, "CMakeLists.txt")
    pattern = re.compile(
        r'set\s*\(\s*HIPDNN_FLATBUFFERS_VERSION\s+"([^"]+)"', re.IGNORECASE
    )
    with open(cmake_file, encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return match.group(1)
    raise RuntimeError(
        f"Could not find HIPDNN_FLATBUFFERS_VERSION in {cmake_file}. "
        "Update run_flatc.py if the cache variable was renamed or moved."
    )


def _read_flatc_flags(flags_file=None):
    """Read the shared flatc flag list. Single source of truth lives in
    cmake/flatc_flags.txt and is also consumed by FlatBuffersGenerate.cmake.
    Lines starting with '#' and blank lines are ignored. The C++ output mode
    (--cpp) is implied by the consumer and is NOT in the file.
    """
    if flags_file is None:
        flags_file = os.path.join(_HIPDNN_DIR, "cmake", "flatc_flags.txt")
    flags = []
    with open(flags_file, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            flags.append(stripped)
    if not flags:
        raise RuntimeError(
            f"No flatc flags parsed from {flags_file}. The shared flag list "
            "is empty or all lines were filtered out."
        )
    return flags


REQUIRED_VER = _read_required_version()
FLATC_EXTRA_FLAGS = _read_flatc_flags()

# Supported SDKs and their namespace paths for generated output
SDKS = {
    "data_sdk": "hipdnn_data_sdk",
    "flatbuffers_sdk": "hipdnn_flatbuffers_sdk",
}


def detect_sdk(schema_path):
    """Detect which SDK a schema file belongs to based on its path."""
    normalized = schema_path.replace("\\", "/")
    for sdk_dir, namespace in SDKS.items():
        if f"/{sdk_dir}/schemas/" in normalized or normalized.startswith(
            f"{sdk_dir}/schemas/"
        ):
            return sdk_dir, namespace
    return None, None


def main():
    """Run flatc compiler on FlatBuffers schema files on Linux or Windows."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

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
        sdk_dir, namespace = detect_sdk(f)
        if sdk_dir is None:
            print(f"WARNING: Cannot determine SDK for {f}, skipping", file=sys.stderr)
            continue

        schemas_dir = os.path.join(script_dir, "..", sdk_dir, "schemas")

        output_dir = os.path.join(
            script_dir,
            "..",
            sdk_dir,
            "include",
            namespace,
            "data_objects",
        )

        try:
            subprocess.run(
                [
                    flatc_path,
                    "-I",
                    schemas_dir,
                    "--cpp",
                    *FLATC_EXTRA_FLAGS,
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
