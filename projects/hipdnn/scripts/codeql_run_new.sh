#!/bin/bash -x
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Run CodeQL (C++) over the hipDNN *superbuild* (hipDNN + providers), analyzing
# host code only. CodeQL's C++ extractor cannot ingest AMDGPU device
# compilations (`-x hip --offload-arch=...`), so any CMake target that compiles
# device code is built OUTSIDE the CodeQL tracer and excluded from the database.
# Host-only targets are compiled under the tracer and analyzed.
#
# For the default `hipdnn-providers` preset the whole superbuild is host C++
# (kernels live in the prebuilt MIOpen/hipBLASLt libraries), so nothing is
# skipped. The skip logic matters for presets that pull in device codegen
# (e.g. hipdnn-providers-all / hipdnn-dev-all, which include hip-kernel-provider)
# or when GPU_TARGETS is set.
#
# See --help for the full list of command-line options.

# Defaults (override via the command-line flags parsed below).
CODEQL_VERSION="v2.21.2"
PRESET="hipdnn-providers"
ROCM_PATH_ARG="/opt/rocm"
GPU_TARGETS=""
BUILD_DIR=""   # default set after repo_root is known (alongside the build)
DB_DIR=""      # default set after repo_root is known (alongside the build)

usage() {
    cat <<'USAGE'
Usage: codeql_run_new.sh [options]

Run CodeQL (C++) over the hipDNN superbuild (hipDNN + providers), host code only.

Options:
  --preset <name>         CMake superbuild preset            (default: hipdnn-providers)
  --rocm-path <path>      ROCm install root                  (default: /opt/rocm)
  --gpu-targets <list>    forwarded to CMake when non-empty  (default: none)
  --codeql-version <tag>  CodeQL bundle tag                  (default: v2.21.2)
  --build-dir <path>      build directory                    (default: <repo>/build)
  --db-dir <path>         CodeQL database directory          (default: <build-dir>/codeql_db)
  -h, --help              show this help

CodeQL writes many small "trap" files; when the checkout is on a networked
filesystem (e.g. an NFS home), point --build-dir / --db-dir at a local disk.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --preset)         PRESET="${2:?missing value for --preset}"; shift 2 ;;
        --rocm-path)      ROCM_PATH_ARG="${2:?missing value for --rocm-path}"; shift 2 ;;
        --gpu-targets)    GPU_TARGETS="${2:?missing value for --gpu-targets}"; shift 2 ;;
        --codeql-version) CODEQL_VERSION="${2:?missing value for --codeql-version}"; shift 2 ;;
        --build-dir)      BUILD_DIR="${2:?missing value for --build-dir}"; shift 2 ;;
        --db-dir)         DB_DIR="${2:?missing value for --db-dir}"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *)                echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

CODEQL_BUNDLE_URL="https://github.com/github/codeql-action/releases/download/codeql-bundle-${CODEQL_VERSION}/codeql-bundle-linux64.tar.gz"

# Get the full path of the script
script_path="$(readlink -f "$0")"
# Get the directory containing the script
script_dir="$(dirname "$script_path")"
# The superbuild source root is the rocm-libraries repository root (not
# projects/hipdnn): the preset lives there and pulls in the providers.
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"

# Build and database directories default alongside the superbuild build tree
# (repo build/). CodeQL trap-file writes thrash on NFS; pass --build-dir / --db-dir
# to relocate onto local disk when the checkout is networked.
build_location="${BUILD_DIR:-$repo_root/build}"
db_location="${DB_DIR:-$build_location/codeql_db}"

echo "The full path of the script is: $script_path"
echo "The directory containing the script is: $script_dir"
echo "The superbuild (source) root is: $repo_root"
echo "Preset: $PRESET   ROCM_PATH: $ROCM_PATH_ARG"

codeql_location="$script_dir/codeql-bundle-linux64"

# Download and extract CodeQL if not present
if [ ! -d "$codeql_location" ]; then
    echo "Downloading CodeQL bundle version ${CODEQL_VERSION}..."
    wget -q --show-progress "$CODEQL_BUNDLE_URL"
    mkdir -p "$codeql_location"
    tar zxf codeql-bundle-linux64.tar.gz -C "$codeql_location"
    rm codeql-bundle-linux64.tar.gz
fi

export PATH="$PATH:$codeql_location/codeql"

# Clean up old directories
if [ -d "$db_location" ]; then
    echo "Removing old db folder: $db_location"
    rm -rf "$db_location"
fi

if [ -d "$build_location" ]; then
    echo "Removing old build folder: $build_location"
    rm -rf "$build_location"
fi

echo "Initializing db folder here: $db_location"
echo "Initializing build folder here: $build_location"

# Determine optimal thread count
THREADS=$(nproc)
echo "System has $THREADS cores available"

# For large projects, sometimes using fewer threads for CodeQL is more efficient
# due to memory constraints. Adjust if needed.
CODEQL_THREADS=$THREADS
BUILD_THREADS=$THREADS

mkdir -p "$build_location"

# Configure the superbuild with Ninja. Export the compile database so we can
# classify device- vs host-code targets, and disable clang-tidy (the CodeQL run
# only needs compilation, not linting).
GPU_TARGETS_ARG=()
if [ -n "$GPU_TARGETS" ]; then
    GPU_TARGETS_ARG=(-DGPU_TARGETS="${GPU_TARGETS}")
fi
cmake --preset "$PRESET" -S "$repo_root" -B "$build_location" \
    -DENABLE_CLANG_TIDY=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DROCM_PATH="$ROCM_PATH_ARG" \
    "${GPU_TARGETS_ARG[@]}"

# Classify CMake targets from the compile database. A target is a "device"
# target if ANY of its translation units is compiled with an AMDGPU device
# offload flag; CodeQL cannot extract those, so we exclude the whole target.
readarray -t DEVICE_TARGETS < <(python3 - "$build_location/compile_commands.json" <<'PY'
import json, re, sys
cc = json.load(open(sys.argv[1]))
dev, host = set(), set()
for e in cc:
    cmd = e.get("command") or " ".join(e.get("arguments", []))
    out = e.get("output", "") or e.get("file", "")
    m = re.search(r"CMakeFiles/([^.]+)\.dir/", out)
    if not m:
        continue
    tgt = m.group(1)
    is_dev = ("--offload-arch" in cmd) or (" -x hip" in cmd) or ('"-x", "hip"' in cmd)
    (dev if is_dev else host).add(tgt)
# A target with any device TU is a device target even if it also has host TUs.
for t in sorted(dev):
    print(t)
PY
)

readarray -t HOST_TARGETS < <(python3 - "$build_location/compile_commands.json" <<'PY'
import json, re, sys
cc = json.load(open(sys.argv[1]))
dev, host = set(), set()
for e in cc:
    cmd = e.get("command") or " ".join(e.get("arguments", []))
    out = e.get("output", "") or e.get("file", "")
    m = re.search(r"CMakeFiles/([^.]+)\.dir/", out)
    if not m:
        continue
    tgt = m.group(1)
    is_dev = ("--offload-arch" in cmd) or (" -x hip" in cmd) or ('"-x", "hip"' in cmd)
    (dev if is_dev else host).add(tgt)
for t in sorted(host - dev):
    print(t)
PY
)

echo "Device-code targets to SKIP (${#DEVICE_TARGETS[@]}): ${DEVICE_TARGETS[*]:-<none>}"
echo "Host targets to ANALYZE (${#HOST_TARGETS[@]}): ${HOST_TARGETS[*]:-<all>}"

if [ "${#DEVICE_TARGETS[@]}" -gt 0 ]; then
    # Pre-build the device-code targets OUTSIDE the CodeQL tracer so that
    # host targets link against them without their device TUs being traced.
    echo "Pre-building device-code targets outside CodeQL tracer..."
    cmake --build "$build_location" -j "$BUILD_THREADS" --target "${DEVICE_TARGETS[@]}"
    BUILD_CMD=(cmake --build "$build_location" -j "$BUILD_THREADS" --target "${HOST_TARGETS[@]}")
else
    # No device code in this preset: build everything under the tracer.
    BUILD_CMD=(cmake --build "$build_location" -j "$BUILD_THREADS")
fi

# Create CodeQL database by building the host targets under the tracer.
echo "Creating CodeQL database with $CODEQL_THREADS threads..."
codeql database create "$db_location" \
    --source-root="$repo_root" \
    --language=cpp \
    --threads=$CODEQL_THREADS \
    --ram=16384 \
    --search-path="$codeql_location/codeql" \
    --command="${BUILD_CMD[*]}" \
    --overwrite

# Analyze with explicit thread settings and increased RAM
echo "Analyzing CodeQL database with $CODEQL_THREADS threads..."
codeql database analyze "$db_location" \
    --format=sarifv2.1.0 \
    --output="$build_location/codeql.sarif" \
    --threads=$CODEQL_THREADS \
    --ram=16384 \
    --rerun \
    --compilation-cache="$build_location/.cache/" \
    --no-default-compilation-cache \
    --verbose

echo "CodeQL analysis completed!"
echo "Results saved to: $build_location/codeql.sarif"
