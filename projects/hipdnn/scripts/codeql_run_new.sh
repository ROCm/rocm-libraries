# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

#!/bin/bash -x
set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Configuration
CODEQL_VERSION="${CODEQL_VERSION:-v2.21.2}"
CODEQL_BUNDLE_URL="https://github.com/github/codeql-action/releases/download/codeql-bundle-${CODEQL_VERSION}/codeql-bundle-linux64.tar.gz"

# Get the full path of the script
script_path="$(readlink -f "$0")"
# Get the directory containing the script
script_dir="$(dirname "$script_path")"
repo_root="$(dirname "$script_dir")"

echo "The full path of the script is: $script_path"
echo "The directory containing the script is: $script_dir"
echo "The repo root is $repo_root"

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

db_location="$script_dir/hipdnn_codeql_db"
build_location="$script_dir/build"

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

# Configure with Ninja using all cores
cmake -G Ninja -S "$repo_root" -B "$build_location" \
    -DENABLE_CLANG_TIDY=OFF

# Create CodeQL database with explicit thread settings
echo "Creating CodeQL database with $CODEQL_THREADS threads..."
codeql database create "$db_location" \
    --source-root="$repo_root" \
    --language=cpp \
    --threads=$CODEQL_THREADS \
    --ram=16384 \
    --search-path="$codeql_location/codeql" \
    --command="ninja -j$BUILD_THREADS" \
    --working-dir="$build_location" \
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
