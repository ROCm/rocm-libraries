#!/bin/bash

# MIT License
#
# Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Complete setup script for OpenCV benchmark
# This script handles everything from installation to running the benchmark

set -e

# Default values
NUM_THREADS=""
NUM_RUNS=""
CLEAN_BUILD=1  # Default to fresh build

# Parse command-line arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --threads <N>     Number of threads to use (default: auto-detect)"
    echo "  -n, --num-runs <N>    Number of benchmark runs (default: 100)"
    echo "  --no-clean            Skip clean build (use existing build)"
    echo "  -h, --help            Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Fresh build, auto-detect threads, 100 runs (default)"
    echo "  $0 -t 64               # Fresh build with 64 threads"
    echo "  $0 -t 32 -n 50         # Fresh build with 32 threads, 50 runs"
    echo "  $0 --no-clean          # Use existing build without cleaning"
    echo ""
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        -n|--num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --no-clean)
            CLEAN_BUILD=0
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "OpenCV Benchmark - Complete Setup"
echo "========================================"
echo ""
if [ -n "$NUM_THREADS" ] || [ -n "$NUM_RUNS" ]; then
    echo "Configuration:"
    [ -n "$NUM_THREADS" ] && echo "  Threads: $NUM_THREADS"
    [ -n "$NUM_RUNS" ] && echo "  Number of runs: $NUM_RUNS"
    echo ""
fi

# Check if dependencies are installed
DEPS_MISSING=0
if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
    DEPS_MISSING=1
fi
if ! dpkg -l | grep -q libxlsxwriter-dev; then
    DEPS_MISSING=1
fi

if [ $DEPS_MISSING -eq 1 ]; then
    echo "Step 1: Installing system dependencies..."
    echo "--------------------------------------"
    echo "This requires sudo privileges."
    echo ""

    if [ "$EUID" -ne 0 ]; then
        echo "Please enter your password to install dependencies:"
        sudo apt-get update
        sudo apt-get install -y libopencv-dev libgomp1 cmake build-essential libxlsxwriter-dev python3-pip
    else
        apt-get update
        apt-get install -y libopencv-dev libgomp1 cmake build-essential libxlsxwriter-dev python3-pip
    fi
    echo ""
else
    echo "✓ System dependencies already installed"
    if pkg-config --exists opencv4; then
        echo "  OpenCV Version: $(pkg-config --modversion opencv4)"
    else
        echo "  OpenCV Version: $(pkg-config --modversion opencv)"
    fi
    echo "  libxlsxwriter: Installed"
    echo ""
fi

# Install Python dependencies
echo "Step 2: Installing Python dependencies..."
echo "--------------------------------------"
if ! python3 -c "import PIL" 2>/dev/null; then
    pip3 install --user Pillow
else
    echo "✓ Pillow already installed"
fi
echo ""

# Check if dataset exists
if [ ! -d "1080p_128images_dataset" ] || [ -z "$(ls -A 1080p_128images_dataset 2>/dev/null)" ]; then
    echo "Step 3: Generating test dataset..."
    echo "--------------------------------------"
    python3 generate_test_dataset.py
    echo ""
else
    echo "✓ Dataset already exists ($(ls -1 1080p_128images_dataset | wc -l) images)"
    echo ""
fi

# Build benchmark
echo "Step 4: Building benchmark..."
echo "--------------------------------------"
if [ $CLEAN_BUILD -eq 1 ]; then
    echo "Performing fresh build (cleaning old build artifacts)..."
    rm -rf build
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    cd ..
    echo "✓ Fresh build complete"
else
    echo "Using existing build (incremental build)..."
    mkdir -p build
    cd build
    if [ ! -f Makefile ]; then
        echo "No existing build found, running cmake..."
        cmake ..
    fi
    make -j$(nproc)
    cd ..
    echo "✓ Incremental build complete"
fi
echo ""

# Build command with optional arguments
BUILD_DIR="build"
cmd="./${BUILD_DIR}/opencv_vs_rpp_host_benchmarking"
cmd_args=""

if [ -n "$NUM_THREADS" ]; then
    cmd_args="$cmd_args --threads $NUM_THREADS"
fi

if [ -n "$NUM_RUNS" ]; then
    cmd_args="$cmd_args --num-runs $NUM_RUNS"
fi

runs_text="${NUM_RUNS:-100}"
threads_text="${NUM_THREADS:-auto-detect}"

echo "========================================"
echo "Starting Benchmark"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Threads: $threads_text"
echo "  Runs: $runs_text"
echo ""
echo "This will take several minutes..."
echo "Running $runs_text iterations of 50+ operations on 128 1080p images"
echo ""

$cmd $cmd_args

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo ""
