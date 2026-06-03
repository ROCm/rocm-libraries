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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "OpenCV Benchmark - Complete Setup"
echo "========================================"
echo ""

# Check if dataset exists
if [ ! -d "1080p_128images_dataset" ] || [ -z "$(ls -A 1080p_128images_dataset 2>/dev/null)" ]; then
    echo "Step 1: Generating test dataset..."
    echo "--------------------------------------"
    python3 generate_test_dataset.py
    echo ""
else
    echo "✓ Dataset already exists ($(ls -1 1080p_128images_dataset | wc -l) images)"
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
    echo "Step 2: Installing dependencies..."
    echo "--------------------------------------"
    echo "This requires sudo privileges."
    echo ""

    if [ "$EUID" -ne 0 ]; then
        echo "Please enter your password to install dependencies:"
        sudo apt-get update
        sudo apt-get install -y libopencv-dev libgomp1 cmake build-essential libxlsxwriter-dev
    else
        apt-get update
        apt-get install -y libopencv-dev libgomp1 cmake build-essential libxlsxwriter-dev
    fi
    echo ""
else
    echo "✓ Dependencies already installed"
    if pkg-config --exists opencv4; then
        echo "  OpenCV Version: $(pkg-config --modversion opencv4)"
    else
        echo "  OpenCV Version: $(pkg-config --modversion opencv)"
    fi
    echo "  libxlsxwriter: Installed"
    echo ""
fi

# Build the benchmark
echo "Step 3: Building benchmark..."
echo "--------------------------------------"
rm -rf build
mkdir -p build
cd build
cmake .. -DENABLE_PARALLEL_THREADS=OFF
make -j$(nproc)
cd ..
echo "✓ Build complete"
echo ""

# Run the benchmark
echo "========================================"
echo "Starting Benchmark"
echo "========================================"
echo ""
echo "This will take several minutes..."
echo "Running 100 iterations of 50+ operations on 128 1080p images"
echo ""

./build/opencv_vs_rpp_host_benchmarking

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
