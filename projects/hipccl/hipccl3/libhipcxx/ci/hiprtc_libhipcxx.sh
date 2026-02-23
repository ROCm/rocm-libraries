#!/bin/bash

# Modifications Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

source "$(dirname "$0")/build_common.sh"

# On RHEL/CentOS-based containers, set LLVM_PATH to help COMGR find the correct clang binary
# This prevents COMGR from defaulting to /bin/clang which causes incorrect GCC detection
if [ -f /etc/redhat-release ] || [ -f /etc/centos-release ]; then
  export LLVM_PATH=/opt/rocm/lib/llvm
fi

print_environment_details


PRESET="libcudacxx-nvrtc-cpp${CXX_STANDARD}"
CMAKE_OPTIONS=""

configure_and_build_preset "libcudacxx NVRTC" "$PRESET" "$CMAKE_OPTIONS"

source "./sccache_stats.sh" "start"
test_preset "libcudacxx NVRTC" "${PRESET}"
source "./sccache_stats.sh" "end"
