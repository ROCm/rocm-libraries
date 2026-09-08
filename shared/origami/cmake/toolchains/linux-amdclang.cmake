# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# This file intentionally selects one coherent toolchain and SDK installation.
# Do not use CMAKE_PREFIX_PATH to substitute a different ROCm SDK, because that
# can mix incompatible compiler and library artifacts. A cross-compiling
# toolchain should instead set CMAKE_SYSROOT to its complete target filesystem.
set(CMAKE_C_COMPILER   "/opt/rocm/lib/llvm/bin/amdclang"   CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "/opt/rocm/lib/llvm/bin/amdclang++" CACHE FILEPATH "C++/HIP compiler")
set(CMAKE_SYSTEM_PREFIX_PATH "/opt/rocm")
