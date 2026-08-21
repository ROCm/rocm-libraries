# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(CMAKE_C_COMPILER   "/opt/rocm/lib/llvm/bin/amdclang"   CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "/opt/rocm/lib/llvm/bin/amdclang++" CACHE FILEPATH "C++/HIP compiler")
set(CMAKE_SYSTEM_PREFIX_PATH "/opt/rocm")
