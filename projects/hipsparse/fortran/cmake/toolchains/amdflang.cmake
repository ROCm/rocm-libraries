# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# AMD ROCm toolchain (amdflang). The recommended default: amdflang is the LLVM-based Fortran compiler shipped with ROCm.
# Usage, from projects/hipsparse/fortran:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/amdflang.cmake

set(CMAKE_Fortran_COMPILER amdflang CACHE FILEPATH "Fortran compiler")
set(CMAKE_C_COMPILER       amdclang CACHE FILEPATH "C compiler")
