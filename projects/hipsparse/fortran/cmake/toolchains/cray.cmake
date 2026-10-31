# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Cray toolchain. ftn and cc are the Cray compiler wrappers, which select the underlying compiler from the loaded PrgEnv module.
# Usage, from projects/hipsparse/fortran:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/cray.cmake

set(CMAKE_Fortran_COMPILER ftn CACHE FILEPATH "Fortran compiler")
set(CMAKE_C_COMPILER       cc CACHE FILEPATH "C compiler")
