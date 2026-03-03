# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Cross-platform ROCm Clang toolchain for the superbuild.
#
# Linux:   defaults to /opt/rocm, uses amdclang/amdclang++
# Windows: ROCM_PATH must be provided, uses clang/clang++
#
# Usage:
#   Linux:   cmake --preset <preset>
#   Windows: cmake --preset <preset> -DROCM_PATH="C:/AMD/ROCm/7.0"

if(WIN32)
    if(NOT DEFINED ROCM_PATH)
        message(FATAL_ERROR
            "ROCM_PATH must be set for Windows builds.\n"
            "  cmake --preset <preset> -DROCM_PATH=\"C:/AMD/ROCm/7.0\""
        )
    endif()

    set(CMAKE_RC_COMPILER "CMAKE_RC_COMPILER-NOTREQUIRED")
    set(_ROCM_C_COMPILER_NAME "clang.exe")
    set(_ROCM_CXX_COMPILER_NAME "clang++.exe")
else()
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm")
    endif()

    set(_ROCM_C_COMPILER_NAME "amdclang")
    set(_ROCM_CXX_COMPILER_NAME "amdclang++")
endif()

set(ROCM_PATH "${ROCM_PATH}" CACHE PATH "Path to ROCm installation")
set(CMAKE_PREFIX_PATH "${ROCM_PATH}" CACHE PATH "Search path for ROCm packages")

set(ROCM_LLVM_PATH "${ROCM_PATH}/lib/llvm")
set(CMAKE_C_COMPILER "${ROCM_LLVM_PATH}/bin/${_ROCM_C_COMPILER_NAME}" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${ROCM_LLVM_PATH}/bin/${_ROCM_CXX_COMPILER_NAME}" CACHE FILEPATH "C++/HIP compiler")

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Enable position independent code")

unset(_ROCM_C_COMPILER_NAME)
unset(_ROCM_CXX_COMPILER_NAME)
