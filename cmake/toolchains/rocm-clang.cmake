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
#
# Note: ROCM_PATH should be passed via -D, not set in the environment,
# as environment variables can interfere with toolchain detection.

# First-run validation (only runs once during initial configuration)
if(NOT _ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED)
    # Warn if ROCM_PATH is set in environment (can interfere with toolchain discovery)
    if(DEFINED ENV{ROCM_PATH})
        message(WARNING
            "\nROCM_PATH is set in the environment and may interfere with toolchain detection.\n"
            "Remove ROCM_PATH from the environment and use the following instead:\n"
            "  cmake -DROCM_PATH=$ENV{ROCM_PATH}\n"
        )
    endif()

    # Validate that a compatible generator is being used
    if(CMAKE_GENERATOR)
        string(TOLOWER "${CMAKE_GENERATOR}" _generator_lower)
        if(NOT (_generator_lower MATCHES "ninja" OR _generator_lower MATCHES "makefile"))
            message(WARNING
                "\nIncompatible generator detected: '${CMAKE_GENERATOR}'\n"
                "The ROCm Clang toolchain requires Ninja or Makefile generators.\n"
                "Use \"cmake -G <generator>\" to select a compatible generator.\n"
            )
        endif()
        unset(_generator_lower)
    endif()
endif()

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

# Forward variables to try_compile() so the toolchain file works correctly during compiler checks
if(NOT _ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED)
    set(_ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED TRUE)
    list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
        ROCM_PATH
        _ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED
    )
endif()
