# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Cross-platform ROCm Clang toolchain for the superbuild.
#
# Linux:   defaults to /opt/rocm, uses clang/clang++
# Windows: ROCM_PATH must be provided, uses clang/clang++
#
# Usage:
#   Linux:   cmake --preset <preset>
#   Linux:   cmake --preset <preset> -DROCM_PATH="/opt/rocm"
#   Windows: cmake --preset <preset> -DROCM_PATH="C:/AMD/ROCm/6.4"
#
# Note: ROCM_PATH should be passed via -D, not set in the environment,
# as environment variables can interfere with toolchain detection.

# Platform-specific configuration
if(WIN32)
    set(_ROCM_COMPILER_EXTENSION ".exe")
    set(CMAKE_RC_COMPILER "CMAKE_RC_COMPILER-NOTREQUIRED")
else()
    set(_ROCM_COMPILER_EXTENSION "")
endif()

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
            message(FATAL_ERROR
                "\nIncompatible generator detected: '${CMAKE_GENERATOR}'\n"
                "The ROCm Clang toolchain requires Ninja or Makefile generators.\n"
                "Use \"cmake -G <generator>\" to select a compatible generator.\n"
            )
        endif()
        unset(_generator_lower)
    endif()
endif()

# Set ROCM_PATH with platform-specific defaults
if(WIN32)
    if(NOT DEFINED ROCM_PATH)
        message(FATAL_ERROR
            "ROCM_PATH must be set for Windows builds.\n"
            "  cmake --preset <preset> -DROCM_PATH=\"C:/AMD/ROCm/6.4\""
        )
    endif()
else()
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()

# Set compiler paths
set(ROCM_LLVM_PATH "${ROCM_PATH}/lib/llvm")
file(TO_NATIVE_PATH "${ROCM_LLVM_PATH}/bin" ROCM_LLVM_BIN_DIR)

if(NOT EXISTS "${ROCM_LLVM_BIN_DIR}")
    message(FATAL_ERROR "ROCm LLVM bin directory does not exist: ${ROCM_LLVM_BIN_DIR}")
endif()

set(CMAKE_C_COMPILER "${ROCM_LLVM_BIN_DIR}/clang${_ROCM_COMPILER_EXTENSION}" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${ROCM_LLVM_BIN_DIR}/clang++${_ROCM_COMPILER_EXTENSION}" CACHE FILEPATH "C++/HIP compiler")

# Cache ROCM_PATH and add to CMAKE_PREFIX_PATH for find_package(hip)
set(ROCM_PATH "${ROCM_PATH}" CACHE PATH "Path to ROCm installation")
if(NOT "${ROCM_PATH}" IN_LIST CMAKE_PREFIX_PATH)
    list(PREPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Enable position independent code")

# Cleanup
unset(_ROCM_COMPILER_EXTENSION)

# Forward variables to try_compile() so the toolchain file works correctly during compiler checks
if(NOT _ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED)
    set(_ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED TRUE)
    list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
        ROCM_PATH
        _ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED
    )
endif()
