# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# The ROCm toolchain can be discovered by cmake if the ROCm bin/hipconfig and lib/llvm/bin/clang++
# programs are available in your system PATH.
#
# If the path does *not* contain those programs, the ROCM_CMAKE_PATH CMake variable can be set to
# the ROCM root folder to specify where CMake should look for the toolchains. E.g.:
#
# * Linux: cmake --preset release -DROCM_CMAKE_PATH="/opt/rocm"
# * Windows: cmake --preset release -DROCM_CMAKE_PATH="C:/AMD/ROCm/7.0"
#
# When ROCM_CMAKE_PATH is provided, the path will be updated to include the following folders during
# the toolchain discovery:
#
# * $ROCM_CMAKE_PATH/bin
# * $ROCM_CMAKE_PATH/lib/llvm/bin
#
# The above folders must be present on your system so that the toolchain system inspection can
# locate the following files:
#
# * $ROCM_CMAKE_PATH/bin/hipconfig
# * $ROCM_CMAKE_PATH/lib/llvm/bin/clang++
#
# ** To skip automatic detection** and force cmake to use hard-coded compiler names from a ROCm
# install, set ROCM_PATH instead of ROCM_CMAKE_PATH.
#
# DO NOT SET ROCM_PATH IN YOUR ENVIRONMENT. Setting ROCM_PATH in the environment will cause the
# compiler check to fail. Instead, use the -D option to cmake. E.g.:
#
# * Linux: cmake --preset release -DROCM_PATH="/opt/rocm"
# * Windows: cmake --preset release -DROCM_PATH="C:/AMD/ROCm/7.0"
#
# The CXX and HIP compilers will be set as $ROCM_PATH/lib/llvm/bin/clang++.

# Platform-specific compiler configuration
if(WIN32)
    set(DEFAULT_ROCM_COMPILER_EXTENSION ".exe")
    set(CMAKE_RC_COMPILER "CMAKE_RC_COMPILER-NOTREQUIRED")
    # No suitable default on windows, use this as a possible example.
    set(DEFAULT_ROCM_CMAKE_PATH "c:/dist/therock")
else()
    # Can use /opt/rocm as the typical default install path on Linux
    set(DEFAULT_ROCM_CMAKE_PATH "/opt/rocm")
endif()

# Common compiler configuration
set(DEFAULT_ROCM_LLVM_BIN_SUFFIX "/lib/llvm/bin")

message(
    VERBOSE
    "hipDNN-ClangToolChain start: CXX=${CMAKE_CXX_COMPILER} TOOLCHAIN_COMPLETED=${_ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED} ROCM_PATH=${ROCM_PATH} ROCM_CMAKE_PATH=${ROCM_CMAKE_PATH} ROCM_CMAKE_LLVM_BIN_PATH=${ROCM_CMAKE_LLVM_BIN_PATH}"
)

if(NOT _ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED)
    if(DEFINED ROCM_PATH)
        message(STATUS "ROCM_PATH provided: ${ROCM_PATH}")
    elseif(DEFINED ROCM_CMAKE_HIPCONFIG_PATH)
        set(TRY_TO_FIND_ROCM_PATH_USING_HIPCONFIG "TRUE")
        message(STATUS "ROCM_CMAKE_HIPCONFIG_PATH provided: ${ROCM_CMAKE_HIPCONFIG_PATH}")
    elseif(DEFINED ROCM_CMAKE_PATH)
        message(STATUS "ROCM_CMAKE_PATH provided: ${ROCM_CMAKE_PATH}")
    elseif(DEFINED ENV{ROCM_CMAKE_PATH})
        set(ROCM_CMAKE_PATH $ENV{ROCM_CMAKE_PATH})
        message(STATUS "ROCM_CMAKE_PATH set from environment: ${ROCM_CMAKE_PATH}")
    else()
        set(TRY_TO_FIND_ROCM_PATH_USING_HIPCONFIG "TRUE")
        message(
            STATUS
                "No helper variables provided; attempting to detect ROCm path and hip package using system PATH and CMAKE_PREFIX_PATH."
        )
    endif()
endif()

# Prioritize the C/C++ compiler search to ROCm compiler names: clang/clang++.
set(CMAKE_CXX_COMPILER_NAMES clang++)
set(CMAKE_C_COMPILER_NAMES clang)

# Warn if ROCM_PATH is set in environment (can interfere with toolchain discovery).
if(DEFINED ENV{ROCM_PATH})
    message(
        WARNING "\nROCM_PATH is set in the environment and may interfere with toolchain detection. "
                "Remove ROCM_PATH from the environment and use the following instead:\n"
                "  cmake -DROCM_PATH=$ENV{ROCM_PATH}\n\n"
    )
endif()

if(NOT DEFINED ROCM_PATH AND NOT DEFINED ROCM_CMAKE_PATH AND TRY_TO_FIND_ROCM_PATH_USING_HIPCONFIG)

    find_program(
        HIPCONFIG_EXECUTABLE hipconfig PATHS ${ROCM_CMAKE_HIPCONFIG_PATH} ${ROCM_CMAKE_PATH}
                                             ${DEFAULT_ROCM_CMAKE_PATH} PATH_SUFFIXES "/bin"
    )
    if(HIPCONFIG_EXECUTABLE)
        message(VERBOSE "Found hipconfig: ${HIPCONFIG_EXECUTABLE}")

        # Try to detect ROCm path using hipconfig --rocmpath
        if(HIPCONFIG_EXECUTABLE)
            execute_process(
                COMMAND ${HIPCONFIG_EXECUTABLE} --rocmpath
                OUTPUT_VARIABLE DETECTED_ROCM_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE HIPCONFIG_RESULT
                ERROR_QUIET
            )
            if(HIPCONFIG_RESULT EQUAL 0 AND DETECTED_ROCM_PATH)
                set(ROCM_CMAKE_PATH "${DETECTED_ROCM_PATH}")
                message(
                    STATUS
                        "Automatically detected ROCM_CMAKE_PATH using hipconfig: ${ROCM_CMAKE_PATH}"
                )
            else()
                message(
                    STATUS
                        "hipconfig found but failed to detect ROCm path; relying on system PATH to locate ROCm toolchain."
                )
            endif()
            execute_process(
                COMMAND ${HIPCONFIG_EXECUTABLE} --hipclangpath
                OUTPUT_VARIABLE DETECTED_HIP_CLANG_PATH OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE HIPCONFIG_RESULT ERROR_QUIET
            )
            if(HIPCONFIG_RESULT EQUAL 0 AND DETECTED_HIP_CLANG_PATH)
                set(ROCM_CMAKE_LLVM_BIN_PATH "${DETECTED_HIP_CLANG_PATH}")
                message(
                    STATUS
                        "Automatically detected ROCM_CMAKE_LLVM_BIN_PATH using hipconfig: ${ROCM_CMAKE_LLVM_BIN_PATH}"
                )
            else()
                message(
                    STATUS
                        "hipconfig found but failed to detect ROCm clang path; relying on system PATH to locate ROCm toolchain."
                )
            endif()
        endif()
    else()
        message(
            STATUS
                "hipconfig not found in PATH; relying on defaults, system PATH, and CMAKE_PREFIX_PATH to locate ROCm toolchain and HIP package."
        )
    endif()
endif()

if(DEFINED ROCM_CMAKE_PATH)
    file(TO_NATIVE_PATH "${ROCM_CMAKE_PATH}/bin" ROCM_BIN_DIR)
    if(EXISTS "${ROCM_BIN_DIR}")
        # Check if ROCM_BIN_DIR is already in PATH Normalize both paths to ensure consistent
        # delimiter comparison
        file(TO_NATIVE_PATH "$ENV{PATH}" _normalized_env_path)
        string(FIND "${_normalized_env_path}" "${ROCM_BIN_DIR}" _rocm_bin_in_path)
        if(_rocm_bin_in_path EQUAL -1)
            # Not in PATH, so add it
            if(WIN32)
                set(ENV{PATH} "${ROCM_BIN_DIR};$ENV{PATH}")
            else()
                set(ENV{PATH} "${ROCM_BIN_DIR}:$ENV{PATH}")
            endif()
            message(STATUS "Added ${ROCM_BIN_DIR} to system PATH for ROCm tool discovery")
        else()
            message(VERBOSE "ROCm bin directory already in system PATH: ${ROCM_BIN_DIR}")
        endif()
        unset(_rocm_bin_in_path)
        unset(_normalized_env_path)
    else()
        message(FATAL_ERROR "ROCm bin directory does not exist: ${ROCM_BIN_DIR}")
    endif()
    if(ROCM_CMAKE_LLVM_BIN_PATH)
        file(TO_NATIVE_PATH ${ROCM_CMAKE_LLVM_BIN_PATH} ROCM_LLVM_BIN_DIR)
    else()
        file(TO_NATIVE_PATH "${ROCM_CMAKE_PATH}/lib/llvm/bin" ROCM_LLVM_BIN_DIR)
    endif()
    if(EXISTS "${ROCM_LLVM_BIN_DIR}")
        # Check if ROCM_LLVM_BIN_DIR is already in PATH Normalize both paths to ensure consistent
        # delimiter comparison
        file(TO_NATIVE_PATH "$ENV{PATH}" _normalized_env_path)
        string(FIND "${_normalized_env_path}" "${ROCM_LLVM_BIN_DIR}" _rocm_bin_in_path)
        if(_rocm_bin_in_path EQUAL -1)
            # Not in PATH, so add it
            if(WIN32)
                set(ENV{PATH} "${ROCM_LLVM_BIN_DIR};$ENV{PATH}")
            else()
                set(ENV{PATH} "${ROCM_LLVM_BIN_DIR}:$ENV{PATH}")
            endif()
            message(STATUS "Added ${ROCM_LLVM_BIN_DIR} to system PATH for ROCm LLVM tool discovery")
        else()
            message(VERBOSE "ROCm LLVM bin directory already in system PATH: ${ROCM_LLVM_BIN_DIR}")
        endif()
        unset(_rocm_bin_in_path)
        unset(_normalized_env_path)
    else()
        message(FATAL_ERROR "ROCm LLVM bin directory does not exist: ${ROCM_LLVM_BIN_DIR}")
    endif()
endif()

# If ROCM_PATH is provided, explicitly set compilers (bypasses toolchain auto-discovery).
#
# Only use ROCM_PATH to hard-code the compiler if ROCM_CMAKE_PATH is not defined to ensure
# consistent behaviour when ROCM_PATH is set by an included package but the usr provided only
# ROCM_CMAKE_PATH.
if(DEFINED ROCM_PATH AND NOT DEFINED ROCM_CMAKE_PATH)
    file(TO_NATIVE_PATH "${ROCM_PATH}${DEFAULT_ROCM_LLVM_BIN_SUFFIX}" ROCM_LLVM_BIN_DIR)

    if(EXISTS ${ROCM_LLVM_BIN_DIR})
        set(CMAKE_C_COMPILER ${ROCM_LLVM_BIN_DIR}/clang${DEFAULT_ROCM_COMPILER_EXTENSION})
        set(CMAKE_CXX_COMPILER ${ROCM_LLVM_BIN_DIR}/clang++${DEFAULT_ROCM_COMPILER_EXTENSION})
        message(STATUS "Using ROCm Clang compilers from ${ROCM_LLVM_BIN_DIR}")
    else()
        message(
            FATAL_ERROR
                "The directory ${ROCM_LLVM_BIN_DIR} does not exist. Cannot set ROCm Clang compilers."
        )
    endif()

    # In case the toolchain is not in the system path, add the ROCm folder to the CMAKE_PREFIX_PATH
    # so that find_package(hip) works.
    if(NOT "${ROCM_PATH}" IN_LIST CMAKE_PREFIX_PATH)
        list(PREPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
        message(STATUS "Added ${ROCM_PATH} to CMAKE_PREFIX_PATH for finding HIP package")
    else()
        message(VERBOSE "ROCM_PATH already in CMAKE_PREFIX_PATH")
    endif()
endif()

if(NOT _ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED)
    # Validate that a compatible generator is being used
    if(CMAKE_GENERATOR)
        string(TOLOWER "${CMAKE_GENERATOR}" _generator_lower)
        if(NOT (_generator_lower MATCHES "ninja" OR _generator_lower MATCHES "makefile"))
            message(WARNING "\nIncompatible generator detected: '${CMAKE_GENERATOR}'\n"
                            "The ROCm Clang toolchain requires Ninja or Makefile generators.\n"
                            "Use \"cmake -G <generator>\" to select a compatible generator.\n"
            )
        endif()
        unset(_generator_lower)
    endif()
endif()

message(
    VERBOSE
    "hipDNN-ClangToolChain stop : CXX=${CMAKE_CXX_COMPILER} TOOLCHAIN_COMPLETED=${_ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED} ROCM_PATH=${ROCM_PATH} ROCM_CMAKE_PATH=${ROCM_CMAKE_PATH} ROCM_CMAKE_LLVM_BIN_PATH=${ROCM_CMAKE_LLVM_BIN_PATH}"
)

if(NOT _ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED)
    set(_ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED TRUE)
    # Forward variables to try_compile() so the toolchain file works correctly during compiler
    # checks
    list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES ROCM_CMAKE_PATH ROCM_CMAKE_LLVM_BIN_PATH
         _ROCM_CLANG_TOOLCHAIN_FIRST_RUN_COMPLETED
    )
    # Only forward ROCM_PATH to try_compile() if ROCM_CMAKE_PATH is not defined to ensure consistent
    # behaviour when ROCM_PATH is set by an included package but the usr provided only
    # ROCM_CMAKE_PATH.
    if(DEFINED ROCM_PATH AND NOT DEFINED ROCM_CMAKE_PATH)
        list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES ROCM_PATH)
    endif()
endif()
