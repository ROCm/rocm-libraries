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
endif()

# Common compiler configuration
set(DEFAULT_ROCM_LLVM_BIN_SUFFIX "/lib/llvm/bin")

if(NOT DEFINED _ROCM_CLANG_TOOLCHAIN_FIRST_RUN)
    if(DEFINED ROCM_CMAKE_PATH)
        message(STATUS "ROCM_CMAKE_PATH provided: ${ROCM_CMAKE_PATH}")
    elseif(DEFINED ROCM_PATH)
        message(STATUS "ROCM_PATH provided: ${ROCM_PATH}")
    else()
        message(
            STATUS
                "Neither ROCM_CMAKE_PATH nor ROCM_PATH are defined; attempting to detect ROCm path using hipconfig."
        )

        # Try to detect ROCm path using hipconfig --rocmpath
        find_program(HIPCONFIG_EXECUTABLE hipconfig)
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
        else()
            message(
                STATUS
                    "hipconfig not found in PATH; relying on system PATH to locate ROCm toolchain."
            )

            # Check if clang++ compiler will be found
            find_program(CLANGPP_EXECUTABLE clang++)
            if(CLANGPP_EXECUTABLE)
                message(STATUS "hipconfig not found in PATH but found AMD/ROCm clang++ compiler.")

                # Warn if CMAKE_PREFIX_PATH is empty
                if(NOT CMAKE_PREFIX_PATH)
                    message(
                        WARNING
                            "When hipconfig is not available, CMAKE_PREFIX_PATH must be set to the ROCm install folder to locate hip package."
                    )
                endif()
            endif()
        endif()
    endif()
endif()

# Prioritize the C/C++ compiler search to ROCm compiler names: clang/clang++.
set(CMAKE_CXX_COMPILER_NAMES clang++)
set(CMAKE_C_COMPILER_NAMES clang)

# Warn if ROCM_PATH is set in environment (can interfere with toolcain discovery).
if(DEFINED ENV{ROCM_PATH})
    message(
        WARNING "\nROCM_PATH is set in the environment and may interfere with toolchain detection. "
                "Remove ROCM_PATH from the environment and use the following instead:\n"
                "  cmake -DROCM_PATH=$ENV{ROCM_PATH}\n\n"
    )
endif()

if(DEFINED ROCM_CMAKE_PATH)
    set(ROCM_BIN_DIR "${ROCM_CMAKE_PATH}/bin")
    if(EXISTS "${ROCM_BIN_DIR}")
        # Check if ROCM_BIN_DIR is already in PATH
        string(FIND "$ENV{PATH}" "${ROCM_BIN_DIR}" _rocm_bin_in_path)
        if(_rocm_bin_in_path EQUAL -1)
            # Not in PATH, so add it
            if(WIN32)
                set(ENV{PATH} "${ROCM_BIN_DIR};$ENV{PATH}")
            else()
                set(ENV{PATH} "${ROCM_BIN_DIR}:$ENV{PATH}")
            endif()
            message(STATUS "Added ${ROCM_BIN_DIR} to PATH for ROCm tools")
        elseif(NOT DEFINED _ROCM_CLANG_TOOLCHAIN_FIRST_RUN)
            message(STATUS "ROCm bin directory already in PATH: ${ROCM_BIN_DIR}")
        endif()
        unset(_rocm_bin_in_path)
    else()
        message(FATAL_ERROR "ROCm bin directory does not exist: ${ROCM_BIN_DIR}")
    endif()
    set(ROCM_LLVM_BIN_DIR "${ROCM_CMAKE_PATH}/lib/llvm/bin")
    if(EXISTS "${ROCM_LLVM_BIN_DIR}")
        # Check if ROCM_BIN_DIR is already in PATH
        string(FIND "$ENV{PATH}" "${ROCM_LLVM_BIN_DIR}" _rocm_bin_in_path)
        if(_rocm_bin_in_path EQUAL -1)
            # Not in PATH, so add it
            if(WIN32)
                set(ENV{PATH} "${ROCM_LLVM_BIN_DIR};$ENV{PATH}")
            else()
                set(ENV{PATH} "${ROCM_LLVM_BIN_DIR}:$ENV{PATH}")
            endif()
            message(STATUS "Added ${ROCM_LLVM_BIN_DIR} to PATH for ROCm LLVM tools")
        elseif(NOT DEFINED _ROCM_CLANG_TOOLCHAIN_FIRST_RUN)
            message(STATUS "ROCm LLVM bin directory already in PATH: ${ROCM_LLVM_BIN_DIR}")
        endif()
        unset(_rocm_bin_in_path)
    else()
        message(FATAL_ERROR "ROCm LLVM bin directory does not exist: ${ROCM_LLVM_BIN_DIR}")
    endif()
endif()

# If ROCM_PATH is provided, explicitly set compilers (bypasses toolchain auto-discovery).
if(DEFINED ROCM_PATH)
    set(ROCM_LLVM_BIN_DIR ${ROCM_PATH}${DEFAULT_ROCM_LLVM_BIN_SUFFIX})

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
    # so that find_package(hip) works. TODO: check if ROCM_PATH is already in system path before
    # adding to CMAKE_PREFIX_PATH.
    if(NOT "${ROCM_PATH}" IN_LIST CMAKE_PREFIX_PATH)
        list(PREPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
        message(STATUS "Added ${ROCM_PATH} to CMAKE_PREFIX_PATH for finding HIP package")
    else()
        message(STATUS "ROCM_PATH already in CMAKE_PREFIX_PATH")
    endif()
endif()

set(_ROCM_CLANG_TOOLCHAIN_FIRST_RUN TRUE)
