# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Platform-specific compiler configuration

if(UNIX)
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
    endif()

    # Unix/Linux: Use ROCm LLVM Clang
    set(ROCM_LLVM_BIN_DIR ${ROCM_PATH}/llvm/bin)
    set(ROCM_LLVM_LIB_DIR ${ROCM_PATH}/llvm/lib)

    if(EXISTS ${ROCM_LLVM_BIN_DIR})
        # Set the C and C++ compilers to clang and clang++ with a specific directory hint
        set(CMAKE_C_COMPILER ${ROCM_LLVM_BIN_DIR}/clang)
        set(CMAKE_CXX_COMPILER ${ROCM_LLVM_BIN_DIR}/clang++)
                
        message(STATUS "Using ROCm Clang compilers from ${ROCM_LLVM_BIN_DIR}")
    else()
        message(FATAL_ERROR "The directory ${ROCM_LLVM_BIN_DIR} does not exist. Cannot auto select clang compilers.")
    endif()

elseif(WIN32)
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "C:/dist/therock" CACHE PATH "Path to ROCm installation")
    endif()

    set(ROCM_LLVM_BIN_DIR ${ROCM_PATH}/lib/llvm/bin)
    set(ROCM_LLVM_LIB_DIR ${ROCM_PATH}/lib/llvm/lib)

    if(EXISTS ${ROCM_LLVM_BIN_DIR})
        set(CMAKE_C_COMPILER ${ROCM_LLVM_BIN_DIR}/clang.exe)
        set(CMAKE_CXX_COMPILER ${ROCM_LLVM_BIN_DIR}/clang++.exe)
        message(STATUS "Using ROCm Clang compilers from ${ROCM_LLVM_BIN_DIR}")
    else()
        message(FATAL_ERROR "The directory ${ROCM_LLVM_BIN_DIR} does not exist. Cannot auto select clang compilers.")
    endif()

    set(CMAKE_RC_COMPILER rc.exe)
endif()
