# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Platform-specific compiler configuration

include(${CMAKE_CURRENT_LIST_DIR}/CheckToolVersion.cmake)

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
        
        findAndCheckLlvmSymbolizer()
        
        message(STATUS "Using ROCm Clang compilers from ${ROCM_LLVM_BIN_DIR}")
    else()
        message(FATAL_ERROR "The directory ${ROCM_LLVM_BIN_DIR} does not exist. Cannot auto select clang compilers.")
    endif()

elseif(WIN32)
    # Windows: Use Clang from TheRock build
    set(WINDOWS_ROCM_DIR "C:/src/TheRock/build/dist/rocm" CACHE PATH "Path to Windows ROCm installation")
    set(WINDOWS_ROCM_LLVM_BIN_DIR "${WINDOWS_ROCM_DIR}/lib/llvm/bin")
    set(WINDOWS_ROCM_CMAKE_DIR "${WINDOWS_ROCM_DIR}/lib/cmake")
    set(CMAKE_SYSTEM_VERSION "10.0.22621.0")
    set(CMAKE_RC_COMPILER rc.exe)

    # Enable exporting all symbols for Windows
    # set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

    if(EXISTS ${WINDOWS_ROCM_LLVM_BIN_DIR})
        # Set the C and C++ compilers to clang and clang++ for Windows
        set(CMAKE_C_COMPILER ${WINDOWS_ROCM_LLVM_BIN_DIR}/clang.exe)
        set(CMAKE_CXX_COMPILER ${WINDOWS_ROCM_LLVM_BIN_DIR}/clang++.exe)
        set(CMAKE_HIP_COMPILER ${WINDOWS_ROCM_LLVM_BIN_DIR}/clang++.exe)
        message(STATUS "Using Windows ROCm Clang compilers from ${WINDOWS_ROCM_LLVM_BIN_DIR}")
    else()
        message(FATAL_ERROR "The directory ${WINDOWS_ROCM_LLVM_BIN_DIR} does not exist. Cannot auto select clang compilers.")
    endif()

    # Set up CMake package search path for TheRock build
    if(EXISTS ${WINDOWS_ROCM_CMAKE_DIR})
        # Add to CMAKE_PREFIX_PATH for find_package() searches
        list(APPEND CMAKE_PREFIX_PATH ${WINDOWS_ROCM_CMAKE_DIR})
        message(STATUS "Added Windows ROCm CMake package search path: ${WINDOWS_ROCM_CMAKE_DIR}")
    else()
        message(FATAL_ERROR "Windows ROCm CMake directory not found: ${WINDOWS_ROCM_CMAKE_DIR}")
    endif()
endif()
