# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# =============================================================================
# CK Tile Dispatcher - ctypes Bindings
# =============================================================================
#
# Provides shared libraries with C API for Python ctypes integration.
#
# Targets:
#   - dispatcher_gemm_lib      : GEMM dispatcher library
#   - dispatcher_mx_gemm_lib   : MX-GEMM (microscaling) dispatcher library (gfx950)
#   - dispatcher_conv_lib      : Convolution dispatcher library (forward + bwd_data)
#   - dispatcher_conv_bwdw_lib : Convolution backward weight library
#   - gpu_helper               : GPU helper executable for Python
#

cmake_minimum_required(VERSION 3.16)

# Helper function to add a ctypes library
function(add_ctypes_library TARGET_NAME SOURCE_FILE)
    cmake_parse_arguments(ARG "CONV" "KERNEL_HEADER" "" ${ARGN})
    
    add_library(${TARGET_NAME} SHARED ${SOURCE_FILE})
    
    target_include_directories(${TARGET_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    
    target_link_libraries(${TARGET_NAME} PRIVATE
        hip::device
    )
    
    # Force-include kernel header if provided
    if(ARG_KERNEL_HEADER AND EXISTS ${ARG_KERNEL_HEADER})
        target_compile_options(${TARGET_NAME} PRIVATE
            -include ${ARG_KERNEL_HEADER}
        )
        if(ARG_CONV)
            target_compile_definitions(${TARGET_NAME} PRIVATE CONV_KERNEL_AVAILABLE)
        endif()
    endif()
    
    set_target_properties(${TARGET_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )
endfunction()

# Helper function for block-scale quant ctypes targets.
#
# Usage:
#   add_quant_ctypes_target(
#     TARGET      <cmake-target-name>
#     SOURCE      <source.cpp>
#     GLOB        <glob-pattern-for-kernel-headers>
#     CACHE_VAR   <GFX_ARCH cache variable name>
#     [MAIN_INCLUDE_ONLY]   # omit dispatcher/include (aquant/abquant collision fix)
#   )
#
# B1: single wrapper replacing five near-identical per-operator stanzas.
# B2: always list(SORT) the kernel glob, so the first entry is deterministic.
# D1: each plain target gets its own cache variable (CACHE_VAR), distinct from the
#     grouped targets that previously shared the same names and silently stomped the
#     plain targets' NOT DEFINED guard.
#
# MAIN_INCLUDE_ONLY: skip ${PROJECT_SOURCE_DIR}/dispatcher/include. That path pulls
#   in generated_tile_backend.hpp, which instantiates SelectedKernel::run(GemmHostArgs&)
#   and conflicts with the quant kernel's launch(QuantGemmHostArgs&). Aquant and abquant
#   set this flag; the plain GEMM and grouped quant targets do not need it.
#
# Arch resolution has NO default. Guessing here is not a convenience: the generated
#   kernel headers pin an arch-specific warp_tile_k, and a mismatch makes the kernel
#   silently emit all zeros rather than fail. An unresolved arch is a hard
#   configure-time error for every target.
function(add_quant_ctypes_target)
    cmake_parse_arguments(QUANT "MAIN_INCLUDE_ONLY" "TARGET;SOURCE;GLOB;CACHE_VAR" "" ${ARGN})

    file(GLOB _headers "${QUANT_GLOB}")
    if(NOT _headers)
        message(STATUS "No kernel found for ${QUANT_TARGET} - skipping (built at runtime by Python bridge)")
        return()
    endif()

    list(SORT _headers)
    list(GET _headers 0 _kernel_header)
    message(STATUS "Found kernel for ${QUANT_TARGET}: ${_kernel_header}")

    if(QUANT_MAIN_INCLUDE_ONLY)
        # Hand-build the target without dispatcher/include to avoid the
        # SelectedKernel::run(GemmHostArgs&) vs launch(QuantGemmHostArgs&) conflict.
        add_library(${QUANT_TARGET} SHARED ${QUANT_SOURCE})
        target_include_directories(${QUANT_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/include)
        target_link_libraries(${QUANT_TARGET} PRIVATE hip::device)
        target_compile_options(${QUANT_TARGET} PRIVATE -include ${_kernel_header})
        set_target_properties(${QUANT_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17)
    else()
        add_ctypes_library(${QUANT_TARGET} ${QUANT_SOURCE} KERNEL_HEADER ${_kernel_header})
    endif()

    # Resolve GFX arch: cache variable wins, then CMAKE_HIP_ARCHITECTURES, else error.
    if(NOT DEFINED ${QUANT_CACHE_VAR} OR "${${QUANT_CACHE_VAR}}" STREQUAL "")
        if(CMAKE_HIP_ARCHITECTURES)
            list(GET CMAKE_HIP_ARCHITECTURES 0 _resolved_arch)
            set(${QUANT_CACHE_VAR} "${_resolved_arch}" CACHE STRING
                "GFX arch for the CMake-built ${QUANT_TARGET} ctypes .so")
        else()
            message(FATAL_ERROR
                "${QUANT_TARGET}: GPU arch is unresolved. Set -D${QUANT_CACHE_VAR}=<gfx###> "
                "or CMAKE_HIP_ARCHITECTURES.")
        endif()
    endif()
    message(STATUS "${QUANT_TARGET} GFX_ARCH: ${${QUANT_CACHE_VAR}}")

    target_compile_definitions(${QUANT_TARGET} PRIVATE
        CK_TILE_SINGLE_KERNEL_INCLUDE
        GFX_ARCH="${${QUANT_CACHE_VAR}}"
    )
endfunction()

# =============================================================================
# GEMM ctypes Library
# =============================================================================

# Find a generated GEMM kernel header for the library.
# The glob "gemm_*.hpp" also matches every block-scale quant family, which
# defines a QuantGemm SelectedKernel whose API is incompatible with the regular
# GEMM ctypes lib; force-including one into dispatcher_gemm_lib / gpu_helper
# fails the build. Filter with an ALLOW regex on the regular GEMM name shape
# (gemm_<dtype>_<layout>_...) rather than a deny-list of quant family names: a
# deny-list silently admits the next quant family somebody adds.
file(GLOB GEMM_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/gemm_*.hpp")
list(FILTER GEMM_KERNEL_HEADERS INCLUDE REGEX
     "/gemm_(fp16|bf16|fp32|fp8|bf8|int8)_[rc][rc][rc]_")
if(GEMM_KERNEL_HEADERS)
    list(SORT GEMM_KERNEL_HEADERS)
    list(GET GEMM_KERNEL_HEADERS 0 GEMM_KERNEL_HEADER)
    message(STATUS "Found GEMM kernel for ctypes lib: ${GEMM_KERNEL_HEADER}")
    
    add_ctypes_library(dispatcher_gemm_lib 
        gemm_ctypes_lib.cpp 
        KERNEL_HEADER ${GEMM_KERNEL_HEADER}
    )
else()
    message(STATUS "No GEMM kernel found for ctypes lib - building without kernel")
    add_library(dispatcher_gemm_lib SHARED gemm_ctypes_lib.cpp)
    target_include_directories(dispatcher_gemm_lib PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    target_link_libraries(dispatcher_gemm_lib PRIVATE hip::device)
endif()

# =============================================================================
# MX-GEMM ctypes Library (microscaling GEMM, gfx950/MI350 only)
# =============================================================================
#
# Force-includes a generated mx_gemm kernel header (SelectedKernel/KERNEL_NAME/
# MxGemmHostArgs/ScaleType/...). The lib also includes the Old-TE common helper
# (common/utils.hpp) for the free-function is_row_major(Layout), so add the
# tile_engine/ops include roots.

set(MX_GEMM_TE_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/tile_engine/ops
    ${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm
    ${PROJECT_SOURCE_DIR}/tile_engine/ops/gemm/mx_gemm
)

# Resolve the configured GPU target GENERICALLY -- do NOT hardcode/default the
# arch to gfx950. mx_gemm is gfx950-only (it calls the gfx950-only
# preShuffleScaleBuffer_gfx950 helper), and that requirement is enforced IN THE
# SOURCE: mx_gemm_ctypes_lib.cpp #errors if GFX_ARCH is unset and static_asserts
# GFX_ARCH == "gfx950" (plus a runtime device guard). Here we simply pass the
# real build target through as -DGFX_ARCH and only wire up the target when the
# build actually targets gfx950, so a gfx942/other dispatcher build is neither
# broken by the static_assert nor silently built for an unsupported arch.
if(DEFINED GPU_TARGETS AND NOT GPU_TARGETS STREQUAL "")
    string(REPLACE ";" " " _mx_gpu_targets_space "${GPU_TARGETS}")
    string(REPLACE " " ";" _mx_gpu_targets_list "${_mx_gpu_targets_space}")
    list(GET _mx_gpu_targets_list 0 MX_GEMM_GPU_TARGET)
elseif(DEFINED CK_TILE_GEMM_GPU_TARGET AND NOT CK_TILE_GEMM_GPU_TARGET STREQUAL "")
    set(MX_GEMM_GPU_TARGET "${CK_TILE_GEMM_GPU_TARGET}")
else()
    set(MX_GEMM_GPU_TARGET "")
endif()

if(NOT MX_GEMM_GPU_TARGET STREQUAL "gfx950")
    message(STATUS
        "MX-GEMM ctypes lib is gfx950-only; skipping for GPU target "
        "'${MX_GEMM_GPU_TARGET}' (configure with GPU_TARGETS=gfx950 to enable).")
else()
    file(GLOB MX_GEMM_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/mx_gemm_*.hpp")
    if(MX_GEMM_KERNEL_HEADERS)
        list(GET MX_GEMM_KERNEL_HEADERS 0 MX_GEMM_KERNEL_HEADER)
        message(STATUS "Found MX-GEMM kernel for ctypes lib: ${MX_GEMM_KERNEL_HEADER}")

        add_ctypes_library(dispatcher_mx_gemm_lib
            mx_gemm_ctypes_lib.cpp
            KERNEL_HEADER ${MX_GEMM_KERNEL_HEADER}
        )
        target_include_directories(dispatcher_mx_gemm_lib PRIVATE ${MX_GEMM_TE_INCLUDE_DIRS})
        # The generated header only exports SelectedKernel/KERNEL_NAME/ScaleType/...
        # under this define. Pass the resolved build target through as -DGFX_ARCH
        # (the source #errors if unset and static_asserts it is gfx950).
        target_compile_definitions(dispatcher_mx_gemm_lib PRIVATE
            CK_TILE_SINGLE_KERNEL_INCLUDE
            GFX_ARCH="${MX_GEMM_GPU_TARGET}"
        )
    else()
        message(STATUS "No MX-GEMM kernel found for ctypes lib - building without kernel")
        add_library(dispatcher_mx_gemm_lib SHARED mx_gemm_ctypes_lib.cpp)
        target_include_directories(dispatcher_mx_gemm_lib PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/dispatcher/include
            ${MX_GEMM_TE_INCLUDE_DIRS}
        )
        target_link_libraries(dispatcher_mx_gemm_lib PRIVATE hip::device)
        # Pass the resolved build target through as -DGFX_ARCH (the source #errors
        # if unset and static_asserts it is gfx950).
        target_compile_definitions(dispatcher_mx_gemm_lib PRIVATE
            GFX_ARCH="${MX_GEMM_GPU_TARGET}")
        set_target_properties(dispatcher_mx_gemm_lib PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD 17
        )
    endif()
endif()

# =============================================================================
# Convolution ctypes Library (supports forward + bwd_data)
# =============================================================================

# Look for forward kernels
file(GLOB CONV_FWD_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/conv_fwd_*.hpp")
# Look for backward data kernels  
file(GLOB CONV_BWDD_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/conv_bwd_data_*.hpp")
# Fallback: any conv kernel (for backwards compatibility)
file(GLOB CONV_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/conv_*.hpp")

add_library(dispatcher_conv_lib SHARED conv_ctypes_lib.cpp)
target_include_directories(dispatcher_conv_lib PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/dispatcher/include
)
target_link_libraries(dispatcher_conv_lib PRIVATE hip::device)
set_target_properties(dispatcher_conv_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
)

# Add forward kernel if available
if(CONV_FWD_KERNEL_HEADERS)
    list(SORT CONV_FWD_KERNEL_HEADERS)
    list(GET CONV_FWD_KERNEL_HEADERS 0 CONV_FWD_KERNEL_HEADER)
    message(STATUS "Found Conv FWD kernel for ctypes lib: ${CONV_FWD_KERNEL_HEADER}")
    target_compile_options(dispatcher_conv_lib PRIVATE -include ${CONV_FWD_KERNEL_HEADER})
    target_compile_definitions(dispatcher_conv_lib PRIVATE CONV_KERNEL_AVAILABLE)
elseif(CONV_KERNEL_HEADERS)
    # Fallback to any conv kernel
    list(SORT CONV_KERNEL_HEADERS)
    list(GET CONV_KERNEL_HEADERS 0 CONV_KERNEL_HEADER)
    message(STATUS "Found Conv kernel for ctypes lib: ${CONV_KERNEL_HEADER}")
    target_compile_options(dispatcher_conv_lib PRIVATE -include ${CONV_KERNEL_HEADER})
    target_compile_definitions(dispatcher_conv_lib PRIVATE CONV_KERNEL_AVAILABLE)
else()
    message(STATUS "No Conv FWD kernel found for ctypes lib - building without kernel")
endif()

# Add backward data kernel if available
if(CONV_BWDD_KERNEL_HEADERS)
    list(SORT CONV_BWDD_KERNEL_HEADERS)
    list(GET CONV_BWDD_KERNEL_HEADERS 0 CONV_BWDD_KERNEL_HEADER)
    message(STATUS "Found Conv BWD_DATA kernel for ctypes lib: ${CONV_BWD_DATA_KERNEL_HEADER}")
    target_compile_options(dispatcher_conv_lib PRIVATE -include ${CONV_BWDD_KERNEL_HEADER})
    target_compile_definitions(dispatcher_conv_lib PRIVATE CONV_BWD_DATA_AVAILABLE)
endif()

# =============================================================================
# Convolution Backward Weight ctypes Library (separate lib for bwd_weight)
# =============================================================================

file(GLOB CONV_BWDW_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/conv_*bwd_weight*.hpp")
if(CONV_BWDW_KERNEL_HEADERS)
    list(SORT CONV_BWDW_KERNEL_HEADERS)
    list(GET CONV_BWDW_KERNEL_HEADERS 0 CONV_BWDW_KERNEL_HEADER)
    message(STATUS "Found Conv BwdWeight kernel for ctypes lib: ${CONV_BWDW_KERNEL_HEADER}")
    
    add_library(dispatcher_conv_bwdw_lib SHARED conv_bwdw_ctypes_lib.cpp)
    target_include_directories(dispatcher_conv_bwdw_lib PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    target_link_libraries(dispatcher_conv_bwdw_lib PRIVATE hip::device)
    target_compile_options(dispatcher_conv_bwdw_lib PRIVATE
        -include ${CONV_BWDW_KERNEL_HEADER}
    )
    target_compile_definitions(dispatcher_conv_bwdw_lib PRIVATE CONV_BWD_WEIGHT_AVAILABLE)
    set_target_properties(dispatcher_conv_bwdw_lib PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )
else()
    message(STATUS "No Conv BwdWeight kernel found for ctypes lib - building without kernel")
    add_library(dispatcher_conv_bwdw_lib SHARED conv_bwdw_ctypes_lib.cpp)
    target_include_directories(dispatcher_conv_bwdw_lib PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    target_link_libraries(dispatcher_conv_bwdw_lib PRIVATE hip::device)
    set_target_properties(dispatcher_conv_bwdw_lib PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )
endif()

# =============================================================================
# GroupedGemm BQuant ctypes Library
# =============================================================================

add_quant_ctypes_target(
    TARGET      dispatcher_grouped_gemm_bquant_lib
    SOURCE      grouped_gemm_bquant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_bquant_*.hpp"
    CACHE_VAR   CK_TILE_BQUANT_GFX_ARCH
)

# =============================================================================
# GroupedGemm RowColQuant ctypes Library
# =============================================================================

add_quant_ctypes_target(
    TARGET      dispatcher_grouped_gemm_rowcolquant_lib
    SOURCE      grouped_gemm_rowcolquant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_rowcolquant_*.hpp"
    CACHE_VAR   CK_TILE_ROWCOLQUANT_GFX_ARCH
)

# =============================================================================
# GroupedGemm TensorQuant ctypes Library
# =============================================================================

add_quant_ctypes_target(
    TARGET      dispatcher_grouped_gemm_tensorquant_lib
    SOURCE      grouped_gemm_tensorquant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_tensorquant_*.hpp"
    CACHE_VAR   CK_TILE_TENSORQUANT_GFX_ARCH
)

# =============================================================================
# GroupedGemm AQuant ctypes Library
# =============================================================================

add_quant_ctypes_target(
    TARGET      dispatcher_grouped_gemm_aquant_lib
    SOURCE      grouped_gemm_aquant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_aquant_*.hpp"
    CACHE_VAR   CK_TILE_AQUANT_GFX_ARCH
)

# =============================================================================
# GroupedGemm ABQuant ctypes Library
# =============================================================================

add_quant_ctypes_target(
    TARGET      dispatcher_grouped_gemm_abquant_lib
    SOURCE      grouped_gemm_abquant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/grouped_gemm_abquant_*.hpp"
    CACHE_VAR   CK_TILE_ABQUANT_GFX_ARCH
)

# =============================================================================
# Gemm TensorQuant ctypes Library
# =============================================================================
# Registry-bypass, single-kernel-per-.so bridge for the TensorQuant GEMM path
# (one scalar scale for A + one scalar scale for B). fp8/bf8, rcr layout only.
# CK_TILE_GEMM_TENSOR_QUANT_GFX_ARCH is DISTINCT from the grouped targets'
# CK_TILE_TENSOR_QUANT_GFX_ARCH to prevent the grouped `set(... CACHE)` call
# from silently pre-filling this variable before the NOT DEFINED guard below.

add_quant_ctypes_target(
    TARGET      dispatcher_gemm_tensor_quant_lib
    SOURCE      gemm_tensor_quant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/gemm_tensor_quant_*.hpp"
    CACHE_VAR   CK_TILE_GEMM_TENSOR_QUANT_GFX_ARCH
)

# =============================================================================
# Gemm BQuant ctypes Library (non-grouped, block-scale GEMM)
# =============================================================================
# Plain (single-problem) B-only quantized block-scale GEMM from
# 38_block_scale_gemm. Distinct from the grouped_gemm_bquant target above.
# Doubly guarded (on the generated header glob and on the source file) so the
# target is skipped cleanly if neither is present.

# =============================================================================
# Gemm BQuant ctypes Library (non-grouped, block-scale GEMM)
# =============================================================================
# Plain (single-problem) B-only quantized block-scale GEMM from
# 38_block_scale_gemm. Distinct from the grouped_gemm_bquant target above.
# CK_TILE_GEMM_BQUANT_GFX_ARCH is DISTINCT from the grouped target's
# CK_TILE_BQUANT_GFX_ARCH: the grouped block runs first and sets that variable
# via `set(... CACHE)`, which means any `if(NOT DEFINED CK_TILE_BQUANT_GFX_ARCH)`
# guard below silently never fires -- the grouped stanza's value is silently
# inherited instead. The distinct name ensures each target is independently
# configurable.

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gemm_bquant_ctypes_lib.cpp)
    add_quant_ctypes_target(
        TARGET      dispatcher_gemm_bquant_lib
        SOURCE      gemm_bquant_ctypes_lib.cpp
        GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/gemm_bquant_*.hpp"
        CACHE_VAR   CK_TILE_GEMM_BQUANT_GFX_ARCH
    )
    if(NOT TARGET dispatcher_gemm_bquant_lib)
        # add_quant_ctypes_target returned early (no kernel header). Build a
        # kernel-less stub so an all-target build still configures and compiles:
        # CK_TILE_SINGLE_KERNEL_INCLUDE is deliberately NOT defined here, which is
        # what makes gemm_bquant_ctypes_lib.cpp compile its no-kernel path (the
        # entry point returns -2 "unsupported"). Same idiom as dispatcher_mx_gemm_lib.
        message(STATUS "No gemm_bquant kernel found for ctypes lib - building without kernel")
        add_library(dispatcher_gemm_bquant_lib SHARED gemm_bquant_ctypes_lib.cpp)
        target_include_directories(dispatcher_gemm_bquant_lib PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/dispatcher/include
        )
        target_link_libraries(dispatcher_gemm_bquant_lib PRIVATE hip::device)
        set_target_properties(dispatcher_gemm_bquant_lib PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD 17
        )
    endif()
endif()

# =============================================================================
# TileEngine -> Dispatcher Bridge ctypes Libraries (append-only shared block)
# =============================================================================
#
# The bridge PRs (#9305 gemm_multi_abd, #9306 batched_gemm, #9328
# batched_contraction) each add a registry-bypass ctypes .so target. To keep the
# branches mutually conflict-proof, this block is BYTE-IDENTICAL on every branch
# that touches this file. Each target is doubly guarded:
#   (1) on its generated kernel-header glob, AND
#   (2) on if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/<op>_ctypes_lib.cpp)
# so a branch that does not ship a given <op>_ctypes_lib.cpp simply skips that
# target at configure time (no error). The Python bridges build these .so files
# at runtime via hipcc; these CMake targets exist for CMake-driven / CI builds.

# --- GEMM Multi-ABD (registry-bypass, array-pointer ABI) ---------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gemm_multi_abd_ctypes_lib.cpp)
    file(GLOB GEMM_MULTI_ABD_KERNEL_HEADERS
        "${CMAKE_BINARY_DIR}/generated_kernels/gemm_*_multiabd_*.hpp")
    if(GEMM_MULTI_ABD_KERNEL_HEADERS)
        list(SORT GEMM_MULTI_ABD_KERNEL_HEADERS)
        list(GET GEMM_MULTI_ABD_KERNEL_HEADERS 0 GEMM_MULTI_ABD_KERNEL_HEADER)
        message(STATUS "Found GEMM Multi-ABD kernel for ctypes lib: ${GEMM_MULTI_ABD_KERNEL_HEADER}")
        add_ctypes_library(dispatcher_gemm_multi_abd_lib
            gemm_multi_abd_ctypes_lib.cpp
            KERNEL_HEADER ${GEMM_MULTI_ABD_KERNEL_HEADER}
        )
        target_compile_definitions(dispatcher_gemm_multi_abd_lib PRIVATE
            CK_TILE_SINGLE_KERNEL_INCLUDE)
    else()
        message(STATUS
            "No GEMM Multi-ABD kernel found for ctypes lib - skipping dispatcher_gemm_multi_abd_lib "
            "(built at runtime by the Python bridge once a kernel header exists)")
    endif()
endif()

# --- Batched GEMM (registry-bypass, batch_count + per-batch strides ABI) ------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/batched_gemm_ctypes_lib.cpp)
    file(GLOB BATCHED_GEMM_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/gemm_*_batched.hpp")
    if(BATCHED_GEMM_KERNEL_HEADERS)
        list(SORT BATCHED_GEMM_KERNEL_HEADERS)
        list(GET BATCHED_GEMM_KERNEL_HEADERS 0 BATCHED_GEMM_KERNEL_HEADER)
        message(STATUS "Found Batched GEMM kernel for ctypes lib: ${BATCHED_GEMM_KERNEL_HEADER}")
        add_library(dispatcher_batched_gemm_lib SHARED batched_gemm_ctypes_lib.cpp)
        target_include_directories(dispatcher_batched_gemm_lib PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/dispatcher/include
        )
        target_link_libraries(dispatcher_batched_gemm_lib PRIVATE hip::device)
        target_compile_options(dispatcher_batched_gemm_lib PRIVATE
            -include ${BATCHED_GEMM_KERNEL_HEADER}
        )
        target_compile_definitions(dispatcher_batched_gemm_lib PRIVATE CK_TILE_SINGLE_KERNEL_INCLUDE)
        set_target_properties(dispatcher_batched_gemm_lib PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD 17
        )
    else()
        message(STATUS "No Batched GEMM kernel found for ctypes lib - skipping dispatcher_batched_gemm_lib")
    endif()
endif()

# --- Batched-Contraction (registry-bypass, force-included kernel) ------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/batched_contraction_ctypes_lib.cpp)
    file(GLOB BC_KERNEL_HEADERS "${CMAKE_BINARY_DIR}/generated_kernels/batched_contraction_*.hpp")
    if(BC_KERNEL_HEADERS)
        list(SORT BC_KERNEL_HEADERS)
        list(GET BC_KERNEL_HEADERS 0 BC_KERNEL_HEADER)
        message(STATUS "Found batched-contraction kernel for ctypes lib: ${BC_KERNEL_HEADER}")
        add_ctypes_library(dispatcher_batched_contraction_lib
            batched_contraction_ctypes_lib.cpp
            KERNEL_HEADER ${BC_KERNEL_HEADER}
        )
        # The generated header only exports SelectedKernel/KERNEL_NAME/CONTRACTION_KEY_*
        # under this define, so the force-include path requires it.
        target_compile_definitions(dispatcher_batched_contraction_lib PRIVATE
            CK_TILE_SINGLE_KERNEL_INCLUDE
        )
    else()
        message(STATUS
            "No batched-contraction kernel found for ctypes lib - skipping "
            "dispatcher_batched_contraction_lib (built at runtime by the Python "
            "bridge once a kernel header exists)")
    endif()
endif()

# =============================================================================
# GPU Helper Executable
# =============================================================================

if(GEMM_KERNEL_HEADERS)
    add_executable(gpu_helper gpu_helper.cpp)
    
    target_include_directories(gpu_helper PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/dispatcher/include
    )
    
    target_link_libraries(gpu_helper PRIVATE
        hip::device
    )
    
    target_compile_options(gpu_helper PRIVATE
        -include ${GEMM_KERNEL_HEADER}
    )
    
    set_target_properties(gpu_helper PROPERTIES
        CXX_STANDARD 17
    )
endif()


# =============================================================================
# Gemm RowColQuant ctypes Library
# =============================================================================
# CK_TILE_GEMM_ROWCOLQUANT_GFX_ARCH is a new name with no grouped-target alias,
# so no cache collision is possible. Included for naming consistency.

add_quant_ctypes_target(
    TARGET      dispatcher_gemm_rowcolquant_lib
    SOURCE      gemm_rowcolquant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/gemm_rowcolquant_*.hpp"
    CACHE_VAR   CK_TILE_GEMM_ROWCOLQUANT_GFX_ARCH
)

# =============================================================================
# AQuant (A-only quantized) GEMM ctypes Library
# =============================================================================
# MAIN_INCLUDE_ONLY: omit dispatcher/include to avoid the GemmHostArgs vs
# QuantGemmHostArgs collision (see add_quant_ctypes_target docs above).
# CK_TILE_GEMM_AQUANT_GFX_ARCH is distinct from grouped CK_TILE_AQUANT_GFX_ARCH
# (D1 fix: grouped block runs first and would stomp the NOT DEFINED guard).

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gemm_aquant_ctypes_lib.cpp)
    add_quant_ctypes_target(
        TARGET      dispatcher_gemm_aquant_lib
        SOURCE      gemm_aquant_ctypes_lib.cpp
        GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/gemm_aquant_*.hpp"
        CACHE_VAR   CK_TILE_GEMM_AQUANT_GFX_ARCH
        MAIN_INCLUDE_ONLY
    )
endif()

# =============================================================================
# Gemm ABQuant (A+B block-scale) ctypes Library
# =============================================================================
# MAIN_INCLUDE_ONLY: same rationale as AQuant above (QuantGemmHostArgs collision).
# CK_TILE_GEMM_ABQUANT_GFX_ARCH is distinct from the grouped target's
# CK_TILE_ABQUANT_GFX_ARCH. There is no default arch: an unresolved arch is a
# configure-time error (see add_quant_ctypes_target).

add_quant_ctypes_target(
    TARGET      dispatcher_gemm_abquant_lib
    SOURCE      gemm_abquant_ctypes_lib.cpp
    GLOB        "${CMAKE_BINARY_DIR}/generated_kernels/gemm_abquant_*.hpp"
    CACHE_VAR   CK_TILE_GEMM_ABQUANT_GFX_ARCH
    MAIN_INCLUDE_ONLY
)

# =============================================================================
