# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(hipblaslt_codegen)

set(TENSILELITE_BUILD_PARALLEL_LEVEL "" CACHE STRING "Number of CPU cores to use for building device libraries (will use nproc if unset).")
set(TENSILELITE_KEEP_BUILD_TMP OFF CACHE STRING "Keep temporary build directory for device libraries (turning this ON  bloat the build size).")
set(TENSILELITE_ASM_DEBUG "" CACHE STRING "Keep debug information for built code objects.")
set(TENSILELITE_LOGIC_FILTER "" CACHE STRING "Cutomsized logic filter, default is *, i.e. all logics.")
set(TENSILELITE_NO_COMPRESS "" CACHE STRING "Do not compress device code object files.")
set(TENSILELITE_EXPERIMENTAL "" CACHE STRING "Process experimental logic files.")
set(TENSILELITE_ENABLE_ASM_COMMENTS OFF CACHE STRING "Turn on asm comments when building gemm libraries.")
set(TENSILELITE_OFFLOADBUNDLER "" CACHE STRING "Path to clang-offload-bundler.")
mark_as_advanced(
    TENSILELITE_BUILD_PARALLEL_LEVEL
    TENSILELITE_KEEP_BUILD_TMP
    TENSILELITE_ASM_DEBUG
    TENSILELITE_LOGIC_FILTER
    TENSILELITE_NO_COMPRESS
    TENSILELITE_EXPERIMENTAL
    TENSILELITE_ENABLE_ASM_COMMENTS
    TENSILELITE_OFFLOADBUNDLER
)

set(HIPBLASLT_LIBLOGIC_PATH "" CACHE STRING "Path to library logic files (will use 'library' if unset).")
set(HIPBLASLT_TENSILE_LIBPATH "${hipblaslt_BINARY_DIR}/Tensile" CACHE STRING "Path to output the device gemm libraries.")

if(NOT HIPBLASLT_LIBLOGIC_PATH)
    set(HIPBLASLT_LIBLOGIC_PATH "${hipblaslt_SOURCE_DIR}/library")
endif()
if(NOT HIPBLASLT_CODEGEN_ROOT)
    set(HIPBLASLT_CODEGEN_ROOT "${hipblaslt_SOURCE_DIR}/tensilelite")
endif()

set(_codegen_host_sanitizer_args "")
if(HIPBLASLT_ENABLE_ASAN
        OR THEROCK_SANITIZER STREQUAL "ASAN"
        OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
    list(APPEND _codegen_host_sanitizer_args HOST_ASAN)
elseif(HIPBLASLT_ENABLE_TSAN OR THEROCK_SANITIZER STREQUAL "TSAN")
    list(APPEND _codegen_host_sanitizer_args HOST_TSAN)
endif()

set(_codegen_product_args "")
if(HIPBLASLT_ENABLE_ASAN)
    list(APPEND _codegen_product_args ASAN "${HIPBLASLT_ENABLE_ASAN}")
endif()
if(HIPBLASLT_ENABLE_YAML)
    list(APPEND _codegen_product_args YAML_FORMAT "${HIPBLASLT_ENABLE_YAML}")
endif()
if(NOT DEFINED HIPBLASLT_ENABLE_LAZY_LOAD OR NOT HIPBLASLT_ENABLE_LAZY_LOAD)
    list(APPEND _codegen_product_args LAZY_LOAD "${HIPBLASLT_ENABLE_LAZY_LOAD}")
endif()
if(HIPBLASLT_ENABLE_GEMM_A2A_FUSION)
    list(APPEND _codegen_product_args GEMM_A2A_FUSION "${HIPBLASLT_ENABLE_GEMM_A2A_FUSION}")
endif()

create_device_library(
    TARGET tensilelite-device-libraries
    LOGIC_PATH "${HIPBLASLT_LIBLOGIC_PATH}"
    OUTPUT_DIR "${HIPBLASLT_TENSILE_LIBPATH}"
    ARCHES ${GPU_TARGETS}
    ${_codegen_host_sanitizer_args}
    ${_codegen_product_args}
)

if(HIPBLASLT_ENABLE_EXTOPS)
    add_subdirectory(extops)
endif()
if(HIPBLASLT_ENABLE_MATRIX_TRANSFORM)
    add_subdirectory(matrix-transform)
endif()
