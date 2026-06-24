# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(HipBLASLtCodegen)

set(TENSILELITE_BUILD_PARALLEL_LEVEL "" CACHE STRING "Number of CPU cores to use for building device libraries (will use nproc if unset).")
set(TENSILELITE_KEEP_BUILD_TMP OFF CACHE STRING "Keep temporary build directory for device libraries (turning this ON  bloat the build size).")
set(TENSILELITE_ASM_DEBUG "" CACHE STRING "Keep debug information for built code objects.")
set(TENSILELITE_LOGIC_FILTER "" CACHE STRING "Cutomsized logic filter, default is *, i.e. all logics.")
set(TENSILELITE_NO_COMPRESS "" CACHE STRING "Do not compress device code object files.")
set(TENSILELITE_EXPERIMENTAL "" CACHE STRING "Process experimental logic files.")
set(TENSILELITE_ENABLE_ASM_COMMENTS OFF CACHE STRING "Turn on asm comments when building gemm libraries.")
set(TENSILELITE_OFFLOADBUNDLER "" CACHE STRING "Path to clang-offload-bundler.")

set(HIPBLASLT_LIBLOGIC_PATH "" CACHE STRING "Path to library logic files (will use 'library' if unset).")
set(HIPBLASLT_TENSILE_LIBPATH "${hipblaslt_BINARY_DIR}/Tensile" CACHE STRING "Path to output the device gemm libraries.")

if(NOT HIPBLASLT_LIBLOGIC_PATH)
    set(HIPBLASLT_LIBLOGIC_PATH "${hipblaslt_SOURCE_DIR}/library")
endif()

set(HIPBLASLT_CODEGEN_ROOT "${hipblaslt_SOURCE_DIR}/tensilelite")

set(_codegen_flags "")
if(HIPBLASLT_ENABLE_ASAN)
    list(APPEND _codegen_flags ASAN)
endif()
if(HIPBLASLT_ENABLE_YAML)
    list(APPEND _codegen_flags YAML_FORMAT)
endif()
if(TENSILELITE_NO_COMPRESS)
    list(APPEND _codegen_flags NO_COMPRESS)
endif()
if(TENSILELITE_EXPERIMENTAL)
    list(APPEND _codegen_flags EXPERIMENTAL)
endif()
if(NOT HIPBLASLT_ENABLE_LAZY_LOAD)
    list(APPEND _codegen_flags NO_LAZY_LOAD)
endif()
if(TENSILELITE_ENABLE_ASM_COMMENTS)
    list(APPEND _codegen_flags ASM_COMMENTS)
endif()
if(TENSILELITE_KEEP_BUILD_TMP)
    list(APPEND _codegen_flags KEEP_BUILD_TMP)
endif()
if(TENSILELITE_ASM_DEBUG)
    list(APPEND _codegen_flags ASM_DEBUG)
endif()

set(_codegen_opts "")
if(TENSILELITE_OFFLOADBUNDLER)
    list(APPEND _codegen_opts OFFLOAD_BUNDLER "${TENSILELITE_OFFLOADBUNDLER}")
endif()
if(TENSILELITE_BUILD_PARALLEL_LEVEL)
    list(APPEND _codegen_opts JOBS "${TENSILELITE_BUILD_PARALLEL_LEVEL}")
endif()
if(TENSILELITE_LOGIC_FILTER)
    list(APPEND _codegen_opts LOGIC_FILTER "${TENSILELITE_LOGIC_FILTER}")
endif()

hipblaslt_create_device_library(
    TARGET tensilelite-device-libraries
    LOGIC_PATH "${HIPBLASLT_LIBLOGIC_PATH}"
    OUTPUT_DIR "${HIPBLASLT_TENSILE_LIBPATH}"
    ARCHES ${GPU_TARGETS}
    CXX_COMPILER "${CMAKE_CXX_COMPILER}"
    ${_codegen_opts}
    ${_codegen_flags}
)

if(HIPBLASLT_ENABLE_EXTOPS)
    add_subdirectory(extops)
endif()
if(HIPBLASLT_ENABLE_MATRIX_TRANSFORM)
    add_subdirectory(matrix-transform)
endif()
