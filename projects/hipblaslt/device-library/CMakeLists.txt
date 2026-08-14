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

set(HIPBLASLT_GFX1250_REVISION "" CACHE STRING "gfx1250 ASIC revision to build for ('v0' or 'v1'); probed by 'invoke build' when unset.")

# Rejected rather than ignored: an unrecognized value matches no branch below
# and would quietly build the shipping revision for someone who asked for v0.
if(NOT HIPBLASLT_GFX1250_REVISION MATCHES "^(v0|v1)?$")
    message(FATAL_ERROR "HIPBLASLT_GFX1250_REVISION must be 'v0' or 'v1', got '${HIPBLASLT_GFX1250_REVISION}'")
endif()

if(NOT HIPBLASLT_LIBLOGIC_PATH)
    set(HIPBLASLT_LIBLOGIC_PATH "${hipblaslt_SOURCE_DIR}/library")
endif()

# The revision changes only TensileLite's arch name; GPU_TARGETS stays as the
# caller wrote it, since it also feeds extops, matrix-transform and target
# validation, which don't know the name. Only the bare gfx1250 reaches here, so
# widen "^gfx1250$" and the IN_LIST tests below if a qualified one is ever added.
set(_tensile_arches ${GPU_TARGETS})
if(HIPBLASLT_GFX1250_REVISION STREQUAL "v0")
    list(TRANSFORM _tensile_arches REPLACE "^gfx1250$" "gfx1250v0")
endif()

# The revisions build different libraries from one command, and the variable is
# cached, so record in the configure log which one this directory is for.
if("gfx1250v0" IN_LIST _tensile_arches)
    message(STATUS "hipBLASLt: gfx1250 ASIC revision v0 (gfx1250v0 logic only; it replaces "
                   "gfx1250's own tuning rather than adding to it)")
elseif("gfx1250" IN_LIST _tensile_arches)
    message(STATUS "hipBLASLt: gfx1250 ASIC revision v1 (shipping)")
elseif(HIPBLASLT_GFX1250_REVISION STREQUAL "v0")
    message(WARNING "HIPBLASLT_GFX1250_REVISION=v0 but GPU_TARGETS names no gfx1250 "
                    "(${GPU_TARGETS}); nothing in this build is for the v0 revision.")
endif()

hipblaslt_create_device_library(
    TARGET tensilelite-device-libraries
    LOGIC_PATH "${HIPBLASLT_LIBLOGIC_PATH}"
    OUTPUT_DIR "${HIPBLASLT_TENSILE_LIBPATH}"
    ARCHES ${_tensile_arches}
    CXX_COMPILER "${CMAKE_CXX_COMPILER}"
    OFFLOAD_BUNDLER "${TENSILELITE_OFFLOADBUNDLER}"
    JOBS "${TENSILELITE_BUILD_PARALLEL_LEVEL}"
    LOGIC_FILTER "${TENSILELITE_LOGIC_FILTER}"
    ASAN "${HIPBLASLT_ENABLE_ASAN}"
    YAML_FORMAT "${HIPBLASLT_ENABLE_YAML}"
    NO_COMPRESS "${TENSILELITE_NO_COMPRESS}"
    EXPERIMENTAL "${TENSILELITE_EXPERIMENTAL}"
    LAZY_LOAD "${HIPBLASLT_ENABLE_LAZY_LOAD}"
    ASM_COMMENTS "${TENSILELITE_ENABLE_ASM_COMMENTS}"
    KEEP_BUILD_TMP "${TENSILELITE_KEEP_BUILD_TMP}"
    ASM_DEBUG "${TENSILELITE_ASM_DEBUG}"
)

if(HIPBLASLT_ENABLE_EXTOPS)
    add_subdirectory(extops)
endif()
if(HIPBLASLT_ENABLE_MATRIX_TRANSFORM)
    add_subdirectory(matrix-transform)
endif()
