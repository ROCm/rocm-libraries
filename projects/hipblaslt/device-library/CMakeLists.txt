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

set(HIPBLASLT_ASIC_REVISION "" CACHE STRING "gfx1250 ASIC revisions to build ('' = both v0 and v1, 'v1' = v1 only, 'v0' = v0 only).")

# Reject an unrecognized value rather than silently building the wrong revisions.
if(NOT HIPBLASLT_ASIC_REVISION MATCHES "^(v0|v1)?$")
    message(FATAL_ERROR "HIPBLASLT_ASIC_REVISION must be '', 'v0' or 'v1', got '${HIPBLASLT_ASIC_REVISION}'")
endif()

if(NOT HIPBLASLT_LIBLOGIC_PATH)
    set(HIPBLASLT_LIBLOGIC_PATH "${hipblaslt_SOURCE_DIR}/library")
endif()

# gfx1250's two ASIC revisions (v0/v1) need different tuned kernels; the runtime
# picks library/gfx1250/ (v1) vs library/gfx1250v0/ by asicRevision. The default
# builds both (v1 here, v0 by the second command below); HIPBLASLT_ASIC_REVISION
# only prunes which. GPU_TARGETS is untouched (it also feeds extops/transform/
# validation on the base name); only the bare gfx1250 reaches here.
set(_main_arches ${GPU_TARGETS})
set(_build_gfx1250v0 OFF)
if("gfx1250" IN_LIST GPU_TARGETS)
    if(HIPBLASLT_ASIC_REVISION STREQUAL "v0")
        list(TRANSFORM _main_arches REPLACE "^gfx1250$" "gfx1250v0")  # v0 in place of v1
        message(STATUS "hipBLASLt: gfx1250 ASIC revision v0 only")
    elseif(HIPBLASLT_ASIC_REVISION STREQUAL "v1")
        message(STATUS "hipBLASLt: gfx1250 ASIC revision v1 only")
    else()
        set(_build_gfx1250v0 ON)  # both
        message(STATUS "hipBLASLt: gfx1250 ASIC revisions v0 and v1 (runtime selects by asicRevision)")
    endif()
elseif(HIPBLASLT_ASIC_REVISION STREQUAL "v0")
    message(WARNING "HIPBLASLT_ASIC_REVISION=v0 but GPU_TARGETS names no gfx1250 "
                    "(${GPU_TARGETS}); nothing in this build is for the v0 revision.")
endif()

hipblaslt_create_device_library(
    TARGET tensilelite-device-libraries
    LOGIC_PATH "${HIPBLASLT_LIBLOGIC_PATH}"
    OUTPUT_DIR "${HIPBLASLT_TENSILE_LIBPATH}"
    ARCHES ${_main_arches}
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

if(_build_gfx1250v0)
    # Needs its own OUTPUT_DIR: the two runs share a compiler target and each
    # writes Kernels.cpp + a build_tmp tree under its output dir (reusing a .o
    # that is already there), so a shared dir would serialize them or cross-reuse
    # objects. Installed by the top-level CMakeLists.txt via the property below.
    set(_gfx1250v0_output_dir "${HIPBLASLT_TENSILE_LIBPATH}-gfx1250v0")
    file(MAKE_DIRECTORY "${_gfx1250v0_output_dir}/library/gfx1250v0")
    set_property(GLOBAL PROPERTY HIPBLASLT_GFX1250V0_LIBRARY_DIR
                 "${_gfx1250v0_output_dir}/library/gfx1250v0")

    hipblaslt_create_device_library(
        TARGET tensilelite-device-libraries-gfx1250v0
        LOGIC_PATH "${HIPBLASLT_LIBLOGIC_PATH}"
        OUTPUT_DIR "${_gfx1250v0_output_dir}"
        ARCHES gfx1250v0
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
endif()

if(HIPBLASLT_ENABLE_EXTOPS)
    add_subdirectory(extops)
endif()
if(HIPBLASLT_ENABLE_MATRIX_TRANSFORM)
    add_subdirectory(matrix-transform)
endif()
