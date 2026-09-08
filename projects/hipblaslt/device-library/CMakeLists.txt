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

# gfx1250-strict is selected like any other target, but it cannot share a
# TensileCreateLibrary run with gfx1250: Tensile identifies a target by the three
# integers it parses out of the gfx name, so both spell ISA 12.5.0 and one run can
# only describe one of them. Peel it into the second command, which gets its own
# OUTPUT_DIR.
set(_main_arches ${GPU_TARGETS})
set(_build_gfx1250_strict OFF)
if("gfx1250-strict" IN_LIST GPU_TARGETS)
    list(REMOVE_ITEM _main_arches "gfx1250-strict")
    set(_build_gfx1250_strict ON)
endif()

if(_main_arches)
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
endif()

if(_build_gfx1250_strict)
    # Needs its own OUTPUT_DIR: the two runs share a compiler target and each
    # writes Kernels.cpp + a build_tmp tree under its output dir (reusing a .o
    # that is already there), so a shared dir would serialize them or cross-reuse
    # objects. Installed by the top-level CMakeLists.txt via the property below.
    set(_gfx1250_strict_output_dir "${HIPBLASLT_TENSILE_LIBPATH}-gfx1250-strict")
    file(MAKE_DIRECTORY "${_gfx1250_strict_output_dir}/library/gfx1250-strict")
    set_property(GLOBAL PROPERTY HIPBLASLT_GFX1250_STRICT_LIBRARY_DIR
                 "${_gfx1250_strict_output_dir}/library/gfx1250-strict")

    hipblaslt_create_device_library(
        TARGET tensilelite-device-libraries-gfx1250-strict
        LOGIC_PATH "${HIPBLASLT_LIBLOGIC_PATH}"
        OUTPUT_DIR "${_gfx1250_strict_output_dir}"
        ARCHES gfx1250-strict
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
