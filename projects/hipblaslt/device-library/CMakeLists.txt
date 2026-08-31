# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

include(hipblaslt_codegen)

set(TENSILELITE_BUILD_PARALLEL_LEVEL "" CACHE STRING
    "Number of CPU cores to use for building device libraries (uses nproc if unset).")
set(TENSILELITE_KEEP_BUILD_TMP OFF CACHE BOOL
    "Keep the temporary device-library build directory.")
set(TENSILELITE_ASM_DEBUG OFF CACHE BOOL
    "Keep debug information in generated device libraries.")
set(TENSILELITE_LOGIC_FILTER "" CACHE STRING
    "Glob used to select logic files; empty selects all logic files.")
set(TENSILELITE_NO_COMPRESS OFF CACHE BOOL
    "Do not compress device code object files.")
set(TENSILELITE_EXPERIMENTAL OFF CACHE BOOL
    "Process experimental logic files.")
set(TENSILELITE_ENABLE_ASM_COMMENTS OFF CACHE BOOL
    "Emit comments in generated assembly.")
set(TENSILELITE_OFFLOADBUNDLER "" CACHE FILEPATH
    "Path to clang-offload-bundler.")
set(TENSILELITE_LIBRARY_FORMAT "msgpack" CACHE STRING
    "Device library serialization format (msgpack or yaml).")
set_property(CACHE TENSILELITE_LIBRARY_FORMAT PROPERTY STRINGS msgpack yaml)
set(Tensile_NO_LAZY_LIBRARY_LOADING OFF CACHE BOOL
    "Disable lazy loading of device libraries.")

set(HIPBLASLT_LIBLOGIC_PATH "" CACHE STRING "Path to library logic files (will use 'library' if unset).")
set(HIPBLASLT_TENSILE_LIBPATH "${hipblaslt_BINARY_DIR}/Tensile" CACHE STRING "Path to output the device gemm libraries.")

set(HIPBLASLT_ASIC_REVISION "" CACHE STRING "gfx1250 ASIC revisions to build ('' = both v0 and v1, 'v1' = v1 only, 'v0' = v0 only).")

# This policy belongs to the caller that owns the logic corpus, not to the
# gfx1250v0 architecture name itself. hipSPARSELt reuses this device-library
# CMakeLists with a corpus that deliberately has no v0 overlay, while
# hipBLASLt's own corpus must retain one. The embedding hipSPARSELt CMakeLists
# overrides this to OFF before adding hipBLASLt as a subdirectory.
if(NOT DEFINED HIPBLASLT_REQUIRE_GFX1250V0_OVERLAY)
    set(HIPBLASLT_REQUIRE_GFX1250V0_OVERLAY ON)
endif()

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
set(_main_requires_gfx1250v0_overlay OFF)
if("gfx1250" IN_LIST GPU_TARGETS)
    if(HIPBLASLT_ASIC_REVISION STREQUAL "v0")
        list(TRANSFORM _main_arches REPLACE "^gfx1250$" "gfx1250v0")  # v0 in place of v1
        set(_main_requires_gfx1250v0_overlay ${HIPBLASLT_REQUIRE_GFX1250V0_OVERLAY})
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

create_device_library(
    LOGIC_PATH "${HIPBLASLT_LIBLOGIC_PATH}"
    OUTPUT_DIR "${HIPBLASLT_TENSILE_LIBPATH}"
    CODEGEN_ROOT "${hipblaslt_SOURCE_DIR}/tensilelite"
    PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
    ARCHES ${_main_arches}
    REQUIRE_GFX1250V0_OVERLAY ${_main_requires_gfx1250v0_overlay}
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

    create_device_library(
        TARGET tensilelite-device-libraries-gfx1250v0
        LOGIC_PATH "${HIPBLASLT_LIBLOGIC_PATH}"
        OUTPUT_DIR "${_gfx1250v0_output_dir}"
        CODEGEN_ROOT "${hipblaslt_SOURCE_DIR}/tensilelite"
        PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
        ARCHES gfx1250v0
        # hipBLASLt owns the gfx1250/gfx1250v0 split for its own corpus, but
        # hipSPARSELt reuses this CMakeLists with a no-split corpus. Let the
        # embedding caller choose whether this target requires the directory.
        REQUIRE_GFX1250V0_OVERLAY ${HIPBLASLT_REQUIRE_GFX1250V0_OVERLAY}
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
