# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# ============================================================================
# AOT catalog engine -- family library build system
# ============================================================================
# Build-backend-agnostic. Each library/<arch>/<family>/ folder is fully self-
# contained: a checked-in family.json (the spec: kernels + constraints + ABI/
# grid), a per-family CMakeLists.txt that does whatever build-time work that
# family needs, and -- for rocKE-AOT families -- a co-located producer script.
#
#   * rocKE-AOT family: the CMakeLists calls aot_add_rocke_family(), which runs
#     the producer at build time to COMPILE the .co from the spec. We do NOT
#     check those .co into git (they are churning build products); TheRock
#     packages the built output via install().
#   * prebuilt-.co / AITER-ASM family (future): the CMakeLists would just stage
#     checked-in .co -- no producer, no rocKE. (Add aot_add_prebuilt_family()
#     when the first such family lands; see the engine README.)
#
# rocKE is an OPTIONAL, PER-FAMILY dependency -- never a system dependency. If
# ROCKE_PYTHON_DIR is unset/invalid, rocKE-AOT families are skipped with a
# warning; the engine still builds and its GPU parity tests skip on an empty
# catalog. A build with only prebuilt families needs no rocKE at all.
#
# Families are AUTO-DISCOVERED: any library/<arch>/<family>/CMakeLists.txt is
# picked up, so a kernel author adds a new arch/kernel set just by dropping in a
# folder (e.g. gfx942/sdpa_dense/).

# AOT_CATALOG_BUILD_DIR is set by the parent (engine) CMakeLists and names the
# build-tree catalog root the runtime/tests read (<root>/<arch>/<family>/...).
if(NOT DEFINED AOT_CATALOG_BUILD_DIR)
    message(FATAL_ERROR "aot-catalog: AOT_CATALOG_BUILD_DIR not set by parent")
endif()

# ---------------------------------------------------------------------------
# rocKE availability (resolved once; per-family CMakeLists read _AOT_ROCKE_OK).
# ---------------------------------------------------------------------------
set(ROCKE_PYTHON_DIR "" CACHE PATH
    "Path to rocKE's ck_dsl python package dir (.../projects/composablekernel/python). \
Required ONLY to build rocKE-AOT kernel families; leave empty to skip them.")
set(ROCKE_COMGR_LIB "" CACHE FILEPATH
    "Optional path to libamd_comgr.so for the rocKE producer (else it uses \
CK_DSL_COMGR_LIB from the environment or /opt/rocm).")

find_package(Python3 COMPONENTS Interpreter QUIET)

set(_AOT_ROCKE_OK TRUE)
if(NOT ROCKE_PYTHON_DIR OR NOT EXISTS "${ROCKE_PYTHON_DIR}/ck_dsl")
    set(_AOT_ROCKE_OK FALSE)
    message(WARNING
        "aot-catalog: ROCKE_PYTHON_DIR not set or invalid -- rocKE-AOT kernel "
        "families will be SKIPPED (engine still builds; GPU parity tests skip on "
        "the empty catalog). Set -DROCKE_PYTHON_DIR=<rocKE>/projects/composablekernel/python "
        "to build kernels.")
elseif(NOT Python3_Interpreter_FOUND)
    set(_AOT_ROCKE_OK FALSE)
    message(WARNING "aot-catalog: Python3 interpreter not found -- rocKE-AOT families skipped.")
endif()

# The producer compiles .co in-process via libamd_comgr.so (ck_dsl's comgr runtime),
# NOT hipcc. Resolve it the SAME way the producer will at build time -- explicit
# ROCKE_COMGR_LIB, else $CK_DSL_COMGR_LIB from the environment, else the ROCm default
# -- so a box that has rocKE's python tree but NO comgr SKIPS the families (build
# still succeeds) instead of hard-failing when the producer throws mid-build.
if(_AOT_ROCKE_OK)
    set(_aot_comgr "")
    if(ROCKE_COMGR_LIB AND EXISTS "${ROCKE_COMGR_LIB}")
        set(_aot_comgr "${ROCKE_COMGR_LIB}")
    elseif(DEFINED ENV{CK_DSL_COMGR_LIB} AND EXISTS "$ENV{CK_DSL_COMGR_LIB}")
        set(_aot_comgr "$ENV{CK_DSL_COMGR_LIB}")
    else()
        find_library(_AOT_COMGR_LIB NAMES amd_comgr
            HINTS "$ENV{ROCM_PATH}/lib" "$ENV{ROCM_PATH}/lib64" /opt/rocm/lib /opt/rocm/lib64)
        if(_AOT_COMGR_LIB)
            set(_aot_comgr "${_AOT_COMGR_LIB}")
        endif()
    endif()
    if(NOT _aot_comgr)
        set(_AOT_ROCKE_OK FALSE)
        message(WARNING
            "aot-catalog: rocKE python found but libamd_comgr.so was not (checked "
            "ROCKE_COMGR_LIB, \$CK_DSL_COMGR_LIB, and the ROCm lib dirs) -- rocKE-AOT "
            "families SKIPPED (engine still builds; GPU parity tests skip on the empty "
            "catalog). Set -DROCKE_COMGR_LIB=<path/to/libamd_comgr.so> to build kernels.")
    endif()
endif()

# The install root for the packaged catalog (mirrors the engine's baked default).
set(_AOT_CATALOG_INSTALL_DIR
    "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/aot_catalog")

# ---------------------------------------------------------------------------
# _aot_arch_requested(<arch> <out_var>) -- is <arch> in this build's target set?
#
# TheRock/ROCm pass the requested GPU targets in GPU_TARGETS (AMDGPU_TARGETS is
# the legacy alias); a family builds only when its arch is among them, so a
# gfx942-only build does NOT compile the gfx1151 kernels (and vice versa). Both
# the family arch and each target are normalized to their gfxNNN base -- feature
# suffixes like ":xnack-" are dropped -- mirroring the runtime device match in
# hip_kernel_provider_common::getDeviceString. When NEITHER list is set (e.g. a
# standalone dev build that never specified targets), ALL families build: the
# original behavior, so nothing silently disappears.
#
# Any future family adder (e.g. aot_add_prebuilt_family) should gate on this too.
# ---------------------------------------------------------------------------
function(_aot_arch_requested arch out_var)
    set(_targets "${GPU_TARGETS}")
    if(NOT _targets)
        set(_targets "${AMDGPU_TARGETS}")
    endif()
    if(NOT _targets)
        set(${out_var} TRUE PARENT_SCOPE) # no target list -> build everything
        return()
    endif()
    string(REGEX REPLACE ":.*$" "" _arch_base "${arch}")
    foreach(_t IN LISTS _targets)
        string(REGEX REPLACE ":.*$" "" _t_base "${_t}")
        if(_t_base STREQUAL _arch_base)
            set(${out_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_var} FALSE PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# aot_add_rocke_family(ARCH <arch> FAMILY <family> PRODUCER <script.py>)
#
# Runs the co-located producer at build time to emit the family's .co into
# ${AOT_CATALOG_BUILD_DIR}/<arch>/<family>/, stages the checked-in family.json
# beside them, and installs the folder for packaging. No-op (with a STATUS note)
# when the arch is not in this build's GPU_TARGETS or when rocKE is unavailable.
# ---------------------------------------------------------------------------
function(aot_add_rocke_family)
    cmake_parse_arguments(F "" "ARCH;FAMILY;PRODUCER" "" ${ARGN})
    if(NOT F_ARCH OR NOT F_FAMILY OR NOT F_PRODUCER)
        message(FATAL_ERROR "aot_add_rocke_family: ARCH, FAMILY and PRODUCER are required")
    endif()

    set(_src ${CMAKE_CURRENT_SOURCE_DIR})
    set(_out ${AOT_CATALOG_BUILD_DIR}/${F_ARCH}/${F_FAMILY})

    # Skip archs this build didn't ask for BEFORE touching rocKE -- an unwanted
    # arch must not build even where rocKE is available.
    _aot_arch_requested(${F_ARCH} _arch_ok)
    if(NOT _arch_ok)
        message(STATUS
            "aot-catalog: skipping ${F_ARCH}/${F_FAMILY} "
            "(arch not in GPU_TARGETS/AMDGPU_TARGETS for this build)")
        return()
    endif()

    if(NOT _AOT_ROCKE_OK)
        message(STATUS "aot-catalog: skipping rocKE family ${F_ARCH}/${F_FAMILY} (rocKE unavailable)")
        return()
    endif()

    # Env for the producer: PYTHONPATH -> ck_dsl; optional comgr override.
    set(_env PYTHONPATH=${ROCKE_PYTHON_DIR})
    if(ROCKE_COMGR_LIB)
        list(APPEND _env CK_DSL_COMGR_LIB=${ROCKE_COMGR_LIB})
    endif()

    set(_stamp ${CMAKE_CURRENT_BINARY_DIR}/${F_FAMILY}.stamp)
    add_custom_command(
        OUTPUT ${_stamp}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_out}
        COMMAND ${CMAKE_COMMAND} -E env ${_env}
                ${Python3_EXECUTABLE} ${_src}/${F_PRODUCER} ${_out}
        COMMAND ${CMAKE_COMMAND} -E copy ${_src}/family.json ${_out}/family.json
        COMMAND ${CMAKE_COMMAND} -E touch ${_stamp}
        DEPENDS ${_src}/${F_PRODUCER} ${_src}/family.json
        COMMENT "aot-catalog: building ${F_ARCH}/${F_FAMILY} kernels via rocKE"
        VERBATIM
    )
    add_custom_target(aot_family_${F_ARCH}_${F_FAMILY} ALL DEPENDS ${_stamp}
        COMMENT "aot-catalog: ${F_ARCH}/${F_FAMILY} family (kernels + family.json)")

    # Record the target so consumers (e.g. the test binary) can depend on the whole
    # set -- an isolated `ninja hip_kernel_provider_tests` must build the catalog
    # first, else the GPU parity tests silently skip on an empty build-tree catalog.
    set_property(GLOBAL APPEND PROPERTY
        AOT_CATALOG_FAMILY_TARGETS aot_family_${F_ARCH}_${F_FAMILY})

    # cmake-lint: disable=E1122
    install(DIRECTORY ${_out}/
            DESTINATION ${_AOT_CATALOG_INSTALL_DIR}/${F_ARCH}/${F_FAMILY}
            FILES_MATCHING PATTERN "*.co" PATTERN "family.json")
endfunction()

# ---------------------------------------------------------------------------
# Auto-discover every library/<arch>/<family>/CMakeLists.txt.
# ---------------------------------------------------------------------------
file(GLOB _family_cmakes RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/*/*/CMakeLists.txt)
foreach(_fc ${_family_cmakes})
    get_filename_component(_fdir ${_fc} DIRECTORY)
    add_subdirectory(${_fdir})
endforeach()
