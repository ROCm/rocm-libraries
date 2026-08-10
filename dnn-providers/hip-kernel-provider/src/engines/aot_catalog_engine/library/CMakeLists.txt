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
    "Path to rocKE's rocke python package dir (.../hip-kernel-provider/rocke/platform/python). \
Required ONLY to build rocKE-AOT kernel families; leave empty to skip them.")
set(ROCKE_COMGR_LIB "" CACHE FILEPATH
    "Optional path to libamd_comgr.so for the rocKE producer (else it uses \
ROCKE_COMGR_LIB from the environment or /opt/rocm).")

find_package(Python3 COMPONENTS Interpreter QUIET)

set(_AOT_ROCKE_OK TRUE)
if(NOT ROCKE_PYTHON_DIR OR NOT EXISTS "${ROCKE_PYTHON_DIR}/rocke")
    set(_AOT_ROCKE_OK FALSE)
    message(WARNING
        "aot-catalog: ROCKE_PYTHON_DIR not set or invalid -- rocKE-AOT kernel "
        "families will be SKIPPED (engine still builds; GPU parity tests skip on "
        "the empty catalog). Set -DROCKE_PYTHON_DIR=<hip-kernel-provider>/rocke/platform/python "
        "to build kernels.")
elseif(NOT Python3_Interpreter_FOUND)
    set(_AOT_ROCKE_OK FALSE)
    message(WARNING "aot-catalog: Python3 interpreter not found -- rocKE-AOT families skipped.")
endif()

# The producer compiles .co in-process via libamd_comgr.so (rocke's comgr runtime),
# NOT hipcc. Resolve it the SAME way the producer will at build time -- explicit
# ROCKE_COMGR_LIB, else $ROCKE_COMGR_LIB from the environment, else the ROCm default
# -- so a box that has rocKE's python tree but NO comgr SKIPS the families (build
# still succeeds) instead of hard-failing when the producer throws mid-build.
if(_AOT_ROCKE_OK)
    set(_aot_comgr "")
    if(ROCKE_COMGR_LIB AND EXISTS "${ROCKE_COMGR_LIB}")
        set(_aot_comgr "${ROCKE_COMGR_LIB}")
    elseif(DEFINED ENV{ROCKE_COMGR_LIB} AND EXISTS "$ENV{ROCKE_COMGR_LIB}")
        set(_aot_comgr "$ENV{ROCKE_COMGR_LIB}")
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
            "ROCKE_COMGR_LIB, \$ROCKE_COMGR_LIB, and the ROCm lib dirs) -- rocKE-AOT "
            "families SKIPPED (engine still builds; GPU parity tests skip on the empty "
            "catalog). Set -DROCKE_COMGR_LIB=<path/to/libamd_comgr.so> to build kernels.")
    endif()
endif()

# The install root for the packaged catalog (mirrors the engine's baked default).
set(_AOT_CATALOG_INSTALL_DIR
    "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/aot_catalog")

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

    # Guard a partial/half-added family folder. The folder was auto-discovered by
    # its CMakeLists alone, but the build-time custom command below needs BOTH the
    # producer script and the checked-in family.json; if either is missing the
    # command fails cryptically mid-build (python can't find the producer, or the
    # family.json copy errors). Warn and skip -- mirroring the rocKE-absence handling
    # above -- so an incomplete folder never breaks the rest of the catalog build.
    if(NOT EXISTS ${_src}/${F_PRODUCER})
        message(WARNING
            "aot-catalog: skipping ${F_ARCH}/${F_FAMILY} -- producer script "
            "'${F_PRODUCER}' not found in ${_src} (partial family folder?)")
        return()
    endif()
    if(NOT EXISTS ${_src}/family.json)
        message(WARNING
            "aot-catalog: skipping ${F_ARCH}/${F_FAMILY} -- family.json not found "
            "in ${_src} (partial family folder?)")
        return()
    endif()

    # Env for the producer: PYTHONPATH -> rocke; optional comgr override.
    set(_env PYTHONPATH=${ROCKE_PYTHON_DIR})
    if(ROCKE_COMGR_LIB)
        list(APPEND _env ROCKE_COMGR_LIB=${ROCKE_COMGR_LIB})
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
# aot_add_family_test(ARCH <arch> SOURCES <test.cpp> [<test2.cpp> ...])
#
# A family declares its OWN tests here, co-located in the family folder, so the
# family folder is a fully self-contained unit: family.json (the spec) + the
# CMakeLists that builds its .co + the tests that validate those .co, all in one
# place. Deleting library/<arch>/<family>/ removes the kernels, the spec AND the
# tests together -- nothing about that family is left dangling elsewhere.
#
# The named sources are appended (as absolute paths) to the
# AOT_CATALOG_FAMILY_TEST_SOURCES global property, which src/tests/CMakeLists.txt
# compiles into hip_kernel_provider_tests (that target owns the include dirs and
# the AOT_CATALOG_TEST_DIR compile def these sources need). Gated on the same
# _aot_arch_requested() check as the kernel build, so a family's tests compile
# only when its arch is in this build's target set.
#
# CONTRACT: a family's tests must depend ONLY on that family -- no test here may
# reference another family's kernels or symbols, so removing any one family can
# never break another's tests. Cross-family or engine-substrate tests (e.g. the
# tune-cache and SDPA-decode unit tests) do NOT belong in a family folder; they
# live at the engine test level (src/tests/engines/aot_catalog_engine/).
# ---------------------------------------------------------------------------
function(aot_add_family_test)
    cmake_parse_arguments(T "" "ARCH" "SOURCES" ${ARGN})
    if(NOT T_ARCH OR NOT T_SOURCES)
        message(FATAL_ERROR "aot_add_family_test: ARCH and SOURCES are required")
    endif()

    _aot_arch_requested(${T_ARCH} _arch_ok)
    if(NOT _arch_ok)
        return() # arch not in this build's target set -> its tests don't compile
    endif()

    foreach(_s IN LISTS T_SOURCES)
        set_property(GLOBAL APPEND PROPERTY
            AOT_CATALOG_FAMILY_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${_s})
    endforeach()
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
