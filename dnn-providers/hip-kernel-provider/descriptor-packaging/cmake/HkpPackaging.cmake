# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Build-time hip UKD -> compile -> prune -> kpack packaging for the
# hip-kernel-provider. All functions are provider-internal and namespaced hkp_*.
# hkp = Hip Kernel-provider Packaging; the hkp_/HKP_ prefixes mark internal
# symbols of this kpack-packaging module.

include_guard(GLOBAL)

# Captured at include time so it survives into functions: inside a function
# CMAKE_CURRENT_LIST_DIR reflects the invoking listfile, not this module.
set(HKP_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(HKP_PKG_DIR "${CMAKE_CURRENT_LIST_DIR}/..")
set(HKP_PYTHON_ROOT "${HKP_PKG_DIR}/python")
set(HKP_TOOL "${HKP_PKG_DIR}/tools/hkp_pack.py")
set(HKP_FIXTURES "${HKP_PKG_DIR}/tests/fixtures")

# rocm-kpack source for the FetchContent tiers of hkp_resolve_kpack. The default
# ref is pinned to a known-good SHA for reproducible builds (the tool depends on
# rocm_kpack's prepare_kernel/get_kernel API). Override the ref to test a newer
# kpack: set HIPKERNELPROVIDER_KPACK_GIT_REF to any SHA, tag, or branch (e.g.
# "main"), and HIPKERNELPROVIDER_KPACK_GIT_REPO to fetch from a fork. A branch
# ref is a moving target: FetchContent re-fetches it only on a clean populate
# (a wiped _deps/rocm_kpack-* or build dir) — `cmake --fresh` is NOT sufficient,
# as it clears the cache but leaves _deps/ intact. Pin to a specific newer SHA
# for a deterministic re-fetch.
set(HIPKERNELPROVIDER_KPACK_GIT_REPO "https://github.com/ROCm/rocm-kpack.git"
    CACHE STRING "rocm-kpack git repository to fetch (override for a fork).")
set(HIPKERNELPROVIDER_KPACK_GIT_REF "e3483286e751060b3a70b792792cc122632c66e8"
    CACHE STRING "rocm-kpack git ref (SHA, tag, or branch) to fetch. Defaults \
to a pinned SHA for reproducibility; set to a branch or newer SHA to test the \
latest tool.")

# ---------------------------------------------------------------------------
# hkp_resolve_kpack(<out_var>)
#   3-tier resolution of the rocm-kpack 'python' directory:
#   (1) -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR override,
#   (2)/(3) FetchContent of a pinned rocm-kpack commit. Sets <out_var> to the
#   resolved python dir. rocm_kpack is load-bearing (the tool cannot pack
#   without it), so an unresolvable dependency is a hard error.
# ---------------------------------------------------------------------------
function(hkp_resolve_kpack out_var)
    if(DEFINED HIPKERNELPROVIDER_KPACK_PYTHON_DIR AND EXISTS "${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}")
        set(${out_var} "${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}" PARENT_SCOPE)
        message(STATUS "hkp: using rocm_kpack from HIPKERNELPROVIDER_KPACK_PYTHON_DIR=${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}")
        return()
    endif()

    # Tiers 2/3: FetchContent of rocm-kpack at the repo/ref configured at the top
    # of this module (HIPKERNELPROVIDER_KPACK_GIT_REPO/REF). Only the python/ tree
    # is consumed (download-only, never configured), so a bare populate is used.
    set(_kpack_repo "${HIPKERNELPROVIDER_KPACK_GIT_REPO}")
    set(_kpack_tag "${HIPKERNELPROVIDER_KPACK_GIT_REF}")
    message(STATUS "hkp: fetching rocm_kpack ${_kpack_repo}@${_kpack_tag}")
    include(FetchContent)
    FetchContent_Declare(
        rocm_kpack
        GIT_REPOSITORY "${_kpack_repo}"
        GIT_TAG "${_kpack_tag}"
    )
    # CMP0169 (CMake >= 3.30) deprecates the single-arg FetchContent_Populate;
    # keep it valid since only the source tree is needed, not a configured build.
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_GetProperties(rocm_kpack)
    if(NOT rocm_kpack_POPULATED)
        FetchContent_Populate(rocm_kpack)
    endif()
    if(EXISTS "${rocm_kpack_SOURCE_DIR}/python/rocm_kpack/kpack.py")
        set(${out_var} "${rocm_kpack_SOURCE_DIR}/python" PARENT_SCOPE)
        message(STATUS "hkp: fetched rocm_kpack into ${rocm_kpack_SOURCE_DIR}/python")
        return()
    endif()

    message(FATAL_ERROR
        "hkp: rocm_kpack could not be resolved (override with "
        "HIPKERNELPROVIDER_KPACK_PYTHON_DIR or ensure the pinned commit is fetchable). "
        "rocm_kpack is required to pack; there is no skip path.")
endfunction()

# ---------------------------------------------------------------------------
# hkp_selected_arches(<out_var>)
#   Normalize GPU_TARGETS (or AMDGPU_TARGETS) into a bare gfx arch list,
#   stripping feature suffixes (gfx942:xnack-). Empty result is legal (install
#   nothing, non-error). No intersection with a fixed fixture set: the tool
#   compiles from authored sources for whatever arch is requested.
# ---------------------------------------------------------------------------
function(hkp_selected_arches out_var)
    set(_targets "")
    if(DEFINED GPU_TARGETS AND GPU_TARGETS)
        set(_targets ${GPU_TARGETS})
    elseif(DEFINED AMDGPU_TARGETS AND AMDGPU_TARGETS)
        set(_targets ${AMDGPU_TARGETS})
    endif()

    set(_selected "")
    foreach(_arch IN LISTS _targets)
        string(REGEX REPLACE ":.*$" "" _bare "${_arch}")
        if(_bare)
            list(APPEND _selected "${_bare}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _selected)
    set(${out_var} "${_selected}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_wire_production(<source_root> <arches> <hipcc> <kpack_python> <install_base>)
#   Wire the production compile -> prune -> pack -> install DAG against an
#   authored source root. The tool is invoked ONCE with the full arch list and
#   loops arches internally. A surviving arch installs its descriptors/<gfx>/
#   shard; a skipped arch produces no folder, so its install() is a no-op via
#   OPTIONAL. Empty arch selection wires nothing.
# ---------------------------------------------------------------------------
function(hkp_wire_production source_root arches hipcc kpack_python install_base)
    if(NOT arches)
        return()
    endif()
    if(NOT hipcc)
        message(FATAL_ERROR
            "hkp: hipcc not found (searched hipcc, hipcc.bat, hipcc.bin.exe); "
            "cannot compile kernels. Ensure the ROCm bin dir is on PATH.")
    endif()

    set(_out_root "${CMAKE_CURRENT_BINARY_DIR}/hkp-descriptors")
    set(_inter_root "${CMAKE_CURRENT_BINARY_DIR}/hkp-intermediate")
    string(REPLACE ";" "," _arch_csv "${arches}")
    set(_stamp "${_out_root}.stamp")
    file(GLOB _source_inputs CONFIGURE_DEPENDS
         "${source_root}/*.json" "${source_root}/*.cpp")
    # Editing the tool's own sources must retrigger the pack step, else the
    # install artifacts go stale against the current pipeline code.
    file(GLOB _tool_sources CONFIGURE_DEPENDS "${HKP_PYTHON_ROOT}/hkp_pack/*.py")

    add_custom_command(
        OUTPUT "${_stamp}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_out_root}"
        COMMAND "${Python3_EXECUTABLE}" "${HKP_TOOL}"
                --source-root "${source_root}"
                --out-root "${_out_root}"
                --arches "${_arch_csv}"
                --hipcc "${hipcc}"
                --inter-root "${_inter_root}"
                --kpack-python-dir "${kpack_python}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
        DEPENDS "${HKP_TOOL}" ${_source_inputs} ${_tool_sources}
        COMMENT "hkp: compiling + pruning + packing descriptors for ${arches}"
        VERBATIM)

    add_custom_target(hkp_descriptor_packaging ALL DEPENDS "${_stamp}"
                      COMMENT "hkp: descriptor packaging")

    # The tool writes only the shippable tree (kpack-form descriptor JSON + the
    # kpack/ subfolder) to _out_root; install it wholesale. A skipped arch
    # produces no folder, so OPTIONAL makes its install a no-op.
    foreach(_arch IN LISTS arches)
        install(DIRECTORY "${_out_root}/${_arch}/"
                DESTINATION "${install_base}/${_arch}"
                OPTIONAL)
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# hkp_add_packaging()
#   Two independent gates. The production pack+install wires only when
#   HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT names an existing authored source
#   folder (empty default = dormant; set-but-missing = fatal). The tests are
#   wired regardless: they drive the fixture slice directly, never the
#   production source root, so fixtures never reach a production build.
# ---------------------------------------------------------------------------
function(hkp_add_packaging)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    hkp_resolve_kpack(_kpack_python)
    hkp_selected_arches(_arches)

    # hipcc is the perl/bat driver that honors --genco; on Windows it is
    # hipcc.exe or hipcc.bat. hipcc.bin.exe is the raw clang driver and is only
    # a last-resort fallback. The compiler is load-bearing: absence is fatal
    # whenever there is anything to pack.
    find_program(HKP_HIPCC NAMES hipcc hipcc.bat hipcc.bin.exe)

    set(_install_base
        "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/arch_content/hip-kernel-provider/descriptors")

    # Second gate: production only compiles+installs from an authored source
    # folder named here. The test fixtures are never the production source.
    set(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT "" CACHE PATH
        "Authored hip-source root the production pack step compiles from. \
Empty leaves production packaging dormant; set to a real authored source folder \
for a release build.")
    if(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT)
        if(NOT IS_DIRECTORY "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT is set but is not "
                "a directory: ${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
        endif()
        hkp_wire_production("${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}"
                            "${_arches}" "${HKP_HIPCC}" "${_kpack_python}"
                            "${_install_base}")
    else()
        message(STATUS
            "hkp: HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT empty; production "
            "packaging dormant (tests still run against the fixtures).")
    endif()

    hkp_register_tests("${_kpack_python}" "${HKP_HIPCC}")
endfunction()

# ---------------------------------------------------------------------------
# hkp_register_tests(<kpack_python> <hipcc>)
#   The pytest ctest entry (real compile -> prune -> pack -> rewrite, over the
#   fixture slice). It compiles for real; there is no skip path. The shipped
#   tree's shape (kpack-form UKDs, no loose .co, empty-arch skip) is asserted at
#   the tool-output level in pytest; the install() rule itself is a stock
#   recursive install(DIRECTORY) and is not separately staged.
# ---------------------------------------------------------------------------
function(hkp_register_tests kpack_python hipcc)
    if(NOT HIPKERNELPROVIDER_ENABLE_TESTS)
        return()
    endif()
    if(NOT hipcc)
        message(FATAL_ERROR
            "hkp: HIPKERNELPROVIDER_ENABLE_TESTS is on but hipcc was not found; "
            "the suite compiles kernels for real via --genco and cannot skip.")
    endif()

    add_test(NAME hkp_pack_pytest
             COMMAND "${Python3_EXECUTABLE}" -m pytest "${HKP_PKG_DIR}/tests" -v)
    # ENVIRONMENT is a ';'-joined list of VAR=VALUE entries; the resolved paths
    # here use OS-native separators (no embedded ';'), so the join is safe.
    set(_pyenv "PYTHONPATH=${HKP_PYTHON_ROOT}" "HKP_HIPCC=${hipcc}")
    if(kpack_python)
        list(APPEND _pyenv "HIPKERNELPROVIDER_KPACK_PYTHON_DIR=${kpack_python}")
    endif()
    set_tests_properties(hkp_pack_pytest PROPERTIES
        LABELS "unit_test;hip-kernel-provider;quick;host"
        ENVIRONMENT "${_pyenv}")
endfunction()
