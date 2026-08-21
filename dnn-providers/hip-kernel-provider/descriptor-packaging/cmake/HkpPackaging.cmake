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
#   (1) -DHIPKERNELPROVIDER_ROCM_KPACK_DIR override,
#   (2)/(3) FetchContent of a pinned rocm-kpack commit. Sets <out_var> to the
#   resolved python dir. rocm_kpack is load-bearing (the tool cannot pack
#   without it), so an unresolvable dependency is a hard error.
# ---------------------------------------------------------------------------
function(hkp_resolve_kpack out_var)
    if(DEFINED HIPKERNELPROVIDER_ROCM_KPACK_DIR AND EXISTS "${HIPKERNELPROVIDER_ROCM_KPACK_DIR}")
        set(${out_var} "${HIPKERNELPROVIDER_ROCM_KPACK_DIR}" PARENT_SCOPE)
        message(STATUS "hkp: using rocm_kpack from HIPKERNELPROVIDER_ROCM_KPACK_DIR=${HIPKERNELPROVIDER_ROCM_KPACK_DIR}")
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
        "HIPKERNELPROVIDER_ROCM_KPACK_DIR or ensure the pinned commit is fetchable). "
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
# hkp_wire_production(HIP_ROOT <dir> ROCKE_ROOT <dir> ARCHES <list> HIPCC <path>
#                     ROCM_KPACK_DIR <dir> INSTALL_BASE <dir> ROCKE_INTERP <path>
#                     ROCKE_COMGR_LIB <path>)
#   Wire the production compile -> prune -> pack -> install DAG against the
#   enabled authored source roots. ONE tool process is invoked with a repeatable
#   --source-root per enabled root (either may be empty), merging both producers
#   into one kpack per arch. The hip root globs *.json + *.cpp for its co-located
#   sources; the rocke root globs *.json only. When a rocke root is enabled the
#   tool runs under ROCKE_INTERP (wheel-provisioned so `import rocke`/`kernels`
#   resolve) and ROCKE_COMGR_LIB, if set, is forwarded to the tool environment.
#   Empty ARCHES wires nothing.
# ---------------------------------------------------------------------------
function(hkp_wire_production)
    set(_one HIP_ROOT ROCKE_ROOT ARCHES HIPCC ROCM_KPACK_DIR INSTALL_BASE
        ROCKE_INTERP ROCKE_COMGR_LIB)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${_one}" "")

    if(NOT ARG_ARCHES)
        return()
    endif()

    set(_out_root "${CMAKE_CURRENT_BINARY_DIR}/hkp-descriptors")
    set(_inter_root "${CMAKE_CURRENT_BINARY_DIR}/hkp-intermediate")
    string(REPLACE ";" "," _arch_csv "${ARG_ARCHES}")
    set(_stamp "${_out_root}.stamp")

    set(_src_args "")
    set(_source_inputs "")
    if(ARG_HIP_ROOT)
        list(APPEND _src_args --source-root "${ARG_HIP_ROOT}")
        file(GLOB _hip_inputs CONFIGURE_DEPENDS
             "${ARG_HIP_ROOT}/*.json" "${ARG_HIP_ROOT}/*.cpp")
        list(APPEND _source_inputs ${_hip_inputs})
    endif()
    if(ARG_ROCKE_ROOT)
        list(APPEND _src_args --source-root "${ARG_ROCKE_ROOT}")
        file(GLOB _rocke_inputs CONFIGURE_DEPENDS "${ARG_ROCKE_ROOT}/*.json")
        list(APPEND _source_inputs ${_rocke_inputs})
    endif()

    # Editing the tool's own sources must retrigger the pack step, else the
    # install artifacts go stale against the current pipeline code.
    file(GLOB _tool_sources CONFIGURE_DEPENDS "${HKP_PYTHON_ROOT}/hkp_pack/*.py")

    # A rocke root needs the wheel-provisioned interpreter so the rocke UKDs
    # import; a hip-only pack runs under the base interpreter (hip compiles shell
    # out to hipcc and are interpreter-agnostic).
    set(_interp "${Python3_EXECUTABLE}")
    set(_interp_dep "")
    if(ARG_ROCKE_ROOT)
        set(_interp "${ARG_ROCKE_INTERP}")
        set(_interp_dep "${ARG_ROCKE_INTERP}")
    endif()

    # ROCKE_COMGR_LIB overrides a shadowed System32 amd_comgr on Windows; forward
    # it into the tool environment when set (runtime resolution, no find_library).
    set(_tool_cmd "${_interp}" "${HKP_TOOL}")
    if(ARG_ROCKE_COMGR_LIB)
        set(_tool_cmd "${CMAKE_COMMAND}" -E env
            "ROCKE_COMGR_LIB=${ARG_ROCKE_COMGR_LIB}" "${_interp}" "${HKP_TOOL}")
    endif()

    add_custom_command(
        OUTPUT "${_stamp}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_out_root}"
        COMMAND ${_tool_cmd}
                ${_src_args}
                --out-root "${_out_root}"
                --arches "${_arch_csv}"
                --hipcc "${ARG_HIPCC}"
                --inter-root "${_inter_root}"
                --rocm-kpack-dir "${ARG_ROCM_KPACK_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_stamp}"
        DEPENDS "${HKP_TOOL}" ${_source_inputs} ${_tool_sources} ${_interp_dep}
        COMMENT "hkp: compiling + pruning + packing descriptors for ${ARG_ARCHES}"
        VERBATIM)

    add_custom_target(hkp_descriptor_packaging ALL DEPENDS "${_stamp}"
                      COMMENT "hkp: descriptor packaging")

    # The tool writes only the shippable tree (kpack-form descriptor JSON + the
    # kpack/ subfolder) to _out_root; install it wholesale. A skipped arch
    # produces no folder, so OPTIONAL makes its install a no-op.
    foreach(_arch IN LISTS ARG_ARCHES)
        install(DIRECTORY "${_out_root}/${_arch}/"
                DESTINATION "${ARG_INSTALL_BASE}/${_arch}"
                OPTIONAL)
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# hkp_probe_rocke_importable(<out_ok>)
#   Configure-time gate for a rocke production root: run the base interpreter
#   with the in-tree rocke platform + library sources on PYTHONPATH and check
#   that `import rocke, kernels` succeeds. A configure probe cannot use the
#   build-time-only wheel interpreter, so it validates the source tree (the same
#   trees the wheels are built from); a wheel-install failure surfaces at build.
# ---------------------------------------------------------------------------
function(hkp_probe_rocke_importable out_ok)
    set(_rocke_root "${HKP_PKG_DIR}/../rocke")
    if(WIN32)
        set(_sep ";")
    else()
        set(_sep ":")
    endif()
    set(_pp "${_rocke_root}/platform/python${_sep}${_rocke_root}/library")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_pp}"
                "${Python3_EXECUTABLE}" -c "import rocke, kernels"
        RESULT_VARIABLE _rc
        OUTPUT_QUIET ERROR_QUIET)
    if(_rc EQUAL 0)
        set(${out_ok} TRUE PARENT_SCOPE)
    else()
        set(${out_ok} FALSE PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# hkp_rocke_wheel_python_interp(<out_interp>)
#   Provision a build-local interpreter carrying the rocke + rocke_library
#   wheels (built by the rocke-wheels target, which requires
#   HIPKERNELPROVIDER_ENABLE_ROCKE). The production tool imports rocke/kernels
#   from these installed wheels rather than the editable dev venv. The venv
#   interpreter is an add_custom_command OUTPUT so the production command can
#   depend on it for build ordering.
# ---------------------------------------------------------------------------
function(hkp_rocke_wheel_python_interp out_interp)
    set(_venv "${CMAKE_CURRENT_BINARY_DIR}/hkp-rocke-venv")
    if(WIN32)
        set(_venv_py "${_venv}/Scripts/python.exe")
    else()
        set(_venv_py "${_venv}/bin/python")
    endif()

    set(_platform_wheel
        "${ROCKE_WHEEL_DIR}/rocke-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")
    set(_library_wheel
        "${ROCKE_WHEEL_DIR}/rocke_library-${ROCKE_WHEEL_VERSION}-py3-none-any.whl")

    add_custom_command(
        OUTPUT "${_venv_py}"
        COMMAND "${Python3_EXECUTABLE}" -m venv --system-site-packages --copies
                "${_venv}"
        COMMAND "${_venv_py}" -m pip install -q --upgrade pip
        COMMAND "${_venv_py}" -m pip install -q
                "${_platform_wheel}" "${_library_wheel}"
        DEPENDS "${_platform_wheel}" "${_library_wheel}"
        COMMENT "hkp: provisioning rocke wheel python interpreter"
        VERBATIM)

    add_custom_target(hkp_rocke_wheel_python_interp ALL DEPENDS "${_venv_py}"
                      COMMENT "hkp: rocke wheel python interpreter")
    set(${out_interp} "${_venv_py}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_add_packaging()
#   Independent production gates for a hip source root and a rocke source root.
#   Each var: empty = that producer dormant; set-but-not-a-dir = fatal; set-and-
#   directory = include that producer. Any of {hip, rocke, both, neither} is
#   legal; when either is enabled ONE tool process packs both into one kpack per
#   arch. Configure hard-fails only when a SET root's configure-discoverable
#   toolchain is missing (hip -> hipcc; rocke -> the ENABLE_ROCKE + importable
#   conjunction). The tests are wired regardless: they drive the fixture slice
#   directly, never a production source root.
# ---------------------------------------------------------------------------
function(hkp_add_packaging)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    hkp_resolve_kpack(_rocm_kpack_dir)
    hkp_selected_arches(_arches)

    # hipcc is the perl/bat driver that honors --genco; on Windows it is
    # hipcc.exe or hipcc.bat. hipcc.bin.exe is the raw clang driver and is only
    # a last-resort fallback.
    find_program(HKP_HIPCC NAMES hipcc hipcc.bat hipcc.bin.exe)

    set(_install_base
        "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/arch_content/hip-kernel-provider/descriptors")

    set(HIPKERNELPROVIDER_PRODUCTION_HIP_SOURCE_ROOT "" CACHE PATH
        "Authored hip-source root the production pack step compiles from. \
Empty leaves hip production packaging dormant; set to a real authored source \
folder for a release build.")
    set(HIPKERNELPROVIDER_PRODUCTION_ROCKE_SOURCE_ROOT "" CACHE PATH
        "Authored rocke-source root the production pack step compiles from \
(descriptors only; the kernel bodies resolve via the importable rocke/kernels \
packages). Empty leaves rocke production packaging dormant.")

    set(_hip_root "")
    if(HIPKERNELPROVIDER_PRODUCTION_HIP_SOURCE_ROOT)
        if(NOT IS_DIRECTORY "${HIPKERNELPROVIDER_PRODUCTION_HIP_SOURCE_ROOT}")
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_HIP_SOURCE_ROOT is set but is "
                "not a directory: ${HIPKERNELPROVIDER_PRODUCTION_HIP_SOURCE_ROOT}")
        endif()
        set(_hip_root "${HIPKERNELPROVIDER_PRODUCTION_HIP_SOURCE_ROOT}")
    endif()

    set(_rocke_root "")
    if(HIPKERNELPROVIDER_PRODUCTION_ROCKE_SOURCE_ROOT)
        if(NOT IS_DIRECTORY "${HIPKERNELPROVIDER_PRODUCTION_ROCKE_SOURCE_ROOT}")
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_ROCKE_SOURCE_ROOT is set but "
                "is not a directory: "
                "${HIPKERNELPROVIDER_PRODUCTION_ROCKE_SOURCE_ROOT}")
        endif()
        set(_rocke_root "${HIPKERNELPROVIDER_PRODUCTION_ROCKE_SOURCE_ROOT}")
    endif()

    # A set hip root requires the configure-discoverable hipcc.
    if(_hip_root AND NOT HKP_HIPCC)
        message(FATAL_ERROR
            "hkp: HIPKERNELPROVIDER_PRODUCTION_HIP_SOURCE_ROOT is set but hipcc "
            "was not found (searched hipcc, hipcc.bat, hipcc.bin.exe). Ensure "
            "the ROCm bin dir is on PATH.")
    endif()

    # A set rocke root requires the full conjunction: ENABLE_ROCKE (which builds
    # the wheels), the wheel-env available, and rocke/kernels importable. Any
    # missing piece is a configure hard-fail naming what is missing.
    set(_rocke_interp "")
    set(_rocke_comgr_lib "${ROCKE_COMGR_LIB}")
    if(_rocke_root)
        if(NOT HIPKERNELPROVIDER_ENABLE_ROCKE)
            message(FATAL_ERROR
                "hkp: a rocke production source root is set but "
                "HIPKERNELPROVIDER_ENABLE_ROCKE is OFF — enable it so the "
                "rocke/kernels wheels are built and importable.")
        endif()
        if(NOT ROCKE_WHEEL_DIR)
            message(FATAL_ERROR
                "hkp: a rocke production source root is set but the rocke "
                "wheel-env is not available (ROCKE_WHEEL_DIR unset).")
        endif()
        hkp_probe_rocke_importable(_rocke_ok)
        if(NOT _rocke_ok)
            message(FATAL_ERROR
                "hkp: a rocke production source root is set but rocke/kernels "
                "are not importable.")
        endif()
        hkp_rocke_wheel_python_interp(_rocke_interp)
    endif()

    if(_hip_root OR _rocke_root)
        hkp_wire_production(
            HIP_ROOT "${_hip_root}"
            ROCKE_ROOT "${_rocke_root}"
            ARCHES "${_arches}"
            HIPCC "${HKP_HIPCC}"
            ROCM_KPACK_DIR "${_rocm_kpack_dir}"
            INSTALL_BASE "${_install_base}"
            ROCKE_INTERP "${_rocke_interp}"
            ROCKE_COMGR_LIB "${_rocke_comgr_lib}")
    else()
        message(STATUS
            "hkp: no production source root set "
            "(HIPKERNELPROVIDER_PRODUCTION_HIP_SOURCE_ROOT / "
            "..._ROCKE_SOURCE_ROOT empty); production packaging dormant "
            "(tests still run against the fixtures).")
    endif()

    hkp_register_tests("${_rocm_kpack_dir}" "${HKP_HIPCC}" "${_hip_root}"
                       "${_rocke_comgr_lib}")
endfunction()

# ---------------------------------------------------------------------------
# hkp_register_tests(<rocm_kpack_dir> <hipcc> <hip_root> <rocke_comgr_lib>)
#   Register the pytest suite as two build-tree ctest entries running disjoint
#   sets: a quick entry (`-m quick`, the no-compile subset) and a standard entry
#   (`-m "not quick"`, the rest). Tier labels come from HKP_PACK_test_categories,
#   whose cascade runs each test once per tier with no overlap. Without pytest on
#   PATH the entries register DISABLED so they list as skipped, not absent.
#
#   Configure hard-fails on a missing hipcc only when a hip production root is
#   set (a tests-only ingestor build configures clean on a bare box; the
#   compile-dependent tests self-skip via the hipcc/rocke fixtures, and CI
#   hard-gates them via the REQUIRE_* env vars forwarded below).
# ---------------------------------------------------------------------------
function(hkp_register_tests rocm_kpack_dir hipcc hip_root rocke_comgr_lib)
    if(NOT HIPKERNELPROVIDER_ENABLE_TESTS)
        return()
    endif()
    if(hip_root AND NOT hipcc)
        message(FATAL_ERROR
            "hkp: a hip production source root is set but hipcc was not found.")
    endif()

    # `python` resolves from PATH at test time; the ENVIRONMENT paths are
    # configure-time absolutes, valid because these entries run only in the
    # build tree on the configuring machine.
    set(_pyenv "PYTHONPATH=${HKP_PYTHON_ROOT}")
    if(hipcc)
        list(APPEND _pyenv "HKP_HIPCC=${hipcc}")
    endif()
    if(rocm_kpack_dir)
        list(APPEND _pyenv "HIPKERNELPROVIDER_ROCM_KPACK_DIR=${rocm_kpack_dir}")
    endif()
    # Forward the rocke comgr override + the CI hard-gate flags so the
    # comgr-dependent tier actually runs (not silently skips) where provisioned.
    if(rocke_comgr_lib)
        list(APPEND _pyenv "ROCKE_COMGR_LIB=${rocke_comgr_lib}")
    endif()
    if(HIPKERNELPROVIDER_KPACK_REQUIRE_HIPCC)
        list(APPEND _pyenv
            "HIPKERNELPROVIDER_KPACK_REQUIRE_HIPCC=${HIPKERNELPROVIDER_KPACK_REQUIRE_HIPCC}")
    endif()
    if(HIPKERNELPROVIDER_KPACK_REQUIRE_COMGR)
        list(APPEND _pyenv
            "HIPKERNELPROVIDER_KPACK_REQUIRE_COMGR=${HIPKERNELPROVIDER_KPACK_REQUIRE_COMGR}")
    endif()

    # When PATH `python` cannot import pytest, register the entries as DISABLED so
    # they appear in the ctest listing as skipped rather than silently absent.
    execute_process(
        COMMAND python -c "import pytest"
        RESULT_VARIABLE _pytest_rc
        OUTPUT_QUIET ERROR_QUIET)
    set(_disabled "")
    if(NOT _pytest_rc EQUAL 0)
        message(STATUS
            "hkp: pytest not importable by PATH `python`; registering "
            "descriptor-packaging pytest tests as DISABLED.")
        set(_disabled DISABLED TRUE)
    endif()

    add_test(NAME hip-kernel-provider-hkp-pack-quick
             COMMAND python -m pytest "${HKP_PKG_DIR}/tests" -m quick -v)
    set_tests_properties(hip-kernel-provider-hkp-pack-quick PROPERTIES
        ENVIRONMENT "${_pyenv}"
        ${_disabled})

    add_test(NAME hip-kernel-provider-hkp-pack
             COMMAND python -m pytest "${HKP_PKG_DIR}/tests" -m "not quick" -v)
    set_tests_properties(hip-kernel-provider-hkp-pack PROPERTIES
        ENVIRONMENT "${_pyenv}"
        ${_disabled})

    # Both entries are add_test()'d in this scope just above, so the YAML's
    # test_patterns match them via the directory-property loop. EXPLICIT_TESTS is
    # avoided: apply_ctest_category_labels joins it with ';', which execute_process
    # re-splits into separate argv, leaking a second name into the parser's
    # positional install-file slot.
    if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED
       AND COMMAND apply_ctest_category_labels)
        apply_ctest_category_labels("${HKP_PACK_CTEST_CATEGORIES_YAML}")
    endif()
endfunction()
