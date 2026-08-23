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

include(KpackPython)

# ---------------------------------------------------------------------------
# hkp_resolve_kpack(<out_var> <python_exe>)
#   Resolve the rocm_kpack python dir, or hard-fail: this pipeline cannot pack
#   without it, so there is no skip path.
#
#   Also verifies <python_exe> can import it. Resolution only proves the
#   directory exists; the import still fails when the interpreter differs from
#   the one the tree's compiled msgpack/zstandard extensions were built for.
#   Probing here reports that at configure time instead of mid-build.
# ---------------------------------------------------------------------------
function(hkp_resolve_kpack out_var python_exe)
    kpack_resolve_python_dir(_python_dir)
    if("${_python_dir}" STREQUAL "")
        kpack_unset_reason(_reason)
        message(FATAL_ERROR "hkp: ${_reason}. rocm_kpack is required to pack "
            "descriptors; there is no skip path.")
    endif()
    kpack_check_python_deps("${python_exe}" "${_python_dir}" _missing)
    if(_missing)
        string(REPLACE ";" ", " _missing_csv "${_missing}")
        message(FATAL_ERROR
            "hkp: ${python_exe} cannot import ${_missing_csv} (rocm_kpack "
            "needs zstandard>=0.20.0 and msgpack). If the resolved tree was "
            "staged for a different Python, install the dependencies for this "
            "interpreter or point -DPython3_EXECUTABLE at the one they were "
            "built for.")
    endif()
    set(${out_var} "${_python_dir}" PARENT_SCOPE)
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

    hkp_resolve_kpack(_kpack_python "${Python3_EXECUTABLE}")
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
#   Register the pytest suite as two build-tree ctest entries running disjoint
#   sets: a quick entry (`-m quick`, the no-compile subset) and a standard entry
#   (`-m "not quick"`, the rest). Tier labels come from HKP_PACK_test_categories,
#   whose cascade runs each test once per tier with no overlap. Without pytest on
#   PATH the entries register DISABLED so they list as skipped, not absent.
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

    # Runs under Python3_EXECUTABLE, the interpreter hkp_resolve_kpack proved
    # can import rocm_kpack. Bare PATH `python` may be a different one. The
    # ENVIRONMENT paths are configure-time absolutes, valid because these
    # entries run only in the build tree on the configuring machine.
    set(_pyenv "PYTHONPATH=${HKP_PYTHON_ROOT}" "HKP_HIPCC=${hipcc}"
        "HIPKERNELPROVIDER_KPACK_PYTHON_DIR=${kpack_python}")

    # Without pytest, register the entries as DISABLED so they appear in the
    # ctest listing as skipped rather than silently absent.
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pytest"
        RESULT_VARIABLE _pytest_rc
        OUTPUT_QUIET ERROR_QUIET)
    set(_disabled "")
    if(NOT _pytest_rc EQUAL 0)
        message(STATUS
            "hkp: pytest not importable by ${Python3_EXECUTABLE}; registering "
            "descriptor-packaging pytest tests as DISABLED.")
        set(_disabled DISABLED TRUE)
    endif()

    add_test(NAME hip-kernel-provider-hkp-pack-quick
             COMMAND "${Python3_EXECUTABLE}" -m pytest "${HKP_PKG_DIR}/tests" -m quick -v)
    set_tests_properties(hip-kernel-provider-hkp-pack-quick PROPERTIES
        ENVIRONMENT "${_pyenv}"
        ${_disabled})

    add_test(NAME hip-kernel-provider-hkp-pack
             COMMAND "${Python3_EXECUTABLE}" -m pytest "${HKP_PKG_DIR}/tests" -m "not quick" -v)
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
