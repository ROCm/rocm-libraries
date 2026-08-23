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

# The authored hip-form source root the integration suite's packaged artifact is built
# from. It lives beside the test that consumes it, not in the product tree: it is a test
# fixture, so it is staged into the build tree and never installed.
set(HKP_DEMO_SOURCE_ROOT
    "${HKP_PKG_DIR}/../src/integration_tests/kernel_ingestor_engine/fixtures/packaged")

# ---------------------------------------------------------------------------
# hkp_resolve_kpack(<out_var>)
#   2-tier resolution of the rocm-kpack 'python' directory:
#   (1) -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR override,
#   (2) a rocm_kpack importable by the build's Python. Sets <out_var> to the
#   resolved python dir. rocm_kpack is load-bearing (the tool cannot pack without
#   it), so an unresolvable dependency is a hard error.
#
#   This resolves the PACKER only. The reader is the platform's rocm_kpack shared
#   library, resolved by find_package(rocm-kpack) in the provider's CMakeLists;
#   the two halves are versioned by the platform rather than pinned together here.
# ---------------------------------------------------------------------------
function(hkp_resolve_kpack out_var)
    if(DEFINED HIPKERNELPROVIDER_KPACK_PYTHON_DIR AND EXISTS "${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}")
        set(${out_var} "${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}" PARENT_SCOPE)
        message(STATUS "hkp: using the rocm_kpack packer from \
HIPKERNELPROVIDER_KPACK_PYTHON_DIR=${HIPKERNELPROVIDER_KPACK_PYTHON_DIR}; the reader is \
the platform's rocm_kpack shared library")
        return()
    endif()

    # Tier 2: a rocm_kpack importable by the build's Python. The reader is the
    # platform's shared library, so an installed rocm_kpack package is its counterpart;
    # there is no source tree to borrow a python/ directory from.
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import os, rocm_kpack; print(os.path.dirname(os.path.dirname(rocm_kpack.__file__)))"
        OUTPUT_VARIABLE _kpack_python
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _kpack_probe
        ERROR_QUIET)
    if(_kpack_probe EQUAL 0 AND EXISTS "${_kpack_python}/rocm_kpack/kpack.py")
        set(${out_var} "${_kpack_python}" PARENT_SCOPE)
        message(STATUS "hkp: using rocm_kpack from ${_kpack_python}")
        return()
    endif()

    message(FATAL_ERROR
        "hkp: rocm_kpack could not be resolved (override with "
        "HIPKERNELPROVIDER_KPACK_PYTHON_DIR, or install the rocm_kpack Python "
        "package into ${Python3_EXECUTABLE}). "
        "rocm_kpack is required to pack; there is no skip path.")
endfunction()

# ---------------------------------------------------------------------------
# hkp_selected_arches(<out_var>)
#   Normalize GPU_TARGETS (or AMDGPU_TARGETS) into a bare gfx arch list,
#   stripping feature suffixes (gfx942:xnack-) and dropping anything that is not
#   a concrete gfx name. Empty result is legal (install nothing, non-error). No
#   intersection with a fixed fixture set: the tool compiles from authored
#   sources for whatever arch is requested.
#
#   The only consumer of GPU_TARGETS in dnn-providers/. The sibling kpack
#   producer, src/engines/asm_sdpa_engine/CMakeLists.txt, declares an explicit
#   list instead because it globs prebuilt .co files; this step compiles from
#   source and can target any real gfx, so it reads GPU_TARGETS.
#
#   Elsewhere in this repo a gfxNNX-style label is a selector matched against a
#   concrete arch (shared/ctest/parse_test_categories.py,
#   test/therock/test_runner.py, .github/scripts/amdgpu_family_matrix.py); here
#   the value reaches hipcc's --offload-arch, where a family name is unusable
#   rather than coarse. Hence drop-with-warning, not passthrough, and no
#   family-to-arch expansion table.
# ---------------------------------------------------------------------------
function(hkp_selected_arches out_var)
    set(_targets "")
    set(_source "")
    if(DEFINED GPU_TARGETS AND GPU_TARGETS)
        set(_targets ${GPU_TARGETS})
        set(_source "GPU_TARGETS")
    elseif(DEFINED AMDGPU_TARGETS AND AMDGPU_TARGETS)
        set(_targets ${AMDGPU_TARGETS})
        set(_source "AMDGPU_TARGETS")
    endif()

    set(_selected "")
    foreach(_arch IN LISTS _targets)
        string(REGEX REPLACE ":.*$" "" _bare "${_arch}")
        if(NOT _bare)
            continue()
        endif()
        if(NOT _bare MATCHES "^gfx[0-9a-f]+$")
            message(WARNING
                "hkp: ignoring '${_arch}' from ${_source}; it is not a concrete gfx "
                "architecture and cannot be passed to hipcc --offload-arch. Nothing "
                "is packed for it. Name real gfx architectures in ${_source} to pack "
                "for them.")
            continue()
        endif()
        list(APPEND _selected "${_bare}")
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
    # install artifacts go stale against the current pipeline code. The fetched
    # rocm_kpack package counts: kpack_resolver.py imports it and it decides the
    # archive format. FetchContent's source dir is name-derived, so moving
    # HIPKERNELPROVIDER_KPACK_GIT_REF does not move this stamp -- without these
    # inputs a `cmake --fresh` onto a new ref rebuilds the reader while the pack
    # stamp survives, leaving an archive written by the old packer and read by the
    # new one. That is the skew the single pin in RocmKpack.cmake exists to prevent.
    file(GLOB _tool_sources CONFIGURE_DEPENDS
         "${HKP_PYTHON_ROOT}/hkp_pack/*.py" "${kpack_python}/rocm_kpack/*.py")

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
# hkp_wire_demo(<source_root> <arches> <hipcc> <kpack_python> <stage_root>)
#   Wire the same compile -> prune -> pack DAG as hkp_wire_production, then a
#   second, cheap step that copies the packed tree into the build tree's staged
#   descriptor directory. Two differences, both deliberate:
#
#   * No install() of any kind. The artifact exists so an integration ctest has a
#     real .kpack to load; it is not shipped surface.
#   * Missing hipcc warns and skips instead of failing. A release that silently
#     ships nothing is worse than a failed build, which is why production is
#     fatal -- but a test fixture must never break a developer's build. The test
#     GTEST_SKIP()s when the artifact is absent.
#
#   The two steps are separate custom commands on purpose. The pack step is
#   expensive and keyed on the authored sources; the copy step is keyed on a
#   stamp placed INSIDE stage_root, which hip-kernel-provider/CMakeLists.txt
#   wipes with file(REMOVE_RECURSE) once per configure. The wipe therefore takes
#   the stamp with it and the next build re-copies -- a stamp outside stage_root
#   would leave the tree wiped and the copy skipped, which looks exactly like
#   "packaging did not run".
#
#   The packer's <arch>/ + kpack/ layout is copied through verbatim: `library` on
#   a packed UKD is relative to the directory of the descriptor that declared it,
#   so flattening the tree breaks resolution at dispatch time with an error that
#   points at the loader rather than at this function.
# ---------------------------------------------------------------------------
function(hkp_wire_demo source_root arches hipcc kpack_python stage_root)
    if(NOT arches)
        message(STATUS
            "hkp: no GPU_TARGETS/AMDGPU_TARGETS selected; the packaged integration "
            "fixture is not staged.")
        return()
    endif()
    if(NOT IS_DIRECTORY "${source_root}")
        message(WARNING
            "hkp: packaged integration fixture source root not found at ${source_root}; "
            "the artifact will not be staged.")
        return()
    endif()
    if(NOT stage_root)
        message(WARNING
            "hkp: HIPDNN_DESCRIPTOR_BUILD_DIR is not set, so there is nowhere to stage "
            "the packaged integration fixture; skipping it.")
        return()
    endif()
    if(NOT hipcc)
        message(WARNING
            "hkp: hipcc not found (searched hipcc, hipcc.bat, hipcc.bin.exe); the "
            "packaged integration fixture cannot be compiled and will not be staged. "
            "IntegrationGpuKernelIngestorKpack will skip. Put the ROCm bin dir on PATH "
            "to build it.")
        return()
    endif()

    set(_out_root "${CMAKE_CURRENT_BINARY_DIR}/hkp-demo-descriptors")
    set(_inter_root "${CMAKE_CURRENT_BINARY_DIR}/hkp-demo-intermediate")
    string(REPLACE ";" "," _arch_csv "${arches}")
    set(_pack_stamp "${_out_root}.stamp")
    file(GLOB _source_inputs CONFIGURE_DEPENDS
         "${source_root}/*.json" "${source_root}/*.cpp")
    # Both halves of the packer, for the reason spelled out in hkp_wire_production.
    file(GLOB _tool_sources CONFIGURE_DEPENDS
         "${HKP_PYTHON_ROOT}/hkp_pack/*.py" "${kpack_python}/rocm_kpack/*.py")

    add_custom_command(
        OUTPUT "${_pack_stamp}"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "${_out_root}"
        COMMAND "${Python3_EXECUTABLE}" "${HKP_TOOL}"
                --source-root "${source_root}"
                --out-root "${_out_root}"
                --arches "${_arch_csv}"
                --hipcc "${hipcc}"
                --inter-root "${_inter_root}"
                --kpack-python-dir "${kpack_python}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_pack_stamp}"
        DEPENDS "${HKP_TOOL}" ${_source_inputs} ${_tool_sources}
        COMMENT "hkp: packing the integration fixture for ${arches}"
        VERBATIM)

    set(_stage_stamp "${stage_root}/.hkp-demo-staged.stamp")
    add_custom_command(
        OUTPUT "${_stage_stamp}"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${_out_root}" "${stage_root}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${_stage_stamp}"
        DEPENDS "${_pack_stamp}"
        COMMENT "hkp: staging the packaged integration fixture into ${stage_root}"
        VERBATIM)

    add_custom_target(hkp_demo_packaging ALL DEPENDS "${_stage_stamp}"
                      COMMENT "hkp: packaged integration fixture")
endfunction()

# ---------------------------------------------------------------------------
# hkp_add_packaging()
#   Three independent gates. The production pack+install wires only when
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

    # The integration suite's packaged artifact, staged into the build tree rather than
    # installed. Gated on the ingestor and nothing else: HIPDNN_DESCRIPTOR_BUILD_DIR is
    # only defined under that gate, and a build with the ingestor on but the tests off
    # still wants the staging rule to be exercised rather than silently absent.
    if(HIPDNN_ENABLE_KERNEL_INGESTOR)
        hkp_wire_demo("${HKP_DEMO_SOURCE_ROOT}" "${_arches}" "${HKP_HIPCC}"
                      "${_kpack_python}" "${HIPDNN_DESCRIPTOR_BUILD_DIR}")
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

    # `python` resolves from PATH at test time; the ENVIRONMENT paths are
    # configure-time absolutes, valid because these entries run only in the
    # build tree on the configuring machine.
    set(_pyenv "PYTHONPATH=${HKP_PYTHON_ROOT}" "HKP_HIPCC=${hipcc}")
    if(kpack_python)
        list(APPEND _pyenv "HIPKERNELPROVIDER_KPACK_PYTHON_DIR=${kpack_python}")
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
