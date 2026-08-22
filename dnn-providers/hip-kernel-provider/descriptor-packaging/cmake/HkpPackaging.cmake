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
# hkp_wire_production(SOURCE_ROOT <dir> ENABLE_ROCKE <bool> ARCHES <list>
#                     HIPCC <path> ROCM_KPACK_DIR <dir> INSTALL_BASE <dir>
#                     ROCKE_INTERP <path> ROCKE_COMGR_LIB <path>)
#   Wire the production compile -> prune -> pack -> install DAG against the ONE
#   authored source root. The root is walked recursively; child folders under it
#   scope the content (hip/, rocKE/, per-integration folders) and each
#   descriptor's authored subpath is preserved into the staged and installed
#   trees. Producer selection is per-UKD on kernel_source.kind, never per-folder,
#   so one root feeds both producers into one kpack per arch.
#
#   ENABLE_ROCKE says whether the rocKE producer may run. It is a switch of its
#   own rather than something inferred from the root, because the root now names
#   a location only: CMake cannot know which producers a tree needs without
#   reading the descriptors. When ON the tool runs under ROCKE_INTERP
#   (wheel-provisioned so `import rocke`/`kernels` resolve) and ROCKE_COMGR_LIB,
#   if set, is forwarded to the tool environment.
#
#   Empty ARCHES wires nothing.
# ---------------------------------------------------------------------------
function(hkp_wire_production)
    set(_one SOURCE_ROOT ENABLE_ROCKE ARCHES HIPCC ROCM_KPACK_DIR INSTALL_BASE
        ROCKE_INTERP ROCKE_COMGR_LIB)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${_one}" "")

    if(NOT ARG_ARCHES)
        return()
    endif()

    set(_out_root "${CMAKE_CURRENT_BINARY_DIR}/hkp-descriptors")
    set(_inter_root "${CMAKE_CURRENT_BINARY_DIR}/hkp-intermediate")
    string(REPLACE ";" "," _arch_csv "${ARG_ARCHES}")
    set(_stamp "${_out_root}.stamp")

    # The authored root is a tree, not a flat folder: glob recursively so a
    # descriptor added in any child folder retriggers the pack step.
    file(GLOB_RECURSE _source_inputs CONFIGURE_DEPENDS
         "${ARG_SOURCE_ROOT}/*.json" "${ARG_SOURCE_ROOT}/*.cpp")

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

    # The rocKE producer needs the wheel-provisioned interpreter so its UKDs
    # import; a hip-only pack runs under the base interpreter (hip compiles shell
    # out to hipcc and are interpreter-agnostic).
    set(_interp "${Python3_EXECUTABLE}")
    set(_interp_dep "")
    if(ARG_ENABLE_ROCKE)
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
                --source-root "${ARG_SOURCE_ROOT}"
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
#   Gate production packaging on ONE source root plus explicit per-producer
#   switches. The root names a location; the switches say which producers may
#   run. They are separate because a path cannot express producer intent —
#   inferring it would mean CMake reading descriptor JSON to find out.
#
#   Root empty = production packaging dormant. Root set but not a directory =
#   fatal. Root set with neither switch on = fatal, because it would otherwise
#   pack nothing and silently install an empty tree. Configure hard-fails when an
#   ON switch's configure-discoverable toolchain is missing (hip -> hipcc;
#   rocke -> the ENABLE_ROCKE + wheel-env + importable conjunction). The tests
#   are wired regardless: they drive the fixture slice directly, never a
#   production source root.
# ---------------------------------------------------------------------------
function(hkp_add_packaging)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    hkp_resolve_kpack(_rocm_kpack_dir "${Python3_EXECUTABLE}")
    hkp_selected_arches(_arches)

    # hipcc is the perl/bat driver that honors --genco; on Windows it is
    # hipcc.exe or hipcc.bat. hipcc.bin.exe is the raw clang driver and is only
    # a last-resort fallback.
    find_program(HKP_HIPCC NAMES hipcc hipcc.bat hipcc.bin.exe)

    set(_install_base
        "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/arch_content/hip-kernel-provider/descriptors")

    set(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT "" CACHE PATH
        "The authored source root the production pack step compiles from. \
Walked recursively; child folders under it scope the content (hip/, rocKE/, \
per-integration folders) and each descriptor's authored subpath is preserved \
into the staged and installed trees. Empty leaves production packaging dormant.")
    set(HIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP OFF CACHE BOOL
        "Allow the hip producer to run over the production source root. \
Requires a configure-discoverable hipcc.")
    set(HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE OFF CACHE BOOL
        "Allow the rocKE producer to run over the production source root. \
Requires HIPKERNELPROVIDER_ENABLE_ROCKE, the rocke wheel-env, and importable \
rocke/kernels packages.")

    set(_source_root "")
    if(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT)
        if(NOT IS_DIRECTORY "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT is set but is "
                "not a directory: ${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
        endif()
        set(_source_root "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}")
    endif()

    # A root with no producer enabled would pack nothing and install an empty
    # tree — the silent-empty-package failure mode. Name both switches so the
    # fix is obvious.
    if(_source_root
       AND NOT HIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP
       AND NOT HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE)
        message(FATAL_ERROR
            "hkp: HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT is set but no "
            "producer is enabled — set HIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP "
            "and/or HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE, otherwise the "
            "pack step would ship an empty tree.")
    endif()

    # The hip producer requires the configure-discoverable hipcc.
    if(_source_root AND HIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP AND NOT HKP_HIPCC)
        message(FATAL_ERROR
            "hkp: HIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP is ON but hipcc was "
            "not found (searched hipcc, hipcc.bat, hipcc.bin.exe). Ensure the "
            "ROCm bin dir is on PATH or CMAKE_PROGRAM_PATH.")
    endif()

    # The rocKE producer requires the full conjunction: ENABLE_ROCKE (which
    # builds the wheels), the wheel-env available, and rocke/kernels importable.
    # Any missing piece is a configure hard-fail naming what is missing.
    set(_rocke_interp "")
    set(_rocke_comgr_lib "${ROCKE_COMGR_LIB}")
    set(_enable_rocke OFF)
    if(_source_root AND HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE)
        set(_enable_rocke ON)
        if(NOT HIPKERNELPROVIDER_ENABLE_ROCKE)
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE is ON but "
                "HIPKERNELPROVIDER_ENABLE_ROCKE is OFF — enable it so the "
                "rocke/kernels wheels are built and importable.")
        endif()
        if(NOT ROCKE_WHEEL_DIR)
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE is ON but the "
                "rocke wheel-env is not available (ROCKE_WHEEL_DIR unset).")
        endif()
        hkp_probe_rocke_importable(_rocke_ok)
        if(NOT _rocke_ok)
            message(FATAL_ERROR
                "hkp: HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE is ON but "
                "rocke/kernels are not importable.")
        endif()
        hkp_rocke_wheel_python_interp(_rocke_interp)
    endif()

    if(_source_root)
        hkp_wire_production(
            SOURCE_ROOT "${_source_root}"
            ENABLE_ROCKE "${_enable_rocke}"
            ARCHES "${_arches}"
            HIPCC "${HKP_HIPCC}"
            ROCM_KPACK_DIR "${_rocm_kpack_dir}"
            INSTALL_BASE "${_install_base}"
            ROCKE_INTERP "${_rocke_interp}"
            ROCKE_COMGR_LIB "${_rocke_comgr_lib}")
    else()
        message(STATUS
            "hkp: no production source root set "
            "(HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT empty); production "
            "packaging dormant (tests still run against the fixtures).")
    endif()

    # The integration suite's packaged artifact, staged into the build tree rather than
    # installed. Gated on the ingestor and nothing else: HIPDNN_DESCRIPTOR_BUILD_DIR is
    # only defined under that gate, and a build with the ingestor on but the tests off
    # still wants the staging rule to be exercised rather than silently absent.
    if(HIPDNN_ENABLE_KERNEL_INGESTOR)
        hkp_wire_demo("${HKP_DEMO_SOURCE_ROOT}" "${_arches}" "${HKP_HIPCC}"
                      "${_rocm_kpack_dir}" "${HIPDNN_DESCRIPTOR_BUILD_DIR}")
    endif()

    hkp_register_tests("${_rocm_kpack_dir}" "${HKP_HIPCC}"
                       "${HIPKERNELPROVIDER_PRODUCTION_ENABLE_HIP}"
                       "${_rocke_comgr_lib}")
endfunction()

# ---------------------------------------------------------------------------
# hkp_register_tests(<rocm_kpack_dir> <hipcc> <hip_enabled> <rocke_comgr_lib>)
#   Register the pytest suite as two build-tree ctest entries running disjoint
#   sets: a quick entry (`-m quick`, the no-compile subset) and a standard entry
#   (`-m "not quick"`, the rest). Tier labels come from HKP_PACK_test_categories,
#   whose cascade runs each test once per tier with no overlap. When
#   Python3_EXECUTABLE cannot import pytest the entries register DISABLED so
#   they list as skipped, not absent.
#
#   Configure hard-fails on a missing hipcc only when the hip producer is
#   enabled (a tests-only ingestor build configures clean on a bare box; the
#   compile-dependent tests self-skip via the hipcc/rocke fixtures, and CI
#   hard-gates them via the REQUIRE_* env vars forwarded below).
# ---------------------------------------------------------------------------
function(hkp_register_tests rocm_kpack_dir hipcc hip_enabled rocke_comgr_lib)
    if(NOT HIPKERNELPROVIDER_ENABLE_TESTS)
        return()
    endif()
    if(hip_enabled AND NOT hipcc)
        message(FATAL_ERROR
            "hkp: the hip production producer is enabled but hipcc was not found.")
    endif()

    # Runs under Python3_EXECUTABLE, the interpreter hkp_resolve_kpack proved
    # can import rocm_kpack. Bare PATH `python` may be a different one. The
    # ENVIRONMENT paths are configure-time absolutes, valid because these
    # entries run only in the build tree on the configuring machine.
    #
    # conftest.py reads HIPKERNELPROVIDER_ROCM_KPACK_DIR, so that is the name
    # forwarded here regardless of which variable resolved it.
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
