# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Packaging probes: CI-only, build-tree-only checks that an integration's descriptors
# pack for an explicit architecture, independent of the build's GPU_TARGETS. A probe
# packs through hkp_wire_pack_target with the toolchain hkp_add_packaging resolved,
# then a ctest entry asserts the packed output. Nothing here installs or ships.
#
# This module is included only by the guarded hook in hkp_add_packaging, so with the
# option OFF it is never read. It holds function definitions only; configure effects
# start when hkp_load_packaging_probes() runs.

include_guard(GLOBAL)

# ---------------------------------------------------------------------------
# _hkp_probe_check_out_root(<name> <out_root>)
#   FATAL when a probe's output root lies inside a descriptor tree the install rules
#   ship; probe output must never be installed.
# ---------------------------------------------------------------------------
function(_hkp_probe_check_out_root name out_root)
    foreach(_shipped IN ITEMS
            "${HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR}"
            "${HIPKERNELPROVIDER_TEST_DESCRIPTOR_BUILD_DIR}")
        if(_shipped)
            string(FIND "${out_root}/" "${_shipped}/" _pos)
            if(_pos EQUAL 0)
                message(FATAL_ERROR
                    "hkp probe '${name}': output root ${out_root} is inside the "
                    "shipped descriptor tree ${_shipped}; probe output must not be "
                    "installed.")
            endif()
        endif()
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# _hkp_probe_derive_root(<name> <from_dir> <kdp> <instance> <out_dir>)
#   Derive the probe root at configure time (FATAL with the tool's stderr when the
#   instance is missing or ambiguous) and re-run configure when a source file changes.
# ---------------------------------------------------------------------------
function(_hkp_probe_derive_root name from_dir kdp instance out_dir)
    file(GLOB_RECURSE _derive_inputs CONFIGURE_DEPENDS "${from_dir}/*")
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" APPEND
                 PROPERTY CMAKE_CONFIGURE_DEPENDS ${_derive_inputs})
    execute_process(
        COMMAND "${Python3_EXECUTABLE}"
                "${HKP_PKG_DIR}/tools/hkp_probe_derive_root.py"
                --from "${from_dir}"
                --kdp "${kdp}"
                --instance-name "${instance}"
                --out "${out_dir}"
        RESULT_VARIABLE _derive_rc
        OUTPUT_VARIABLE _derive_out
        ERROR_VARIABLE _derive_err)
    if(NOT _derive_rc EQUAL 0)
        message(FATAL_ERROR
            "hkp probe '${name}': deriving the probe root failed (exit "
            "${_derive_rc}).\n${_derive_err}")
    endif()
    message(STATUS
        "hkp: probe ${name}: derived root ${out_dir} (instance '${instance}' of ${kdp})")
endfunction()

# ---------------------------------------------------------------------------
# hkp_add_packaging_probe(NAME <n> ARCH <gfxNNN> KIND rocke
#                         DERIVE_FROM <dir> KDP <file> INSTANCE <ukd name>
#                         [PACK_JOBS <j>])
#   Declare one probe: pack a descriptor root for ARCH and assert the output.
#
#   DERIVE_FROM copies <dir> at configure time into the build tree with the named KDP
#   trimmed to the one UKD whose `name` is INSTANCE, so the probe packs the real
#   descriptors but compiles a single kernel and the assertion expects exactly one
#   UKD. A missing or ambiguous INSTANCE is a configure error, never a skip.
#
#   Creates pack target hkp_packaging_probe_<n> (stamp and output under
#   ${CMAKE_BINARY_DIR}/hkp-probes/<n>/out) and ctest entry hkp-probe-<n>. Only
#   callable from a probes file loaded by hkp_load_packaging_probes().
# ---------------------------------------------------------------------------
function(hkp_add_packaging_probe)
    cmake_parse_arguments(PARSE_ARGV 0 ARG ""
        "NAME;ARCH;KIND;DERIVE_FROM;KDP;INSTANCE;PACK_JOBS" "")

    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "hkp probe: unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    foreach(_required NAME ARCH KIND DERIVE_FROM KDP INSTANCE)
        if(NOT ARG_${_required})
            message(FATAL_ERROR
                "hkp probe: hkp_add_packaging_probe requires ${_required}.")
        endif()
    endforeach()
    if(NOT ARG_KIND STREQUAL "rocke")
        message(FATAL_ERROR
            "hkp probe '${ARG_NAME}': unsupported KIND '${ARG_KIND}'. Supported "
            "kinds: rocke.")
    endif()

    get_property(_toolchain_set GLOBAL PROPERTY HKP_PROBE_TOOLCHAIN_SET)
    if(NOT _toolchain_set)
        message(FATAL_ERROR
            "hkp probe '${ARG_NAME}': hkp_add_packaging_probe is only valid from a "
            "probes file loaded by hkp_load_packaging_probes().")
    endif()
    get_property(_names GLOBAL PROPERTY HKP_PROBE_NAMES)
    if(ARG_NAME IN_LIST _names)
        message(FATAL_ERROR "hkp probe '${ARG_NAME}' is declared twice.")
    endif()

    get_property(_kpack_dir GLOBAL PROPERTY HKP_PROBE_ROCM_KPACK_DIR)
    get_property(_hipcc GLOBAL PROPERTY HKP_PROBE_HIPCC)
    get_property(_comgr_lib GLOBAL PROPERTY HKP_PROBE_COMGR_LIB)
    get_property(_rocke_args GLOBAL PROPERTY HKP_PROBE_ROCKE_ARGS)

    set(_probe_dir "${CMAKE_BINARY_DIR}/hkp-probes/${ARG_NAME}")
    set(_out_root "${_probe_dir}/out")

    _hkp_probe_check_out_root("${ARG_NAME}" "${_out_root}")

    set(_root "${_probe_dir}/root")
    _hkp_probe_derive_root("${ARG_NAME}" "${ARG_DERIVE_FROM}" "${ARG_KDP}"
                           "${ARG_INSTANCE}" "${_root}")

    if(NOT IS_DIRECTORY "${_root}")
        message(FATAL_ERROR
            "hkp probe '${ARG_NAME}': probe root is not a directory: ${_root}")
    endif()
    _hkp_root_covers_any_arch(_covers "${_root}" "${ARG_ARCH}")
    if(NOT _covers)
        message(FATAL_ERROR
            "hkp probe '${ARG_NAME}': no *.kdp.json under ${_root} ships for "
            "${ARG_ARCH}, so the pack would prune everything.")
    endif()

    set(_pack_jobs 1)
    if(ARG_PACK_JOBS)
        set(_pack_jobs "${ARG_PACK_JOBS}")
    endif()
    hkp_wire_pack_target(
        NAME "probe_${ARG_NAME}"
        SOURCE_ROOT "${_root}"
        ARCHES "${ARG_ARCH}"
        HIPCC "${_hipcc}"
        ROCM_KPACK_DIR "${_kpack_dir}"
        OUT_ROOT "${_out_root}"
        ${_rocke_args}
        PACK_JOBS "${_pack_jobs}")

    set(_assert_args
        --out-root "${_out_root}"
        --arch "${ARG_ARCH}"
        --kind "${ARG_KIND}"
        --kpack-python-dir "${_kpack_dir}"
        --stamp-name "${HKP_PACK_STAMP_NAME}"
        --expect-ukds 1)
    if(_comgr_lib)
        list(APPEND _assert_args --expect-comgr "${_comgr_lib}")
    endif()
    add_test(NAME "hkp-probe-${ARG_NAME}"
             COMMAND "${Python3_EXECUTABLE}"
                     "${HKP_PKG_DIR}/tools/hkp_probe_assert.py" ${_assert_args})
    set_tests_properties("hkp-probe-${ARG_NAME}" PROPERTIES
        ENVIRONMENT "PYTHONPATH=${HKP_PYTHON_ROOT}")

    set_property(GLOBAL APPEND PROPERTY HKP_PROBE_NAMES "${ARG_NAME}")
endfunction()

# ---------------------------------------------------------------------------
# hkp_load_packaging_probes(<rocm_kpack_dir> <hipcc> <rocke_comgr_lib> [<rocke args>...])
#   Store the toolchain hkp_add_packaging resolved, load the declared probes from
#   probes/probes.cmake, register the probe tooling tests, write the manifest of ctest
#   names (${CMAKE_BINARY_DIR}/hkp-probes/manifest.txt) and define the aggregate target
#   hkp_packaging_probes. Configuration fails when no probe is declared or pytest is not
#   importable by Python3_EXECUTABLE.
# ---------------------------------------------------------------------------
function(hkp_load_packaging_probes rocm_kpack_dir hipcc rocke_comgr_lib)
    set_property(GLOBAL PROPERTY HKP_PROBE_ROCM_KPACK_DIR "${rocm_kpack_dir}")
    set_property(GLOBAL PROPERTY HKP_PROBE_HIPCC "${hipcc}")
    set_property(GLOBAL PROPERTY HKP_PROBE_COMGR_LIB "${rocke_comgr_lib}")
    set_property(GLOBAL PROPERTY HKP_PROBE_ROCKE_ARGS "${ARGN}")
    set_property(GLOBAL PROPERTY HKP_PROBE_NAMES "")
    set_property(GLOBAL PROPERTY HKP_PROBE_TOOLCHAIN_SET TRUE)

    include("${HKP_PKG_DIR}/probes/probes.cmake")

    get_property(_names GLOBAL PROPERTY HKP_PROBE_NAMES)
    if(NOT _names)
        message(FATAL_ERROR
            "hkp probe: HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES is ON but "
            "${HKP_PKG_DIR}/probes/probes.cmake declared no probe.")
    endif()

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pytest"
        RESULT_VARIABLE _pytest_rc
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT _pytest_rc EQUAL 0)
        message(FATAL_ERROR
            "hkp: pytest is not importable by ${Python3_EXECUTABLE}, so the "
            "descriptor-packaging tests cannot run. Install pytest for that "
            "interpreter, or configure with "
            "-DHIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES=OFF.")
    endif()

    # Same environment contract as the pytest entries of hkp_register_tests: the probe
    # tooling tests read the kpack dir, hipcc and comgr from it and fail when absent.
    set(_pyenv "PYTHONPATH=${HKP_PYTHON_ROOT}" "HKP_HIPCC=${hipcc}")
    if(rocm_kpack_dir)
        list(APPEND _pyenv "HIPKERNELPROVIDER_ROCM_KPACK_DIR=${rocm_kpack_dir}")
    endif()
    if(rocke_comgr_lib)
        list(APPEND _pyenv "ROCKE_COMGR_LIB=${rocke_comgr_lib}")
    endif()
    add_test(NAME hkp-probe-tools
             COMMAND "${Python3_EXECUTABLE}" -m pytest "${HKP_PKG_DIR}/probes/tests" -v)
    set_tests_properties(hkp-probe-tools PROPERTIES ENVIRONMENT "${_pyenv}")

    set(_test_names hkp-probe-tools)
    set(_targets "")
    foreach(_name IN LISTS _names)
        list(APPEND _test_names "hkp-probe-${_name}")
        list(APPEND _targets "hkp_packaging_probe_${_name}")
    endforeach()
    list(SORT _test_names)
    string(REPLACE ";" "\n" _manifest "${_test_names}")
    file(WRITE "${CMAKE_BINARY_DIR}/hkp-probes/manifest.txt" "${_manifest}\n")

    add_custom_target(hkp_packaging_probes COMMENT "hkp: packaging probes")
    add_dependencies(hkp_packaging_probes ${_targets})
endfunction()
