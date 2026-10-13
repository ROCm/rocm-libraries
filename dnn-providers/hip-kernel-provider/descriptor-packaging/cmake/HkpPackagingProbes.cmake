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
# _hkp_probe_derive_root(<name> <from_dir> <arch> <out_dir> <expect_var> [<ukd>...])
#   Derive the probe root at configure time (FATAL with the tool's stderr when the root
#   cannot be derived, e.g. a kept KDP references a standalone UKD, or a listed UKD is
#   kept nowhere) and re-run configure when a source file or the derive tooling
#   changes. Each <ukd> is passed as one --ukd; with none, derive selects per compile
#   group (one UKD, or a value sweep for kinds with a sweep field). The tool writes
#   <out_dir>/../expect.json, the list of kept UKDs the assertion expects, each with the
#   pass it packs under (<arch>, or a generic target containing it); <expect_var> receives its path. FATAL when that list is empty: no
#   KDP under <from_dir> ships for <arch>, so the pack would prune everything. Deriving
#   is idempotent, so a reconfigure leaves the derived root's mtimes alone and does not
#   re-pack.
# ---------------------------------------------------------------------------
function(_hkp_probe_derive_root name from_dir arch out_dir expect_var)
    file(GLOB_RECURSE _derive_inputs CONFIGURE_DEPENDS "${from_dir}/*")
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" APPEND
                 PROPERTY CMAKE_CONFIGURE_DEPENDS ${_derive_inputs}
                          "${HKP_PKG_DIR}/tools/hkp_probe_derive_root.py"
                          "${HKP_PKG_DIR}/tools/hkp_probe_kinds.py"
                          "${HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON}")
    get_property(_filter_args GLOBAL PROPERTY HKP_PROBE_FILTER_ARGS)
    set(_ukd_args "")
    foreach(_ukd IN LISTS ARGN)
        list(APPEND _ukd_args --ukd "${_ukd}")
    endforeach()
    execute_process(
        COMMAND "${Python3_EXECUTABLE}"
                "${HKP_PKG_DIR}/tools/hkp_probe_derive_root.py"
                --from "${from_dir}"
                --generic-targets-json "${HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON}"
                --arch "${arch}"
                --out "${out_dir}"
                ${_filter_args}
                ${_ukd_args}
        RESULT_VARIABLE _derive_rc
        OUTPUT_VARIABLE _derive_out
        ERROR_VARIABLE _derive_err)
    # Exit 3: every listed UKD ships for <arch> but this build's filters (rocKE OFF)
    # remove it. The probe does not apply to this build; <expect_var> is set empty.
    set(${expect_var} "" PARENT_SCOPE)
    if(_derive_rc EQUAL 3)
        string(STRIP "${_derive_err}" _derive_err)
        message(STATUS "hkp: probe ${name} skipped: ${_derive_err}")
        return()
    endif()
    if(NOT _derive_rc EQUAL 0)
        message(FATAL_ERROR
            "hkp probe '${name}': deriving the probe root failed (exit "
            "${_derive_rc}).\n${_derive_err}")
    endif()

    get_filename_component(_expect "${out_dir}/../expect.json" ABSOLUTE)
    file(READ "${_expect}" _expect_json)
    string(JSON _expect_count LENGTH "${_expect_json}")
    if(_expect_count EQUAL 0)
        message(FATAL_ERROR
            "hkp probe '${name}': no *.kdp.json under ${from_dir} ships for ${arch}, "
            "so the pack would prune everything.")
    endif()
    message(STATUS
        "hkp: probe ${name}: derived root ${out_dir} (${_expect_count} UKD(s) for ${arch})")
    set(${expect_var} "${_expect}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# _hkp_probe_check_name(<name>)
#   FATAL unless <name> is usable as a target, ctest and directory name, is not the
#   reserved `tools`, is not declared yet, and the call comes from a probes file loaded
#   by hkp_load_packaging_probes().
# ---------------------------------------------------------------------------
function(_hkp_probe_check_name name)
    if(NOT name MATCHES "^[A-Za-z0-9_.+-]+$" OR name MATCHES "^\\.+$")
        message(FATAL_ERROR
            "hkp probe '${name}': NAME must match [A-Za-z0-9_.+-]+ and not be "
            "dots only (it names a target, a ctest entry and a build directory); "
            "pass NAME explicitly.")
    endif()
    if(name STREQUAL "tools")
        message(FATAL_ERROR
            "hkp probe '${name}': NAME 'tools' is reserved for the probe "
            "tooling test hkp-probe-tools.")
    endif()
    get_property(_toolchain_set GLOBAL PROPERTY HKP_PROBE_TOOLCHAIN_SET)
    if(NOT _toolchain_set)
        message(FATAL_ERROR
            "hkp probe '${name}': hkp_add_packaging_probe is only valid from a "
            "probes file loaded by hkp_load_packaging_probes().")
    endif()
    get_property(_names GLOBAL PROPERTY HKP_PROBE_NAMES)
    if(name IN_LIST _names)
        message(FATAL_ERROR "hkp probe '${name}' is declared twice.")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# _hkp_probe_check_ukds(<name> [<ukd>...])
#   FATAL when a UKDS entry is empty or listed twice.
# ---------------------------------------------------------------------------
function(_hkp_probe_check_ukds name)
    set(_seen "")
    foreach(_ukd IN LISTS ARGN)
        if(_ukd STREQUAL "")
            message(FATAL_ERROR "hkp probe '${name}': UKDS holds an empty UKD name.")
        endif()
        if(_ukd IN_LIST _seen)
            message(FATAL_ERROR
                "hkp probe '${name}': UKDS lists '${_ukd}' more than once.")
        endif()
        list(APPEND _seen "${_ukd}")
    endforeach()
endfunction()

# ---------------------------------------------------------------------------
# _hkp_probe_add_assert_test(<name> <out_root> <arch> <expect_file>)
#   Register ctest entry hkp-probe-<name>, which asserts the probe's packed output with
#   the kpack dir and comgr library hkp_load_packaging_probes() stored.
# ---------------------------------------------------------------------------
function(_hkp_probe_add_assert_test name out_root arch expect_file)
    get_property(_kpack_dir GLOBAL PROPERTY HKP_PROBE_ROCM_KPACK_DIR)
    get_property(_comgr_lib GLOBAL PROPERTY HKP_PROBE_COMGR_LIB)
    set(_assert_args
        --out-root "${out_root}"
        --arch "${arch}"
        --expect "${expect_file}"
        --kpack-python-dir "${_kpack_dir}"
        --stamp-name "${HKP_PACK_STAMP_NAME}")
    if(_comgr_lib)
        list(APPEND _assert_args --expect-comgr "${_comgr_lib}")
    endif()
    add_test(NAME "hkp-probe-${name}"
             COMMAND "${Python3_EXECUTABLE}"
                     "${HKP_PKG_DIR}/tools/hkp_probe_assert.py" ${_assert_args})
    set_tests_properties("hkp-probe-${name}" PROPERTIES
        ENVIRONMENT "PYTHONPATH=${HKP_PYTHON_ROOT}")
endfunction()

# ---------------------------------------------------------------------------
# hkp_add_packaging_probe(ARCH <gfxNNN> [ROOT <dir>] [NAME <n>] [UKDS <ukd-name>...]
#                         [PACK_JOBS <j>])
#   Declare one probe: pack a descriptor root for ARCH and assert the output.
#
#   ROOT defaults to the production descriptors, HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_
#   SOURCE_ROOT; a relative ROOT is resolved against the directory of the probes file.
#   It is copied at configure time into the build tree with every KDP that ships for
#   ARCH trimmed (see hkp_probe_derive_root.py). Without UKDS, each compile group of
#   such a KDP keeps one UKD, or for a kind with a sweep field (rocke: spec) enough UKDs
#   that each value of each spec field varying within the group is packed at least once
#   (combinations of values are not guaranteed). So the probe packs the real
#   descriptors but compiles each compile path, and each value, once. With UKDS, the probe packs exactly the listed UKDs (by
#   UKD name): KDPs keeping none of them are left out of the derived root, and
#   configure fails when a listed name is kept by no KDP shipping for ARCH, except that
#   a probe for which this build's filters (rocKE OFF) remove any listed UKD, or every UKD
#   shipping for ARCH when none are listed, is skipped whole with a STATUS line,
#   declaring nothing. The
#   assertion expects exactly the kept UKDs, each with the provenance of its producer
#   kind (see hkp_probe_assert.py). Adding a pack under an already probed ARCH needs no
#   new declaration unless the probe lists UKDS.
#
#   ARCH must be one concrete target as hkp_is_concrete_arch() defines it, the same
#   rule GPU_TARGETS goes through (gfx950, gfx1250-strict; not gfx9-4-generic or a
#   TheRock family name). NAME defaults to <ARCH> for the
#   production root and <ROOT basename>_<ARCH> for another ROOT; it must match
#   [A-Za-z0-9_.+-]+, must not be dots only, and must not be `tools` (hkp-probe-tools
#   is the probe tooling test).
#   UKDS entries must be non-empty and unique. PACK_JOBS caps the pack's worker
#   processes; omitted, the packer sizes its pool against the machine, min(32, ncpu),
#   as the production root does. Creates pack target
#   hkp_packaging_probe_<NAME> (stamp and output under
#   ${CMAKE_BINARY_DIR}/hkp-probes/<NAME>/out) and ctest entry hkp-probe-<NAME>. The
#   pack targets are part of `all` (hkp_wire_pack_target declares them so); the
#   aggregate hkp_packaging_probes builds every probe's pack. Only callable from a
#   probes file loaded by hkp_load_packaging_probes(), which also calls it for the
#   automatic arches.
# ---------------------------------------------------------------------------
function(hkp_add_packaging_probe)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "ARCH;ROOT;NAME;PACK_JOBS" "UKDS")

    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "hkp probe: unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT ARG_ARCH)
        message(FATAL_ERROR "hkp probe: hkp_add_packaging_probe requires ARCH.")
    endif()
    hkp_is_concrete_arch(_concrete "${ARG_ARCH}")
    if(NOT _concrete)
        message(FATAL_ERROR
            "hkp probe: ARCH '${ARG_ARCH}' is not one concrete architecture (a "
            "processor id such as gfx950, optionally with a distinct-target word such "
            "as gfx1250-strict; not a generic target or TheRock family name).")
    endif()
    if(NOT ARG_ROOT)
        set(ARG_ROOT "${HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_SOURCE_ROOT}")
        set(_default_name "${ARG_ARCH}")
    else()
        # A function sees the CMAKE_CURRENT_LIST_DIR of its caller: the probes file.
        get_filename_component(ARG_ROOT "${ARG_ROOT}" ABSOLUTE
                               BASE_DIR "${CMAKE_CURRENT_LIST_DIR}")
        get_filename_component(_root_base "${ARG_ROOT}" NAME)
        set(_default_name "${_root_base}_${ARG_ARCH}")
    endif()
    if(NOT ARG_NAME)
        set(ARG_NAME "${_default_name}")
    endif()
    _hkp_probe_check_name("${ARG_NAME}")
    if(NOT IS_DIRECTORY "${ARG_ROOT}")
        message(FATAL_ERROR
            "hkp probe '${ARG_NAME}': ROOT is not a directory: '${ARG_ROOT}'")
    endif()
    # `UKDS ""` (e.g. an empty variable) parses as a present but empty list.
    if("UKDS" IN_LIST ARG_KEYWORDS_MISSING_VALUES
       OR ("UKDS" IN_LIST ARGV AND ARG_UKDS STREQUAL ""))
        message(FATAL_ERROR
            "hkp probe '${ARG_NAME}': UKDS needs at least one UKD name.")
    endif()
    _hkp_probe_check_ukds("${ARG_NAME}" "${ARG_UKDS}")

    get_property(_kpack_dir GLOBAL PROPERTY HKP_PROBE_ROCM_KPACK_DIR)
    get_property(_hipcc GLOBAL PROPERTY HKP_PROBE_HIPCC)
    get_property(_root_args GLOBAL PROPERTY HKP_PROBE_ROOT_ARGS)

    set(_probe_dir "${CMAKE_BINARY_DIR}/hkp-probes/${ARG_NAME}")
    set(_out_root "${_probe_dir}/out")

    _hkp_probe_check_out_root("${ARG_NAME}" "${_out_root}")

    set(_root "${_probe_dir}/root")
    _hkp_probe_derive_root("${ARG_NAME}" "${ARG_ROOT}" "${ARG_ARCH}" "${_root}"
                           _expect_file ${ARG_UKDS})
    if(NOT _expect_file)
        return()
    endif()

    # Omitted PACK_JOBS passes nothing to the packer, which then uses its own default.
    set(_pack_jobs_args "")
    if(ARG_PACK_JOBS)
        list(APPEND _pack_jobs_args PACK_JOBS "${ARG_PACK_JOBS}")
    endif()
    hkp_wire_pack_target(
        NAME "probe_${ARG_NAME}"
        SOURCE_ROOT "${_root}"
        ARCHES "${ARG_ARCH}"
        HIPCC "${_hipcc}"
        ROCM_KPACK_DIR "${_kpack_dir}"
        OUT_ROOT "${_out_root}"
        ${_root_args}
        ${_pack_jobs_args})

    _hkp_probe_add_assert_test("${ARG_NAME}" "${_out_root}" "${ARG_ARCH}"
                               "${_expect_file}")

    set_property(GLOBAL APPEND PROPERTY HKP_PROBE_NAMES "${ARG_NAME}")
endfunction()

# ---------------------------------------------------------------------------
# _hkp_probe_automatic_arches(<out_var>)
#   The architectures to probe automatically: every explicit `arch` entry on a KDP or
#   UKD under HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_SOURCE_ROOT that is a concrete
#   target (hkp_is_concrete_arch), minus the build's own targets (hkp_selected_arches).
#   Descriptors with no `arch` (wildcards) name no arch: they ship for every build
#   target, so the build's own production pack covers them. A generic target in
#   HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON is probed through its first member (the
#   packer compiles the generic for any member and merges it into the member's folder),
#   unless the build targets one of its members, whose production pack compiles it
#   already; the STATUS line shows each generic -> member mapping. A TheRock family name
#   is skipped with a STATUS line: no build packs for it, because GPU_TARGETS drops it.
#   FATAL with the tool's stderr when the packer's loader rejects the root, including a
#   `*-generic` entry absent from the table. Configure
#   re-runs when a file under the root, the generic target table or the derive tooling
#   changes.
# ---------------------------------------------------------------------------
function(_hkp_probe_automatic_arches out_var)
    get_property(_filter_args GLOBAL PROPERTY HKP_PROBE_FILTER_ARGS)
    set(_root "${HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_SOURCE_ROOT}")
    file(GLOB_RECURSE _root_inputs CONFIGURE_DEPENDS "${_root}/*")
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" APPEND
                 PROPERTY CMAKE_CONFIGURE_DEPENDS ${_root_inputs}
                          "${HKP_PKG_DIR}/tools/hkp_probe_derive_root.py"
                          "${HKP_PKG_DIR}/tools/hkp_probe_kinds.py"
                          "${HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON}")
    hkp_selected_arches(_targets _targets_source)
    set(_exclude_args "")
    foreach(_target IN LISTS _targets)
        list(APPEND _exclude_args --exclude "${_target}")
    endforeach()
    execute_process(
        COMMAND "${Python3_EXECUTABLE}"
                "${HKP_PKG_DIR}/tools/hkp_probe_derive_root.py"
                --list-arches --from "${_root}"
                --generic-targets-json "${HIPDNN_PLUGIN_SDK_GPU_GENERIC_TARGETS_JSON}"
                ${_exclude_args}
                ${_filter_args}
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR
            "hkp probe: listing the architectures of the production root ${_root} "
            "failed (exit ${_rc}).\n${_err}")
    endif()
    string(REGEX REPLACE "\n+$" "" _out "${_out}")
    string(REPLACE "\n" ";" _lines "${_out}")

    # Each line is an arch followed by the generic targets it stands for.
    set(_named "")
    set(_auto "")
    set(_skipped "")
    set(_mapped "")
    foreach(_line IN LISTS _lines)
        separate_arguments(_words UNIX_COMMAND "${_line}")
        list(POP_FRONT _words _arch)
        list(APPEND _named "${_arch}")
        hkp_is_concrete_arch(_concrete "${_arch}")
        if(_concrete)
            list(APPEND _auto "${_arch}")
            foreach(_generic IN LISTS _words)
                list(APPEND _mapped "${_generic} -> ${_arch}")
            endforeach()
        else()
            list(APPEND _skipped "${_arch}")
        endif()
    endforeach()
    if(_mapped)
        list(JOIN _mapped ", " _mapped_text)
        message(STATUS
            "hkp: generic targets probed through their first member: [${_mapped_text}]")
    endif()
    if(_skipped)
        list(JOIN _skipped ", " _skipped_text)
        message(STATUS
            "hkp: no automatic probe for non-concrete arches [${_skipped_text}]: no "
            "build packs for them.")
    endif()
    if(_targets)
        list(REMOVE_ITEM _auto ${_targets})
    endif()
    list(JOIN _auto ", " _auto_text)
    list(JOIN _named ", " _named_text)
    list(JOIN _targets ", " _targets_text)
    message(STATUS
        "hkp: automatic probe arches: [${_auto_text}] (probe arches of the explicit "
        "arches under ${_root}: [${_named_text}]; build targets: [${_targets_text}])")
    set(${out_var} "${_auto}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# _hkp_probe_filter_args(<out_var> <enable_rocke> [<exclude_folder>...])
#   The derive tool's --exclude-folder / --disable-kind flags for the filters the
#   production roots are wired with, the disabled kinds from _hkp_disabled_kind_names().
# ---------------------------------------------------------------------------
function(_hkp_probe_filter_args out_var enable_rocke)
    _hkp_disabled_kind_names(_kinds "${enable_rocke}")
    set(_args "")
    foreach(_folder IN LISTS ARGN)
        list(APPEND _args --exclude-folder "${_folder}")
    endforeach()
    foreach(_kind IN LISTS _kinds)
        list(APPEND _args --disable-kind "${_kind}")
    endforeach()
    set(${out_var} "${_args}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# hkp_load_packaging_probes(<rocm_kpack_dir> <hipcc> <rocke_comgr_lib> [<root args>...])
#   Store the toolchain hkp_add_packaging resolved (<root args> are the keywords every
#   production root is wired with: ENABLE_ROCKE, the ROCKE_* toolchain when ON, and
#   EXCLUDE_FOLDERS). A build lacking a producer (rocKE OFF) still probes: derive and the
#   pack read the root under the same excluded folders and disabled kinds the production
#   pack gets, so rocKE content is neither packed nor expected and every other kind is
#   probed as usual; an arch left with no content names no probe. Load the named
#   probes from probes/probes.cmake, then declare one automatic probe per architecture the production
#   root names that the build does not target: the explicit `arch` entries of its KDPs
#   and UKDs (hkp_probe_derive_root.py --list-arches; wildcard descriptors name none,
#   generic targets map to a member, see _hkp_probe_automatic_arches) minus
#   hkp_selected_arches(), each as hkp_add_packaging_probe(ARCH <arch>). Register
#   the probe tooling tests, write the manifest of ctest names
#   (${CMAKE_BINARY_DIR}/hkp-probes/manifest.txt) and define the aggregate target
#   hkp_packaging_probes. Zero probes is valid: every explicit arch is a build target,
#   so the production pack covers it. Configuration fails when the production root
#   cannot be read or pytest is not importable by Python3_EXECUTABLE.
# ---------------------------------------------------------------------------
function(hkp_load_packaging_probes rocm_kpack_dir hipcc rocke_comgr_lib)
    set_property(GLOBAL PROPERTY HKP_PROBE_ROCM_KPACK_DIR "${rocm_kpack_dir}")
    set_property(GLOBAL PROPERTY HKP_PROBE_HIPCC "${hipcc}")
    set_property(GLOBAL PROPERTY HKP_PROBE_COMGR_LIB "${rocke_comgr_lib}")
    set_property(GLOBAL PROPERTY HKP_PROBE_ROOT_ARGS "${ARGN}")
    # The filters the production roots were wired with, read from those very arguments:
    # derive and the arch listing see the root as the pack step does, so what a probe
    # expects is what its pack ships.
    cmake_parse_arguments(_F "" "ENABLE_ROCKE" "EXCLUDE_FOLDERS" ${ARGN})
    _hkp_probe_filter_args(_filter_args "${_F_ENABLE_ROCKE}" ${_F_EXCLUDE_FOLDERS})
    set_property(GLOBAL PROPERTY HKP_PROBE_FILTER_ARGS "${_filter_args}")
    set_property(GLOBAL PROPERTY HKP_PROBE_NAMES "")
    set_property(GLOBAL PROPERTY HKP_PROBE_TOOLCHAIN_SET TRUE)

    include("${HKP_PKG_DIR}/probes/probes.cmake")
    _hkp_probe_automatic_arches(_auto_arches)
    foreach(_arch IN LISTS _auto_arches)
        hkp_add_packaging_probe(ARCH "${_arch}")
    endforeach()
    # Calls from anywhere but the probes file and the automatic probes above hit the
    # guard in hkp_add_packaging_probe.
    set_property(GLOBAL PROPERTY HKP_PROBE_TOOLCHAIN_SET FALSE)

    get_property(_names GLOBAL PROPERTY HKP_PROBE_NAMES)

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
    # Tests that pack rocKE content live under probes/tests/rocke/ and nowhere else; a
    # build with rocKE OFF has no rocKE toolchain to run them, so it never collects them
    # (as hkp_register_tests does for tests/rocke).
    set(_rocke_ignore "")
    if(NOT _F_ENABLE_ROCKE)
        set(_rocke_ignore "--ignore=${HKP_PKG_DIR}/probes/tests/rocke")
    endif()
    add_test(NAME hkp-probe-tools
             COMMAND "${Python3_EXECUTABLE}" -m pytest "${HKP_PKG_DIR}/probes/tests" -v
                     ${_rocke_ignore})
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
    if(_targets)
        add_dependencies(hkp_packaging_probes ${_targets})
    endif()
endfunction()
