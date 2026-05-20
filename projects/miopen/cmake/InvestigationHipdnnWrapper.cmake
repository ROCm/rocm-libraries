# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Phase 1 wiring for RFC 0001 (MIOpen -> hipDNN forwarding wrapper).
#
# This module is included from the top-level CMakeLists.txt ONLY when
# MIOPEN_ENABLE_HIPDNN_WRAPPER is ON. The flag-off build never sees this
# file's effects, preserving the §1 byte-equivalence constraint.
#
# The two Phase 1 source artifacts (Q1 rename header and Q4 wrapper source)
# are hand-maintained in src/private/ and consumed by the public/private split
# wired in src/CMakeLists.txt. This module retains the count-parity CTest
# (the only piece of automation that survives both generators' retirement —
# RFC §6 Q1, Q4) plus the optional consumer-smoke CTest (Q5).

if(NOT MIOPEN_ENABLE_HIPDNN_WRAPPER)
    message(FATAL_ERROR
        "InvestigationHipdnnWrapper.cmake included with the flag OFF — guard the include site")
endif()

set(_INV_TOOLS_DIR     "${PROJECT_SOURCE_DIR}/tools/wrapper")
set(_INV_PRIVATE_DIR   "${PROJECT_SOURCE_DIR}/src/private")
set(_INV_PUBLIC_HEADER "${PROJECT_SOURCE_DIR}/include/miopen/miopen.h")
set(_INV_RENAME_HEADER "${_INV_PRIVATE_DIR}/miopen_private_rename.h")
set(_INV_WRAPPER_SRC   "${_INV_PRIVATE_DIR}/wrapper.cpp")

# Validation tests, surfaced through CTest. These run unconditionally in
# flag-on builds; the §4 validation matrix calls for them in CI.
if(BUILD_TESTING)
    # Q1/Q4 drift parity: miopen.h, miopen_private_rename.h, and wrapper.cpp
    # must all agree on the public entry-point count. This is the single
    # check that survives both generators' retirement (RFC §6 Q1, Q4).
    add_test(
        NAME    investigation_q4_stub_count
        COMMAND "${CMAKE_COMMAND}"
            -DHEADER=${_INV_PUBLIC_HEADER}
            -DRENAME=${_INV_RENAME_HEADER}
            -DSRC=${_INV_WRAPPER_SRC}
            -P "${_INV_TOOLS_DIR}/check_stub_count.cmake")
    set_tests_properties(investigation_q4_stub_count PROPERTIES
        LABELS "investigation;hipdnn_wrapper")

    # Q5 consumer smoke: only meaningful after `cmake --install`. Off by
    # default; enable with -DMIOPEN_INVESTIGATION_INSTALL_PREFIX=...
    if(MIOPEN_INVESTIGATION_INSTALL_PREFIX)
        add_test(
            NAME    investigation_q5_consumer_smoke
            COMMAND bash "${_INV_TOOLS_DIR}/check_consumer_smoke.sh"
                --prefix "${MIOPEN_INVESTIGATION_INSTALL_PREFIX}")
        set_tests_properties(investigation_q5_consumer_smoke PROPERTIES
            LABELS "investigation;hipdnn_wrapper")

        # Q6 find_package smoke (RemainingWork item 7): builds an external
        # CMake consumer via find_package(miopen) and verifies DT_NEEDED on
        # the wrapper. Same opt-in gate as Q5 — needs an installed prefix.
        add_test(
            NAME    investigation_q6_find_package_smoke
            COMMAND bash "${_INV_TOOLS_DIR}/check_find_package_smoke.sh"
                --prefix "${MIOPEN_INVESTIGATION_INSTALL_PREFIX}")
        set_tests_properties(investigation_q6_find_package_smoke PROPERTIES
            LABELS "investigation;hipdnn_wrapper")
    endif()

    # Q2 superset diff (RemainingWork item 5): assert the flag-on libMIOpen.so
    # exports a superset of the flag-off baseline's public miopen* symbols and
    # that SONAME matches. Opt-in: needs a baseline dump created with
    # `tools/wrapper/symbol_diff.sh dump <flag-off-libMIOpen.so> --out <prefix>`
    # passed via -DMIOPEN_WRAPPER_FLAGOFF_BASELINE=<prefix>. The test first
    # dumps the wrapper-on candidate to ${CMAKE_BINARY_DIR}/wrapper_symbols.*
    # via a fixture, then invokes the diff.
    if(MIOPEN_WRAPPER_FLAGOFF_BASELINE)
        add_test(
            NAME    investigation_q2_dump_candidate
            COMMAND bash "${_INV_TOOLS_DIR}/symbol_diff.sh" dump
                "$<TARGET_FILE:MIOpen>"
                --out "${CMAKE_BINARY_DIR}/wrapper_symbols")
        set_tests_properties(investigation_q2_dump_candidate PROPERTIES
            LABELS         "investigation;hipdnn_wrapper"
            FIXTURES_SETUP q2_candidate_dump)
        add_test(
            NAME    investigation_q2_symbol_superset
            COMMAND bash "${_INV_TOOLS_DIR}/symbol_diff.sh" diff
                --baseline  "${MIOPEN_WRAPPER_FLAGOFF_BASELINE}"
                --candidate "${CMAKE_BINARY_DIR}/wrapper_symbols")
        set_tests_properties(investigation_q2_symbol_superset PROPERTIES
            LABELS            "investigation;hipdnn_wrapper"
            FIXTURES_REQUIRED q2_candidate_dump)
    endif()
endif()

message(STATUS "InvestigationHipdnnWrapper: hand-maintained sources at ${_INV_TOOLS_DIR}")
message(STATUS "InvestigationHipdnnWrapper: run \"ctest -L investigation\" to validate")
