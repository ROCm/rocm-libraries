# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Tests hipblaslt_version.cmake. Run as
#   cmake -DWORK_DIR=<dir> -P hipblaslt_version_tests.cmake
# Each case runs this script again with -DCASE=... in a child process, so a case
# that must stop configuration with FATAL_ERROR can be checked too.

cmake_minimum_required(VERSION 3.25)

include("${CMAKE_CURRENT_LIST_DIR}/hipblaslt_version.cmake")

if(DEFINED CASE)
    if(CASE STREQUAL "project")
        hipblaslt_resolve_project_version("${EXPLICIT}" "${ROCM}" "${ROOT}" _result)
    else()
        hipblaslt_build_version("${PROJECT}" "${PACKAGE}" "${REVISION}" _result)
    endif()
    message(STATUS "RESULT=[${_result}]")
    return()
endif()

if(NOT DEFINED WORK_DIR)
    message(FATAL_ERROR "hipblaslt_version_tests: WORK_DIR is not set")
endif()
file(REMOVE_RECURSE "${WORK_DIR}")

# Creates a fake ROCm root <name> with an optional manifest identity and
# .info/version line ("-" for none).
function(make_root name manifest info)
    set(_root "${WORK_DIR}/${name}")
    file(MAKE_DIRECTORY "${_root}")
    if(NOT manifest STREQUAL "-")
        file(WRITE "${_root}/share/therock/therock_manifest.json"
            "{\"rocm_package_version\": \"${manifest}\", \"rocm_version\": \"ignored\"}\n")
    endif()
    if(NOT info STREQUAL "-")
        file(WRITE "${_root}/.info/version" "${info}\n")
    endif()
endfunction()

make_root(manifest_only "10.3.0a20261116" "-")
make_root(info_only "-" "10.2.1-123")
make_root(local_manifest "git" "10.3.0")
make_root(both_agree "10.3.0rc1" "10.3.0")
make_root(disagree "10.3.0rc1" "10.2.0")
make_root(garbage "not-a-version" "nope")
make_root(empty "-" "-")

set(_failures 0)

# Runs one case and checks it printed <expected>, or failed when <expected>
# is FATAL. <expect_warning> is TRUE when the case must warn.
function(check name expected expect_warning)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" ${ARGN} -P "${CMAKE_CURRENT_LIST_FILE}"
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err)
    set(_all "${_out}${_err}")
    set(_ok TRUE)
    if(expected STREQUAL "FATAL")
        if(_rc EQUAL 0)
            set(_ok FALSE)
        endif()
    else()
        string(FIND "${_all}" "RESULT=[${expected}]" _at)
        if(NOT _rc EQUAL 0 OR _at EQUAL -1)
            set(_ok FALSE)
        endif()
    endif()
    string(FIND "${_all}" "CMake Warning" _warned)
    if(expect_warning AND _warned EQUAL -1)
        set(_ok FALSE)
    elseif(NOT expect_warning AND NOT _warned EQUAL -1)
        set(_ok FALSE)
    endif()
    if(_ok)
        message(STATUS "PASS ${name}")
    else()
        message(STATUS "FAIL ${name}: expected ${expected} (warning: ${expect_warning}), "
            "exit ${_rc}, output:\n${_all}")
        math(EXPR _failures "${_failures} + 1")
        set(_failures "${_failures}" PARENT_SCOPE)
    endif()
endfunction()

set(P -DCASE=project)
check(explicit_wins 1.2.3 FALSE ${P} -DEXPLICIT=1.2.3 -DROCM=10.3.0 "-DROOT=${WORK_DIR}/manifest_only")
check(rocm_version_beats_root 10.3.0 FALSE ${P} -DEXPLICIT= -DROCM=10.3.0 "-DROOT=${WORK_DIR}/info_only")
check(manifest_only 10.3.0 FALSE ${P} -DEXPLICIT= -DROCM= "-DROOT=${WORK_DIR}/manifest_only")
check(info_only 10.2.1 FALSE ${P} -DEXPLICIT= -DROCM= "-DROOT=${WORK_DIR}/info_only")
check(local_manifest_uses_info 10.3.0 FALSE ${P} -DEXPLICIT= -DROCM= "-DROOT=${WORK_DIR}/local_manifest")
check(both_agree 10.3.0 FALSE ${P} -DEXPLICIT= -DROCM= "-DROOT=${WORK_DIR}/both_agree")
check(disagree_falls_back 0.0.0 TRUE ${P} -DEXPLICIT= -DROCM= "-DROOT=${WORK_DIR}/disagree")
check(garbage_falls_back 0.0.0 TRUE ${P} -DEXPLICIT= -DROCM= "-DROOT=${WORK_DIR}/garbage")
check(empty_root_falls_back 0.0.0 TRUE ${P} -DEXPLICIT= -DROCM= "-DROOT=${WORK_DIR}/empty")
check(missing_root_falls_back 0.0.0 TRUE ${P} -DEXPLICIT= -DROCM= "-DROOT=${WORK_DIR}/missing")
check(two_parts_is_fatal FATAL FALSE ${P} -DEXPLICIT=10.3 -DROCM= "-DROOT=${WORK_DIR}/empty")
check(part_above_999_is_fatal FATAL FALSE ${P} -DEXPLICIT= -DROCM=10.1000.0 "-DROOT=${WORK_DIR}/empty")
check(part_999_is_allowed 999.999.999 FALSE ${P} -DEXPLICIT=999.999.999 -DROCM= "-DROOT=${WORK_DIR}/empty")

set(B -DCASE=build -DPROJECT=10.3.0 -DREVISION=f4e24d58b)
check(release "10.3.0 (git f4e24d58b)" FALSE ${B} -DPACKAGE=10.3.0)
check(release_candidate "10.3.0rc2 (git f4e24d58b)" FALSE ${B} -DPACKAGE=10.3.0rc2)
check(nightly "10.3.0a20261103 (git f4e24d58b)" FALSE ${B} -DPACKAGE=10.3.0a20261103)
check(bkc "10.3.0a20261103+bkc.20261110 (git f4e24d58b)" FALSE ${B} -DPACKAGE=10.3.0a20261103+bkc.20261110)
check(dev "10.3.0.dev0+f4e24d58b (git f4e24d58b)" FALSE ${B} -DPACKAGE=10.3.0.dev0+f4e24d58b)
check(local_git "10.3.0.dev0 (git f4e24d58b)" FALSE ${B} -DPACKAGE=git)
check(standalone "10.3.0.dev0 (git f4e24d58b)" FALSE ${B} -DPACKAGE=)
check(mismatched_package_warns "10.4.0a20261103 (git f4e24d58b)" TRUE ${B} -DPACKAGE=10.4.0a20261103)
check(no_revision "10.3.0.dev0 (git unknown)" FALSE -DCASE=build -DPROJECT=10.3.0 -DPACKAGE= -DREVISION=)
string(REPEAT "a" 250 _long)
check(too_long_is_fatal FATAL FALSE -DCASE=build -DPROJECT=10.3.0 -DPACKAGE= "-DREVISION=${_long}")

if(_failures GREATER 0)
    message(FATAL_ERROR "${_failures} hipblaslt_version case(s) failed")
endif()
message(STATUS "All hipblaslt_version cases passed")
