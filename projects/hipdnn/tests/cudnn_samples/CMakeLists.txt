# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Compiles NVIDIA's unmodified cudnn-frontend samples/cpp corpus against the *installed*
# hipDNN cuDNN-compatibility shim, and runs what compiles.
#
# This is a standalone project configured against an install tree, not a subdirectory of
# the hipDNN build. It installs to share/hipdnn/cudnn_samples/ and is configured at test
# time the way a consumer would configure it.

cmake_minimum_required(VERSION 3.25)

# Must precede project(): on Windows, CMake's clang platform module has already set
# CMAKE_BUILD_TYPE to Debug by the time project() returns, so a default applied after it
# would never fire. Before project() the variable reflects only what the user asked for.
#
# It matters because the debug CRT this selects cannot link against a release-built Catch2
# or hipDNN -- lld-link rejects the _ITERATOR_DEBUG_LEVEL mismatch. That happens at the
# link step, after every TU has compiled cleanly, so it reads as a corpus failure rather
# than a configuration one.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(hipdnn_cudnn_samples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
enable_testing()

set(HIPDNN_CUDNN_SAMPLES_CORPUS_DIR "" CACHE PATH
    "Existing cudnn-frontend checkout to use instead of cloning the pinned corpus")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---------------------------------------------------------------------------------------
# Is the shim installed at all?
# ---------------------------------------------------------------------------------------
# The package exports cudnn_compatibility as an optional component, so this reports what
# hipDNN was *built* with rather than what happens to be lying on disk. With the shim off
# we register one skipped test and stop: no clone, no staging, no build. A job that
# quietly returns success while testing nothing is worse than a red job, so the skip has
# to be a distinct ctest status and not an empty pass.
find_package(hipdnn_frontend CONFIG REQUIRED OPTIONAL_COMPONENTS cudnn_compatibility)

if(NOT hipdnn_frontend_cudnn_compatibility_FOUND)
    message(STATUS
        "hipDNN was built without HIPDNN_ENABLE_CUDNN_COMPATIBILITY; "
        "registering a single skipped test and building no samples")
    # SKIP_RETURN_CODE only fires on that exact exit status, so the command has to produce
    # it. `cmake -E echo` exits 0, which ctest would report as a pass -- the precise
    # failure mode this path exists to prevent.
    add_test(NAME cudnn_samples_skipped
             COMMAND ${Python3_EXECUTABLE} -c
                     "import sys; print('cuDNN sample corpus skipped: hipDNN was built without HIPDNN_ENABLE_CUDNN_COMPATIBILITY'); sys.exit(77)")
    set_tests_properties(cudnn_samples_skipped PROPERTIES SKIP_RETURN_CODE 77)
    return()
endif()

set(REPORT_DIR "${CMAKE_BINARY_DIR}/cudnn_samples_report")
file(MAKE_DIRECTORY "${REPORT_DIR}")
set(PROBE "${CMAKE_CURRENT_SOURCE_DIR}/sample_probe.py")

# ---------------------------------------------------------------------------------------
# Catch2
# ---------------------------------------------------------------------------------------
# 3.3.0 is the floor for SKIP(); pin the fallback to the version already used elsewhere in
# this repo rather than upstream cudnn-frontend's older floor.
find_package(Catch2 3.3.0 QUIET)
if(NOT Catch2_FOUND)
    message(STATUS "Catch2 >= 3.3.0 not found; fetching v3.13.0")
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.13.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(Catch2)
endif()

# ---------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/manifest.json" MANIFEST)

string(JSON CORPUS_REPO GET "${MANIFEST}" corpus repo)
string(JSON CORPUS_TAG GET "${MANIFEST}" corpus tag)
string(JSON CORPUS_SPARSE_PATH GET "${MANIFEST}" corpus sparse_path)
string(JSON CORPUS_FE_VERSION GET "${MANIFEST}" corpus frontend_version)
string(JSON EXPECT_CUDART_VERSION GET "${MANIFEST}" environment CUDART_VERSION)
string(JSON EXPECT_CUDNN_VERSION GET "${MANIFEST}" environment CUDNN_VERSION)
string(JSON SAMPLE_COUNT LENGTH "${MANIFEST}" samples)

# ---------------------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------------------
if(HIPDNN_CUDNN_SAMPLES_CORPUS_DIR)
    set(CORPUS_ROOT "${HIPDNN_CUDNN_SAMPLES_CORPUS_DIR}")
    message(STATUS "Using the supplied cudnn-frontend checkout: ${CORPUS_ROOT}")
else()
    set(CORPUS_ROOT "${CMAKE_BINARY_DIR}/corpus")
    if(NOT EXISTS "${CORPUS_ROOT}/${CORPUS_SPARSE_PATH}")
        message(STATUS "Cloning ${CORPUS_REPO} ${CORPUS_TAG} (sparse)")
        file(REMOVE_RECURSE "${CORPUS_ROOT}")
        execute_process(
            COMMAND git clone --depth 1 --branch ${CORPUS_TAG} --filter=blob:none --sparse
                    ${CORPUS_REPO} "${CORPUS_ROOT}"
            RESULT_VARIABLE _rc ERROR_VARIABLE _out OUTPUT_VARIABLE _out)
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR "E22 CORPUS: clone of ${CORPUS_REPO} ${CORPUS_TAG} failed\n${_out}")
        endif()
        # `include` is not staged or compiled against; it is here only so the version pin
        # below can be checked against the corpus's own declaration.
        execute_process(
            COMMAND git -C "${CORPUS_ROOT}" sparse-checkout set ${CORPUS_SPARSE_PATH} include
            RESULT_VARIABLE _rc ERROR_VARIABLE _out OUTPUT_VARIABLE _out)
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR "E22 CORPUS: sparse-checkout failed\n${_out}")
        endif()
    endif()
endif()

set(CORPUS_CPP "${CORPUS_ROOT}/${CORPUS_SPARSE_PATH}")
if(NOT EXISTS "${CORPUS_CPP}")
    message(FATAL_ERROR "E22 CORPUS: ${CORPUS_CPP} does not exist")
endif()

# The shim static_asserts on this version, so a drifted corpus would otherwise fail deep
# inside an unrelated TU with a confusing diagnostic.
set(_fe_version_hdr "${CORPUS_ROOT}/include/cudnn_frontend_version.h")
if(NOT EXISTS "${_fe_version_hdr}")
    message(FATAL_ERROR "E20 PIN: ${_fe_version_hdr} is missing; cannot verify the corpus version")
endif()
file(READ "${_fe_version_hdr}" _fe_version_header)
# CUDNN_FRONTEND_VERSION is itself a computed expression, so matching it directly would
# capture the literal 10000 out of the arithmetic. Read the three components and redo the
# arithmetic the way the header does.
foreach(_part MAJOR MINOR PATCH)
    if(NOT _fe_version_header MATCHES "#define[ \t]+CUDNN_FRONTEND_${_part}_VERSION[ \t]+([0-9]+)")
        message(FATAL_ERROR
            "E20 PIN: cannot read CUDNN_FRONTEND_${_part}_VERSION from ${_fe_version_hdr}")
    endif()
    set(_fe_${_part} "${CMAKE_MATCH_1}")
endforeach()
math(EXPR _fe_version "(${_fe_MAJOR} * 10000) + (${_fe_MINOR} * 100) + ${_fe_PATCH}")
if(NOT _fe_version EQUAL CORPUS_FE_VERSION)
    message(FATAL_ERROR
        "E20 PIN: corpus reports CUDNN_FRONTEND_VERSION ${_fe_version} "
        "(${_fe_MAJOR}.${_fe_MINOR}.${_fe_PATCH}), manifest expects ${CORPUS_FE_VERSION}")
endif()

# Stage samples/cpp only. One level up, samples/CMakeLists.txt pulls in legacy_samples/,
# which carries a second, name-colliding utils/helpers.h.
set(STAGE_DIR "${CMAKE_BINARY_DIR}/staged/cpp")
file(GLOB_RECURSE CORPUS_SOURCES RELATIVE "${CORPUS_CPP}" "${CORPUS_CPP}/*.cpp")
foreach(rel IN LISTS CORPUS_SOURCES)
    configure_file("${CORPUS_CPP}/${rel}" "${STAGE_DIR}/${rel}" COPYONLY)
endforeach()
# Ours, not upstream's. The corpus is not self-contained by its own include lines -- it
# leans on an upstream force-included PCH for <cudnn.h>, <memory>, <string> and
# <iostream> -- and upstream's copy also redefines handle helpers the shim already owns.
# Upstream's utils/ is deliberately not staged, which turns the one unavoidable change
# into a substitution rather than an edit.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/utils/helpers.h" "${STAGE_DIR}/utils/helpers.h" COPYONLY)

# ---------------------------------------------------------------------------------------
# Version pins that only a compile can check
# ---------------------------------------------------------------------------------------
# CUDNN_VERSION is a computed macro, so scraping the installed header for a literal would
# not work. CUDART_VERSION reaches the corpus via the overlay, which derives it from the
# shim's own claim; this asserts the value that derivation lands on.
set(_pin_probe "${CMAKE_BINARY_DIR}/version_pin_probe.cpp")
file(WRITE "${_pin_probe}" "\
#include <cuda_runtime.h>
#include <cudnn.h>
static_assert(CUDNN_VERSION == ${EXPECT_CUDNN_VERSION}, \"installed CUDNN_VERSION != manifest\");
static_assert(CUDART_VERSION == ${EXPECT_CUDART_VERSION}, \"overlay CUDART_VERSION != manifest\");
")
# STATIC_LIBRARY makes this compile-only while LINK_LIBRARIES still applies the target's
# usage requirements -- which is how the probe picks up HIP's include dir, since that
# arrives transitively rather than on hipdnn_frontend's own include property.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
try_compile(PIN_PROBE_OK "${CMAKE_BINARY_DIR}/version_pin_probe"
    SOURCES "${_pin_probe}"
    LINK_LIBRARIES hipdnn_frontend
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CURRENT_SOURCE_DIR}/overlay"
    CXX_STANDARD 17
    OUTPUT_VARIABLE PIN_PROBE_OUTPUT)
unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
if(NOT PIN_PROBE_OK)
    message(FATAL_ERROR
        "E21 PIN: installed cuDNN version macros disagree with the manifest "
        "(expected CUDNN_VERSION=${EXPECT_CUDNN_VERSION}, CUDART_VERSION=${EXPECT_CUDART_VERSION})\n"
        "${PIN_PROBE_OUTPUT}")
endif()

# ---------------------------------------------------------------------------------------
# Per-sample registration
# ---------------------------------------------------------------------------------------
if(HIPDNN_ENABLE_SDPA)
    message(STATUS "Installed hipDNN has SDPA enabled; the whole manifest applies")
else()
    message(STATUS "Installed hipDNN has SDPA disabled; requires_sdpa entries become EXCLUDED")
endif()

# Used by the EXCLUDED rot checks to confirm a header the shim is supposed to lack is
# still absent from the installed tree.
get_target_property(_fe_includes hipdnn_frontend INTERFACE_INCLUDE_DIRECTORIES)

set(SEEN_SOURCES "")
if(NOT SAMPLE_COUNT MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "E23 MANIFEST: the manifest declares no samples")
endif()
math(EXPR LAST_SAMPLE "${SAMPLE_COUNT} - 1")

# RANGE carries an explicit start and step throughout this file. cmake-lint accepts
# only the three-argument form, and that form is also the only one that rejects a
# degenerate count outright: `RANGE n` and `RANGE 0 n` both walk [0, -1] when the
# count is zero, whereas `RANGE 0 -1 1` is a hard CMake error. The positive-integer
# guards on each length make that unreachable in any case.
foreach(i RANGE 0 ${LAST_SAMPLE} 1)
    string(JSON entry GET "${MANIFEST}" samples ${i})
    string(JSON tu GET "${entry}" path)
    string(JSON tier GET "${entry}" tier)

    if(NOT EXISTS "${STAGE_DIR}/${tu}")
        message(FATAL_ERROR "E23 MANIFEST: entry '${tu}' names no staged source")
    endif()
    if(tu IN_LIST SEEN_SOURCES)
        message(FATAL_ERROR "E23 MANIFEST: '${tu}' appears more than once")
    endif()
    list(APPEND SEEN_SOURCES "${tu}")

    string(JSON reason ERROR_VARIABLE _ignored GET "${entry}" reason)
    if(reason MATCHES "NOTFOUND$")
        set(reason "")
    endif()
    string(JSON requires_sdpa ERROR_VARIABLE _ignored GET "${entry}" requires_sdpa)
    if(requires_sdpa STREQUAL "ON" AND NOT HIPDNN_ENABLE_SDPA)
        set(tier "EXCLUDED")
        set(reason "SDPA disabled in the installed hipDNN")
    endif()

    string(MAKE_C_IDENTIFIER "${tu}" tu_id)
    set(target "tu_${tu_id}")

    # ---- EXCLUDED: no target, but the reason is re-checked rather than trusted.
    if(tier STREQUAL "EXCLUDED")
        string(JSON rot ERROR_VARIABLE _ignored GET "${entry}" rot_check)
        if(NOT rot MATCHES "NOTFOUND$")
            string(JSON _inc_len ERROR_VARIABLE _ignored LENGTH "${rot}" includes_present)
            # Matching a positive integer rejects the NOTFOUND sentinel and an empty
            # array in one test; on an empty one the loop below would walk [0, -1].
            if(_inc_len MATCHES "^[1-9][0-9]*$")
                file(READ "${STAGE_DIR}/${tu}" _tu_text)
                math(EXPR _last "${_inc_len} - 1")
                foreach(j RANGE 0 ${_last} 1)
                    string(JSON _inc GET "${rot}" includes_present ${j})
                    if(NOT _tu_text MATCHES "#include[ \t]*<${_inc}>")
                        message(FATAL_ERROR
                            "E24 ROT: '${tu}' is EXCLUDED because it includes <${_inc}>, "
                            "but that include is gone. Re-tier it.")
                    endif()
                endforeach()
            endif()
            string(JSON _abs_len ERROR_VARIABLE _ignored LENGTH "${rot}" headers_absent)
            if(_abs_len MATCHES "^[1-9][0-9]*$")
                math(EXPR _last "${_abs_len} - 1")
                foreach(j RANGE 0 ${_last} 1)
                    string(JSON _hdr GET "${rot}" headers_absent ${j})
                    foreach(_dir IN LISTS _fe_includes)
                        if(EXISTS "${_dir}/hipdnn_compatibility/cudnn/${_hdr}")
                            message(FATAL_ERROR
                                "E24 ROT: '${tu}' is EXCLUDED because the shim ships no "
                                "${_hdr}, but it now does. Re-tier it.")
                        endif()
                    endforeach()
                endforeach()
            endif()
        endif()
        # Static outcome, so the sidecar is written now: there is no ctest case to write it
        # later, and the report must still account for every manifest entry.
        file(WRITE "${REPORT_DIR}/${tu_id}.json"
             "{\n  \"tu\": \"${tu}\",\n  \"tier\": \"EXCLUDED\",\n  \"outcome\": \"excluded\",\n  \"reason\": \"${reason}\"\n}\n")
        continue()
    endif()

    # ---- Sources: normally one, but a manifest "group" adds the TUs holding definitions
    # this one only declares. Upstream links all 64 into a single binary and so never has
    # to know about the dependency.
    set(sources "${STAGE_DIR}/${tu}")
    string(JSON _grp_len ERROR_VARIABLE _ignored LENGTH "${entry}" group)
    if(_grp_len MATCHES "^[1-9][0-9]*$")
        math(EXPR _last "${_grp_len} - 1")
        foreach(j RANGE 0 ${_last} 1)
            string(JSON _dep GET "${entry}" group ${j})
            if(NOT EXISTS "${STAGE_DIR}/${_dep}")
                message(FATAL_ERROR "E23 MANIFEST: '${tu}' groups '${_dep}', which is not staged")
            endif()
            list(APPEND sources "${STAGE_DIR}/${_dep}")
        endforeach()
    endif()

    # Deliberately part of the default target: the driver builds everything once with
    # `-k 0`, so the per-target compile tests below are no-ops for the ones that succeeded.
    add_executable(${target} ${sources})
    # The overlay must be a plain -I, never SYSTEM. hipDNN's installed include dir arrives
    # as -isystem, and compilers search every -I before every -isystem, so marking the
    # overlay SYSTEM would let the real headers win and silently defeat it.
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/overlay" "${STAGE_DIR}")
    # hipdnn_frontend, not hipdnn_frontend_dynamic: a direct backend link has no runtime
    # discovery step to go wrong, and discovery is not what this job tests.
    target_link_libraries(${target} PRIVATE hipdnn_frontend Catch2::Catch2WithMain)
    # No -Werror. Upstream builds the corpus with it; we deliberately do not.

    # Compilation is itself a ctest case. A driver that stopped at the first build error
    # would report nothing about the other 55 samples, which is the most likely failure.
    add_test(NAME compile_${tu_id}
             COMMAND ${Python3_EXECUTABLE} "${PROBE}"
                     --mode compile --tu "${tu}" --tier "${tier}" --report-dir "${REPORT_DIR}"
                     -- ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}" --target ${target})
    set_tests_properties(compile_${tu_id} PROPERTIES FIXTURES_SETUP fx_${tu_id})

    if(tier STREQUAL "XFAIL_COMPILE")
        string(JSON anchor GET "${entry}" anchor)
        if(anchor STREQUAL "")
            message(FATAL_ERROR "E23 MANIFEST: XFAIL_COMPILE entry '${tu}' has an empty anchor")
        endif()
        # PASS_REGULAR_EXPRESSION alone, never with WILL_FAIL. Once a pass regex is set,
        # ctest ignores the return code, so WILL_FAIL would invert the regex verdict: a TU
        # that started compiling would pass, and a TU failing for a *different* reason
        # would also pass. The regex on its own is correct in all three cases, which is
        # what stops the manifest rotting silently.
        set_tests_properties(compile_${tu_id} PROPERTIES PASS_REGULAR_EXPRESSION "${anchor}")
    else()
        # --allow-running-no-tests is mandatory: Catch2 exits non-zero when every case in a
        # run skips, and most TUs hold exactly one TEST_CASE while the pinned arch
        # predicates keep many of them closed.
        add_test(NAME run_${tu_id}
                 COMMAND ${Python3_EXECUTABLE} "${PROBE}"
                         --mode run --tu "${tu}" --tier "${tier}" --report-dir "${REPORT_DIR}"
                         -- $<TARGET_FILE:${target}> --allow-running-no-tests)
        # A sample that wedges on the GPU must fail its own case rather than hang the job
        # until the workflow-level timeout kills it and reports nothing about anything.
        set_tests_properties(run_${tu_id} PROPERTIES
            FIXTURES_REQUIRED fx_${tu_id}
            TIMEOUT 300)
    endif()
endforeach()

# Every staged source must be accounted for, or a corpus that grew a TU goes untested.
foreach(rel IN LISTS CORPUS_SOURCES)
    if(NOT rel IN_LIST SEEN_SOURCES)
        message(FATAL_ERROR "E23 MANIFEST: staged source '${rel}' has no manifest entry")
    endif()
endforeach()

list(LENGTH CORPUS_SOURCES _staged_count)
message(STATUS "cuDNN sample corpus: ${_staged_count} staged, ${SAMPLE_COUNT} manifest entries")
