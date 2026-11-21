###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in
 # all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # THE SOFTWARE.
 #
 ###############################################################################

 add_hiptensor_unit_test(logger_test ${CMAKE_CURRENT_SOURCE_DIR}/logger_test.cpp)
 add_hiptensor_unit_test(yaml_test ${CMAKE_CURRENT_SOURCE_DIR}/yaml_test.cpp)
 add_hiptensor_unit_test(elementwise_op_test
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_op_test.cpp
     )
 add_hiptensor_unit_test(util_test
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/util_test.cpp
     )
 add_hiptensor_unit_test(hiptensor_options_test
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/hiptensor_options_test.cpp
     )
 add_hiptensor_unit_test(plan_lifetime_test
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/plan_lifetime_test.cpp
     )
 add_hiptensor_unit_test(plan_cache_lru_test
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/plan_cache_lru_test.cpp
     )
 add_hiptensor_unit_test(batched_contraction_test
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/batched_contraction_test.cpp
     )
add_hiptensor_unit_test(elementwise_mode_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_mode_test.cpp
    )
add_hiptensor_unit_test(elementwise_broadcast_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_broadcast_test.cpp
    )
add_hiptensor_unit_test(device_init_test
    ${CMAKE_CURRENT_SOURCE_DIR}/device_init_test.cpp
    )
add_hiptensor_unit_test(concurrency_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/concurrency_test.cpp
    )
add_hiptensor_unit_test(backend_seam_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/backend_seam_test.cpp
    )
# The rocKE backend's kernels through the public API (HIPTENSOR_BACKEND=rocke, no fallback).
if(HIPTENSOR_ENABLE_ROCKE_BACKEND)
    # The C++ pack builder lists the kernels, attributes, routes and JIT classes rocKE's Python
    # reference lists (the manifests the rocKE tree keeps in library/builders/hiptensor/
    # manifests). No GPU, no compile; a build from prebuilt packs has no builder to check.
    set(_ROCKE_MANIFESTS "${HIPTENSOR_ROCKE_SOURCE_DIR}/library/builders/hiptensor/manifests")
    if(TARGET hiptensor_rocke_pack_builder AND EXISTS "${_ROCKE_MANIFESTS}")
        set(_ROCKE_MANIFEST_ARCHS)
        foreach(target IN LISTS HIPTENSOR_ROCKE_SUPPORTED_TARGETS)
            list(APPEND _ROCKE_MANIFEST_ARCHS --arch ${target})
        endforeach()
        add_test(NAME rocke_pack_manifest_test
                 COMMAND hiptensor_rocke_pack_builder --list ${_ROCKE_MANIFEST_ARCHS}
                         --check-manifests "${_ROCKE_MANIFESTS}")
        # rocKE's cross-check of this builder (library/tests/test_hiptensor_pack_cross_check.py,
        # which skips without ROCKE_HIPTENSOR_PACK_BUILDER): the C++ builder's manifests, its
        # untuned reduction and tuned elementwise picks, and the packs it writes for this
        # build's pack targets byte for byte against rocKE's Python writer with the COMGR the
        # builder links. Needs a Python with pytest; no GPU (every kernel compiles twice: long).
        find_package(Python3 COMPONENTS Interpreter QUIET)
        set(_ROCKE_HAS_PYTEST 1)
        if(Python3_EXECUTABLE)
            execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import pytest"
                            RESULT_VARIABLE _ROCKE_HAS_PYTEST OUTPUT_QUIET ERROR_QUIET)
        endif()
        hiptensor_rocke_pack_targets("${GPU_TARGETS}" _ROCKE_CROSS_CHECK_TARGETS)
        if(DEFINED HIPTENSOR_ROCKE_PACK_TARGETS AND NOT HIPTENSOR_ROCKE_PACK_TARGETS STREQUAL "")
            set(_ROCKE_CROSS_CHECK_TARGETS ${HIPTENSOR_ROCKE_PACK_TARGETS})
        endif()
        if(_ROCKE_HAS_PYTEST EQUAL 0 AND _ROCKE_CROSS_CHECK_TARGETS)
            string(REPLACE ";" ":" _ROCKE_CROSS_CHECK_ARCHS "${_ROCKE_CROSS_CHECK_TARGETS}")
            add_test(NAME rocke_pack_cross_check_test
                     COMMAND "${Python3_EXECUTABLE}" -m pytest -p no:cacheprovider -v -rs
                             "${HIPTENSOR_ROCKE_SOURCE_DIR}/library/tests/test_hiptensor_pack_cross_check.py")
            set_tests_properties(rocke_pack_cross_check_test PROPERTIES
                ENVIRONMENT "ROCKE_HIPTENSOR_PACK_BUILDER=$<TARGET_FILE:hiptensor_rocke_pack_builder>;ROCKE_HIPTENSOR_PACK_ARCHS=${_ROCKE_CROSS_CHECK_ARCHS};PYTHONPATH=${HIPTENSOR_ROCKE_SOURCE_DIR}/platform/python:${HIPTENSOR_ROCKE_SOURCE_DIR}/library:$ENV{PYTHONPATH};PYTHONDONTWRITEBYTECODE=1;ROCKE_BACKEND=python"
                TIMEOUT 7200)
        else()
            message(STATUS "rocKE pack cross-check: no Python with pytest or no pack target; "
                           "rocke_pack_cross_check_test not added")
        endif()
    elseif(TARGET hiptensor_rocke_pack_builder)
        message(STATUS "rocKE pack manifests: ${_ROCKE_MANIFESTS} not found; "
                       "rocke_pack_manifest_test not added")
    endif()
    add_hiptensor_unit_test(rocke_backend_test
        ${HIPTENSOR_COMMON_TEST_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/rocke_backend_test.cpp
        )
    # The shipped selection models (HIPTENSOR_ROCKE_SELECT_MODEL_DIR) read and walked by the
    # library's model code against the exporter's check vectors (CPU only).
    add_hiptensor_unit_test(rocke_select_model_test
        ${CMAKE_CURRENT_SOURCE_DIR}/rocke_select_model_test.cpp
        ${PROJECT_SOURCE_DIR}/library/src/backends/rocke/rocke_select_model.cpp
        )
    target_compile_definitions(rocke_select_model_test PRIVATE
        HIPTENSOR_ROCKE_SELECT_MODEL_DIR="${HIPTENSOR_ROCKE_SELECT_MODEL_DIR}")
    # Every contraction on the f64 route (operands converted to f64, one rounding of the result).
    add_test(NAME rocke_backend_test_f64_route
             COMMAND rocke_backend_test --gtest_filter=*RockeContraction*)
    set_tests_properties(rocke_backend_test_f64_route PROPERTIES
        ENVIRONMENT "HIPTENSOR_ROCKE_CONTRACTION_ROUTE=f64"
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    if(HIPTENSOR_ENABLE_ROCKE_JIT)
        # The JIT and kernel cache, and every rocke_backend_test case with every elementwise
        # launch specialized (HIPTENSOR_ROCKE_JIT=all: also the launches without a vector form).
        add_hiptensor_unit_test(rocke_jit_test
            ${HIPTENSOR_COMMON_TEST_SOURCES}
            ${CMAKE_CURRENT_SOURCE_DIR}/rocke_jit_test.cpp
            )
        # The contraction JIT (rocKE's tensor GETT planner at plan creation).
        add_hiptensor_unit_test(rocke_gett_jit_test
            ${HIPTENSOR_COMMON_TEST_SOURCES}
            ${CMAKE_CURRENT_SOURCE_DIR}/rocke_gett_jit_test.cpp
            )
        # rocKE's C++ engine: the route tables and their plans the JIT's routes are compared with
        target_link_libraries(rocke_gett_jit_test PRIVATE hiptensor_rocke_core)
        add_test(NAME rocke_backend_test_jit COMMAND rocke_backend_test)
        set_tests_properties(rocke_backend_test_jit PROPERTIES
            ENVIRONMENT "HIPTENSOR_ROCKE_TEST_JIT=1;HIPTENSOR_ROCKE_JIT=all"
            WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endif()
endif()
 # add_hiptensor_unit_test(interface_test
     # ${HIPTENSOR_COMMON_TEST_SOURCES}
     # ${CMAKE_CURRENT_SOURCE_DIR}/interface_test.cpp
     # )
