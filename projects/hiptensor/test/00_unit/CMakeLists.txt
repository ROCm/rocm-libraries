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
add_hiptensor_unit_test(elementwise_instance_selection_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_instance_selection_test.cpp
    )
add_hiptensor_unit_test(api_safety_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/api_safety_test.cpp
    )
add_hiptensor_unit_test(plan_cache_unit_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/plan_cache_unit_test.cpp
    )
add_hiptensor_unit_test(backend_seam_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/backend_seam_test.cpp
    )
# The rocKE backend's kernels through the public API (HIPTENSOR_BACKEND=rocke, no fallback).
if(HIPTENSOR_ENABLE_ROCKE_BACKEND)
    add_hiptensor_unit_test(rocke_backend_test
        ${HIPTENSOR_COMMON_TEST_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/rocke_backend_test.cpp
        )
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
        add_test(NAME rocke_backend_test_jit COMMAND rocke_backend_test)
        set_tests_properties(rocke_backend_test_jit PROPERTIES
            ENVIRONMENT "HIPTENSOR_ROCKE_TEST_JIT=1;HIPTENSOR_ROCKE_JIT=all"
            WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endif()
endif()
add_hiptensor_unit_test(contraction_selection_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/contraction_selection_test.cpp
    )
# The actor-critic cases that need the tables for row-major strides, which name kernels that
# cannot run column-major problems. They skip with that message when the tables are not selected.
set(_HIPTENSOR_ROW_MAJOR_TABLES_ENV "HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR=OFF")
if(WIN32)
    list(APPEND _HIPTENSOR_ROW_MAJOR_TABLES_ENV
        "PATH=${CMAKE_INSTALL_PREFIX}/bin$<SEMICOLON>${CMAKE_RUNTIME_OUTPUT_DIRECTORY}$<SEMICOLON>$ENV{PATH}")
endif()
add_test(NAME contraction_selection_test_row_major_tables
         COMMAND contraction_selection_test --gtest_filter=ActorCriticRowMajorTableTest.*)
set_tests_properties(contraction_selection_test_row_major_tables PROPERTIES
    ENVIRONMENT "${_HIPTENSOR_ROW_MAJOR_TABLES_ENV}"
    FAIL_REGULAR_EXPRESSION "Runs only with the row-major actor-critic tables"
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
add_hiptensor_unit_test(large_tensor_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/large_tensor_test.cpp
    )
add_hiptensor_unit_test(operation_validation_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/operation_validation_test.cpp
    )
add_hiptensor_unit_test(workspace_size_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/workspace_size_test.cpp
    )

# Has its own main: the shared one stops early on the hosts this test is about.
add_hiptensor_unit_test(device_support_test
    ${CMAKE_CURRENT_SOURCE_DIR}/device_support_test.cpp
    )
# The same binary with every GPU hidden, which simulates a host without a GPU.
set(_HIPTENSOR_NO_VISIBLE_DEVICE_ENV "HIP_VISIBLE_DEVICES=-1" "ROCR_VISIBLE_DEVICES=-1")
if(WIN32)
    list(APPEND _HIPTENSOR_NO_VISIBLE_DEVICE_ENV
        "PATH=${CMAKE_INSTALL_PREFIX}/bin$<SEMICOLON>${CMAKE_RUNTIME_OUTPUT_DIRECTORY}$<SEMICOLON>$ENV{PATH}")
endif()
add_test(NAME device_support_test_no_visible_device
         COMMAND device_support_test --gtest_filter=*NoVisibleDevice*)
set_tests_properties(device_support_test_no_visible_device PROPERTIES
    ENVIRONMENT "${_HIPTENSOR_NO_VISIBLE_DEVICE_ENV}"
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# Fills the device memory, so it must not share the GPU with other tests.
add_hiptensor_unit_test(alloc_failure_test
    ${HIPTENSOR_COMMON_TEST_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/alloc_failure_test.cpp
    )
set_tests_properties(alloc_failure_test PROPERTIES RUN_SERIAL TRUE)
file(APPEND "${INSTALL_TEST_FILE}" "set_tests_properties(alloc_failure_test PROPERTIES RUN_SERIAL TRUE)\n")
file(APPEND "${INSTALL_SMOKE_TEST_FILE}" "set_tests_properties(\"alloc_failure_test smoke\" PROPERTIES RUN_SERIAL TRUE)\n")
file(APPEND "${INSTALL_REGRESSION_TEST_FILE}" "set_tests_properties(\"alloc_failure_test regression\" PROPERTIES RUN_SERIAL TRUE)\n")
 # add_hiptensor_unit_test(interface_test
     # ${HIPTENSOR_COMMON_TEST_SOURCES}
     # ${CMAKE_CURRENT_SOURCE_DIR}/interface_test.cpp
     # )
