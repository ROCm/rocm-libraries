#[[
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]

cmake_minimum_required(VERSION 3.10)

add_test(
  NAME
    rpp_cpu_test_u08_01
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/HOST_NEW"
                              "${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "uniqueFunctionalities_host"
            0 0
)
add_test(NAME rpp_cpu_test_u08_02 COMMAND uniqueFunctionalities_host 0 1 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
add_test(NAME rpp_cpu_test_u08_03 COMMAND uniqueFunctionalities_host 0 2 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
add_test(NAME rpp_cpu_test_u08_04 COMMAND uniqueFunctionalities_host 0 3 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
add_test(NAME rpp_cpu_test_u08_05 COMMAND uniqueFunctionalities_host 0 4 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
add_test(NAME rpp_cpu_test_u08_06 COMMAND uniqueFunctionalities_host 0 5 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
add_test(NAME rpp_cpu_test_u08_07 COMMAND uniqueFunctionalities_host 0 6 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
add_test(NAME rpp_cpu_test_u08_08 COMMAND uniqueFunctionalities_host 0 7 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
add_test(NAME rpp_cpu_test_u08_09 COMMAND uniqueFunctionalities_host 0 8 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
# TBD - ARG 0 9 - fails with seg fault
#add_test(NAME rpp_cpu_test_u8_10 COMMAND uniqueFunctionalities_host 0 9 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
add_test(NAME rpp_cpu_test_u08_11 COMMAND uniqueFunctionalities_host 0 10 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
add_test(NAME rpp_cpu_test_u08_12 COMMAND uniqueFunctionalities_host 0 11 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
add_test(NAME rpp_cpu_test_f16_13 COMMAND uniqueFunctionalities_host 1 0 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
add_test(NAME rpp_cpu_test_f32_14 COMMAND uniqueFunctionalities_host 2 0 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)
add_test(NAME rpp_cpu_test_i08_15 COMMAND uniqueFunctionalities_host 5 0 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_cpu_test)

if( "${BACKEND}" STREQUAL "HIP")
    add_test(
        NAME
            rpp_hip_test_u08_01
        COMMAND
            "${CMAKE_CTEST_COMMAND}"
                    --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/HIP_NEW"
                              "${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test"
                    --build-generator "${CMAKE_GENERATOR}"
                    --test-command "uniqueFunctionalities_hip"
                    0 0
    )
    add_test(NAME rpp_hip_test_u08_02 COMMAND uniqueFunctionalities_hip 0 1 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    add_test(NAME rpp_hip_test_u08_03 COMMAND uniqueFunctionalities_hip 0 2 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    add_test(NAME rpp_hip_test_u08_04 COMMAND uniqueFunctionalities_hip 0 3 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    add_test(NAME rpp_hip_test_u08_05 COMMAND uniqueFunctionalities_hip 0 4 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    add_test(NAME rpp_hip_test_u08_06 COMMAND uniqueFunctionalities_hip 0 5 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    add_test(NAME rpp_hip_test_u08_07 COMMAND uniqueFunctionalities_hip 0 8 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    add_test(NAME rpp_hip_test_f16_08 COMMAND uniqueFunctionalities_hip 1 1 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    add_test(NAME rpp_hip_test_f32_09 COMMAND uniqueFunctionalities_hip 2 1 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    add_test(NAME rpp_hip_test_i08_10 COMMAND uniqueFunctionalities_hip 5 1 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    # TBD - NEED to investigate these test as they are currently turned off
    #add_test(NAME rpp_hip_test_f16_13 COMMAND uniqueFunctionalities_hip 1 0 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    #add_test(NAME rpp_hip_test_f32_14 COMMAND uniqueFunctionalities_hip 2 0 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    #add_test(NAME rpp_hip_test_u08_f16_15 COMMAND uniqueFunctionalities_hip 3 0 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    #add_test(NAME rpp_hip_test_u08_f32_16 COMMAND uniqueFunctionalities_hip 4 0 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    #add_test(NAME rpp_hip_test_i08_17 COMMAND uniqueFunctionalities_hip 5 0 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
    #add_test(NAME rpp_hip_test_u08_i08_18 COMMAND uniqueFunctionalities_hip 6 0 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rpp_hip_test)
elseif( "${BACKEND}" STREQUAL "OCL")
    # TBD: Add OCL Tests
endif()
