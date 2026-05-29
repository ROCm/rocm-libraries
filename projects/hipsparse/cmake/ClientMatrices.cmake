# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(NOT CMAKE_MATRICES_DIR)
  message(FATAL_ERROR "Unspecified CMAKE_MATRICES_DIR")
endif()

if(NOT CONVERT_SOURCE)
  set(CONVERT_SOURCE "${CMAKE_SOURCE_DIR}/deps/convert.cpp")
endif()

# convert relative path to absolute
get_filename_component(PROJECT_BINARY_DIR "${PROJECT_BINARY_DIR}"
                       ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
get_filename_component(CMAKE_MATRICES_DIR "${CMAKE_MATRICES_DIR}"
                       ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/clients")

include("${CMAKE_CURRENT_LIST_DIR}/hipsparse_test_matrices_common.cmake")

if(HIPSPARSE_ENABLE_ASAN)
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "${CONVERT_SOURCE}" -O3 -fsanitize=address -static-libasan -o "${PROJECT_BINARY_DIR}/clients/mtx2csr.exe"
    RESULT_VARIABLE STATUS)
else()
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "${CONVERT_SOURCE}" -O3 -o "${PROJECT_BINARY_DIR}/clients/mtx2csr.exe"
    RESULT_VARIABLE STATUS)
endif()

if(STATUS AND NOT STATUS EQUAL 0)
  message(FATAL_ERROR "mtx2csr.exe failed to build, aborting.")
endif()

hipsparse_prepare_test_matrices("${PROJECT_BINARY_DIR}/clients/mtx2csr.exe" "mtx2csr.exe")
